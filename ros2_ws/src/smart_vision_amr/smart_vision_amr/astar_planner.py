"""A* ROS adapter: OccupancyGrid + persistent cells + PoseStamped goal -> Path."""

import json
import time
from math import cos, hypot, isfinite, pi, sin, sqrt

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Int32MultiArray, String
from tf2_ros import Buffer, TransformException, TransformListener
from collections import deque

from smart_vision_amr.astar import AStarPlanner
from smart_vision_amr.clearance_cost import clearance_penalties
from smart_vision_amr.grid import Grid
from smart_vision_amr.map_utils import (cell_to_world, inflate_cells, occupancy_to_grid,
                                        world_to_cell, yaw_from_quaternion)
from smart_vision_amr.reverse_recovery import (Breadcrumb, GridGeometry,
                                               select_reverse_egress)


class AStarPlannerNode(Node):
    def __init__(self):
        super().__init__("astar_planner")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("occupied_threshold", 65)
        self.declare_parameter("obstacle_inflation_cells", 2)
        self.declare_parameter("replan_retry_period", 2.0)
        self.declare_parameter("robot_radius", 0.22)
        self.declare_parameter("start_egress_enabled", True)
        self.declare_parameter("start_egress_clearance_margin", 0.02)
        self.declare_parameter("start_egress_max_cells", 7)
        self.declare_parameter("live_obstacle_layer_required", False)
        self.declare_parameter("live_obstacle_timeout", 2.0)
        self.declare_parameter("preferred_clearance", 0.45)
        self.declare_parameter("clearance_penalty_gain", 1.5)
        self.declare_parameter("reverse_recovery_enabled", True)
        self.declare_parameter("reverse_recovery_min_distance", 0.10)
        self.declare_parameter("reverse_recovery_max_distance", 0.30)
        self.declare_parameter("reverse_recovery_clearance_margin", 0.02)
        self.declare_parameter("reverse_recovery_speed", 0.04)
        self.declare_parameter("reverse_recovery_max_attempts", 3)
        self.map = None
        self.raw_static_grid = None
        self.map_signature = None
        self.remembered = set()
        self.observed = set()
        self.last_observation_time = None
        self.last_goal = None
        self.planned_cells = []
        self.waiting_for_transform = False
        self.waiting_for_replan = False
        self.navigation_suspended = False
        self.goal_completed = False
        self.last_plan_time = float("-inf")
        self.invalid_path_published = False
        self.stop_stamp_ns = None
        self.breadcrumbs = deque(maxlen=300)
        self.breadcrumb_odom_anchor = None
        self.latest_odom = None
        self.odom_samples = deque(maxlen=100)
        self.breadcrumb_map_key = None
        self.recovery_active = False
        self.recovery_requested_at = None
        self.recovery_cancel_due_map_change = False
        self.recovery_attempts = 0
        self.recovery_attempted_cells = set()
        self.planner = AStarPlanner()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.path_pub = self.create_publisher(Path, "/smart_vision_amr/path", 10)
        self.status_pub = self.create_publisher(String, "/smart_vision_amr/planner_status", latched)
        self.recovery_pub = self.create_publisher(String, "/smart_vision_amr/recovery_request", 10)
        self.create_subscription(OccupancyGrid, self.get_parameter("map_topic").value, self.on_map, latched)
        self.create_subscription(Int32MultiArray, "/smart_vision_amr/remembered_cells", self.on_memory, latched)
        self.create_subscription(Int32MultiArray, "/smart_vision_amr/observed_obstacles",
                                 self.on_observed, 10)
        self.create_subscription(PoseStamped, "/smart_vision_amr/goal", self.on_goal, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(String, "/smart_vision_amr/navigation_status",
                                 self.on_navigation_status, 10)
        self.create_timer(0.25, self.retry_transform)
        self.publish_status("IDLE", "Waiting for an explicit goal")

    def on_map(self, msg):
        # A previous trail is only meaningful in the exact saved map in which
        # its map-frame poses were measured. Repeated publications of that map
        # are harmless; a reload, new SLAM content, or source-frame change is not.
        info = msg.info
        origin = info.origin
        map_key = (msg.header.frame_id, info.width, info.height,
                   info.resolution, info.map_load_time.sec, info.map_load_time.nanosec,
                   origin.position.x, origin.position.y, origin.position.z,
                   origin.orientation.x, origin.orientation.y,
                   origin.orientation.z, origin.orientation.w, tuple(msg.data))
        map_replaced = self.breadcrumb_map_key is not None and map_key != self.breadcrumb_map_key
        if map_replaced:
            self._clear_breadcrumbs()
        self.breadcrumb_map_key = map_key
        threshold = self.get_parameter("occupied_threshold").value
        inflation = self.get_parameter("obstacle_inflation_cells").value
        self.map = (msg, occupancy_to_grid(msg, threshold, inflation))
        self.raw_static_grid = occupancy_to_grid(msg, threshold, 0)
        self.map_signature = (threshold, inflation)
        if self._goal_is_active():
            if getattr(self, "recovery_active", False):
                if map_replaced and not self.recovery_cancel_due_map_change:
                    # The reverse swept volume was certified against the old
                    # map. The follower consumes this explicit cancellation
                    # even before BackUp starts; an empty Path alone cannot
                    # resolve the cross-topic request/path ordering race.
                    self.recovery_cancel_due_map_change = True
                    goal = self.last_goal
                    cancel = String()
                    cancel.data = json.dumps({
                        "goal_id": f"{goal.header.stamp.sec}:{goal.header.stamp.nanosec}",
                        "cancel": True,
                        "reason": "Map changed during reverse recovery"})
                    self.recovery_pub.publish(cancel)
                    self.invalid_path_published = False
                    self._invalidate_path("Map changed during reverse recovery; canceling reverse")
                return
            if any(self.map[1].is_static_obstacle(cell) for cell in self.planned_cells):
                self._invalidate_path("Updated map blocks the active route")
            self._request_replan("Map updated")

    def _goal_is_active(self):
        return (self.last_goal is not None and not self.navigation_suspended
                and not self.goal_completed)

    def _retry_period(self):
        # Keep retry rate bounded even if a mistaken runtime parameter is zero.
        return max(0.25, float(self.get_parameter("replan_retry_period").value))

    def retry_transform(self):
        """Retry no-path, missing-map and missing-TF conditions without losing the goal."""
        if self.map is not None:
            signature = (self.get_parameter("occupied_threshold").value,
                         self.get_parameter("obstacle_inflation_cells").value)
            if signature != getattr(self, "map_signature", None):
                self.on_map(self.map[0])
        if self._goal_is_active() and not getattr(self, "recovery_active", False):
            self._record_breadcrumb()
        if getattr(self, "recovery_active", False):
            requested = self.recovery_requested_at
            # The follower has a 5 s cancel wait plus a 15 s BackUp watchdog.
            # If no correlated result arrives, release the retained goal from
            # an otherwise permanent RECOVERING hold. The attempted cell stays
            # marked so a lost response cannot trigger repeated reversals.
            if requested is not None and time.monotonic() - requested > 25.0:
                self.recovery_active = False
                self.recovery_requested_at = None
                self.recovery_cancel_due_map_change = False
                self._clear_breadcrumbs()
                self._invalidate_path("Reverse recovery timed out; goal retained")
                self.last_plan_time = time.monotonic()
            return
        if self._goal_is_active() and not self._observations_ready():
            self._invalidate_path("Waiting for fresh LiDAR obstacle observations; goal retained")
            return
        if self.waiting_for_replan and self._goal_is_active():
            self._request_replan("Retrying retained goal")

    def _observations_ready(self):
        if not self.get_parameter("live_obstacle_layer_required").value:
            return True
        timeout = max(.25, float(self.get_parameter("live_obstacle_timeout").value))
        return (self.last_observation_time is not None
                and time.monotonic()-self.last_observation_time < timeout)

    def _request_replan(self, reason):
        if not self._goal_is_active() or getattr(self, "recovery_active", False):
            return
        self.waiting_for_replan = True
        if time.monotonic() - self.last_plan_time >= self._retry_period():
            self.plan_to(self.last_goal)

    def _invalidate_path(self, reason):
        # One empty Path cancels the unsafe action. Repeated retries must not
        # continuously cancel a replacement action or flood navigation_status.
        self.planned_cells = []
        if not self.invalid_path_published:
            path = Path()
            path.header.frame_id = self.get_parameter("map_frame").value
            path.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(path)
            self.invalid_path_published = True
        self.waiting_for_replan = True
        self.publish_status("WAIT_REPLAN", reason)

    def publish_status(self, status, message):
        goal = self.last_goal
        payload = {"status": status, "message": message, "goal": None, "goal_stamp": None}
        if goal is not None:
            payload["goal"] = {"x": goal.pose.position.x, "y": goal.pose.position.y,
                               "frame_id": goal.header.frame_id or self.get_parameter("map_frame").value}
            payload["goal_stamp"] = {"sec": goal.header.stamp.sec,
                                     "nanosec": goal.header.stamp.nanosec}
        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)

    def on_navigation_status(self, msg):
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if payload.get("manual_stop"):
            stopped = payload.get("stop_stamp") or {}
            if "sec" not in stopped or "nanosec" not in stopped:
                return
            boundary = stopped["sec"] * 1_000_000_000 + stopped["nanosec"]
            self.stop_stamp_ns = max(boundary, self.stop_stamp_ns or 0)
            goal_stamp = (0 if self.last_goal is None else
                          self.last_goal.header.stamp.sec * 1_000_000_000
                          + self.last_goal.header.stamp.nanosec)
            # /goal and the cancel service have no cross-topic ordering. A stop
            # arriving late must not cancel a deliberately newer stamped goal.
            if goal_stamp == 0 or goal_stamp <= self.stop_stamp_ns:
                self.navigation_suspended = True
                self.recovery_active = False
                self.recovery_requested_at = None
                self.recovery_cancel_due_map_change = False
                self._clear_breadcrumbs()
                self.waiting_for_replan = False
                self.publish_status("IDLE", "Manually stopped; send a newly ROS-stamped goal to resume")
            return
        if self.last_goal is None or not self._status_matches_goal(payload):
            return
        recovery_status = payload.get("status")
        if recovery_status == "RECOVERING":
            self.publish_status("RECOVERING", "Verified short reverse is in progress")
            return
        if recovery_status == "RECOVERY_STOPPED":
            # The follower has now observed the BackUp action's terminal
            # result. Any route planned while cancellation was still pending
            # may have been discarded, so plan again from the current pose.
            self.recovery_active = False
            self.recovery_requested_at = None
            self.recovery_cancel_due_map_change = False
            self._clear_breadcrumbs()
            if self._goal_is_active():
                self.last_plan_time = float("-inf")
                self._request_replan("Reverse action stopped; replanning retained goal")
            return
        if recovery_status in ("RECOVERY_SUCCEEDED", "RECOVERY_FAILED"):
            if not getattr(self, "recovery_active", False):
                return
            self.recovery_active = False
            self.recovery_requested_at = None
            self.recovery_cancel_due_map_change = False
            self._clear_breadcrumbs()
            # Backing up changes the start pose. Replan from fresh TF, retaining
            # the same explicit mission goal. Failed recovery is never repeated
            # from the same cell because it is recorded in attempted_cells.
            if recovery_status == "RECOVERY_SUCCEEDED":
                self.last_plan_time = float("-inf")
                self._request_replan("Reverse recovery finished; replanning")
            else:
                self._invalidate_path("Reverse recovery failed; goal retained for safe replanning")
                self.last_plan_time = time.monotonic()
            return
        if payload.get("status") == "SUCCEEDED":
            self.goal_completed = True
            self.waiting_for_replan = False
            self.publish_status("IDLE", "Goal completed")
        elif payload.get("status") == "FAILED" and self._goal_is_active():
            # Nav2 may abort while the geometric route is unchanged. Retry the
            # retained goal; the follower discards its failed duplicate cache.
            self._invalidate_path("Controller failed; waiting to replan retained goal")
            self.last_plan_time = time.monotonic()

    def _status_matches_goal(self, payload):
        target = payload.get("goal") or {}
        goal = self.last_goal
        frame = goal.header.frame_id or self.get_parameter("map_frame").value
        if (target.get("frame_id") != frame
                or target.get("x") != goal.pose.position.x
                or target.get("y") != goal.pose.position.y):
            return False
        return payload.get("goal_stamp") == {"sec": goal.header.stamp.sec,
                                              "nanosec": goal.header.stamp.nanosec}

    def on_memory(self, msg):
        if self.map is None:
            return
        width, height = self.map[0].info.width, self.map[0].info.height
        updated = {(index % width, index // width) for index in msg.data
                   if 0 <= index < width * height}
        if updated != self.remembered:
            added = updated - self.remembered
            removed = self.remembered - updated
            self.remembered = updated
            # Empty-route recovery must also react to obstacle removal. While
            # driving, unrelated memory churn must not cancel a valid Nav2 goal.
            added_blocks = self._changes_block_path(added)
            removed_blocks = self._changes_block_path(removed)
            if self._goal_is_active() and (self.waiting_for_replan
                    or added_blocks or removed_blocks):
                if added_blocks:
                    self._invalidate_path("Remembered obstacle intersects the route")
                self._request_replan("Remembered obstacles changed")

    def on_observed(self, msg):
        if self.map is None:
            return
        self.last_observation_time = time.monotonic()
        width, height = self.map[0].info.width, self.map[0].info.height
        updated = {(index % width, index // width) for index in msg.data
                   if 0 <= index < width * height}
        if updated != self.observed:
            added = updated - self.observed
            self.observed = updated
            # Laser scans vary at sensor frequency. Stop on a newly blocked
            # route, and rate-limit recovery even when no planned cells remain.
            # The observation publisher already removed geometrically matched
            # static-wall echoes, so do not apply a second cell-distance filter.
            if self._goal_is_active() and (self.waiting_for_replan
                    or self._changes_block_path(added)):
                if self._changes_block_path(added):
                    self._invalidate_path("Observed obstacle intersects the route")
                self._request_replan("Observed obstacles changed")
        elif self.waiting_for_replan and self._goal_is_active():
            # Fresh empty/unchanged scans also recover a stale-sensor hold.
            self._request_replan("LiDAR obstacle observations restored")

    def _changes_block_path(self, cells):
        if not cells or not self.planned_cells:
            return False
        width, height = self.map[0].info.width, self.map[0].info.height
        inflation = self.get_parameter("obstacle_inflation_cells").value
        blocked = inflate_cells(cells, width, height, inflation)
        return any(cell in blocked for cell in self.planned_cells)

    def on_goal(self, goal):
        stamp_ns = goal.header.stamp.sec * 1_000_000_000 + goal.header.stamp.nanosec
        if self.stop_stamp_ns is not None and (stamp_ns == 0 or stamp_ns <= self.stop_stamp_ns):
            self.get_logger().warning(
                "Ignored delayed/unstamped goal after manual stop; send a newly ROS-stamped goal")
            return
        # A delivery may start the next sector directly from the previous one.
        # Preserve that same physical trail across normal goal boundaries; the
        # reverse attempt budget, however, belongs to the new explicit goal.
        # An interrupted reverse has no known end pose and invalidates it.
        if self.recovery_active:
            self._clear_breadcrumbs()
        self.last_goal = goal
        self.navigation_suspended = False
        self.goal_completed = False
        self.recovery_active = False
        self.recovery_requested_at = None
        self.recovery_cancel_due_map_change = False
        self.recovery_attempts = 0
        self.recovery_attempted_cells = set()
        # An explicit request is a new attempt even when coordinates are identical.
        self.invalid_path_published = False
        self._record_breadcrumb()
        self.plan_to(goal)

    def _clear_breadcrumbs(self):
        self.breadcrumbs.clear()
        self.breadcrumb_odom_anchor = None

    def on_odom(self, msg):
        """Keep the latest wheel/IMU odometry as an independent motion witness."""
        stamp = msg.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        pose = msg.pose.pose
        sample = (time.monotonic(), stamp_ns, float(pose.position.x),
                  float(pose.position.y), yaw_from_quaternion(pose.orientation),
                  msg.header.frame_id)
        if (stamp_ns <= 0 or not all(isfinite(value) for value in sample[:5])
                or not sample[5]):
            self.latest_odom = None
            self.odom_samples.clear()
            self._clear_breadcrumbs()
            return
        self.latest_odom = sample
        self.odom_samples.append(sample)

    @staticmethod
    def _angle_delta(a, b):
        return (a - b + pi) % (2.0 * pi) - pi

    def _record_breadcrumb(self, transform=None):
        """Record map poses only when fresh, synchronized odometry verifies travel.

        AMCL may move ``map -> odom`` while the chassis is stationary. Using TF
        alone would turn that correction into a fictitious retreat trail.
        """
        if not self._goal_is_active():
            return False
        if transform is None:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.get_parameter("map_frame").value,
                    self.get_parameter("base_frame").value, rclpy.time.Time())
            except TransformException:
                self._clear_breadcrumbs()
                return False
        stamp = getattr(getattr(transform, "header", None), "stamp", None)
        now = time.monotonic()
        if stamp is None or not self.odom_samples:
            self._clear_breadcrumbs()
            return False
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        # Choose the odometry sample closest to the map TF timestamp. The
        # newest /odom can describe motion that the delayed AMCL TF has not
        # reached yet, so comparing those two would reject real travel.
        odom = min(self.odom_samples, key=lambda item: abs(item[1] - stamp_ns))
        now_ns = getattr(self.get_clock().now(), "nanoseconds", None)
        if (stamp_ns <= 0 or now - odom[0] > .75 or now < odom[0]
                or abs(stamp_ns - odom[1]) > 150_000_000
                or (now_ns is not None and (now_ns - stamp_ns > 1_000_000_000
                                            or stamp_ns - now_ns > 200_000_000))):
            self._clear_breadcrumbs()
            return False
        translation = transform.transform.translation
        rotation = getattr(transform.transform, "rotation", None)
        if rotation is None:
            self._clear_breadcrumbs()
            return False
        yaw = yaw_from_quaternion(rotation)
        sample = Breadcrumb(now, float(translation.x),
                            float(translation.y), yaw)
        if not all(isfinite(value) for value in (sample.x, sample.y, sample.yaw)):
            self._clear_breadcrumbs()
            return False
        history = self.breadcrumbs
        anchor = self.breadcrumb_odom_anchor
        if not history or anchor is None:
            self._clear_breadcrumbs()
            history.append(sample)
            self.breadcrumb_odom_anchor = odom
            return True
        previous = history[-1]
        distance = hypot(sample.x - previous.x, sample.y - previous.y)
        turn = self._angle_delta(sample.yaw, previous.yaw)
        odom_dx, odom_dy = odom[2] - anchor[2], odom[3] - anchor[3]
        odom_distance = hypot(odom_dx, odom_dy)
        odom_turn = self._angle_delta(odom[4], anchor[4])
        # Express the odometry delta in the map heading at the anchor. A map
        # correction moves the TF pose but does not move this odometry delta.
        rotation_to_map = previous.yaw - anchor[4]
        expected_dx = cos(rotation_to_map) * odom_dx - sin(rotation_to_map) * odom_dy
        expected_dy = sin(rotation_to_map) * odom_dx + cos(rotation_to_map) * odom_dy
        error = hypot(sample.x - previous.x - expected_dx,
                      sample.y - previous.y - expected_dy)
        odom_forward = odom_dx * cos(anchor[4]) + odom_dy * sin(anchor[4])
        odom_lateral = abs(-odom_dx * sin(anchor[4]) + odom_dy * cos(anchor[4]))
        map_forward = ((sample.x - previous.x) * cos(previous.yaw)
                       + (sample.y - previous.y) * sin(previous.yaw))
        if (odom[5] != anchor[5] or odom[1] < anchor[1]
                or distance > .15 or odom_distance > .15
                or abs(turn) > .15 or abs(odom_turn) > .15
                or abs(turn - odom_turn) > .08
                or error > .015 + .20 * odom_distance
                or odom_lateral > .015 + .20 * odom_distance
                or (distance >= .025 and (odom_forward < .015 or map_forward < .015))
                or odom_forward < -.005):
            self._clear_breadcrumbs()
            history.append(sample)
            self.breadcrumb_odom_anchor = odom
            return True
        if distance < .025 or odom_distance < .025:
            # Keep the verified location, not an AMCL-only correction. Refresh
            # its age while stationary; movement accumulates from the anchor.
            history[-1] = Breadcrumb(now, previous.x, previous.y, previous.yaw)
            return True
        history.append(sample)
        self.breadcrumb_odom_anchor = odom
        return True

    def _maybe_request_reverse(self, start, end, map_msg, grid, raw_grid,
                               raw_dynamic, inflated_dynamic, cell_penalty,
                               trail_verified):
        """Request a bounded Nav2 backup only when a verified A* exit exists."""
        if (not trail_verified or not self.get_parameter("reverse_recovery_enabled").value
                or getattr(self, "recovery_active", False)
                or start in self.recovery_attempted_cells
                or self.recovery_attempts >= int(self.get_parameter("reverse_recovery_max_attempts").value)
                or not self._observations_ready()
                or grid.is_static_obstacle(end) or end in inflated_dynamic):
            return False

        # Dynamic raw scan hits and confirmed memory have the same hard-stop
        # status as raw map occupancy while checking the reverse swept volume.
        raw_with_live = Grid(tuple(tuple(raw_grid.rows[y][x] or (x, y) in raw_dynamic
                                        for x in range(raw_grid.width))
                                   for y in range(raw_grid.height)))
        inflated_with_live = Grid(tuple(tuple(grid.rows[y][x] or (x, y) in inflated_dynamic
                                             for x in range(grid.width))
                                        for y in range(grid.height)))
        origin = map_msg.info.origin
        geometry = GridGeometry(map_msg.info.resolution, origin.position.x,
                                origin.position.y, yaw_from_quaternion(origin.orientation))

        def can_reach_retained_goal(candidate_cell):
            return bool(self.planner.plan(grid, candidate_cell, end, inflated_dynamic,
                                          cell_penalty=cell_penalty))

        candidate = select_reverse_egress(
            tuple(self.breadcrumbs), raw_with_live, inflated_with_live, geometry,
            can_reach_retained_goal, now=time.monotonic(),
            robot_radius=float(self.get_parameter("robot_radius").value),
            clearance_margin=float(self.get_parameter("reverse_recovery_clearance_margin").value),
            min_distance=float(self.get_parameter("reverse_recovery_min_distance").value),
            max_distance=float(self.get_parameter("reverse_recovery_max_distance").value),
            max_history_age=90.0, max_sample_gap=30.0)
        if candidate is None:
            return False

        self.recovery_attempts += 1
        self.recovery_attempted_cells.add(start)
        self.recovery_active = True
        self.recovery_requested_at = time.monotonic()
        goal = self.last_goal
        request = String()
        request.data = json.dumps({
            "goal_id": f"{goal.header.stamp.sec}:{goal.header.stamp.nanosec}",
            "distance": candidate.distance_m,
            "speed": float(self.get_parameter("reverse_recovery_speed").value),
            "reason": "No forward A* route; verified recent straight track leads to a safe A* start",
            "start_cell": list(start), "exit_cell": list(candidate.target_cell)})
        self.recovery_pub.publish(request)
        self.publish_status("RECOVERING", f"Backing up {candidate.distance_m:.2f} m to replan retained goal")
        self.get_logger().warning(
            f"No forward route; requested collision-checked {candidate.distance_m:.2f} m reverse "
            f"to {candidate.target_cell} (attempt {self.recovery_attempts})")
        return True

    def plan_to(self, goal):
        if getattr(self, "recovery_active", False):
            return
        self.last_plan_time = time.monotonic()
        self.waiting_for_replan = True
        self.publish_status("PLANNING", "Planning to retained goal")
        if self.map is None:
            self.waiting_for_transform = True
            self._invalidate_path("Waiting for /map")
            return
        if not self._observations_ready():
            self._invalidate_path("Waiting for fresh LiDAR obstacle observations; goal retained")
            return
        map_msg, grid = self.map
        signature = (self.get_parameter("occupied_threshold").value,
                     self.get_parameter("obstacle_inflation_cells").value)
        if signature != getattr(self, "map_signature", None):
            grid = occupancy_to_grid(map_msg, signature[0], signature[1])
            self.map = (map_msg, grid)
            self.raw_static_grid = occupancy_to_grid(map_msg, signature[0], 0)
            self.map_signature = signature
        raw_grid = self.raw_static_grid
        frame = self.get_parameter("map_frame").value
        if goal.header.frame_id and goal.header.frame_id != frame:
            self._invalidate_path(f"Goal frame must be {frame}, got {goal.header.frame_id}")
            self.waiting_for_replan = False
            self.publish_status("INVALID_GOAL", "Goal frame differs from map frame")
            return
        try:
            tf = self.tf_buffer.lookup_transform(frame, self.get_parameter("base_frame").value,
                                                 rclpy.time.Time())
        except TransformException as exc:
            self.waiting_for_transform = True
            self._invalidate_path(f"Waiting for robot transform: {exc}")
            return
        self.waiting_for_transform = False
        trail_verified = self._record_breadcrumb(transform=tf)
        start = world_to_cell(map_msg.info, tf.transform.translation.x, tf.transform.translation.y)
        end = world_to_cell(map_msg.info, goal.pose.position.x, goal.pose.position.y)
        width, height = map_msg.info.width, map_msg.info.height
        inflation = self.get_parameter("obstacle_inflation_cells").value
        # The change-memory node classifies each scan by comparing its measured
        # range with the first raw-map obstacle along that beam. These observed
        # cells are dynamic obstacles; expected static-wall echoes are absent.
        remembered_effective = self.remembered
        observed_effective = self.observed
        remembered_blocked = inflate_cells(remembered_effective, width, height, inflation)
        observed_blocked = inflate_cells(observed_effective, width, height, inflation)
        blocked = remembered_blocked | observed_blocked
        cell_penalty = clearance_penalties(
            raw_grid, observed_effective | remembered_effective,
            map_msg.info.resolution,
            float(self.get_parameter("preferred_clearance").value),
            float(self.get_parameter("clearance_penalty_gain").value))
        # A scan obstacle can enter the inflation band around the robot itself
        # even when the footprint is still clear. Permit the A* search to leave
        # that one dynamic cell; static-map and confirmed-memory collisions stay
        # hard failures, while Nav2's local costmap retains its collision safety.
        escape_cells = self._dynamic_escape_halo(
            start, observed_blocked | remembered_blocked, self.observed, self.remembered,
            grid, raw_grid, inflation,
            self.get_parameter("robot_radius").value / map_msg.info.resolution,
            (self.get_parameter("start_egress_clearance_margin").value
             / map_msg.info.resolution + sqrt(2.0) / 2.0))
        allow_dynamic_escape = bool(escape_cells)
        egress_steps = 0
        try:
            if grid.is_static_obstacle(start) and self.get_parameter("start_egress_enabled").value:
                resolution = map_msg.info.resolution
                # The measured pose may lie anywhere inside its cell: preserve
                # an extra half-cell diagonal in addition to footprint margin.
                margin = self.get_parameter("start_egress_clearance_margin").value + resolution*sqrt(2.0)/2.0
                cells = self.planner.plan_with_start_egress(
                    grid, start, end, blocked, raw_grid=raw_grid,
                    raw_blocked=self.observed | self.remembered,
                    robot_radius_cells=self.get_parameter("robot_radius").value/resolution,
                    margin_cells=margin/resolution,
                    max_egress_cells=min(int(inflation)+1, int(self.get_parameter("start_egress_max_cells").value)),
                    allow_blocked_start=allow_dynamic_escape, escape_cells=escape_cells,
                    cell_penalty=cell_penalty)
                if cells:
                    egress_steps = next((index for index, cell in enumerate(cells)
                                         if not grid.is_static_obstacle(cell)), 0)
            else:
                cells = self.planner.plan(grid, start, end, blocked,
                                          allow_blocked_start=allow_dynamic_escape,
                                          escape_cells=escape_cells,
                                          cell_penalty=cell_penalty)
        except ValueError:
            self._invalidate_path(f"Robot or goal lies outside the map: {start} -> {end}")
            return
        self.planned_cells = cells
        if not cells:
            reference_path = self._static_reference_path(
                grid, raw_grid, start, end, map_msg.info.resolution, inflation)
            reference_hits = [cell for cell in reference_path if cell in blocked]
            self.get_logger().error(
                f"No path from {start} to {end} (static_start={grid.is_static_obstacle(start)}, "
                f"raw_static_start={raw_grid.is_static_obstacle(start)}, "
                f"remembered_start={start in remembered_blocked}, "
                f"observed_start={start in observed_blocked}, "
                f"raw_observed_start={start in self.observed}, "
                f"dynamic_escape_cells={len(escape_cells)}, "
                f"static_goal={grid.is_static_obstacle(end)}, "
                f"remembered_goal={end in remembered_blocked}, "
                f"observed_goal={end in observed_blocked}, "
                f"observed_raw={len(self.observed)}, "
                f"observed_dynamic={len(observed_effective)}, "
                f"observed_inflated={len(observed_blocked)}, "
                f"remembered_raw={len(self.remembered)}, "
                f"remembered_inflated={len(remembered_blocked)}, "
                f"static_reference_cells={len(reference_path)}, "
                f"static_reference_dynamic_hits={len(reference_hits)}, "
                f"first_blocked_reference_cells={reference_hits[:8]})",
                throttle_duration_sec=10.0)
            self._invalidate_path(f"No safe A* route from {start} to {end}; retrying")
            self._maybe_request_reverse(start, end, map_msg, grid, raw_grid,
                                        observed_effective | remembered_effective,
                                        blocked, cell_penalty, trail_verified)
            return
        self.waiting_for_replan = False
        self.invalid_path_published = False
        path = Path()
        path.header.frame_id = frame
        path.header.stamp = self.get_clock().now().to_msg()
        from math import atan2, sin, cos, isfinite
        for index, cell in enumerate(cells):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = cell_to_world(map_msg.info, cell)
            if index + 1 < len(cells):
                next_x, next_y = cell_to_world(map_msg.info, cells[index + 1])
                yaw = atan2(next_y - pose.pose.position.y, next_x - pose.pose.position.x)
                pose.pose.orientation.z, pose.pose.orientation.w = sin(yaw / 2.0), cos(yaw / 2.0)
            else:
                q = goal.pose.orientation
                norm = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
                if isfinite(norm) and norm > 1e-9:
                    pose.pose.orientation.x, pose.pose.orientation.y = q.x / norm, q.y / norm
                    pose.pose.orientation.z, pose.pose.orientation.w = q.z / norm, q.w / norm
                elif path.poses:
                    pose.pose.orientation = path.poses[-1].pose.orientation
                else:
                    pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)
        detail = f"A* route ready: {len(path.poses)} poses"
        if egress_steps:
            detail += f"; safe start egress {egress_steps} cells"
            self.get_logger().warning(
                f"Start is only inside map inflation; physically clear egress {egress_steps} cells before normal A*")
        self.publish_status("READY", detail)
        self.get_logger().info(f"Published A* path with {len(path.poses)} poses")

    def _static_reference_path(self, grid, raw_grid, start, end, resolution, inflation):
        """Diagnostic only: show where dynamic cells block an otherwise safe map route."""
        try:
            if not grid.is_static_obstacle(start):
                return self.planner.plan(grid, start, end)
            if not self.get_parameter("start_egress_enabled").value:
                return []
            margin = (self.get_parameter("start_egress_clearance_margin").value
                      + resolution * sqrt(2.0) / 2.0)
            return self.planner.plan_with_start_egress(
                grid, start, end, set(), raw_grid=raw_grid, raw_blocked=set(),
                robot_radius_cells=self.get_parameter("robot_radius").value / resolution,
                margin_cells=margin / resolution,
                max_egress_cells=min(
                    int(inflation) + 1,
                    int(self.get_parameter("start_egress_max_cells").value)))
        except ValueError:
            return []

    @staticmethod
    def _dynamic_escape_halo(start, soft_blocked, raw_observed, raw_remembered,
                             grid, raw_grid, inflation, radius_cells, margin_cells):
        """Allow egress from a soft inflation halo around observed/remembered cells.

        Raw occupied returns, confirmed memory cells, and static map obstacles remain
        hard blocks. Only the bounded inflated border may be crossed on egress
        when the complete circular robot footprint clears raw occupied cells.
        """
        required_clearance = float(radius_cells) + float(margin_cells)
        if not isfinite(required_clearance) or required_clearance < 0:
            return set()
        raw_blocked = raw_observed | raw_remembered

        def footprint_clear(cell):
            x, y = cell
            if not raw_grid.contains(cell):
                return False
            if min(x + .5, raw_grid.width - x - .5,
                   y + .5, raw_grid.height - y - .5) < required_clearance:
                return False
            reach = int(required_clearance + 1.5)
            for oy in range(max(0, y - reach), min(raw_grid.height, y + reach + 1)):
                for ox in range(max(0, x - reach), min(raw_grid.width, x + reach + 1)):
                    obstacle = (ox, oy)
                    if (raw_grid.is_static_obstacle(obstacle) or obstacle in raw_blocked):
                        distance = hypot(max(abs(x - ox) - .5, 0.0),
                                         max(abs(y - oy) - .5, 0.0))
                        if distance + 1e-9 < required_clearance:
                            return False
            return True

        if (start not in soft_blocked or start in raw_observed
                or start in raw_remembered or grid.is_static_obstacle(start)
                or not footprint_clear(start)):
            return set()
        limit = max(1, int(inflation) + 1)
        escaped = {start}
        queue = deque([(start, 0)])
        while queue:
            cell, depth = queue.popleft()
            if depth >= limit:
                continue
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor in escaped or neighbor not in soft_blocked:
                    continue
                if (neighbor in raw_observed or neighbor in raw_remembered
                        or grid.is_static_obstacle(neighbor)
                        or not footprint_clear(neighbor)):
                    continue
                escaped.add(neighbor)
                queue.append((neighbor, depth + 1))
        return escaped


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
