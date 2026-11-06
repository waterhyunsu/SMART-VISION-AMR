"""A* ROS adapter: OccupancyGrid + persistent cells + PoseStamped goal -> Path."""

import json
import time
from uuid import uuid4
from math import atan2, ceil, cos, hypot, isfinite, pi, sin, sqrt

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Int32MultiArray, String
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener
from collections import deque

from smart_vision_amr.astar import AStarPlanner
from smart_vision_amr.continuous_start_egress import certified_start_only_edges
from smart_vision_amr.local_detour import plan_local_detour
from smart_vision_amr.metric_connector import (plan_connector, occupied_rectangles,
                                               polyline_clearance, point_sweep_clearance)
from smart_vision_amr.clearance_cost import clearance_penalties
from smart_vision_amr.grid import Grid
from smart_vision_amr.failure_memory import FailureMemory, EXECUTION_FAILURE_REASONS
from smart_vision_amr.obstacle_hold import ObstacleHold
from smart_vision_amr.recovery_episode import RecoveryCoordinator, RecoveryPhase
from smart_vision_amr.odometry_trail import OdometryTrail
from smart_vision_amr.map_utils import (cell_to_world, inflate_cells, inflate_cells_footprint, occupancy_to_grid,
                                        first_static_obstacle_distance, split_mapped_wall_returns, world_to_cell,
                                        yaw_from_quaternion)
from smart_vision_amr.reverse_recovery import (Breadcrumb, GridGeometry,
                                               select_reverse_egress)
from smart_vision_amr.path_geometry import (route_cells, smooth_grid_path, path_progress,
                                           point_at_progress, path_length)


def _normalized_pose_yaw(quaternion):
    """Match the follower's normalized yaw for motion/retreat evidence.

    Keep this local to measured chassis poses: map-origin conversion retains
    its existing semantics. Invalid quaternion norms fail existing finite
    checks instead of creating a fabricated zero-heading breadcrumb.
    """
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    norm = x*x+y*y+z*z+w*w
    if not all(isfinite(value) for value in (x, y, z, w, norm)) or not .8 <= norm <= 1.2:
        return float("nan")
    return atan2(2*(w*z+x*y), norm-2*(y*y+z*z))


def _point_segment_distance(x, y, start, end):
    """Continuous point-to-segment distance, without raster/tuple equality."""
    dx, dy = end[0]-start[0], end[1]-start[1]
    squared = dx*dx+dy*dy
    fraction = (max(0., min(1., ((x-start[0])*dx+(y-start[1])*dy)/squared))
                if squared else 0.)
    return hypot(x-start[0]-fraction*dx, y-start[1]-fraction*dy)


LOCAL_DETOUR_DEFAULTS = {
    "local_detour_enabled": True, "local_detour_trigger_distance": 1.5,
    "local_detour_max_rejoin_distance": 2.5, "local_detour_max_length": 4.0,
    "local_detour_max_lateral_distance": .9, "local_detour_max_expansions": 1500,
    "local_detour_max_planning_time": .06, "local_detour_max_candidates": 10,
    "local_detour_max_attempts": 3, "local_detour_clearance_margin": .02,
    "local_detour_occlusion_extension": .5, "local_detour_rejoin_buffer": .15,
    "local_detour_observation_margin": .05,
}


FAILURE_DEFAULTS = {
    **LOCAL_DETOUR_DEFAULTS,
    "failure_memory_enabled": True, "failure_memory_ttl": 120.0,
    "failure_memory_radius": .25, "failure_memory_heading_tolerance": pi/3,
    "failure_memory_max_entries": 64, "failure_memory_penalty": 2.0,
    "failure_event_max_age": 3.0, "rejoin_progress_distance": .35,
    "rejoin_cross_track_tolerance": .20,
    "obstacle_hold_enabled": True, "obstacle_retry_limit": 3,
    "controller_retry_limit": 3,
    "obstacle_clear_duration": 2.0,
    "reverse_clearance_neutral_egress_enabled": False,
}


class AStarPlannerNode(Node):
    def __init__(self):
        super().__init__("astar_planner")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("occupied_threshold", 65)
        self.declare_parameter("obstacle_inflation_cells", 2)
        self.declare_parameter("mapped_wall_return_tolerance_cells", 0)
        self.declare_parameter("replan_retry_period", 2.0)
        self.declare_parameter("robot_radius", 0.22)
        self.declare_parameter("start_egress_enabled", True)
        self.declare_parameter("start_egress_clearance_margin", 0.02)
        self.declare_parameter("start_egress_max_cells", 7)
        self.declare_parameter("live_obstacle_layer_required", False)
        self.declare_parameter("live_obstacle_timeout", 2.0)
        self.declare_parameter("live_route_confirmation_scans", 1)
        self.declare_parameter("preferred_clearance", 0.45)
        self.declare_parameter("clearance_penalty_gain", 1.5)
        self.declare_parameter("path_smoothing_enabled", True)
        self.declare_parameter("path_smoothing_iterations", 2)
        self.declare_parameter("path_smoothing_shortcut_distance", 0.80)
        self.declare_parameter("path_resample_spacing", 0.05)
        self.declare_parameter("reverse_recovery_enabled", True)
        self.declare_parameter("reverse_recovery_min_distance", 0.10)
        self.declare_parameter("reverse_recovery_max_distance", 0.30)
        self.declare_parameter("reverse_recovery_clearance_margin", 0.02)
        self.declare_parameter("reverse_recovery_speed", 0.04)
        self.declare_parameter("reverse_recovery_max_attempts", 3)
        for name, value in FAILURE_DEFAULTS.items():
            self.declare_parameter(name, value)
        self.map = None
        self.raw_static_grid = None
        self.map_signature = None
        self.remembered = set()
        # The latched memory overlay may arrive before the latched map. Keep
        # its latest raw indices until map dimensions are known.
        self.pending_memory_indices = None
        self.memory_map_key = None
        self.observed = set()
        self.last_observation_time = None
        self.pending_live_route_cells = set()
        self.pending_live_route_count = 0
        self.pending_live_route_at = None
        self.pending_live_route_last_at = None
        self.last_goal = None
        self.planned_cells = []
        self.waiting_for_transform = False
        self.waiting_for_replan = False
        self.navigation_suspended = False
        self.goal_completed = False
        self.last_plan_time = float("-inf")
        self.invalid_path_published = False
        self.geometry_signature = None
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
        self._init_failure_state()
        self.planner = AStarPlanner()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.path_pub = self.create_publisher(Path, "/smart_vision_amr/path", 10)
        # /path and /planner_status are different ROS topics. The exact path
        # stamp lets consumers correlate them regardless of arrival order.
        self.last_path_stamp_ns = -1
        self.current_path_stamp_ns = None
        self.status_pub = self.create_publisher(String, "/smart_vision_amr/planner_status", latched)
        self.recovery_pub = self.create_publisher(String, "/smart_vision_amr/recovery_request", 10)
        self.failure_status_pub = self.create_publisher(
            String, "/smart_vision_amr/failure_memory_status", latched)
        self.create_subscription(OccupancyGrid, self.get_parameter("map_topic").value, self.on_map, latched)
        self.create_subscription(Int32MultiArray, "/smart_vision_amr/remembered_cells", self.on_memory, latched)
        self.create_subscription(Int32MultiArray, "/smart_vision_amr/observed_obstacles",
                                 self.on_observed, 10)
        self.create_subscription(PoseStamped, "/smart_vision_amr/goal", self.on_goal, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(String, "/smart_vision_amr/navigation_status",
                                 self.on_navigation_status, 10)
        self.create_subscription(String, "/smart_vision_amr/execution_failure",
                                 self.on_execution_failure, 10)
        self.create_subscription(String, "/smart_vision_amr/localization_health",
                                 self.on_localization_health, latched)
        self.create_subscription(String, "/smart_vision_amr/motion_status",
                                 self.on_motion_status, latched)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, "/smart_vision_amr/observed_visibility",
                                 self.on_visibility, qos_profile_sensor_data)
        self.create_timer(0.25, self.retry_transform)
        self.create_timer(.5, self.publish_failure_status,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.publish_status("IDLE", "Waiting for an explicit goal")

    def on_map(self, msg):
        # A previous trail is only meaningful in the exact saved map in which
        # its map-frame poses were measured. Repeated publications of that map
        # are harmless; a reload, new SLAM content, or source-frame change is not.
        info = msg.info
        origin = info.origin
        # Memory indices refer to a particular map's cells. Match the memory
        # producer's map identity (geometry + content, excluding load time),
        # so reloading identical content does not discard valid memory.
        memory_map_key = (msg.header.frame_id, info.width, info.height,
                          info.resolution, origin.position.x, origin.position.y,
                          origin.position.z, origin.orientation.x,
                          origin.orientation.y, origin.orientation.z,
                          origin.orientation.w, tuple(msg.data))
        map_key = (memory_map_key[:4]
                   + (info.map_load_time.sec, info.map_load_time.nanosec)
                   + memory_map_key[4:])
        previous_memory_map_key = getattr(self, "memory_map_key", None)
        memory_map_replaced = (previous_memory_map_key is not None
                               and memory_map_key != previous_memory_map_key)
        previous_map_key = self.breadcrumb_map_key
        map_replaced = previous_map_key is not None and map_key != previous_map_key
        if map_replaced:
            self._clear_breadcrumbs()
            self._reset_local_reference()
            self.local_grid_cache = None
        self.breadcrumb_map_key = map_key
        threshold = self.get_parameter("occupied_threshold").value
        inflation = self.get_parameter("obstacle_inflation_cells").value
        signature = (threshold, inflation,
                     self.get_parameter("mapped_wall_return_tolerance_cells").value)
        unchanged_map = (self.map is not None and previous_map_key == map_key
                and getattr(self, "map_signature", None) == signature
                and self.raw_static_grid is not None)
        if unchanged_map:
            # Identical latched-map republishes preserve a pending stop/probe
            # certificate's semantic map identity. Real reload/content/frame
            # or parameter changes still rebuild and revoke the old evidence.
            self.map = (msg, self.map[1])
        else:
            self.map = (msg, occupancy_to_grid(msg, threshold, inflation))
            self.raw_static_grid = occupancy_to_grid(msg, threshold, 0)
            self._clear_start_scan_history()
        self.map_signature = signature
        self.memory_map_key = memory_map_key
        if memory_map_replaced:
            # The memory publisher resets on a new saved map. Never interpret
            # old linear indices in the replacement map while its new overlay
            # is still in flight.
            self.remembered.clear()
            self.pending_memory_indices = None
        elif previous_memory_map_key is None:
            pending = getattr(self, "pending_memory_indices", None)
            self.pending_memory_indices = None
            if pending is not None:
                self.remembered = self._memory_cells(pending)
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
                        "episode_id": self.active_recovery_episode_id,
                        "reason": "Map changed during reverse recovery"})
                    self.recovery_pub.publish(cancel)
                    self.invalid_path_published = False
                    self._invalidate_path("Map changed during reverse recovery; canceling reverse")
                return
            if any(self.map[1].is_static_obstacle(cell) for cell in self._remaining_path_cells()):
                self._invalidate_path("Updated map blocks the active route")
            elif unchanged_map and self.planned_world_points and not self.waiting_for_replan:
                # A latched identical map is not a new obstacle or a failed
                # local repair. Keep following its already certified suffix.
                return
            self._request_replan("Map updated")

    def _goal_is_active(self):
        return (self.last_goal is not None and not self.navigation_suspended
                and not self.goal_completed)

    def _init_failure_state(self):
        self.active_start_certificate = None
        self.completed_start_prefix = None
        self.failure_memory = FailureMemory()
        self.failure_state, self.failure_reason = "CLEAR", ""
        self.controller_retry_events = set()
        self.controller_pending_retry_events = set()
        # Actual attempted world segments accompany accepted terminal events.
        # FailureMemory's directional radius is a routing preference, not a
        # physical swept footprint. Never reconstruct a larger obstacle from it.
        self.execution_failure_segments = {}
        self.last_failure_id = None
        self.last_failure_event_stamp_ns = 0
        self.seen_failure_ids = deque(maxlen=256)
        self.pending_failure_id = None
        self.pending_failure_since = None
        self.localization_health = self.latest_health_at = None
        self.latest_scan = self.latest_scan_at = None
        self.planned_world_points = []
        self.rejoin_anchor = None
        self.rejoin_verified = False
        self.navigation_progress_at = None
        self.obstacle_hold = ObstacleHold()
        # A blocked published path is canceled immediately, but only an
        # actual no-route A* result may spend this physical-obstacle budget.
        self.last_no_route_obstacle_charge_at = float("-inf")
        self.observed_revision = 0
        self._clear_pending_live_route()
        self.last_no_route_observed_revision = 0
        self.obstacle_episode_failure_ids = set()
        self.obstacle_failure_cells = {}
        self.last_visibility_at = None
        self.last_visibility_stamp_ns = 0
        self.latest_motion_status = self.latest_motion_status_at = None
        self.local_reference_points = []
        self.local_reference_key = None
        self.local_grid_cache = None
        self.local_cell_penalty = {}
        self.local_detour_attempts = 0
        self.local_attempt_id = 0
        self.global_plan_count = 0
        self.local_rejoin_anchor = None
        self.pending_reverse_probe = None
        self.recovery_coordinator = RecoveryCoordinator()
        self.odometry_trail = OdometryTrail()
        self.projected_odom_trail_verified = False
        self.active_recovery_episode_id = None
        self.last_invalidated_path_stamp_ns = None
        self.wait_moving_cells = set()
        self.wait_moving_count = 0
        self.wait_moving_at = self.wait_moving_last_at = None
        self.plan_metrics = {"plan_kind": "NONE", "local_detour_state": "NONE",
                             "local_attempt_id": 0, "global_plan_count": 0}

    def on_motion_status(self, msg):
        try:
            value = json.loads(msg.data)
            if isinstance(value, dict):
                self.latest_motion_status, self.latest_motion_status_at = value, time.monotonic()
                if not self._automatic_resume_allowed():
                    self._clear_start_certificate()
                if (value.get("mode") in {"manual", "localization", "hold"}
                        or value.get("estop_latched") is True or value.get("reset_pending") is True):
                    self.pending_reverse_probe = None
                    self.recovery_coordinator.reset()
        except (ValueError, TypeError):
            pass

    def _reset_local_reference(self):
        self.pending_reverse_probe = None
        self.metric_route_scope = self.metric_reverse_budget = self.metric_no_path_key = None
        self.recovery_coordinator.reset()
        self.wait_moving_cells = set()
        self.wait_moving_count = 0
        self.wait_moving_at = self.wait_moving_last_at = None
        self._clear_start_certificate()
        self.start_scan_context_invalid = self.start_prefix_input_pending = False
        self.local_reference_points = []
        self.local_reference_key = None
        self.local_detour_attempts = 0
        self.local_rejoin_anchor = None
        self._clear_pending_live_route()
        self.plan_metrics.update(local_detour_state="NONE")

    def _clear_pending_live_route(self):
        self.pending_live_route_cells = set()
        self.pending_live_route_count = 0
        self.pending_live_route_at = None
        self.pending_live_route_last_at = None

    def _live_route_block_confirmed(self, cells):
        """Confirm a distant scan hit before canceling the entire FollowPath.

        Raw scans still reach Nav2's local costmap and the robot-side near-scan
        guard on every frame. Close hits and missing pose evidence cancel now.
        """
        if not cells:
            self._clear_pending_live_route()
            return False
        required = self._live_route_required_scans()
        if required <= 1:
            self._clear_pending_live_route()
            return True
        pose = self._fresh_motion_pose()
        if pose is None:
            self._clear_pending_live_route()
            return True
        radius = float(self.get_parameter("robot_radius").value)
        # 0.18 m covers the short reaction/braking interval at the configured
        # 0.24 m/s speed ceiling. Within this distance there is no debounce.
        if not isfinite(radius) or radius <= 0:
            self._clear_pending_live_route()
            return True
        for cell in cells:
            x, y = cell_to_world(self.map[0].info, cell)
            if hypot(x-pose[0], y-pose[1]) <= radius+.18:
                self._clear_pending_live_route()
                return True
        now = time.monotonic()
        previous = self.pending_live_route_cells
        consistent = (previous and self.pending_live_route_last_at is not None
                      and 0 <= now-self.pending_live_route_last_at <= .5
                      and any(abs(x-px) <= 1 and abs(y-py) <= 1
                              for x, y in cells for px, py in previous))
        if not consistent:
            self.pending_live_route_at = now
            self.pending_live_route_count = 0
        self.pending_live_route_cells = set(cells)
        self.pending_live_route_last_at = now
        self.pending_live_route_count += 1
        if (self.pending_live_route_count >= min(required, 10)
                and now-self.pending_live_route_at >= .15):
            self._clear_pending_live_route()
            return True
        return False

    def _live_route_required_scans(self):
        try:
            return max(1, min(int(self.get_parameter("live_route_confirmation_scans").value), 10))
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return 1

    def _react_to_live_route_blockers(self, blockers):
        self.get_logger().warning(
            f"Fresh observed cells persist on remaining route: cells={sorted(blockers)[:16]} "
            f"observed_count={len(self.observed)} path_cells={len(self.planned_cells)}")
        if self._start_physical_recovery(blockers, "Observed obstacle intersects the route"):
            return
        self._invalidate_path("Observed obstacle intersects the route")
        self._request_replan("Observed obstacles changed")

    def _start_physical_recovery(self, blockers, reason, *, terminal_repair=False, raw_hits=(),
                                 resumed=False):
        if (not blockers or self.map is None or not self._goal_is_active()
                or self.recovery_active
                or (not self.planned_world_points and not terminal_repair and not resumed)):
            return False
        coordinator = self.recovery_coordinator
        episode = coordinator.current
        if not resumed:
            if episode is not None and episode.phase is not RecoveryPhase.FOLLOW:
                return True
            if not coordinator.begin(uuid4().hex, self._local_route_key(), blockers,
                                     self.observed, self.remembered):
                return False
        # A new physical cycle gets a bounded local search, not a preemptive
        # hold inherited from another approach's retry counter.
        self.obstacle_hold.reset(repr(self._memory_scopes()))
        info = self.map[0].info
        neighborhood = inflate_cells(set(blockers), info.width, info.height,
                                     max(1, ceil(.15/info.resolution)))
        self.obstacle_episode_failure_ids = {
            key for key, cells in self.obstacle_failure_cells.items() if cells & neighborhood}
        self.failure_state, self.failure_reason = "CLEAR", reason
        self.plan_metrics.update(failure_class="obstacle", recovery_stage="local_detour")
        if resumed:
            if self._reuse_clear_forward_suffix():
                return True
            if getattr(self, "clear_suffix_inputs_pending", False):
                self._hold_start_prefix_inputs()
                return True
        if self._try_local_detour(blockers, terminal_repair=terminal_repair or resumed, raw_hits=raw_hits):
            if coordinator.current.phase is RecoveryPhase.LOCAL:
                coordinator.local_result(True)
            coordinator.current.path_stamp_ns = self.current_path_stamp_ns
            return True
        if getattr(self, "metric_connector_input_pending", False):
            # Retain LOCAL ownership while the same paired input is pending.
            # A missing acquisition TF is not a physical rear-probe verdict.
            return True
        coordinator.local_result(False)
        return self._queue_reverse_before_global(blockers, reason,
            terminal_repair=terminal_repair or resumed, raw_hits=raw_hits)

    def _reuse_clear_forward_suffix(self):
        """Positive removal may restore the same measured-safe forward route."""
        self.clear_suffix_inputs_pending = False
        if (len(self.local_reference_points) < 2 or self.local_reference_key != self._local_route_key()
                or not self._observations_ready() or not self._automatic_resume_allowed()):
            return False
        pose = self._fresh_motion_pose()
        if pose is None:
            return False
        reference = self.local_reference_points
        progress = path_progress(reference, pose[:2])
        points = [pose[:2], point_at_progress(reference, progress)]
        traveled = 0.
        for a, b in zip(reference, reference[1:]):
            traveled += hypot(b[0]-a[0], b[1]-a[1])
            if traveled > progress+1e-9:
                points.append(b)
        points = [p for index, p in enumerate(points) if not index or hypot(p[0]-points[index-1][0],p[1]-points[index-1][1])>1e-8]
        if len(points) < 2:
            return False
        info, grid = self.map[0].info, self.map[1]
        yaw = yaw_from_quaternion(info.origin.orientation)
        c, s = cos(yaw), sin(yaw)
        local = [((c*(x-info.origin.position.x)+s*(y-info.origin.position.y))/info.resolution,
                  (-s*(x-info.origin.position.x)+c*(y-info.origin.position.y))/info.resolution) for x,y in points]
        cells = route_cells(local)
        blocked = (self._live_obstacle_blocked(self.observed)
                   | self._live_obstacle_blocked(self.remembered))
        if any(grid.is_static_obstacle(cell) or cell in blocked for cell in cells):
            return False
        # The delivered suffix may turn within its first 11 cm. Prove those
        # actual segments; a shortcut chord is neither its body sweep nor a
        # safe substitute for the ordered route.
        prefix, traveled = [points[0]], 0.
        for a, b in zip(points, points[1:]):
            traveled += hypot(b[0]-a[0], b[1]-a[1])
            if traveled >= .11:
                prefix.append(point_at_progress(points, .11))
                break
            prefix.append(b)
        if not self._prefix_sweep_clear(prefix):
            self.clear_suffix_inputs_pending = self.plan_metrics.get(
                "start_prefix_validation", {}).get("input_pending") is True
            return False
        path = Path()
        path.header.frame_id = self.get_parameter("map_frame").value
        path.header.stamp = self.get_clock().now().to_msg()
        for index, point in enumerate(points):
            item=PoseStamped();item.header=path.header
            item.pose.position.x,item.pose.position.y=point
            if index+1<len(points):
                angle=atan2(points[index+1][1]-point[1],points[index+1][0]-point[0])
                item.pose.orientation.z,item.pose.orientation.w=sin(angle/2),cos(angle/2)
            else:
                item.pose.orientation=self.last_goal.pose.orientation
            path.poses.append(item)
        self.planned_cells=list(cells);self.planned_world_points=points
        self.waiting_for_replan=False;self.invalid_path_published=False
        self.plan_metrics.update(plan_kind="LOCAL_DETOUR",local_detour_state="REJOINED",
                                 detour_reason="verified_clear_suffix",failure_class="none",recovery_stage="tracking")
        self._publish_path(path)
        self.publish_status("READY","Measured removal restored the existing safe forward route")
        return True

    def _resume_recovery_wait(self, *, positive_free=False, moving_obstacle_confirmed=False):
        coordinator=self.recovery_coordinator
        if not coordinator.resume_wait(uuid4().hex,observed=self.observed,remembered=self.remembered,
                positive_free=positive_free,moving_obstacle_confirmed=moving_obstacle_confirmed):
            return False
        return self._start_physical_recovery(coordinator.current.blockers,
            "Fresh physical evidence changed the blocked route",resumed=True)

    def _wait_after_recovery_cycle(self, *, relevant_cells=()):
        coordinator=self.recovery_coordinator
        if coordinator.current is None or coordinator.current.phase is not RecoveryPhase.GLOBAL:
            return False
        # Remember the actual hard constraints on this failed start/suffix,
        # not just the object which originally opened the episode. The memory
        # callback may confirm their removal after visibility already removed
        # the original trigger cell. Repeated/unchanged memory cannot wake.
        info = self.map[0].info
        route = set(relevant_cells)
        pose = self._fresh_motion_pose()
        if pose is not None:
            route.add(world_to_cell(info, *pose[:2]))
        if not route and self.local_reference_points:
            angle = yaw_from_quaternion(info.origin.orientation)
            c, s = cos(angle), sin(angle)
            points = [((c*(x-info.origin.position.x)+s*(y-info.origin.position.y))/info.resolution,
                       (-s*(x-info.origin.position.x)+c*(y-info.origin.position.y))/info.resolution)
                      for x,y in self.local_reference_points]
            route.update(route_cells(points))
        relevance = (set(self.remembered) & inflate_cells(route, info.width, info.height,
            int(self.get_parameter("obstacle_inflation_cells").value)))
        coordinator.global_result(False,observed=self.observed,remembered=self.remembered,
                                  relevant_remembered=relevance)
        self.wait_moving_cells = set()
        self.wait_moving_count = 0
        self.wait_moving_at = self.wait_moving_last_at = None
        episode=coordinator.current
        self.obstacle_hold.ensure_goal_scope(repr(self._memory_scopes()))
        self.obstacle_hold.register_failure(episode.blockers,now=time.monotonic())
        self.obstacle_hold.holding=True  # One complete cycle, never timer-only repeated global attempts.
        self.failure_state,self.failure_reason="OBSTACLE_HOLD","장애물로 주행 대기 · 실제 환경 변화 또는 명시적 재개를 기다립니다."
        self.waiting_for_replan=True
        self.plan_metrics.update(failure_class="obstacle",recovery_stage="await_clearance")
        self.publish_status("WAIT_REPLAN",self.failure_reason)
        return True

    def _confirmed_wait_change(self, current_hits, now):
        """Three new stamped scans must show a nearby, materially moved object.

        TTL overlay disappearance and one-cell endpoint jitter never count.
        This only grants another bounded recovery cycle, never drive clearance.
        """
        episode = self.recovery_coordinator.current
        if episode is None or episode.phase is not RecoveryPhase.WAIT_OBSTACLE:
            return False
        info = self.map[0].info
        old_band = inflate_cells(episode.observed | episode.remembered,
                                info.width, info.height, 1)
        reach = max(2, ceil(.8/info.resolution))
        changed = {cell for cell in set(current_hits)-old_band
                   if self.raw_static_grid.contains(cell)
                   and not self.raw_static_grid.is_static_obstacle(cell)
                   and any(hypot(cell[0]-old[0], cell[1]-old[1]) <= reach
                           for old in episode.blockers)}
        if not changed:
            self.wait_moving_cells = set()
            self.wait_moving_count = 0
            self.wait_moving_at = self.wait_moving_last_at = None
            return False
        consistent = (self.wait_moving_cells and self.wait_moving_last_at is not None
            and 0 <= now-self.wait_moving_last_at <= .5
            and any(abs(a[0]-b[0]) <= 1 and abs(a[1]-b[1]) <= 1
                    for a in changed for b in self.wait_moving_cells))
        if not consistent:
            self.wait_moving_count, self.wait_moving_at = 0, now
        self.wait_moving_cells, self.wait_moving_last_at = changed, now
        self.wait_moving_count += 1
        return self.wait_moving_count >= 3 and now-self.wait_moving_at >= .15

    def _recovery_wait_owns_retry(self):
        episode=self.recovery_coordinator.current
        if episode is None or episode.phase is not RecoveryPhase.WAIT_OBSTACLE:
            return False
        if episode.scope != self._local_route_key() or not self._goal_is_active():
            self.recovery_coordinator.reset()
            return False
        # A fresh timestamp/unchanged overlay/TTL expiration cannot reopen
        # a completed failed cycle. Callbacks supply positive wake evidence.
        return True

    @staticmethod
    def _payload_stamp_ns(payload, name="path_stamp"):
        stamp=payload.get(name)
        if not isinstance(stamp,dict):
            return None
        sec,nsec=stamp.get("sec"),stamp.get("nanosec")
        if type(sec) is not int or type(nsec) is not int or sec<0 or not 0<=nsec<1_000_000_000:
            return None
        return sec*1_000_000_000+nsec

    def _status_owns_path(self,payload,*,allow_invalidated=False):
        stamp=self._payload_stamp_ns(payload)
        expected=(self.current_path_stamp_ns if self.planned_world_points
                  else self.last_invalidated_path_stamp_ns if allow_invalidated else None)
        return type(stamp) is int and stamp>0 and stamp==expected

    def _queue_reverse_before_global(self, blockers, reason, *, terminal_repair=False, raw_hits=()):
        """Cancel an actually attempted blocked route before testing retreat.

        A new explicit goal still starts with ordinary A*. For an obstacle on
        an active route, global replacement waits for the existing stop and
        independently verified reverse gates; NOT_STOPPED is not a rejection
        that silently skips the reverse stage.
        """
        if (not blockers or self.map is None or not self._goal_is_active()
                or self.recovery_active or self.obstacle_hold.holding
                or self.failure_state == "CONTROLLER_HOLD"
                or (not self.planned_world_points and not terminal_repair)):
            return False
        self.pending_reverse_probe = {
            "key": self._local_route_key(), "raw_grid": self.raw_static_grid,
            "geometry": self._start_certificate_geometry(),
            "reference": tuple(self.local_reference_points),
            "blocking_cells": set(blockers), "raw_hits": set(raw_hits),
            "reason": reason, "created_at": time.monotonic()}
        self.failure_state, self.failure_reason = "WAIT_REVERSE", "Stopping before a verified reverse probe"
        self.plan_metrics.update(failure_class="obstacle", recovery_stage="await_reverse_stop")
        self._invalidate_path(reason)
        self._process_pending_reverse()
        return True

    def _process_pending_reverse(self):
        """Return true while this retained-goal stop/reverse probe owns retry."""
        pending = getattr(self, "pending_reverse_probe", None)
        if pending is None:
            return False
        if (not self._goal_is_active() or self.map is None
                or self.failure_state == "CONTROLLER_HOLD" or self.obstacle_hold.holding
                or pending["key"] != self._local_route_key()
                or pending["raw_grid"] is not self.raw_static_grid
                or pending["geometry"] != self._start_certificate_geometry()
                or pending["reference"] != tuple(self.local_reference_points)
                or time.monotonic() < pending["created_at"]):
            self.pending_reverse_probe = None
            self.recovery_coordinator.reset()
            return False
        if (not self._observations_ready() or not self._automatic_resume_allowed()
                or self._fresh_motion_pose() is None):
            self.failure_state, self.failure_reason = "SENSOR_HOLD", "Fresh motion evidence required before reverse probe"
            self.plan_metrics.update(failure_class="sensor", recovery_stage="await_reverse_inputs")
            self.publish_status("WAIT_REPLAN", self.failure_reason)
            return True
        if not self._reverse_start_stopped():
            self.failure_state, self.failure_reason = "WAIT_REVERSE", "Waiting for measured chassis stop before reverse probe"
            self.plan_metrics.update(failure_class="obstacle", recovery_stage="await_reverse_stop")
            self.publish_status("WAIT_REPLAN", self.failure_reason)
            return True

        message, grid = self.map
        pose = self._fresh_motion_pose()
        if pose is None:
            return True
        start = world_to_cell(message.info, *pose[:2])
        goal = self.last_goal
        end = world_to_cell(message.info, goal.pose.position.x, goal.pose.position.y)
        observed = set(self.observed) | pending["raw_hits"]
        remembered = set(self.remembered)
        reserved_observed = self._global_occlusion_reserve(start, observed)
        reserved_remembered = self._global_occlusion_reserve(start, remembered, remembered=True)
        blocked = (self._live_obstacle_blocked(reserved_observed)
                   | self._live_obstacle_blocked(reserved_remembered))
        costs = clearance_penalties(self.raw_static_grid, reserved_observed | reserved_remembered,
            message.info.resolution, float(self.get_parameter("preferred_clearance").value),
            float(self.get_parameter("clearance_penalty_gain").value))
        trail_verified = self._record_breadcrumb()
        self.failure_state, self.failure_reason = "WAIT_REVERSE", "Checking the recent rear trail before global replanning"
        self.plan_metrics.update(failure_class="obstacle", recovery_stage="reverse_probe")
        self.recovery_coordinator.stop_proved()
        if self._maybe_request_reverse(start, end, message, grid, self.raw_static_grid,
                observed | remembered, blocked, costs, trail_verified,
                blocking_evidence=pending["blocking_cells"]):
            self.recovery_coordinator.reverse_verdict(True)
            self.pending_reverse_probe = None
            return True
        rejection = getattr(self, "last_reverse_rejection_code", "NO_SAFE_EGRESS")
        if rejection in {"NOT_STOPPED", "POSE_UNREADY", "SENSORS_UNREADY", "AUTONOMY_UNAVAILABLE",
                         "CERTIFICATE_EXPIRED", "ODOM_CHILD_FRAME_MISSING"}:
            # Cancellation/inputs can still be settling. Never turn a missing
            # stop proof into permission to send another forward action.
            self.plan_metrics.update(recovery_stage="await_reverse_stop" if rejection == "NOT_STOPPED"
                                     else "await_reverse_inputs")
            self.publish_status("WAIT_REPLAN", "Reverse probe waiting: "+rejection)
            return True
        self.pending_reverse_probe = None
        self.recovery_coordinator.reverse_verdict(False)
        budget = getattr(self, "metric_reverse_budget", None)
        self.metric_reverse_budget = None
        episode = self.recovery_coordinator.current
        if (budget is not None and budget["key"] == self._local_route_key()
                and episode is not None and budget.get("episode_id") == episode.episode_id):
            config = self._local_configuration()
            if (config is not None and config["enabled"]
                    and self._try_metric_connector(config, max_seconds=budget["seconds"],
                        max_expansions=budget["expansions"], end=end, after_reverse=True,
                        raw_hits=pending["raw_hits"])):
                return True
            if getattr(self, "metric_connector_input_pending", False):
                return True
            self.plan_metrics["metric_side_probe_result"] = {name: self.plan_metrics.get(name) for name in
                ("metric_connector_reason", "metric_connector_duration_ms", "metric_connector_expansions")}
        self.failure_state, self.failure_reason = "WAIT_ALTERNATIVE", "No verified rear exit; replanning from the stopped pose"
        self.plan_metrics.update(failure_class="obstacle", recovery_stage="global_replan",
                                 reverse_probe_rejection=rejection)
        self.last_plan_time = float("-inf")
        self.plan_to(self.last_goal)
        return True

    def _live_route_blockers(self, observed):
        """Find route-blocking scan cells with one TF lookup per scan frame."""
        if not observed or not self.planned_cells:
            return set()
        metric = self._metric_route_blockers(observed)
        if metric is not None:
            return metric
        remaining = set(self._remaining_path_cells())
        remaining -= self._validated_start_prefix_exemption(remaining, observed=observed)
        if getattr(self, "start_prefix_input_pending", False):
            self._hold_start_prefix_inputs()
            return set()
        blocked_route = remaining & self._live_obstacle_blocked(observed)
        if not blocked_route:
            return set()
        info = self.map[0].info
        try:
            radius = (float(self.get_parameter("robot_radius").value)
                      + float(self.get_parameter("local_detour_clearance_margin").value))
            reach = max(int(self.get_parameter("obstacle_inflation_cells").value),
                        ceil(radius / info.resolution + 1.))
            if not isfinite(radius) or radius <= 0 or info.resolution <= 0:
                raise ValueError("Invalid live inflation radius")
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
            # Invalid parameters already fail closed in _live_obstacle_blocked.
            # Do not drop a hit while identifying its source cell.
            return set(observed)
        candidates = observed & inflate_cells(blocked_route, info.width, info.height, reach)
        return {cell for cell in candidates
                if self._live_obstacle_blocked({cell}) & blocked_route}

    def _metric_route_blockers(self, cells):
        """Validate a metric route's actual remaining sweep, not crossed squares.

        The route contains no exempt obstacle cells. A newly measured square
        touching any remaining circle sweep, including the current body, is
        still a blocker. Refining robot centres must not make a whole 5cm
        candidate square the execution geometry again.
        """
        scope = getattr(self, "metric_route_scope", None)
        if scope is None:
            return None
        if (scope != (self._local_route_key(), self.raw_static_grid,
                      self.current_path_stamp_ns, self._start_certificate_geometry())
                or not self._goal_is_active() or not self._automatic_resume_allowed()):
            self.metric_route_scope = None
            return None
        current = self._fresh_motion_pose()
        if current is None or not self._observations_ready():
            self._hold_start_prefix_inputs()
            return set()
        points = self.planned_world_points
        progress = path_progress(points, current[:2])
        remaining = [current[:2], point_at_progress(points, progress)]
        traveled = 0.
        for a, b in zip(points, points[1:]):
            traveled += hypot(b[0]-a[0], b[1]-a[1])
            if traveled > progress+1e-9:
                remaining.append(b)
        if len(remaining) < 2:
            return set()
        info = self.map[0].info
        angle = yaw_from_quaternion(info.origin.orientation)
        c, s = cos(angle), sin(angle)
        local = [((c*(x-info.origin.position.x)+s*(y-info.origin.position.y))/info.resolution,
                  (-s*(x-info.origin.position.x)+c*(y-info.origin.position.y))/info.resolution)
                 for x, y in remaining]
        radius = (float(self.get_parameter("robot_radius").value)
                  + float(self.get_parameter("local_detour_clearance_margin").value))/info.resolution
        # Every currently supplied observed/memory square stays intact.
        blockers = {cell for cell in cells if polyline_clearance(local,
            [(cell[0], cell[1], cell[0]+1., cell[1]+1.)],
            self.raw_static_grid.width, self.raw_static_grid.height)+1e-9 < radius}
        if blockers:
            return blockers
        # Original raw rays and acquisition-time TF remain independently hard.
        short, traveled = [remaining[0]], 0.
        for a,b in zip(remaining,remaining[1:]):
            traveled += hypot(b[0]-a[0],b[1]-a[1])
            if traveled >= .11:
                short.append(point_at_progress(remaining,.11))
                break
            short.append(b)
        if not self._prefix_sweep_clear(short, observed=cells):
            if self.plan_metrics.get("start_prefix_validation", {}).get("input_pending"):
                self._hold_start_prefix_inputs()
                return set()
            # Do not invent an observed object for a static/raw-only failure.
            self._invalidate_path("Current raw body or forward sweep blocks metric route")
            return set()
        return set()

    def _try_metric_connector(self, config, *, max_seconds, max_expansions,
                              end=None, after_reverse=False, raw_hits=()):
        """Bounded body-safe departure/rejoin after coarse quantization fails."""
        began = time.perf_counter()
        self.metric_connector_input_pending = False
        self.plan_metrics.update(metric_connector_reason="input_unavailable",
                                 metric_connector_expansions=0, metric_connector_duration_ms=0.)
        if max_seconds <= 0 or max_expansions < 1:
            self.plan_metrics["metric_connector_reason"] = "time_budget" if max_seconds <= 0 else "expansion_budget"
            return False
        if (not self._goal_is_active() or self.failure_state == "CONTROLLER_HOLD"
                or self.obstacle_hold.holding or self.pending_failure_id is not None or self.recovery_active
                or not self._automatic_resume_allowed() or not self._observations_ready()
                or self.raw_static_grid is None):
            return False
        current = self._fresh_motion_pose()
        if current is None:
            return False
        info, grid = self.map[0].info, self.map[1]
        raw, overlay, remembered = self.raw_static_grid, set(self.observed), set(self.remembered)
        observed = overlay | set(raw_hits)
        key = self._local_route_key()
        angle = yaw_from_quaternion(info.origin.orientation)
        c, s = cos(angle), sin(angle)
        def local(point):
            dx, dy = point[0]-info.origin.position.x, point[1]-info.origin.position.y
            return ((c*dx+s*dy)/info.resolution, (-s*dx+c*dy)/info.resolution)
        def world(point):
            return cell_to_world(info, (point[0]-.5, point[1]-.5))
        body = float(self.get_parameter("robot_radius").value)
        start_margin = float(self.get_parameter("start_egress_clearance_margin").value)
        if not isfinite(body) or body <= 0 or not isfinite(start_margin) or start_margin < 0:
            self.plan_metrics["metric_connector_reason"] = "invalid_configuration"
            return False
        actual_clearance = polyline_clearance([local(current[:2])]*2,
            occupied_rectangles(raw, observed | remembered), raw.width, raw.height)*info.resolution
        if actual_clearance+1e-9 < body:
            self.plan_metrics["metric_connector_reason"] = "actual_body_intersection"
            return False
        reference = self.local_reference_points
        tails = []
        retained_indices = []
        candidates_used = []
        measured = self._live_obstacle_blocked(observed | remembered)
        def side_opening(point):
            heading = atan2(point[1]-current[1], point[0]-current[0])-current[2]
            error = abs(atan2(sin(heading), cos(heading)))
            return not after_reverse or pi/4 <= error <= 3*pi/4
        if len(reference) >= 2 and self.local_reference_key == key:
            progress, traveled = path_progress(reference, current[:2]), 0.
            for index, (a, b) in enumerate(zip(reference, reference[1:]), 1):
                traveled += hypot(b[0]-a[0], b[1]-a[1])
                if not progress+.30 <= traveled <= progress+min(1.20, config["max_rejoin_distance"]):
                    continue
                cell = world_to_cell(info, *b)
                if not grid.is_static_obstacle(cell) and cell not in measured and side_opening(b):
                    tails.append([local(p) for p in reference[index:]])
                    retained_indices.append(index)
                    candidates_used.append(b)
                if time.perf_counter()-began > max_seconds*.35:
                    break
                if len(tails) >= config["max_candidates"]:
                    break
        # A static-start failure may have no usable retained suffix. Join only
        # a normal coarse-free anchor; all later nodes obey the original grid.
        if not tails and end is not None:
            start = world_to_cell(info, *current[:2])
            candidates = []
            reach = min(12, ceil(.60/info.resolution))
            for dy in range(-reach, reach+1):
                for dx in range(-reach, reach+1):
                    cell = start[0]+dx, start[1]+dy
                    if (2 <= hypot(dx, dy) <= reach and grid.contains(cell)
                            and not grid.is_static_obstacle(cell) and cell not in measured
                            and side_opening(cell_to_world(info, cell))):
                        candidates.append(cell)
            candidates.sort(key=lambda p: hypot(p[0]-end[0], p[1]-end[1])+hypot(p[0]-start[0],p[1]-start[1]))
            edges, costs = self._failure_constraints(grid)
            for candidate in candidates[:config["max_candidates"]]:
                if time.perf_counter()-began > max_seconds*.35:
                    break
                path = self.planner.plan(grid, candidate, end, measured,
                                         blocked_edges=edges, edge_penalty=costs)
                if path:
                    tail = [(x+.5, y+.5) for x, y in path]
                    exact_goal = local((self.last_goal.pose.position.x, self.last_goal.pose.position.y))
                    if tail[-1] != exact_goal:
                        tail.append(exact_goal)
                    tails.append(tail)
                    retained_indices.append(None)
                    candidates_used.append(cell_to_world(info, candidate))
                    break
        if not tails:
            self.plan_metrics["metric_connector_reason"] = "no_coarse_free_complete_tail"
            return False
        budget = max_seconds-(time.perf_counter()-began)
        radius = (body+max(config["clearance_margin"], start_margin))/info.resolution
        start = world_to_cell(info, *current[:2])
        reserved = (self._global_occlusion_reserve(start, observed)
                    |self._global_occlusion_reserve(start, remembered, remembered=True))
        virtual = self._live_obstacle_blocked(reserved)-measured
        budget = max_seconds-(time.perf_counter()-began)
        result = plan_connector(raw, observed | remembered, local(current[:2]), tails,
            radius=radius, spacing=.01/info.resolution, margin=0.,
            lateral=min(body/2., config["max_lateral_distance"])/info.resolution,
            max_seconds=max(0., budget), max_expansions=max_expansions,
            preferred=float(self.get_parameter("preferred_clearance").value)/info.resolution,
            gain=float(self.get_parameter("clearance_penalty_gain").value), soft_cells=virtual)
        self.plan_metrics.update(metric_connector_reason=result.reason,
            metric_connector_duration_ms=(time.perf_counter()-began)*1000.,
            metric_connector_search_ms=result.duration_s*1000.,
            metric_connector_expansions=result.expansions,
            metric_connector_side_only=after_reverse, metric_connector_anchors=candidates_used)
        if not result.points:
            return False
        points = [world(p) for p in result.points]
        retained = retained_indices[result.target_index]
        if retained is not None:
            prefix = len(points)-len(tails[result.target_index])
            points[prefix:] = reference[retained:]
        if after_reverse and not side_opening(point_at_progress(points, min(.15, path_length(points)))):
            self.plan_metrics["metric_connector_reason"] = "no_side_leading_first_segment"
            return False
        # One fresh snapshot, same goal/map and actual authority at publication.
        fresh = self._fresh_motion_pose()
        if (fresh is None or key != self._local_route_key() or self.raw_static_grid is not raw
                or self.observed != overlay or self.remembered != remembered
                or not self._automatic_resume_allowed() or not self._observations_ready()):
            self.plan_metrics["metric_connector_reason"] = "fresh_snapshot_or_authority_changed"
            return False
        points[0] = fresh[:2]
        if polyline_clearance([local(p) for p in points], occupied_rectangles(raw, observed | remembered),
                              raw.width, raw.height)+1e-9 < radius:
            self.plan_metrics["metric_connector_reason"] = "fresh_start_sweep_changed"
            return False
        if not self._prefix_sweep_clear(points, cells_verified=True):
            self.plan_metrics["metric_connector_reason"] = self.plan_metrics.get(
                "start_prefix_validation", {}).get("reason", "raw_prefix_unverified")
            if self.plan_metrics.get("start_prefix_validation", {}).get("input_pending"):
                self.metric_connector_input_pending = True
                self._hold_start_prefix_inputs()
            return False
        if time.perf_counter()-began > max_seconds:
            self.plan_metrics["metric_connector_reason"] = "final_validation_time_budget"
            return False
        path = Path()
        path.header.frame_id = self.get_parameter("map_frame").value
        path.header.stamp = self.get_clock().now().to_msg()
        for index, point in enumerate(points):
            pose = PoseStamped(); pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = point
            if index+1 < len(points):
                yaw = atan2(points[index+1][1]-point[1], points[index+1][0]-point[0])
                pose.pose.orientation.z, pose.pose.orientation.w = sin(yaw/2), cos(yaw/2)
            else:
                pose.pose.orientation = self.last_goal.pose.orientation
            path.poses.append(pose)
        elapsed = time.perf_counter()-began
        self.plan_metrics["metric_connector_duration_ms"] = elapsed*1000.
        if elapsed > max_seconds:
            self.plan_metrics["metric_connector_reason"] = "final_validation_time_budget"
            return False
        self.planned_world_points = points
        self.planned_cells = sorted(route_cells([local(p) for p in points]))
        self.waiting_for_replan = self.invalid_path_published = False
        self.failure_state, self.failure_reason = "CLEAR", ""
        index = retained_indices[result.target_index]
        if index is None:
            self.local_reference_points = list(points)
            self.local_reference_key = key
            reference = self.local_reference_points
        anchor = world(tails[result.target_index][0])
        self.local_rejoin_anchor = {"reference": list(reference),
            "progress": path_progress(reference, anchor), "pose": fresh,
            "odom": self.latest_odom}
        self._publish_path(path)
        self.metric_route_scope = (self._local_route_key(), raw, self.current_path_stamp_ns,
                                   self._start_certificate_geometry())
        self.plan_metrics.update(plan_kind="LOCAL_METRIC_CONNECTOR", local_detour_state="ACTIVE",
            recovery_stage="side_connector_after_rear_rejected" if after_reverse else "metric_local_detour",
            failure_class="none", metric_connector_reason="certified_connector",
            rejoin_point={"x": anchor[0], "y": anchor[1]}, metric_retained_suffix=retained is not None)
        self.publish_status("READY", "Continuous-body local connector; original occupied squares retained")
        return True

    def _local_route_key(self):
        goal = self.last_goal
        if goal is None:
            return None
        p, q = goal.pose.position, goal.pose.orientation
        return (self.breadcrumb_map_key, goal.header.frame_id,
                goal.header.stamp.sec, goal.header.stamp.nanosec,
                p.x, p.y, q.x, q.y, q.z, q.w)

    def _invalid_local_configuration(self, reason):
        self.plan_metrics.update(local_detour_state="FAILED", detour_reason="invalid_configuration",
                                 detour_configuration_error=str(reason)[:160])
        return None

    def _local_configuration(self):
        """Runtime edits may disable this optional path, never crash planning."""
        try:
            config = {name.removeprefix("local_detour_"): self.get_parameter(name).value
                      for name in LOCAL_DETOUR_DEFAULTS}
            if type(config["enabled"]) is not bool:
                raise ValueError("local_detour_enabled must be boolean")
            if not config["enabled"]:
                return config
            for name in ("trigger_distance", "max_rejoin_distance", "max_length",
                         "max_lateral_distance", "max_planning_time"):
                value = config[name]
                if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                    raise ValueError("local_detour_"+name+" must be finite and positive")
            if config["max_planning_time"] > .1:
                raise ValueError("local_detour_max_planning_time must be at most 0.1 seconds")
            margin = config["clearance_margin"]
            if (type(margin) not in (int, float) or not isfinite(margin)
                    or not 0 <= margin <= 1.):
                raise ValueError("local_detour_clearance_margin must be finite in 0..1 metres")
            for name in ("occlusion_extension", "rejoin_buffer", "observation_margin"):
                value = config[name]
                if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1.:
                    raise ValueError("local_detour_"+name+" must be finite in 0..1 metres")
            for name, maximum in (("max_expansions", 10000), ("max_candidates", 64),
                                  ("max_attempts", 10)):
                if type(config[name]) is not int or not 1 <= config[name] <= maximum:
                    raise ValueError("local_detour_"+name+" must be an integer in 1.."+str(maximum))
            return config
        except (ValueError, TypeError, OverflowError) as exc:
            return self._invalid_local_configuration(exc)

    def _local_footprint_grid(self, config=None):
        """Cache static/unknown footprint inflation; never clear start cells."""
        config = self._local_configuration() if config is None else config
        if config is None or not config["enabled"]:
            return None, 0
        try:
            return self._build_local_footprint_grid(config)
        except (ValueError, TypeError, OverflowError) as exc:
            self._invalid_local_configuration(exc)
            return None, 0

    def _build_local_footprint_grid(self, config):
        message = self.map[0]
        radius = float(self.get_parameter("robot_radius").value)+config["clearance_margin"]
        if not isfinite(radius) or radius <= 0 or not isfinite(message.info.resolution) or message.info.resolution <= 0:
            raise ValueError("Local footprint radius and map resolution must be finite and positive")
        inflation = max(int(self.get_parameter("obstacle_inflation_cells").value),
                        ceil(radius/message.info.resolution))
        key = (self.breadcrumb_map_key, self.get_parameter("occupied_threshold").value,
               inflation, radius)
        if self.local_grid_cache is None or self.local_grid_cache[0] != key:
            grid = occupancy_to_grid(message, key[1], inflation)
            margin = radius/message.info.resolution
            # Outside-map space is unknown too; the complete footprint must
            # remain inside, even when no occupied edge cells were published.
            rows = tuple(tuple(value or x+.5 < margin or grid.width-x-.5 < margin
                               or y+.5 < margin or grid.height-y-.5 < margin
                               for x, value in enumerate(row))
                         for y, row in enumerate(grid.rows))
            self.local_grid_cache = (key, Grid(rows))
        return self.local_grid_cache[1], inflation

    def _local_occlusion_reserve(self, raw_hits, reference, start, inflation, config):
        """Reserve unseen depth locally without claiming extra sensor returns.

        A visible face does not establish that the old route is free directly
        behind it. Extend only nearby corridor-blocking hits along the original
        forward tangent; keep actual observations and persistent memory intact.
        """
        reserved = set(raw_hits)
        extension = config["occlusion_extension"]/self.map[0].info.resolution
        if extension <= 0:
            return reserved
        origin = start[0]+.5, start[1]+.5
        progress = path_progress(reference, origin)
        trigger = config["trigger_distance"]/self.map[0].info.resolution
        width, height = self.map[0].info.width, self.map[0].info.height
        for cell in raw_hits:
            center = cell[0]+.5, cell[1]+.5
            if hypot(center[0]-origin[0], center[1]-origin[1]) > trigger:
                continue
            along = path_progress(reference, center)
            point = point_at_progress(reference, along)
            if along < progress-1. or hypot(center[0]-point[0], center[1]-point[1]) > inflation+1.:
                continue
            ahead = point_at_progress(reference, along+1.)
            dx, dy = ahead[0]-point[0], ahead[1]-point[1]
            length = hypot(dx, dy)
            if length < 1e-9:
                continue
            end = center[0]+extension*dx/length, center[1]+extension*dy/length
            reserved.update(cell for cell in route_cells([center, end])
                            if 0 <= cell[0] < width and 0 <= cell[1] < height)
        return reserved

    def _global_occlusion_reserve(self, start, raw_hits, *, remembered=False):
        """Reserve unseen depth on a retained route without inventing LiDAR hits.

        A local repair can fail and hand the goal back to global A*. Both
        planners must then protect the same bounded hidden space behind a
        visible or remembered obstacle face. The reserve is planning-only:
        scan observations and persistent memory retain measured cells alone.
        """
        if (not raw_hits or len(self.local_reference_points) < 2
                or self.local_reference_key != self._local_route_key()):
            return raw_hits
        reserve_hits = raw_hits
        if remembered:
            # An isolated persisted pixel has no measured face extent. A
            # different cluster elsewhere must not cause this singleton to
            # gain an invented depth and falsely seal a narrow passage.
            reserve_hits = {cell for cell in raw_hits
                            if any((cell[0]+dx, cell[1]+dy) in raw_hits
                                   for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                   if dx or dy)}
            if not reserve_hits:
                return raw_hits
        info = self.map[0].info
        try:
            resolution = float(info.resolution)
            extension = float(self.get_parameter("local_detour_occlusion_extension").value)
            trigger = float(self.get_parameter("local_detour_trigger_distance").value)
            radius = (float(self.get_parameter("robot_radius").value)
                      + float(self.get_parameter("local_detour_clearance_margin").value))
            if (not all(isfinite(value) for value in (resolution, extension, trigger, radius))
                    or resolution <= 0 or not 0 <= extension <= 1.
                    or trigger <= 0 or radius <= 0):
                # Invalid optional local tuning must not disable the global
                # planner; its measured hits still remain hard obstacles.
                return raw_hits
            inflation = max(int(self.get_parameter("obstacle_inflation_cells").value),
                            ceil(radius / resolution))
            yaw = yaw_from_quaternion(info.origin.orientation)
            c, sn = cos(yaw), sin(yaw)
            origin = info.origin.position
            reference = [((c*(x-origin.x)+sn*(y-origin.y))/resolution,
                          (-sn*(x-origin.x)+c*(y-origin.y))/resolution)
                         for x, y in self.local_reference_points]
            return raw_hits | self._local_occlusion_reserve(
                reserve_hits, reference, start, inflation,
                {"occlusion_extension": extension, "trigger_distance": trigger})
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
            return raw_hits

    def _live_obstacle_blocked(self, cells, *, observation_margin=0.):
        """Inflate observed or remembered hits by the same complete footprint.

        Both the scan hit and robot center may lie anywhere in their raster
        cells; the exact cell-square separation avoids an extra circular halo.
        Confirmation changes persistence, never collision geometry. Static
        map inflation stays unchanged; Nav2 independently checks raw scans.
        """
        message = self.map[0]
        width, height, resolution = message.info.width, message.info.height, message.info.resolution
        robot_radius = float(self.get_parameter("robot_radius").value)
        margin = float(self.get_parameter("local_detour_clearance_margin").value)
        if (not all(isfinite(value) for value in (resolution, robot_radius, margin, observation_margin))
                or resolution <= 0 or robot_radius <= 0 or margin < 0 or observation_margin < 0):
            # An invalid runtime parameter must never silently reduce collision
            # clearance. Preserve the old square exclusion until it is fixed.
            inflation = max(0, int(self.get_parameter("obstacle_inflation_cells").value))
            return inflate_cells(cells, width, height, inflation)
        radius_cells = (robot_radius + margin + observation_margin) / resolution
        return inflate_cells_footprint(cells, width, height, radius_cells)

    def _soft_occlusion_penalty(self, costs, conservative_blocked, measured_blocked):
        """Only invented hidden-depth cells may become finite planning costs.

        Measured hits, confirmed memory, static/unknown occupancy and their
        unchanged body inflation remain hard in the separate blocked mask.
        A cost preference is not evidence that unseen space is actually free.
        """
        penalty = dict(costs)
        amount = max(1., float(self.get_parameter("clearance_penalty_gain").value))
        for cell in conservative_blocked - measured_blocked:
            penalty[cell] = penalty.get(cell, 0.) + amount
        return penalty

    def _occlusion_fallback_ready(self, start):
        """Do not relax an unmeasured reserve on stale or unrelated evidence."""
        if (not self._goal_is_active() or not self._automatic_resume_allowed()
                or not self._observations_ready()
                or len(self.local_reference_points) < 2
                or self.local_reference_key != self._local_route_key()):
            return False
        pose = self._fresh_motion_pose()
        return pose is not None and world_to_cell(self.map[0].info, pose[0], pose[1]) == start

    def _retained_start_certificate(self, start, allowed, grid, raw_grid,
                                    conservative_blocked, observed, remembered,
                                    *, require_occlusion_context=True):
        """Revalidate only this call's existing edges, never certify new exits."""
        if (not allowed or not self.get_parameter("start_egress_enabled").value
                or self.raw_static_grid is not raw_grid
                or self.observed != observed or self.remembered != remembered
                or (require_occlusion_context and not self._occlusion_fallback_ready(start))):
            return set()
        info = self.map[0].info
        if start in self._live_obstacle_blocked(remembered):
            return set()
        pose = self._fresh_motion_pose()
        if pose is None or world_to_cell(info, pose[0], pose[1]) != start:
            return set()
        yaw = yaw_from_quaternion(info.origin.orientation)
        dx, dy = pose[0]-info.origin.position.x, pose[1]-info.origin.position.y
        point = ((cos(yaw)*dx+sin(yaw)*dy)/info.resolution,
                 (-sin(yaw)*dx+cos(yaw)*dy)/info.resolution)
        current = certified_start_only_edges(
            raw_grid, grid, observed | remembered, conservative_blocked, start, point,
            float(self.get_parameter("robot_radius").value)/info.resolution,
            float(self.get_parameter("start_egress_clearance_margin").value)/info.resolution)
        return allowed & current

    @staticmethod
    def _start_only_forbidden_edges(start, allowed):
        """One certified departure never grants reentry or diagonal shortcuts."""
        forbidden = {(start, (start[0]+dx, start[1]+dy))
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     if (dx or dy) and (start, (start[0]+dx, start[1]+dy)) not in allowed}
        forbidden.update(((start[0]+dx, start[1]+dy), start)
                         for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
        for dx in (-1, 1):
            for dy in (-1, 1):
                horizontal, vertical = (start[0]+dx, start[1]), (start[0], start[1]+dy)
                forbidden.update(((horizontal, vertical), (vertical, horizontal)))
        return forbidden

    def _measured_start_certificate(self, start, grid, raw_grid, measured, observed, remembered):
        """Only after primary NoPath: certify a single-cell measured-mask exit."""
        if (not self.get_parameter("start_egress_enabled").value
                or self.raw_static_grid is not raw_grid
                or self.observed != observed or self.remembered != remembered
                or not self._occlusion_fallback_ready(start)):
            return set()
        info = self.map[0].info
        if start in self._live_obstacle_blocked(remembered):
            return set()
        pose = self._fresh_motion_pose()
        if pose is None or world_to_cell(info, *pose[:2]) != start:
            return set()
        yaw = yaw_from_quaternion(info.origin.orientation)
        dx, dy = pose[0]-info.origin.position.x, pose[1]-info.origin.position.y
        point = ((cos(yaw)*dx+sin(yaw)*dy)/info.resolution,
                 (-sin(yaw)*dx+cos(yaw)*dy)/info.resolution)
        return certified_start_only_edges(raw_grid, grid, observed | remembered, measured,
            start, point, float(self.get_parameter("robot_radius").value)/info.resolution,
            float(self.get_parameter("start_egress_clearance_margin").value)/info.resolution)

    def _selected_dynamic_escape_certificate(self, start, cells, grid, raw_grid,
                                              measured, observed, remembered):
        """Own one already-selected escape edge, never the surrounding halo.

        A successful dynamic-halo search is not itself the retained first-edge
        proof used by scan callbacks. Bind its actual chosen edge with the
        same fresh continuous-body certificate, without changing the search
        or granting a second inflated cell, diagonal shortcut or reentry.
        """
        diagnostic = {"reason": "checking", "edge": None}
        self.plan_metrics["selected_start_certificate_diagnostics"] = diagnostic
        if (len(cells) < 2 or cells[0] != start or not self._goal_is_active()
                or not self._automatic_resume_allowed() or not self._observations_ready()):
            diagnostic["reason"] = "fresh_goal_authority_or_observations_unavailable"
            return set()
        selected = (cells[0], cells[1])
        diagnostic["edge"] = [list(cell) for cell in selected]
        if start in self._live_obstacle_blocked(remembered):
            diagnostic["reason"] = "actual_memory_inflation_start"
            return set()
        allowed = self._retained_start_certificate(
            start, {selected}, grid, raw_grid, measured | {start}, observed, remembered,
            require_occlusion_context=False)
        if not allowed:
            diagnostic["reason"] = "selected_edge_fresh_body_certificate_rejected"
            return set()
        if any(grid.is_static_obstacle(cell) or cell in measured for cell in cells[1:]):
            diagnostic["reason"] = "later_measured_or_static_cell_blocked"
            return set()
        forbidden = self._start_only_forbidden_edges(start, allowed)
        if any(edge in forbidden for edge in zip(cells, cells[1:])):
            diagnostic["reason"] = "reentry_or_adjacent_diagonal_forbidden"
            return set()
        diagnostic["reason"] = "selected_first_edge_certified"
        return allowed

    def _clear_start_certificate(self, reason="context_reset"):
        if (getattr(self, "active_start_certificate", None) is not None
                or getattr(self, "completed_start_prefix", None) is not None):
            self.plan_metrics.update(start_certificate_state="REVOKED",
                                     start_certificate_reason=reason)
        self.active_start_certificate = self.completed_start_prefix = None

    def _start_certificate_geometry(self):
        return tuple(self.get_parameter(name).value for name in
                     ("robot_radius", "start_egress_clearance_margin", "obstacle_inflation_cells",
                      "local_detour_clearance_margin", "occupied_threshold"))

    def _install_start_certificate(self, edge):
        """Bind only the chosen published first edge; never retain other exits."""
        pose = self._fresh_motion_pose()
        if (pose is None or self.latest_odom is None or len(self.planned_world_points) < 2
                or world_to_cell(self.map[0].info, *pose[:2]) != edge[0]):
            self.plan_metrics.update(start_certificate_state="NOT_INSTALLED",
                start_certificate_reason="fresh_install_pose_or_odom_unavailable")
            return
        self.active_start_certificate = {
            "edge": edge, "key": self._local_route_key(), "raw_grid": self.raw_static_grid,
            "path_stamp": self.current_path_stamp_ns, "geometry": self._start_certificate_geometry(),
            "prefix": tuple(self.planned_world_points[:2]), "pose": pose,
            "odom": self.latest_odom, "created_at": time.monotonic()}
        self.plan_metrics.update(start_certificate_state="ACTIVE",
                                 start_certificate_reason="chosen_first_edge_installed")

    def _prefix_sweep_clear(self, points, *, observed=(), cells_verified=False):
        """Fresh complete circular body/sweep against raw cells and raw returns."""
        def verdict(clear, reason, **details):
            self.plan_metrics["start_prefix_validation"] = {
                "clear": clear, "reason": reason, **details}
            return clear

        info, raw = self.map[0].info, self.raw_static_grid
        yaw = yaw_from_quaternion(info.origin.orientation)
        c, s, resolution = cos(yaw), sin(yaw), info.resolution
        local = [((c*(x-info.origin.position.x)+s*(y-info.origin.position.y))/resolution,
                  (-s*(x-info.origin.position.x)+c*(y-info.origin.position.y))/resolution)
                 for x, y in points]
        radius = (float(self.get_parameter("robot_radius").value)
                  + float(self.get_parameter("start_egress_clearance_margin").value))
        required = radius/resolution
        dynamic = self.observed | self.remembered | set(observed)
        for a, b in (() if cells_verified else zip(local, local[1:])):
            reach = ceil(required+1.)
            occupied = [(x, y) for x in range(max(0, int(min(a[0], b[0]))-reach),
                                               min(raw.width, ceil(max(a[0], b[0]))+reach+1))
                        for y in range(max(0, int(min(a[1], b[1]))-reach),
                                       min(raw.height, ceil(max(a[1], b[1]))+reach+1))
                        if raw.is_static_obstacle((x, y)) or (x, y) in dynamic]
            length = hypot(b[0]-a[0], b[1]-a[1])
            steps = max(1, ceil(length/.2))
            threshold = required+length/(2*steps)
            for step in range(steps+1):
                x, y = a[0]+(b[0]-a[0])*step/steps, a[1]+(b[1]-a[1])*step/steps
                if min(x, y, raw.width-x, raw.height-y)+1e-9 < threshold:
                    return verdict(False, "raw_map_boundary")
                if any(hypot(max(ox-x, 0., x-ox-1), max(oy-y, 0., y-oy-1))+1e-9 < threshold
                       for ox, oy in occupied):
                    return verdict(False, "raw_static_unknown_or_dynamic_cell_intersection")
        selected = self._coherent_start_scan()
        if selected is None:
            return verdict(False, "coherent_raw_scan_input_pending", input_pending=True)
        scan, tf = selected
        try:
            origin, q = tf.transform.translation, tf.transform.rotation
            heading = _normalized_pose_yaw(q)
            if not all(isfinite(v) for v in (origin.x, origin.y, heading)):
                return verdict(False, "raw_scan_transform_geometry_invalid")
            raw_points = []
            for index, reading in enumerate(scan.ranges):
                if not isfinite(reading) or not scan.range_min <= reading <= scan.range_max:
                    continue
                angle = heading+scan.angle_min+index*scan.angle_increment
                x, y = origin.x+reading*cos(angle), origin.y+reading*sin(angle)
                raw_points.append((x, y))
            if point_sweep_clearance(points, raw_points) < radius:
                return verdict(False, "raw_scan_body_or_sweep_intersection")
        except (TransformException, AttributeError, ValueError, TypeError) as exc:
            return verdict(False, "raw_scan_transform_unavailable_or_invalid",
                lookup_error=(type(exc).__name__+": "+str(exc))[:240],
                scan_stamp_ns=scan.header.stamp.sec*1_000_000_000+scan.header.stamp.nanosec)
        return verdict(True, "fresh_raw_body_and_sweep_clear")

    def _validated_start_prefix_exemption(self, remaining, *, observed=()):
        """Skip only a freshly swept-clear current/just-traversed first cell."""
        self.start_prefix_input_pending = False
        if getattr(self, "start_scan_context_invalid", False):
            self.start_prefix_input_pending = True
            return set()
        state = (getattr(self, "active_start_certificate", None)
                 or getattr(self, "completed_start_prefix", None))
        if state is None:
            return set()
        pose = self._fresh_motion_pose()
        if (not self._goal_is_active() or not self._automatic_resume_allowed()
                or not self._observations_ready() or pose is None
                or state["key"] != self._local_route_key()
                or state["raw_grid"] is not self.raw_static_grid
                or state["path_stamp"] != self.current_path_stamp_ns
                or state["geometry"] != self._start_certificate_geometry()
                or len(self.planned_world_points) < 2
                or tuple(self.planned_world_points[:2]) != state["prefix"]
                or time.monotonic() < state["created_at"]):
            reason = ("goal_or_authority_unavailable" if not self._goal_is_active() or not self._automatic_resume_allowed()
                      else "fresh_observations_or_pose_unavailable" if not self._observations_ready() or pose is None
                      else "certificate_context_geometry_or_path_changed")
            self._clear_start_certificate(reason)
            return set()
        start, destination = state["edge"]
        current = world_to_cell(self.map[0].info, *pose[:2])
        actual_observed = set(self.observed) | set(observed)
        if start not in remaining:
            self._clear_start_certificate("certified_start_left_remaining_route")
            return set()
        if getattr(self, "active_start_certificate", None) is not None:
            if current == start:
                info = self.map[0].info
                memory = self._live_obstacle_blocked(self.remembered)
                measured = self._live_obstacle_blocked(actual_observed) | memory
                angle = yaw_from_quaternion(info.origin.orientation)
                dx, dy = pose[0]-info.origin.position.x, pose[1]-info.origin.position.y
                point = ((cos(angle)*dx+sin(angle)*dy)/info.resolution,
                         (-sin(angle)*dx+cos(angle)*dy)/info.resolution)
                # A previously owned departure remains meaningful while its
                # current raster cell flickers free. The synthetic start bit
                # only asks the same raw-body proof to retain this exact edge;
                # it never exempts a destination or any later measured cell.
                allowed = (certified_start_only_edges(self.raw_static_grid, self.map[1],
                    actual_observed | self.remembered, measured | {start}, start, point,
                    float(self.get_parameter("robot_radius").value)/info.resolution,
                    float(self.get_parameter("start_egress_clearance_margin").value)/info.resolution)
                    if start not in memory else set())
                if (state["edge"] not in allowed or not self._prefix_sweep_clear(
                        [pose[:2], *state["prefix"]], observed=actual_observed)):
                    if (state["edge"] in allowed and self.plan_metrics.get(
                            "start_prefix_validation", {}).get("input_pending") is True):
                        self.start_prefix_input_pending = True
                        self.plan_metrics.update(start_certificate_state="INPUT_PENDING",
                            start_certificate_reason="coherent_raw_scan_input_pending")
                        return set()
                    reason = ("actual_memory_inflation_start" if start in memory
                              else "selected_edge_no_longer_certified" if state["edge"] not in allowed
                              else self.plan_metrics.get("start_prefix_validation", {}).get("reason", "raw_sweep_rejected"))
                    self._clear_start_certificate(reason)
                    return set()
                return {start}
            odom, initial = self.latest_odom, state["odom"]
            moved = hypot(odom[2]-initial[2], odom[3]-initial[3]) if odom else 0.
            angle = state["pose"][2]-initial[4]
            ox, oy = (odom[2]-initial[2], odom[3]-initial[3]) if odom else (0., 0.)
            dx, dy = cos(angle)*ox-sin(angle)*oy, sin(angle)*ox+cos(angle)*oy
            if (current != destination or odom is None or odom[5] != initial[5]
                    or odom[1] <= initial[1] or moved < .005
                    or hypot(pose[0]-state["pose"][0]-dx, pose[1]-state["pose"][1]-dy) > .01+.2*moved
                    or _point_segment_distance(*pose[:2], *state["prefix"]) > .5*self.map[0].info.resolution):
                self._clear_start_certificate("departure_pose_or_independent_odom_mismatch")
                return set()
            self.active_start_certificate = None
            self.completed_start_prefix = state
        elif current == start:
            self._clear_start_certificate("certified_start_reentered")  # No reuse on return.
            return set()
        progress = path_progress(self.planned_world_points, pose[:2])
        end = min(path_length(self.planned_world_points), progress+.11)
        sweep = [pose[:2], point_at_progress(self.planned_world_points, progress)]
        traveled = 0.
        for a, b in zip(self.planned_world_points, self.planned_world_points[1:]):
            traveled += hypot(b[0]-a[0], b[1]-a[1])
            if progress < traveled < end:
                sweep.append(b)
        sweep.append(point_at_progress(self.planned_world_points, end))
        if not self._prefix_sweep_clear(sweep, observed=actual_observed):
            if self.plan_metrics.get("start_prefix_validation", {}).get("input_pending") is True:
                self.start_prefix_input_pending = True
                self.plan_metrics.update(start_certificate_state="INPUT_PENDING",
                    start_certificate_reason="coherent_raw_scan_input_pending")
                return set()
            self._clear_start_certificate(self.plan_metrics.get(
                "start_prefix_validation", {}).get("reason", "raw_forward_sweep_rejected"))
            return set()
        return {start}  # Only the original cell, never the whole 27 cm backbuffer.

    def _hold_start_prefix_inputs(self):
        """Missing coherent evidence is a sensor stop, never an obstacle retry."""
        self.failure_state, self.failure_reason = (
            "SENSOR_HOLD", "Waiting for event-coherent raw scan and acquisition-time TF")
        self.plan_metrics.update(failure_class="sensor", recovery_stage="await_start_prefix_inputs")
        self._clear_start_certificate("coherent_raw_scan_input_pending")
        self._invalidate_path(self.failure_reason)

    def _try_local_detour(self, changed_cells, *, terminal_repair=False, raw_hits=()):
        """Queue a certified nonempty replacement before invalidating Nav2.

        The existing follower owns cancel/terminal/new-goal ordering. Planning
        needs fresh independent evidence, not an extra stop or fixed delay.
        """
        self.metric_connector_input_pending = False
        config = self._local_configuration()
        if config is None or not config["enabled"]:
            return False
        if (not self._goal_is_active()
                or (not self.planned_world_points and not terminal_repair)
                or len(self.local_reference_points) < 2
                or self.local_reference_key != self._local_route_key()
                or self.recovery_active or self.pending_failure_id is not None
                or self.obstacle_hold.holding
                or self.failure_state in ("SENSOR_HOLD", "CONTROLLER_HOLD")
                or not self._automatic_resume_allowed()):
            return False
        current = self._fresh_motion_pose()
        if current is None or not self._observations_ready():
            return False
        # A genuinely blocked active prefix gets another bounded local repair
        # before global A*. Keep its original reference/suffix and attempt
        # budget: neither another scan nor a terminal abort earns a new budget.
        if self.local_detour_attempts >= config["max_attempts"]:
            self.plan_metrics.update(local_detour_state="FAILED", detour_reason="attempt_budget")
            return False
        message = self.map[0]
        resolution = message.info.resolution
        start = world_to_cell(message.info, current[0], current[1])
        trigger = config["trigger_distance"]
        nearby = [cell for cell in changed_cells
                  if hypot(cell_to_world(message.info, cell)[0]-current[0],
                           cell_to_world(message.info, cell)[1]-current[1]) <= trigger]
        if not nearby:
            return False
        began = time.perf_counter()
        grid, inflation = self._local_footprint_grid(config)
        if grid is None:
            return False
        edges, edge_cost = self._failure_constraints(grid)
        yaw = yaw_from_quaternion(message.info.origin.orientation)
        c, sn = cos(yaw), sin(yaw)
        def local(point):
            x, y = point[0]-message.info.origin.position.x, point[1]-message.info.origin.position.y
            return ((c*x+sn*y)/resolution, (-sn*x+c*y)/resolution)
        reference = [local(point) for point in self.local_reference_points]
        local_observed = self.observed | set(raw_hits)
        actual_hits = local_observed | self.remembered
        reserved_observed = self._local_occlusion_reserve(local_observed, reference, start, inflation, config)
        reserved_remembered = self._local_occlusion_reserve(self.remembered, reference, start, inflation, config)
        reserved_hits = reserved_observed | reserved_remembered
        # Scan endpoints can cross a raster boundary as the viewing angle
        # changes. Reserve that uncertainty before the unchanged footprint
        # inflation, so a one-cell refinement need not cancel a safe bypass.
        observation_cells = ceil(config["observation_margin"]/resolution)
        planning_hits = inflate_cells(reserved_hits, grid.width, grid.height, observation_cells)
        blocked = (self._live_obstacle_blocked(
            reserved_observed, observation_margin=config["observation_margin"])
            | self._live_obstacle_blocked(
                reserved_remembered, observation_margin=config["observation_margin"]))
        measured_blocked = (self._live_obstacle_blocked(
            local_observed, observation_margin=config["observation_margin"])
            | self._live_obstacle_blocked(
                self.remembered, observation_margin=config["observation_margin"]))
        virtual_blocked = blocked - measured_blocked
        # A cached static-only field cannot price newly exposed obstacle faces.
        # This mask is a conservative local planning constraint, not new data.
        dynamic_penalty = clearance_penalties(self.raw_static_grid, planning_hits, resolution,
            float(self.get_parameter("preferred_clearance").value),
            float(self.get_parameter("clearance_penalty_gain").value))
        self.local_detour_attempts += 1
        self.local_attempt_id += 1
        if grid.is_static_obstacle(start) or start in measured_blocked:
            # Coarse start-blocked queries cannot take even one edge. Spend a
            # bounded share on actual centres; reserve the rest for a second
            # side-opening check only after the measured rear is rejected.
            deadline = config["max_planning_time"]-(time.perf_counter()-began)
            if self._try_metric_connector(config, max_seconds=max(0., deadline*.65),
                    max_expansions=max(1, int(config["max_expansions"]*.65)), raw_hits=raw_hits):
                return True
            if self.metric_connector_input_pending:
                return False
            spent = time.perf_counter()-began
            used = self.plan_metrics.get("metric_connector_expansions", 0)
            self.metric_reverse_budget = {"key": self._local_route_key(),
                "episode_id": (self.recovery_coordinator.current.episode_id
                               if self.recovery_coordinator.current is not None else None),
                "seconds": max(0., config["max_planning_time"]-spent),
                "expansions": max(0, config["max_expansions"]-used)}
            self.plan_metrics.update(local_detour_state="FAILED", detour_reason="metric_start_connector_unavailable",
                                     adapter_duration_ms=spent*1000.)
            return False
        fallback_attempted = fallback_used = False
        self.plan_metrics.update(occlusion_fallback=False, occlusion_fallback_attempted=False,
            occlusion_constraint_mode="HARD_RESERVE", occlusion_virtual_cells=len(virtual_blocked))
        try:
            search_started = time.perf_counter()
            # Reserve a share for the confidence fallback, rather than letting
            # conservative anchors consume the entire event's candidate budget.
            share = .6 if virtual_blocked else 1.
            candidate_share = .5 if virtual_blocked else 1.
            result = plan_local_detour(grid, start, reference, blocked,
                resolution=resolution, blocked_edges=edges, cell_penalty=dynamic_penalty,
                edge_penalty=edge_cost,
                max_rejoin_distance=config["max_rejoin_distance"],
                max_detour_length=config["max_length"],
                max_lateral_distance=config["max_lateral_distance"],
                max_expansions=max(1, int(config["max_expansions"]*share)),
                max_candidates=max(1, int(config["max_candidates"]*candidate_share)),
                max_seconds=config["max_planning_time"]*share,
                rejoin_buffer=config["rejoin_buffer"])
            primary_reason = result.reason
            total_expansions = result.expansions
            total_candidates = result.candidates_examined
            remaining_time = config["max_planning_time"] - (time.perf_counter()-search_started)
            remaining_expansions = config["max_expansions"] - total_expansions
            remaining_candidates = config["max_candidates"] - total_candidates
            if (not result.points and primary_reason in (
                    "start_blocked", "no_forward_rejoin", "no_safe_anchor", "no_local_route",
                    "time_budget", "expansion_budget")
                    and virtual_blocked and remaining_time > 0
                    and remaining_expansions > 0 and remaining_candidates > 0
                    and self._occlusion_fallback_ready(start)):
                soft_penalty = self._soft_occlusion_penalty(dynamic_penalty, blocked, measured_blocked)
                # Fresh TF checks and cost construction consume this same
                # event's deadline; do not give their elapsed time back.
                remaining_time = config["max_planning_time"] - (time.perf_counter()-search_started)
                if remaining_time > 0:
                    fallback_attempted = True
                    dynamic_penalty = soft_penalty
                    result = plan_local_detour(grid, start, reference, measured_blocked,
                        resolution=resolution, blocked_edges=edges, cell_penalty=dynamic_penalty,
                        edge_penalty=edge_cost,
                        max_rejoin_distance=config["max_rejoin_distance"],
                        max_detour_length=config["max_length"],
                        max_lateral_distance=config["max_lateral_distance"],
                        max_expansions=remaining_expansions, max_candidates=remaining_candidates,
                        max_seconds=remaining_time, rejoin_buffer=config["rejoin_buffer"])
                    total_expansions += result.expansions
                    total_candidates += result.candidates_examined
                    fallback_used = bool(result.points)
            search_duration_ms = (time.perf_counter()-search_started)*1000.
        except (ValueError, TypeError, OverflowError) as exc:
            self._invalid_local_configuration(exc)
            return False
        elapsed = (time.perf_counter()-began)*1000.
        self.plan_metrics.update(local_attempt_id=self.local_attempt_id,
            search_duration_ms=search_duration_ms, adapter_duration_ms=elapsed,
            expansions=total_expansions, candidates_examined=total_candidates, detour_reason=result.reason,
            local_detour_state="FAILED", rejoin_progress=result.rejoin_progress,
            occlusion_reserved_cells=len(reserved_hits-actual_hits),
            occlusion_extension=config["occlusion_extension"], rejoin_buffer=config["rejoin_buffer"],
            observation_reserved_cells=len(planning_hits-reserved_hits),
            observation_margin=config["observation_margin"],
            occlusion_fallback_attempted=fallback_attempted, occlusion_fallback=fallback_used,
            occlusion_constraint_mode="SOFT_VIRTUAL" if fallback_attempted else "HARD_RESERVE",
            occlusion_primary_reason=primary_reason)
        if not result.points:
            remaining = config["max_planning_time"]-(time.perf_counter()-began)
            if self._try_metric_connector(config, max_seconds=max(0., remaining*.65),
                    max_expansions=max(0, config["max_expansions"]-total_expansions), raw_hits=raw_hits):
                return True
            if self.metric_connector_input_pending:
                return False
            self.metric_reverse_budget = {"key": self._local_route_key(),
                "episode_id": (self.recovery_coordinator.current.episode_id
                               if self.recovery_coordinator.current is not None else None),
                "seconds": max(0., config["max_planning_time"]-(time.perf_counter()-began)),
                "expansions": max(0, config["max_expansions"]-total_expansions
                                  -self.plan_metrics.get("metric_connector_expansions", 0))}
            return False
        world = [cell_to_world(message.info, (x-.5, y-.5)) for x, y in result.points]
        # Keep the suffix bit-for-bit, without conversion roundoff or a global
        # smoothing pass moving the original waypoints.
        prefix_count = len(result.points)-(len(reference)-result.rejoin_index)
        world[prefix_count:] = self.local_reference_points[result.rejoin_index:]
        if fallback_used:
            self.get_logger().warning(
                f"Local conservative search failed ({primary_reason}); using soft virtual-depth costs "
                "with all measured obstacles, memory and body inflation unchanged")
        path = Path()
        path.header.frame_id = self.get_parameter("map_frame").value
        now_ns = self.get_clock().now().nanoseconds
        path.header.stamp = rclpy.time.Time(nanoseconds=max(now_ns,
            self.last_failure_event_stamp_ns+1)).to_msg()
        for index, point in enumerate(world):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = point
            if index+1 < len(world):
                angle = atan2(world[index+1][1]-point[1], world[index+1][0]-point[0])
                pose.pose.orientation.z, pose.pose.orientation.w = sin(angle/2.), cos(angle/2.)
            else:
                q = self.last_goal.pose.orientation
                norm = sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
                if isfinite(norm) and norm > 1e-9:
                    pose.pose.orientation.x, pose.pose.orientation.y = q.x/norm, q.y/norm
                    pose.pose.orientation.z, pose.pose.orientation.w = q.z/norm, q.w/norm
                else:
                    pose.pose.orientation.w = 1.
            path.poses.append(pose)
        self.planned_world_points = world
        self.planned_cells = sorted(route_cells(result.points))
        self.waiting_for_replan = self.invalid_path_published = False
        self.local_rejoin_anchor = {"reference": list(self.local_reference_points),
            "progress": result.rejoin_progress, "pose": current, "odom": self.latest_odom}
        rejoin = self.local_reference_points[result.rejoin_index]
        self.plan_metrics.update(plan_kind="LOCAL_DETOUR", local_detour_state="ACTIVE",
            rejoin_point={"x": rejoin[0], "y": rejoin[1]}, global_plan_count=self.global_plan_count)
        if terminal_repair:
            self.plan_metrics.update(failure_class="obstacle", recovery_stage="local_detour")
        self._publish_path(path)
        self.publish_status("READY", "Local prefix repaired; original forward suffix and goal retained")
        return True

    def _terminal_obstacle_blockers(self, segment):
        """Measured objects inside the attempted body sweep, never an abort alone.

        Fresh raw endpoints may precede the projected obstacle layer. Use them
        only in this local search; do not manufacture memory or edit the map.
        A known mapped-wall return is not an unexpected removable object.
        """
        info = self.map[0].info
        body = float(self.get_parameter("robot_radius").value)
        start, end = segment
        raw = self.raw_static_grid
        def intersects(x, y):
            return _point_segment_distance(x, y, start, end) <= body
        blockers = {cell for cell in self.observed | self.remembered
                    if raw.contains(cell) and not raw.is_static_obstacle(cell)
                    and intersects(*cell_to_world(info, cell))}
        scan = self.latest_scan
        if scan is None or not scan.header.frame_id:
            return blockers, set()
        try:
            transform = self.tf_buffer.lookup_transform(
                self.get_parameter("map_frame").value, scan.header.frame_id,
                rclpy.time.Time.from_msg(scan.header.stamp))
            origin, q = transform.transform.translation, transform.transform.rotation
            heading = _normalized_pose_yaw(q)
            if not all(isfinite(value) for value in (origin.x, origin.y, heading)):
                return blockers, set()
        except (TransformException, ValueError, AttributeError):
            return blockers, set()
        known = occupancy_to_grid(self.map[0],
            self.get_parameter("occupied_threshold").value, 0, unknown_is_obstacle=False)
        tolerance = min(.05, info.resolution)
        raw_hits = set()
        for index, reading in enumerate(scan.ranges):
            if not isfinite(reading) or not scan.range_min <= reading <= scan.range_max:
                continue
            angle = heading+scan.angle_min+index*scan.angle_increment
            x, y = origin.x+reading*cos(angle), origin.y+reading*sin(angle)
            if not intersects(x, y):
                continue
            cell = world_to_cell(info, x, y)
            if not raw.contains(cell) or raw.is_static_obstacle(cell):
                continue
            wall = first_static_obstacle_distance(
                info, known, origin.x, origin.y, angle, scan.range_max)
            if wall is None or abs(wall-reading) > tolerance:
                raw_hits.add(cell)
        return blockers | raw_hits, raw_hits

    def _update_local_rejoin(self):
        anchor = self.local_rejoin_anchor
        if anchor is None or not self._goal_is_active() or not self._automatic_resume_allowed():
            return
        pose = self._fresh_motion_pose()
        odom = self.latest_odom
        if pose is None or odom is None or odom[5] != anchor["odom"][5]:
            return
        points = anchor["reference"]
        progress = path_progress(points, pose[:2])
        projection = point_at_progress(points, progress)
        ox, oy = odom[2]-anchor["odom"][2], odom[3]-anchor["odom"][3]
        rotation = anchor["pose"][2]-anchor["odom"][4]
        ex, ey = cos(rotation)*ox-sin(rotation)*oy, sin(rotation)*ox+cos(rotation)*oy
        error = hypot(pose[0]-anchor["pose"][0]-ex, pose[1]-anchor["pose"][1]-ey)
        if (progress >= anchor["progress"]+.10 and hypot(ox, oy) >= .20
                and odom[1] > anchor["odom"][1] and error <= .05+.20*hypot(ox, oy)
                and self.navigation_progress_at is not None
                and time.monotonic()-self.navigation_progress_at < 1.
                and hypot(pose[0]-projection[0], pose[1]-projection[1])
                    <= float(self.get_parameter("rejoin_cross_track_tolerance").value)):
            self.local_detour_attempts = 0
            self.local_rejoin_anchor = None
            self.plan_metrics.update(local_detour_state="REJOINED")
            self.publish_status("READY", "Original suffix rejoin verified by map pose and independent odometry")

    def _automatic_resume_allowed(self):
        status = self.latest_motion_status or {}
        stamp = status.get("sent_at_ns")
        return (self.latest_motion_status_at is not None
                and 0 <= time.monotonic()-self.latest_motion_status_at < 1.0
                and type(stamp) is int
                and -.25e9 <= self.get_clock().now().nanoseconds-stamp < 1e9
                and status.get("mode") == "auto" and status.get("available") is True
                and status.get("autonomous_allowed") is True
                and status.get("estop_latched") is not True
                and status.get("reset_pending") is not True)

    def _register_obstacle_failure(self, x, y, reference_cells=None, *,
                                   failure_id=None, candidate_cells=None):
        """Only nearby unmapped physical hits can arm automatic obstacle waiting."""
        if not self.get_parameter("obstacle_hold_enabled").value or self.map is None:
            return False
        raw = self.raw_static_grid or occupancy_to_grid(self.map[0],
                        self.get_parameter("occupied_threshold").value, 0)
        # Observed hits already passed the sensor's range-based wall classifier.
        # Preserve real near-wall objects and confirmed memory; only an actual
        # occupied/unknown map cell cannot be a removable dynamic blocker.
        hits = {cell for cell in self.observed | self.remembered
                if not raw.is_static_obstacle(cell)}
        if candidate_cells is not None:
            hits.intersection_update(candidate_cells)
        centers = {cell: cell_to_world(self.map[0].info, cell) for cell in hits}
        if reference_cells is not None:
            info = self.map[0].info
            corridor = inflate_cells(set(reference_cells), info.width, info.height,
                                    max(int(self.get_parameter("obstacle_inflation_cells").value)+1,
                                        ceil((float(self.get_parameter("robot_radius").value)+.05)
                                             / info.resolution)))
            nearby = hits & corridor
        else:
            nearby = {cell for cell, p in centers.items() if hypot(p[0]-x, p[1]-y) < 1.5}
        if not nearby:
            return False
        closest = min(nearby, key=lambda cell: hypot(centers[cell][0]-x, centers[cell][1]-y))
        cx, cy = centers[closest]
        # Track the locally blocking object rather than unrelated room objects.
        tracked = {cell for cell in nearby if hypot(centers[cell][0]-cx, centers[cell][1]-cy) <= .60}
        # The retry budget belongs to the retained goal, but removal evidence
        # belongs to its current blocker. A previously passed object behind the
        # robot must not make the current object's removal impossible to prove.
        self.obstacle_hold.blocked_cells.intersection_update(tracked)
        if failure_id is not None:
            self.obstacle_failure_cells[failure_id] = set(tracked)
        live_ids = {entry["id"] for entry in self.failure_memory.snapshot(
            now=time.monotonic())["entries"]}
        self.obstacle_failure_cells = {key: cells for key, cells in self.obstacle_failure_cells.items()
                                       if key in live_ids}
        info = self.map[0].info
        neighborhood = inflate_cells(tracked, info.width, info.height,
                                     max(1, ceil(.15/info.resolution)))
        self.obstacle_episode_failure_ids = {key for key, cells in self.obstacle_failure_cells.items()
                                            if cells & neighborhood}
        self.obstacle_hold.retry_limit = max(1, int(self.get_parameter("obstacle_retry_limit").value))
        self.obstacle_hold.clear_duration = max(.5, float(self.get_parameter("obstacle_clear_duration").value))
        before = self.obstacle_hold.attempts
        holding = self.obstacle_hold.register_failure(tracked, now=time.monotonic())
        self.plan_metrics.update(failure_class="obstacle",
            recovery_stage="await_clearance" if holding else "hard_replan")
        self.get_logger().warning(
            f"Observed blocker retry {before}->{self.obstacle_hold.attempts}/"
            f"{self.obstacle_hold.retry_limit}: cells={sorted(tracked)} "
            f"source={'execution_failure' if failure_id is not None else 'no_safe_alternative'} "
            f"holding={holding}")
        if not holding:
            return False
        self.failure_state, self.failure_reason = "OBSTACLE_HOLD", (
            "장애물이 있어 경로 주행을 잠시 멈춥니다. 장애물을 치우면 경로를 재개합니다.")
        self._invalidate_path(self.failure_reason)
        return True

    def on_visibility(self, msg):
        if not self._goal_is_active() or self.map is None:
            return
        now = time.monotonic()
        stamp = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        current = self.map[0]
        matching = (msg.header.frame_id == current.header.frame_id
                    and msg.info.width == current.info.width and msg.info.height == current.info.height
                    and msg.info.resolution == current.info.resolution
                    and msg.info.origin == current.info.origin
                    and len(msg.data) == current.info.width*current.info.height)
        source_fresh = (matching and stamp > self.last_visibility_stamp_ns and stamp > 0
                        and stamp >= getattr(self, "start_scan_min_source_ns", 0)
                        and stamp not in getattr(self, "start_scan_retired_stamps", ())
                        and -.25e9 <= self.get_clock().now().nanoseconds-stamp < .75e9)
        if source_fresh:
            self.last_visibility_stamp_ns, self.last_visibility_at = stamp, now
            self._record_start_visibility(stamp, now)
            if self._recovery_wait_owns_retry():
                current_hits = {(index % msg.info.width, index // msg.info.width)
                                for index, value in enumerate(msg.data) if value > 0}
                if (self._fresh_motion_pose() is not None and self._automatic_resume_allowed()
                        and self._confirmed_wait_change(current_hits, now)):
                    self.observed.update(current_hits)
                    if self._resume_recovery_wait(moving_obstacle_confirmed=True):
                        return
            if (self._live_route_required_scans() > 1 and self.planned_cells
                    and not self.waiting_for_replan):
                # This grid marks only *this* scan's occupied cells. In
                # contrast, /observed_obstacles repeats cached hits for its
                # TTL and cannot prove multiple independent detections.
                current_hits = {(index % msg.info.width, index // msg.info.width)
                                for index, value in enumerate(msg.data) if value > 0}
                blockers = self._live_route_blockers(current_hits)
                if blockers and self._live_route_block_confirmed(blockers):
                    self.observed.update(current_hits)
                    self._react_to_live_route_blockers(blockers)
                elif not blockers:
                    self._clear_pending_live_route()
        if not self.obstacle_hold.holding:
            return
        pose_fresh = source_fresh and self._fresh_motion_pose() is not None
        mux_ready = pose_fresh and self._automatic_resume_allowed()
        fresh = bool(source_fresh and pose_fresh and mux_ready)
        if not fresh:
            self.get_logger().info(
                f"Obstacle hold: free-space evidence unavailable "
                f"map_geometry={matching} source_fresh={source_fresh} "
                f"pose_fresh={pose_fresh} mux_ready={mux_ready}",
                throttle_duration_sec=1.0)
            self.obstacle_hold.observe_free(set(), set(), fresh=False, now=now)
            return
        width = msg.info.width
        # Only inspect the bounded tracked object, not an entire map-sized set.
        tracked = set(self.obstacle_hold.blocked_cells)
        if any(not self.map[1].contains(cell) for cell in tracked):
            self.obstacle_hold.observe_free(set(), set(), fresh=False, now=now)
            return
        free = {cell for cell in tracked if msg.data[cell[1]*width+cell[0]] == 0}
        # Small ordinary AMCL/raster shifts cannot make a nearby stationary
        # object appear removed merely by moving its endpoint to the next cell.
        radius = max(1, ceil(.10/msg.info.resolution))
        neighbors = inflate_cells(tracked, width, msg.info.height, radius)
        near_hits = {cell for cell in neighbors if msg.data[cell[1]*width+cell[0]] > 0}
        occupied = tracked if near_hits else set()
        if not tracked <= free or occupied:
            self.get_logger().info(
                f"Obstacle hold: tracked={sorted(tracked)} free={sorted(free)} "
                f"near_hits={sorted(near_hits)[:8]}", throttle_duration_sec=1.0)
        if not self.obstacle_hold.observe_free(free, occupied, fresh=True, now=now):
            return
        self.get_logger().info("Obstacle hold: measured free-space interval completed; retrying retained goal")
        # These cells were positively traversed by fresh valid LiDAR rays for
        # two seconds. Clear only failure IDs associated with this same object.
        self.failure_memory.release_verified(self.obstacle_episode_failure_ids, now=now)
        for key in self.obstacle_episode_failure_ids:
            self.obstacle_failure_cells.pop(key, None)
        self.obstacle_episode_failure_ids.clear()
        self.observed.difference_update(free)
        self.remembered.difference_update(free)
        self.recovery_attempts, self.recovery_attempted_cells = 0, set()
        self.rejoin_anchor, self.rejoin_verified = None, False
        self.failure_state, self.failure_reason = "CLEAR", "장애물 제거 확인 · 기존 목표로 경로를 재생성합니다."
        self.last_plan_time = float("-inf")
        if self._recovery_wait_owns_retry():
            self._resume_recovery_wait(positive_free=True)
            return
        self._request_replan(self.failure_reason)

    def _memory_scopes(self):
        if self.map is None or self.last_goal is None:
            return "", ""
        msg, _ = self.map
        origin = msg.info.origin
        # Map updates do not erase a failed approach. Geometry/frame changes
        # naturally isolate it; goal resends use coordinates, not request stamps.
        map_scope = repr((msg.header.frame_id, msg.info.width, msg.info.height,
                          msg.info.resolution, origin.position.x, origin.position.y,
                          origin.orientation.x, origin.orientation.y,
                          origin.orientation.z, origin.orientation.w))
        goal = self.last_goal
        return map_scope, repr((goal.header.frame_id or self.get_parameter("map_frame").value,
                                goal.pose.position.x, goal.pose.position.y))

    def _failure_constraints(self, grid):
        if not self.get_parameter("failure_memory_enabled").value:
            return set(), {}
        info = self.map[0].info
        geometry = GridGeometry(info.resolution, info.origin.position.x,
                                info.origin.position.y, yaw_from_quaternion(info.origin.orientation))
        map_scope, goal_scope = self._memory_scopes()
        failed, costs = self.failure_memory.constraints(grid, geometry, now=time.monotonic(),
                                                       map_scope=map_scope, goal_scope=goal_scope)
        # Execution failure proves that an approach failed, not that space is
        # occupied. Only measured/static/remembered/unknown cells and certified
        # first-edge geometry may prohibit motion. Keep every failed approach
        # as the existing bounded directional cost, never a hard graph edge.
        penalty = float(self.get_parameter("failure_memory_penalty").value)
        for edge in failed:
            costs[edge] = max(costs.get(edge, 0.), penalty)
        return set(), costs

    def _controller_retry_policy(self, failure_costs):
        """Count accepted terminal event IDs; failure history is never occupancy.

        Physical recovery ownership is decided once by the actual attempted
        body corridor. Historical side-wall proximity, missing old segments,
        or an unrelated later scan cannot create a second motion policy here.
        """
        episode = self.recovery_coordinator.current
        if episode is not None and episode.phase is not RecoveryPhase.FOLLOW:
            return bool(failure_costs), failure_costs
        self.controller_retry_events.update(self.controller_pending_retry_events)
        self.controller_pending_retry_events.clear()
        attempts = len(self.controller_retry_events)
        if not attempts and not failure_costs:
            return False, failure_costs
        limit = max(1, int(self.get_parameter("controller_retry_limit").value))
        self.plan_metrics.update(failure_class="controller", recovery_stage="soft_replan",
            controller_retry_count=attempts, controller_retry_limit=limit,
            controller_failure_evidence="Accepted execution events are directional costs, not occupancy")
        if attempts >= limit:
            self.failure_state = "CONTROLLER_HOLD"
            self.failure_reason = (
                "Controller retries exhausted without a confirmed obstacle; "
                "goal retained, operator restart required")
            self.plan_metrics.update(recovery_stage="await_operator")
            self._invalidate_path(self.failure_reason)
            self.waiting_for_replan = False
            self.publish_status("CONTROLLER_HOLD", self.failure_reason)
            return False, failure_costs
        return bool(failure_costs), failure_costs

    def on_localization_health(self, msg):
        try:
            payload = json.loads(msg.data)
            if isinstance(payload, dict):
                self.localization_health, self.latest_health_at = payload, time.monotonic()
        except (ValueError, TypeError):
            pass

    def on_scan(self, msg):
        now = time.monotonic()
        stamp = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        previous_ros = getattr(self, "start_scan_last_ros_ns", None)
        if previous_ros is not None and self.get_clock().now().nanoseconds < previous_ros:
            self._clear_start_scan_history()
        previous = getattr(self, "start_scan_last_source_ns", None)
        frame = getattr(self, "start_scan_frame", None)
        if frame is not None and frame != msg.header.frame_id:
            self._clear_start_scan_history()
            previous = getattr(self, "start_scan_last_source_ns", None)
        if previous is not None and stamp <= previous:
            # Repeated source time cannot renew either FIRST receipt. A raw
            # source rewind alone is not proof that the whole ROS epoch reset.
            if stamp < previous:
                self._clear_start_scan_history()
            return
        if stamp <= 0 or not msg.header.frame_id:
            self._clear_start_scan_history()
            return
        history = getattr(self, "start_scan_history", None)
        if history is None:
            history = self.start_scan_history = deque(maxlen=8)
        if history and now < history[-1][1]:
            self._clear_start_scan_history()
            history = self.start_scan_history
        if (stamp < getattr(self, "start_scan_min_source_ns", 0)
                or stamp in getattr(self, "start_scan_retired_stamps", ())):
            return
        self.latest_scan, self.latest_scan_at = msg, now
        self.start_scan_last_source_ns, self.start_scan_frame = stamp, msg.header.frame_id
        self.start_scan_last_ros_ns = self.get_clock().now().nanoseconds
        history.append((stamp, now, msg))

    def _clear_start_scan_history(self):
        if (getattr(self, "active_start_certificate", None) is not None
                or getattr(self, "completed_start_prefix", None) is not None):
            self.start_scan_context_invalid = True
            self._clear_start_certificate("raw_scan_input_epoch_or_context_changed")
        now_ns = self.get_clock().now().nanoseconds
        # Queued messages from the previous context must not acquire new
        # FIRST receipts after a frame/clock/map/odometry reset. Bounded by
        # the two eight-event histories; repeated resets retain that barrier.
        retired = deque(getattr(self, "start_scan_retired_stamps", ()), maxlen=32)
        for event in (*getattr(self, "start_scan_history", ()),
                      *getattr(self, "start_visibility_history", ())):
            if event[0] not in retired:
                retired.append(event[0])
        self.start_scan_retired_stamps = retired
        previous_ros = getattr(self, "start_scan_last_ros_ns", None)
        if previous_ros is not None and now_ns < previous_ros:
            # A demonstrated ROS clock reset permits new low source stamps;
            # ordinary map/frame resets keep the old source high-water marks.
            self.start_scan_last_source_ns = None
            self.last_visibility_stamp_ns = 0
        self.start_scan_frame = None
        self.start_scan_min_source_ns = now_ns
        self.start_scan_last_ros_ns = now_ns
        self.start_scan_history = deque(maxlen=8)
        self.start_visibility_history = deque(maxlen=8)
        self.last_visibility_at = None

    def _record_start_visibility(self, stamp, received):
        """Record only the original source stamp of a received projected grid."""
        if (stamp <= 0 or stamp < getattr(self, "start_scan_min_source_ns", 0)
                or stamp in getattr(self, "start_scan_retired_stamps", ())):
            return
        history = getattr(self, "start_visibility_history", None)
        if history is None:
            history = self.start_visibility_history = deque(maxlen=8)
        if history and stamp <= history[-1][0]:
            return  # Duplicate/reordered projection never renews its receipt.
        history.append((stamp, received))

    def _coherent_start_scan(self):
        """Newest fully paired projected/raw event with its acquisition TF.

        Both original FIRST receipts and the shared source stamp must remain
        within 250 ms. A newer raw stream without a projected counterpart can
        never replace this proof; current cells and the robot mux guard remain
        independently mandatory. The same rule applies to new and owned edges.
        """
        now, ros_ns = time.monotonic(), self.get_clock().now().nanoseconds
        latest = getattr(self, "latest_scan", None)
        frame = None if latest is None else latest.header.frame_id
        diagnostic = {"latest_visibility_stamp_ns": getattr(self, "last_visibility_stamp_ns", 0),
                      "selected_stamp_ns": None, "reason": "no_fresh_paired_event"}
        if latest is not None:
            diagnostic["newest_raw_scan_stamp_ns"] = (
                latest.header.stamp.sec*1_000_000_000+latest.header.stamp.nanosec)
        self.plan_metrics["start_scan_selection"] = diagnostic
        history = getattr(self, "start_scan_history", ())
        if ((history and now < history[-1][1])
                or ros_ns < getattr(self, "start_scan_last_ros_ns", ros_ns)):
            self._clear_start_scan_history()
            diagnostic["reason"] = "raw_scan_clock_epoch_changed"
            return None
        if not frame:
            diagnostic["reason"] = "raw_scan_frame_missing"
            return None
        projected = {stamp: received for stamp, received in
            getattr(self, "start_visibility_history", ())
            if 0 <= now-received < .25 and -.25e9 <= ros_ns-stamp < .25e9}
        candidates = [(stamp, received, scan) for stamp, received, scan in
            getattr(self, "start_scan_history", ())
            if stamp in projected and scan.header.frame_id == frame
            and stamp == scan.header.stamp.sec*1_000_000_000+scan.header.stamp.nanosec
            and 0 <= now-received < .25 and -.25e9 <= ros_ns-stamp < .25e9]
        for stamp, received, scan in reversed(candidates):
            if (len(scan.ranges) < 2 or not all(isfinite(value) for value in
                    (scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max))
                    or scan.angle_increment == 0 or not 0 <= scan.range_min < scan.range_max):
                diagnostic["reason"] = "paired_scan_geometry_invalid"
                return None
            valid = sum((isfinite(value) and scan.range_min <= value <= scan.range_max)
                        or (value == float("inf")) for value in scan.ranges)
            if valid / len(scan.ranges) < .8:
                diagnostic["reason"] = "paired_scan_coverage_invalid"
                return None
            try:
                transform = self.tf_buffer.lookup_transform(self.get_parameter("map_frame").value,
                    scan.header.frame_id, rclpy.time.Time.from_msg(scan.header.stamp))
                origin, rotation = transform.transform.translation, transform.transform.rotation
                heading = _normalized_pose_yaw(rotation)
                if not all(isfinite(value) for value in (origin.x, origin.y, heading)):
                    raise ValueError("Invalid acquisition transform geometry")
            except TransformException as exc:
                diagnostic.update(reason="paired_acquisition_tf_unavailable",
                    lookup_error=(type(exc).__name__+": "+str(exc))[:240])
                continue
            except (AttributeError, ValueError, TypeError) as exc:
                diagnostic.update(reason="paired_acquisition_transform_invalid",
                    lookup_error=(type(exc).__name__+": "+str(exc))[:240])
                return None
            diagnostic.update(reason="coherent_event_ready", selected_stamp_ns=stamp,
                source_age_s=(ros_ns-stamp)*1e-9, raw_first_receipt_age_s=now-received,
                visibility_first_receipt_age_s=now-projected[stamp])
            return scan, transform
        return None

    def _motion_pose_input_diagnostic(self, now, ros_ns, health, odom, scan):
        """Describe the existing input gates without authorizing any motion."""
        sent = health.get("sent_at_ns")
        stamp = (None if scan is None else scan.header.stamp.sec*1_000_000_000
                 + scan.header.stamp.nanosec)
        def age(value, current):
            return current-value if isinstance(value, (int, float)) and isfinite(value) else None
        def source_age(value):
            elapsed = age(value, ros_ns)
            return elapsed/1e9 if elapsed is not None else None
        detail = {"first_failed_reason": "INPUTS_VALID", "node_clock_ns": ros_ns,
            "checked_monotonic": now, "health_stamp_ns": sent if type(sent) is int else None,
            "health_receipt_age_s": age(self.latest_health_at, now),
            "health_source_age_s": source_age(sent) if type(sent) is int else None,
            "odom_stamp_ns": odom[1] if odom is not None and type(odom[1]) is int else None,
            "odom_receipt_age_s": age(odom[0], now) if odom is not None else None,
            "odom_source_age_s": source_age(odom[1]) if odom is not None else None,
            "scan_stamp_ns": stamp, "scan_receipt_age_s": age(self.latest_scan_at, now),
            "scan_source_age_s": source_age(stamp),
            "tf_stamp_ns": None, "tf_source_age_s": None, "lookup_error": ""}
        # Order is identical to _fresh_motion_pose's unchanged short-circuit
        # predicate. Lambdas preserve its missing-input short circuit as well.
        checks = (
            ("health_missing", lambda: self.latest_health_at is None),
            ("health_receipt_age", lambda: not 0 <= now-self.latest_health_at < 1.0),
            ("health_stamp_invalid", lambda: isinstance(sent, bool) or not isinstance(sent, int)),
            ("health_source_age", lambda: not -.25e9 <= ros_ns-sent < 1e9),
            ("health_level", lambda: health.get("level") != "ok"),
            ("autonomy_not_allowed", lambda: health.get("autonomous_allowed") is not True),
            ("health_stop_latched", lambda: health.get("stop_latched") is True),
            ("odom_missing", lambda: odom is None),
            ("odom_receipt_age", lambda: not 0 <= now-odom[0] < .75),
            ("odom_source_age", lambda: not -.25e9 <= ros_ns-odom[1] < 1e9),
            ("scan_missing", lambda: scan is None or self.latest_scan_at is None),
            ("scan_receipt_age", lambda: not 0 <= now-self.latest_scan_at < .75),
            ("scan_stamp_invalid", lambda: stamp <= 0),
            ("scan_source_age", lambda: not -.25e9 <= ros_ns-stamp < 1e9),
            ("scan_empty", lambda: not scan.ranges),
            ("scan_range_max", lambda: not isfinite(scan.range_max) or scan.range_max <= 0),
            ("scan_range_min", lambda: not isfinite(scan.range_min) or not 0 <= scan.range_min < scan.range_max),
            ("scan_angles", lambda: not isfinite(scan.angle_min) or not isfinite(scan.angle_increment)
             or scan.angle_increment == 0),
            ("scan_coverage", lambda: sum((isfinite(v) and scan.range_min <= v <= scan.range_max)
             or v == float("inf") for v in scan.ranges)/len(scan.ranges) < .8),
        )
        for reason, failed in checks:
            if failed():
                detail["first_failed_reason"] = reason
                break
        return detail

    def _fresh_motion_pose(self):
        """Independent receipt/source age checks; a replay is not fresh motion."""
        now, ros_ns = time.monotonic(), self.get_clock().now().nanoseconds
        health, odom, scan = self.localization_health or {}, self.latest_odom, self.latest_scan
        sent = health.get("sent_at_ns")
        scan_stamp = (0 if scan is None else scan.header.stamp.sec*1_000_000_000
                      + scan.header.stamp.nanosec)
        diagnostic = self._motion_pose_input_diagnostic(now, ros_ns, health, odom, scan)
        self.plan_metrics["motion_pose_diagnostics"] = diagnostic
        if (self.latest_health_at is None or not 0 <= now-self.latest_health_at < 1.0
                or isinstance(sent, bool) or not isinstance(sent, int)
                or not -.25e9 <= ros_ns-sent < 1e9
                or health.get("level") != "ok" or health.get("autonomous_allowed") is not True
                or health.get("stop_latched") is True or odom is None
                or not 0 <= now-odom[0] < .75 or not -.25e9 <= ros_ns-odom[1] < 1e9
                or scan is None or self.latest_scan_at is None
                or not 0 <= now-self.latest_scan_at < .75
                or scan_stamp <= 0 or not -.25e9 <= ros_ns-scan_stamp < 1e9
                or not scan.ranges or not isfinite(scan.range_max) or scan.range_max <= 0
                or not isfinite(scan.range_min) or not 0 <= scan.range_min < scan.range_max
                or not isfinite(scan.angle_min) or not isfinite(scan.angle_increment)
                or scan.angle_increment == 0
                or sum((isfinite(v) and scan.range_min <= v <= scan.range_max)
                       or v == float("inf") for v in scan.ranges)/len(scan.ranges) < .8):
            return None
        try:
            tf = self.tf_buffer.lookup_transform(self.get_parameter("map_frame").value,
                    self.get_parameter("base_frame").value, rclpy.time.Time())
            stamp = tf.header.stamp.sec*1_000_000_000 + tf.header.stamp.nanosec
            p, q = tf.transform.translation, tf.transform.rotation
            diagnostic.update(tf_stamp_ns=stamp, tf_source_age_s=(ros_ns-stamp)/1e9)
            if (stamp <= 0 or not -.25e9 <= ros_ns-stamp < 1e9
                    or not all(isfinite(v) for v in (p.x, p.y, q.x, q.y, q.z, q.w))
                    or not .8 <= q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w <= 1.2):
                diagnostic["first_failed_reason"] = (
                    "tf_stamp_invalid" if stamp <= 0 else
                    "tf_source_age" if not -.25e9 <= ros_ns-stamp < 1e9 else
                    "tf_geometry")
                return None
            self.last_fresh_motion_pose_stamp_ns = stamp
            result = float(p.x), float(p.y), _normalized_pose_yaw(q)
            diagnostic.update(first_failed_reason="OK", pose=list(result))
            return result
        except (TransformException, ValueError, AttributeError) as exc:
            diagnostic.update(first_failed_reason="tf_lookup_or_geometry_exception",
                              lookup_error=(type(exc).__name__+": "+str(exc))[:200])
            return None

    def on_execution_failure(self, msg):
        if (not self._goal_is_active() or not self.get_parameter("failure_memory_enabled").value):
            return
        # A terminal result may arrive before the latched map on startup.
        # Without map geometry there is no safe scope or corridor for its
        # failure memory, so retain the goal and wait for /map instead.
        if self.map is None:
            self._invalidate_path("Waiting for /map after execution failure")
            return
        try:
            data = json.loads(msg.data)
            episode=self.recovery_coordinator.current
            if ((episode is not None and episode.phase is not RecoveryPhase.FOLLOW)
                    or not self._status_owns_path(data,allow_invalidated=True)):
                return
            event_id, created = data.get("event_id"), data.get("created_at_ns")
            if (type(data.get("schema_version")) is not int or data["schema_version"] != 1
                    or data.get("terminal_confirmed") is not True
                    or not isinstance(event_id, str) or not 1 <= len(event_id) <= 128
                    or event_id in self.seen_failure_ids or not self._status_matches_goal(data)
                    or data.get("reason") not in EXECUTION_FAILURE_REASONS
                    or isinstance(created, bool) or not isinstance(created, int)
                    or not -.25e9 <= self.get_clock().now().nanoseconds-created
                    <= float(self.get_parameter("failure_event_max_age").value)*1e9):
                return
            pose, segment = data["pose"], data["segment"]
            values = (pose["x"], pose["y"], pose["yaw"], segment["heading"],
                      segment["start"]["x"], segment["start"]["y"],
                      segment["end"]["x"], segment["end"]["y"])
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
                   for v in values):
                return
            current = self._fresh_motion_pose()
            if current is None or hypot(current[0]-pose["x"], current[1]-pose["y"]) > .35:
                self.failure_state, self.failure_reason = "SENSOR_HOLD", "Failure evidence is not fresh"
                return
        except (ValueError, TypeError, KeyError, AttributeError):
            return
        # Place the restriction ahead of the stopped body, along the attempted
        # route. Retreat and departure in a different direction remain possible.
        heading = segment["heading"]
        map_scope, goal_scope = self._memory_scopes()
        self.last_failure_id = self.failure_memory.record_failure(
            x=pose["x"]+.20*cos(heading), y=pose["y"]+.20*sin(heading),
            approach_heading=heading, reason=data["reason"], now=time.monotonic(),
            map_scope=map_scope, goal_scope=goal_scope, failure_id=event_id)
        live_ids = {entry["id"] for entry in self.failure_memory.snapshot(
            now=time.monotonic())["entries"]}
        self.execution_failure_segments = {
            key: segments for key, segments in self.execution_failure_segments.items()
            if key in live_ids}
        attempted = ((float(segment["start"]["x"]), float(segment["start"]["y"])),
                     (float(segment["end"]["x"]), float(segment["end"]["y"])))
        segments = self.execution_failure_segments.setdefault(self.last_failure_id, [])
        if attempted not in segments:
            segments.append(attempted)
            del segments[:-16]  # Same bounded history as a coalesced entry's events.
        self.seen_failure_ids.append(event_id)
        limit = max(1, int(self.get_parameter("controller_retry_limit").value))
        if len(self.controller_retry_events) + len(self.controller_pending_retry_events) < limit:
            self.controller_pending_retry_events.add(event_id)
        self.last_failure_event_stamp_ns = created
        self.pending_failure_id = self.pending_failure_since = None
        self.rejoin_anchor, self.rejoin_verified = None, False
        self.failure_state, self.failure_reason = "AVOIDING", data["reason"]
        self.plan_metrics.update(failure_class="execution", recovery_stage="classify_failure",
            controller_failure_evidence="classification pending")
        self._invalidate_path("Execution failure remembered; seeking a different approach")
        # A Nav2 progress abort is not itself proof that a physical object
        # blocked the chassis. Only count an observed hit intersecting the
        # attempted local segment (with the configured footprint margin).
        if self.map is not None:
            info = self.map[0].info
            start_cell = world_to_cell(info, segment["start"]["x"], segment["start"]["y"])
            end_cell = world_to_cell(info, segment["end"]["x"], segment["end"]["y"])
            attempted_cells = route_cells([(start_cell[0]+.5, start_cell[1]+.5),
                                           (end_cell[0]+.5, end_cell[1]+.5)])
            # The confirmed terminal already stopped the old action. Its
            # invalidation deliberately cleared the live path, not the matched
            # original reference. A measured obstacle in this exact attempted
            # body corridor may repair that reference before global replanning.
            # Missing/clear sensor evidence retains the controller policy above.
            blockers, raw_hits = self._terminal_obstacle_blockers(attempted)
            if blockers:
                self.controller_pending_retry_events.discard(event_id)
                self.controller_retry_events.discard(event_id)
                self.obstacle_failure_cells[self.last_failure_id] = set(blockers)
                if self._start_physical_recovery(blockers,
                        "Physical execution blocker remains after local repair",
                        terminal_repair=True, raw_hits=raw_hits):
                    return
            edges, costs = self._failure_constraints(self.map[1])
            self._controller_retry_policy(costs)
            if self.failure_state == "CONTROLLER_HOLD":
                return
        self.last_plan_time = float("-inf")
        self._request_replan("Execution failure recorded")

    def _remaining_path_cells(self):
        """Only newly blocked *remaining* route cancels a healthy active action."""
        points = self.planned_world_points
        if len(points) < 2 or self.map is None:
            return self.planned_cells
        try:
            tf = self.tf_buffer.lookup_transform(self.get_parameter("map_frame").value,
                    self.get_parameter("base_frame").value, rclpy.time.Time())
            stamp = tf.header.stamp.sec*1_000_000_000 + tf.header.stamp.nanosec
            if not -.25e9 <= self.get_clock().now().nanoseconds-stamp < 1e9:
                return self.planned_cells
            p = tf.transform.translation
            progress = max(0., path_progress(points, (p.x, p.y))
                           -float(self.get_parameter("robot_radius").value)-.05)
            remaining = [point_at_progress(points, progress)]
            traveled = 0.
            for a, b in zip(points, points[1:]):
                traveled += hypot(b[0]-a[0], b[1]-a[1])
                if traveled >= progress:
                    remaining.append(b)
            info = self.map[0].info
            c, s = cos(yaw_from_quaternion(info.origin.orientation)), sin(yaw_from_quaternion(info.origin.orientation))
            def local(point):
                x, y = point[0]-info.origin.position.x, point[1]-info.origin.position.y
                return ((c*x+s*y)/info.resolution, (-s*x+c*y)/info.resolution)
            return list(route_cells([local(point) for point in remaining]))
        except (TransformException, AttributeError, ValueError):
            return self.planned_cells

    def _update_rejoin(self):
        self._update_local_rejoin()
        anchor = self.rejoin_anchor
        if anchor is None or self.rejoin_verified or not self._goal_is_active():
            return
        pose = self._fresh_motion_pose()
        if pose is None:
            return
        points, odom = anchor["points"], self.latest_odom
        if len(points) < 2 or odom[5] != anchor["odom"][5]:
            return
        progress = path_progress(points, pose[:2])
        projection = point_at_progress(points, progress)
        distance = float(self.get_parameter("rejoin_progress_distance").value)
        # Compare a chassis displacement in both coordinate systems; neither
        # an AMCL correction nor an odometry reset alone proves a safe rejoin.
        ox, oy = odom[2]-anchor["odom"][2], odom[3]-anchor["odom"][3]
        rotation = anchor["pose"][2]-anchor["odom"][4]
        ex, ey = cos(rotation)*ox-sin(rotation)*oy, sin(rotation)*ox+cos(rotation)*oy
        error = hypot(pose[0]-anchor["pose"][0]-ex, pose[1]-anchor["pose"][1]-ey)
        if (progress-anchor["progress"] >= distance
                and hypot(odom[2]-anchor["odom"][2], odom[3]-anchor["odom"][3]) >= min(.25, distance)
                and odom[1] > anchor["odom"][1]
                and error <= .05+.20*hypot(ox, oy)
                and self.navigation_progress_at is not None
                and time.monotonic()-self.navigation_progress_at < 1.0
                and hypot(pose[0]-projection[0], pose[1]-projection[1])
                    <= float(self.get_parameter("rejoin_cross_track_tolerance").value)):
            if not self.rejoin_verified:
                self.controller_retry_events.clear()
                self.controller_pending_retry_events.clear()
                self.plan_metrics.update(controller_retry_count=0, recovery_stage="tracking")
            self.rejoin_verified = True
            self.recovery_attempts, self.recovery_attempted_cells = 0, set()
            self.failure_state, self.failure_reason = "AVOIDING", "Alternative route progress verified"

    def publish_failure_status(self):
        # Steady heartbeat also expires memory while simulation /clock is paused.
        try:
            settings = dict(ttl_sec=float(self.get_parameter("failure_memory_ttl").value),
                radius_m=float(self.get_parameter("failure_memory_radius").value),
                heading_tolerance_rad=float(self.get_parameter("failure_memory_heading_tolerance").value),
                max_entries=int(self.get_parameter("failure_memory_max_entries").value),
                penalty_cost=float(self.get_parameter("failure_memory_penalty").value))
            validated = FailureMemory(**settings)
            for name in settings:
                setattr(self.failure_memory, name, getattr(validated, name))
            map_scope, goal_scope = self._memory_scopes()
            snapshot = self.failure_memory.snapshot(now=time.monotonic(),
                map_scope=map_scope, goal_scope=goal_scope)
            self._update_rejoin()
            active = [{**entry, "expires_in": entry["expires_in_sec"],
                       "failures": entry["occurrences"]} for entry in snapshot["entries"]]
            state = self.failure_state
            if not self._goal_is_active() or not self.get_parameter("failure_memory_enabled").value:
                state = "CLEAR"
                active = []
            elif active and self._fresh_motion_pose() is None:
                state = "SENSOR_HOLD"
            elif self.recovery_active:
                state = "RECOVERING"
            elif self.obstacle_hold.holding:
                state = "OBSTACLE_HOLD" if self._fresh_motion_pose() is not None else "SENSOR_HOLD"
            elif (not active and self.rejoin_anchor is None
                  and state in ("CLEAR", "AVOIDING")):
                state = "CLEAR"
            packet = String()
            goal = self.last_goal
            packet.data = json.dumps({"state": state, "reason": self.failure_reason,
                "count": len(active), "active": active, "last_failure_id": self.last_failure_id,
                "rejoin_verified": self.rejoin_verified, "recovery_attempts": self.recovery_attempts,
                "goal_stamp": None if goal is None else {"sec": goal.header.stamp.sec,
                                                         "nanosec": goal.header.stamp.nanosec},
                "sent_at_ns": self.get_clock().now().nanoseconds})
            detail = json.loads(packet.data)
            hold = self.obstacle_hold.snapshot(now=time.monotonic())
            detail.update(retry_count=hold["attempts"], retry_limit=hold["retry_limit"],
                blocked_obstacle_count=hold["tracked_cell_count"],
                clear_wait_seconds=max(0., hold["clear_duration"]-hold["clear_elapsed"]))
            detail.update(self._failure_diagnostics())
            packet.data = json.dumps(detail)
            self.failure_status_pub.publish(packet)
            if (self._goal_is_active() and self.planned_world_points and not self.waiting_for_replan
                    and self.plan_metrics.get("local_detour_state") == "ACTIVE"):
                self.publish_status("READY", "Local detour active; original forward suffix retained")
        except (ValueError, TypeError):
            self.get_logger().warning("Invalid failure-memory limits; keeping last valid settings",
                                      throttle_duration_sec=5.)

    def _retry_period(self):
        # Keep retry rate bounded even if a mistaken runtime parameter is zero.
        return max(0.25, float(self.get_parameter("replan_retry_period").value))

    def retry_transform(self):
        """Retry no-path, missing-map and missing-TF conditions without losing the goal."""
        if self._recovery_wait_owns_retry():
            if (self.last_visibility_at is None or time.monotonic()-self.last_visibility_at>.5
                    or self._fresh_motion_pose() is None or not self._automatic_resume_allowed()):
                self.obstacle_hold.observe_free(set(),set(),fresh=False,now=time.monotonic())
            return
        if self.failure_state == "CONTROLLER_HOLD":
            return
        if self.obstacle_hold.holding:
            now = time.monotonic()
            if (self.last_visibility_at is None or now-self.last_visibility_at > .5
                    or self._fresh_motion_pose() is None or not self._automatic_resume_allowed()):
                self.obstacle_hold.observe_free(set(), set(), fresh=False, now=now)
            return
        if self.pending_failure_id is not None:
            # FAILED and the reliable event use different topics. Never retry
            # the old route while its spatial evidence is still uncorrelated.
            if (self.pending_failure_since is None
                    or time.monotonic()-self.pending_failure_since < 3.0
                    or self._fresh_motion_pose() is None):
                return
            # Lost/rejected evidence cannot poison the map or permanently lose
            # the mission. Retry once sensors recover, without inventing memory.
            self.pending_failure_id = self.pending_failure_since = None
            self.failure_state, self.failure_reason = "WAIT_ALTERNATIVE", "Unconfirmed failure; retrying with fresh sensors"
            # Loss of the reliable diagnostic topic is not permission to skip
            # a currently measured physical blocker on the retained reference.
            pose = self._fresh_motion_pose()
            if (pose is not None and self._observations_ready()
                    and self._automatic_resume_allowed() and len(self.local_reference_points) >= 2):
                progress = path_progress(self.local_reference_points, pose[:2])
                attempted = (pose[:2], point_at_progress(self.local_reference_points, progress+.3))
                blockers, raw_hits = self._terminal_obstacle_blockers(attempted)
                if blockers and self._start_physical_recovery(blockers,
                        "Fresh physical blocker after missing terminal diagnosis",
                        terminal_repair=True, raw_hits=raw_hits):
                    return
        if self.map is not None:
            signature = (self.get_parameter("occupied_threshold").value,
                         self.get_parameter("obstacle_inflation_cells").value,
                         self.get_parameter("mapped_wall_return_tolerance_cells").value)
            if signature != getattr(self, "map_signature", None):
                self.on_map(self.map[0])
        if self._process_pending_reverse():
            return
        geometry = tuple(self.get_parameter(name).value for name in (
            "path_smoothing_enabled", "path_smoothing_iterations",
            "path_smoothing_shortcut_distance", "path_resample_spacing"))
        if geometry != getattr(self, "geometry_signature", None):
            self.geometry_signature = geometry
            if self._goal_is_active():
                self._request_replan("Path geometry settings changed")
        if self._goal_is_active() and not getattr(self, "recovery_active", False):
            self._record_breadcrumb()
        if getattr(self, "recovery_active", False):
            requested = self.recovery_requested_at
            # The follower has a 5 s cancel wait plus a 15 s BackUp watchdog.
            # A missing terminal cannot prove the BackUp action stopped.
            # Retain its ownership and request cancellation; only a matching
            # terminal may advance to GLOBAL. Never race another drive action.
            if requested is not None and time.monotonic() - requested > 25.0:
                if not self.recovery_cancel_due_map_change:
                    self.recovery_cancel_due_map_change = True
                    goal=self.last_goal
                    self.recovery_pub.publish(String(data=json.dumps({
                        "goal_id": f"{goal.header.stamp.sec}:{goal.header.stamp.nanosec}",
                        "episode_id": self.active_recovery_episode_id,
                        "cancel": True, "reason": "Reverse terminal response timeout"})))
                    self._invalidate_path("Waiting for confirmed reverse cancellation; goal retained")
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
        if self._recovery_wait_owns_retry():
            return
        if (not self._goal_is_active() or getattr(self, "recovery_active", False)
                or self.pending_failure_id is not None or self.obstacle_hold.holding
                or self.failure_state == "CONTROLLER_HOLD"):
            return
        if self._process_pending_reverse():
            return
        if (self.failure_state == "SENSOR_HOLD"
                and self.plan_metrics.get("recovery_stage") == "await_start_prefix_inputs"):
            if self._coherent_start_scan() is None:
                return
            self.start_scan_context_invalid = False
            episode = self.recovery_coordinator.current
            if episode is not None and episode.phase is RecoveryPhase.LOCAL:
                self._start_physical_recovery(episode.blockers,
                    "Coherent inputs restored the retained local recovery stage", resumed=True)
                return
        episode = self.recovery_coordinator.current
        if (self.map is not None and self.failure_state == "SENSOR_HOLD" and not self.planned_world_points
                and (episode is None or episode.phase is RecoveryPhase.FOLLOW)
                and len(self.local_reference_points) >= 2
                and self.local_reference_key == self._local_route_key()
                and self._observations_ready() and self._automatic_resume_allowed()):
            pose = self._fresh_motion_pose()
            if pose is not None:
                progress = path_progress(self.local_reference_points, pose[:2])
                attempted = (pose[:2], point_at_progress(self.local_reference_points, progress+.3))
                blockers, raw_hits = self._terminal_obstacle_blockers(attempted)
                if blockers and self._start_physical_recovery(blockers,
                        "Fresh physical blocker on sensor-restored retained route",
                        terminal_repair=True, raw_hits=raw_hits):
                    return
        self.waiting_for_replan = True
        if time.monotonic() - self.last_plan_time >= self._retry_period():
            self.plan_to(self.last_goal)

    def _invalidate_path(self, reason):
        self._clear_start_certificate()
        # One empty Path cancels the unsafe action. Repeated retries must not
        # continuously cancel a replacement action or flood navigation_status.
        if self.planned_cells:
            self.last_invalidated_path_stamp_ns=self.current_path_stamp_ns
        self.planned_cells = []
        self.planned_world_points = []
        self._clear_pending_live_route()
        if self.plan_metrics.get("local_detour_state") == "ACTIVE":
            self.plan_metrics.update(local_detour_state="FAILED", detour_reason=reason)
            self.local_rejoin_anchor = None
        if not self.invalid_path_published:
            path = Path()
            path.header.frame_id = self.get_parameter("map_frame").value
            path.header.stamp = self.get_clock().now().to_msg()
            self._publish_path(path)
            self.invalid_path_published = True
        self.waiting_for_replan = self.failure_state != "CONTROLLER_HOLD"
        self.publish_status("CONTROLLER_HOLD" if self.failure_state == "CONTROLLER_HOLD"
                            else "WAIT_REPLAN", reason)

    def _publish_path(self, path):
        """Give every published path, including an empty invalidation, a unique stamp."""
        self._clear_start_certificate()
        self.metric_route_scope = None
        if path.poses:
            self.metric_reverse_budget = None
            self.metric_no_path_key = None
        if path.poses:
            self.pending_reverse_probe = None
        stamp = path.header.stamp
        requested_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        next_ns = max(1, requested_ns, getattr(self, 'last_path_stamp_ns', -1) + 1,
                      getattr(self, 'last_failure_event_stamp_ns', 0) + 1)
        path.header.stamp = rclpy.time.Time(nanoseconds=next_ns).to_msg()
        for pose in path.poses:
            pose.header.stamp = path.header.stamp
        self.last_path_stamp_ns = next_ns
        self.current_path_stamp_ns = next_ns
        self.path_pub.publish(path)
        if path.poses:
            coordinator=self.recovery_coordinator
            if coordinator.current is not None:
                if coordinator.current.phase is RecoveryPhase.LOCAL:
                    coordinator.local_result(True)
                elif coordinator.current.phase is RecoveryPhase.GLOBAL:
                    coordinator.global_result(True,path_stamp_ns=next_ns)
                coordinator.current.path_stamp_ns=next_ns

    def _failure_diagnostics(self):
        """Current state overrides metadata left by an earlier recovery branch."""
        fields = ("failure_class", "recovery_stage", "controller_retry_count",
                  "controller_retry_limit", "controller_failure_evidence")
        detail = {name: self.plan_metrics[name] for name in fields if name in self.plan_metrics}
        state = self.failure_state
        if state == "CONTROLLER_HOLD":
            detail.update(failure_class="controller", recovery_stage="await_operator")
        elif self.recovery_active or state == "RECOVERING":
            detail.update(failure_class="recovery", recovery_stage="reverse")
        elif self.obstacle_hold.holding or state == "OBSTACLE_HOLD":
            detail.update(failure_class="obstacle", recovery_stage="await_clearance")
        elif state == "SENSOR_HOLD":
            detail.update(failure_class="sensor", recovery_stage="await_fresh_evidence")
        elif state == "CLEAR" and not self.waiting_for_replan:
            detail.update(failure_class="none", recovery_stage="tracking")
        detail.update(self.recovery_coordinator.snapshot())
        return detail

    def publish_status(self, status, message):
        goal = self.last_goal
        payload = {"status": status, "message": message, "goal": None, "goal_stamp": None,
                   **self.plan_metrics, **self._failure_diagnostics()}
        clock = self.get_clock().now()
        payload["sent_at_ns"] = getattr(clock, "nanoseconds", 0)
        path_stamp_ns = getattr(self, 'current_path_stamp_ns', None)
        if path_stamp_ns is not None:
            sec, nanosec = divmod(path_stamp_ns, 1_000_000_000)
            payload["path_stamp"] = {"sec": sec, "nanosec": nanosec}
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
        if not isinstance(payload, dict):
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
                self._reset_local_reference()
                self.recovery_active = False
                self.active_recovery_episode_id = None
                self.recovery_requested_at = None
                self.recovery_cancel_due_map_change = False
                self._clear_breadcrumbs()
                self.waiting_for_replan = False
                self.pending_failure_id = None
                self.rejoin_anchor = None
                self.obstacle_hold.reset()
                self.obstacle_episode_failure_ids.clear()
                self.obstacle_failure_cells.clear()
                self.publish_status("IDLE", "Manually stopped; send a newly ROS-stamped goal to resume")
            return
        if self.last_goal is None or not self._status_matches_goal(payload):
            return
        status = payload.get("status")
        if status in ("RECOVERING", "RECOVERY_STOPPED", "RECOVERY_SUCCEEDED", "RECOVERY_FAILED"):
            # Every reverse dispatch has its own ID, including initial NoPath.
            # A late terminal cannot release another attempt or erase its trail.
            owned = getattr(self, "active_recovery_episode_id", None)
            if (not self.recovery_active or not isinstance(owned, str)
                    or payload.get("episode_id") != owned):
                return
            episode = self.recovery_coordinator.current
            if episode is not None and (episode.episode_id != owned
                    or episode.phase is not RecoveryPhase.REVERSING):
                return
            if status == "RECOVERING":
                self.publish_status("RECOVERING", "Verified short reverse is in progress")
                return
            if episode is not None and not self.recovery_coordinator.reverse_terminal(owned):
                return
            self.recovery_active = False
            self.active_recovery_episode_id = None
            self.recovery_requested_at = None
            self.recovery_cancel_due_map_change = False
            self._clear_breadcrumbs()
            self.failure_state, self.failure_reason = "WAIT_ALTERNATIVE", "Reverse terminal confirmed; checking retained goal"
            self.last_plan_time = float("-inf")
            self._request_replan(self.failure_reason)
            return
        episode = self.recovery_coordinator.current
        if (self.failure_state == "CONTROLLER_HOLD"
                or episode is not None and episode.phase is not RecoveryPhase.FOLLOW):
            return
        if status == "SUCCEEDED":
            stamp = self._payload_stamp_ns(payload)
            if (not self._status_owns_path(payload)
                    or not self.recovery_coordinator.may_accept_success(stamp)):
                return
            self.goal_completed = True
            self._reset_local_reference()
            self.waiting_for_replan = False
            self.pending_failure_id = None
            self.rejoin_anchor = None
            self.failure_state = "CLEAR"
            self.obstacle_hold.reset()
            self.obstacle_episode_failure_ids.clear()
            self.obstacle_failure_cells.clear()
            self.publish_status("IDLE", "Goal completed")
            return
        if not self._status_owns_path(payload, allow_invalidated=True):
            return
        if status == "MOVING":
            self.navigation_progress_at = time.monotonic()
        elif status == "FAILED" and self._goal_is_active():
            if payload.get("route_failure") is True:
                event_id = payload.get("execution_failure_id")
                if event_id in self.seen_failure_ids:
                    return
                if isinstance(event_id, str) and event_id:
                    self.pending_failure_id = event_id
                    self.pending_failure_since = time.monotonic()
                    self.failure_state, self.failure_reason = "WAIT_ALTERNATIVE", "Waiting for terminal failure evidence"
            self._invalidate_path("Controller failed; waiting to replan retained goal")
            self.last_plan_time = time.monotonic()
        elif (status == "WAIT_REPLAN" and payload.get("route_failure") is False
              and self._goal_is_active()):
            self.pending_failure_id = self.pending_failure_since = None
            self.failure_state, self.failure_reason = "SENSOR_HOLD", "Sensor recovery required; no failure recorded"
            self._invalidate_path(self.failure_reason)
            self.last_plan_time = time.monotonic()

    def _status_matches_goal(self, payload):
        target = payload.get("goal") or {}
        if not isinstance(target, dict):
            return False
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
            # Preserve even an empty message: it can supersede an earlier
            # nonempty latched overlay before the first map is delivered.
            self.pending_memory_indices = tuple(msg.data)
            return
        updated = self._memory_cells(msg.data)
        if updated != self.remembered:
            added = updated - self.remembered
            removed = self.remembered - updated
            self.remembered = updated
            if self._recovery_wait_owns_retry():
                self._resume_recovery_wait()
                return
            # Empty-route recovery must also react to obstacle removal. While
            # driving, unrelated memory churn must not cancel a valid Nav2 goal.
            if self._goal_is_active() and (self.waiting_for_replan
                    or self._changes_block_path(added, layer="memory")
                    or self._changes_block_path(removed, layer="memory")):
                if self._changes_block_path(added, layer="memory"):
                    if self._start_physical_recovery(
                            added, "Remembered obstacle intersects the route"):
                        return
                    self._invalidate_path("Remembered obstacle intersects the route")
                self._request_replan("Remembered obstacles changed")

    def _memory_cells(self, indices):
        width, height = self.map[0].info.width, self.map[0].info.height
        return {(index % width, index // width) for index in indices
                if 0 <= index < width * height}

    def on_observed(self, msg):
        if self.map is None:
            return
        self.last_observation_time = time.monotonic()
        # Overlay publications may repeat a cached hit. Only the stamped
        # visibility grid proves a new physical scan for route confirmation.
        self.observed_revision += 1
        width, height = self.map[0].info.width, self.map[0].info.height
        updated = {(index % width, index // width) for index in msg.data
                   if 0 <= index < width * height}
        changed = updated != self.observed
        self.observed = updated
        if self._recovery_wait_owns_retry():
            return  # Raw disappearance or TTL replay is not positive free-space evidence.
        if not self._goal_is_active():
            self._clear_pending_live_route()
            return
        if (self._live_route_required_scans() <= 1 and not self.waiting_for_replan
                and changed):
            # Scan endpoints can flicker between neighboring raster cells.
            # Confirm route blockers across fresh frames before invalidating
            # the complete Nav2 action; a nearby hit bypasses this filter.
            blockers = self._live_route_blockers(updated)
            if blockers:
                self._react_to_live_route_blockers(blockers)
                return
        if self.waiting_for_replan:
            # Fresh empty/unchanged scans also recover a stale-sensor hold.
            self._request_replan("LiDAR obstacle observations restored")

    def _changes_block_path(self, cells, *, layer="observed", filter_near_wall=False):
        if not cells or not self.planned_cells:
            return False
        effective = (self._split_mapped_wall_returns(cells)[0]
                     if filter_near_wall else cells)
        metric = self._metric_route_blockers(effective)
        if metric is not None:
            return bool(metric)
        blocked = self._live_obstacle_blocked(effective)
        remaining = set(self._remaining_path_cells())
        remaining -= self._validated_start_prefix_exemption(remaining, observed=effective if layer != "memory" else ())
        if getattr(self, "start_prefix_input_pending", False):
            self._hold_start_prefix_inputs()
            return False
        return bool(blocked & remaining)

    def _split_mapped_wall_returns(self, cells):
        raw_grid = getattr(self, "raw_static_grid", None)
        if raw_grid is None:
            map_msg = self.map[0]
            raw_grid = occupancy_to_grid(map_msg,
                                         self.get_parameter("occupied_threshold").value, 0)
            self.raw_static_grid = raw_grid
        return split_mapped_wall_returns(
            cells, raw_grid,
            self.get_parameter("mapped_wall_return_tolerance_cells").value)

    def on_goal(self, goal):
        stamp_ns = goal.header.stamp.sec * 1_000_000_000 + goal.header.stamp.nanosec
        if self.stop_stamp_ns is not None and (stamp_ns == 0 or stamp_ns <= self.stop_stamp_ns):
            self.get_logger().warning(
                "Ignored delayed/unstamped goal after manual stop; send a newly ROS-stamped goal")
            return
        if self.failure_state == "CONTROLLER_HOLD" and self.last_goal is not None:
            previous = (self.last_goal.header.stamp.sec * 1_000_000_000
                        + self.last_goal.header.stamp.nanosec)
            if stamp_ns <= previous:
                self.get_logger().warning('Controller retry hold requires a newly stamped explicit goal')
                return
        # A delivery may start the next sector directly from the previous one.
        # Preserve that same physical trail across normal goal boundaries; the
        # reverse attempt budget, however, belongs to the new explicit goal.
        # An interrupted reverse has no known end pose and invalidates it.
        if self.recovery_active:
            self._clear_breadcrumbs()
        self.last_goal = goal
        self.controller_retry_events.clear()
        self.controller_pending_retry_events.clear()
        self.plan_metrics.update(failure_class="none", recovery_stage="tracking",
            controller_retry_count=0,
            controller_retry_limit=max(1, int(self.get_parameter("controller_retry_limit").value)))
        self.current_path_stamp_ns = None
        self._reset_local_reference()
        self.navigation_suspended = False
        self.goal_completed = False
        self.obstacle_hold.reset(repr(self._memory_scopes()))
        if self.obstacle_hold.attempts == 0:
            self.last_no_route_obstacle_charge_at = float("-inf")
            self.last_no_route_observed_revision = self.observed_revision
        self.last_visibility_at = None
        if self.obstacle_hold.holding:
            self.failure_state = "OBSTACLE_HOLD"
            self.invalid_path_published = False
            self._invalidate_path(self.failure_reason)
            return
        if self.obstacle_hold.attempts == 0:
            self.obstacle_episode_failure_ids.clear()
            self.obstacle_failure_cells.clear()
        self.pending_failure_id = None
        self.pending_failure_since = None
        self.rejoin_anchor, self.rejoin_verified = None, False
        self.navigation_progress_at = None
        self.last_failure_id = None
        self.last_failure_event_stamp_ns = 0
        self.failure_state, self.failure_reason = "CLEAR", ""
        self.planned_world_points = []
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
        self.odometry_trail.clear()
        self.projected_odom_trail_verified = False
        self._clear_start_scan_history()

    def on_odom(self, msg):
        """Keep the latest wheel/IMU odometry as an independent motion witness."""
        child_frame = getattr(msg, "child_frame_id", "")
        previous_child = getattr(self, "latest_odom_child_frame", None)
        if previous_child is not None and child_frame != previous_child:
            self._clear_breadcrumbs()
            self.odom_samples.clear()
        self.latest_odom_child_frame = child_frame
        stamp = msg.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        pose = msg.pose.pose
        sample = (time.monotonic(), stamp_ns, float(pose.position.x),
                  float(pose.position.y), _normalized_pose_yaw(pose.orientation),
                  msg.header.frame_id)
        if (stamp_ns <= 0 or not all(isfinite(value) for value in sample[:5])
                or not sample[5]):
            self.latest_odom = None
            self.odom_samples.clear()
            self._clear_breadcrumbs()
            return
        self.latest_odom = sample
        self.odom_samples.append(sample)
        prior = self.odometry_trail.latest
        discontinuity = (prior is not None and (
            sample[5] != self.odometry_trail.frame
            or child_frame != self.odometry_trail.child_frame
            or sample[0] < prior.received_at or sample[1] < prior.stamp_ns
            or sample[0]-prior.received_at >= .75 or sample[1]-prior.stamp_ns >= 750_000_000
            or hypot(sample[2]-prior.x, sample[3]-prior.y) > .15
            or abs(self._angle_delta(sample[4], prior.yaw)) > .15))
        accepted = self.odometry_trail.observe(sample, child_frame)
        if not accepted:
            self.latest_odom = None
            self.odom_samples.clear()
        elif discontinuity:
            self.odom_samples.clear()
            self.odom_samples.append(sample)
        if discontinuity or not accepted:
            self.breadcrumbs.clear()
            self._clear_start_scan_history()
        self.projected_odom_trail_verified = False

    @staticmethod
    def _angle_delta(a, b):
        return (a - b + pi) % (2.0 * pi) - pi

    def _record_breadcrumb(self, transform=None):
        """Project actual odometry through one fresh, synchronized map pair.

        AMCL corrections rigidly move the whole trace; they never create a
        travelled point or erase source-verified odometry. Missing TF revokes
        the projected certificate, while source/frame/reset/jump discontinuity
        is handled when the actual odometry message arrives.
        """
        self.projected_odom_trail_verified = False
        def unavailable():
            self.breadcrumbs.clear()
            self.breadcrumb_odom_anchor = None
            return False
        if not self._goal_is_active():
            return unavailable()
        if transform is None:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.get_parameter("map_frame").value,
                    self.get_parameter("base_frame").value, rclpy.time.Time())
            except TransformException:
                return unavailable()
        stamp = getattr(getattr(transform, "header", None), "stamp", None)
        now = time.monotonic()
        if stamp is None or not self.odom_samples:
            return unavailable()
        stamp_ns = stamp.sec*1_000_000_000+stamp.nanosec
        odom = min(self.odom_samples, key=lambda item: abs(item[1]-stamp_ns))
        now_ns = getattr(self.get_clock().now(), "nanoseconds", None)
        if (stamp_ns <= 0 or not 0 <= now-odom[0] < .75
                or abs(stamp_ns-odom[1]) > 150_000_000
                or (now_ns is not None and not -.2e9 <= now_ns-stamp_ns <= 1e9)):
            return unavailable()
        # Prefer an exact/interpolated pair so motion during DDS skew does not
        # masquerade as a correction to map->odom. If the latest TF is ahead
        # of received odometry, ask tf2 at an actually received source stamp.
        if stamp_ns != odom[1]:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.get_parameter("map_frame").value,
                    self.get_parameter("base_frame").value,
                    rclpy.time.Time(nanoseconds=odom[1]))
                paired_stamp = transform.header.stamp.sec*1_000_000_000+transform.header.stamp.nanosec
                if paired_stamp != odom[1]:
                    return unavailable()
            except (TransformException, ValueError, AttributeError):
                return unavailable()
        translation = transform.transform.translation
        rotation = getattr(transform.transform, "rotation", None)
        if rotation is None:
            return unavailable()
        pose = (float(translation.x), float(translation.y), _normalized_pose_yaw(rotation))
        if not all(isfinite(value) for value in pose):
            return unavailable()
        if self.odometry_trail.latest is None:
            # An explicit reset may leave the latest *real received* sample as
            # the only seed. It cannot manufacture an earlier travelled trail.
            if not self.odometry_trail.observe(odom, self.latest_odom_child_frame):
                return unavailable()
        projected = self.odometry_trail.project(pose, odom, now=now)
        if not projected:
            return unavailable()
        self.breadcrumbs.clear()
        self.breadcrumbs.extend(projected)
        self.breadcrumb_odom_anchor = odom
        self.projected_odom_trail_verified = True
        return True

    def _reverse_start_stopped(self):
        """Require independently measured stillness before certifying a retreat.

        The invalid/empty Path is already published before this check. Wait
        for at least .20 s of dense odometry; a command of zero or a single
        pose is insufficient evidence that the old FollowPath has stopped.
        Allow bounded 10 Hz timing jitter without weakening the pose bound.
        """
        samples = tuple(self.odom_samples)
        if len(samples) < 4:
            return False
        latest = samples[-1]
        now = time.monotonic()
        window = [sample for sample in samples if 0 <= latest[0]-sample[0] <= .50]
        if (len(window) < 4 or not 0 <= now-latest[0] < .20
                or any(sample[5] != latest[5] for sample in window)
                or latest[0]-window[0][0] < .20-1e-9):
            return False
        for previous, sample in zip(window, window[1:]):
            if (not 0 < sample[0]-previous[0] <= .18+1e-9
                    or not 0 < sample[1]-previous[1] <= 180_000_000
                    or sample[5] != latest[5]):
                return False
        return all(hypot(sample[2]-latest[2], sample[3]-latest[3]) <= .002
                   and abs(self._angle_delta(sample[4], latest[4])) <= .005
                   for sample in window)

    def _reverse_unavailable(self, code, start):
        """Expose the first failed reverse gate without flooding retry logs."""
        previous_code = getattr(self, "last_reverse_rejection_code", None)
        self.last_reverse_rejection_code = code
        now = time.monotonic()
        previous = getattr(self, "last_reverse_rejection_log_at", float("-inf"))
        if code != previous_code or now < previous or now - previous >= 5.0:
            self.get_logger().warning(f"REVERSE_REJECT[{code}] start={start}")
            self.last_reverse_rejection_log_at = now
        return False

    def _maybe_request_reverse(self, start, end, map_msg, grid, raw_grid,
                               raw_dynamic, inflated_dynamic, cell_penalty,
                               trail_verified, blocking_evidence=None):
        """Request a bounded Nav2 backup only when a verified A* exit exists."""
        episode=self.recovery_coordinator.current
        if episode is not None and episode.phase is not RecoveryPhase.REVERSE_VERDICT:
            return self._reverse_unavailable("EPISODE_STAGE",start)
        if not trail_verified:
            return self._reverse_unavailable("TRAIL_UNVERIFIED", start)
        if not self.get_parameter("reverse_recovery_enabled").value:
            return self._reverse_unavailable("DISABLED", start)
        if getattr(self, "recovery_active", False):
            return self._reverse_unavailable("ALREADY_ACTIVE", start)
        if episode is None and start in self.recovery_attempted_cells:
            return self._reverse_unavailable("START_ALREADY_ATTEMPTED", start)
        if episode is None and self.recovery_attempts >= int(self.get_parameter("reverse_recovery_max_attempts").value):
            return self._reverse_unavailable("ATTEMPT_LIMIT", start)
        if not self._goal_is_active():
            return self._reverse_unavailable("NO_ACTIVE_GOAL", start)
        if not self._observations_ready():
            return self._reverse_unavailable("SENSORS_UNREADY", start)
        if not self._automatic_resume_allowed():
            return self._reverse_unavailable("AUTONOMY_UNAVAILABLE", start)
        if not self._reverse_start_stopped():
            return self._reverse_unavailable("NOT_STOPPED", start)
        if not getattr(self, "latest_odom_child_frame", ""):
            return self._reverse_unavailable("ODOM_CHILD_FRAME_MISSING", start)
        if grid.is_static_obstacle(end):
            return self._reverse_unavailable("GOAL_STATIC_BLOCKED", start)
        if end in inflated_dynamic:
            return self._reverse_unavailable("GOAL_DYNAMIC_BLOCKED", start)
        current = self._fresh_motion_pose()
        physical_radius = float(self.get_parameter("robot_radius").value)
        if current is None:
            return self._reverse_unavailable("POSE_UNREADY", start)
        if not self.breadcrumbs:
            return self._reverse_unavailable("TRAIL_MISSING", start)
        if not isfinite(physical_radius) or physical_radius <= 0:
            return self._reverse_unavailable("RADIUS_INVALID", start)
        # Keep measurement time and odom baseline before any map/A* work. Do
        # not refresh certificate age after a costly candidate search.
        certified_odom = tuple(self.latest_odom)
        certified_child = self.latest_odom_child_frame
        certified_map_stamp_ns = getattr(self, "last_fresh_motion_pose_stamp_ns", None)
        if type(certified_map_stamp_ns) is not int or certified_map_stamp_ns <= 0:
            return self._reverse_unavailable("MAP_POSE_STAMP_MISSING", start)
        certificate_at_ns = min(self.get_clock().now().nanoseconds,
                                certified_odom[1], certified_map_stamp_ns)
        latest = self.breadcrumbs[-1]
        if (hypot(current[0]-latest.x, current[1]-latest.y) > .05
                or abs(self._angle_delta(current[2], latest.yaw)) > .15):
            return self._reverse_unavailable("TRAIL_POSE_MISMATCH", start)

        # A newly seen object can leave a physically clear corridor parallel
        # to a wall. Requiring clearance to that *wall* to improve as well as
        # clearance to the blocker unnecessarily rejects a useful retreat.
        # Nominate only fresh, unmapped physical hits ahead of this chassis;
        # the selector still checks every raw wall/hit over the entire sweep.
        blocking_cells = set()
        if (self.last_observation_time is not None
                and 0 <= time.monotonic()-self.last_observation_time < .75):
            body = physical_radius
            for cell in raw_dynamic & self.observed & set(blocking_evidence or ()):
                if raw_grid.is_static_obstacle(cell):
                    continue
                x, y = cell_to_world(map_msg.info, cell)
                dx, dy = x-current[0], y-current[1]
                forward = dx*cos(current[2])+dy*sin(current[2])
                lateral = abs(-dx*sin(current[2])+dy*cos(current[2]))
                if 0 < forward <= .80 and lateral <= body+.10:
                    blocking_cells.add(cell)

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

        failure_edges, failure_costs = self._failure_constraints(grid)

        # Keep the conservative goal oracle first. Only unmeasured depth may
        # become a cost after that search fails; actual hits, confirmed memory,
        # static/unknown cells and failed edges retain their original masks.
        observed_snapshot, remembered_snapshot = set(self.observed), set(self.remembered)
        measured_dynamic = (self._live_obstacle_blocked(
            observed_snapshot | (set(raw_dynamic)-remembered_snapshot))
            | self._live_obstacle_blocked(remembered_snapshot))
        virtual_dynamic = set(inflated_dynamic)-measured_dynamic
        reverse_scope = self._local_route_key()
        reverse_reference = tuple(self.local_reference_points)
        oracle_wait_reason = None
        self.plan_metrics.update(reverse_goal_occlusion_fallback=False,
                                 reverse_goal_occlusion_fallback_attempted=False)

        def fresh_reverse_oracle_context():
            nonlocal oracle_wait_reason
            if (self.map is None or self.map[0] is not map_msg
                    or self.raw_static_grid is not raw_grid
                    or self._local_route_key() != reverse_scope
                    or tuple(self.local_reference_points) != reverse_reference
                    or self.observed != observed_snapshot or self.remembered != remembered_snapshot):
                return False
            if not self._observations_ready():
                oracle_wait_reason = "SENSORS_UNREADY"
                return False
            if not self._automatic_resume_allowed():
                oracle_wait_reason = "AUTONOMY_UNAVAILABLE"
                return False
            if not self._reverse_start_stopped():
                oracle_wait_reason = "NOT_STOPPED"
                return False
            if not self._occlusion_fallback_ready(start):
                oracle_wait_reason = "POSE_UNREADY"
                return False
            return True

        def can_reach_retained_goal(candidate_cell):
            primary = self.planner.plan(grid, candidate_cell, end, inflated_dynamic,
                                       cell_penalty=cell_penalty,
                                       blocked_edges=failure_edges, edge_penalty=failure_costs)
            if primary:
                return True
            if not virtual_dynamic or not fresh_reverse_oracle_context():
                return False
            self.plan_metrics["reverse_goal_occlusion_fallback_attempted"] = True
            soft_penalty = self._soft_occlusion_penalty(
                cell_penalty, set(inflated_dynamic), measured_dynamic)
            alternative = self.planner.plan(grid, candidate_cell, end, measured_dynamic,
                                           cell_penalty=soft_penalty,
                                           blocked_edges=failure_edges, edge_penalty=failure_costs)
            if not alternative or not fresh_reverse_oracle_context():
                return False
            self.plan_metrics["reverse_goal_occlusion_fallback"] = True
            return True

        clearance_neutral_egress = (
            self.get_parameter("reverse_clearance_neutral_egress_enabled").value is True
            and not blocking_cells and inflated_with_live.is_static_obstacle(start)
            and not raw_with_live.is_static_obstacle(start))
        self.plan_metrics["reverse_clearance_neutral_egress"] = clearance_neutral_egress

        history = tuple(self.breadcrumbs)
        self.plan_metrics["reverse_probe_diagnostics"] = {
            "trail_count": len(history),
            "trail_frame": "odom_projected" if self.projected_odom_trail_verified else "legacy_unverified",
            "source_odom_frame": self.odometry_trail.frame,
            "source_odom_child_frame": self.odometry_trail.child_frame,
            "source_odom_reset_reason": self.odometry_trail.last_reset_reason,
            "oldest_at": history[0].stamp if history and isfinite(history[0].stamp) else None,
            "newest_at": history[-1].stamp if history and isfinite(history[-1].stamp) else None,
            "current_pose": list(current),
            "recent_trail": [[point.stamp, point.x, point.y, point.yaw]
                             for point in history[-32:]
                             if all(isfinite(v) for v in (point.stamp, point.x, point.y, point.yaw))],
            "neutral_enabled": clearance_neutral_egress,
            "conservative_dynamic_count": len(inflated_dynamic),
            "measured_dynamic_count": len(measured_dynamic),
            "virtual_dynamic_count": len(virtual_dynamic), "candidate": None}

        candidate = select_reverse_egress(
            history, raw_with_live, inflated_with_live, geometry,
            can_reach_retained_goal, now=time.monotonic(),
            robot_radius=physical_radius,
            # Dispatch may differ by <=3 mm / .01 rad from this stationary
            # certificate. The separate .01 m envelope covers that handover
            # (3 mm + .30 m*sin(.01) < 7 mm), beyond the physical margin.
            clearance_margin=float(self.get_parameter("reverse_recovery_clearance_margin").value)+.01,
            min_distance=float(self.get_parameter("reverse_recovery_min_distance").value),
            max_distance=float(self.get_parameter("reverse_recovery_max_distance").value),
            max_history_age=90.0, max_sample_gap=30.0,
            blocking_cells=blocking_cells, current_pose=current,
            clearance_neutral_egress_enabled=clearance_neutral_egress,
            verified_command_corridor=self.projected_odom_trail_verified,
            # A physically clear start can still be trapped by the static
            # inflation band. Parallel retreat need not increase distance to
            # an unrelated side wall, but must exit inflation and preserve
            # the selector's complete physical sweep and A* goal checks.
            allow_parallel_static_egress=(
                grid.is_static_obstacle(start)
                and not raw_with_live.is_static_obstacle(start)
                and not blocking_cells))
        if candidate is None:
            if oracle_wait_reason is not None:
                return self._reverse_unavailable(oracle_wait_reason, start)
            return self._reverse_unavailable("NO_SAFE_EGRESS", start)
        self.plan_metrics["reverse_probe_diagnostics"]["candidate"] = {
            "distance_m": candidate.distance_m, "target_cell": list(candidate.target_cell),
            "target_xy": [candidate.target_x, candidate.target_y],
            "start_clearance_m": candidate.start_clearance_m,
            "target_clearance_m": candidate.target_clearance_m}
        if not 0 <= self.get_clock().now().nanoseconds-certificate_at_ns < 750_000_000:
            return self._reverse_unavailable("CERTIFICATE_EXPIRED", start)

        self.recovery_attempts += 1
        self.recovery_attempted_cells.add(start)
        self.recovery_active = True
        self.recovery_requested_at = time.monotonic()
        self.failure_state, self.failure_reason = "RECOVERING", "Verified recent track retreat"
        goal = self.last_goal
        before_global = getattr(self, "pending_reverse_probe", None) is not None
        self.active_recovery_episode_id=(episode.episode_id if episode is not None else uuid4().hex)
        reason = ("Local repair unavailable; verified recent retreat before global replanning"
                  if before_global else
                  "No forward A* route; verified recent straight track leads to a safe A* start")
        request = String()
        request.data = json.dumps({
            "goal_id": f"{goal.header.stamp.sec}:{goal.header.stamp.nanosec}",
            "episode_id": self.active_recovery_episode_id,
            "source_path_stamp": (None if self.last_invalidated_path_stamp_ns is None else {
                "sec": self.last_invalidated_path_stamp_ns//1_000_000_000,
                "nanosec": self.last_invalidated_path_stamp_ns%1_000_000_000}),
            "distance": candidate.distance_m,
            "speed": float(self.get_parameter("reverse_recovery_speed").value),
            "reason": reason,
            "recovery_trigger": "blocked_active_route" if before_global else "global_no_path",
            "start_cell": list(start), "exit_cell": list(candidate.target_cell),
            "blocking_cells": [list(cell) for cell in sorted(blocking_cells)],
            "certified_start": {
                "frame_id": certified_odom[5],
                "child_frame_id": certified_child,
                "stamp_ns": certified_odom[1],
                "x": certified_odom[2], "y": certified_odom[3],
                "yaw": certified_odom[4],
                "certified_at_ns": certificate_at_ns,
                "max_position_drift_m": .003, "max_yaw_drift_rad": .01,
                "sweep_reserve_m": .01,
                "map_frame": self.get_parameter("map_frame").value,
                "map_base_frame": self.get_parameter("base_frame").value,
                "map_x": current[0], "map_y": current[1], "map_yaw": current[2],
                "map_stamp_ns": certified_map_stamp_ns}})
        self.recovery_pub.publish(request)
        self.publish_status("RECOVERING", f"Backing up {candidate.distance_m:.2f} m to replan retained goal")
        self.get_logger().warning(
            f"{'Local repair unavailable' if before_global else 'No forward route'}; "
            f"requested collision-checked {candidate.distance_m:.2f} m reverse "
            f"to {candidate.target_cell} (attempt {self.recovery_attempts})")
        return True

    def plan_to(self, goal):
        if self._recovery_wait_owns_retry():
            return
        if self._process_pending_reverse():
            return
        episode=self.recovery_coordinator.current
        if episode is not None and episode.phase not in (RecoveryPhase.GLOBAL,RecoveryPhase.FOLLOW):
            return
        if (episode is not None and episode.phase is RecoveryPhase.GLOBAL
                and (self._fresh_motion_pose() is None or not self._automatic_resume_allowed())):
            self.failure_state,self.failure_reason="SENSOR_HOLD","Fresh pose and authority required for post-reverse planning"
            self.publish_status("WAIT_REPLAN",self.failure_reason)
            return
        if (getattr(self, "recovery_active", False) or self.pending_failure_id is not None
                or self.obstacle_hold.holding
                or self.failure_state == "CONTROLLER_HOLD"):
            return
        if self.failure_state == "SENSOR_HOLD" and self._fresh_motion_pose() is None:
            self._invalidate_path("Waiting for fresh independent sensor evidence before retry")
            return
        self.last_plan_time = time.monotonic()
        global_started = time.perf_counter()
        self.plan_metrics.update(plan_kind="GLOBAL", failure_edge_fallback=False,
            local_detour_state="FAILED" if self.plan_metrics.get("local_detour_state") == "FAILED" else "NONE",
            occlusion_fallback=False, occlusion_fallback_attempted=False,
            occlusion_constraint_mode="HARD_RESERVE", occlusion_virtual_cells=0,
            occlusion_primary_reason="", occlusion_reused_start_certificate=False,
            occlusion_measured_start_certificate=False,
            selected_start_certificate_diagnostics={"reason": "not_required", "edge": None})
        self.geometry_signature = tuple(self.get_parameter(name).value for name in (
            "path_smoothing_enabled", "path_smoothing_iterations",
            "path_smoothing_shortcut_distance", "path_resample_spacing"))
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
                     self.get_parameter("obstacle_inflation_cells").value,
                     self.get_parameter("mapped_wall_return_tolerance_cells").value)
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
        heading = _normalized_pose_yaw(tf.transform.rotation)
        no_path_key = (self._local_route_key(), self.raw_static_grid,
            round(tf.transform.translation.x/.01), round(tf.transform.translation.y/.01),
            round(heading/(pi/36)) if isfinite(heading) else None,
            frozenset(self.observed), frozenset(self.remembered),
            getattr(self, "last_failure_id", None), self._start_certificate_geometry())
        trail_key = tuple((round(point.x/.025), round(point.y/.025))
                          for point in self.breadcrumbs)
        no_path_key += (getattr(self, "projected_odom_trail_verified", False), trail_key,
                        getattr(self, "latest_odom_child_frame", None))
        no_path_key += (tuple(self.get_parameter(name).value for name in LOCAL_DETOUR_DEFAULTS),
                        self.get_parameter("preferred_clearance").value,
                        self.get_parameter("clearance_penalty_gain").value)
        if no_path_key == getattr(self, "metric_no_path_key", None):
            pose = self._fresh_motion_pose()
            info = self.map[0].info
            if pose is not None:
                angle = yaw_from_quaternion(info.origin.orientation)
                dx,dy=pose[0]-info.origin.position.x,pose[1]-info.origin.position.y
                point=((cos(angle)*dx+sin(angle)*dy)/info.resolution,
                       (-sin(angle)*dx+cos(angle)*dy)/info.resolution)
                distance=polyline_clearance([point,point],occupied_rectangles(
                    self.raw_static_grid,self.observed|self.remembered),
                    self.raw_static_grid.width,self.raw_static_grid.height)*info.resolution
                if distance+1e-9 < float(self.get_parameter("robot_radius").value):
                    if not self._wait_after_recovery_cycle():
                        self.publish_status("WAIT_REPLAN", "Current measured circular body intersects an occupied square; goal retained")
                    return
            self.metric_no_path_key = None
        width, height = map_msg.info.width, map_msg.info.height
        inflation = self.get_parameter("obstacle_inflation_cells").value
        # Both layers contain range-classified dynamic hits. A second proximity
        # filter here could hide a real object beside a mapped wall. Confirmed
        # memory also retains full inflation wherever it was observed.
        remembered_effective = set(self.remembered)
        observed_effective = set(self.observed)
        planning_observed = self._global_occlusion_reserve(start, observed_effective)
        planning_remembered = self._global_occlusion_reserve(
            start, remembered_effective, remembered=True)
        # Diagnostic only: show how many live returns are already covered by
        # the inflated static map. Do not suppress a real near-wall object.
        observed_inside_static_inflation = sum(
            1 for cell in self.observed if grid.is_static_obstacle(cell))
        remembered_blocked = self._live_obstacle_blocked(planning_remembered)
        observed_blocked = self._live_obstacle_blocked(planning_observed)
        blocked = remembered_blocked | observed_blocked
        measured_remembered_blocked = self._live_obstacle_blocked(remembered_effective)
        measured_observed_blocked = self._live_obstacle_blocked(observed_effective)
        measured_blocked = measured_remembered_blocked | measured_observed_blocked
        virtual_blocked = blocked - measured_blocked
        self.plan_metrics.update(occlusion_virtual_cells=len(virtual_blocked))
        cell_penalty = clearance_penalties(
            raw_grid, planning_observed | planning_remembered,
            map_msg.info.resolution,
            float(self.get_parameter("preferred_clearance").value),
            float(self.get_parameter("clearance_penalty_gain").value))
        failure_edges, failure_costs = self._failure_constraints(grid)
        soft_controller_edges, failure_costs = self._controller_retry_policy(failure_costs)
        if self.failure_state == "CONTROLLER_HOLD":
            return
        if failure_costs and self._fresh_motion_pose() is None:
            self.failure_state, self.failure_reason = "SENSOR_HOLD", "Fresh motion evidence required for failure recovery"
            self._invalidate_path(self.failure_reason)
            return
        # A scan obstacle can enter the inflation band around the robot itself
        # even when the footprint is still clear. Permit the A* search to leave
        # that one dynamic cell; static-map and confirmed-memory collisions stay
        # hard failures, while Nav2's local costmap retains its collision safety.
        escape_cells = self._dynamic_escape_halo(
            start, observed_blocked | remembered_blocked, observed_effective, remembered_effective,
            grid, raw_grid, inflation,
            self.get_parameter("robot_radius").value / map_msg.info.resolution,
            (self.get_parameter("start_egress_clearance_margin").value
             / map_msg.info.resolution + sqrt(2.0) / 2.0))
        safety_egress_edges = set()
        certified_start_edges = set()
        observation_age = (None if self.last_observation_time is None else
                           time.monotonic()-self.last_observation_time)
        # This timestamp is optional diagnostic metadata, not motion evidence.
        # Existing plan_to callers can provide a transform without a header.
        diagnostic_stamp = getattr(getattr(tf, "header", None), "stamp", None)
        diagnostic_sec = getattr(diagnostic_stamp, "sec", None)
        diagnostic_nsec = getattr(diagnostic_stamp, "nanosec", None)
        diagnostic_stamp_ns = (diagnostic_sec*1_000_000_000+diagnostic_nsec
                               if type(diagnostic_sec) is int and type(diagnostic_nsec) is int
                               else None)
        certificate_diagnostic = {
            "start": list(start), "helper_called": False, "allowed_edges": [],
            "enabled": bool(self.get_parameter("start_egress_enabled").value),
            "escape_cells_before": [list(cell) for cell in sorted(escape_cells)],
            "static_start": grid.is_static_obstacle(start),
            "raw_static_start": raw_grid.is_static_obstacle(start),
            "raw_observed_start": start in observed_effective,
            "observed_inflated_start": start in observed_blocked,
            "remembered_inflated_start": start in remembered_blocked,
            "measured_remembered_inflated_start": start in measured_remembered_blocked,
            "observation_receipt_age_s": observation_age,
            "start_tf_stamp_ns": diagnostic_stamp_ns,
            "start_tf_pose": [float(tf.transform.translation.x), float(tf.transform.translation.y)],
            "fresh_pose": None, "fresh_pose_drift_m": None,
            "first_failed_reason": "NOT_REQUIRED"}
        certificate_gates = (
            ("disabled", not certificate_diagnostic["enabled"]),
            ("escape_already_present", bool(escape_cells)),
            ("not_in_inflation", not (grid.is_static_obstacle(start) or start in observed_blocked)),
            ("raw_static_start", raw_grid.is_static_obstacle(start)),
            ("raw_observed_start", start in observed_effective),
            ("remembered_inflation_start", start in remembered_blocked),
            ("observations_missing", observation_age is None),
            ("observation_receipt_age", observation_age is not None and not 0 <= observation_age < .75))
        certificate_diagnostic["first_failed_reason"] = next(
            (reason for reason, failed in certificate_gates if failed), "AWAIT_FRESH_POSE")
        certificate_diagnostic["neighbor_masks"] = [
            {"cell": [start[0]+dx, start[1]+dy],
             "static": grid.is_static_obstacle((start[0]+dx, start[1]+dy)),
             "measured": (start[0]+dx, start[1]+dy) in measured_blocked,
             "conservative": (start[0]+dx, start[1]+dy) in blocked,
             "virtual_only": (start[0]+dx, start[1]+dy) in virtual_blocked}
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        self.plan_metrics["start_certificate_diagnostics"] = certificate_diagnostic
        if (self.get_parameter("start_egress_enabled").value
                and not escape_cells
                and (grid.is_static_obstacle(start) or start in observed_blocked)
                and not raw_grid.is_static_obstacle(start)
                and start not in observed_effective and start not in remembered_blocked
                and self.last_observation_time is not None
                and 0 <= time.monotonic() - self.last_observation_time < .75):
            # Correct a static/live inflation start-cell quantization artifact.
            # Use the calibrated robot radius from this deployment's profile,
            # then retain the configured clearance margin.
            # Global inflation, raw occupied/unknown cells and later edges stay
            # unchanged. This does not certify unmodeled hardware extensions.
            current = self._fresh_motion_pose()
            certificate_diagnostic["motion_pose"] = dict(
                self.plan_metrics.get("motion_pose_diagnostics", {}))
            if current is None:
                certificate_diagnostic["first_failed_reason"] = "fresh_motion_pose_unavailable"
            else:
                certificate_diagnostic.update(fresh_pose=list(current),
                    fresh_pose_drift_m=hypot(current[0]-tf.transform.translation.x,
                                            current[1]-tf.transform.translation.y))
                certificate_diagnostic["first_failed_reason"] = (
                    "fresh_pose_cell_changed" if world_to_cell(map_msg.info, current[0], current[1]) != start else
                    "fresh_pose_drift" if certificate_diagnostic["fresh_pose_drift_m"] > .05 else
                    "AWAIT_HELPER")
            if (current is not None
                    and world_to_cell(map_msg.info, current[0], current[1]) == start
                    and hypot(current[0] - tf.transform.translation.x,
                              current[1] - tf.transform.translation.y) <= .05):
                origin = map_msg.info.origin.position
                angle = yaw_from_quaternion(map_msg.info.origin.orientation)
                dx, dy = current[0] - origin.x, current[1] - origin.y
                pose_cells = ((cos(angle) * dx + sin(angle) * dy) / map_msg.info.resolution,
                              (-sin(angle) * dx + cos(angle) * dy) / map_msg.info.resolution)
                physical_radius = float(self.get_parameter("robot_radius").value)
                certificate_diagnostic["helper_called"] = True
                safe_edges = certified_start_only_edges(
                    raw_grid, grid, observed_effective | remembered_effective,
                    blocked, start, pose_cells, physical_radius / map_msg.info.resolution,
                    float(self.get_parameter("start_egress_clearance_margin").value)
                    / map_msg.info.resolution)
                certificate_diagnostic.update(
                    allowed_edges=[[list(a), list(b)] for a, b in sorted(safe_edges)],
                    first_failed_reason="CERTIFIED" if safe_edges else "no_certified_edge")
                if safe_edges:
                    certified_start_edges = safe_edges
                    escape_cells = {start}
                    safety_egress_edges = {
                        (start, (start[0] + dx, start[1] + dy))
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                        if (dx or dy) and (start, (start[0] + dx, start[1] + dy)) not in safe_edges
                    }
                    # Clearing the start must not change later diagonal
                    # corner checks. Block all reentry and both directions of
                    # every neighbor-to-neighbor diagonal that would use this
                    # cleared cell as an orthogonal side cell.
                    safety_egress_edges.update(
                        ((start[0]+dx, start[1]+dy), start)
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
                    for dx in (-1, 1):
                        for dy in (-1, 1):
                            horizontal = (start[0]+dx, start[1])
                            vertical = (start[0], start[1]+dy)
                            safety_egress_edges.update(((horizontal, vertical), (vertical, horizontal)))
                    self.get_logger().info(
                        f"Fresh continuous pose certified {len(safe_edges)} start-only egress edges "
                        f"with physical radius {physical_radius:.3f} m and unchanged clearance margin")
        # A statically inflated start cannot be passed to ordinary A*. Build
        # a private planning view clearing that one *certified* cell only. The
        # stored map, goal, neighboring inflation and every later edge remain
        # unchanged; the directed first-edge constraints are never softened.
        planning_grid = grid
        if certified_start_edges and grid.is_static_obstacle(start):
            planning_grid = Grid(tuple(tuple(
                False if (x, y) == start else grid.rows[y][x]
                for x in range(grid.width)) for y in range(grid.height)))
        allow_dynamic_escape = bool(escape_cells)
        plan_blocked_edges = safety_egress_edges
        egress_steps = 0
        self.global_plan_count += 1
        self.plan_metrics.update(global_plan_count=self.global_plan_count)
        try:
            if (grid.is_static_obstacle(start) and not certified_start_edges
                    and self.get_parameter("start_egress_enabled").value):
                resolution = map_msg.info.resolution
                # The measured pose may lie anywhere inside its cell: preserve
                # an extra half-cell diagonal in addition to footprint margin.
                margin = self.get_parameter("start_egress_clearance_margin").value + resolution*sqrt(2.0)/2.0
                cells = self.planner.plan_with_start_egress(
                    grid, start, end, blocked, raw_grid=raw_grid,
                    raw_blocked=observed_effective | remembered_effective,
                    robot_radius_cells=self.get_parameter("robot_radius").value/resolution,
                    margin_cells=margin/resolution,
                    max_egress_cells=min(int(inflation)+1, int(self.get_parameter("start_egress_max_cells").value)),
                    allow_blocked_start=allow_dynamic_escape, escape_cells=escape_cells,
                    cell_penalty=cell_penalty, blocked_edges=plan_blocked_edges,
                    edge_penalty=failure_costs)
                if cells:
                    egress_steps = next((index for index, cell in enumerate(cells)
                                         if not grid.is_static_obstacle(cell)), 0)
            else:
                cells = self.planner.plan(planning_grid, start, end, blocked,
                                          allow_blocked_start=allow_dynamic_escape,
                                          escape_cells=escape_cells,
                                          cell_penalty=cell_penalty,
                                          blocked_edges=plan_blocked_edges,
                                          edge_penalty=failure_costs)
        except ValueError:
            self.plan_metrics.update(global_duration_ms=(time.perf_counter()-global_started)*1000.)
            self._invalidate_path(f"Robot or goal lies outside the map: {start} -> {end}")
            return
        certification_blocked = blocked
        reused_start_certificate = False
        if (not cells and virtual_blocked and self._occlusion_fallback_ready(start)):
            # The primary conservative search is unchanged. Only its virtual
            # depth may become a cost; measured memory/static/unknown stay hard.
            needs_start_certificate = grid.is_static_obstacle(start) or start in measured_blocked
            retained = (self._measured_start_certificate(
                start, grid, raw_grid, measured_blocked, observed_effective, remembered_effective)
                if needs_start_certificate else set())
            certificate_valid = not needs_start_certificate or bool(retained)
            fallback_grid = grid
            fallback_escape = {start} if retained else set()
            fallback_safety = self._start_only_forbidden_edges(start, retained) if retained else set()
            fallback_edges = fallback_safety
            if retained and grid.is_static_obstacle(start):
                fallback_grid = Grid(tuple(tuple(
                    False if (x, y) == start else grid.rows[y][x]
                    for x in range(grid.width)) for y in range(grid.height)))
            self.plan_metrics.update(occlusion_fallback_attempted=True,
                occlusion_primary_reason="no_route_with_hard_virtual_reserve",
                occlusion_constraint_mode="SOFT_VIRTUAL",
                occlusion_measured_start_certificate=bool(retained))
            soft_penalty = self._soft_occlusion_penalty(cell_penalty, blocked, measured_blocked)
            try:
                fallback_cells = (self.planner.plan(
                    fallback_grid, start, end, measured_blocked, cell_penalty=soft_penalty,
                    allow_blocked_start=bool(retained), escape_cells=fallback_escape,
                    blocked_edges=fallback_edges, edge_penalty=failure_costs)
                    if certificate_valid else [])
            except ValueError:
                fallback_cells = []
            if fallback_cells:
                cells, blocked, cell_penalty = fallback_cells, measured_blocked, soft_penalty
                planning_grid = fallback_grid
                observed_blocked, remembered_blocked = measured_observed_blocked, measured_remembered_blocked
                certified_start_edges = retained
                certification_blocked = measured_blocked
                escape_cells, allow_dynamic_escape = fallback_escape, bool(retained)
                safety_egress_edges, plan_blocked_edges = fallback_safety, fallback_edges
                reused_start_certificate = bool(retained)
                self.plan_metrics.update(occlusion_fallback=True,
                    occlusion_reused_start_certificate=reused_start_certificate)
                self.get_logger().warning(
                    "Global hard virtual-depth reserve has no route; using soft reserve costs "
                    "with all measured obstacles, memory and body inflation unchanged")
        self.plan_metrics.update(global_duration_ms=(time.perf_counter()-global_started)*1000.)
        # Retained wire key: this now reports bounded execution costs, not a
        # hard-edge fallback permission or a certificate of sensor clearance.
        failure_edge_fallback = soft_controller_edges
        if soft_controller_edges:
            self.plan_metrics.update(failure_edge_fallback=True)
        if not cells:
            config = self._local_configuration()
            if (config is not None and config["enabled"]
                    and self.get_parameter("start_egress_enabled").value
                    and self._try_metric_connector(config,
                        max_seconds=config["max_planning_time"],
                        max_expansions=config["max_expansions"], end=end)):
                return
            if getattr(self, "metric_connector_input_pending", False):
                return
            self.get_logger().warning("ASTAR_START_CERTIFICATE_DIAGNOSTICS "+json.dumps({
                "motion_pose": self.plan_metrics.get("motion_pose_diagnostics"),
                "start_certificate": self.plan_metrics.get("start_certificate_diagnostics")},
                sort_keys=True, allow_nan=False), throttle_duration_sec=10.0)
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
                f"observed_near_map_wall=0, "
                f"observed_inside_static_inflation={observed_inside_static_inflation}, "
                f"observed_effective={len(observed_effective)}, "
                f"observed_inflated={len(observed_blocked)}, "
                f"remembered_raw={len(self.remembered)}, "
                f"remembered_inflated={len(remembered_blocked)}, "
                f"failed_edges={len(failure_edges)}, "
                f"failed_edge_penalties={len(failure_costs)}, "
                f"clearance_penalty_cells={len(cell_penalty)}, "
                f"wall_match_tolerance_cells={signature[2]}, "
                f"static_reference_cells={len(reference_path)}, "
                f"static_reference_dynamic_hits={len(reference_hits)}, "
                f"first_blocked_reference_cells={reference_hits[:8]})",
                throttle_duration_sec=10.0)
            self.failure_state, self.failure_reason = "WAIT_ALTERNATIVE", "No safe alternative; goal retained"
            self.plan_metrics.update(failure_class="obstacle" if blocked else "route",
                recovery_stage="global_replan")
            self._invalidate_path(f"No safe A* route from {start} to {end}; retrying")
            if self._wait_after_recovery_cycle(relevant_cells=reference_path):
                return
            # A focused obstacle retreat needs causal evidence: a physical
            # fresh hit intersects the static route's body corridor and the
            # same retained goal is reachable with only live hits removed.
            # A static/failure-memory dead end alone cannot authorize it.
            body_corridor_m = float(self.get_parameter("robot_radius").value)+.05
            reverse_blockers = {cell for cell in observed_effective
                if any(hypot(cell[0]-waypoint[0], cell[1]-waypoint[1])
                       * map_msg.info.resolution <= body_corridor_m
                       for waypoint in reference_path)}
            if reverse_blockers:
                try:
                    without_live = self.planner.plan(
                        planning_grid, start, end, remembered_blocked,
                        allow_blocked_start=allow_dynamic_escape,
                        escape_cells=escape_cells, cell_penalty=cell_penalty,
                        blocked_edges=plan_blocked_edges, edge_penalty=failure_costs)
                except ValueError:
                    without_live = []
                if not without_live:
                    reverse_blockers.clear()
            # A verified reverse remains available before latching a persistent
            # obstacle hold; a robot can clear the blocked start without waiting
            # for a stationary object to disappear.
            if self._maybe_request_reverse(start, end, map_msg, grid, raw_grid,
                                        observed_effective | remembered_effective,
                                        blocked, cell_penalty, trail_verified,
                                        blocking_evidence=reverse_blockers):
                return
            # A changed scan invalidating a path is not a failed *alternative*.
            # Charge only a completed no-route search with a newly observed,
            # physical hit close enough to touch the robot's swept body along
            # the static reference. Inflated planner cells alone may merely
            # reflect a deliberately conservative clearance. The observation
            # message sequence also prevents retry_transform() from spending
            # the whole budget on one cached scan and unchanged A* result.
            now = time.monotonic()
            body_corridor_m = (float(self.get_parameter("robot_radius").value)
                               + .05)
            physical_hits = [cell for cell in observed_effective
                             if any(hypot(cell[0]-waypoint[0],
                                          cell[1]-waypoint[1])
                                    * map_msg.info.resolution <= body_corridor_m
                                    for waypoint in reference_path)]
            if (physical_hits and self.observed_revision > self.last_no_route_observed_revision
                    and self.last_observation_time is not None
                    and 0 <= now-self.last_observation_time < .75
                    and self._automatic_resume_allowed()
                    and now-self.last_no_route_obstacle_charge_at >=
                    max(2.0, self._retry_period())):
                pose = self._fresh_motion_pose()
                route_without_live_hit = []
                if pose is not None:
                    # The live return must actually be pivotal. A no-route
                    # caused by a static wall or old
                    # memory is not evidence that moving this object helps.
                    try:
                        route_without_live_hit = self.planner.plan(
                            grid, start, end, remembered_blocked,
                            cell_penalty=cell_penalty,
                            blocked_edges=failure_edges, edge_penalty=failure_costs)
                    except ValueError:
                        route_without_live_hit = []
                if pose is not None and route_without_live_hit:
                    before = self.obstacle_hold.attempts
                    self._register_obstacle_failure(
                        pose[0], pose[1], physical_hits,
                        candidate_cells=physical_hits)
                    if self.obstacle_hold.attempts > before:
                        self.last_no_route_obstacle_charge_at = now
                        self.last_no_route_observed_revision = self.observed_revision
            if (self.failure_state != "SENSOR_HOLD" and self.plan_metrics.get("metric_connector_reason")
                    == "actual_body_intersection"):
                self.metric_no_path_key = no_path_key
            return
        if (not certified_start_edges and escape_cells and start in blocked):
            chosen = self._selected_dynamic_escape_certificate(
                start, cells, grid, raw_grid, measured_blocked,
                observed_effective, remembered_effective)
            if not chosen:
                self._invalidate_path("Dynamic start escape has no fresh chosen-first-edge certificate")
                reason = self.plan_metrics["selected_start_certificate_diagnostics"]["reason"]
                if reason in {"actual_memory_inflation_start", "later_measured_or_static_cell_blocked",
                              "reentry_or_adjacent_diagonal_forbidden"}:
                    self._wait_after_recovery_cycle(relevant_cells=cells)
                return
            certified_start_edges = chosen
            certification_blocked = measured_blocked | {start}
        path = Path()
        path.header.frame_id = frame
        now = self.get_clock().now()
        # Cross-topic ordering has no DDS guarantee. A replacement must be
        # strictly newer than the failed request even within one /clock tick.
        cutoff = self.last_failure_event_stamp_ns
        path.header.stamp = (rclpy.time.Time(nanoseconds=max(now.nanoseconds, cutoff+1)).to_msg()
                             if cutoff else now.to_msg())
        points = [(x + .5, y + .5) for x, y in cells]
        smoothed = False
        if (self.get_parameter("path_smoothing_enabled").value and not certified_start_edges
                and not soft_controller_edges):
            resolution = map_msg.info.resolution
            points, smoothed = smooth_grid_path(
                cells, lambda cell: grid.is_static_obstacle(cell) or cell in blocked,
                spacing=max(.01, float(self.get_parameter("path_resample_spacing").value)) / resolution,
                shortcut_distance=max(.01, float(self.get_parameter(
                    "path_smoothing_shortcut_distance").value)) / resolution,
                iterations=int(self.get_parameter("path_smoothing_iterations").value),
                cell_penalty=cell_penalty)
        # Dynamic obstacle invalidation must cover the route actually sent to
        # Nav2, including all crossed cells, rather than only the old A* cells.
        planned_cells = list(cells) + sorted(route_cells(points) - set(cells))
        world_points = [cell_to_world(map_msg.info, (x-.5, y-.5)) for x, y in points]
        if failure_costs:
            current = self._fresh_motion_pose()
            if current is None:
                self.failure_state, self.failure_reason = "SENSOR_HOLD", "Fresh pose required for execution recovery"
                self._invalidate_path(self.failure_reason)
                return
            self.failure_state, self.failure_reason = (
                "REJOINING",
                "Following bounded execution-approach costs; verifying real progress")
            if self.rejoin_anchor is None:
                self.rejoin_anchor = {"progress": path_progress(world_points, current[:2]),
                                      "odom": self.latest_odom, "points": world_points,
                                      "pose": current}
                self.rejoin_verified = False
        self.planned_cells = planned_cells
        self.waiting_for_replan = False
        self.invalid_path_published = False
        self.planned_world_points = world_points
        self.local_reference_points = list(world_points)
        self.local_reference_key = self._local_route_key()
        self.local_cell_penalty = cell_penalty
        self.local_rejoin_anchor = None
        self._local_footprint_grid()
        self.plan_metrics.update(global_duration_ms=(time.perf_counter()-global_started)*1000.)
        if not failure_costs:
            self.failure_state, self.failure_reason = "CLEAR", ""
            self.plan_metrics.update(failure_class="none", recovery_stage="tracking")
        for index, point in enumerate(world_points):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = point
            if index + 1 < len(world_points):
                next_x, next_y = world_points[index + 1]
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
        if (certified_start_edges and (cells[0], cells[1]) not in
                self._retained_start_certificate(
                    start, certified_start_edges, grid, raw_grid, certification_blocked,
                    observed_effective, remembered_effective,
                    require_occlusion_context=bool(self.plan_metrics.get("occlusion_fallback")))):
            self.plan_metrics.update(occlusion_fallback=False)
            self._invalidate_path("Fresh start-only certificate expired before route publication")
            return
        if certified_start_edges:
            current = self._fresh_motion_pose()
            if current is None:
                self._hold_start_prefix_inputs()
                return
            if not self._prefix_sweep_clear([current[:2], *world_points[:2]]):
                if self.plan_metrics.get("start_prefix_validation", {}).get("input_pending"):
                    self._hold_start_prefix_inputs()
                else:
                    self._invalidate_path("Raw body or sweep blocks the selected first edge")
                    self._wait_after_recovery_cycle(relevant_cells=cells)
                return
        self._publish_path(path)
        if certified_start_edges and len(cells) >= 2:
            self._install_start_certificate((cells[0], cells[1]))
        detail = f"A* route ready: {len(path.poses)} poses"
        detail += "; collision-checked smoothing" if smoothed else "; original grid route"
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
