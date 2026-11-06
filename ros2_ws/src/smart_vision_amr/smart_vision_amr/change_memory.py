"""LiDAR visibility based change observation and persistent memory node."""

import hashlib
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from smart_vision_amr.obstacle_memory import ObstacleMemory
from smart_vision_amr.scan_outlier_guard import NearScanOutlierGuard
from smart_vision_amr.map_utils import (bresenham, cell_to_world, world_to_cell,
                                       occupancy_to_grid, first_static_obstacle_distance,
                                       distance_to_known_wall_area, wall_echo_reference_grid)


def wall_aligned_echo_indices(samples, minimum_beams=6, minimum_span_m=0.10):
    """Return indices in contiguous, spatially extended mapped-wall echoes.

    ``samples`` contains only returns whose range is within the bounded wall
    band. A single close return remains an obstacle; a run must span a real
    stretch of wall, not merely several rays hitting one small object.
    """
    runs = []
    current = []
    for index in sorted(samples):
        if current and index != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(index)
    if current:
        runs.append(current)
    matched = set()
    for run in runs:
        if len(run) < minimum_beams:
            continue
        points = [samples[index] for index in run]
        span = math.hypot(max(x for x, _ in points) - min(x for x, _ in points),
                          max(y for _, y in points) - min(y for _, y in points))
        if span >= minimum_span_m:
            matched.update(run)
    return matched


class ChangeMemoryNode(Node):
    def __init__(self):
        super().__init__("change_memory")
        # Serialize application state, but do not share its group with the
        # automatically created /clock subscription. A projection must not
        # freeze ROS time while TF arrives on the other executor worker.
        self.application_callbacks = MutuallyExclusiveCallbackGroup()
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("memory_path", "~/.ros/smart_vision_amr/environment_memory.json")
        self.declare_parameter("occupied_threshold", 65)
        # Compare measured range with the first known wall, not wall proximity.
        self.declare_parameter("mapped_wall_range_tolerance_m", 0.05)
        # 0 disables the cap. Only obstacle hits are range-limited; free-ray
        # evidence still traverses to the actual return or scan.range_max.
        self.declare_parameter("live_obstacle_max_range_m", 0.0)
        # Optional UI-only layer. Zero disables it; it never feeds A*, the
        # persistent memory, visibility proof or any motor collision guard.
        self.declare_parameter("display_obstacle_max_range_m", 0.0)
        self.declare_parameter("confirm_observations", 5)
        self.declare_parameter("clear_observations", 5)
        self.declare_parameter("confirm_min_duration_sec", 2.0)
        self.declare_parameter("clear_min_duration_sec", 2.0)
        self.declare_parameter("min_observation_interval_sec", 0.25)
        self.declare_parameter("save_every_scans", 5)
        self.declare_parameter("memory_enabled", True)
        # Runtime-only memory survives sector goals but starts empty with each
        # navigation process. Disk persistence remains available explicitly.
        self.declare_parameter("memory_persist_to_disk", True)
        # This short-lived layer is independent of optional persisted memory.
        self.declare_parameter("live_obstacle_ttl_sec", 2.0)
        self.declare_parameter("live_scan_max_age_sec", 1.0)
        self.declare_parameter("live_transform_max_age_sec", 1.0)
        self.declare_parameter("scan_tf_wait_sec", 0.20)
        self.declare_parameter("scan_tf_queue_size", 5)
        self.declare_parameter("scan_tf_stamp_tolerance_sec", 0.02)
        self.declare_parameter("scan_future_tolerance_sec", 0.15)
        self.declare_parameter("live_localization_jump_distance", 0.25)
        self.declare_parameter("live_localization_jump_angle", 0.25)
        self.raw_static_grid = None
        self.wall_echo_grid = None
        self.map = None
        self.memory = None
        self.live_obstacles = {}
        self.display_obstacles = {}
        self.near_scan_guard = NearScanOutlierGuard()
        self.last_scan_stamp_ns = None
        self.last_map_to_odom = None
        self.scans_since_save = 0
        self.pending_scans = deque()
        self.projection_dropped = 0
        self.projection_valid = 0
        self.last_projection_diagnostic = float("-inf")
        self.last_valid_projection_diagnostic = float("-inf")
        self.memory_path = Path(os.path.expanduser(self.get_parameter("memory_path").value))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.overlay_pub = self.create_publisher(Int32MultiArray, "/smart_vision_amr/remembered_cells", latched)
        self.observation_pub = self.create_publisher(
            Int32MultiArray, "/smart_vision_amr/observed_obstacles", 10)
        self.display_observation_pub = self.create_publisher(
            Int32MultiArray, "/smart_vision_amr/display_obstacles", 10)
        self.visibility_pub = self.create_publisher(
            OccupancyGrid, "/smart_vision_amr/observed_visibility", qos_profile_sensor_data)
        self.projection_status_pub = self.create_publisher(
            String, "/smart_vision_amr/scan_projection_status", latched)
        self.create_subscription(OccupancyGrid, self.get_parameter("map_topic").value,
                                 self.on_map, latched, callback_group=self.application_callbacks)
        # Projection consumes the newest real measurement after a slow frame,
        # rather than replaying the sensor-data default's five-frame backlog.
        # This changes only this observer's queue, never raw /scan delivery to
        # the motion guard or Nav2. Skipped scans do not count as observations.
        latest_scan_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value,
                                 self.on_scan, latest_scan_qos,
                                 callback_group=self.application_callbacks)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initial_pose, 10,
                                 callback_group=self.application_callbacks)
        self.create_service(Trigger, "/smart_vision_amr/save_memory", self.save_service,
                            callback_group=self.application_callbacks)
        self.create_service(Trigger, "/smart_vision_amr/clear_memory", self.clear_service,
                            callback_group=self.application_callbacks)
        from rclpy.clock import Clock, ClockType
        self.create_timer(.02, self.retry_pending_scans,
                          clock=Clock(clock_type=ClockType.STEADY_TIME),
                          callback_group=self.application_callbacks)

    def map_signature(self, msg):
        info = msg.info
        metadata = {"frame": msg.header.frame_id, "width": info.width, "height": info.height,
                    "resolution": info.resolution, "origin": [info.origin.position.x,
                    info.origin.position.y, info.origin.position.z, info.origin.orientation.x,
                    info.origin.orientation.y, info.origin.orientation.z, info.origin.orientation.w]}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(bytes((int(value) + 1) & 0xff for value in msg.data))
        return digest.hexdigest()

    def on_map(self, msg):
        signature = self.map_signature(msg)
        if self.map is not None and signature == self.map["signature"]:
            return
        self.map = {"message": msg, "signature": signature}
        self._known_free_base_cells()
        self.raw_static_grid = occupancy_to_grid(
            msg, self.get_parameter("occupied_threshold").value,
            unknown_is_obstacle=False)
        # Only the unknown fringe attached to a known wall shares its static
        # boundary. Isolated unknown frontiers must not hide a new object.
        self.wall_echo_grid = wall_echo_reference_grid(
            msg, self.get_parameter("occupied_threshold").value)
        self.live_obstacles.clear()
        getattr(self, "display_obstacles", {}).clear()
        if hasattr(self, "near_scan_guard"):
            self.near_scan_guard.reset()
        self.last_scan_stamp_ns = None
        self.last_map_to_odom = None
        self.pending_scans.clear()
        path = self.memory_path
        if not self.get_parameter("memory_enabled").value:
            self.memory = self._new_memory(msg)
            self.get_logger().info("Persistent environment memory disabled for baseline condition")
        elif not self.get_parameter("memory_persist_to_disk").value:
            self.memory = self._new_memory(msg)
            self.get_logger().info("Runtime obstacle memory started empty; disk restore disabled")
        elif path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("map_signature") != signature:
                    raise ValueError("saved memory belongs to another map; refusing to apply")
                self.memory = ObstacleMemory.load(path, self._as_grid(msg))
                self.get_logger().info(f"Restored {len(self.memory.confirmed_cells)} remembered cells")
            except Exception as exc:
                self.memory = self._new_memory(msg)
                self.get_logger().error(f"Memory not restored: {exc}")
        else:
            self.memory = self._new_memory(msg)
        self.publish_overlay()

    def on_initial_pose(self, message):
        if self.map is None or self.memory is None:
            return
        if message.header.frame_id.lstrip('/') != self.get_parameter("map_frame").value.lstrip('/'):
            return
        pose = message.pose.pose
        q = pose.orientation
        if not all(math.isfinite(value) for value in (
                pose.position.x, pose.position.y, pose.position.z,
                q.x, q.y, q.z, q.w)):
            return
        if q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w < 1e-12:
            return
        cell = world_to_cell(self.map["message"].info, pose.position.x, pose.position.y)
        if not (0 <= cell[0] < self.memory.width and 0 <= cell[1] < self.memory.height):
            return
        self._discard_relocated_memory("Explicit initial pose accepted")
        self.last_map_to_odom = None

    def _discard_relocated_memory(self, reason):
        """Discard cells projected with an obsolete map localization solution."""
        self.live_obstacles.clear()
        getattr(self, "display_obstacles", {}).clear()
        if hasattr(self, "near_scan_guard"):
            self.near_scan_guard.reset()
        self.pending_scans.clear()
        self.memory = self._new_memory(self.map["message"])
        self.scans_since_save = 0
        # A latched empty overlay removes the old cells for planners and UIs.
        # /observed_obstacles remains measurement-driven: no fabricated scan.
        self.publish_overlay()
        try:
            self.persist()
        except OSError as exc:
            # Never restore the old coordinate solution on the next launch.
            try:
                self.memory_path.unlink(missing_ok=True)
            except OSError as unlink_exc:
                self.get_logger().error(
                    f"Cannot invalidate old obstacle memory file: {unlink_exc}")
            self.get_logger().error(f"Cannot save cleared obstacle memory: {exc}")
        self.get_logger().warning(f"{reason}; rebuilding obstacle memory from fresh LiDAR scans")

    def reset_live_on_localization_jump(self, target_frame, measurement_time):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, self.get_parameter("odom_frame").value, measurement_time)
        except TransformException:
            return
        stamp = transform.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        max_age_ns = max(.05, float(self.get_parameter(
            "live_transform_max_age_sec").value)) * 1e9
        if (stamp_ns == 0 or abs(stamp_ns - measurement_time.nanoseconds) > max_age_ns
                or transform.header.frame_id.lstrip('/') != target_frame.lstrip('/')
                or transform.child_frame_id.lstrip('/') != self.get_parameter(
                    "odom_frame").value.lstrip('/')):
            return
        q = transform.transform.rotation
        if (not all(math.isfinite(value) for value in (
                transform.transform.translation.x, transform.transform.translation.y,
                transform.transform.translation.z, q.x, q.y, q.z, q.w))
                or q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w < 1e-12):
            return
        pose = (transform.transform.translation.x, transform.transform.translation.y,
                self._yaw(q))
        previous = self.last_map_to_odom
        self.last_map_to_odom = pose
        if previous is None:
            return
        delta_yaw = abs(math.atan2(math.sin(pose[2]-previous[2]), math.cos(pose[2]-previous[2])))
        if (math.hypot(pose[0]-previous[0], pose[1]-previous[1])
                > self.get_parameter("live_localization_jump_distance").value
                or delta_yaw > self.get_parameter("live_localization_jump_angle").value):
            self._discard_relocated_memory("Verified map-to-odom localization jump")

    @staticmethod
    def cell_in_unknown_sector(cell, info, sensor_xy, invalid_angles, angular_step):
        """Retain only cells intersecting an invalid beam's narrow angular cone."""
        if not invalid_angles:
            return False
        x, y = cell_to_world(info, cell)
        dx, dy = x-sensor_xy[0], y-sensor_xy[1]
        distance = math.hypot(dx, dy)
        half_diagonal = info.resolution*math.sqrt(2.)*.5
        half_width = (abs(angular_step)*.5
                      + math.asin(min(1., half_diagonal/max(distance, half_diagonal))))
        bearing = math.atan2(dy, dx)
        return any(abs(math.atan2(math.sin(angle-bearing), math.cos(angle-bearing))) <= half_width
                   for angle in invalid_angles)

    def _as_grid(self, msg):
        from smart_vision_amr.map_utils import occupancy_to_grid
        return occupancy_to_grid(msg, self.get_parameter("occupied_threshold").value)

    def _new_memory(self, msg):
        memory = ObstacleMemory(msg.info.width, msg.info.height,
                                self.get_parameter("confirm_observations").value,
                                self.get_parameter("clear_observations").value,
                                confirm_min_duration_sec=self.get_parameter("confirm_min_duration_sec").value,
                                clear_min_duration_sec=self.get_parameter("clear_min_duration_sec").value,
                                min_observation_interval_sec=self.get_parameter(
                                    "min_observation_interval_sec").value)
        memory.map_signature = self.map["signature"]
        return memory

    def on_scan(self, scan):
        if self.map is None or self.memory is None:
            return
        if not self._scan_is_current(scan):
            return
        stamp_ns = scan.header.stamp.sec * 1_000_000_000 + scan.header.stamp.nanosec
        if any(item[0] == stamp_ns for item in self.pending_scans):
            return
        # A zero-time TF query means "latest" in TF2, never measurement time.
        if stamp_ns == 0:
            self._projection_diagnostic("DROPPED", "unstamped_scan", scan)
            return
        received_at = time.monotonic()
        if self._project_scan(scan, received_at=received_at):
            return
        bound = max(1, min(50, int(self.get_parameter("scan_tf_queue_size").value)))
        while len(self.pending_scans) >= bound:
            _, _, dropped = self.pending_scans.popleft()
            self._projection_diagnostic("DROPPED", "tf_queue_full", dropped)
        self.pending_scans.append((stamp_ns, received_at, scan))
        self.pending_scans = deque(sorted(self.pending_scans, key=lambda item: item[0]))
        self._projection_diagnostic("PENDING_TF", "waiting_for_measurement_transform", scan)

    def retry_pending_scans(self):
        """Project the newest acquisition-ready scan, never a stale FIFO burst.

        A newer frame may still be awaiting TF, in which case an older ready
        frame is useful. Once a newer frame succeeds, the existing stamp
        high-water mark supersedes older queued frames. Neither the original
        measurement time nor its first receipt time is renewed by a retry.
        """
        wait = max(.01, min(1., float(self.get_parameter("scan_tf_wait_sec").value)))
        retained = deque()
        while self.pending_scans:
            stamp, arrived, scan = self.pending_scans.pop()
            if not self._scan_is_current(scan):
                continue
            if time.monotonic() - arrived >= wait:
                self._projection_diagnostic("DROPPED", "measurement_tf_timeout", scan)
                continue
            previous_memory = self.memory
            if not self._project_scan(scan, received_at=arrived):
                retained.appendleft((stamp, arrived, scan))
            if self.memory is not previous_memory:
                # A verified localization jump clears the live queue. Do not
                # restore a newer waiting scan from the obsolete projection
                # context after _discard_relocated_memory cleared it.
                retained.clear()
        self.pending_scans = retained

    def _scan_is_current(self, scan):
        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = scan.header.stamp.sec * 1_000_000_000 + scan.header.stamp.nanosec
        max_age = max(.05, float(self.get_parameter("live_scan_max_age_sec").value))
        future = max(0., min(.25, float(self.get_parameter("scan_future_tolerance_sec").value)))
        if (now_ns-stamp_ns) > max_age * 1e9 or stamp_ns-now_ns > future*1e9:
            self._projection_diagnostic("DROPPED", "scan_clock_skew_or_stale", scan)
            return False
        if self.last_scan_stamp_ns is not None and stamp_ns <= self.last_scan_stamp_ns:
            return False
        if (not all(math.isfinite(value) for value in (scan.angle_min, scan.angle_increment,
                                                      scan.range_min, scan.range_max))
                or scan.angle_increment == 0.0 or not 0 <= scan.range_min < scan.range_max):
            self._projection_diagnostic("DROPPED", "invalid_scan_geometry", scan)
            return False
        return True

    def _projection_diagnostic(self, status, reason, scan, tf_skew=None, *, timing=None):
        if status == "DROPPED":
            self.projection_dropped += 1
            self.get_logger().warning(f"LiDAR projection rejected: {reason}", throttle_duration_sec=5.)
        now = time.monotonic()
        if status == "VALID":
            # A PENDING_TF event on every 10 Hz scan must not suppress every
            # subsequent VALID diagnostic indefinitely.
            if now-getattr(self, "last_valid_projection_diagnostic", float("-inf")) < 1.:
                return
            self.last_valid_projection_diagnostic = now
        self.last_projection_diagnostic = now
        stamp_ns = scan.header.stamp.sec*1_000_000_000 + scan.header.stamp.nanosec
        message = String()
        payload = {"status": status, "reason": reason,
            "scan_stamp_ns": stamp_ns,
            "scan_age_ms": (self.get_clock().now().nanoseconds-stamp_ns)/1e6,
            "tf_skew_ms": tf_skew, "queue_depth": len(self.pending_scans),
            "valid_scans": self.projection_valid, "dropped_scans": self.projection_dropped}
        if timing is not None:
            payload.update(timing)
        message.data = json.dumps(payload)
        self.projection_status_pub.publish(message)

    def _project_scan(self, scan, *, received_at=None):
        """Return false only when TF may still arrive within the bounded wait."""
        started_at = time.monotonic()
        if received_at is None:
            received_at = started_at
        stamp_ns = scan.header.stamp.sec * 1_000_000_000 + scan.header.stamp.nanosec
        target_frame = self.get_parameter("map_frame").value
        sensor_frame = scan.header.frame_id or self.get_parameter("base_frame").value
        measurement_time = rclpy.time.Time.from_msg(scan.header.stamp)
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, sensor_frame, measurement_time)
        except TransformException as exc:
            # A LiDAR measurement often reaches this process 10-100 ms before
            # the matching odom/AMCL transform. The queued retry below handles
            # that normal ordering; warn only if the bounded wait expires.
            self.get_logger().debug(f"Scan-stamped TF not yet ready: {exc}", throttle_duration_sec=5.0)
            return False
        tf_ns = tf.header.stamp.sec * 1_000_000_000 + tf.header.stamp.nanosec
        tf_max_age = max(.05, float(self.get_parameter("live_transform_max_age_sec").value))
        tolerance = min(tf_max_age, max(.000001, float(self.get_parameter(
            "scan_tf_stamp_tolerance_sec").value)))
        skew_ms = (tf_ns-stamp_ns)/1e6 if tf_ns else 0.
        if tf_ns and abs(stamp_ns-tf_ns) > tolerance * 1e9:
            self._projection_diagnostic("DROPPED", "tf_measurement_stamp_skew", scan, skew_ms)
            return True
        q = tf.transform.rotation
        if not all(math.isfinite(value) for value in (tf.transform.translation.x,
                tf.transform.translation.y, q.x, q.y, q.z, q.w)):
            self._projection_diagnostic("DROPPED", "invalid_transform", scan, skew_ms)
            return True
        self.reset_live_on_localization_jump(target_frame, measurement_time)
        info = self.map["message"].info
        # One immutable lookup per projection replaces thousands of ROS
        # parameter lookups along the rays. Map/threshold changes invalidate
        # this cache; unknown cells still never become free evidence.
        base_free_cells = self._known_free_base_cells()
        origin = world_to_cell(info, tf.transform.translation.x, tf.transform.translation.y)
        yaw = self._yaw(tf.transform.rotation)
        occupied_cells, free_cells, uncertain_cells = set(), set(), set()
        unmatched_hits = []
        display_hits, display_occupied = [], set()
        wall_band_candidates = {}
        wall_band_samples = {}
        display_wall_candidates, display_wall_samples = {}, {}
        invalid_angles = []
        valid_beams = 0
        # Hold even a multi-beam near-minimum cluster as unknown until a
        # second scan confirms it. Raw /scan still reaches Nav2 unchanged.
        guard = getattr(self, "near_scan_guard", None)
        if guard is None:  # Lightweight test harnesses bypass Node.__init__.
            guard = self.near_scan_guard = NearScanOutlierGuard()
        outliers = guard.classify(scan, stamp_ns)
        if not outliers.accepted:
            return True
        wall_tolerance = float(self.get_parameter("mapped_wall_range_tolerance_m").value)
        # Invalid tolerance must never discard actual obstacles.
        wall_tolerance = max(0., wall_tolerance) if math.isfinite(wall_tolerance) else 0.
        obstacle_range = float(self.get_parameter("live_obstacle_max_range_m").value)
        obstacle_range = obstacle_range if math.isfinite(obstacle_range) and obstacle_range > 0. else math.inf
        display_range = float(self.get_parameter("display_obstacle_max_range_m").value)
        display_range = display_range if math.isfinite(display_range) and display_range > 0. else 0.
        for index, distance in enumerate(scan.ranges):
            angle = yaw + scan.angle_min + index * scan.angle_increment
            if index in outliers.unconfirmed:
                uncertain_cells.add(world_to_cell(
                    info, tf.transform.translation.x + distance*math.cos(angle),
                    tf.transform.translation.y + distance*math.sin(angle)))
                invalid_angles.append(angle)
                continue
            # NaN, zero/below-minimum and out-of-range packets are not evidence
            # that the ray is free. Positive infinity is the LaserScan no-hit case.
            if (math.isnan(distance) or distance == -math.inf
                    or distance < scan.range_min
                    or (math.isfinite(distance) and distance > scan.range_max)):
                invalid_angles.append(angle)
                continue
            valid_beams += 1
            valid_hit = scan.range_min <= distance <= scan.range_max and distance < scan.range_max
            ray_length = distance if valid_hit else scan.range_max
            endpoint = (tf.transform.translation.x + ray_length * math.cos(angle),
                        tf.transform.translation.y + ray_length * math.sin(angle))
            end_cell = world_to_cell(info, *endpoint)
            ray = list(bresenham(origin, end_cell))
            for cell in ray[:-1] if valid_hit else ray:
                if cell in base_free_cells:
                    free_cells.add(cell)
            if valid_hit and distance <= max(obstacle_range, display_range) and self.raw_static_grid.contains(end_cell):
                wall_distance = first_static_obstacle_distance(
                    info, self.raw_static_grid, tf.transform.translation.x,
                    tf.transform.translation.y, angle, scan.range_max)
                # A nearby object in front of a wall remains a dynamic hit.
                # Unknown cells are not evidence of a mapped wall.
                wall_echo = (wall_distance is not None
                             and abs(wall_distance-distance) <= wall_tolerance)
                # Separate beam groups prevent an added distant UI return
                # from changing the planner's existing near-wall classifier.
                layers = []
                if distance <= obstacle_range:
                    layers.append((wall_band_samples, wall_band_candidates, unmatched_hits))
                if distance <= display_range:
                    layers.append((display_wall_samples, display_wall_candidates, display_hits))
                if wall_echo:
                    for samples, candidates, hits in layers:
                        samples[index] = endpoint
                elif (wall_distance is not None
                        and wall_tolerance < wall_distance-distance <= 0.10
                        and distance_to_known_wall_area(
                            info, self.wall_echo_grid, *endpoint, 0.10) is not None):
                    # A map/laser edge can differ by one or two 5 cm cells.
                    # Suppress that extra overlay inflation only when many
                    # adjacent beams trace an extended mapped wall. A short
                    # near-wall return remains a live obstacle. Nav2 receives
                    # the original LaserScan regardless of this classification.
                    for samples, candidates, hits in layers:
                        samples[index] = endpoint
                        candidates[index] = (endpoint, end_cell)
                else:
                    for samples, candidates, hits in layers:
                        hits.append((endpoint, end_cell))
        if not valid_beams:
            return True
        wall_band = wall_aligned_echo_indices(wall_band_samples)
        unmatched_hits.extend(hit for index, hit in wall_band_candidates.items()
                              if index not in wall_band)
        if display_range:
            display_band = wall_aligned_echo_indices(display_wall_samples)
            display_hits.extend(hit for index, hit in display_wall_candidates.items()
                                if index not in display_band)
        for hits, cells in ((unmatched_hits, occupied_cells), (display_hits, display_occupied)):
            for endpoint, end_cell in hits:
                # Use the same original wall/fringe classifier in both layers.
                # Display cells never enter planner visibility or memory.
                if distance_to_known_wall_area(
                        info, self.wall_echo_grid, *endpoint, wall_tolerance) is not None:
                    continue
                cells.add(end_cell)
        self.last_scan_stamp_ns = stamp_ns
        self.projection_valid += 1
        free_cells.difference_update(occupied_cells)
        # Adjacent clear rays can traverse the same 5 cm map cell as one tiny
        # uncertain return. That cell stays unknown until the next scan too.
        free_cells.difference_update(uncertain_cells)
        now = time.monotonic()
        ttl = max(.05, float(self.get_parameter("live_obstacle_ttl_sec").value))
        # Keep briefly occluded returns, but clear a cell as soon as a valid ray
        # traverses it. A few hardware edge dropouts must not freeze every old
        # obstacle in the map: protect only each dropout's actual angular cone.
        # Neither TTL nor relocalization edits the saved SLAM grid.
        sensor_xy = (tf.transform.translation.x, tf.transform.translation.y)
        cell_radius = info.resolution * math.sqrt(2.) * .5
        self.live_obstacles = {cell: seen for cell, seen in self.live_obstacles.items()
                               if self.raw_static_grid.contains(cell) and cell not in free_cells
                               and math.dist(cell_to_world(info, cell), sensor_xy)
                                   <= obstacle_range + cell_radius
                               and (now-seen < ttl or self.cell_in_unknown_sector(
                                   cell, info, sensor_xy, invalid_angles, scan.angle_increment))}
        self.live_obstacles.update((cell, now) for cell in occupied_cells)
        if display_range:
            display_free = free_cells-display_occupied
            self.display_obstacles = {
                cell: seen for cell, seen in getattr(self, "display_obstacles", {}).items()
                if self.raw_static_grid.contains(cell) and cell not in display_free
                and math.dist(cell_to_world(info, cell), sensor_xy) <= display_range+cell_radius
                and (now-seen < ttl or self.cell_in_unknown_sector(
                    cell, info, sensor_xy, invalid_angles, scan.angle_increment))}
            self.display_obstacles.update((cell, now) for cell in display_occupied)
            display = Int32MultiArray()
            display.data = sorted(y*self.memory.width+x for x, y in self.display_obstacles)
            self.display_observation_pub.publish(display)
        else:
            getattr(self, "display_obstacles", {}).clear()
        # A missing hit/TTL expiry is not removal evidence. Publish exactly
        # which cells this measurement's valid rays traversed, with its stamp.
        if hasattr(self, "visibility_pub"):
            visibility = OccupancyGrid()
            visibility.header.frame_id = target_frame
            visibility.header.stamp = scan.header.stamp
            visibility.info = info
            data = [-1] * (info.width*info.height)
            for x, y in free_cells:
                data[y*info.width+x] = 0
            for x, y in occupied_cells:
                data[y*info.width+x] = 100
            visibility.data = data
            self.visibility_pub.publish(visibility)
        # Publish the stamped proof before the geometry-trigger message. A
        # consumer may immediately plan from observed_obstacles; it should not
        # first be forced to wait for this same frame's visibility publication.
        # DDS topic order is not assumed: consumers still match original stamps.
        observed = Int32MultiArray()
        observed.data = sorted(y * self.memory.width + x for x, y in self.live_obstacles)
        self.observation_pub.publish(observed)
        published_at = time.monotonic()
        self._projection_diagnostic(
            "VALID", "measurement_transform_applied", scan, skew_ms,
            timing={"projection_duration_ms": (published_at-started_at)*1000.,
                    "first_receipt_to_publish_ms": (published_at-received_at)*1000.})
        if not self.get_parameter("memory_enabled").value:
            return True
        changed = False
        observed_at = time.time()
        for cell in free_cells:
            changed = self.memory.observe_free(cell, observed_at) or changed
        for cell in occupied_cells:
            # Live returns may include raw occupied/unknown cells; persistent
            # changes only describe additions to known free cells of the map.
            if cell in base_free_cells:
                changed = self.memory.observe_occupied(cell, observed_at) or changed
        self.scans_since_save += 1
        if changed:
            self.publish_overlay()
            self.persist()
            self.scans_since_save = 0
        interval = self.get_parameter("save_every_scans").value
        if interval > 0 and self.scans_since_save >= interval:
            self.persist()
            self.scans_since_save = 0
        return True

    def _is_free_in_base_map(self, cell):
        return cell in self._known_free_base_cells()

    def _known_free_base_cells(self):
        """Cache only known-free map cells, with no changed occupancy policy."""
        msg = self.map["message"]
        threshold = self.get_parameter("occupied_threshold").value
        if (getattr(self, "_base_free_message", None) is not msg
                or getattr(self, "_base_free_threshold", None) != threshold):
            self._base_free_cells = frozenset(
                (index % msg.info.width, index // msg.info.width)
                for index, value in enumerate(msg.data) if 0 <= value < threshold)
            self._base_free_message = msg
            self._base_free_threshold = threshold
        return self._base_free_cells

    @staticmethod
    def _yaw(q):
        from math import atan2
        return atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def publish_overlay(self):
        if self.memory is None:
            return
        msg = Int32MultiArray()
        msg.data = [index for x, y in sorted(self.memory.confirmed_cells)
                    for index in (y * self.memory.width + x,)]
        self.overlay_pub.publish(msg)

    def persist(self):
        if (self.memory is not None and self.get_parameter("memory_enabled").value
                and self.get_parameter("memory_persist_to_disk").value):
            self.memory.map_signature = self.map["signature"]
            self.memory.save(self.memory_path)

    def save_service(self, request, response):
        if not self.get_parameter("memory_enabled").value:
            response.success, response.message = False, "Memory is disabled in baseline mode"
            return response
        if not self.get_parameter("memory_persist_to_disk").value:
            response.success, response.message = False, "Disk persistence is disabled for this navigation session"
            return response
        try:
            self.persist()
            response.success, response.message = True, "Memory saved"
        except Exception as exc:
            response.success, response.message = False, str(exc)
        return response

    def clear_service(self, request, response):
        if self.map is None:
            response.success, response.message = False, "Base map is not available"
            return response
        self.memory = self._new_memory(self.map["message"])
        self.persist()
        self.publish_overlay()
        response.success, response.message = True, "Memory cleared"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ChangeMemoryNode()
    # The scan projection can be CPU-heavy. TransformListener already uses a
    # reentrant callback group. Application state has its own serialized group,
    # so a second worker keeps both TF and /clock arriving during a projection.
    executor = MultiThreadedExecutor(num_threads=2, context=node.context)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.persist()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
