"""LiDAR visibility based change observation and persistent memory node."""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from smart_vision_amr.obstacle_memory import ObstacleMemory
from smart_vision_amr.map_utils import bresenham, cell_to_world, world_to_cell


class ChangeMemoryNode(Node):
    def __init__(self):
        super().__init__("change_memory")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("memory_path", "~/.ros/smart_vision_amr/environment_memory.json")
        self.declare_parameter("occupied_threshold", 65)
        self.declare_parameter("confirm_observations", 5)
        self.declare_parameter("clear_observations", 5)
        self.declare_parameter("confirm_min_duration_sec", 2.0)
        self.declare_parameter("clear_min_duration_sec", 2.0)
        self.declare_parameter("min_observation_interval_sec", 0.25)
        self.declare_parameter("save_every_scans", 5)
        self.declare_parameter("memory_enabled", True)
        # This short-lived layer is independent of optional persisted memory.
        self.declare_parameter("live_obstacle_ttl_sec", 2.0)
        self.declare_parameter("live_scan_max_age_sec", 1.0)
        self.declare_parameter("live_transform_max_age_sec", 1.0)
        self.declare_parameter("live_localization_jump_distance", 0.25)
        self.declare_parameter("live_localization_jump_angle", 0.25)
        self.map = None
        self.memory = None
        self.live_obstacles = {}
        self.last_scan_stamp_ns = None
        self.last_map_to_odom = None
        self.scans_since_save = 0
        self.memory_path = Path(os.path.expanduser(self.get_parameter("memory_path").value))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.overlay_pub = self.create_publisher(Int32MultiArray, "/smart_vision_amr/remembered_cells", latched)
        self.observation_pub = self.create_publisher(
            Int32MultiArray, "/smart_vision_amr/observed_obstacles", 10)
        self.create_subscription(OccupancyGrid, self.get_parameter("map_topic").value, self.on_map, latched)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value,
                                 self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initial_pose, 10)
        self.create_service(Trigger, "/smart_vision_amr/save_memory", self.save_service)
        self.create_service(Trigger, "/smart_vision_amr/clear_memory", self.clear_service)

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
        self.live_obstacles.clear()
        self.last_scan_stamp_ns = None
        self.last_map_to_odom = None
        path = self.memory_path
        if not self.get_parameter("memory_enabled").value:
            self.memory = self._new_memory(msg)
            self.get_logger().info("Persistent environment memory disabled for baseline condition")
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
        if message.header.frame_id.lstrip('/') != self.get_parameter("map_frame").value.lstrip('/'):
            return
        # These cells were projected using the old localization solution. The
        # next valid scan rebuilds them; do not announce a fresh empty scan here.
        self.live_obstacles.clear()
        self.last_map_to_odom = None

    def reset_live_on_localization_jump(self, target_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, self.get_parameter("odom_frame").value, rclpy.time.Time())
        except TransformException:
            return
        pose = (transform.transform.translation.x, transform.transform.translation.y,
                self._yaw(transform.transform.rotation))
        if not all(math.isfinite(value) for value in pose):
            return
        previous = self.last_map_to_odom
        self.last_map_to_odom = pose
        if previous is None:
            return
        delta_yaw = abs(math.atan2(math.sin(pose[2]-previous[2]), math.cos(pose[2]-previous[2])))
        if (math.hypot(pose[0]-previous[0], pose[1]-previous[1])
                > self.get_parameter("live_localization_jump_distance").value
                or delta_yaw > self.get_parameter("live_localization_jump_angle").value):
            self.live_obstacles.clear()
            self.get_logger().warning(
                "Localization frame changed; rebuilding transient LiDAR obstacles from fresh scans")

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
        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = scan.header.stamp.sec * 1_000_000_000 + scan.header.stamp.nanosec
        max_age = max(.05, float(self.get_parameter("live_scan_max_age_sec").value))
        if abs(now_ns-stamp_ns) > max_age * 1e9:
            return
        if self.last_scan_stamp_ns is not None and stamp_ns <= self.last_scan_stamp_ns:
            return
        if (not all(math.isfinite(value) for value in (scan.angle_min, scan.angle_increment,
                                                      scan.range_min, scan.range_max))
                or scan.angle_increment == 0.0 or not 0 <= scan.range_min < scan.range_max):
            return
        target_frame = self.get_parameter("map_frame").value
        sensor_frame = scan.header.frame_id or self.get_parameter("base_frame").value
        try:
            # Gazebo Classic may publish scans just ahead of the latest TF on WSL. Use
            # the newest transform so valid scans are not dropped on small clock skew.
            tf = self.tf_buffer.lookup_transform(target_frame, sensor_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warning(f"Waiting for map-to-base TF: {exc}", throttle_duration_sec=5.0)
            return
        tf_ns = tf.header.stamp.sec * 1_000_000_000 + tf.header.stamp.nanosec
        tf_max_age = max(.05, float(self.get_parameter("live_transform_max_age_sec").value))
        if tf_ns and abs(stamp_ns-tf_ns) > tf_max_age * 1e9:
            return
        q = tf.transform.rotation
        if not all(math.isfinite(value) for value in (tf.transform.translation.x,
                tf.transform.translation.y, q.x, q.y, q.z, q.w)):
            return
        self.reset_live_on_localization_jump(target_frame)
        info = self.map["message"].info
        origin = world_to_cell(info, tf.transform.translation.x, tf.transform.translation.y)
        yaw = self._yaw(tf.transform.rotation)
        occupied_cells, free_cells = set(), set()
        invalid_angles = []
        valid_beams = 0
        for index, distance in enumerate(scan.ranges):
            angle = yaw + scan.angle_min + index * scan.angle_increment
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
                if self._is_free_in_base_map(cell):
                    free_cells.add(cell)
            if valid_hit and self._is_free_in_base_map(end_cell):
                occupied_cells.add(end_cell)
        if not valid_beams:
            return
        self.last_scan_stamp_ns = stamp_ns
        free_cells.difference_update(occupied_cells)
        now = time.monotonic()
        ttl = max(.05, float(self.get_parameter("live_obstacle_ttl_sec").value))
        # Keep briefly occluded returns, but clear a cell as soon as a valid ray
        # traverses it. A few hardware edge dropouts must not freeze every old
        # obstacle in the map: protect only each dropout's actual angular cone.
        # Neither TTL nor relocalization edits the saved SLAM grid.
        sensor_xy = (tf.transform.translation.x, tf.transform.translation.y)
        self.live_obstacles = {cell: seen for cell, seen in self.live_obstacles.items()
                               if self._is_free_in_base_map(cell) and cell not in free_cells
                               and (now-seen < ttl or self.cell_in_unknown_sector(
                                   cell, info, sensor_xy, invalid_angles, scan.angle_increment))}
        self.live_obstacles.update((cell, now) for cell in occupied_cells)
        observed = Int32MultiArray()
        observed.data = sorted(y * self.memory.width + x for x, y in self.live_obstacles)
        self.observation_pub.publish(observed)
        if not self.get_parameter("memory_enabled").value:
            return
        changed = False
        observed_at = time.time()
        for cell in free_cells:
            changed = self.memory.observe_free(cell, observed_at) or changed
        for cell in occupied_cells:
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

    def _is_free_in_base_map(self, cell):
        msg = self.map["message"]
        x, y = cell
        if not (0 <= x < msg.info.width and 0 <= y < msg.info.height):
            return False
        value = msg.data[y * msg.info.width + x]
        return 0 <= value < self.get_parameter("occupied_threshold").value

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
        if self.memory is not None and self.get_parameter("memory_enabled").value:
            self.memory.map_signature = self.map["signature"]
            self.memory.save(self.memory_path)

    def save_service(self, request, response):
        if not self.get_parameter("memory_enabled").value:
            response.success, response.message = False, "Memory is disabled in baseline mode"
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.persist()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
