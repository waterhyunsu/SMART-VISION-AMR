"""LiDAR visibility based change observation and persistent memory node."""

import hashlib
import json
import os
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from smart_vision_amr.obstacle_memory import ObstacleMemory
from smart_vision_amr.map_utils import bresenham, world_to_cell


class ChangeMemoryNode(Node):
    def __init__(self):
        super().__init__("change_memory")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("memory_path", "~/.ros/smart_vision_amr/environment_memory.json")
        self.declare_parameter("occupied_threshold", 65)
        self.declare_parameter("confirm_observations", 5)
        self.declare_parameter("clear_observations", 5)
        self.declare_parameter("confirm_min_duration_sec", 2.0)
        self.declare_parameter("clear_min_duration_sec", 2.0)
        self.declare_parameter("min_observation_interval_sec", 0.25)
        self.declare_parameter("save_every_scans", 5)
        self.declare_parameter("memory_enabled", True)
        self.map = None
        self.memory = None
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
        target_frame = self.get_parameter("map_frame").value
        sensor_frame = scan.header.frame_id or self.get_parameter("base_frame").value
        try:
            # Gazebo Classic may publish scans just ahead of the latest TF on WSL. Use
            # the newest transform so valid scans are not dropped on small clock skew.
            tf = self.tf_buffer.lookup_transform(target_frame, sensor_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warning(f"Waiting for map-to-base TF: {exc}", throttle_duration_sec=5.0)
            return
        info = self.map["message"].info
        origin = world_to_cell(info, tf.transform.translation.x, tf.transform.translation.y)
        yaw = self._yaw(tf.transform.rotation)
        occupied_cells, free_cells = set(), set()
        for index, distance in enumerate(scan.ranges):
            angle = yaw + scan.angle_min + index * scan.angle_increment
            valid_hit = scan.range_min <= distance <= scan.range_max and distance < scan.range_max
            ray_length = distance if valid_hit else scan.range_max
            endpoint = (tf.transform.translation.x + ray_length * __import__("math").cos(angle),
                        tf.transform.translation.y + ray_length * __import__("math").sin(angle))
            end_cell = world_to_cell(info, *endpoint)
            ray = list(bresenham(origin, end_cell))
            for cell in ray[:-1] if valid_hit else ray:
                if self._is_free_in_base_map(cell):
                    free_cells.add(cell)
            if valid_hit and self._is_free_in_base_map(end_cell):
                occupied_cells.add(end_cell)
        free_cells.difference_update(occupied_cells)
        observed = Int32MultiArray()
        observed.data = sorted(y * self.memory.width + x for x, y in occupied_cells)
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
        rclpy.shutdown()
