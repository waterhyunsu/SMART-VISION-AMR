"""Forward planned paths to Nav2's FollowPath controller action."""

import json
import math

import rclpy
from geometry_msgs.msg import PolygonStamped, PoseStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener

from smart_vision_amr.path_geometry import paths_are_equivalent


class PathFollower(Node):
    def __init__(self):
        super().__init__("path_follower")
        self.declare_parameter("action_name", "/follow_path")
        self.declare_parameter("controller_id", "")
        self.declare_parameter("goal_checker_id", "")
        self.declare_parameter("controller_frame", "odom")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.client = ActionClient(self, FollowPath, self.get_parameter("action_name").value)
        self.status_pub = self.create_publisher(String, "/smart_vision_amr/navigation_status", 10)
        self.active_goal = None
        self.goal_request_pending = False
        self.pending_path = None
        self.cancel_requested = False
        self.cancel_when_accepted = False
        self.latest_path_key = None
        self.active_path = None
        self.safety_fault_reason = None
        self.latest_scan = None
        self.latest_odom = None
        self.latest_footprint = None
        self.latest_costmap = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(PolygonStamped, "/local_costmap/published_footprint",
                                 self._on_footprint, latched)
        self.create_subscription(OccupancyGrid, "/local_costmap/costmap",
                                 self._on_costmap, latched)
        self.create_subscription(Path, "/smart_vision_amr/path", self.on_path, 10)
        self.create_service(Trigger, "/smart_vision_amr/cancel_navigation", self.cancel_navigation)
        self.create_timer(0.2, self._watch_navigation_inputs)

    def _on_scan(self, message):
        self.latest_scan = message

    def _on_odom(self, message):
        self.latest_odom = message

    def _on_footprint(self, message):
        self.latest_footprint = message

    def _on_costmap(self, message):
        self.latest_costmap = message

    def _message_age(self, message):
        stamp = message.header.stamp
        return self.get_clock().now().nanoseconds * 1e-9 - (stamp.sec + stamp.nanosec * 1e-9)

    def _navigation_input_error(self):
        for label, message, max_age in (
                ("scan", self.latest_scan, 0.75),
                ("odom", self.latest_odom, 0.75),
                ("local costmap footprint", self.latest_footprint, 0.75),
                ("local costmap", self.latest_costmap, 1.5)):
            if message is None:
                return f"No {label} received"
            age = self._message_age(message)
            if age > max_age or age < -0.5:
                return f"{label} is stale (age {age:.2f}s)"
        if self.latest_odom.header.frame_id != self.latest_costmap.header.frame_id:
            return "Odom and local costmap frames differ"
        if not self.latest_footprint.polygon.points:
            return "Local costmap footprint is empty"
        x = self.latest_odom.pose.pose.position.x
        y = self.latest_odom.pose.pose.position.y
        points = self.latest_footprint.polygon.points
        footprint_x = sum(point.x for point in points) / len(points)
        footprint_y = sum(point.y for point in points) / len(points)
        if math.hypot(x - footprint_x, y - footprint_y) > 0.25:
            return "Local costmap footprint disagrees with odometry"
        info = self.latest_costmap.info
        center_x = info.origin.position.x + info.width * info.resolution / 2
        center_y = info.origin.position.y + info.height * info.resolution / 2
        if math.hypot(x - center_x, y - center_y) > 0.35:
            return "Local costmap is not centered on the robot"
        return None

    def _watch_navigation_inputs(self):
        if self.active_goal is None or self.cancel_requested:
            return
        issue = self._navigation_input_error()
        if issue:
            self.safety_fault_reason = issue
            self.pending_path = None
            self.get_logger().error(f"Cancelling FollowPath: {issue}")
            self._cancel_active_goal()

    def cancel_navigation(self, request, response):
        if self.active_goal is None:
            response.success, response.message = True, "No active navigation goal"
            return response
        self.active_goal.cancel_goal_async()
        response.success, response.message = True, "Cancellation requested"
        return response

    def on_path(self, path):
        if not path.poses:
            self.get_logger().error("Planner reported no path; cancelling active navigation")
            self.pending_path = None
            self.latest_path_key = None
            self.active_path = None
            if self.active_goal is not None:
                self._cancel_active_goal()
            elif self.goal_request_pending:
                self.cancel_when_accepted = True
            message = String()
            message.data = json.dumps({"status": "FAILED", "message": "Planner returned no path"})
            self.status_pub.publish(message)
            return
        key = self._path_key(path)
        if key == self.latest_path_key:
            return
        self.latest_path_key = key
        if self.goal_request_pending or self.active_goal is not None:
            reference = self.pending_path or self.active_path
            if reference is not None and paths_are_equivalent(
                    self._points(reference), self._points(path)):
                self.get_logger().debug("Keeping active FollowPath goal; update only advances along the same route")
                return
            self.pending_path = path
            self._cancel_active_goal()
            return
        self._send_path(path)

    @staticmethod
    def _path_key(path):
        return tuple((round(p.pose.position.x, 3), round(p.pose.position.y, 3),
                      round(p.pose.orientation.z, 3), round(p.pose.orientation.w, 3))
                     for p in path.poses)

    @staticmethod
    def _points(path):
        return [(p.pose.position.x, p.pose.position.y) for p in path.poses]

    def _cancel_active_goal(self):
        if self.active_goal is not None and not self.cancel_requested:
            self.cancel_requested = True
            self.active_goal.cancel_goal_async()

    def _send_path(self, path):
        issue = self._navigation_input_error()
        if issue:
            self.latest_path_key = None
            self.get_logger().error(f"Refusing FollowPath: {issue}")
            self.publish_status("FAILED", issue)
            return
        if not self.client.wait_for_server(timeout_sec=2.0):
            self.latest_path_key = None
            self.get_logger().error("Nav2 FollowPath action is unavailable")
            self.publish_status("FAILED", "Nav2 FollowPath action is unavailable")
            return
        controller_frame = self.get_parameter("controller_frame").value
        if not path.header.frame_id:
            self.latest_path_key = None
            self.get_logger().error("Planner path has no frame")
            self.publish_status("FAILED", "Planner path has no frame")
            return
        try:
            controller_path = self._transform_path(path, controller_frame)
        except TransformException as exc:
            self.latest_path_key = None
            self.get_logger().error(f"Cannot transform path to {controller_frame}: {exc}")
            self.publish_status("FAILED", f"Cannot transform path to {controller_frame}: {exc}")
            return
        goal = FollowPath.Goal()
        goal.path = controller_path
        goal.controller_id = self.get_parameter("controller_id").value
        goal.goal_checker_id = self.get_parameter("goal_checker_id").value
        self.active_path = path
        self.goal_request_pending = True
        future = self.client.send_goal_async(goal, feedback_callback=self.on_feedback)
        future.add_done_callback(self.on_goal_response)
        self.get_logger().info(
            f"Sent path with {len(path.poses)} poses to Nav2 in {controller_frame} frame")

    def _transform_path(self, path, target_frame):
        if path.header.frame_id == target_frame:
            return path
        transform = self.tf_buffer.lookup_transform(
            target_frame, path.header.frame_id, rclpy.time.Time())
        stamp = transform.header.stamp
        age = self.get_clock().now().nanoseconds * 1e-9 - (stamp.sec + stamp.nanosec * 1e-9)
        if age > 0.75 or age < -1.5:
            raise TransformException(f"{path.header.frame_id} to {target_frame} TF is stale (age {age:.2f}s)")
        result = Path()
        result.header.frame_id = target_frame
        result.header.stamp = self.get_clock().now().to_msg()
        for pose in path.poses:
            transformed = PoseStamped()
            transformed.header = result.header
            transformed.pose = do_transform_pose(pose.pose, transform)
            result.poses.append(transformed)
        return result

    def on_goal_response(self, future):
        self.goal_request_pending = False
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Nav2 rejected FollowPath goal")
            if self.pending_path is None:
                self.publish_status("FAILED", "Nav2 rejected FollowPath goal")
            self.cancel_when_accepted = False
            self._send_pending_path()
            return
        self.active_goal = goal_handle
        if self.pending_path is not None or self.cancel_when_accepted:
            self._cancel_active_goal()
        goal_handle.get_result_async().add_done_callback(self.on_result)

    def on_feedback(self, feedback):
        pass

    def on_result(self, future):
        result = future.result()
        self.get_logger().info(f"FollowPath finished with status {result.status}")
        from action_msgs.msg import GoalStatus
        status = {GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
                  GoalStatus.STATUS_CANCELED: "CANCELED"}.get(result.status, "FAILED")
        if self.safety_fault_reason:
            status = "FAILED"
        if self.pending_path is None:
            self.publish_status(status, self.safety_fault_reason or f"Nav2 status {result.status}")
        self.active_goal = None
        self.cancel_requested = False
        self.cancel_when_accepted = False
        self.safety_fault_reason = None
        self._send_pending_path()

    def _send_pending_path(self):
        if self.pending_path is None or self.active_goal is not None or self.goal_request_pending:
            return
        path, self.pending_path = self.pending_path, None
        self._send_path(path)

    def publish_status(self, status, message_text):
        message = String()
        message.data = json.dumps({"status": status, "message": message_text})
        self.status_pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
