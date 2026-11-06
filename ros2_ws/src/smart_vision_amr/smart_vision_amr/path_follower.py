"""Forward planned paths and bounded recovery to Nav2 action servers."""

from collections import deque
from math import atan2, cos, hypot, isfinite, pi, radians, sin
import time
import hashlib
import uuid

import rclpy
from geometry_msgs.msg import PolygonStamped, PoseStamped, Twist
from nav2_msgs.action import BackUp, FollowPath, Spin
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from rcl_interfaces.msg import ParameterEvent, ParameterType
from rcl_interfaces.srv import GetParameters
import json

from smart_vision_amr.path_geometry import (anticipatory_curve_speed_limit, curvature_speed_limit, path_length, path_progress,
                                           paths_are_equivalent, paths_share_remaining_geometry,
                                           point_at_progress, upcoming_curvature)
from smart_vision_amr.execution_progress import ExecutionProgress, ProgressSample


class PathFollower(Node):
    INITIAL_ALIGNMENT_TOTAL_START_LIMIT = 2
    REPLAN_ALIGNMENT_TOTAL_START_LIMIT = 6
    REPLAN_ALIGNMENT_PATH_START_LIMIT = 2
    RECOVERY_REQUEST_BUFFER_SECONDS = 3.0
    BACKUP_CANCEL_RETRY_SECONDS = 0.5
    BACKUP_CANCEL_SAFETY_ESTOP_SECONDS = 1.0
    BACKUP_CANCEL_GENERAL_ESTOP_SECONDS = 5.0
    FOOTPRINT_ODOM_WAIT_SECONDS = .25
    RECOVERY_CERTIFICATE_RECEIPT_MAX_AGE = 1.0
    # FollowPath cancellation may take longer than one second. Dispatch still
    # requires the original odometry sample, a fresh map TF and sensor inputs,
    # and a chassis pose inside the certified 3 mm / .01 rad start envelope.
    RECOVERY_CERTIFICATE_MAX_AGE = 2.5
    RECOVERY_START_POSITION_TOLERANCE = .003
    RECOVERY_START_YAW_TOLERANCE = .01
    RECOVERY_SWEEP_RESERVE = .01
    SAFETY_CONTROL_SOURCE = "path_follower_recovery_safety"

    def __init__(self):
        super().__init__("path_follower")
        self.declare_parameter("action_name", "/follow_path")
        self.declare_parameter("controller_id", "")
        self.declare_parameter("goal_checker_id", "")
        self.declare_parameter("resume_goal_tolerance", 0.1)
        self.declare_parameter("navigation_inputs_required", True)
        # Initial FollowPath dispatch still requires scan/odom fresher than
        # 0.75 s. Once active, the robot-side localization guard holds velocity
        # at 0.7 s; allow a bounded receipt gap before cancelling the route.
        self.declare_parameter("navigation_sensor_continuity_timeout", 1.0)
        self.declare_parameter("navigation_footprint_topic", "/local_costmap/published_footprint")
        self.declare_parameter("navigation_costmap_topic", "/local_costmap/costmap")
        self.navigation_inputs_required = bool(self.get_parameter("navigation_inputs_required").value)
        self.navigation_sensor_continuity_timeout = float(
            self.get_parameter("navigation_sensor_continuity_timeout").value)
        if not .75 <= self.navigation_sensor_continuity_timeout <= 1.0:
            raise ValueError('navigation_sensor_continuity_timeout must be in [0.75, 1.0] s')
        self.navigation_input_fault = None
        self.navigation_input_cutoff_ns = None
        self.navigation_footprint_reference = None
        self.navigation_odom_history = deque(maxlen=256)
        self.navigation_odom_epoch = 0
        self.navigation_footprint_receipt_pose = None
        self.navigation_footprint_receipt_ros_ns = None
        self.navigation_pending_footprint = None
        self.navigation_footprint_alignment_fault = None
        for name, default in (("follow_request_timeout", 5.0),
                              ("follow_feedback_timeout", 5.0),
                              ("follow_cancel_timeout", 3.0),
                              ("follow_cancel_retry_period", 0.5),
                              ("curve_speed_limit_enabled", True),
                              ("curve_linear_limit_preserves_angular_speed", False),
                              ("curve_speed_limit_topic", "/speed_limit"),
                              ("curve_preview_distance", 0.75),
                              ("curve_base_frame", "base_footprint"),
                              ("curve_pose_timeout", 1.0),
                              ("curve_minimum_chord", 0.15),
                              ("curve_alignment_angle", 0.65),
                              ("curve_goal_heading_distance", 0.30),
                              ("curve_max_linear_velocity", 0.25),
                              ("curve_min_linear_speed_limit", 0.12),
                              ("curve_max_angular_velocity", 0.35),
                              ("curve_max_lateral_acceleration", 0.10),
                              ("curve_braking_deceleration", 0.10),
                              ("curve_braking_margin", 0.15),
                              ("initial_heading_alignment_enabled", True),
                              ("initial_heading_alignment_threshold", 0.55),
                              ("initial_heading_alignment_prune_distance", 0.40),
                              ("initial_heading_alignment_timeout", 12.0)):
            self.declare_parameter(name, default)
        for name, default in (("execution_failure_enabled", True),
                              ("execution_progress_window", 6.0),
                              ("execution_min_progress", 0.05),
                              ("execution_heading_progress", 0.15),
                              ("execution_evidence_timeout", 1.0)):
            self.declare_parameter(name, default)
        self.execution_progress = ExecutionProgress(
            window=float(self.get_parameter('execution_progress_window').value),
            min_progress=float(self.get_parameter('execution_min_progress').value),
            heading_progress=float(self.get_parameter('execution_heading_progress').value))
        self.execution_reference = None
        self.execution_route_id = None
        self.pending_execution_failure = None
        self.execution_discarded_reason = None
        self.execution_replan_cutoff_ns = None
        self.latest_navigation_command = None
        self.latest_navigation_command_at = None
        self.localization_health = None
        self.localization_health_at = None
        self.execution_last_ros_ns = None
        self.execution_clock_advanced_at = None
        self.resume_goal_tolerance = self.get_parameter("resume_goal_tolerance").value
        self.client = ActionClient(self, FollowPath, self.get_parameter("action_name").value)
        self.backup_client = ActionClient(self, BackUp, "/backup")
        self.spin_client = ActionClient(self, Spin, "/spin")
        self.initial_alignment = None
        self.initial_alignment_total_starts = 0
        self._route_alignment_reference = None
        self._pending_replan_alignment = None
        self.replan_alignment_total_starts = 0
        self.replan_alignment_routes = []
        # These are hard bounds on the planner's requested reverse maneuver.
        for name, default in (("recovery_min_distance", 0.10),
                              ("recovery_max_distance", 0.35),
                              ("recovery_min_speed", 0.03),
                              ("recovery_max_speed", 0.12),
                              ("recovery_robot_radius", 0.22),
                              ("recovery_rear_margin", 0.05),
                              ("recovery_sensor_timeout", 1.0),
                              ("recovery_rear_half_angle_deg", 35.0),
                              ("recovery_max_time", 15.0),
                              ("recovery_cancel_timeout", 5.0),
                              ("recovery_settle_timeout", 2.0),
                              ("recovery_scan_frame", "base_scan")):
            self.declare_parameter(name, default)
        self.status_pub = self.create_publisher(String, "/smart_vision_amr/navigation_status", 10)
        self.execution_failure_pub = self.create_publisher(
            String, '/smart_vision_amr/execution_failure', 10)
        self.safety_control_pub = self.create_publisher(
            String, "/smart_vision_amr/manual_control", 1)
        self.safety_control_seq = 0
        self.speed_limit_pub = self.create_publisher(
            SpeedLimit, self.get_parameter("curve_speed_limit_topic").value, 1)
        self.follow_requested_at = None
        self.follow_feedback_at = None
        self.follow_has_feedback = False
        self.last_executed_intent = None
        self.follow_cancel_since = None
        self.follow_last_cancel_at = None
        self.follow_cancel_acknowledged = False
        self.follow_watchdog_fault = None
        self.follow_estop_sent = False
        self.follow_last_estop_at = None
        self.follow_distance_remaining = None
        self.follow_path_progress = None
        self.follow_request_seq = 0
        self.follow_speed_limit = 0.0
        self.controller_base_speed = None
        self.controller_speed_future = None
        self.controller_speed_requested_at = None
        self.controller_speed_event_at = float('-inf')
        self.controller_speed_next_request = float('-inf')
        self.controller_speed_query_index = 0
        self.last_speed_limit_at = None
        self.last_follow_diagnostic_at = None
        self.active_goal = None
        self.goal_request_pending = False
        self.pending_path = None
        self.cancel_requested = False
        self.cancel_when_accepted = False
        self.navigation_suspended = False
        self.require_goal_match = False
        self.requested_goal = None
        self.suspended_paths = deque(maxlen=10)
        self.latest_path_key = None
        self.active_path = None
        self.goal_generation = 0
        self.active_context = None
        self.request_context = None
        self.path_invalidated = False
        self.recovery_cancel_path_cutoff_ns = None
        self.stop_stamp_ns = None
        self.recovery_request = None
        self.pending_recovery_request = None
        self.pending_recovery_cancel = None
        self.recovery_seq = 0
        self.backup_goal = None
        self.backup_action_context = None
        self.backup_goal_request_pending = False
        self.backup_cancel_when_accepted = False
        self.backup_cancel_requested = False
        self.backup_cancel_pending_since = None
        self.backup_last_cancel_at = None
        self.backup_cancel_safety_fault = False
        self.backup_cancel_replan_required = False
        self.backup_cancel_estop_sent = False
        self.backup_last_estop_at = None
        self.latest_scan = None
        self.latest_scan_at = None
        self.latest_odom = None
        self.latest_odom_at = None
        self.latest_footprint = self.latest_costmap = None
        self.latest_footprint_at = self.latest_costmap_at = None
        self.motion_status = None
        self.motion_status_at = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.controller_speed_client = self.create_client(GetParameters, '/controller_server/get_parameters')
        self.create_subscription(ParameterEvent, '/parameter_events', self.on_controller_parameters, 10)
        self.create_subscription(Path, "/smart_vision_amr/path", self.on_path, 10)
        self.create_subscription(PoseStamped, "/smart_vision_amr/goal", self.on_goal, 10)
        self.create_subscription(String, "/smart_vision_amr/recovery_request",
                                 self.on_recovery_request, 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
        # Nav2 publishes the full local grid and an oriented polygon, including
        # when robot_radius (rather than an explicit footprint) is configured.
        volatile = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(PolygonStamped, self.get_parameter("navigation_footprint_topic").value,
                                 self.on_footprint, volatile)
        self.create_subscription(OccupancyGrid, self.get_parameter("navigation_costmap_topic").value,
                                 self.on_costmap, volatile)
        self.create_subscription(Twist, '/smart_vision_amr/navigation_cmd_vel',
                                 self.on_navigation_command, 1)
        durable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/smart_vision_amr/motion_status",
                                 self.on_motion_status, durable)
        self.create_subscription(String, '/smart_vision_amr/localization_health',
                                 self.on_localization_health, durable)
        self.create_service(Trigger, "/smart_vision_amr/cancel_navigation", self.cancel_navigation)
        # Steady time still detects stale sensors when Gazebo /clock is paused.
        from rclpy.clock import Clock, ClockType
        self.create_timer(0.1, self._monitor_recovery,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))

    def cancel_navigation(self, request, response):
        # A planner may keep publishing obstacle-driven replans for its last goal.
        # Only a new explicit goal may resume navigation after a manual stop.
        context = self._goal_context()
        self._reset_execution_progress(clear_pending=True)
        now = self.get_clock().now().to_msg()
        now_ns = now.sec * 1_000_000_000 + now.nanosec
        accepted_stamp = context.get("goal_stamp") or {}
        accepted_ns = (accepted_stamp.get("sec", 0) * 1_000_000_000
                       + accepted_stamp.get("nanosec", 0))
        self.stop_stamp_ns = max(now_ns, accepted_ns, self.stop_stamp_ns or 0)
        self.goal_generation += 1
        self.navigation_suspended = True
        self.navigation_input_fault = None
        self.navigation_input_cutoff_ns = None
        self.require_goal_match = True
        self.requested_goal = None
        self.pending_recovery_request = None
        self.pending_recovery_cancel = None
        self.suspended_paths.clear()
        self.pending_path = None
        self.latest_path_key = None
        self.active_path = None
        self._cancel_initial_alignment()
        self._abort_recovery()
        if self.goal_request_pending:
            self.cancel_when_accepted = True
        self._cancel_active_goal()
        self.publish_status("CANCELED", "Navigation manually stopped", context=context,
                            manual_stop=True)
        # The web joystick must wait for the actual action result, not merely
        # the acceptance of a cancellation request, before handing over motion.
        response.success = (self.active_goal is None and not self.goal_request_pending
                            and self.backup_goal is None and not self.backup_goal_request_pending
                            and self.initial_alignment is None)
        response.message = ("Navigation suspended until a new goal" if response.success
                            else "Waiting for the previous navigation action to finish")
        return response

    def on_goal(self, goal):
        stamp_ns = goal.header.stamp.sec * 1_000_000_000 + goal.header.stamp.nanosec
        if self.stop_stamp_ns is not None and (stamp_ns == 0 or stamp_ns <= self.stop_stamp_ns):
            message = "Ignored delayed/unstamped goal after manual stop; send a newly ROS-stamped goal"
            self.get_logger().warning(message)
            self.publish_status("IGNORED_GOAL", message,
                                context={"goal": None, "goal_stamp": None})
            return
        self.goal_generation += 1
        self._cancel_initial_alignment()
        self.initial_alignment_total_starts = 0
        self._route_alignment_reference = None
        self.replan_alignment_total_starts = 0
        self.replan_alignment_routes = []
        self._reset_execution_progress(clear_pending=True)
        self._abort_recovery()
        self.requested_goal = goal
        self.navigation_suspended = False
        self.navigation_input_fault = None
        self.navigation_input_cutoff_ns = None
        self.require_goal_match = True
        self.path_invalidated = False
        self.recovery_cancel_path_cutoff_ns = None
        canceled = self.pending_recovery_cancel
        pending_recovery = getattr(self, "pending_recovery_request", None)
        if (canceled is not None and pending_recovery is not None
                and canceled["goal_stamp_ns"] == stamp_ns
                and pending_recovery["goal_stamp_ns"] == stamp_ns
                and not self._same_recovery_episode(canceled, pending_recovery["payload"])):
            # A late cancel for an earlier same-goal episode must not discard
            # the independently owned buffered request for the new episode.
            self.pending_recovery_cancel = canceled = None
        if canceled is not None:
            if canceled["goal_stamp_ns"] < stamp_ns:
                self.pending_recovery_cancel = None
            elif canceled["goal_stamp_ns"] == stamp_ns:
                self.pending_recovery_request = None
                self.path_invalidated = True
                self.recovery_cancel_path_cutoff_ns = canceled["path_cutoff_ns"]
                self.publish_status(
                    "RECOVERY_FAILED", "Reverse request canceled before goal arrived",
                    context=self._recovery_context(canceled),
                )
        # A same-coordinate explicit goal is a fresh attempt, even after an abort.
        self.latest_path_key = None
        if self.pending_path is not None and not self._path_matches_goal(self.pending_path):
            self.pending_path = None
        if self.active_goal is not None:
            self._cancel_active_goal()
        elif self.goal_request_pending:
            self.cancel_when_accepted = True
        pending_recovery = getattr(self, "pending_recovery_request", None)
        if pending_recovery is not None:
            requested_stamp = pending_recovery["goal_stamp_ns"]
            if requested_stamp <= stamp_ns:
                self.pending_recovery_request = None
                if requested_stamp == stamp_ns:
                    if time.monotonic() - pending_recovery["received_at"] <= self.RECOVERY_REQUEST_BUFFER_SECONDS:
                        self._accept_recovery_request(pending_recovery["payload"])
                    else:
                        self.publish_status("RECOVERY_FAILED", "Reverse request expired before goal arrived",
                                            context=self._recovery_context(pending_recovery["payload"]))
        # DDS does not order callbacks across /goal and /path. The planner may
        # publish a new goal's path before this subscriber sees the goal itself.
        candidates = list(reversed(self.suspended_paths))
        self.suspended_paths.clear()
        for path in candidates:
            if self._path_matches_goal(path):
                self.on_path(path)
                break

    def _path_matches_goal(self, path):
        if self.requested_goal is None:
            return False
        goal = self.requested_goal
        if (goal.header.frame_id and path.header.frame_id
                and goal.header.frame_id != path.header.frame_id):
            return False
        goal_stamp = goal.header.stamp.sec * 1_000_000_000 + goal.header.stamp.nanosec
        path_stamp = path.header.stamp.sec * 1_000_000_000 + path.header.stamp.nanosec
        if goal_stamp and path_stamp < goal_stamp:
            return False
        # An empty path safely stops navigation; it has no endpoint to compare.
        if not path.poses:
            return True
        end, target = path.poses[-1].pose.position, goal.pose.position
        return hypot(end.x - target.x, end.y - target.y) <= self.resume_goal_tolerance

    def on_path(self, path):
        if self.navigation_suspended:
            self.suspended_paths.append(path)
            return
        if (path.poses and getattr(self, 'navigation_input_fault', None) is not None
                and (self.active_goal is not None or self.goal_request_pending)):
            # Sensor recovery does not prove the old action has stopped. Ask
            # for a newly planned route after its terminal result instead.
            return
        input_cutoff = getattr(self, 'navigation_input_cutoff_ns', None)
        if (path.poses and input_cutoff is not None
                and path.header.stamp.sec*1_000_000_000+path.header.stamp.nanosec <= input_cutoff):
            return
        if getattr(self, 'pending_execution_failure', None) is not None and path.poses:
            # The failed FollowPath still owns motion until its terminal result.
            # No intermediate copy of the old path may bypass that barrier.
            # A discarded diagnosis is only result attribution: fresh routes
            # still queue behind the existing action's cancellation barrier.
            return
        if self.require_goal_match and not self._path_matches_goal(path):
            self.suspended_paths.append(path)
            return
        if path.poses:
            owned_recovery = ((self.recovery_request or {}).get("context")
                              or (getattr(self, "backup_action_context", None)
                                  if self.backup_goal is not None or self.backup_goal_request_pending else None))
            if (owned_recovery is not None
                    and self._valid_episode_id(owned_recovery.get("episode_id"))
                    and self._context_is_current(owned_recovery)):
                # The episode coordinator alone advances reverse -> global.
                # A same-goal delayed route, even one stamped +1 ns beyond
                # /clock, cannot take this request/action's motion ownership.
                # Explicit new goals increment generation and cancel through
                # on_goal; critical input/manual/map cancellation is unchanged.
                return
        cutoff = getattr(self, 'execution_replan_cutoff_ns', None)
        if path.poses and cutoff is not None:
            stamp_ns = path.header.stamp.sec*1_000_000_000+path.header.stamp.nanosec
            if stamp_ns <= cutoff:
                # A delayed route that predates the failure event must not
                # restart while the planner is still consuming that event.
                return
            self.execution_replan_cutoff_ns = None
        if path.poses and self.recovery_cancel_path_cutoff_ns is not None:
            stamp = path.header.stamp
            path_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            if path_ns <= self.recovery_cancel_path_cutoff_ns:
                return
        if path.poses and self.recovery_request is not None:
            # /path and /recovery_request are separate DDS topics. A delayed
            # copy of the route that was just invalidated may arrive after the
            # reverse request. It must never replace the safe recovery action.
            stamp = path.header.stamp
            path_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            if path_ns <= self.recovery_request["requested_ros_ns"]:
                return
            # A fresh safe route supersedes a pending reverse maneuver.
            self._abort_recovery()
        if self.backup_goal_request_pending or self.backup_goal is not None:
            if path.poses:
                self.pending_path = path
                self.latest_path_key = self._path_key(path)
                self.path_invalidated = False
                self._cancel_backup_goal()
            elif self.recovery_request is not None:
                # The planner's *initial* empty Path can arrive after the
                # recovery request because these are different DDS topics.
                # A later, newly stamped empty Path is an explicit timeout
                # or safety invalidation and cancels the active reverse.
                stamp = path.header.stamp
                path_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
                if path_ns > self.recovery_request["requested_ros_ns"] + 500_000_000:
                    self._fail_recovery("Planner canceled reverse recovery")
            return
        if self.recovery_request is not None and not path.poses:
            return
        waiting_alignment = getattr(self, '_pending_replan_alignment', None)
        if waiting_alignment is not None:
            if not path.poses:
                self._pending_replan_alignment = None
                self.pending_path = None
                self.latest_path_key = None
                self.path_invalidated = True
                self.publish_status('WAIT_REPLAN', 'Planner has no safe route; heading setup discarded',
                                    context=waiting_alignment['context'])
            else:
                # This queue owns no action. Keep the original post-terminal
                # stop boundary while replacing only its exact route/context.
                waiting_alignment['path'] = path
                waiting_alignment['context'] = self._goal_context(path)
                self.pending_path = path
                self.latest_path_key = self._path_key(path)
                self._try_pending_replan_alignment()
            return
        if self.initial_alignment is not None:
            if path.poses:
                key = self._path_key(path)
                if key != self.latest_path_key:
                    self.pending_path = path
                    self.latest_path_key = key
                    equivalent = (paths_share_remaining_geometry(
                        self._points(self.initial_alignment['path']), self._points(path))
                        if self.initial_alignment.get('kind') == 'material_replan' else
                        paths_are_equivalent(self._points(self.initial_alignment['path']), self._points(path)))
                    if not equivalent:
                        self._cancel_initial_alignment()
            else:
                self.pending_path = None
                self.latest_path_key = None
                self.path_invalidated = True
                self._cancel_initial_alignment()
                self.publish_status('WAIT_REPLAN', 'Planner has no safe route; stopping heading alignment')
            return
        if not path.poses:
            if not self.path_invalidated:
                self.get_logger().warning("Planner has no safe path; stopping while retaining the goal")
                self.publish_status("WAIT_REPLAN", "Planner has no safe route; waiting for replan")
            self.path_invalidated = True
            self.pending_path = None
            self.latest_path_key = None
            self.active_path = None
            if self.active_goal is not None:
                self._cancel_active_goal()
            elif self.goal_request_pending:
                self.cancel_when_accepted = True
            return
        self.path_invalidated = False
        key = self._path_key(path)
        if key == self.latest_path_key:
            if self.pending_path is not None:
                self.pending_path = path
            elif (self.goal_request_pending or self.active_goal is not None):
                context = self.active_context or self.request_context
                if (self.active_path is not None
                        and paths_share_remaining_geometry(self._points(self.active_path), self._points(path))):
                    if self.cancel_requested or self.cancel_when_accepted:
                        self.pending_path = path
                    else:
                        self._ack_equivalent_path_alias(path, context)
            return
        self.latest_path_key = key
        if self.goal_request_pending or self.active_goal is not None:
            reference = self.pending_path or self.active_path
            reference_context = self.active_context or self.request_context
            same_goal = reference_context is None or self._context_is_current(reference_context)
            if same_goal and reference is not None and paths_share_remaining_geometry(
                    self._points(reference), self._points(path)):
                self.get_logger().debug("Keeping active FollowPath goal; update only advances along the same route")
                if self.pending_path is not None or self.cancel_requested or self.cancel_when_accepted:
                    # A cancellation already in flight still owns its stop
                    # barrier. Update the queued route, never its old action.
                    self.pending_path = path
                else:
                    self._ack_equivalent_path_alias(path, reference_context)
                return
            self.pending_path = path
            self._cancel_active_goal()
            return
        self._send_path(path)

    def _ack_equivalent_path_alias(self, path, context):
        """Bind an accepted equivalent route to the same running action.

        Async response/feedback/result closures share this owned dict. A new
        route stamp changes only its accepted alias, never goal generation,
        action geometry, cancellation ownership or recovery identity.
        """
        if context is None or not self._context_is_current(context):
            return
        stamp = {"sec": path.header.stamp.sec, "nanosec": path.header.stamp.nanosec}
        previous = context.get("path_stamp") or {}
        incoming_ns = stamp["sec"] * 1_000_000_000 + stamp["nanosec"]
        previous_ns = previous.get("sec", 0) * 1_000_000_000 + previous.get("nanosec", 0)
        if incoming_ns <= previous_ns:
            return
        context["path_stamp"] = stamp
        self.publish_status("MOVING", "Accepted equivalent remaining route without restarting FollowPath",
                            context=context, route_alias_accepted=True)

    @staticmethod
    def _path_key(path):
        return tuple((round(p.pose.position.x, 3), round(p.pose.position.y, 3),
                      round(p.pose.orientation.z, 3), round(p.pose.orientation.w, 3))
                     for p in path.poses)

    @staticmethod
    def _points(path):
        return [(p.pose.position.x, p.pose.position.y) for p in path.poses]

    def _cancel_active_goal(self):
        if (self.active_goal is not None or self.goal_request_pending) and self.follow_cancel_since is None:
            self.follow_cancel_since = time.monotonic()
        if self.goal_request_pending:
            self.cancel_when_accepted = True
        if self.active_goal is not None and not self.cancel_requested:
            self.cancel_requested = True
            self._send_follow_cancel()

    def _send_follow_cancel(self):
        handle = self.active_goal
        if handle is None:
            return
        self.follow_last_cancel_at = time.monotonic()
        try:
            handle.cancel_goal_async().add_done_callback(
                lambda future, accepted=handle: self._on_follow_cancel_response(future, accepted))
        except Exception as exc:
            self.get_logger().error(f"FollowPath cancel request failed: {exc}")

    def _on_follow_cancel_response(self, future, handle):
        if handle is not self.active_goal:
            return
        try:
            response = future.result()
            # A cancellation acknowledgment is not a terminal action result.
            self.follow_cancel_acknowledged = bool(response.goals_canceling)
        except Exception as exc:
            self.get_logger().warning(f"FollowPath cancellation is not acknowledged: {exc}")

    def _send_path(self, path):
        if (self.navigation_suspended or self.recovery_request is not None
                or self.backup_goal_request_pending or self.backup_goal is not None
                or self.initial_alignment is not None):
            return
        if self.active_goal is not None or self.goal_request_pending:
            self.pending_path = path
            self._cancel_active_goal()
            return
        if getattr(self, 'navigation_inputs_required', False):
            issue = self._navigation_input_error()
            if issue is not None:
                self.latest_path_key = None
                if self.requested_goal is None:
                    # DDS does not order /goal and /path callbacks.
                    self.suspended_paths.append(path)
                    self.require_goal_match = True
                else:
                    self._hold_navigation_inputs(issue, self._goal_context(path), force_status=True)
                return
            self.navigation_input_fault = None
            self.navigation_input_cutoff_ns = None
        reference = getattr(self, '_route_alignment_reference', None)
        initial_setup = reference is None or reference['generation'] != self.goal_generation
        if self.get_parameter('initial_heading_alignment_enabled').value:
            error = self._initial_heading_error(path)
            if error is None:
                if getattr(self, 'navigation_inputs_required', False):
                    self._hold_navigation_inputs('Fresh map pose unavailable for route heading',
                                                 self._goal_context(path), force_status=True)
                    return
            elif abs(error) > float(self.get_parameter('initial_heading_alignment_threshold').value):
                if not initial_setup:
                    material = self._material_replan_prefix(path)
                    if material is None:
                        self._hold_navigation_inputs('Fresh map pose unavailable for changed-route prefix',
                                                     self._goal_context(path), force_status=True)
                        return
                    if material:
                        self._pending_replan_alignment = {
                            'path': path, 'context': self._goal_context(path),
                            'requested_ns': self.get_clock().now().nanoseconds,
                            'requested_at': time.monotonic()}
                        self.pending_path = path
                        self.publish_status('WAIT_CANCEL',
                            'Waiting for measured stop before changed-route heading alignment',
                            context=self._goal_context(path), heading_alignment_phase='WAIT_STOP',
                            initial_heading_error_rad=error)
                        self._try_pending_replan_alignment()
                        return
                    self._dispatch_follow_path(path)
                    return
                if self.initial_alignment_total_starts >= self.INITIAL_ALIGNMENT_TOTAL_START_LIMIT:
                    # Physical Spin starts are never refunded by route churn.
                    # Only a new explicit goal renews this bounded budget.
                    self.navigation_suspended = True
                    self.path_invalidated = True
                    self.pending_path = None
                    self.latest_path_key = None
                    self.publish_status('CONTROLLER_HOLD',
                                        'Initial heading start budget exhausted; explicit mission resume required',
                                        context=self._goal_context(path),
                                        initial_alignment_starts=self.initial_alignment_total_starts,
                                        initial_alignment_start_limit=self.INITIAL_ALIGNMENT_TOTAL_START_LIMIT)
                    return
                self._start_initial_alignment(path, error)
                return
        self._dispatch_follow_path(path)

    def _material_replan_prefix(self, path):
        """A changed near-term route, not an alias, suffix advance or distant edit."""
        reference = getattr(self, '_route_alignment_reference', None)
        if reference is None or reference['generation'] != self.goal_generation:
            return False
        points = self._points(path)
        if paths_share_remaining_geometry(reference['points'], points):
            return False
        try:
            pose = self._heading_alignment_pose(path)
            if pose is None:
                return None
            def prefix(route):
                progress = path_progress(route, pose[:2])
                length = min(.40, max(0., path_length(route)-progress))
                count = max(1, int(length/.02)+1)
                return [point_at_progress(route, progress+length*i/count)
                        for i in range(count+1)]
            return not paths_are_equivalent(prefix(reference['points']), prefix(points),
                                            tolerance=.03, start_tolerance=.03)
        except (TransformException, ValueError, TypeError):
            return None

    def _replan_alignment_route(self, path):
        routes = getattr(self, 'replan_alignment_routes', None)
        if routes is None:
            routes = self.replan_alignment_routes = []
        points = tuple(self._points(path))
        for record in routes:
            if paths_share_remaining_geometry(record['points'], points):
                return record
        record = {'points': points, 'starts': 0}
        routes.append(record)
        return record

    def _replan_alignment_stopped(self, not_before_ns, not_before_receipt):
        """Dense source-stamped odometry after FollowPath terminal, never a zero command."""
        if not self._source_is_fresh(self.latest_odom, self.latest_odom_at):
            return False
        history = tuple(getattr(self, 'navigation_odom_history', ()))
        if len(history) < 4:
            return False
        latest = history[-1]
        source_age = (self.get_clock().now().nanoseconds-latest[0])*1e-9
        receipt_age = time.monotonic()-self.latest_odom_at
        if (not 0 <= receipt_age < .20 or not -.25 <= source_age < .20
                or latest[0]-not_before_ns < 200_000_000):
            return False
        receipts = dict(getattr(self, 'navigation_odom_receipts', ()))
        window = [v for v in history if 0 <= latest[0]-v[0] <= 500_000_000
                  and v[0] >= not_before_ns
                  and receipts.get(v[0], float('-inf')) >= not_before_receipt]
        if (len(window) < 4 or latest[0]-window[0][0] < 200_000_000
                or any(v[1:3] != latest[1:3] for v in window)):
            return False
        received = [receipts.get(v[0]) for v in window]
        if (any(at is None or at < not_before_receipt for at in received)
                or received[-1]-received[0] < .20-1e-9
                or not 0 <= time.monotonic()-received[-1] < .20
                or any(not 0 < b-a <= .18+1e-9 for a, b in zip(received, received[1:]))):
            return False
        if any(not 0 < b[0]-a[0] <= 180_000_000 for a, b in zip(window, window[1:])):
            return False
        twist = self.latest_odom.twist.twist
        if (not all(isfinite(v) for v in (twist.linear.x, twist.angular.z))
                or abs(twist.linear.x) >= .025 or abs(twist.angular.z) >= .06):
            return False
        return all(hypot(v[3]-latest[3], v[4]-latest[4]) <= .002
                   and abs(atan2(sin(v[5]-latest[5]), cos(v[5]-latest[5]))) <= .005
                   for v in window)

    def _try_pending_replan_alignment(self):
        state = getattr(self, '_pending_replan_alignment', None)
        if state is None:
            return
        if (self.navigation_suspended or not self._context_is_current(state['context'])
                or self.recovery_request is not None):
            self._pending_replan_alignment = None
            return
        if (self.active_goal is not None or self.goal_request_pending
                or self.backup_goal is not None or self.backup_goal_request_pending
                or self.initial_alignment is not None):
            return
        if not self._alignment_motion_ready():
            return
        issue = self._navigation_input_error() if getattr(self, 'navigation_inputs_required', False) else None
        if issue is not None:
            self._hold_navigation_inputs(issue, state['context'], force_status=True)
            return
        if not self._replan_alignment_stopped(state['requested_ns'], state['requested_at']):
            return
        path = state['path']
        error = self._initial_heading_error(path)
        if error is None:
            self._hold_navigation_inputs('Fresh map pose unavailable for changed-route heading',
                                         state['context'], force_status=True)
            return
        material = self._material_replan_prefix(path)
        if material is None:
            self._hold_navigation_inputs('Fresh map pose unavailable for changed-route prefix',
                                         state['context'], force_status=True)
            return
        self._pending_replan_alignment = None
        self.pending_path = None
        if (not material
                or abs(error) <= float(self.get_parameter('initial_heading_alignment_threshold').value)):
            self._dispatch_follow_path(path)
            return
        route = self._replan_alignment_route(path)
        total = getattr(self, 'replan_alignment_total_starts', 0)
        if (route['starts'] >= self.REPLAN_ALIGNMENT_PATH_START_LIMIT
                or total >= self.REPLAN_ALIGNMENT_TOTAL_START_LIMIT):
            self.navigation_suspended = True
            self.path_invalidated = True
            self.latest_path_key = None
            self.publish_status('CONTROLLER_HOLD',
                'Changed-route heading budget exhausted; explicit mission resume required',
                context=state['context'], heading_alignment_phase='HOLD',
                replan_alignment_starts=total,
                replan_alignment_start_limit=self.REPLAN_ALIGNMENT_TOTAL_START_LIMIT,
                route_alignment_starts=route['starts'],
                route_alignment_start_limit=self.REPLAN_ALIGNMENT_PATH_START_LIMIT)
            return
        self._start_initial_alignment(path, error, kind='material_replan')

    def _heading_alignment_pose(self, path):
        transform = self.tf_buffer.lookup_transform(
            path.header.frame_id or 'map', self.get_parameter('curve_base_frame').value,
            rclpy.time.Time())
        stamp = transform.header.stamp
        age = (self.get_clock().now().nanoseconds
               - stamp.sec*1_000_000_000-stamp.nanosec)/1e9
        if not -.25 <= age < .75:
            return None
        p, q = transform.transform.translation, transform.transform.rotation
        if (not all(isfinite(value) for value in (p.x, p.y, q.x, q.y, q.z, q.w))
                or not .8 <= q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w <= 1.2):
            return None
        return p.x, p.y, atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

    def _initial_heading_error(self, path):
        points = self._points(path)
        if len(points) < 2 or path_length(points) < .2:
            return 0.
        try:
            pose = self._heading_alignment_pose(path)
            if pose is None:
                return None
            x, y, yaw = pose
            progress = path_progress(points, (x, y))
            projected = point_at_progress(points, progress)
            cross_track = hypot(x-projected[0], y-projected[1])
            # The physical DWB config accepts up to 0.40 m for path pruning.
            # A robot slightly off the path can still use its collision-checked
            # FollowPath action, but an in-place Spin about that offset may
            # face the wrong first segment and strand a valid replan.
            prune_distance = float(self.get_parameter(
                'initial_heading_alignment_prune_distance').value)
            if not isfinite(prune_distance) or prune_distance < .25:
                return None
            if cross_track > prune_distance:
                return None
            if cross_track > .25:
                return 0.
            target = point_at_progress(points, min(path_length(points), progress+.15))
            if hypot(target[0]-x, target[1]-y) < .05:
                return 0.
            heading = atan2(target[1]-y, target[0]-x)
            return atan2(sin(heading-yaw), cos(heading-yaw))
        except (TransformException, ValueError):
            return None

    def _start_initial_alignment(self, path, error, *, kind='initial'):
        client = getattr(self, 'spin_client', None)
        if client is None or not client.wait_for_server(timeout_sec=.2):
            self.latest_path_key = None
            self.publish_status('WAIT_REPLAN', 'Nav2 collision-checked Spin action unavailable',
                                context=self._goal_context(path))
            return
        goal = Spin.Goal()
        goal.target_yaw = float(error)
        # Under localization warnings the mux may cap yaw to .25 rad/s.
        # Budget the requested angle at that bounded rate, including settling.
        allowance = min(20., max(float(self.get_parameter(
            'initial_heading_alignment_timeout').value), abs(error)/.25+4.))
        goal.time_allowance.sec = max(1, int(allowance+.999))
        state = {'path': path, 'context': self._goal_context(path), 'handle': None,
                 'requested_at': time.monotonic(), 'cancel_at': None, 'last_cancel_at': None,
                 'cancel': False,
                 'time_allowance': goal.time_allowance.sec, 'watchdog_fault': None,
                 'kind': kind}
        self.initial_alignment = state
        self.pending_path = path
        if kind == 'initial':
            self.initial_alignment_total_starts += 1
        else:
            self._replan_alignment_route(path)['starts'] += 1
            self.replan_alignment_total_starts = getattr(self, 'replan_alignment_total_starts', 0)+1
        try:
            future = client.send_goal_async(goal)
            future.add_done_callback(lambda result, sent=state:
                                     self._on_initial_alignment_response(result, sent))
        except Exception as exc:
            # An action transport error does not prove the rotation was rejected.
            self._hold_unconfirmed_initial_alignment(state, f'Spin acceptance unconfirmed: {exc}')
            return
        self.publish_status('MOVING', 'Rotating in place to face the initial A* segment',
                            context=state['context'], initial_heading_error_rad=error,
                            heading_alignment_kind=kind, heading_alignment_phase='SPIN')

    def _on_initial_alignment_response(self, future, state):
        if state is not self.initial_alignment:
            return
        try:
            handle = future.result()
        except Exception as exc:
            self._hold_unconfirmed_initial_alignment(state, f'Spin acceptance unconfirmed: {exc}')
            return
        if not handle.accepted:
            self.initial_alignment = None
            had_safety_stop = self.follow_estop_sent
            self._clear_confirmed_initial_alignment_watch(state)
            if self.navigation_suspended or had_safety_stop:
                return
            if state['cancel'] and self.pending_path is not None:
                if self._alignment_motion_ready():
                    self._send_pending_path()
                    return
            self.pending_path = None
            self.latest_path_key = None
            self.publish_status('WAIT_REPLAN', 'Nav2 rejected collision-checked heading turn',
                                context=state['context'])
            return
        state['handle'] = handle
        if state['cancel'] or self.navigation_suspended or not self._context_is_current(state['context']):
            self._cancel_initial_alignment()
        try:
            handle.get_result_async().add_done_callback(
                lambda result, sent=state: self._on_initial_alignment_result(result, sent))
        except Exception as exc:
            self._hold_unconfirmed_initial_alignment(state, f'Spin result unconfirmed: {exc}')

    def _on_initial_alignment_result(self, future, state):
        if state is not self.initial_alignment:
            return
        from action_msgs.msg import GoalStatus
        try:
            result = future.result()
            if result.status not in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED,
                                     GoalStatus.STATUS_ABORTED):
                raise ValueError(f'nonterminal Spin status {result.status}')
        except Exception as exc:
            self._hold_unconfirmed_initial_alignment(state, f'Spin result unconfirmed: {exc}')
            return
        self.initial_alignment = None
        had_safety_stop = self.follow_estop_sent
        self._clear_confirmed_initial_alignment_watch(state)
        if self.navigation_suspended or had_safety_stop:
            return
        if state['cancel'] or not self._context_is_current(state['context']):
            if self._alignment_motion_ready():
                self._send_pending_path()
            else:
                self.pending_path = None
                self.latest_path_key = None
                self.publish_status('WAIT_REPLAN', 'Heading turn stopped; waiting for auto control',
                                    context=state['context'])
            self._try_start_recovery()
            return
        path, self.pending_path = self.pending_path, None
        if result.status != GoalStatus.STATUS_SUCCEEDED or path is None:
            self.latest_path_key = None
            self.publish_status('WAIT_REPLAN', 'Collision-checked heading turn did not complete',
                                context=state['context'])
            return
        issue = self._navigation_input_error() if getattr(self, 'navigation_inputs_required', False) else None
        if issue is not None:
            self._hold_navigation_inputs(issue, state['context'], force_status=True)
            return
        if state.get('kind') == 'material_replan' and not self._alignment_motion_ready():
            self.latest_path_key = None
            self.publish_status('WAIT_REPLAN', 'Heading turn stopped; waiting for auto control',
                                context=state['context'], route_failure=False, input_hold=True)
            return
        error = self._initial_heading_error(path)
        if error is None or abs(error) > float(
                self.get_parameter('initial_heading_alignment_threshold').value):
            self.latest_path_key = None
            self.publish_status('WAIT_REPLAN', 'Heading turn ended without alignment; replan from current pose',
                                context=state['context'])
            return
        self._dispatch_follow_path(path)

    def _hold_unconfirmed_initial_alignment(self, state, reason):
        # Keep the safety watchdog attributable to this exact Spin. A terminal
        # callback may release its resend duty, never another action's fault.
        previous = state.get('watchdog_fault')
        state['watchdog_fault'] = reason
        if self.follow_watchdog_fault is None or self.follow_watchdog_fault == previous:
            self.follow_watchdog_fault = reason
        self._send_follow_safety_estop()

    def _clear_confirmed_initial_alignment_watch(self, state):
        owned_fault = state.get('watchdog_fault')
        if (owned_fault is not None and self.follow_watchdog_fault == owned_fault
                and self.active_goal is None and not self.goal_request_pending):
            # Terminal action/rejection proves the Spin stopped. Leave the
            # navigation suspension and the robot's E-stop latch untouched;
            # the operator must still reset and dispatch a fresh goal.
            self._clear_follow_watch()

    def _alignment_motion_ready(self):
        motion = self.motion_status or {}
        received = self.motion_status_at
        return (received is not None and 0 <= time.monotonic()-received < 1.
                and motion.get('mode') == 'auto'
                and motion.get('available') is True
                and motion.get('autonomous_allowed') is True
                and motion.get('estop_latched') is not True
                and motion.get('reset_pending') is not True)

    def _cancel_initial_alignment(self):
        waiting = getattr(self, '_pending_replan_alignment', None)
        if waiting is not None and self.pending_path is waiting['path']:
            self.pending_path = None
            self.latest_path_key = None
        self._pending_replan_alignment = None
        state = self.initial_alignment
        if state is None:
            return
        if not state['cancel']:
            state['cancel'] = True
            state['cancel_at'] = time.monotonic()
        handle = state['handle']
        if handle is not None and (state['last_cancel_at'] is None
                                   or time.monotonic()-state['last_cancel_at'] >= .5):
            state['last_cancel_at'] = time.monotonic()
            try:
                handle.cancel_goal_async()
            except Exception as exc:
                self.get_logger().error(f'Spin cancel request failed: {exc}')

    def _monitor_initial_alignment(self):
        self._try_pending_replan_alignment()
        state = self.initial_alignment
        if state is None:
            return
        now = time.monotonic()
        if (state['handle'] is None and not state['cancel']
                and now-state['requested_at'] > max(.5, float(
                    self.get_parameter('follow_request_timeout').value))):
            self._hold_unconfirmed_initial_alignment(state, 'Spin goal acceptance timed out')
            return
        if not state['cancel'] and now-state['requested_at'] > state['time_allowance']+3.:
            self._cancel_initial_alignment()
        if state['cancel']:
            self._cancel_initial_alignment()
            if now-state['cancel_at'] > max(.5, float(
                    self.get_parameter('follow_cancel_timeout').value)) and not self.follow_estop_sent:
                self._hold_unconfirmed_initial_alignment(state, 'Spin cancellation/result timed out')

    def _dispatch_follow_path(self, path):
        if not self.client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 FollowPath action is unavailable")
            self.latest_path_key = None
            self.publish_status("FAILED", "Nav2 FollowPath action is unavailable")
            return
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = self.get_parameter("controller_id").value
        goal.goal_checker_id = self.get_parameter("goal_checker_id").value
        self.active_path = path
        self._select_execution_reference(path)
        context = self._goal_context(path)
        self.request_context = context
        self.goal_request_pending = True
        self.follow_requested_at = time.monotonic()
        self.follow_feedback_at = None
        self.follow_has_feedback = False
        self.last_executed_intent = None
        self.follow_distance_remaining = path_length(self._points(path))
        self.follow_request_seq += 1
        sequence = self.follow_request_seq
        try:
            future = self.client.send_goal_async(
                goal, feedback_callback=lambda feedback, sent=sequence: self.on_feedback(feedback, sent))
            future.add_done_callback(lambda result, sent=context, seq=sequence:
                                     self.on_goal_response(result, sent, seq))
        except Exception as exc:
            # Send errors do not prove the server never accepted the request.
            self._hold_unconfirmed_follow(f"FollowPath acceptance is unconfirmed: {exc}")
            return
        self._route_alignment_reference = {
            'generation': self.goal_generation, 'points': tuple(self._points(path)),
            'path_stamp_ns': path.header.stamp.sec*1_000_000_000+path.header.stamp.nanosec}
        # Initial setup and materially different near-term detours have
        # independent bounded budgets; equivalent route progress never spins.
        self._publish_curve_limit()
        self.get_logger().info(f"Sent path with {len(path.poses)} poses to Nav2")

    def on_goal_response(self, future, context=None, sequence=None):
        if sequence is not None and sequence != self.follow_request_seq:
            return
        context = context or self.request_context or self._goal_context(self.active_path)
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._hold_unconfirmed_follow(f"FollowPath acceptance is unconfirmed: {exc}")
            return
        self.goal_request_pending = False
        self.request_context = None
        if not goal_handle.accepted:
            self.get_logger().error("Nav2 rejected FollowPath goal")
            self.latest_path_key = None
            if (self.pending_path is None and self._context_is_current(context)
                    and not self.navigation_suspended):
                if getattr(self, 'navigation_input_fault', None) is not None:
                    self.navigation_input_cutoff_ns = self.get_clock().now().nanoseconds
                    self.publish_status('WAIT_REPLAN', 'Navigation request ended during sensor hold',
                                        context=context, route_failure=False, input_hold=True)
                else:
                    self.publish_status("FAILED", "Nav2 rejected FollowPath goal", context=context)
            self.cancel_when_accepted = False
            self._clear_follow_watch()
            self._publish_speed_limit(0.0)
            self._try_start_recovery()
            self._send_pending_path()
            return
        self.active_goal = goal_handle
        self.active_context = context
        self.follow_feedback_at = time.monotonic()
        if (self.navigation_suspended or self.pending_path is not None
                or self.cancel_when_accepted or self.recovery_request is not None
                or not self._context_is_current(context)):
            self._cancel_active_goal()
        else:
            self.publish_status("MOVING", "Following A* route", context=context)
        try:
            goal_handle.get_result_async().add_done_callback(
                lambda result, sent=context, accepted=goal_handle: self.on_result(result, sent, accepted))
        except Exception as exc:
            self._hold_unconfirmed_follow(f"FollowPath result is unavailable: {exc}")

    def on_feedback(self, feedback, sequence=None):
        if sequence is not None and sequence != self.follow_request_seq:
            return
        if self.active_goal is None and not self.goal_request_pending:
            return
        try:
            remaining = float(feedback.feedback.distance_to_goal)
            speed = float(feedback.feedback.speed)
        except (AttributeError, TypeError, ValueError):
            return
        if not isfinite(remaining) or remaining < 0 or not isfinite(speed):
            return
        self.follow_feedback_at = time.monotonic()
        self.follow_has_feedback = True
        self.follow_distance_remaining = remaining
        self._publish_curve_limit()
        if (self.last_follow_diagnostic_at is None or
                time.monotonic()-self.last_follow_diagnostic_at >= 1.0):
            self.last_follow_diagnostic_at = time.monotonic()
            if not self.cancel_requested and not self.navigation_suspended:
                self.publish_status("MOVING", "Following A* route", context=self.active_context,
                                    distance_remaining=(max(0., path_length(self._points(self.active_path))-
                                                            self.follow_path_progress)
                                                        if self.active_path is not None and
                                                        self.follow_path_progress is not None else None),
                                    controller_distance_remaining=remaining, measured_speed=speed,
                                    curve_speed_limit=self.follow_speed_limit)

    def on_result(self, future, context=None, handle=None):
        if handle is not None and handle is not self.active_goal:
            return
        context = context or self.active_context or self._goal_context(self.active_path)
        from action_msgs.msg import GoalStatus
        try:
            result = future.result()
            if result.status not in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED,
                                      GoalStatus.STATUS_ABORTED):
                self._hold_unconfirmed_follow(f"FollowPath returned nonterminal status {result.status}")
                return
            status = {GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
                      GoalStatus.STATUS_CANCELED: "CANCELED"}.get(result.status, "FAILED")
            detail = f"Nav2 status {result.status}"
        except Exception as exc:
            self._hold_unconfirmed_follow(f"FollowPath result is unconfirmed: {exc}")
            return
        self.get_logger().info(f"FollowPath finished: {detail}")
        if (status == 'SUCCEEDED' and getattr(self, 'navigation_input_fault', None) is not None
                and self._context_is_current(context) and not self.navigation_suspended):
            # A confirmed terminal success wins a transient input/cancel race,
            # just as it wins an unconfirmed execution-failure diagnosis below.
            self.navigation_input_fault = self.navigation_input_cutoff_ns = None
            self.path_invalidated = False
        failure_detection_enabled = bool(self.get_parameter('execution_failure_enabled').value)
        if not failure_detection_enabled:
            self._discard_execution_diagnosis('Execution failure detection disabled')
        self._discard_scan_guard_execution()
        failure_event = None
        pending_failure = getattr(self, 'pending_execution_failure', None)
        discarded_reason = getattr(self, 'execution_discarded_reason', None)
        self.execution_discarded_reason = None
        if pending_failure is not None and status == 'SUCCEEDED':
            # The goal can finish just before a cancellation reaches Nav2.
            # A confirmed success disproves the pending stuck diagnosis; it
            # must not poison an otherwise successfully traversed approach.
            self.pending_execution_failure = None
            pending_failure = None
            if self._context_is_current(context) and not self.navigation_suspended:
                self.path_invalidated = False
        if discarded_reason is not None and status == 'SUCCEEDED':
            # Success also wins a pending diagnosis invalidated by a transient
            # condition; explicit operator stop still suspends the goal below.
            if self._context_is_current(context) and not self.navigation_suspended:
                self.path_invalidated = False
        # Only confirmed server-side ABORTED results with fresh spatial evidence
        # qualify. Cancellations, watchdogs and unavailable sensors never do.
        if (failure_detection_enabled and pending_failure is None and status == 'FAILED' and not self.cancel_requested
                and not self.follow_watchdog_fault and not self.navigation_suspended
                and not self.path_invalidated and self.recovery_request is None
                and getattr(self, 'follow_has_feedback', False)
                and self._context_is_current(context)):
            observation = self._execution_observation(require_intent=True)
            if observation is None:
                # Nav2 commonly publishes zero immediately before ABORTED.
                # A bounded command actually accepted during this same action
                # remains evidence, provided every spatial prerequisite is
                # still healthy now. Never infer intent from an arbitrary abort.
                observation = self._execution_observation(require_intent=False)
                recent = getattr(self, 'last_executed_intent', None)
                horizon = max(6., self.execution_progress.window*2.) if getattr(self, 'execution_progress', None) else 6.
                if (observation is None or recent is None
                        or recent['request_seq'] != self.follow_request_seq
                        or not self._execution_command_is_fresh(time.monotonic())
                        or not 0 <= time.monotonic()-recent['at'] <= horizon):
                    observation = None
            if observation is not None:
                pending_failure = self._execution_payload(
                    'controller_aborted', observation, {'terminal_status': int(result.status)})
        if pending_failure is not None:
            # Recheck after cancellation: a sensor/TF fault that appeared while
            # stopping invalidates the spatial diagnosis; retain no route memory.
            if (self._context_is_current(context) and not self.navigation_suspended
                    and not self.follow_watchdog_fault
                    and self._execution_observation(require_intent=False) is not None):
                failure_event = pending_failure
                failure_event['terminal_confirmed'] = True
                failure_event['created_at_ns'] = self.get_clock().now().nanoseconds
                self.execution_replan_cutoff_ns = failure_event['created_at_ns']
                packet = String()
                packet.data = json.dumps(failure_event, allow_nan=False)
                self.execution_failure_pub.publish(packet)
            self.pending_execution_failure = None
            self.pending_path = None
            self.latest_path_key = None
            self.path_invalidated = True
        if status != "SUCCEEDED":
            # Failure must not permanently deduplicate the same safe replacement route.
            self.latest_path_key = None
        if (self.pending_path is None and self.recovery_request is None
                and not self.navigation_suspended
                and not self.path_invalidated and self._context_is_current(context)
                and (discarded_reason is None or status == 'SUCCEEDED')):
            self.publish_status(status, detail, context=context)
        self.active_goal = None
        self.active_context = None
        self.active_path = None
        self.cancel_requested = False
        self.cancel_when_accepted = False
        watchdog_fault = self.follow_watchdog_fault
        self._clear_follow_watch()
        self._publish_speed_limit(0.0)
        if failure_event is not None:
            self.publish_status('FAILED', 'Stopped; replan while avoiding failed approach',
                                context=context, route_failure=True,
                                execution_failure_id=failure_event['event_id'],
                                execution_failure_reason=failure_event['reason'])
        elif pending_failure is not None and not self.navigation_suspended:
            self.publish_status('WAIT_REPLAN', 'Execution diagnosis discarded: fresh sensor/TF evidence unavailable',
                                context=context, route_failure=False)
        elif discarded_reason is not None and status != 'SUCCEEDED' and not self.navigation_suspended:
            self.publish_status('WAIT_REPLAN', 'Execution diagnosis discarded: '+discarded_reason,
                                context=context, route_failure=False)
        if getattr(self, 'navigation_input_fault', None) is not None and not self.navigation_suspended:
            self.navigation_input_cutoff_ns = self.get_clock().now().nanoseconds
            self.publish_status('WAIT_REPLAN', 'Navigation stopped; waiting for fresh navigation inputs',
                                context=context, route_failure=False, input_hold=True)
        if watchdog_fault and not self.navigation_suspended:
            # Routes computed before stop confirmation start at an uncertain
            # pose. Ask the retained-goal planner for a fresh start instead.
            self.pending_path = None
            self.publish_status("FAILED", f"FollowPath stopped after watchdog: {watchdog_fault}", context=context)
        self._try_start_recovery()
        self._send_pending_path()

    def _send_pending_path(self):
        if (self.navigation_suspended or self.pending_path is None
                or self.active_goal is not None or self.goal_request_pending
                or self.initial_alignment is not None
                or self.recovery_request is not None
                or self.backup_goal_request_pending or self.backup_goal is not None):
            return
        path, self.pending_path = self.pending_path, None
        self._send_path(path)

    def _clear_follow_watch(self):
        self.follow_requested_at = self.follow_feedback_at = None
        self.follow_has_feedback = False
        self.last_executed_intent = None
        self.follow_cancel_since = self.follow_last_cancel_at = None
        self.follow_cancel_acknowledged = False
        self.follow_watchdog_fault = None
        self.follow_estop_sent = False
        self.follow_last_estop_at = None
        self.follow_distance_remaining = None
        self.follow_path_progress = None

    def _hold_unconfirmed_follow(self, reason):
        """Retain the handle/request barrier until Nav2 proves the action ended."""
        self.follow_watchdog_fault = reason
        self._reset_execution_progress(clear_pending=True)
        self.latest_path_key = None
        self.pending_path = None
        self.path_invalidated = True
        if self.follow_cancel_since is None:
            self.follow_cancel_since = time.monotonic()
        self._cancel_active_goal()
        self._send_follow_safety_estop()

    def _send_follow_safety_estop(self):
        self.safety_control_seq = max(self.safety_control_seq + 1, time.monotonic_ns())
        packet = String()
        packet.data = json.dumps({"source_id": self.SAFETY_CONTROL_SOURCE,
            "seq": self.safety_control_seq, "sent_at_ns": self.get_clock().now().nanoseconds,
            "mode": "estop", "linear": 0.0, "angular": 0.0})
        self.safety_control_pub.publish(packet)
        first = not self.follow_estop_sent
        self.follow_estop_sent = True
        self.follow_last_estop_at = time.monotonic()
        if first:
            reason = self.follow_watchdog_fault or "FollowPath cancellation is unconfirmed"
            self.get_logger().error(reason + "; requesting latched safety stop")
            self.cancel_navigation(Trigger.Request(), Trigger.Response())
            self.publish_status("SAFETY_STOP", reason,
                                cancel_acknowledged=self.follow_cancel_acknowledged)

    def _monitor_follow_path(self):
        self._watch_navigation_inputs()
        self._monitor_initial_alignment()
        now = time.monotonic()
        self._update_controller_speed(now)
        requested = self.follow_requested_at
        if (self.goal_request_pending and requested is not None
                and now-requested >= max(.5, float(self.get_parameter("follow_request_timeout").value))
                and not self.follow_estop_sent):
            self._hold_unconfirmed_follow("FollowPath goal acceptance timed out")
        if self.active_goal is not None and not self.cancel_requested:
            last = self.follow_feedback_at
            if (last is not None and now-last >= max(.5, float(self.get_parameter("follow_feedback_timeout").value))):
                self.follow_watchdog_fault = "FollowPath feedback timed out"
                self.latest_path_key = None
                self.path_invalidated = True
                self._cancel_active_goal()
                self.publish_status("WAIT_CANCEL", self.follow_watchdog_fault,
                                    feedback_age_sec=now-last)
        if self.follow_cancel_since is not None:
            retry = max(.1, float(self.get_parameter("follow_cancel_retry_period").value))
            if self.active_goal is not None and (self.follow_last_cancel_at is None
                                                  or now-self.follow_last_cancel_at >= retry):
                self._send_follow_cancel()
            if (now-self.follow_cancel_since >= max(.5, float(self.get_parameter("follow_cancel_timeout").value))
                    and not self.follow_estop_sent):
                self.follow_watchdog_fault = self.follow_watchdog_fault or "FollowPath cancellation/result timed out"
                self._send_follow_safety_estop()
        if self.follow_estop_sent:
            status = self.motion_status or {}
            acknowledged = (self.motion_status_at is not None and now-self.motion_status_at < 1.
                            and status.get("estop_latched") is True)
            if not acknowledged and (self.follow_last_estop_at is None or now-self.follow_last_estop_at >= .5):
                self._send_follow_safety_estop()
        elif (self.active_goal is not None or self.goal_request_pending) and not self.cancel_requested:
            if self.last_speed_limit_at is None or now-self.last_speed_limit_at >= .2:
                self._publish_curve_limit()
        self._monitor_execution_progress()

    def _reset_execution_progress(self, *, clear_pending=False):
        detector = getattr(self, 'execution_progress', None)
        if detector is not None:
            detector.reset()
        self.execution_reference = None
        self.execution_route_id = None
        if clear_pending:
            self.pending_execution_failure = None
            self.execution_replan_cutoff_ns = None
            self.last_executed_intent = None
            self.execution_discarded_reason = None

    def _discard_execution_diagnosis(self, reason):
        """A transient safety/ownership interruption breaks spatial causality."""
        self.last_executed_intent = None
        detector = getattr(self, 'execution_progress', None)
        if detector is not None:
            detector.reset(getattr(self, 'execution_route_id', None))
        if getattr(self, 'pending_execution_failure', None) is not None:
            self.pending_execution_failure = None
            self.execution_discarded_reason = reason
            self.pending_path = None
            self.latest_path_key = None
            self.path_invalidated = True

    def _discard_scan_guard_execution(self):
        """A final-mux safety veto is not failed execution of DWB's command."""
        motion = getattr(self, 'motion_status', None) or {}
        if motion.get('scan_guard_hold') is not True:
            return
        reason = 'LiDAR safety guard interrupted execution'
        detail = motion.get('scan_guard_reason')
        if isinstance(detail, str) and detail:
            reason += ': '+detail[:160]
        self._discard_execution_diagnosis(reason)
        # Even a spontaneous ABORT without a pending diagnosis must use the
        # existing sensor-hold retry route, rather than charge a controller
        # failure using intent cached just before this physical output veto.
        self.execution_discarded_reason = reason

    def _select_execution_reference(self, path):
        detector = getattr(self, 'execution_progress', None)
        if detector is None:
            return
        if not self.get_parameter('execution_failure_enabled').value:
            self._discard_execution_diagnosis('Execution failure detection disabled')
            return
        points = self._points(path)
        reference = getattr(self, 'execution_reference', None)
        if reference is None or not paths_are_equivalent(reference, points):
            self.execution_reference = points
            self.execution_route_id = uuid.uuid4().hex
            detector.reset(self.execution_route_id)

    def on_navigation_command(self, message):
        self.latest_navigation_command = message
        self.latest_navigation_command_at = time.monotonic()

    def on_localization_health(self, message):
        try:
            health = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if isinstance(health, dict):
            self.localization_health = health
            self.localization_health_at = time.monotonic()
            if (health.get('level') != 'ok' or health.get('autonomous_allowed') is not True
                    or health.get('stop_latched') is True):
                self._discard_execution_diagnosis('Localization quality interrupted')

    def _execution_command_is_fresh(self, now):
        command = getattr(self, 'latest_navigation_command', None)
        arrival = getattr(self, 'latest_navigation_command_at', None)
        timeout = max(.2, float(self.get_parameter('execution_evidence_timeout').value))
        return (command is not None and arrival is not None and 0 <= now-arrival < timeout
                and self.follow_requested_at is not None and arrival >= self.follow_requested_at
                and all(isfinite(v) for v in (command.linear.x, command.angular.z)))

    def _execution_observation(self, *, require_intent=True):
        if getattr(self, 'execution_progress', None) is None or self.active_path is None:
            return None
        if (getattr(self, 'navigation_inputs_required', False)
                and self._navigation_input_error() is not None):
            return None
        now = time.monotonic()
        timeout = max(.2, float(self.get_parameter('execution_evidence_timeout').value))
        ros_ns = self.get_clock().now().nanoseconds
        if now < getattr(self, 'execution_clock_fault_until', float('-inf')):
            return None
        if ros_ns != getattr(self, 'execution_last_ros_ns', None):
            previous_ros = getattr(self, 'execution_last_ros_ns', None)
            self.execution_last_ros_ns = ros_ns
            self.execution_clock_advanced_at = now
            if previous_ros is not None and ros_ns < previous_ros:
                self.execution_clock_fault_until = now+.5
                return None
        if (ros_ns <= 0 or self.execution_clock_advanced_at is None
                or now-self.execution_clock_advanced_at > .5):
            return None
        if not (self._source_is_fresh(self.latest_scan, self.latest_scan_at)
                and self._source_is_fresh(self.latest_odom, self.latest_odom_at)):
            return None
        health = getattr(self, 'localization_health', None) or {}
        health_at = getattr(self, 'localization_health_at', None)
        health_stamp = health.get('sent_at_ns')
        if (health_at is None or now-health_at >= timeout
                or not isinstance(health_stamp, int) or isinstance(health_stamp, bool)
                or not -.25e9 <= ros_ns-health_stamp < timeout*1e9
                or health.get('autonomous_allowed') is not True
                or health.get('level') != 'ok' or health.get('stop_latched') is True):
            return None
        motion = self.motion_status or {}
        motion_stamp = motion.get('sent_at_ns')
        if (self.motion_status_at is None or now-self.motion_status_at >= timeout
                or not isinstance(motion_stamp, int) or isinstance(motion_stamp, bool)
                or not -.25e9 <= ros_ns-motion_stamp < timeout*1e9
                or motion.get('mode') != 'auto' or motion.get('available') is not True
                or motion.get('estop_latched') is True or motion.get('reset_pending') is True
                or motion.get('scan_guard_hold') is True
                or motion.get('autonomous_allowed') is not True):
            return None
        command = getattr(self, 'latest_navigation_command', None)
        command_at = getattr(self, 'latest_navigation_command_at', None)
        if require_intent and (command is None or command_at is None or now-command_at >= timeout):
            return None
        if require_intent and (self.follow_requested_at is None
                or command_at < self.follow_requested_at
                or self.motion_status_at < self.follow_requested_at):
            return None
        linear = motion.get('output_linear', 0.)
        angular = motion.get('output_angular', 0.)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (linear, angular)):
            return None
        if not isfinite(linear) or not isfinite(angular):
            return None
        if require_intent:
            raw = (command.linear.x, command.angular.z)
            if not all(isfinite(v) for v in raw):
                return None
            detector = self.execution_progress
            if ((abs(linear) < detector.min_linear and abs(angular) < detector.min_angular)
                    or (abs(raw[0]) < detector.min_linear and abs(raw[1]) < detector.min_angular)):
                return None
        # A valid message age is insufficient when the numeric payload is corrupt.
        odom = self.latest_odom
        odom_q = odom.pose.pose.orientation
        if not all(isfinite(v) for v in (odom.pose.pose.position.x, odom.pose.pose.position.y,
                odom.twist.twist.linear.x, odom.twist.twist.angular.z,
                odom_q.x, odom_q.y, odom_q.z, odom_q.w,
                *odom.pose.covariance, *odom.twist.covariance)):
            return None
        if not .8 <= odom_q.x**2+odom_q.y**2+odom_q.z**2+odom_q.w**2 <= 1.2:
            return None
        scan = self.latest_scan
        if (not scan.ranges or not isfinite(scan.angle_min)
                or not isfinite(scan.angle_increment) or scan.angle_increment == 0
                or not isfinite(scan.range_max) or scan.range_max <= 0
                or not isfinite(scan.range_min) or not 0 <= scan.range_min < scan.range_max
                or sum((isfinite(v) and scan.range_min <= v <= scan.range_max)
                       or v == float('inf') for v in scan.ranges)/len(scan.ranges) < .8):
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.active_path.header.frame_id or 'map',
                self.get_parameter('curve_base_frame').value, rclpy.time.Time())
        except (TransformException, ValueError):
            return None
        stamp = transform.header.stamp
        tf_ns = stamp.sec*1_000_000_000+stamp.nanosec
        p, q = transform.transform.translation, transform.transform.rotation
        if (tf_ns <= 0 or not -.25e9 <= ros_ns-tf_ns < timeout*1e9
                or not all(isfinite(v) for v in (p.x, p.y, q.x, q.y, q.z, q.w))
                or not .8 <= q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w <= 1.2):
            return None
        yaw = atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        points = self._points(self.active_path)
        if len(points) < 2:
            return None
        progress = path_progress(points, (p.x, p.y))
        a = point_at_progress(points, progress)
        b = point_at_progress(points, min(path_length(points), progress+.22))
        heading = atan2(b[1]-a[1], b[0]-a[0])
        if hypot(p.x-points[-1][0], p.y-points[-1][1]) <= .30:
            final_q = self.active_path.poses[-1].pose.orientation
            heading = atan2(2*(final_q.w*final_q.z+final_q.x*final_q.y),
                            1-2*(final_q.y*final_q.y+final_q.z*final_q.z))
        error = atan2(sin(heading-yaw), cos(heading-yaw))
        stable_points = self.execution_reference or points
        if require_intent:
            self.last_executed_intent = {'at': now, 'request_seq': self.follow_request_seq,
                                         'linear': linear, 'angular': angular}
        return {'at': now, 'ros_ns': ros_ns, 'pose': {'x': p.x, 'y': p.y, 'yaw': yaw},
                'segment': {'start': {'x': a[0], 'y': a[1]},
                            'end': {'x': b[0], 'y': b[1]}, 'heading': heading},
                'progress': path_progress(stable_points, (p.x, p.y)),
                'yaw_error': error, 'odom_yaw': atan2(
                    2*(odom_q.w*odom_q.z+odom_q.x*odom_q.y),
                    1-2*(odom_q.y*odom_q.y+odom_q.z*odom_q.z)),
                'linear': linear, 'angular': angular}

    def _execution_payload(self, reason, observation, evidence):
        context = self.active_context or self._goal_context(self.active_path)
        fingerprint = hashlib.sha256(repr(self._path_key(self.active_path)).encode()).hexdigest()[:24]
        return {'schema_version': 1, 'event_id': uuid.uuid4().hex,
                'created_at_ns': observation['ros_ns'], 'detected_at_ns': observation['ros_ns'],
                'goal': context.get('goal'), 'goal_stamp': context.get('goal_stamp'),
                'path_stamp': context.get('path_stamp'),
                'pose': observation['pose'], 'segment': observation['segment'],
                'reason': reason, 'path_fingerprint': fingerprint, 'evidence': evidence,
                'terminal_confirmed': False}

    def _monitor_execution_progress(self):
        detector = getattr(self, 'execution_progress', None)
        if detector is None:
            return
        if not self.get_parameter('execution_failure_enabled').value:
            self._discard_execution_diagnosis('Execution failure detection disabled')
            return
        # Runtime tuning must affect the detector as well as ROS's parameter
        # list. Reset evidence on a changed window to avoid mixing thresholds.
        try:
            limits = (float(self.get_parameter('execution_progress_window').value),
                      float(self.get_parameter('execution_min_progress').value),
                      float(self.get_parameter('execution_heading_progress').value))
            if limits != (detector.window, detector.min_progress, detector.heading_progress):
                detector = ExecutionProgress(window=limits[0], min_progress=limits[1],
                                             heading_progress=limits[2])
                self.execution_progress = detector
        except (TypeError, ValueError):
            detector.reset(self.execution_route_id)
            self.get_logger().warning('Invalid execution detection limits; route memory detection suspended',
                                      throttle_duration_sec=10.)
            return
        if (not self.get_parameter('execution_failure_enabled').value
                or self.active_goal is None or self.cancel_requested
                or self.navigation_suspended or self.path_invalidated
                or self.recovery_request is not None or self.follow_watchdog_fault
                or getattr(self, 'pending_execution_failure', None) is not None):
            detector.reset(self.execution_route_id)
            return
        observation = self._execution_observation()
        if observation is None:
            observation = self._execution_observation(require_intent=False)
            if observation is None or not self._execution_command_is_fresh(time.monotonic()):
                # A health/sensor/clock ownership fault breaks continuity, so
                # an earlier command can no longer explain a later abort.
                self.last_executed_intent = None
                detector.reset(self.execution_route_id)
                return
        sample = ProgressSample(observation['at'], observation['ros_ns'],
            observation['pose']['x'], observation['pose']['y'], observation['yaw_error'],
            observation['progress'], observation['linear'], observation['angular'],
            observation['pose']['yaw'], observation['odom_yaw'])
        decision = detector.observe(sample, self.execution_route_id)
        if decision is None:
            return
        reason, evidence = decision
        self.pending_execution_failure = self._execution_payload(reason, observation, evidence)
        self.pending_path = None
        self.latest_path_key = None
        self.path_invalidated = True
        self._cancel_active_goal()
        self.publish_status('WAIT_CANCEL', 'No execution progress; awaiting confirmed stop before replanning',
                            context=self.active_context, route_failure_pending=True,
                            execution_failure_reason=reason)

    def _publish_speed_limit(self, speed):
        self.follow_speed_limit = speed
        message = SpeedLimit()
        message.header.stamp = self.get_clock().now().to_msg()
        message.percentage = False
        # Nav2 defines 0.0 as "restore configured speed", not a stop command.
        message.speed_limit = float(speed)
        self.speed_limit_pub.publish(message)
        self.last_speed_limit_at = time.monotonic()

    def _publish_curve_limit(self):
        if not self.get_parameter("curve_speed_limit_enabled").value:
            if self.follow_speed_limit:
                self._publish_speed_limit(0.)
            return
        if self.active_path is None or len(self.active_path.poses) < 2:
            return
        points = self._points(self.active_path)
        length = path_length(points)
        progress = None
        alignment = False
        final_heading = False
        if getattr(self, 'tf_buffer', None) is not None:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.active_path.header.frame_id or 'map',
                    self.get_parameter('curve_base_frame').value, rclpy.time.Time())
                stamp = transform.header.stamp
                age = (self.get_clock().now().nanoseconds-stamp.sec*1_000_000_000-stamp.nanosec)/1e9
                if -.25 <= age < max(.1, float(self.get_parameter('curve_pose_timeout').value)):
                    position = transform.transform.translation
                    progress = path_progress(points, (position.x, position.y))
                    final_heading = hypot(position.x-points[-1][0], position.y-points[-1][1]) <= float(
                        self.get_parameter('curve_goal_heading_distance').value)
                    q = transform.transform.rotation
                    if q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w > .5:
                        yaw = atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                        target = point_at_progress(points, progress+.25)
                        alpha = atan2(target[1]-position.y, target[0]-position.x)-yaw
                        alignment = abs(atan2(sin(alpha), cos(alpha))) > float(
                            self.get_parameter('curve_alignment_angle').value)
            except (TransformException, ValueError):
                pass
        self.follow_path_progress = progress
        if ((alignment or final_heading)
                and not self.get_parameter('curve_linear_limit_preserves_angular_speed').value):
            # Stock DWB scales angular limits with a linear cap. Our physical
            # RouteTrackingDWB preserves yaw authority, so keep its bend cap
            # during heading corrections instead of restoring full linear speed.
            self._publish_speed_limit(0.)
            return
        # Humble FollowPath feedback can report distance_to_goal=0 throughout
        # a valid route. Use actual map pose for progress; missing pose retains
        # the worst remaining-route curvature cap instead of releasing it early.
        horizon = (length if progress is None else
                   max(.1, float(self.get_parameter('curve_preview_distance').value)))
        curvature = upcoming_curvature(points, progress or 0., horizon,
            max(.05, float(self.get_parameter('curve_minimum_chord').value)))
        try:
            speed = curvature_speed_limit(curvature,
                float(self.get_parameter("curve_max_linear_velocity").value),
                float(self.get_parameter("curve_max_angular_velocity").value),
                float(self.get_parameter("curve_max_lateral_acceleration").value))
            minimum = float(self.get_parameter('curve_min_linear_speed_limit').value)
            maximum = float(self.get_parameter('curve_max_linear_velocity').value)
            if not isfinite(minimum) or not 0 < minimum <= maximum:
                raise ValueError('curve minimum speed limit must be within the configured speed range')
            speed = max(minimum, speed)
            if progress is not None:
                speed = anticipatory_curve_speed_limit(
                    points, progress, horizon, maximum,
                    float(self.get_parameter('curve_max_angular_velocity').value),
                    float(self.get_parameter('curve_max_lateral_acceleration').value),
                    minimum_speed=minimum,
                    deceleration=float(self.get_parameter('curve_braking_deceleration').value),
                    braking_margin=float(self.get_parameter('curve_braking_margin').value),
                    minimum_chord=max(.05, float(self.get_parameter('curve_minimum_chord').value)))
        except ValueError as exc:
            self.get_logger().error(f"Invalid curve speed configuration: {exc}")
            self._hold_unconfirmed_follow("Invalid curve speed limit configuration")
            return
        base = self.controller_base_speed
        # Humble DWB ignores an absolute limit >= its configured maximum,
        # leaving an older low limit in place. Its explicit reset value is 0.
        # While the asynchronous maximum query is pending, straight sections
        # still reset at the declared cap; lower curved caps stay conservative.
        if base is not None:
            self._publish_speed_limit(0. if speed >= base-1e-6 else max(1e-6, speed))
        else:
            self._publish_speed_limit(0. if speed >= float(self.get_parameter(
                'curve_max_linear_velocity').value)-1e-6 else max(1e-6, speed))

    def _controller_speed_names(self):
        plugin = self.get_parameter('controller_id').value or 'FollowPath'
        return [plugin+'.max_speed_xy', plugin+'.desired_linear_vel']

    def on_controller_parameters(self, event):
        if event.node.rstrip('/') != '/controller_server':
            return
        names = self._controller_speed_names()
        for parameter in list(event.new_parameters)+list(event.changed_parameters):
            if (parameter.name in names and parameter.value.type == ParameterType.PARAMETER_DOUBLE
                    and isfinite(parameter.value.double_value) and parameter.value.double_value > 0):
                self.controller_base_speed = parameter.value.double_value
                self.controller_speed_event_at = time.monotonic()

    def _update_controller_speed(self, now):
        # Test harnesses and older deployments may omit the optional service.
        if getattr(self, 'controller_speed_client', None) is None:
            return
        future = self.controller_speed_future
        if future is not None:
            if future.done():
                try:
                    response = future.result()
                    if self.controller_speed_event_at <= self.controller_speed_requested_at:
                        found = False
                        for value in response.values:
                            if (value.type == ParameterType.PARAMETER_DOUBLE
                                    and isfinite(value.double_value) and value.double_value > 0):
                                self.controller_base_speed = value.double_value
                                found = True
                                break
                        if not found:
                            self.controller_speed_query_index = (getattr(
                                self, 'controller_speed_query_index', 0)+1) % 2
                except Exception as exc:
                    self.get_logger().warning(f'Controller speed query failed: {exc}')
                self.controller_speed_future = None
            elif now-self.controller_speed_requested_at > 3.:
                future.cancel()
                self.controller_speed_future = None
            return
        if (self.controller_base_speed is None and now >= self.controller_speed_next_request
                and self.controller_speed_client.service_is_ready()):
            self.controller_speed_next_request = now+2.
            self.controller_speed_requested_at = now
            request = GetParameters.Request()
            # Humble's parameter service may reject the entire request when
            # any one requested plugin-specific parameter is undeclared.
            request.names = [self._controller_speed_names()[getattr(
                self, 'controller_speed_query_index', 0)]]
            self.controller_speed_future = self.controller_speed_client.call_async(request)

    def on_scan(self, message):
        self.latest_scan = message
        self.latest_scan_at = time.monotonic()

    def on_odom(self, message):
        self.latest_odom = message
        self.latest_odom_at = time.monotonic()
        history = getattr(self, 'navigation_odom_history', None)
        if history is None:
            history = self.navigation_odom_history = deque(maxlen=256)
            self.navigation_odom_epoch = 0
        receipts = getattr(self, 'navigation_odom_receipts', None)
        if receipts is None:
            receipts = self.navigation_odom_receipts = deque(maxlen=256)
        try:
            sample = self._navigation_odom_sample(message)
            if history:
                previous = history[-1]
                dt = (sample[0]-previous[0])*1e-9
                continuous = (sample[1:3] == previous[1:3] and 0 < dt <= .25
                              and hypot(sample[3]-previous[3], sample[4]-previous[4]) <= .03+dt
                              and abs(atan2(sin(sample[5]-previous[5]),
                                            cos(sample[5]-previous[5]))) <= .05+3*dt)
                if sample == previous:
                    return  # Repeated source data cannot extend the history.
                if not continuous:
                    history.clear()
                    receipts.clear()
                    self.navigation_odom_epoch += 1
            history.append(sample)
            receipts.append((sample[0], self.latest_odom_at))
            while history and sample[0]-history[0][0] > 3_000_000_000:
                history.popleft()
        except (ValueError, AttributeError, TypeError):
            history.clear()
            receipts.clear()
            self.navigation_odom_epoch += 1
        self._promote_navigation_footprint()
        self._try_pending_replan_alignment()

    def on_footprint(self, message):
        received_at = time.monotonic()
        received_ros_ns = self.get_clock().now().nanoseconds
        receipt_pose = None
        pending_odom_fresh = False
        try:
            sample = self._navigation_odom_sample(self.latest_odom)
            source_age = (received_ros_ns-sample[0])*1e-9
            received_age = received_at-self.latest_odom_at
            if -.25 <= source_age < .25 and 0 <= received_age < .25:
                receipt_pose = (getattr(self, 'navigation_odom_epoch', 0), sample)
            # A fresh stamped polygon needs exact source-time alignment, not
            # the stricter receipt-time proof used only for legacy old headers.
            # Keep the ordinary navigation odometry deadlines while waiting.
            pending_odom_fresh = (isfinite(source_age) and isfinite(received_age)
                                  and -.25 <= source_age < .75
                                  and 0 <= received_age < .75)
        except (ValueError, AttributeError, TypeError):
            pass
        self._navigation_pending_footprint_error(received_at)
        record = {'message': message, 'received_at': received_at,
                  'received_ros_ns': received_ros_ns, 'receipt_pose': receipt_pose}
        try:
            stamp = message.header.stamp
            stamp_ns = stamp.sec*1_000_000_000+stamp.nanosec
            history = getattr(self, 'navigation_odom_history', ())
            newest = history[-1] if history else None
            previous = getattr(self, 'latest_footprint', None)
            # Only ordinary cross-topic arrival order may wait. Malformed
            # geometry and headers still replace the old input and fail closed.
            self._navigation_footprint_geometry(message.polygon.points, (0., 0., 0.))
            wait = (bool(message.header.frame_id) and pending_odom_fresh
                    and newest is not None and newest == sample
                    and (previous is None or message.header.frame_id == previous.header.frame_id)
                    and -.25e9 <= received_ros_ns-stamp_ns < 1.5e9
                    and 0 < stamp_ns-newest[0] <= self.FOOTPRINT_ODOM_WAIT_SECONDS*1e9)
        except (ValueError, AttributeError, TypeError):
            wait = False
        if wait:
            pending = getattr(self, 'navigation_pending_footprint', None)
            if pending is not None:
                # Keep the earliest acquisition: replacing it at sensor rate
                # could keep chasing newer stamps ahead of delayed odometry.
                return
            record.update(stamp_ns=stamp_ns, epoch=self.navigation_odom_epoch,
                          frames=newest[1:3],
                          wait_started_at=received_at)
            self.navigation_pending_footprint = record
            return
        self.navigation_pending_footprint = None
        self.navigation_footprint_alignment_fault = None
        self._install_navigation_footprint(record)

    def _install_navigation_footprint(self, record):
        self.latest_footprint = record['message']
        self.latest_footprint_at = record['received_at']
        self.navigation_footprint_receipt_ros_ns = record['received_ros_ns']
        self.navigation_footprint_receipt_pose = record['receipt_pose']

    def _navigation_pending_footprint_error(self, now):
        pending = getattr(self, 'navigation_pending_footprint', None)
        if pending is not None:
            elapsed = now-pending['wait_started_at']
            if pending['epoch'] != getattr(self, 'navigation_odom_epoch', 0):
                issue = 'Odometry changed while aligning footprint'
            elif not isfinite(elapsed) or not 0 <= elapsed < self.FOOTPRINT_ODOM_WAIT_SECONDS:
                issue = 'Timed out waiting for footprint acquisition odometry'
            else:
                issue = None
            if issue is not None:
                self.navigation_pending_footprint = None
                self.navigation_footprint_alignment_fault = issue
        return getattr(self, 'navigation_footprint_alignment_fault', None)

    def _promote_navigation_footprint(self):
        self._navigation_pending_footprint_error(time.monotonic())
        pending = getattr(self, 'navigation_pending_footprint', None)
        if pending is None:
            return
        try:
            self._navigation_odom_at(pending['stamp_ns'], *pending['frames'])
        except ValueError:
            return  # No extrapolation: the next continuous odometry must bracket it.
        self.navigation_pending_footprint = None
        self.navigation_footprint_alignment_fault = None
        self._install_navigation_footprint(pending)

    def on_costmap(self, message):
        self.latest_costmap = message
        self.latest_costmap_at = time.monotonic()

    @staticmethod
    def _navigation_pose(position, orientation):
        values = (position.x, position.y, position.z,
                  orientation.x, orientation.y, orientation.z, orientation.w)
        norm = sum(value*value for value in values[3:])
        if (not all(isfinite(value) for value in values) or not .8 <= norm <= 1.2
                or abs(orientation.x) > .05 or abs(orientation.y) > .05):
            raise ValueError('Invalid planar pose')
        yaw = atan2(2*(orientation.w*orientation.z+orientation.x*orientation.y),
                    norm-2*(orientation.y**2+orientation.z**2))
        return position.x, position.y, yaw

    def _navigation_odom_sample(self, message):
        stamp = message.header.stamp
        stamp_ns = stamp.sec*1_000_000_000+stamp.nanosec
        frame = message.header.frame_id
        base = message.child_frame_id or self.get_parameter('curve_base_frame').value
        if stamp_ns <= 0 or not frame or not base:
            raise ValueError('Missing odometry timestamp or frame')
        return (stamp_ns, frame, base,
                *self._navigation_pose(message.pose.pose.position, message.pose.pose.orientation))

    def _navigation_odom_at(self, stamp_ns, frame, base):
        history = getattr(self, 'navigation_odom_history', ())
        for sample in history:
            if sample[0] == stamp_ns and sample[1:3] == (frame, base):
                return sample[3:]
        for first, last in zip(history, list(history)[1:]):
            if (first[1:3] == last[1:3] == (frame, base)
                    and first[0] < stamp_ns < last[0]
                    and last[0]-first[0] <= 250_000_000):
                fraction = (stamp_ns-first[0])/(last[0]-first[0])
                yaw = first[5]+fraction*atan2(sin(last[5]-first[5]), cos(last[5]-first[5]))
                return (first[3]+fraction*(last[3]-first[3]),
                        first[4]+fraction*(last[4]-first[4]), yaw)
        first = history[0] if history else None
        last = history[-1] if history else None
        def frames(sample):
            return None if sample is None else tuple(str(value)[:80] for value in sample[1:3])
        raise ValueError(
            'No continuous odometry at footprint acquisition time: '
            f'requested_ns={stamp_ns} first_ns={None if first is None else first[0]} '
            f'last_ns={None if last is None else last[0]} count={len(history)} '
            f'epoch={getattr(self, "navigation_odom_epoch", 0)} '
            f'requested_frames={frames((None, frame, base))} '
            f'first_frames={frames(first)} last_frames={frames(last)}')

    def _navigation_transform(self, target, source, now, stamp_ns=None):
        if not target or not source:
            raise ValueError('Missing navigation frame')
        if target == source:
            return 0., 0., 0.
        query = (rclpy.time.Time() if stamp_ns is None
                 else rclpy.time.Time(nanoseconds=stamp_ns))
        transform = self.tf_buffer.lookup_transform(target, source, query)
        stamp = transform.header.stamp
        stamp_seconds = stamp.sec+stamp.nanosec*1e-9
        # A zero stamp is valid for a wholly static TF chain.
        reference = now if stamp_ns is None else stamp_ns*1e-9
        if stamp_seconds != 0 and not -.25 <= reference-stamp_seconds < 1.0:
            raise ValueError('Stale navigation frame transform')
        return self._navigation_pose(transform.transform.translation, transform.transform.rotation)

    @staticmethod
    def _navigation_point(point, transform):
        x, y, yaw = transform
        return x+cos(yaw)*point[0]-sin(yaw)*point[1], y+sin(yaw)*point[0]+cos(yaw)*point[1]

    def _navigation_footprint_geometry(self, points, transform):
        if any(not all(isfinite(value) for value in (point.x, point.y, point.z)) for point in points):
            raise ValueError('Invalid footprint coordinates')
        polygon = [self._navigation_point((point.x, point.y), transform) for point in points]
        if len(polygon) > 1 and polygon[0] == polygon[-1]:
            polygon.pop()
        if len(polygon) < 3 or len(set(polygon)) != len(polygon):
            raise ValueError('Invalid footprint polygon')
        edges = list(zip(polygon, polygon[1:]+polygon[:1]))
        def cross(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        for first, (a, b) in enumerate(edges):
            for second in range(first+2, len(edges)):
                if first == 0 and second == len(edges)-1:
                    continue
                c, d = edges[second]
                if (cross(a, b, c)*cross(a, b, d) <= 0 and cross(c, d, a)*cross(c, d, b) <= 0
                        and max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0]))
                        and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))):
                    raise ValueError('Self-intersecting footprint polygon')
        twice_area = sum(a[0]*b[1]-b[0]*a[1] for a, b in edges)
        radius = float(self.get_parameter('recovery_robot_radius').value)
        if not isfinite(radius) or radius <= 0 or abs(twice_area)/2 < .1*radius**2:
            raise ValueError('Footprint area is too small')
        center_x = sum((a[0]+b[0])*(a[0]*b[1]-b[0]*a[1]) for a, b in edges)/(3*twice_area)
        center_y = sum((a[1]+b[1])*(a[0]*b[1]-b[0]*a[1]) for a, b in edges)/(3*twice_area)
        centered = [(px-center_x, py-center_y) for px, py in polygon]
        radii = [hypot(px, py) for px, py in centered]
        if not .5*radius <= max(radii) <= 2*radius+.05:
            raise ValueError('Footprint size differs from the robot')
        return polygon, center_x, center_y, centered, radii

    def _navigation_input_error(self, *, scan_odom_timeout=.75):
        """Compare geometry with odometry from the same proven source/receipt time."""
        now = self.get_clock().now().nanoseconds*1e-9
        received_now = time.monotonic()
        alignment_issue = self._navigation_pending_footprint_error(received_now)
        if alignment_issue is not None:
            return alignment_issue
        footprint_fresh_header = False
        for label, message, received_at, timeout in (
                ('scan', self.latest_scan, self.latest_scan_at, scan_odom_timeout),
                ('odom', self.latest_odom, self.latest_odom_at, scan_odom_timeout),
                ('local costmap', self.latest_costmap, self.latest_costmap_at, 2.),
                ('footprint', self.latest_footprint, self.latest_footprint_at, 1.5)):
            if message is None:
                return 'Missing '+label
            if (received_at is None or not isfinite(received_at)
                    or not 0 <= received_now-received_at < timeout):
                return 'Stale '+label+' receipt'
            stamp = message.header.stamp
            stamp_seconds = stamp.sec+stamp.nanosec*1e-9
            age = now-stamp_seconds
            if not isfinite(age) or age < -.25:
                return 'Invalid '+label+' timestamp'
            fresh_header = stamp_seconds > 0 and age < timeout
            if label == 'footprint':
                footprint_fresh_header = fresh_header
            elif not fresh_header:
                return 'Stale '+label+' header'
        scan, odom, costmap = self.latest_scan, self.latest_odom, self.latest_costmap
        if (len(scan.ranges) < 2 or not all(isfinite(value) for value in
                (scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max))
                or scan.angle_increment == 0 or not 0 <= scan.range_min < scan.range_max
                or sum((isfinite(value) and scan.range_min <= value <= scan.range_max)
                       or value == float('inf') for value in scan.ranges)/len(scan.ranges) < .8):
            return 'Invalid scan geometry or ranges'
        geometry_stage = 'odometry pose'
        try:
            x, y, yaw = self._navigation_pose(odom.pose.pose.position, odom.pose.pose.orientation)
            if not all(isfinite(value) for value in (
                    odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z,
                    odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z,
                    *odom.pose.covariance, *odom.twist.covariance)):
                return 'Invalid odometry values'
            base = odom.child_frame_id or self.get_parameter('curve_base_frame').value
            if not scan.header.frame_id:
                return 'Missing scan frame'
            if scan.header.frame_id != self.get_parameter('recovery_scan_frame').value:
                geometry_stage = 'scan frame transform'
                self._navigation_transform(base, scan.header.frame_id, now)
            info = costmap.info
            if (info.width <= 0 or info.height <= 0 or not isfinite(info.resolution)
                    or info.resolution <= 0 or len(costmap.data) != info.width*info.height
                    or any(value < -1 or value > 100 for value in costmap.data)):
                return 'Invalid local costmap dimensions or data'
            geometry_stage = 'local costmap origin'
            origin = self._navigation_pose(info.origin.position, info.origin.orientation)
            center = self._navigation_point((info.width*info.resolution/2,
                                             info.height*info.resolution/2), origin)
            geometry_stage = 'local costmap frame transform'
            center = self._navigation_point(center, self._navigation_transform(
                odom.header.frame_id, costmap.header.frame_id, now))
            if hypot(x-center[0], y-center[1]) > .35:
                return 'Local costmap is not centered on odometry'
            if footprint_fresh_header:
                stamp = self.latest_footprint.header.stamp
                footprint_ns = stamp.sec*1_000_000_000+stamp.nanosec
                geometry_stage = 'footprint timestamp and odometry history'
                footprint_pose = self._navigation_odom_at(footprint_ns, odom.header.frame_id, base)
            else:
                proof = getattr(self, 'navigation_footprint_receipt_pose', None)
                if (proof is None or proof[0] != getattr(self, 'navigation_odom_epoch', 0)
                        or proof[1][1:3] != (odom.header.frame_id, base)):
                    return 'Old footprint has no fresh receipt-time odometry proof'
                footprint_pose = proof[1][3:]
                footprint_ns = self.navigation_footprint_receipt_ros_ns
            geometry_stage = 'footprint frame transform'
            footprint_transform = self._navigation_transform(
                odom.header.frame_id, self.latest_footprint.header.frame_id, now, footprint_ns)
        except (TransformException, ValueError, AttributeError) as exc:
            self.get_logger().warning(
                f'Navigation geometry validation failed at {geometry_stage}: '
                f'{type(exc).__name__}: {exc}', throttle_duration_sec=2.0)
            return 'Invalid navigation pose or unavailable frame transform'
        try:
            polygon, center_x, center_y, centered, radii = self._navigation_footprint_geometry(
                self.latest_footprint.polygon.points, footprint_transform)
        except ValueError as exc:
            return str(exc)
        radius = float(self.get_parameter('recovery_robot_radius').value)
        footprint_x, footprint_y, footprint_yaw = footprint_pose
        if hypot(footprint_x-center_x, footprint_y-center_y) > .10:
            return 'Footprint disagrees with current odometry'
        if not footprint_fresh_header and hypot(x-center_x, y-center_y) > .10:
            return 'Old footprint position no longer matches current odometry'
        # A valid source stamp permits motion compensation. The legacy old-
        # header exception retains its current position/orientation guards;
        # downstream consumers cannot recover geometry time from that header.
        body_yaw = footprint_yaw if footprint_fresh_header else yaw
        body = tuple(self._navigation_point(point, (0., 0., -body_yaw)) for point in centered)
        angles = sorted(atan2(py, px) for px, py in centered)
        gaps = [b-a for a, b in zip(angles, angles[1:]+[angles[0]+2*pi])]
        circular = (len(polygon) >= 8 and max(gaps) <= pi/3
                    and max(radii)-min(radii) <= .05*radius
                    and .8*radius <= sum(radii)/len(radii) <= radius+.08)
        reference = getattr(self, 'navigation_footprint_reference', None)
        if not footprint_fresh_header and not circular:
            if reference is None or reference[0] != (odom.header.frame_id, base):
                return 'Old asymmetric footprint has no current body-frame reference'
            previous = reference[1]
            mismatch = max(max(min(hypot(a[0]-b[0], a[1]-b[1]) for b in other) for a in shape)
                           for shape, other in ((body, previous), (previous, body)))
            if mismatch > .05:
                return 'Old footprint orientation or shape disagrees with odometry'
        if footprint_fresh_header:
            self.navigation_footprint_reference = ((odom.header.frame_id, base), body)
        return None

    def _hold_navigation_inputs(self, issue, context, *, force_status=False):
        changed = getattr(self, 'navigation_input_fault', None) != issue
        self.navigation_input_fault = issue
        self.navigation_input_cutoff_ns = self.get_clock().now().nanoseconds
        self._reset_execution_progress(clear_pending=True)
        self.path_invalidated = True
        self.latest_path_key = None
        self.pending_path = None
        self._cancel_initial_alignment()
        self._cancel_active_goal()
        if changed or force_status:
            # Each rejected planner retry must re-arm its retained-goal loop;
            # steady watchdog ticks only report a newly changed fault.
            self.publish_status('WAIT_REPLAN', 'Navigation input unavailable: '+issue,
                                context=context, route_failure=False, input_hold=True)

    def _watch_navigation_inputs(self):
        if (not getattr(self, 'navigation_inputs_required', False) or self.navigation_suspended
                or self.recovery_request is not None or self.backup_goal is not None
                or self.backup_goal_request_pending
                or (self.active_goal is None and not self.goal_request_pending
                    and self.initial_alignment is None)):
            return
        issue = self._navigation_input_error(
            scan_odom_timeout=getattr(self, 'navigation_sensor_continuity_timeout', 1.0))
        if issue is not None:
            context = self.active_context or self.request_context or self._goal_context(self.active_path)
            self._hold_navigation_inputs(issue, context)

    def on_motion_status(self, message):
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(status, dict):
            return
        self.motion_status = status
        self.motion_status_at = time.monotonic()
        self._discard_scan_guard_execution()
        if (status.get('mode') != 'auto' or status.get('available') is not True
                or status.get('autonomous_allowed') is not True
                or status.get('estop_latched') is True or status.get('reset_pending') is True):
            self._discard_execution_diagnosis('Manual, hold, safety stop or unavailable motion output')
            self._cancel_initial_alignment()
            if status.get('estop_latched') is True:
                self.pending_path = None
                self.latest_path_key = None
                self.path_invalidated = True
        if (self.recovery_request is not None
                and (status.get("mode") != "auto"
                     or status.get("available") is not True
                     or status.get("estop_latched") is True)):
            self._fail_recovery("Manual control, hold, emergency stop or unavailable motion output")

    def on_recovery_request(self, message):
        """Accept only a bounded reverse request for the current explicit goal."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        if ('episode_id' in payload and not self._valid_episode_id(payload['episode_id'])):
            return
        goal_id = payload.get("goal_id")
        try:
            sec_text, nsec_text = goal_id.split(":")
            sec, nsec = int(sec_text), int(nsec_text)
        except (AttributeError, TypeError, ValueError):
            return
        if sec < 0 or not 0 <= nsec < 1_000_000_000 or not (sec or nsec):
            return
        requested_stamp_ns = sec * 1_000_000_000 + nsec
        current = self.requested_goal
        current_stamp_ns = (0 if current is None else
                            current.header.stamp.sec * 1_000_000_000
                            + current.header.stamp.nanosec)
        if payload.get("cancel") is True:
            # The planner invalidated the map while a reverse maneuver may be
            # pending or already moving.  An empty Path on another DDS topic
            # cannot be relied upon to overtake the BackUp action request.
            if requested_stamp_ns < current_stamp_ns or self.navigation_suspended:
                return
            owned = ((self.recovery_request or {}).get("context")
                     or (getattr(self, "backup_action_context", None)
                         if self.backup_goal is not None or self.backup_goal_request_pending else None))
            pending = getattr(self, "pending_recovery_request", None)
            if (requested_stamp_ns == current_stamp_ns and owned is not None
                    and not self._same_recovery_episode(payload, owned)):
                return
            if (pending is not None and pending["goal_stamp_ns"] == requested_stamp_ns
                    and not self._same_recovery_episode(payload, pending["payload"])):
                return
            stamp = self.get_clock().now().to_msg()
            cutoff_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            self.pending_recovery_cancel = {
                "goal_stamp_ns": requested_stamp_ns,
                "received_at": time.monotonic(),
                "path_cutoff_ns": cutoff_ns,
                "episode_id": payload.get("episode_id"),
            }
            if requested_stamp_ns > current_stamp_ns:
                pending = self.pending_recovery_request
                if (pending is not None and
                        pending["goal_stamp_ns"] == requested_stamp_ns):
                    self.pending_recovery_request = None
                return
            self.pending_recovery_request = None
            self.pending_path = None
            self.latest_path_key = None
            self.active_path = None
            self.path_invalidated = True
            self.recovery_cancel_path_cutoff_ns = cutoff_ns
            context = ((self.recovery_request or {}).get("context")
                       or self._recovery_context(payload))
            self._abort_recovery(safety_fault=True)
            self._cancel_initial_alignment()
            if self.goal_request_pending:
                self.cancel_when_accepted = True
            self._cancel_active_goal()
            self.publish_status(
                "RECOVERY_FAILED",
                str(payload.get("reason", "Planner canceled reverse recovery"))[:160],
                context=context,
            )
            return
        canceled = self.pending_recovery_cancel
        if (canceled is not None
                and canceled["goal_stamp_ns"] == requested_stamp_ns
                and self._same_recovery_episode(canceled, payload)
                and time.monotonic() - canceled["received_at"]
                <= self.RECOVERY_REQUEST_BUFFER_SECONDS):
            # Drop delayed same-goal requests from the invalidated map. A
            # newer planner attempt can retry after this short DDS window.
            if requested_stamp_ns == current_stamp_ns:
                self.publish_status("RECOVERY_FAILED", "Reverse request was canceled by map change",
                                    context=self._recovery_context(payload))
            return
        if requested_stamp_ns > current_stamp_ns:
            # The planner can publish a no-path recovery request before this
            # subscriber receives the matching /goal on a different DDS topic.
            # Buffer only a newer request, for a short bounded interval. It
            # cannot actuate anything until the exact stamped goal arrives.
            if (self.stop_stamp_ns is not None and requested_stamp_ns <= self.stop_stamp_ns):
                return
            pending = getattr(self, "pending_recovery_request", None)
            if pending is None or requested_stamp_ns >= pending["goal_stamp_ns"]:
                self.pending_recovery_request = {
                    "goal_stamp_ns": requested_stamp_ns,
                    "received_at": time.monotonic(), "payload": payload,
                }
            return
        if requested_stamp_ns != current_stamp_ns or self.navigation_suspended:
            self.get_logger().warning("Ignoring reverse request for a stale or unstamped goal")
            return
        self._accept_recovery_request(payload)

    def _accept_recovery_request(self, payload):
        """Validate a request only after its exact stamped goal is current."""
        if ('episode_id' in payload and not self._valid_episode_id(payload['episode_id'])):
            return
        context = self._recovery_context(payload)
        stamp = context.get("goal_stamp") or {}
        expected_id = f'{stamp.get("sec", 0)}:{stamp.get("nanosec", 0)}'
        if (self.requested_goal is None or expected_id == "0:0"
                or payload.get("goal_id") != expected_id):
            self.get_logger().warning("Ignoring reverse request for a stale or unstamped goal")
            return
        if self.recovery_request is not None:
            return
        if self.backup_goal is not None or self.backup_goal_request_pending:
            self.publish_status("RECOVERY_FAILED",
                                "Previous BackUp action has not finished cancellation",
                                context=context)
            return
        distance, speed = payload.get("distance"), payload.get("speed")
        if (isinstance(distance, bool) or isinstance(speed, bool)
                or not isinstance(distance, (int, float))
                or not isinstance(speed, (int, float))
                or not isfinite(distance) or not isfinite(speed)
                or not max(0.10, self.get_parameter("recovery_min_distance").value)
                <= distance <= min(0.35, self.get_parameter("recovery_max_distance").value)
                or not max(0.03, self.get_parameter("recovery_min_speed").value)
                <= speed <= min(0.12, self.get_parameter("recovery_max_speed").value)):
            self.publish_status("RECOVERY_FAILED", "Reverse distance or speed is outside safety bounds",
                                context=context)
            return
        if distance / speed + 2.0 > min(15.0, self.get_parameter("recovery_max_time").value):
            self.publish_status("RECOVERY_FAILED", "Reverse request exceeds the action time limit",
                                context=context)
            return
        try:
            certificate = self._parse_recovery_start(payload.get('certified_start'))
        except ValueError as exc:
            self.publish_status('RECOVERY_FAILED', str(exc), context=context)
            return
        self.recovery_seq += 1
        now = self.get_clock().now().to_msg()
        self.recovery_request = {
            "seq": self.recovery_seq, "context": context,
            "distance": float(distance), "speed": float(speed),
            "reason": str(payload.get("reason", "Safe route unavailable"))[:160],
            "settle_deadline": None, "started_at": None, "traveled": 0.0,
            "requested_at": time.monotonic(),
            "requested_ros_ns": now.sec * 1_000_000_000 + now.nanosec,
            "certified_start": certificate,
            "certified_odom_epoch": getattr(self, 'navigation_odom_epoch', 0),
        }
        # This owned identity survives clearing recovery_request while a
        # BackUp acceptance/cancel/result is still in flight.
        self.backup_action_context = context
        self.pending_path = None
        self.latest_path_key = None
        self.path_invalidated = True
        self._cancel_initial_alignment()
        if self.goal_request_pending:
            self.cancel_when_accepted = True
        self._cancel_active_goal()
        self.publish_status("RECOVERING", "Waiting for FollowPath to stop before checked reverse",
                            context=context)
        self._try_start_recovery()

    def _parse_recovery_start(self, payload):
        """A planner sweep certificate is mandatory, including after goal buffering."""
        if not isinstance(payload, dict):
            raise ValueError('Reverse requires a certified stationary start')
        frames = tuple(payload.get(name) for name in
                       ('frame_id', 'child_frame_id', 'map_frame', 'map_base_frame'))
        stamps = tuple(payload.get(name) for name in
                       ('stamp_ns', 'map_stamp_ns', 'certified_at_ns'))
        fields = ('x', 'y', 'yaw', 'map_x', 'map_y', 'map_yaw', 'max_position_drift_m',
                  'max_yaw_drift_rad', 'sweep_reserve_m')
        if (any(not isinstance(frame, str) or not frame.strip() for frame in frames)
                or any(type(stamp) is not int or not 0 < stamp < 2**63 for stamp in stamps)
                or any(isinstance(payload.get(name), bool)
                       or not isinstance(payload.get(name), (int, float))
                       or not isfinite(payload[name]) for name in fields)
                or not 0 < payload['max_position_drift_m'] <= self.RECOVERY_START_POSITION_TOLERANCE
                or not 0 < payload['max_yaw_drift_rad'] <= self.RECOVERY_START_YAW_TOLERANCE
                or payload['sweep_reserve_m'] < self.RECOVERY_SWEEP_RESERVE):
            raise ValueError('Invalid reverse start certificate geometry or bounds')
        result = {name: payload[name] for name in (*fields, 'frame_id', 'child_frame_id',
                                                 'map_frame', 'map_base_frame', 'map_stamp_ns',
                                                 'stamp_ns', 'certified_at_ns')}
        if self._recovery_certificate_age_error(
                result, max_age=self.RECOVERY_CERTIFICATE_RECEIPT_MAX_AGE) is not None:
            raise ValueError('Reverse start certificate is stale or has invalid source time')
        return result

    def _recovery_certificate_age_error(self, certificate, *, max_age=None):
        if max_age is None:
            max_age = self.RECOVERY_CERTIFICATE_MAX_AGE
        now_ns = self.get_clock().now().nanoseconds
        issued_age = (now_ns-certificate['certified_at_ns'])*1e-9
        source_age_at_issue = (certificate['certified_at_ns']-certificate['stamp_ns'])*1e-9
        map_age_at_issue = (certificate['certified_at_ns']-certificate['map_stamp_ns'])*1e-9
        if (not -.25 <= issued_age < max_age
                or not -.25 <= source_age_at_issue < .75
                or not -.25 <= map_age_at_issue < .75):
            return 'Reverse start certificate is stale or has invalid source time'
        return None

    def _recovery_start_error(self, request):
        """Recheck the certified odometry epoch and actual axis after cancellation."""
        certificate = request['certified_start']
        issue = self._recovery_certificate_age_error(certificate)
        if issue is not None:
            return issue
        if request['certified_odom_epoch'] != getattr(self, 'navigation_odom_epoch', 0):
            return 'Odometry epoch changed after reverse start certification'
        try:
            current = self._navigation_odom_sample(self.latest_odom)
        except (ValueError, TypeError, AttributeError):
            return 'Reverse start requires valid current odometry'
        frames = certificate['frame_id'], certificate['child_frame_id']
        if current[1:3] != frames:
            return 'Odometry frame changed after reverse start certification'
        source_age = (self.get_clock().now().nanoseconds-current[0])*1e-9
        receipt_age = time.monotonic()-self.latest_odom_at
        if (not isfinite(receipt_age) or not 0 <= receipt_age < .75
                or not -.25 <= source_age < .75):
            return 'Reverse start requires fresh source and receipt odometry'
        stamp = certificate['stamp_ns']
        history = getattr(self, 'navigation_odom_history', ())
        baseline = next((sample for sample in history
                         if sample[0] == stamp and sample[1:3] == frames), None)
        if baseline is None:
            if history and history[-1][1:3] == frames and 0 < stamp-history[-1][0] <= 250_000_000:
                return 'WAIT_CERTIFIED_ODOM'
            return 'Certified reverse odometry sample is unavailable'
        yaw_error = lambda first, last: abs(atan2(sin(first-last), cos(first-last)))
        if (hypot(baseline[3]-certificate['x'], baseline[4]-certificate['y']) > 1e-6
                or yaw_error(baseline[5], certificate['yaw']) > 1e-6):
            return 'Reverse certificate disagrees with measured source odometry'
        if (current[0] < stamp
                or hypot(current[3]-baseline[3], current[4]-baseline[4]) > certificate['max_position_drift_m']
                or yaw_error(current[5], baseline[5]) > certificate['max_yaw_drift_rad']):
            return 'Robot moved outside the certified reverse start during handover'
        # The swept raw grid uses map coordinates. Stable odometry alone
        # cannot detect an AMCL/map->odom correction during the handover.
        try:
            transform = self.tf_buffer.lookup_transform(certificate['map_frame'],
                certificate['map_base_frame'], rclpy.time.Time())
            if (transform.header.frame_id != certificate['map_frame']
                    or transform.child_frame_id != certificate['map_base_frame']):
                return 'Reverse certificate map transform frames differ'
            map_stamp = transform.header.stamp.sec*1_000_000_000+transform.header.stamp.nanosec
            map_age = (self.get_clock().now().nanoseconds-map_stamp)*1e-9
            if map_stamp <= 0 or not -.25 <= map_age < .75:
                return 'Reverse start requires a fresh map transform'
            if map_stamp < certificate['map_stamp_ns']:
                return 'WAIT_CERTIFIED_TF'
            x, y, yaw = self._navigation_pose(transform.transform.translation,
                                               transform.transform.rotation)
        except TransformException:
            return 'WAIT_CERTIFIED_TF'
        except (ValueError, TypeError, AttributeError):
            return 'Invalid map transform after reverse start certification'
        if (hypot(x-certificate['map_x'], y-certificate['map_y']) > certificate['max_position_drift_m']
                or yaw_error(yaw, certificate['map_yaw']) > certificate['max_yaw_drift_rad']):
            return 'Map pose changed outside the certified reverse start during handover'
        return None

    def _abort_recovery(self, *, safety_fault=False):
        """Invalidate callbacks from an old maneuver and cancel its Nav2 action."""
        if self.recovery_request is not None:
            self.backup_action_context = self.recovery_request["context"]
        self.recovery_seq += 1
        self.recovery_request = None
        if self.backup_goal_request_pending or self.backup_goal is not None:
            if self.backup_cancel_pending_since is None:
                self.backup_cancel_pending_since = time.monotonic()
            self.backup_cancel_safety_fault |= safety_fault
            self.backup_cancel_replan_required = True
        if self.backup_goal_request_pending:
            self.backup_cancel_when_accepted = True
        self._cancel_backup_goal()

    def _cancel_backup_goal(self):
        if self.backup_goal is not None and not self.backup_cancel_requested:
            self.backup_cancel_requested = True
            self._send_backup_cancel()

    def _send_backup_cancel(self):
        if self.backup_goal is None:
            return
        self.backup_last_cancel_at = time.monotonic()
        try:
            self.backup_goal.cancel_goal_async()
        except Exception as exc:
            self.get_logger().error(f"Nav2 BackUp cancel request failed: {exc}")

    def _send_recovery_safety_estop(self):
        # The mux remembers source sequences across follower process restarts.
        # Steady-clock nanoseconds keep this sequence strictly increasing.
        self.safety_control_seq = max(self.safety_control_seq + 1, time.monotonic_ns())
        message = String()
        message.data = json.dumps({
            "source_id": self.SAFETY_CONTROL_SOURCE,
            "seq": self.safety_control_seq,
            "sent_at_ns": self.get_clock().now().nanoseconds,
            "mode": "estop", "linear": 0.0, "angular": 0.0,
        })
        self.safety_control_pub.publish(message)
        first_packet = not self.backup_cancel_estop_sent
        self.backup_cancel_estop_sent = True
        self.backup_last_estop_at = time.monotonic()
        if not first_packet:
            return
        self.get_logger().error("BackUp cancellation is unconfirmed; requesting latched safety stop")
        # A later web reset must not resurrect the old mission goal. The
        # operator must explicitly issue a new goal after acknowledging estop.
        self.cancel_navigation(Trigger.Request(), Trigger.Response())

    def _clear_backup_cancel_watch(self):
        self.backup_cancel_pending_since = None
        self.backup_last_cancel_at = None
        self.backup_cancel_safety_fault = False
        self.backup_cancel_replan_required = False
        self.backup_cancel_estop_sent = False
        self.backup_last_estop_at = None

    def _notify_backup_stopped_after_cancel(self, detail):
        # A route produced while BackUp was still in flight starts at an
        # unconfirmed pose. Its planner must run again after stop confirmation.
        discarded = self.pending_path
        discarded_context = (self._goal_context(discarded)
                             if discarded is not None and discarded.poses
                             and self._path_matches_goal(discarded) else None)
        self.pending_path = None
        self.latest_path_key = None
        self.path_invalidated = True
        if not self.navigation_suspended and self.requested_goal is not None:
            self.publish_status("RECOVERY_STOPPED", detail,
                                context=(getattr(self, "backup_action_context", None)
                                         or self._goal_context()))
            if discarded_context is not None:
                # The old BackUp terminal remains owned by its old episode.
                # Separately ask for a fresh start for the actual newer route
                # discarded at this barrier, without relabeling that terminal.
                self.publish_status("WAIT_REPLAN", "Pending route discarded at confirmed BackUp stop; refresh start pose",
                                    context=discarded_context, route_failure=False, input_hold=True)

    def _monitor_backup_cancellation(self):
        since = self.backup_cancel_pending_since
        if since is None:
            return
        now = time.monotonic()
        if (self.backup_goal is not None and
                (self.backup_last_cancel_at is None or
                 now - self.backup_last_cancel_at >= self.BACKUP_CANCEL_RETRY_SECONDS)):
            self._send_backup_cancel()
        status = self.motion_status or {}
        status_fresh = (self.motion_status_at is not None
                        and now - self.motion_status_at < 1.0)
        if self.backup_cancel_estop_sent:
            # A single best-effort packet is not proof the mux latched it.
            # Retransmit until its status confirms the stop or a human-held
            # manual mode suppresses automatic output.
            if status_fresh and (status.get("estop_latched") is True
                                 or status.get("mode") != "auto"):
                return
            if (self.backup_last_estop_at is None or
                    now - self.backup_last_estop_at >= self.BACKUP_CANCEL_RETRY_SECONDS):
                self._send_recovery_safety_estop()
            return
        delay = (self.BACKUP_CANCEL_SAFETY_ESTOP_SECONDS
                 if self.backup_cancel_safety_fault
                 else self.BACKUP_CANCEL_GENERAL_ESTOP_SECONDS)
        if now - since < delay:
            return
        # A confirmed manual/hold/estop mode already suppresses Nav2 output.
        # A stale status cannot prove that the robot has stopped.
        if (status_fresh and
                (status.get("mode") != "auto" or status.get("estop_latched") is True)):
            return
        self._send_recovery_safety_estop()

    def _fail_recovery(self, reason):
        request = self.recovery_request
        if request is None:
            return
        context = request["context"]
        self._abort_recovery(safety_fault=True)
        self.pending_path = None
        self.latest_path_key = None
        self.path_invalidated = True
        if self._context_is_current(context) and not self.navigation_suspended:
            self.publish_status("RECOVERY_FAILED", reason, context=context)
        self._send_pending_path()

    def _hold_unconfirmed_backup(self, reason, *, request_unresolved=False):
        """Fail closed when an action future cannot prove BackUp has stopped."""
        context = (getattr(self, "backup_action_context", None)
                   or self._goal_context())
        if request_unresolved:
            # The server may have accepted the request despite losing its
            # response. Do not dispatch any later FollowPath without proof.
            self.backup_goal_request_pending = True
        if self.backup_cancel_pending_since is None:
            self.backup_cancel_pending_since = time.monotonic()
        self.backup_cancel_safety_fault = True
        if self.recovery_request is not None:
            self._fail_recovery(reason)
        else:
            self.publish_status("RECOVERY_FAILED", reason, context=context)
        self.cancel_navigation(Trigger.Request(), Trigger.Response())

    def _source_is_fresh(self, message, arrival):
        timeout = float(self.get_parameter("recovery_sensor_timeout").value)
        if message is None or arrival is None or time.monotonic() - arrival >= timeout:
            return False
        stamp = message.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        now = self.get_clock().now().to_msg()
        now_ns = now.sec * 1_000_000_000 + now.nanosec
        age = (now_ns - stamp_ns) / 1e9
        return stamp_ns > 0 and -0.25 <= age < timeout

    @staticmethod
    def _rear_clearance(scan, required, half_angle):
        """Require LiDAR coverage across the entire rear cone, not just one ray."""
        if (not scan.ranges or not isfinite(scan.angle_min)
                or not isfinite(scan.angle_increment)
                or scan.angle_increment == 0 or scan.range_max < required
                or not isfinite(scan.range_max) or not isfinite(scan.range_min)
                or scan.range_min < 0 or scan.range_min > scan.range_max
                or not isfinite(half_angle)
                or not 0 < half_angle < pi / 2):
            return False, 0.0
        bins = [0, 0, 0]
        nearest = float("inf")
        cone_beams = 0
        valid_beams = 0
        for index, value in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            relative = atan2(sin(angle - pi), cos(angle - pi))
            if abs(relative) > half_angle:
                continue
            cone_beams += 1
            if isfinite(value):
                if value < scan.range_min:
                    return False, value
                distance = min(value, scan.range_max)
            elif value == float("inf"):
                distance = scan.range_max
            else:
                # NaN and -inf do not prove that the swept rear cone is free.
                # A narrow obstacle can occupy fewer than 20% of the beams,
                # so the aggregate coverage test alone is insufficient.
                return False, 0.0
            bin_index = min(2, int((relative + half_angle) * 3 / (2 * half_angle)))
            bins[bin_index] += 1
            valid_beams += 1
            nearest = min(nearest, distance)
        coverage = valid_beams / cone_beams if cone_beams else 0.0
        return (coverage >= 0.8 and all(count >= 2 for count in bins)
                and nearest > required), nearest

    def _recovery_preflight(self, request, *, require_stationary):
        if getattr(self, 'navigation_inputs_required', False):
            issue = self._navigation_input_error()
            if issue is not None:
                return 'Navigation input unavailable: '+issue
        status = self.motion_status
        if (status is None or self.motion_status_at is None
                or time.monotonic() - self.motion_status_at >= 1.0
                or status.get("mode") != "auto" or status.get("available") is not True
                or status.get("estop_latched") is True):
            return "Automatic motion is not available"
        if not self._source_is_fresh(self.latest_scan, self.latest_scan_at):
            return "Rear LiDAR scan is stale or missing"
        if not self._source_is_fresh(self.latest_odom, self.latest_odom_at):
            return "Odometry is stale or missing"
        if self.latest_scan.header.frame_id != self.get_parameter("recovery_scan_frame").value:
            return "LiDAR frame differs from the verified base-aligned rear scan frame"
        radius = float(self.get_parameter("recovery_robot_radius").value)
        margin = float(self.get_parameter("recovery_rear_margin").value)
        if not isfinite(radius) or radius <= 0 or not isfinite(margin) or margin < 0:
            return "Recovery footprint radius or rear margin is invalid"
        remaining = max(0.0, request["distance"] - request["traveled"])
        required = remaining + radius + margin
        half_angle = radians(float(self.get_parameter("recovery_rear_half_angle_deg").value))
        clear, nearest = self._rear_clearance(self.latest_scan, required, half_angle)
        if not clear:
            return f"Rear LiDAR clearance {nearest:.2f} m is below {required:.2f} m"
        if require_stationary:
            twist = self.latest_odom.twist.twist
            if abs(twist.linear.x) >= 0.025 or abs(twist.angular.z) >= 0.06:
                return "WAIT_STATIONARY"
        return None

    def _try_start_recovery(self):
        request = self.recovery_request
        if request is None or self.navigation_suspended:
            return
        if (self.active_goal is not None or self.goal_request_pending
                or self.initial_alignment is not None
                or self.backup_goal is not None or self.backup_goal_request_pending):
            return
        if not self._context_is_current(request["context"]):
            self._abort_recovery()
            return
        if request["settle_deadline"] is None:
            request["settle_deadline"] = (time.monotonic()
                                          + float(self.get_parameter("recovery_settle_timeout").value))
        reason = self._recovery_preflight(request, require_stationary=True)
        if reason is None:
            reason = self._recovery_start_error(request)
        if reason in {'WAIT_CERTIFIED_ODOM', 'WAIT_CERTIFIED_TF'} and time.monotonic() < request['settle_deadline']:
            return
        if reason == "WAIT_STATIONARY" and time.monotonic() < request["settle_deadline"]:
            return
        if reason is not None:
            self._fail_recovery("Robot did not stop before reverse" if reason == "WAIT_STATIONARY" else reason)
            return
        if not self.backup_client.wait_for_server(timeout_sec=0.2):
            self._fail_recovery("Nav2 BackUp action is unavailable")
            return
        # Server discovery may have waited. Its success cannot refresh the
        # source times or the certified chassis pose before the action send.
        reason = self._recovery_start_error(request)
        if reason is not None:
            self._fail_recovery(reason)
            return
        goal = BackUp.Goal()
        goal.target.x = -request["distance"]
        goal.speed = request["speed"]
        allowance = request["distance"] / request["speed"] + 2.0
        goal.time_allowance.sec = int(allowance)
        goal.time_allowance.nanosec = int((allowance - int(allowance)) * 1e9)
        self.backup_goal_request_pending = True
        request["started_at"] = time.monotonic()
        sequence = request["seq"]
        try:
            future = self.backup_client.send_goal_async(
                goal, feedback_callback=lambda feedback, sent=sequence:
                self._on_backup_feedback(feedback, sent))
        except Exception as exc:
            self.backup_goal_request_pending = False
            self._fail_recovery(f"Nav2 BackUp request failed: {exc}")
            return
        future.add_done_callback(lambda response, sent=sequence:
                                 self._on_backup_goal_response(response, sent))
        self.publish_status("RECOVERING", "Nav2 collision-checked reverse started",
                            context=request["context"])

    def _on_backup_feedback(self, feedback, sequence):
        request = self.recovery_request
        if request is None or request["seq"] != sequence:
            return
        try:
            traveled = float(feedback.feedback.distance_traveled)
        except (AttributeError, TypeError, ValueError):
            return
        if isfinite(traveled):
            request["traveled"] = max(request["traveled"], max(0.0, traveled))

    def _on_backup_goal_response(self, future, sequence):
        try:
            handle = future.result()
            accepted = handle.accepted
        except Exception as exc:
            self._hold_unconfirmed_backup(
                f"Nav2 BackUp acceptance is unconfirmed: {exc}",
                request_unresolved=True,
            )
            return
        self.backup_goal_request_pending = False
        if not accepted:
            replan_required = self.backup_cancel_replan_required
            self._clear_backup_cancel_watch()
            if self.recovery_request is not None and self.recovery_request["seq"] == sequence:
                self._fail_recovery("Nav2 rejected BackUp request")
            elif replan_required:
                self._notify_backup_stopped_after_cancel(
                    "Nav2 rejected BackUp; plan from current pose")
            self._send_pending_path()
            return
        self.backup_goal = handle
        if (self.backup_cancel_when_accepted or self.recovery_request is None
                or self.recovery_request["seq"] != sequence
                or not self._context_is_current(self.recovery_request["context"])):
            self._cancel_backup_goal()
        self.backup_cancel_when_accepted = False
        try:
            handle.get_result_async().add_done_callback(
                lambda result, sent=sequence, accepted_handle=handle:
                self._on_backup_result(result, sent, accepted_handle))
        except Exception as exc:
            self._hold_unconfirmed_backup(f"Nav2 BackUp result is unavailable: {exc}")

    def _on_backup_result(self, future, sequence, handle):
        from action_msgs.msg import GoalStatus
        try:
            status = future.result().status
        except Exception as exc:
            self._hold_unconfirmed_backup(f"Nav2 BackUp result is unconfirmed: {exc}")
            return
        if status not in (GoalStatus.STATUS_SUCCEEDED,
                          GoalStatus.STATUS_CANCELED,
                          GoalStatus.STATUS_ABORTED):
            self._hold_unconfirmed_backup(
                f"Nav2 BackUp returned nonterminal status {status}")
            return
        confirmed_cancel = False
        if self.backup_goal is handle:
            self.backup_goal = None
            self.backup_cancel_requested = False
            confirmed_cancel = self.backup_cancel_replan_required
            if confirmed_cancel:
                self._notify_backup_stopped_after_cancel(
                    "BackUp stopped; plan from current pose")
            self._clear_backup_cancel_watch()
        request = self.recovery_request
        if request is not None and request["seq"] == sequence:
            self.recovery_request = None
            self.latest_path_key = None
            if status == GoalStatus.STATUS_SUCCEEDED:
                # The planner must calculate a new route from the new pose.
                self.pending_path = None
                self.publish_status("RECOVERY_SUCCEEDED", "Checked reverse completed; replan retained goal",
                                    context=request["context"])
            else:
                self.publish_status("RECOVERY_FAILED", f"Nav2 BackUp ended with status {status}",
                                    context=request["context"])
        self._send_pending_path()

    def _monitor_recovery(self):
        self._monitor_follow_path()
        self._monitor_backup_cancellation()
        pending = getattr(self, "pending_recovery_request", None)
        if (pending is not None and time.monotonic() - pending["received_at"]
                > self.RECOVERY_REQUEST_BUFFER_SECONDS):
            self.pending_recovery_request = None
        canceled = self.pending_recovery_cancel
        if (canceled is not None and time.monotonic() - canceled["received_at"]
                > self.RECOVERY_REQUEST_BUFFER_SECONDS):
            self.pending_recovery_cancel = None
        request = self.recovery_request
        if request is None:
            return
        if (request["started_at"] is None
                and time.monotonic() - request["requested_at"]
                > min(5.0, float(self.get_parameter("recovery_cancel_timeout").value))):
            self._fail_recovery("Timed out waiting for FollowPath cancellation and BackUp start")
            return
        if self.backup_goal_request_pending or self.backup_goal is not None:
            reason = self._recovery_preflight(request, require_stationary=False)
            if reason is not None:
                self._fail_recovery(reason)
            elif (request["started_at"] is not None and time.monotonic() - request["started_at"]
                  > min(15.0, float(self.get_parameter("recovery_max_time").value))):
                self._fail_recovery("Nav2 BackUp exceeded the recovery time limit")
        else:
            self._try_start_recovery()

    def _goal_context(self, path=None):
        goal = self.requested_goal
        if goal is None and path is not None and path.poses:
            goal = PoseStamped()
            goal.header = path.header
            goal.pose = path.poses[-1].pose
        payload = {"generation": self.goal_generation, "goal": None, "goal_stamp": None}
        if goal is not None:
            payload["goal"] = {"x": goal.pose.position.x, "y": goal.pose.position.y,
                               "frame_id": goal.header.frame_id or "map"}
            payload["goal_stamp"] = {"sec": goal.header.stamp.sec,
                                     "nanosec": goal.header.stamp.nanosec}
        if path is not None:
            payload["path_stamp"] = {"sec": path.header.stamp.sec,
                                     "nanosec": path.header.stamp.nanosec}
        return payload

    @staticmethod
    def _valid_episode_id(value):
        return isinstance(value, str) and 1 <= len(value) <= 128

    @staticmethod
    def _same_recovery_episode(first, second):
        # The ID is optional for pre-episode external clients. All current
        # planner requests include it, so different nonempty IDs never match.
        left, right = first.get("episode_id"), second.get("episode_id")
        return left is None or right is None or left == right

    def _recovery_context(self, payload):
        context = self._goal_context()
        episode_id = payload.get("episode_id")
        if self._valid_episode_id(episode_id):
            context["episode_id"] = episode_id
        return context

    def _context_is_current(self, context):
        return context.get("generation") == self.goal_generation

    def publish_status(self, status, message_text, context=None, manual_stop=False, **metrics):
        context = self._goal_context() if context is None else context
        message = String()
        message.data = json.dumps({"status": status, "message": message_text,
                                   "goal": context.get("goal"),
                                   "goal_stamp": context.get("goal_stamp"),
                                   "path_stamp": context.get("path_stamp"),
                                   "episode_id": context.get("episode_id"),
                                   "manual_stop": manual_stop,
                                   "stop_stamp": ({"sec": self.stop_stamp_ns // 1_000_000_000,
                                                   "nanosec": self.stop_stamp_ns % 1_000_000_000}
                                                  if self.stop_stamp_ns is not None else None),
                                   **metrics})
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
        if rclpy.ok():
            rclpy.shutdown()
