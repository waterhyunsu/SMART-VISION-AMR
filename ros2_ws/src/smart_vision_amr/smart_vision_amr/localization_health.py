"""Continuously compare localization confidence with independent scan geometry.

The node never publishes an initial pose or a velocity command. Critical
faults use the common motion mux stop, with an explicit operator reset required
after recovery. Scan projection uses the measurement-time transform.
"""

import json
import math
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import SetParametersAtomically
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener

from smart_vision_amr.ekf_sensor_guard import odometry_issues
from smart_vision_amr.localization_geometry import (
    occupied_distance, scan_map_agreement, scan_points,
)
from smart_vision_amr.localization_health_math import (
    LocalizationHealthCore, angle_difference, quaternion_yaw,
)
from smart_vision_amr.sensor_stamp_guard import SensorStampGuard


def stamp_ns(header):
    return header.stamp.sec*1_000_000_000+header.stamp.nanosec


def stamp_rejection_reason(guard, stamp, now_ns):
    if stamp <= 0:
        return 'zero_or_invalid_header'
    if now_ns <= 0:
        return 'clock_not_ready'
    if guard.last_stamp_ns is not None and stamp <= guard.last_stamp_ns:
        return 'duplicate_or_reordered_header'
    age = now_ns-stamp
    if age > guard.max_age_ns:
        return 'stale_header'
    if age < -guard.future_tolerance_ns:
        return 'future_header_or_clock_skew'
    return 'timestamp_rejected'


class HealthTransformListener(TransformListener):
    """Health-only TF QoS and ingress observations, without changing TF data.

    Static TF keeps TransformListener's reliable/transient-local default.
    Dynamic receipt diagnostics never satisfy a pose or measurement-age guard;
    the unchanged TF buffer lookup remains the sole transform input to checks.
    """

    def __init__(self, buffer, node, *, best_effort=True, map_frame='map',
                 odom_frame='odom', base_frame='base_footprint', clock=None):
        if type(best_effort) is not bool:
            raise ValueError('dynamic_tf_best_effort must be boolean')
        self.ingress_clock = clock or time.monotonic
        self.dynamic_best_effort = best_effort
        self.dynamic_received_count = self.dynamic_transform_count = 0
        self.dynamic_received_at = self.dynamic_latest_stamp = None
        self.dynamic_receive_interval = None
        self.dynamic_links = {
            'map_odom': (map_frame.lstrip('/'), odom_frame.lstrip('/')),
            'odom_base': (odom_frame.lstrip('/'), base_frame.lstrip('/'))}
        self.dynamic_link_samples = {}
        options = {}
        if best_effort:
            options['qos'] = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                                        durability=DurabilityPolicy.VOLATILE)
        # Omitting qos when disabled restores the installed Humble listener's
        # original reliable depth-100 profile. Never override static_qos.
        super().__init__(buffer, node, **options)

    def callback(self, data):
        now = self.ingress_clock()
        self.dynamic_receive_interval = (None if self.dynamic_received_at is None
                                         else now-self.dynamic_received_at)
        self.dynamic_received_at = now
        self.dynamic_received_count += 1
        self.dynamic_transform_count += len(data.transforms)
        for transform in data.transforms:
            stamp = stamp_ns(transform.header)
            self.dynamic_latest_stamp = (stamp if self.dynamic_latest_stamp is None
                                         else max(self.dynamic_latest_stamp, stamp))
            pair = (transform.header.frame_id.lstrip('/'), transform.child_frame_id.lstrip('/'))
            for name, frames in self.dynamic_links.items():
                if pair == frames:
                    previous = self.dynamic_link_samples.get(name)
                    self.dynamic_link_samples[name] = {
                        'stamp': stamp, 'received': now,
                        'count': 1 if previous is None else previous['count']+1,
                        'interval': None if previous is None else now-previous['received']}
        super().callback(data)

    def diagnostics(self, now, ros_ns):
        result = {
            'dynamic_tf_qos': 'BEST_EFFORT' if self.dynamic_best_effort else 'RELIABLE',
            'dynamic_tf_received_count': self.dynamic_received_count,
            'dynamic_tf_transform_count': self.dynamic_transform_count,
            'dynamic_tf_receive_age_seconds': (None if self.dynamic_received_at is None
                                                else now-self.dynamic_received_at),
            'dynamic_tf_receive_interval_seconds': self.dynamic_receive_interval,
            'dynamic_tf_latest_header_age_seconds': (None if self.dynamic_latest_stamp is None
                                                       else (ros_ns-self.dynamic_latest_stamp)/1e9)}
        for name in self.dynamic_links:
            sample = self.dynamic_link_samples.get(name)
            result.update({
                f'dynamic_tf_{name}_count': 0 if sample is None else sample['count'],
                f'dynamic_tf_{name}_header_age_seconds': (None if sample is None
                                                           else (ros_ns-sample['stamp'])/1e9),
                f'dynamic_tf_{name}_receive_age_seconds': (None if sample is None
                                                            else now-sample['received']),
                f'dynamic_tf_{name}_receive_interval_seconds': (None if sample is None else sample['interval'])})
        return result


class LocalizationHealth(Node):
    SAFETY_SOURCE = 'localization_safety'

    def __init__(self, *, context=None):
        super().__init__('localization_health', context=context)
        defaults = {
            'stable_seconds': 2.0, 'fault_seconds': 0.5,
            'spatial_fault_seconds': 1.25, 'sensor_timeout': 0.5,
            'tf_timeout': 1.0, 'future_tolerance': 1.5,
            'position_warning': 0.35, 'position_acquisition_limit': 0.50,
            'position_critical': 0.8,
            'yaw_warning': 0.5, 'yaw_critical': 1.0,
            'agreement_warning': 0.55, 'agreement_critical': 0.20,
            'amcl_moving_timeout': 4.0, 'amcl_pose_timeout': 8.0,
            'motion_dt_limit': 1.0,
            'mapping_mode': False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.core = LocalizationHealthCore(**{
            name: (bool(self.get_parameter(name).value) if name == 'mapping_mode'
                   else float(self.get_parameter(name).value)) for name in defaults})
        for name, value in {'base_frame': 'base_footprint', 'map_frame': 'map',
                            'odom_frame': 'odom', 'agreement_tolerance': 0.18,
                            'jump_distance_m': 0.65, 'jump_yaw_rad': 0.8,
                            'nomotion_update_interval': 2.0,
                            'nomotion_update_trigger_age': 1.0,
                            'nomotion_request_timeout': 1.0,
                            'stationary_refresh_max_requests': 10,
                            'stationary_refresh_timeout': 30.0,
                            'agreement_tf_wait_sec': 0.20,
                            'agreement_tf_queue_size': 10,
                            'dynamic_tf_best_effort': True}.items():
            self.declare_parameter(name, value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = HealthTransformListener(self.tf_buffer, self,
            best_effort=self.get_parameter('dynamic_tf_best_effort').value,
            map_frame=self.map_frame, odom_frame=self.odom_frame, base_frame=self.base_frame)
        self.map_metadata = self.map_distance = None
        self.scan = self.odom = None
        self.scan_at = self.odom_at = self.amcl_at = None
        self.scan_valid = self.odom_valid = False
        self.last_received_at = {'scan': None, 'odom': None}
        self.last_received_stamp = {'scan': None, 'odom': None}
        self.received_intervals = {'scan': None, 'odom': None}
        self.header_intervals = {'scan': None, 'odom': None}
        self.rejected_count = {'scan': 0, 'odom': 0}
        self.rejection_reason = {'scan': '', 'odom': ''}
        self.last_rejection_reason = {'scan': '', 'odom': ''}
        self.position_std = self.yaw_std = None
        self.last_amcl_stamp_ns = None
        self.previous_amcl = None
        self.jump_until = 0.0
        self.agreement = self.agreement_at = self.agreement_stamp = None
        self.agreement_scan = self.agreement_scan_at = None
        self.stationary_refresh_scan_stamp = None
        self.agreement_pending = deque()
        self.agreement_tf_state = 'WAITING_FOR_SCAN'
        self.agreement_tf_error = ''
        self.agreement_tf_wait_ms = None
        self.agreement_tf_drop_count = 0
        self.last_reset_seq = {}
        self.last_manual_confirmed_stamp_ns = 0
        self.last_manual_confirmed_amcl_stamp_ns = 0
        self.safety_seq = 0
        self.last_estop_at = None
        self.last_level = None
        self.no_motion_client = self.create_client(Empty, '/request_nomotion_update')
        # AMCL's SetInitialPose service does not publish /initialpose. The web
        # bridge confirms the resulting posterior here, after its own pose/TF
        # checks, so a stopped manual pose can start a fresh acquisition too.
        self.create_service(SetParametersAtomically,
            '/smart_vision_amr/localization_health/confirm_manual_pose',
            self.confirm_manual_pose)
        self.no_motion_future = None
        self.no_motion_started_at = None
        self.last_no_motion_at = None
        self.no_motion_requests = self.no_motion_responses = self.no_motion_timeouts = 0
        self.no_motion_errors = 0
        self.stationary_refresh_started = None
        self.stationary_refresh_requests = 0
        self.last_refresh_scan_stamp = None
        self.refresh_mode, self.refresh_state = 'idle', 'WAITING_FOR_INPUTS'
        self.stamps = {name: SensorStampGuard(self.core.sensor_timeout, 0.25)
                       for name in ('scan', 'odom')}
        durable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(
            String, '/smart_vision_amr/localization_health', durable)
        self.control_pub = self.create_publisher(String, '/smart_vision_amr/manual_control', 10)
        # During SLAM, every changing /map would trigger a full distance
        # transform on the Pi without contributing to manual safety.
        if not self.core.mapping_mode:
            self.create_subscription(OccupancyGrid, '/map', self.on_map, durable)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        if not self.core.mapping_mode:
            self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.on_amcl, durable)
            self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.on_initialpose, 10)
        # Reset is a one-shot intent, not a velocity lease. Keep it through a
        # short callback backlog; source/sequence/stamp guards still reject old
        # packets. Velocity consumers retain their latest-sample QoS.
        safety_intents = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                   durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(String, '/smart_vision_amr/manual_control',
                                 self.on_control, safety_intents)
        self.create_timer(0.25, self.on_tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        if not self.core.mapping_mode:
            self.create_timer(0.05, lambda: self.update_agreement(time.monotonic()),
                              clock=Clock(clock_type=ClockType.STEADY_TIME))

    def on_map(self, message):
        info = message.info
        self.agreement_scan = self.agreement_scan_at = None
        self.stationary_refresh_scan_stamp = None
        try:
            if (message.header.frame_id != self.map_frame or info.resolution <= 0.0
                    or not math.isfinite(info.resolution)
                    or info.width*info.height != len(message.data)):
                raise ValueError('invalid map dimensions/frame')
            yaw = quaternion_yaw(info.origin.orientation)
            if not all(math.isfinite(value) for value in
                       (info.origin.position.x, info.origin.position.y)):
                raise ValueError('invalid map origin')
            self.map_metadata = {
                'width': info.width, 'height': info.height, 'resolution': info.resolution,
                'origin': {'x': info.origin.position.x, 'y': info.origin.position.y, 'yaw': yaw}}
            data = np.asarray(message.data, dtype=np.int16).reshape(info.height, info.width)
            self.map_distance = occupied_distance(data, info.resolution)
            self.agreement = self.agreement_at = self.agreement_stamp = None
            if hasattr(self, 'agreement_pending'):
                self.agreement_pending.clear()
        except (ValueError, TypeError):
            self.map_metadata = self.map_distance = None

    def on_scan(self, message):
        now, ros_ns = time.monotonic(), self.get_clock().now().nanoseconds
        stamp = stamp_ns(message.header)
        LocalizationHealth._record_receive_timing(self, 'scan', now, stamp)
        self.last_received_at['scan'], self.last_received_stamp['scan'] = now, stamp
        values = np.asarray(message.ranges, dtype=float)
        finite = np.isfinite(values) & (values >= message.range_min) & (values <= message.range_max)
        structurally_valid = (values.ndim == 1 and len(values) >= 12
                           and np.count_nonzero(finite) >= 12
                           and math.isfinite(message.angle_min)
                           and math.isfinite(message.angle_increment)
                           and message.angle_increment != 0.0)
        if not structurally_valid:
            self.scan_valid = False
            self.reject('scan', 'invalid_scan_values')
        elif not self.stamps['scan'].accept(stamp, ros_ns):
            # An extra replay must not flip a good fresh scan invalid; it also
            # must not refresh its receive age. The last good sample ages out.
            self.reject('scan', stamp_rejection_reason(self.stamps['scan'], stamp, ros_ns))
        else:
            self.scan_valid = True
            self.scan, self.scan_at = message, now
            self.rejection_reason['scan'] = ''
            if hasattr(self, 'agreement_pending'):
                bound = int(LocalizationHealth._refresh_setting(self, 'agreement_tf_queue_size', 10))
                bound = max(1, min(50, bound))
                while len(self.agreement_pending) >= bound:
                    self.agreement_pending.popleft()
                    self.agreement_tf_drop_count += 1
                self.agreement_pending.append((stamp, now, message))

    def on_odom(self, message):
        now, ros_ns = time.monotonic(), self.get_clock().now().nanoseconds
        stamp = stamp_ns(message.header)
        LocalizationHealth._record_receive_timing(self, 'odom', now, stamp)
        self.last_received_at['odom'], self.last_received_stamp['odom'] = now, stamp
        issues = odometry_issues(message, base_frame=self.base_frame)
        if issues:
            self.odom_valid = False
            self.reject('odom', '; '.join(issues))
        elif not self.stamps['odom'].accept(stamp, ros_ns):
            self.reject('odom', stamp_rejection_reason(self.stamps['odom'], stamp, ros_ns))
        else:
            self.odom_valid = True
            self.odom, self.odom_at = message, now
            self.rejection_reason['odom'] = ''

    def reject(self, name, reason):
        self.rejected_count[name] += 1
        self.rejection_reason[name] = self.last_rejection_reason[name] = str(reason)

    @staticmethod
    def _record_receive_timing(node, name, now, stamp):
        """Keep receipt spacing separate from source clock spacing; never retime data."""
        received = node.last_received_at[name]
        previous = node.last_received_stamp[name]
        if hasattr(node, 'received_intervals'):
            node.received_intervals[name] = None if received is None else now-received
            node.header_intervals[name] = None if previous is None else (stamp-previous)/1e9

    def on_amcl(self, message):
        now = time.monotonic()
        stamp = stamp_ns(message.header)
        if stamp <= 0 or (self.last_amcl_stamp_ns is not None and stamp <= self.last_amcl_stamp_ns):
            return
        try:
            covariance = message.pose.covariance
            if (message.header.frame_id != self.map_frame
                    or any(not math.isfinite(value) for value in covariance)
                    or any(covariance[index] < 0.0 for index in (0, 7, 35))):
                raise ValueError('invalid AMCL covariance/frame')
            pose = message.pose.pose
            yaw = quaternion_yaw(pose.orientation)
            if not all(math.isfinite(value) for value in (pose.position.x, pose.position.y)):
                raise ValueError('invalid AMCL position')
            xy = (covariance[1]+covariance[6])/2.0
            if covariance[0]*covariance[7]-xy*xy < -1e-9:
                raise ValueError('nonphysical AMCL position covariance')
            # Largest principal-axis uncertainty includes correlated x/y error.
            largest = (covariance[0]+covariance[7]+math.sqrt(
                (covariance[0]-covariance[7])**2+4.0*xy*xy))/2.0
            self.position_std = math.sqrt(largest)
            self.yaw_std = math.sqrt(covariance[35])
            # Check correction size against independently integrated odom.
            # Initial acquisition/initialpose has no previous baseline.
            if self.odom is not None:
                odom_pose = self.odom.pose.pose
                current = (pose.position.x, pose.position.y, yaw,
                           odom_pose.position.x, odom_pose.position.y,
                           quaternion_yaw(odom_pose.orientation))
                if self.previous_amcl is not None:
                    previous = self.previous_amcl
                    map_delta = math.hypot(current[0]-previous[0], current[1]-previous[1])
                    odom_delta = math.hypot(current[3]-previous[3], current[4]-previous[4])
                    yaw_residual = abs(angle_difference(
                        angle_difference(current[2], previous[2]),
                        angle_difference(current[5], previous[5])))
                    if (abs(map_delta-odom_delta) > float(self.get_parameter('jump_distance_m').value)
                            or yaw_residual > float(self.get_parameter('jump_yaw_rad').value)):
                        self.jump_until = now+max(2.0, 2.0*self.core.fault_seconds)
                self.previous_amcl = current
            self.amcl_at = now
            self.last_amcl_stamp_ns = stamp
            # A new, validated AMCL pose ends the stationary refresh episode.
            # A service response or an otherwise healthy safety tick does not.
            self.stationary_refresh_started = None
            self.stationary_refresh_requests = 0
        except (ValueError, TypeError):
            self.position_std = self.yaw_std = None

    def on_initialpose(self, _message):
        self.previous_amcl = None
        self.position_std = self.yaw_std = self.amcl_at = None
        self.agreement = self.agreement_at = self.agreement_stamp = None
        self.agreement_scan = self.agreement_scan_at = None
        self.stationary_refresh_scan_stamp = None
        self.jump_until = 0.0
        if hasattr(self, 'agreement_pending'):
            self.agreement_pending.clear()
        self.stationary_refresh_started = None
        self.stationary_refresh_requests = 0
        self.last_refresh_scan_stamp = None
        self.refresh_mode, self.refresh_state = 'idle', 'WAITING_FOR_INPUTS'
        self.core.reinitialize()

    def confirm_manual_pose(self, request, response):
        """Acknowledge one *observed* AMCL posterior, never the pose request.

        The dedicated service carries two ROS timestamps as integer fields.
        A delayed/duplicated request cannot restart acquisition, and a service
        receipt never releases the existing stop latch or authorizes driving.
        """
        result = response.result
        result.successful = False
        result.reason = 'invalid_request'
        parameters = request.parameters
        names = {'initial_pose_stamp_ns', 'verified_amcl_stamp_ns'}
        if (len(parameters) != 2 or {item.name for item in parameters} != names
                or any(item.value.type != ParameterType.PARAMETER_INTEGER for item in parameters)):
            return response
        values = {item.name: item.value.integer_value for item in parameters}
        initial = values['initial_pose_stamp_ns']
        verified = values['verified_amcl_stamp_ns']
        now_ros = self.get_clock().now().nanoseconds
        now = time.monotonic()
        if initial <= 0 or verified <= initial:
            result.reason = 'invalid_or_stale_stamp'
            return response
        if (initial == self.last_manual_confirmed_stamp_ns
                and verified == self.last_manual_confirmed_amcl_stamp_ns):
            # The first response may be lost after this node has already
            # reinitialized. Re-acknowledge only the exact accepted request.
            result.successful = True
            result.reason = 'already_confirmed_stop_latch_retained'
            return response
        amcl_header_age = (now_ros-verified)/1e9
        if (initial <= self.last_manual_confirmed_stamp_ns or now_ros <= 0
                or initial > now_ros+int(self.core.future_tolerance*1e9)
                or not -self.core.future_tolerance <= amcl_header_age <= self.core.amcl_pose_timeout):
            result.reason = 'invalid_or_stale_stamp'
            return response
        if self.last_amcl_stamp_ns is None or self.last_amcl_stamp_ns < verified:
            result.reason = 'amcl_not_seen_yet'
            return response
        if (self.amcl_at is None
                or not 0. <= now-self.amcl_at <= self.core.amcl_pose_timeout
                or self.position_std is None or self.yaw_std is None
                or not math.isfinite(self.position_std) or not math.isfinite(self.yaw_std)):
            result.reason = 'amcl_not_fresh'
            return response
        if (self.map_distance is None or self.scan is None or self.odom is None
                or not self.scan_valid or not self.odom_valid
                or self.scan_at is None or self.odom_at is None
                or not 0. <= now-self.scan_at <= self.core.sensor_timeout
                or not 0. <= now-self.odom_at <= self.core.sensor_timeout):
            result.reason = 'sensor_not_fresh'
            return response
        for sample in (self.scan, self.odom):
            age = (now_ros-stamp_ns(sample.header))/1e9
            if not -.25 <= age <= self.core.sensor_timeout:
                result.reason = 'sensor_stamp_not_fresh'
                return response
        twist = self.odom.twist.twist
        if (not math.isfinite(twist.linear.x) or not math.isfinite(twist.angular.z)
                or abs(twist.linear.x) >= .01 or abs(twist.angular.z) >= .03):
            result.reason = 'robot_not_stopped'
            return response
        if amcl_header_age > 2.:
            # A stationary AMCL posterior need not refresh on every laser scan.
            # Permit the same bounded age as the normal health watchdog only
            # when a *new* scan still agrees with the saved map. Service
            # receipt alone is never proof of a current pose or safe motion.
            if (self.agreement is None or not math.isfinite(self.agreement)
                    or self.agreement_at is None
                    or not 0. <= now-self.agreement_at <= self.core.sensor_timeout
                    or self.agreement_stamp is None or self.agreement_stamp <= initial):
                result.reason = 'scan_map_not_fresh'
                return response
        try:
            for frame in (self.map_frame, self.odom_frame):
                transform = self._tf(frame, self.base_frame)
                quaternion_yaw(transform.transform.rotation)
                if (transform.header.frame_id.lstrip('/') != frame.lstrip('/')
                        or transform.child_frame_id.lstrip('/') != self.base_frame.lstrip('/')
                        or not all(math.isfinite(value) for value in (
                            transform.transform.translation.x,
                            transform.transform.translation.y))):
                    raise ValueError('invalid TF frame or pose')
                stamp = stamp_ns(transform.header)
                age = (now_ros-stamp)/1e9
                if not -self.core.future_tolerance <= age <= self.core.tf_timeout:
                    raise ValueError('stale TF')
                if frame == self.map_frame and stamp <= initial:
                    raise ValueError('map TF predates initial pose')
        except (TransformException, ValueError, AttributeError):
            result.reason = 'tf_not_fresh'
            return response
        self.core.reinitialize()
        # Keep the newly verified posterior: clearing covariance/amcl_at here
        # can deadlock a stationary robot waiting for another laser update.
        self.previous_amcl = None
        self.jump_until = 0.
        self.agreement = self.agreement_at = self.agreement_stamp = None
        self.agreement_scan = self.agreement_scan_at = None
        self.agreement_pending.clear()
        self.last_manual_confirmed_stamp_ns = initial
        self.last_manual_confirmed_amcl_stamp_ns = verified
        result.successful = True
        result.reason = 'confirmed_for_acquisition_stop_latch_retained'
        return response

    def _tf(self, target, source, stamp=None):
        return self.tf_buffer.lookup_transform(
            target, source, rclpy.time.Time() if stamp is None else rclpy.time.Time.from_msg(stamp))

    def update_agreement(self, now):
        if self.scan is None or self.map_distance is None:
            return
        pending = getattr(self, 'agreement_pending', None)
        # Newer scans can continually outrun a delayed TF stream. Keep a small
        # bounded queue and apply the newest measurement whose exact TF exists.
        candidates = (list(pending) if pending is not None else
                      [(stamp_ns(self.scan.header), self.scan_at, self.scan)])
        if not candidates:
            return
        wait = min(.5, max(.02, LocalizationHealth._refresh_setting(
            self, 'agreement_tf_wait_sec', .20)))
        ros_ns = self.get_clock().now().nanoseconds
        keep = []
        applied = None
        for stamp, received, scan in reversed(candidates):
            if self.agreement_stamp is not None and stamp <= self.agreement_stamp:
                continue
            age = (ros_ns-stamp)/1e9
            if (received is None or not 0 <= now-received <= self.core.sensor_timeout
                    or not -.25 <= age <= self.core.sensor_timeout):
                self.agreement_tf_drop_count = getattr(self, 'agreement_tf_drop_count', 0)+1
                self.agreement_tf_state = 'SCAN_STALE_OR_CLOCK_SKEW'
                continue
            # Both transforms correspond to exactly the laser measurement,
            # not the newest pose available when this callback is scheduled.
            try:
                scan_time = scan.header.stamp
                laser_tf = self._tf(self.base_frame, scan.header.frame_id, scan_time)
                robot_tf = self._tf(self.map_frame, self.base_frame, scan_time)
                for transform in (laser_tf, robot_tf):
                    tf_stamp = stamp_ns(transform.header)
                    if tf_stamp and abs(tf_stamp-stamp) > 20_000_000:
                        raise ValueError('transform does not match laser measurement time')
                laser, robot = laser_tf.transform, robot_tf.transform
                if not all(math.isfinite(value) for value in (
                        laser.translation.x, laser.translation.y,
                        robot.translation.x, robot.translation.y)):
                    raise ValueError('nonfinite scan transform')
                points = scan_points(scan, (laser.translation.x, laser.translation.y,
                                           quaternion_yaw(laser.rotation)))
                self.agreement = scan_map_agreement(
                    points, {'x': robot.translation.x, 'y': robot.translation.y,
                             'yaw': quaternion_yaw(robot.rotation)},
                    self.map_metadata, self.map_distance,
                    tolerance=float(self.get_parameter('agreement_tolerance').value))
                self.agreement_at, self.agreement_stamp = now, stamp
                self.agreement_scan, self.agreement_scan_at = scan, received
                self.agreement_tf_state, self.agreement_tf_error = 'READY', ''
                self.agreement_tf_wait_ms = (now-received)*1000.
                applied = stamp
                break
            except (TransformException, ValueError) as exc:
                self.agreement_tf_error = str(exc)[:300]
                if now-received <= wait:
                    keep.append((stamp, received, scan))
                    self.agreement_tf_state = 'WAITING_FOR_MEASUREMENT_TF'
                else:
                    self.agreement_tf_state = 'MEASUREMENT_TF_TIMEOUT'
                    self.agreement_tf_drop_count = getattr(self, 'agreement_tf_drop_count', 0)+1
        if pending is not None:
            # On success discard all older scans; they cannot improve freshness.
            self.agreement_pending = deque(sorted(
                (item for item in keep if applied is None or item[0] > applied), key=lambda item: item[0]))

    def on_tick(self):
        now, ros_ns = time.monotonic(), self.get_clock().now().nanoseconds
        if not self.core.mapping_mode:
            self.request_initial_covariance(now)
            self.update_agreement(now)
        metrics = {
            'map_available': self.map_distance is not None,
            'scan_age_seconds': None if self.scan_at is None else now-self.scan_at,
            'odom_age_seconds': None if self.odom_at is None else now-self.odom_at,
            'scan_header_age_seconds': None if self.scan is None else (ros_ns-stamp_ns(self.scan.header))/1e9,
            'odom_header_age_seconds': None if self.odom is None else (ros_ns-stamp_ns(self.odom.header))/1e9,
            'scan_valid': bool(self.scan_valid), 'odom_valid': bool(self.odom_valid),
            'position_std_m': self.position_std, 'yaw_std_rad': self.yaw_std,
            'amcl_age_seconds': None if self.amcl_at is None else now-self.amcl_at,
            'amcl_update_epoch': self.amcl_at,
            'amcl_header_age_seconds': None if self.last_amcl_stamp_ns is None else (ros_ns-self.last_amcl_stamp_ns)/1e9,
            'moving': self.odom is not None and (
                abs(self.odom.twist.twist.linear.x) > 0.015
                or abs(self.odom.twist.twist.angular.z) > 0.05),
            'pose_jump': now < self.jump_until,
            'scan_map_agreement': self.agreement,
            'agreement_age_seconds': None if self.agreement_at is None else now-self.agreement_at,
            'agreement_header_age_seconds': None if self.agreement_stamp is None else (ros_ns-self.agreement_stamp)/1e9,
            'map_tf_age_seconds': None, 'odom_tf_valid': False, 'odom_tf_age_seconds': None,
            'map_tf_lookup_error': '', 'odom_tf_lookup_error': '',
            'amcl_update_requests': self.no_motion_requests,
            'amcl_update_responses': self.no_motion_responses,
            'amcl_update_timeouts': self.no_motion_timeouts,
            'amcl_update_errors': self.no_motion_errors,
            'scan_tf_state': self.agreement_tf_state,
            'scan_tf_error': self.agreement_tf_error,
            'scan_tf_wait_ms': self.agreement_tf_wait_ms,
            'scan_tf_drop_count': self.agreement_tf_drop_count,
            'scan_tf_pending_count': len(self.agreement_pending),
            'amcl_refresh_mode': self.refresh_mode,
            'amcl_refresh_state': self.refresh_state,
            'amcl_stationary_update_requests': self.stationary_refresh_requests,
        }
        metrics.update(self.tf_listener.diagnostics(now, ros_ns))
        for name in ('odom', 'scan'):
            received, raw_stamp = self.last_received_at[name], self.last_received_stamp[name]
            metrics.update({
                f'{name}_last_received_age_seconds': None if received is None else now-received,
                f'{name}_raw_header_age_seconds': None if raw_stamp is None else (ros_ns-raw_stamp)/1e9,
                f'{name}_rejected_count': self.rejected_count[name],
                f'{name}_rejection_reason': self.rejection_reason[name],
                f'{name}_last_rejection_reason': self.last_rejection_reason[name],
                f'{name}_receive_interval_seconds': self.received_intervals[name],
                f'{name}_header_interval_seconds': self.header_intervals[name],
            })
        if not self.core.mapping_mode:
            try:
                transform = self._tf(self.map_frame, self.base_frame)
                quaternion_yaw(transform.transform.rotation)
                if not all(math.isfinite(value) for value in (
                        transform.transform.translation.x, transform.transform.translation.y)):
                    raise ValueError('nonfinite map TF')
                metrics['map_tf_age_seconds'] = (ros_ns-stamp_ns(transform.header))/1e9
            except (TransformException, ValueError) as exc:
                metrics['map_tf_lookup_error'] = str(exc)[:300]
        try:
            transform = self._tf(self.odom_frame, self.base_frame)
            quaternion_yaw(transform.transform.rotation)
            age = (ros_ns-stamp_ns(transform.header))/1e9
            metrics['odom_tf_age_seconds'] = age
            metrics['odom_tf_valid'] = -0.25 <= age <= self.core.tf_timeout
        except (TransformException, ValueError) as exc:
            metrics['odom_tf_lookup_error'] = str(exc)[:300]
        status = self.core.evaluate(now, metrics)
        status['stamp_ns'] = ros_ns
        status['sent_at_ns'] = ros_ns
        if status['level'] != self.last_level:
            self.get_logger().info(f"Localization health: {status['level']} {status['reasons']}")
            self.last_level = status['level']
        if status['stop_latched'] and (self.last_estop_at is None or now-self.last_estop_at >= 0.5):
            self.publish_estop(now)
        message = String()
        message.data = json.dumps(status, allow_nan=False)
        self.publisher.publish(message)

    def request_initial_covariance(self, now):
        """Request a real filter-only laser update without weakening the guard.

        AMCL may otherwise wait for motion before refreshing covariance, while
        the safety mux correctly waits for covariance before allowing motion.
        Once localized, slow movement can stay below AMCL's update thresholds
        and exhaust the motion watchdog. A healthy but stationary robot can
        also lose its map TF when AMCL stops publishing; request a bounded
        update before that happens. A request/response never refreshes the
        AMCL observation epoch; only ``on_amcl`` may do that.
        """
        setting = lambda name, default: LocalizationHealth._refresh_setting(self, name, default)
        # Never permit a runtime typo to flood AMCL or indefinitely retain a
        # pending request. The separate four-second motion watchdog is unchanged.
        interval = max(2., setting('nomotion_update_interval', 2.))
        trigger_age = max(.1, setting('nomotion_update_trigger_age', 1.))
        deadline = min(1., max(.1, setting('nomotion_request_timeout', 1.)))
        if self.no_motion_future is not None:
            future = self.no_motion_future
            started = getattr(self, 'no_motion_started_at', self.last_no_motion_at)
            if not future.done():
                if started is None:
                    self.no_motion_started_at = now
                    return
                if now-started < deadline:
                    return
                try:
                    future.cancel()
                    self.no_motion_client.remove_pending_request(future)
                except Exception:
                    pass
                self.no_motion_timeouts = getattr(self, 'no_motion_timeouts', 0)+1
                self.refresh_state = 'TIMEOUT'
            elif callable(getattr(future, 'result', None)):
                try:
                    future.result()
                    self.no_motion_responses = getattr(self, 'no_motion_responses', 0)+1
                    self.refresh_state = 'RESPONDED'
                except Exception:
                    self.no_motion_errors = getattr(self, 'no_motion_errors', 0)+1
            self.no_motion_future = None
            self.no_motion_started_at = None
        if (self.scan is None or self.map_distance is None
                or self.scan_at is None or not 0 <= now-self.scan_at <= self.core.sensor_timeout
                or (self.last_no_motion_at is not None and now-self.last_no_motion_at < interval)):
            return
        if not self.no_motion_client.service_is_ready():
            self.refresh_state = 'SERVICE_UNAVAILABLE'
            return
        mode = 'bootstrap'
        if self.position_std is not None:
            if LocalizationHealth._stationary_update_ready(self, now, trigger_age):
                mode = 'stationary'
                maximum = max(1, min(30, int(setting('stationary_refresh_max_requests', 10))))
                timeout = max(2., min(60., setting('stationary_refresh_timeout', 30.)))
                started = getattr(self, 'stationary_refresh_started', None)
                count = getattr(self, 'stationary_refresh_requests', 0)
                if count >= maximum or (started is not None and now-started >= timeout):
                    self.refresh_mode, self.refresh_state = mode, 'STATIONARY_LIMIT_REACHED'
                    return
            elif LocalizationHealth._moving_update_ready(self, now, trigger_age):
                mode = 'moving'
            else:
                self.refresh_state = 'WAITING_FOR_MOTION_OR_ACQUISITION'
                return
        scan_stamp = (self.stationary_refresh_scan_stamp if mode == 'stationary'
                      else stamp_ns(self.scan.header))
        if scan_stamp <= (getattr(self, 'last_refresh_scan_stamp', None) or 0):
            self.refresh_state = 'WAITING_FOR_NEW_SCAN'
            return
        self.last_no_motion_at = now
        try:
            self.no_motion_future = self.no_motion_client.call_async(Empty.Request())
            self.no_motion_started_at = now
            self.no_motion_requests = getattr(self, 'no_motion_requests', 0)+1
            self.last_refresh_scan_stamp = scan_stamp
            self.refresh_mode, self.refresh_state = mode, 'REQUESTED'
            if mode == 'stationary':
                if getattr(self, 'stationary_refresh_started', None) is None:
                    self.stationary_refresh_started = now
                self.stationary_refresh_requests = getattr(self, 'stationary_refresh_requests', 0)+1
        except Exception:
            self.no_motion_future = None
            self.no_motion_started_at = None
            self.no_motion_errors = getattr(self, 'no_motion_errors', 0)+1
            self.refresh_state = 'REQUEST_FAILED'

    @staticmethod
    def _refresh_setting(node, name, default):
        getter = getattr(node, 'get_parameter', None)
        try:
            value = getter(name).value if callable(getter) else default
            return float(value) if not isinstance(value, bool) and math.isfinite(value) and value > 0 else default
        except (AttributeError, KeyError, TypeError, ValueError):
            return default

    def _stationary_update_ready(self, now, trigger_age=1.):
        """Refresh uncertain acquisition from fresh scans without issuing motion.

        A filter service response never edits covariance, confirms a pose, or
        releases a safety latch. Healthy idle tracking requests a bounded
        update only after its real AMCL pose becomes old.
        """
        self.stationary_refresh_scan_stamp = None
        acquiring = not getattr(self.core, 'acquired', True)
        elevated = (self.position_std >= self.core.position_warning
                    or (self.yaw_std is not None and self.yaw_std >= self.core.yaw_warning))
        last = getattr(self.core, 'last_result', None)
        stale_posterior = (isinstance(last, dict)
                           and 'amcl_pose_stale' in last.get('reasons', ()))
        healthy_tracking = (not acquiring and not getattr(self.core, 'stop_latched', False)
                            and isinstance(last, dict) and last.get('level') == 'ok'
                            and last.get('autonomous_allowed') is True)
        if not (acquiring or elevated or healthy_tracking or stale_posterior):
            return False
        if not LocalizationHealth._refresh_evidence_ready(self, now, trigger_age):
            return False
        twist = self.odom.twist.twist
        if abs(twist.linear.x) > .015 or abs(twist.angular.z) > .05:
            return False
        # The newest arrival may repeatedly outrun TF. A recent measurement
        # already checked by the bounded agreement queue is also valid evidence;
        # retain its actual receipt/header and consume that proof's sequence.
        candidates = [(self.scan, self.scan_at),
                      (getattr(self, 'agreement_scan', None),
                       getattr(self, 'agreement_scan_at', None))]
        ros_ns = self.get_clock().now().nanoseconds
        for scan, received in candidates:
            if (scan is None or received is None
                    or not 0 <= now-received <= self.core.sensor_timeout
                    or not -.25 <= (ros_ns-stamp_ns(scan.header))/1e9 <= self.core.sensor_timeout):
                continue
            try:
                for target in (self.base_frame, self.map_frame):
                    source = scan.header.frame_id if target == self.base_frame else self.base_frame
                    stamped = self._tf(target, source, scan.header.stamp)
                    quaternion_yaw(stamped.transform.rotation)
                    if not all(math.isfinite(value) for value in (
                            stamped.transform.translation.x, stamped.transform.translation.y)):
                        raise ValueError('nonfinite measurement transform')
                    tf_stamp = stamp_ns(stamped.header)
                    if tf_stamp and abs(tf_stamp-stamp_ns(scan.header)) > 20_000_000:
                        raise ValueError('transform does not match measurement time')
                self.stationary_refresh_scan_stamp = stamp_ns(scan.header)
                return True
            except (TransformException, ValueError, AttributeError):
                continue
        return False

    def _refresh_evidence_ready(self, now, trigger_age):
        amcl_at, amcl_stamp = getattr(self, 'amcl_at', None), getattr(self, 'last_amcl_stamp_ns', None)
        odom, odom_at = getattr(self, 'odom', None), getattr(self, 'odom_at', None)
        if (amcl_at is None or not 0 <= now-amcl_at or now-amcl_at < trigger_age
                or amcl_stamp is None or odom is None or odom_at is None
                or not 0 <= now-odom_at <= self.core.sensor_timeout
                or not getattr(self, 'scan_valid', False) or not getattr(self, 'odom_valid', False)):
            return False
        ros_ns = self.get_clock().now().nanoseconds
        if (ros_ns <= 0 or ros_ns-amcl_stamp < trigger_age*1e9
                or self.scan_at is None or not 0 <= now-self.scan_at <= self.core.sensor_timeout):
            return False
        for sample in (self.scan, odom):
            if not -.25 <= (ros_ns-stamp_ns(sample.header))/1e9 <= self.core.sensor_timeout:
                return False
        if not all(math.isfinite(value) for value in (
                odom.twist.twist.linear.x, odom.twist.twist.angular.z)):
            return False
        try:
            transform = self._tf(self.odom_frame, self.base_frame)
            quaternion_yaw(transform.transform.rotation)
            age = (ros_ns-stamp_ns(transform.header))/1e9
            if (not -.25 <= age <= self.core.tf_timeout or not all(math.isfinite(value) for value in (
                    transform.transform.translation.x, transform.transform.translation.y))):
                return False
        except (TransformException, ValueError, AttributeError):
            return False
        return True

    def _moving_update_ready(self, now, trigger_age=1.):
        """Only healthy, measured slow motion can request periodic AMCL work."""
        amcl_at, amcl_stamp = getattr(self, 'amcl_at', None), getattr(self, 'last_amcl_stamp_ns', None)
        odom, odom_at = getattr(self, 'odom', None), getattr(self, 'odom_at', None)
        if (amcl_at is None or now-amcl_at < trigger_age or amcl_stamp is None
                or odom is None or odom_at is None or not 0 <= now-odom_at <= self.core.sensor_timeout
                or not getattr(self, 'scan_valid', False) or not getattr(self, 'odom_valid', False)):
            return False
        ros_ns = self.get_clock().now().nanoseconds
        if (ros_ns <= 0 or ros_ns-amcl_stamp < trigger_age*1e9
                or not 0 <= now-self.scan_at <= self.core.sensor_timeout):
            return False
        for sample in (self.scan, odom):
            age = (ros_ns-stamp_ns(sample.header))/1e9
            if not -.25 <= age <= self.core.sensor_timeout:
                return False
        linear, angular = odom.twist.twist.linear.x, odom.twist.twist.angular.z
        if (not math.isfinite(linear) or not math.isfinite(angular)
                or (abs(linear) <= .015 and abs(angular) <= .05)):
            return False
        try:
            for target in (self.map_frame, self.odom_frame):
                transform = self._tf(target, self.base_frame)
                age = (ros_ns-stamp_ns(transform.header))/1e9
                quaternion_yaw(transform.transform.rotation)
                if (not -.25 <= age <= self.core.tf_timeout
                        or not all(math.isfinite(value) for value in (
                            transform.transform.translation.x, transform.transform.translation.y))):
                    return False
        except (TransformException, ValueError, AttributeError):
            return False
        return True

    def publish_estop(self, now):
        self.safety_seq = max(self.safety_seq+1, time.monotonic_ns())
        message = String()
        message.data = json.dumps({
            'source_id': self.SAFETY_SOURCE, 'seq': self.safety_seq,
            'sent_at_ns': self.get_clock().now().nanoseconds,
            'mode': 'estop', 'linear': 0.0, 'angular': 0.0})
        self.control_pub.publish(message)
        self.last_estop_at = now

    def on_control(self, message):
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if not isinstance(payload, dict) or payload.get('mode') != 'reset':
            return
        source, seq, sent = (payload.get(key) for key in ('source_id', 'seq', 'sent_at_ns'))
        if (not isinstance(source, str) or not source or source.endswith('_safety')
                or type(seq) is not int or seq <= self.last_reset_seq.get(source, -1)
                or type(sent) is not int or abs(self.get_clock().now().nanoseconds-sent) > 1e9):
            return
        self.last_reset_seq[source] = seq
        if self.core.stop_latched and not self.core.operator_reset(time.monotonic()):
            self.publish_estop(time.monotonic())


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationHealth()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
