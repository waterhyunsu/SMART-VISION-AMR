"""Latch a hardware-only motion stop when an EKF input stream is lost.

The guard uses both receive time and ROS header freshness. It starts with a
latched estop, because the EKF can publish wheel-only odom before the IMU is
calibrated. It arms only after all required streams have been healthy for a
continuous period and the operator explicitly resets the stop. After a trip,
it re-sends estop until the motion mux acknowledges it. Recovery of sensor
data alone never resets the mux.
"""

import json
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Log, SetParametersResult
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import BatteryState, Imu, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from smart_vision_amr.sensor_stamp_guard import SensorStampGuard
from smart_vision_amr.runtime_timing import GUARD_METRICS, RuntimeTiming, timed_call

ROS_ERROR_LEVEL = Log.ERROR[0] if isinstance(Log.ERROR, bytes) else int(Log.ERROR)


def publisher_issues(get_publishers, expected_publishers):
    """Report any missing, duplicate, or wrong-owner ROS publishers."""
    issues = []
    for topic, expected_node in expected_publishers.items():
        publishers = get_publishers(topic)
        if len(publishers) != 1:
            issues.append(f'{topic} publishers={len(publishers)} (expected 1)')
        elif (expected_node is not None and
              (publishers[0].node_name != expected_node or
               publishers[0].node_namespace != '/')):
            issues.append(f'{topic} publisher is not /{expected_node}')
    return issues


def configured_publishers(default, ekf_publisher, raw_wheel_topic):
    """Retarget only Gazebo's EKF/raw-odom graph names; keep every safety check."""
    if (not isinstance(ekf_publisher, str) or not ekf_publisher or '/' in ekf_publisher
            or not isinstance(raw_wheel_topic, str)
            or not raw_wheel_topic.startswith('/') or raw_wheel_topic == '/'
            or (raw_wheel_topic in default and raw_wheel_topic != '/wheel/odom_raw')):
        raise ValueError('Expected EKF publisher and raw wheel odom topic must be valid')
    expected = default.copy()
    expected['/odom'] = ekf_publisher
    expected.pop('/wheel/odom_raw')
    expected[raw_wheel_topic] = None
    return expected


def odometry_issues(message, *, max_position_variance=0.0,
                    max_yaw_variance=0.0, max_linear_velocity_variance=0.25,
                    max_angular_velocity_variance=1.0, base_frame='base_footprint'):
    """Validate the filter's actual output, not merely its ROS publisher."""
    pose, twist = message.pose.pose, message.twist.twist
    q = pose.orientation
    values = (pose.position.x, pose.position.y, pose.position.z,
              q.x, q.y, q.z, q.w, twist.linear.x, twist.linear.y,
              twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z)
    if not all(math.isfinite(value) for value in values):
        return ['nonfinite odometry value']
    if not 0.95 <= sum(value*value for value in (q.x, q.y, q.z, q.w)) <= 1.05:
        return ['invalid odometry quaternion']
    if message.header.frame_id != 'odom' or message.child_frame_id != base_frame:
        return ['unexpected odometry frames']
    for covariance in (message.pose.covariance, message.twist.covariance):
        if (not all(math.isfinite(value) for value in covariance)
                or any(covariance[index] < 0.0 for index in (0, 7, 35))):
            return ['invalid odometry covariance']
    # Local absolute x/y/yaw are unobservable from velocity + gyro alone:
    # their covariance legitimately grows even while AMCL keeps map pose
    # accurate. Only opt into a local-pose magnitude limit explicitly.
    if (max_position_variance > 0.0 and
            max(message.pose.covariance[0], message.pose.covariance[7]) > max_position_variance):
        return ['odometry position covariance too large']
    if max_yaw_variance > 0.0 and message.pose.covariance[35] > max_yaw_variance:
        return ['odometry yaw covariance too large']
    if max(message.twist.covariance[0], message.twist.covariance[7]) > max_linear_velocity_variance:
        return ['odometry linear velocity covariance too large']
    if message.twist.covariance[35] > max_angular_velocity_variance:
        return ['odometry angular velocity covariance too large']
    return []


def battery_voltage_issues(voltage, minimum_voltage):
    """A missing/invalid voltage cannot satisfy an enabled battery gate."""
    if not isinstance(voltage, (int, float)) or not math.isfinite(voltage) or voltage <= 0.0:
        return ['invalid battery voltage']
    # BatteryState.voltage is float32; values such as 11.4 can arrive a few
    # millionths below their decimal representation after serialization.
    if voltage < minimum_voltage - 0.001:
        return [f'battery voltage {voltage:.2f} V below {minimum_voltage:.2f} V']
    return []


def dynamixel_log_failure(message):
    """Only OpenCR transaction failures reported by the hardware driver."""
    return (message.level >= ROS_ERROR_LEVEL and
            message.name.lstrip('/') == 'DynamixelSDKWrapper' and
            message.msg.startswith(('Failed to read[', 'Failed to write[')))


class EkfSensorGuardCore:
    REQUIRED = ('wheel', 'imu', 'scan', 'odom', 'odom_tf')

    def __init__(self, arm_seconds=2.0, stale_seconds=1.0,
                 max_header_age_seconds=1.0, future_tolerance_seconds=0.25,
                 minimum_battery_voltage=0.0, battery_rearm_margin_voltage=0.2,
                 battery_stale_seconds=5.0, odom_tf_stale_seconds=1.5):
        if (arm_seconds <= 0.0 or stale_seconds <= 0.0 or
                type(odom_tf_stale_seconds) not in (int, float) or
                not math.isfinite(odom_tf_stale_seconds) or
                odom_tf_stale_seconds <= 0.0):
            raise ValueError('Guard durations must be positive')
        self.arm_seconds = arm_seconds
        self.stale_seconds = stale_seconds
        self.odom_tf_stale_seconds = float(odom_tf_stale_seconds)
        self.stamps = {name: SensorStampGuard(max_header_age_seconds,
                                             future_tolerance_seconds)
                       for name in self.REQUIRED}
        # TF ingress can briefly lag its 20 Hz EKF source. Keep the tighter
        # one-second wheel/IMU/scan/odom watchdog unchanged.
        self.stamps['odom_tf'].max_age_ns = int(self.odom_tf_stale_seconds * 1e9)
        self.stamps['battery'] = SensorStampGuard(battery_stale_seconds,
                                                 future_tolerance_seconds)
        self.last_seen = {name: None for name in self.stamps}
        self.value_issues = {}
        self.graph_issues = ['ROS publisher graph not yet checked']
        self.graph_checked_at = None
        self.healthy_since = None
        self.armed = False
        self.latched = True
        self.trip_reason = 'EKF inputs not yet validated'
        self.minimum_battery_voltage = 0.0
        self.battery_rearm_margin_voltage = 0.0
        self.battery_stale_seconds = 0.0
        self.battery_recovery_required = False
        self.required = self.REQUIRED
        self.set_battery_stale_seconds(battery_stale_seconds)
        self.set_battery_limits(minimum_battery_voltage, battery_rearm_margin_voltage)

    def set_battery_stale_seconds(self, seconds):
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or seconds <= 0.0):
            raise ValueError('battery_stale_seconds must be finite and positive')
        self.battery_stale_seconds = float(seconds)
        self.stamps['battery'].max_age_ns = int(seconds * 1e9)

    def set_battery_limits(self, minimum_voltage, rearm_margin):
        if (type(minimum_voltage) not in (int, float) or
                not math.isfinite(minimum_voltage) or minimum_voltage < 0.0 or
                type(rearm_margin) not in (int, float) or
                not math.isfinite(rearm_margin) or rearm_margin < 0.0):
            raise ValueError('Battery voltage limits must be finite and nonnegative')
        if (minimum_voltage, rearm_margin) == (self.minimum_battery_voltage,
                                                self.battery_rearm_margin_voltage):
            return
        self.minimum_battery_voltage = float(minimum_voltage)
        self.battery_rearm_margin_voltage = float(rearm_margin)
        if minimum_voltage == 0.0:
            self.battery_recovery_required = False
        self.required = (self.REQUIRED + ('battery',)
                         if minimum_voltage > 0.0 else self.REQUIRED)
        # A changed limit needs a newly measured sample and a fresh arm interval.
        self.last_seen['battery'] = None
        self.value_issues.pop('battery', None)
        self.healthy_since = None

    def observe_battery(self, voltage, stamp_ns, wall_ns, monotonic_seconds):
        if self.minimum_battery_voltage <= 0.0:
            return True
        threshold = self.minimum_battery_voltage + (
            self.battery_rearm_margin_voltage if self.battery_recovery_required else 0.0)
        if not self.observe('battery', stamp_ns, wall_ns, monotonic_seconds):
            return False
        issues = battery_voltage_issues(voltage, threshold)
        if issues:
            self.battery_recovery_required = True
            self.reject('battery', issues)
            if not self.latched:
                # A brief low sample must not be overwritten by a healthy
                # sample before the next 10 Hz safety timer fires.
                self.force_trip(f'battery: {issues[0]}')
            return False
        return True

    def force_trip(self, reason):
        """A direct hardware fault invalidates the current arm immediately."""
        self.armed = False
        self.latched = True
        self.healthy_since = None
        self.trip_reason = reason

    def latch_external_stop(self):
        """Honor an operator/mux stop without erasing healthy sensor history."""
        self.armed = False
        self.latched = True
        self.trip_reason = 'Motion mux emergency stop'

    def observe(self, name, stamp_ns, wall_ns, monotonic_seconds):
        if name not in self.stamps:
            raise ValueError(f'Unknown sensor stream: {name}')
        if not self.stamps[name].accept(stamp_ns, wall_ns):
            return False
        self.last_seen[name] = monotonic_seconds
        self.value_issues.pop(name, None)
        return True

    def reject(self, name, issues, *, invalidate=True):
        if invalidate:
            self.last_seen[name] = None
        self.value_issues[name] = ', '.join(issues)

    def missing(self, monotonic_seconds):
        missing = [f'{name}: {self.value_issues[name]}' if name in self.value_issues else name
                   for name in self.required
                   if self.last_seen[name] is None or
                   monotonic_seconds - self.last_seen[name] >= (
                       self.battery_stale_seconds if name == 'battery' else
                       self.odom_tf_stale_seconds if name == 'odom_tf' else
                       self.stale_seconds)]
        if (self.graph_checked_at is None or
                monotonic_seconds - self.graph_checked_at >= self.stale_seconds):
            missing.append('ROS publisher graph stale')
        else:
            missing.extend(self.graph_issues)
        return missing

    def update_graph(self, issues, monotonic_seconds):
        self.graph_issues = list(issues)
        self.graph_checked_at = monotonic_seconds

    def tick(self, monotonic_seconds):
        """Return a trip reason once; never auto-clear a latched fault."""
        missing = self.missing(monotonic_seconds)
        if missing:
            self.healthy_since = None
        elif self.healthy_since is None:
            self.healthy_since = monotonic_seconds

        if self.latched:
            return None
        if not self.armed:
            if (self.healthy_since is not None and
                    monotonic_seconds - self.healthy_since >= self.arm_seconds):
                self.armed = True
            return None
        if missing:
            self.armed = False
            self.latched = True
            self.trip_reason = ', '.join(missing) + ' sensor stream stale'
            return self.trip_reason
        return None

    def operator_reset(self, monotonic_seconds):
        """Allow re-arming only after an external reset and stable inputs."""
        if not self.latched or self.missing(monotonic_seconds):
            return False
        if (self.healthy_since is None or
                monotonic_seconds - self.healthy_since < self.arm_seconds):
            return False
        self.latched = False
        self.armed = True
        self.trip_reason = ''
        self.battery_recovery_required = False
        return True


class EkfSensorGuard(Node):
    SAFETY_SOURCE = 'ekf_sensor_safety'
    EXPECTED_PUBLISHERS = {
        '/odom': 'ekf_hardware',
        '/joint_states': None,
        '/imu': None,
        '/scan': None,
        '/wheel/odom_raw': None,
        '/wheel/odom': 'wheel_joint_odometry',
        '/imu/ekf': 'imu_ekf_conditioner',
    }

    def __init__(self):
        super().__init__('ekf_sensor_guard')
        # The physical runtime uses the stock wheel-only diagnostic topic and
        # /ekf_hardware. Gazebo's isolated EKF has different graph names; make
        # only those names configurable while retaining the same guard checks.
        self.declare_parameter('expected_ekf_odom_publisher', 'ekf_hardware')
        self.declare_parameter('expected_raw_wheel_odom_topic', '/wheel/odom_raw')
        ekf_publisher = self.get_parameter('expected_ekf_odom_publisher').value
        raw_wheel_topic = self.get_parameter('expected_raw_wheel_odom_topic').value
        self.expected_publishers = configured_publishers(
            self.EXPECTED_PUBLISHERS, ekf_publisher, raw_wheel_topic)
        self.declare_parameter('arm_seconds', 2.0)
        self.declare_parameter('stale_seconds', 1.0)
        self.declare_parameter('odom_tf_stale_seconds', 1.5)
        self.declare_parameter('dynamic_tf_best_effort', True)
        self.declare_parameter('max_position_variance', 0.0)
        self.declare_parameter('max_yaw_variance', 0.0)
        self.declare_parameter('max_linear_velocity_variance', 0.25)
        self.declare_parameter('max_angular_velocity_variance', 1.0)
        self.declare_parameter('base_frame', 'base_footprint')
        # Zero disables this optional threshold until the pack's loaded
        # voltage has been measured on the particular physical robot.
        self.declare_parameter('minimum_battery_voltage', 0.0)
        self.declare_parameter('battery_rearm_margin_voltage', 0.2)
        self.declare_parameter('battery_stale_seconds', 5.0)
        self.declare_parameter('enable_timing_diagnostics', False)
        self._timing = (RuntimeTiming(GUARD_METRICS, warning=self.get_logger().warning)
                        if self.get_parameter('enable_timing_diagnostics').value else None)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.tf_buffer = Buffer(node=self)
        # /tf is an ongoing stream. Best-effort depth 20 prevents delayed
        # reliable samples from starving the guard's latest transform. The
        # listener's separate /tf_static QoS remains reliable/transient-local.
        tf_qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE)
        self.tf_listener = TransformListener(
            self.tf_buffer, self,
            **({'qos': tf_qos} if self.get_parameter('dynamic_tf_best_effort').value else {}))
        self.core = EkfSensorGuardCore(
            arm_seconds=float(self.get_parameter('arm_seconds').value),
            stale_seconds=float(self.get_parameter('stale_seconds').value),
            odom_tf_stale_seconds=float(self.get_parameter('odom_tf_stale_seconds').value),
            minimum_battery_voltage=float(self.get_parameter('minimum_battery_voltage').value),
            battery_rearm_margin_voltage=float(
                self.get_parameter('battery_rearm_margin_voltage').value),
            battery_stale_seconds=float(self.get_parameter('battery_stale_seconds').value))
        self.add_on_set_parameters_callback(self.on_parameters)
        self.acknowledged = False
        self.last_estop_at = None
        self.safety_seq = 0
        self.pending_reset = None
        self.last_reset_seq = {}
        self.mux_instance_id = None
        self.retired_mux_instances = set()
        self.last_status_seq = -1
        self.last_tf_check_at = None
        self.last_owner_check_at = None
        self.control_pub = self.create_publisher(
            String, '/smart_vision_amr/manual_control', 10)
        self.create_subscription(Odometry, '/wheel/odom',
                                 lambda msg: self.on_sensor('wheel', msg),
                                 qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu/ekf',
                                 lambda msg: self.on_sensor('imu', msg),
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan',
                                 lambda msg: self.on_sensor('scan', msg),
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom,
                                 qos_profile_sensor_data)
        self.create_subscription(BatteryState, '/battery_state', self.on_battery,
                                 qos_profile_sensor_data)
        self.create_subscription(Log, '/rosout', self.on_rosout,
                                 QoSProfile(depth=20,
                                            reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.VOLATILE))
        durable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/smart_vision_amr/motion_status',
                                 self.on_motion_status, durable)
        # Retain one-shot reset and subsequent stop intents through a short
        # callback backlog. on_control keeps the source/sequence/1s-age guards
        # and an external stop still invalidates a pending reset.
        self.create_subscription(String, '/smart_vision_amr/manual_control',
                                 self.on_control,
                                 QoSProfile(depth=10,
                                            reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.VOLATILE))
        self.create_timer(0.1, self.on_tick,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))
        self._publish_estop(time.monotonic())
        self.get_logger().warning(
            'EKF motion held until wheel, IMU, LiDAR, /odom output, odom TF and sole EKF /odom '
            'publisher are healthy for 2 s and the operator explicitly resets')

    def on_parameters(self, parameters):
        minimum = self.core.minimum_battery_voltage
        margin = self.core.battery_rearm_margin_voltage
        battery_stale = self.core.battery_stale_seconds
        for parameter in parameters:
            if parameter.name == 'minimum_battery_voltage':
                minimum = parameter.value
            elif parameter.name == 'battery_rearm_margin_voltage':
                margin = parameter.value
            elif parameter.name == 'battery_stale_seconds':
                battery_stale = parameter.value
        try:
            if (type(battery_stale) not in (int, float) or
                    not math.isfinite(battery_stale) or battery_stale <= 0.0):
                raise ValueError('battery_stale_seconds must be finite and positive')
            self.core.set_battery_limits(minimum, margin)
            self.core.set_battery_stale_seconds(battery_stale)
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))
        return SetParametersResult(successful=True)

    def on_battery(self, message):
        stamp = message.header.stamp
        was_latched = self.core.latched
        self.core.observe_battery(
            message.voltage, stamp.sec*1_000_000_000+stamp.nanosec,
            self.get_clock().now().nanoseconds, time.monotonic())
        if not was_latched and self.core.latched:
            self.get_logger().error(f'{self.core.trip_reason}; emergency stop latched')
            self._publish_estop(time.monotonic())

    def on_rosout(self, message):
        if not dynamixel_log_failure(message):
            return
        if self.core.trip_reason.startswith('OpenCR transaction failed:'):
            return  # Repeated driver errors must not flood safety publications.
        reason = f'OpenCR transaction failed: {message.msg}'
        self.core.force_trip(reason)
        self.get_logger().error(f'{reason}; emergency stop latched')
        self._publish_estop(time.monotonic())

    def on_sensor(self, name, message):
        stamp = message.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        self.core.observe(name, stamp_ns, self.get_clock().now().nanoseconds,
                          time.monotonic())

    def on_odom(self, message):
        issues = odometry_issues(
            message, base_frame=self.base_frame,
            max_position_variance=float(self.get_parameter('max_position_variance').value),
            max_yaw_variance=float(self.get_parameter('max_yaw_variance').value),
            max_linear_velocity_variance=float(self.get_parameter('max_linear_velocity_variance').value),
            max_angular_velocity_variance=float(self.get_parameter('max_angular_velocity_variance').value))
        if issues:
            self.core.reject('odom', issues)
        else:
            self.on_sensor('odom', message)

    def update_tf(self, now):
        try:
            transform = timed_call(getattr(self, '_timing', None), 'tf_lookup',
                                   self.tf_buffer.lookup_transform,
                                   'odom', self.base_frame, rclpy.time.Time())
            t, q = transform.transform.translation, transform.transform.rotation
            values = (t.x, t.y, t.z, q.x, q.y, q.z, q.w)
            if (not all(math.isfinite(value) for value in values)
                    or not 0.95 <= sum(value*value for value in (q.x, q.y, q.z, q.w)) <= 1.05):
                self.core.reject('odom_tf', ['invalid TF values'])
                return
            stamp = transform.header.stamp
            stamp_ns = stamp.sec*1_000_000_000+stamp.nanosec
            wall_ns = self.get_clock().now().nanoseconds
            if not self.core.observe('odom_tf', stamp_ns, wall_ns, now):
                age = (wall_ns-stamp_ns)/1e9
                self.core.reject('odom_tf', [f'odom to base TF header rejected (age {age:.2f}s)'],
                                 invalidate=False)
        except TransformException as exc:
            # A single missed lookup is not a sustained TF outage. Keep the
            # last valid sample until its independent TF deadline expires.
            self.core.reject('odom_tf', [f'odom to base TF unavailable: {exc}'],
                             invalidate=False)

    def on_tick(self):
        timing = getattr(self, '_timing', None)
        if timing is not None:
            timing.gap('tick_gap', .1)
        now = time.monotonic()
        if (self.last_tf_check_at is None or
                now - self.last_tf_check_at >= 0.2):
            self.last_tf_check_at = now
            self.update_tf(now)
        if (self.last_owner_check_at is None or
                now - self.last_owner_check_at >= 0.5):
            self.last_owner_check_at = now
            try:
                expected = self.expected_publishers.copy()
                if self.core.minimum_battery_voltage > 0.0:
                    expected['/battery_state'] = None
                issues = timed_call(timing, 'ownership_query', publisher_issues,
                                    self.get_publishers_info_by_topic, expected)
            except Exception as exc:
                self.get_logger().warning(f'Cannot inspect ROS publishers: {exc}')
                issues = [f'ROS publisher graph query failed: {exc}']
            self.core.update_graph(issues, now)
        reason = self.core.tick(now)
        if reason is not None:
            self.get_logger().error(f'EKF input lost: {reason}; emergency stop latched')
        # Keep a slow safety heartbeat even after acknowledgement. If a web
        # mux restarts and its transient unlocked status is lost, a one-shot
        # estop would otherwise leave it unlocked indefinitely.
        resend_after = 1.0 if self.acknowledged else 0.2
        if (self.core.latched and
                (self.last_estop_at is None or now - self.last_estop_at >= resend_after)):
            self._publish_estop(now)

    def _publish_estop(self, now):
        # The mux remembers safety-source sequences across guard restarts.
        self.safety_seq = max(self.safety_seq + 1, time.monotonic_ns())
        message = String()
        payload = {
            'source_id': self.SAFETY_SOURCE,
            'seq': self.safety_seq,
            'sent_at_ns': self.get_clock().now().nanoseconds,
            'mode': 'estop', 'linear': 0.0, 'angular': 0.0,
        }
        timing = getattr(self, '_timing', None)
        if timing is not None:
            # A publish cannot report its own completed duration. The next
            # safety heartbeat carries the previous result, without restamping
            # or retransmitting this control packet after a slow publication.
            payload['timing_diagnostics'] = {'metrics': timing.snapshot()}
        message.data = json.dumps(payload)
        timed_call(timing, 'control_publish', self.control_pub.publish, message)
        self.last_estop_at = now

    def on_control(self, message):
        """Track reset intent and invalidate it when an external stop follows."""
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if not isinstance(payload, dict) or payload.get('mode') not in {'reset', 'estop'}:
            return
        mode = payload['mode']
        source = payload.get('source_id')
        seq = payload.get('seq')
        stamp_ns = payload.get('sent_at_ns')
        now_ns = self.get_clock().now().nanoseconds
        safety_sources = ('ekf_sensor_safety', 'path_follower_recovery_safety',
                          'localization_safety')
        if (not isinstance(source, str) or not source or
                (mode == 'reset' and source in safety_sources) or
                (mode == 'estop' and source == EkfSensorGuard.SAFETY_SOURCE) or
                type(seq) is not int or seq <= self.last_reset_seq.get(source, -1) or
                type(stamp_ns) is not int or abs(now_ns - stamp_ns) > 1_000_000_000):
            return
        if source.startswith('mux-service:'):
            retired = getattr(self, 'retired_mux_instances', set())
            if source in retired:
                return
            current = getattr(self, 'mux_instance_id', None)
            if current != source:
                if current is not None:
                    retired.add(current)
                self.retired_mux_instances = retired
                self.mux_instance_id = source
                self.last_status_seq = -1
        if mode == 'estop':
            self.last_reset_seq[source] = seq
            self.pending_reset = None
            self.acknowledged = False
            if not self.core.latched:
                self.core.latch_external_stop()
            return
        self.last_reset_seq[source] = seq
        self.pending_reset = (source, time.monotonic())

    def on_motion_status(self, message):
        try:
            status = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if not isinstance(status, dict):
            return
        metadata = ('mux_instance_id', 'status_seq', 'sent_at_ns')
        if any(key in status for key in metadata):
            instance, seq, stamp = (status.get(key) for key in metadata)
            if (not isinstance(instance, str) or not instance or type(seq) is not int
                    or seq < 0 or type(stamp) is not int
                    or abs(self.get_clock().now().nanoseconds-stamp) > 1_000_000_000
                    or instance in getattr(self, 'retired_mux_instances', set())):
                return
            current = getattr(self, 'mux_instance_id', None)
            if current != instance:
                if current is not None:
                    self.retired_mux_instances.add(current)
                self.mux_instance_id = instance
                self.last_status_seq = -1
            if seq <= getattr(self, 'last_status_seq', -1):
                return
            self.last_status_seq = seq
        elif getattr(self, 'mux_instance_id', None) is not None:
            return  # Legacy packets cannot bypass an identified mux stream.
        if status.get('estop_latched') is True:
            if not self.core.latched:
                self.core.latch_external_stop()
            self.acknowledged = True
        elif status.get('estop_latched') is False and self.core.latched and self.acknowledged:
            # A mux/web restart also produces true->false without a reset.
            # Require a fresh, matching web reset packet to distinguish it.
            self.acknowledged = False
            now = time.monotonic()
            reset = self.pending_reset
            self.pending_reset = None
            explicit_reset = (reset is not None and reset[0] == status.get('source_id')
                              and now - reset[1] <= 1.0)
            if explicit_reset and self.core.operator_reset(now):
                self.get_logger().info('Operator reset accepted; EKF sensor guard re-armed')
            else:
                healthy_for = (None if self.core.healthy_since is None else
                               round(now-self.core.healthy_since, 3))
                self.get_logger().error(
                    'Motion reset rejected: no fresh explicit operator reset '
                    'or EKF inputs not stable; '
                    f'explicit_reset={explicit_reset}, '
                    f'reset_source={reset[0] if reset else None}, '
                    f'mux_source={status.get("source_id")}, '
                    f'healthy_for_s={healthy_for}, '
                    f'missing={self.core.missing(now)}')
                self._publish_estop(now)


def main(args=None):
    rclpy.init(args=args)
    node = EkfSensorGuard()
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
