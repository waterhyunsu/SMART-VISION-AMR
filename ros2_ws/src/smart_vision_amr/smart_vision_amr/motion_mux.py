"""One velocity publisher, with leased manual control ahead of navigation.

The core deliberately has no ROS imports so timing and ownership races can be
tested without a robot.  Navigation never publishes directly to /cmd_vel in
the supplied launch files; only this node does.
"""

import json
import math
import time


class MotionMuxCore:
    """Arbitrate commands using a monotonic dead-man lease, never ROS time."""

    MODES = frozenset({'manual', 'hold', 'auto', 'estop', 'reset'})
    RECOVERY_SAFETY_SOURCE = 'path_follower_recovery_safety'
    EKF_SENSOR_SAFETY_SOURCE = 'ekf_sensor_safety'
    SAFETY_REASONS = {
        RECOVERY_SAFETY_SOURCE: 'Reverse recovery safety stop',
        EKF_SENSOR_SAFETY_SOURCE: 'EKF wheel, IMU, or LiDAR input lost',
    }

    def __init__(self, clock=time.monotonic, manual_timeout=.5, auto_timeout=.5,
                 max_linear_velocity=.25, max_angular_velocity=1.0):
        self.clock = clock
        self.manual_timeout = float(manual_timeout)
        self.auto_timeout = float(auto_timeout)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        if any(not math.isfinite(value) or value <= 0 for value in (
                self.manual_timeout, self.auto_timeout, self.max_linear_velocity,
                self.max_angular_velocity)):
            raise ValueError('Velocity limits and timeouts must be finite and positive')
        self.mode = 'auto'
        self.reason = 'Navigation control'
        self.source_id = None
        self.retired_sources = set()
        self.sequences = {}
        self.manual_at = None
        self.manual_velocity = (0.0, 0.0)
        self.auto_at = None
        self.auto_velocity = (0.0, 0.0)
        self.estop_latched = False

    def _velocity(self, linear, angular):
        if (isinstance(linear, bool) or isinstance(angular, bool)
                or not isinstance(linear, (int, float))
                or not isinstance(angular, (int, float))):
            raise ValueError('Velocity must be numeric')
        linear, angular = float(linear), float(angular)
        if not math.isfinite(linear) or not math.isfinite(angular):
            raise ValueError('Velocity must be finite')
        return (max(-self.max_linear_velocity, min(self.max_linear_velocity, linear)),
                max(-self.max_angular_velocity, min(self.max_angular_velocity, angular)))

    def _expire(self):
        if (self.mode == 'manual' and self.manual_at is not None
                and self.clock() - self.manual_at >= self.manual_timeout):
            self.mode = 'hold'
            self.reason = 'Manual control lease expired; explicit resume required'
            self.manual_velocity = (0.0, 0.0)
            self.auto_at = None

    def receive_auto(self, linear, angular):
        """Only a post-transition command may restart automatic motion."""
        self._expire()
        try:
            velocity = self._velocity(linear, angular)
        except ValueError:
            self.auto_at = None
            return False
        # Drop, rather than retain, navigation output during an override.
        if self.mode != 'auto' or self.estop_latched:
            return False
        self.auto_velocity, self.auto_at = velocity, self.clock()
        return True

    def receive_control(self, payload, ros_now_ns):
        """Accept one atomic state/velocity packet from the current web bridge.

        ROS stamps reject delayed DDS packets.  Lease expiry still works while
        /clock is paused because lease receipt is timed with monotonic time.
        """
        self._expire()
        if not isinstance(payload, dict):
            return False
        source, seq, mode = (payload.get('source_id'), payload.get('seq'),
                             payload.get('mode'))
        stamp = payload.get('sent_at_ns')
        if (not isinstance(source, str) or not source or len(source) > 128
                or type(seq) is not int or seq < 0
                or not isinstance(mode, str) or mode not in self.MODES
                or (source in self.SAFETY_REASONS and mode != 'estop')
                or type(stamp) is not int or stamp < 0
                or type(ros_now_ns) is not int
                or abs(ros_now_ns-stamp) > 1_000_000_000
                or source in self.retired_sources
                or seq <= self.sequences.get(source, -1)):
            return False
        try:
            velocity = self._velocity(payload.get('linear', 0.0),
                                      payload.get('angular', 0.0))
        except ValueError:
            return False
        if source in self.SAFETY_REASONS:
            # Independent safety watchdogs can stop even a live manual lease.
            # Keep the web bridge as owner so its explicit emergency-reset
            # button remains the only way to clear the latched stop.
            self.sequences[source] = seq
            self.estop_latched = True
            self.mode, self.reason = 'estop', self.SAFETY_REASONS[source]
            self.manual_velocity, self.auto_at = (0.0, 0.0), None
            return True
        if self.estop_latched and mode not in {'reset', 'estop'}:
            return False
        now = self.clock()
        if (self.source_id is not None and source != self.source_id
                and self.mode == 'manual' and mode != 'estop'):
            return False
        if source != self.source_id:
            if self.source_id is not None:
                self.retired_sources.add(self.source_id)
            self.source_id = source
        self.sequences[source] = seq

        if mode == 'estop':
            self.estop_latched = True
            self.mode, self.reason = 'estop', 'Emergency stop latched'
            self.manual_velocity, self.auto_at = (0.0, 0.0), None
            return True
        if self.estop_latched:
            if mode != 'reset':
                return False
            self.estop_latched = False
            self.mode, self.reason = 'hold', 'Emergency reset; explicit resume required'
            self.manual_velocity, self.auto_at = (0.0, 0.0), None
            return True
        if mode in {'hold', 'reset'}:
            self.mode, self.reason = 'hold', 'Motion held'
            self.manual_velocity, self.auto_at = (0.0, 0.0), None
        elif mode == 'manual':
            self.mode, self.reason = 'manual', 'Leased manual control has priority'
            self.manual_velocity, self.manual_at = velocity, now
            self.auto_at = None
        else:
            # Repeated auto heartbeats must not erase fresh navigation commands.
            if self.mode != 'auto':
                self.auto_at = None
            self.mode, self.reason = 'auto', 'Navigation control'
            self.manual_velocity = (0.0, 0.0)
        return True

    def output(self):
        self._expire()
        if self.estop_latched:
            return 0.0, 0.0
        if self.mode == 'manual':
            return self.manual_velocity
        if (self.mode == 'auto' and self.auto_at is not None
                and self.clock()-self.auto_at < self.auto_timeout):
            return self.auto_velocity
        return 0.0, 0.0

    def status(self):
        self._expire()
        return {'mode': self.mode, 'available': True, 'reason': self.reason,
                'source_id': self.source_id, 'estop_latched': self.estop_latched}


# Keep the arbitration core importable for non-ROS unit tests as well.
try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
except ImportError:
    Node = object


class MotionMuxNode(Node):
    def __init__(self, context=None, use_sim_time=None):
        overrides = []
        if use_sim_time is not None:
            from rclpy.parameter import Parameter
            overrides.append(Parameter('use_sim_time', value=bool(use_sim_time)))
        super().__init__('motion_mux', context=context, parameter_overrides=overrides)
        for name, default in (
                ('manual_timeout', .5), ('auto_timeout', .5),
                ('max_linear_velocity', .25), ('max_angular_velocity', 1.0)):
            self.declare_parameter(name, default)
        self.core = MotionMuxCore(**{
            name: self.get_parameter(name).value for name in (
                'manual_timeout', 'auto_timeout', 'max_linear_velocity',
                'max_angular_velocity')})
        self.output_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.output_available = False
        self.output_conflict = 'Checking exclusive /cmd_vel ownership'
        self.output_publishers = 0
        self._ownership_checked_at = -math.inf
        durable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(
            String, '/smart_vision_amr/motion_status', durable)
        self.create_subscription(Twist, '/smart_vision_amr/navigation_cmd_vel',
                                 self.on_auto, 1)
        self.create_subscription(String, '/smart_vision_amr/manual_control',
                                 self.on_control, 1)
        # ROS timers stop when simulation time pauses. Dead-man watchdogs
        # must continue to run even then, so use a steady clock explicitly.
        from rclpy.clock import Clock, ClockType
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.05, self.publish_output, clock=self.steady_clock)
        self.create_timer(.1, self.publish_status, clock=self.steady_clock)
        self.publish_status()

    def on_auto(self, message):
        self.core.receive_auto(message.linear.x, message.angular.z)

    def on_control(self, message):
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if self.core.receive_control(payload, self.get_clock().now().nanoseconds):
            self.publish_output()
            self.publish_status()

    def publish_output(self):
        self.check_output_ownership()
        if not self.output_available:
            # Even a zero command would interfere with an existing, unmanaged
            # controller. Wait for an explicit exclusive handover instead.
            return
        message = Twist()
        message.linear.x, message.angular.z = self.core.output()
        self.output_pub.publish(message)

    def check_output_ownership(self):
        """Back off completely if any other publisher can command the robot."""
        now = time.monotonic()
        if now-self._ownership_checked_at < .2:
            return
        self._ownership_checked_at = now
        try:
            endpoints = self.get_publishers_info_by_topic(self.output_pub.topic_name)
            self.output_publishers = len(endpoints)
            own_endpoint = (len(endpoints) == 1
                            and endpoints[0].node_name == self.get_name()
                            and endpoints[0].node_namespace == self.get_namespace())
            self.output_available = own_endpoint
            if own_endpoint:
                self.output_conflict = ''
            elif not endpoints:
                self.output_conflict = 'Waiting for local /cmd_vel publisher discovery'
            else:
                names = [f'{item.node_namespace.rstrip("/")}/{item.node_name}'
                         for item in endpoints]
                self.output_conflict = ('Unmanaged /cmd_vel publisher: '
                                        + ', '.join(names))
        except Exception as exc:
            # Failure to inspect the graph must never imply exclusive access.
            self.output_available = False
            self.output_conflict = 'Cannot verify exclusive /cmd_vel ownership: '+str(exc)

    def publish_status(self):
        self.check_output_ownership()
        status = self.core.status()
        status['available'] = self.output_available
        status['output_publishers'] = self.output_publishers
        if not self.output_available:
            status['reason'] = self.output_conflict
        message = String()
        message.data = json.dumps(status, separators=(',', ':'))
        self.status_pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = MotionMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.core.mode = 'hold'
            node.publish_output()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
