"""One velocity publisher, with leased manual control ahead of navigation.

The core deliberately has no ROS imports so timing and ownership races can be
tested without a robot.  Navigation never publishes directly to /cmd_vel in
the supplied launch files; only this node does.
"""

import json
import math
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext

import numpy as np

from smart_vision_amr.runtime_timing import (
    MUX_METRICS, RuntimeTiming, guard_timing_report, timed_call,
)
from smart_vision_amr.scan_outlier_guard import (NearScanOutlierGuard,
    rebase_scan_points, scan_points_in_base, swept_body_decision)
from smart_vision_amr.command_ramp import CommandRamp
from smart_vision_amr.braking_envelope import BrakingEnvelope


class MotionMuxCore:
    """Arbitrate commands using a monotonic dead-man lease, never ROS time."""

    MODES = frozenset({'manual', 'localization', 'hold', 'auto', 'estop', 'reset'})
    RECOVERY_SAFETY_SOURCE = 'path_follower_recovery_safety'
    EKF_SENSOR_SAFETY_SOURCE = 'ekf_sensor_safety'
    LOCALIZATION_SAFETY_SOURCE = 'localization_safety'
    SAFETY_REASONS = {
        RECOVERY_SAFETY_SOURCE: 'Reverse recovery safety stop',
        EKF_SENSOR_SAFETY_SOURCE: 'EKF wheel, IMU, or LiDAR input lost',
        LOCALIZATION_SAFETY_SOURCE: 'Localization confidence unsafe',
    }

    def __init__(self, clock=time.monotonic, manual_timeout=.5, auto_timeout=.5,
                 max_linear_velocity=.25, max_angular_velocity=1.0,
                 max_auto_age_seconds=.35, auto_future_tolerance_seconds=.25,
                 require_localization_health=False,
                 localization_health_timeout=1.0,
                 localization_warning_speed_factor=.5,
                 localization_rotation_limit=.25, startup_estop=False,
                 marker_alignment_rotation_limit=.15,
                 auto_turn_slowdown_enabled=False,
                 auto_turn_start_angular=.08, auto_turn_full_angular=.35,
                 auto_turn_entry_linear_velocity=.15,
                 auto_turn_min_linear_velocity=.10):
        self.clock = clock
        self.manual_timeout = float(manual_timeout)
        self.auto_timeout = float(auto_timeout)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.auto_turn_slowdown_enabled = bool(auto_turn_slowdown_enabled)
        self.auto_turn_start_angular = float(auto_turn_start_angular)
        self.auto_turn_full_angular = float(auto_turn_full_angular)
        self.auto_turn_entry_linear_velocity = float(auto_turn_entry_linear_velocity)
        self.auto_turn_min_linear_velocity = float(auto_turn_min_linear_velocity)
        if any(not math.isfinite(value) or value <= 0 for value in (
                self.manual_timeout, self.auto_timeout, self.max_linear_velocity,
                self.max_angular_velocity)):
            raise ValueError('Velocity limits and timeouts must be finite and positive')
        if (not all(math.isfinite(value) for value in (
                self.auto_turn_start_angular, self.auto_turn_full_angular,
                self.auto_turn_entry_linear_velocity,
                self.auto_turn_min_linear_velocity))
                or not 0 < self.auto_turn_start_angular < self.auto_turn_full_angular
                <= self.max_angular_velocity
                or not 0 < self.auto_turn_min_linear_velocity
                <= self.auto_turn_entry_linear_velocity <= self.max_linear_velocity):
            raise ValueError('Invalid autonomous turn slowdown limits')
        self.mode = 'auto'
        self.reason = 'Navigation control'
        self.source_id = None
        self.retired_sources = set()
        self.sequences = {}
        self.manual_at = None
        self.manual_velocity = (0.0, 0.0)
        self.auto_at = None
        self.auto_velocity = (0.0, 0.0)
        self.estop_latched = bool(startup_estop)
        if self.estop_latched:
            self.mode, self.reason = 'estop', 'Startup interlock; explicit safety reset required'
        self.max_auto_age_ns = int(max_auto_age_seconds * 1e9)
        self.auto_future_tolerance_ns = int(auto_future_tolerance_seconds*1e9)
        self.auto_source_id = None
        self.retired_auto_sources = set()
        self.auto_sequences = {}
        self.auto_last_stamp = -1
        self.auto_rejected_reason = ''
        # A raw near-minimum LiDAR return is a physical collision uncertainty,
        # independent of whether A* has already classified/map-projected it.
        self.scan_guard_hold = False
        self.scan_guard_reason = ''
        self.scan_guard_code = ''
        self.scan_guard_support = ''
        self.scan_guard_metrics = {}
        self.require_localization_health = bool(require_localization_health)
        self.localization_health_timeout = float(localization_health_timeout)
        self.localization_warning_speed_factor = float(localization_warning_speed_factor)
        self.localization_rotation_limit = float(localization_rotation_limit)
        self.marker_alignment_rotation_limit = float(marker_alignment_rotation_limit)
        if (self.max_auto_age_ns <= 0 or self.auto_future_tolerance_ns < 0
                or self.localization_health_timeout <= 0
                or not 0 < self.localization_warning_speed_factor <= 1
                or not 0 < self.localization_rotation_limit <= self.max_angular_velocity
                or not 0 < self.marker_alignment_rotation_limit <= self.max_angular_velocity):
            raise ValueError('Invalid command age or localization safety limits')
        self.localization_at = None
        self.localization_health = None
        self.manual_operation = None
        self.control_received = self.control_accepted = self.control_rejected = 0
        self.auto_received = self.auto_accepted = self.auto_rejected = 0
        self.last_control = self.last_auto = None
        self.last_control_rejection = self.last_auto_rejection = None
        self.control_rejected_reason = ''
        self.control_by_origin = {}
        self.auto_by_origin = {}

    def _receipt(self, channel, payload, accepted, reason, origin):
        """Record once at the public boundary, including pre-core rejects.

        Nested application helpers do not create receipts. Keep only the most
        recent packet and rejection; counters never enter the publish bypass key.
        """
        origin = origin if origin in {'topic', 'service', 'core', 'legacy'} else 'core'
        origins = getattr(self, channel+'_by_origin')
        counts = origins.setdefault(origin, {'received': 0, 'accepted': 0, 'rejected': 0})
        counts['received'] += 1
        counts['accepted' if accepted else 'rejected'] += 1
        packet = payload if isinstance(payload, dict) else {}
        source, seq, mode = (packet.get('source_id'), packet.get('seq'), packet.get('mode'))
        receipt = {'source_id': source[:128] if isinstance(source, str) else None,
                   'seq': seq if type(seq) is int else None,
                   'mode': mode[:32] if isinstance(mode, str) else None,
                   'accepted': bool(accepted), 'reason': reason, 'origin': origin,
                   'received_at_monotonic_ns': int(self.clock()*1e9)}
        setattr(self, channel+'_received', getattr(self, channel+'_received')+1)
        outcome = channel+('_accepted' if accepted else '_rejected')
        setattr(self, outcome, getattr(self, outcome)+1)
        setattr(self, 'last_'+channel, receipt)
        if not accepted:
            setattr(self, 'last_'+channel+'_rejection', dict(receipt))
        return bool(accepted)

    def reject_control(self, payload, reason, origin='topic'):
        self.control_rejected_reason = reason
        return self._receipt('control', payload, False, reason, origin)

    def reject_auto(self, payload, reason, origin='topic'):
        self.auto_rejected_reason = reason
        return self._receipt('auto', payload, False, reason, origin)

    def _velocity(self, linear, angular):
        if (isinstance(linear, bool) or isinstance(angular, bool)
                or not isinstance(linear, (int, float))
                or not isinstance(angular, (int, float))):
            raise ValueError('Velocity must be numeric')
        try:
            linear, angular = float(linear), float(angular)
        except OverflowError as exc:
            raise ValueError('Velocity must be finite') from exc
        if not math.isfinite(linear) or not math.isfinite(angular):
            raise ValueError('Velocity must be finite')
        return (max(-self.max_linear_velocity, min(self.max_linear_velocity, linear)),
                max(-self.max_angular_velocity, min(self.max_angular_velocity, angular)))

    def _expire(self):
        if (self.mode in {'manual', 'localization'} and self.manual_at is not None
                and self.clock() - self.manual_at >= self.manual_timeout):
            self.mode = 'hold'
            self.reason = 'Operator control lease expired; explicit resume required'
            self.manual_velocity = (0.0, 0.0)
            self.auto_at = None

    def receive_auto(self, linear, angular):
        accepted = self._apply_auto(linear, angular)
        return self._receipt('auto', {'mode': 'auto'}, accepted,
                             '' if accepted else self.auto_rejected_reason, 'legacy')

    def _apply_auto(self, linear, angular):
        """Only a post-transition command may restart automatic motion."""
        self._expire()
        try:
            velocity = self._velocity(linear, angular)
        except ValueError:
            self.auto_at = None
            self.auto_rejected_reason = 'Invalid autonomous velocity'
            return False
        if self.auto_turn_slowdown_enabled and velocity[0] > 0.:
            # Keep the controller's angular authority while reducing the
            # physical turn radius. Nav2's /speed_limit scales angular speed
            # too, so it cannot safely provide this final-command coupling.
            yaw_rate = abs(velocity[1])
            if yaw_rate < self.auto_turn_start_angular:
                fraction = yaw_rate/self.auto_turn_start_angular
                linear_cap = (self.max_linear_velocity - fraction *
                              (self.max_linear_velocity-self.auto_turn_entry_linear_velocity))
            else:
                fraction = min(1., (yaw_rate-self.auto_turn_start_angular)
                               / (self.auto_turn_full_angular-self.auto_turn_start_angular))
                linear_cap = (self.auto_turn_entry_linear_velocity - fraction *
                              (self.auto_turn_entry_linear_velocity-self.auto_turn_min_linear_velocity))
            velocity = min(velocity[0], linear_cap), velocity[1]
        # Drop, rather than retain, navigation output during an override.
        if self.mode != 'auto' or self.estop_latched:
            self.auto_rejected_reason = 'Autonomous command blocked by '+self.mode+' control'
            return False
        self.auto_rejected_reason = ''
        self.auto_velocity, self.auto_at = velocity, self.clock()
        return True

    def receive_stamped_auto(self, payload, ros_now_ns):
        accepted = self._receive_stamped_auto(payload, ros_now_ns)
        return self._receipt('auto', payload, accepted,
                             '' if accepted else self.auto_rejected_reason, 'topic')

    def _receive_stamped_auto(self, payload, ros_now_ns):
        """Validate forwarding age and sequence before renewing the steady lease.

        A new bridge UUID may take over only with a stamp newer than every
        previously accepted source. A retired bridge can never resume. Commands
        rejected during manual/hold also consume their sequence so releasing a
        manual lease cannot replay navigation output buffered during override.
        """
        if not isinstance(payload, dict):
            self.auto_rejected_reason = 'Autonomous packet must be a JSON object'
            return False
        source, seq, stamp = (payload.get('source_id'), payload.get('seq'),
                              payload.get('sent_at_ns'))
        if (not isinstance(source, str) or not source or len(source) > 128
                or type(seq) is not int or seq < 0 or type(stamp) is not int
                or type(ros_now_ns) is not int or stamp < 0
                or stamp > ros_now_ns + self.auto_future_tolerance_ns
                or ros_now_ns-stamp > self.max_auto_age_ns
                or source in self.retired_auto_sources
                or seq <= self.auto_sequences.get(source, -1)
                or stamp < self.auto_last_stamp
                or (source != self.auto_source_id and stamp <= self.auto_last_stamp)):
            self.auto_rejected_reason = 'Delayed, replayed, or invalid autonomous command'
            return False
        try:
            self._velocity(payload.get('linear'), payload.get('angular'))
        except ValueError:
            self.auto_rejected_reason = 'Invalid autonomous velocity'
            self.auto_at = None
            return False
        if source != self.auto_source_id:
            if self.auto_source_id is not None:
                self.retired_auto_sources.add(self.auto_source_id)
            self.auto_source_id = source
        self.auto_sequences[source], self.auto_last_stamp = seq, stamp
        self.auto_rejected_reason = ''
        return self._apply_auto(payload['linear'], payload['angular'])

    def receive_localization_health(self, payload):
        if not isinstance(payload, dict) or payload.get('level') not in {
                'ok', 'healthy', 'warning', 'critical'}:
            return False
        self.localization_health = dict(payload)
        self.localization_at = self.clock()
        return True

    def _localization_policy(self):
        if (self.localization_at is None
                or self.clock()-self.localization_at > self.localization_health_timeout):
            return (0.0, False, False, 'Localization health missing or stale') if (
                self.require_localization_health) else (1.0, True, False, '')
        health = self.localization_health
        level = health['level']
        # A low scan/map match alone is advisory while localization still
        # explicitly permits autonomous motion. Halving an already small DWB
        # command can leave the physical drive below its response threshold.
        advisory_map_mismatch = (level == 'warning'
                                 and health.get('autonomous_allowed') is True
                                 and health.get('reasons') == ['scan_map_agreement_low'])
        factor = (0.0 if level == 'critical' else
                  self.localization_warning_speed_factor
                  if level == 'warning' and not advisory_map_mismatch else 1.0)
        return (factor, health.get('autonomous_allowed', True) is True,
                health.get('allow_localization', False) is True,
                '; '.join(str(x) for x in health.get('reasons', [])))

    def receive_control(self, payload, ros_now_ns, origin='core'):
        self.control_rejected_reason = ''
        accepted = self._receive_control(payload, ros_now_ns)
        return self._receipt('control', payload, accepted,
                             '' if accepted else self.control_rejected_reason, origin)

    def _reject_control(self, reason):
        self.control_rejected_reason = reason
        return False

    def _receive_control(self, payload, ros_now_ns):
        """Accept one atomic state/velocity packet from the current web bridge.

        ROS stamps reject delayed DDS packets.  Lease expiry still works while
        /clock is paused because lease receipt is timed with monotonic time.
        """
        self._expire()
        try:
            velocity = self._validated_control_velocity(payload, ros_now_ns)
        except ValueError as exc:
            return self._reject_control(str(exc))
        source, seq, mode = (payload['source_id'], payload['seq'], payload['mode'])
        if source in self.SAFETY_REASONS and mode == 'estop':
            # Independent safety watchdogs stop without taking the UI's lease.
            self.sequences[source] = seq
            self.estop_latched = True
            self.mode, self.reason = 'estop', self.SAFETY_REASONS[source]
            self.manual_velocity, self.auto_at = (0.0, 0.0), None
            return True
        now = self.clock()
        if source != self.source_id:
            if (self.source_id is not None
                    and not source.startswith('mux-service:')
                    and not self.source_id.startswith('mux-service:')):
                self.retired_sources.add(self.source_id)
            self.source_id = source
        self.sequences[source] = seq

        if mode == 'estop':
            self.estop_latched = True
            self.mode, self.reason = 'estop', 'Emergency stop latched'
            self.manual_velocity, self.auto_at = (0.0, 0.0), None
            return True
        if self.estop_latched:
            self.estop_latched = False
            self.mode, self.reason = 'hold', 'Emergency reset; explicit resume required'
            self.manual_velocity, self.auto_at = (0.0, 0.0), None
            return True
        if mode in {'hold', 'reset'}:
            self.mode, self.reason = 'hold', 'Motion held'
            self.manual_velocity, self.auto_at = (0.0, 0.0), None
        elif mode in {'manual', 'localization'}:
            self.mode, self.reason = mode, ('Leased localization rotation' if mode == 'localization'
                                           else 'Leased manual control has priority')
            self.manual_velocity, self.manual_at = velocity, now
            self.auto_at = None
            self.manual_operation = payload.get('operation')
        else:
            if self.mode != 'auto':
                self.auto_at = None
            self.mode, self.reason = 'auto', 'Navigation control'
            self.manual_velocity = (0.0, 0.0)
        return True

    def _validated_control_velocity(self, payload, ros_now_ns):
        """Read-only admission, also used before a busy output-lock stop.

        State and sequence are checked again when the packet is committed.
        A preliminary stop never grants control or renews any lease.
        """
        if not isinstance(payload, dict):
            raise ValueError('Control packet must be a JSON object')
        source, seq, mode = (payload.get('source_id'), payload.get('seq'),
                             payload.get('mode'))
        stamp = payload.get('sent_at_ns')
        if not isinstance(source, str) or not source or len(source) > 128:
            raise ValueError('Control source_id must be a nonempty string (max 128 characters)')
        if type(seq) is not int or seq < 0:
            raise ValueError('Control seq must be a nonnegative integer')
        if not isinstance(mode, str) or mode not in self.MODES:
            raise ValueError('Unknown control mode')
        if source in self.SAFETY_REASONS and mode != 'estop':
            raise ValueError('Safety watchdog sources may only request estop')
        if type(stamp) is not int or stamp < 0 or type(ros_now_ns) is not int:
            raise ValueError('Control sent_at_ns must be a nonnegative integer')
        safety_estop = source in self.SAFETY_REASONS and mode == 'estop'
        # A delayed safety watchdog must still latch a stop. Only its estop
        # packets bypass the old-stamp limit; future stamps and all other
        # control packets retain the same one-second limit.
        if (stamp-ros_now_ns > 1_000_000_000
                or (ros_now_ns-stamp > 1_000_000_000 and not safety_estop)):
            raise ValueError('Control timestamp differs by over 1s; check PC/robot clock sync and packet delay')
        if source in self.retired_sources:
            raise ValueError('Control source was retired; reconnect the current bridge')
        if seq <= self.sequences.get(source, -1):
            raise ValueError('Replayed or out-of-order control sequence')
        velocity = self._velocity(payload.get('linear', 0.0), payload.get('angular', 0.0))
        if mode == 'localization' and velocity[0] != 0.0:
            raise ValueError('Localization control permits yaw only; linear must be zero')
        if safety_estop:
            return velocity
        if self.estop_latched and mode not in {'reset', 'estop'}:
            raise ValueError('Emergency stop is latched; explicit safety reset required')
        if self.mode == 'manual' and mode == 'localization':
            raise ValueError('Manual joystick lease has priority; acknowledge hold before localization')
        if (self.source_id is not None and source != self.source_id
                and (self.mode == 'manual' and mode != 'estop'
                     or self.mode == 'localization' and mode not in {'manual', 'estop'})):
            raise ValueError('Another operator owns a live '+self.mode+' lease; release it or wait for expiry')
        return velocity

    def output(self):
        self._expire()
        if self.estop_latched:
            return 0.0, 0.0
        if self.mode == 'auto' and self.scan_guard_hold:
            return 0.0, 0.0
        velocity = (0.0, 0.0)
        if self.mode in {'manual', 'localization'}:
            velocity = self.manual_velocity
        elif (self.mode == 'auto' and self.auto_at is not None
                and self.clock()-self.auto_at < self.auto_timeout):
            velocity = self.auto_velocity
        factor, auto_allowed, allow_rotation, _ = self._localization_policy()
        if self.mode == 'manual' and self.manual_operation == 'marker_alignment':
            # This is mission-driven observation, not the operator's bootstrap
            # localization exception. Even compatibility runtimes must have a
            # fresh, autonomous-ready health report before rotating for a tag.
            health_fresh = (self.localization_at is not None
                            and self.clock()-self.localization_at <= self.localization_health_timeout)
            if (not health_fresh or not auto_allowed
                    or self.localization_health.get('autonomous_allowed') is not True or factor <= 0.
                    or velocity[0] != 0.):
                return 0.0, 0.0
            angular = max(-self.marker_alignment_rotation_limit,
                          min(self.marker_alignment_rotation_limit, velocity[1]))
            return 0.0, angular*factor
        if (self.mode == 'localization'
                or self.mode == 'manual' and self.manual_operation == 'localization'):
            if allow_rotation and velocity[0] == 0.0:
                return 0.0, max(-self.localization_rotation_limit,
                                min(self.localization_rotation_limit, velocity[1]))
            return 0.0, 0.0
        if self.mode == 'auto' and not auto_allowed:
            return 0.0, 0.0
        return velocity[0]*factor, velocity[1]*factor

    def status(self):
        self._expire()
        factor, auto_allowed, allow_localization, health_reason = self._localization_policy()
        linear, angular = self.output()
        return {'mode': self.mode, 'available': True, 'reason': self.reason,
                'source_id': self.source_id, 'estop_latched': self.estop_latched,
                'scan_guard_hold': self.scan_guard_hold,
                'scan_guard_reason': self.scan_guard_reason,
                'scan_guard_code': self.scan_guard_code,
                'scan_guard_support': self.scan_guard_support,
                'scan_guard_metrics': dict(self.scan_guard_metrics),
                'control_seq': self.sequences.get(self.source_id, -1),
                'allow_localization': allow_localization,
                'auto_source_id': self.auto_source_id,
                'auto_rejected_reason': self.auto_rejected_reason,
                'control_rejected_reason': self.control_rejected_reason,
                **{name: getattr(self, name) for name in (
                    'control_received', 'control_accepted', 'control_rejected',
                    'auto_received', 'auto_accepted', 'auto_rejected',
                    'control_by_origin', 'auto_by_origin',
                    'last_control', 'last_control_rejection', 'last_auto', 'last_auto_rejection')},
                'output_owner': 'manual' if self.mode == 'manual' else self.mode,
                'localization_speed_factor': factor,
                'autonomous_allowed': auto_allowed,
                'localization_reason': health_reason,
                'output_linear': linear, 'output_angular': angular}


# Keep the arbitration core importable for non-ROS unit tests as well.
try:
    import rclpy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
except ImportError:
    Node = object


class MotionMuxNode(Node):
    OWNERSHIP_MAX_AGE = .5
    GUARD_COMPUTE_BUDGET = .05

    def __init__(self, context=None, use_sim_time=None):
        overrides = []
        if use_sim_time is not None:
            from rclpy.parameter import Parameter
            overrides.append(Parameter('use_sim_time', value=bool(use_sim_time)))
        super().__init__('motion_mux', context=context, parameter_overrides=overrides)
        # The control subscription runs in a different executor group from
        # navigation, graph queries, and status writes. Serialize the small
        # arbitration/publish critical section across those groups.
        self._state_lock = threading.RLock()
        self._braking_feedback_lock = threading.Lock()
        self._output_publish_lock = threading.Lock()
        self._control_ingress_lock = threading.Lock()
        self._deferred_auto_control = None
        self._urgent_stop_event = threading.Event()
        self._urgent_stop_generation = 0
        self._status_write_lock = threading.Lock()
        self._closing = False
        for name, default in (
                ('manual_timeout', .5), ('auto_timeout', .5),
                ('max_linear_velocity', .25), ('max_angular_velocity', 1.0),
                ('max_auto_age_seconds', .35),
                ('auto_future_tolerance_seconds', .25),
                ('require_localization_health', False),
                ('localization_health_timeout', 1.0),
                ('localization_warning_speed_factor', .5),
                ('localization_rotation_limit', .25),
                ('marker_alignment_rotation_limit', .15),
                ('auto_turn_slowdown_enabled', False),
                ('auto_turn_start_angular', .08),
                ('auto_turn_full_angular', .35),
                ('auto_turn_entry_linear_velocity', .15),
                ('auto_turn_min_linear_velocity', .10),
                ('gentle_start_enabled', False),
                ('gentle_start_linear_acceleration', .15),
                ('gentle_start_angular_acceleration', .80),
                ('gentle_start_max_publish_gap', .50),
                ('startup_estop', False), ('allow_topic_reset', True),
                ('allow_legacy_auto_input', True), ('enable_timing_diagnostics', False),
                ('near_scan_guard_enabled', True),
                ('scan_guard_robot_radius', .22),
                ('scan_guard_base_frame', 'base_footprint'),
                ('scan_guard_horizon', .5), ('scan_guard_sensor_timeout', 1.0),
                ('scan_guard_fixed_frame', 'odom'), ('scan_guard_pose_timeout', .25)):
            self.declare_parameter(name, default)
        for name, default in (('braking_wheel_separation', .287),
                              ('braking_wheel_deceleration', .15),
                              ('braking_reaction_seconds', .25),
                              ('braking_stop_reserve', .02)):
            self.declare_parameter(name, default)
        self.core = MotionMuxCore(**{
            name: self.get_parameter(name).value for name in (
                'manual_timeout', 'auto_timeout', 'max_linear_velocity',
                'max_angular_velocity', 'max_auto_age_seconds',
                'auto_future_tolerance_seconds',
                'require_localization_health', 'localization_health_timeout',
                'localization_warning_speed_factor', 'localization_rotation_limit',
                'marker_alignment_rotation_limit',
                'auto_turn_slowdown_enabled', 'auto_turn_start_angular',
                'auto_turn_full_angular', 'auto_turn_entry_linear_velocity',
                'auto_turn_min_linear_velocity',
                'startup_estop')})
        self.allow_topic_reset = self.get_parameter('allow_topic_reset').value
        self.command_ramp = CommandRamp(
            enabled=self.get_parameter('gentle_start_enabled').value,
            linear_acceleration=self.get_parameter('gentle_start_linear_acceleration').value,
            angular_acceleration=self.get_parameter('gentle_start_angular_acceleration').value,
            max_publish_gap=self.get_parameter('gentle_start_max_publish_gap').value)
        self.last_published_output = (0., 0.)
        brake_parameters = dict(
            wheel_separation=self.get_parameter('braking_wheel_separation').value,
            wheel_deceleration=self.get_parameter('braking_wheel_deceleration').value,
            reaction_seconds=self.get_parameter('braking_reaction_seconds').value,
            stop_reserve=self.get_parameter('braking_stop_reserve').value)
        self.braking_envelope = BrakingEnvelope(**brake_parameters)
        # Two 20 Hz output periods after a fresh measured feedback sample.
        # The earlier command cap uses the full configured reaction allowance.
        self.measured_braking_envelope = BrakingEnvelope(
            **{**brake_parameters, 'reaction_seconds': min(.10, brake_parameters['reaction_seconds']),
               # The command cap owns the 2cm stopping allowance. Residual
               # motion checks physical body contact, avoiding a fictitious
               # 2cm movement when stationary feedback has tiny age bounds.
               'stop_reserve': 0.})
        self._braking_feedback = None
        self._braking_feedback_revision = 0
        self._braking_last_stamp_ns = -1
        self._braking_scale = 1.
        self._braking_cloud = None
        self.near_scan_guard = NearScanOutlierGuard()
        self.near_scan_guard_last_stamp_ns = -1
        self._scan_guard_enabled = bool(self.get_parameter('near_scan_guard_enabled').value)
        self._scan_guard_radius = float(self.get_parameter('scan_guard_robot_radius').value)
        self._scan_guard_base_frame = str(self.get_parameter('scan_guard_base_frame').value)
        self._scan_guard_horizon = float(self.get_parameter('scan_guard_horizon').value)
        self._scan_guard_sensor_timeout = float(self.get_parameter('scan_guard_sensor_timeout').value)
        if (not self._scan_guard_base_frame
                or not math.isfinite(self._scan_guard_radius) or self._scan_guard_radius <= 0.
                or not 0. < self._scan_guard_horizon <= 2.
                or not 0. < self._scan_guard_sensor_timeout <= 1.):
            raise ValueError('Invalid scan guard geometry or freshness limits')
        from tf2_ros import Buffer, TransformListener
        self.scan_guard_tf_buffer = Buffer()
        self.scan_guard_tf_listener = TransformListener(self.scan_guard_tf_buffer, self)
        self._scan_guard_tf_cache = {}
        self._scan_guard_sample = None
        self._scan_guard_pending = None
        self._scan_guard_fixed_frame = str(self.get_parameter('scan_guard_fixed_frame').value)
        self._scan_guard_pose_timeout = float(self.get_parameter('scan_guard_pose_timeout').value)
        if not self._scan_guard_fixed_frame or not 0. < self._scan_guard_pose_timeout <= .25:
            raise ValueError('Invalid scan guard odometry frame or pose timeout')
        if self._scan_guard_enabled:
            self.core.scan_guard_hold = True
            self.core.scan_guard_code = 'WAIT_SCAN'
            self.core.scan_guard_reason = 'Waiting for raw LiDAR and verified laser mounting TF'
        self._timing = (RuntimeTiming(MUX_METRICS, warning=self.get_logger().warning)
                        if self.get_parameter('enable_timing_diagnostics').value else None)
        self._guard_timing = None
        self.output_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.output_available = False
        self.output_conflict = 'Checking exclusive /cmd_vel ownership'
        self.output_publishers = 0
        self._ownership_checked_at = -math.inf
        self._ownership_completed_at = -math.inf
        self._ownership_query_inflight = False
        durable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(
            String, '/smart_vision_amr/motion_status', durable)
        self.control_pub = self.create_publisher(
            String, '/smart_vision_amr/manual_control', 10)
        self.service_source = 'mux-service:'+uuid.uuid4().hex
        self.service_seq = 0
        self.auto_started_ns = self.get_clock().now().nanoseconds
        self._health_sent_ns = -1
        self.status_sequence = 0
        self._status_published_at = -math.inf
        self._status_transition = None
        self.pending_reset = None
        self._reset_services_active = 0
        self.last_zero_published_at = None
        # Stamped velocity and control leases are only useful while fresh.
        # Prefer the latest packet over reliable DDS queue replay. Emergency
        # stop/reset services and the independent safety publishers stay reliable.
        latest_control = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                    durability=DurabilityPolicy.VOLATILE)
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
        # /clock uses the default group. Never put a callback which waits for
        # the geometric output lock in front of that clock subscription.
        # All ordinary callbacks sharing _state_lock also share a group:
        # they cannot fill four executor workers waiting for one slow guard.
        self.output_group = MutuallyExclusiveCallbackGroup()
        self.navigation_input_group = self.output_group
        self.health_input_group = self.output_group
        self.braking_input_group = MutuallyExclusiveCallbackGroup()
        if self.get_parameter('allow_legacy_auto_input').value:
            self.create_subscription(Twist, '/smart_vision_amr/navigation_cmd_vel',
                                     self.on_auto, 1, callback_group=self.navigation_input_group)
        self.create_subscription(String, '/smart_vision_amr/navigation_control',
                                 self.on_navigation_control, latest_control,
                                 callback_group=self.navigation_input_group)
        self.create_subscription(String, '/smart_vision_amr/localization_health',
                                 self.on_localization_health, latest_control,
                                 callback_group=self.health_input_group)
        self.scan_guard_group = self.output_group
        if self.get_parameter('near_scan_guard_enabled').value:
            self.create_subscription(LaserScan, '/scan', self.on_scan_guard,
                                     latest_control,
                                     callback_group=self.scan_guard_group)
            self.create_subscription(Odometry, '/odom', self.on_braking_odom,
                                     latest_control,
                                     callback_group=self.braking_input_group)
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.status_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(String, '/smart_vision_amr/manual_control',
                                 self.on_control, latest_control,
                                 callback_group=self.control_group)
        # ROS timers stop when simulation time pauses. Dead-man watchdogs
        # must continue to run even then, so use a steady clock explicitly.
        from rclpy.clock import Clock, ClockType
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        # Reentrant services yield while steady timers/subscriptions process the
        # reset intent and independent sensor watchdogs can veto it.
        self.safety_group = ReentrantCallbackGroup()
        self.create_service(Trigger, '/smart_vision_amr/emergency_stop',
                            self.on_emergency_stop, callback_group=self.safety_group)
        self.create_service(Trigger, '/smart_vision_amr/reset_emergency_stop',
                            self.on_reset_emergency_stop, callback_group=self.safety_group)
        # Reset verification needs 25 ms polling only during its short
        # handshake. An idle timer otherwise needlessly competes for workers.
        self.reset_timer = self.create_timer(
            .025, self.advance_reset, clock=self.steady_clock,
            callback_group=self.safety_group)
        self.reset_timer.cancel()
        # The separate status group refreshes graph ownership. A graph lookup
        # must not delay the output group and its latest command/scan inputs.
        self.create_timer(.05, lambda: self.publish_output(refresh_ownership=False), clock=self.steady_clock,
                          callback_group=self.output_group)
        self.create_timer(.1, self.publish_status, clock=self.steady_clock,
                          callback_group=self.status_group)
        self.publish_status()

    def _state_guard(self):
        # Small test harnesses construct only the fields they exercise.
        return getattr(self, '_state_lock', nullcontext())

    @contextmanager
    def _try_state_guard(self, blocking=False):
        lock = getattr(self, '_state_lock', None)
        acquired = lock is None or lock.acquire(blocking=blocking)
        try:
            yield acquired
        finally:
            if acquired and lock is not None:
                lock.release()

    def _latch_urgent_zero(self):
        """Stop before waiting for geometry; grant no new control authority.

        Final normal publication uses this same short lock, so an old computed
        command cannot follow this zero while the stop awaits core admission.
        """
        lock = getattr(self, '_output_publish_lock', None)
        if lock is None:  # Minimal non-ROS arbitration fixtures.
            return None
        with lock:
            self._urgent_stop_generation += 1
            token = self._urgent_stop_generation
            self._urgent_stop_event.set()
            completed = getattr(self, '_ownership_completed_at', -math.inf)
            if self.output_available and 0. <= time.monotonic()-completed <= self.OWNERSHIP_MAX_AGE:
                self.output_pub.publish(Twist())
                self.last_published_output = (0., 0.)
                self.last_output_was_zero = True
                self.last_zero_published_at = time.monotonic()
            return token

    def _finish_urgent_zero(self, token):
        if token is not None:
            with self._output_publish_lock:
                if token == self._urgent_stop_generation:
                    self._urgent_stop_event.clear()

    def _consume_deferred_auto_locked(self):
        """Admit a real resume after contention, preserving its original age.

        This mailbox never grants motion. A newer accepted stop/manual packet
        clears it, and normal core timestamp/sequence/authority checks apply.
        """
        with getattr(self, '_control_ingress_lock', nullcontext()):
            item = getattr(self, '_deferred_auto_control', None)
            self._deferred_auto_control = None
        if item is None:
            return
        payload, received_at = item
        if getattr(self, '_closing', False) or (getattr(self, 'pending_reset', None) or {}).get('finishing'):
            self.core.reject_control(payload, 'Safety transition pending; explicit resume required')
        elif not 0. <= time.monotonic()-received_at <= 1.:
            self.core.reject_control(payload, 'Control admission delayed by over 1s; explicit resume required')
        else:
            self.core.receive_control(payload, self.get_clock().now().nanoseconds, origin='topic')

    def on_auto(self, message):
        with self._state_guard():
            if not getattr(self, '_closing', False):
                self.core.receive_auto(message.linear.x, message.angular.z)

    def on_navigation_control(self, message):
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            with self._state_guard():
                self.core.reject_auto(None, 'Malformed autonomous JSON')
            return
        # Do not accept a packet already in transit when this mux restarted.
        with self._state_guard():
            if getattr(self, '_closing', False):
                return
            if (isinstance(payload, dict) and type(payload.get('sent_at_ns')) is int
                    and payload['sent_at_ns'] < self.auto_started_ns):
                self.core.reject_auto(payload, 'Autonomous command predates mux startup')
            else:
                self.core.receive_stamped_auto(payload, self.get_clock().now().nanoseconds)

    def on_localization_health(self, message):
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        with self._state_guard():
            if getattr(self, '_closing', False):
                return
            stamp = payload.get('sent_at_ns')
            if stamp is not None or self.core.require_localization_health:
                now_ns = self.get_clock().now().nanoseconds
                if (type(stamp) is not int or stamp < self._health_sent_ns
                        or stamp < 0 or stamp > now_ns+50_000_000
                        or now_ns-stamp > int(self.core.localization_health_timeout*1e9)):
                    return
                self._health_sent_ns = stamp
            accepted = self.core.receive_localization_health(payload)
            # Healthy heartbeats need no new geometric computation. The 20Hz
            # output timer will consume them; a veto still publishes zero now.
            if accepted and self.core.output() == (0., 0.):
                self._publish_output_locked(urgent_zero=True)

    def _scan_guard_extrinsic(self, frame):
        """Resolve fixed mounting once; never infer a missing laser frame."""
        if not frame:
            return None
        base = self._scan_guard_base_frame
        if frame == base:
            return (0., 0., 0.)
        cached = self._scan_guard_tf_cache.get(frame)
        if cached is not None:
            return cached
        try:
            transform = self.scan_guard_tf_buffer.lookup_transform(base, frame, rclpy.time.Time())
            p, q = transform.transform.translation, transform.transform.rotation
            values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
            norm = q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w
            if not all(math.isfinite(value) for value in values) or not .8 <= norm <= 1.2:
                return None
            scale = math.sqrt(norm)
            x, y, z, w = q.x/scale, q.y/scale, q.z/scale, q.w/scale
            # A non-horizontal scan plane needs a 3-D model, not a guessed yaw.
            if (abs(2*(x*z-w*y)) > .1 or abs(2*(y*z+w*x)) > .1
                    or 1.-2*(x*x+y*y) < .99):
                return None
            result = (p.x, p.y, math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        except Exception:
            return None
        self._scan_guard_tf_cache[frame] = result
        return result

    def _scan_guard_pose(self, stamp_ns=0):
        """Nonblocking continuous-odometry pose, at acquisition or latest TF."""
        self._scan_guard_pose_lookup = {'failure_kind': '', 'lookup_error': ''}
        try:
            transform = self.scan_guard_tf_buffer.lookup_transform(
                self._scan_guard_fixed_frame, self._scan_guard_base_frame,
                rclpy.time.Time(nanoseconds=stamp_ns))
            p, q = transform.transform.translation, transform.transform.rotation
            norm = q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w
            if (not all(math.isfinite(v) for v in (p.x, p.y, q.x, q.y, q.z, q.w))
                    or not .8 <= norm <= 1.2):
                self._scan_guard_pose_lookup = {
                    'failure_kind': 'geometry', 'lookup_error': 'Invalid odometry transform geometry'}
                return None
            scale = math.sqrt(norm)
            x, y, z, w = q.x/scale, q.y/scale, q.z/scale, q.w/scale
            if (abs(2*(x*z-w*y)) > .1 or abs(2*(y*z+w*x)) > .1
                    or 1.-2*(x*x+y*y) < .99):
                self._scan_guard_pose_lookup = {
                    'failure_kind': 'geometry', 'lookup_error': 'Non-planar odometry transform'}
                return None
            stamp = transform.header.stamp
            return ((p.x, p.y, math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))),
                    stamp.sec*1_000_000_000+stamp.nanosec)
        except Exception as exc:
            self._scan_guard_pose_lookup = {
                'failure_kind': 'lookup',
                'lookup_error': (type(exc).__name__+': '+str(exc))[:200]}
            return None

    def _scan_guard_current_pose(self, now_ns):
        """Interpolate future TF; bridge only brief lookup gaps, never bad poses.

        Cached poses keep their original ROS timestamp. Neither failed lookups
        nor reuse renew the age budget used for the directional motion bound.
        """
        self._scan_guard_pose_lookup = {}
        pose = self._scan_guard_pose()
        lookup = dict(getattr(self, '_scan_guard_pose_lookup', {}))
        source, error = 'latest', lookup.get('lookup_error', '')
        invalid = lookup.get('failure_kind') == 'geometry'
        unavailable_age = None
        future_lead = 0.
        uncertainty_age = None
        source_stamp_ns = pose[1] if pose is not None else None
        received_at = time.monotonic()
        if pose is not None:
            age = (now_ns-pose[1])/1.e9
            future_lead = max(0., -age)
            if age >= self._scan_guard_pose_timeout or age < -self._scan_guard_pose_timeout:
                # A stale or far-future stream cannot be hidden by older data.
                self._scan_guard_pose_cache = None
                self._scan_guard_pose_diagnostic = {
                    'pose_source': 'stale' if age >= 0. else 'future_out_of_bounds',
                    'pose_age_seconds': age, 'lookup_error': error,
                    'future_lead_seconds': future_lead,
                    'pose_uncertainty_seconds': None}
                return None
            if age < -.05:
                source = 'interpolated'
                pose = self._scan_guard_pose(now_ns)
                lookup = dict(getattr(self, '_scan_guard_pose_lookup', {}))
                error = lookup.get('lookup_error', '')
                invalid = lookup.get('failure_kind') == 'geometry'
        if pose is None:
            cached = getattr(self, '_scan_guard_pose_cache', None)
            if invalid:
                self._scan_guard_pose_cache = None
                source = 'invalid_geometry'
            elif cached is not None:
                cached_pose, received_at, future_lead, source_stamp_ns = cached
                age = (now_ns-cached_pose[1])/1.e9
                unavailable_age = age
                if (-.05 <= age < self._scan_guard_pose_timeout
                        and time.monotonic()-received_at < self._scan_guard_pose_timeout):
                    pose, source = cached_pose, 'cached'
            if pose is None:
                source = 'invalid_geometry' if invalid else 'lookup_unavailable'
        else:
            previous = getattr(self, '_scan_guard_pose_cache', None)
            if previous is not None and previous[3] == source_stamp_ns:
                # Re-reading the same TF is not new motion evidence, even if
                # a catching-up /clock allows a newer interpolated pose.
                received_at = previous[1]
                future_lead = max(future_lead, previous[2])
            self._scan_guard_pose_cache = (pose, received_at, future_lead, source_stamp_ns)
        if pose is not None:
            # A future TF may be genuine prediction or a delayed local /clock
            # callback. Preserve that uncertainty after interpolation and
            # throughout cache reuse instead of declaring an exact age of zero.
            uncertainty_age = max(abs((now_ns-pose[1])/1.e9),
                                  future_lead+max(0., time.monotonic()-received_at))
        self._scan_guard_pose_diagnostic = {
            'pose_source': source, 'lookup_error': error,
            'pose_age_seconds': (now_ns-pose[1])/1.e9 if pose is not None else unavailable_age,
            'future_lead_seconds': future_lead,
            'pose_uncertainty_seconds': uncertainty_age}
        return pose

    def _validate_pending_scan_locked(self):
        pending = getattr(self, '_scan_guard_pending', None)
        if pending is None:
            return
        pose = self._scan_guard_pose(pending['stamp_ns'])
        self._scan_guard_acquisition_diagnostic = dict(
            getattr(self, '_scan_guard_pose_lookup', {}))
        if pose is not None:
            pending['scan_pose'] = pose[0]
            self._scan_guard_sample = pending
            self._scan_guard_pending = None

    def on_scan_guard(self, scan):
        """Keep temporal evidence intact, then guard the actual chassis sweep."""
        received_at = time.monotonic()
        stamp_ns = scan.header.stamp.sec*1_000_000_000+scan.header.stamp.nanosec
        now_ns = self.get_clock().now().nanoseconds
        if (stamp_ns <= 0 or stamp_ns <= self.near_scan_guard_last_stamp_ns
                or not -.25e9 <= now_ns-stamp_ns < 1e9):
            return
        decision = self.near_scan_guard.classify(scan, stamp_ns)
        extrinsic = self._scan_guard_extrinsic(scan.header.frame_id) if decision.accepted else None
        fault = ('INVALID_SCAN' if not decision.accepted else
                 'TF_UNAVAILABLE' if extrinsic is None else '')
        points = scan_points_in_base(scan, extrinsic) if not fault else ()
        inside = frozenset(index for index, x, y in points
                           if math.hypot(x, y) <= self._scan_guard_radius)
        token = (self._latch_urgent_zero() if self.core.mode == 'auto'
                 and (fault or inside) else None)
        with self._state_guard():
            if getattr(self, '_closing', False):
                self._finish_urgent_zero(token)
                return
            if decision.accepted:
                self.near_scan_guard_last_stamp_ns = stamp_ns
            sample = {'points': points, 'decision': decision, 'stamp_ns': stamp_ns,
                      'received_at': received_at, 'fault': fault}
            if fault:
                self._scan_guard_sample, self._scan_guard_pending = sample, None
            else:
                # A scan stamp can lead the newest odometry TF by a few ms.
                # Keep a previous validated scan alive while waiting for the
                # interpolation, rather than injecting a zero at every scan.
                self._scan_guard_pending = sample
            # Admission never repeats the complete stopping-envelope proof.
            # The 20 Hz output timer validates this newest evidence before its
            # next nonzero output. Keep a prior hold until that proof succeeds.
            self._guard_proof = None
            self._scan_guard_checks = None
            if fault or inside:
                self.core.scan_guard_hold = True
                self.core.scan_guard_code = fault or 'BODY_INTERSECTION'
                self.core.scan_guard_support = (
                    'unconfirmed' if inside & decision.unconfirmed else
                    'confirmed' if inside & decision.confirmed else 'raw' if inside else '')
                self.core.scan_guard_reason = (
                    'Invalid raw LiDAR scan; measured clearance is unverified' if fault == 'INVALID_SCAN' else
                    'LiDAR mounting TF unavailable; chassis sweep is unverified' if fault else
                    'Raw LiDAR return intersects current chassis')
                self.core.scan_guard_metrics = {'collision_indices': sorted(inside)[:8],
                    'inside_count': len(inside), 'pending_scan': self._scan_guard_pending is not None}
                if self.core.mode == 'auto':
                    self._publish_output_locked(urgent_zero=True)
            self._finish_urgent_zero(token)

    def on_braking_odom(self, message):
        """Keep original source/receipt times; duplicates cannot renew speed evidence."""
        received_at = time.monotonic()
        # This callback never waits behind scan/trajectory computation. A
        # dedicated latest-only DDS queue plus this tiny lock coalesces bursts.
        with getattr(self, '_braking_feedback_lock', nullcontext()):
            stamp = message.header.stamp
            stamp_ns = stamp.sec*1000000000+stamp.nanosec
            now_ns = self.get_clock().now().nanoseconds
            if now_ns < getattr(self, '_braking_last_ros_ns', now_ns):
                self._braking_feedback = None
                self._braking_feedback_revision = getattr(self, '_braking_feedback_revision', 0)+1
                self._braking_last_stamp_ns = -1
            self._braking_last_ros_ns = now_ns
            previous = getattr(self, '_braking_feedback', None)
            watermark = getattr(self, '_braking_last_stamp_ns', -1)
            v, w = message.twist.twist.linear.x, message.twist.twist.angular.z
            valid = (message.header.frame_id == self._scan_guard_fixed_frame
                     and message.child_frame_id == self._scan_guard_base_frame
                     and stamp_ns > 0 and -.25 <= (now_ns-stamp_ns)/1.e9 < .25
                     and math.isfinite(v) and math.isfinite(w)
                     and abs(v) <= self.core.max_linear_velocity+.15
                     and abs(w) <= self.core.max_angular_velocity+.5)
            if not valid or stamp_ns < watermark:
                self._braking_feedback = None
                self._braking_feedback_revision = getattr(self, '_braking_feedback_revision', 0)+1
                return
            if stamp_ns == watermark:
                if previous is not None and (v, w) != previous['velocity']:
                    self._braking_feedback = None
                    self._braking_feedback_revision = getattr(self, '_braking_feedback_revision', 0)+1
                return
            self._braking_last_stamp_ns = stamp_ns
            self._braking_feedback = {'velocity': (float(v), float(w)),
                'stamp_ns': stamp_ns, 'received_at': received_at}

    def _limit_braking_output_locked(self, linear, angular):
        cloud = getattr(self, '_braking_cloud', None)
        if cloud is None or self.core.mode != 'auto':
            return linear, angular
        points, uncertainty = cloud
        scale, _ = self.braking_envelope.limit(points, linear, angular,
            radius=self._scan_guard_radius, **uncertainty)
        return linear*scale, angular*scale

    def _refresh_scan_guard_locked(self, extra_twist=None):
        if not getattr(self, '_scan_guard_enabled', False):
            return
        self._validate_pending_scan_locked()
        self._scan_guard_checks = None
        self._guard_proof = None
        guard_started_at = time.monotonic()
        self._braking_cloud = None
        self._braking_scale = 1.
        sample = self._scan_guard_sample
        code, support = 'WAIT_SCAN', ''
        metrics = {'scan_age_seconds': None, 'pose_age_seconds': None,
                   'lookup_error': '', 'pose_source': 'not_requested'}
        reason = 'Waiting for raw LiDAR and verified acquisition-time odometry TF'
        if sample is None and getattr(self, '_scan_guard_pending', None) is not None:
            metrics.update(
                scan_age_seconds=(self.get_clock().now().nanoseconds-
                                  self._scan_guard_pending['stamp_ns'])/1.e9,
                pose_source='waiting_acquisition_pose',
                lookup_error=getattr(self, '_scan_guard_acquisition_diagnostic', {}).get('lookup_error', ''))
        if sample is not None:
            now_ns = self.get_clock().now().nanoseconds
            age = (now_ns-sample['stamp_ns'])/1.e9
            code = sample['fault']
            latest_pose = None if code else self._scan_guard_current_pose(now_ns)
            pose_age = (now_ns-latest_pose[1])/1.e9 if latest_pose is not None else math.inf
            metrics['scan_age_seconds'] = age
            if not code:
                metrics.update(getattr(self, '_scan_guard_pose_diagnostic', {}))
            elif code == 'TF_UNAVAILABLE':
                metrics.update(pose_source='mounting_unavailable', lookup_error='Laser mounting TF unavailable')
            if code == 'TF_UNAVAILABLE':
                reason = 'LiDAR mounting TF unavailable; chassis sweep is unverified'
            elif code == 'INVALID_SCAN':
                reason = 'Invalid raw LiDAR scan; measured clearance is unverified'
            elif (not -.25 <= age < self._scan_guard_sensor_timeout
                  or time.monotonic()-sample['received_at'] >= self._scan_guard_sensor_timeout):
                code, reason = 'STALE_SCAN', 'Raw LiDAR scan is stale; measured clearance is unverified'
            elif latest_pose is None or not -.05 <= pose_age < self._scan_guard_pose_timeout:
                code, reason = 'POSE_UNAVAILABLE', 'Fresh continuous odometry TF unavailable for scan compensation'
            else:
                requested = ((self.core.auto_velocity) if self.core.mode == 'auto'
                    and self.core.auto_at is not None
                    and self.core.clock()-self.core.auto_at < self.core.auto_timeout else (0., 0.))
                twists = {requested, getattr(self, 'last_published_output', (0., 0.))}
                if extra_twist is not None:
                    twists.add(extra_twist)
                # Prepare one immutable cloud per proof, rather than converting
                # every beam again for each candidate and final emitted curve.
                points = np.asarray(rebase_scan_points(
                    sample['points'], sample['scan_pose'], latest_pose[0]), dtype=float).reshape((-1, 3))
                # The full scan age is compensated exactly above. Bound only
                # the much shorter motion since the latest available pose:
                # longitudinal speed, lateral turning drift and heading error.
                dt = max(abs(pose_age), metrics.get('pose_uncertainty_seconds') or 0.) + self.GUARD_COMPUTE_BUDGET
                uncertainty = {
                    'longitudinal_uncertainty': self.core.max_linear_velocity*dt,
                    'lateral_uncertainty': .5*self.core.max_linear_velocity*self.core.max_angular_velocity*dt*dt,
                    'yaw_uncertainty': self.core.max_angular_velocity*dt}
                checks = [(points, sample['decision'], uncertainty)]
                pending = getattr(self, '_scan_guard_pending', None)
                pending_tf_invalid = (pending is not None and
                    getattr(self, '_scan_guard_acquisition_diagnostic', {}).get('failure_kind') == 'geometry')
                brake_points, brake_uncertainty = points, uncertainty
                if pending is not None:
                    # A newly measured imminent collision still stops on the
                    # first scan, even while its stamped TF is being awaited.
                    # These points remain in their verified mounting's BASE
                    # frame at acquisition. Never shift them using the older
                    # scan's pose or claim an unobserved acquisition transform.
                    # Enclose possible differential-drive motion since that
                    # source/first receipt, including heading of the stop arc.
                    pending_dt = max(abs((now_ns-pending['stamp_ns'])/1.e9),
                                     max(0., time.monotonic()-pending['received_at'])) + self.GUARD_COMPUTE_BUDGET
                    pending_bounds = {
                        'longitudinal_uncertainty': self.core.max_linear_velocity*pending_dt,
                        'lateral_uncertainty': .5*self.core.max_linear_velocity*self.core.max_angular_velocity*pending_dt*pending_dt,
                        'yaw_uncertainty': self.core.max_angular_velocity*pending_dt}
                    pending_points = np.asarray(pending['points'], dtype=float).reshape((-1, 3))
                    checks.append((pending_points, pending['decision'], pending_bounds))
                    # One union and the containing uncertainty intervals mean
                    # the chosen common v/w scale is checked against BOTH
                    # clouds. Taking the minimum of two independently selected
                    # scales would be unsafe for non-monotone curved sweeps.
                    brake_points = np.concatenate((points, pending_points), axis=0)
                    brake_uncertainty = {name: max(uncertainty[name], pending_bounds[name])
                                         for name in uncertainty}
                    metrics.update(braking_pending_scan=True,
                        braking_pending_acquisition_tf_verified=False,
                        braking_pending_source_stamp_ns=pending['stamp_ns'],
                        braking_pending_motion_bound_seconds=pending_dt)
                self._scan_guard_checks = checks
                self._guard_proof = {'started_at': guard_started_at, 'started_ros_ns': now_ns, 'sample': sample,
                    'pending': pending, 'pose_stamp_ns': latest_pose[1], 'feedback': None}
                brake_hold = pending_tf_invalid
                # Test harnesses without a brake model exercise the original
                # raw-sweep path; production always constructs both models.
                braking = getattr(self, 'braking_envelope', None)
                if braking is not None and self.core.mode == 'auto':
                    self._braking_cloud = (brake_points, brake_uncertainty)
                    self._braking_scale, capped = braking.limit(
                        brake_points, *requested, radius=self._scan_guard_radius, **brake_uncertainty)
                    twists.discard(requested)
                    twists.add(tuple(value*self._braking_scale for value in requested))
                    with getattr(self, '_braking_feedback_lock', nullcontext()):
                        feedback = getattr(self, '_braking_feedback', None)
                        self._guard_proof['feedback_revision'] = getattr(self, '_braking_feedback_revision', 0)
                    self._guard_proof['feedback'] = feedback
                    feedback_age = ((now_ns-feedback['stamp_ns'])/1.e9
                                    if feedback is not None else math.inf)
                    feedback_fresh = (feedback is not None and -.25 <= feedback_age < .25
                        and 0. <= time.monotonic()-feedback['received_at'] < .25)
                    metrics.update(braking_speed_scale=self._braking_scale,
                        braking_wheel_deceleration_mps2=braking.wheel_deceleration,
                        braking_reaction_seconds=braking.reaction_seconds,
                        braking_feedback_age_seconds=feedback_age if math.isfinite(feedback_age) else None,
                        braking_command_distance_m=capped.distance)
                    if not feedback_fresh:
                        brake_hold = True
                        metrics['braking_feedback_unavailable'] = True
                    else:
                        # A delayed velocity sample is not instantaneous
                        # velocity. Bound acceleration since its source stamp
                        # using the actual output ramp's configured limit.
                        v, w = feedback['velocity']
                        # The final publish must finish within one 20Hz period;
                        # cover acceleration during that computation as well.
                        elapsed = abs(feedback_age)+self.GUARD_COMPUTE_BUDGET
                        ramp = getattr(self, 'command_ramp', None)
                        av = getattr(ramp, 'linear_acceleration', .15)
                        aw = getattr(ramp, 'angular_acceleration', .80)
                        self._guard_proof['velocity_intervals'] = (v-av*elapsed, v+av*elapsed,
                                                                  w-aw*elapsed, w+aw*elapsed)
                        measured = self.measured_braking_envelope.check_feedback(brake_points,
                            v, w, linear_delta=av*elapsed, angular_delta=aw*elapsed,
                            radius=self._scan_guard_radius, **brake_uncertainty)
                        brake_hold = brake_hold or bool(measured.collision)
                        metrics.update(braking_measured_velocity=list(feedback['velocity']),
                            braking_linear_velocity_interval=[v-av*elapsed, v+av*elapsed],
                            braking_angular_velocity_interval=[w-aw*elapsed, w+aw*elapsed],
                            braking_measured_distance_m=measured.distance,
                            braking_measured_clearance_m=(measured.clearance
                                if math.isfinite(measured.clearance) else None),
                            braking_collision_indices=sorted(measured.collision)[:8])
                results, unconfirmed, confirmed = [], set(), set()
                for cloud, decision, bounds in checks:
                    for twist in twists:
                        result = swept_body_decision(cloud, *twist,
                            radius=self._scan_guard_radius, horizon=self._scan_guard_horizon, **bounds)
                        results.append(result)
                        unconfirmed.update(result.swept & decision.unconfirmed)
                        confirmed.update(result.swept & decision.confirmed)
                inside = frozenset().union(*(result.inside for result in results))
                hits = frozenset().union(*(result.swept for result in results))
                support = 'unconfirmed' if unconfirmed else 'confirmed' if confirmed else 'raw' if hits else ''
                code = 'BODY_INTERSECTION' if inside else 'SWEPT_INTERSECTION' if hits else 'CLEAR'
                prefix = ('Unconfirmed near-range LiDAR return' if support == 'unconfirmed' else
                          'Confirmed near-range LiDAR obstacle' if support == 'confirmed' else 'LiDAR return')
                reason = (prefix+' intersects current chassis' if inside else
                          prefix+' intersects requested chassis sweep' if hits else '')
                body = min(result.minimum_body_clearance for result in results)
                sweep = min(result.minimum_swept_clearance for result in results)
                metrics.update({'robot_radius_m': self._scan_guard_radius,
                           'horizon_seconds': self._scan_guard_horizon,
                           'minimum_body_clearance_m': body if math.isfinite(body) else None,
                           'minimum_swept_clearance_m': sweep if math.isfinite(sweep) else None,
                           'collision_count': len(hits), 'inside_count': len(inside),
                           'collision_indices': sorted(hits)[:8], 'scan_age_seconds': age,
                           'pose_age_seconds': pose_age,
                           'longitudinal_uncertainty_m': uncertainty['longitudinal_uncertainty'],
                           'lateral_uncertainty_m': uncertainty['lateral_uncertainty'],
                           'pending_scan': pending is not None})
                if code == 'CLEAR' and brake_hold:
                    code = ('POSE_UNAVAILABLE' if pending_tf_invalid else 'BRAKING_FEEDBACK_UNAVAILABLE' if
                            metrics.get('braking_feedback_unavailable') else 'BRAKING_INTERSECTION')
                    reason = ('Pending scan acquisition transform has invalid geometry' if pending_tf_invalid else
                              'Fresh measured wheel velocity unavailable for stopping envelope' if
                              metrics.get('braking_feedback_unavailable') else
                              'Measured residual motion intersects its raw-LiDAR stopping envelope')
        self.core.scan_guard_code = code
        self.core.scan_guard_support = support
        self.core.scan_guard_metrics = metrics
        self.core.scan_guard_hold = code != 'CLEAR'
        self.core.scan_guard_reason = reason

    def _final_output_guard_locked(self, linear, angular):
        """Check only the emitted curve using this output's prepared snapshot.

        TF, the requested brake cap and measured-inertia tube were already
        checked once. Nothing here refreshes a source lease or redoes them.
        """
        proof = getattr(self, '_guard_proof', None)
        if proof is None or self.core.scan_guard_hold:
            return
        metrics = self.core.scan_guard_metrics
        code = None
        for cloud, decision, bounds in self._scan_guard_checks:
            result = swept_body_decision(cloud, linear, angular,
                radius=self._scan_guard_radius, horizon=self._scan_guard_horizon, **bounds)
            if result.swept:
                code = 'BODY_INTERSECTION' if result.inside else 'SWEPT_INTERSECTION'
                metrics['final_output_collision_indices'] = sorted(result.swept)[:8]
                break
        if code is not None:
            self.core.scan_guard_hold = True
            self.core.scan_guard_code = code
            self.core.scan_guard_reason = 'Final emitted curve or original evidence is no longer verified: '+code

    def _guard_evidence_current_locked(self):
        """Final cheap check, after all geometry and publication-lock waits.

        Both clocks must stay inside the advance already included in every
        pose/pending/velocity bound. No cache hit renews the original evidence.
        """
        proof = getattr(self, '_guard_proof', None)
        if proof is None or self.core.scan_guard_hold:
            return
        now, ros_ns = time.monotonic(), self.get_clock().now().nanoseconds
        duration = now-proof['started_at']
        ros_advance = (ros_ns-proof['started_ros_ns'])/1.e9
        metrics = self.core.scan_guard_metrics
        metrics.update(guard_compute_seconds=duration, guard_ros_advance_seconds=ros_advance)
        code = None
        if (not 0. <= duration <= self.GUARD_COMPUTE_BUDGET
                or not 0. <= ros_advance <= self.GUARD_COMPUTE_BUDGET):
            code = 'CONTROL_DEADLINE'
        for sample in (proof['sample'], proof['pending']):
            if sample is not None and (not -.25 <= (ros_ns-sample['stamp_ns'])/1.e9 < self._scan_guard_sensor_timeout
                    or not 0. <= now-sample['received_at'] < self._scan_guard_sensor_timeout):
                code = 'STALE_SCAN'
        if not -.05 <= (ros_ns-proof['pose_stamp_ns'])/1.e9 < self._scan_guard_pose_timeout:
            code = 'POSE_UNAVAILABLE'
        feedback = proof['feedback']
        if feedback is not None:
            metrics['braking_feedback_receipt_age_seconds'] = now-feedback['received_at']
            if (not -.25 <= (ros_ns-feedback['stamp_ns'])/1.e9 < .25
                    or not 0. <= now-feedback['received_at'] < .25):
                code = 'BRAKING_FEEDBACK_UNAVAILABLE'
            with getattr(self, '_braking_feedback_lock', nullcontext()):
                latest = getattr(self, '_braking_feedback', None)
                revision = getattr(self, '_braking_feedback_revision', 0)
            if (latest is None or revision != proof['feedback_revision']):
                code = 'BRAKING_FEEDBACK_UNAVAILABLE'
            elif latest['stamp_ns'] != feedback['stamp_ns']:
                lo_v, hi_v, lo_w, hi_w = proof['velocity_intervals']
                v, w = latest['velocity']
                if not lo_v-1.e-9 <= v <= hi_v+1.e-9 or not lo_w-1.e-9 <= w <= hi_w+1.e-9:
                    code = 'BRAKING_FEEDBACK_UNAVAILABLE'
        if code is not None:
            self.core.scan_guard_hold = True
            self.core.scan_guard_code = code
            self.core.scan_guard_reason = 'Original evidence or output compute deadline expired: '+code

    def _service_packet(self, mode):
        self.service_seq += 1
        return {'source_id': self.service_source, 'seq': self.service_seq,
                'sent_at_ns': self.get_clock().now().nanoseconds,
                'mode': mode, 'linear': 0.0, 'angular': 0.0}

    def on_emergency_stop(self, request, response):
        del request
        token = self._latch_urgent_zero()
        with self._state_guard():
            canceled = self.pending_reset['future'] if self.pending_reset is not None else None
            if canceled is not None:
                self.pending_reset = None
                self.reset_timer.cancel()
            packet = self._service_packet('estop')
            self.core.receive_control(packet, self.get_clock().now().nanoseconds, origin='service')
            with getattr(self, '_control_ingress_lock', nullcontext()):
                self._deferred_auto_control = None
            self._finish_urgent_zero(token)
            self._publish_output_locked(urgent_zero=True)
            success = bool(self.output_available and self.core.estop_latched
                           and self.last_zero_published_at is not None
                           and time.monotonic()-self.last_zero_published_at < .1)
            conflict = self.output_conflict
        if canceled is not None:
            canceled.set_result(False)
        self.control_pub.publish(String(data=json.dumps(packet)))
        self.publish_status(force=True, refresh_ownership=False)
        response.success = success
        response.message = ('Emergency stop latched; zero velocity published. '
                            'Physical standstill is not measured.' if response.success else
                            'Stop latched but exclusive /cmd_vel output is unavailable: '
                            + conflict)
        return response

    async def on_reset_emergency_stop(self, request, response):
        with self._state_guard():
            self._reset_services_active += 1
        try:
            return await self._handle_reset_emergency_stop(request, response)
        finally:
            with self._state_guard():
                self._reset_services_active -= 1

    async def _handle_reset_emergency_stop(self, request, response):
        del request
        from rclpy.task import Future
        with self._state_guard():
            if getattr(self, '_closing', False):
                response.success, response.message = False, 'Motion mux is shutting down'
                return response
            if self.pending_reset is not None:
                response.success, response.message = False, 'A reset is already pending'
                return response
            future = Future()
            packet = self._service_packet('reset')
            reset = {'future': future, 'packet': packet,
                     'at': time.monotonic(), 'applied': False}
            self.pending_reset = reset
        # Publish intent before changing state; ekf_sensor_guard correlates this
        # source with the subsequent unlatched motion_status and can re-latch.
        with self._state_guard():
            still_pending = self.pending_reset is reset
        if still_pending:
            try:
                self.control_pub.publish(String(data=json.dumps(packet)))
            except Exception:
                with self._state_guard():
                    if self.pending_reset is reset:
                        self.pending_reset = None
                response.success, response.message = False, 'Cannot publish safety reset intent'
                return response
        with self._state_guard():
            if self.pending_reset is reset:
                self.reset_timer.reset()
        accepted = await future
        response.success = bool(accepted)
        response.message = ('Emergency reset acknowledged; motion remains held. '
                            'Explicit resume required.' if accepted else
                            'Emergency reset rejected by safety or output ownership guard')
        return response

    def advance_reset(self):
        timing = getattr(self, '_timing', None)
        if timing is not None:
            timing.gap('reset_tick_gap', .025)
        applied = completed = False
        with self._state_guard():
            reset = self.pending_reset
            if reset is None or getattr(self, '_closing', False):
                self.reset_timer.cancel()
                return
            if reset.get('finishing'):
                return
            age = time.monotonic()-reset['at']
            if not reset['applied'] and age >= .1:
                reset['applied'] = True
                packet = reset['packet']
                reset['accepted'] = self.core.receive_control(
                    packet, self.get_clock().now().nanoseconds, origin='service')
                self._publish_output_locked()
                applied = True
            # Give safety guards their full veto window after the unlatched
            # status was actually published, even when that write was slow.
            if (reset.get('reported_at') is not None
                    and time.monotonic()-reset['reported_at'] >= .4):
                factor, _, _, _ = self.core._localization_policy()
                accepted = (reset.get('accepted', False) and not self.core.estop_latched
                            and self.core.source_id == self.service_source
                            and self.output_available and factor > 0)
                if not accepted:
                    self.core.estop_latched = True
                    self.core.mode, self.core.reason = 'estop', 'Safety reset rejected'
                else:
                    # Late manual/navigation packets may have arrived while guards
                    # verified the reset. Completion always returns to explicit hold.
                    self.core.mode, self.core.reason = 'hold', 'Safety reset confirmed; explicit resume required'
                    self.core.manual_velocity, self.core.auto_at = (0.0, 0.0), None
                reset['finishing'] = True
                self.reset_timer.cancel()
                self._publish_output_locked()
                completed = True
        if applied:
            try:
                self.publish_status(force=True, refresh_ownership=False)
            except Exception:
                with self._state_guard():
                    if self.pending_reset is reset:
                        self.core.estop_latched = True
                        self.core.mode, self.core.reason = 'estop', 'Safety reset status unavailable'
                        self._publish_output_locked()
                        self.pending_reset = None
                        self.reset_timer.cancel()
                        reset['future'].set_result(False)
                return
            with self._state_guard():
                if self.pending_reset is reset:
                    reset['reported_at'] = time.monotonic()
        if completed:
            try:
                # Report a completed hold while retaining the internal zero
                # barrier until the service result has been committed.
                self.publish_status(force=True, refresh_ownership=False)
            except Exception:
                with self._state_guard():
                    if self.pending_reset is reset:
                        self.core.estop_latched = True
                        self.core.mode, self.core.reason = 'estop', 'Safety reset status unavailable'
                        self._publish_output_locked()
                        self.pending_reset = None
                        reset['future'].set_result(False)
                return
            with self._state_guard():
                if self.pending_reset is reset:
                    self.pending_reset = None
                    accepted = bool(accepted and not self.core.estop_latched)
                    reset['future'].set_result(accepted)

    def on_control(self, message):
        return timed_call(getattr(self, '_timing', None), 'control_callback',
                          self._on_control, message)

    def _on_control(self, message):
        received_at = time.monotonic()
        timing = getattr(self, '_timing', None)
        receipt_ns = self.get_clock().now().nanoseconds if timing is not None else None
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            with self._state_guard():
                self.core.reject_control(None, 'Malformed control JSON')
            return
        # Locally created service packets are published for guard observation,
        # never applied by DDS loopback ahead of the reset-intent grace period.
        if isinstance(payload, dict) and payload.get('source_id') == self.service_source:
            return
        token, prevalidated = None, False
        try:
            velocity = self.core._validated_control_velocity(payload, self.get_clock().now().nanoseconds)
            prevalidated = True
            if (payload['mode'] in {'estop', 'hold'} or
                    payload['mode'] in {'manual', 'localization'} and
                    (velocity == (0., 0.) or payload['mode'] != self.core.mode)):
                token = self._latch_urgent_zero()
        except ValueError:
            pass  # Core admission records the original rejection below.
        # A slow status write would hold this mutually exclusive callback group
        # and queue later ESTOP packets. The independent status timer publishes
        # the resulting state, while command acceptance and /cmd_vel remain
        # serialized below.
        canceled = None
        # An auto heartbeat waiting here would occupy the control callback
        # group ahead of a later ESTOP. It grants no motion lease on its own;
        # coalesce it like the latest-only DDS queue while geometry is busy.
        auto_heartbeat = isinstance(payload, dict) and payload.get('mode') == 'auto'
        with self._try_state_guard(blocking=not auto_heartbeat) as acquired:
            if not acquired:
                # Preserve an explicit ownership/mode transition. Repeated
                # auto heartbeats from the existing owner need no replay.
                if prevalidated and not (self.core.mode == 'auto' and self.core.source_id == payload.get('source_id')
                                         and not self.core.estop_latched):
                    with getattr(self, '_control_ingress_lock', nullcontext()):
                        old = getattr(self, '_deferred_auto_control', None)
                        if (old is None or old[0]['source_id'] != payload['source_id']
                                or payload['seq'] > old[0]['seq']):
                            self._deferred_auto_control = (payload, received_at)
                return
            if getattr(self, '_closing', False):
                self._finish_urgent_zero(token)
                return
            if timing is not None and isinstance(payload, dict):
                stamp = payload.get('sent_at_ns')
                if type(stamp) is int:
                    timing.observe('control_receipt_age', (receipt_ns-stamp)/1e9)
                if payload.get('source_id') == self.core.EKF_SENSOR_SAFETY_SOURCE:
                    report = guard_timing_report(payload)
                    if report is not None:
                        sequence = payload.get('seq')
                        self._guard_timing = {
                            'reported_on_seq': sequence if type(sequence) is int else None,
                            'metrics': report}
            if (isinstance(payload, dict) and payload.get('mode') == 'reset'
                    and not self.allow_topic_reset):
                self.core.reject_control(
                    payload, 'Topic reset is disabled; use reset_emergency_stop service')
                self._finish_urgent_zero(token)
                return
            if (self.pending_reset is not None and self.pending_reset.get('finishing')
                    and (not isinstance(payload, dict) or payload.get('mode') != 'estop')):
                self.core.reject_control(payload, 'Safety reset finalizing; explicit resume required')
                self._finish_urgent_zero(token)
                return
            previous = (self.core.mode, self.core.source_id, self.core.estop_latched)
            if self.core.receive_control(
                    payload, self.get_clock().now().nanoseconds, origin='topic'):
                reset = self.pending_reset
                if (isinstance(payload, dict) and payload.get('mode') == 'estop'
                        and reset is not None
                        and (payload.get('source_id') != self.core.EKF_SENSOR_SAFETY_SOURCE
                             or reset['applied'])):
                    # An operator STOP or a fresh safety veto must not be
                    # cleared by the delayed service reset tick. The EKF
                    # guard's pre-apply heartbeat is still part of its normal
                    # reset handshake; it may veto after status goes unlatched.
                    canceled = reset['future']
                    self.pending_reset = None
                    self.reset_timer.cancel()
                # A newer admitted packet wins over any older uncommitted
                # ownership transition, including a different AUTO source.
                with getattr(self, '_control_ingress_lock', nullcontext()):
                    queued = getattr(self, '_deferred_auto_control', None)
                    if queued is not None and queued[1] <= received_at:
                        self._deferred_auto_control = None
                self._finish_urgent_zero(token)
                # A repeated auto lease does not change the command or revoke
                # permission. The timer/scan output will consume it once.
                if (self.core.mode != 'auto' or
                        previous != (self.core.mode, self.core.source_id, self.core.estop_latched)):
                    self._publish_output_locked(urgent_zero=self.core.output() == (0., 0.))
            else:
                self._finish_urgent_zero(token)
        if canceled is not None:
            canceled.set_result(False)

    def _publish_output_locked(self, urgent_zero=False):
        self._consume_deferred_auto_locked()
        if not urgent_zero and self.core.mode == 'auto':
            self._refresh_scan_guard_locked()
        timing = getattr(self, '_timing', None)
        if timing is not None:
            timing.gap('output_gap', .05)
        completed_at = getattr(self, '_ownership_completed_at', None)
        if (self.output_available and completed_at is not None
                and time.monotonic()-completed_at > self.OWNERSHIP_MAX_AGE):
            # A stalled graph query must not leave an old positive result
            # authorizing motion indefinitely.
            self.output_available = False
            self.output_conflict = 'Exclusive /cmd_vel ownership check is stale'
        if not self.output_available:
            # Even a zero command would interfere with an existing, unmanaged
            # controller. Wait for an explicit exclusive handover instead.
            ramp = getattr(self, 'command_ramp', None)
            if ramp is not None:
                ramp.reset()
            return
        message = Twist()
        # Guards briefly observe an unlatched state to acknowledge/veto the
        # explicit reset. Keep the motor output at zero throughout that window,
        # even if a delayed UI/navigation packet is accepted by the core.
        message.linear.x, message.angular.z = ((0.0, 0.0) if
            getattr(self, 'pending_reset', None) is not None
            or getattr(self, '_closing', False) or urgent_zero else self.core.output())
        if self.core.mode == 'auto':
            scale = getattr(self, '_braking_scale', 1.)
            message.linear.x *= scale
            message.angular.z *= scale
        ramp = getattr(self, 'command_ramp', None)
        if ramp is not None:
            owner = (self.core.mode, self.core.source_id,
                     self.core.auto_source_id if self.core.mode == 'auto' else self.core.manual_operation)
            message.linear.x, message.angular.z = ramp.apply(
                message.linear.x, message.angular.z, now=time.monotonic(), owner=owner)
        # The ramp can briefly have a different curvature from the new target.
        # Certify that actual output as well, retaining the full raw-body check.
        if not urgent_zero and self.core.mode == 'auto' and getattr(self, '_scan_guard_enabled', False):
            message.linear.x, message.angular.z = self._limit_braking_output_locked(
                message.linear.x, message.angular.z)
            self._final_output_guard_locked(message.linear.x, message.angular.z)
            if self.core.scan_guard_hold:
                message.linear.x, message.angular.z = 0., 0.
                if ramp is not None:
                    ramp.reset()
        with getattr(self, '_output_publish_lock', nullcontext()):
            if not urgent_zero and self.core.mode == 'auto' and getattr(self, '_scan_guard_enabled', False):
                self._guard_evidence_current_locked()
            pending_stop = getattr(self, '_urgent_stop_event', None)
            if (pending_stop is not None and pending_stop.is_set()
                    or self.core.mode == 'auto' and self.core.scan_guard_hold):
                message.linear.x, message.angular.z = 0., 0.
                if ramp is not None:
                    ramp.reset()
            try:
                timed_call(timing, 'cmd_vel_publish', self.output_pub.publish, message)
            except Exception:
                if ramp is not None:
                    ramp.reset()
                raise
            self.last_published_output = (message.linear.x, message.angular.z)
            self.last_output_was_zero = message.linear.x == 0.0 and message.angular.z == 0.0
            if self.last_output_was_zero:
                self.last_zero_published_at = time.monotonic()

    def publish_output(self, refresh_ownership=True):
        if refresh_ownership:
            self.check_output_ownership()
        with self._state_guard():
            self._publish_output_locked()

    def check_output_ownership(self, nonblocking=False):
        """Back off completely if any other publisher can command the robot."""
        now = time.monotonic()
        with self._try_state_guard(blocking=not nonblocking) as acquired:
            if not acquired:
                return
            if getattr(self, '_ownership_query_inflight', False):
                return
            if now-self._ownership_checked_at < .2:
                return
            self._ownership_checked_at = now
            self._ownership_query_inflight = True
        try:
            endpoints = timed_call(getattr(self, '_timing', None), 'ownership_query',
                                   self.get_publishers_info_by_topic, self.output_pub.topic_name)
            own_endpoint = (len(endpoints) == 1
                            and endpoints[0].node_name == self.get_name()
                            and endpoints[0].node_namespace == self.get_namespace())
            publishers = len(endpoints)
            if own_endpoint:
                conflict = ''
            elif not endpoints:
                conflict = 'Waiting for local /cmd_vel publisher discovery'
            else:
                names = [f'{item.node_namespace.rstrip("/")}/{item.node_name}'
                         for item in endpoints]
                conflict = 'Unmanaged /cmd_vel publisher: '+', '.join(names)
        except Exception as exc:
            # Failure to inspect the graph must never imply exclusive access.
            own_endpoint, publishers = False, 0
            conflict = 'Cannot verify exclusive /cmd_vel ownership: '+str(exc)
        with self._state_guard():
            completed_at = time.monotonic()
            duration = completed_at-now
            if duration > self.OWNERSHIP_MAX_AGE:
                # A slow graph lookup may have taken its snapshot long before
                # returning. Retry promptly and do not trust a stale success.
                own_endpoint = False
                conflict = 'Exclusive /cmd_vel ownership check timed out'
                self._ownership_checked_at = -math.inf
            self._ownership_query_inflight = False
            self._ownership_completed_at = completed_at
            self.output_publishers = publishers
            self.output_available = own_endpoint
            self.output_conflict = conflict

    def publish_status(self, force=False, refresh_ownership=True):
        timing = getattr(self, '_timing', None)
        with getattr(self, '_status_write_lock', nullcontext()):
            if timing is not None:
                timing.gap('status_gap', .1)
            if refresh_ownership:
                # Ownership refresh has its own callback group. Waiting here
                # briefly for the output snapshot cannot block command input;
                # skipping instead can phase-lock the 10 Hz status timer with
                # the 20 Hz output timer and starve every ownership refresh.
                self.check_output_ownership()
            with self._try_state_guard(blocking=force) as acquired:
                if not acquired:
                    return False
                status = self.core.status()
                # The arbitration target may be ahead of a gentle start.
                # Report the actual emitted command; reads do not advance it.
                status['requested_linear'] = status['output_linear']
                status['requested_angular'] = status['output_angular']
                status['output_linear'], status['output_angular'] = getattr(
                    self, 'last_published_output', (status['output_linear'], status['output_angular']))
                ramp = getattr(self, 'command_ramp', None)
                if ramp is not None:
                    status['command_ramp'] = ramp.snapshot()
                reset = getattr(self, 'pending_reset', None)
                reset_pending = reset is not None and not reset.get('finishing')
                transition = (status['mode'], status['source_id'], status['estop_latched'],
                              status['reason'], self.output_available, self.output_conflict,
                              status['localization_speed_factor'], status['autonomous_allowed'],
                              reset_pending)
                now = time.monotonic()
                if (not force and transition == getattr(self, '_status_transition', None)
                        and now-getattr(self, '_status_published_at', -math.inf) < .1):
                    return False
                self._status_published_at, self._status_transition = now, transition
                self.status_sequence = getattr(self, 'status_sequence', 0)+1
                status['status_seq'] = self.status_sequence
                status['sent_at_ns'] = self.get_clock().now().nanoseconds
                status['mux_instance_id'] = getattr(self, 'service_source', '')
                status['available'] = self.output_available
                status['output_publishers'] = self.output_publishers
                status['stop_acknowledged'] = bool(
                    self.output_available and getattr(self, 'last_output_was_zero', False)
                    and (reset_pending
                         or self.core.output() == (0.0, 0.0))
                    and getattr(self, 'last_zero_published_at', None) is not None
                    and time.monotonic()-self.last_zero_published_at < .2)
                status['reset_pending'] = reset_pending
                if status['reset_pending']:
                    status['output_owner'] = 'safety_reset'
                if not self.output_available:
                    status['reason'] = self.output_conflict
                if timing is not None:
                    # Publish duration is available on the next status sample.
                    status['timing_diagnostics'] = {
                        'enabled': True, 'mux': timing.snapshot(),
                        'ekf_sensor_guard': getattr(self, '_guard_timing', None)}
                message = String()
                message.data = json.dumps(status, separators=(',', ':'))
            timed_call(timing, 'status_publish', self.status_pub.publish, message)
            return True

    def begin_shutdown(self):
        with self._state_guard():
            self._closing = True
            canceled = self.pending_reset['future'] if self.pending_reset is not None else None
            if canceled is not None:
                self.pending_reset = None
                self.reset_timer.cancel()
            self.core.mode, self.core.reason = 'hold', 'Motion mux shutting down'
            self.core.estop_latched = True
            self.core.manual_velocity, self.core.auto_at = (0.0, 0.0), None
            self._publish_output_locked()
        if canceled is not None:
            canceled.set_result(False)


def main(args=None):
    import signal
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    # Publish the final stop before invalidating the context on normal launch
    # shutdown. SIGKILL/power loss still requires the motor firmware watchdog.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = MotionMuxNode()
    # Keep the node registered for the entire run. Repeated rclpy.spin_once(node)
    # removes and re-adds it on every pass, which can starve control subscriptions
    # behind the output/status timers on a busy Raspberry Pi.
    executor = MultiThreadedExecutor(num_threads=4, context=node.context)
    executor.add_node(node)
    stop_requested = [False]

    def stop(_signum, _frame):
        stop_requested[0] = True

    previous = {signum: signal.signal(signum, stop)
                for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        while rclpy.ok() and not stop_requested[0]:
            executor.spin_once(timeout_sec=.05)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.begin_shutdown()
            # Resolving an awaited reset Future queues its service task for a
            # later executor spin. Drain that task before shutdown waits for
            # worker completion, or Humble can wait forever.
            deadline = time.monotonic()+2.0
            while time.monotonic() < deadline:
                with node._state_guard():
                    if not node._reset_services_active:
                        break
                executor.spin_once(timeout_sec=.01)
            time.sleep(.05)  # Give DDS's transport thread time to flush the stop.
        executor.shutdown(timeout_sec=2.0)
        if rclpy.ok():
            # All callbacks have finished; no old worker can follow this with
            # a nonzero command.
            node.publish_output(refresh_ownership=False)
            time.sleep(.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    main()
