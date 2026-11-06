"""Deterministic localization health decisions, independent of a ROS graph."""

import math


def quaternion_yaw(q):
    values = (q.x, q.y, q.z, q.w)
    if (not all(math.isfinite(value) for value in values)
            or not 0.95 <= sum(value*value for value in values) <= 1.05):
        raise ValueError('invalid quaternion')
    return math.atan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))


def angle_difference(a, b):
    return math.atan2(math.sin(a-b), math.cos(a-b))


class LocalizationHealthCore:
    """Acquire confidence, debounce faults, and require an explicit stop reset.

    Missing localization at boot is a warning so an operator may deliberately
    run assisted localization. Once acquired, sustained critical faults latch
    a stop. Merely recovering sensor streams never resumes a mission.
    """

    def __init__(self, *, stable_seconds=2.0, fault_seconds=0.5,
                 spatial_fault_seconds=1.25, sensor_timeout=0.5,
                 tf_timeout=1.0, future_tolerance=1.5,
                 position_warning=0.35, position_acquisition_limit=0.50,
                 position_critical=0.8,
                 yaw_warning=0.5, yaw_critical=1.0,
                 agreement_warning=0.55, agreement_critical=0.20,
                 amcl_moving_timeout=4.0, amcl_pose_timeout=8.0,
                 motion_dt_limit=1.0,
                 mapping_mode=False):
        values = (stable_seconds, fault_seconds, spatial_fault_seconds,
                  sensor_timeout, tf_timeout, position_warning,
                  position_acquisition_limit,
                  position_critical, yaw_warning, yaw_critical, amcl_moving_timeout,
                  amcl_pose_timeout,
                  motion_dt_limit)
        if (not all(math.isfinite(value) and value > 0.0 for value in values)
                or not 0.0 <= future_tolerance < 10.0
                or not 0.0 <= agreement_critical < agreement_warning <= 1.0
                or not position_warning < position_acquisition_limit < position_critical
                or yaw_warning >= yaw_critical):
            raise ValueError('Invalid localization health limits')
        self.stable_seconds = stable_seconds
        self.fault_seconds = fault_seconds
        # TF/scan-to-map checks may miss a few scheduler ticks under load.
        # They still block motion on the first bad tick; only the irreversible
        # stop latch waits longer. Sensor loss and pose jumps keep fault_seconds.
        self.spatial_fault_seconds = spatial_fault_seconds
        self.sensor_timeout = sensor_timeout
        self.tf_timeout = tf_timeout
        self.future_tolerance = future_tolerance
        self.position_warning = position_warning
        self.position_acquisition_limit = position_acquisition_limit
        self.position_critical = position_critical
        self.yaw_warning = yaw_warning
        self.yaw_critical = yaw_critical
        self.agreement_warning = agreement_warning
        self.agreement_critical = agreement_critical
        self.amcl_moving_timeout = amcl_moving_timeout
        self.amcl_pose_timeout = amcl_pose_timeout
        self.motion_dt_limit = motion_dt_limit
        self.mapping_mode = bool(mapping_mode)
        self.amcl_motion_without_update_seconds = 0.0
        self.last_amcl_update_epoch = None
        self.last_motion_evaluation = None
        self.previous_moving = False
        self.motion_dt_clamped = 0
        self.acquired = False
        self.stop_latched = False
        self.good_since = None
        self.sensor_good_since = None
        self.reinitialization_requested = False
        self.fault_since = {}
        self.last_result = None

    def reinitialize(self):
        """A new pose starts acquisition; it never releases a previous stop."""
        self.acquired = False
        self.reinitialization_requested = True
        self.good_since = None
        self.sensor_good_since = None
        self.fault_since.clear()
        self.amcl_motion_without_update_seconds = 0.0
        self.last_amcl_update_epoch = self.last_motion_evaluation = None
        self.previous_moving = False
        self.motion_dt_clamped = 0

    def evaluate(self, now, metrics):
        if not isinstance(now, (float, int)) or not math.isfinite(now):
            raise ValueError('Localization evaluation needs finite monotonic time')
        moving = metrics.get('moving') is True
        raw_dt = 0.0 if self.last_motion_evaluation is None else max(0.0, now-self.last_motion_evaluation)
        dt = min(raw_dt, self.motion_dt_limit)
        if raw_dt > self.motion_dt_limit:
            self.motion_dt_clamped += 1
        epoch = metrics.get('amcl_update_epoch')
        new_update = (type(epoch) in (int, float) and math.isfinite(epoch)
                      and 0.0 <= epoch <= now
                      and (self.last_amcl_update_epoch is None or epoch > self.last_amcl_update_epoch))
        if new_update:
            self.last_amcl_update_epoch = epoch
            self.amcl_motion_without_update_seconds = 0.0
            dt = min(dt, max(0.0, now-epoch))
        # Trapezoidal transition accounting avoids charging the preceding
        # stationary interval to a new low-speed rotation. A stop never resets
        # the accumulated budget; repeated short stop/go pulses still expire.
        self.amcl_motion_without_update_seconds += dt*(int(self.previous_moving)+int(moving))/2.0
        self.last_motion_evaluation, self.previous_moving = now, moving
        warning, critical, advisories = [], [], []
        sensor_ok = True
        for name in ('scan', 'odom'):
            age = metrics.get(f'{name}_age_seconds')
            header_age = metrics.get(f'{name}_header_age_seconds')
            valid = (age is not None and header_age is not None
                     and math.isfinite(age) and math.isfinite(header_age)
                     and 0.0 <= age <= self.sensor_timeout
                     and -0.25 <= header_age <= self.sensor_timeout
                     and metrics.get(f'{name}_valid', True))
            if not valid:
                sensor_ok = False
                critical.append(f'{name}_unavailable')
        if not metrics.get('odom_tf_valid', False):
            sensor_ok = False
            critical.append('odom_tf_unavailable')
        if sensor_ok:
            if self.sensor_good_since is None:
                self.sensor_good_since = now
        else:
            self.sensor_good_since = None
        if self.mapping_mode:
            # A changing SLAM map has no stable AMCL pose or scan/map baseline.
            # Keep the independent sensor/TF interlock and never permit auto.
            critical = sorted(set(critical))
            for reason in list(self.fault_since):
                if reason not in critical:
                    del self.fault_since[reason]
            for reason in critical:
                self.fault_since.setdefault(reason, now)
            if any(now-self.fault_since[reason] >= self.fault_seconds
                   for reason in critical):
                self.stop_latched = True
            self.acquired = False
            self.good_since = None
            reasons = ['mapping_mode', *critical]
            if self.stop_latched and not critical:
                reasons.append('operator_reset_required')
            result = {
                'level': 'critical' if critical or self.stop_latched else 'warning',
                'reasons': reasons, 'metrics': dict(metrics),
                'stop_latched': self.stop_latched,
                'autonomous_allowed': False,
                'allow_localization': sensor_ok,
                'reinitialization_requested': self.reinitialization_requested,
                'stable_seconds': 0.0 if self.sensor_good_since is None
                                  else now-self.sensor_good_since,
                'mapping_mode': True,
            }
            self.last_result = result
            return result
        tf_age = metrics.get('map_tf_age_seconds')
        if (tf_age is None or not math.isfinite(tf_age)
                or not -self.future_tolerance <= tf_age <= self.tf_timeout):
            critical.append('map_tf_unavailable')
        if not metrics.get('map_available'):
            critical.append('map_unavailable')
        for name, warn, stop in (
                ('position_std_m', self.position_warning, self.position_critical),
                ('yaw_std_rad', self.yaw_warning, self.yaw_critical)):
            value = metrics.get(name)
            if value is None or not math.isfinite(value) or value < 0.0:
                critical.append('amcl_covariance_unavailable')
            elif value >= stop:
                critical.append(f'{name}_high')
            elif value >= warn:
                warning.append(f'{name}_elevated')
        # A cached low covariance is not proof of a current pose. Require both
        # a recent ROS receipt and a recent source stamp before departure and
        # throughout autonomous motion. Only on_amcl can refresh these ages.
        amcl_age = metrics.get('amcl_age_seconds')
        amcl_header_age = metrics.get('amcl_header_age_seconds')
        if (amcl_age is None or amcl_header_age is None
                or not math.isfinite(amcl_age) or not math.isfinite(amcl_header_age)
                or not 0. <= amcl_age <= self.amcl_pose_timeout
                or not -.25 <= amcl_header_age <= self.amcl_pose_timeout):
            critical.append('amcl_pose_stale')
        if self.amcl_motion_without_update_seconds >= self.amcl_moving_timeout:
            critical.append('amcl_not_updating_while_moving')
        if metrics.get('pose_jump'):
            critical.append('localization_jump')
        score = metrics.get('scan_map_agreement')
        agreement_age = metrics.get('agreement_age_seconds')
        agreement_header_age = metrics.get('agreement_header_age_seconds')
        if (score is None or not math.isfinite(score) or agreement_age is None
                or agreement_age > 2.0*self.sensor_timeout
                or (agreement_header_age is not None and (
                    not math.isfinite(agreement_header_age)
                    or not -.25 <= agreement_header_age <= 2.0*self.sensor_timeout))):
            critical.append('scan_map_check_unavailable')
        elif score < self.agreement_critical:
            # Movable classroom fixtures and temporary obstacles can make the
            # saved-wall hit ratio arbitrarily low even with a valid AMCL pose.
            # Keep the measurement visible, but let independent map identity,
            # sensor freshness, TF, and covariance checks decide motion.
            advisories.append('scan_map_disagreement')
        elif score < self.agreement_warning:
            advisories.append('scan_map_agreement_low')

        critical = sorted(set(critical))
        for reason in list(self.fault_since):
            if reason not in critical:
                del self.fault_since[reason]
        for reason in critical:
            self.fault_since.setdefault(reason, now)
        # Testbed bootstrap may accept a measured, moderately uncertain pose.
        # Keep its warning active so motion_mux limits speed to half. Any
        # missing/stale sensor, TF, map check, or yaw warning still prevents
        # initial acquisition. A low finite match score is diagnostic only.
        bounded_position_warning = (
            not self.acquired and warning == ['position_std_m_elevated']
            and metrics.get('position_std_m', math.inf) <= self.position_acquisition_limit)
        if not critical and (not warning or bounded_position_warning):
            if self.good_since is None:
                self.good_since = now
            if now-self.good_since >= self.stable_seconds:
                self.acquired = True
        else:
            self.good_since = None
        spatial_faults = {'map_tf_unavailable', 'odom_tf_unavailable',
                          'scan_map_check_unavailable'}
        sustained = [reason for reason in critical
                     if now-self.fault_since[reason] >= (
                         self.spatial_fault_seconds if reason in spatial_faults
                         else self.fault_seconds)]
        if self.acquired and sustained:
            self.stop_latched = True
        level = ('critical' if self.stop_latched else
                 'warning' if warning or critical or not self.acquired else 'ok')
        reasons = sorted(set(warning+critical))
        if not self.acquired:
            reasons.append('initializing')
        if self.stop_latched and not critical:
            reasons.append('operator_reset_required')
        result = {
            'level': level, 'reasons': reasons, 'advisories': advisories,
            'metrics': dict(metrics),
            'stop_latched': self.stop_latched,
            'autonomous_allowed': self.acquired and not critical and not self.stop_latched,
            'allow_localization': sensor_ok,
            'reinitialization_requested': self.reinitialization_requested,
            'stable_seconds': 0.0 if self.good_since is None else now-self.good_since,
        }
        result['metrics']['amcl_motion_without_update_seconds'] = self.amcl_motion_without_update_seconds
        result['metrics']['amcl_motion_dt_clamped'] = self.motion_dt_clamped
        self.last_result = result
        return result

    def operator_reset(self, now):
        if self.last_result is None:
            return False
        if self.mapping_mode:
            if (self.sensor_good_since is None
                    or now-self.sensor_good_since < self.stable_seconds):
                return False
            self.stop_latched = False
            return True
        fully_healthy = (self.good_since is not None
                         and now-self.good_since >= self.stable_seconds)
        # An explicit initialpose + reset may release only the stop latch into
        # acquisition mode. Waiting for a perfect match first would deadlock
        # when the robot needs a bounded localization sweep to regain it.
        reacquisition_ready = (not self.acquired and self.reinitialization_requested
                               and self.sensor_good_since is not None
                               and now-self.sensor_good_since >= self.stable_seconds)
        if not fully_healthy and not reacquisition_ready:
            return False
        self.stop_latched = False
        return True
