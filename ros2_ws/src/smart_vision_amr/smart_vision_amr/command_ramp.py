"""Gentle acceleration of published twists, with immediate stop precedence.

This helper advances only when the final publisher emits a command. Reading
status must never consume acceleration budget. Actual turns preserve requested
curvature with one common scale. Small heading corrections during translation
instead slew only yaw, so alternating tiny corrections cannot stop translation.
"""
from math import copysign, isfinite


class CommandRamp:
    SMALL_HEADING_ANGULAR = .08
    TRANSLATION_EPSILON = .02

    def __init__(self, *, enabled=False, linear_acceleration=.20,
                 angular_acceleration=.80, max_publish_gap=.50):
        values = (linear_acceleration, angular_acceleration, max_publish_gap)
        if not all(isfinite(value) and value > 0. for value in values):
            raise ValueError('Command ramp acceleration and gap limits must be finite and positive')
        self.enabled = bool(enabled)
        self.linear_acceleration = float(linear_acceleration)
        self.angular_acceleration = float(angular_acceleration)
        self.max_publish_gap = float(max_publish_gap)
        self.max_acceleration_step = min(.10, self.max_publish_gap)
        self.last_at = None
        self.owner = None
        self.last_output = (0., 0.)
        self.target = (0., 0.)
        self.reason = 'idle'
        self.limited = False

    def _record(self, now, owner, output, reason):
        self.last_at, self.owner = now, owner
        self.last_output = output
        self.reason = reason
        self.limited = output != self.target
        return output

    def reset(self, *, now=None, owner=None):
        self.last_at, self.owner = now, owner
        self.last_output = self.target = (0., 0.)
        self.reason, self.limited = 'reset', False

    def apply(self, linear, angular, *, now, owner=None):
        if not all(isfinite(value) for value in (linear, angular, now)):
            raise ValueError('Command ramp requires finite twist and steady time')
        self.target = float(linear), float(angular)
        if self.target == (0., 0.):
            return self._record(now, owner, (0., 0.), 'stop')
        if not self.enabled:
            return self._record(now, owner, self.target, 'disabled')
        if self.last_at is None:
            return self._record(now, owner, (0., 0.), 'first_sample')
        if owner != self.owner:
            return self._record(now, owner, (0., 0.), 'owner_changed')
        dt = now-self.last_at
        if dt < 0. or dt > self.max_publish_gap:
            return self._record(now, owner, (0., 0.), 'publish_gap')
        # Do not turn scheduler delay into a stored acceleration allowance.
        # 100~200 ms delivery jitter may continue gently, but one publication
        # can advance at most a 100 ms acceleration step (0.02 m/s by default).
        dt = min(dt, self.max_acceleration_step)
        previous_linear, previous_angular = self.last_output
        if previous_linear*linear < 0.:
            # Never switch the drive direction in a single nonzero command.
            # The next successful publish starts the new direction from zero.
            return self._record(now, owner, (0., 0.), 'direction_changed')
        small_heading_correction = (
            abs(linear) > self.TRANSLATION_EPSILON
            and max(abs(previous_angular), abs(angular)) < self.SMALL_HEADING_ANGULAR)
        if previous_angular*angular < 0. and not small_heading_correction:
            # Reversing a turn also passes through a real zero publication;
            # do not jump directly from an established spin to its opposite.
            return self._record(now, owner, (0., 0.), 'yaw_direction_changed')
        if small_heading_correction:
            # Tracking a nearly straight aisle naturally alternates tiny yaw
            # signs. Stopping the whole chassis at each crossing prevents any
            # acceleration and creates false no-progress failures. Continue
            # the bounded linear ramp and pass yaw through its own signed
            # acceleration interval; meaningful turns still use the common
            # curvature scale below and reverse/stop precedence is unchanged.
            allowed_linear = abs(previous_linear)+self.linear_acceleration*dt
            forward = copysign(min(abs(linear), allowed_linear), linear)
            yaw_step = self.angular_acceleration*dt
            yaw = min(previous_angular+yaw_step, max(previous_angular-yaw_step, angular))
            return self._record(now, owner, (forward, yaw),
                                'small_heading_slew' if yaw != angular else
                                'acceleration_limited' if forward != linear else 'tracking')
        factor = 1.
        for target, previous, acceleration in (
                (linear, previous_linear, self.linear_acceleration),
                (angular, previous_angular, self.angular_acceleration)):
            if target == 0.:
                continue
            # Reducing a command is immediate. A sign change cannot borrow the
            # previous opposite-direction speed as an acceleration allowance.
            aligned = max(0., copysign(1., target)*previous)
            allowed = aligned+acceleration*dt
            factor = min(factor, allowed/abs(target))
        output = linear*factor, angular*factor
        return self._record(now, owner, output,
                            'acceleration_limited' if factor < 1. else 'tracking')

    def snapshot(self):
        return {'enabled': self.enabled, 'limited': self.limited,
                'reason': self.reason, 'linear_acceleration': self.linear_acceleration,
                'angular_acceleration': self.angular_acceleration,
                'max_acceleration_step': self.max_acceleration_step,
                'target_linear': self.target[0], 'target_angular': self.target[1],
                'output_linear': self.last_output[0], 'output_angular': self.last_output[1]}
