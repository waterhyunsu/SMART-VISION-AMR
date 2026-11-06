"""Bounded, ROS-independent detection of failed *execution*, not sensor faults.

The caller supplies samples only while localization, sensor, clock and arbiter
ownership checks pass. Losing any prerequisite must call ``reset``. A route
refresh can keep its history by passing an equivalent route identity.
"""
from collections import deque
from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, sin


@dataclass(frozen=True)
class ProgressSample:
    at: float
    ros_ns: int
    x: float
    y: float
    yaw_error: float
    path_progress: float
    command_linear: float
    command_angular: float
    yaw: float = 0.0
    odom_yaw: float = 0.0


class ExecutionProgress:
    def __init__(self, window=6.0, min_progress=0.05, heading_progress=0.15,
                 min_linear=0.025, min_angular=0.08, oscillation_changes=4):
        values = (window, min_progress, heading_progress, min_linear, min_angular)
        if any(not isfinite(v) or v <= 0 for v in values) or oscillation_changes < 2:
            raise ValueError('Execution progress limits must be positive and finite')
        self.window = max(2., float(window))
        self.min_progress = float(min_progress)
        self.heading_progress = float(heading_progress)
        self.min_linear, self.min_angular = float(min_linear), float(min_angular)
        self.oscillation_changes = int(oscillation_changes)
        self.samples = deque(maxlen=500)
        self.route_id = None
        self.last_intent_at = None

    def reset(self, route_id=None):
        self.samples.clear()
        self.route_id = route_id
        self.last_intent_at = None

    def observe(self, sample, route_id):
        if route_id != self.route_id:
            self.reset(route_id)
        if not all(isfinite(v) for v in (sample.at, sample.x, sample.y, sample.yaw_error,
                                         sample.path_progress, sample.command_linear,
                                         sample.command_angular, sample.yaw,
                                         sample.odom_yaw)) or sample.ros_ns <= 0:
            self.reset(route_id)
            return None
        previous = self.samples[-1] if self.samples else None
        if previous and (sample.at <= previous.at or sample.at-previous.at > .6
                         or sample.ros_ns <= previous.ros_ns):
            self.reset(route_id)
        # DWB sign reversals normally pass briefly through zero. Preserve such
        # short gaps while the caller still has healthy spatial evidence; a
        # longer deliberate pause cannot accumulate failed-execution time.
        if abs(sample.command_linear) < self.min_linear and abs(sample.command_angular) < self.min_angular:
            if self.last_intent_at is None or sample.at-self.last_intent_at > .6:
                self.reset(route_id)
            else:
                self.samples.append(sample)
            return None
        self.last_intent_at = sample.at
        self.samples.append(sample)
        while len(self.samples) > 1 and sample.at-self.samples[1].at >= self.window:
            self.samples.popleft()
        first = self.samples[0]
        duration = sample.at-first.at
        if duration < self.window or sample.ros_ns-first.ros_ns < 500_000_000:
            return None
        displacement = hypot(sample.x-first.x, sample.y-first.y)
        progress = sample.path_progress-first.path_progress
        heading_gain = abs(first.yaw_error)-abs(sample.yaw_error)
        # The nearest path segment can change while the chassis turns through a
        # corner. Compare the measured net yaw with the heading target at the
        # beginning of this window as well, so a real alignment is not called
        # a failed execution merely because the moving tangent keeps the
        # instantaneous yaw error nearly constant.
        yaw_delta = atan2(sin(sample.yaw-first.yaw), cos(sample.yaw-first.yaw))
        odom_yaw_delta = atan2(sin(sample.odom_yaw-first.odom_yaw),
                               cos(sample.odom_yaw-first.odom_yaw))
        yaw_consistent = abs(atan2(sin(yaw_delta-odom_yaw_delta),
                                   cos(yaw_delta-odom_yaw_delta))) <= .15
        yaw_travel = sum(abs(atan2(sin(b.yaw-a.yaw), cos(b.yaw-a.yaw)))
                         for a, b in zip(self.samples, list(self.samples)[1:]))
        yaw_directness = abs(yaw_delta) / yaw_travel if yaw_travel > 1e-9 else 0.
        anchored_error = atan2(sin(first.yaw_error-yaw_delta),
                               cos(first.yaw_error-yaw_delta))
        anchored_heading_gain = abs(first.yaw_error)-abs(anchored_error)
        turns = [1 if s.command_angular > self.min_angular else -1
                 for s in self.samples if abs(s.command_angular) >= self.min_angular]
        changes = sum(a != b for a, b in zip(turns, turns[1:]))
        evidence = {'window_seconds': duration, 'displacement': displacement,
                    'path_progress_delta': progress, 'heading_progress': heading_gain,
                    'anchored_heading_progress': anchored_heading_gain,
                    'net_yaw_change': yaw_delta, 'odom_yaw_change': odom_yaw_delta,
                    'yaw_consistent': yaw_consistent, 'yaw_directness': yaw_directness,
                    'sign_changes': changes, 'sample_count': len(self.samples)}
        # Heading improvement is useful motion for a differential-drive robot.
        # Repeated reversals without spatial progress are an oscillation even
        # if one sample happens to face the initial target more closely.
        if progress >= self.min_progress or displacement >= self.min_progress:
            return None
        if changes >= self.oscillation_changes:
            return 'execution_oscillation', evidence
        if (yaw_consistent and yaw_directness >= .5
                and abs(yaw_delta) >= min(.1, self.heading_progress / 2.)
                and max(heading_gain, anchored_heading_gain) >= self.heading_progress):
            return None
        return 'execution_no_progress', evidence
