"""Conservative two-scan check for returns at the LiDAR minimum range.

This filters only the *map-change evidence* stream. The original LaserScan is
left intact for Nav2's collision costmap and the immediate motion interlock.
Gazebo and real LiDAR edge echoes can cover several adjacent beams in *one*
scan, so spatial support alone is not evidence of a persistent obstacle.
Every near-minimum cluster is unknown until the next scan, not free space.
"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class NearScanDecision:
    unconfirmed: frozenset[int]
    confirmed: frozenset[int]
    accepted: bool = True


class NearScanOutlierGuard:
    def __init__(self, *, above_min_m=0.10, absolute_max_m=0.30,
                 temporal_delta_m=0.12, temporal_window_sec=0.75):
        values = (above_min_m, absolute_max_m, temporal_delta_m,
                  temporal_window_sec)
        if not all(isfinite(value) and value > 0 for value in values):
            raise ValueError("near-scan guard limits must be finite and positive")
        self.above_min_m = above_min_m
        self.absolute_max_m = absolute_max_m
        self.temporal_delta_m = temporal_delta_m
        self.temporal_window_ns = int(temporal_window_sec * 1e9)
        self.reset()

    def reset(self):
        self.last_stamp_ns = -1
        self.pending = ()  # (bearing, distance) seen in the immediately previous scan

    def classify(self, scan, stamp_ns):
        """Return suspect beam indices; never infer a free ray from a suspect.

        A matching bearing and range on the *next* scan establishes temporal
        support, including for a wide adjacent-beam cluster. Out-of-order or
        replayed samples cannot confirm a ghost or clear the pending hold.
        """
        if (type(stamp_ns) is not int or stamp_ns <= self.last_stamp_ns
                or not isfinite(scan.range_min) or not isfinite(scan.range_max)
                or not isfinite(scan.angle_min) or not isfinite(scan.angle_increment)
                or scan.angle_increment == 0 or not 0 <= scan.range_min < scan.range_max
                or not scan.ranges
                or not any((isfinite(value) and scan.range_min <= value <= scan.range_max)
                           or value == float('inf') for value in scan.ranges)):
            return NearScanDecision(frozenset(), frozenset(), False)
        limit = min(self.absolute_max_m, scan.range_min + self.above_min_m)
        candidates = [(i, scan.angle_min + i * scan.angle_increment, float(value))
                      for i, value in enumerate(scan.ranges)
                      if isfinite(value) and scan.range_min <= value <= limit]
        previous = self.pending if stamp_ns - self.last_stamp_ns <= self.temporal_window_ns else ()
        unconfirmed, confirmed = set(), set()
        for index, bearing, distance in candidates:
            temporal = any(
                abs(old_bearing-bearing) <= max(.04, 2.5*abs(scan.angle_increment))
                and abs(old_distance-distance) <= self.temporal_delta_m
                for old_bearing, old_distance in previous)
            (confirmed if temporal else unconfirmed).add(index)
        self.pending = tuple((bearing, distance) for i, bearing, distance in candidates
                             if i in unconfirmed or i in confirmed)
        self.last_stamp_ns = stamp_ns
        return NearScanDecision(frozenset(unconfirmed), frozenset(confirmed))


@dataclass(frozen=True)
class SweptBodyDecision:
    inside: frozenset[int]
    swept: frozenset[int]
    minimum_body_clearance: float
    minimum_swept_clearance: float


def scan_points_in_base(scan, extrinsic):
    """Transform valid raw returns by the known planar base <- laser mounting.

    No ranges, evidence classes or sensor messages are modified. Infinite and
    invalid returns contain no measured obstacle point and remain unchanged.
    """
    from math import cos, sin
    tx, ty, yaw = extrinsic
    if not all(isfinite(value) for value in extrinsic):
        raise ValueError('Laser mounting transform must be finite')
    c, s = cos(yaw), sin(yaw)
    points = []
    for index, value in enumerate(scan.ranges):
        if not isfinite(value) or not scan.range_min <= value <= scan.range_max:
            continue
        bearing = scan.angle_min+index*scan.angle_increment
        x, y = value*cos(bearing), value*sin(bearing)
        points.append((index, tx+c*x-s*y, ty+s*x+c*y))
    return tuple(points)


def rebase_scan_points(points, scan_pose, current_pose):
    """Move acquisition-time base coordinates through continuous odometry."""
    from math import cos, sin
    if not all(isfinite(value) for pose in (scan_pose, current_pose) for value in pose):
        raise ValueError('Scan and current odometry poses must be finite')
    sx, sy, sa = scan_pose
    tx, ty, ta = current_pose
    c, d = cos(sa-ta), sin(sa-ta)
    dx, dy = cos(ta)*(sx-tx)+sin(ta)*(sy-ty), -sin(ta)*(sx-tx)+cos(ta)*(sy-ty)
    return tuple((index, dx+c*x-d*y, dy+d*x+c*y) for index, x, y in points)


def swept_body_decision(points, linear, angular, *, radius=.22, horizon=.5,
                        longitudinal_uncertainty=0., lateral_uncertainty=0.,
                        yaw_uncertainty=0.):
    """Point/circular-body sweep, including bounded age of the latest pose.

    Scan delay is handled by odometry rebasing, never by radial padding. Only
    motion since the latest odometry pose adds a longitudinal interval and a
    bounded lateral/yaw uncertainty. Parallel walls therefore do not inherit
    the full speed*scan_age as a fictitious all-direction collision radius.
    """
    # The brake envelope already uses this exact vectorized directed-arc
    # distance, including horizontal pose intervals. Reuse it here so every
    # raw point and every emitted curve remains checked without Python loops
    # per beam in the 20 Hz output callback. No evidence or age is cached.
    from .braking_envelope import _cloud, _clearances
    if (not all(isfinite(v) for v in (linear, angular, radius, horizon,
            longitudinal_uncertainty, lateral_uncertainty, yaw_uncertainty))
            or radius <= 0. or not 0. < horizon <= 2.
            or min(longitudinal_uncertainty, lateral_uncertainty, yaw_uncertainty) < 0.):
        raise ValueError('Invalid scan collision sweep limits')
    cloud = _cloud(points)
    body = _clearances(cloud, 0., 0., radius, longitudinal_uncertainty,
                       lateral_uncertainty, 0.)
    sweep = _clearances(cloud, linear*horizon, angular*horizon, radius,
                        longitudinal_uncertainty, lateral_uncertainty, yaw_uncertainty)
    inside = frozenset(int(index) for index in cloud[body <= 1.e-9, 0])
    swept = frozenset(int(index) for index in cloud[sweep <= 1.e-9, 0])
    return SweptBodyDecision(inside, swept,
        float(body.min()) if len(cloud) else float('inf'),
        float(sweep.min()) if len(cloud) else float('inf'))
