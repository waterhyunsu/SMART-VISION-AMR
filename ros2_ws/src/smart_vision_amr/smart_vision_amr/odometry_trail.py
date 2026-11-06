"""A bounded trail of actually received odometry, independent of AMCL.

Map transforms may be corrected after travel. Store the measured trace in its
continuous odometry frame and project every point with one current transform.
"""
from collections import deque
from dataclasses import dataclass
from math import cos, hypot, isfinite, pi, sin

from smart_vision_amr.reverse_recovery import Breadcrumb


@dataclass(frozen=True)
class OdometryTrailPoint:
    received_at: float
    stamp_ns: int
    x: float
    y: float
    yaw: float


class OdometryTrail:
    def __init__(self, maxlen=300):
        self.points = deque(maxlen=maxlen)
        self.latest = None
        self.frame = self.child_frame = None
        self.last_reset_reason = "empty"

    def clear(self, reason="explicit_reset"):
        self.points.clear()
        self.latest = None
        self.frame = self.child_frame = None
        self.last_reset_reason = reason

    def observe(self, sample, child_frame):
        """Accept one real source/receipt-stamped sample; discontinuities reset.

        Keep the original 15 cm / .15 rad inter-sample jump bounds, and reject
        time/frame resets and unobserved gaps. No map pose can create a point.
        """
        received, stamp, x, y, yaw, frame = sample
        if (type(stamp) is not int or stamp <= 0 or not frame or not child_frame
                or not all(isfinite(v) for v in (received, x, y, yaw))):
            self.clear("invalid_sample")
            return False
        point = OdometryTrailPoint(received, stamp, x, y, yaw)
        prior = self.latest
        if prior is not None:
            reason = None
            if frame != self.frame or child_frame != self.child_frame:
                reason = "frame_changed"
            elif received < prior.received_at or stamp < prior.stamp_ns:
                reason = "clock_reset"
            elif (received-prior.received_at >= .75
                  or stamp-prior.stamp_ns >= 750_000_000):
                reason = "unobserved_gap"
            elif (hypot(x-prior.x, y-prior.y) > .15
                  or abs((yaw-prior.yaw+pi) % (2*pi)-pi) > .15):
                reason = "pose_jump"
            elif stamp == prior.stamp_ns:
                if (hypot(x-prior.x, y-prior.y) > 1e-9
                        or abs((yaw-prior.yaw+pi) % (2*pi)-pi) > 1e-9):
                    self.clear("changed_duplicate_stamp")
                    return False
                return True  # Duplicate receipt does not refresh source age.
            if reason is not None:
                self.clear(reason)
        self.frame, self.child_frame, self.latest = frame, child_frame, point
        if not self.points or hypot(x-self.points[-1].x, y-self.points[-1].y)+1e-9 >= .025:
            self.points.append(point)
        return True

    def project(self, paired_map_pose, paired_odom, *, now, max_age=90.):
        """Rigidly re-express actual odometry using one same-time pose pair."""
        if (self.latest is None or paired_odom[5] != self.frame
                or now < self.latest.received_at or now-self.latest.received_at >= .75
                or not all(isfinite(v) for v in paired_map_pose)):
            return []
        rotation = paired_map_pose[2]-paired_odom[4]
        c, s = cos(rotation), sin(rotation)
        tx = paired_map_pose[0]-c*paired_odom[2]+s*paired_odom[3]
        ty = paired_map_pose[1]-s*paired_odom[2]-c*paired_odom[3]
        points = [p for p in self.points if 0 <= now-p.received_at <= max_age
                  and p.stamp_ns <= paired_odom[1]]
        # The actual paired source pose is the tail, never an AMCL-created
        # extra movement. Keep a currently stationary tail fresh independently.
        tail = OdometryTrailPoint(paired_odom[0], paired_odom[1], *paired_odom[2:5])
        # The already-required measured-stop bound is 2 mm. Do not interpret
        # stationary wheel noise inside that bound as a tiny forward segment
        # which would hide the actual older reverse corridor.
        if points and hypot(points[-1].x-tail.x, points[-1].y-tail.y) <= .002:
            points[-1] = tail
        else:
            points.append(tail)
        return [Breadcrumb(p.received_at, tx+c*p.x-s*p.y,
                           ty+s*p.x+c*p.y, p.yaw+rotation) for p in points]
