"""Raw-point stopping envelope for a differential drive.

The brake rate is a configured lower bound, not an online estimate of friction.
The default covers the deliberately slow .15 m/s² Gazebo wheel-brake fixture;
it must be measured before claiming that bound for a particular physical robot.
No obstacle point or chassis radius is changed by this module.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import cos, isfinite, pi, sin
import numpy as np

try:
    from ._geometry_native import clearances as _native_clearances
except ImportError:
    _native_clearances = None


@dataclass(frozen=True)
class BrakeDecision:
    collision: frozenset
    clearance: float
    distance: float
    yaw: float
    wheel_brake_seconds: float
    wheel_curve_bound: float


def _cloud(points):
    values = (points if isinstance(points, np.ndarray) else
              np.asarray(tuple(points), dtype=float).reshape((-1, 3)))
    if (values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values))):
        raise ValueError('Nonfinite raw obstacle point')
    return values


def _arc_distances(x, y, distance, angle):
    """Vector form of the raw guard's exact point-to-directed-arc distance."""
    radius = distance/angle
    ex, ey = radius*sin(angle), radius*(1-cos(angle))
    initial = -pi/2 if radius > 0 else pi/2
    advance = ((np.arctan2(y-radius, x)-initial) if angle > 0
               else (initial-np.arctan2(y-radius, x))) % (2*pi)
    result = np.minimum(np.hypot(x, y), np.hypot(x-ex, y-ey))
    return np.where((abs(angle) >= 2*pi) | (advance <= abs(angle)+1.e-12),
        np.minimum(result, np.abs(np.hypot(x, y-radius)-abs(radius))), result)


def _clearances(points, distance, angle, radius, longitudinal, lateral, yaw):
    """Exact horizontal uncertainty intervals, matching swept_body_decision.

    Vectorization keeps 360-beam brake candidate searches out of the Python
    per-point loop; it changes neither the point set nor collision threshold.
    """
    if _native_clearances is not None:
        cloud = np.ascontiguousarray(points, dtype=np.float64)
        return np.frombuffer(_native_clearances(cloud, distance, angle, radius,
                             longitudinal, lateral, yaw), dtype=np.float64)
    return _clearances_numpy(points, distance, angle, radius, longitudinal, lateral, yaw)


def _clearances_numpy(points, distance, angle, radius, longitudinal, lateral, yaw):
    """Portable reference retained when the optional native module is absent."""
    x, y = points[:, 1], points[:, 2]
    lo, hi = x-longitudinal, x+longitudinal
    gap = np.maximum(np.maximum(lo, -hi), 0.)
    if distance == 0.:
        result = np.hypot(gap, y)
    elif abs(angle) < 1.e-6:
        along = np.maximum(np.maximum(min(0., distance)-hi,
                                     lo-max(0., distance)), 0.)
        result = np.hypot(along, y)-abs(distance*angle)*.5
    else:
        r = distance/angle
        ex, ey = r*sin(angle), r*(1-cos(angle))
        result = np.minimum(_arc_distances(lo, y, distance, angle),
                            _arc_distances(hi, y, distance, angle))
        result = np.minimum(result, np.hypot(gap, y))
        result = np.minimum(result, np.hypot(np.maximum(np.maximum(lo-ex, ex-hi), 0.), y-ey))
        if abs(angle) >= pi:
            result = np.minimum(result, np.hypot(gap, y-2*r))
        discriminant = r*r-(y-r)**2
        root = np.sqrt(np.maximum(discriminant, 0.))
        for value in (root, -root):
            # A circle/interval intersection lies on this directed arc only
            # when its angle is included in the stopping sweep.
            initial = -pi/2 if r > 0 else pi/2
            advance = ((np.arctan2(y-r, value)-initial) if angle > 0
                       else (initial-np.arctan2(y-r, value))) % (2*pi)
            on_arc = (abs(angle) >= 2*pi) | (advance <= abs(angle)+1.e-12)
            hit = (discriminant >= 0.) & (lo <= value) & (value <= hi) & on_arc
            result = np.where(hit, 0., result)
    return result-radius-lateral-2*abs(distance)*sin(min(pi, yaw)/2)


def _trig_interval(low, high, *, cosine):
    fn = np.cos if cosine else np.sin
    minimum, maximum = np.minimum(fn(low), fn(high)), np.maximum(fn(low), fn(high))
    peak = 0. if cosine else pi/2
    contains_max = np.ceil((low-peak)/(2*pi)) <= np.floor((high-peak)/(2*pi))
    contains_min = np.ceil((low-peak-pi)/(2*pi)) <= np.floor((high-peak-pi)/(2*pi))
    return np.where(contains_min, -1., minimum), np.where(contains_max, 1., maximum)


def _box_clearances(points, boxes, radius):
    """Exact nearest distance and hits, with a conservative broad phase.

    Distance to the union's enclosing box is a lower bound for every time
    box. An exact seed gives an upper bound for the global nearest distance.
    Keep every possible hit and every point that could improve that seed;
    all excluded raw points are proved farther away, not treated as absent.
    """
    if not len(points):
        return frozenset(), float('inf')
    selected = points
    if len(points) >= 32:
        x, y = points[:, 1], points[:, 2]
        dx = np.maximum(np.maximum(np.min(boxes[0])-x, x-np.max(boxes[1])), 0.)
        dy = np.maximum(np.maximum(np.min(boxes[2])-y, y-np.max(boxes[3])), 0.)
        lower_squared = dx*dx+dy*dy
        seed = int(np.argmin(lower_squared))
        dx = np.maximum(np.maximum(boxes[0]-x[seed], x[seed]-boxes[1]), 0.)
        dy = np.maximum(np.maximum(boxes[2]-y[seed], y[seed]-boxes[3]), 0.)
        best_squared = float(np.min(dx*dx+dy*dy))
        cutoff = np.nextafter(max(best_squared, (radius+1.e-9)**2), np.inf)
        selected = points[lower_squared <= cutoff]
    x, y = selected[:, 1, None], selected[:, 2, None]
    dx = np.maximum(np.maximum(boxes[0]-x, x-boxes[1]), 0.)
    dy = np.maximum(np.maximum(boxes[2]-y, y-boxes[3]), 0.)
    clearance = np.sqrt(dx*dx+dy*dy)-radius
    hits = frozenset(int(v) for v in selected[np.any(clearance <= 1.e-9, axis=1), 0])
    return hits, float(np.min(clearance))


class BrakingEnvelope:
    def __init__(self, *, wheel_separation=.287, wheel_deceleration=.15,
                 reaction_seconds=.10, stop_reserve=.02):
        if (not all(isfinite(v) for v in (wheel_separation, wheel_deceleration,
                                         reaction_seconds, stop_reserve))
                or wheel_separation <= 0 or wheel_deceleration <= 0
                or not 0 <= reaction_seconds <= .5 or not 0 <= stop_reserve <= .05):
            raise ValueError('Invalid wheel braking envelope')
        self.wheel_separation = wheel_separation
        self.wheel_deceleration = wheel_deceleration
        self.reaction_seconds = reaction_seconds
        self.stop_reserve = stop_reserve

    @staticmethod
    def _advance(x, y, yaw, v, w, dt):
        if abs(w) < 1.e-10:
            return x+v*dt*cos(yaw), y+v*dt*sin(yaw), yaw+w*dt
        return (x+v/w*(sin(yaw+w*dt)-sin(yaw)),
                y-v/w*(cos(yaw+w*dt)-cos(yaw)), yaw+w*dt)

    @lru_cache(maxsize=256)
    def _curve_bound(self, v, w, duration, distance, angle):
        """Bound independent equal wheel braking against the common-scale arc.

        The small wheel can reach zero before the large wheel. Its center path
        is not exactly the common-curvature arc. Check that second physical
        brake model as well, using a Lipschitz bound between 5 ms samples.
        Straight motion and a pure pivot have identical center paths.
        """
        if abs(v) < 1.e-10 or abs(w) < 1.e-10:
            return 0.
        x, y, yaw = self._advance(0., 0., 0., v, w, self.reaction_seconds)
        left, right = v-w*self.wheel_separation/2, v+w*self.wheel_separation/2
        count = max(1, int(duration/.005)+1)
        dt = duration/count
        t = (np.arange(count)+.5)*dt
        a = np.maximum(0., abs(left)-self.wheel_deceleration*t)*np.sign(left)
        b = np.maximum(0., abs(right)-self.wheel_deceleration*t)*np.sign(right)
        v_mid, w_mid = (a+b)/2, (b-a)/self.wheel_separation
        heading = yaw+np.cumsum(w_mid)*dt-w_mid*dt/2
        xs = np.r_[x, x+np.cumsum(v_mid*np.cos(heading)*dt)]
        ys = np.r_[y, y+np.cumsum(v_mid*np.sin(heading)*dt)]
        bound = float(np.max(_arc_distances(xs, ys, distance, angle)))
        return bound+max(abs(left), abs(right))*dt

    def check(self, points, linear, angular, *, radius=.22,
              longitudinal_uncertainty=0., lateral_uncertainty=0., yaw_uncertainty=0.):
        if (not all(isfinite(v) for v in (linear, angular, radius,
                longitudinal_uncertainty, lateral_uncertainty, yaw_uncertainty))
                or radius <= 0. or min(longitudinal_uncertainty,
                    lateral_uncertainty, yaw_uncertainty) < 0.):
            raise ValueError('Nonfinite braking velocity')
        points = _cloud(points)
        wheel_speed = max(abs(linear-angular*self.wheel_separation/2),
                          abs(linear+angular*self.wheel_separation/2))
        duration = wheel_speed/self.wheel_deceleration
        effective_time = self.reaction_seconds+duration/2
        # This is a longitudinal stopping allowance, not radial body padding.
        if abs(linear) > 1.e-9:
            effective_time += self.stop_reserve/abs(linear)
        distance, angle = linear*effective_time, angular*effective_time
        curve_bound = self._curve_bound(linear, angular, duration, distance, angle)
        clearances = _clearances(points, distance, angle, radius, longitudinal_uncertainty,
                                lateral_uncertainty+curve_bound, yaw_uncertainty)
        hits = frozenset(int(v) for v in points[clearances <= 1.e-9, 0])
        minimum = float(np.min(clearances)) if len(points) else float('inf')
        return BrakeDecision(hits, minimum,
                             abs(distance), abs(angle), duration, curve_bound)

    def check_feedback(self, points, linear, angular, *, linear_delta=0., angular_delta=0.,
                       radius=.22, longitudinal_uncertainty=0., lateral_uncertainty=0.,
                       yaw_uncertainty=0.):
        """Enclose every signed feedback velocity, including yaw sign reversal.

        For f(u)=sign(u)*max(|u|-a*t,0), (f(v+h)+f(v-h))/2 is monotone in v
        and in |h| for a fixed sign of v. The difference f(v+h)-f(v-h) is
        monotone in h and |v| for a fixed sign of h. Thus the Cartesian input
        rectangle's marginal v/w extrema occur at its edges or its zero axes.

        Integrating these marginal velocity intervals and all sin/cos extrema
        encloses a continuous position box for *each whole time step*. This
        is an interval enclosure, not four trajectories with unchecked gaps.
        """
        values = (linear, angular, linear_delta, angular_delta, radius,
                  longitudinal_uncertainty, lateral_uncertainty, yaw_uncertainty)
        if (not all(isfinite(v) for v in values) or radius <= 0.
                or min(values[2:4]+values[5:]) < 0.):
            raise ValueError('Invalid measured braking interval')
        points = _cloud(points)
        vs = np.array([linear-linear_delta, np.clip(0., linear-linear_delta,
                        linear+linear_delta), linear+linear_delta])
        ws = np.array([angular-angular_delta, np.clip(0., angular-angular_delta,
                        angular+angular_delta), angular+angular_delta])
        v, w = np.meshgrid(vs, ws)
        left = v.ravel()-w.ravel()*self.wheel_separation/2
        right = v.ravel()+w.ravel()*self.wheel_separation/2
        duration = max(np.max(np.abs(left)), np.max(np.abs(right)))/self.wheel_deceleration
        count = max(1, int((duration+self.reaction_seconds)/.02)+1)
        times = np.unique(np.r_[np.linspace(0., duration+self.reaction_seconds, count+1),
                                self.reaction_seconds])
        elapsed = np.maximum(0., times-self.reaction_seconds)
        a = np.maximum(0., np.abs(left[:, None])-self.wheel_deceleration*elapsed)*np.sign(left[:, None])
        b = np.maximum(0., np.abs(right[:, None])-self.wheel_deceleration*elapsed)*np.sign(right[:, None])
        velocities, yaws = (a+b)/2, (b-a)/self.wheel_separation
        # Every fixed-input signed v and w approaches zero monotonically, so
        # endpoint extrema also enclose the full time interval between them.
        vlo = np.minimum(np.min(velocities[:, :-1], axis=0), np.min(velocities[:, 1:], axis=0))
        vhi = np.maximum(np.max(velocities[:, :-1], axis=0), np.max(velocities[:, 1:], axis=0))
        wlo = np.minimum(np.min(yaws[:, :-1], axis=0), np.min(yaws[:, 1:], axis=0))
        whi = np.maximum(np.max(yaws[:, :-1], axis=0), np.max(yaws[:, 1:], axis=0))
        dt = np.diff(times)
        theta_lo = np.r_[-yaw_uncertainty, -yaw_uncertainty+np.cumsum(wlo*dt)]
        theta_hi = np.r_[yaw_uncertainty, yaw_uncertainty+np.cumsum(whi*dt)]
        heading_lo = np.minimum(theta_lo[:-1], theta_lo[1:])
        heading_hi = np.maximum(theta_hi[:-1], theta_hi[1:])
        coslo, coshi = _trig_interval(heading_lo, heading_hi, cosine=True)
        sinlo, sinhi = _trig_interval(heading_lo, heading_hi, cosine=False)

        def integrate(low, high, uncertainty):
            candidates = np.stack((vlo*low, vlo*high, vhi*low, vhi*high))
            lo = np.r_[-uncertainty, -uncertainty+np.cumsum(np.min(candidates, axis=0)*dt)]
            hi = np.r_[uncertainty, uncertainty+np.cumsum(np.max(candidates, axis=0)*dt)]
            return lo, hi

        xlo, xhi = integrate(coslo, coshi, longitudinal_uncertainty)
        ylo, yhi = integrate(sinlo, sinhi, lateral_uncertainty)
        boxes = [np.minimum(xlo[:-1], xlo[1:]), np.maximum(xhi[:-1], xhi[1:]),
                 np.minimum(ylo[:-1], ylo[1:]), np.maximum(yhi[:-1], yhi[1:])]
        has_translation = max(abs(vs)) > 1.e-9
        if self.stop_reserve and has_translation:
            # Continue the stopped heading by the same longitudinal reserve;
            # retain both directions if the signed velocity interval crosses 0.
            lo, hi = _trig_interval(theta_lo[-1], theta_hi[-1], cosine=True)
            a, b = _trig_interval(theta_lo[-1], theta_hi[-1], cosine=False)
            signs = [1.] if vs[0] >= 0. else [-1.] if vs[-1] <= 0. else [-1., 1.]
            dx = [s*z*self.stop_reserve for s in signs for z in (lo, hi)]
            dy = [s*z*self.stop_reserve for s in signs for z in (a, b)]
            for i, bound in enumerate((xlo[-1]+min(0., *dx), xhi[-1]+max(0., *dx),
                                       ylo[-1]+min(0., *dy), yhi[-1]+max(0., *dy))):
                boxes[i] = np.r_[boxes[i], bound]
        hits, minimum = _box_clearances(points, boxes, radius)
        travel = float(np.max(np.hypot(np.maximum(np.abs(xlo), np.abs(xhi)),
                                      np.maximum(np.abs(ylo), np.abs(yhi)))))
        if has_translation:
            travel += self.stop_reserve
        return BrakeDecision(hits, minimum, travel,
            float(max(np.max(np.abs(theta_lo)), np.max(np.abs(theta_hi)))), duration, 0.)

    def limit(self, points, linear, angular, *, radius=.22, **uncertainty):
        """Return a checked common v/w scale; a clear aisle retains full speed.

        Curved stop trajectories need not be monotonic with speed. Every chosen
        candidate is checked; the bracket is only an optimization, never a
        claim that untested slower curves are collision-free.
        """
        # This cache contains geometry results only, never sensor freshness or
        # permission. The mux still validates original source/receipt on every
        # publish. Its repeated before/after-ramp check often uses one cloud.
        # The output proof already owns a prepared array. Converting it back
        # to hundreds of Python tuples just to build a one-use cache key is
        # more expensive than the geometry. Keep the array for this proof;
        # original scan/feedback age checks remain the mux's responsibility.
        if isinstance(points, np.ndarray):
            return self._limit_prepared(_cloud(points), linear, angular, radius, uncertainty)
        key = tuple(tuple(row) for row in points)
        return self._limit_cached(key, linear, angular, radius,
                                  tuple(sorted(uncertainty.items())))

    @lru_cache(maxsize=32)
    def _limit_cached(self, points, linear, angular, radius, uncertainty_items):
        return self._limit_prepared(_cloud(points), linear, angular, radius,
                                    dict(uncertainty_items))

    def _limit_prepared(self, points, linear, angular, radius, uncertainty):
        full = self.check(points, linear, angular, radius=radius, **uncertainty)
        if not full.collision:
            return 1., full
        unsafe = 1.
        for step in range(7, -1, -1):
            scale = step/8
            result = self.check(points, linear*scale, angular*scale,
                                radius=radius, **uncertainty)
            if not result.collision:
                safe = scale
                for _ in range(5):
                    middle = (safe+unsafe)/2
                    trial = self.check(points, linear*middle, angular*middle,
                                       radius=radius, **uncertainty)
                    if trial.collision:
                        unsafe = middle
                    else:
                        safe, result = middle, trial
                return safe, result
            unsafe = scale
        return 0., self.check(points, 0., 0., radius=radius, **uncertainty)
