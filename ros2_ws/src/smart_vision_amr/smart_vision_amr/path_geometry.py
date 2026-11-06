"""Small ROS-independent helpers for comparing navigation path geometry."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, floor, hypot, isfinite, sqrt
from typing import Callable, Sequence


Point = tuple[float, float]


def _segment_events(start: Point, end: Point) -> list[float]:
    """Parameters at every tile boundary of a continuous line segment."""
    if not all(isfinite(v) for point in (start, end) for v in point):
        raise ValueError("segment coordinates must be finite")
    events = {0.0, 1.0}
    for origin, finish in zip(start, end):
        delta = finish - origin
        if abs(delta) < 1e-12:
            continue
        for boundary in range(ceil(min(origin, finish)), floor(max(origin, finish)) + 1):
            event = (boundary - origin) / delta
            if 0.0 <= event <= 1.0:
                events.add(event)
    return sorted(events)


def _point_cells(x: float, y: float) -> set[tuple[int, int]]:
    xs, ys = {floor(x)}, {floor(y)}
    if abs(x - round(x)) < 1e-10:
        xs.update((round(x) - 1, round(x)))
    if abs(y - round(y)) < 1e-10:
        ys.update((round(y) - 1, round(y)))
    return {(cx, cy) for cx in xs for cy in ys}


def segment_cells(start: Point, end: Point) -> set[tuple[int, int]]:
    """Cells touched by a continuous segment, including corner/edge contact.

    Coordinates use tile boundaries at integers, so cell (x, y)'s center is
    (x + .5, y + .5). Checking only Bresenham cell centers misses corner cuts.
    Grid-crossing events and their intervening intervals cover the exact line.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    times = _segment_events(start, end)
    samples = times + [(a + b) * .5 for a, b in zip(times, times[1:])]
    touched = set()
    for t in samples:
        x, y = start[0] + t * dx, start[1] + t * dy
        touched.update(_point_cells(x, y))
    return touched


def path_traversal_cost(points: Sequence[Point],
                        cell_penalty: Mapping[tuple[int, int], float]) -> float:
    """Integrate length and nonnegative soft cost through exact tile intervals.

    This objective lets continuous smoothing retain a clearance-aware A* route.
    A shortcut across a costlier wall-side strip must earn its shorter length;
    collision freedom alone does not establish that it is a better route.
    Boundary-aligned segments use the larger adjacent cost. Zero-length corner
    contacts do not add artificial cost, and resampling leaves this integral
    unchanged. Distances and costs are in grid-cell units.
    """
    result = 0.0
    for a, b in zip(points, points[1:]):
        length = hypot(b[0] - a[0], b[1] - a[1])
        times = _segment_events(a, b)
        for left, right in zip(times, times[1:]):
            t = (left + right) * .5
            x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            penalty = max((cell_penalty.get(cell, 0.0) for cell in _point_cells(x, y)),
                          default=0.0)
            if not isfinite(penalty) or penalty < 0.0:
                raise ValueError("cell penalties must be finite and nonnegative")
            result += length * (right - left) * (1.0 + penalty)
    return result


def route_cells(points: Sequence[Point]) -> set[tuple[int, int]]:
    if len(points) == 1:
        return segment_cells(points[0], points[0])
    return {cell for a, b in zip(points, points[1:]) for cell in segment_cells(a, b)}


def collision_free(points: Sequence[Point], blocked: Callable[[tuple[int, int]], bool]) -> bool:
    return bool(points) and not any(blocked(cell) for cell in route_cells(points))


def resample_path(points: Sequence[Point], spacing: float) -> list[Point]:
    """Densify every segment without moving endpoints or connecting corners."""
    if not points or not isfinite(spacing) or spacing <= 0:
        raise ValueError("a nonempty path and positive finite spacing are required")
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        length = hypot(b[0] - a[0], b[1] - a[1])
        if length < 1e-12:
            continue
        count = max(1, ceil(length / spacing))
        result.extend((a[0] + (b[0] - a[0]) * index / count,
                       a[1] + (b[1] - a[1]) * index / count)
                      for index in range(1, count + 1))
    return result


def smooth_grid_path(cells: Sequence[tuple[int, int]],
                     blocked: Callable[[tuple[int, int]], bool], *,
                     spacing: float = 1.0, shortcut_distance: float = 12.0,
                     iterations: int = 2,
                     cell_penalty: Mapping[tuple[int, int], float] | None = None
                     ) -> tuple[list[Point], bool]:
    """Visibility shortcuts + Chaikin corner rounding, certified on inflated grid.

    All original cells, unknown cells, dynamic inflation, and map edges are
    hard blocks. An already certified start-egress path may begin in inflation;
    leave that original route untouched rather than weakening this checker.
    Unsafe corners remain sharp while certified shortcuts and other rounded
    corners are kept. If the original route is not clear, leave it unchanged.
    Supplied soft costs are retained: no shortcut or rounded corner may increase
    the integrated length-plus-clearance cost of the part it replaces.
    """
    original = [(x + .5, y + .5) for x, y in cells]
    if len(original) < 3 or not collision_free(original, blocked):
        return original, False
    if (not isfinite(shortcut_distance) or shortcut_distance <= 0
            or not isfinite(spacing) or spacing <= 0):
        return original, False
    penalties = cell_penalty or {}

    def acceptable(candidate: Sequence[Point], reference: Sequence[Point]) -> bool:
        if not collision_free(candidate, blocked):
            return False
        return (not penalties or path_traversal_cost(candidate, penalties)
                <= path_traversal_cost(reference, penalties) + 1e-8)

    simplified = [original[0]]
    index = 0
    while index < len(original) - 1:
        candidate = index + 1
        for end in range(index + 2, len(original)):
            if hypot(original[end][0] - original[index][0],
                     original[end][1] - original[index][1]) > shortcut_distance:
                break
            if acceptable([original[index], original[end]], original[index:end + 1]):
                candidate = end
        simplified.append(original[candidate])
        index = candidate
    # A tight corner can be clear as a grid route but too close to a wall to
    # round. Certify each corner independently: one unsafe bend must not throw
    # away visibility shortcuts and safe rounding elsewhere on the route.
    candidate = simplified
    for _ in range(max(0, min(4, int(iterations)))):
        rounded = [candidate[0]]
        for index in range(1, len(candidate) - 1):
            before, corner, after = candidate[index-1:index+2]
            entry = (.75*corner[0] + .25*before[0],
                     .75*corner[1] + .25*before[1])
            exit = (.75*corner[0] + .25*after[0],
                    .75*corner[1] + .25*after[1])
            if acceptable([before, entry, exit, after], [before, corner, after]):
                rounded.extend((entry, exit))
            else:
                rounded.append(corner)
        rounded.append(candidate[-1])
        if not collision_free(rounded, blocked):
            break
        candidate = rounded
    result = resample_path(candidate, spacing)
    if not collision_free(result, blocked):
        return original, False
    if penalties and (path_traversal_cost(result, penalties)
                      > path_traversal_cost(original, penalties) + 1e-8):
        # Independently accepted corners should compose without an increase;
        # preserve the certified simplification if floating-point boundary
        # interactions invalidate that assumption.
        result = resample_path(simplified, spacing)
    return result, result != original


def path_length(points: Sequence[Point]) -> float:
    return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def path_progress(points: Sequence[Point], position: Point) -> float:
    """Arc length at the nearest projection, preferring earlier ties at crossings."""
    if not all(isfinite(value) for value in position):
        raise ValueError('position must be finite')
    best = (float('inf'), 0.)
    traveled = 0.
    for a, b in zip(points, points[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        squared = dx*dx+dy*dy
        length = sqrt(squared)
        t = 0. if squared < 1e-12 else max(0., min(1., ((position[0]-a[0])*dx+(position[1]-a[1])*dy)/squared))
        distance_squared = (position[0]-a[0]-t*dx)**2+(position[1]-a[1]-t*dy)**2
        candidate = (distance_squared, traveled+t*length)
        if candidate < best:
            best = candidate
        traveled += length
    return best[1]


def point_at_progress(points: Sequence[Point], distance: float) -> Point:
    """Interpolate a look-ahead target by distance along the polyline."""
    if not points:
        raise ValueError('a nonempty path is required')
    remaining = max(0., distance)
    for a, b in zip(points, points[1:]):
        length = hypot(b[0]-a[0], b[1]-a[1])
        if length > 1e-12 and remaining <= length:
            t = remaining/length
            return a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])
        remaining -= length
    return points[-1]


def upcoming_curvature(points: Sequence[Point], progress: float, horizon: float,
                       minimum_chord: float = .15) -> float:
    """Largest circumcircle curvature in the upcoming arc-length interval."""
    return max((curvature for start, end, _, curvature in
                _curvature_samples(points, minimum_chord)
                if end >= progress and start <= progress+horizon), default=0.)


def _curvature_samples(points: Sequence[Point], minimum_chord: float):
    """Yield local support bounds, vertex distance and curvature along a route."""
    if len(points) < 3:
        return
    distances = [0.0]
    for a, b in zip(points, points[1:]):
        distances.append(distances[-1] + hypot(b[0]-a[0], b[1]-a[1]))
    for index in range(1, len(points)-1):
        before, after = index-1, index+1
        while before > 0 and distances[index]-distances[before] < minimum_chord:
            before -= 1
        while after < len(points)-1 and distances[after]-distances[index] < minimum_chord:
            after += 1
        a, b, c = points[before], points[index], points[after]
        ab, bc, ac = hypot(b[0]-a[0], b[1]-a[1]), hypot(c[0]-b[0], c[1]-b[1]), hypot(c[0]-a[0], c[1]-a[1])
        denominator = ab * bc * ac
        if denominator <= 1e-12:
            continue
        cross = abs((b[0]-a[0]) * (c[1]-a[1]) - (b[1]-a[1]) * (c[0]-a[0]))
        yield (distances[index-1], distances[index+1], distances[index],
               2.0 * cross / denominator)


def anticipatory_curve_speed_limit(points: Sequence[Point], progress: float,
                                  horizon: float, max_speed: float,
                                  max_angular_speed: float,
                                  max_lateral_acceleration: float,
                                  minimum_speed: float = .12,
                                  deceleration: float = .10,
                                  braking_margin: float = .15,
                                  minimum_chord: float = .15) -> float:
    """Brake before bends with v² <= bend_speed² + 2*a*remaining_distance.

    Unlike applying the worst upcoming bend cap to the entire preview window,
    this bound falls continuously as the robot approaches a corner. It is a
    route speed ceiling, never a substitute for collision or emergency stops.
    """
    if (not all(isfinite(v) for v in (progress, horizon, minimum_speed,
                                     deceleration, braking_margin, minimum_chord))
            or progress < 0 or horizon <= 0 or deceleration <= 0
            or not 0 < minimum_speed <= max_speed
            or braking_margin < 0 or minimum_chord <= 0):
        raise ValueError('Invalid anticipatory braking limits')
    # Validate the common velocity and acceleration limits even on a straight.
    speed = curvature_speed_limit(0., max_speed, max_angular_speed,
                                  max_lateral_acceleration)
    for start, end, vertex, curvature in _curvature_samples(points, minimum_chord):
        if end < progress or start > progress+horizon:
            continue
        bend_speed = max(minimum_speed, curvature_speed_limit(
            curvature, max_speed, max_angular_speed, max_lateral_acceleration))
        remaining = max(0., vertex-progress-braking_margin)
        speed = min(speed, sqrt(bend_speed*bend_speed+2*deceleration*remaining))
    return speed


def curvature_speed_limit(curvature: float, max_speed: float,
                          max_angular_speed: float, max_lateral_acceleration: float) -> float:
    """Upper bound from angular velocity and lateral acceleration (v² κ)."""
    if not all(isfinite(v) and v > 0 for v in (max_speed, max_angular_speed, max_lateral_acceleration)):
        raise ValueError("speed/acceleration limits must be finite and positive")
    if not isfinite(curvature) or curvature < 0:
        raise ValueError("curvature must be finite and nonnegative")
    if curvature < 1e-9:
        return max_speed
    return min(max_speed, max_angular_speed / curvature,
               sqrt(max_lateral_acceleration / curvature))


def paths_are_equivalent(reference: Sequence[Point], candidate: Sequence[Point],
                         tolerance: float = 0.20, start_tolerance: float = 0.35) -> bool:
    """Return whether a new path follows the same corridor after robot progress.

    The new path may start farther along the old path. Compare its geometry with
    the corresponding remaining suffix, so advancing the start pose alone does
    not look like a route change. A sufficiently large deviation still replaces
    the active controller goal.
    """
    if not reference or not candidate:
        return False

    start = candidate[0]
    nearest = min(range(len(reference)),
                  key=lambda i: hypot(reference[i][0] - start[0], reference[i][1] - start[1]))
    if hypot(reference[nearest][0] - start[0], reference[nearest][1] - start[1]) > start_tolerance:
        return False
    suffix = reference[nearest:]

    def directed_distance(source: Sequence[Point], target: Sequence[Point]) -> float:
        return max(min(hypot(x - tx, y - ty) for tx, ty in target) for x, y in source)

    deviation = max(directed_distance(suffix, candidate), directed_distance(candidate, suffix))
    return deviation <= tolerance


def paths_share_remaining_geometry(reference: Sequence[Point], candidate: Sequence[Point],
                                   tolerance: float = .03) -> bool:
    """Certify the same ordered polyline suffix, not merely the same corridor.

    Clip at the candidate start's segment projection so pure progress and
    sparse/dense resampling do not cancel FollowPath. Densified point-to-segment
    distances include a Lipschitz reserve: every point between samples remains
    within ``tolerance``. Matching progress must be monotonic in both directions;
    ambiguous crossings conservatively require a replacement action. This
    stricter check is only for active-action deduplication, never Spin/progress.
    """
    if (not reference or not candidate or not isfinite(tolerance)
            or not 0. < tolerance <= .03
            or not all(isfinite(value) for path in (reference, candidate)
                       for point in path for value in point)):
        return False

    def segments(path):
        result, traveled = [], 0.
        for a, b in zip(path, path[1:]):
            dx, dy = b[0]-a[0], b[1]-a[1]
            squared = dx*dx+dy*dy
            length = sqrt(squared)
            result.append((a[0], a[1], dx, dy, squared, length, traveled,
                           min(a[0], b[0]), max(a[0], b[0]),
                           min(a[1], b[1]), max(a[1], b[1])))
            traveled += length
        return result

    def project(path, prepared, point):
        if not prepared:
            return hypot(point[0]-path[0][0], point[1]-path[0][1]), 0.
        best = (float('inf'), 0.)
        px, py = point
        for ax, ay, dx, dy, squared, length, traveled, lx, hx, ly, hy in prepared:
            # Most distant segments cannot improve the nearest projection.
            # Reusing bounds avoids O(samples * segments) square roots.
            bx = lx-px if px < lx else px-hx if px > hx else 0.
            by = ly-py if py < ly else py-hy if py > hy else 0.
            if bx*bx+by*by > best[0]:
                continue
            t = (0. if squared < 1.e-24 else
                 max(0., min(1., ((px-ax)*dx+(py-ay)*dy)/squared)))
            item = ((px-ax-t*dx)**2+(py-ay-t*dy)**2, traveled+t*length)
            best = min(best, item)
        return sqrt(best[0]), best[1]

    distance, progress = project(reference, segments(reference), candidate[0])
    if distance > tolerance:
        return False
    suffix = [point_at_progress(reference, progress)]
    traveled = 0.
    for a, b in zip(reference, reference[1:]):
        traveled += hypot(b[0]-a[0], b[1]-a[1])
        if traveled > progress+1.e-10:
            suffix.append(b)

    def ordered_near(source, target):
        previous = 0.
        prepared = segments(target)
        # Adjacent samples are <= tolerance/2 apart. Distance to a fixed
        # polyline is 1-Lipschitz, so tolerance/4 covers unsampled interiors.
        for point in resample_path(source, tolerance/2.):
            distance, advance = project(target, prepared, point)
            if distance > tolerance*.75+1.e-10 or advance+1.e-8 < previous:
                return False
            previous = advance
        return True

    return ordered_near(suffix, candidate) and ordered_near(candidate, suffix)
