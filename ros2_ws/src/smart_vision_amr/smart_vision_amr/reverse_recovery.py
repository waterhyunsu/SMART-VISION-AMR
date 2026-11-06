"""Choose a short, straight and map-clear retreat from recent odometry.

This module only selects a candidate. It never commands motion. The caller
must separately verify live rear LiDAR/local-costmap clearance and hand the
retreat to a collision-checking controller with manual/estop precedence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, floor, hypot, isfinite, pi, sin
from typing import Callable, Sequence

from smart_vision_amr.grid import Cell, Grid


@dataclass(frozen=True)
class Breadcrumb:
    """Odometry pose sampled during forward motion; stamp is monotonic seconds."""

    stamp: float
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class GridGeometry:
    """OccupancyGrid geometry in metres, including its origin rotation."""

    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float = 0.0


@dataclass(frozen=True)
class ReverseCandidate:
    """A verified straight retreat endpoint, in the same frame as history."""

    distance_m: float
    target_x: float
    target_y: float
    target_cell: Cell
    start_clearance_m: float
    target_clearance_m: float
    breadcrumb_count: int


def _local_xy(geometry: GridGeometry, x: float, y: float) -> tuple[float, float]:
    dx, dy = x - geometry.origin_x, y - geometry.origin_y
    c, s = cos(geometry.origin_yaw), sin(geometry.origin_yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _cell(geometry: GridGeometry, x: float, y: float) -> Cell:
    lx, ly = _local_xy(geometry, x, y)
    return floor(lx / geometry.resolution), floor(ly / geometry.resolution)


def _raw_clearance(
    grid: Grid, geometry: GridGeometry, occupied: tuple[Cell, ...], x: float, y: float
) -> float:
    """Exact point-to-occupied-cell-square/map-edge distance in metres."""

    lx, ly = _local_xy(geometry, x, y)
    resolution = geometry.resolution
    width_m, height_m = grid.width * resolution, grid.height * resolution
    if not 0.0 <= lx < width_m or not 0.0 <= ly < height_m:
        return float("-inf")
    distance = min(lx, ly, width_m - lx, height_m - ly)
    half = resolution / 2.0
    for cx, cy in occupied:
        ox, oy = (cx + 0.5) * resolution, (cy + 0.5) * resolution
        dx = max(abs(lx - ox) - half, 0.0)
        if dx > distance:
            continue
        dy = max(abs(ly - oy) - half, 0.0)
        if dy > distance:
            continue
        distance = min(distance, hypot(dx, dy))
    return distance


def _obstacle_clearance(
    geometry: GridGeometry, occupied: tuple[Cell, ...], x: float, y: float
) -> float:
    """Distance to a nominated physical blocker, excluding unrelated walls."""
    lx, ly = _local_xy(geometry, x, y)
    resolution = geometry.resolution
    return min((hypot(max(cx * resolution - lx, 0.0,
                          lx - (cx + 1) * resolution),
                      max(cy * resolution - ly, 0.0,
                          ly - (cy + 1) * resolution))
                for cx, cy in occupied), default=float("inf"))


def _nearby_sweep_obstacles(geometry, occupied, x, y, start_clearance, travel_bound):
    """Keep every possible nearest square over the bounded centre sweep.

    At any point within L of the start, nearest clearance is at most d0+L
    (distance is 1-Lipschitz). A square attaining that minimum is therefore
    within d0+2L of the start. The enclosing axis-aligned box keeps a superset;
    omitted squares cannot affect even the monotonic-clearance comparison.
    Map boundaries are evaluated separately by _raw_clearance as before.
    """
    lx, ly = _local_xy(geometry, x, y)
    reach, resolution = start_clearance+2*travel_bound, geometry.resolution
    return tuple((cx, cy) for cx, cy in occupied
                 if ((cx+1)*resolution >= lx-reach and cx*resolution <= lx+reach
                     and (cy+1)*resolution >= ly-reach and cy*resolution <= ly+reach))


def _valid_settings(values: Sequence[float]) -> bool:
    return all(isfinite(value) and value >= 0.0 for value in values)


def select_reverse_egress(
    history: Sequence[Breadcrumb],
    raw_grid: Grid,
    inflated_grid: Grid,
    geometry: GridGeometry,
    can_reach_goal: Callable[[Cell], bool],
    *,
    now: float,
    robot_radius: float,
    clearance_margin: float = 0.02,
    min_distance: float = 0.10,
    max_distance: float = 0.30,
    min_clearance_gain: float = 0.02,
    max_latest_age: float = 1.0,
    max_history_age: float = 15.0,
    max_sample_gap: float = 2.0,
    max_pose_jump: float = 0.15,
    max_heading_change: float = 0.15,
    max_lateral_error: float = 0.03,
    sample_step: float = 0.01,
    blocking_cells: set[Cell] | None = None,
    current_pose: tuple[float, float, float] | None = None,
    allow_parallel_static_egress: bool = False,
    clearance_neutral_egress_enabled: bool = False,
    verified_command_corridor: bool = False,
) -> ReverseCandidate | None:
    """Return the nearest safe straight retreat, or ``None``.

    ``history`` is ordered oldest to newest and must be in the same stationary
    map frame as ``geometry``. Its stamps are monotonic-clock seconds, not ROS
    timestamps. The newest pose is the current robot pose; earlier poses must
    lie behind it along the current chassis heading. ``can_reach_goal`` is an
    A* callback from a candidate cell to the *retained* mission goal, with
    current dynamic obstacles applied. A callback exception fails closed.

    Clearance is measured to raw occupied/unknown *cell areas* and map edges.
    Dense samples and a half-sample Lipschitz margin certify the intervening
    swept centre line, rather than just checking breadcrumb endpoints. This is
    not a substitute for live LiDAR or Nav2 collision checking during motion.

    When fresh observed ``blocking_cells`` are supplied, clearance must improve
    relative to that obstacle, while the entire chassis still clears *every*
    raw occupied/unknown square and boundary throughout the retreat. Otherwise
    the original conservative clearance-to-all-obstacles monotonic rule applies.
    This permits a useful retreat beside a wall without treating a small,
    physically clear change in distance to an unrelated wall as a collision.
    ``current_pose`` supplies the separately verified fresh chassis pose. It
    never replaces or rewrites an odometry breadcrumb: the actual BackUp ray
    is certified from this pose, while the original history still establishes
    the recent forward trail and its continuity.

    An independently certified, physically clear start may lie inside the
    *static inflation* of an obstacle even when a wall beside the robot keeps
    its nearest clearance constant throughout the retreat. The caller may
    enable ``allow_parallel_static_egress`` only for that situation. The
    selected endpoint must leave all inflation, retain the existing complete
    physical sweep clearance, and have a route to the retained goal.

    ``clearance_neutral_egress_enabled`` additionally permits small clearance
    decreases along a physically clear inflation escape. It applies only
    without a focused physical blocker and with an inflated/raw-free start.
    It waives the monotonic/gain preference, never the complete history and
    commanded-axis footprint checks, inflation-free endpoint, or goal oracle.
    The caller must independently authorize the fresh, stopped recovery.

    ``verified_command_corridor`` is available only to a caller that verified
    the odometry source/frame/continuity and projected its retained trace with
    one current paired transform. Historical headings need not equal today's
    heading: past positions must still remain within the existing narrow
    chassis-axis corridor. The complete raw trace and actual commanded straight
    chord are both checked, with unchanged footprint margins, endpoint and
    retained-goal requirements. Stationary duplicate positions do not erase an
    otherwise valid positional trace. The adapter proves continuity on every
    source sample, so gaps between sparse retained *spatial* anchors do not
    reapply ``max_sample_gap``; their original timestamps and history-age bound
    remain unchanged. This flag never establishes source or stop evidence by
    itself.
    """

    settings = (now, robot_radius, clearance_margin, min_distance, max_distance,
                min_clearance_gain, max_latest_age, max_history_age, max_sample_gap,
                max_pose_jump, max_heading_change, max_lateral_error, sample_step,
                geometry.resolution)
    if (not _valid_settings(settings) or geometry.resolution <= 0.0
            or not all(isfinite(value) for value in (
                geometry.origin_x, geometry.origin_y, geometry.origin_yaw))
            or sample_step <= 0.0 or min_distance <= 0.0
            or max_distance < min_distance or max_sample_gap <= 0.0
            or max_pose_jump <= 0.0
            or raw_grid.width != inflated_grid.width
            or raw_grid.height != inflated_grid.height
            or len(history) < 2
            or type(allow_parallel_static_egress) is not bool
            or type(clearance_neutral_egress_enabled) is not bool
            or type(verified_command_corridor) is not bool):
        return None

    latest = history[-1]
    if (not _valid_settings((latest.stamp,))
            or not all(isfinite(value) for value in (latest.x, latest.y))
            or not isfinite(latest.yaw)
            or latest.stamp > now
            or now - latest.stamp > max_latest_age):
        return None
    if current_pose is None:
        current_pose = latest.x, latest.y, latest.yaw
    if (len(current_pose) != 3 or not all(isfinite(value) for value in current_pose)
            or hypot(current_pose[0]-latest.x, current_pose[1]-latest.y) > .05+1e-9
            or abs((current_pose[2]-latest.yaw+pi) % (2*pi)-pi) > .15+1e-9):
        return None
    current_x, current_y, current_yaw = current_pose

    occupied = tuple((x, y) for y, row in enumerate(raw_grid.rows)
                     for x, blocked in enumerate(row) if blocked)
    if blocking_cells is not None and any(
            not isinstance(cell, tuple) or len(cell) != 2
            or any(type(value) is not int for value in cell)
            for cell in blocking_cells):
        return None
    focused = tuple(sorted(blocking_cells or ()))
    if focused and any(not raw_grid.contains(cell) or not raw_grid.is_static_obstacle(cell)
                       for cell in focused):
        return None
    start_clearance = _raw_clearance(raw_grid, geometry, occupied, current_x, current_y)
    # Never reinterpret insufficient physical clearance as a safe start. The
    # additional half-sample margin accounts for the unsampled point between
    # consecutive collision checks (distance-to-obstacle is 1-Lipschitz).
    required = robot_radius + clearance_margin + sample_step / 2.0
    if start_clearance + 1e-9 < required:
        return None
    parallel_static_egress = (allow_parallel_static_egress and not focused
                              and inflated_grid.is_static_obstacle(
                                  _cell(geometry, current_x, current_y))
                              and not raw_grid.is_static_obstacle(
                                  _cell(geometry, current_x, current_y)))
    clearance_neutral_egress = (clearance_neutral_egress_enabled and not focused
                                and inflated_grid.is_static_obstacle(
                                    _cell(geometry, current_x, current_y))
                                and not raw_grid.is_static_obstacle(
                                    _cell(geometry, current_x, current_y)))
    travel_bound = max(hypot(max_distance, max_lateral_error),
                       hypot(latest.x-current_x, latest.y-current_y))
    occupied = _nearby_sweep_obstacles(geometry, occupied, current_x, current_y,
                                      start_clearance, travel_bound)

    backward_x, backward_y = -cos(current_yaw), -sin(current_yaw)
    if (verified_command_corridor
            and abs((latest.x-current_x)*backward_y
                    -(latest.y-current_y)*backward_x) > max_lateral_error):
        return None
    start_progress_clearance = (_obstacle_clearance(geometry, focused, current_x, current_y)
                                if focused else start_clearance)
    previous, previous_clearance = latest, start_clearance
    previous_progress_clearance = start_progress_clearance
    used = 1
    for breadcrumb in reversed(history[:-1]):
        if (not _valid_settings((breadcrumb.stamp,))
                or not all(isfinite(value) for value in (breadcrumb.x, breadcrumb.y))
                or not isfinite(breadcrumb.yaw)
                or breadcrumb.stamp >= previous.stamp
                or latest.stamp - breadcrumb.stamp > max_history_age
                or (not verified_command_corridor
                    and previous.stamp - breadcrumb.stamp > max_sample_gap)):
            break
        segment = hypot(previous.x - breadcrumb.x, previous.y - breadcrumb.y)
        if segment > max_pose_jump:
            break
        if segment <= 1e-6:
            if verified_command_corridor:
                previous = breadcrumb
                used += 1
                continue
            break
        heading_error = (breadcrumb.yaw - current_yaw + pi) % (2.0 * pi) - pi
        if not verified_command_corridor and abs(heading_error) > max_heading_change:
            break
        dx, dy = breadcrumb.x - current_x, breadcrumb.y - current_y
        backward_distance = dx * backward_x + dy * backward_y
        lateral_error = abs(dx * backward_y - dy * backward_x)
        if (backward_distance <= 0.0 or lateral_error > max_lateral_error
                or backward_distance > max_distance + 1e-9):
            break

        steps = max(1, ceil(segment / sample_step))
        # The requested sample step bounds clearance loss anywhere between
        # sampled centres. Check the *entire* segment before accepting its end.
        for index in range(1, steps + 1):
            fraction = index / steps
            x = previous.x + fraction * (breadcrumb.x - previous.x)
            y = previous.y + fraction * (breadcrumb.y - previous.y)
            clearance = _raw_clearance(raw_grid, geometry, occupied, x, y)
            progress_clearance = (_obstacle_clearance(geometry, focused, x, y)
                                  if focused else clearance)
            if (clearance + 1e-9 < required
                    or (not clearance_neutral_egress
                        and progress_clearance + 1e-9 < previous_progress_clearance)):
                return None
            previous_clearance = clearance
            previous_progress_clearance = progress_clearance

        previous = breadcrumb
        used += 1
        # Nav2 BackUp moves on the current chassis axis. A breadcrumb can
        # legitimately differ sideways by a few centimetres; certify the
        # commanded straight ray too, rather than assuming the old polyline
        # and the actual manoeuvre sweep the same space.
        target_x = current_x + backward_distance * backward_x
        target_y = current_y + backward_distance * backward_y
        cell = _cell(geometry, target_x, target_y)
        gain = previous_progress_clearance - start_progress_clearance
        if (backward_distance < min_distance - 1e-9
                or (not parallel_static_egress and not clearance_neutral_egress
                    and gain < min_clearance_gain - 1e-9)
                or inflated_grid.is_static_obstacle(cell)):
            continue
        straight_steps = max(1, ceil(backward_distance / sample_step))
        straight_clearance = start_clearance
        straight_progress = start_progress_clearance
        for index in range(1, straight_steps + 1):
            distance = backward_distance * index / straight_steps
            x, y = current_x + distance * backward_x, current_y + distance * backward_y
            clearance = _raw_clearance(raw_grid, geometry, occupied, x, y)
            progress_clearance = (_obstacle_clearance(geometry, focused, x, y)
                                  if focused else clearance)
            if (clearance + 1e-9 < required
                    or (not clearance_neutral_egress
                        and progress_clearance + 1e-9 < straight_progress)):
                return None
            straight_clearance, straight_progress = clearance, progress_clearance
        if (not parallel_static_egress and not clearance_neutral_egress
                and straight_progress - start_progress_clearance
                    < min_clearance_gain - 1e-9):
            continue
        try:
            reachable = bool(can_reach_goal(cell))
        except Exception:
            reachable = False
        if reachable:
            return ReverseCandidate(backward_distance, target_x, target_y,
                                    cell, start_clearance, straight_clearance, used)
    return None
