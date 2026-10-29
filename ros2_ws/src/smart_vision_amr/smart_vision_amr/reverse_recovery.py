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
            or len(history) < 2):
        return None

    latest = history[-1]
    if (not _valid_settings((latest.stamp,))
            or not all(isfinite(value) for value in (latest.x, latest.y))
            or not isfinite(latest.yaw)
            or latest.stamp > now
            or now - latest.stamp > max_latest_age):
        return None

    occupied = tuple((x, y) for y, row in enumerate(raw_grid.rows)
                     for x, blocked in enumerate(row) if blocked)
    start_clearance = _raw_clearance(raw_grid, geometry, occupied, latest.x, latest.y)
    # Never reinterpret insufficient physical clearance as a safe start. The
    # additional half-sample margin accounts for the unsampled point between
    # consecutive collision checks (distance-to-obstacle is 1-Lipschitz).
    required = robot_radius + clearance_margin + sample_step / 2.0
    if start_clearance + 1e-9 < required:
        return None

    backward_x, backward_y = -cos(latest.yaw), -sin(latest.yaw)
    previous, previous_clearance = latest, start_clearance
    used = 1
    for breadcrumb in reversed(history[:-1]):
        if (not _valid_settings((breadcrumb.stamp,))
                or not all(isfinite(value) for value in (breadcrumb.x, breadcrumb.y))
                or not isfinite(breadcrumb.yaw)
                or breadcrumb.stamp >= previous.stamp
                or latest.stamp - breadcrumb.stamp > max_history_age
                or previous.stamp - breadcrumb.stamp > max_sample_gap):
            break
        segment = hypot(previous.x - breadcrumb.x, previous.y - breadcrumb.y)
        if segment > max_pose_jump or segment <= 1e-6:
            break
        heading_error = (breadcrumb.yaw - latest.yaw + pi) % (2.0 * pi) - pi
        if abs(heading_error) > max_heading_change:
            break
        dx, dy = breadcrumb.x - latest.x, breadcrumb.y - latest.y
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
            if (clearance + 1e-9 < required
                    or clearance + 1e-9 < previous_clearance):
                return None
            previous_clearance = clearance

        previous = breadcrumb
        used += 1
        cell = _cell(geometry, breadcrumb.x, breadcrumb.y)
        if (backward_distance < min_distance - 1e-9
                or previous_clearance - start_clearance < min_clearance_gain - 1e-9
                or inflated_grid.is_static_obstacle(cell)):
            continue
        try:
            reachable = bool(can_reach_goal(cell))
        except Exception:
            reachable = False
        if reachable:
            return ReverseCandidate(backward_distance, breadcrumb.x, breadcrumb.y,
                                    cell, start_clearance, previous_clearance, used)
    return None
