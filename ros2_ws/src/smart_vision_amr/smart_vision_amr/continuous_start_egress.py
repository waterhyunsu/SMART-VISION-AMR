"""Certify one start-cell escape with the current continuous robot pose.

The normal inflated grid stays authoritative everywhere except its current
start cell.  A raw occupied/unknown map cell or a LiDAR hit is never relaxed.
"""

from __future__ import annotations

from math import ceil, hypot, isfinite

from .grid import Cell, Grid

Edge = tuple[Cell, Cell]


def certified_start_only_edges(
    raw_grid: Grid,
    inflated_grid: Grid,
    raw_dynamic: set[Cell],
    dynamic_inflated: set[Cell],
    start: Cell,
    pose_cells: tuple[float, float],
    radius_cells: float,
    margin_cells: float,
) -> set[Edge]:
    """Return swept-clear outgoing edges from one inflated, raw-free start.

    The half-cell-diagonal reserve in ordinary grid inflation protects an
    unknown pose anywhere inside a cell. Here the fresh continuous pose is
    known, so the actual pose, path start center, and first edge are checked
    against complete raw occupied squares and the map boundary. Samples are
    spaced by at most 0.2 cell; adding half that spacing to the required
    clearance certifies the entire swept segment by the 1-Lipschitz distance
    property. All later path cells still obey the unchanged inflated grid.
    """
    if (raw_grid.width != inflated_grid.width or raw_grid.height != inflated_grid.height
            or not raw_grid.contains(start) or raw_grid.is_static_obstacle(start)
            or start in raw_dynamic
            or not (inflated_grid.is_static_obstacle(start) or start in dynamic_inflated)
            or not all(isfinite(value) for value in (*pose_cells, radius_cells, margin_cells))
            or radius_cells <= 0.0 or margin_cells < 0.0):
        return set()
    px, py = pose_cells
    if not (start[0] <= px < start[0] + 1.0 and start[1] <= py < start[1] + 1.0):
        return set()

    required = radius_cells + margin_cells
    reach = ceil(required + 2.0)
    minimum_x = max(0, start[0] - reach)
    maximum_x = min(raw_grid.width - 1, start[0] + reach)
    minimum_y = max(0, start[1] - reach)
    maximum_y = min(raw_grid.height - 1, start[1] + reach)
    occupied = tuple(
        (x, y)
        for y in range(minimum_y, maximum_y + 1)
        for x in range(minimum_x, maximum_x + 1)
        if raw_grid.is_static_obstacle((x, y)) or (x, y) in raw_dynamic
    )

    def point_clear(point: tuple[float, float], reserve: float) -> bool:
        x, y = point
        threshold = required + reserve
        if min(x, y, raw_grid.width - x, raw_grid.height - y) + 1e-9 < threshold:
            return False
        for ox, oy in occupied:
            distance = hypot(max(ox - x, 0.0, x - (ox + 1.0)),
                             max(oy - y, 0.0, y - (oy + 1.0)))
            if distance + 1e-9 < threshold:
                return False
        return True

    def swept_clear(a: tuple[float, float], b: tuple[float, float]) -> bool:
        length = hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, ceil(length / 0.2))
        reserve = length / (2.0 * steps)
        return all(point_clear((a[0] + (b[0] - a[0]) * step / steps,
                                a[1] + (b[1] - a[1]) * step / steps), reserve)
                   for step in range(steps + 1))

    center = (start[0] + 0.5, start[1] + 0.5)
    if not point_clear(pose_cells, 0.0) or not swept_clear(pose_cells, center):
        return set()
    allowed: set[Edge] = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0:
                continue
            neighbor = (start[0] + dx, start[1] + dy)
            if (inflated_grid.is_static_obstacle(neighbor)
                    or neighbor in dynamic_inflated or neighbor in raw_dynamic):
                continue
            if dx and dy and any(
                inflated_grid.is_static_obstacle(side) or side in dynamic_inflated
                for side in ((start[0] + dx, start[1]), (start[0], start[1] + dy))
            ):
                continue
            destination = (neighbor[0] + 0.5, neighbor[1] + 0.5)
            if (swept_clear(center, destination)
                    and swept_clear(pose_cells, destination)):
                allowed.add((start, neighbor))
    return allowed
