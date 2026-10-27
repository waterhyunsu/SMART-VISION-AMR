"""Small ROS-independent helpers for comparing navigation path geometry."""

from __future__ import annotations

from math import hypot
from typing import Callable, Sequence


Point = tuple[float, float]
Cell = tuple[int, int]


def supercover_cells(start: Cell, end: Cell) -> list[Cell]:
    """Return every grid cell touched by the segment between cell centers.

    When the segment passes exactly through a grid corner, include both side
    cells too. This prevents path simplification from clipping obstacle corners.
    """
    x, y = start
    dx, dy = end[0] - x, end[1] - y
    nx, ny = abs(dx), abs(dy)
    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0
    ix = iy = 0
    cells = [(x, y)]

    while ix < nx or iy < ny:
        decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if decision == 0:
            cells.append((x + step_x, y))
            cells.append((x, y + step_y))
            x += step_x
            y += step_y
            ix += 1
            iy += 1
            cells.append((x, y))
        elif decision < 0:
            x += step_x
            ix += 1
            cells.append((x, y))
        else:
            y += step_y
            iy += 1
            cells.append((x, y))

    return cells


def line_is_clear(start: Cell, end: Cell, is_blocked: Callable[[Cell], bool]) -> bool:
    """Return whether a straight grid segment avoids every blocked cell."""
    return all(not is_blocked(cell) for cell in supercover_cells(start, end))


def smooth_grid_path(path: Sequence[Cell], is_blocked: Callable[[Cell], bool]) -> list[Cell]:
    """Remove unnecessary A* waypoints while preserving collision-free lines."""
    if len(path) <= 2:
        return list(path)

    result = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1 and not line_is_clear(
                path[anchor], path[candidate], is_blocked):
            candidate -= 1
        result.append(path[candidate])
        anchor = candidate
    return result


def paths_are_equivalent(reference: Sequence[Point], candidate: Sequence[Point],
                         tolerance: float = 0.12, start_tolerance: float = 0.35) -> bool:
    """Return whether a new path follows the same corridor after robot progress.

    The new path may start farther along the old path. Compare its geometry with
    the corresponding remaining suffix, so advancing the start pose alone does
    not look like a route change. Small scan-grid jitter is ignored, while a
    material route change still replaces the active controller goal.
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
