"""Small ROS-independent helpers for comparing navigation path geometry."""

from __future__ import annotations

from math import hypot
from typing import Sequence


Point = tuple[float, float]


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
