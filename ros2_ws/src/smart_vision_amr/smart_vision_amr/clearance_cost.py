"""Nonnegative A* costs that prefer space around the robot's footprint."""

from __future__ import annotations

from math import isfinite

import cv2
import numpy as np

from smart_vision_amr.grid import Grid


def clearance_penalties(raw_grid: Grid, dynamic_obstacles: set[tuple[int, int]],
                        resolution: float, preferred_clearance: float,
                        gain: float) -> dict[tuple[int, int], float]:
    """Penalize near-wall *free* cells without ever making a blocked cell free.

    Clearance is approximated from raw occupied/unknown map cells, current
    dynamic returns, and the map boundary. These bounded costs only express a
    preference; inflated occupancy and full-footprint checks decide collisions.
    Even a narrow corridor with cost on every free cell remains traversable.
    Returned costs have the same units as one cardinal A* step and lie in 0..gain.
    """
    if not all(isfinite(value) for value in (resolution, preferred_clearance, gain)):
        raise ValueError("clearance parameters must be finite")
    if resolution <= 0 or preferred_clearance <= 0 or gain <= 0:
        return {}
    free = np.logical_not(np.asarray(raw_grid.rows, dtype=np.bool_)).astype(np.uint8)
    for x, y in dynamic_obstacles:
        if 0 <= x < raw_grid.width and 0 <= y < raw_grid.height:
            free[y, x] = 0
    # An unknown world beyond the map edge is also an obstacle. The padding
    # makes that boundary visible to distanceTransform.
    padded = np.pad(free, 1, mode="constant", constant_values=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    clearance = np.maximum(0.0, (distance - 0.5) * resolution)
    costly = (free != 0) & (clearance < preferred_clearance)
    ys, xs = np.nonzero(costly)
    return {(int(x), int(y)): float(gain * np.clip(
                (preferred_clearance - clearance[y, x]) / preferred_clearance, 0.0, 1.0))
            for y, x in zip(ys, xs)}
