"""Short-lived, directional execution failures, independent of the SLAM map.

A failed *approach* is not an occupied cell. Only directed transitions close
and parallel to that approach are prohibited. Reverse/perpendicular motion
and approaches outside the small affected area stay available. Callers use a
steady clock, stable map geometry scope, and target-coordinate goal scope.
Changing live obstacle observations must not reset this memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, ceil, cos, floor, hypot, isfinite, pi, sin
from typing import Iterable

from .astar import Edge, NEIGHBORS
from .grid import Grid
from .reverse_recovery import GridGeometry

EXECUTION_FAILURE_REASONS = frozenset({
    "execution_no_progress", "execution_oscillation", "controller_aborted",
})


def _angle_difference(a: float, b: float) -> float:
    return atan2(sin(a - b), cos(a - b))


def _segment_distance(x: float, y: float, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = dx * dx + dy * dy
    fraction = (max(0.0, min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / norm))
                if norm else 0.0)
    return hypot(x - a[0] - fraction * dx, y - a[1] - fraction * dy)


@dataclass
class FailureEntry:
    id: int
    x: float
    y: float
    approach_heading: float
    reason: str
    first_seen: float
    last_seen: float
    expires_at: float
    map_scope: str
    goal_scope: str
    occurrences: int = 1
    event_ids: list[str] = field(default_factory=list)


class FailureMemory:
    """Bounded temporary state; never mutates or persists a supplied map.

    ``radius_m`` is a centre-line approach area, not a replacement collision
    radius. A surrounding ring up to twice that radius adds ``penalty_cost``
    cardinal-step units to matching directed edges. Costs never grant clearance.
    Failure IDs may be released only after independently verified successful
    execution, not merely after publishing a path or seeing a free LiDAR ray.
    """

    FORMAT_VERSION = 1

    def __init__(self, ttl_sec: float = 120.0, radius_m: float = 0.25,
                 heading_tolerance_rad: float = pi / 3, max_entries: int = 64,
                 penalty_cost: float = 2.0):
        if (not all(isfinite(value) for value in (
                ttl_sec, radius_m, heading_tolerance_rad, penalty_cost))
                or ttl_sec <= 0 or radius_m <= 0
                or not 0 < heading_tolerance_rad < pi / 2
                or penalty_cost < 0 or isinstance(max_entries, bool)
                or not isinstance(max_entries, int) or max_entries < 1):
            raise ValueError("failure-memory settings must be finite and positive; "
                             "heading tolerance must be below 90 degrees")
        self.ttl_sec = float(ttl_sec)
        self.radius_m = float(radius_m)
        self.heading_tolerance_rad = float(heading_tolerance_rad)
        self.max_entries = max_entries
        self.penalty_cost = float(penalty_cost)
        self._entries: dict[int, FailureEntry] = {}
        self._next_id = 1
        self.expired_count = 0
        self.evicted_count = 0
        self.released_count = 0

    @staticmethod
    def _validate_time(now):
        if not isfinite(now) or now < 0:
            raise ValueError("failure-memory time must be finite steady-clock seconds")

    def expire(self, now: float) -> int:
        self._validate_time(now)
        expired = [key for key, item in self._entries.items() if now >= item.expires_at]
        for key in expired:
            del self._entries[key]
        self.expired_count += len(expired)
        return len(expired)

    def record_failure(self, *, x: float, y: float, approach_heading: float,
                       reason: str, now: float, map_scope: str, goal_scope: str,
                       failure_id: str | None = None) -> int | None:
        """Record a real execution failure; non-motion reasons are ignored.

        A repeated event ID cannot extend TTL or increase occurrence counts.
        Nearby failures in the same direction/scope coalesce without moving the
        original centre (which would let repeated reports drift the restriction).
        Only a genuinely later event can extend a coalesced entry's TTL.
        """
        self.expire(now)
        if reason not in EXECUTION_FAILURE_REASONS:
            return None
        if (not all(isfinite(value) for value in (x, y, approach_heading))
                or not isinstance(map_scope, str) or not map_scope
                or not isinstance(goal_scope, str) or not goal_scope
                or (failure_id is not None
                    and (not isinstance(failure_id, str) or not failure_id))):
            raise ValueError("failure requires finite pose and nonempty scopes/event ID")
        heading = atan2(sin(approach_heading), cos(approach_heading))
        scoped = [entry for entry in self._entries.values()
                  if entry.map_scope == map_scope and entry.goal_scope == goal_scope]
        if failure_id is not None:
            for entry in scoped:
                if failure_id in entry.event_ids:
                    return entry.id
        matches = [entry for entry in scoped
                   if hypot(entry.x - x, entry.y - y) <= self.radius_m
                   and abs(_angle_difference(entry.approach_heading, heading))
                   <= self.heading_tolerance_rad / 2]
        if matches:
            entry = min(matches, key=lambda item: hypot(item.x - x, item.y - y))
            if now < entry.last_seen:
                return entry.id
            entry.last_seen = now
            entry.expires_at = now + self.ttl_sec
            entry.reason = reason
            entry.occurrences += 1
            if failure_id is not None:
                entry.event_ids.append(failure_id)
                entry.event_ids[:] = entry.event_ids[-16:]
            return entry.id
        if len(self._entries) >= self.max_entries:
            oldest = min(self._entries.values(), key=lambda item: (item.last_seen, item.id))
            del self._entries[oldest.id]
            self.evicted_count += 1
        entry = FailureEntry(
            self._next_id, float(x), float(y), heading, reason, now, now,
            now + self.ttl_sec, map_scope, goal_scope,
            event_ids=[failure_id] if failure_id is not None else [])
        self._entries[entry.id] = entry
        self._next_id += 1
        return entry.id

    def release_verified(self, failure_ids: Iterable[int], *, now: float) -> int:
        """Release IDs after verified execution or removal of their physical blocker."""
        self.expire(now)
        removed = 0
        for key in set(failure_ids):
            if key in self._entries:
                del self._entries[key]
                removed += 1
        self.released_count += removed
        return removed

    def _matching(self, now, map_scope, goal_scope):
        self.expire(now)
        return [entry for entry in self._entries.values()
                if entry.map_scope == map_scope and entry.goal_scope == goal_scope]

    def segment_blocked(self, a: tuple[float, float], b: tuple[float, float], *,
                        now: float, map_scope: str, goal_scope: str) -> bool:
        """Check a directed world segment, including smoothing shortcuts.

        This only enforces failure memory; the caller must also collision-check
        the complete robot footprint against inflated occupancy/live obstacles.
        """
        if not all(isfinite(value) for value in (*a, *b)):
            raise ValueError("segment coordinates must be finite")
        entries = self._matching(now, map_scope, goal_scope)
        if hypot(b[0] - a[0], b[1] - a[1]) <= 1e-12:
            return False
        heading = atan2(b[1] - a[1], b[0] - a[0])
        return any(abs(_angle_difference(heading, entry.approach_heading))
                   <= self.heading_tolerance_rad + 1e-12
                   and _segment_distance(entry.x, entry.y, a, b)
                   <= self.radius_m + 1e-12 for entry in entries)

    def constraints(self, grid: Grid, geometry: GridGeometry, *, now: float,
                    map_scope: str, goal_scope: str) -> tuple[set[Edge], dict[Edge, float]]:
        """Rasterize local directed constraints, leaving the grid untouched."""
        if (not all(isfinite(value) for value in (
                geometry.resolution, geometry.origin_x,
                geometry.origin_y, geometry.origin_yaw)) or geometry.resolution <= 0):
            raise ValueError("grid geometry must be finite with positive resolution")
        entries = self._matching(now, map_scope, goal_scope)
        blocked: set[Edge] = set()
        costs: dict[Edge, float] = {}
        c, s = cos(geometry.origin_yaw), sin(geometry.origin_yaw)
        resolution = geometry.resolution

        def world(cell):
            lx, ly = (cell[0] + .5) * resolution, (cell[1] + .5) * resolution
            return (geometry.origin_x + c * lx - s * ly,
                    geometry.origin_y + s * lx + c * ly)

        for entry in entries:
            dx, dy = entry.x - geometry.origin_x, entry.y - geometry.origin_y
            center = (floor((c * dx + s * dy) / resolution),
                      floor((-s * dx + c * dy) / resolution))
            # An extra cell catches segments whose endpoints lie outside the
            # radius but whose intervening centre line crosses it.
            reach = ceil(2 * self.radius_m / resolution) + 1
            for y in range(max(0, center[1] - reach), min(grid.height, center[1] + reach + 1)):
                for x in range(max(0, center[0] - reach), min(grid.width, center[0] + reach + 1)):
                    a = x, y
                    wa = world(a)
                    for mx, my, _ in NEIGHBORS:
                        b = x + mx, y + my
                        if not grid.contains(b):
                            continue
                        heading = geometry.origin_yaw + atan2(my, mx)
                        if abs(_angle_difference(heading, entry.approach_heading)) > self.heading_tolerance_rad + 1e-12:
                            continue
                        distance = _segment_distance(entry.x, entry.y, wa, world(b))
                        edge = a, b
                        if distance <= self.radius_m + 1e-12:
                            blocked.add(edge)
                        elif distance <= 2 * self.radius_m and self.penalty_cost:
                            # Repeated/coalesced reports do not compound costs.
                            costs[edge] = max(costs.get(edge, 0.0), self.penalty_cost)
        for edge in blocked:
            costs.pop(edge, None)
        return blocked, costs

    def snapshot(self, *, now: float, map_scope: str | None = None,
                 goal_scope: str | None = None) -> dict:
        self.expire(now)
        entries = [entry for entry in self._entries.values()
                   if (map_scope is None or entry.map_scope == map_scope)
                   and (goal_scope is None or entry.goal_scope == goal_scope)]
        return {
            "format_version": self.FORMAT_VERSION,
            "count": len(entries), "total_count": len(self._entries),
            "ttl_sec": self.ttl_sec, "radius_m": self.radius_m,
            "heading_tolerance_rad": self.heading_tolerance_rad,
            "expired_count": self.expired_count, "evicted_count": self.evicted_count,
            "released_count": self.released_count,
            "entries": [{
                "id": entry.id, "x": entry.x, "y": entry.y,
                "approach_heading": entry.approach_heading, "reason": entry.reason,
                "map_scope": entry.map_scope, "goal_scope": entry.goal_scope,
                "first_seen": entry.first_seen, "last_seen": entry.last_seen,
                "age_sec": max(0.0, now - entry.last_seen),
                "expires_in_sec": max(0.0, entry.expires_at - now),
                "occurrences": entry.occurrences,
            } for entry in sorted(entries, key=lambda item: item.id)],
        }
