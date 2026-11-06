"""Bound spatial retries until obstacle removal or a certified alternate route.

Missing hits, expired transient observations, and unknown/stale sensor rays are
not proof of free space. This pure state machine never commands motion or edits
occupancy. A caller may resume the retained goal after ``observe_free`` proves
removal or after independently certifying a genuinely different safe route;
normal localization, planning and safety checks still apply.
"""
from __future__ import annotations

from math import isfinite
from typing import Iterable

from .grid import Cell


def _cells(value: Iterable[Cell]) -> set[Cell]:
    result = set()
    for cell in value:
        if (not isinstance(cell, (list, tuple)) or len(cell) != 2
                or any(not isinstance(v, int) or isinstance(v, bool) for v in cell)):
            raise ValueError('obstacle hold cells must be integer pairs')
        result.add(tuple(cell))
    return result


class ObstacleHold:
    def __init__(self, retry_limit: int = 3, clear_duration: float = 2.0):
        if (not isinstance(retry_limit, int) or isinstance(retry_limit, bool)
                or retry_limit < 1 or isinstance(clear_duration, bool)
                or not isinstance(clear_duration, (int, float))
                or not isfinite(clear_duration) or clear_duration <= 0):
            raise ValueError('obstacle hold retry limit and clear duration must be positive and finite')
        self.retry_limit = retry_limit
        self.clear_duration = float(clear_duration)
        self.resume_count = 0
        self.last_resume_at = None
        self.reset()

    def reset(self, goal_scope: str | None = None):
        """Cancel this retry/hold sequence, including manual-stop auto-resume."""
        if goal_scope is not None and (not isinstance(goal_scope, str) or not goal_scope):
            raise ValueError('goal scope must be a nonempty string or None')
        self.goal_scope = goal_scope
        self.attempts = 0
        self.holding = False
        self.blocked_cells: set[Cell] = set()
        self.clear_since = None
        self.last_observation_at = None
        self._last_time = None

    def ensure_goal_scope(self, goal_scope: str):
        """Changing targets resets the budget; resending the same target does not."""
        if not isinstance(goal_scope, str) or not goal_scope:
            raise ValueError('goal scope must be a nonempty string')
        if goal_scope != self.goal_scope:
            self.reset(goal_scope)

    def _time(self, now: float) -> bool:
        if (isinstance(now, bool) or not isinstance(now, (int, float))
                or not isfinite(now) or now < 0):
            raise ValueError('obstacle hold time must be finite steady-clock seconds')
        advanced = self._last_time is None or now >= self._last_time
        if not advanced:
            # A clock reset cannot turn a partial free interval into approval.
            self.clear_since = None
        self._last_time = float(now)
        return advanced

    def register_failure(self, blocked_cells: Iterable[Cell], *, now: float) -> bool:
        """Count one confirmed failed attempt and retain its physical hit cells.

        The caller deduplicates failure event IDs and excludes sensor/manual
        faults. Already-held reports cannot spend an unbounded retry budget.
        No physical hit cells means removal cannot be verified automatically.
        """
        cells = _cells(blocked_cells)
        self._time(now)
        self.blocked_cells.update(cells)
        self.attempts = min(self.retry_limit, self.attempts+1)
        self.holding = self.attempts >= self.retry_limit
        self.clear_since = None
        return self.holding

    def observe_free(self, free_cells: Iterable[Cell], occupied_cells: Iterable[Cell], *,
                     fresh: bool, now: float) -> bool:
        """Return true once after every tracked physical cell is explicitly free.

        All tracked cells must remain positively free for ``clear_duration``.
        A current occupied return wins over a free ray. Any unknown/missing
        tracked cell or unhealthy sensor evidence restarts that interval.
        ``fresh`` must reflect source stamp, receipt age, valid rays and TF, not
        merely an available topic name. The caller must also report interruption
        with ``fresh=False`` when no new scan arrives.
        """
        free, occupied = _cells(free_cells), _cells(occupied_cells)
        advanced = self._time(now)
        self.last_observation_at = float(now)
        if (not self.holding or fresh is not True or not advanced
                or not self.blocked_cells
                or bool(self.blocked_cells & occupied)
                or not self.blocked_cells <= free):
            self.clear_since = None
            return False
        if self.clear_since is None:
            self.clear_since = float(now)
            return False
        if now-self.clear_since < self.clear_duration:
            return False
        self.holding = False
        self.attempts = 0
        self.blocked_cells.clear()
        self.clear_since = None
        self.last_resume_at = float(now)
        self.resume_count += 1
        return True

    def release_for_alternative(self, *, now: float) -> bool:
        """Release a held goal after its caller certifies a different safe route.

        This changes only the retry latch. It neither asserts that the object
        disappeared nor edits the observed, remembered, or saved map layers.
        """
        self._time(now)
        if not self.holding:
            return False
        self.holding = False
        self.attempts = 0
        self.blocked_cells.clear()
        self.clear_since = None
        self.last_resume_at = float(now)
        self.resume_count += 1
        return True

    def snapshot(self, *, now: float) -> dict:
        self._time(now)
        return {
            'state': 'OBSTACLE_HOLD' if self.holding else 'RETRY' if self.attempts else 'CLEAR',
            'holding': self.holding, 'attempts': self.attempts,
            'retry_limit': self.retry_limit, 'clear_duration': self.clear_duration,
            'goal_scope': self.goal_scope,
            'tracked_cells': [[x, y] for x, y in sorted(self.blocked_cells)],
            'tracked_cell_count': len(self.blocked_cells),
            'removal_verifiable': bool(self.blocked_cells),
            'clear_since': self.clear_since,
            'clear_elapsed': (max(0., now-self.clear_since)
                              if self.clear_since is not None else 0.),
            'last_observation_at': self.last_observation_at,
            'last_resume_at': self.last_resume_at, 'resume_count': self.resume_count,
        }
