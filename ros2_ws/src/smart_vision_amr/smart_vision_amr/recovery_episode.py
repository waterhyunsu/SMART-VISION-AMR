"""Pure ownership of one physical-blocker recovery cycle.

Evidence and actuators belong to the ROS adapter. Timers, retry counters and
unrelated terminal events cannot grant a phase transition here.
"""
from dataclasses import dataclass
from enum import Enum


class RecoveryPhase(str, Enum):
    LOCAL = "LOCAL"
    STOP_PROOF = "STOP_PROOF"
    REVERSE_VERDICT = "REVERSE_VERDICT"
    REVERSING = "REVERSING"
    GLOBAL = "GLOBAL"
    FOLLOW = "FOLLOW"
    WAIT_OBSTACLE = "WAIT_OBSTACLE"


@dataclass
class RecoveryEpisode:
    episode_id: str
    scope: tuple
    phase: RecoveryPhase
    blockers: frozenset
    observed: frozenset
    remembered: frozenset
    wait_reason: str = ""
    path_stamp_ns: int | None = None
    positive_free_consumed: bool = False
    relevant_remembered: frozenset = frozenset()


class RecoveryCoordinator:
    def __init__(self):
        self.current: RecoveryEpisode | None = None

    def reset(self):
        self.current = None

    def begin(self, episode_id, scope, blockers, observed, remembered):
        if self.current is not None and self.current.phase is not RecoveryPhase.FOLLOW:
            return False
        if not isinstance(episode_id, str) or not 1 <= len(episode_id) <= 128 or not blockers:
            return False
        self.current = RecoveryEpisode(episode_id, scope, RecoveryPhase.LOCAL,
            frozenset(blockers), frozenset(observed), frozenset(remembered))
        return True

    def _move(self, source, destination):
        if self.current is None or self.current.phase is not source:
            return False
        self.current.phase = destination
        self.current.wait_reason = ""
        return True

    def local_result(self, success):
        return self._move(RecoveryPhase.LOCAL,
                          RecoveryPhase.FOLLOW if success else RecoveryPhase.STOP_PROOF)

    def stop_proved(self):
        return self._move(RecoveryPhase.STOP_PROOF, RecoveryPhase.REVERSE_VERDICT)

    def reverse_verdict(self, accepted):
        return self._move(RecoveryPhase.REVERSE_VERDICT,
                          RecoveryPhase.REVERSING if accepted else RecoveryPhase.GLOBAL)

    def reverse_terminal(self, episode_id):
        if self.current is None or episode_id != self.current.episode_id:
            return False
        return self._move(RecoveryPhase.REVERSING, RecoveryPhase.GLOBAL)

    def global_result(self, success, *, path_stamp_ns=None, observed=(), remembered=(),
                      relevant_remembered=()):
        if not self._move(RecoveryPhase.GLOBAL,
                          RecoveryPhase.FOLLOW if success else RecoveryPhase.WAIT_OBSTACLE):
            return False
        self.current.path_stamp_ns = path_stamp_ns if success else None
        if not success:
            self.current.observed = frozenset(observed)
            self.current.remembered = frozenset(remembered)
            self.current.relevant_remembered = frozenset(relevant_remembered)
            self.current.wait_reason = "No measured-safe route after local and reverse checks"
        return True

    def may_plan_global(self):
        return self.current is None or self.current.phase in (RecoveryPhase.GLOBAL, RecoveryPhase.FOLLOW)

    def may_accept_success(self, path_stamp_ns):
        return (self.current is None or
                self.current.phase is RecoveryPhase.FOLLOW
                and type(path_stamp_ns) is int and path_stamp_ns == self.current.path_stamp_ns)

    def wait_wake_allowed(self, *, positive_free=False, observed=(), remembered=(), moving_obstacle_confirmed=False):
        episode = self.current
        if episode is None or episode.phase is not RecoveryPhase.WAIT_OBSTACLE:
            return False
        # Missing live hits and repeated timestamps are not removal evidence.
        memory_removed = ((episode.remembered-set(remembered))
                          & (episode.blockers | episode.relevant_remembered))
        moved = moving_obstacle_confirmed and frozenset(observed) != episode.observed
        return bool((positive_free and not episode.positive_free_consumed) or memory_removed or moved)

    def resume_wait(self, episode_id, *, observed, remembered,
                    positive_free=False, moving_obstacle_confirmed=False):
        if not self.wait_wake_allowed(positive_free=positive_free, observed=observed, remembered=remembered,
                                     moving_obstacle_confirmed=moving_obstacle_confirmed):
            return False
        previous = self.current
        self.reset()
        opened = self.begin(episode_id, previous.scope, previous.blockers, observed, remembered)
        if opened:
            self.current.positive_free_consumed = (
                previous.positive_free_consumed or bool(positive_free))
            if moving_obstacle_confirmed and frozenset(observed) != previous.observed:
                self.current.positive_free_consumed = False
        return opened

    def snapshot(self):
        if self.current is None:
            return {}
        return {"recovery_episode_id": self.current.episode_id,
                "recovery_phase": self.current.phase.value,
                "recovery_wait_reason": self.current.wait_reason[:240]}
