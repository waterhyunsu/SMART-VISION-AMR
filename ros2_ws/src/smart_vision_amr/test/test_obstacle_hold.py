"""A few failed executions may hold; only positive free rays can release it."""
import json

import pytest

from smart_vision_amr.obstacle_hold import ObstacleHold


def held(**settings):
    hold = ObstacleHold(**settings)
    hold.ensure_goal_scope('map:goal-a')
    for i in range(hold.retry_limit):
        hold.register_failure({(5, 6)}, now=float(i))
    assert hold.holding
    return hold


def test_three_attempt_budget_is_bounded_and_third_failure_holds():
    hold = ObstacleHold()
    assert not hold.register_failure({(5, 6)}, now=1.)
    assert not hold.register_failure({(5, 7)}, now=2.)
    assert hold.register_failure({(5, 8)}, now=3.)
    assert hold.blocked_cells == {(5, 6), (5, 7), (5, 8)}
    for now in (4., 5., 6.):
        assert hold.register_failure({(5, 8)}, now=now)
    assert hold.attempts == 3


def test_all_actual_obstacle_cells_must_be_positive_free_for_two_seconds():
    hold = held()
    assert not hold.observe_free({(5, 6)}, set(), fresh=True, now=3.)
    assert not hold.observe_free({(5, 6)}, set(), fresh=True, now=4.9)
    assert hold.observe_free({(5, 6)}, set(), fresh=True, now=5.)
    assert not hold.holding and hold.attempts == 0
    assert not hold.observe_free({(5, 6)}, set(), fresh=True, now=6.)
    assert hold.resume_count == 1


def test_missing_hits_and_ttl_passage_are_never_free_space_evidence():
    hold = held()
    assert not hold.observe_free(set(), set(), fresh=True, now=200.)
    assert not hold.observe_free(set(), set(), fresh=True, now=10_000.)
    assert hold.holding
    assert hold.clear_since is None


def test_certified_alternative_releases_retry_latch_without_claiming_removal():
    hold = held()
    scope = hold.goal_scope
    assert hold.release_for_alternative(now=4.)
    assert not hold.holding and hold.attempts == 0
    assert hold.goal_scope == scope and hold.resume_count == 1
    assert not hold.release_for_alternative(now=5.)


def test_occupied_return_wins_and_restarts_clear_confirmation():
    hold = held()
    hold.observe_free({(5, 6)}, set(), fresh=True, now=3.)
    assert not hold.observe_free({(5, 6)}, {(5, 6)}, fresh=True, now=4.9)
    assert hold.clear_since is None
    assert not hold.observe_free({(5, 6)}, set(), fresh=True, now=5.)
    assert hold.observe_free({(5, 6)}, set(), fresh=True, now=7.)


@pytest.mark.parametrize('fault', ['stale', 'unknown', 'partial_free'])
def test_sensor_fault_or_unknown_rays_cannot_finish_an_old_clear_interval(fault):
    hold = held()
    hold.blocked_cells.add((5, 7))
    hold.observe_free({(5, 6), (5, 7)}, set(), fresh=True, now=3.)
    free = {(5, 6)} if fault == 'partial_free' else set() if fault == 'unknown' else {(5, 6), (5, 7)}
    assert not hold.observe_free(free, set(), fresh=fault != 'stale', now=5.)
    assert hold.clear_since is None
    assert not hold.observe_free({(5, 6), (5, 7)}, set(), fresh=True, now=6.)
    assert hold.observe_free({(5, 6), (5, 7)}, set(), fresh=True, now=8.)


def test_empty_physical_evidence_can_hold_but_cannot_autonomously_resume():
    hold = ObstacleHold(retry_limit=1)
    assert hold.register_failure(set(), now=1.)
    assert not hold.observe_free({(1, 1)}, set(), fresh=True, now=2.)
    assert not hold.observe_free({(1, 1)}, set(), fresh=True, now=4.)
    assert not hold.snapshot(now=4.)['removal_verifiable']
    assert hold.holding


def test_manual_stop_reset_cancels_automatic_resume():
    hold = held()
    hold.observe_free({(5, 6)}, set(), fresh=True, now=3.)
    hold.reset()
    assert not hold.observe_free({(5, 6)}, set(), fresh=True, now=10.)
    assert not hold.holding and hold.attempts == 0
    assert not hold.blocked_cells


def test_same_goal_resend_preserves_hold_but_a_new_target_has_its_own_budget():
    hold = held()
    hold.ensure_goal_scope('map:goal-a')
    assert hold.holding and hold.attempts == 3
    hold.ensure_goal_scope('map:goal-b')
    assert not hold.holding and hold.attempts == 0
    assert hold.goal_scope == 'map:goal-b'


def test_backwards_time_cannot_approve_removal():
    hold = held()
    hold.observe_free({(5, 6)}, set(), fresh=True, now=10.)
    assert not hold.observe_free({(5, 6)}, set(), fresh=True, now=5.)
    assert hold.clear_since is None
    assert not hold.observe_free({(5, 6)}, set(), fresh=True, now=6.)
    assert hold.observe_free({(5, 6)}, set(), fresh=True, now=8.)


def test_snapshot_is_json_and_sources_are_copied():
    hold = ObstacleHold(retry_limit=1)
    source = {(5, 6)}
    hold.register_failure(source, now=1.)
    source.clear()
    snapshot = hold.snapshot(now=2.)
    json.dumps(snapshot, allow_nan=False)
    assert snapshot['tracked_cells'] == [[5, 6]]
    assert snapshot['state'] == 'OBSTACLE_HOLD'


@pytest.mark.parametrize('settings', [{'retry_limit': 0}, {'retry_limit': True},
                                      {'retry_limit': 1.5}, {'clear_duration': 0},
                                      {'clear_duration': float('nan')}, {'clear_duration': float('inf')}])
def test_invalid_limits_cannot_disable_the_hold_silently(settings):
    with pytest.raises(ValueError):
        ObstacleHold(**settings)


@pytest.mark.parametrize('now', [-1., True, float('nan'), float('inf')])
def test_invalid_time_cannot_change_retry_state(now):
    hold = ObstacleHold()
    with pytest.raises(ValueError):
        hold.register_failure({(1, 1)}, now=now)
    assert hold.attempts == 0
