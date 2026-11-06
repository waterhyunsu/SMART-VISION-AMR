"""Pure phase ownership: time and counters never imply drive permission."""
from smart_vision_amr.recovery_episode import RecoveryCoordinator, RecoveryPhase


def waiting(*, observed=((5,5),), remembered=()):
    coordinator=RecoveryCoordinator()
    assert coordinator.begin('episode-1',('map','goal'),{(5,5)},observed,remembered)
    assert coordinator.local_result(False)
    assert coordinator.stop_proved()
    assert coordinator.reverse_verdict(False)
    assert coordinator.global_result(False,observed=observed,remembered=remembered)
    return coordinator


def test_only_owned_terminal_advances_reverse_once():
    c=RecoveryCoordinator()
    assert c.begin('current',('map','goal'),{(5,5)},{(5,5)},())
    assert not c.may_plan_global()
    assert not c.reverse_terminal('current')
    assert c.local_result(False) and c.stop_proved() and c.reverse_verdict(True)
    assert not c.reverse_terminal('old')
    assert c.current.phase is RecoveryPhase.REVERSING
    assert c.reverse_terminal('current')
    assert not c.reverse_terminal('current')
    assert not c.reverse_verdict(True)
    assert c.may_plan_global()


def test_local_and_global_success_require_bound_path_stamp():
    c=RecoveryCoordinator();c.begin('e',(),{(5,5)},(),())
    assert not c.may_accept_success(10)
    assert c.local_result(True)
    assert not c.may_accept_success(10)
    c.current.path_stamp_ns=10
    assert c.may_accept_success(10) and not c.may_accept_success(9)
    c=waiting();assert c.resume_wait('e2',observed=(),remembered=(),positive_free=True)
    assert c.local_result(False) and c.stop_proved() and c.reverse_verdict(False)
    assert c.global_result(True,path_stamp_ns=20)
    assert c.may_accept_success(20) and not c.may_accept_success(10)


def test_identical_positive_free_cannot_reopen_each_failed_cycle():
    c=waiting()
    assert c.resume_wait('e2',observed=(),remembered=(),positive_free=True)
    assert c.local_result(False) and c.stop_proved() and c.reverse_verdict(False)
    assert c.global_result(False,observed=(),remembered=())
    for index in range(3):
        assert not c.resume_wait(f'e{index+3}',observed=(),remembered=(),positive_free=True)
    assert c.current.episode_id=='e2' and c.current.phase is RecoveryPhase.WAIT_OBSTACLE


def test_live_disappearance_or_republication_cannot_wake_wait():
    c=waiting()
    assert not c.resume_wait('e2',observed=(),remembered=())
    assert not c.resume_wait('e3',observed={(5,5)},remembered=(),moving_obstacle_confirmed=True)
    assert c.resume_wait('e4',observed={(8,5)},remembered=(),moving_obstacle_confirmed=True)


def test_confirmed_memory_deletion_is_consumed_after_one_cycle():
    c=waiting(remembered={(5,5)})
    assert c.resume_wait('e2',observed=(),remembered=())
    assert c.local_result(False) and c.stop_proved() and c.reverse_verdict(False)
    assert c.global_result(False,observed=(),remembered=())
    assert not c.resume_wait('e3',observed=(),remembered=())


def test_new_physical_episode_only_after_follow_or_explicit_reset():
    c=waiting()
    assert not c.begin('new',('map','goal'),{(8,5)},(),())
    c.reset()
    assert c.begin('new',('map','newgoal'),{(8,5)},(),())
    assert c.snapshot()['recovery_phase']=='LOCAL'
