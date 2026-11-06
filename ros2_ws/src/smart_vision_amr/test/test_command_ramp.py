"""Actual publication ramps; safety stops and status reads never delay motion control."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from smart_vision_amr.command_ramp import CommandRamp
from smart_vision_amr.motion_mux import MotionMuxCore, MotionMuxNode


def ramp():
    result = CommandRamp(enabled=True)
    result.apply(0., 0., now=10., owner='navigation')
    return result


def test_forward_start_takes_at_least_one_second_to_reach_straight_speed():
    limiter = ramp()
    velocities = [limiter.apply(.22, 0., now=10.+index*.05, owner='navigation')[0]
                  for index in range(1, 24)]
    assert velocities[0] == pytest.approx(.01)
    assert velocities[9] == pytest.approx(.10)
    assert velocities[19] == pytest.approx(.20)
    assert velocities[21] == pytest.approx(.22)
    assert max(b-a for a, b in zip([0.]+velocities, velocities)) <= .010000001


def test_turn_start_scales_whole_twist_without_widening_evaluated_arc():
    limiter = ramp()
    for index in range(1, 25):
        linear, angular = limiter.apply(.10, .35, now=10.+index*.05, owner='navigation')
        assert angular/linear == pytest.approx(.35/.10)
        assert 0. < linear <= .10
        assert 0. < angular <= .35


def test_reverse_transition_emits_zero_then_ramps_new_direction():
    limiter = ramp()
    for index in range(1, 24):
        limiter.apply(.22, 0., now=10.+index*.05, owner='navigation')
    assert limiter.apply(-.10, 0., now=11.20, owner='navigation') == (0., 0.)
    assert limiter.reason == 'direction_changed'
    assert limiter.apply(-.10, 0., now=11.25, owner='navigation')[0] == pytest.approx(-.01)
    for index in range(1, 11):
        velocity = limiter.apply(-.10, 0., now=11.25+index*.05, owner='navigation')
    assert velocity == pytest.approx((-.10, 0.))


@pytest.mark.parametrize('linear', [0., .10])
def test_yaw_direction_transition_also_crosses_a_published_zero(linear):
    limiter = ramp()
    for index in range(1, 24):
        limiter.apply(linear, -.5, now=10.+index*.05, owner='navigation')
    assert limiter.apply(linear, .5, now=11.20, owner='navigation') == (0., 0.)
    assert limiter.reason == 'yaw_direction_changed'
    forward, angular = limiter.apply(linear, .5, now=11.25, owner='navigation')
    assert angular == pytest.approx(.04)
    if linear:
        assert forward/angular == pytest.approx(linear/.5)


@pytest.mark.parametrize('correction', [.01, .035])
def test_alternating_small_forward_heading_corrections_do_not_trigger_false_no_progress(correction):
    limiter = ramp()
    distance, previous_yaw = 0., 0.
    for index in range(1, 121):
        target_yaw = correction if index % 2 else -correction
        linear, yaw = limiter.apply(.22, target_yaw,
                                    now=10.+index*.05, owner='navigation')
        distance += linear*.05
        assert linear > 0., (index, limiter.snapshot())
        assert abs(yaw-previous_yaw) <= .80*.05+1e-9
        previous_yaw = yaw
    assert distance > .8
    assert linear == pytest.approx(.22)


def test_tiny_heading_reversals_still_stop_an_in_place_turn_before_opposite_spin():
    limiter = ramp()
    limiter.apply(0., .01, now=10.05, owner='navigation')
    assert limiter.apply(0., -.01, now=10.1, owner='navigation') == (0., 0.)


def test_stop_and_lower_speed_are_immediate_not_slewed():
    limiter = ramp()
    for index in range(1, 24):
        limiter.apply(.22, 0., now=10.+index*.05, owner='navigation')
    assert limiter.apply(.04, 0., now=11.16, owner='navigation') == (.04, 0.)
    assert limiter.apply(0., 0., now=11.16, owner='navigation') == (0., 0.)
    assert limiter.apply(.22, 0., now=11.21, owner='navigation')[0] == pytest.approx(.01)


@pytest.mark.parametrize('next_time', [10.6, 9.9])
def test_scheduler_gap_or_backwards_clock_restarts_from_zero(next_time):
    limiter = ramp()
    assert limiter.apply(.22, 0., now=next_time, owner='navigation') == (0., 0.)
    assert limiter.apply(.22, 0., now=next_time+.05, owner='navigation')[0] == pytest.approx(.01)


def test_one_hundred_to_two_hundred_ms_jitter_neither_resets_nor_accumulates_acceleration():
    limiter = ramp()
    assert limiter.apply(.22, 0., now=10.2, owner='navigation')[0] == pytest.approx(.02)
    assert limiter.apply(.22, 0., now=10.3, owner='navigation')[0] == pytest.approx(.04)
    assert limiter.apply(.22, 0., now=10.5, owner='navigation')[0] == pytest.approx(.06)


def test_owner_change_cannot_reuse_previous_command_acceleration_budget():
    limiter = ramp()
    limiter.apply(.22, 0., now=10.1, owner='navigation')
    assert limiter.apply(.22, 0., now=10.15, owner='joystick') == (0., 0.)
    assert limiter.apply(.22, 0., now=10.2, owner='joystick')[0] == pytest.approx(.01)


def test_status_reads_do_not_consume_acceleration_budget():
    limiter = ramp()
    expected = limiter.apply(.22, 0., now=10.05, owner='navigation')
    for _ in range(50):
        assert limiter.snapshot()['output_linear'] == expected[0]
    assert limiter.apply(.22, 0., now=10.05, owner='navigation') == expected


@pytest.mark.parametrize('settings', [dict(linear_acceleration=0.),
    dict(angular_acceleration=-1.), dict(max_publish_gap=float('nan'))])
def test_invalid_ramp_limits_fail_before_motion(settings):
    with pytest.raises(ValueError):
        CommandRamp(**settings)


class MuxHarness(MotionMuxNode):
    def __init__(self, clock):
        self.core = MotionMuxCore(clock=clock, max_linear_velocity=.22)
        self.command_ramp = CommandRamp(enabled=True)
        self.last_published_output = (0., 0.)
        self.output_available = True
        self.output_conflict = ''
        self.output_publishers = 1
        self.output_pub = Mock()
        self.status_pub = Mock()
        self.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            nanoseconds=int(clock()*1e9)))


def test_mux_status_reports_actual_published_ramp_and_estop_bypasses_it(monkeypatch):
    # The upstream full suite provides ROS message stand-ins through other tests.
    # Keep this extracted test independently runnable without importing ROS.
    import smart_vision_amr.motion_mux as mux
    monkeypatch.setattr(mux, 'Twist', lambda: SimpleNamespace(
        linear=SimpleNamespace(x=0., y=0., z=0.),
        angular=SimpleNamespace(x=0., y=0., z=0.)), raising=False)
    monkeypatch.setattr(mux, 'String', lambda **kwargs: SimpleNamespace(**kwargs), raising=False)
    now = [10.]
    node = MuxHarness(lambda: now[0])
    with patch('smart_vision_amr.motion_mux.time.monotonic', side_effect=lambda: now[0]):
        node.publish_output(refresh_ownership=False)
        node.core.receive_auto(.22, 0.)
        now[0] += .05
        node.publish_output(refresh_ownership=False)
        actual = node.output_pub.publish.call_args.args[0]
        assert actual.linear.x == pytest.approx(.01)
        before = node.command_ramp.snapshot()
        for _ in range(10):
            node.publish_status(force=True, refresh_ownership=False)
            status = json.loads(node.status_pub.publish.call_args.args[0].data)
            assert status['output_linear'] == actual.linear.x
            assert status['requested_linear'] == .22
            assert not status['stop_acknowledged']
        assert node.command_ramp.snapshot() == before
        node.core.receive_control({'source_id': 'operator', 'seq': 1, 'mode': 'estop',
                                   'sent_at_ns': int(now[0]*1e9)}, int(now[0]*1e9))
        node.publish_output(refresh_ownership=False)
        stopped = node.output_pub.publish.call_args.args[0]
        assert (stopped.linear.x, stopped.angular.z) == (0., 0.)
        node.publish_status(force=True, refresh_ownership=False)
        status = json.loads(node.status_pub.publish.call_args.args[0].data)
        assert status['output_linear'] == 0.
        assert status['stop_acknowledged']
