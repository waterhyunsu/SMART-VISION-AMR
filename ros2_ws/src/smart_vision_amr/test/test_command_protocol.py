"""Delayed/replayed automatic commands and localization-limited final output."""

import math

import pytest

from smart_vision_amr.command_stamper import CommandStamperCore
from smart_vision_amr.motion_mux import MotionMuxCore


def test_forwarding_packets_carry_unique_restart_source_and_order():
    first, second = CommandStamperCore(), CommandStamperCore()
    one = first.packet(.1, -.2, 10_000_000_000)
    two = first.packet(.2, .0, 10_100_000_000)
    assert one['source_id'] != second.source_id
    assert one['seq'] < two['seq']
    assert one['sent_at_ns'] == 10_000_000_000
    assert (one['linear'], one['angular']) == (.1, -.2)
    for velocity in (math.nan, math.inf, True, '0.1'):
        with pytest.raises(ValueError):
            first.packet(velocity, 0.0, 20)


def test_transport_delay_sequence_and_source_retirement_are_enforced():
    now = [10.0]
    mux = MotionMuxCore(clock=lambda: now[0])
    bridge = CommandStamperCore('old')
    packet = bridge.packet(.1, -.2, 20_000_000_000)
    assert mux.receive_stamped_auto(packet, 20_050_000_000)
    assert mux.output() == (.1, -.2)
    assert not mux.receive_stamped_auto(packet, 20_100_000_000)
    assert not mux.receive_stamped_auto(bridge.packet(.2, .1, 20_000_000_000),
                                        20_400_000_000)
    assert not mux.receive_stamped_auto(bridge.packet(.2, .1, 20_700_000_000),
                                        20_400_000_000)
    restarted = CommandStamperCore('new')
    new_packet = restarted.packet(.2, .1, 20_100_000_000)
    assert mux.receive_stamped_auto(new_packet, 20_100_000_000)
    assert not mux.receive_stamped_auto(bridge.packet(.25, .9, 20_150_000_000),
                                        20_150_000_000)
    now[0] += .6
    assert mux.output() == (0.0, 0.0)


def test_delayed_unknown_source_cannot_replace_a_new_bridge():
    mux = MotionMuxCore()
    assert mux.receive_stamped_auto(CommandStamperCore('new').packet(.1, 0., 500), 500)
    assert not mux.receive_stamped_auto(CommandStamperCore('old').packet(.2, 0., 499), 501)
    assert mux.status()['auto_source_id'] == 'new'


def test_commands_during_manual_override_cannot_replay_on_release():
    mux = MotionMuxCore()
    bridge = CommandStamperCore('nav')
    manual = {'source_id': 'web', 'seq': 1, 'mode': 'manual',
              'sent_at_ns': 20, 'angular': .1}
    assert mux.receive_control(manual, 20)
    packet = bridge.packet(.2, .0, 21)
    assert not mux.receive_stamped_auto(packet, 21)
    assert mux.receive_control({**manual, 'seq': 2, 'mode': 'auto'}, 20)
    assert not mux.receive_stamped_auto(packet, 21)
    assert mux.output() == (0.0, 0.0)
    assert mux.receive_stamped_auto(bridge.packet(.1, .0, 22), 22)


def test_localization_warning_limits_both_manual_and_autonomous_speed():
    mux = MotionMuxCore()
    assert mux.receive_localization_health({'level': 'warning', 'reasons': ['covariance']})
    mux.receive_auto(.2, .6)
    assert mux.output() == (.1, .3)
    mux.receive_control({'source_id': 'web', 'seq': 1, 'mode': 'manual',
                         'sent_at_ns': 1, 'linear': -.2, 'angular': -.6}, 1)
    assert mux.output() == (-.1, -.3)
    assert mux.status()['localization_speed_factor'] == .5


def test_localization_bootstrap_and_stale_health_fail_closed():
    now = [0.0]
    mux = MotionMuxCore(clock=lambda: now[0], require_localization_health=True)
    mux.receive_auto(.2, .0)
    assert mux.output() == (0.0, 0.0)
    mux.receive_localization_health({'level': 'warning', 'autonomous_allowed': False,
                                      'allow_localization': True})
    assert mux.output() == (0.0, 0.0)
    packet = {'source_id': 'web', 'seq': 1, 'mode': 'manual',
              'operation': 'localization', 'sent_at_ns': 0, 'angular': .8}
    assert mux.receive_control(packet, 0)
    assert mux.output() == (0.0, .25)
    assert mux.receive_control({**packet, 'seq': 2, 'linear': .05}, 0)
    # Marking an operation as localization never permits translation through an
    # unsafe pose. A warning allows normal manual driving at its reduced speed.
    mux.receive_localization_health({'level': 'critical', 'autonomous_allowed': False,
                                      'allow_localization': True})
    assert mux.output() == (0.0, 0.0)
    assert mux.receive_control({**packet, 'seq': 3}, 0)
    assert mux.output() == (0.0, .25)
    now[0] = 1.1
    assert mux.output() == (0.0, 0.0)


def test_service_reset_does_not_retire_existing_ui_and_safety_still_wins():
    mux = MotionMuxCore()
    packet = {'source_id': 'web', 'seq': 1, 'sent_at_ns': 20, 'mode': 'auto'}
    assert mux.receive_control(packet, 20)
    service = {**packet, 'source_id': 'mux-service:test', 'mode': 'estop'}
    assert mux.receive_control(service, 20)
    assert mux.receive_control({**service, 'seq': 2, 'mode': 'reset'}, 20)
    assert mux.receive_control({**packet, 'seq': 2}, 20)
    assert mux.receive_control({**packet, 'source_id': 'localization_safety',
                                'seq': 99, 'mode': 'estop'}, 20)
    assert mux.output() == (0.0, 0.0)
    assert mux.status()['estop_latched']


def test_physical_startup_interlock_prevents_a_running_stamper_resuming_an_old_goal():
    mux = MotionMuxCore(startup_estop=True, require_localization_health=True)
    bridge = CommandStamperCore('still-running-pc')
    mux.receive_localization_health({'level': 'healthy', 'autonomous_allowed': True})
    for now_ns in (100, 101, 102):
        assert not mux.receive_stamped_auto(bridge.packet(.2, .4, now_ns), now_ns)
        assert mux.output() == (0., 0.)
    reset = {'source_id': 'mux-service:operator', 'seq': 1, 'sent_at_ns': 103,
             'mode': 'reset'}
    assert mux.receive_control(reset, 103)
    # Reset itself remains held; explicit resume and a new command are required.
    assert mux.mode == 'hold' and mux.output() == (0., 0.)
    assert not mux.receive_stamped_auto(bridge.packet(.2, .4, 104), 104)
    assert mux.receive_control({**reset, 'source_id': 'web', 'mode': 'auto'}, 103)
    assert mux.receive_stamped_auto(bridge.packet(.1, .2, 105), 105)
    assert mux.output() == (.1, .2)
