"""Sensor freshness checks used before feeding the hardware EKF."""

from smart_vision_amr.sensor_stamp_guard import SensorStampGuard


def test_rejects_reordered_old_and_future_samples():
    guard = SensorStampGuard(max_age_sec=1.0, future_tolerance_sec=0.25)
    now = 5_000_000_000
    assert guard.accept(4_900_000_000, now)
    assert not guard.accept(4_800_000_000, now)
    assert not guard.accept(4_900_000_000, now)
    assert not guard.accept(3_000_000_000, now)
    assert not guard.accept(5_500_000_000, now)
    assert guard.accept(5_050_000_000, now)


def test_dropped_sample_does_not_advance_last_stamp():
    guard = SensorStampGuard()
    now = 10_000_000_000
    assert not guard.accept(8_000_000_000, now)
    assert guard.accept(9_500_000_000, now)
