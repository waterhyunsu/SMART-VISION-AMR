"""Persisted obstacle evidence is confirmed and cleared only after count and time."""

from smart_vision_amr.obstacle_memory import ObstacleMemory


def memory():
    return ObstacleMemory(
        5, 5, confirm_observations=5, clear_observations=5,
        confirm_min_duration_sec=2.0, clear_min_duration_sec=2.0,
        min_observation_interval_sec=0.25)


def test_five_observations_before_two_seconds_are_not_enough():
    tracker = memory()
    for stamp in (0.0, 0.5, 1.0, 1.5, 1.75):
        assert not tracker.observe_occupied((2, 2), stamp)
    assert (2, 2) not in tracker.confirmed_cells
    assert tracker.observe_occupied((2, 2), 2.0)


def test_two_seconds_before_five_observations_are_not_enough():
    tracker = memory()
    for stamp in (0.0, 0.7, 1.4, 2.1):
        assert not tracker.observe_occupied((2, 2), stamp)
    assert (2, 2) not in tracker.confirmed_cells
    assert tracker.observe_occupied((2, 2), 2.8)


def test_fast_duplicate_beams_do_not_count_as_new_observations():
    tracker = memory()
    for stamp in (0.0, 0.1, 0.2, 0.25):
        tracker.observe_occupied((2, 2), stamp)
    assert tracker.cells[(2, 2)].occupied_observations == 2


def test_free_observation_resets_unconfirmed_occupied_evidence():
    tracker = memory()
    for stamp in (0.0, 0.5, 1.0, 1.5):
        tracker.observe_occupied((2, 2), stamp)
    tracker.observe_free((2, 2), 2.0)
    assert not tracker.observe_occupied((2, 2), 2.5)
    assert tracker.cells[(2, 2)].occupied_observations == 1


def test_confirmed_cell_requires_five_free_observations_over_two_seconds():
    tracker = memory()
    for stamp in (0.0, 0.5, 1.0, 1.5, 2.0):
        tracker.observe_occupied((2, 2), stamp)
    for stamp in (3.0, 3.5, 4.0, 4.5, 4.75):
        assert not tracker.observe_free((2, 2), stamp)
    assert (2, 2) in tracker.confirmed_cells
    assert tracker.observe_free((2, 2), 5.0)
    assert (2, 2) not in tracker.confirmed_cells
