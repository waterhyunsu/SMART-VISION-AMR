"""Wider-route costs do not alter obstacle occupancy or map data."""

from smart_vision_amr.clearance_cost import clearance_penalties
from smart_vision_amr.grid import Grid


def test_wall_dynamic_return_and_map_edge_add_only_nonnegative_penalties():
    raw = Grid.from_ascii("""
        ..........
        ..........
        .....#....
        ..........
        ..........
        ..........
        ..........
        ..........
        ..........
        ..........
    """)
    before = raw.rows
    costs = clearance_penalties(raw, {(8, 8)}, .1, .45, 1.5)
    assert raw.rows == before
    assert (5, 2) not in costs  # occupied map cell stays occupied
    assert (8, 8) not in costs  # live obstacle is not a candidate route cell
    assert costs[(5, 3)] > costs[(5, 6)]
    assert costs[(1, 7)] < costs[(0, 7)]  # map boundary is unknown
    assert all(value >= 0 for value in costs.values())


def test_cost_can_be_disabled_without_modifying_the_grid():
    raw = Grid.from_ascii("...")
    assert clearance_penalties(raw, set(), .1, .45, 0.0) == {}
    assert clearance_penalties(raw, set(), .1, 0.0, 1.5) == {}
    assert raw.render() == "..."
