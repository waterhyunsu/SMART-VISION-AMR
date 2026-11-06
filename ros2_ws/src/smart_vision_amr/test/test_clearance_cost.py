"""Wider-route costs do not alter obstacle occupancy or map data."""

import pytest

from smart_vision_amr.astar import AStarPlanner
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


def test_center_preference_is_bounded_and_keeps_a_narrow_corridor_open():
    raw = Grid.from_ascii("\n".join(["###########"] + ["#.........#"] * 5
                                    + ["###########"]))
    costs = clearance_penalties(raw, set(), .05, .45, 1.5)
    assert max(costs.values()) <= 1.5
    assert costs[(5, 1)] > costs[(5, 3)]
    assert costs[(5, 1)] == costs[(5, 5)]
    # Every free cell is inside the preferred-clearance band, but the preference
    # must never turn a physically feasible corridor into a hard no-path result.
    assert len(costs) == 45
    route = AStarPlanner().plan(raw, (1, 1), (9, 1), cell_penalty=costs)
    assert route and route[0] == (1, 1) and route[-1] == (9, 1)
    assert any(y > 1 for _, y in route[1:-1])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_nonfinite_clearance_parameters_are_rejected(value, index):
    settings = [.05, .45, 1.5]
    settings[index] = value
    with pytest.raises(ValueError, match="finite"):
        clearance_penalties(Grid.from_ascii("..."), set(), *settings)
