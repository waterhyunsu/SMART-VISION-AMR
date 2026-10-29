"""Optional clearance costs must change preferences, never collision rules."""

from math import hypot

import pytest

from smart_vision_amr.astar import AStarPlanner
from smart_vision_amr.grid import Grid


def length(path):
    return sum(hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(path, path[1:]))


def test_penalty_prefers_longer_clear_route_without_changing_default():
    grid = Grid.from_ascii("\n".join(["......."] * 5))
    start, goal = (0, 2), (6, 2)
    costly_cells = {(x, 2): 3.0 for x in range(1, 6)}
    planner = AStarPlanner()

    direct = planner.plan(grid, start, goal)
    weighted = planner.plan(grid, start, goal, cell_penalty=costly_cells)

    assert direct == [(x, 2) for x in range(7)]
    assert planner.plan(grid, start, goal, cell_penalty={}) == direct
    assert weighted[0] == start and weighted[-1] == goal
    assert not set(weighted) & costly_cells.keys()
    assert length(weighted) > length(direct)


def test_penalty_is_added_on_entry_in_cardinal_step_units():
    grid = Grid.from_ascii("\n".join(["....."] * 3))
    planner = AStarPlanner()
    # The straight route costs four steps. A detour costs about 4.83 steps.
    # A half-step penalty on one middle cell is cheaper than a detour, while
    # a full-step penalty makes the detour preferable.
    mildly_weighted = planner.plan(grid, (0, 1), (4, 1),
                                   cell_penalty={(2, 1): 0.5})
    strongly_weighted = planner.plan(grid, (0, 1), (4, 1),
                                     cell_penalty={(2, 1): 1.0})
    assert mildly_weighted == [(x, 1) for x in range(5)]
    assert (2, 1) not in strongly_weighted


def test_penalty_never_opens_static_or_dynamic_obstacles():
    grid = Grid.from_ascii("""
        .#.
        #..
        ...
    """)
    planner = AStarPlanner()
    assert planner.plan(grid, (0, 0), (2, 2), cell_penalty={(1, 0): 0.0}) == []
    open_grid = Grid.from_ascii("\n".join(["..."] * 3))
    assert planner.plan(open_grid, (0, 0), (2, 2), {(2, 2)},
                        cell_penalty={(2, 2): 0.0}) == []


def test_egress_tail_accepts_same_optional_penalties():
    grid = Grid.from_ascii("\n".join(["......."] * 5))
    planner = AStarPlanner()
    penalty = {(x, 2): 3.0 for x in range(1, 6)}
    direct = planner.plan(grid, (0, 2), (6, 2), cell_penalty=penalty)
    via_egress = planner.plan_with_start_egress(
        grid, (0, 2), (6, 2), raw_grid=grid, cell_penalty=penalty)
    assert via_egress == direct


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_penalty_is_rejected_even_when_its_cell_is_unreachable(bad):
    grid = Grid.from_ascii("\n".join(["..."] * 3))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        AStarPlanner().plan(grid, (0, 0), (2, 2), cell_penalty={(99, 99): bad})
