"""Directional execution constraints apply to normal A* and exceptional egress."""

import pytest

from smart_vision_amr.astar import AStarPlanner
from smart_vision_amr.grid import Grid


def test_directed_edge_is_hard_while_reverse_remains_available():
    grid = Grid.from_ascii(".....")
    blocked = {((1, 0), (2, 0))}
    planner = AStarPlanner()
    assert planner.plan(grid, (0, 0), (4, 0), blocked_edges=blocked) == []
    assert planner.plan(grid, (4, 0), (0, 0), blocked_edges=blocked)


def test_edge_penalty_can_choose_alternate_without_opening_obstacles():
    grid = Grid.from_ascii(".....\n.....\n.....")
    planner = AStarPlanner()
    edge = (1, 1), (2, 1)
    direct = planner.plan(grid, (0, 1), (4, 1))
    alternate = planner.plan(grid, (0, 1), (4, 1), edge_penalty={edge: 3})
    assert edge in set(zip(direct, direct[1:]))
    assert edge not in set(zip(alternate, alternate[1:]))
    wall = Grid.from_ascii("..#..\n..#..\n..#..")
    assert planner.plan(wall, (0, 1), (4, 1), edge_penalty={edge: 0}) == []


def halo():
    raw = Grid(tuple(tuple(x == 7 and y == 4 for x in range(15))
                     for y in range(15)))
    # Artificial static inflation halo, with raw physical footprint unchanged.
    inflated = Grid(tuple(tuple(x == 7 and 4 <= y <= 10 for x in range(15))
                          for y in range(15)))
    return inflated, raw


def test_start_egress_prefix_honors_blocked_edges_and_preserves_other_exit():
    grid, raw = halo()
    blocked = {((7, 7), (8, 7))}
    route = AStarPlanner().plan_with_start_egress(
        grid, (7, 7), (12, 7), raw_grid=raw, robot_radius_cells=.1,
        margin_cells=0, max_egress_cells=2, blocked_edges=blocked)
    assert route and route[-1] == (12, 7)
    assert not set(zip(route, route[1:])) & blocked
    assert route[1] == (6, 7)


def test_start_egress_fully_blocked_execution_exits_waits_without_relaxing():
    grid, raw = halo()
    start = 7, 7
    blocked = {(start, (start[0] + dx, start[1] + dy))
               for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))}
    assert AStarPlanner().plan_with_start_egress(
        grid, start, (12, 7), raw_grid=raw, robot_radius_cells=.1,
        margin_cells=0, blocked_edges=blocked) == []


def test_start_egress_costs_apply_in_prefix_and_tail():
    grid, raw = halo()
    # Right is normally the cheaper tail, but this high local execution cost
    # must make the prefix leave via the left instead of ignoring the cost.
    right_edge = (7, 7), (8, 7)
    route = AStarPlanner().plan_with_start_egress(
        grid, (7, 7), (12, 7), raw_grid=raw, robot_radius_cells=.1,
        margin_cells=0, max_egress_cells=2, edge_penalty={right_edge: 50})
    assert route and route[1] == (6, 7)


@pytest.mark.parametrize("bad", [-.1, float("nan"), float("inf")])
def test_nonfinite_or_negative_edge_penalty_rejected_even_outside_grid(bad):
    grid = Grid.from_ascii("...")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        AStarPlanner().plan(grid, (0, 0), (2, 0),
                            edge_penalty={((99, 99), (100, 99)): bad})


@pytest.mark.parametrize("edge", [((0, 0), (0, 0)), ((0, 0), (2, 0)),
                                  ((0.0, 0), (1, 0)), ((0,), (1,)), (0, 1)])
def test_invalid_edge_shape_rejected(edge):
    with pytest.raises(ValueError, match="adjacent integer-cell"):
        AStarPlanner().plan(Grid.from_ascii("..."), (0, 0), (2, 0),
                            blocked_edges={edge})
