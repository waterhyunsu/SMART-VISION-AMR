"""Static-inflation egress uses real A* without ROS or live robot motion."""

from math import hypot
from types import SimpleNamespace

import pytest

from smart_vision_amr.astar import AStarPlanner
from smart_vision_amr.map_utils import occupancy_to_grid


def grids(width=45, height=45, occupied=(), unknown=(), inflation=6):
    message = SimpleNamespace(
        info=SimpleNamespace(width=width, height=height),
        data=[0] * (width * height))
    for x, y in occupied:
        message.data[y * width + x] = 100
    for x, y in unknown:
        message.data[y * width + x] = -1
    return (occupancy_to_grid(message, inflation_cells=inflation),
            occupancy_to_grid(message))


def plan(grid, raw, start, goal, **kwargs):
    return AStarPlanner().plan_with_start_egress(
        grid, start, goal, raw_grid=raw, **kwargs)


def test_reported_corner_halo_exits_one_cell_and_keeps_full_inflation_tail():
    # Equivalent local geometry to the saved map's failed (56,129)->(40,42).
    grid, raw = grids(202, 164, occupied={(59, 123)})
    start, goal = (56, 129), (40, 42)
    assert grid.is_static_obstacle(start)
    assert not raw.is_static_obstacle(start)
    assert AStarPlanner().plan(grid, start, goal) == []
    result = plan(grid, raw, start, goal)
    assert result[0] == start
    assert result[1] == (56, 130)
    assert result[-1] == goal
    assert all(not grid.is_static_obstacle(cell) for cell in result[1:])
    assert all(not raw.is_static_obstacle(cell) for cell in result)


@pytest.mark.parametrize("unknown", [False, True])
def test_raw_occupied_and_unknown_start_never_egress(unknown):
    kwargs = {"unknown" if unknown else "occupied": {(20, 20)}}
    grid, raw = grids(**kwargs)
    assert plan(grid, raw, (20, 20), (35, 35)) == []


def test_inflated_goal_cannot_be_relaxed_by_static_or_dynamic_escape():
    grid, raw = grids(occupied={(20, 20)})
    assert plan(grid, raw, (17, 26), (20, 26),
                escape_cells={(20, 26)}, allow_blocked_start=True) == []
    grid, raw = grids()
    goal = (35, 35)
    assert AStarPlanner().plan_with_start_egress(
        grid, (20, 20), goal, {goal}, raw_grid=raw,
        escape_cells={goal}, allow_blocked_start=True) == []


def test_actual_footprint_clearance_rejects_start_too_close_to_raw_cell_area():
    grid, raw = grids(occupied={(23, 20)})
    # Three cells between centers means only 2.5 cells to the occupied square.
    assert plan(grid, raw, (20, 20), (35, 35),
                robot_radius_cells=2.6, margin_cells=0.0) == []


def test_map_boundary_is_a_physical_footprint_limit():
    grid, raw = grids(occupied={(7, 20)})
    assert not raw.is_static_obstacle((2, 20))
    assert plan(grid, raw, (2, 20), (35, 35)) == []


def test_bounded_prefix_cannot_cross_deep_halo_until_budget_is_sufficient():
    grid, raw = grids(occupied={(20, 20)})
    start, goal = (20, 24), (20, 35)
    options = {"robot_radius_cells": 0.5, "margin_cells": 0.0}
    assert plan(grid, raw, start, goal, max_egress_cells=2, **options) == []
    result = plan(grid, raw, start, goal, max_egress_cells=3, **options)
    assert result[:4] == [(20, 24), (20, 25), (20, 26), (20, 27)]
    assert result[-1] == goal


def test_raw_observed_or_confirmed_cells_are_hard_even_in_escape_set():
    grid, raw = grids(occupied={(20, 20)})
    start, goal = (17, 26), (35, 35)
    assert plan(grid, raw, start, goal, raw_blocked={start},
                escape_cells={start}, allow_blocked_start=True) == []
    grid, raw = grids()
    wall = {(22, y) for y in range(grid.height)}
    assert plan(grid, raw, (10, 20), (35, 20), raw_blocked=wall,
                escape_cells=wall, allow_blocked_start=True) == []


def test_raw_obstacle_wall_cannot_be_crossed_by_prefix_or_tail():
    wall = {(22, y) for y in range(45)}
    grid, raw = grids(occupied=wall)
    assert plan(grid, raw, (16, 20), (35, 20)) == []


def test_prefix_is_cardinal_and_never_decreases_physical_clearance():
    obstacle = (20, 20)
    grid, raw = grids(occupied={obstacle})
    result = plan(grid, raw, (17, 26), (35, 35))
    assert result
    exit_index = next(i for i, cell in enumerate(result)
                      if not grid.is_static_obstacle(cell))
    prefix = result[:exit_index + 1]
    distances = [hypot(max(abs(x - obstacle[0]) - 0.5, 0.0),
                       max(abs(y - obstacle[1]) - 0.5, 0.0))
                 for x, y in prefix]
    assert all(b >= a for a, b in zip(distances, distances[1:]))
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
               for a, b in zip(prefix, prefix[1:]))
    assert all(not grid.is_static_obstacle(cell) for cell in result[exit_index:])


def test_normally_free_start_keeps_existing_astar_result():
    grid, raw = grids(occupied={(20, 20)})
    start, goal = (10, 10), (35, 35)
    assert plan(grid, raw, start, goal) == AStarPlanner().plan(grid, start, goal)


def test_no_prefix_budget_keeps_halo_blocked():
    grid, raw = grids(occupied={(20, 20)})
    assert plan(grid, raw, (17, 26), (35, 35), max_egress_cells=0) == []


def test_outside_start_and_mismatched_raw_grid_fail_explicitly():
    grid, raw = grids()
    with pytest.raises(ValueError, match="inside"):
        plan(grid, raw, (-1, 20), (35, 35))
    _, wrong_raw = grids(width=44)
    with pytest.raises(ValueError, match="same dimensions"):
        plan(grid, wrong_raw, (20, 20), (35, 35))


@pytest.mark.parametrize("radius,margin", [(float("nan"), 0), (1, -1), (-1, 1)])
def test_invalid_footprint_settings_cannot_enable_egress(radius, margin):
    grid, raw = grids(occupied={(20, 20)})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        plan(grid, raw, (17, 26), (35, 35),
             robot_radius_cells=radius, margin_cells=margin)
