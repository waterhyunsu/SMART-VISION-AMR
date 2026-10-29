"""Dynamic-halo egress must clear raw obstacle squares with the robot footprint."""

from smart_vision_amr.astar_planner import AStarPlannerNode
from smart_vision_amr.grid import Grid
from smart_vision_amr.map_utils import inflate_cells


def free_grid(size=25):
    return Grid(tuple(tuple(False for _ in range(size)) for _ in range(size)))


def halo(start, raw, radius=4.4, margin=0.4):
    grid = free_grid()
    blocked = inflate_cells(raw, grid.width, grid.height, 5)
    return AStarPlannerNode._dynamic_escape_halo(
        start, blocked, raw, set(), grid, grid, 5, radius, margin)


def test_start_one_cell_from_scan_obstacle_cannot_escape():
    assert halo((11, 10), {(10, 10)}) == set()


def test_inflated_start_with_clear_footprint_can_leave_halo():
    escaped = halo((15, 14), {(10, 10)})
    assert (15, 14) in escaped
    assert (10, 10) not in escaped


def test_escape_cells_near_raw_obstacle_or_map_edge_are_not_allowed():
    escaped = halo((15, 14), {(10, 10)})
    assert (15, 10) not in escaped  # square-to-center clearance is only 4.5 cells
    assert halo((1, 14), {(5, 10)}) == set()  # the robot overhangs the map edge
