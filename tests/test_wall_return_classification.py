"""Geometric wall-return classification follows the map's actual ray surface."""

from math import cos, pi, sin, sqrt
from types import SimpleNamespace

import pytest

from smart_vision_amr import map_utils
from smart_vision_amr.grid import Grid
from nav_msgs.msg import OccupancyGrid


def make_grid_and_info(*, origin_x=0.0, origin_y=0.0, yaw=0.0):
    info = SimpleNamespace(
        resolution=0.05,
        width=10,
        height=10,
        origin=SimpleNamespace(
            position=SimpleNamespace(x=origin_x, y=origin_y),
            orientation=SimpleNamespace(z=sin(yaw / 2), w=cos(yaw / 2), x=0.0, y=0.0),
        ),
    )
    grid = Grid(tuple(tuple(x == 5 for x in range(info.width))
                      for _ in range(info.height)))
    return info, grid


def ray_distance(info, grid, x, y, angle, maximum=1.0):
    helper = getattr(map_utils, "first_static_obstacle_distance", None)
    assert callable(helper), "map_utils must expose range-aware static ray traversal"
    return helper(info, grid, x, y, angle, maximum)


def test_first_static_wall_distance_uses_cell_entry_not_cell_center():
    info, grid = make_grid_and_info()

    assert ray_distance(info, grid, 0.125, 0.125, 0.0) == pytest.approx(0.125)
    assert ray_distance(info, grid, 0.125, 0.125, pi / 2) is None


def test_first_static_wall_distance_respects_rotated_map_origin():
    info, grid = make_grid_and_info(origin_x=1.0, origin_y=-2.0, yaw=pi / 2)
    # Local map coordinate (0.125, 0.125) rotated into the world frame.
    world_x, world_y = 0.875, -1.875

    assert ray_distance(info, grid, world_x, world_y, pi / 2) == pytest.approx(0.125)


def test_ray_corner_checks_cells_touched_at_the_corner():
    info, _ = make_grid_and_info()
    rows = [[False] * info.width for _ in range(info.height)]
    rows[2][3] = True
    grid = Grid(tuple(tuple(row) for row in rows))

    assert ray_distance(info, grid, 0.125, 0.125, pi / 4) == pytest.approx(.025 * sqrt(2))


def test_unknown_cells_stay_blocked_for_planning_but_are_not_wall_echoes():
    msg = OccupancyGrid()
    msg.info.resolution = .05
    msg.info.width = msg.info.height = 10
    msg.info.origin.orientation.w = 1.0
    msg.data = [0] * 100
    msg.data[2 * 10 + 4] = -1
    planning_grid = map_utils.occupancy_to_grid(msg, 65, 0)
    wall_reference_grid = map_utils.occupancy_to_grid(
        msg, 65, 0, unknown_is_obstacle=False)

    assert ray_distance(msg.info, planning_grid, .125, .125, 0.0) == pytest.approx(.075)
    assert ray_distance(msg.info, wall_reference_grid, .125, .125, 0.0) is None
    assert ray_distance(msg.info, wall_reference_grid, .125, .125, pi, .2) is None
