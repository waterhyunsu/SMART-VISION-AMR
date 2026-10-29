"""Straight reverse recovery selection never actuates the robot."""

from math import pi

import pytest

from smart_vision_amr.grid import Grid
from smart_vision_amr.reverse_recovery import (
    Breadcrumb, GridGeometry, select_reverse_egress,
)


def grids(*, occupied={(0, 10)}, inflated_limit=5, width=20, height=20):
    raw = Grid(tuple(tuple((x, y) in occupied for x in range(width))
                     for y in range(height)))
    inflated = Grid(tuple(tuple(x <= inflated_limit or (x, y) in occupied
                                for x in range(width)) for y in range(height)))
    return raw, inflated


def trail(*, now=10.0, start_x=0.35, y=1.05, yaw=pi):
    # Oldest first. During forward motion the robot approached the wall at x=0.
    return [Breadcrumb(now - 3 + i, start_x + 0.3 - 0.1 * i, y, yaw)
            for i in range(4)]


def select(history=None, *, raw=None, inflated=None, route=lambda cell: True,
           now=10.0, geometry=None, robot_radius=0.22, **kwargs):
    if raw is None or inflated is None:
        raw, inflated = grids()
    return select_reverse_egress(
        history if history is not None else trail(), raw, inflated,
        geometry if geometry is not None else GridGeometry(0.1, 0.0, 0.0),
        route, now=now, robot_radius=robot_radius, **kwargs)


def test_selects_nearest_outside_inflation_with_reachable_retained_goal():
    checked = []
    candidate = select(route=lambda cell: checked.append(cell) or cell == (6, 10))
    assert candidate is not None
    assert candidate.distance_m == pytest.approx(0.30)
    assert (candidate.target_x, candidate.target_y) == pytest.approx((0.65, 1.05))
    assert candidate.target_cell == (6, 10)
    assert candidate.breadcrumb_count == 4
    assert candidate.start_clearance_m == pytest.approx(0.25)
    assert candidate.target_clearance_m == pytest.approx(0.55)
    assert checked == [(6, 10)]


def test_rejects_insufficient_physical_margin_and_unknown_cells():
    # Raw obstacle-cell edge ends at 0.10 m; 0.31 m leaves only 0.21 m.
    assert select(trail(start_x=0.31)) is None
    raw, inflated = grids(occupied={(4, 10)})
    assert select(raw=raw, inflated=inflated) is None


def test_rejects_stale_or_discontinuous_history():
    assert select(trail(now=8.0), now=10.0) is None
    history = trail()
    history[2] = Breadcrumb(4.0, history[2].x, history[2].y, history[2].yaw)
    assert select(history) is None
    history = [Breadcrumb(9.0, 0.65, 1.05, pi),
               Breadcrumb(10.0, 0.35, 1.05, pi)]
    assert select(history) is None  # 0.30 m odometry jump


def test_rejects_curved_or_reversed_breadcrumbs():
    history = trail()
    history[1] = Breadcrumb(history[1].stamp, history[1].x, 1.10, pi)
    assert select(history) is None
    history = trail()
    history[0] = Breadcrumb(history[0].stamp, history[0].x,
                            history[0].y, pi + 0.30)
    assert select(history) is None
    history = trail()
    history[0] = Breadcrumb(history[0].stamp, 0.25, history[0].y, pi)
    assert select(history) is None


def test_rejects_raw_obstacle_crossing_even_with_clear_endpoints():
    raw, inflated = grids(occupied={(5, 10)}, inflated_limit=5)
    history = [Breadcrumb(9.0, 0.65, 1.05, pi),
               Breadcrumb(10.0, 0.45, 1.05, pi)]
    assert select(history, raw=raw, inflated=inflated, robot_radius=0.01) is None


def test_rejects_missing_astar_tail_or_no_exit_within_distance_bound():
    assert select(route=lambda cell: False) is None
    raw, inflated = grids(inflated_limit=8)
    assert select(raw=raw, inflated=inflated) is None
    assert select(max_distance=0.20) is None


def test_rotated_negative_origin_and_negative_world_coordinates():
    geometry = GridGeometry(0.1, -2.0, -3.0, pi / 2.0)
    history = [Breadcrumb(p.stamp, -3.05, -3.0 + p.x, -pi / 2.0)
               for p in trail()]
    candidate = select(history, geometry=geometry)
    assert candidate is not None
    assert candidate.target_cell == (6, 10)
    assert candidate.target_y == pytest.approx(-2.35)


def test_callback_exception_fails_closed():
    def broken_route(_cell):
        raise RuntimeError("planner unavailable")

    assert select(route=broken_route) is None
