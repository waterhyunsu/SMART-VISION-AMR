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


def parallel_wall_scene(*, rear_hit=False):
    occupied = {(x, 0) for x in range(30)} | {(16, 3)}
    if rear_hit:
        occupied.add((8, 3))
    raw = Grid(tuple(tuple((x, y) in occupied for x in range(30))
                     for y in range(20)))
    inflated = Grid(tuple(tuple(y <= 2 or (12 <= x <= 20 and y <= 7)
                                for x in range(30)) for y in range(20)))
    history = [Breadcrumb(7.+i, .95+.10*i, .38, 0.) for i in range(4)]
    return raw, inflated, history


def test_fresh_blocker_focused_retreat_works_parallel_to_wall_with_full_footprint():
    raw, inflated, history = parallel_wall_scene()
    # Distance to the parallel wall is constant; the original all-obstacle
    # gain rule cannot create space even though the obstacle is farther away.
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24) is None
    candidate = select(history, raw=raw, inflated=inflated, robot_radius=.24,
                       blocking_cells={(16, 3)})
    assert candidate is not None
    assert candidate.distance_m == pytest.approx(.10)
    assert candidate.target_x == pytest.approx(1.15)
    assert candidate.start_clearance_m == pytest.approx(.28)
    assert candidate.target_clearance_m == pytest.approx(.28)


def static_parallel_wall_scene(*, y=.33):
    """A side wall stays .28 m away; a diagonal static obstacle traps the start."""
    occupied = {(x, 0) for x in range(80)} | {(33, 10)}
    raw = Grid(tuple(tuple((x, row) in occupied for x in range(80))
                     for row in range(40)))
    # The side-wall inflation ends at row 5. The diagonal obstacle's five-cell
    # square inflation covers the clear start (28, 6), but not the rear exit.
    inflated = Grid(tuple(tuple(row <= 5 or (28 <= x <= 38 and 5 <= row <= 15)
                          for x in range(80)) for row in range(40)))
    history = [Breadcrumb(7.+index, 1.125+.1*index, y, 0.)
               for index in range(4)]
    return raw, inflated, history


def test_static_inflated_start_may_retreat_parallel_to_wall_with_full_clearance():
    raw, inflated, history = static_parallel_wall_scene()
    geometry = GridGeometry(.05, 0., 0.)
    # Ordinary reverse makes no clearance gain because the side wall remains
    # the nearest obstacle. The explicitly authorized static-inflation case
    # exits its band on the old straight trail with an intact .28 m envelope.
    assert select(history, raw=raw, inflated=inflated, geometry=geometry,
                  robot_radius=.24, clearance_margin=.03) is None
    candidate = select(history, raw=raw, inflated=inflated, geometry=geometry,
                       robot_radius=.24, clearance_margin=.03,
                       allow_parallel_static_egress=True,
                       route=lambda cell: cell == (26, 6))
    assert candidate is not None
    assert candidate.distance_m == pytest.approx(.10)
    assert candidate.target_cell == (26, 6)
    assert candidate.start_clearance_m == pytest.approx(.28)
    assert candidate.target_clearance_m == pytest.approx(.28)


def test_static_parallel_reverse_still_rejects_22_5_cm_gap_and_unreachable_exit():
    raw, inflated, history = static_parallel_wall_scene(y=.275)
    geometry = GridGeometry(.05, 0., 0.)
    # Wall cell ends at y=.05; .275-.05=.225 m, below the unchanged
    # .24 m physical radius + .03 m margins + .005 m sample reserve.
    assert select(history, raw=raw, inflated=inflated, geometry=geometry,
                  robot_radius=.24, clearance_margin=.03,
                  allow_parallel_static_egress=True) is None
    raw, inflated, history = static_parallel_wall_scene()
    assert select(history, raw=raw, inflated=inflated, geometry=geometry,
                  robot_radius=.24, clearance_margin=.03,
                  allow_parallel_static_egress=True,
                  route=lambda _: False) is None


def test_focused_blocker_never_bypasses_rear_raw_hit_or_retained_goal_route():
    raw, inflated, history = parallel_wall_scene(rear_hit=True)
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(16, 3)}) is None
    raw, inflated, history = parallel_wall_scene()
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(16, 3)}, route=lambda _: False) is None


@pytest.mark.parametrize('focus', [{(15, 3)}, {(99, 3)}, {(16.0, 3)}, {'bad'}])
def test_focus_must_be_valid_current_raw_obstacle_cells(focus):
    raw, inflated, history = parallel_wall_scene()
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells=focus) is None


def test_focused_retreat_refuses_insufficient_wall_clearance_and_stale_history():
    raw, inflated, history = parallel_wall_scene()
    unsafe = [Breadcrumb(p.stamp, p.x, .35, p.yaw) for p in history]
    assert select(unsafe, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(16, 3)}) is None
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(16, 3)}, now=12.) is None


def test_commanded_straight_axis_is_checked_separately_from_small_lateral_track_drift():
    from smart_vision_amr.reverse_recovery import _raw_clearance
    raw, inflated = grids(occupied={(0, 10), (5, 7)})
    history = [Breadcrumb(7.+i, .67-.10*i, 1.07 if i < 3 else 1.05, pi)
               for i in range(4)]
    geometry = GridGeometry(.1, 0., 0.)
    occupied = ((0, 10), (5, 7))
    # The history polyline clears every raw cell with >=.265 m. The actual
    # BackUp chassis-axis ray y=1.05 would approach the side cell within .25 m.
    for a, b in zip(history, history[1:]):
        for i in range(21):
            u = i/20
            assert _raw_clearance(raw, geometry, occupied,
                a.x+u*(b.x-a.x), a.y+u*(b.y-a.y)) >= .265
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(0, 10)}) is None


def test_actual_pose_offset_is_not_relabelled_as_a_safe_old_breadcrumb():
    raw, inflated, history = parallel_wall_scene()
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(16, 3)}) is not None
    # The old trail is safe, but the actual chassis is five centimetres
    # closer to the wall. Certifying only the old trail would approve it.
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(16, 3)}, current_pose=(1.25, .33, 0.)) is None
    assert history[-1].y == .38


def test_actual_heading_drives_ray_sweep_and_endpoint_not_breadcrumb_heading():
    raw, _, history = parallel_wall_scene()
    inflated = Grid(tuple(tuple(y <= 2 or (10 <= x <= 20 and y <= 7)
                                for x in range(30)) for y in range(20)))
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(16, 3)}) is not None
    # 0.1 rad is within the adapter's trail continuity tolerance, but the
    # actual .30 m reverse ray turns toward the wall and violates clearance.
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(16, 3)}, current_pose=(1.25, .38, .10)) is None
    assert history[-1].yaw == 0.


@pytest.mark.parametrize('actual', [(1.31, .38, 0.), (1.25, .38, .16),
                                      (float('nan'), .38, 0.)])
def test_actual_pose_still_requires_bounded_continuity_with_verified_trail(actual):
    raw, inflated, history = parallel_wall_scene()
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(16, 3)}, current_pose=actual) is None


def test_handover_envelope_is_additional_to_physical_margin():
    raw, inflated, history = parallel_wall_scene()
    history = [Breadcrumb(p.stamp, p.x, .37, p.yaw) for p in history]
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(16, 3)}, clearance_margin=.02) is not None
    assert select(history, raw=raw, inflated=inflated, robot_radius=.24,
                  blocking_cells={(16, 3)}, clearance_margin=.02+.01) is None


def test_bounded_sweep_filter_matches_full_map_clearance_with_rotated_origin():
    from math import cos, sin
    from random import Random
    from smart_vision_amr.reverse_recovery import _raw_clearance, _nearby_sweep_obstacles
    rng = Random(901)
    occupied = tuple(sorted({(rng.randrange(100), rng.randrange(100)) for _ in range(1500)}))
    raw = Grid(tuple(tuple((x, y) in occupied for x in range(100)) for y in range(100)))
    geometry = GridGeometry(.05, -3., -4., .65)
    c, s = cos(.65), sin(.65)
    def world(x, y): return -3.+c*x-s*y, -4.+s*x+c*y
    x, y = world(2.5, 2.5)
    d0 = _raw_clearance(raw, geometry, occupied, x, y)
    nearby = _nearby_sweep_obstacles(geometry, occupied, x, y, d0, .31)
    assert len(nearby) < len(occupied)/5
    for i in range(80):
        angle, distance = rng.uniform(-pi, pi), rng.uniform(0., .31)
        px, py = x+cos(angle)*distance, y+sin(angle)*distance
        assert _raw_clearance(raw, geometry, nearby, px, py) == pytest.approx(
            _raw_clearance(raw, geometry, occupied, px, py), abs=1e-12)
