"""Bounded local route splicing with independent collision and suffix proofs.

Pure geometry only: no ROS initialization, publishers, services or actuation.
"""
from copy import deepcopy
from math import hypot
from types import SimpleNamespace

import pytest

from smart_vision_amr.grid import Grid
from smart_vision_amr.local_detour import plan_local_detour
from smart_vision_amr.map_utils import inflate_cells, occupancy_to_grid
from smart_vision_amr.path_geometry import route_cells


def scene(*, occupied=(), unknown=(), width=80, height=60, inflation=0):
    message = SimpleNamespace(info=SimpleNamespace(width=width, height=height),
                              data=[0]*(width*height))
    for x, y in occupied:
        message.data[y*width+x] = 100
    for x, y in unknown:
        message.data[y*width+x] = -1
    return occupancy_to_grid(message, inflation_cells=inflation)


def reference():
    # Keep fractional, already-published suffix coordinates: reconstructing
    # the suffix from A* cell centers would silently change the original route.
    return [(x+.5, 25.5 if x < 42 else 25.625) for x in range(10, 66)]


def physical_hits():
    return inflate_cells({(24, 25)}, 80, 60, 4)


def detour(grid=None, points=None, blocked=None, **limits):
    options = {'resolution': .05, 'max_seconds': .5}
    options.update(limits)
    return plan_local_detour(grid or scene(), (10, 25), points or reference(),
                             physical_hits() if blocked is None else blocked, **options)


def assert_clear(result, grid, blocked):
    assert result.points is not None, result
    covered = route_cells(result.points)
    assert not any(grid.is_static_obstacle(cell) for cell in covered)
    assert not covered & blocked


def test_local_prefix_rejoins_forward_and_retains_the_exact_original_suffix():
    points, blocked, grid = reference(), physical_hits(), scene()
    before = deepcopy((points, blocked, grid.rows))
    result = detour(grid, points, blocked)
    assert_clear(result, grid, blocked)
    assert result.prefix_cells[0] == (10, 25)
    assert result.rejoin_index > 14, 'anchor must be beyond the obstructed original segment'
    suffix = points[result.rejoin_index:]
    assert list(result.points[-len(suffix):]) == suffix
    assert result.points[-1] == points[-1]
    assert before == (points, blocked, grid.rows), 'a patch must not edit source map, hits or reference'
    length = sum(hypot(b[0]-a[0], b[1]-a[1]) for a, b in
                 zip(result.prefix_cells, result.prefix_cells[1:]))*.05
    assert length <= 4.


@pytest.mark.parametrize('buffer', [0., .15, .30])
def test_rejoin_buffer_starts_after_the_last_blocked_run_not_the_first(buffer):
    points = [(x+.5, 25.5) for x in range(10, 66)]
    blocked = inflate_cells({(24, 25), (38, 25)}, 80, 60, 3)
    result = detour(points=points, blocked=blocked, rejoin_buffer=buffer)
    assert_clear(result, scene(), blocked)
    last_unsafe_end = max(index+1 for index, (a, b) in enumerate(zip(points, points[1:]))
                          if route_cells([a, b]) & blocked)
    safe_progress = (points[last_unsafe_end][0]-points[0][0])*.05
    assert result.rejoin_progress >= safe_progress+buffer-1e-9
    assert result.points[-len(points[result.rejoin_index:]):] == points[result.rejoin_index:]


def test_forward_buffer_that_exceeds_local_horizon_requires_global_fallback():
    points = [(x+.5, 25.5) for x in range(10, 66)]
    blocked = inflate_cells({(24, 25), (38, 25)}, 80, 60, 3)
    # The first suffix-safe anchor is at1.60 m, inside this1.70 m horizon,
    # but the required .30 m buffer is not. Never silently shrink the buffer.
    result = detour(points=points, blocked=blocked, max_rejoin_distance=1.7,
                    rejoin_buffer=.30)
    assert result.points is None
    assert result.reason == 'no_forward_rejoin'
    assert result.expansions == 0


@pytest.mark.parametrize('kind', ['static', 'unknown'])
def test_new_prefix_and_original_suffix_both_respect_static_and_unknown_cells(kind):
    cells = {(x, 30) for x in range(15, 36)}
    grid = scene(**{'occupied' if kind == 'static' else 'unknown': cells}, inflation=2)
    result = detour(grid)
    assert_clear(result, grid, physical_hits())
    # A new obstacle outside the local splice but on its retained suffix is
    # a global routing problem, never permission to preserve a blocked suffix.
    grid = scene(**{'occupied' if kind == 'static' else 'unknown': {(60, 25)}})
    assert detour(grid).points is None


def test_dynamic_inflation_preserves_footprint_clearance_even_at_the_nearest_passage():
    grid = scene()
    blocked = physical_hits()
    result = detour(grid, blocked=blocked)
    assert_clear(result, grid, blocked)
    for x, y in result.prefix_cells:
        # The raw return is a filled square, not a zero-size point.
        distance = hypot(max(abs(x-24)-.5, 0.), max(abs(y-25)-.5, 0.))*.05
        assert distance >= .22


def test_no_local_escape_from_an_occupied_start_or_occupied_goal():
    assert detour(scene(occupied={(10, 25)})).points is None
    assert detour(scene(unknown={(65, 25)})).points is None
    assert detour(blocked=physical_hits() | {(10, 25)}).points is None


def test_diagonal_contact_cannot_squeeze_between_two_orthogonal_obstacles():
    grid = Grid.from_ascii('..\n..')
    result = plan_local_detour(grid, (0, 0), [(.5, .5), (1.5, 1.5)],
        {(1, 0), (0, 1)}, resolution=1., max_lateral_distance=2., max_seconds=.5)
    assert result.points is None


def test_fully_blocked_corridor_falls_back_instead_of_inventing_an_escape():
    wall = {(24, y) for y in range(60)}
    result = detour(blocked=wall)
    assert result.points is None
    assert result.expansions <= 1500


def test_directed_failure_memory_blocks_a_local_patch_without_banning_the_reverse_edge():
    forward_barrier = {((18, y), (19, y+dy)) for y in range(60)
                       for dy in (-1, 0, 1) if 0 <= y+dy < 60}
    assert detour(blocked_edges=forward_barrier).points is None
    reverse_barrier = {(b, a) for a, b in forward_barrier}
    result = detour(blocked_edges=reverse_barrier)
    assert_clear(result, scene(), physical_hits())
    assert not set(zip(result.prefix_cells, result.prefix_cells[1:])) & reverse_barrier


def test_expansion_budget_is_a_hard_failure_without_a_partial_route():
    result = detour(max_expansions=1)
    assert result.points is None
    assert result.expansions <= 1


def test_time_budget_is_checked_during_search_and_never_returns_a_partial_route():
    calls = []

    def advancing_clock():
        calls.append(len(calls)*.002)
        return calls[-1]

    result = detour(max_seconds=.001, clock=advancing_clock)
    assert result.points is None
    assert result.search_duration_ms >= 1.
    assert len(calls) >= 2


def test_distant_obstruction_cannot_turn_a_local_patch_into_a_whole_goal_replan():
    far = inflate_cells({(62, 25)}, 80, 60, 2)
    assert detour(blocked=far, max_rejoin_distance=1.5).points is None


def test_lateral_budget_rejects_a_bypass_requiring_a_wide_excursion():
    assert detour(max_lateral_distance=.1).points is None


def test_candidate_budget_does_not_search_for_an_unbounded_number_of_anchors():
    result = detour(max_candidates=1, max_expansions=1)
    assert result.points is None
    assert result.expansions <= 1


def test_failed_directed_approach_on_untouched_suffix_requires_global_fallback():
    # The obstacle being bypassed is nearby, but retaining this later forbidden
    # approach would undo the already-recorded execution failure.
    forbidden = {((60, 25), (61, 25))}
    assert detour(blocked_edges=forbidden).points is None


@pytest.mark.parametrize('bad_point', [(1e12, 25.5), (-1., 25.5), (80., 25.5), (float('nan'), 1.)])
def test_invalid_or_outside_reference_is_rejected_before_supercover_allocation(monkeypatch, bad_point):
    def never_project_unbounded_segment(*_args, **_kwargs):
        raise AssertionError('invalid reference reached expensive collision geometry')

    monkeypatch.setattr('smart_vision_amr.local_detour.segment_cells', never_project_unbounded_segment)
    result = detour(points=[(10.5, 25.5), bad_point])
    assert result.points is None
    assert result.reason == 'invalid_reference'


def test_deadline_applies_while_reading_a_dense_reference_before_search():
    calls = []

    def limited_clock():
        calls.append(len(calls)*.0001)
        return calls[-1]

    dense = [(10.5+index*.002, 25.5) for index in range(20000)]
    result = detour(points=dense, max_seconds=.001, clock=limited_clock)
    assert result.points is None
    assert result.expansions == 0
    assert result.reason == 'time_budget'
    assert len(calls) < 100, 'budget must stop preprocessing, not just A* expansion'
