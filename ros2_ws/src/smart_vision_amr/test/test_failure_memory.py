"""Execution memory changes directed planning, never occupancy or sensor policy."""

import json
from math import pi

import pytest

from smart_vision_amr.astar import AStarPlanner
from smart_vision_amr.failure_memory import FailureMemory
from smart_vision_amr.grid import Grid
from smart_vision_amr.reverse_recovery import Breadcrumb, GridGeometry, select_reverse_egress


def record(memory, **kwargs):
    return memory.record_failure(**dict({
        "x": 1.05, "y": 1.05, "approach_heading": 0.0,
        "reason": "execution_no_progress", "now": 10.0,
        "map_scope": "classroom-v1", "goal_scope": "map:1.85,1.05",
    }, **kwargs))


def constraints(memory, grid, **kwargs):
    return memory.constraints(grid, kwargs.pop("geometry", GridGeometry(.1, 0, 0)),
                              **dict({"now": 10.0, "map_scope": "classroom-v1",
                                      "goal_scope": "map:1.85,1.05"}, **kwargs))


def test_failed_approach_changes_astar_route_but_reverse_and_other_approach_work():
    grid = Grid.from_ascii("\n".join(["." * 21] * 21))
    original_rows = grid.rows
    memory = FailureMemory()
    record(memory)
    blocked, penalty = constraints(memory, grid)
    planner = AStarPlanner()
    initial = planner.plan(grid, (2, 10), (18, 10))
    alternate = planner.plan(grid, (2, 10), (18, 10),
                             blocked_edges=blocked, edge_penalty=penalty)
    assert initial and alternate and alternate != initial
    assert not (set(zip(alternate, alternate[1:])) & blocked)
    assert ((10, 10), (11, 10)) in blocked
    assert ((11, 10), (10, 10)) not in blocked
    assert ((10, 10), (10, 11)) not in blocked
    reverse = planner.plan(grid, (18, 10), (2, 10), blocked_edges=blocked)
    assert reverse == [(x, 10) for x in range(18, 1, -1)]
    # No cells or source map content were changed as a side effect.
    assert grid.rows is original_rows
    assert not any(any(row) for row in grid.rows)


def test_single_corridor_waits_and_expiry_restores_without_forcing_same_failure():
    grid = Grid.from_ascii("###########\n...........\n###########")
    memory = FailureMemory(ttl_sec=20, radius_m=.11)
    record(memory, x=.55, y=.15)
    blocked, penalty = constraints(memory, grid)
    planner = AStarPlanner()
    assert planner.plan(grid, (1, 1), (9, 1), blocked_edges=blocked,
                        edge_penalty=penalty) == []
    blocked, penalty = constraints(memory, grid, now=30.0)
    assert not blocked and not penalty
    assert planner.plan(grid, (1, 1), (9, 1), blocked_edges=blocked)
    assert memory.snapshot(now=30)["expired_count"] == 1


def test_scope_uses_geometry_and_target_not_live_occupancy_mutations():
    memory = FailureMemory()
    record(memory)
    grid = Grid.from_ascii("\n".join(["." * 21] * 21))
    before, _ = constraints(memory, grid)
    # A new occupancy object/overlay is not a reset of execution experience.
    changed = Grid.from_ascii("#" + "." * 20 + "\n" + "\n".join(["." * 21] * 20))
    after, _ = constraints(memory, changed)
    assert before == after
    assert constraints(memory, grid, map_scope="other-map") == (set(), {})
    assert constraints(memory, grid, goal_scope="map:other-goal") == (set(), {})
    assert memory.snapshot(now=11)["count"] == 1


@pytest.mark.parametrize("reason", ["scan_timeout", "tf_timeout", "odom_stale",
                                   "action_timeout", "cancelled", "manual_takeover",
                                   "controller_watchdog", "emergency_stop"])
def test_sensor_communication_watchdog_and_operator_events_never_poison_routes(reason):
    memory = FailureMemory()
    assert record(memory, reason=reason) is None
    assert memory.snapshot(now=11)["count"] == 0


def test_duplicate_event_does_not_extend_ttl_and_same_conditions_coalesce():
    memory = FailureMemory(ttl_sec=20)
    first = record(memory, failure_id="controller-a:1")
    assert record(memory, now=15, failure_id="controller-a:1") == first
    same = record(memory, now=16, x=1.08, approach_heading=.03,
                  reason="execution_oscillation", failure_id="controller-a:2")
    assert same == first
    item = memory.snapshot(now=17)["entries"][0]
    assert item["occurrences"] == 2
    assert item["expires_in_sec"] == 19
    assert item["x"] == 1.05  # no drift of the stored failed approach
    assert item["reason"] == "execution_oscillation"
    record(memory, now=18, approach_heading=pi, failure_id="controller-a:3")
    assert memory.snapshot(now=18)["count"] == 2


def test_old_event_cannot_rewind_or_extend_newer_failure():
    memory = FailureMemory(ttl_sec=20)
    first = record(memory, now=15, failure_id="new")
    assert record(memory, now=12, failure_id="old") == first
    item = memory.snapshot(now=16)["entries"][0]
    assert item["last_seen"] == 15
    assert item["occurrences"] == 1


def test_memory_bounded_and_only_explicit_verified_ids_release_it():
    memory = FailureMemory(max_entries=2)
    a = record(memory, goal_scope="a", now=10)
    b = record(memory, goal_scope="b", now=11)
    c = record(memory, goal_scope="c", now=12)
    assert memory.snapshot(now=12)["evicted_count"] == 1
    assert memory.release_verified([a, b, b, 999], now=12) == 1
    state = memory.snapshot(now=12)
    assert state["entries"][0]["id"] == c
    assert state["released_count"] == 1
    json.dumps(state, allow_nan=False)


def test_rotated_map_constraints_keep_world_approach_heading():
    grid = Grid.from_ascii("\n".join(["." * 21] * 21))
    memory = FailureMemory(radius_m=.11)
    # local cell (10,10) becomes world (-1.05, 1.05), local +y is world -x.
    record(memory, x=-1.05, y=1.05, approach_heading=pi)
    blocked, _ = constraints(memory, grid, geometry=GridGeometry(.1, 0, 0, pi / 2))
    assert ((10, 10), (10, 11)) in blocked
    assert ((10, 11), (10, 10)) not in blocked
    assert ((10, 10), (11, 10)) not in blocked


def test_smoothing_shortcut_cannot_hide_a_failed_approach():
    memory = FailureMemory()
    record(memory)
    kwargs = {"now": 11, "map_scope": "classroom-v1", "goal_scope": "map:1.85,1.05"}
    assert memory.segment_blocked((.2, 1.05), (1.8, 1.05), **kwargs)
    assert not memory.segment_blocked((1.8, 1.05), (.2, 1.05), **kwargs)
    assert not memory.segment_blocked((1.05, .2), (1.05, 1.8), **kwargs)
    assert not memory.segment_blocked((.2, 1.5), (1.8, 1.5), **kwargs)
    assert not memory.segment_blocked((1.05, 1.05), (1.05, 1.05), **kwargs)


def test_reverse_candidate_reachability_must_include_same_failure_constraints():
    # Only one narrow horizontal corridor. Retreat itself is physically clear,
    # but no candidate should be offered if all continue into the failed edge.
    grid = Grid.from_ascii("\n".join(["#" * 30] + ["." * 30] * 5 + ["#" * 30]))
    geometry = GridGeometry(.1, 0, 0)
    memory = FailureMemory(radius_m=.6)
    record(memory, x=1.8, y=.35)
    blocked, penalty = constraints(memory, grid)
    history = [Breadcrumb(10 + i * .1, 1.0 + i * .05, .35, 0)
               for i in range(5)]
    planner = AStarPlanner()
    goal = (25, 3)
    settings = dict(now=10.4, robot_radius=.05, clearance_margin=0,
                    min_clearance_gain=0, min_distance=.1, max_distance=.25,
                    max_pose_jump=.1)
    baseline = select_reverse_egress(history, grid, grid, geometry,
        lambda cell: bool(planner.plan(grid, cell, goal)), **settings)
    remembered = select_reverse_egress(history, grid, grid, geometry,
        lambda cell: bool(planner.plan(grid, cell, goal, blocked_edges=blocked,
                                       edge_penalty=penalty)), **settings)
    assert baseline is not None
    assert remembered is None


@pytest.mark.parametrize("kwargs", [
    {"ttl_sec": 0}, {"radius_m": -.1}, {"heading_tolerance_rad": pi},
    {"max_entries": 0}, {"max_entries": True}, {"penalty_cost": -1},
    {"radius_m": float("nan")}, {"ttl_sec": float("inf")},
])
def test_invalid_settings_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        FailureMemory(**kwargs)


@pytest.mark.parametrize("kwargs", [{"x": float("nan")}, {"map_scope": ""},
                                    {"goal_scope": ""}, {"failure_id": ""}])
def test_invalid_failure_cannot_create_bad_json_or_edges(kwargs):
    memory = FailureMemory()
    with pytest.raises(ValueError):
        record(memory, **kwargs)
    assert memory.snapshot(now=11)["count"] == 0
