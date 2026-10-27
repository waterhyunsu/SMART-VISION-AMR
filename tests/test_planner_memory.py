"""Offline planner/memory regressions; importing ROS types starts no ROS nodes."""

import json
import unittest
from math import hypot
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import MapMetaData

from smart_vision_amr.astar import AStarPlanner
from smart_vision_amr.astar_planner import AStarPlannerNode
from smart_vision_amr.grid import Grid
from smart_vision_amr.map_utils import cell_to_world
from smart_vision_amr.obstacle_memory import ObstacleMemory


def grid_with_obstacles(width, height, obstacles):
    return Grid(tuple(tuple((x, y) in obstacles for x in range(width))
                      for y in range(height)))


class CombinedInflationEscapeTests(unittest.TestCase):
    def test_overlapping_halos_preserve_raw_blocks_and_bound_escape(self):
        start = (3, 3)
        raw_obstacles = {(2, 2), (5, 3)}
        raw_grid = grid_with_obstacles(11, 11, raw_obstacles)
        static_halo = {(x, y) for y in range(11) for x in range(5)}
        dynamic_halo = {(x, y) for y in range(11) for x in range(3, 11)}
        remembered, observed = {(3, 5)}, {(4, 2)}
        inflated = grid_with_obstacles(11, 11, static_halo | raw_obstacles)

        escape = AStarPlannerNode._combined_inflation_escape_halo(
            start, inflated, raw_grid, dynamic_halo, remembered, observed, 2)

        self.assertIn(start, escape)
        self.assertIn((4, 3), escape)  # Overlapping static and dynamic margin.
        self.assertIn((6, 6), escape)  # Three diagonal steps are still allowed.
        self.assertNotIn((7, 3), escape)  # Beyond inflation + one cell.
        self.assertFalse(escape & (raw_obstacles | remembered | observed))
        for x, y in escape:
            self.assertLessEqual(max(abs(x - start[0]), abs(y - start[1])), 3)

        # Even with the escape exemption, A* cannot end on a raw obstacle.
        for endpoint in raw_obstacles | remembered | observed:
            with self.subTest(endpoint=endpoint):
                self.assertEqual(AStarPlanner().plan(
                    inflated, start, endpoint, dynamic_halo,
                    allow_blocked_start=True, escape_cells=escape), [])

    def test_raw_blocked_start_never_gets_an_escape_exemption(self):
        start = (2, 2)
        empty = grid_with_obstacles(5, 5, set())
        inflated = grid_with_obstacles(5, 5, {start})
        cases = (
            (grid_with_obstacles(5, 5, {start}), set(), set()),
            (empty, {start}, set()),
            (empty, set(), {start}),
        )
        for raw, remembered, observed in cases:
            with self.subTest(remembered=remembered, observed=observed):
                self.assertEqual(AStarPlannerNode._combined_inflation_escape_halo(
                    start, inflated, raw, {start}, remembered, observed, 2), set())


class GoalFallbackTests(unittest.TestCase):
    def setUp(self):
        self.info = MapMetaData()
        self.info.width = self.info.height = 7
        self.info.resolution = 0.05
        self.info.origin.orientation.w = 1.0
        self.grid = grid_with_obstacles(7, 7, set())
        self.goal = PoseStamped()
        self.goal.pose.position.x, self.goal.pose.position.y = cell_to_world(
            self.info, (2, 2))
        self.planner = SimpleNamespace(
            planner=AStarPlanner(),
            get_parameter=lambda name: SimpleNamespace(value=0.05))

    def test_fallback_finds_reachable_cell_within_five_centimeters(self):
        result = AStarPlannerNode._nearest_reachable_goal(
            self.planner, self.info, self.goal, (0, 2), (2, 2), self.grid,
            {(2, 2)}, False, set())
        self.assertIsNotNone(result)
        endpoint, path, distance = result
        self.assertNotEqual(endpoint, (2, 2))
        self.assertEqual(path[0], (0, 2))
        self.assertEqual(path[-1], endpoint)
        self.assertLessEqual(distance, 0.05)
        self.assertAlmostEqual(distance, hypot(
            cell_to_world(self.info, endpoint)[0] - self.goal.pose.position.x,
            cell_to_world(self.info, endpoint)[1] - self.goal.pose.position.y))

    def test_fallback_does_not_expand_beyond_five_centimeters(self):
        # The robot has free cells to move through, but every cell close enough
        # to the requested goal is blocked. A farther endpoint is not acceptable.
        blocked = {(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)}
        self.assertIsNone(AStarPlannerNode._nearest_reachable_goal(
            self.planner, self.info, self.goal, (0, 2), (2, 2), self.grid,
            blocked, False, set()))


class PlannerRetryTests(unittest.TestCase):
    def make_state(self):
        return SimpleNamespace(
            last_goal=object(), planned_cells=[(1, 1), (2, 1)],
            waiting_for_transform=False, no_path=False,
            retry_after_failure=False, map=object(), plan_to=Mock(),
            get_logger=lambda: Mock())

    def test_failure_retains_goal_for_timer_retry(self):
        state = self.make_state()
        goal = state.last_goal
        AStarPlannerNode.on_navigation_status(
            state, SimpleNamespace(data=json.dumps({"status": "FAILED"})))
        self.assertIs(state.last_goal, goal)
        self.assertTrue(state.retry_after_failure)
        AStarPlannerNode.retry_transform(state)
        state.plan_to.assert_called_once_with(goal)

    def test_canceled_goal_cannot_retry_even_if_no_path_flag_remains(self):
        state = self.make_state()
        state.no_path = state.retry_after_failure = state.waiting_for_transform = True
        AStarPlannerNode.on_navigation_status(
            state, SimpleNamespace(data=json.dumps({"status": "CANCELED"})))
        self.assertIsNone(state.last_goal)
        self.assertEqual(state.planned_cells, [])
        self.assertFalse(state.retry_after_failure)
        self.assertFalse(state.waiting_for_transform)
        AStarPlannerNode.retry_transform(state)
        state.plan_to.assert_not_called()

    def test_no_path_retries_without_waiting_for_new_obstacle_messages(self):
        state = self.make_state()
        state.no_path = True
        AStarPlannerNode.retry_transform(state)
        state.plan_to.assert_called_once_with(state.last_goal)


class PlannerReplanHysteresisTests(unittest.TestCase):
    def _planner_state(self, *, remembered, observed, planned_cells):
        info = MapMetaData()
        info.width = info.height = 11
        message = SimpleNamespace(info=info)
        state = SimpleNamespace(
            map=(message, None), remembered=set(remembered), observed=set(observed),
            last_goal=object(), planned_cells=list(planned_cells), plan_to=Mock(),
            get_parameter=lambda name: SimpleNamespace(value=2))
        state._newly_blocked_dynamic_cells = AStarPlannerNode._newly_blocked_dynamic_cells
        return state

    def test_lidar_return_already_covered_by_memory_does_not_replan(self):
        # The raw scan cell is new, but its inflated blocked area is already
        # covered by the remembered obstacle. It must not churn the same route.
        state = self._planner_state(
            remembered={(5, 5)}, observed=set(), planned_cells=[(5, 5)])

        AStarPlannerNode.on_observed(
            state, SimpleNamespace(data=[5 * 11 + 5]))

        state.plan_to.assert_not_called()

    def test_newly_blocked_path_cell_still_triggers_replan(self):
        state = self._planner_state(
            remembered=set(), observed=set(), planned_cells=[(5, 5)])

        AStarPlannerNode.on_observed(
            state, SimpleNamespace(data=[5 * 11 + 5]))

        state.plan_to.assert_called_once_with(state.last_goal)

    def test_memory_confirmation_of_live_scan_cell_does_not_replan(self):
        state = self._planner_state(
            remembered=set(), observed={(5, 5)}, planned_cells=[(5, 5)])

        AStarPlannerNode.on_memory(
            state, SimpleNamespace(data=[5 * 11 + 5]))

        state.plan_to.assert_not_called()

    def test_transient_no_path_keeps_last_route_until_grace_expires(self):
        should_publish = AStarPlannerNode._should_publish_empty_path

        self.assertFalse(should_publish(True, 10.0, 10.49, 0.5))
        self.assertTrue(should_publish(True, 10.0, 10.5, 0.5))
        self.assertTrue(should_publish(False, 10.0, 10.0, 0.5))


class MemoryConfirmationTests(unittest.TestCase):
    def make_memory(self):
        return ObstacleMemory(
            5, 5, confirm_observations=5, clear_observations=5,
            confirm_min_duration_sec=2.0, clear_min_duration_sec=2.0,
            min_observation_interval_sec=0.25)

    def test_five_observations_are_insufficient_before_two_seconds(self):
        memory = self.make_memory()
        for stamp in (0.0, 0.5, 1.0, 1.5, 1.75):
            self.assertFalse(memory.observe_occupied((2, 2), stamp))
        self.assertNotIn((2, 2), memory.confirmed_cells)
        self.assertTrue(memory.observe_occupied((2, 2), 2.0))

    def test_two_seconds_are_insufficient_before_five_observations(self):
        memory = self.make_memory()
        for stamp in (0.0, 0.7, 1.4, 2.1):
            self.assertFalse(memory.observe_occupied((2, 2), stamp))
        self.assertNotIn((2, 2), memory.confirmed_cells)
        self.assertTrue(memory.observe_occupied((2, 2), 2.8))

    def test_fast_duplicate_observations_do_not_increase_count(self):
        memory = self.make_memory()
        for stamp in (0.0, 0.1, 0.2, 0.25):
            memory.observe_occupied((2, 2), stamp)
        self.assertEqual(memory.cells[(2, 2)].occupied_observations, 2)

    def test_visible_free_observation_restarts_unconfirmed_occupied_evidence(self):
        memory = self.make_memory()
        for stamp in (0.0, 0.5, 1.0, 1.5):
            memory.observe_occupied((2, 2), stamp)
        memory.observe_free((2, 2), 2.0)
        self.assertFalse(memory.observe_occupied((2, 2), 2.5))
        self.assertEqual(memory.cells[(2, 2)].occupied_observations, 1)
        self.assertEqual(memory.cells[(2, 2)].first_occupied_at, 2.5)
        self.assertNotIn((2, 2), memory.confirmed_cells)

    def test_confirmed_cell_clears_only_after_count_and_free_duration(self):
        memory = self.make_memory()
        for stamp in (0.0, 0.5, 1.0, 1.5, 2.0):
            memory.observe_occupied((2, 2), stamp)
        self.assertIn((2, 2), memory.confirmed_cells)
        for stamp in (3.0, 3.5, 4.0, 4.5, 4.75):
            self.assertFalse(memory.observe_free((2, 2), stamp))
        self.assertIn((2, 2), memory.confirmed_cells)
        self.assertTrue(memory.observe_free((2, 2), 5.0))
        self.assertNotIn((2, 2), memory.confirmed_cells)

    def test_unobserved_time_does_not_reset_evidence(self):
        memory = self.make_memory()
        for stamp in (0.0, 0.3, 0.6, 0.9):
            memory.observe_occupied((2, 2), stamp)
        self.assertTrue(memory.observe_occupied((2, 2), 100.0))


if __name__ == "__main__":
    unittest.main()
