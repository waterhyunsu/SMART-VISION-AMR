"""Retained-goal recovery tests use real A* and messages without actuating a robot."""

import json
import unittest
from collections import deque
from math import cos, sin
from types import SimpleNamespace
from unittest.mock import Mock, patch

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Int32MultiArray, String
from tf2_ros import TransformException

from smart_vision_amr.astar import AStarPlanner
from smart_vision_amr.astar_planner import AStarPlannerNode
from smart_vision_amr.map_utils import occupancy_to_grid


class PlannerHarness(AStarPlannerNode):
    def __init__(self):
        self.params = {"map_frame": "map", "base_frame": "base_link",
                       "occupied_threshold": 65, "obstacle_inflation_cells": 0,
                       "replan_retry_period": 2.0, "robot_radius": 0.22,
                       "start_egress_enabled": True, "start_egress_clearance_margin": 0.02,
                       "start_egress_max_cells": 7, "live_obstacle_layer_required": False,
                       "live_obstacle_timeout": 2.0,
                       "mapped_wall_return_tolerance_cells": 0}
        self.params["preferred_clearance"] = 0.45
        self.params["clearance_penalty_gain"] = 1.5
        self.params.update({"reverse_recovery_enabled": True,
                            "reverse_recovery_min_distance": 0.10,
                            "reverse_recovery_max_distance": 0.30,
                            "reverse_recovery_clearance_margin": 0.02,
                            "reverse_recovery_speed": 0.04,
                            "reverse_recovery_max_attempts": 3})
        self.map = None
        self.remembered = set()
        self.observed = set()
        self.last_observation_time = None
        self.last_goal = None
        self.planned_cells = []
        self.waiting_for_transform = False
        self.waiting_for_replan = False
        self.navigation_suspended = False
        self.goal_completed = False
        self.last_plan_time = float("-inf")
        self.invalid_path_published = False
        self.stop_stamp_ns = None
        self.breadcrumbs = deque(maxlen=300)
        self.breadcrumb_odom_anchor = None
        self.latest_odom = None
        self.odom_samples = deque(maxlen=100)
        self.breadcrumb_map_key = None
        self.recovery_active = False
        self.recovery_requested_at = None
        self.recovery_cancel_due_map_change = False
        self.recovery_attempts = 0
        self.recovery_attempted_cells = set()
        self.planner = AStarPlanner()
        self.tf_buffer = Mock()
        self.tf_buffer.lookup_transform.return_value = SimpleNamespace(
            transform=SimpleNamespace(translation=SimpleNamespace(x=0.5, y=1.5)))
        self.path_pub, self.status_pub, self.recovery_pub, self.logger = Mock(), Mock(), Mock(), Mock()

    def get_logger(self):
        return self.logger

    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: self.last_goal.header.stamp))


def make_map(blocked=()):
    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.info.resolution = 1.0
    msg.info.width, msg.info.height = 5, 3
    msg.info.origin.orientation.w = 1.0
    msg.data = [0] * 15
    for x, y in blocked:
        msg.data[y * 5 + x] = 100
    return msg


def make_goal(stamp=10):
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp.sec = stamp
    goal.pose.position.x, goal.pose.position.y = 4.5, 1.5
    goal.pose.orientation.w = 1.0
    return goal


def make_narrow_corridor():
    """A 0.75 m passage whose 0.25 m static margins still permit a route."""
    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.info.resolution = 0.05
    msg.info.width, msg.info.height = 16, 20
    msg.info.origin.orientation.w = 1.0
    msg.data = [0] * (16 * 20)
    for y in range(20):
        msg.data[y * 16] = 100
        msg.data[y * 16 + 15] = 100
    return msg


def make_reverse_egress_map():
    """A real footprint clears a wall, but the nominal A* start is inflated."""
    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.info.resolution = 0.1
    msg.info.width = msg.info.height = 20
    msg.info.origin.orientation.w = 1.0
    msg.data = [0] * 400
    msg.data[10 * 20] = 100
    return msg


def status(node, name, **extra):
    goal = node.last_goal
    payload = {"status": name,
               "goal": {"x": goal.pose.position.x, "y": goal.pose.position.y, "frame_id": "map"},
               "goal_stamp": {"sec": goal.header.stamp.sec, "nanosec": goal.header.stamp.nanosec}}
    payload.update(extra)
    msg = String()
    msg.data = json.dumps(payload)
    return msg


class PlannerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.node = PlannerHarness()
        msg = make_map()
        self.node.map = msg, occupancy_to_grid(msg)
        self.clock = patch("smart_vision_amr.astar_planner.time.monotonic", return_value=100.0)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)

    def set_verified_pose(self, x, y=1.05, *, yaw=3.141592653589793,
                          odom_x=None, stamp_ns=0, record=True):
        """Supply synchronized map TF and independent odometry to the planner."""
        odom = Odometry()
        odom.header.frame_id = 'odom'
        odom.header.stamp.sec = 10
        odom.header.stamp.nanosec = stamp_ns
        odom.pose.pose.position.x = x if odom_x is None else odom_x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = sin(yaw / 2.0)
        odom.pose.pose.orientation.w = cos(yaw / 2.0)
        self.node.on_odom(odom)
        self.node.tf_buffer.lookup_transform.return_value = SimpleNamespace(
            header=SimpleNamespace(stamp=odom.header.stamp),
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=x, y=y),
                rotation=SimpleNamespace(x=0., y=0.,
                                         z=sin(yaw / 2.0), w=cos(yaw / 2.0))))
        if record:
            self.node._record_breadcrumb()

    def reverse_map(self):
        map_msg = make_reverse_egress_map()
        self.node.map = map_msg, occupancy_to_grid(map_msg, 65, 5)
        self.node.raw_static_grid = occupancy_to_grid(map_msg, 65, 0)
        self.node.map_signature = (65, 5, 0)
        self.node.params['obstacle_inflation_cells'] = 5

    def forward_trail_to_inflated_start(self, final_x=.35):
        """Travel 0.30 m in one straight odom-verified segment under goal 10."""
        self.reverse_map()
        self.set_verified_pose(final_x + .30, record=False)
        old_goal = make_goal(stamp=10)
        old_goal.pose.position.x, old_goal.pose.position.y = 1.55, 1.05
        self.node.on_goal(old_goal)
        for step, x in enumerate((final_x + .20, final_x + .10, final_x), 1):
            self.now.return_value = 100.0 + step * .1
            self.set_verified_pose(x, stamp_ns=step * 100_000_000)
        self.assertGreaterEqual(len(self.node.breadcrumbs), 4)

    def test_no_path_then_obstacle_clear_retries_same_goal_at_bounded_rate(self):
        self.node.observed = {(2, 0), (2, 1), (2, 2)}
        goal = make_goal()
        self.node.on_goal(goal)
        self.assertIs(self.node.last_goal, goal)
        self.assertTrue(self.node.waiting_for_replan)
        self.assertFalse(self.node.path_pub.publish.call_args.args[0].poses)

        self.now.return_value = 100.2
        self.node.on_observed(Int32MultiArray(data=[]))
        self.node.retry_transform()
        self.assertEqual(self.node.path_pub.publish.call_count, 1)
        self.now.return_value = 102.1
        self.node.retry_transform()
        self.assertTrue(self.node.path_pub.publish.call_args.args[0].poses)
        self.assertFalse(self.node.waiting_for_replan)
        self.assertEqual(json.loads(self.node.status_pub.publish.call_args.args[0].data)["status"], "READY")
        self.assertIs(self.node.last_goal, goal)

    def test_repeated_failed_attempts_publish_only_one_empty_cancel(self):
        self.node.observed = {(2, 0), (2, 1), (2, 2)}
        self.node.on_goal(make_goal())
        for elapsed in (2.1, 4.2, 6.3):
            self.now.return_value = 100.0 + elapsed
            self.node.retry_transform()
        self.assertEqual(self.node.path_pub.publish.call_count, 1)
        self.assertTrue(self.node.waiting_for_replan)

    def test_observed_obstacle_on_active_path_stops_before_throttled_replan(self):
        self.node.on_goal(make_goal())
        self.now.return_value = 100.1
        self.node.on_observed(Int32MultiArray(data=[2, 7, 12]))
        self.assertFalse(self.node.path_pub.publish.call_args.args[0].poses)
        self.assertFalse(self.node.planned_cells)
        self.assertTrue(self.node.waiting_for_replan)

    def test_unmapped_obstacle_replans_around_it_without_changing_saved_map_or_goal(self):
        goal = make_goal()
        self.node.on_goal(goal)
        original_map = list(self.node.map[0].data)
        self.assertIn((2, 1), self.node.planned_cells)
        self.now.return_value = 100.1
        self.node.on_observed(Int32MultiArray(data=[7]))
        self.assertFalse(self.node.path_pub.publish.call_args.args[0].poses)
        self.now.return_value = 102.1
        self.node.retry_transform()
        self.assertTrue(self.node.planned_cells)
        self.assertNotIn((2, 1), self.node.planned_cells)
        self.assertIs(self.node.last_goal, goal)
        self.assertEqual(list(self.node.map[0].data), original_map)

    def test_required_live_layer_holds_on_missing_or_stale_scans_then_recovers_same_goal(self):
        self.node.params['live_obstacle_layer_required'] = True
        goal = make_goal()
        self.node.on_goal(goal)
        self.assertFalse(self.node.path_pub.publish.call_args.args[0].poses)
        self.now.return_value = 102.1
        self.node.on_observed(Int32MultiArray(data=[]))
        self.assertTrue(self.node.planned_cells)
        self.now.return_value = 104.2
        self.node.retry_transform()
        self.assertFalse(self.node.planned_cells)
        self.assertTrue(self.node.waiting_for_replan)
        self.node.on_observed(Int32MultiArray(data=[]))
        self.assertTrue(self.node.planned_cells)
        self.assertIs(self.node.last_goal, goal)

    def test_updated_static_map_stops_active_route_before_retry_interval(self):
        self.node.on_goal(make_goal())
        self.now.return_value = 100.1
        self.node.on_map(make_map(blocked=[(2, 1)]))
        self.assertFalse(self.node.path_pub.publish.call_args.args[0].poses)
        self.assertTrue(self.node.waiting_for_replan)

    def test_controller_abort_invalidates_route_and_retries_identical_goal(self):
        self.node.on_goal(make_goal())
        first = self.node.path_pub.publish.call_args.args[0]
        self.node.on_navigation_status(status(self.node, "FAILED"))
        self.assertFalse(self.node.path_pub.publish.call_args.args[0].poses)
        self.now.return_value = 102.1
        self.node.retry_transform()
        retried = self.node.path_pub.publish.call_args.args[0]
        self.assertEqual([(p.pose.position.x, p.pose.position.y) for p in first.poses],
                         [(p.pose.position.x, p.pose.position.y) for p in retried.poses])

    def test_manual_stop_and_success_suppress_auto_retry_until_explicit_goal(self):
        self.node.on_goal(make_goal())
        self.node.on_navigation_status(status(self.node, "CANCELED", manual_stop=True,
                                                   stop_stamp={"sec": 10, "nanosec": 0}))
        count = self.node.path_pub.publish.call_count
        self.now.return_value = 110.0
        self.node.on_map(make_map())
        self.node.on_observed(Int32MultiArray(data=[7]))
        self.node.retry_transform()
        self.assertEqual(self.node.path_pub.publish.call_count, count)
        self.node.on_goal(make_goal(stamp=11))
        self.assertFalse(self.node.navigation_suspended)
        self.assertGreater(self.node.path_pub.publish.call_count, count)
        self.node.on_navigation_status(status(self.node, "SUCCEEDED"))
        count = self.node.path_pub.publish.call_count
        self.now.return_value = 120.0
        self.node.on_map(make_map())
        self.node.retry_transform()
        self.assertEqual(self.node.path_pub.publish.call_count, count)

    def test_stop_before_queued_goal_suppresses_stale_and_unstamped_requests(self):
        stop = String()
        stop.data = json.dumps({"status": "CANCELED", "manual_stop": True,
                                "goal": None, "goal_stamp": None,
                                "stop_stamp": {"sec": 20, "nanosec": 0}})
        self.node.on_navigation_status(stop)
        self.node.on_goal(make_goal(stamp=10))
        self.node.on_goal(make_goal(stamp=0))
        self.node.on_goal(make_goal(stamp=20))
        self.assertIsNone(self.node.last_goal)
        self.node.path_pub.publish.assert_not_called()
        self.node.on_goal(make_goal(stamp=21))
        self.assertEqual(self.node.last_goal.header.stamp.sec, 21)
        self.assertFalse(self.node.navigation_suspended)
        self.assertTrue(self.node.planned_cells)

    def test_late_stop_does_not_suspend_genuinely_newer_goal(self):
        self.node.on_goal(make_goal(stamp=21))
        self.node.on_navigation_status(status(self.node, "CANCELED", manual_stop=True,
                                             stop_stamp={"sec": 20, "nanosec": 0}))
        self.assertFalse(self.node.navigation_suspended)
        self.assertFalse(self.node.waiting_for_replan)

    def test_late_old_goal_success_does_not_complete_current_goal(self):
        self.node.on_goal(make_goal(stamp=10))
        previous_status = status(self.node, "SUCCEEDED")
        self.node.on_goal(make_goal(stamp=11))
        self.node.on_navigation_status(previous_status)
        self.assertFalse(self.node.goal_completed)

    def test_no_forward_route_requests_one_verified_reverse_then_replans_same_goal(self):
        self.forward_trail_to_inflated_start()
        self.node.on_navigation_status(status(self.node, "SUCCEEDED"))
        self.node.recovery_attempts = 2  # previous sector's budget is not inherited
        goal = make_goal(stamp=11)
        goal.pose.position.x, goal.pose.position.y = 1.55, 1.05
        self.node.on_goal(goal)
        request = json.loads(self.node.recovery_pub.publish.call_args.args[0].data)
        self.assertEqual(request["goal_id"], "11:0")
        self.assertAlmostEqual(request["distance"], .30)
        self.assertEqual(request["exit_cell"], [6, 10])
        self.assertTrue(self.node.recovery_active)
        self.assertEqual(self.node.recovery_attempts, 1)
        self.node.retry_transform()
        self.assertEqual(self.node.recovery_pub.publish.call_count, 1)

        # The Nav2 backup moved the robot to the selected earlier pose.
        self.now.return_value = 100.4
        self.set_verified_pose(.65, stamp_ns=400_000_000, record=False)
        self.node.tf_buffer.lookup_transform.return_value.transform.translation.x = 0.65
        self.node.on_navigation_status(status(self.node, "RECOVERY_SUCCEEDED"))
        self.assertFalse(self.node.recovery_active)
        # Replanning may seed the new pose, but the pre-reverse trail is gone.
        self.assertLessEqual(len(self.node.breadcrumbs), 1)
        self.assertIs(self.node.last_goal, goal)
        self.assertTrue(self.node.planned_cells)
        self.assertEqual(self.node.planned_cells[0], (6, 10))

    def test_reverse_rejected_when_start_footprint_margin_is_insufficient(self):
        self.forward_trail_to_inflated_start(final_x=.31)
        goal = make_goal(stamp=11)
        goal.pose.position.x, goal.pose.position.y = 1.55, 1.05
        self.node.on_goal(goal)
        self.node.recovery_pub.publish.assert_not_called()

    def test_amcl_map_correction_without_odom_travel_never_creates_reverse_history(self):
        self.reverse_map()
        self.set_verified_pose(.65, record=False)
        old_goal = make_goal(stamp=10)
        old_goal.pose.position.x, old_goal.pose.position.y = 1.55, 1.05
        self.node.on_goal(old_goal)
        for step, x in enumerate((.63, .60, .55, .45, .35), 1):
            self.now.return_value = 100.0 + step * .1
            # Map->odom moves, but /odom reports no physical chassis travel.
            self.set_verified_pose(x, odom_x=.65, stamp_ns=step * 100_000_000)
            self.assertLessEqual(len(self.node.breadcrumbs), 1)
        new_goal = make_goal(stamp=11)
        new_goal.pose.position.x, new_goal.pose.position.y = 1.55, 1.05
        self.node.on_goal(new_goal)
        self.node.recovery_pub.publish.assert_not_called()

    def test_stale_odom_clears_cross_goal_reverse_history(self):
        self.forward_trail_to_inflated_start()
        self.now.return_value = 101.2
        new_goal = make_goal(stamp=11)
        new_goal.pose.position.x, new_goal.pose.position.y = 1.55, 1.05
        self.node.on_goal(new_goal)
        self.assertFalse(self.node.breadcrumbs)
        self.node.recovery_pub.publish.assert_not_called()

    def test_odom_pose_jump_resets_the_verified_trail(self):
        self.reverse_map()
        self.set_verified_pose(.65, record=False)
        old_goal = make_goal(stamp=10)
        old_goal.pose.position.x, old_goal.pose.position.y = 1.55, 1.05
        self.node.on_goal(old_goal)
        self.now.return_value = 100.1
        self.set_verified_pose(.55, stamp_ns=100_000_000)
        self.assertEqual(len(self.node.breadcrumbs), 2)
        self.now.return_value = 100.2
        self.set_verified_pose(.35, stamp_ns=200_000_000)
        self.assertEqual(len(self.node.breadcrumbs), 1)

    def test_manual_stop_erases_trail_before_a_future_goal(self):
        self.forward_trail_to_inflated_start()
        self.node.on_navigation_status(status(
            self.node, 'CANCELED', manual_stop=True,
            stop_stamp={'sec': 10, 'nanosec': 0}))
        self.assertFalse(self.node.breadcrumbs)
        self.assertIsNone(self.node.breadcrumb_odom_anchor)

    def test_changed_map_content_clears_verified_history(self):
        self.forward_trail_to_inflated_start()
        original = self.node.map[0]
        self.node.on_map(original)
        self.assertTrue(self.node.breadcrumbs)
        updated = make_reverse_egress_map()
        updated.data[19 * 20 + 19] = 100
        self.node.on_map(updated)
        self.assertFalse(self.node.breadcrumbs)

    def test_map_change_cancels_active_reverse_before_replanning_retained_goal(self):
        self.forward_trail_to_inflated_start()
        self.node.on_map(self.node.map[0])
        goal = make_goal(stamp=11)
        goal.pose.position.x, goal.pose.position.y = 1.55, 1.05
        self.node.on_goal(goal)
        self.assertTrue(self.node.recovery_active)
        updated = make_reverse_egress_map()
        updated.info.map_load_time.sec = 123  # same cells, reloaded map source
        self.node.on_map(updated)
        cancel = json.loads(self.node.recovery_pub.publish.call_args.args[0].data)
        self.assertEqual(cancel['goal_id'], '11:0')
        self.assertIs(cancel['cancel'], True)
        self.assertFalse(self.node.breadcrumbs)
        self.assertFalse(self.node.path_pub.publish.call_args.args[0].poses)
        self.assertIs(self.node.last_goal, goal)
        self.assertTrue(self.node.recovery_active)  # waits for follower cancel result

        self.node.on_navigation_status(status(self.node, 'RECOVERY_FAILED'))
        self.assertFalse(self.node.recovery_active)
        self.assertTrue(self.node.waiting_for_replan)
        self.assertIs(self.node.last_goal, goal)

    def test_missing_backup_result_cannot_hold_retained_goal_forever(self):
        self.node.on_goal(make_goal())
        self.node.recovery_active = True
        self.node.recovery_requested_at = 100.0
        self.node.recovery_attempted_cells.add((0, 1))
        self.now.return_value = 125.1
        self.node.retry_transform()
        self.assertFalse(self.node.recovery_active)
        self.assertTrue(self.node.waiting_for_replan)
        self.assertIsNotNone(self.node.last_goal)
        self.assertEqual(self.node.recovery_attempted_cells, {(0, 1)})

    def test_confirmed_backup_stop_replans_current_goal_immediately(self):
        goal = make_goal()
        self.node.on_goal(goal)
        self.node.path_pub.publish.reset_mock()
        self.node.recovery_active = True
        self.node.recovery_requested_at = 100.0
        self.node.waiting_for_replan = False
        self.node.on_navigation_status(status(self.node, "RECOVERY_STOPPED"))
        self.assertFalse(self.node.recovery_active)
        self.assertIs(self.node.last_goal, goal)
        self.assertTrue(self.node.path_pub.publish.call_args.args[0].poses)
        self.assertFalse(self.node.waiting_for_replan)

    def test_static_start_collision_remains_blocked(self):
        msg = make_map(blocked=[(0, 1)])
        self.node.map = msg, occupancy_to_grid(msg)
        self.node.on_goal(make_goal())
        self.assertFalse(self.node.planned_cells)
        self.assertTrue(self.node.waiting_for_replan)
        self.assertFalse(self.node.path_pub.publish.call_args.args[0].poses)

    def corner_start(self):
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.resolution = .05
        msg.info.width = msg.info.height = 30
        msg.info.origin.orientation.w = 1.
        msg.data = [0]*900
        msg.data[14*30+13] = 100
        self.node.params['obstacle_inflation_cells'] = 6
        self.node.tf_buffer.lookup_transform.return_value.transform.translation.x = .525
        self.node.tf_buffer.lookup_transform.return_value.transform.translation.y = 1.025
        self.node.on_map(msg)
        goal = make_goal()
        goal.pose.position.x, goal.pose.position.y = .425, .375
        return goal

    def test_physically_clear_inflated_start_recovers_without_changing_goal_or_margin(self):
        goal = self.corner_start()
        self.node.on_goal(goal)
        self.assertIs(self.node.last_goal, goal)
        self.assertTrue(self.node.map[1].is_static_obstacle((10, 20)))
        self.assertFalse(self.node.raw_static_grid.is_static_obstacle((10, 20)))
        self.assertEqual(self.node.planned_cells[:2], [(10, 20), (10, 21)])
        self.assertTrue(all(not self.node.map[1].is_static_obstacle(cell)
                            for cell in self.node.planned_cells[1:]))
        self.assertEqual(self.node.params['obstacle_inflation_cells'], 6)
        message = json.loads(self.node.status_pub.publish.call_args.args[0].data)
        self.assertEqual(message['status'], 'READY')
        self.assertIn('safe start egress 1 cells', message['message'])
        self.assertEqual(message['goal_stamp'], {'sec': 10, 'nanosec': 0})

    def test_start_egress_stays_blocked_when_footprint_clearance_is_insufficient(self):
        goal = self.corner_start()
        self.node.params['robot_radius'] = .50
        self.node.on_goal(goal)
        self.assertFalse(self.node.planned_cells)
        self.assertTrue(self.node.waiting_for_replan)

    def test_runtime_inflation_change_rebuilds_cached_map_without_new_map_message(self):
        goal = self.corner_start()
        self.node.params['start_egress_enabled'] = False
        self.node.on_goal(goal)
        self.assertFalse(self.node.planned_cells)
        self.node.params['obstacle_inflation_cells'] = 5
        self.now.return_value = 102.1
        self.node.retry_transform()
        self.assertTrue(self.node.planned_cells)
        self.assertFalse(self.node.map[1].is_static_obstacle((10, 20)))

    def test_map_recovery_keeps_goal(self):
        self.node.map = None
        self.node.on_goal(make_goal())
        self.assertTrue(self.node.waiting_for_replan)
        self.now.return_value = 102.1
        self.node.on_map(make_map())
        self.assertTrue(self.node.planned_cells)

    def test_missing_tf_stops_and_recovers_only_after_transform_returns(self):
        self.node.tf_buffer.lookup_transform.side_effect = TransformException("missing transform")
        goal = make_goal()
        self.node.on_goal(goal)
        self.assertFalse(self.node.path_pub.publish.call_args.args[0].poses)
        self.assertTrue(self.node.waiting_for_transform)
        self.node.tf_buffer.lookup_transform.side_effect = None
        self.now.return_value = 102.1
        self.node.retry_transform()
        self.assertTrue(self.node.path_pub.publish.call_args.args[0].poses)
        self.assertFalse(self.node.waiting_for_transform)
        self.assertIs(self.node.last_goal, goal)

    def test_out_of_map_pose_waits_instead_of_crashing(self):
        self.node.tf_buffer.lookup_transform.return_value.transform.translation.x = -1.0
        self.node.on_goal(make_goal())
        self.assertTrue(self.node.waiting_for_replan)
        self.assertFalse(self.node.path_pub.publish.call_args.args[0].poses)

    def test_goal_quaternion_preserved_and_zero_quaternion_falls_back(self):
        goal = make_goal()
        goal.pose.orientation.z, goal.pose.orientation.w = 2.0, 0.0
        self.node.on_goal(goal)
        end = self.node.path_pub.publish.call_args.args[0].poses[-1].pose.orientation
        self.assertEqual((end.z, end.w), (1.0, 0.0))
        goal.pose.orientation.z = 0.0
        self.node.on_goal(goal)
        end = self.node.path_pub.publish.call_args.args[0].poses[-1].pose.orientation
        self.assertAlmostEqual(end.z * end.z + end.w * end.w, 1.0)

    def narrow_corridor(self):
        self.node.params['obstacle_inflation_cells'] = 5
        self.node.tf_buffer.lookup_transform.return_value.transform.translation.x = .375
        self.node.tf_buffer.lookup_transform.return_value.transform.translation.y = .075
        self.node.on_map(make_narrow_corridor())
        goal = make_goal()
        goal.pose.position.x, goal.pose.position.y = .375, .925
        return goal

    def test_near_wall_lidar_returns_do_not_double_inflate_shut_corridor(self):
        goal = self.narrow_corridor()
        self.node.observed = {(2, 10), (13, 10)}
        self.node.on_goal(goal)
        self.assertFalse(self.node.planned_cells)
        diagnostic = self.node.logger.error.call_args.args[0]
        self.assertIn('observed_raw=2', diagnostic)
        self.assertIn('static_reference_dynamic_hits=', diagnostic)
        self.assertIn('first_blocked_reference_cells=', diagnostic)

        self.node.params['mapped_wall_return_tolerance_cells'] = 2
        self.now.return_value = 102.1
        self.node.retry_transform()
        self.assertTrue(self.node.planned_cells)
        self.assertIs(self.node.last_goal, goal)
        self.assertEqual(self.node.observed, {(2, 10), (13, 10)})
        self.assertEqual(self.node.params['obstacle_inflation_cells'], 5)

    def test_wall_match_does_not_hide_new_obstacle_in_corridor(self):
        goal = self.narrow_corridor()
        self.node.params['mapped_wall_return_tolerance_cells'] = 2
        self.node.on_goal(goal)
        self.assertTrue(self.node.planned_cells)
        self.now.return_value = 100.1
        self.node.on_observed(Int32MultiArray(data=[10 * 16 + 2, 10 * 16 + 13]))
        self.assertTrue(self.node.planned_cells)
        self.node.on_observed(Int32MultiArray(data=[10 * 16 + 8]))
        self.assertFalse(self.node.planned_cells)
        self.now.return_value = 102.1
        self.node.retry_transform()
        self.assertTrue(self.node.waiting_for_replan)
        self.assertFalse(self.node.planned_cells)
        diagnostic = self.node.logger.error.call_args.args[0]
        self.assertIn('observed_effective=1', diagnostic)
        self.assertIn('observed_near_map_wall=0', diagnostic)

    def test_confirmed_wall_edge_memory_keeps_full_footprint_clearance(self):
        goal = self.narrow_corridor()
        self.node.params['mapped_wall_return_tolerance_cells'] = 2
        self.node.remembered = {(2, 10), (13, 10)}
        self.node.on_goal(goal)
        self.assertFalse(self.node.planned_cells)
        self.assertTrue(self.node.waiting_for_replan)

    def test_new_wall_edge_memory_stops_route_that_loses_full_clearance(self):
        goal = self.narrow_corridor()
        self.node.params['mapped_wall_return_tolerance_cells'] = 2
        self.node.on_goal(goal)
        self.assertTrue(self.node.planned_cells)
        self.node.on_memory(Int32MultiArray(data=[10 * 16 + 2, 10 * 16 + 13]))
        self.assertFalse(self.node.planned_cells)
        self.assertTrue(self.node.waiting_for_replan)

    def test_confirmed_obstacle_away_from_wall_still_blocks_corridor(self):
        goal = self.narrow_corridor()
        self.node.params['mapped_wall_return_tolerance_cells'] = 2
        self.node.remembered = {(8, 10)}
        self.node.on_goal(goal)
        self.assertFalse(self.node.planned_cells)
        self.assertTrue(self.node.waiting_for_replan)


if __name__ == "__main__":
    unittest.main()
