"""Exercise asynchronous stop/replan races without starting ROS or Nav2 processes."""

import json
import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger

from smart_vision_amr.path_follower import PathFollower


class FollowerHarness(PathFollower):
    """Use real follower callbacks and ROS messages, with fake action futures."""

    def __init__(self):
        self.client = Mock()
        self.client.wait_for_server.return_value = True
        self.backup_client = Mock()
        self.backup_client.wait_for_server.return_value = True
        self.logger = Mock()
        self.status_pub = Mock()
        self.safety_control_pub = Mock()
        self.safety_control_seq = 0
        self.active_goal = None
        self.goal_request_pending = False
        self.pending_path = None
        self.cancel_requested = False
        self.cancel_when_accepted = False
        self.navigation_suspended = False
        self.require_goal_match = False
        self.requested_goal = None
        self.suspended_paths = deque(maxlen=10)
        self.resume_goal_tolerance = 0.1
        self.latest_path_key = None
        self.active_path = None
        self.goal_generation = 0
        self.active_context = None
        self.request_context = None
        self.path_invalidated = False
        self.recovery_cancel_path_cutoff_ns = None
        self.stop_stamp_ns = None
        self.recovery_request = None
        self.pending_recovery_cancel = None
        self.recovery_seq = 0
        self.backup_goal = None
        self.backup_goal_request_pending = False
        self.backup_cancel_when_accepted = False
        self.backup_cancel_requested = False
        self.backup_cancel_pending_since = None
        self.backup_last_cancel_at = None
        self.backup_cancel_safety_fault = False
        self.backup_cancel_replan_required = False
        self.backup_cancel_estop_sent = False
        self.backup_last_estop_at = None
        self.latest_scan = None
        self.latest_scan_at = None
        self.latest_odom = None
        self.latest_odom_at = None
        self.motion_status = None
        self.motion_status_at = None
        self.clock_seconds = 5
        self.parameters = {
            "recovery_min_distance": 0.10, "recovery_max_distance": 0.35,
            "recovery_min_speed": 0.03, "recovery_max_speed": 0.07,
            "recovery_robot_radius": 0.22, "recovery_rear_margin": 0.05,
            "recovery_sensor_timeout": 1.0,
            "recovery_rear_half_angle_deg": 35.0,
            "recovery_max_time": 15.0,
            "recovery_cancel_timeout": 5.0,
            "recovery_settle_timeout": 2.0,
            "recovery_scan_frame": "base_scan",
        }

    def get_clock(self):
        stamp = PoseStamped().header.stamp
        stamp.sec = self.clock_seconds
        return SimpleNamespace(now=lambda: SimpleNamespace(
            to_msg=lambda: stamp,
            nanoseconds=self.clock_seconds * 1_000_000_000,
        ))

    def get_logger(self):
        return self.logger

    def get_parameter(self, name):
        return SimpleNamespace(value=self.parameters.get(name, ""))


def make_path(y=0.0, stamp=0):
    path = Path()
    path.header.frame_id = "map"
    path.header.stamp.sec = stamp
    for x in (0.0, 1.0, 2.0):
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    return path


def goal_for(path, stamp=0):
    goal = PoseStamped()
    goal.header.frame_id = path.header.frame_id
    goal.header.stamp.sec = stamp
    goal.pose = path.poses[-1].pose
    return goal


def result_future(status=GoalStatus.STATUS_CANCELED):
    return Mock(result=Mock(return_value=SimpleNamespace(status=status)))


class CancelNavigationTest(unittest.TestCase):
    def setUp(self):
        self.node = FollowerHarness()

    def cancel(self):
        return self.node.cancel_navigation(Trigger.Request(), Trigger.Response())

    def test_stop_clears_queued_path_even_without_an_active_handle(self):
        self.node.pending_path = make_path()
        self.node.latest_path_key = ((1.0, 0.0, 0.0, 1.0),)
        self.node.active_path = make_path()

        self.assertTrue(self.cancel().success)
        self.assertTrue(self.node.navigation_suspended)
        self.assertIsNone(self.node.pending_path)
        self.assertIsNone(self.node.latest_path_key)
        self.assertIsNone(self.node.active_path)
        self.node._send_pending_path()
        self.node.client.send_goal_async.assert_not_called()

    def test_active_stop_cancels_once_and_result_never_restarts_queued_path(self):
        goal = Mock()
        self.node.active_goal = goal
        self.node.pending_path = make_path(1.0)

        self.assertFalse(self.cancel().success)
        self.assertFalse(self.cancel().success)
        goal.cancel_goal_async.assert_called_once_with()
        self.node.on_result(result_future())
        self.assertIsNone(self.node.active_goal)
        self.assertTrue(self.cancel().success)
        self.node.client.send_goal_async.assert_not_called()

    def test_stop_while_goal_request_in_flight_cancels_after_acceptance(self):
        self.node.goal_request_pending = True
        self.node.pending_path = make_path(1.0)
        self.cancel()
        self.assertTrue(self.node.cancel_when_accepted)

        goal = Mock(accepted=True)
        self.node.on_goal_response(Mock(result=Mock(return_value=goal)))
        goal.cancel_goal_async.assert_called_once_with()
        self.assertFalse(self.node.goal_request_pending)
        goal.get_result_async.return_value.add_done_callback.assert_called_once()
        self.node.on_result(result_future())
        self.node.client.send_goal_async.assert_not_called()

    def test_obstacle_replans_stay_suppressed_until_explicit_goal(self):
        path = make_path(stamp=7)
        self.cancel()
        self.node.status_pub.publish.reset_mock()
        self.node.on_path(path)
        self.node.on_path(make_path(1.0))
        self.node.on_path(Path())
        self.node._send_path(path)
        self.node.client.send_goal_async.assert_not_called()
        self.node.status_pub.publish.assert_not_called()

        self.node.on_goal(goal_for(path, stamp=6))
        self.node.on_path(path)
        self.assertFalse(self.node.navigation_suspended)
        self.node.client.send_goal_async.assert_called_once()

    def test_new_goal_before_old_acceptance_waits_for_old_cancellation(self):
        self.node.goal_request_pending = True
        self.cancel()
        new_path = make_path(1.0, stamp=7)
        self.node.on_goal(goal_for(new_path, stamp=6))
        self.node.on_path(new_path)
        self.node.client.send_goal_async.assert_not_called()

        old_goal = Mock(accepted=True)
        self.node.on_goal_response(Mock(result=Mock(return_value=old_goal)))
        old_goal.cancel_goal_async.assert_called_once_with()
        self.node.on_result(result_future())
        self.node.client.send_goal_async.assert_called_once()
        sent_goal = self.node.client.send_goal_async.call_args.args[0]
        self.assertEqual(sent_goal.path, new_path)

    def test_default_replan_still_replaces_active_route(self):
        old_goal = Mock()
        self.node.active_goal = old_goal
        self.node.active_path = make_path()
        new_path = make_path(1.0)
        self.node.on_path(new_path)

        old_goal.cancel_goal_async.assert_called_once_with()
        self.node.on_result(result_future())
        self.node.client.send_goal_async.assert_called_once()
        self.assertFalse(self.node.navigation_suspended)

    def test_new_path_before_goal_is_buffered_and_old_candidate_is_ignored(self):
        self.cancel()
        new_path = make_path(1.0, stamp=11)
        self.node.on_path(new_path)
        self.node.on_path(make_path(0.0, stamp=12))
        self.node.client.send_goal_async.assert_not_called()

        self.node.on_goal(goal_for(new_path, stamp=10))
        self.node.client.send_goal_async.assert_called_once()
        sent_goal = self.node.client.send_goal_async.call_args.args[0]
        self.assertEqual(sent_goal.path, new_path)

    def test_goal_before_path_rejects_old_endpoint_and_old_timestamp(self):
        self.cancel()
        new_path = make_path(1.0, stamp=11)
        self.node.on_goal(goal_for(new_path, stamp=10))
        self.node.on_path(make_path(0.0, stamp=12))
        self.node.on_path(make_path(1.0, stamp=9))
        self.node.client.send_goal_async.assert_not_called()

        self.node.on_path(new_path)
        self.node.client.send_goal_async.assert_called_once()

    def test_cancel_again_discards_buffered_paths_and_previous_explicit_goal(self):
        self.cancel()
        self.node.on_path(make_path(1.0, stamp=11))
        self.node.on_goal(goal_for(make_path(2.0), stamp=10))
        self.cancel()
        self.assertIsNone(self.node.requested_goal)
        self.assertFalse(self.node.suspended_paths)
        self.node.on_path(make_path(2.0, stamp=12))
        self.node.client.send_goal_async.assert_not_called()

    def test_delayed_goal_and_path_cannot_resume_after_stop(self):
        self.node.clock_seconds = 20
        self.cancel()
        old_path = make_path(stamp=11)
        self.node.on_goal(goal_for(old_path, stamp=10))
        self.node.on_path(old_path)
        self.assertTrue(self.node.navigation_suspended)
        self.node.client.send_goal_async.assert_not_called()
        self.node.on_goal(goal_for(make_path(stamp=22), stamp=21))
        self.node.on_path(make_path(stamp=22))
        self.assertFalse(self.node.navigation_suspended)
        self.node.client.send_goal_async.assert_called_once()

    def test_unstamped_goal_is_rejected_after_stop(self):
        self.cancel()
        self.node.on_goal(goal_for(make_path()))
        self.node.on_path(make_path())
        self.assertTrue(self.node.navigation_suspended)
        self.node.client.send_goal_async.assert_not_called()
        payload = json.loads(self.node.status_pub.publish.call_args.args[0].data)
        self.assertEqual(payload["status"], "IGNORED_GOAL")

    def test_direct_paths_still_work_before_first_cancel(self):
        self.node.on_path(make_path())
        self.node.client.send_goal_async.assert_called_once()

    def test_followup_goal_consumes_early_path_after_previous_run_completed(self):
        self.cancel()
        first_path = make_path(1.0, stamp=11)
        self.node.on_goal(goal_for(first_path, stamp=10))
        self.node.on_path(first_path)
        accepted = Mock(accepted=True)
        self.node.on_goal_response(Mock(result=Mock(return_value=accepted)))
        self.node.on_result(result_future(GoalStatus.STATUS_SUCCEEDED))
        self.node.client.send_goal_async.reset_mock()

        next_path = make_path(2.0, stamp=21)
        self.node.on_path(next_path)
        self.node.client.send_goal_async.assert_not_called()
        self.node.on_goal(goal_for(next_path, stamp=20))
        self.node.client.send_goal_async.assert_called_once()


class FollowerRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.node = FollowerHarness()

    def start_goal(self, path=None, stamp=10):
        path = path or make_path(stamp=stamp + 1)
        goal = goal_for(path, stamp=stamp)
        self.node.on_goal(goal)
        self.node.on_path(path)
        context = self.node.request_context
        handle = Mock(accepted=True)
        self.node.on_goal_response(Mock(result=Mock(return_value=handle)), context)
        return path, goal, context

    def statuses(self):
        return [json.loads(call.args[0].data) for call in self.node.status_pub.publish.call_args_list]

    def test_aborted_route_can_be_sent_again_unchanged(self):
        path, goal, context = self.start_goal()
        self.node.on_result(result_future(GoalStatus.STATUS_ABORTED), context)
        self.assertIsNone(self.node.latest_path_key)
        self.node.on_path(path)
        self.assertEqual(self.node.client.send_goal_async.call_count, 2)
        failed = [s for s in self.statuses() if s["status"] == "FAILED"][-1]
        self.assertEqual(failed["goal"], {"x": 2.0, "y": 0.0, "frame_id": "map"})
        self.assertEqual(failed["goal_stamp"], {"sec": 10, "nanosec": 0})

    def test_explicit_identical_goal_restarts_after_success(self):
        path, goal, context = self.start_goal()
        self.node.on_result(result_future(GoalStatus.STATUS_SUCCEEDED), context)
        self.node.on_goal(goal_for(path, stamp=12))
        self.node.on_path(make_path(stamp=13))
        self.assertEqual(self.node.client.send_goal_async.call_count, 2)

    def test_old_succeeded_result_does_not_complete_new_goal(self):
        path, goal, old_context = self.start_goal()
        self.node.on_goal(goal_for(make_path(1.0), stamp=20))
        self.node.on_result(result_future(GoalStatus.STATUS_SUCCEEDED), old_context)
        self.assertNotIn("SUCCEEDED", [s["status"] for s in self.statuses()])

    def test_same_xy_new_heading_replaces_active_route(self):
        path, goal, old_context = self.start_goal()
        new_path = make_path(stamp=21)
        new_path.poses[-1].pose.orientation.z = 1.0
        new_path.poses[-1].pose.orientation.w = 0.0
        self.node.on_goal(goal_for(new_path, stamp=20))
        self.node.on_path(new_path)
        self.assertIs(self.node.pending_path, new_path)
        self.node.on_result(result_future(), old_context)
        self.assertEqual(self.node.client.send_goal_async.call_count, 2)

    def test_repeated_empty_paths_wait_without_status_or_cancel_spam(self):
        path, goal, context = self.start_goal()
        empty = Path()
        empty.header.frame_id, empty.header.stamp.sec = "map", 11
        self.node.on_path(empty)
        self.node.on_path(empty)
        self.node.on_result(result_future(), context)
        self.assertEqual([s["status"] for s in self.statuses()].count("WAIT_REPLAN"), 1)
        self.assertNotIn("CANCELED", [s["status"] for s in self.statuses()])
        self.node.on_path(path)
        self.assertEqual(self.node.client.send_goal_async.call_count, 2)

    def test_manual_cancel_status_retains_stopped_goal_correlation(self):
        self.start_goal()
        self.node.cancel_navigation(Trigger.Request(), Trigger.Response())
        stopped = self.statuses()[-1]
        self.assertTrue(stopped["manual_stop"])
        self.assertEqual(stopped["goal_stamp"], {"sec": 10, "nanosec": 0})

    def test_rejected_path_can_be_sent_again(self):
        path = make_path()
        self.node.on_path(path)
        self.node.on_goal_response(Mock(result=Mock(return_value=Mock(accepted=False))))
        self.node.on_path(path)
        self.assertEqual(self.node.client.send_goal_async.call_count, 2)


if __name__ == "__main__":
    unittest.main()
