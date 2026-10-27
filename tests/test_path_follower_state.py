"""Exercise action state transitions without creating a ROS node or ROS graph.

Run with ROS Humble sourced and ros2_ws/src/smart_vision_amr on PYTHONPATH.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Trigger

from smart_vision_amr.path_follower import PathFollower


def path(*points):
    message = Path()
    message.header.frame_id = "map"
    for x, y in points:
        pose = PoseStamped()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        message.poses.append(pose)
    return message


def future(value):
    return SimpleNamespace(result=lambda: value)


def goal_handle(accepted=True):
    handle = Mock()
    handle.accepted = accepted
    return handle


class FollowerHarness(PathFollower):
    """Use the production callbacks with fake I/O, without Node.__init__."""

    def __init__(self):
        self.active_goal = None
        self.goal_request_pending = False
        self.pending_path = None
        self.pending_path_ready_at = None
        self.cancel_requested = False
        self.cancel_when_accepted = False
        self.failure_reported_for_cancel = False
        self.user_cancel_requested = False
        self.latest_path_key = None
        self.active_path = None
        self.safety_fault_reason = None
        self.statuses = []
        self.sent = []
        self.log = Mock()

    def get_logger(self):
        return self.log

    def get_parameter(self, name):
        assert name == "path_change_stability_sec"
        return SimpleNamespace(value=0.6)

    def publish_status(self, status, message_text):
        self.statuses.append(status)

    def _send_path(self, message):
        self.sent.append(message)
        self.active_path = message
        self.goal_request_pending = True


class PathFollowerStateTests(unittest.TestCase):
    def setUp(self):
        clock = patch("smart_vision_amr.path_follower.time.monotonic", return_value=100.0)
        self.clock = clock.start()
        self.addCleanup(clock.stop)
        self.follower = FollowerHarness()
        self.original = path((0, 0), (1, 0), (2, 0))
        self.detour = path((0, 0), (1, 0.5), (2, 0))

    def start_active(self):
        self.follower.on_path(self.original)
        handle = goal_handle()
        self.follower.on_goal_response(future(handle))
        return handle

    def cancel(self):
        response = self.follower.cancel_navigation(Trigger.Request(), Trigger.Response())
        self.assertTrue(response.success)

    def finish(self, status):
        self.follower.on_result(future(SimpleNamespace(status=status)))

    def test_rejected_goal_allows_identical_path_retry(self):
        self.follower.on_path(self.original)
        self.follower.on_goal_response(future(goal_handle(accepted=False)))
        self.assertEqual(self.follower.statuses, ["FAILED"])
        self.assertIsNone(self.follower.latest_path_key)
        self.follower.on_path(self.original)
        self.assertEqual(len(self.follower.sent), 2)
        self.assertTrue(self.follower.goal_request_pending)

    def test_idle_cancel_clears_retained_target_and_pending_path(self):
        self.follower.pending_path = self.detour
        self.follower.pending_path_ready_at = 100.0
        self.follower.latest_path_key = self.follower._path_key(self.detour)
        self.cancel()
        self.follower._process_pending_path()
        self.assertEqual(self.follower.statuses, ["CANCELED"])
        self.assertEqual(self.follower.sent, [])
        self.assertIsNone(self.follower.pending_path)
        self.assertIsNone(self.follower.pending_path_ready_at)
        self.assertIsNone(self.follower.latest_path_key)

    def test_cancel_during_acceptance_wait_cancels_late_accepted_goal(self):
        self.follower.on_path(self.original)
        self.cancel()
        handle = goal_handle()
        self.follower.on_goal_response(future(handle))
        handle.cancel_goal_async.assert_called_once()
        self.finish(GoalStatus.STATUS_CANCELED)
        self.assertEqual(self.follower.statuses, ["CANCELED"])
        self.assertIsNone(self.follower.active_goal)
        self.assertFalse(self.follower.goal_request_pending)

    def test_cancel_during_acceptance_wait_suppresses_late_rejection(self):
        self.follower.on_path(self.original)
        self.cancel()
        self.follower.on_goal_response(future(goal_handle(accepted=False)))
        self.assertEqual(self.follower.statuses, ["CANCELED"])
        self.assertFalse(self.follower.user_cancel_requested)
        self.assertFalse(self.follower.cancel_when_accepted)

    def test_active_cancel_discards_queued_reroute(self):
        handle = self.start_active()
        self.follower.on_path(self.detour)
        self.cancel()
        handle.cancel_goal_async.assert_called_once()
        self.finish(GoalStatus.STATUS_CANCELED)
        self.clock.return_value = 101.0
        self.follower._process_pending_path()
        self.assertEqual(self.follower.statuses, ["CANCELED"])
        self.assertEqual(self.follower.sent, [self.original])

    def test_late_result_after_user_cancel_does_not_overwrite_new_target(self):
        for terminal_status in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_SUCCEEDED,
                                GoalStatus.STATUS_ABORTED):
            with self.subTest(terminal_status=terminal_status):
                self.follower = FollowerHarness()
                self.start_active()
                self.cancel()
                replacement = path((0, 0), (0, 1), (0, 2))
                self.follower.on_path(replacement)
                self.assertEqual(self.follower.sent, [self.original])
                self.finish(terminal_status)
                self.assertEqual(self.follower.statuses, ["CANCELED"])
                self.assertEqual(self.follower.sent, [self.original, replacement])
                self.assertEqual(self.follower.latest_path_key,
                                 self.follower._path_key(replacement))

    def test_rejected_canceled_request_preserves_new_target(self):
        self.follower.on_path(self.original)
        self.cancel()
        replacement = path((0, 0), (0, 2))
        self.follower.on_path(replacement)
        self.follower.on_goal_response(future(goal_handle(accepted=False)))
        self.assertEqual(self.follower.statuses, ["CANCELED"])
        self.assertEqual(self.follower.sent, [self.original, replacement])
        self.assertEqual(self.follower.latest_path_key,
                         self.follower._path_key(replacement))

    def test_equivalent_route_after_committed_reroute_cancel_is_restarted(self):
        handle = self.start_active()
        self.follower.on_path(self.detour)
        self.clock.return_value = 100.7
        self.follower._process_pending_path()
        handle.cancel_goal_async.assert_called_once()
        self.follower.on_path(self.original)
        self.assertIs(self.follower.pending_path, self.original)
        self.finish(GoalStatus.STATUS_CANCELED)
        self.assertEqual(self.follower.statuses, [])
        self.assertEqual(self.follower.sent, [self.original, self.original])

    def test_equivalent_route_after_cancel_on_acceptance_is_restarted(self):
        self.follower.on_path(self.original)
        self.follower.on_path(self.detour)
        self.clock.return_value = 100.7
        self.follower._process_pending_path()
        self.assertTrue(self.follower.cancel_when_accepted)
        self.follower.on_path(self.original)
        handle = goal_handle()
        self.follower.on_goal_response(future(handle))
        handle.cancel_goal_async.assert_called_once()
        self.finish(GoalStatus.STATUS_CANCELED)
        self.assertEqual(self.follower.statuses, [])
        self.assertEqual(self.follower.sent, [self.original, self.original])

    def test_equivalent_route_before_debounce_keeps_active_goal(self):
        handle = self.start_active()
        self.follower.on_path(self.detour)
        self.clock.return_value = 100.2
        self.follower.on_path(self.original)
        self.clock.return_value = 101.0
        self.follower._process_pending_path()
        handle.cancel_goal_async.assert_not_called()
        self.assertIsNone(self.follower.pending_path)
        self.assertEqual(self.follower.sent, [self.original])

    def test_safety_cancel_still_reports_failure_and_allows_retry(self):
        handle = self.start_active()
        self.follower._navigation_input_error = Mock(return_value="odom is stale")
        self.follower._watch_navigation_inputs()
        handle.cancel_goal_async.assert_called_once()
        self.finish(GoalStatus.STATUS_CANCELED)
        self.assertEqual(self.follower.statuses, ["FAILED"])
        self.assertIsNone(self.follower.latest_path_key)


if __name__ == "__main__":
    unittest.main()
