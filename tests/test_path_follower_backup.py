"""Exercise bounded Nav2 BackUp recovery without moving a robot."""

import json
import math
import time
import unittest
from unittest.mock import Mock

from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger

from test_path_follower_cancel import (
    FollowerHarness, goal_for, make_path, result_future,
)


class FollowerBackupTest(unittest.TestCase):
    def setUp(self):
        self.node = FollowerHarness()
        self.node.clock_seconds = 12
        path = make_path(stamp=11)
        self.node.on_goal(goal_for(path, stamp=10))
        self.node.on_path(path)
        self.follow_context = self.node.request_context
        self.follow_handle = Mock(accepted=True)
        self.node.on_goal_response(
            Mock(result=Mock(return_value=self.follow_handle)),
            self.follow_context,
        )
        self.node.status_pub.publish.reset_mock()

    def statuses(self):
        return [
            json.loads(call.args[0].data)["status"]
            for call in self.node.status_pub.publish.call_args_list
        ]

    def feed_sensors(self, rear_range=2.0, scan_age=0, odom_age=0, mode="auto"):
        scan = LaserScan()
        scan.header.frame_id = "base_scan"
        scan.header.stamp.sec = self.node.clock_seconds - scan_age
        scan.angle_min = -math.pi
        scan.angle_increment = 2 * math.pi / 359
        scan.range_min = 0.12
        scan.range_max = 3.5
        scan.ranges = [rear_range] * 360
        self.node.on_scan(scan)
        odom = Odometry()
        odom.header.stamp.sec = self.node.clock_seconds - odom_age
        self.node.on_odom(odom)
        self.node.on_motion_status(
            String(data=json.dumps({
                "mode": mode, "available": True, "estop_latched": False,
            }))
        )

    def request(self, goal_id="10:0", distance=0.2, speed=0.04):
        self.node.on_recovery_request(
            String(data=json.dumps({
                "goal_id": goal_id, "distance": distance,
                "speed": speed, "reason": "inflated start",
            }))
        )

    def finish_follow_cancel(self):
        self.node.on_result(
            result_future(GoalStatus.STATUS_CANCELED), self.follow_context
        )

    def accept_backup(self):
        handle = Mock(accepted=True)
        sequence = self.node.recovery_request["seq"]
        self.node._on_backup_goal_response(
            Mock(result=Mock(return_value=handle)), sequence
        )
        return handle, sequence

    def test_recovery_waits_for_follow_path_to_finish(self):
        self.feed_sensors()
        self.request()
        self.follow_handle.cancel_goal_async.assert_called_once_with()
        self.node.backup_client.send_goal_async.assert_not_called()
        self.finish_follow_cancel()
        self.node.backup_client.send_goal_async.assert_called_once()
        goal = self.node.backup_client.send_goal_async.call_args.args[0]
        self.assertAlmostEqual(goal.target.x, -0.2)
        self.assertAlmostEqual(goal.speed, 0.04)
        self.assertGreater(goal.time_allowance.sec, 0)

    def test_wrong_goal_or_large_reverse_is_rejected(self):
        self.feed_sensors()
        self.request(goal_id="9:0")
        self.request(distance=1.0)
        self.request(speed=0.3)
        self.assertIsNone(self.node.recovery_request)
        self.node.backup_client.send_goal_async.assert_not_called()

    def test_stale_or_blocked_rear_lidar_prevents_reverse(self):
        self.feed_sensors(scan_age=3)
        self.request()
        self.finish_follow_cancel()
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.node.backup_client.send_goal_async.assert_not_called()
        self.feed_sensors(rear_range=0.25)
        self.request()
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.node.backup_client.send_goal_async.assert_not_called()

    def test_missing_rear_scan_coverage_is_not_treated_as_clear(self):
        self.feed_sensors()
        for index in range(len(self.node.latest_scan.ranges)):
            self.node.latest_scan.ranges[index] = float("nan")
        self.request()
        self.finish_follow_cancel()
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.node.backup_client.send_goal_async.assert_not_called()

    def test_narrow_invalid_gap_directly_behind_blocks_reverse(self):
        self.feed_sensors()
        scan = self.node.latest_scan
        for index in range(len(scan.ranges)):
            angle = scan.angle_min + index * scan.angle_increment
            rear_error = math.atan2(math.sin(angle - math.pi), math.cos(angle - math.pi))
            if abs(rear_error) < 0.03:
                scan.ranges[index] = float("nan")
        self.request()
        self.finish_follow_cancel()
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.node.backup_client.send_goal_async.assert_not_called()

    def test_invalid_lidar_geometry_fails_closed_without_crashing(self):
        self.feed_sensors()
        self.node.latest_scan.angle_min = float("nan")
        self.request()
        self.finish_follow_cancel()
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.node.backup_client.send_goal_async.assert_not_called()

    def test_request_arriving_before_matching_goal_is_buffered_without_motion(self):
        self.node.clock_seconds = 14
        self.node.on_recovery_request(String(data=json.dumps({
            "goal_id": "13:0", "distance": 0.2, "speed": 0.04,
        })))
        self.assertIsNone(self.node.recovery_request)
        self.assertIsNotNone(self.node.pending_recovery_request)
        self.node.backup_client.send_goal_async.assert_not_called()

        next_path = make_path(y=1.0, stamp=14)
        self.node.on_goal(goal_for(next_path, stamp=13))
        self.assertIsNone(self.node.pending_recovery_request)
        self.assertIsNotNone(self.node.recovery_request)
        self.feed_sensors()
        self.finish_follow_cancel()
        self.node.backup_client.send_goal_async.assert_called_once()

    def test_buffered_request_is_dropped_on_mismatched_new_goal_or_cancel(self):
        self.node.on_recovery_request(String(data=json.dumps({
            "goal_id": "13:0", "distance": 0.2, "speed": 0.04,
        })))
        newer_path = make_path(y=1.0, stamp=15)
        self.node.on_goal(goal_for(newer_path, stamp=14))
        self.assertIsNone(self.node.pending_recovery_request)
        self.assertIsNone(self.node.recovery_request)
        self.node.backup_client.send_goal_async.assert_not_called()

        self.node.on_recovery_request(String(data=json.dumps({
            "goal_id": "16:0", "distance": 0.2, "speed": 0.04,
        })))
        self.node.cancel_navigation(Trigger.Request(), Trigger.Response())
        self.assertIsNone(self.node.pending_recovery_request)
        self.node.backup_client.send_goal_async.assert_not_called()

    def test_success_replans_same_goal_without_reusing_old_path(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, sequence = self.accept_backup()
        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_SUCCEEDED), sequence, handle
        )
        self.assertIn("RECOVERY_SUCCEEDED", self.statuses())
        self.assertIsNone(self.node.recovery_request)
        self.assertIsNotNone(self.node.requested_goal)
        self.assertEqual(self.node.client.send_goal_async.call_count, 1)

    def test_manual_cancel_waits_for_backup_result(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, sequence = self.accept_backup()
        response = self.node.cancel_navigation(Trigger.Request(), Trigger.Response())
        self.assertFalse(response.success)
        handle.cancel_goal_async.assert_called_once_with()
        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_SUCCEEDED), sequence, handle
        )
        self.assertNotIn("RECOVERY_SUCCEEDED", self.statuses())
        self.assertTrue(
            self.node.cancel_navigation(Trigger.Request(), Trigger.Response()).success
        )

    def test_new_goal_waits_for_backup_cancellation(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, sequence = self.accept_backup()
        new_path = make_path(y=1.0, stamp=21)
        self.node.on_goal(goal_for(new_path, stamp=20))
        self.node.on_path(new_path)
        handle.cancel_goal_async.assert_called_once_with()
        self.assertEqual(self.node.client.send_goal_async.call_count, 1)
        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_CANCELED), sequence, handle
        )
        self.assertEqual(self.node.client.send_goal_async.call_count, 1)
        self.assertIsNone(self.node.pending_path)
        self.assertIn("RECOVERY_STOPPED", self.statuses())
        self.node.on_path(make_path(y=1.0, stamp=22))
        self.assertEqual(self.node.client.send_goal_async.call_count, 2)
        self.assertNotIn("RECOVERY_SUCCEEDED", self.statuses())

    def test_sensor_timeout_during_backup_cancels_action(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, sequence = self.accept_backup()
        self.node.latest_scan_at = time.monotonic() - 5
        self.node._monitor_recovery()
        handle.cancel_goal_async.assert_called_once_with()
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_CANCELED), sequence, handle
        )
        self.assertNotIn("RECOVERY_SUCCEEDED", self.statuses())

    def test_unconfirmed_cancel_after_sensor_fault_retries_and_latches_stop(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, sequence = self.accept_backup()
        self.node.latest_scan_at = time.monotonic() - 5
        self.node._monitor_recovery()
        self.assertIsNone(self.node.recovery_request)
        handle.cancel_goal_async.assert_called_once_with()
        self.node.safety_control_pub.publish.assert_not_called()

        self.node.backup_cancel_pending_since = time.monotonic() - 1.1
        self.node.backup_last_cancel_at = time.monotonic() - 0.6
        self.node._monitor_recovery()
        self.assertEqual(handle.cancel_goal_async.call_count, 2)
        packet = json.loads(self.node.safety_control_pub.publish.call_args.args[0].data)
        self.assertEqual({key: value for key, value in packet.items() if key != "seq"}, {
            "source_id": "path_follower_recovery_safety",
            "sent_at_ns": 12_000_000_000, "mode": "estop",
            "linear": 0.0, "angular": 0.0,
        })
        self.assertGreater(packet["seq"], 0)
        self.assertTrue(self.node.navigation_suspended)
        self.assertIsNone(self.node.requested_goal)
        self.assertIs(handle, self.node.backup_goal)
        self.node._monitor_recovery()
        self.assertEqual(self.node.safety_control_pub.publish.call_count, 1)

        self.node.backup_last_estop_at = time.monotonic() - 0.6
        self.node._monitor_recovery()
        self.assertEqual(self.node.safety_control_pub.publish.call_count, 2)
        repeated_packet = json.loads(
            self.node.safety_control_pub.publish.call_args.args[0].data
        )
        self.assertGreater(repeated_packet["seq"], packet["seq"])
        self.node.on_motion_status(String(data=json.dumps({
            "mode": "estop", "available": True, "estop_latched": True,
        })))
        self.node.backup_last_estop_at = time.monotonic() - 0.6
        self.node._monitor_recovery()
        self.assertEqual(self.node.safety_control_pub.publish.call_count, 2)

        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_CANCELED), sequence, handle
        )
        self.assertIsNone(self.node.backup_goal)
        self.node.client.send_goal_async.assert_called_once()

    def test_confirmed_cancel_before_watchdog_does_not_latch_stop(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, sequence = self.accept_backup()
        self.node.latest_scan_at = time.monotonic() - 5
        self.node._monitor_recovery()
        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_CANCELED), sequence, handle
        )
        self.node._monitor_recovery()
        self.node.safety_control_pub.publish.assert_not_called()
        self.assertIn("RECOVERY_STOPPED", self.statuses())

    def test_replan_waits_for_confirmed_backup_cancel_then_requests_fresh_path(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, sequence = self.accept_backup()
        self.node.latest_scan_at = time.monotonic() - 5
        self.node._monitor_recovery()
        later_path = make_path(y=1.0, stamp=13)
        self.node.on_goal(goal_for(later_path, stamp=12))
        self.node.on_path(later_path)
        self.assertIs(self.node.pending_path, later_path)
        self.assertNotIn("RECOVERY_STOPPED", self.statuses())

        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_CANCELED), sequence, handle
        )
        self.assertIsNone(self.node.pending_path)
        self.assertEqual(self.node.client.send_goal_async.call_count, 1)
        stopped = [json.loads(call.args[0].data) for call in
                   self.node.status_pub.publish.call_args_list
                   if json.loads(call.args[0].data)["status"] == "RECOVERY_STOPPED"]
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0]["goal_stamp"], {"sec": 12, "nanosec": 0})

    def test_manual_override_does_not_take_away_manual_control(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, sequence = self.accept_backup()
        self.node.on_motion_status(
            String(data=json.dumps({"mode": "manual", "available": True}))
        )
        self.node.backup_cancel_pending_since = time.monotonic() - 2
        self.node.backup_last_cancel_at = time.monotonic() - 1
        self.node._monitor_recovery()
        self.assertEqual(handle.cancel_goal_async.call_count, 2)
        self.node.safety_control_pub.publish.assert_not_called()
        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_CANCELED), sequence, handle
        )

    def test_explicit_same_goal_cancel_stops_backup_and_rejects_old_path(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, sequence = self.accept_backup()
        self.node.on_recovery_request(String(data=json.dumps({
            "goal_id": "10:0", "cancel": True,
            "reason": "Map changed during reverse recovery",
        })))
        handle.cancel_goal_async.assert_called_once_with()
        self.assertIsNone(self.node.recovery_request)
        self.assertIsNotNone(self.node.requested_goal)
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.node.on_path(make_path(stamp=11))
        self.assertIsNone(self.node.pending_path)
        self.assertEqual(self.node.client.send_goal_async.call_count, 1)
        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_CANCELED), sequence, handle
        )
        self.assertEqual(self.node.client.send_goal_async.call_count, 1)

    def test_explicit_cancel_before_backup_acceptance_cancels_late_handle(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        sequence = self.node.recovery_request["seq"]
        self.node.on_recovery_request(String(data=json.dumps({
            "goal_id": "10:0", "cancel": True,
        })))
        self.assertTrue(self.node.backup_goal_request_pending)
        handle = Mock(accepted=True)
        self.node._on_backup_goal_response(
            Mock(result=Mock(return_value=handle)), sequence
        )
        handle.cancel_goal_async.assert_called_once_with()
        self.assertIs(handle, self.node.backup_goal)
        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_CANCELED), sequence, handle
        )
        self.assertIsNone(self.node.backup_goal)

    def test_rejected_backup_after_safety_cancel_replans_from_current_pose(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        sequence = self.node.recovery_request["seq"]
        self.node.latest_scan_at = time.monotonic() - 5
        self.node._monitor_recovery()
        later_path = make_path(stamp=13)
        self.node.on_path(later_path)
        self.assertIs(self.node.pending_path, later_path)
        self.node._on_backup_goal_response(
            Mock(result=Mock(return_value=Mock(accepted=False))), sequence
        )
        self.assertIsNone(self.node.pending_path)
        self.assertIn("RECOVERY_STOPPED", self.statuses())
        self.assertEqual(self.node.client.send_goal_async.call_count, 1)

    def test_explicit_cancel_ignores_other_goal(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, _ = self.accept_backup()
        self.node.on_recovery_request(String(data=json.dumps({
            "goal_id": "9:0", "cancel": True,
        })))
        self.assertIsNotNone(self.node.recovery_request)
        handle.cancel_goal_async.assert_not_called()

    def test_cancel_before_matching_goal_discards_buffered_reverse_request(self):
        self.node.clock_seconds = 14
        self.node.on_recovery_request(String(data=json.dumps({
            "goal_id": "13:0", "distance": 0.2, "speed": 0.04,
        })))
        self.assertIsNotNone(self.node.pending_recovery_request)
        self.node.on_recovery_request(String(data=json.dumps({
            "goal_id": "13:0", "cancel": True,
        })))
        self.assertIsNone(self.node.pending_recovery_request)
        self.assertIsNotNone(self.node.pending_recovery_cancel)

        next_path = make_path(y=1.0, stamp=14)
        self.node.on_goal(goal_for(next_path, stamp=13))
        self.assertIsNone(self.node.recovery_request)
        self.assertTrue(self.node.path_invalidated)
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.node.on_recovery_request(String(data=json.dumps({
            "goal_id": "13:0", "distance": 0.2, "speed": 0.04,
        })))
        self.assertIsNone(self.node.recovery_request)
        self.node.backup_client.send_goal_async.assert_not_called()

    def test_lost_backup_acceptance_never_dispatches_new_path(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        sequence = self.node.recovery_request["seq"]
        self.node._on_backup_goal_response(
            Mock(result=Mock(side_effect=RuntimeError("response lost"))), sequence
        )
        self.assertTrue(self.node.backup_goal_request_pending)
        self.assertTrue(self.node.navigation_suspended)
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.node.on_path(make_path(y=1.0, stamp=13))
        self.node._send_pending_path()
        self.assertEqual(self.node.client.send_goal_async.call_count, 1)
        self.node.backup_cancel_pending_since = time.monotonic() - 2
        self.node._monitor_recovery()
        self.node.safety_control_pub.publish.assert_called_once()

    def test_lost_backup_terminal_result_never_dispatches_new_path(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, sequence = self.accept_backup()
        self.node.on_path(make_path(y=1.0, stamp=13))
        self.node._on_backup_result(
            Mock(result=Mock(side_effect=RuntimeError("result lost"))),
            sequence, handle,
        )
        self.assertIs(self.node.backup_goal, handle)
        self.assertTrue(self.node.navigation_suspended)
        self.assertEqual(self.node.client.send_goal_async.call_count, 1)
        self.node.backup_cancel_pending_since = time.monotonic() - 2
        self.node._monitor_recovery()
        self.node.safety_control_pub.publish.assert_called_once()

    def test_unknown_backup_result_status_is_not_proof_of_stop(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, sequence = self.accept_backup()
        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_UNKNOWN), sequence, handle
        )
        self.assertIs(self.node.backup_goal, handle)
        self.assertTrue(self.node.navigation_suspended)
        self.assertNotIn("RECOVERY_SUCCEEDED", self.statuses())
        self.assertEqual(self.node.client.send_goal_async.call_count, 1)

    def test_manual_motion_mode_cancels_backup(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, _ = self.accept_backup()
        self.node.on_motion_status(
            String(data=json.dumps({"mode": "manual", "available": True}))
        )
        handle.cancel_goal_async.assert_called_once_with()
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.assertIsNotNone(self.node.requested_goal)

    def test_manual_override_before_backup_acceptance_cancels_when_accepted(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        sequence = self.node.recovery_request["seq"]
        self.node.on_motion_status(
            String(data=json.dumps({"mode": "estop", "available": True,
                                    "estop_latched": True}))
        )
        self.assertIn("RECOVERY_FAILED", self.statuses())
        handle = Mock(accepted=True)
        self.node._on_backup_goal_response(
            Mock(result=Mock(return_value=handle)), sequence
        )
        handle.cancel_goal_async.assert_called_once_with()
        self.node._on_backup_result(
            result_future(GoalStatus.STATUS_CANCELED), sequence, handle
        )
        self.assertNotIn("RECOVERY_SUCCEEDED", self.statuses())

    def test_missing_backup_server_reports_failure(self):
        self.feed_sensors()
        self.node.backup_client.wait_for_server.return_value = False
        self.request()
        self.finish_follow_cancel()
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.assertIsNone(self.node.recovery_request)

    def test_late_initial_empty_path_is_ignored_but_newer_empty_cancels(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, _ = self.accept_backup()
        initial_empty = make_path()
        initial_empty.poses.clear()
        initial_empty.header.stamp.sec = 12
        self.node.on_path(initial_empty)
        handle.cancel_goal_async.assert_not_called()
        later_empty = make_path()
        later_empty.poses.clear()
        later_empty.header.stamp.sec = 13
        self.node.on_path(later_empty)
        handle.cancel_goal_async.assert_called_once_with()
        self.assertIn("RECOVERY_FAILED", self.statuses())

    def test_delayed_old_nonempty_route_cannot_abort_reverse_recovery(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, _ = self.accept_backup()
        delayed = make_path(stamp=11)
        self.node.on_path(delayed)
        self.assertIsNotNone(self.node.recovery_request)
        handle.cancel_goal_async.assert_not_called()
        self.assertEqual(self.node.client.send_goal_async.call_count, 1)

    def test_cancellation_wait_is_bounded(self):
        self.feed_sensors()
        self.request()
        self.node.recovery_request["requested_at"] = time.monotonic() - 6
        self.node._monitor_recovery()
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.assertIsNone(self.node.recovery_request)
        self.node.backup_client.send_goal_async.assert_not_called()

    def test_busy_backup_reports_failure_for_matching_goal(self):
        self.feed_sensors()
        self.request()
        self.finish_follow_cancel()
        handle, _ = self.accept_backup()
        self.node._fail_recovery("simulated sensor loss")
        self.assertIsNone(self.node.recovery_request)
        self.node.status_pub.publish.reset_mock()
        self.request()
        self.assertIn("RECOVERY_FAILED", self.statuses())
        self.node.backup_client.send_goal_async.assert_called_once()
        handle.cancel_goal_async.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
