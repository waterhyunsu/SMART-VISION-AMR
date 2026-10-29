"""The follower must stop on stale navigation data and retry the retained goal."""

import json
import time
from unittest.mock import Mock

from geometry_msgs.msg import Point32, PolygonStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan

from test_path_follower_cancel import FollowerHarness, goal_for, make_path


def healthy_follower():
    follower = FollowerHarness()
    follower.navigation_inputs_required = True
    follower.latest_scan = LaserScan()
    follower.latest_scan.header.stamp.sec = 5
    follower.latest_scan.header.frame_id = "base_scan"
    follower.latest_odom = Odometry()
    follower.latest_odom.header.stamp.sec = 5
    follower.latest_odom.header.frame_id = "odom"
    follower.latest_footprint = PolygonStamped()
    follower.latest_footprint.header.stamp.sec = 5
    follower.latest_footprint.header.frame_id = "odom"
    for x, y in ((-.2, -.2), (.2, -.2), (.2, .2), (-.2, .2)):
        point = Point32()
        point.x, point.y = x, y
        follower.latest_footprint.polygon.points.append(point)
    follower.latest_costmap = OccupancyGrid()
    follower.latest_costmap.header.stamp.sec = 5
    follower.latest_costmap.header.frame_id = "odom"
    follower.latest_costmap.info.width = 60
    follower.latest_costmap.info.height = 60
    follower.latest_costmap.info.resolution = .05
    follower.latest_costmap.info.origin.position.x = -1.5
    follower.latest_costmap.info.origin.position.y = -1.5
    received = time.monotonic()
    follower.latest_scan_at = follower.latest_odom_at = received
    follower.latest_footprint_at = follower.latest_costmap_at = received
    return follower


def test_fresh_and_consistent_inputs_allow_navigation():
    follower = healthy_follower()
    assert follower._navigation_input_error() is None


def test_stale_or_missing_inputs_prevent_motion():
    follower = healthy_follower()
    follower.latest_scan.header.stamp.sec = 3
    assert "scan" in follower._navigation_input_error()
    follower.latest_scan.header.stamp.sec = 5
    follower.latest_footprint = None
    assert "footprint" in follower._navigation_input_error()


def test_disagreeing_footprint_prevents_motion():
    follower = healthy_follower()
    for point in follower.latest_footprint.polygon.points:
        point.x += .5
    assert "footprint" in follower._navigation_input_error()


def test_frozen_ros_clock_cannot_hide_stale_scan_receipt():
    follower = healthy_follower()
    follower.latest_scan_at = time.monotonic() - 2.0
    assert "scan" in follower._navigation_input_error()


def test_stale_input_cancels_active_action_and_fresh_input_can_retry():
    follower = healthy_follower()
    path = make_path(stamp=5)
    follower.requested_goal = goal_for(path, stamp=5)
    follower.active_path = path
    follower.active_goal = Mock()
    follower.latest_path_key = follower._path_key(path)
    follower.latest_scan.header.stamp.sec = 3

    follower._watch_navigation_inputs()

    follower.active_goal.cancel_goal_async.assert_called_once()
    assert follower.path_invalidated is True
    assert follower.latest_path_key is None
    status = json.loads(follower.status_pub.publish.call_args.args[0].data)
    assert status["status"] == "FAILED"
    assert status["goal"] == {"x": 2.0, "y": 0.0, "frame_id": "map"}

    follower.active_goal = None
    follower.cancel_requested = False
    follower.latest_scan.header.stamp.sec = 5
    follower._send_path(path)
    follower.client.send_goal_async.assert_called_once()


def test_manual_stop_is_never_auto_resumed_by_input_guard():
    follower = healthy_follower()
    follower.navigation_suspended = True
    follower.active_goal = Mock()
    follower.latest_scan.header.stamp.sec = 3

    follower._watch_navigation_inputs()

    follower.active_goal.cancel_goal_async.assert_not_called()


def test_path_arriving_before_goal_waits_for_goal_context_when_inputs_are_stale():
    follower = healthy_follower()
    path = make_path(stamp=5)
    follower.latest_scan.header.stamp.sec = 3

    follower.on_path(path)

    follower.status_pub.publish.assert_not_called()
    assert list(follower.suspended_paths) == [path]
    assert follower.require_goal_match is True
