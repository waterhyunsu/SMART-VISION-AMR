"""Actual LaserScan projection and transient clearing without any actuator node."""
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

from smart_vision_amr.change_memory import ChangeMemoryNode


class ScanHarness(ChangeMemoryNode):
    def __init__(self):
        self.params = {'memory_enabled': False, 'map_frame': 'map', 'base_frame': 'base_footprint',
                       'odom_frame': 'odom',
                       'occupied_threshold': 65, 'confirm_observations': 5,
                       'clear_observations': 5, 'confirm_min_duration_sec': 2.,
                       'clear_min_duration_sec': 2., 'min_observation_interval_sec': .25,
                       'save_every_scans': 0,
                       'live_obstacle_ttl_sec': 2., 'live_scan_max_age_sec': 1.,
                       'live_transform_max_age_sec': 1.,
                       'mapped_wall_return_tolerance_cells': 1,
                       'live_localization_jump_distance': .25,
                       'live_localization_jump_angle': .25}
        self.map = self.memory = None
        self.memory_path = Path('/this-test-never-writes-memory.json')
        self.live_obstacles = {}
        self.last_scan_stamp_ns = None
        self.scans_since_save = 0
        self.observation_pub, self.overlay_pub, self.logger = Mock(), Mock(), Mock()
        self.tf_buffer = Mock()
        self.transform = TransformStamped()
        self.transform.header.stamp.sec = 100
        self.transform.transform.translation.x = .5
        self.transform.transform.translation.y = 1.5
        self.transform.transform.rotation.w = 1.
        self.map_to_odom = TransformStamped()
        self.map_to_odom.transform.rotation.w = 1.
        self.tf_buffer.lookup_transform.side_effect = (
            lambda target, source, when: self.map_to_odom if source == 'odom' else self.transform)
        self.ros_ns = 100_000_000_000
        self.monotonic = patch('smart_vision_amr.change_memory.time.monotonic', return_value=10.)
        self.clock = self.monotonic.start()
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.resolution = 1.
        msg.info.width = msg.info.height = 6
        msg.info.origin.orientation.w = 1.
        msg.data = [0]*36
        self.on_map(msg)

    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=self.ros_ns))

    def get_logger(self):
        return self.logger

    def scan(self, distance=2., angle=0., advance=.1):
        self.ros_ns += round(advance*1e9)
        msg = LaserScan()
        msg.header.frame_id = 'base_scan'
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(self.ros_ns, 1_000_000_000)
        self.transform.header.stamp = deepcopy(msg.header.stamp)
        msg.angle_min, msg.angle_increment = angle, .01
        msg.range_min, msg.range_max = .12, 4.
        msg.ranges = [float(distance)]
        self.on_scan(msg)
        return msg

    def observed(self):
        return set(self.observation_pub.publish.call_args.args[0].data)


def harness():
    """Keep patch cleanup local even when a regression assertion fails."""
    import contextlib

    @contextlib.contextmanager
    def managed():
        node = ScanHarness()
        try:
            yield node
        finally:
            node.monotonic.stop()
    return managed()


def test_live_obstacles_work_with_persistent_memory_disabled_and_never_mutate_map():
    with harness() as node:
        original = list(node.map['message'].data)
        node.scan()
        assert node.observed() == {8}  # Hit (2, 1), not present in the saved map.
        assert not node.memory.confirmed_cells
        assert list(node.map['message'].data) == original
        node.scan(math.inf)
        assert node.observed() == set()  # A valid free ray clears it immediately.
        assert list(node.map['message'].data) == original


def test_briefly_occluded_obstacle_survives_one_scan_then_expires_without_map_memory():
    with harness() as node:
        node.scan()
        node.clock.return_value = 10.2
        node.scan(1., math.pi/2)
        assert 8 in node.observed()
        node.clock.return_value = 12.1
        node.scan(1., math.pi/2)
        assert 8 not in node.observed()


def test_invalid_beams_do_not_clear_obstacles_or_heartbeat_the_planner():
    with harness() as node:
        node.scan()
        count = node.observation_pub.publish.call_count
        for value in (math.nan, -math.inf, 0., -.1, 5.):
            node.scan(value)
        assert node.observation_pub.publish.call_count == count
        assert node.observed() == {8}


def test_partially_invalid_scan_does_not_expire_an_obstacle_in_unknown_sector():
    with harness() as node:
        node.scan()
        node.clock.return_value = 15.
        scan = node.scan(math.nan)
        scan.angle_increment = math.pi/2
        scan.ranges = [math.nan, 1.]  # Forward unknown, left valid.
        node.on_scan(scan)
        assert 8 in node.observed()
        assert (2, 1) in node.live_obstacles


def test_physical_401_beam_scan_dropouts_protect_only_their_angular_sectors():
    with harness() as node:
        node.scan()  # An old obstacle directly ahead, cell (2, 1).
        node.live_obstacles[(2, 4)] = 10.  # Unrelated old return at about 56 degrees.
        node.live_obstacles[(-1, 1)] = 10.  # An invalid legacy index must always be removed.
        node.clock.return_value = 15.
        scan = node.scan(math.nan)
        scan.angle_min, scan.angle_increment = 0., 2*math.pi/400
        scan.ranges = [.6]*401  # Nearby hits occlude the older cells; no free-ray clearing.
        for index in (0, 100, 200):
            scan.ranges[index] = math.nan
        node.on_scan(scan)
        assert (2, 1) in node.live_obstacles  # The forward dropout is still unknown.
        assert (2, 4) not in node.live_obstacles  # Three dropouts cannot freeze the whole map.
        assert (-1, 1) not in node.live_obstacles


def test_relocalization_translation_jump_rebuilds_transient_cells_only():
    with harness() as node:
        node.scan()
        original_map = list(node.map['message'].data)
        persistent = node.memory
        node.map_to_odom.transform.translation.x = .5
        node.scan(1., math.pi/2)
        assert (2, 1) not in node.live_obstacles
        assert node.live_obstacles  # The fresh scan has already rebuilt its actual hits.
        assert node.memory is persistent
        assert list(node.map['message'].data) == original_map


def test_relocalization_yaw_jump_rebuilds_transient_cells():
    with harness() as node:
        node.scan()
        node.map_to_odom.transform.rotation.z = math.sin(.3/2)
        node.map_to_odom.transform.rotation.w = math.cos(.3/2)
        node.scan(1., math.pi/2)
        assert (2, 1) not in node.live_obstacles


def test_normal_robot_motion_does_not_count_as_a_localization_frame_jump():
    with harness() as node:
        node.scan()
        node.transform.transform.translation.y += .5
        node.scan(1., math.pi/2)
        assert (2, 1) in node.live_obstacles  # map->odom stayed fixed; the robot moved.


def test_initial_pose_invalidates_old_cells_without_announcing_fresh_empty_observations():
    with harness() as node:
        node.scan()
        count = node.observation_pub.publish.call_count
        stamp = node.last_scan_stamp_ns
        persistent = node.memory
        original_map = list(node.map['message'].data)
        message = PoseWithCovarianceStamped()
        message.header.frame_id = 'odom'
        node.on_initial_pose(message)
        assert node.live_obstacles
        message.header.frame_id = 'map'
        node.on_initial_pose(message)
        assert node.live_obstacles == {}
        assert node.last_scan_stamp_ns == stamp  # A duplicate scan cannot revive the old layer.
        assert node.observation_pub.publish.call_count == count
        assert node.memory is persistent
        assert list(node.map['message'].data) == original_map
        node.scan(1., math.pi/2)
        assert node.observation_pub.publish.call_count == count+1
        assert (2, 1) not in node.live_obstacles


def test_stale_scan_and_transform_do_not_replace_live_obstacles():
    with harness() as node:
        scan = node.scan()
        count = node.observation_pub.publish.call_count
        node.ros_ns += 2_000_000_000
        scan.header.stamp.nanosec += 1
        node.on_scan(scan)
        assert node.observation_pub.publish.call_count == count
        scan.header.stamp.sec, scan.header.stamp.nanosec = divmod(node.ros_ns, 1_000_000_000)
        node.on_scan(scan)  # Current scan, stale map-to-LiDAR transform.
        assert node.observation_pub.publish.call_count == count


def test_duplicate_scan_does_not_keep_obstacle_layer_alive():
    with harness() as node:
        scan = node.scan()
        node.on_scan(scan)
        assert node.observation_pub.publish.call_count == 1


def test_new_map_discards_live_cells_instead_of_applying_old_indices():
    with harness() as node:
        node.scan()
        msg = deepcopy(node.map['message'])
        msg.info.origin.position.x = 20.
        node.on_map(msg)
        assert node.live_obstacles == {}
        assert node.last_scan_stamp_ns is None


def test_scan_to_live_overlay_to_astar_detour_uses_same_goal_without_slam_map_write():
    from test_planner_recovery import PlannerHarness, make_goal
    with harness() as sensor:
        planner = PlannerHarness()
        planner.params['live_obstacle_layer_required'] = True
        planner.on_map(sensor.map['message'])
        goal = make_goal()
        with patch('smart_vision_amr.astar_planner.time.monotonic', return_value=100.) as clock:
            sensor.scan(math.inf)
            planner.on_observed(sensor.observation_pub.publish.call_args.args[0])
            planner.on_goal(goal)
            assert (2, 1) in planner.planned_cells
            clock.return_value = 100.1
            sensor.scan(2.)
            planner.on_observed(sensor.observation_pub.publish.call_args.args[0])
            assert not planner.path_pub.publish.call_args.args[0].poses
            clock.return_value = 102.1
            # Another sensor sample keeps the live layer fresh while the retry
            # throttle permits replacement of the canceled direct route.
            sensor.scan(2.)
            planner.on_observed(sensor.observation_pub.publish.call_args.args[0])
            assert planner.planned_cells
            assert (2, 1) not in planner.planned_cells
            assert planner.last_goal is goal
            assert all(value == 0 for value in sensor.map['message'].data)


def test_wall_echo_is_suppressed_but_nearer_wall_adjacent_object_is_live_and_remembered(tmp_path):
    with harness() as node:
        node.params['memory_enabled'] = True
        node.memory_path = tmp_path / 'memory.json'
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.resolution = .05
        msg.info.width = msg.info.height = 50
        msg.info.origin.orientation.w = 1.
        msg.data = [0] * (50 * 50)
        for index in range(50):
            msg.data[index] = msg.data[49 * 50 + index] = 100
            msg.data[index * 50] = msg.data[index * 50 + 49] = 100
        node.on_map(msg)
        node.transform.transform.translation.x = 43.5 * .05
        node.transform.transform.translation.y = 10.5 * .05

        wall_clock = patch('smart_vision_amr.change_memory.time.time', return_value=100.0)
        observed_at = wall_clock.start()
        try:
            # The wall begins at 0.275 m along this beam; a return 0.025 m early
            # agrees with the mapped wall within the one-cell range tolerance.
            node.scan(.25)
            assert node.observed() == set()
            assert (48, 25) not in node.memory.cells

            # The diagonal ray lands in cell (47, 25), only two cells from the
            # mapped wall. The return is much closer than the wall's predicted range.
            obstacle_angle = math.atan2(.75, .20)
            obstacle_range = math.hypot(.75, .20)
            for index in range(5):
                node.clock.return_value = 10.1 + index * .525
                observed_at.return_value = 100.1 + index * .525
                node.scan(obstacle_range, angle=obstacle_angle, advance=.525)
        finally:
            wall_clock.stop()
        assert node.observed() == {25 * 50 + 47}
        memory = node.memory.cells[(47, 25)]
        assert memory.occupied_observations == 5
        assert memory.state == 'confirmed'

        from test_planner_recovery import PlannerHarness
        planner = PlannerHarness()
        planner.params['obstacle_inflation_cells'] = 5
        planner.params['preferred_clearance'] = 0.0
        planner.params['clearance_penalty_gain'] = 0.0
        planner.on_map(msg)
        planner_tf = planner.tf_buffer.lookup_transform.return_value
        planner_tf.transform.translation.x = 43.5 * .05
        planner_tf.transform.translation.y = 10.5 * .05
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp.sec = 10
        goal.pose.position.x = 43.5 * .05
        goal.pose.position.y = 40.5 * .05
        goal.pose.orientation.w = 1.0

        with patch('smart_vision_amr.astar_planner.time.monotonic', return_value=100.0) as now:
            planner.on_goal(goal)
            assert (43, 25) in planner.planned_cells
            planner.on_observed(node.observation_pub.publish.call_args.args[0])
            assert not planner.planned_cells
            now.return_value = 102.1
            planner.retry_transform()
            assert planner.planned_cells
            assert all(cell[0] < 42 or not 20 <= cell[1] <= 30
                       for cell in planner.planned_cells)
            assert planner.last_goal is goal


def test_unknown_frontier_does_not_filter_a_nearby_live_obstacle():
    with harness() as node:
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.resolution = .05
        msg.info.width = msg.info.height = 50
        msg.info.origin.orientation.w = 1.
        msg.data = [0] * (50 * 50)
        # The sensor ray ends in known free cell (46, 10), 2.5 cm before
        # unknown cell (47, 10). Unknown space is not a mapped wall return.
        msg.data[10 * 50 + 47] = -1
        node.on_map(msg)
        node.transform.transform.translation.x = 43.5 * .05
        node.transform.transform.translation.y = 10.5 * .05

        node.scan(.15)

        assert node.observed() == {10 * 50 + 46}
