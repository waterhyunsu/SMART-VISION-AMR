"""A* ROS adapter: OccupancyGrid + persistent cells + PoseStamped goal -> Path."""

import json
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Int32MultiArray, String
from tf2_ros import Buffer, TransformException, TransformListener
from collections import deque

from smart_vision_amr.astar import AStarPlanner
from smart_vision_amr.map_utils import cell_to_world, inflate_cells, occupancy_to_grid, world_to_cell


class AStarPlannerNode(Node):
    def __init__(self):
        super().__init__("astar_planner")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("occupied_threshold", 65)
        self.declare_parameter("obstacle_inflation_cells", 2)
        self.map = None
        self.raw_static_grid = None
        self.remembered = set()
        self.observed = set()
        self.last_goal = None
        self.planned_cells = []
        self.waiting_for_transform = False
        self.planner = AStarPlanner()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.path_pub = self.create_publisher(Path, "/smart_vision_amr/path", 10)
        self.create_subscription(OccupancyGrid, self.get_parameter("map_topic").value, self.on_map, latched)
        self.create_subscription(Int32MultiArray, "/smart_vision_amr/remembered_cells", self.on_memory, latched)
        self.create_subscription(Int32MultiArray, "/smart_vision_amr/observed_obstacles",
                                 self.on_observed, 10)
        self.create_subscription(PoseStamped, "/smart_vision_amr/goal", self.on_goal, 10)
        self.create_subscription(String, "/smart_vision_amr/navigation_status",
                                 self.on_navigation_status, 10)
        self.create_timer(1.0, self.retry_transform)

    def on_navigation_status(self, msg):
        try:
            status = json.loads(msg.data).get("status")
        except (ValueError, AttributeError):
            return
        if status == "FAILED":
            # Keep the active goal and last route so a later scan/memory update
            # can trigger a fresh plan after Nav2 aborts for lack of progress.
            self.get_logger().warning(
                "Navigation failed; keeping goal for obstacle-driven replanning")
            return
        if status in ("SUCCEEDED", "CANCELED"):
            self.last_goal = None
            self.planned_cells = []
            self.waiting_for_transform = False

    def on_map(self, msg):
        threshold = self.get_parameter("occupied_threshold").value
        inflation = self.get_parameter("obstacle_inflation_cells").value
        self.map = (msg, occupancy_to_grid(msg, threshold, inflation))
        self.raw_static_grid = occupancy_to_grid(msg, threshold, 0)
        if self.last_goal is not None:
            self.plan_to(self.last_goal)

    def retry_transform(self):
        if self.waiting_for_transform and self.last_goal is not None and self.map is not None:
            self.plan_to(self.last_goal)

    def on_memory(self, msg):
        if self.map is None:
            return
        width, height = self.map[0].info.width, self.map[0].info.height
        updated = {(index % width, index // width) for index in msg.data
                   if 0 <= index < width * height}
        if updated != self.remembered:
            added = updated - self.remembered
            removed = self.remembered - updated
            self.remembered = updated
            # Replan for either transition only when it changes the current
            # route. Unrelated memory churn must not cancel a valid Nav2 goal.
            if self.last_goal is not None and (self._changes_block_path(added)
                                               or self._changes_block_path(removed)):
                self.plan_to(self.last_goal)

    def on_observed(self, msg):
        if self.map is None:
            return
        width, height = self.map[0].info.width, self.map[0].info.height
        updated = {(index % width, index // width) for index in msg.data
                   if 0 <= index < width * height}
        if updated != self.observed:
            added = updated - self.observed
            self.observed = updated
            # Laser scans vary at sensor frequency. Replan only when newly
            # observed cells actually intersect the current A* route after inflation.
            if self.last_goal is not None and self._changes_block_path(added):
                self.plan_to(self.last_goal)

    def _changes_block_path(self, cells):
        if not cells or not self.planned_cells:
            return False
        width, height = self.map[0].info.width, self.map[0].info.height
        inflation = self.get_parameter("obstacle_inflation_cells").value
        blocked = inflate_cells(cells, width, height, inflation)
        return any(cell in blocked for cell in self.planned_cells)

    def on_goal(self, goal):
        self.last_goal = goal
        self.plan_to(goal)

    def plan_to(self, goal):
        if self.map is None:
            self.get_logger().warning("Cannot plan before receiving /map")
            self.waiting_for_transform = True
            return
        map_msg, grid = self.map
        frame = self.get_parameter("map_frame").value
        if goal.header.frame_id and goal.header.frame_id != frame:
            self.get_logger().error(f"Goal frame must be {frame}, got {goal.header.frame_id}")
            return
        try:
            tf = self.tf_buffer.lookup_transform(frame, self.get_parameter("base_frame").value,
                                                 rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warning(f"Cannot plan without robot pose: {exc}")
            self.waiting_for_transform = True
            return
        self.waiting_for_transform = False
        start = world_to_cell(map_msg.info, tf.transform.translation.x, tf.transform.translation.y)
        end = world_to_cell(map_msg.info, goal.pose.position.x, goal.pose.position.y)
        width, height = map_msg.info.width, map_msg.info.height
        inflation = self.get_parameter("obstacle_inflation_cells").value
        remembered_blocked = inflate_cells(self.remembered, width, height, inflation)
        observed_blocked = inflate_cells(self.observed, width, height, inflation)
        blocked = remembered_blocked | observed_blocked
        # A scan obstacle can enter the inflation band around the robot itself
        # even when the footprint is still clear. Permit the A* search to leave
        # that one dynamic cell; static-map and confirmed-memory collisions stay
        # hard failures, while Nav2's local costmap retains its collision safety.
        dynamic_escape_cells = self._dynamic_escape_halo(
            start, observed_blocked | remembered_blocked, self.observed, self.remembered,
            grid, inflation)
        static_escape_cells = self._static_inflation_escape_halo(
            start, grid, self.raw_static_grid, inflation,
            remembered_blocked | observed_blocked, self.remembered, self.observed)
        escape_cells = dynamic_escape_cells | static_escape_cells
        allow_blocked_start = bool(escape_cells)
        cells = self.planner.plan(grid, start, end, blocked,
                                  allow_blocked_start=allow_blocked_start,
                                  escape_cells=escape_cells)
        self.planned_cells = cells
        if not cells:
            self.get_logger().error(
                f"No path from {start} to {end} (static_start={grid.is_static_obstacle(start)}, "
                f"remembered_start={start in remembered_blocked}, "
                f"observed_start={start in observed_blocked}, "
                f"raw_observed_start={start in self.observed}, "
                f"dynamic_escape_cells={len(dynamic_escape_cells)}, "
                f"static_escape_cells={len(static_escape_cells)}, "
                f"static_goal={grid.is_static_obstacle(end)}, "
                f"remembered_goal={end in remembered_blocked}, "
                f"observed_goal={end in observed_blocked})")
            path = Path()
            path.header.frame_id = frame
            path.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(path)
            return
        path = Path()
        path.header.frame_id = frame
        path.header.stamp = self.get_clock().now().to_msg()
        from math import atan2, sin, cos
        for index, cell in enumerate(cells):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = cell_to_world(map_msg.info, cell)
            if index + 1 < len(cells):
                next_x, next_y = cell_to_world(map_msg.info, cells[index + 1])
                yaw = atan2(next_y - pose.pose.position.y, next_x - pose.pose.position.x)
                pose.pose.orientation.z, pose.pose.orientation.w = sin(yaw / 2.0), cos(yaw / 2.0)
            else:
                pose.pose.orientation = goal.pose.orientation
            path.poses.append(pose)
        self.path_pub.publish(path)
        self.get_logger().info(f"Published A* path with {len(path.poses)} poses")

    @staticmethod
    def _static_inflation_escape_halo(start, inflated_grid, raw_grid, inflation,
                                      dynamic_blocked, raw_remembered, raw_observed):
        """Allow bounded egress from static inflation when the raw map cell is free.

        Unknown and occupied map cells, live scan returns, and confirmed memory
        remain hard blocks. Only the connected inflated-only border around the
        robot's current cell can be crossed; the configured inflation remains
        fully enforced everywhere else.
        """
        if (raw_grid is None or not raw_grid.contains(start)
                or raw_grid.is_static_obstacle(start)
                or not inflated_grid.is_static_obstacle(start)
                or start in raw_remembered or start in raw_observed
                or start in dynamic_blocked):
            return set()

        limit = max(1, int(inflation) + 1)
        escaped = {start}
        queue = deque([(start, 0)])
        while queue:
            cell, depth = queue.popleft()
            if depth >= limit:
                continue
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                neighbor = (cell[0] + dx, cell[1] + dy)
                if (neighbor in escaped or not inflated_grid.contains(neighbor)
                        or not inflated_grid.is_static_obstacle(neighbor)
                        or raw_grid.is_static_obstacle(neighbor)
                        or neighbor in dynamic_blocked
                        or neighbor in raw_remembered or neighbor in raw_observed):
                    continue
                escaped.add(neighbor)
                queue.append((neighbor, depth + 1))
        return escaped

    @staticmethod
    def _dynamic_escape_halo(start, soft_blocked, raw_observed, raw_remembered, grid, inflation):
        """Allow egress from a soft inflation halo around observed/remembered cells.

        Raw occupied returns, confirmed memory cells, and static map obstacles remain
        hard blocks. Only the bounded inflated border may be crossed on egress.
        """
        if (start not in soft_blocked or start in raw_observed
                or start in raw_remembered or grid.is_static_obstacle(start)):
            return set()
        limit = max(1, int(inflation) + 1)
        escaped = {start}
        queue = deque([(start, 0)])
        while queue:
            cell, depth = queue.popleft()
            if depth >= limit:
                continue
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor in escaped or neighbor not in soft_blocked:
                    continue
                if (neighbor in raw_observed or neighbor in raw_remembered
                        or grid.is_static_obstacle(neighbor)):
                    continue
                escaped.add(neighbor)
                queue.append((neighbor, depth + 1))
        return escaped


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
