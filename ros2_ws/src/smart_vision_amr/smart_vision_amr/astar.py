"""Deterministic 8-connected A* with octile costs and safe diagonal moves."""

from __future__ import annotations

from collections.abc import Mapping
from heapq import heappop, heappush
from itertools import count
from collections import deque
from math import hypot, isfinite

from .grid import Cell, Grid

CARDINAL_COST = 1.0
DIAGONAL_COST = 2**0.5
NEIGHBORS = (
    (-1, 0, CARDINAL_COST),
    (1, 0, CARDINAL_COST),
    (0, -1, CARDINAL_COST),
    (0, 1, CARDINAL_COST),
    (-1, -1, DIAGONAL_COST),
    (1, -1, DIAGONAL_COST),
    (-1, 1, DIAGONAL_COST),
    (1, 1, DIAGONAL_COST),
)


def octile_distance(a: Cell, b: Cell) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (DIAGONAL_COST - 1.0) * min(dx, dy)


class AStarPlanner:
    """Grid A* that treats confirmed remembered changes as occupied cells."""

    def plan_with_start_egress(
            self, grid: Grid, start: Cell, goal: Cell,
            remembered: set[Cell] | None = None, *, raw_grid: Grid,
            raw_blocked: set[Cell] | None = None, robot_radius_cells: float = 4.4,
            margin_cells: float = 0.4, max_egress_cells: int = 7,
            escape_cells: set[Cell] | None = None,
            allow_blocked_start: bool = False,
            cell_penalty: Mapping[Cell, float] | None = None) -> list[Cell]:
        """Leave only a physically clear start inside static inflation.

        Inflation stays unchanged for the goal and the ordinary A* tail. A
        bounded cardinal prefix may leave its start halo, provided the complete
        circular footprint stays clear of raw occupied/unknown cells, raw scan
        returns, confirmed memory, and the map boundary. Clearance must never
        decrease along that prefix. No static obstacle is globally unblocked.
        Distances are in grid cells, including fractional footprint radii.
        """
        remembered = remembered or set()
        raw_blocked = raw_blocked or set()
        blocked = remembered | raw_blocked
        safe_escape_cells = (escape_cells or set()) - raw_blocked
        if not grid.contains(start) or not grid.contains(goal):
            raise ValueError("start and goal must be inside the grid")
        if (raw_grid.width != grid.width or raw_grid.height != grid.height):
            raise ValueError("raw and inflated grids must have the same dimensions")
        # Goal inflation is never relaxed, even if it was nominated as part of
        # a dynamic escape halo for the start of an earlier attempt.
        if grid.is_static_obstacle(goal) or goal in blocked:
            return []
        if raw_grid.is_static_obstacle(start) or start in raw_blocked:
            return []
        if not grid.is_static_obstacle(start):
            return self.plan(grid, start, goal, blocked,
                             allow_blocked_start=allow_blocked_start,
                             escape_cells=safe_escape_cells,
                             cell_penalty=cell_penalty)
        radius = float(robot_radius_cells)
        margin = float(margin_cells)
        if (not isfinite(radius) or not isfinite(margin)
                or radius < 0.0 or margin < 0.0):
            raise ValueError("footprint radius and margin must be finite and nonnegative")
        limit = int(max_egress_cells)
        if limit < 1:
            return []
        required_clearance = radius + margin
        occupied = {(x, y) for y, row in enumerate(raw_grid.rows)
                    for x, blocked in enumerate(row) if blocked}
        occupied.update(cell for cell in raw_blocked if raw_grid.contains(cell))
        clearance_cache: dict[Cell, float] = {}

        def clearance(cell):
            if cell in clearance_cache:
                return clearance_cache[cell]
            if not raw_grid.contains(cell):
                return float("-inf")
            x, y = cell
            # Map cells have unit square area centered on integer cell indices.
            # Center-to-center distance would overestimate physical clearance.
            distance = min(x + 0.5, grid.width - x - 0.5,
                           y + 0.5, grid.height - y - 0.5)
            for ox, oy in occupied:
                dx = max(abs(x - ox) - 0.5, 0.0)
                dy = max(abs(y - oy) - 0.5, 0.0)
                distance = min(distance, hypot(dx, dy))
            clearance_cache[cell] = distance
            return distance

        epsilon = 1e-9
        if clearance(start) + epsilon < required_clearance:
            return []
        queue = deque([(start, 0)])
        came_from: dict[Cell, Cell] = {}
        visited = {start}
        while queue:
            current, depth = queue.popleft()
            if (not grid.is_static_obstacle(current)
                    and current not in blocked):
                tail = self.plan(grid, current, goal, blocked,
                                 allow_blocked_start=allow_blocked_start,
                                 escape_cells=safe_escape_cells,
                                 cell_penalty=cell_penalty)
                if tail:
                    return self._reconstruct(came_from, current) + tail[1:]
                # This is an exit, so do not extend the exceptional prefix into
                # ordinary space and later re-enter inflation to reach a goal.
                continue
            if depth >= limit:
                continue
            previous_clearance = clearance(current)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = current[0] + dx, current[1] + dy
                if (neighbor in visited or raw_grid.is_static_obstacle(neighbor)
                        or neighbor in raw_blocked):
                    continue
                next_clearance = clearance(neighbor)
                if (next_clearance + epsilon < required_clearance
                        or next_clearance + epsilon < previous_clearance):
                    continue
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append((neighbor, depth + 1))
        return []

    def plan(self, grid: Grid, start: Cell, goal: Cell, remembered: set[Cell] | None = None,
             *, allow_blocked_start: bool = False,
             escape_cells: set[Cell] | None = None,
             cell_penalty: Mapping[Cell, float] | None = None) -> list[Cell]:
        """Find a shortest route, optionally charging extra cost when entering cells.

        ``cell_penalty`` is in units of one cardinal step. It can favor a
        longer route with greater obstacle clearance, but never makes a blocked
        cell traversable. The octile heuristic remains admissible because every
        added penalty is nonnegative.
        """
        remembered = remembered or set()
        escape_cells = escape_cells or set()
        if cell_penalty is None:
            cell_penalty = {}
        else:
            for penalty in cell_penalty.values():
                if not isfinite(penalty) or penalty < 0:
                    raise ValueError("cell penalties must be finite and nonnegative")
        if not grid.contains(start) or not grid.contains(goal):
            raise ValueError("start and goal must be inside the grid")
        if (grid.is_static_obstacle(start)
                or (start in remembered and not allow_blocked_start)
                or self._blocked(grid, goal, remembered, escape_cells)):
            return []
        if start == goal:
            return [start]

        serial = count()
        frontier: list[tuple[float, int, Cell]] = []
        heappush(frontier, (octile_distance(start, goal), next(serial), start))
        came_from: dict[Cell, Cell] = {}
        g_score = {start: 0.0}
        closed: set[Cell] = set()

        while frontier:
            _, _, current = heappop(frontier)
            if current in closed:
                continue
            if current == goal:
                return self._reconstruct(came_from, goal)
            closed.add(current)

            for dx, dy, move_cost in NEIGHBORS:
                neighbor = (current[0] + dx, current[1] + dy)
                if self._blocked(grid, neighbor, remembered, escape_cells) or neighbor in closed:
                    continue
                if dx and dy:
                    side_a = (current[0] + dx, current[1])
                    side_b = (current[0], current[1] + dy)
                    if (self._blocked(grid, side_a, remembered, escape_cells)
                            or self._blocked(grid, side_b, remembered, escape_cells)):
                        continue

                tentative = g_score[current] + move_cost + cell_penalty.get(neighbor, 0.0)
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + octile_distance(neighbor, goal)
                heappush(frontier, (priority, next(serial), neighbor))

        return []

    @staticmethod
    def _blocked(grid: Grid, cell: Cell, remembered: set[Cell], escape_cells: set[Cell] | None = None) -> bool:
        # Only the adapter may nominate cells in the dynamic LiDAR inflation halo.
        # Static occupancy and confirmed memory are never bypassed.
        return grid.is_static_obstacle(cell) or (cell in remembered and cell not in (escape_cells or set()))

    @staticmethod
    def _reconstruct(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
