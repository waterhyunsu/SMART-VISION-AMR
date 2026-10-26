"""Deterministic 8-connected A* with octile costs and safe diagonal moves."""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count

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

    def plan(self, grid: Grid, start: Cell, goal: Cell, remembered: set[Cell] | None = None,
             *, allow_blocked_start: bool = False,
             escape_cells: set[Cell] | None = None) -> list[Cell]:
        remembered = remembered or set()
        escape_cells = escape_cells or set()
        if not grid.contains(start) or not grid.contains(goal):
            raise ValueError("start and goal must be inside the grid")
        if ((grid.is_static_obstacle(start) and start not in escape_cells)
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

                tentative = g_score[current] + move_cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + octile_distance(neighbor, goal)
                heappush(frontier, (priority, next(serial), neighbor))

        return []

    @staticmethod
    def _blocked(grid: Grid, cell: Cell, remembered: set[Cell], escape_cells: set[Cell] | None = None) -> bool:
        # The adapter may nominate cells only in a bounded inflation halo around
        # the robot's current cell. Raw occupied and unknown cells stay blocked.
        escaped = escape_cells or set()
        return ((grid.is_static_obstacle(cell) and cell not in escaped)
                or (cell in remembered and cell not in escaped))

    @staticmethod
    def _reconstruct(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
