"""Bounded local grid prefix repair with an untouched forward route suffix.

The caller supplies a footprint-inflated static grid and live/memory cells.
No occupancy is cleared and no reverse/start-egress exception is permitted.
Coordinates in ``reference_points`` use tile boundaries at integer values.
"""
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import ceil, floor, hypot, isfinite
import time

from .astar import DIAGONAL_COST, NEIGHBORS
from .path_geometry import segment_cells


@dataclass(frozen=True)
class LocalDetourResult:
    points: list | None
    prefix_cells: list
    rejoin_index: int | None
    expansions: int
    reason: str
    search_duration_ms: float
    rejoin_progress: float | None = None
    candidates_examined: int = 0


def plan_local_detour(grid, start, reference_points, blocked=None, *, resolution=.05,
                      blocked_edges=None, cell_penalty=None, edge_penalty=None,
                      max_rejoin_distance=2.5, max_detour_length=4.,
                      max_lateral_distance=.9, max_expansions=1500,
                      max_candidates=10, max_seconds=.06, min_rejoin_distance=.30,
                      rejoin_buffer=.15, clock=None):
    """Return a certified local prefix, or an explicit bounded failure result.

    Only forward anchors beyond the obstructed portion qualify. Every original
    point from the selected anchor to the goal is returned unchanged. Search,
    anchor validation and suffix certification share one monotonic time budget.
    Directed failed edges remain prohibited on the prefix and retained suffix.
    """
    clock = clock or time.perf_counter
    began = clock()
    expansions = 0
    candidates_examined = 0

    def result(reason, points=None, prefix=None, index=None, progress=None):
        return LocalDetourResult(points, prefix or [], index, expansions, reason,
                                 max(0., (clock()-began)*1000.), progress, candidates_examined)

    limits = (resolution, max_rejoin_distance, max_detour_length,
              max_lateral_distance, max_seconds, min_rejoin_distance)
    if (not all(type(v) in (int, float) and isfinite(v) and v > 0 for v in limits)
            or type(max_expansions) is not int or max_expansions < 1
            or type(max_candidates) is not int or max_candidates < 1
            or type(rejoin_buffer) not in (int, float) or not isfinite(rejoin_buffer)
            or not 0 <= rejoin_buffer <= 1.):
        raise ValueError('Local detour limits must be finite, positive and bounded')
    def expired():
        return clock()-began >= max_seconds

    if not hasattr(reference_points, '__len__') or not 2 <= len(reference_points) <= 20000:
        return result('invalid_reference')
    points = []
    for point in reference_points:
        if expired():
            return result('time_budget')
        if (not hasattr(point, '__len__') or len(point) != 2
                or not all(type(v) in (int, float) and isfinite(v) for v in point)
                or not grid.contains((floor(point[0]), floor(point[1])))):
            return result('invalid_reference')
        points.append(tuple(point))
    blocked, blocked_edges = set(blocked or ()), set(blocked_edges or ())
    cell_penalty, edge_penalty = cell_penalty or {}, edge_penalty or {}
    for penalties in (cell_penalty, edge_penalty):
        for index, value in enumerate(penalties.values()):
            if index % 128 == 0 and expired():
                return result('time_budget')
            if not isfinite(value) or value < 0:
                raise ValueError('Local detour penalties must be finite and nonnegative')

    occupancy = {}

    def occupied(cell):
        if cell not in occupancy:
            occupancy[cell] = grid.is_static_obstacle(cell) or cell in blocked
        return occupancy[cell]

    if occupied(start):
        return result('start_blocked')

    def segment_safe(a, b):
        # Subdivide before supercover construction, so a long source segment
        # cannot allocate a map-sized set before the deadline is checked.
        count = max(1, ceil(max(abs(b[0]-a[0]), abs(b[1]-a[1]))/8))
        for part in range(count):
            if expired():
                return False
            first = tuple(a[axis]+(b[axis]-a[axis])*part/count for axis in (0, 1))
            last = tuple(a[axis]+(b[axis]-a[axis])*(part+1)/count for axis in (0, 1))
            if any(occupied(cell) for cell in segment_cells(first, last)):
                return False
        if not blocked_edges:
            return not expired()
        # Quarter-cell samples retain directed edge order, unlike the unordered
        # supercover set used above for conservative collision certification.
        steps = max(1, ceil(hypot(b[0]-a[0], b[1]-a[1])*4))
        previous = (floor(a[0]), floor(a[1]))
        for step in range(1, steps+1):
            if step % 32 == 0 and expired():
                return False
            t = step/steps
            cell = (floor(a[0]+t*(b[0]-a[0])), floor(a[1]+t*(b[1]-a[1])))
            if cell != previous and (previous, cell) in blocked_edges:
                return False
            previous = cell
        return True

    distances = [0.]
    for a, b in zip(points, points[1:]):
        if expired():
            return result('time_budget')
        distances.append(distances[-1]+hypot(b[0]-a[0], b[1]-a[1])*resolution)
    origin = (start[0]+.5, start[1]+.5)
    # Dense published samples on one straight segment have identical geometry.
    # Collapse only the projection workspace, never the retained suffix itself.
    geometry_segments = []
    first = 0
    for index in range(1, len(points)-1):
        if expired():
            return result('time_budget')
        ax, ay = points[index][0]-points[first][0], points[index][1]-points[first][1]
        bx, by = points[index+1][0]-points[index][0], points[index+1][1]-points[index][1]
        if abs(ax*by-ay*bx) > 1e-9 or ax*bx+ay*by < 0:
            geometry_segments.append((first, index))
            first = index
    geometry_segments.append((first, len(points)-1))

    def descriptors(segments):
        prepared = []
        for index, end in segments:
            if expired():
                break
            a, b = points[index], points[end]
            dx, dy = b[0]-a[0], b[1]-a[1]
            prepared.append((a[0], a[1], dx, dy, dx*dx+dy*dy,
                             distances[index], distances[end]-distances[index]))
        return prepared

    def projection(point, segments):
        best = (float('inf'), 0.)
        for ax, ay, dx, dy, square, along, length in segments:
            if expired():
                break
            t = 0. if square < 1e-12 else max(0., min(1., ((point[0]-ax)*dx+(point[1]-ay)*dy)/square))
            candidate = ((point[0]-ax-t*dx)**2+(point[1]-ay-t*dy)**2, along+t*length)
            if candidate < best:
                best = candidate
        return best

    cross, progress = projection(origin, descriptors(geometry_segments))
    if cross**.5*resolution > min(.35, max_lateral_distance):
        return result('start_off_reference')
    if expired():
        return result('time_budget')
    # Backwards validation computes exactly which anchors retain a safe suffix.
    suffix_safe = [False]*len(points)
    suffix_safe[-1] = segment_safe(points[-1], points[-1])
    last_blocked = None
    for index in range(len(points)-2, -1, -1):
        if expired():
            return result('time_budget')
        safe = segment_safe(points[index], points[index+1])
        suffix_safe[index] = safe and suffix_safe[index+1]
        if not safe and distances[index+1] >= progress:
            last_blocked = distances[index+1] if last_blocked is None else max(last_blocked, distances[index+1])
    if last_blocked is None:
        return result('reference_not_blocked')
    candidates = [i for i, distance in enumerate(distances)
                  if suffix_safe[i] and distance >= max(progress+min_rejoin_distance, last_blocked+rejoin_buffer)
                  and distance <= progress+max_rejoin_distance]
    if not candidates:
        return result('no_forward_rejoin')
    # Sampling spreads a strict candidate cap across the complete local horizon.
    if len(candidates) > max_candidates:
        candidates = [candidates[round(i*(len(candidates)-1)/(max_candidates-1))]
                      for i in range(max_candidates)] if max_candidates > 1 else [candidates[-1]]
    anchors = {}
    for index in candidates:
        candidates_examined += 1
        cell = (floor(points[index][0]), floor(points[index][1]))
        if not occupied(cell) and segment_safe((cell[0]+.5, cell[1]+.5), points[index]):
            anchors.setdefault(cell, index)
    if not anchors:
        return result('no_safe_anchor')
    local_segments = descriptors([(index, end) for index, end in geometry_segments
                                  if distances[end] >= progress-.05
                                  and distances[index] <= progress+max_rejoin_distance])
    geometry = {}

    def local_geometry(cell):
        if cell not in geometry:
            geometry[cell] = projection((cell[0]+.5, cell[1]+.5), local_segments)
        return geometry[cell]

    heuristics = {}
    diagonal_extra = DIAGONAL_COST-1.

    def heuristic(cell):
        if cell not in heuristics:
            best = float('inf')
            for ax, ay in anchors:
                dx, dy = abs(cell[0]-ax), abs(cell[1]-ay)
                distance = (dx+diagonal_extra*dy if dx >= dy
                            else dy+diagonal_extra*dx)
                if distance < best:
                    best = distance
            heuristics[cell] = best
        return heuristics[cell]

    serial = count()
    frontier = [(heuristic(start), next(serial), start)]
    costs, lengths, parents, closed = {start: 0.}, {start: 0.}, {}, set()
    while frontier:
        if expired():
            return result('time_budget')
        _, _, current = heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        expansions += 1
        if current in anchors:
            index = anchors[current]
            connector = hypot(current[0]+.5-points[index][0], current[1]+.5-points[index][1])*resolution
            if lengths[current]+connector > max_detour_length:
                continue
            prefix = [current]
            while prefix[-1] in parents:
                prefix.append(parents[prefix[-1]])
            prefix.reverse()
            repaired = [(x+.5, y+.5) for x, y in prefix]
            if repaired[-1] == points[index]:
                repaired.pop()
            repaired += points[index:]
            return result('local_detour_ready', repaired, prefix, index, distances[index])
        if expansions >= max_expansions:
            return result('expansion_budget')
        _, current_progress = local_geometry(current)
        for dx, dy, movement in NEIGHBORS:
            if expired():
                return result('time_budget')
            neighbor = (current[0]+dx, current[1]+dy)
            edge = current, neighbor
            if neighbor in closed or occupied(neighbor) or edge in blocked_edges:
                continue
            if dx and dy and (occupied((current[0]+dx, current[1]))
                              or occupied((current[0], current[1]+dy))):
                continue
            length = lengths[current]+movement*resolution
            if length > max_detour_length:
                continue
            cost = costs[current]+movement+cell_penalty.get(neighbor, 0.)+edge_penalty.get(edge, 0.)
            if cost >= costs.get(neighbor, float('inf')):
                continue
            cross, forward = local_geometry(neighbor)
            if (cross**.5*resolution > max_lateral_distance or forward < progress-.025
                    or forward+.025 < current_progress):
                continue
            costs[neighbor], lengths[neighbor], parents[neighbor] = cost, length, current
            heappush(frontier, (cost+heuristic(neighbor), next(serial), neighbor))
    return result('no_local_route')
