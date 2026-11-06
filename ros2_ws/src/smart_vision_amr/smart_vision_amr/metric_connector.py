"""Bounded continuous-body connectors, retaining the original occupied squares.

Coordinates are in original map cells, not a raster with smaller obstacles.
Only the candidate robot centres are refined. No ROS state or sensor leases
are held here; the adapter must revalidate inputs before publishing a result.
"""

from dataclasses import dataclass
from heapq import heappop, heappush
from math import ceil, floor, hypot, isfinite
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class ConnectorResult:
    points: tuple = ()
    reason: str = "no_connector"
    expansions: int = 0
    duration_s: float = 0.
    target_index: int = -1


def occupied_rectangles(grid, dynamic=()):
    """Merge identical horizontal runs vertically; preserve their exact union."""
    cells = set(dynamic)
    active, result = {}, []
    for y, row in enumerate(grid.rows):
        runs, first = [], None
        for x in range(grid.width + 1):
            occupied = x < grid.width and (row[x] or (x, y) in cells)
            if occupied and first is None:
                first = x
            elif not occupied and first is not None:
                runs.append((first, x)); first = None
        next_active = {}
        for run in runs:
            next_active[run] = active.pop(run, y)
        for (left, right), bottom in active.items():
            result.append((left, bottom, right, y))
        active = next_active
    result.extend((left, bottom, right, grid.height)
                  for (left, right), bottom in active.items())
    return np.asarray(result, dtype=float).reshape((-1, 4))


def _segment_clearances(points, rectangles, width, height):
    """Exact per-segment distances; suffix queries share this same geometry."""
    p = np.asarray(points, dtype=float)
    rectangles = np.asarray(rectangles, dtype=float).reshape((-1, 4))
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        return np.asarray([-float("inf")])
    boundary = np.minimum.reduce((p[:, 0], p[:, 1], width-p[:, 0], height-p[:, 1]))
    minimum = np.minimum(boundary[:-1], boundary[1:])
    if not len(rectangles):
        return minimum
    # A rectangle farther than this current upper bound cannot reduce it.
    reach = max(0., float(minimum.max()))
    rect = rectangles[(rectangles[:, 2] >= p[:, 0].min()-reach)
                      & (rectangles[:, 0] <= p[:, 0].max()+reach)
                      & (rectangles[:, 3] >= p[:, 1].min()-reach)
                      & (rectangles[:, 1] <= p[:, 1].max()+reach)]
    if not len(rect):
        return minimum
    lo, hi = rect[:, :2], rect[:, 2:]
    a, d = p[:-1, None, :], np.diff(p, axis=0)[:, None, :]
    shape = (len(p)-1, len(rect))
    enter, leave, possible = np.zeros(shape), np.ones(shape), np.ones(shape, bool)
    for axis in (0, 1):
        moving = np.abs(d[..., axis]) > 1e-14
        t1 = np.divide(lo[None, :, axis]-a[..., axis], d[..., axis],
                       out=np.full(shape, -np.inf), where=moving)
        t2 = np.divide(hi[None, :, axis]-a[..., axis], d[..., axis],
                       out=np.full(shape, np.inf), where=moving)
        enter = np.maximum(enter, np.minimum(t1, t2))
        leave = np.minimum(leave, np.maximum(t1, t2))
        possible &= moving | ((lo[None, :, axis] <= a[..., axis])
                              & (a[..., axis] <= hi[None, :, axis]))
    minimum[np.any(possible & (enter <= leave), axis=1)] = 0.
    for endpoint in (a, a+d):
        gap = np.maximum(np.maximum(lo[None, :, :]-endpoint, endpoint-hi[None, :, :]), 0.)
        minimum = np.minimum(minimum, np.hypot(gap[..., 0], gap[..., 1]).min(axis=1))
    squared = np.square(d).sum(axis=2)
    for ix, iy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        corner = np.column_stack(((hi if ix else lo)[:, 0], (hi if iy else lo)[:, 1]))
        t = np.divide(((corner[None, :, :]-a)*d).sum(axis=2), squared,
                      out=np.zeros(shape), where=squared > 0)
        gap = a+np.clip(t, 0., 1.)[..., None]*d-corner[None, :, :]
        minimum = np.minimum(minimum, np.hypot(gap[..., 0], gap[..., 1]).min(axis=1))
    return minimum


def polyline_clearance(points, rectangles, width, height):
    """Exact segment-to-rectangle distance, including unknown map exterior."""
    return float(_segment_clearances(points, rectangles, width, height).min())


def point_sweep_clearance(points, hits):
    """Exact distance from original raw endpoints to every polyline segment."""
    if not len(hits):
        return float("inf")
    p, hit = np.asarray(points, dtype=float), np.asarray(hits, dtype=float)
    a, delta = p[:-1, None, :], np.diff(p, axis=0)[:, None, :]
    squared = np.square(delta).sum(axis=2)
    t = np.divide(((hit[None, :, :]-a)*delta).sum(axis=2), squared,
                  out=np.zeros((len(p)-1, len(hit))), where=squared > 0)
    gap = a+np.clip(t, 0., 1.)[..., None]*delta-hit[None, :, :]
    return float(np.hypot(gap[..., 0], gap[..., 1]).min())


def plan_connector(grid, dynamic, start, tails, *, radius, spacing, margin,
                   max_seconds=.06, max_expansions=1500, lateral=6.,
                   preferred=9., gain=1.5, soft_cells=()):
    """Join a short safe connector to an independently checked complete tail.

    All preparation and final validation share the deadline. The 5mm-equivalent
    midpoint field is analytic distance to the original rectangles, not EDT of
    shrunken cells. Lipschitz reserves certify every searched edge; a separate
    exact check covers the continuous start, anchor and complete output.
    """
    began = perf_counter()
    expansions = 0
    def done(reason, points=(), index=-1):
        return ConnectorResult(tuple(points), reason, expansions, perf_counter()-began, index)
    if (not all(isfinite(v) for v in (*start, radius, spacing, margin, max_seconds, lateral, preferred, gain))
            or radius <= 0 or spacing <= 0 or margin < 0 or max_seconds <= 0
            or max_expansions < 1 or lateral <= 0 or preferred < 0 or gain < 0):
        return done("invalid_configuration")
    required = radius+margin
    rectangles = occupied_rectangles(grid, dynamic)
    selected, best_quality = None, -float("inf")
    longest = max(tails, key=len, default=())
    longest_tuple = tuple(map(tuple, longest))
    suffix = np.minimum.accumulate(_segment_clearances(longest, rectangles,
        grid.width, grid.height)[::-1])[::-1] if len(longest) >= 2 else None
    for index, tail in enumerate(tails):
        if perf_counter()-began >= max_seconds:
            return done("time_budget")
        if len(tail) < 2:
            continue
        common_suffix = (len(tail) <= len(longest) and tuple(map(tuple, tail))
                         == longest_tuple[-len(tail):])
        clearance = (float(suffix[len(longest)-len(tail)]) if common_suffix else
                     polyline_clearance(tail, rectangles, grid.width, grid.height))
        if clearance+1e-9 < required:
            continue
        # Prefer a robust remaining suffix rather than the first barely-free
        # smoothed point. This is a finite preference, never hidden padding:
        # a genuinely narrow but body-clear suffix remains a candidate.
        quality = max(1., gain)*min(clearance, preferred if preferred > 0 else clearance)
        quality -= hypot(tail[0][0]-start[0], tail[0][1]-start[1])
        if quality > best_quality:
            selected, best_quality = (index, tail), quality
    if selected is None:
        return done("no_safe_complete_tail")
    index, tail = selected
    anchor = tail[0]
    # Bounding candidate centres never changes occupancy outside this window.
    x0 = max(0, floor((min(start[0], anchor[0])-lateral)/spacing))
    x1 = min(ceil(grid.width/spacing), ceil((max(start[0], anchor[0])+lateral)/spacing))
    y0 = max(0, floor((min(start[1], anchor[1])-lateral)/spacing))
    y1 = min(ceil(grid.height/spacing), ceil((max(start[1], anchor[1])+lateral)/spacing))
    if (x1-x0+1)*(y1-y0+1) > 40000:
        return done("window_budget")
    half = spacing/2
    xs, ys = (np.arange(x0*2+1, x1*2+2)*half,
              np.arange(y0*2+1, y1*2+2)*half)
    x, y = np.meshgrid(xs, ys)
    distance = np.minimum.reduce((x, y, grid.width-x, grid.height-y))
    keep = rectangles[(rectangles[:, 2] >= xs[0]-preferred-required)
                      & (rectangles[:, 0] <= xs[-1]+preferred+required)
                      & (rectangles[:, 3] >= ys[0]-preferred-required)
                      & (rectangles[:, 1] <= ys[-1]+preferred+required)]
    for left, bottom, right, top in keep:
        distance = np.minimum(distance, np.hypot(np.maximum(np.maximum(left-x, x-right), 0.),
                                                np.maximum(np.maximum(bottom-y, y-top), 0.)))
        if perf_counter()-began >= max_seconds:
            return done("time_budget")
    field = distance[::2, ::2]
    soft = set(soft_cells)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
    def node(p):
        return round(p[0]/spacing-.5)-x0, round(p[1]/spacing-.5)-y0
    def point(p):
        return (p[0]+x0+.5)*spacing, (p[1]+y0+.5)*spacing
    first, target = node(start), node(anchor)
    inside = lambda p: 0 <= p[0] < field.shape[1] and 0 <= p[1] < field.shape[0]
    if (not inside(target)
            or polyline_clearance([point(target), anchor], rectangles, grid.width, grid.height)+1e-9 < required):
        return done("endpoint_connector_blocked")
    frontier, best, parent = [], {}, {}
    # The current pose isn't rounded into an occupied node. Seed only nearby
    # centres whose complete actual-pose connector is independently clear.
    for dx,dy in ((0,0),*directions):
        seed=first[0]+dx,first[1]+dy
        if (not inside(seed) or field[seed[1],seed[0]]+1e-9 < required
                or polyline_clearance([start,point(seed)],rectangles,grid.width,grid.height)+1e-9 < required):
            continue
        cost=hypot(point(seed)[0]-start[0],point(seed)[1]-start[1])/spacing
        best[seed]=cost
        heappush(frontier,(cost+hypot(seed[0]-target[0],seed[1]-target[1]),cost,seed))
    if not frontier:
        return done("endpoint_connector_blocked")
    found = None
    while frontier:
        if perf_counter()-began >= max_seconds*.85:
            return done("time_budget")
        _, cost, current = heappop(frontier)
        if cost != best.get(current):
            continue
        if expansions >= max_expansions:
            return done("expansion_budget")
        expansions += 1
        if current == target:
            found = [current]
            while current in parent:
                current = parent[current]; found.append(current)
            found.reverse()
            break
        cx, cy = current
        for dx, dy in directions:
            nx, ny = cx+dx, cy+dy
            if not (0 <= nx < field.shape[1] and 0 <= ny < field.shape[0]):
                continue
            length = hypot(dx, dy)
            # Endpoints plus midpoint and a quarter-edge Lipschitz reserve.
            clearance=min(field[cy,cx],field[ny,nx],distance[2*cy+dy,2*cx+dx])
            if clearance+1e-9 < required+spacing*length/4:
                if (min(field[cy,cx],field[ny,nx])+1e-9 < required
                        or polyline_clearance([point(current),point((nx,ny))],rectangles,
                                              grid.width,grid.height)+1e-9 < required):
                    continue
            penalty = gain*max(0., (preferred-field[ny, nx])/preferred) if preferred > 0 else 0.
            if (floor(point((nx, ny))[0]), floor(point((nx, ny))[1])) in soft:
                penalty += max(1., gain)
            new = cost+length*(1.+penalty)
            neighbor = nx, ny
            if new < best.get(neighbor, float("inf")):
                best[neighbor], parent[neighbor] = new, current
                heappush(frontier, (new+hypot(nx-target[0], ny-target[1]), new, neighbor))
    if found is None:
        return done("no_connector")
    points = [tuple(start)]+[point(p) for p in found]+[tuple(tail[0])]
    canonical = [points[0]]
    for p in points[1:]:
        if hypot(p[0]-canonical[-1][0], p[1]-canonical[-1][1]) < 1e-9:
            continue
        if len(canonical) > 1:
            a, b = canonical[-2:]
            if abs((b[0]-a[0])*(p[1]-b[1])-(b[1]-a[1])*(p[0]-b[0])) < 1e-10:
                if (b[0]-a[0])*(p[0]-b[0])+(b[1]-a[1])*(p[1]-b[1]) >= 0:
                    canonical.pop()
        canonical.append(p)
    # Remove the nearest-lattice backfold only with an exact connector proof.
    if len(canonical) > 2 and polyline_clearance(canonical[:1]+canonical[2:3], rectangles,
                                                grid.width, grid.height)+1e-9 >= required:
        canonical.pop(1)
    # Only the new connector is canonicalized; the retained suffix remains
    # bit-for-bit and in the original order (including its exact final goal).
    canonical += list(tail[1:])
    if polyline_clearance(canonical, rectangles, grid.width, grid.height)+1e-9 < required:
        return done("output_collision")
    if perf_counter()-began >= max_seconds:
        return done("time_budget")
    return done("certified_connector", canonical, index)
