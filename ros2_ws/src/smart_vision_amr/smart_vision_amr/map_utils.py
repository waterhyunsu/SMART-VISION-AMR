"""OccupancyGrid conversion helpers shared by the ROS nodes."""

from math import cos, sin, atan2, ceil, floor, hypot, isfinite

from smart_vision_amr.grid import Grid


def yaw_from_quaternion(q):
    return atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def world_to_cell(info, x, y):
    yaw = yaw_from_quaternion(info.origin.orientation)
    dx, dy = x - info.origin.position.x, y - info.origin.position.y
    local_x = cos(yaw) * dx + sin(yaw) * dy
    local_y = -sin(yaw) * dx + cos(yaw) * dy
    return floor(local_x / info.resolution), floor(local_y / info.resolution)


def cell_to_world(info, cell):
    x, y = cell
    lx = (x + 0.5) * info.resolution
    ly = (y + 0.5) * info.resolution
    yaw = yaw_from_quaternion(info.origin.orientation)
    return (info.origin.position.x + cos(yaw) * lx - sin(yaw) * ly,
            info.origin.position.y + sin(yaw) * lx + cos(yaw) * ly)


def occupancy_to_grid(message, occupied_threshold=65, inflation_cells=0,
                      unknown_is_obstacle=True):
    width, height = message.info.width, message.info.height
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            value = message.data[y * width + x]
            # Unknown remains blocked for planning. A wall-echo reference must
            # use only known occupied cells, never interpret unknown as a wall.
            row.append((unknown_is_obstacle and value < 0) or value >= occupied_threshold)
        rows.append(tuple(row))
    if inflation_cells > 0:
        # Expand only from source rows. Blocked/unknown cells are already true,
        # so this avoids a large per-cell inflation set for mostly-unknown maps.
        source_rows = [tuple(x for x, blocked in enumerate(row) if blocked) for row in rows]
        for y in range(height):
            expanded = set()
            for source_y in range(max(0, y - inflation_cells), min(height, y + inflation_cells + 1)):
                for x in source_rows[source_y]:
                    expanded.update(range(max(0, x - inflation_cells), min(width, x + inflation_cells + 1)))
            if expanded:
                rows[y] = tuple(value or x in expanded for x, value in enumerate(rows[y]))
    return Grid(tuple(rows))


def wall_echo_reference_grid(message, occupied_threshold=65):
    """Known walls and unknown pixels touching one, for endpoint echo matching.

    A narrow unknown fringe along a mapped wall belongs to the same static
    barrier for path planning. An isolated unknown frontier does not: a new
    obstacle just before that frontier must remain visible to the planner.
    """
    width, height = message.info.width, message.info.height
    values = message.data
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            value = values[y * width + x]
            adjoining_wall = value < 0 and any(
                values[ny * width + nx] >= occupied_threshold
                for ny in range(max(0, y - 1), min(height, y + 2))
                for nx in range(max(0, x - 1), min(width, x + 2))
            )
            row.append(value >= occupied_threshold or adjoining_wall)
        rows.append(tuple(row))
    return Grid(tuple(rows))


def inflate_cells(cells, width, height, radius):
    if radius <= 0:
        return set(cells)
    result = set()
    for x, y in cells:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    result.add((nx, ny))
    return result


def inflate_cells_circular(cells, width, height, radius):
    """Block centers whose circular footprint reaches an occupied unit cell.

    ``radius`` is measured in cells and includes the caller's footprint and
    position uncertainty. Unlike square inflation, a diagonal cell is blocked
    only if the circle actually approaches its occupied square. Raw hit cells
    are always retained, even for a zero radius.
    """
    if not isfinite(radius) or radius < 0:
        raise ValueError("Circular inflation radius must be finite and nonnegative")
    reach = ceil(radius + .5)
    offsets = [(dx, dy) for dy in range(-reach, reach + 1)
               for dx in range(-reach, reach + 1)
               if hypot(max(abs(dx) - .5, 0.), max(abs(dy) - .5, 0.)) <= radius + 1e-9]
    return {(x + dx, y + dy) for x, y in cells for dx, dy in offsets
            if 0 <= x + dx < width and 0 <= y + dy < height}


def inflate_cells_footprint(cells, width, height, radius):
    """Exact circular body versus two complete square grid cells.

    The hit can lie anywhere in its cell, and the robot center anywhere in the
    candidate cell. Their nearest separation is ``max(|dx|-1, 0)`` cells on
    each axis. This is the tightest full-cell mask that does not admit a
    geometric overlap at the requested body radius.
    """
    if not isfinite(radius) or radius < 0:
        raise ValueError("Footprint inflation radius must be finite and nonnegative")
    reach = ceil(radius + 1.)
    offsets = [(dx, dy) for dy in range(-reach, reach + 1)
               for dx in range(-reach, reach + 1)
               if hypot(max(abs(dx) - 1., 0.), max(abs(dy) - 1., 0.)) <= radius + 1e-9]
    return {(x + dx, y + dy) for x, y in cells for dx, dy in offsets
            if 0 <= x + dx < width and 0 <= y + dy < height}


def first_static_obstacle_distance(info, grid, origin_x, origin_y, angle, max_distance):
    """Return the first raw-map obstacle entry distance along a world-frame ray.

    The traversal uses the map's rotated local grid and checks both side cells
    when a ray crosses a grid corner. Leaving the known map is not treated as a
    mapped wall. The caller controls whether unknown cells block ``grid``; the
    LiDAR wall-echo classifier should pass a known-occupied-only grid.
    """
    resolution = float(info.resolution)
    values = (resolution, float(origin_x), float(origin_y), float(angle), float(max_distance))
    if not all(isfinite(value) for value in values) or resolution <= 0 or max_distance < 0:
        return None

    origin = info.origin
    map_yaw = yaw_from_quaternion(origin.orientation)
    dx_world = origin_x - origin.position.x
    dy_world = origin_y - origin.position.y
    local_x = (cos(map_yaw) * dx_world + sin(map_yaw) * dy_world) / resolution
    local_y = (-sin(map_yaw) * dx_world + cos(map_yaw) * dy_world) / resolution
    direction_x = cos(angle - map_yaw)
    direction_y = sin(angle - map_yaw)
    if not all(isfinite(value) for value in (local_x, local_y, direction_x, direction_y)):
        return None
    # A ray exactly along a grid edge touches both rows/columns. Match the
    # conservative corner treatment below, including rotated cardinal rays.
    edge_offsets = [(0, 0)]
    if abs(direction_x) <= 1e-12:
        direction_x = 0.0
        if abs(local_x - round(local_x)) <= 1e-12:
            local_x = float(round(local_x))
            edge_offsets.append((-1, 0))
    if abs(direction_y) <= 1e-12:
        direction_y = 0.0
        if abs(local_y - round(local_y)) <= 1e-12:
            local_y = float(round(local_y))
            edge_offsets.append((0, -1))

    def touches_obstacle(cell):
        return any(grid.contains((cell[0] + ox, cell[1] + oy))
                   and grid.is_static_obstacle((cell[0] + ox, cell[1] + oy))
                   for ox, oy in edge_offsets)

    cell_x, cell_y = floor(local_x), floor(local_y)
    cell = (cell_x, cell_y)
    if not grid.contains(cell):
        return None
    if touches_obstacle(cell):
        return 0.0

    step_x = 1 if direction_x > 0 else -1 if direction_x < 0 else 0
    step_y = 1 if direction_y > 0 else -1 if direction_y < 0 else 0
    delta_x = resolution / abs(direction_x) if step_x else float("inf")
    delta_y = resolution / abs(direction_y) if step_y else float("inf")
    next_x = (((cell_x + 1 - local_x) * resolution / direction_x)
              if step_x > 0 else
              ((local_x - cell_x) * resolution / -direction_x)
              if step_x < 0 else float("inf"))
    next_y = (((cell_y + 1 - local_y) * resolution / direction_y)
              if step_y > 0 else
              ((local_y - cell_y) * resolution / -direction_y)
              if step_y < 0 else float("inf"))

    while True:
        entry = min(next_x, next_y)
        if entry > max_distance:
            return None
        if abs(next_x - next_y) <= 1e-12:
            # At an exact corner the ray touches both side cells. Treat either
            # occupied side as the first collision, even if the diagonal is free.
            side_cells = ((cell_x + step_x, cell_y), (cell_x, cell_y + step_y))
            if any(grid.contains(side) and touches_obstacle(side)
                   for side in side_cells):
                return max(0.0, entry)
            cell_x += step_x
            cell_y += step_y
            next_x += delta_x
            next_y += delta_y
        elif next_x < next_y:
            cell_x += step_x
            next_x += delta_x
        else:
            cell_y += step_y
            next_y += delta_y

        cell = (cell_x, cell_y)
        if not grid.contains(cell):
            return None
        if touches_obstacle(cell):
            return max(0.0, entry)


def distance_to_known_wall_area(info, grid, x, y, max_distance):
    """Distance from a world point to nearby *known occupied* cell areas.

    A grazing laser ray can land just outside an occupied grid corner while
    its exact-angle map ray misses that cell. Check only a bounded neighborhood
    of the measured endpoint. Unknown cells are never assumed to be walls.
    """
    resolution = float(info.resolution)
    values = (resolution, float(x), float(y), float(max_distance))
    if not all(isfinite(value) for value in values) or resolution <= 0 or max_distance < 0:
        return None
    yaw = yaw_from_quaternion(info.origin.orientation)
    dx, dy = x - info.origin.position.x, y - info.origin.position.y
    local_x = cos(yaw) * dx + sin(yaw) * dy
    local_y = -sin(yaw) * dx + cos(yaw) * dy
    cx, cy = floor(local_x / resolution), floor(local_y / resolution)
    reach = ceil(max_distance / resolution) + 1
    nearest = None
    for iy in range(cy - reach, cy + reach + 1):
        for ix in range(cx - reach, cx + reach + 1):
            if not grid.contains((ix, iy)) or not grid.is_static_obstacle((ix, iy)):
                continue
            gap_x = max(ix * resolution - local_x, 0.0,
                        local_x - (ix + 1) * resolution)
            gap_y = max(iy * resolution - local_y, 0.0,
                        local_y - (iy + 1) * resolution)
            distance = hypot(gap_x, gap_y)
            if distance <= max_distance + 1e-9:
                nearest = distance if nearest is None else min(nearest, distance)
    return nearest


def split_mapped_wall_returns(cells, raw_static_grid, tolerance_cells):
    """Separate dynamic returns from near-coincident mapped wall returns.

    Legacy proximity helper, retained for API compatibility only. Live scan
    classification uses first_static_obstacle_distance at measurement time.

    A LiDAR endpoint a cell or two away from a mapped wall can be caused by
    localization/rasterization error. Inflating that endpoint *again* can close
    a narrow corridor already protected by static map inflation. Only the A*
    overlay consumes this filtered set; the raw LaserScan is still available to
    Nav2's local costmap for collision avoidance.
    """
    tolerance = max(0, int(tolerance_cells))
    dynamic, mapped_wall = set(), set()
    for x, y in cells:
        near_mapped_wall = any(
            raw_static_grid.is_static_obstacle((x + dx, y + dy))
            for dy in range(-tolerance, tolerance + 1)
            for dx in range(-tolerance, tolerance + 1)
        )
        (mapped_wall if near_mapped_wall else dynamic).add((x, y))
    return dynamic, mapped_wall


def bresenham(start, end):
    """Yield integer cells from start through end, inclusive."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy
    while True:
        yield x0, y0
        if (x0, y0) == (x1, y1):
            break
        twice = 2 * error
        if twice > -dy:
            error -= dy
            x0 += sx
        if twice < dx:
            error += dx
            y0 += sy
