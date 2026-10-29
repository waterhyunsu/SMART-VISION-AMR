"""OccupancyGrid conversion helpers shared by the ROS nodes."""

from math import atan2, cos, floor, isfinite, sin

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
            # Unknown is conservatively blocked for global planning, but can be
            # excluded when building a known-wall-only LiDAR ray reference.
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
    cell_x, cell_y = floor(local_x), floor(local_y)
    cell = (cell_x, cell_y)
    if not grid.contains(cell):
        return None
    if grid.is_static_obstacle(cell):
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
            if any(grid.contains(side) and grid.is_static_obstacle(side)
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
        if grid.is_static_obstacle(cell):
            return max(0.0, entry)


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
