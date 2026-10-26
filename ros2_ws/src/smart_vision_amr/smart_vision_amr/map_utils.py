"""OccupancyGrid conversion helpers shared by the ROS nodes."""

from math import cos, sin, atan2, floor

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


def occupancy_to_grid(message, occupied_threshold=65, inflation_cells=0):
    width, height = message.info.width, message.info.height
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            value = message.data[y * width + x]
            # Unknown is conservatively blocked for global planning.
            row.append(value < 0 or value >= occupied_threshold)
        rows.append(tuple(row))
    if inflation_cells > 0:
        offsets = [(dx, dy)
                   for dy in range(-inflation_cells, inflation_cells + 1)
                   for dx in range(-inflation_cells, inflation_cells + 1)
                   if dx * dx + dy * dy <= inflation_cells * inflation_cells]
        sources = [(x, y) for y, row in enumerate(rows)
                   for x, blocked in enumerate(row) if blocked]
        rows = [list(row) for row in rows]
        for x, y in sources:
            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    rows[ny][nx] = True
    return Grid(tuple(tuple(row) for row in rows))


def inflate_cells(cells, width, height, radius):
    if radius <= 0:
        return set(cells)
    result = set()
    radius_squared = radius * radius
    offsets = [(dx, dy)
               for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)
               if dx * dx + dy * dy <= radius_squared]
    for x, y in cells:
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                result.add((nx, ny))
    return result


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
