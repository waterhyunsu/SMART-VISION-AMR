"""Conservative in-place sweep clearance and independent scan/map agreement."""
import math

import cv2
import numpy as np


def scan_points(message, laser_pose):
    """Return finite hit endpoints in base frame, respecting the real sensor offset."""
    ranges = np.asarray(message.ranges, dtype=float)
    angles = message.angle_min + np.arange(len(ranges))*message.angle_increment
    valid = np.isfinite(ranges) & (ranges >= message.range_min) & (ranges <= message.range_max)
    angles = angles[valid] + laser_pose[2]
    return np.column_stack((laser_pose[0]+ranges[valid]*np.cos(angles),
                            laser_pose[1]+ranges[valid]*np.sin(angles)))


def rotation_clearance(message, laser_pose, radius=.30):
    """No translation: all observed obstacle endpoints must clear the sweep circle.

    Positive infinity means no return within the sensor range. A few sparse
    zero/NaN LDS dropouts may be bounded by nearby finite hits: runs of at most
    two beams, at most 2% of beams, and at most three degrees between the
    bounding beams. Inferred clearance needs an extra 8 cm for one missing beam
    or 10 cm for two after the laser-to-base offset and angular uncertainty are
    applied. Larger or unbounded gaps stay unsafe;
    neither the physical sweep radius nor valid close returns are relaxed.
    """
    values = np.asarray(message.ranges, dtype=float)
    if (values.ndim != 1 or len(values) < 30
            or not math.isfinite(radius) or radius <= 0.
            or len(laser_pose) != 3 or not all(math.isfinite(value) for value in laser_pose)
            or not math.isfinite(message.angle_min)
            or not math.isfinite(message.angle_increment)
            or abs(message.angle_increment)*(len(values)-1) < 6.0
            or not math.isfinite(message.range_min) or message.range_min < 0.
            or not math.isfinite(message.range_max) or message.range_max <= radius
            or message.range_max <= message.range_min):
        return False, None

    finite_hits = (np.isfinite(values) & (values >= max(.001, message.range_min))
                   & (values <= message.range_max))
    dropouts = np.isnan(values) | (values == 0.)
    # Negative infinity, negative ranges, positive sub-minimum hits and invalid
    # over-range finite values must never be converted into free space.
    if not np.all(finite_hits | np.isposinf(values) | dropouts):
        return False, None

    hit_angles = (message.angle_min+np.flatnonzero(finite_hits)*message.angle_increment
                  + laser_pose[2])
    points = np.column_stack((laser_pose[0]+values[finite_hits]*np.cos(hit_angles),
                              laser_pose[1]+values[finite_hits]*np.sin(hit_angles)))
    if len(points) < 8:
        return False, None
    distance = float(np.min(np.linalg.norm(points, axis=1)))

    if np.any(dropouts):
        step = abs(message.angle_increment)
        if np.count_nonzero(dropouts)/len(values) > .02:
            return False, None
        # The end bins may describe the same bearing. Never bridge a gap that
        # wraps around the scan's representation without an interior bound.
        if dropouts[0] or dropouts[-1]:
            return False, None
        indices = np.flatnonzero(dropouts)
        run_start = run_end = int(indices[0])
        runs = []
        for index in indices[1:]:
            index = int(index)
            if index == run_end+1:
                run_end = index
            else:
                runs.append((run_start, run_end))
                run_start = run_end = index
        runs.append((run_start, run_end))
        for start, end in runs:
            run_length = end-start+1
            gap_angle = (run_length+1)*step
            if (run_length > 2 or
                    gap_angle > math.radians(2. if run_length == 1 else 3.)+1e-12):
                return False, None
            before, after = (start-1) % len(values), (end+1) % len(values)
            if not finite_hits[before] or not finite_hits[after]:
                return False, None
            neighbor_ranges = values[[before, after]]
            neighbor_angles = (message.angle_min
                + np.array([before, after])*message.angle_increment + laser_pose[2])
            neighbors = np.column_stack((laser_pose[0]+neighbor_ranges*np.cos(neighbor_angles),
                                         laser_pose[1]+neighbor_ranges*np.sin(neighbor_angles)))
            neighbor_distance = float(np.min(np.linalg.norm(neighbors, axis=1)))
            bounded_range = float(np.min(neighbor_ranges))
            missing_angles = (message.angle_min
                + np.arange(start, end+1)*message.angle_increment + laser_pose[2])
            bounded_distance = float(np.min(np.hypot(
                laser_pose[0]+bounded_range*np.cos(missing_angles),
                laser_pose[1]+bounded_range*np.sin(missing_angles))))
            # A hidden hit may lie up to ``run_length`` beam intervals from
            # the nearest measured endpoint; subtract the full chord length.
            bounded_distance -= 2.*bounded_range*math.sin(run_length*step*.5)
            clearance = min(neighbor_distance, bounded_distance)
            if clearance < radius+(.08 if run_length == 1 else .10):
                return False, clearance
            distance = min(distance, clearance)
    return distance >= radius, distance


def occupied_distance(data, resolution):
    data = np.asarray(data)
    distance = cv2.distanceTransform((data < 65).astype(np.uint8),
                                     cv2.DIST_L2, cv2.DIST_MASK_PRECISE)*resolution
    distance[data < 0] = np.inf
    return distance


def scan_map_agreement(points, pose, metadata, distance, tolerance=.15):
    """Fraction of static-wall hits with an unobstructed mapped line of sight.

    Matching endpoints behind an intervening wall are inconsistent, even when
    AMCL covariance is small. Unknown/outside cells also cannot validate a ray.
    Rays start at the base center: skip its first 25 cm (including the 6.4 cm
    Waffle Pi sensor offset) and the last 20 cm near the measured wall surface.
    This geometric check rejects aliases, but cannot distinguish perfectly
    symmetric rooms without another observation or operator confirmation.
    """
    if len(points) < 12:
        return 0.
    points = np.asarray(points)[np.linspace(0, len(points)-1, min(160, len(points)), dtype=int)]
    c, s = math.cos(pose['yaw']), math.sin(pose['yaw'])
    world = points @ np.array([[c, s], [-s, c]]) + [pose['x'], pose['y']]
    origin = metadata['origin']
    c, s = math.cos(origin['yaw']), math.sin(origin['yaw'])
    rotation = np.array([[c, -s], [s, c]])
    local = (world-[origin['x'], origin['y']]) @ rotation
    robot = (np.array([pose['x'], pose['y']])-[origin['x'], origin['y']]) @ rotation
    cells = np.floor(local/metadata['resolution']).astype(int)
    valid = ((cells[:, 0] >= 0) & (cells[:, 0] < metadata['width'])
             & (cells[:, 1] >= 0) & (cells[:, 1] < metadata['height']))
    matched = np.zeros(len(points), dtype=bool)
    matched[valid] = distance[cells[valid, 1], cells[valid, 0]] <= tolerance
    # Generate the same half-cell samples in bounded batches. A Python loop
    # creating several arrays for each of 160 rays was costly on the Pi at the
    # scan rate; batching leaves the wall/unknown-cell decisions unchanged.
    indices = np.flatnonzero(matched)
    if len(indices):
        deltas = local[indices]-robot
        lengths = np.linalg.norm(deltas, axis=1)
        long = lengths > .45
        indices, deltas, lengths = indices[long], deltas[long], lengths[long]
        resolution = metadata['resolution']
        for first in range(0, len(indices), 32):
            batch_indices = indices[first:first+32]
            batch_deltas = deltas[first:first+32]
            batch_lengths = lengths[first:first+32]
            # Each ray keeps the original count and np.linspace endpoints.
            counts = np.maximum(2, np.ceil((batch_lengths-.45)/(resolution*.5)).astype(int)+1)
            starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
            owners = np.repeat(np.arange(len(batch_indices)), counts)
            offsets = np.arange(int(np.sum(counts)))-starts[owners]
            start = .25/batch_lengths
            stop = 1.-.20/batch_lengths
            fractions = start[owners]+offsets*((stop-start)/(counts-1))[owners]
            fractions[starts+counts-1] = stop
            ray = np.floor((robot+fractions[:, None]*batch_deltas[owners])/resolution).astype(int)
            inside = ((ray[:, 0] >= 0) & (ray[:, 0] < metadata['width'])
                      & (ray[:, 1] >= 0) & (ray[:, 1] < metadata['height']))
            # Clip only for safe indexing; an out-of-bounds sample still fails.
            x = np.clip(ray[:, 0], 0, metadata['width']-1)
            y = np.clip(ray[:, 1], 0, metadata['height']-1)
            clearance = distance[y, x]
            clear = inside & np.isfinite(clearance) & (clearance > 0.)
            matched[batch_indices] = np.logical_and.reduceat(clear, starts)
    return float(np.mean(matched))
