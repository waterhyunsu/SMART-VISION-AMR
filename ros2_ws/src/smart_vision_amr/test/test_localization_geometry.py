"""Independent wall/line-of-sight checks for confident but incorrect AMCL poses."""
import math

import numpy as np
import pytest

from smart_vision_amr.localization_geometry import occupied_distance, scan_map_agreement


def scene():
    data = np.zeros((80, 80), dtype=np.int16)
    data[:, 60] = 100  # Measured wall at x=3.025 m.
    metadata = {'width': 80, 'height': 80, 'resolution': .05,
                'origin': {'x': 0., 'y': 0., 'yaw': 0.}}
    pose = {'x': .525, 'y': 2.025, 'yaw': 0.}
    points = np.column_stack((np.full(20, 2.5), np.linspace(-.7, .7, 20)))
    return data, metadata, pose, points


def test_clean_rays_to_known_wall_pass():
    data, metadata, pose, points = scene()
    assert scan_map_agreement(points, pose, metadata, occupied_distance(data, .05)) == 1.


def test_matching_endpoints_behind_an_intervening_wall_are_rejected():
    data, metadata, pose, points = scene()
    data[:, 30] = 100  # Endpoint-only matching would incorrectly return 1.0.
    assert scan_map_agreement(points, pose, metadata, occupied_distance(data, .05)) == 0.


def test_unknown_ray_interior_cannot_confirm_position_even_if_endpoint_matches():
    data, metadata, pose, points = scene()
    data[:, 30] = -1
    assert scan_map_agreement(points, pose, metadata, occupied_distance(data, .05)) == 0.


def test_unknown_endpoint_stays_invalid():
    data, metadata, pose, points = scene()
    data[:, 60] = -1
    assert scan_map_agreement(points, pose, metadata, occupied_distance(data, .05)) == 0.


def test_ray_check_excludes_robot_body_and_wall_surface_margins():
    data, metadata, pose, points = scene()
    data[:, 11] = 100  # 5 cm from center: inside the body exclusion zone.
    data[:, 58] = 100  # 10 cm before endpoints: within the wall surface margin.
    assert scan_map_agreement(points, pose, metadata, occupied_distance(data, .05)) == 1.


def test_rotated_map_origin_and_robot_orientation_preserve_ray_geometry():
    data, metadata, pose, points = scene()
    metadata['origin'] = {'x': -3., 'y': 4., 'yaw': math.pi/2}
    pose = {'x': -5.025, 'y': 4.525, 'yaw': math.pi/2}
    assert scan_map_agreement(points, pose, metadata, occupied_distance(data, .05)) == 1.


def test_subsampling_is_bounded_without_ignoring_last_part_of_scan():
    data, metadata, pose, points = scene()
    dense = np.repeat(points, 16, axis=0)[:319]
    dense[-100:, 0] = 1.  # These endpoints fail the measured-wall match.
    score = scan_map_agreement(dense, pose, metadata, occupied_distance(data, .05))
    assert .65 < score < .75


def _per_ray_agreement(points, pose, metadata, distance, tolerance=.15):
    """Original per-ray calculation, kept as an independent regression oracle."""
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
    for index in np.flatnonzero(matched):
        delta = local[index]-robot
        length = float(np.linalg.norm(delta))
        if length <= .45:
            continue
        count = max(2, int(math.ceil((length-.45)/(metadata['resolution']*.5)))+1)
        fractions = np.linspace(.25/length, 1.-.20/length, count)
        ray = np.floor((robot+fractions[:, None]*delta)/metadata['resolution']).astype(int)
        inside = ((ray[:, 0] >= 0) & (ray[:, 0] < metadata['width'])
                  & (ray[:, 1] >= 0) & (ray[:, 1] < metadata['height']))
        if not np.all(inside):
            matched[index] = False
            continue
        clearance = distance[ray[:, 1], ray[:, 0]]
        if not np.all(np.isfinite(clearance) & (clearance > 0.)):
            matched[index] = False
    return float(np.mean(matched))


@pytest.mark.parametrize('seed', range(24))
def test_batched_agreement_matches_per_ray_on_mixed_walls_unknown_and_rotations(seed):
    rng = np.random.default_rng(seed)
    data = np.zeros((80, 80), dtype=np.int16)
    data[:, 60] = 100
    data[:, 11] = 100  # Short endpoints in the robot's 25 cm exclusion zone.
    for _ in range(4):
        x = int(rng.integers(21, 55))
        y = int(rng.integers(0, 60))
        data[y:y+int(rng.integers(6, 20)), x] = 100
    for _ in range(2):
        x = int(rng.integers(25, 55))
        y = int(rng.integers(0, 70))
        data[y:y+5, x:x+3] = -1
    angle = float(rng.uniform(-math.pi, math.pi))
    origin = {'x': float(rng.uniform(-2, 2)),
              'y': float(rng.uniform(-2, 2)), 'yaw': angle}
    metadata = {'width': 80, 'height': 80, 'resolution': .05, 'origin': origin}
    # Rotate the entire room together; the same physical rays must be checked.
    local_robot = np.array([.525, 2.025])
    c, s = math.cos(angle), math.sin(angle)
    world_robot = np.array([origin['x'], origin['y']]) + local_robot @ np.array([[c, s], [-s, c]])
    pose = {'x': float(world_robot[0]), 'y': float(world_robot[1]), 'yaw': angle}
    far = np.column_stack((np.full(200, 2.5), np.linspace(-2.1, 2.1, 200)))
    far[:, 0] += rng.uniform(-.02, .02, len(far))
    near = np.column_stack((np.full(30, .05), np.linspace(-.2, .2, 30)))
    points = np.vstack((far, near))
    distance = occupied_distance(data, .05)
    assert scan_map_agreement(points, pose, metadata, distance) == _per_ray_agreement(
        points, pose, metadata, distance)
