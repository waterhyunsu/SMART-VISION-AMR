"""Pure geometry checks for sparse LDS dropout handling; no ROS or motor I/O."""
import math
from types import SimpleNamespace

import pytest

from smart_vision_amr.localization_geometry import rotation_clearance


def scan():
    return SimpleNamespace(ranges=[1.2]*401, angle_min=-math.pi,
        angle_increment=2.*math.pi/400, range_min=.12, range_max=3.5)


def test_realistic_401_beam_scan_tolerates_three_isolated_zero_nan_dropouts():
    message = scan()
    for index, value in ((35, 0.), (173, math.nan), (310, 0.)):
        message.ranges[index] = value
    safe, distance = rotation_clearance(message, (-.064, 0., math.pi))
    assert safe
    assert distance >= 1.1


def test_zero_dropout_stays_unknown_when_driver_reports_zero_range_min():
    message = scan()
    message.range_min = 0.
    message.ranges[30] = 0.
    assert rotation_clearance(message, (0., 0., 0.))[0]


@pytest.mark.parametrize('indices', [(0,), (400,), (0, 400), (200, 201, 202)])
def test_unbounded_or_three_beam_dropout_gap_is_unsafe(indices):
    message = scan()
    for index in indices:
        message.ranges[index] = 0.
    assert not rotation_clearance(message, (0., 0., 0.))[0]


def test_two_consecutive_lds_dropouts_with_42cm_body_clearance_are_safe():
    message = scan()
    message.ranges[99] = message.ranges[102] = .43
    message.ranges[100:102] = [0., math.nan]
    # The physical laser is 6.4 cm behind the body center. The missing beams
    # are lateral, so the bounded body-frame distance remains above 40 cm.
    safe, distance = rotation_clearance(message, (-.064, 0., 0.))
    assert safe
    assert .40 <= distance < .43


def test_two_consecutive_lds_dropouts_with_39cm_clearance_are_unsafe():
    message = scan()
    message.ranges[100:102] = [0., math.nan]
    message.ranges[99] = message.ranges[102] = .39
    safe, distance = rotation_clearance(message, (0., 0., 0.))
    assert not safe
    assert distance < .40


def test_two_beam_gap_with_invalid_boundary_stays_unsafe():
    message = scan()
    message.ranges[99:103] = [math.inf, 0., math.nan, .45]
    assert not rotation_clearance(message, (0., 0., 0.))[0]


def test_valid_close_hit_blocks_even_with_safe_two_beam_gap():
    message = scan()
    message.ranges[100:102] = [0., math.nan]
    message.ranges[200] = .29
    safe, distance = rotation_clearance(message, (0., 0., 0.))
    assert not safe
    assert distance == pytest.approx(.29)


def test_more_than_two_percent_missing_is_unsafe_even_when_isolated():
    message = scan()
    for index in range(10, 361, 40):  # Nine isolated bins: 2.24% of 401 beams.
        message.ranges[index] = math.nan
    assert not rotation_clearance(message, (0., 0., 0.))[0]


def test_wide_angular_gap_is_not_bridged():
    message = scan()
    message.ranges = [1.2]*181
    message.angle_increment = 2.*math.pi/180
    message.ranges[80] = 0.
    assert not rotation_clearance(message, (0., 0., 0.))[0]


def test_two_beam_gap_wider_than_three_degrees_is_not_bridged():
    message = scan()
    message.ranges = [1.2]*301
    message.angle_increment = 2.*math.pi/300
    message.ranges[80:82] = [0., math.nan]
    assert not rotation_clearance(message, (0., 0., 0.))[0]


@pytest.mark.parametrize('value', [math.inf, -math.inf, .05, -.1])
def test_dropout_neighbors_must_be_valid_finite_hits(value):
    message = scan()
    message.ranges[99] = value
    message.ranges[100] = 0.
    assert not rotation_clearance(message, (0., 0., 0.))[0]


def test_dropout_neighbors_need_additional_eight_centimeter_body_clearance():
    message = scan()
    message.ranges[99:102] = [.31, 0., .31]
    safe, distance = rotation_clearance(message, (0., 0., 0.))
    assert not safe
    assert distance < .38


def test_angular_uncertainty_cannot_promote_a_marginal_gap_to_clear():
    message = scan()
    message.ranges[99:102] = [.382, 0., .382]
    safe, distance = rotation_clearance(message, (0., 0., 0.))
    assert not safe
    assert distance < .38


def test_sensor_offset_and_yaw_apply_to_inferred_clearance():
    message = scan()
    message.ranges[199:202] = [.5, 0., .5]
    assert rotation_clearance(message, (0., 0., 0.))[0]
    safe, distance = rotation_clearance(message, (.15, 0., math.pi))
    assert not safe
    assert distance < .38


@pytest.mark.parametrize('value', [0., math.nan, math.inf, -math.inf])
def test_all_bad_or_no_finite_hit_scan_cannot_authorize_rotation(value):
    message = scan()
    message.ranges = [value]*401
    assert not rotation_clearance(message, (0., 0., 0.))[0]


def test_valid_close_hit_still_blocks_after_other_dropouts_are_bridged():
    message = scan()
    message.ranges[35] = 0.
    message.ranges[310] = math.nan
    message.ranges[200] = .29
    safe, distance = rotation_clearance(message, (0., 0., 0.))
    assert not safe
    assert distance == pytest.approx(.29)


def test_smallest_body_sweep_radius_remains_thirty_centimeters():
    message = scan()
    message.ranges[200] = .299
    assert not rotation_clearance(message, (0., 0., 0.))[0]
    message.ranges[200] = .301
    assert rotation_clearance(message, (0., 0., 0.))[0]
