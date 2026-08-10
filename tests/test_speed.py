import time

import pytest

from modules.speed import SpeedEstimator, calibrate_pixels_per_meter


def test_calibrate_pixels_per_meter():
    assert calibrate_pixels_per_meter(pixel_distance=340, real_world_meters=5) == 68.0


def test_calibrate_rejects_zero_or_negative_distance():
    with pytest.raises(ValueError):
        calibrate_pixels_per_meter(pixel_distance=100, real_world_meters=0)


def test_estimator_rejects_bad_calibration():
    with pytest.raises(ValueError):
        SpeedEstimator(pixels_per_meter=0)


def test_first_sighting_has_no_speed():
    est = SpeedEstimator(pixels_per_meter=68)
    assert est.calculate_speed(0, (0, 0, 50, 50)) == 0


def test_stationary_object_has_zero_speed():
    est = SpeedEstimator(pixels_per_meter=68)
    est.calculate_speed(0, (0, 0, 50, 50))
    time.sleep(0.05)
    assert est.calculate_speed(0, (0, 0, 50, 50)) == 0


def test_moving_object_reports_nonzero_speed():
    est = SpeedEstimator(pixels_per_meter=68)
    est.calculate_speed(0, (0, 0, 50, 50))
    time.sleep(0.05)
    speed = est.calculate_speed(0, (40, 0, 90, 50))
    assert speed > 0


def test_speed_math_is_correct():
    # 68 px = 1 meter. Moving 68px in 0.5s = 1 m in 0.5s = 2 m/s = 7.2 km/h.
    # Box (43,0,93,50) has center_x = (43+93)//2 = 68, i.e. 68px from origin.
    est = SpeedEstimator(pixels_per_meter=68)
    est.prev[0] = (0, time.time() - 0.5)
    speed = est.calculate_speed(0, (43, 0, 93, 50))
    assert speed == 7  # int-truncated


def test_two_tracked_objects_dont_interfere():
    est = SpeedEstimator(pixels_per_meter=68)
    est.calculate_speed(0, (0, 0, 50, 50))
    est.calculate_speed(1, (500, 500, 550, 550))
    time.sleep(0.05)

    moving = est.calculate_speed(0, (40, 0, 90, 50))
    stationary = est.calculate_speed(1, (500, 500, 550, 550))

    assert moving > 0
    assert stationary == 0
