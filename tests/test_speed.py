import time

import pytest

from modules.speed import (
    LinearPlaneCalibration,
    SpeedConfig,
    SpeedEstimator,
    calibrate_pixels_per_meter,
)


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


# --- Phase 6: rigorous video-time estimator ---------------------------------

def _feed(est, track_id, timestamps, step_px=68):
    """Feed a constant-velocity trajectory (step_px per frame) at the given
    video timestamps; return the last SpeedEstimate."""
    last = None
    for i, t in enumerate(timestamps):
        x = i * step_px
        last = est.estimate(track_id, (x, 0, x + 50, 50), timestamp=t)
    return last


def test_estimate_matches_known_ground_truth_speed():
    # 68 px = 1 m. 68 px per 0.1 s of video = 10 m/s = 36 km/h.
    est = SpeedEstimator(68)
    last = _feed(est, 0, [i * 0.1 for i in range(6)])
    assert last.valid
    assert abs(last.kmh - 36.0) < 0.6
    assert last.uncertainty == 0.0  # constant velocity -> no spread


def test_processing_wall_clock_does_not_affect_estimate():
    # Same video timestamps, but one run sleeps between frames (slower machine).
    # The estimated vehicle speed must be identical.
    fast = _feed(SpeedEstimator(68), 0, [i * 0.1 for i in range(6)])

    slow_est = SpeedEstimator(68)
    slow = None
    for i in range(6):
        time.sleep(0.01)  # simulate a slow processor
        x = i * 68
        slow = slow_est.estimate(0, (x, 0, x + 50, 50), timestamp=i * 0.1)

    assert fast.kmh == slow.kmh == 36.0
    assert fast.valid and slow.valid


def test_impossible_jump_is_rejected():
    est = SpeedEstimator(68, SpeedConfig(max_plausible_kmh=100))
    est.estimate(0, (0, 0, 50, 50), 0.0)
    est.estimate(0, (68, 0, 118, 50), 0.1)     # 36 km/h
    est.estimate(0, (136, 0, 186, 50), 0.2)    # 36 km/h
    est.estimate(0, (6800, 0, 6850, 50), 0.3)  # teleport -> rejected
    after = est.estimate(0, (6868, 0, 6918, 50), 0.4)  # 36 from teleport point
    assert after.kmh < 100  # the median stays sane, not blown up by the jump


def test_no_estimate_before_minimum_samples():
    est = SpeedEstimator(68, SpeedConfig(min_samples=3))
    e0 = est.estimate(0, (0, 0, 50, 50), 0.0)
    e1 = est.estimate(0, (68, 0, 118, 50), 0.1)
    assert not e0.valid and not e1.valid  # fewer than 3 samples


def test_no_estimate_before_minimum_duration():
    est = SpeedEstimator(68, SpeedConfig(min_samples=1, min_duration_s=0.5))
    e = None
    for i in range(3):
        x = i * 68
        e = est.estimate(0, (x, 0, x + 50, 50), timestamp=i * 0.05)  # only 0.1s total
    assert not e.valid  # duration under the 0.5s minimum


def test_uncertainty_grows_with_variable_speed():
    est = SpeedEstimator(68)
    # alternating fast/slow steps -> non-zero spread
    positions = [0, 68, 68 + 20, 68 + 20 + 68, 68 + 20 + 68 + 20, 244]
    last = None
    for i, x in enumerate(positions):
        last = est.estimate(0, (x, 0, x + 50, 50), timestamp=i * 0.1)
    assert last.valid
    assert last.uncertainty > 0.0


def test_linear_plane_calibration_interpolates():
    cal = LinearPlaneCalibration(near_y=1000, near_ppm=100, far_y=0, far_ppm=20)
    assert cal(1000) == 100
    assert cal(0) == 20
    assert cal(500) == 60  # midpoint
    assert cal(-100) > 0   # clamped positive on extrapolation
