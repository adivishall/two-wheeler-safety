"""Speed validation tests (Phase 8): homography calibration + synthetic
ground-truth trajectory metrics. Model-free (numpy + cv2 only)."""

import math

import pytest

from modules.speed import (
    HomographyPlaneCalibration,
    LinearPlaneCalibration,
)
from modules.speed_eval import (
    IMAGE_POINTS,
    WORLD_POINTS,
    compare_calibrations,
    estimate_kmh,
    evaluate_calibration,
    synthetic_trajectory,
)

# -- homography calibration -------------------------------------------------

def test_homography_recovers_known_world_distance():
    cal = HomographyPlaneCalibration(IMAGE_POINTS, WORLD_POINTS)
    # The near edge of the trapezoid is 6 m wide in the world.
    d = cal.metric_distance(IMAGE_POINTS[0], IMAGE_POINTS[1])
    assert abs(d - 6.0) < 1e-3


def test_homography_projects_reference_points_exactly():
    cal = HomographyPlaneCalibration(IMAGE_POINTS, WORLD_POINTS)
    for img_pt, world_pt in zip(IMAGE_POINTS, WORLD_POINTS):
        wx, wy = cal.project(*img_pt)
        assert abs(wx - world_pt[0]) < 1e-3
        assert abs(wy - world_pt[1]) < 1e-3


def test_homography_rejects_wrong_shape():
    with pytest.raises(ValueError):
        HomographyPlaneCalibration([(0, 0), (1, 1)], WORLD_POINTS)


def test_linear_metric_distance_matches_scalar_formula():
    # metric_distance must equal the estimator's historical midpoint formula so
    # existing linear-calibration behaviour is unchanged.
    cal = LinearPlaneCalibration(near_y=1000, near_ppm=100, far_y=0, far_ppm=20)
    p1, p2 = (100, 400), (160, 600)
    expected = math.hypot(60, 200) / cal((400 + 600) / 2.0)
    assert abs(cal.metric_distance(p1, p2) - expected) < 1e-9


# -- synthetic trajectories + metrics ---------------------------------------

def test_synthetic_trajectory_shapes():
    traj = synthetic_trajectory(40.0, n=15, direction="approach")
    assert traj.true_kmh == 40.0
    assert len(traj.frames) == 15
    box, t = traj.frames[0]
    assert len(box) == 4 and t == 0.0


def test_homography_beats_constant_on_approach():
    # The headline Phase-8 result: for motion toward the camera a single
    # pixels-per-metre is perspective-blind and unusable; the homography is far
    # more accurate.
    approach = compare_calibrations(direction="approach")
    res = approach["results"]
    assert res["homography"]["mae"] < res["constant"]["mae"]
    assert approach["best"] == "homography"
    # constant ppm misses overspeeders entirely on approach.
    assert res["constant"]["overspeed_recall"] == 0.0
    assert res["homography"]["overspeed_recall"] > res["constant"]["overspeed_recall"]


def test_constant_ppm_is_adequate_for_lateral_motion():
    # The honest flip side: for lateral motion at the calibrated depth, a
    # constant/linear calibration is already accurate (that is what it is for).
    lateral = compare_calibrations(direction="lateral")
    res = lateral["results"]
    assert res["constant"]["mae"] < 5.0
    assert lateral["best"] in ("constant", "linear")


def test_estimate_kmh_returns_value_for_valid_trajectory():
    est = estimate_kmh("homography", synthetic_trajectory(50.0, noise_px=0.0))
    assert est is not None
    assert abs(est - 50.0) < 3.0  # near-exact when the plane matches and no noise


def test_evaluate_calibration_reports_all_fields():
    m = evaluate_calibration("homography", direction="approach")
    d = m.as_dict()
    for key in ("mae", "rmse", "bias", "overspeed_precision", "overspeed_recall"):
        assert key in d
    assert m.n >= 1
