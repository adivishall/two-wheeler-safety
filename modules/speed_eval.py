"""Speed-estimation validation on synthetic ground-truth trajectories (Phase 8).

The runtime speed estimator (`modules.speed.SpeedEstimator`) converts pixel
motion to km/h via a calibration. This module *measures* how accurate that is by
generating trajectories with a **known** world speed on a perspective ground
plane, running them through the real estimator under different calibrations, and
reporting MAE / RMSE / bias and overspeed precision/recall — so the better
calibration is chosen by evidence, not assertion.

The scene is defined by four image<->world reference points (a road trapezoid).
A vehicle moves down the lane at a constant world speed; its image trajectory is
produced with the world->image homography, then fed to the estimator, which
recovers the speed with:

* **constant ppm** — one pixels-per-metre, calibrated from the near-field lane
  width (the classic single-reference method); perspective-blind.
* **linear plane** — `LinearPlaneCalibration`, ppm varying with image row.
* **homography** — `HomographyPlaneCalibration`, full perspective in both axes.

Everything is deterministic; requires only numpy + cv2 (CI deps), no model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from modules.speed import (
    HomographyPlaneCalibration,
    LinearPlaneCalibration,
    SpeedConfig,
    SpeedEstimator,
    calibrate_pixels_per_meter,
)

# A road trapezoid: near edge low+wide in the image, far edge high+narrow.
# world metres: lane 6 m wide, running 0 m (near) to 40 m (far) from the camera.
IMAGE_POINTS = [(60, 420), (580, 420), (470, 180), (170, 180)]
WORLD_POINTS = [(0.0, 0.0), (6.0, 0.0), (6.0, 40.0), (0.0, 40.0)]


def _world_to_image_matrix():
    import cv2
    import numpy as np

    return cv2.getPerspectiveTransform(
        np.asarray(WORLD_POINTS, dtype=np.float32),
        np.asarray(IMAGE_POINTS, dtype=np.float32),
    )


def _apply(matrix, x, y):
    import numpy as np

    vec = matrix @ np.array([x, y, 1.0], dtype=np.float64)
    w = vec[2] if vec[2] != 0 else 1e-9
    return (float(vec[0] / w), float(vec[1] / w))


@dataclass
class Trajectory:
    true_kmh: float
    frames: list  # list[(box, timestamp)]


def synthetic_trajectory(
    true_kmh: float, *, fps: float = 25.0, n: int = 20, box_px: int = 40,
    direction: str = "approach", noise_px: float = 0.0, seed: int = 0,
) -> Trajectory:
    """A vehicle at constant ``true_kmh`` projected to image space via the
    world->image homography.

    ``direction``:
      * ``"approach"`` — moves down the lane *toward* the camera (world y falls);
        the perspective-hard case where a constant ppm fails.
      * ``"lateral"`` — crosses the frame at a fixed near-field depth (world y
        constant); the case a constant ppm was designed for.

    ``noise_px`` adds deterministic sub-pixel jitter to the box centre so a
    perfectly-matched calibration does not score an unrealistic exact 0 error.
    """
    import numpy as np

    w2i = _world_to_image_matrix()
    speed_mps = true_kmh / 3.6
    rng = np.random.default_rng(seed)
    frames = []
    for f in range(n):
        t = f / fps
        if direction == "approach":
            world_x = 3.0
            world_y = max(1.0, 38.0 - speed_mps * t)
        elif direction == "lateral":
            world_x = -6.0 + speed_mps * t  # cross the lane at a fixed depth
            world_y = 2.0  # near field, where the constant ppm is calibrated
        else:
            raise ValueError(f"unknown direction: {direction}")
        cx, cy = _apply(w2i, world_x, world_y)
        if noise_px:
            cx += float(rng.normal(0, noise_px))
            cy += float(rng.normal(0, noise_px))
        half = box_px / 2
        box = (cx - half, cy - half, cx + half, cy + half)
        frames.append((box, t))
    return Trajectory(true_kmh=true_kmh, frames=frames)


def _constant_ppm() -> float:
    """Near-field pixels-per-metre from the lane width at the near edge (the
    reference an operator can actually measure)."""
    (nlx, _nly), (nrx, _nry) = IMAGE_POINTS[0], IMAGE_POINTS[1]
    near_width_px = abs(nrx - nlx)
    near_width_m = WORLD_POINTS[1][0] - WORLD_POINTS[0][0]
    return calibrate_pixels_per_meter(near_width_px, near_width_m)


def _linear_plane() -> LinearPlaneCalibration:
    """Row-based calibration from near/far lane widths (ppm at each row)."""
    near_y = (IMAGE_POINTS[0][1] + IMAGE_POINTS[1][1]) / 2.0
    far_y = (IMAGE_POINTS[2][1] + IMAGE_POINTS[3][1]) / 2.0
    near_ppm = abs(IMAGE_POINTS[1][0] - IMAGE_POINTS[0][0]) / 6.0
    far_ppm = abs(IMAGE_POINTS[2][0] - IMAGE_POINTS[3][0]) / 6.0
    return LinearPlaneCalibration(near_y, near_ppm, far_y, far_ppm)


def _make_estimator(kind: str) -> SpeedEstimator:
    # A short min-duration so a synthetic clip yields an estimate; the estimator
    # logic (median smoothing, outlier rejection) is otherwise unchanged.
    cfg = SpeedConfig(min_samples=3, min_duration_s=0.1, max_plausible_kmh=200.0)
    if kind == "constant":
        return SpeedEstimator(_constant_ppm(), cfg)
    if kind == "linear":
        return SpeedEstimator(_constant_ppm(), cfg, calibration=_linear_plane())
    if kind == "homography":
        return SpeedEstimator(
            _constant_ppm(), cfg,
            calibration=HomographyPlaneCalibration(IMAGE_POINTS, WORLD_POINTS),
        )
    raise ValueError(f"unknown calibration kind: {kind}")


def estimate_kmh(kind: str, traj: Trajectory) -> float | None:
    """Run one trajectory through the estimator; return the final valid km/h."""
    est = _make_estimator(kind)
    last = None
    for box, t in traj.frames:
        e = est.estimate(0, box, timestamp=t)
        if e.valid:
            last = e
    return last.kmh if last else None


@dataclass
class SpeedMetrics:
    kind: str
    n: int
    mae: float
    rmse: float
    bias: float  # mean(est - true); +ve = overestimates
    overspeed_precision: float
    overspeed_recall: float

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def _error_metrics(pairs):
    """pairs: list[(true, est)] with est not None."""
    errs = [est - true for true, est in pairs]
    n = len(errs)
    mae = sum(abs(e) for e in errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    bias = sum(errs) / n
    return mae, rmse, bias


def _overspeed_prf(pairs, limit):
    tp = fp = fn = 0
    for true, est in pairs:
        actual = true > limit
        predicted = est > limit
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall


DEFAULT_SPEEDS = (20.0, 30.0, 40.0, 50.0, 60.0, 70.0)


def evaluate_calibration(
    kind: str, *, speeds=DEFAULT_SPEEDS, limit: float = 40.0, fps: float = 25.0,
    direction: str = "approach", noise_px: float = 1.5,
) -> SpeedMetrics:
    pairs = []
    for i, kmh in enumerate(speeds):
        traj = synthetic_trajectory(
            kmh, fps=fps, direction=direction, noise_px=noise_px, seed=i,
        )
        est = estimate_kmh(kind, traj)
        if est is not None:
            pairs.append((kmh, est))
    mae, rmse, bias = _error_metrics(pairs)
    p, r = _overspeed_prf(pairs, limit)
    return SpeedMetrics(kind, len(pairs), mae, rmse, bias, p, r)


def compare_calibrations(*, direction: str = "approach", **kwargs) -> dict:
    """Evaluate all three calibrations for one motion direction; return
    {kind: metrics} + the best kind (lowest MAE)."""
    results = {
        kind: evaluate_calibration(kind, direction=direction, **kwargs)
        for kind in ("constant", "linear", "homography")
    }
    best = min(results, key=lambda k: results[k].mae)
    return {
        "direction": direction,
        "results": {k: v.as_dict() for k, v in results.items()},
        "best": best,
    }
