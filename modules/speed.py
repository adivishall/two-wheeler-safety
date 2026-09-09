"""Per-vehicle speed estimation.

Speed comes from how far a tracked box moves between frames, converted from
pixels to metres via camera calibration. Two things matter for correctness:

1. **Use video time, not wall-clock time.** The rigorous path (:meth:`estimate`)
   takes an explicit ``timestamp`` in *video seconds* (``frame_index / fps``).
   A slower computer processes fewer frames per real second but the video
   timestamps are unchanged, so the estimated vehicle speed does not depend on
   how fast the machine happens to run. (The legacy :meth:`calculate_speed`,
   kept for backward compatibility, still uses wall-clock time and is only
   suitable for live capture where processing keeps up with the source.)

2. **Calibration.** ``pixels_per_meter`` converts pixel distance to metres.
   Get it from :func:`calibrate_pixels_per_meter` using a known real-world
   reference distance in-frame. This is only valid for motion roughly
   perpendicular to the camera; a vehicle moving toward/away reads artificially
   slow. For a coarse perspective correction, pass an optional
   :class:`LinearPlaneCalibration` so pixels-per-metre varies with image row.

The rigorous estimator also smooths over a window (median, robust to a single
bad frame), rejects physically impossible jumps, waits for a minimum number of
samples and track duration before reporting, and exposes an uncertainty (the
spread of recent instantaneous estimates).
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass


def calibrate_pixels_per_meter(pixel_distance, real_world_meters):
    """Compute the pixels_per_meter calibration value SpeedEstimator needs.

    Mark two points in your camera's frame that are a known real-world
    distance apart (e.g. two road markings), measure the pixel distance
    between them in a frame, and pass both here:

        calibrate_pixels_per_meter(pixel_distance=340, real_world_meters=5)
    """
    if real_world_meters <= 0:
        raise ValueError("real_world_meters must be > 0")
    return pixel_distance / real_world_meters


class LinearPlaneCalibration:
    """Optional perspective-aware calibration.

    Objects higher up the frame are farther from the camera, so fewer pixels
    span a metre. Given two ``(image_row_y, pixels_per_meter)`` references, this
    linearly interpolates/extrapolates ``pixels_per_meter`` for any row and
    clamps it positive. A coarse road-plane approximation, not a full homography.
    """

    def __init__(self, near_y, near_ppm, far_y, far_ppm):
        if near_ppm <= 0 or far_ppm <= 0:
            raise ValueError("pixels_per_meter references must be > 0")
        if near_y == far_y:
            raise ValueError("near_y and far_y must differ")
        self.near_y = float(near_y)
        self.near_ppm = float(near_ppm)
        self.far_y = float(far_y)
        self.far_ppm = float(far_ppm)

    def __call__(self, y):
        t = (y - self.far_y) / (self.near_y - self.far_y)
        ppm = self.far_ppm + (self.near_ppm - self.far_ppm) * t
        return max(1e-6, ppm)


@dataclass(frozen=True)
class SpeedConfig:
    smoothing_window: int = 5  # instantaneous samples kept for the median
    max_plausible_kmh: float = 150.0  # instantaneous speeds above this are dropped
    min_samples: int = 3  # samples required before a speed is reported
    min_duration_s: float = 0.2  # track lifetime required before reporting


DEFAULT_SPEED_CONFIG = SpeedConfig()


@dataclass
class SpeedEstimate:
    kmh: float
    uncertainty: float  # spread (std) of recent instantaneous estimates, km/h
    samples: int
    valid: bool


class SpeedEstimator:
    def __init__(
        self,
        pixels_per_meter,
        config: SpeedConfig = DEFAULT_SPEED_CONFIG,
        calibration: LinearPlaneCalibration | None = None,
    ):
        if pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be > 0")

        self.pixels_per_meter = pixels_per_meter
        self.config = config
        self.calibration = calibration

        self.prev = {}  # legacy wall-clock path: track_id -> (center_x, timestamp)
        self._hist = {}  # track_id -> deque[(t, cx, cy)]  (video-time path)
        self._recent = {}  # track_id -> deque[instantaneous kmh]
        self._first_t = {}  # track_id -> first video timestamp seen

    def _ppm(self, y):
        return self.calibration(y) if self.calibration is not None else self.pixels_per_meter

    def calculate_speed(self, track_id, box):
        """Legacy wall-clock estimate (int km/h). Prefer :meth:`estimate` with a
        video timestamp; this remains for live capture and existing callers."""
        x1, _, x2, _ = box
        center_x = (x1 + x2) // 2
        now = time.time()

        speed_kmh = 0

        if track_id in self.prev:
            prev_x, prev_time = self.prev[track_id]
            time_diff = now - prev_time

            if time_diff > 0:
                pixel_distance = abs(center_x - prev_x)
                meters = pixel_distance / self.pixels_per_meter
                speed_kmh = int((meters / time_diff) * 3.6)  # m/s -> km/h

        self.prev[track_id] = (center_x, now)

        return speed_kmh

    def estimate(self, track_id, box, timestamp) -> SpeedEstimate:
        """Rigorous, video-time speed estimate.

        ``timestamp`` is in video seconds (``frame_index / fps``), so the result
        is independent of processing wall-clock time.
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        hist = self._hist.setdefault(
            track_id, deque(maxlen=self.config.smoothing_window + 1)
        )
        recent = self._recent.setdefault(
            track_id, deque(maxlen=self.config.smoothing_window)
        )
        self._first_t.setdefault(track_id, timestamp)

        if hist:
            pt, pcx, pcy = hist[-1]
            dt = timestamp - pt
            if dt > 0:
                ppm = self._ppm((cy + pcy) / 2.0)
                meters = math.hypot(cx - pcx, cy - pcy) / ppm
                inst = (meters / dt) * 3.6
                if 0.0 <= inst <= self.config.max_plausible_kmh:
                    recent.append(inst)  # reject physically impossible jumps
        hist.append((timestamp, cx, cy))

        samples = len(recent)
        duration = timestamp - self._first_t[track_id]
        if samples < self.config.min_samples or duration < self.config.min_duration_s:
            return SpeedEstimate(0.0, 0.0, samples, False)

        vals = list(recent)
        kmh = statistics.median(vals)  # robust to a single bad frame
        uncertainty = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return SpeedEstimate(round(kmh, 1), round(uncertainty, 1), samples, True)
