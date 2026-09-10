"""End-to-end video-pipeline integration test with synthetic data.

Exercises the real pipeline — VehicleTracker -> Hungarian association ->
PlateStabilizer (temporal OCR) -> HelmetStateMachine (temporal confirmation) ->
compute_confidence -> build_evidence -> record_fn — with a *fake* model and
reader so no torch/ultralytics/easyocr or weights are needed. Only OpenCV's
video I/O is stubbed (deterministic synthetic frames), everything else is the
production code path.
"""

import numpy as np
import pytest

from modules.video_detector import process_video

FRAME_W, FRAME_H = 320, 400
N_FRAMES = 10

# A no-helmet rider (body) with a plate sitting under it. Same boxes every frame
# so the track is stable and the violation persists to confirmation.
BODY_BOX = (100, 100, 200, 300)      # WithoutHelmet
PLATE_BOX = (120, 280, 180, 320)     # Plate, horizontally under the body
PLATE_TEXT = "MH12AB1234"


class _FakeBox:
    def __init__(self, cls, xyxy, conf):
        self.cls = [cls]
        self.xyxy = [list(xyxy)]
        self.conf = [conf]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Mimics an ultralytics model: model(frame)[0].boxes iterable."""

    names = {0: "Plate", 1: "WithHelmet", 2: "WithoutHelmet", 3: "TripleRiding"}

    def __call__(self, frame, verbose=False):
        return [_FakeResult([
            _FakeBox(0, PLATE_BOX, 0.9),
            _FakeBox(2, BODY_BOX, 0.8),
        ])]


class _FakeReader:
    """Mimics EasyOCR: readtext(crop, detail=1) -> [(bbox, text, conf), ...]."""

    def readtext(self, crop, detail=1):
        return [([[0, 0], [10, 0], [10, 5], [0, 5]], PLATE_TEXT, 0.92)]


class _FakeCapture:
    def __init__(self, path):
        self._i = 0

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return N_FRAMES
        if prop == cv2.CAP_PROP_FPS:
            return 25.0
        return 0

    def read(self):
        if self._i >= N_FRAMES:
            return False, None
        self._i += 1
        return True, np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    def release(self):
        pass


class _FakeWriter:
    def __init__(self, *args, **kwargs):
        self.frames = 0

    def write(self, frame):
        self.frames += 1

    def release(self):
        pass


@pytest.fixture
def stub_video_io(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)


def test_video_pipeline_records_confirmed_no_helmet(stub_video_io, tmp_path):
    recorded = []

    def record_fn(plate, violation, evidence_path):
        recorded.append((plate, violation, evidence_path))
        return 500

    out = str(tmp_path / "out.mp4")
    summary = process_video(
        "unused.mp4", _FakeModel(), _FakeReader(), out,
        record_fn=record_fn, streak_threshold=3,
    )

    # Processed all frames and tracked/confirmed exactly one vehicle.
    assert summary["frames"] == N_FRAMES
    assert summary["plates_tracked"] == 1

    # Exactly one no_helmet violation recorded, for the stabilized plate,
    # de-duplicated across the many frames the rider was in view.
    assert len(summary["violations"]) == 1
    v = summary["violations"][0]
    assert v["violation"] == "no_helmet"
    assert v["plate"] == PLATE_TEXT
    assert 0.0 <= v["confidence"] <= 1.0
    assert set(v["confidence_breakdown"]) >= {
        "detection", "temporal", "association", "ocr", "final"
    }

    # record_fn was called once, and a real evidence file exists on disk.
    assert len(recorded) == 1
    plate, violation, evidence_path = recorded[0]
    assert plate == PLATE_TEXT and violation == "no_helmet"
    import os
    assert os.path.exists(evidence_path)


def test_contradiction_is_not_recorded(stub_video_io, tmp_path):
    """A rider the model calls BOTH helmet and no-helmet is ambiguous and must
    not produce a fine (the contradiction safeguard, at pipeline scope)."""

    class _ContradictingModel(_FakeModel):
        def __call__(self, frame, verbose=False):
            return [_FakeResult([
                _FakeBox(0, PLATE_BOX, 0.9),
                _FakeBox(2, BODY_BOX, 0.8),           # WithoutHelmet
                _FakeBox(1, (98, 98, 202, 302), 0.7),  # WithHelmet, same rider
            ])]

    recorded = []
    out = str(tmp_path / "out.mp4")
    summary = process_video(
        "unused.mp4", _ContradictingModel(), _FakeReader(), out,
        record_fn=lambda *a: recorded.append(a) or 500, streak_threshold=3,
    )
    assert recorded == []
    assert summary["violations"] == []


def test_could_not_open_video_raises(monkeypatch, tmp_path):
    import cv2

    class _Closed(_FakeCapture):
        def isOpened(self):
            return False

    monkeypatch.setattr(cv2, "VideoCapture", _Closed)
    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)
    with pytest.raises(ValueError):
        process_video("nope.mp4", _FakeModel(), _FakeReader(),
                      str(tmp_path / "o.mp4"), record_fn=lambda *a: 0)
