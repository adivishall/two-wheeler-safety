"""End-to-end validation of the shipped video/photo paths where errors interact.

Everything below runs the REAL ``process_video`` / ``analyze_image`` / Flask
routes — tracker, association, OCR voting, state machines, speed, confidence,
evidence, DB. Only three things are faked, all at the I/O edge:

* the video source (``cv2.VideoCapture`` / ``VideoWriter``) — scripted frames;
* the detector — returns scripted boxes for the current frame;
* EasyOCR — reads the plate id *from the pixels* of the crop it is given.
  Each plate is painted into the frame with a unique grey level, so an OCR call
  on a crop that is NOT the plate (e.g. a stale box after the plate vanished)
  reads nothing and is counted. That is how the stale-box bug is caught.

Scenarios follow the spec's list: side-by-side bikes, violator next to a
compliant rider, plate occlusion, competing OCR readings, disappear/reappear,
simultaneous violations, noisy detector output, missing speed calibration,
downscaled speed, cancellation, and the persisted record behind the API.
"""

from __future__ import annotations

import importlib
import os
import random
import sys
import threading
import time

import numpy as np
import pytest

from modules.video_detector import process_video

H, W = 400, 640
FPS = 25.0
CLS = {"Plate": 0, "WithHelmet": 1, "WithoutHelmet": 2, "TripleRiding": 3}
NAMES = {v: k for k, v in CLS.items()}


def _grey(plate_id: int) -> int:
    return 40 + 30 * plate_id  # distinct, non-zero, < 255 for ids 0..6


class Script:
    """Per-frame detections. ``frames[i]`` is a list of
    ``(label, box, conf, plate_id_or_None)``; plates get painted into the frame."""

    def __init__(self, frames, texts, width=W, height=H, paint_scale=1):
        self.frames = frames
        self.texts = texts  # plate_id -> str | list[str] (read in turn)
        self.width, self.height = width, height
        # Boxes are in PROCESSED-frame pixels (what the detector sees); when the
        # source is larger and gets downscaled, paint at source scale.
        self.paint_scale = paint_scale
        self.i = -1  # index of the frame last handed out
        self.ocr_calls = 0
        self.stale_ocr_calls = 0
        self._text_cursor: dict[int, int] = {}

    def frame(self, i):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for label, box, _conf, pid in self.frames[i]:
            if label == "Plate" and pid is not None:
                k = self.paint_scale
                x1, y1, x2, y2 = (v * k for v in box)
                img[y1:y2, x1:x2] = _grey(pid)
        return img

    def read(self, crop):
        self.ocr_calls += 1
        if crop is None or crop.size == 0:
            self.stale_ocr_calls += 1
            return []
        vals, counts = np.unique(crop[..., 0], return_counts=True)
        v = int(vals[np.argmax(counts)])
        pid = next((p for p in self.texts if _grey(p) == v), None)
        if pid is None:
            self.stale_ocr_calls += 1  # read pixels that are not a plate
            return []
        t = self.texts[pid]
        if isinstance(t, list):
            k = self._text_cursor.get(pid, 0)
            self._text_cursor[pid] = k + 1
            t = t[k % len(t)]
        return [([[0, 0], [1, 0], [1, 1], [0, 1]], t, 0.9)]


class _Box:
    def __init__(self, cls, xyxy, conf):
        self.cls, self.xyxy, self.conf = [cls], [list(xyxy)], [conf]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    names = NAMES

    def __init__(self, script: Script):
        self.script = script
        self.kwargs_seen: list = []

    def __call__(self, frame, **kwargs):
        self.kwargs_seen.append(kwargs)
        dets = self.script.frames[self.script.i]
        return [_Result([_Box(CLS[lb], box, conf) for lb, box, conf, _ in dets])]


class FakeReader:
    def __init__(self, script: Script):
        self.script = script

    def readtext(self, crop, detail=1):
        return self.script.read(crop)


def install_video(monkeypatch, script: Script, *, gate: threading.Event | None = None,
                  block_after: int | None = None):
    import cv2

    class Cap:
        def __init__(self, _path):
            script.i = -1

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return len(script.frames)
            if prop == cv2.CAP_PROP_FPS:
                return FPS
            return 0

        def read(self):
            nxt = script.i + 1
            if nxt >= len(script.frames):
                return False, None
            if gate is not None and block_after is not None and nxt >= block_after:
                gate.wait(timeout=5)
            script.i = nxt
            return True, script.frame(nxt)

        def release(self):
            pass

    class Writer:
        def __init__(self, *a, **k):
            pass

        def write(self, f):
            pass

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Cap)
    monkeypatch.setattr(cv2, "VideoWriter", Writer)


def rider(x, label, pid, *, y=100, w=60, h=100, plate=True, conf=0.85):
    """A bike at x: rider box + (optionally) its plate box under it."""
    out = [(label, (x, y, x + w, y + h), conf, None)]
    if plate:
        out.append(("Plate", (x + 10, y + h + 5, x + w - 10, y + h + 30), 0.9, pid))
    return out


def run(monkeypatch, tmp_path, script, **kw):
    install_video(monkeypatch, script)
    recorded = []

    def record_fn(plate, violation, evidence_path, **extra):
        recorded.append({"plate": plate, "violation": violation,
                         "evidence_path": evidence_path, **extra})
        return 500

    summary = process_video("clip.mp4", FakeModel(script), FakeReader(script),
                            str(tmp_path / "out.mp4"), record_fn=record_fn, **kw)
    return summary, recorded


N = 30  # frames in view (~1.2 s): enough for the >=12-frame helmet gate


# ---------------------------------------------------------------------------
# 1 + 2. two bikes side by side: one violator, one compliant
# ---------------------------------------------------------------------------

def test_side_by_side_fines_only_the_violator_with_its_own_plate(monkeypatch, tmp_path):
    frames = [rider(80 + 3 * f, "WithoutHelmet", 0) + rider(300 + 3 * f, "WithHelmet", 1)
              for f in range(N)]
    script = Script(frames, {0: "MH12AB1234", 1: "KA05MN6789"})
    summary, rec = run(monkeypatch, tmp_path, script)
    assert [(r["plate"], r["violation"]) for r in rec] == [("MH12AB1234", "no_helmet")]
    assert summary["plates_tracked"] == 2


# ---------------------------------------------------------------------------
# 3. plate temporarily occluded: no OCR on a stale box, correct evidence
# ---------------------------------------------------------------------------

def test_plate_occlusion_never_ocrs_a_stale_box(monkeypatch, tmp_path):
    frames = []
    for f in range(N):
        frames.append(rider(80 + 3 * f, "WithoutHelmet", 0, plate=not 8 <= f < 20))
    script = Script(frames, {0: "MH12AB1234"})
    summary, rec = run(monkeypatch, tmp_path, script, ocr_lock_confidence=1.1)
    # Regression: the plate box used to persist after the plate vanished and be
    # cropped + OCR'd every frame of the occlusion, feeding the vote pixels
    # from wherever the plate USED to be.
    assert script.stale_ocr_calls == 0
    assert script.ocr_calls == N - 12
    assert [(r["plate"], r["violation"]) for r in rec] == [("MH12AB1234", "no_helmet")]
    ev = rec[0]["evidence"]
    assert ev["metadata_path"] and os.path.exists(tmp_path / ev["metadata_path"])


# ---------------------------------------------------------------------------
# 4. OCR produces competing readings -> abstain, and say so
# ---------------------------------------------------------------------------

def test_competing_ocr_readings_are_not_fined(monkeypatch, tmp_path):
    frames = [rider(80 + 3 * f, "WithoutHelmet", 0) for f in range(N)]
    # Two valid plates, alternating: never an outright majority.
    script = Script(frames, {0: ["MH12AB1234", "MH12AB1284"]})
    summary, rec = run(monkeypatch, tmp_path, script)
    assert rec == []
    held = summary["unfined_confirmations"]
    assert [(h["violation"], h["reason"]) for h in held] == [("no_helmet", "contested")]


def test_one_bad_ocr_frame_does_not_change_the_plate(monkeypatch, tmp_path):
    frames = [rider(80 + 3 * f, "WithoutHelmet", 0) for f in range(N)]
    reads = ["MH12AB1234"] * 9 + ["MH12AB1284"]  # 1 in 10 reads is wrong
    script = Script(frames, {0: reads})
    _summary, rec = run(monkeypatch, tmp_path, script, ocr_lock_confidence=1.1)
    assert [r["plate"] for r in rec] == ["MH12AB1234"]


# ---------------------------------------------------------------------------
# 5. vehicle disappears and reappears -> one fine, not two
# ---------------------------------------------------------------------------

def test_reappearing_vehicle_is_fined_once(monkeypatch, tmp_path):
    frames = []
    for f in range(80):
        frames.append([] if 30 <= f < 50 else rider(80 + 3 * (f % 50), "WithoutHelmet", 0))
    script = Script(frames, {0: "MH12AB1234"})
    summary, rec = run(monkeypatch, tmp_path, script)
    assert [(r["plate"], r["violation"]) for r in rec] == [("MH12AB1234", "no_helmet")]
    assert [d["violation"] for d in summary["suppressed_duplicates"]] == ["no_helmet"]


# ---------------------------------------------------------------------------
# 6. multiple simultaneous violations on different bikes
# ---------------------------------------------------------------------------

def test_simultaneous_violations_on_two_bikes(monkeypatch, tmp_path):
    frames = [rider(60 + 3 * f, "WithoutHelmet", 0) + rider(320 + 3 * f, "TripleRiding", 1)
              for f in range(N)]
    script = Script(frames, {0: "MH12AB1234", 1: "KA05MN6789"})
    _summary, rec = run(monkeypatch, tmp_path, script)
    assert sorted((r["plate"], r["violation"]) for r in rec) == [
        ("KA05MN6789", "triple_riding"), ("MH12AB1234", "no_helmet")]


# ---------------------------------------------------------------------------
# 7 + 8. noisy detector output / intermittent frames
# ---------------------------------------------------------------------------

def test_noisy_detector_does_not_fine_a_helmeted_rider(monkeypatch, tmp_path):
    rng = random.Random(7)
    frames = []
    for f in range(60):
        r = rng.random()
        if r < 0.15:
            label = "WithoutHelmet"  # the measured val-split flip rate
        elif r < 0.30:
            frames.append(rider(80 + 2 * f, "WithHelmet", 0)[1:])  # rider box missed
            continue
        else:
            label = "WithHelmet"
        frames.append(rider(80 + 2 * f, label, 0))
    script = Script(frames, {0: "MH12AB1234"})
    _summary, rec = run(monkeypatch, tmp_path, script)
    assert rec == []


def test_intermittent_violator_is_still_fined(monkeypatch, tmp_path):
    frames = [rider(80 + 3 * f, "WithoutHelmet", 0) if f % 7 else []
              for f in range(40)]  # every 7th frame drops everything
    script = Script(frames, {0: "MH12AB1234"})
    _summary, rec = run(monkeypatch, tmp_path, script)
    assert [(r["plate"], r["violation"]) for r in rec] == [("MH12AB1234", "no_helmet")]


# ---------------------------------------------------------------------------
# 9. speed: unavailable without calibration; scaled correctly when resized
# ---------------------------------------------------------------------------

def test_no_calibration_means_no_speed_and_no_overspeed(monkeypatch, tmp_path):
    frames = [rider(20 + 12 * f, "WithHelmet", 0) for f in range(N)]
    script = Script(frames, {0: "MH12AB1234"})
    summary, rec = run(monkeypatch, tmp_path, script, speed_limit_kmh=1)
    assert rec == []


def test_speed_calibration_is_scaled_with_the_frame(monkeypatch, tmp_path):
    """Source frames are 1280 wide, processed at 640 (scale 0.5). The plate
    moves 10 processed px/frame = 20 SOURCE px/frame; calibration is 40 source
    px/m, so the true speed is 20/40 m * 25 fps = 12.5 m/s = 45 km/h. The old
    code divided processed pixels by source ppm and reported 22.5 km/h."""
    frames = [rider(20 + 10 * f, "WithHelmet", 0) for f in range(N)]
    script = Script(frames, {0: "MH12AB1234"}, width=1280, height=2 * H, paint_scale=2)
    summary, rec = run(monkeypatch, tmp_path, script, max_width=640,
                       pixels_per_meter=40.0, speed_limit_kmh=40)
    speed = [r for r in rec if r["violation"] == "overspeed"]
    assert len(speed) == 1
    import json

    meta = json.load(open(tmp_path / speed[0]["evidence"]["metadata_path"]))
    assert meta["speed"]["kmh"] == pytest.approx(45.0, abs=0.5)
    assert meta["config_snapshot"]["pixels_per_meter_processed"] == 20.0


# ---------------------------------------------------------------------------
# 10. cancellation mid-run
# ---------------------------------------------------------------------------

def test_cancellation_stops_and_reports_why(monkeypatch, tmp_path):
    frames = [rider(80 + 3 * f, "WithoutHelmet", 0) for f in range(N)]
    script = Script(frames, {0: "MH12AB1234"})
    calls = {"n": 0}

    def cancel_after_5():
        calls["n"] += 1
        return calls["n"] > 5

    summary, rec = run(monkeypatch, tmp_path, script, cancel_check=cancel_after_5)
    assert summary["stopped_reason"] == "cancelled"
    assert summary["frames"] == 5
    assert rec == []  # 5 frames can't meet the helmet gate: nothing half-decided


def test_applied_thresholds_reach_the_detector_and_the_evidence(monkeypatch, tmp_path):
    frames = [rider(80 + 3 * f, "WithoutHelmet", 0) for f in range(N)]
    script = Script(frames, {0: "MH12AB1234"})
    install_video(monkeypatch, script)
    model = FakeModel(script)
    rec = []
    process_video("clip.mp4", model, FakeReader(script), str(tmp_path / "o.mp4"),
                  record_fn=lambda p, v, e, **x: rec.append(x) or 500,
                  conf_threshold=0.4, streak_threshold=4)
    assert model.kwargs_seen[0]["conf"] == 0.4
    import json

    meta = json.load(open(tmp_path / rec[0]["evidence"]["metadata_path"]))
    snap = meta["config_snapshot"]
    assert snap["conf_threshold"] == 0.4 and snap["confirm_window"] == 4
    assert meta["plate_votes"]["stable"] == "MH12AB1234"


# ---------------------------------------------------------------------------
# Photo path: per-vehicle attribution
# ---------------------------------------------------------------------------

class FakeImageModel(FakeModel):
    def predict(self, source=None, **kwargs):
        self.kwargs_seen.append(kwargs)
        return [_Result([_Box(CLS[lb], box, conf) for lb, box, conf, _ in
                         self.script.frames[0]])]


def _photo(tmp_path, dets, texts):
    import cv2

    script = Script([dets], texts)
    path = str(tmp_path / "photo.png")
    cv2.imwrite(path, script.frame(0))
    return path, script


def test_photo_with_two_bikes_fines_the_violators_own_plate(tmp_path):
    from modules.detector import analyze_image

    # The violator's plate is detected FIRST, the compliant rider's LAST: the old
    # photo path fined every violation against the last plate it OCR'd.
    dets = rider(300, "WithHelmet", 1)[:1] + rider(80, "WithoutHelmet", 0) + \
        rider(300, "WithHelmet", 1)[1:]
    path, script = _photo(tmp_path, dets, {0: "MH12AB1234", 1: "KA05MN6789"})
    out = analyze_image(path, FakeImageModel(script), FakeReader(script),
                        evidence_dir=str(tmp_path / "ev"))
    fined = [(v["plate"], r["type"]) for v in out["vehicles"] for r in v["violations"]]
    assert fined == [("MH12AB1234", "no_helmet")]
    assert out["plate"] == "MH12AB1234"


def test_photo_violation_without_a_valid_plate_abstains(tmp_path):
    from modules.detector import analyze_image

    path, script = _photo(tmp_path, rider(80, "WithoutHelmet", 0), {0: "12#4"})
    out = analyze_image(path, FakeImageModel(script), FakeReader(script),
                        evidence_dir=str(tmp_path / "ev"))
    v = out["vehicles"][0]
    assert v["violations"] == []
    assert v["abstained"] == [{"type": "no_helmet", "reason": "plate_not_valid_format"}]


# ---------------------------------------------------------------------------
# Through the web app: job lifecycle, persisted record, API response
# ---------------------------------------------------------------------------

MP4_HEAD = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAFFIC_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "evidence"))
    for k in ("DETECT_API_KEY", "VIEWER_API_KEYS", "REVIEWER_API_KEYS", "ADMIN_API_KEYS"):
        monkeypatch.delenv(k, raising=False)
    app_module = importlib.reload(sys.modules["app"]) if "app" in sys.modules \
        else importlib.import_module("app")
    app_module.app.config["TESTING"] = True
    return app_module


def _wait(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/video_status/{job_id}").get_json()
        if st["status"] != "processing":
            return st
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def _upload(client):
    from io import BytesIO

    return client.post("/analyze_video", data={"video": (BytesIO(MP4_HEAD), "clip.mp4")},
                       content_type="multipart/form-data").get_json()


def test_video_job_persists_a_traceable_record(monkeypatch, tmp_path, app_env):
    frames = [rider(80 + 3 * f, "WithoutHelmet", 0) for f in range(N)]
    script = Script(frames, {0: "MH12AB1234"})
    install_video(monkeypatch, script)
    monkeypatch.setattr(app_env, "get_models",
                        lambda: (FakeModel(script), FakeReader(script)))
    with app_env.app.test_client() as c:
        started = _upload(c)
        st = _wait(c, started["job_id"])
        assert st["status"] == "done"
        session = app_env.db.get_session(started["session_id"])
        assert session["status"] == "completed"
        # processing_fps is measured throughput, not the clip's 25 fps
        assert session["processing_fps"] and session["processing_fps"] != FPS
        items = c.get(f"/api/violations?session_id={started['session_id']}").get_json()
        assert items["total"] == 1
        detail = c.get(f"/api/violations/{items['items'][0]['id']}").get_json()
        # The DB links the whole evidence package, including the JSON sidecar
        # (hashes, model version, thresholds) — not just one image.
        assert detail["evidence"]["metadata"].endswith(".json")
        assert detail["evidence"]["plate_crop"]
        name = detail["evidence"]["metadata"].rsplit("/", 1)[1]
        from modules.evidence import verify_evidence

        assert verify_evidence(str(tmp_path / "evidence"), name).ok


def test_cancelled_video_job_is_not_recorded_as_completed(monkeypatch, tmp_path, app_env):
    frames = [rider(80 + 3 * f, "WithoutHelmet", 0) for f in range(N)]
    script = Script(frames, {0: "MH12AB1234"})
    gate = threading.Event()
    install_video(monkeypatch, script, gate=gate, block_after=3)
    monkeypatch.setattr(app_env, "get_models",
                        lambda: (FakeModel(script), FakeReader(script)))
    with app_env.app.test_client() as c:
        started = _upload(c)
        assert c.post(f"/video_cancel/{started['job_id']}").status_code == 200
        gate.set()
        st = _wait(c, started["job_id"])
        assert st["status"] == "cancelled"
        session = app_env.db.get_session(started["session_id"])
        assert session["status"] == "cancelled"  # was stored as "completed"


def test_renamed_non_video_is_rejected_before_a_worker_starts(app_env):
    from io import BytesIO

    with app_env.app.test_client() as c:
        r = c.post("/analyze_video", data={"video": (BytesIO(b"GIF89a" + b"0" * 40),
                                                     "clip.mp4")},
                   content_type="multipart/form-data")
        assert r.status_code == 400
        assert "not a supported video" in r.get_json()["error"]
