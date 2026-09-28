"""Regression tests for the systems/security issues found in the flagship audit.

Each test pins one genuine defect so it cannot come back silently:

* write routes that recorded fines (/analyze, /analyze_video) and /video_cancel
  were not role-gated even when auth was enabled;
* the payment update validated the old status outside the write, so two
  concurrent requests could overwrite a terminal state;
* ``/api/stats?recent=-1`` became ``LIMIT -1`` (SQLite: no limit) and dumped the
  whole table;
* the rate limiter kept a deque for every IP that ever called, forever;
* a RIFF container (AVI/WAV) passed the image sniffer as "WEBP", and video
  uploads were never content-sniffed at all;
* documented detection env vars were read but never applied.
"""

import importlib
import sys
import threading
from io import BytesIO

import pytest

from modules.db import Database
from modules.validation import RateLimiter, sniff_image, sniff_video


def _client(monkeypatch, tmp_path, env=None):
    monkeypatch.setenv("TRAFFIC_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "evidence"))
    for k in ("DETECT_API_KEY", "VIEWER_API_KEYS", "REVIEWER_API_KEYS", "ADMIN_API_KEYS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    mod = importlib.reload(sys.modules["app"]) if "app" in sys.modules \
        else importlib.import_module("app")
    mod.app.config["TESTING"] = True
    return mod, mod.app.test_client()


@pytest.fixture
def authed(monkeypatch, tmp_path):
    return _client(monkeypatch, tmp_path, {"VIEWER_API_KEYS": "view-key",
                                           "REVIEWER_API_KEYS": "rev-key"})


@pytest.mark.parametrize("route,field,name", [
    ("/analyze", "image", "p.jpg"),
    ("/analyze_video", "video", "c.mp4"),
])
def test_upload_routes_that_record_fines_need_the_reviewer_role(authed, route, field, name):
    _mod, c = authed
    data = {field: (BytesIO(b"x" * 32), name)}
    assert c.post(route, data=data, content_type="multipart/form-data").status_code == 403
    data = {field: (BytesIO(b"x" * 32), name)}
    r = c.post(route, data=data, content_type="multipart/form-data",
               headers={"X-API-Key": "view-key"})
    assert r.status_code == 403  # a viewer can read, not write
    data = {field: (BytesIO(b"x" * 32), name)}
    r = c.post(route, data=data, content_type="multipart/form-data",
               headers={"X-API-Key": "rev-key"})
    assert r.status_code == 400  # past the gate; rejected on content instead


def test_video_cancel_needs_the_reviewer_role(authed):
    _mod, c = authed
    assert c.post("/video_cancel/abc").status_code == 403
    assert c.post("/video_cancel/abc", headers={"X-API-Key": "rev-key"}).status_code == 404


def test_upload_routes_stay_open_without_role_keys(monkeypatch, tmp_path):
    _mod, c = _client(monkeypatch, tmp_path)
    r = c.post("/analyze", data={"image": (BytesIO(b"not an image"), "p.jpg")},
               content_type="multipart/form-data")
    assert r.status_code == 400  # reached validation: local/demo use unchanged


def test_payment_update_is_compare_and_set(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "t.db"))
    db.record_fine("MH12AB1234", "no_helmet", None)
    vid = db.list_violations()["items"][0]["id"]

    # Simulate the race: request A read 'unpaid' and validated 'unpaid -> paid';
    # before it writes, request B moves the row to the terminal 'cancelled'.
    real_valid = Database.valid_payment_transition

    def racing_valid(cls, current, new):
        ok = real_valid(current, new)
        if new == "paid":
            other = Database(db.path)
            conn = other._connect()
            with conn:
                conn.execute("UPDATE violations SET status='cancelled' WHERE id=?", (vid,))
            conn.close()
        return ok

    monkeypatch.setattr(Database, "valid_payment_transition", classmethod(racing_valid))
    with pytest.raises(ValueError, match="concurrently"):
        db.set_payment_status(vid, "paid", actor="a")
    assert db.get_violation(vid)["status"] == "cancelled"  # terminal state kept


def test_concurrent_payment_updates_leave_exactly_one_terminal_state(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.record_fine("MH12AB1234", "no_helmet", None)
    vid = db.list_violations()["items"][0]["id"]
    results = []

    def attempt(status):
        try:
            results.append((status, db.set_payment_status(vid, status)))
        except ValueError:
            results.append((status, False))

    threads = [threading.Thread(target=attempt, args=(s,)) for s in ("paid", "cancelled") * 4]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final = db.get_violation(vid)["status"]
    winners = {s for s, ok in results if ok}
    assert final in {"paid", "cancelled"}
    assert winners == {final}  # nobody "succeeded" into a state that was overwritten


def test_stats_recent_limit_is_clamped(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    for i in range(3):
        db.record_fine(f"MH12AB12{i:02d}", "no_helmet", None)
    assert len(db.stats(recent_limit=-1)["recent"]) == 0
    assert len(db.stats(recent_limit=10**9)["recent"]) == 3


def test_rate_limiter_forgets_idle_clients():
    rl = RateLimiter(max_requests=5, window_s=60.0)
    for i in range(1000):
        rl.allow(f"10.0.{i // 256}.{i % 256}", now=0.0)
    assert rl.tracked_keys() == 1000
    rl.allow("late", now=200.0)  # past the window and the sweep interval
    assert rl.tracked_keys() == 1


def test_rate_limiter_still_limits():
    rl = RateLimiter(max_requests=2, window_s=60.0)
    assert rl.allow("a", now=0) and rl.allow("a", now=1)
    assert not rl.allow("a", now=2)
    assert rl.allow("a", now=61.5)


@pytest.mark.parametrize("head,ok", [
    (b"RIFF\x00\x00\x00\x00WEBPVP8 ", True),
    (b"RIFF\x00\x00\x00\x00AVI LIST", False),  # an AVI is not an image
    (b"RIFF\x00\x00\x00\x00WAVEfmt ", False),
    (b"\xff\xd8\xff\xe0", True),
])
def test_image_sniffer_distinguishes_riff_containers(head, ok):
    assert sniff_image(head) is ok


@pytest.mark.parametrize("head,ok", [
    (b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00", True),   # mp4
    (b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00", True),   # mov
    (b"RIFF\x00\x00\x00\x00AVI LIST", True),
    (b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81", True),  # mkv/webm
    (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01", False),
    (b"short", False),
])
def test_video_sniffer(head, ok):
    assert sniff_video(head) is ok


def test_detection_env_vars_reach_the_pipeline(monkeypatch):
    from modules.config import DetectionConfig
    from modules.pipeline import pipeline_config_from_detection

    for k, v in {"STREAK_THRESHOLD": "7", "HELMET_MIN_CONF": "0.55",
                 "TRIPLE_MIN_CONF": "0.45", "SPEED_LIMIT_KMH": "50",
                 "HELMET_MIN_OBSERVED": "15", "HELMET_MIN_FRACTION": "0.8"}.items():
        monkeypatch.setenv(k, v)
    cfg = pipeline_config_from_detection(DetectionConfig.from_env())
    assert (cfg.confirm_window, cfg.helmet_min_conf, cfg.triple_min_conf,
            cfg.speed_limit_kmh, cfg.helmet_min_observed, cfg.helmet_min_fraction) == \
        (7, 0.55, 0.45, 50.0, 15, 0.8)
    snap = cfg.snapshot()
    assert snap["confirm_window"] == 7 and snap["helmet_min_conf"] == 0.55


def test_verify_endpoint_never_fakes_a_pass_for_records_without_a_sidecar(monkeypatch,
                                                                          tmp_path):
    _mod, c = _client(monkeypatch, tmp_path)
    c.post("/detect", json={"plate": "MH12AB1234", "violation": "no_helmet",
                            "image_path": "evidence/x.jpg"})
    vid = c.get("/api/violations").get_json()["items"][0]["id"]
    r = c.get(f"/api/violations/{vid}/verify").get_json()
    assert r["ok"] is None and r["checked"] == 0
    assert c.get("/api/violations/999999/verify").status_code == 404


@pytest.mark.parametrize("name,ok", [
    ("a_b.json", True), ("../x.json", False), ("x.jpg", False), ("..", False),
    ("dir/x.json", False), ("", False),
])
def test_safe_json_name(name, ok):
    from modules.validation import safe_json_name

    assert (safe_json_name(name) is not None) is ok
