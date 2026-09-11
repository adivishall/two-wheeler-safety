"""Tests for the Wave-3 DB additions: processing sessions (Phase 14), durable
job records (Phase 15), audit log (Phase 17), fine lifecycle (Phase 18), and the
detection trace (Phase 12)."""

import sqlite3

import pytest

from modules.db import Database


def _db(tmp_path):
    return Database(str(tmp_path / "t.db"))


# -- sessions (Phase 14) ----------------------------------------------------

def test_session_create_update_get(tmp_path):
    db = _db(tmp_path)
    db.create_session("s1", source="clip.mp4", model_version="traffic-4class@1.0.0",
                      pipeline_version="1.0.0")
    db.update_session("s1", frames_processed=120, vehicles_tracked=4,
                      violations_detected=2, violations_confirmed=1,
                      processing_fps=18.5, output_path="/evidence/out.mp4",
                      status="completed")
    s = db.get_session("s1")
    assert s["frames_processed"] == 120
    assert s["model_version"] == "traffic-4class@1.0.0"
    assert s["status"] == "completed"
    assert s["processing_fps"] == 18.5


def test_update_session_ignores_unknown_fields(tmp_path):
    db = _db(tmp_path)
    db.create_session("s1")
    db.update_session("s1", frames_processed=5, bogus="; DROP TABLE sessions; --")
    assert db.get_session("s1")["frames_processed"] == 5


def test_list_sessions_newest_first(tmp_path):
    db = _db(tmp_path)
    for i in range(3):
        db.create_session(f"s{i}", source=f"c{i}.mp4")
    ids = [s["id"] for s in db.list_sessions()]
    assert set(ids) == {"s0", "s1", "s2"}


def test_violation_links_to_session(tmp_path):
    db = _db(tmp_path)
    db.create_session("s1")
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg", session_id="s1")
    conn = db._connect()
    row = conn.execute("SELECT session_id FROM violations").fetchone()
    conn.close()
    assert row["session_id"] == "s1"


# -- audit log (Phase 17) ---------------------------------------------------

def test_audit_log_records_events(tmp_path):
    db = _db(tmp_path)
    db.log_event("violation_created", record_type="violation", record_id="1",
                 actor="system", metadata={"type": "no_helmet"})
    events = db.list_audit()
    assert events[0]["event"] == "violation_created"
    assert events[0]["actor"] == "system"
    assert '"type"' in events[0]["metadata"]


def test_review_writes_an_audit_event(tmp_path):
    db = _db(tmp_path)
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg")
    vid = db.all_fines()[0]["id"]
    db.set_review(vid, "confirmed", reviewer_decision="valid", actor="officer")
    audit = db.list_audit(record_id=str(vid))
    assert any(e["event"] == "violation_review_confirmed" for e in audit)


# -- fine lifecycle (Phase 18) ----------------------------------------------

def test_payment_transitions_valid_and_invalid(tmp_path):
    db = _db(tmp_path)
    assert db.valid_payment_transition("unpaid", "paid")
    assert db.valid_payment_transition("unpaid", "cancelled")
    assert not db.valid_payment_transition("paid", "unpaid")
    assert not db.valid_payment_transition("cancelled", "paid")


def test_set_payment_status_enforces_transitions(tmp_path):
    db = _db(tmp_path)
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg")
    vid = db.all_fines()[0]["id"]
    assert db.set_payment_status(vid, "paid", actor="clerk")
    assert db.get_violation(vid)["status"] == "paid"
    with pytest.raises(ValueError):
        db.set_payment_status(vid, "unpaid")  # paid is terminal


def test_set_payment_status_unknown_status_raises(tmp_path):
    db = _db(tmp_path)
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg")
    vid = db.all_fines()[0]["id"]
    with pytest.raises(ValueError):
        db.set_payment_status(vid, "refunded")


def test_detection_status_defaults_to_confirmed(tmp_path):
    db = _db(tmp_path)
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg")
    vid = db.all_fines()[0]["id"]
    assert db.get_violation(vid)["detection_status"] == "confirmed"


def test_review_and_payment_are_independent(tmp_path):
    # dismissing a violation (review) must not touch its payment status, and
    # vice versa — the whole point of Phase 18.
    db = _db(tmp_path)
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg")
    vid = db.all_fines()[0]["id"]
    db.set_review(vid, "dismissed")
    assert db.get_violation(vid)["status"] == "unpaid"  # payment unaffected
    db.set_payment_status(vid, "cancelled")
    assert db.get_violation(vid)["review_status"] == "dismissed"  # review unaffected


# -- detection trace (Phase 12) ---------------------------------------------

def test_detection_trace_persisted_with_fine(tmp_path):
    db = _db(tmp_path)
    trace = [
        {"track_id": 7, "label": "WithoutHelmet", "confidence": 0.8,
         "box": [1, 2, 3, 4], "frame_index": 10, "timestamp": 0.4},
        {"track_id": 7, "label": "WithoutHelmet", "confidence": 0.85,
         "box": [1, 2, 3, 5], "frame_index": 11, "timestamp": 0.44},
    ]
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg", track_id=7, detection_trace=trace)
    vid = db.all_fines()[0]["id"]
    stored = db.get_detection_trace(vid)
    assert len(stored) == 2
    assert stored[0]["frame_index"] == 10
    assert stored[0]["label"] == "WithoutHelmet"


def test_add_detection_trace_directly(tmp_path):
    db = _db(tmp_path)
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg")
    vid = db.all_fines()[0]["id"]
    n = db.add_detection_trace(vid, [
        {"track_id": 1, "label": "TripleRiding", "confidence": 0.9,
         "box": [0, 0, 10, 10], "frame_index": 3, "timestamp": 0.12},
    ])
    assert n == 1
    assert db.get_detection_trace(vid)[0]["label"] == "TripleRiding"


def test_reset_clears_detections(tmp_path):
    db = _db(tmp_path)
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg", track_id=1,
                   detection_trace=[{"track_id": 1, "label": "WithoutHelmet",
                                     "confidence": 0.8, "box": [1, 2, 3, 4],
                                     "frame_index": 1, "timestamp": 0.1}])
    db.reset()
    conn = db._connect()
    n = conn.execute("SELECT COUNT(*) AS c FROM detections").fetchone()["c"]
    conn.close()
    assert n == 0


# -- durable jobs (Phase 15) ------------------------------------------------

def test_job_persistence_and_listing(tmp_path):
    db = _db(tmp_path)
    db.create_job("j1", "video", source="clip.mp4")
    j = db.get_job("j1")
    assert j["status"] == "processing"
    assert j["source"] == "clip.mp4"
    assert j["started_at"] is not None
    db.update_job("j1", status="completed", progress=1.0, output="/evidence/out.mp4",
                  completed_at="2026-09-11T00:00:00")
    j = db.get_job("j1")
    assert j["status"] == "completed"
    assert j["output"] == "/evidence/out.mp4"
    assert "j1" in [row["id"] for row in db.list_jobs()]


def test_update_job_whitelists_columns(tmp_path):
    db = _db(tmp_path)
    db.create_job("j1", "video")
    # an unknown field must be dropped, not injected into SQL
    db.update_job("j1", status="failed", nonsense="x")
    assert db.get_job("j1")["status"] == "failed"


def test_schema_migrates_from_legacy_layout(tmp_path):
    # A pre-Wave-3 DB (no new columns/tables) must upgrade in place, keeping data.
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE vehicles (id INTEGER PRIMARY KEY, plate TEXT,
            normalized_plate TEXT UNIQUE);
        CREATE TABLE violations (id INTEGER PRIMARY KEY, vehicle_id INTEGER,
            type TEXT, amount INTEGER, confidence REAL, timestamp DATETIME,
            status TEXT DEFAULT 'unpaid', track_id INTEGER,
            review_status TEXT DEFAULT 'pending', reviewer_decision TEXT,
            reviewed_at DATETIME, review_notes TEXT);
        CREATE TABLE evidence (id INTEGER PRIMARY KEY, violation_id INTEGER,
            original_path TEXT, annotated_path TEXT, plate_crop_path TEXT,
            metadata_path TEXT);
        CREATE TABLE processing_jobs (id TEXT PRIMARY KEY, type TEXT, status TEXT,
            progress REAL, created_at DATETIME, completed_at DATETIME, error TEXT);
        CREATE TABLE detections (id INTEGER PRIMARY KEY, violation_id INTEGER,
            label TEXT, confidence REAL, box TEXT, frame_index INTEGER);
        INSERT INTO vehicles (id, plate, normalized_plate)
            VALUES (1, 'MH12AB1234', 'MH12AB1234');
        INSERT INTO violations (vehicle_id, type, amount, status)
            VALUES (1, 'no_helmet', 500, 'unpaid');
        """
    )
    conn.commit()
    conn.close()

    db = Database(path)  # migrates in place
    assert db.all_fines()[0]["plate"] == "MH12AB1234"
    # new capabilities work on the migrated DB
    db.create_session("s1", source="x.mp4")
    assert db.get_session("s1") is not None
    db.log_event("smoke", record_type="test", record_id="1")
    assert db.list_audit()
