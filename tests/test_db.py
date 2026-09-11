"""Normalized database tests, including safe migration of a legacy fines table."""

import sqlite3

from modules.db import Database


def test_record_and_lookup_roundtrip(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    amt = db.record_fine("MH12AB1234", "no_helmet", "evidence/x.jpg")
    assert amt == 500
    res = db.get_fines("MH12AB1234")
    assert len(res["fines"]) == 1
    assert res["fines"][0]["amount"] == 500
    assert res["fines"][0]["image"] == "/evidence/x.jpg"
    assert res["total"] == 500


def test_default_fine_amounts(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    for viol, amt in [("no_helmet", 500), ("triple_riding", 1000),
                      ("overspeed", 700), ("mystery", 300)]:
        db.record_fine("MH12AB1234", viol, "e.jpg")
        got = [f for f in db.get_fines("MH12AB1234")["fines"] if f["violation"] == viol]
        assert got[0]["amount"] == amt


def test_lookup_is_case_and_whitespace_insensitive(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.record_fine("  mh12 ab1234 ", "no_helmet", "e.jpg")
    assert len(db.get_fines("MH12AB1234")["fines"]) == 1
    assert len(db.get_fines("mh12ab1234")["fines"]) == 1


def test_total_counts_only_unpaid(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg")
    db.record_fine("MH12AB1234", "triple_riding", "e.jpg", status="paid")
    res = db.get_fines("MH12AB1234")
    assert res["total"] == 500  # the paid triple_riding is excluded


def test_one_vehicle_row_for_repeated_plate(tmp_path):
    path = str(tmp_path / "t.db")
    db = Database(path)
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg")
    db.record_fine("MH12AB1234", "overspeed", "e.jpg")
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0] == 2
    conn.close()


def test_all_fines_shape(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg")
    rows = db.all_fines()
    assert len(rows) == 1
    assert set(rows[0]) == {
        "id", "plate", "violation", "amount", "image_path", "timestamp", "status"
    }


def test_confidence_and_track_id_persist(tmp_path):
    path = str(tmp_path / "t.db")
    db = Database(path)
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg", confidence=0.97, track_id=3)
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT confidence, track_id FROM violations").fetchone()
    conn.close()
    assert abs(row[0] - 0.97) < 1e-6 and row[1] == 3


def test_analytics_confidence_histogram_and_review_outcomes(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    # Three confidences that fall in distinct 10-way bins, one with no score.
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg", confidence=0.05)
    db.record_fine("MH12AB1234", "overspeed", "e.jpg", confidence=0.55)
    db.record_fine("KA05CD9", "triple_riding", "e.jpg", confidence=1.0)
    db.record_fine("KA05CD9", "no_helmet", "e.jpg")  # confidence NULL

    a = db.analytics(conf_buckets=10)
    conf = a["confidence"]
    assert conf["count"] == 3
    assert conf["missing"] == 1
    assert sum(conf["histogram"]) == 3
    assert conf["histogram"][0] == 1   # 0.05 -> first bin
    assert conf["histogram"][5] == 1   # 0.55 -> sixth bin
    assert conf["histogram"][9] == 1   # 1.0 clamps into the top bin, not past it
    assert abs(conf["mean"] - (0.05 + 0.55 + 1.0) / 3) < 1e-3  # mean rounded to 4dp

    # Review outcomes: confirm two, dismiss one -> rate 2/3 over decided rows.
    ids = [v["id"] for v in db.list_violations()["items"]]
    db.set_review(ids[0], "confirmed")
    db.set_review(ids[1], "confirmed")
    db.set_review(ids[2], "dismissed")
    ro = db.analytics()["review_outcomes"]
    assert ro["confirmed"] == 2 and ro["dismissed"] == 1
    assert ro["decided"] == 3
    assert abs(ro["confirmation_rate"] - 2 / 3) < 1e-3  # rate rounded to 4dp


def test_analytics_sessions_and_over_time(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.create_session("s1", source="clip.mp4", model_version="traffic-4class@1")
    db.update_session("s1", status="completed", processing_fps=12.0,
                      frames_processed=100, vehicles_tracked=4,
                      violations_detected=2)
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg", session_id="s1")

    a = db.analytics()
    assert a["sessions"]["by_status"].get("completed") == 1
    assert a["sessions"]["throughput_fps"]["avg"] == 12.0
    assert a["sessions"]["totals"]["frames"] == 100
    assert a["sessions"]["by_model"][0]["model"] == "traffic-4class@1"
    # over_time has one calendar day with a single violation today.
    assert sum(d["n"] for d in a["over_time"]) == 1


def test_list_violations_filter_by_session(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.create_session("s1")
    db.create_session("s2")
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg", session_id="s1")
    db.record_fine("KA05CD9", "overspeed", "e.jpg", session_id="s2")
    res = db.list_violations(session_id="s1")
    assert res["total"] == 1
    assert res["items"][0]["plate"] == "MH12AB1234"


def test_reset_clears_records_but_keeps_schema(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.record_fine("MH12AB1234", "no_helmet", "e.jpg")
    db.reset()
    assert db.get_fines("MH12AB1234")["fines"] == []
    # schema still usable
    db.record_fine("KA05AB1", "overspeed", "e.jpg")
    assert len(db.all_fines()) == 1


def test_legacy_fines_table_is_migrated(tmp_path):
    # Simulate an existing traffic.db from the old flat schema.
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE fines (
            id INTEGER PRIMARY KEY AUTOINCREMENT, plate TEXT, violation TEXT,
            amount INTEGER, image_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, status TEXT DEFAULT 'unpaid')"""
    )
    conn.execute(
        "INSERT INTO fines (plate, violation, amount, image_path, timestamp, status) "
        "VALUES ('MH02DL4596','no_helmet',500,'evidence/old.jpg','2025-01-01 10:00:00','unpaid')"
    )
    conn.commit()
    conn.close()

    db = Database(path)  # initialize() should migrate the legacy row
    res = db.get_fines("MH02DL4596")
    assert len(res["fines"]) == 1
    assert res["fines"][0]["amount"] == 500
    assert res["fines"][0]["time"] == "2025-01-01 10:00:00"

    # Re-opening must not double-migrate.
    db2 = Database(path)
    assert len(db2.get_fines("MH02DL4596")["fines"]) == 1


def test_jobs_lifecycle(tmp_path):
    path = str(tmp_path / "t.db")
    db = Database(path)
    db.create_job("job1", "video")
    db.update_job("job1", status="done", progress=100)
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT status, progress FROM processing_jobs WHERE id='job1'").fetchone()
    conn.close()
    assert row[0] == "done" and row[1] == 100


# --- dashboard analytics ---------------------------------------------------

def _seed(db):
    db.record_fine("MH12AB1234", "no_helmet", "evidence/a.jpg", confidence=0.91)
    db.record_fine("MH12AB1234", "overspeed", "evidence/b.jpg", confidence=0.55, status="paid")
    db.record_fine("KA05CD9", "triple_riding", "evidence/c.jpg", confidence=0.80)


def test_stats_totals_and_breakdown(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    _seed(db)
    s = db.stats()
    assert s["total_violations"] == 3
    assert s["total_fines"] == 2200
    assert s["unpaid_fines"] == 1500  # 500 + 1000, paid overspeed excluded
    assert s["paid_fines"] == 700
    assert s["vehicles"] == 2
    assert {r["type"] for r in s["by_type"]} == {"no_helmet", "overspeed", "triple_riding"}
    assert s["by_review_status"] == {"pending": 3}


def test_stats_empty_db_is_zeroed(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    s = db.stats()
    assert s["total_violations"] == 0
    assert s["total_fines"] == 0
    assert s["recent"] == []


# --- filtered / paginated listing ------------------------------------------

def test_list_filters_by_type_and_confidence(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    _seed(db)
    assert db.list_violations(violation_type="no_helmet")["total"] == 1
    hi = db.list_violations(min_confidence=0.8)["items"]
    assert all(i["confidence"] >= 0.8 for i in hi)
    assert len(hi) == 2


def test_list_filter_by_plate_is_normalized(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    _seed(db)
    assert db.list_violations(plate="ka05cd9")["total"] == 1
    assert db.list_violations(plate="  ka 05 cd 9 ")["total"] == 1


def test_list_pagination_and_total(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    for i in range(7):
        db.record_fine("MH12AB1234", "no_helmet", "e.jpg")
    page = db.list_violations(limit=3, offset=0)
    assert page["total"] == 7
    assert len(page["items"]) == 3
    assert db.list_violations(limit=3, offset=6)["items"].__len__() == 1


def test_list_limit_is_clamped(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    _seed(db)
    # An absurd limit is clamped, never an unbounded dump.
    assert db.list_violations(limit=100000)["limit"] <= 500


def test_list_sort_whitelist_rejects_injection(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    _seed(db)
    # A non-whitelisted sort column silently falls back to id (no SQL error).
    res = db.list_violations(sort="amount; DROP TABLE violations")
    assert res["total"] == 3


# --- detail + review -------------------------------------------------------

def test_get_violation_detail_has_evidence(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    _seed(db)
    vid = db.list_violations()["items"][0]["id"]
    detail = db.get_violation(vid)
    assert detail["plate"]
    assert "evidence" in detail and set(detail["evidence"]) == {
        "original", "annotated", "plate_crop", "metadata"
    }
    assert db.get_violation(999999) is None


def test_set_review_transitions_and_validates(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    _seed(db)
    vid = db.list_violations()["items"][0]["id"]
    assert db.set_review(vid, "confirmed", reviewer_decision="valid", notes="clear") is True
    d = db.get_violation(vid)
    assert d["review_status"] == "confirmed"
    assert d["reviewed_at"] is not None
    # reset to pending clears the timestamp
    db.set_review(vid, "pending")
    assert db.get_violation(vid)["reviewed_at"] is None
    # unknown state rejected, missing id -> False
    import pytest
    with pytest.raises(ValueError):
        db.set_review(vid, "bogus")
    assert db.set_review(999999, "confirmed") is False


def test_review_columns_added_to_legacy_normalized_db(tmp_path):
    """A normalized DB created before the review workflow gets the columns
    added by migration (ALTER TABLE), not a broken query."""
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE vehicles (id INTEGER PRIMARY KEY AUTOINCREMENT, plate TEXT,
            normalized_plate TEXT UNIQUE, registration_state TEXT, rto_code TEXT,
            rto_name TEXT, created_at DATETIME, updated_at DATETIME);
        CREATE TABLE violations (id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER, type TEXT, amount INTEGER, confidence REAL,
            timestamp DATETIME, status TEXT DEFAULT 'unpaid', track_id INTEGER);
        CREATE TABLE evidence (id INTEGER PRIMARY KEY AUTOINCREMENT,
            violation_id INTEGER, original_path TEXT, annotated_path TEXT,
            plate_crop_path TEXT, metadata_path TEXT);
        INSERT INTO vehicles (plate, normalized_plate) VALUES ('MH01XX1','MH01XX1');
        INSERT INTO violations (vehicle_id, type, amount, status)
            VALUES (1,'no_helmet',500,'unpaid');
        """
    )
    conn.commit()
    conn.close()

    db = Database(path)  # initialize() should ALTER in the review columns
    row = db.list_violations()["items"][0]
    assert row["review_status"] == "pending"
    assert db.set_review(1, "dismissed", notes="false positive") is True
