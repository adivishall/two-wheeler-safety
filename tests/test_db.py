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
