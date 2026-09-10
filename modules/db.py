"""SQLite persistence for the fine checker.

The original app used one flat ``fines`` table. This module normalizes storage
into ``vehicles`` / ``violations`` / ``evidence`` / ``processing_jobs`` (plus a
reserved ``detections`` table) while **preserving the existing lookup
behaviour** exactly, so the web UI and its tests are unchanged.

Design points:

* One :class:`Database` object owns a path and opens a fresh connection per
  operation (SQLite is fine with this and it keeps the web threads simple).
  Foreign keys are enforced; related inserts run in a single transaction.
* :meth:`initialize` creates tables/indexes ``IF NOT EXISTS`` and, if a legacy
  ``fines`` table is present and the new schema is empty, migrates those rows
  once. It never drops the legacy table, so existing ``traffic.db`` data
  survives an upgrade.
* Plates are matched on ``normalized_plate`` (letters/digits, upper-cased), so
  lookup stays case/whitespace-insensitive.
"""

from __future__ import annotations

import os
import sqlite3

from modules.config import DEFAULT_FINE, DEFAULT_FINE_AMOUNTS, fine_amount
from modules.plate_info import decode_plate, normalize_plate

# Re-exported from modules.config so existing importers of these names keep
# working; the single source of truth now lives in the config module.
DEFAULT_FINE_AMOUNT = DEFAULT_FINE
__all__ = ["Database", "get_database", "fine_amount", "DEFAULT_FINE_AMOUNTS"]


class Database:
    def __init__(self, path: str):
        self.path = path
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    # -- schema -------------------------------------------------------------

    def initialize(self) -> None:
        conn = self._connect()
        try:
            with conn:  # one transaction for all DDL
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS vehicles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate TEXT NOT NULL,
                        normalized_plate TEXT NOT NULL UNIQUE,
                        registration_state TEXT,
                        rto_code TEXT,
                        rto_name TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS violations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        vehicle_id INTEGER NOT NULL REFERENCES vehicles(id),
                        type TEXT NOT NULL,
                        amount INTEGER NOT NULL,
                        confidence REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        status TEXT NOT NULL DEFAULT 'unpaid',
                        track_id INTEGER
                    );

                    CREATE TABLE IF NOT EXISTS evidence (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        violation_id INTEGER NOT NULL REFERENCES violations(id),
                        original_path TEXT,
                        annotated_path TEXT,
                        plate_crop_path TEXT,
                        metadata_path TEXT
                    );

                    CREATE TABLE IF NOT EXISTS processing_jobs (
                        id TEXT PRIMARY KEY,
                        type TEXT,
                        status TEXT,
                        progress REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        completed_at DATETIME,
                        error TEXT
                    );

                    -- Reserved for future per-detection logging; not written by
                    -- the current pipeline (which aggregates detections into
                    -- confirmed violations before persisting).
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        violation_id INTEGER REFERENCES violations(id),
                        label TEXT,
                        confidence REAL,
                        box TEXT,
                        frame_index INTEGER
                    );

                    CREATE INDEX IF NOT EXISTS idx_vehicles_norm
                        ON vehicles(normalized_plate);
                    CREATE INDEX IF NOT EXISTS idx_violations_vehicle
                        ON violations(vehicle_id);
                    CREATE INDEX IF NOT EXISTS idx_violations_status
                        ON violations(status);
                    CREATE INDEX IF NOT EXISTS idx_evidence_violation
                        ON evidence(violation_id);
                    """
                )
            self._migrate_legacy_fines(conn)
        finally:
            conn.close()

    def _migrate_legacy_fines(self, conn: sqlite3.Connection) -> None:
        has_legacy = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fines'"
        ).fetchone()
        if not has_legacy:
            return
        already = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        if already:
            return  # migrate only into an empty new schema, once
        rows = conn.execute(
            "SELECT plate, violation, amount, image_path, timestamp, status FROM fines"
        ).fetchall()
        for row in rows:
            self._insert_violation(
                conn,
                plate=row["plate"] or "",
                violation=row["violation"],
                amount=row["amount"],
                image_path=row["image_path"],
                timestamp=row["timestamp"],
                status=row["status"],
            )
        conn.commit()

    # -- vehicles + violations ---------------------------------------------

    def _upsert_vehicle(self, conn: sqlite3.Connection, plate: str) -> int:
        norm = normalize_plate(plate)
        info = decode_plate(norm)
        existing = conn.execute(
            "SELECT id FROM vehicles WHERE normalized_plate = ?", (norm,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE vehicles SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (existing["id"],),
            )
            return existing["id"]
        cur = conn.execute(
            """INSERT INTO vehicles
               (plate, normalized_plate, registration_state, rto_code, rto_name)
               VALUES (?, ?, ?, ?, ?)""",
            (
                norm,
                norm,
                info.get("state"),
                info.get("rto_code"),
                info.get("rto"),
            ),
        )
        return cur.lastrowid

    def _insert_violation(
        self,
        conn: sqlite3.Connection,
        *,
        plate: str,
        violation: str,
        amount: int,
        image_path: str | None,
        confidence: float | None = None,
        track_id: int | None = None,
        status: str = "unpaid",
        timestamp: str | None = None,
        evidence: dict | None = None,
    ) -> int:
        vehicle_id = self._upsert_vehicle(conn, plate)
        if timestamp is not None:
            cur = conn.execute(
                """INSERT INTO violations
                   (vehicle_id, type, amount, confidence, status, track_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (vehicle_id, violation, amount, confidence, status, track_id, timestamp),
            )
        else:
            cur = conn.execute(
                """INSERT INTO violations
                   (vehicle_id, type, amount, confidence, status, track_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (vehicle_id, violation, amount, confidence, status, track_id),
            )
        violation_id = cur.lastrowid
        ev = evidence or {}
        conn.execute(
            """INSERT INTO evidence
               (violation_id, original_path, annotated_path, plate_crop_path, metadata_path)
               VALUES (?, ?, ?, ?, ?)""",
            (
                violation_id,
                ev.get("original_path"),
                ev.get("annotated_path", image_path),
                ev.get("plate_crop_path"),
                ev.get("metadata_path"),
            ),
        )
        return violation_id

    def record_fine(
        self,
        plate: str,
        violation: str,
        image_path: str | None,
        *,
        amount: int | None = None,
        confidence: float | None = None,
        track_id: int | None = None,
        status: str = "unpaid",
        timestamp: str | None = None,
        evidence: dict | None = None,
    ) -> int:
        """Insert one fine (vehicle upserted, violation + evidence in a single
        transaction). Returns the amount charged."""
        amt = amount if amount is not None else fine_amount(violation)
        conn = self._connect()
        try:
            with conn:
                self._insert_violation(
                    conn,
                    plate=plate,
                    violation=violation,
                    amount=amt,
                    image_path=image_path,
                    confidence=confidence,
                    track_id=track_id,
                    status=status,
                    timestamp=timestamp,
                    evidence=evidence,
                )
        finally:
            conn.close()
        return amt

    # -- lookups ------------------------------------------------------------

    def get_fines(self, plate: str) -> dict:
        """Return ``{"fines": [...], "total": unpaid_sum}`` for a plate, matching
        the historical response shape (fields: violation, amount, status, time,
        image)."""
        norm = normalize_plate(plate)
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT v.type, v.amount, v.status, v.timestamp,
                          COALESCE(e.annotated_path, e.original_path) AS image_path
                   FROM violations v
                   JOIN vehicles ve ON v.vehicle_id = ve.id
                   LEFT JOIN evidence e ON e.violation_id = v.id
                   WHERE ve.normalized_plate = ?
                   ORDER BY v.timestamp""",
                (norm,),
            ).fetchall()
        finally:
            conn.close()

        fines = []
        total = 0
        for row in rows:
            image_file = (row["image_path"] or "").split("/")[-1]
            fines.append({
                "violation": row["type"],
                "amount": row["amount"],
                "status": row["status"],
                "time": row["timestamp"],
                "image": f"/evidence/{image_file}",
            })
            if row["status"] == "unpaid":
                total += row["amount"]
        return {"fines": fines, "total": total}

    def all_fines(self) -> list:
        """Return every fine in the historical ``/fines`` shape."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT v.id, ve.plate, v.type, v.amount,
                          COALESCE(e.annotated_path, e.original_path) AS image_path,
                          v.timestamp, v.status
                   FROM violations v
                   JOIN vehicles ve ON v.vehicle_id = ve.id
                   LEFT JOIN evidence e ON e.violation_id = v.id
                   ORDER BY v.id""",
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "id": row["id"],
                "plate": row["plate"],
                "violation": row["type"],
                "amount": row["amount"],
                "image_path": row["image_path"],
                "timestamp": row["timestamp"],
                "status": row["status"],
            }
            for row in rows
        ]

    # -- processing jobs ----------------------------------------------------

    def create_job(self, job_id: str, job_type: str) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO processing_jobs (id, type, status, progress) "
                    "VALUES (?, ?, 'processing', 0)",
                    (job_id, job_type),
                )
        finally:
            conn.close()

    def reset(self) -> None:
        """Delete all vehicles/violations/evidence. Used by the demo seeder's
        wipe mode; does not touch the schema or the legacy table."""
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM evidence")
                conn.execute("DELETE FROM violations")
                conn.execute("DELETE FROM vehicles")
        finally:
            conn.close()

    def update_job(self, job_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    f"UPDATE processing_jobs SET {cols} WHERE id = ?",
                    (*fields.values(), job_id),
                )
        finally:
            conn.close()


def get_database(path: str | None = None) -> Database:
    return Database(path or os.environ.get("TRAFFIC_DB_PATH", "traffic.db"))
