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
                        -- Payment lifecycle (kept as `status` for backwards
                        -- compatibility): unpaid -> paid | cancelled.
                        status TEXT NOT NULL DEFAULT 'unpaid',
                        track_id INTEGER,
                        -- Three independent lifecycles, never overloaded onto
                        -- one field (Phase 18):
                        --  detection_status: the pipeline's own state
                        --      (confirmed when persisted).
                        --  review_status: human-in-the-loop
                        --      pending -> confirmed | dismissed.
                        --  status (above): payment.
                        detection_status TEXT NOT NULL DEFAULT 'confirmed',
                        review_status TEXT NOT NULL DEFAULT 'pending',
                        reviewer_decision TEXT,
                        reviewed_at DATETIME,
                        review_notes TEXT,
                        session_id TEXT REFERENCES sessions(id)
                    );

                    -- One processing session per source run (Phase 14): ties a
                    -- video/image run to the violations, model, and metrics it
                    -- produced.
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        source TEXT,
                        started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        ended_at DATETIME,
                        model_version TEXT,
                        pipeline_version TEXT,
                        frames_processed INTEGER DEFAULT 0,
                        vehicles_tracked INTEGER DEFAULT 0,
                        violations_detected INTEGER DEFAULT 0,
                        violations_confirmed INTEGER DEFAULT 0,
                        violations_dismissed INTEGER DEFAULT 0,
                        processing_fps REAL,
                        output_path TEXT,
                        status TEXT DEFAULT 'processing',
                        error TEXT
                    );

                    -- Append-only audit trail for important state changes
                    -- (Phase 17): who did what, when. Separate from model
                    -- evidence.
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event TEXT NOT NULL,
                        record_type TEXT,
                        record_id TEXT,
                        actor TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    );

                    CREATE TABLE IF NOT EXISTS evidence (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        violation_id INTEGER NOT NULL REFERENCES violations(id),
                        original_path TEXT,
                        annotated_path TEXT,
                        plate_crop_path TEXT,
                        metadata_path TEXT
                    );

                    -- Durable job metadata (Phase 15). Worker state stays
                    -- in-process (JobManager); this table is the persisted
                    -- record for observability and restart diagnostics.
                    CREATE TABLE IF NOT EXISTS processing_jobs (
                        id TEXT PRIMARY KEY,
                        type TEXT,
                        status TEXT,
                        progress REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        started_at DATETIME,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        completed_at DATETIME,
                        error TEXT,
                        output TEXT,
                        source TEXT
                    );

                    -- Per-detection trace supporting a confirmed violation
                    -- (Phase 12): lets an engineer reconstruct why a violation
                    -- was confirmed. Bounded per violation (see config); NOT a
                    -- log of every detection in a video.
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        violation_id INTEGER REFERENCES violations(id),
                        track_id INTEGER,
                        label TEXT,
                        confidence REAL,
                        box TEXT,
                        frame_index INTEGER,
                        timestamp REAL
                    );

                    CREATE INDEX IF NOT EXISTS idx_vehicles_norm
                        ON vehicles(normalized_plate);
                    CREATE INDEX IF NOT EXISTS idx_violations_vehicle
                        ON violations(vehicle_id);
                    CREATE INDEX IF NOT EXISTS idx_violations_status
                        ON violations(status);
                    CREATE INDEX IF NOT EXISTS idx_violations_type
                        ON violations(type);
                    CREATE INDEX IF NOT EXISTS idx_violations_timestamp
                        ON violations(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_evidence_violation
                        ON evidence(violation_id);
                    """
                )
                self._migrate_add_columns(conn)
                # The review-status index references a column that only exists
                # after the migration above, so create it separately.
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_violations_review "
                    "ON violations(review_status)"
                )
            self._migrate_legacy_fines(conn)
        finally:
            conn.close()

    def _migrate_add_columns(self, conn: sqlite3.Connection) -> None:
        """Add columns introduced after the first normalized schema shipped.

        A ``traffic.db`` created before the review workflow existed won't have
        the review columns; ``CREATE TABLE IF NOT EXISTS`` never alters an
        existing table, so add any missing columns here (idempotent, run inside
        ``initialize``'s transaction). New databases already have them from the
        CREATE above and skip every branch.
        """
        self._add_missing_columns(conn, "violations", {
            "review_status": "TEXT NOT NULL DEFAULT 'pending'",
            "reviewer_decision": "TEXT",
            "reviewed_at": "DATETIME",
            "review_notes": "TEXT",
            "detection_status": "TEXT NOT NULL DEFAULT 'confirmed'",
            "session_id": "TEXT",
        })
        self._add_missing_columns(conn, "processing_jobs", {
            "started_at": "DATETIME",
            "updated_at": "DATETIME",
            "output": "TEXT",
            "source": "TEXT",
        })
        self._add_missing_columns(conn, "detections", {
            "track_id": "INTEGER",
            "timestamp": "REAL",
        })

    @staticmethod
    def _add_missing_columns(conn, table: str, columns: dict) -> None:
        """Idempotently ALTER TABLE ADD COLUMN for any column not already present
        (CREATE TABLE IF NOT EXISTS never alters an existing table)."""
        existing = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for column, decl in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

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
        session_id: str | None = None,
    ) -> int:
        vehicle_id = self._upsert_vehicle(conn, plate)
        if timestamp is not None:
            cur = conn.execute(
                """INSERT INTO violations
                   (vehicle_id, type, amount, confidence, status, track_id,
                    session_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (vehicle_id, violation, amount, confidence, status, track_id,
                 session_id, timestamp),
            )
        else:
            cur = conn.execute(
                """INSERT INTO violations
                   (vehicle_id, type, amount, confidence, status, track_id, session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (vehicle_id, violation, amount, confidence, status, track_id, session_id),
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
        session_id: str | None = None,
        detection_trace: list | None = None,
    ) -> int:
        """Insert one fine (vehicle upserted, violation + evidence + optional
        detection trace in a single transaction). Returns the amount charged."""
        import json as _json
        amt = amount if amount is not None else fine_amount(violation)
        conn = self._connect()
        try:
            with conn:
                violation_id = self._insert_violation(
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
                    session_id=session_id,
                )
                for r in detection_trace or []:
                    box = r.get("box")
                    conn.execute(
                        """INSERT INTO detections
                           (violation_id, track_id, label, confidence, box,
                            frame_index, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (violation_id, r.get("track_id"), r.get("label"),
                         r.get("confidence"),
                         _json.dumps(box) if box is not None else None,
                         r.get("frame_index"), r.get("timestamp")),
                    )
                self._log_event(
                    conn, "violation_created", "violation", str(violation_id),
                    "system", {"type": violation, "amount": amt,
                               "session_id": session_id},
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

    # -- dashboard analytics ------------------------------------------------

    def stats(self, recent_limit: int = 5) -> dict:
        """Aggregate counts for the dashboard overview, computed from the real
        data (never fabricated). Returns totals, a per-type breakdown, a
        review-status breakdown, the vehicle count, and the most recent
        violations."""
        conn = self._connect()
        try:
            totals = conn.execute(
                """SELECT
                     COUNT(*) AS total_violations,
                     COALESCE(SUM(amount), 0) AS total_fines,
                     COALESCE(SUM(CASE WHEN status='unpaid' THEN amount ELSE 0 END), 0)
                         AS unpaid_fines,
                     COALESCE(SUM(CASE WHEN status='paid' THEN amount ELSE 0 END), 0)
                         AS paid_fines
                   FROM violations"""
            ).fetchone()

            by_type = conn.execute(
                "SELECT type, COUNT(*) AS n, COALESCE(SUM(amount),0) AS amount "
                "FROM violations GROUP BY type ORDER BY n DESC"
            ).fetchall()

            by_review = conn.execute(
                "SELECT review_status, COUNT(*) AS n "
                "FROM violations GROUP BY review_status"
            ).fetchall()

            vehicles = conn.execute(
                "SELECT COUNT(*) AS n FROM vehicles"
            ).fetchone()["n"]

            recent = conn.execute(
                """SELECT v.id, ve.plate, v.type, v.amount, v.status,
                          v.confidence, v.timestamp, v.review_status
                   FROM violations v JOIN vehicles ve ON v.vehicle_id = ve.id
                   ORDER BY v.id DESC LIMIT ?""",
                (recent_limit,),
            ).fetchall()
        finally:
            conn.close()

        return {
            "total_violations": totals["total_violations"],
            "total_fines": totals["total_fines"],
            "unpaid_fines": totals["unpaid_fines"],
            "paid_fines": totals["paid_fines"],
            "vehicles": vehicles,
            "by_type": [dict(r) for r in by_type],
            "by_review_status": {r["review_status"]: r["n"] for r in by_review},
            "recent": [dict(r) for r in recent],
        }

    def analytics(self, *, days: int = 30, conf_buckets: int = 10) -> dict:
        """Deeper dashboard analytics, all computed from real rows (never
        fabricated). Returns:

        * ``over_time``   — violations + fine total per calendar day (UTC),
          oldest first, for the last ``days`` days that have data.
        * ``by_type`` / ``by_payment_status`` / ``by_review_status`` — counts.
        * ``review_outcomes`` — pending / confirmed / dismissed with the
          confirmation rate over *decided* rows (dismissed counts against it).
        * ``confidence`` — a histogram of the stored confidence scores over
          ``conf_buckets`` equal bins in [0, 1], plus count/mean and how many
          rows carried no score. Scores are raw model outputs, NOT calibrated
          probabilities (see docs/EVALUATION.md).
        * ``plate_recognition`` — how many vehicles decoded to a registration
          region vs. not (an honest proxy for OCR/plate success on stored
          fines; true OCR accuracy is measured offline by evaluate_ocr.py).
        * ``sessions`` — run counts by status, throughput (processing_fps)
          summary over completed runs, totals, and a per-model breakdown.
        """
        conf_buckets = max(1, min(int(conf_buckets), 50))
        days = max(1, min(int(days), 3650))
        conn = self._connect()
        try:
            over_time = conn.execute(
                """SELECT date(timestamp) AS day, COUNT(*) AS n,
                          COALESCE(SUM(amount), 0) AS amount
                   FROM violations
                   WHERE timestamp >= date('now', ?)
                   GROUP BY day ORDER BY day""",
                (f"-{days - 1} days",),
            ).fetchall()

            by_type = conn.execute(
                "SELECT type, COUNT(*) AS n, COALESCE(SUM(amount),0) AS amount "
                "FROM violations GROUP BY type ORDER BY n DESC"
            ).fetchall()
            by_payment = conn.execute(
                "SELECT status, COUNT(*) AS n FROM violations GROUP BY status"
            ).fetchall()
            by_review = conn.execute(
                "SELECT review_status, COUNT(*) AS n "
                "FROM violations GROUP BY review_status"
            ).fetchall()

            # Confidence histogram. A score of exactly 1.0 lands in the top bin
            # rather than a phantom bucket past the end.
            conf_rows = conn.execute(
                "SELECT confidence FROM violations WHERE confidence IS NOT NULL"
            ).fetchall()
            missing_conf = conn.execute(
                "SELECT COUNT(*) AS n FROM violations WHERE confidence IS NULL"
            ).fetchone()["n"]

            plate_reco = conn.execute(
                """SELECT
                     SUM(CASE WHEN registration_state IS NOT NULL
                              AND registration_state != '' THEN 1 ELSE 0 END) AS decoded,
                     COUNT(*) AS total
                   FROM vehicles"""
            ).fetchone()

            sess_by_status = conn.execute(
                "SELECT status, COUNT(*) AS n FROM sessions GROUP BY status"
            ).fetchall()
            sess_fps = conn.execute(
                """SELECT COUNT(*) AS n, AVG(processing_fps) AS avg_fps,
                          MIN(processing_fps) AS min_fps, MAX(processing_fps) AS max_fps
                   FROM sessions
                   WHERE status='completed' AND processing_fps IS NOT NULL"""
            ).fetchone()
            sess_totals = conn.execute(
                """SELECT COALESCE(SUM(frames_processed),0) AS frames,
                          COALESCE(SUM(vehicles_tracked),0) AS vehicles,
                          COALESCE(SUM(violations_detected),0) AS violations
                   FROM sessions"""
            ).fetchone()
            sess_by_model = conn.execute(
                """SELECT COALESCE(model_version, 'unknown') AS model,
                          COUNT(*) AS runs
                   FROM sessions GROUP BY model_version ORDER BY runs DESC"""
            ).fetchall()
        finally:
            conn.close()

        # Bucket confidence in Python: portable and exact about the [1.0] edge.
        hist = [0] * conf_buckets
        conf_values = [r["confidence"] for r in conf_rows]
        for score in conf_values:
            idx = int(score * conf_buckets)
            if idx >= conf_buckets:
                idx = conf_buckets - 1
            if idx < 0:
                idx = 0
            hist[idx] += 1
        confidence = {
            "buckets": conf_buckets,
            "histogram": hist,
            "edges": [round(i / conf_buckets, 3) for i in range(conf_buckets + 1)],
            "count": len(conf_values),
            "missing": missing_conf,
            "mean": round(sum(conf_values) / len(conf_values), 4) if conf_values else None,
        }

        def _round(value, ndigits=3):
            return round(value, ndigits) if value is not None else None

        review_map = {r["review_status"]: r["n"] for r in by_review}
        confirmed = review_map.get("confirmed", 0)
        dismissed = review_map.get("dismissed", 0)
        decided = confirmed + dismissed
        review_outcomes = {
            "pending": review_map.get("pending", 0),
            "confirmed": confirmed,
            "dismissed": dismissed,
            "decided": decided,
            "confirmation_rate": round(confirmed / decided, 4) if decided else None,
        }

        decoded = (plate_reco["decoded"] or 0) if plate_reco else 0
        veh_total = (plate_reco["total"] or 0) if plate_reco else 0

        return {
            "over_time": [dict(r) for r in over_time],
            "by_type": [dict(r) for r in by_type],
            "by_payment_status": {r["status"]: r["n"] for r in by_payment},
            "by_review_status": review_map,
            "review_outcomes": review_outcomes,
            "confidence": confidence,
            "plate_recognition": {
                "decoded": decoded,
                "total": veh_total,
                "rate": round(decoded / veh_total, 4) if veh_total else None,
            },
            "sessions": {
                "by_status": {r["status"]: r["n"] for r in sess_by_status},
                "throughput_fps": {
                    "runs": sess_fps["n"],
                    "avg": _round(sess_fps["avg_fps"]),
                    "min": _round(sess_fps["min_fps"]),
                    "max": _round(sess_fps["max_fps"]),
                },
                "totals": {
                    "frames": sess_totals["frames"],
                    "vehicles_tracked": sess_totals["vehicles"],
                    "violations_detected": sess_totals["violations"],
                },
                "by_model": [dict(r) for r in sess_by_model],
            },
        }

    # -- filtered / paginated listing ---------------------------------------

    # Columns a caller may sort by, whitelisted so ``sort`` can never inject SQL.
    _SORTABLE = {"id", "timestamp", "amount", "confidence", "type", "status"}

    def list_violations(
        self,
        *,
        plate: str | None = None,
        violation_type: str | None = None,
        status: str | None = None,
        review_status: str | None = None,
        session_id: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        sort: str = "id",
        descending: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Return ``{"items": [...], "total": n, "limit": l, "offset": o}``.

        Every filter is optional and applied with a parameterized WHERE clause
        (no string interpolation of values); ``limit`` is clamped to a sane
        maximum so a caller can never request an unbounded dump. Confidence
        filters ignore rows with a NULL confidence, since a range check on NULL
        is undefined.
        """
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        sort_col = sort if sort in self._SORTABLE else "id"
        direction = "DESC" if descending else "ASC"

        where = []
        params: list = []
        if plate:
            where.append("ve.normalized_plate = ?")
            params.append(normalize_plate(plate))
        if violation_type:
            where.append("v.type = ?")
            params.append(violation_type)
        if status:
            where.append("v.status = ?")
            params.append(status)
        if review_status:
            where.append("v.review_status = ?")
            params.append(review_status)
        if session_id:
            where.append("v.session_id = ?")
            params.append(session_id)
        if min_confidence is not None:
            where.append("v.confidence IS NOT NULL AND v.confidence >= ?")
            params.append(float(min_confidence))
        if max_confidence is not None:
            where.append("v.confidence IS NOT NULL AND v.confidence <= ?")
            params.append(float(max_confidence))
        if date_from:
            where.append("v.timestamp >= ?")
            params.append(date_from)
        if date_to:
            where.append("v.timestamp <= ?")
            params.append(date_to)

        clause = (" WHERE " + " AND ".join(where)) if where else ""

        conn = self._connect()
        try:
            total = conn.execute(
                f"SELECT COUNT(*) AS n FROM violations v "
                f"JOIN vehicles ve ON v.vehicle_id = ve.id{clause}",
                params,
            ).fetchone()["n"]

            rows = conn.execute(
                f"""SELECT v.id, ve.plate, ve.registration_state, ve.rto_name,
                           v.type, v.amount, v.status, v.confidence, v.timestamp,
                           v.track_id, v.review_status, v.reviewer_decision,
                           v.reviewed_at, v.review_notes,
                           COALESCE(e.annotated_path, e.original_path) AS image_path
                    FROM violations v
                    JOIN vehicles ve ON v.vehicle_id = ve.id
                    LEFT JOIN evidence e ON e.violation_id = v.id
                    {clause}
                    ORDER BY v.{sort_col} {direction}
                    LIMIT ? OFFSET ?""",
                (*params, limit, offset),
            ).fetchall()
        finally:
            conn.close()

        items = []
        for row in rows:
            item = dict(row)
            image_file = (item.pop("image_path") or "").split("/")[-1]
            item["evidence"] = f"/evidence/{image_file}" if image_file else None
            items.append(item)

        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def get_violation(self, violation_id: int) -> dict | None:
        """Full detail for one violation, including its evidence package paths
        (served under ``/evidence/<name>``). Returns None if not found."""
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT v.*, ve.plate, ve.registration_state, ve.rto_code,
                          ve.rto_name
                   FROM violations v JOIN vehicles ve ON v.vehicle_id = ve.id
                   WHERE v.id = ?""",
                (violation_id,),
            ).fetchone()
            if row is None:
                return None
            ev = conn.execute(
                "SELECT original_path, annotated_path, plate_crop_path, "
                "metadata_path FROM evidence WHERE violation_id = ?",
                (violation_id,),
            ).fetchone()
        finally:
            conn.close()

        result = dict(row)

        def _url(name):
            base = (name or "").split("/")[-1]
            return f"/evidence/{base}" if base else None

        result["evidence"] = {
            "original": _url(ev["original_path"]) if ev else None,
            "annotated": _url(ev["annotated_path"]) if ev else None,
            "plate_crop": _url(ev["plate_crop_path"]) if ev else None,
            "metadata": _url(ev["metadata_path"]) if ev else None,
        }
        return result

    # -- human review -------------------------------------------------------

    REVIEW_STATES = {"pending", "confirmed", "dismissed"}

    def set_review(
        self,
        violation_id: int,
        review_status: str,
        *,
        reviewer_decision: str | None = None,
        notes: str | None = None,
        actor: str | None = None,
    ) -> bool:
        """Record a human review decision on a violation. Returns True if a row
        was updated, False if the id doesn't exist. Raises ValueError for an
        unknown ``review_status`` so the API can 400 rather than store garbage."""
        if review_status not in self.REVIEW_STATES:
            raise ValueError(
                f"review_status must be one of {sorted(self.REVIEW_STATES)}"
            )
        reviewed_at = None if review_status == "pending" else "CURRENT_TIMESTAMP"
        conn = self._connect()
        try:
            with conn:
                # reviewed_at is set to now for a decision, cleared back to NULL
                # if a row is reset to pending.
                if reviewed_at:
                    cur = conn.execute(
                        """UPDATE violations
                           SET review_status = ?, reviewer_decision = ?,
                               review_notes = ?, reviewed_at = CURRENT_TIMESTAMP
                           WHERE id = ?""",
                        (review_status, reviewer_decision, notes, violation_id),
                    )
                else:
                    cur = conn.execute(
                        """UPDATE violations
                           SET review_status = ?, reviewer_decision = ?,
                               review_notes = ?, reviewed_at = NULL
                           WHERE id = ?""",
                        (review_status, reviewer_decision, notes, violation_id),
                    )
                if cur.rowcount > 0:
                    self._log_event(
                        conn, f"violation_review_{review_status}", "violation",
                        str(violation_id), actor,
                        {"reviewer_decision": reviewer_decision, "notes": notes},
                    )
            return cur.rowcount > 0
        finally:
            conn.close()

    # -- payment lifecycle (Phase 18) ---------------------------------------

    # Payment status is kept in `status` for backwards compatibility; it is a
    # dedicated lifecycle, independent of detection_status and review_status.
    PAYMENT_STATES = {"unpaid", "paid", "cancelled"}
    _PAYMENT_TRANSITIONS = {
        "unpaid": {"paid", "cancelled", "unpaid"},
        "paid": {"paid"},           # terminal
        "cancelled": {"cancelled"},  # terminal
    }

    @classmethod
    def valid_payment_transition(cls, current: str, new: str) -> bool:
        return new in cls._PAYMENT_TRANSITIONS.get(current, set())

    def set_payment_status(
        self, violation_id: int, new_status: str, *, actor: str | None = None
    ) -> bool:
        """Move a violation's payment status along its lifecycle
        (unpaid -> paid | cancelled; paid/cancelled are terminal). Raises
        ValueError for an unknown status or an illegal transition. Returns False
        if the violation doesn't exist."""
        if new_status not in self.PAYMENT_STATES:
            raise ValueError(
                f"payment status must be one of {sorted(self.PAYMENT_STATES)}"
            )
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT status FROM violations WHERE id = ?", (violation_id,)
            ).fetchone()
            if row is None:
                return False
            current = row["status"]
            if not self.valid_payment_transition(current, new_status):
                raise ValueError(
                    f"illegal payment transition {current!r} -> {new_status!r}"
                )
            with conn:
                conn.execute(
                    "UPDATE violations SET status = ? WHERE id = ?",
                    (new_status, violation_id),
                )
                self._log_event(
                    conn, f"payment_{new_status}", "violation",
                    str(violation_id), actor, {"from": current},
                )
            return True
        finally:
            conn.close()

    # -- audit log (Phase 17) -----------------------------------------------

    @staticmethod
    def _log_event(conn, event, record_type, record_id, actor, metadata=None):
        """Append one audit row inside an existing transaction/connection."""
        import json as _json
        conn.execute(
            "INSERT INTO audit_log (event, record_type, record_id, actor, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (event, record_type, record_id, actor,
             _json.dumps(metadata) if metadata is not None else None),
        )

    def log_event(
        self, event: str, *, record_type: str | None = None,
        record_id: str | None = None, actor: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Append an audit event (its own transaction)."""
        conn = self._connect()
        try:
            with conn:
                self._log_event(conn, event, record_type, record_id, actor, metadata)
        finally:
            conn.close()

    def list_audit(self, *, limit: int = 100, record_id: str | None = None) -> list:
        limit = max(1, min(int(limit), 1000))
        conn = self._connect()
        try:
            if record_id is not None:
                rows = conn.execute(
                    "SELECT * FROM audit_log WHERE record_id = ? "
                    "ORDER BY id DESC LIMIT ?", (record_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    # -- processing sessions (Phase 14) -------------------------------------

    def create_session(
        self, session_id: str, *, source: str | None = None,
        model_version: str | None = None, pipeline_version: str | None = None,
    ) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """INSERT OR REPLACE INTO sessions
                       (id, source, model_version, pipeline_version, status)
                       VALUES (?, ?, ?, ?, 'processing')""",
                    (session_id, source, model_version, pipeline_version),
                )
        finally:
            conn.close()

    _SESSION_FIELDS = {
        "source", "ended_at", "model_version", "pipeline_version",
        "frames_processed", "vehicles_tracked", "violations_detected",
        "violations_confirmed", "violations_dismissed", "processing_fps",
        "output_path", "status", "error",
    }

    def update_session(self, session_id: str, **fields) -> None:
        fields = {k: v for k, v in fields.items() if k in self._SESSION_FIELDS}
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    f"UPDATE sessions SET {cols} WHERE id = ?",
                    (*fields.values(), session_id),
                )
        finally:
            conn.close()

    def get_session(self, session_id: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def list_sessions(self, *, limit: int = 50) -> list:
        limit = max(1, min(int(limit), 500))
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    # -- detection trace (Phase 12) -----------------------------------------

    def add_detection_trace(self, violation_id: int, rows: list) -> int:
        """Persist the supporting per-frame detections for a confirmed
        violation. ``rows`` is a list of dicts with keys track_id, label,
        confidence, box (any JSON-able), frame_index, timestamp. Returns the
        number stored. Bounded by the caller (see config trace retention)."""
        import json as _json
        conn = self._connect()
        try:
            with conn:
                for r in rows:
                    box = r.get("box")
                    conn.execute(
                        """INSERT INTO detections
                           (violation_id, track_id, label, confidence, box,
                            frame_index, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (violation_id, r.get("track_id"), r.get("label"),
                         r.get("confidence"),
                         _json.dumps(box) if box is not None else None,
                         r.get("frame_index"), r.get("timestamp")),
                    )
        finally:
            conn.close()
        return len(rows)

    def get_detection_trace(self, violation_id: int) -> list:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT track_id, label, confidence, box, frame_index, timestamp "
                "FROM detections WHERE violation_id = ? ORDER BY frame_index, id",
                (violation_id,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    # -- processing jobs (Phase 15: durable metadata) -----------------------

    _JOB_FIELDS = {
        "type", "status", "progress", "started_at", "updated_at",
        "completed_at", "error", "output", "source",
    }

    def create_job(self, job_id: str, job_type: str, *, source: str | None = None) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """INSERT OR REPLACE INTO processing_jobs
                       (id, type, status, progress, started_at, updated_at, source)
                       VALUES (?, ?, 'processing', 0,
                               CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)""",
                    (job_id, job_type, source),
                )
        finally:
            conn.close()

    def update_job(self, job_id: str, **fields) -> None:
        # Whitelist columns so an unexpected key can never build bad SQL.
        fields = {k: v for k, v in fields.items() if k in self._JOB_FIELDS}
        if not fields:
            return
        fields.setdefault("updated_at", None)  # placeholder replaced below
        set_parts = []
        values = []
        for k, v in fields.items():
            if k == "updated_at" and v is None:
                set_parts.append("updated_at = CURRENT_TIMESTAMP")
            else:
                set_parts.append(f"{k} = ?")
                values.append(v)
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    f"UPDATE processing_jobs SET {', '.join(set_parts)} WHERE id = ?",
                    (*values, job_id),
                )
        finally:
            conn.close()

    def get_job(self, job_id: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM processing_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def list_jobs(self, *, limit: int = 50) -> list:
        limit = max(1, min(int(limit), 500))
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM processing_jobs ORDER BY created_at DESC, id DESC "
                "LIMIT ?", (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def reset(self) -> None:
        """Delete all vehicles/violations/evidence/detections. Used by the demo
        seeder's wipe mode; does not touch the schema or the legacy table."""
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM detections")
                conn.execute("DELETE FROM evidence")
                conn.execute("DELETE FROM violations")
                conn.execute("DELETE FROM vehicles")
        finally:
            conn.close()


def get_database(path: str | None = None) -> Database:
    return Database(path or os.environ.get("TRAFFIC_DB_PATH", "traffic.db"))
