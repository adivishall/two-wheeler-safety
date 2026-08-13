"""Seed the fine database with a small, realistic set of records for demos.

Runs the real detector on a few bundled sample photos so each seeded plate
comes with a genuine annotated evidence image (plate and photo stay
consistent), then records a curated mix of violations — a plate with two
unpaid fines, a triple-riding fine, and an already-paid one — spread over
the last couple of weeks.

    python3 seed_demo.py           # wipe existing fines, then seed
    python3 seed_demo.py --keep    # keep existing fines, just add the demo set

After it runs it prints exactly which plates to look up in the web UI.
"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta, timezone

from modules.detector import load_models, analyze_image

DB_PATH = os.environ.get("TRAFFIC_DB_PATH", "traffic.db")
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "runs/detect/traffic_model-2/weights/best.pt"
)

# Each entry: a source photo, and the violations to record for the plate the
# detector reads from it. (violation, amount ₹, status, days_ago)
DEMO_RECORDS = [
    ("plate4.jpeg", [
        ("no_helmet", 500, "unpaid", 3),
        ("overspeed", 700, "unpaid", 1),
    ]),
    ("plate8.jpeg", [
        ("triple_riding", 1000, "unpaid", 6),
    ]),
    ("plate11.jpeg", [
        ("no_helmet", 500, "paid", 12),
    ]),
]


def ensure_table(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS fines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT,
        violation TEXT,
        amount INTEGER,
        image_path TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'unpaid'
    )
    """)


def main():
    keep = "--keep" in sys.argv

    if not os.path.exists(MODEL_PATH):
        print(f"Model weights not found at {MODEL_PATH}. Train first or set MODEL_PATH.")
        return

    print("Loading model + OCR…")
    model, reader = load_models(MODEL_PATH)

    conn = sqlite3.connect(DB_PATH)
    ensure_table(conn)

    if not keep:
        conn.execute("DELETE FROM fines")
        print(f"Cleared existing fines in {DB_PATH}.")

    seeded = {}  # plate -> [(violation, amount, status)]

    for source, records in DEMO_RECORDS:
        if not os.path.exists(source):
            print(f"  skip {source}: file not found")
            continue

        result = analyze_image(source, model, reader)
        plate = result["plate"]

        if not plate:
            print(f"  skip {source}: no plate could be read")
            continue

        evidence = result["evidence_file"] or ""

        for violation, amount, status, days_ago in records:
            timestamp = (
                datetime.now(timezone.utc) - timedelta(days=days_ago)
            ).strftime("%Y-%m-%d %H:%M:%S")

            conn.execute(
                """
                INSERT INTO fines (plate, violation, amount, image_path, timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (plate, violation, amount, evidence, timestamp, status),
            )
            seeded.setdefault(plate, []).append((violation, amount, status))

    conn.commit()
    conn.close()

    print("\nSeeded demo records — look these up in the web UI:\n")
    for plate, rows in seeded.items():
        due = sum(a for _, a, s in rows if s == "unpaid")
        parts = ", ".join(f"{v.replace('_', ' ')} ₹{a} ({s})" for v, a, s in rows)
        due_str = f"₹{due} due" if due else "all paid"
        print(f"  {plate:14} {parts}   →  {due_str}")
    print()


if __name__ == "__main__":
    main()
