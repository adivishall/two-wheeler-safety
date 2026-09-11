"""One-command demo of the Two-Wheeler Safety Monitor.

    python3 demo.py            # seed a self-contained demo, then serve it
    make demo                  # same

Why this exists: `seed_demo.py` runs the *real* detector (needs the ~2 GB
torch/ultralytics/easyocr stack and the trained weights). This demo is
**dependency-light on purpose** — it seeds a small, clearly-labelled synthetic
dataset (using bundled real plate photos as evidence) so a reviewer can explore
the whole UI — dashboard, analytics, per-session drill-down, plate lookup, and
the human-review workflow — on a fresh clone with only the model-free deps
installed. Nothing here is presented as a real detection; the confidence scores
and sessions are illustrative demo data (the app labels confidence as a score,
not a probability, everywhere it's shown).

Flags:
    --real       Seed with the actual detector instead (delegates to seed_demo),
                 for a genuine end-to-end run when weights are available.
    --no-serve   Seed only; don't start the server.
    --keep       Don't wipe existing records before seeding.
    --port N     Serve on port N (default 5000).

The demo set is documented in docs/DEMO.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

from modules.db import Database
from modules.logging_setup import configure_logging, get_logger

log = get_logger("demo")

DB_PATH = os.environ.get("TRAFFIC_DB_PATH", "traffic.db")
EVIDENCE_DIR = os.environ.get("EVIDENCE_DIR", "evidence")

# Bundled real plate photos used as demo evidence, copied into the evidence dir
# under stable names. (source file, evidence basename)
DEMO_IMAGES = {
    "MH02DL4596": ("demo_shots/MH02DL4596_1786603781.jpg", "demo_MH02DL4596.jpg"),
    "MH12HS8818": ("demo_shots/MH12HS8818_1786603782.jpg", "demo_MH12HS8818.jpg"),
    "KA05CD9876": ("plate4.jpeg", "demo_KA05CD9876.jpg"),
    "DL8CAF5031": ("plate8.jpeg", "demo_DL8CAF5031.jpg"),
    "TN22BB4410": ("plate11.jpeg", "demo_TN22BB4410.jpg"),
}

# Two illustrative processing sessions (as if two clips were analysed).
DEMO_SESSIONS = [
    {"id": "demo-sess-junction", "source": "junction_cam_01.mp4",
     "fps": 14.6, "frames": 1820, "vehicles": 37},
    {"id": "demo-sess-highway", "source": "highway_evening.mp4",
     "fps": 9.2, "frames": 940, "vehicles": 18},
]

# Curated violation set: (plate, violation, confidence, days_ago, session, review, paid)
# Chosen to make every dashboard panel meaningful: a spread of types, a confidence
# range that fills the histogram, a mix of pending/confirmed/dismissed reviews,
# and a few paid fines — across ~10 days so the over-time chart has shape.
DEMO_VIOLATIONS = [
    ("MH02DL4596", "no_helmet",     0.97, 1,  "demo-sess-junction", "confirmed", True),
    ("MH02DL4596", "overspeed",     0.88, 1,  "demo-sess-highway",  "confirmed", False),
    ("MH12HS8818", "triple_riding", 0.93, 2,  "demo-sess-junction", "confirmed", True),
    ("MH12HS8818", "no_helmet",     0.71, 2,  "demo-sess-junction", "pending",   False),
    ("KA05CD9876", "overspeed",     0.84, 3,  "demo-sess-highway",  "confirmed", False),
    ("KA05CD9876", "no_helmet",     0.55, 3,  "demo-sess-junction", "dismissed", False),
    ("DL8CAF5031", "triple_riding", 0.79, 4,  "demo-sess-junction", "confirmed", False),
    ("DL8CAF5031", "no_helmet",     0.62, 5,  "demo-sess-highway",  "pending",   False),
    ("TN22BB4410", "overspeed",     0.90, 6,  "demo-sess-highway",  "confirmed", True),
    ("TN22BB4410", "no_helmet",     0.44, 6,  "demo-sess-junction", "dismissed", False),
    ("MH02DL4596", "triple_riding", 0.81, 7,  "demo-sess-junction", "pending",   False),
    ("KA05CD9876", "triple_riding", 0.68, 8,  "demo-sess-junction", "pending",   False),
    ("MH12HS8818", "overspeed",     0.76, 9,  "demo-sess-highway",  "confirmed", False),
    ("DL8CAF5031", "overspeed",     0.59, 10, "demo-sess-highway",  "pending",   False),
]

MODEL_VERSION = "traffic-4class@1.0.0"
PIPELINE_VERSION = "1.0.0"


def _copy_demo_evidence() -> dict[str, str]:
    """Copy the bundled demo photos into the evidence dir; return plate ->
    stored evidence path (only for images that were found)."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    stored = {}
    for plate, (src, name) in DEMO_IMAGES.items():
        if os.path.exists(src):
            dst = os.path.join(EVIDENCE_DIR, name)
            shutil.copyfile(src, dst)
            stored[plate] = f"{EVIDENCE_DIR}/{name}"
        else:
            log.warning("demo image missing, evidence will be blank: %s", src)
    return stored


def seed_synthetic(keep: bool) -> Database:
    db = Database(DB_PATH)
    if not keep:
        db.reset()
        log.info("cleared existing records in %s", DB_PATH)

    evidence = _copy_demo_evidence()

    for s in DEMO_SESSIONS:
        db.create_session(s["id"], source=s["source"],
                          model_version=MODEL_VERSION, pipeline_version=PIPELINE_VERSION)
        db.update_session(
            s["id"], status="completed", processing_fps=s["fps"],
            frames_processed=s["frames"], vehicles_tracked=s["vehicles"],
            ended_at=datetime.now(timezone.utc).isoformat(),
        )

    now = datetime.now(timezone.utc)
    for plate, violation, conf, days_ago, session, review, paid in DEMO_VIOLATIONS:
        ts = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        vid = _record_and_id(db, plate, violation, evidence.get(plate, ""),
                             conf, ts, session)
        if review != "pending":
            db.set_review(vid, review, actor="demo")
        if paid:
            db.set_payment_status(vid, "paid", actor="demo")

    return db


def _record_and_id(db, plate, violation, evidence_path, conf, ts, session) -> int:
    """Record a fine and return its violation id (the newest for this plate)."""
    db.record_fine(plate, violation, evidence_path,
                   confidence=conf, timestamp=ts, session_id=session)
    row = db.list_violations(plate=plate, limit=1)["items"][0]
    return row["id"]


def _print_summary(db: Database) -> None:
    s = db.stats()
    print("\n" + "=" * 56)
    print("  Demo data seeded (synthetic — for exploring the UI)")
    print("=" * 56)
    print(f"  Violations : {s['total_violations']}")
    print(f"  Vehicles   : {s['vehicles']}")
    print(f"  Fines total: ₹{s['total_fines']}  (unpaid ₹{s['unpaid_fines']})")
    print(f"  Sessions   : {len(db.list_sessions())}")
    print("\n  Plates to look up in the Plate tab:")
    for plate in DEMO_IMAGES:
        print(f"    • {plate}")
    print("\n  In the Dashboard tab: expand Analytics, and click a")
    print("  session's 'View' to drill into the violations it produced.")
    print("=" * 56 + "\n")


def main(argv=None) -> int:
    configure_logging()
    ap = argparse.ArgumentParser(description="One-command demo.")
    ap.add_argument("--real", action="store_true",
                    help="seed with the real detector (delegates to seed_demo.py)")
    ap.add_argument("--no-serve", action="store_true", help="seed only, don't serve")
    ap.add_argument("--keep", action="store_true", help="keep existing records")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    args = ap.parse_args(argv)

    if args.real:
        # Genuine end-to-end seed; needs weights + the full stack.
        import seed_demo
        seed_demo.main()
        db = Database(DB_PATH)
    else:
        db = seed_synthetic(keep=args.keep)

    _print_summary(db)

    if args.no_serve:
        print("Seeded (not serving). Start the app with:  python3 app.py")
        return 0

    # Import the app only now (after the DB is seeded at DB_PATH) and serve.
    os.environ.setdefault("TRAFFIC_DB_PATH", DB_PATH)
    import app as app_module
    print(f"Open the demo at:  http://127.0.0.1:{args.port}\n")
    app_module.app.run(host="127.0.0.1", port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
