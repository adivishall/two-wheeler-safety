"""A human review is also a label: which stage was wrong, the true plate, who
decided — and an export that feeds calibration without demo data leaking in."""

import pytest

from modules.db import Database


def _db(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.initialize()
    db.create_session("run1", source="clip.mp4", model_version="m@1", pipeline_version="1.1.0")
    db.create_session("demo", source="[curated demo seed] not pipeline output")
    return db


_SIDECAR = {"original_path": "o.jpg", "annotated_path": "a.jpg",
            "plate_crop_path": "p.jpg", "metadata_path": "m.json"}


def _fine(db, plate, confidence, session, *, sidecar=True):
    db.record_fine(plate, "no_helmet", None, confidence=confidence, session_id=session,
                   evidence=_SIDECAR if sidecar else None)
    return db.list_violations()["items"][0]["id"]


def test_dismissal_records_the_failing_stage_and_the_true_plate(tmp_path):
    db = _db(tmp_path)
    vid = _fine(db, "MH12AB1234", 0.8, "run1")
    assert db.set_review(vid, "dismissed", reason="wrong_plate",
                         corrected_plate="mh 12 ab 1284", actor="rev-alice")
    row = db.review_labels()[0]
    assert row["review_reason"] == "wrong_plate"
    assert row["corrected_plate"] == "MH12AB1284"
    assert row["reviewed_by"] == "rev-alice" and row["correct"] == 0


@pytest.mark.parametrize("status, kwargs", [
    ("dismissed", {"reason": "blurry"}),                        # unknown reason
    ("confirmed", {"reason": "wrong_plate"}),                   # reason on a confirmation
    ("dismissed", {"reason": "wrong_violation", "corrected_plate": "MH12AB1234"}),
    ("dismissed", {"reason": "wrong_plate", "corrected_plate": " - "}),
])
def test_unusable_labels_are_rejected(tmp_path, status, kwargs):
    db = _db(tmp_path)
    vid = _fine(db, "MH12AB1234", 0.8, "run1")
    with pytest.raises(ValueError):
        db.set_review(vid, status, **kwargs)


def test_resetting_to_pending_clears_the_label(tmp_path):
    db = _db(tmp_path)
    vid = _fine(db, "MH12AB1234", 0.8, "run1")
    db.set_review(vid, "dismissed", reason="duplicate", actor="rev")
    db.set_review(vid, "pending")
    d = db.get_violation(vid)
    assert d["review_reason"] is None and d["reviewed_by"] is None
    assert db.review_labels() == []


def test_export_excludes_demo_sessions_unscored_and_undecided_rows(tmp_path):
    db = _db(tmp_path)
    good = _fine(db, "MH12AB1234", 0.9, "run1")
    demo = _fine(db, "KA05CD1111", 0.7, "demo")
    unscored = _fine(db, "KA05CD2222", None, "run1")
    _fine(db, "KA05CD3333", 0.6, "run1")  # never reviewed
    for vid in (good, demo, unscored):
        db.set_review(vid, "confirmed", actor="rev")
    rows = db.review_labels()
    assert [r["violation_id"] for r in rows] == [good]
    assert rows[0]["correct"] == 1 and rows[0]["session_id"] == "run1"
    assert rows[0]["model_version"] == "m@1"


def test_review_api_accepts_a_structured_dismissal(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_DB_PATH", str(tmp_path / "app.db"))
    import importlib

    import app as app_module

    app_module = importlib.reload(app_module)
    client = app_module.app.test_client()
    app_module.db.record_fine("MH12AB1234", "no_helmet", None, confidence=0.8)
    vid = app_module.db.list_violations()["items"][0]["id"]
    bad = client.post(f"/api/violations/{vid}/review",
                      json={"review_status": "confirmed", "reason": "wrong_plate"})
    assert bad.status_code == 400
    ok = client.post(f"/api/violations/{vid}/review",
                     json={"review_status": "dismissed", "reason": "wrong_plate",
                           "corrected_plate": "MH12AB1284"})
    assert ok.status_code == 200
    assert app_module.db.get_violation(vid)["corrected_plate"] == "MH12AB1284"


def test_calibration_cli_reports_not_measured_without_real_reviews(tmp_path):
    import calibrate_confidence

    db = _db(tmp_path)
    demo = _fine(db, "KA05CD1111", 0.7, "demo")
    db.set_review(demo, "confirmed", actor="rev")  # demo data: must not count
    rc = calibrate_confidence.main(["--db", str(tmp_path / "t.db"),
                                    "--out", str(tmp_path), "--name", "cal"])
    assert rc == 0
    md = (tmp_path / "cal.md").read_text()
    assert "NOT MEASURED" in md and "**0** reviewed" in md


def test_calibration_cli_groups_folds_by_session_and_saves_only_if_adopted(tmp_path):
    import csv
    import random

    import calibrate_confidence

    rng = random.Random(3)
    path = tmp_path / "labels.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["confidence", "correct", "session_id"])
        for i in range(600):  # overconfident: true accuracy is half the score
            s = rng.random()
            w.writerow([round(s, 4), int(rng.random() < s * 0.5), f"s{i // 20}"])
    out = tmp_path / "cal.json"
    rc = calibrate_confidence.main(["--labels", str(path), "--save", str(out)])
    assert rc == 0 and out.exists()  # miscalibrated -> a calibrator is adopted
    scores, labels, groups = calibrate_confidence.read_labels(str(path))
    assert groups is not None and len(set(groups)) == 30


def test_rows_without_a_pipeline_sidecar_are_not_labels(tmp_path):
    """The pre-1.1.0 demo seeder wrote sessions that look like real runs
    ("junction_cam_01.mp4", a real model version) with hand-picked scores and
    'reviews' — but no evidence sidecar. They must never reach calibration."""
    db = _db(tmp_path)
    db.create_session("demo-sess-junction", source="junction_cam_01.mp4",
                      model_version="traffic-4class@1.0.0")
    legacy = _fine(db, "MH02DL4596", 0.97, "demo-sess-junction", sidecar=False)
    real = _fine(db, "MH12AB1234", 0.8, "run1")
    for vid in (legacy, real):
        db.set_review(vid, "confirmed", actor="demo")
    assert [r["violation_id"] for r in db.review_labels()] == [real]


def test_duplicate_and_unusable_evidence_are_not_correctness_labels(tmp_path):
    db = _db(tmp_path)
    for plate, reason in (("MH12AB1111", "duplicate"), ("MH12AB2222", "evidence_unusable"),
                          ("MH12AB3333", "wrong_violation")):
        vid = _fine(db, plate, 0.7, "run1")
        db.set_review(vid, "dismissed", reason=reason, actor="rev")
    by_reason = {r["review_reason"]: r["correct"] for r in db.review_labels()}
    assert by_reason == {"duplicate": None, "evidence_unusable": None, "wrong_violation": 0}
