"""Dashboard evaluation panel: reads generated reports, never invents metrics.

The hard requirement these tests protect is that a missing or broken report
produces an explicit "not run" state with the command to run — never a zero, a
placeholder, or a 500.
"""

from __future__ import annotations

import json
import os

from modules.eval_report import (
    detector_summary,
    evaluation_panel,
    find_detector_report,
    leakage_summary,
    pipeline_summary,
)


def write(tmp_path, name, payload):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w") as fh:
        json.dump(payload, fh)
    return path


DETECTOR = {
    "generated_at": "2026-09-13T10:00:00+00:00",
    "model": "runs/detect/m/weights/best.pt",
    "model_version": "traffic-4class@1.0.0",
    "data": "d.yaml",
    "split": "test",
    "official": {
        "map50": 0.72, "map50_95": 0.53,
        "mean_precision": 0.76, "mean_recall": 0.78,
        "per_class": {"Plate": {"map50": 0.86}},
    },
    "error_analysis": {
        "images": 175,
        "confidence": {"separation": 0.14},
        "confidence_curve": {"overall": {"spearman": 0.9, "verdict": "useful"}},
    },
}


# ---------------------------------------------------------------------------
# Empty states
# ---------------------------------------------------------------------------

def test_detector_summary_missing_dir_is_explicit(tmp_path):
    out = detector_summary(os.path.join(str(tmp_path), "nope"))
    assert out["available"] is False
    assert "evaluate_model.py" in out["command"]
    assert "map50" not in out  # no placeholder number


def test_pipeline_summary_missing_file_is_explicit(tmp_path):
    out = pipeline_summary(str(tmp_path))
    assert out["available"] is False
    assert out["command"] == "python3 evaluate_pipeline.py"


def test_leakage_summary_missing_file_is_explicit(tmp_path):
    out = leakage_summary(str(tmp_path))
    assert out["available"] is False
    assert "audit_dataset.py" in out["command"]


def test_malformed_json_does_not_raise(tmp_path):
    """A truncated report must degrade to the empty state, not take the
    dashboard down."""
    with open(os.path.join(str(tmp_path), "latest.json"), "w") as fh:
        fh.write("{not json")
    with open(os.path.join(str(tmp_path), "dataset_leakage.json"), "w") as fh:
        fh.write("[]")
    assert pipeline_summary(str(tmp_path))["available"] is False
    assert leakage_summary(str(tmp_path))["available"] is False


def test_non_detector_json_is_not_mistaken_for_one(tmp_path):
    write(tmp_path, "something.json", {"unrelated": True})
    assert find_detector_report(str(tmp_path)) is None
    assert detector_summary(str(tmp_path))["available"] is False


# ---------------------------------------------------------------------------
# Populated states
# ---------------------------------------------------------------------------

def test_detector_summary_reads_the_generated_report(tmp_path):
    write(tmp_path, "eval_x.json", DETECTOR)
    out = detector_summary(str(tmp_path))
    assert out["available"] is True
    assert out["map50"] == 0.72
    assert out["model_version"] == "traffic-4class@1.0.0"
    assert out["split"] == "test"
    assert out["images"] == 175
    assert out["per_class"]["Plate"]["map50"] == 0.86
    assert out["confidence"]["spearman"] == 0.9
    assert out["age_days"] is not None


def test_newest_generated_at_wins_not_newest_mtime(tmp_path):
    """A report restored from elsewhere must be ordered by when it was produced,
    not when it landed on disk."""
    old = dict(DETECTOR, generated_at="2020-01-01T00:00:00+00:00")
    old["official"] = dict(DETECTOR["official"], map50=0.1)
    new = dict(DETECTOR, generated_at="2026-09-13T10:00:00+00:00")
    write(tmp_path, "a_new.json", new)
    write(tmp_path, "z_old.json", old)  # written last, but older
    assert detector_summary(str(tmp_path))["map50"] == 0.72


def test_detector_report_without_a_manifest_reports_none_not_a_guess(tmp_path):
    write(tmp_path, "e.json", dict(DETECTOR, model_version=None))
    assert detector_summary(str(tmp_path))["model_version"] is None


def test_pipeline_summary_passes_through_and_ages(tmp_path):
    write(tmp_path, "latest.json", {
        "generated_at": "2026-09-13T10:00:00+00:00",
        "end_to_end": {"precision": 1.0, "recall": 1.0},
        "error_budget": {"bottleneck": "ocr"},
    })
    out = pipeline_summary(str(tmp_path))
    assert out["available"] is True
    assert out["end_to_end"]["precision"] == 1.0
    assert out["error_budget"]["bottleneck"] == "ocr"
    assert out["age_days"] is not None


def test_bad_timestamp_yields_no_age_rather_than_raising(tmp_path):
    write(tmp_path, "latest.json", {"generated_at": "not-a-date"})
    assert pipeline_summary(str(tmp_path))["age_days"] is None


def test_leakage_summary_surfaces_the_clean_flag(tmp_path):
    write(tmp_path, "dataset_leakage.json", {
        "clean": False, "method": "dHash",
        "splits": {"test": {"held_out_images": 194, "leaked_images": 19,
                            "leak_rate": 0.0979}},
    })
    out = leakage_summary(str(tmp_path))
    assert out["clean"] is False
    assert out["splits"]["test"]["leaked"] == 19


def test_evaluation_panel_keeps_detector_and_pipeline_separate(tmp_path):
    write(tmp_path, "eval_x.json", DETECTOR)
    panel = evaluation_panel(str(tmp_path))
    assert set(panel) >= {"detector", "pipeline", "dataset", "note"}
    assert "never combined" in panel["note"]
    # One section being available must not fabricate the others.
    assert panel["detector"]["available"] is True
    assert panel["pipeline"]["available"] is False
