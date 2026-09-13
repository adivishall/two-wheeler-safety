"""Load generated evaluation results for the dashboard.

Phase 24 asks the dashboard to show real model/pipeline metrics. The rule that
makes that safe is: the dashboard **reads generated files, it never computes or
invents metrics**. If an evaluation has not been run, the panel says so and
names the command to run — an empty state, never a placeholder number.

Three sources, all produced by the evaluation tooling:

* ``eval/results/latest.json``           — `evaluate_pipeline.py` summary
* ``eval/results/<detector eval>.json``  — `evaluate_model.py` output
* ``eval/results/dataset_leakage.json``  — `audit_dataset.py` output

Everything is defensive: a missing, empty or malformed file yields
``available: False`` with a reason rather than raising, because a dashboard must
not 500 because someone deleted a report.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

DEFAULT_RESULTS_DIR = "eval/results"
PIPELINE_SUMMARY = "latest.json"
LEAKAGE_REPORT = "dataset_leakage.json"

# Detector reports are named by the caller, so they are found by shape rather
# than by an exact filename: any JSON carrying an "official" metrics block.
_DETECTOR_MARKER = "official"


def _read_json(path: str):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _age_days(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - when).total_seconds() / 86400, 1)


def find_detector_report(results_dir: str = DEFAULT_RESULTS_DIR) -> dict | None:
    """The most recently generated detector evaluation in ``results_dir``.

    Preferring the newest ``generated_at`` (not the newest mtime) means a report
    copied or restored from elsewhere is still ordered by when it was actually
    produced.
    """
    if not os.path.isdir(results_dir):
        return None
    candidates = []
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json"):
            continue
        data = _read_json(os.path.join(results_dir, name))
        if isinstance(data, dict) and _DETECTOR_MARKER in data:
            candidates.append((data.get("generated_at") or "", name, data))
    if not candidates:
        return None
    candidates.sort()
    _when, name, data = candidates[-1]
    data = dict(data)
    data["_file"] = name
    return data


def detector_summary(results_dir: str = DEFAULT_RESULTS_DIR) -> dict:
    """Headline detector metrics for the dashboard, or an explicit empty state."""
    report = find_detector_report(results_dir)
    if not report:
        return {
            "available": False,
            "reason": "no detector evaluation found",
            "command": (
                "python3 evaluate_model.py --model <weights.pt> "
                "--data <data.yaml> --split test"
            ),
        }
    official = report.get("official") or {}
    ea = report.get("error_analysis") or {}
    curve = (ea.get("confidence_curve") or {}).get("overall") or {}
    return {
        "available": True,
        "file": report.get("_file"),
        "model": report.get("model"),
        "model_version": report.get("model_version"),
        "dataset": report.get("data"),
        "split": report.get("split"),
        "generated_at": report.get("generated_at"),
        "age_days": _age_days(report.get("generated_at")),
        "images": ea.get("images"),
        "map50": official.get("map50"),
        "map50_95": official.get("map50_95"),
        "mean_precision": official.get("mean_precision"),
        "mean_recall": official.get("mean_recall"),
        "per_class": official.get("per_class") or {},
        "confidence": {
            "separation": (ea.get("confidence") or {}).get("separation"),
            "spearman": curve.get("spearman"),
            "verdict": curve.get("verdict"),
        },
    }


def pipeline_summary(results_dir: str = DEFAULT_RESULTS_DIR) -> dict:
    """The `evaluate_pipeline.py` summary, or an explicit empty state."""
    data = _read_json(os.path.join(results_dir, PIPELINE_SUMMARY))
    if not isinstance(data, dict):
        return {
            "available": False,
            "reason": "no pipeline evaluation found",
            "command": "python3 evaluate_pipeline.py",
        }
    out = dict(data)
    out["available"] = True
    out["age_days"] = _age_days(data.get("generated_at"))
    return out


def leakage_summary(results_dir: str = DEFAULT_RESULTS_DIR) -> dict:
    """Split-hygiene status: is the held-out data actually held out?"""
    data = _read_json(os.path.join(results_dir, LEAKAGE_REPORT))
    if not isinstance(data, dict):
        return {
            "available": False,
            "reason": "no dataset audit found",
            "command": "python3 audit_dataset.py --data <data.yaml>",
        }
    return {
        "available": True,
        "clean": data.get("clean"),
        "method": data.get("method"),
        "splits": {
            name: {
                "images": r.get("held_out_images"),
                "leaked": r.get("leaked_images"),
                "leak_rate": r.get("leak_rate"),
            }
            for name, r in (data.get("splits") or {}).items()
        },
    }


def evaluation_panel(results_dir: str = DEFAULT_RESULTS_DIR) -> dict:
    """Everything the dashboard's evaluation section renders, in one payload."""
    return {
        "results_dir": results_dir,
        "detector": detector_summary(results_dir),
        "pipeline": pipeline_summary(results_dir),
        "dataset": leakage_summary(results_dir),
        "note": (
            "Detector metrics and pipeline metrics answer different questions "
            "and are never combined: the detector number bounds what is "
            "possible, the pipeline number measures the logic built on top of "
            "it. All values are read from generated report files."
        ),
    }
