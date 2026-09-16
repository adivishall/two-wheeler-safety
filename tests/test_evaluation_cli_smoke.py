"""Smoke tests for the evaluation CLIs (Phase 25).

The full model evaluation needs weights, a dataset and a GPU-ish machine, so it
is run by hand. That leaves the tooling free to rot silently between runs. These
tests keep it honest by exercising every entry point in a way CI can afford:
``--help`` for the ones that need a model, and a real end-to-end run for the
model-free ones (which is cheap and produces actual files to assert on).
"""

from __future__ import annotations

import csv
import json
import os

import pytest

import audit_dataset
import compare_models
import evaluate_model
import evaluate_ocr
import evaluate_pipeline
import evaluate_system

# ---------------------------------------------------------------------------
# Argument parsing works without the heavy stack installed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [
    evaluate_model, evaluate_ocr, evaluate_pipeline, evaluate_system,
    compare_models, audit_dataset,
])
def test_help_does_not_need_torch(module, capsys):
    with pytest.raises(SystemExit) as exc:
        module.parse_args(["--help"]) if hasattr(module, "parse_args") \
            else module.main(["--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out


def test_evaluate_model_rejects_missing_weights(tmp_path):
    rc = evaluate_model.main([
        "--model", str(tmp_path / "nope.pt"), "--data", str(tmp_path / "d.yaml"),
    ])
    assert rc == 2  # a clear failure, not a traceback


def test_evaluate_model_rejects_missing_data(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"not really weights")
    rc = evaluate_model.main(["--model", str(weights),
                              "--data", str(tmp_path / "missing.yaml")])
    assert rc == 2


def test_audit_dataset_rejects_missing_yaml(tmp_path):
    assert audit_dataset.main(["--data", str(tmp_path / "missing.yaml")]) == 2


def test_compare_models_rejects_missing_yaml(tmp_path):
    assert compare_models.main(["--data", str(tmp_path / "missing.yaml")]) == 2


def test_compare_models_reports_when_no_checkpoints_exist(tmp_path):
    data = tmp_path / "d.yaml"
    data.write_text("train: t\nval: v\nnc: 1\nnames: [a]\n")
    rc = compare_models.main(["--data", str(data),
                              "--runs-root", str(tmp_path / "empty")])
    assert rc == 2


def test_evaluate_ocr_rejects_missing_labels(tmp_path):
    assert evaluate_ocr.main(["--labels", str(tmp_path / "missing.csv")]) == 2


def test_evaluate_ocr_requires_a_source():
    """--labels / --sequences / --simulate are mutually exclusive and required;
    silently doing nothing would be worse than failing."""
    with pytest.raises(SystemExit):
        evaluate_ocr.parse_args([])


# ---------------------------------------------------------------------------
# Model-free CLIs actually run end to end
# ---------------------------------------------------------------------------

def test_evaluate_system_runs_and_writes_json(tmp_path):
    out = tmp_path / "system.json"
    assert evaluate_system.main(["--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["totals"]["true_positives"] > 0


def test_evaluate_ocr_simulate_runs_without_easyocr(tmp_path):
    rc = evaluate_ocr.main([
        "--simulate", "--out", str(tmp_path), "--name", "sim",
    ])
    assert rc == 0
    data = json.loads((tmp_path / "sim.json").read_text())
    assert data["mode"] == "simulate"
    assert set(data["default_noise"]["policies"]) == {"last", "best_conf", "temporal"}


def test_evaluate_pipeline_writes_every_output_format(tmp_path):
    rc = evaluate_pipeline.main([
        "--out", str(tmp_path), "--name", "p", "--trials", "2", "--quiet",
    ])
    assert rc == 0
    for suffix in (".json", ".md", ".csv"):
        assert (tmp_path / f"p{suffix}").exists()
    assert (tmp_path / "latest.json").exists()


def test_evaluate_pipeline_csv_is_well_formed(tmp_path):
    evaluate_pipeline.main(["--out", str(tmp_path), "--name", "p",
                            "--trials", "2", "--quiet"])
    with open(tmp_path / "p.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert set(rows[0]) == {"section", "subject", "metric", "value"}
    assert {r["section"] for r in rows} >= {"system", "violations", "error_budget"}
    # Every value must be numeric — a stray string would break any consumer.
    for r in rows:
        float(r["value"])


def test_evaluate_pipeline_only_runs_requested_sections(tmp_path):
    evaluate_pipeline.main(["--only", "speed", "--out", str(tmp_path),
                            "--name", "s", "--quiet"])
    data = json.loads((tmp_path / "s.json").read_text())
    assert data["sections"] == ["speed"]
    assert "speed" in data
    assert "system" not in data


def test_evaluate_pipeline_markdown_separates_model_from_pipeline(tmp_path):
    """The report must not let a reader mistake pipeline metrics for detector
    metrics — that conflation is the single most misleading thing this project
    could publish."""
    evaluate_pipeline.main(["--out", str(tmp_path), "--name", "p",
                            "--trials", "2", "--quiet"])
    md = (tmp_path / "p.md").read_text()
    assert "pipeline logic" in md
    assert "MODEL_EVALUATION.md" in md
    assert "No model weights" in md


def test_evaluate_pipeline_summary_is_small_enough_for_a_dashboard(tmp_path):
    evaluate_pipeline.main(["--out", str(tmp_path), "--name", "p",
                            "--trials", "2", "--quiet"])
    summary = json.loads((tmp_path / "latest.json").read_text())
    assert set(summary) >= {"end_to_end", "violations", "error_budget",
                            "ocr_policy", "speed", "generated_at"}
    assert os.path.getsize(tmp_path / "latest.json") < 4096


def test_evaluate_pipeline_is_reproducible(tmp_path):
    """Two runs must agree exactly, apart from the timestamp — a metric that
    moves between runs cannot be cited."""
    def run(name):
        evaluate_pipeline.main(["--out", str(tmp_path), "--name", name,
                                "--trials", "3", "--quiet"])
        d = json.loads((tmp_path / f"{name}.json").read_text())
        d.pop("generated_at")
        return d

    assert run("a") == run("b")
