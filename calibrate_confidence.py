"""Is the violation confidence score calibrated? — measured on human review outcomes.

Labels are human decisions on real pipeline output: each reviewed violation's
stored ``confidence`` joined to confirmed (correct) / dismissed (incorrect).

    python3 calibrate_confidence.py --db traffic.db --out eval/results --name calibration
    python3 calibrate_confidence.py --labels review_labels.csv
    python3 calibrate_confidence.py --db traffic.db --export review_labels.csv

``--db`` reads ``Database.review_labels()`` (pipeline output only: demo seeds,
hand-recorded and unscored rows are excluded) and groups cross-validation folds
by session, so a calibrator is
never evaluated on reviews from a video it was fitted on. A CSV needs
``confidence,correct`` and may carry ``session_id``.

Every calibrator is judged **out of fold**; one is adopted only if it improves
Brier score with a CI excluding zero, and only with at least 30 correct and 30
incorrect outcomes. ``--save`` writes it only then. Nothing is applied
automatically, and the raw score is never overwritten: until this reports
``calibration_helps`` on real reviews, the score is a ranking, not a probability.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from modules.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    fit_and_evaluate,
    save_calibrator,
)

_TRUE = {"1", "true", "yes", "confirmed", "correct", "y", "t"}


def read_labels(path: str):
    """``(scores, labels, groups)`` from a CSV; ``groups`` is None without a
    ``session_id`` column."""
    scores, labels, groups = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row.get("confidence") or row.get("score")
            lab = row.get("correct") or row.get("label") or row.get("decision")
            if raw is None or lab is None:
                continue
            scores.append(float(raw))
            labels.append(1 if str(lab).strip().lower() in _TRUE else 0)
            groups.append(row.get("session_id") or None)
    return scores, labels, (groups if any(groups) else None)


def labels_from_db(path: str):
    from modules.db import Database

    rows = [r for r in Database(path).review_labels() if r["correct"] is not None]
    return ([float(r["confidence"]) for r in rows], [r["correct"] for r in rows],
            [r["session_id"] or f"violation{r['violation_id']}" for r in rows], rows)


def render_markdown(report: dict, source: str) -> str:
    lines = ["# Confidence calibration — human review outcomes", "",
             f"- Source: `{source}` — only violations the pipeline produced (scored, with "
             "an evidence sidecar); demo seeds and hand-recorded rows are excluded",
             f"- Outcomes: **{report['n']}** reviewed violations — "
             f"{report['correct']} confirmed, {report['incorrect']} dismissed",
             f"- Verdict: **{report['verdict']}**; score used: **{report['best']}**", ""]
    if report["verdict"] == "insufficient_data":
        lines += ["**NOT MEASURED.** " + report.get("reason", "") + ". The confidence "
                  "score stays a ranking score; it is not a probability, and no "
                  "calibrator is fitted.", ""]
        if report["n"] == 0:
            return "\n".join(lines) + "\n"
    lines += ["| score | ECE | MCE | Brier | in-sample ECE | Brier gain vs raw [95% CI] |",
              "|---|---:|---:|---:|---:|---|",
              f"| raw | {report['raw']['ece']} | {report['raw']['mce']} | "
              f"{report['raw']['brier']} | — | — |"]
    for name in ("platt", "isotonic"):
        if name in report:
            m, g = report[name], report[name]["brier_gain_vs_raw"]
            lines.append(f"| {name} (out of fold) | {m['ece']} | {m['mce']} | {m['brier']} | "
                         f"{m['in_sample_ece']} | {g['mean']} [{g['ci_low']}, {g['ci_high']}] |")
    lines += ["", "## If only scores ≥ t were auto-accepted", "",
              "| t | kept | precision | recall |", "|---:|---:|---:|---:|"]
    for r in report["threshold_table"]:
        lines.append(f"| {r['threshold']} | {r['kept']} | {r['precision']} | {r['recall']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Confidence calibration on review outcomes.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", help="SQLite database with human reviews")
    src.add_argument("--labels", help="CSV: confidence,correct[,session_id]")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--export", default=None, help="write the review labels to this CSV")
    ap.add_argument("--out", default=None, help="write <name>.json/.md here")
    ap.add_argument("--name", default="calibration")
    ap.add_argument("--save", default=None,
                    help="write the adopted calibrator here (only if one is adopted)")
    args = ap.parse_args(argv)

    if args.db:
        scores, labels, groups, rows = labels_from_db(args.db)
        source = args.db
        if args.export:
            fields = ["violation_id", "session_id", "type", "confidence", "correct",
                      "review_reason", "corrected_plate", "reviewed_by", "model_version"]
            with open(args.export, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {len(rows)} review labels to {args.export}")
    else:
        scores, labels, groups = read_labels(args.labels)
        source = args.labels

    report = fit_and_evaluate(scores, labels, n_bins=args.bins, folds=args.folds,
                              groups=groups)
    print(f"n={report['n']}  verdict={report['verdict']}  score used={report['best']}")
    if report["verdict"] == "insufficient_data":
        print(f"NOT MEASURED: {report.get('reason')}")

    if args.out:
        from modules.provenance import portable_path, run_provenance

        os.makedirs(args.out, exist_ok=True)
        payload = {"source": portable_path(source), "report": report,
                   "provenance": run_provenance(config={"folds": args.folds,
                                                        "bins": args.bins})}
        with open(os.path.join(args.out, f"{args.name}.json"), "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        with open(os.path.join(args.out, f"{args.name}.md"), "w") as fh:
            fh.write(render_markdown(report, portable_path(source)))

    if args.save:
        if report["best"] == "raw":
            print("no calibrator adopted; nothing saved.")
            return 0
        cal = (PlattCalibrator() if report["best"] == "platt" else IsotonicCalibrator())
        cal.fit(scores, labels)
        save_calibrator(cal, args.save)
        print(f"saved {report['best']} calibrator to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
