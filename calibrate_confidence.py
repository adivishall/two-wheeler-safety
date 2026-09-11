"""Confidence-calibration experiment CLI (Phase 10).

Reads labelled (confidence, correct) pairs and reports whether the raw
confidence score can be turned into a calibrated probability — ECE, MCE and
Brier before/after Platt and isotonic calibration — and optionally saves the
best calibrator for the runtime to load.

Labels CSV: a header plus columns ``confidence,correct`` where ``correct`` is
1/0 (or true/false) from ground truth — e.g. exported from the human-review
workflow, joining each recorded violation's stored ``confidence`` to the
reviewer's confirmed/dismissed decision.

    python3 calibrate_confidence.py --labels review_labels.csv
    python3 calibrate_confidence.py --labels review_labels.csv \
        --save models/confidence_calibrator.json

Nothing is applied automatically: a probability is only ever exposed after a
calibrator is fitted here on real data and the report shows it actually helps.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from modules.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    fit_and_evaluate,
    save_calibrator,
)

_TRUE = {"1", "true", "yes", "confirmed", "correct", "y", "t"}


def read_labels(path: str):
    scores, labels = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row.get("confidence") or row.get("score")
            lab = row.get("correct") or row.get("label") or row.get("decision")
            if raw is None or lab is None:
                continue
            scores.append(float(raw))
            labels.append(1 if str(lab).strip().lower() in _TRUE else 0)
    return scores, labels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Confidence calibration experiment.")
    ap.add_argument("--labels", required=True, help="CSV: confidence,correct")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--save", default=None,
                    help="write the best calibrator here (only if it beats raw)")
    args = ap.parse_args(argv)

    scores, labels = read_labels(args.labels)
    if not scores:
        print("no usable (confidence, correct) rows", file=sys.stderr)
        return 2

    report = fit_and_evaluate(scores, labels, n_bins=args.bins)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nn={report['n']}  best={report['best']}")
    print(f"raw ECE={report['raw']['ece']}  Brier={report['raw']['brier']}")

    if args.save:
        if report["best"] == "raw":
            print("\ncalibration did not improve on the raw score; nothing saved.")
            return 0
        best = report["best"]
        cal = (PlattCalibrator() if best == "platt" else IsotonicCalibrator())
        cal.fit(scores, labels)
        save_calibrator(cal, args.save)
        print(f"\nsaved {best} calibrator to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
