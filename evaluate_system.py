"""Whole-system evaluation CLI (Phases 6/7/9).

Runs the deterministic synthetic scenarios in `modules.system_eval` through the
real tracking / association / OCR-voting / temporal-confirmation / confidence /
fine logic and reports:

* **system**      — TP / FP / FN, wrong-vehicle, wrong-plate, duplicate fines.
* **tracking**    — ID switches, fragmentation, false tracks, GT coverage.
* **association** — correct / wrong / missed / ambiguous rider<->plate pairs.

This is deliberately separate from the raw YOLO metrics in `evaluate_model.py`:
it measures the *pipeline logic*, not detector quality, and needs no model
weights. Nothing is fabricated — the scenarios are fixed and the numbers are
produced by running the shipped code.

    python3 evaluate_system.py            # print the summary
    python3 evaluate_system.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys

from modules.system_eval import evaluate_all


def _print(out: dict) -> None:
    print("=== whole-system evaluation (synthetic scenarios) ===\n")
    header = f"{'scenario':16s} {'TP':>3} {'FP':>3} {'FN':>3} {'wV':>3} {'wP':>3} " \
             f"{'dup':>3} | {'idsw':>4} {'frag':>4} {'false':>5} {'cov':>5} | " \
             f"{'assoc_acc':>9}"
    print(header)
    print("-" * len(header))
    for name, r in out["scenarios"].items():
        s, t, a = r["system"], r["tracking"], r["association"]
        print(f"{name:16s} {s['true_positives']:>3} {s['false_positives']:>3} "
              f"{s['false_negatives']:>3} {s['wrong_vehicle']:>3} "
              f"{s['wrong_plate']:>3} {s['duplicates']:>3} | "
              f"{t['id_switches']:>4} {t['fragmentations']:>4} "
              f"{t['false_tracks']:>5} {t['coverage']:>5.2f} | "
              f"{a['accuracy']:>9.2f}")
    tot = out["totals"]
    print("\nSYSTEM TOTALS: "
          f"TP={tot['true_positives']} FP={tot['false_positives']} "
          f"FN={tot['false_negatives']} wrong_vehicle={tot['wrong_vehicle']} "
          f"wrong_plate={tot['wrong_plate']} duplicates={tot['duplicates']}")
    print(f"precision={tot['precision']:.3f}  recall={tot['recall']:.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Whole-system synthetic evaluation.")
    ap.add_argument("--json", default=None, help="also write the full report here")
    args = ap.parse_args(argv)

    out = evaluate_all()
    _print(out)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
