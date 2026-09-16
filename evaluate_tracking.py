"""Crossing / overlap tracking evaluation and association-config A/B.

`evaluate_system.py` reported `crossing` as the pipeline's weak point. This
tool exists to attribute that weakness to a layer and to measure a fix, rather
than reaching for a more sophisticated tracker because one exists.

    python3 evaluate_tracking.py                 # A/B the old vs new config
    python3 evaluate_tracking.py --diagnose      # frame-by-frame attribution
    python3 evaluate_tracking.py --json eval/results/tracking_comparison.json

The headline finding, reproducible with ``--diagnose``: in the crossing
scenario the two vehicles' bodies reach IoU 0.43, above the old
``body_merge_iou`` of 0.3, so ``merge_bodies`` collapses two vehicles into one
**before the tracker is called**. Every ID switch happens on a frame where the
tracker was handed one anchor for two vehicles. No motion model can fix that,
so no Kalman filter was added.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from modules.association import merge_bodies
from modules.geometry import containment, iou
from modules.system_eval import _flatten, _gt_boxes, _owner_by_overlap
from modules.tracking_eval import (
    NEW_CONFIG,
    OLD_CONFIG,
    check_merge_behaviour,
    compare_configs,
    scenarios,
)
from modules.vehicle_tracker import VehicleTracker


def diagnose(scenario_name: str = "two_crossing_speed20") -> None:
    """Print the frame-by-frame attribution for one scenario under both configs."""
    by_name = {s.name: s for s in scenarios()}
    sc = by_name.get(scenario_name)
    if sc is None:
        print(f"unknown scenario: {scenario_name}")
        print("available:", ", ".join(sorted(by_name)))
        return

    for label, cfg in (("OLD", OLD_CONFIG), ("NEW", NEW_CONFIG)):
        print(f"\n=== {scenario_name} under {label} config "
              f"(merge_iou={cfg.body_merge_iou}, "
              f"containment={cfg.body_merge_containment}) ===")
        print(f"{'f':>3} {'GT':>3} {'bodies':>7} {'tracks':>7} {'bodyIoU':>8} "
              f"{'contain':>8}  mapping")
        tracker = VehicleTracker(config=cfg)
        last: dict = {}
        switches = 0
        for f, frame in enumerate(sc.frames):
            dets, _ = _flatten(frame)
            gt_boxes = _gt_boxes(frame)
            bodies = merge_bodies(dets, cfg)
            tracks = tracker.update(dets, f)
            body_boxes = [d.box for vf in frame for d in vf.dets if d.label != "Plate"]
            ov = con = 0.0
            if len(body_boxes) >= 2:
                ov = iou(body_boxes[0], body_boxes[1])
                con = max(containment(body_boxes[0], body_boxes[1]),
                          containment(body_boxes[1], body_boxes[0]))
            mapping = {}
            for t in tracks:
                g = _owner_by_overlap(t.box, gt_boxes)
                if g is not None:
                    mapping[g] = t.track_id
            sw = [g for g in mapping if g in last and last[g] != mapping[g]]
            switches += len(sw)
            last.update(mapping)
            flag = "  <-- ID SWITCH" if sw else ""
            merged = "  <-- MERGED" if gt_boxes and len(bodies) < len(gt_boxes) else ""
            print(f"{f:>3} {len(gt_boxes):>3} {len(bodies):>7} {len(tracks):>7} "
                  f"{ov:>8.3f} {con:>8.3f}  {dict(sorted(mapping.items()))}"
                  f"{merged}{flag}")
        print(f"total ID switches: {switches}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Tracking/association evaluation for crossing vehicles.",
    )
    ap.add_argument("--diagnose", nargs="?", const="two_crossing_speed20",
                    default=None, metavar="SCENARIO",
                    help="print frame-by-frame attribution for one scenario")
    ap.add_argument("--json", default="eval/results/tracking_comparison.json",
                    help="where to write the machine-readable comparison")
    args = ap.parse_args(argv)

    if args.diagnose:
        diagnose(args.diagnose)
        return 0

    report = compare_configs()
    report["merge_behaviour"] = {
        "old": check_merge_behaviour(OLD_CONFIG),
        "new": check_merge_behaviour(NEW_CONFIG),
    }

    cfg = report["config"]
    print("=== association config A/B ===")
    print(f"OLD: merge_iou={cfg['old']['body_merge_iou']}, "
          f"containment={cfg['old']['body_merge_containment']}")
    print(f"NEW: merge_iou={cfg['new']['body_merge_iou']}, "
          f"containment={cfg['new']['body_merge_containment']}\n")

    header = (f"{'scenario':28s} {'OLD idsw':>9} {'NEW idsw':>9} "
              f"{'OLD assoc':>10} {'NEW assoc':>10} "
              f"{'OLD fines':>12} {'NEW fines':>12}")
    print(header)
    print("-" * len(header))
    for name, row in report["scenarios"].items():
        o, n = row["old"], row["new"]
        fo, fn = o["system"], n["system"]
        print(f"{name:28s} {o['tracking']['id_switches']:>9} "
              f"{n['tracking']['id_switches']:>9} "
              f"{o['association']['accuracy']:>10.3f} "
              f"{n['association']['accuracy']:>10.3f} "
              f"{fo['true_positives']}/{fo['false_positives']}/"
              f"{fo['false_negatives']:<7} "
              f"{fn['true_positives']}/{fn['false_positives']}/"
              f"{fn['false_negatives']}")

    print()
    for key in ("old", "new"):
        t = report["totals"][key]
        print(f"{key.upper():4s} id_switches={t['id_switches']:<3} "
              f"spurious_merged_frames={t['merged_frames']:<3} "
              f"fragmentations={t['fragmentations']:<3} "
              f"assoc_accuracy={t['assoc_accuracy']:.3f} "
              f"wrong_assoc={t['assoc_wrong']:<3} "
              f"fines TP/FP/FN={t['fine_tp']}/{t['fine_fp']}/{t['fine_fn']}")

    mb = report["merge_behaviour"]
    print(f"\nmerge regression cases: OLD "
          f"{mb['old']['_summary']['correct']}/{mb['old']['_summary']['total']} "
          f"correct, NEW "
          f"{mb['new']['_summary']['correct']}/{mb['new']['_summary']['total']}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
