"""Manage a labelled field dataset (see docs/FIELD_EVALUATION.md).

    python3 field_dataset.py validate  DIR [--check-files]
    python3 field_dataset.py assign-splits DIR [--external-camera CAM ...]
                                     [--group-by sequence|camera_day] [--salt S]
    python3 field_dataset.py coverage  DIR
    python3 field_dataset.py export-train DIR OUT [--min-confidence probable]

``assign-splits`` writes ``DIR/splits.lock.json``; existing assignments never
change. ``export-train`` writes DEVELOPMENT frames only, in YOLO format, and
refuses (exit 3) if any would-be training image is an evaluation frame, a copy
of one, or a near-duplicate of one.
"""

from __future__ import annotations

import argparse
import json
import sys

from modules.field_data import (
    DEFAULT_RATIOS,
    FieldDataError,
    LeakageError,
    assign_splits,
    coverage,
    export_training,
    load_dataset,
    read_lock,
)


def _ratios(text: str | None) -> dict:
    if not text:
        return dict(DEFAULT_RATIOS)
    out = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        out[name.strip()] = float(value)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Field dataset: validate, split, export.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("root")
    v.add_argument("--check-files", action="store_true")
    a = sub.add_parser("assign-splits")
    a.add_argument("root")
    a.add_argument("--external-camera", action="append", default=[])
    a.add_argument("--group-by", default="sequence", choices=["sequence", "camera_day"])
    a.add_argument("--salt", default=None)
    a.add_argument("--ratios", default=None,
                   help="e.g. development=0.6,validation=0.2,held_out=0.2")
    c = sub.add_parser("coverage")
    c.add_argument("root")
    e = sub.add_parser("export-train")
    e.add_argument("root")
    e.add_argument("out")
    e.add_argument("--min-confidence", default="probable",
                   choices=["uncertain", "probable", "certain"])
    e.add_argument("--no-perceptual", action="store_true",
                   help="skip the near-duplicate check (faster; weaker guard)")
    args = ap.parse_args(argv)

    try:
        ds = load_dataset(args.root, check_files=getattr(args, "check_files", False))
    except FieldDataError as exc:
        for issue in exc.issues:
            print(f"ERROR  {issue}")
        print(f"\n{len(exc.issues)} error(s); dataset not usable.")
        return 2

    if args.cmd == "validate":
        for w in ds.warnings:
            print(f"WARN   {w}")
        print(f"ok: {len(ds.cameras)} cameras, {len(ds.sequences)} sequences, "
              f"{len(ds.vehicles)} vehicles, {len(ds.frames)} frames, "
              f"{len(ds.warnings)} warning(s)")
        return 0
    try:
        if args.cmd == "assign-splits":
            lock = assign_splits(ds, salt=args.salt, ratios=_ratios(args.ratios),
                                 external_cameras=args.external_camera,
                                 group_by=args.group_by)
            counts: dict = {}
            for rec in lock["assignments"].values():
                counts[rec["split"]] = counts.get(rec["split"], 0) + 1
            print(f"groups per split: {counts}  (lock {lock['lock_hash']})")
            return 0
        lock = read_lock(args.root)
        if args.cmd == "coverage":
            print(json.dumps(coverage(ds, lock), indent=2, sort_keys=True))
            return 0
        if lock is None:
            print("no splits.lock.json: run assign-splits first", file=sys.stderr)
            return 2
        manifest = export_training(ds, lock, args.out, min_confidence=args.min_confidence,
                                   perceptual=not args.no_perceptual)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except LeakageError as exc:
        for problem in exc.problems:
            print(f"LEAK   {problem}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
