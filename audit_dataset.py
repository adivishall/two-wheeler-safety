"""Dataset split-hygiene CLI — verify the held-out data is actually held out.

Reports, per held-out split, how many images are near-duplicates of a training
image (perceptual dHash, see ``modules/dataset_audit.py``), plus near-duplicate
pairs *within* the split. A clean report is the licence to quote held-out metrics
as generalisation evidence; a dirty one means the metrics are inflated and the
split must be rebuilt before anything is claimed.

    python3 audit_dataset.py --data master_traffic_violation_dataset/data.yaml
    python3 audit_dataset.py --data ... --max-distance 3 --json eval/results/leakage.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from modules.dataset_audit import audit_splits
from modules.logging_setup import configure_logging, get_logger

log = get_logger("audit_dataset")


def split_dirs(data_yaml: str) -> tuple[str, dict[str, str]]:
    """Resolve ``data.yaml`` into (train_images_dir, {split: images_dir})."""
    import yaml

    with open(data_yaml) as fh:
        cfg = yaml.safe_load(fh)
    root = cfg.get("path") or os.path.dirname(os.path.abspath(data_yaml))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), root)

    def resolve(entry):
        if not entry:
            return None
        return entry if os.path.isabs(entry) else os.path.join(root, entry)

    train = resolve(cfg.get("train"))
    held = {}
    for split in ("val", "test"):
        # data.yaml conventionally spells the validation split "val"; the
        # on-disk directory is often "valid/images".
        d = resolve(cfg.get(split))
        if d and os.path.isdir(d):
            held[split] = d
    return train, held


def _write_clean_yaml(source_yaml: str, out_root: str, built: dict) -> str:
    """Write a data.yaml for the de-leaked splits, reusing the source class names.

    Ultralytics requires both ``train:`` and ``val:`` keys even when only
    ``test`` is evaluated. ``train`` is pointed back at the *original* training
    images (absolute) — nothing ever trains from this file, and pointing it at a
    clean split would be a lie about what the model saw; ``val`` falls back to
    whichever clean split exists so the file is always loadable.
    """
    import yaml

    with open(source_yaml) as fh:
        cfg = yaml.safe_load(fh)
    src_root = cfg.get("path") or os.path.dirname(os.path.abspath(source_yaml))
    if not os.path.isabs(src_root):
        src_root = os.path.join(os.path.dirname(os.path.abspath(source_yaml)), src_root)
    out = {"path": os.path.abspath(out_root), "nc": cfg.get("nc"),
           "names": cfg.get("names")}
    for split in built:
        out[split] = f"{split}/images"
    train_entry = cfg.get("train")
    if train_entry:
        out["train"] = (
            train_entry if os.path.isabs(train_entry)
            else os.path.join(src_root, train_entry)
        )
    out.setdefault("val", next(f"{s}/images" for s in built))
    path = os.path.join(out_root, "data.yaml")
    os.makedirs(out_root, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(out, fh, sort_keys=False)
    return path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Check held-out splits for leakage from the training split.",
    )
    ap.add_argument("--data", required=True, help="dataset data.yaml")
    ap.add_argument("--max-distance", type=int, default=5,
                    help="max dHash Hamming distance counted as a duplicate (default: 5)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images hashed per split (0 = all); for a quick smoke run")
    ap.add_argument("--json", default="eval/results/dataset_leakage.json",
                    help="where to write the machine-readable report")
    ap.add_argument("--write-clean-split", metavar="DIR", default=None,
                    help="materialise de-leaked copies of the held-out splits "
                         "(symlinked YOLO trees + data.yaml) under DIR, so the "
                         "honest generalisation number can be measured")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)
    if not os.path.exists(args.data):
        log.error("data.yaml not found: %s", args.data)
        return 2

    train_dir, held = split_dirs(args.data)
    if not train_dir or not os.path.isdir(train_dir):
        log.error("train images dir not found (%s)", train_dir)
        return 2
    if not held:
        log.error("no val/test split directories found in %s", args.data)
        return 2

    log.info("hashing %s and %d held-out split(s)", train_dir, len(held))
    report = audit_splits(train_dir, held, max_distance=args.max_distance,
                          limit=args.limit)

    print("\n=== dataset split hygiene ===")
    print(f"method: {report['method']}")
    print(f"train images hashed: {report['train_images']}\n")
    for split, r in report["splits"].items():
        print(f"{split:6s} images={r['held_out_images']:<5d} "
              f"leaked={r['leaked_images']:<4d} ({r['leak_rate']:.3%})  "
              f"internal duplicate pairs={r['internal_duplicate_pairs']}")
        for ex in r["examples"][:5]:
            print(f"        d={ex['distance']}  {os.path.basename(ex['held_out_image'])}"
                  f"  <->  {os.path.basename(ex['train_image'])}")
    print(f"\nCLEAN: {report['clean']}")

    if args.write_clean_split:
        from modules.dataset_audit import build_clean_split
        built = {}
        for split, r in report["splits"].items():
            built[split] = build_clean_split(
                held[split], r["leaked_paths"], args.write_clean_split,
                split_name=split,
            )
            print(f"clean {split}: kept {built[split]['kept']}, "
                  f"dropped {built[split]['dropped_leaked']} leaked "
                  f"-> {built[split]['root']}/{split}")
        yaml_path = _write_clean_yaml(args.data, args.write_clean_split, built)
        report["clean_split"] = {"data_yaml": yaml_path, "splits": built}
        print(f"clean data.yaml: {yaml_path}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"Wrote {args.json}")
    # Exit 1 on leakage so CI / a retraining loop can gate on it.
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
