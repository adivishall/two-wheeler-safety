"""Build a de-duplicated, leakage-free training split.

`audit_dataset.py` measured two problems with the shipped training data:

1. **47.7% of the training images are perceptual duplicates** of another
   training image (11,195 files, 5,860 unique scenes, some repeated 22 times).
   The dataset's "perfect" 4,862 / 4,862 / 4,862 class balance is achieved by
   that duplication, and it hides a real imbalance in *unique* content:

       Plate          6,605 nominal ->  3,534 unique  (1.9x duplicated)
       WithoutHelmet  4,862 nominal ->  2,848 unique  (1.7x)
       TripleRiding   4,862 nominal ->  2,292 unique  (2.1x)
       WithHelmet     4,862 nominal ->  1,867 unique  (2.6x)  <- rarest

   `WithHelmet` — the weakest class — is the most heavily duplicated and has
   34% less unique data than `WithoutHelmet`, the class it is most confused
   with. Nominal counts made that invisible.

2. **Held-out images appear in training** (9.8% of test, 8.1% of val). The
   previous fix shrank the held-out splits; this one removes the offending
   *training* images instead, which is the better direction: the evaluation set
   stays at full size and keeps its statistical power, and the model simply
   never sees held-out content.

This script materialises the cleaned split as a symlinked YOLO tree, so it
costs no disk and is obviously derived. Nothing is copied or modified in place.

    python3 build_train_split.py --data master_traffic_violation_dataset/data.yaml \\
        --out datasets/train_clean

The result is a normal YOLO dataset that `train_traffic.py` consumes unchanged.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from modules.dataset_audit import hamming, hash_directory
from modules.logging_setup import configure_logging, get_logger

log = get_logger("build_train_split")


def choose_unique(hashes: dict[str, int]) -> dict[int, str]:
    """One representative path per distinct dHash, chosen deterministically.

    Sorting before the first-wins pick means the same representative is chosen
    on every machine and every rerun, so a training set built twice is the same
    training set.
    """
    keep: dict[int, str] = {}
    for path in sorted(hashes):
        keep.setdefault(hashes[path], path)
    return keep


def drop_heldout_matches(
    candidates: list[str],
    train_hashes: dict[str, int],
    heldout_hashes: dict[str, int],
    *,
    max_distance: int = 5,
) -> tuple[list[str], int]:
    """Remove training images that duplicate any held-out image.

    Exact matches go through a set lookup; the near-match pass only runs when a
    non-zero distance is requested. Removing from *training* rather than from
    the held-out split keeps the evaluation set at full size — 194 test images
    is already small enough that dropping 19 of them costs real precision in the
    metric.
    """
    held_exact = set(heldout_hashes.values())
    held_list = sorted(held_exact)
    kept, dropped = [], 0
    for path in candidates:
        h = train_hashes[path]
        if h in held_exact:
            dropped += 1
            continue
        if max_distance > 0 and any(
            hamming(h, hh) <= max_distance for hh in held_list
        ):
            dropped += 1
            continue
        kept.append(path)
    return kept, dropped


def class_counts(paths: list[str], class_names: list[str]) -> dict[str, int]:
    """Per-class instance counts over the label files of ``paths``."""
    counts: collections.Counter = collections.Counter()
    for p in paths:
        label = os.path.splitext(
            p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        )[0] + ".txt"
        if not os.path.exists(label):
            continue
        with open(label) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 5:
                    idx = int(float(parts[0]))
                    if 0 <= idx < len(class_names):
                        counts[class_names[idx]] += 1
    return dict(counts)


def _link(src: str, dst: str) -> None:
    if os.path.islink(dst) or os.path.exists(dst):
        os.unlink(dst)
    os.symlink(src, dst)


def materialise(paths: list[str], out_root: str, split: str = "train") -> int:
    """Symlink the chosen images and their labels into a YOLO split tree."""
    img_dir = os.path.join(out_root, split, "images")
    lbl_dir = os.path.join(out_root, split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    n = 0
    for src in paths:
        base = os.path.basename(src)
        label_src = os.path.splitext(
            src.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        )[0] + ".txt"
        _link(os.path.abspath(src), os.path.join(img_dir, base))
        if os.path.exists(label_src):
            _link(os.path.abspath(label_src),
                  os.path.join(lbl_dir, os.path.splitext(base)[0] + ".txt"))
        n += 1
    return n


def load_cfg(data_yaml: str) -> dict:
    import yaml

    with open(data_yaml) as fh:
        return yaml.safe_load(fh)


def resolve(cfg: dict, data_yaml: str, entry) -> str | None:
    if not entry:
        return None
    root = cfg.get("path") or os.path.dirname(os.path.abspath(data_yaml))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), root)
    return entry if os.path.isabs(entry) else os.path.join(root, entry)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Build a de-duplicated, leakage-free training split.",
    )
    ap.add_argument("--data", required=True, help="source dataset data.yaml")
    ap.add_argument("--out", default="datasets/train_clean",
                    help="where to materialise the cleaned split")
    ap.add_argument("--max-distance", type=int, default=5,
                    help="dHash distance treated as a duplicate (default: 5)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="only drop held-out matches, keep internal duplicates "
                         "(ablation: isolates the leakage fix from the dedup)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)
    if not os.path.exists(args.data):
        log.error("data.yaml not found: %s", args.data)
        return 2

    cfg = load_cfg(args.data)
    names = cfg.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    train_dir = resolve(cfg, args.data, cfg.get("train"))
    if not train_dir or not os.path.isdir(train_dir):
        log.error("train images dir not found: %s", train_dir)
        return 2

    log.info("hashing training images in %s", train_dir)
    train_hashes = hash_directory(train_dir)
    log.info("hashed %d training images", len(train_hashes))

    if args.keep_duplicates:
        candidates = sorted(train_hashes)
        unique_n = len(candidates)
    else:
        unique = choose_unique(train_hashes)
        candidates = sorted(unique.values())
        unique_n = len(candidates)
    log.info("after dedup: %d images (%d removed)",
             unique_n, len(train_hashes) - unique_n)

    heldout: dict[str, int] = {}
    for split in ("val", "test"):
        d = resolve(cfg, args.data, cfg.get(split))
        if d and os.path.isdir(d):
            log.info("hashing held-out split %s", split)
            heldout.update(hash_directory(d))
    log.info("hashed %d held-out images", len(heldout))

    kept, leaked = drop_heldout_matches(
        candidates, train_hashes, heldout, max_distance=args.max_distance
    )
    log.info("dropped %d training images that duplicate held-out data", leaked)

    before = class_counts(sorted(train_hashes), names)
    after = class_counts(kept, names)

    n = materialise(kept, args.out)

    # The cleaned split trains; validation and test still point at the ORIGINAL
    # held-out directories, untouched and at full size. That is the whole point
    # of cleaning the training side instead of the evaluation side.
    out_yaml = os.path.join(args.out, "data.yaml")
    val_dir = resolve(cfg, args.data, cfg.get("val"))
    test_dir = resolve(cfg, args.data, cfg.get("test"))
    import yaml

    doc = {
        "path": os.path.abspath(args.out),
        "train": "train/images",
        "val": val_dir,
        "nc": cfg.get("nc"),
        "names": names,
    }
    if test_dir:
        doc["test"] = test_dir
    os.makedirs(args.out, exist_ok=True)
    with open(out_yaml, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)

    report = {
        "source": args.data,
        "out": os.path.abspath(args.out),
        "data_yaml": out_yaml,
        "images_before": len(train_hashes),
        "images_after_dedup": unique_n,
        "dropped_internal_duplicates": len(train_hashes) - unique_n,
        "dropped_heldout_matches": leaked,
        "images_final": n,
        "class_counts_before": before,
        "class_counts_after": after,
        "duplication_factor": {
            k: round(before[k] / after[k], 2) for k in before if after.get(k)
        },
        "note": (
            "Validation and test still point at the ORIGINAL held-out "
            "directories at full size; only training images were removed."
        ),
    }
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/train_split_clean.json", "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(f"\n{'class':16s} {'before':>8} {'after':>8} {'dup factor':>11}")
    for c in names:
        print(f"{c:16s} {before.get(c, 0):>8} {after.get(c, 0):>8} "
              f"{report['duplication_factor'].get(c, 0):>10.2f}x")
    print(f"\nimages: {len(train_hashes)} -> {n} "
          f"({report['dropped_internal_duplicates']} duplicates, "
          f"{leaked} held-out matches removed)")
    print(f"data.yaml: {out_yaml}")
    print("Report:    eval/results/train_split_clean.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
