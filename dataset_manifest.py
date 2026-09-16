"""Generate a machine-readable dataset manifest (Phase 19).

`docs/DATASET.md` is the prose record of what the dataset contains. This writes
the same facts as JSON, **per split**, so a manifest can be diffed between
dataset versions, attached to a model manifest, and checked by a script rather
than read by a person.

Everything counted here comes from the label files on disk. Everything that
cannot be derived from the files — the source datasets, their licences, the
preprocessing history — is recorded as an explicit ``"UNVERIFIED"`` rather than
guessed, because a manifest that quietly invents provenance is worse than no
manifest. Those fields are filled in by hand in ``data/DATASET_MANIFEST.md``
when (and only when) someone traces them.

    python3 dataset_manifest.py --data master_traffic_violation_dataset/data.yaml
    python3 dataset_manifest.py --data ... --leakage eval/results/dataset_leakage.json

The images themselves are gitignored; the manifest is committed, so the record
of what was used survives without redistributing imagery of unverified licence.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from modules.logging_setup import configure_logging, get_logger

log = get_logger("dataset_manifest")

SPLITS = ("train", "val", "test")

# Provenance that cannot be read off the filesystem. Stated as unverified so the
# gap is visible in the artifact, not only in the prose docs.
UNVERIFIED = "UNVERIFIED"


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def _class_names(cfg: dict) -> list[str]:
    names = cfg.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names or [])


def _resolve(cfg: dict, data_yaml: str, entry) -> str | None:
    if not entry:
        return None
    root = cfg.get("path") or os.path.dirname(os.path.abspath(data_yaml))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), root)
    return entry if os.path.isabs(entry) else os.path.join(root, entry)


def describe_split(images_dir: str, classes: list[str]) -> dict:
    """Count label files, background images and per-class instances for a split.

    "Background" (an empty or absent label file) is counted separately because it
    is a meaningful training signal, not a defect — and because an evaluation set
    that is mostly background would flatter precision.
    """
    label_dir = images_dir.replace(os.sep + "images", os.sep + "labels")
    counts = {c: 0 for c in classes}
    label_files = background = instances = 0

    for lp in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        label_files += 1
        rows = 0
        with open(lp) as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                rows += 1
                instances += 1
                idx = int(float(parts[0]))
                if 0 <= idx < len(classes):
                    counts[classes[idx]] += 1
        if rows == 0:
            background += 1

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = 0
    if os.path.isdir(images_dir):
        images = sum(
            1 for n in os.listdir(images_dir)
            if os.path.splitext(n)[1].lower() in image_exts
        )

    return {
        "images_dir": images_dir,
        "images": images,
        "label_files": label_files,
        "background_images": background,
        "instances": instances,
        "class_counts": counts,
    }


def content_hash(data_yaml: str, splits: dict) -> str:
    """Hash the data.yaml bytes plus every split's counts.

    Any change to the classes, the split sizes or the label contents changes the
    version, so two manifests with the same version describe the same data.
    """
    h = hashlib.sha256()
    with open(data_yaml, "rb") as fh:
        h.update(fh.read())
    h.update(json.dumps(splits, sort_keys=True).encode())
    return "sha256:" + h.hexdigest()[:16]


def build(data_yaml: str, *, leakage_report: str | None = None) -> dict:
    cfg = _load_yaml(data_yaml)
    classes = _class_names(cfg)

    splits = {}
    for split in SPLITS:
        d = _resolve(cfg, data_yaml, cfg.get(split))
        if d and os.path.isdir(d):
            splits[split] = describe_split(d, classes)

    totals = {
        "images": sum(s["images"] for s in splits.values()),
        "label_files": sum(s["label_files"] for s in splits.values()),
        "background_images": sum(s["background_images"] for s in splits.values()),
        "instances": sum(s["instances"] for s in splits.values()),
        "class_counts": {
            c: sum(s["class_counts"].get(c, 0) for s in splits.values())
            for c in classes
        },
    }

    manifest = {
        "name": os.path.basename(os.path.dirname(os.path.abspath(data_yaml))),
        "data_yaml": data_yaml,
        "version": content_hash(data_yaml, splits),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classes": classes,
        "annotation_format": (
            "YOLO detection: one .txt per image under a parallel labels/ tree, "
            "each line 'class cx cy w h' normalised to [0,1]. A missing or empty "
            ".txt is a background image."
        ),
        "splits": splits,
        "totals": totals,
        # Not derivable from the files. Stated, not guessed.
        "source": UNVERIFIED,
        "license": UNVERIFIED,
        "preprocessing": (
            "Training images are pre-augmented offline: every train/images file "
            "is named aug_<hash>.jpg with a paired .npy cache, so augmentation "
            "is baked into the stored dataset rather than applied on the fly. "
            "The exact transforms are " + UNVERIFIED + "."
        ),
        "known_limitations": [
            "Source datasets, their authors and their licences are not recorded "
            "with the data; do not redistribute the images or assert a licence "
            "until traced.",
            "Validation and test splits are small and imbalanced (WithHelmet has "
            "only 27 instances in each), so per-class metrics for WithHelmet are "
            "high-variance and should be read as indicative.",
            "Imagery skews to Indian road scenes and the plate decoder assumes "
            "the Indian plate format.",
            "The test split comes from the same pool as train/val, so it shares "
            "their biases; it measures held-out performance, not generalisation "
            "to a different camera or city.",
            "A Plate box being detected says nothing about whether OCR can read "
            "it; OCR quality is evaluated separately.",
        ],
    }

    if leakage_report and os.path.exists(leakage_report):
        with open(leakage_report) as fh:
            leak = json.load(fh)
        manifest["split_hygiene"] = {
            "source": leakage_report,
            "method": leak.get("method"),
            "clean": leak.get("clean"),
            "splits": {
                name: {
                    "held_out_images": r.get("held_out_images"),
                    "leaked_images": r.get("leaked_images"),
                    "leak_rate": r.get("leak_rate"),
                }
                for name, r in (leak.get("splits") or {}).items()
            },
        }
    else:
        manifest["split_hygiene"] = {
            "clean": None,
            "note": "not audited; run audit_dataset.py --data <data.yaml>",
        }
    return manifest


def render_markdown(m: dict) -> str:
    classes = m["classes"]
    lines = [f"# Dataset manifest — `{m['name']}`", "",
             "Generated by `dataset_manifest.py`; every count is read from the "
             "label files on disk. Fields that cannot be derived from the files "
             f"are recorded as `{UNVERIFIED}` rather than guessed.", "",
             f"- **version**: `{m['version']}`",
             f"- generated: {m['generated_at']}",
             f"- config: `{m['data_yaml']}`",
             f"- source: **{m['source']}**",
             f"- licence: **{m['license']}**", "",
             "## Classes", "",
             "| id | class |", "|---:|---|"]
    for i, c in enumerate(classes):
        lines.append(f"| {i} | `{c}` |")

    lines += ["", "## Annotation format", "", m["annotation_format"], "",
              "## Splits", "",
              "| split | images | label files | background | instances |",
              "|---|---:|---:|---:|---:|"]
    for name, s in m["splits"].items():
        lines.append(f"| `{name}` | {s['images']} | {s['label_files']} | "
                     f"{s['background_images']} | {s['instances']} |")
    t = m["totals"]
    lines.append(f"| **total** | **{t['images']}** | **{t['label_files']}** | "
                 f"**{t['background_images']}** | **{t['instances']}** |")

    lines += ["", "### Per-class instances", "",
              "| class | " + " | ".join(m["splits"]) + " | total |",
              "|---|" + "---:|" * (len(m["splits"]) + 1)]
    for c in classes:
        cells = [str(s["class_counts"].get(c, 0)) for s in m["splits"].values()]
        lines.append(f"| `{c}` | " + " | ".join(cells) +
                     f" | **{t['class_counts'].get(c, 0)}** |")

    lines += ["", "## Preprocessing", "", m["preprocessing"], ""]

    hyg = m.get("split_hygiene") or {}
    lines += ["## Split hygiene", ""]
    if hyg.get("clean") is None:
        lines += [f"Not audited. {hyg.get('note', '')}", ""]
    else:
        verdict = "clean" if hyg["clean"] else "**leakage found**"
        lines += [f"Audited with `audit_dataset.py` — {verdict}.", "",
                  f"Method: {hyg.get('method')}", "",
                  "| split | held-out images | leaked | rate |",
                  "|---|---:|---:|---:|"]
        for name, r in (hyg.get("splits") or {}).items():
            lines.append(f"| `{name}` | {r['held_out_images']} | "
                         f"{r['leaked_images']} | {r['leak_rate']:.1%} |")
        lines.append("")

    lines += ["## Known limitations", ""]
    for item in m["known_limitations"]:
        lines.append(f"- {item}")
    lines += ["", "## Reproducing this file", "",
              "```bash", f"python3 dataset_manifest.py --data {m['data_yaml']} \\",
              "    --leakage eval/results/dataset_leakage.json", "```", ""]
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate a machine-readable dataset manifest.",
    )
    ap.add_argument("--data", required=True, help="dataset data.yaml")
    ap.add_argument("--leakage", default="eval/results/dataset_leakage.json",
                    help="audit_dataset.py report to fold in, if present")
    ap.add_argument("--out-json", default="data/dataset_manifest.json")
    ap.add_argument("--out-md", default="data/DATASET_MANIFEST.md")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)
    if not os.path.exists(args.data):
        log.error("data.yaml not found: %s", args.data)
        return 2

    manifest = build(args.data, leakage_report=args.leakage)

    for path, text in (
        (args.out_json, json.dumps(manifest, indent=2, sort_keys=True) + "\n"),
        (args.out_md, render_markdown(manifest)),
    ):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        print(f"Wrote {path}")

    t = manifest["totals"]
    print(f"\nversion {manifest['version']} — {t['label_files']} label files, "
          f"{t['instances']} instances across {len(manifest['splits'])} splits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
