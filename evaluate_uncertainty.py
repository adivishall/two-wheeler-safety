"""Detector metrics with confidence intervals, and model comparisons that say
whether a difference is real.

    # select on VALIDATION (thresholds + model choice) ...
    python3 evaluate_uncertainty.py --split val --name uncertainty_val \\
        --models runs/detect/traffic_model-2/weights/best.pt \\
                 runs/detect/traffic_model_v2_dedup/weights/best.pt
    # ... then report once on TEST, applying the thresholds chosen on val
    python3 evaluate_uncertainty.py --split test --name uncertainty_test \\
        --thresholds-from eval/results/uncertainty_val.json --models ...

For each model: AP@50 per class and mAP@50 with image-bootstrap 95% CIs. For
each model after the first: a *paired* bootstrap of the AP difference against
the first (the baseline), on the same resampled images. On a val run it also
picks each class's F1-optimal confidence threshold; a test run given
``--thresholds-from`` applies those thresholds unchanged, so the test numbers
are not tuned on test.

Predictions are made once per (weights hash, split fingerprint, settings) and
cached under ``eval/cache/`` (gitignored), so re-running the statistics is cheap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from modules import yolo_io
from modules.detection_stats import (
    apply_thresholds,
    bootstrap_ap,
    f1_optimal_thresholds,
    paired_bootstrap,
    records_from_json,
    records_to_json,
)
from modules.logging_setup import configure_logging, get_logger

log = get_logger("uncertainty")

DEFAULT_DATA = "eval/clean_splits/data.yaml"
DEFAULT_MODEL = "runs/detect/traffic_model-2/weights/best.pt"


def model_label(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    return parts[-3] if len(parts) >= 3 and parts[-2] == "weights" else os.path.basename(path)


def load_or_predict(model_path, data_yaml, split, *, imgsz, device, cache_dir,
                    corruption=None, corrupt_fn=None):
    """Cached AP records for one model on one split (optionally corrupted)."""
    from modules.model_manifest import sha256_file
    from modules.provenance import split_fingerprint

    data = yolo_io.load_data_yaml(data_yaml)
    key = yolo_io.cache_key(
        weights=sha256_file(model_path), split=split_fingerprint(data_yaml, split),
        conf=yolo_io.AP_CONF, nms=yolo_io.AP_NMS_IOU, imgsz=imgsz, corruption=corruption,
    )
    path = os.path.join(cache_dir, f"pred_{key}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return records_from_json(json.load(fh)["records"]), data["names"]

    from ultralytics import YOLO

    log.info("predicting %s on %s/%s%s", model_label(model_path), data_yaml, split,
             f" [{corruption}]" if corruption else "")
    model = YOLO(model_path)
    pairs = list(yolo_io.iter_image_label_pairs(yolo_io.split_dir(data, split)))
    records = yolo_io.predict_split(model, pairs, len(data["names"]), imgsz=imgsz,
                                    device=device, corrupt=corrupt_fn)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"model": model_path, "split": split, "corruption": corruption,
                   "records": records_to_json(records)}, fh)
    return records, data["names"]


def _fmt_ci(d: dict, key: str = "ap50") -> str:
    v = d.get(key, d.get("value"))
    if v is None:
        return "—"
    return f"{v:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def render_markdown(p: dict) -> str:
    names = p["class_names"]
    lines = [
        f"# Detector metrics with uncertainty — `{p['split']}` split",
        "",
        f"- Generated: {p['provenance']['generated_at']}",
        f"- Data: `{p['data']}` split **{p['split']}** "
        f"({p['images']} images, fingerprint `{p['provenance']['dataset']['split_version']}`)",
        f"- Bootstrap: {p['n_boot']} image resamples, 95% percentile intervals, seed {p['seed']}",
        f"- AP@50 recomputed from cached predictions (conf ≥ {yolo_io.AP_CONF}, "
        f"NMS IoU {yolo_io.AP_NMS_IOU}, imgsz {p['imgsz']}); 101-point interpolation "
        "as in Ultralytics.",
        "",
        "## AP@50 per class, with 95% CI",
        "",
        "| model | " + " | ".join(names) + " | mAP@50 |",
        "|---|" + "---:|" * (len(names) + 1),
    ]
    for m in p["models"]:
        b = m["bootstrap"]
        cells = [_fmt_ci(b["per_class"][n]) for n in names]
        lines.append(f"| `{m['label']}` | " + " | ".join(cells) + f" | {_fmt_ci(b['map50'])} |")
    inst = p["models"][0]["bootstrap"]["per_class"]
    lines += ["", "Instances: " + ", ".join(f"{n} {inst[n]['instances']}" for n in names), ""]
    if len(p["models"]) > 1:
        base = p["models"][0]["label"]
        lines += [
            f"## Paired differences vs `{base}`",
            "",
            "AP(model) − AP(baseline) on the same resampled images. **Significant** "
            "only when the 95% CI excludes 0.",
            "",
            "| model | class | Δ AP@50 | 95% CI | P(model better) | verdict |",
            "|---|---|---:|---|---:|---|",
        ]
        for m in p["models"][1:]:
            pb = m["paired_vs_baseline"]
            for n in [*names, "map50"]:
                d = pb["map50"] if n == "map50" else pb["per_class"][n]
                if d["diff"] is None:  # no instances of this class in the split
                    lines.append(f"| `{m['label']}` | {n} | — | — | — | no instances |")
                    continue
                verdict = ("**better**" if d["significant"] and d["diff"] > 0 else
                           "**worse**" if d["significant"] else "not distinguishable")
                lines.append(
                    f"| `{m['label']}` | {'mAP@50' if n == 'map50' else n} | "
                    f"{d['diff']:+.3f} | [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] | "
                    f"{d['p_a_better']:.2f} | {verdict} |")
        lines.append("")
    for m in p["models"]:
        if "thresholds_selected" in m:
            lines += [f"## F1-optimal confidence thresholds (chosen on this split) — "
                      f"`{m['label']}`", "",
                      "| class | threshold | precision | recall | F1 |",
                      "|---|---:|---:|---:|---:|"]
            for n in names:
                t = m["thresholds_selected"][n]
                lines.append(f"| {n} | {t['threshold']} | {t['precision']} | "
                             f"{t['recall']} | {t['f1']} |")
            lines.append("")
        if "thresholds_applied" in m:
            src_note = ("its own val-selected thresholds"
                        if m.get("thresholds_source") == m["label"]
                        else f"`{m.get('thresholds_source')}`'s thresholds (none selected for it)")
            lines += [f"## `{m['label']}` at {src_note} from `{p['thresholds_from']}`", "",
                      "| class | threshold | precision | recall | F1 |",
                      "|---|---:|---:|---:|---:|"]
            for n in names:
                t = m["thresholds_applied"][n]
                lines.append(f"| {n} | {t['threshold']} | {t['precision']} | "
                             f"{t['recall']} | {t['f1']} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", default=[DEFAULT_MODEL],
                    help="weights to evaluate; the FIRST is the baseline")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bootstrap", type=int, default=2000, help="resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thresholds-from", default=None,
                    help="a val-split JSON from this tool; its per-class thresholds "
                         "(for the baseline model) are applied unchanged")
    ap.add_argument("--cache-dir", default="eval/cache")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--name", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)
    for m in args.models:
        if not os.path.exists(m):
            log.error("weights not found: %s", m)
            return 2
    if not os.path.exists(args.data):
        log.error("data.yaml not found: %s", args.data)
        return 2
    if args.split == "test" and not args.thresholds_from:
        log.warning("reporting on TEST without --thresholds-from: thresholds will be "
                    "reported for reference only, never selected here")

    from modules.provenance import run_provenance

    device = yolo_io.resolve_device(args.device)
    # Each model gets the thresholds selected for IT on val. (Applying the
    # baseline's thresholds to every model biased any comparison at operating
    # thresholds toward the baseline.) A model with no val selection falls back
    # to the baseline's, and the report says so.
    val_thresholds: dict[str, dict] = {}
    base_label = None
    if args.thresholds_from:
        with open(args.thresholds_from) as fh:
            src = json.load(fh)
        for m in src["models"]:
            if "thresholds_selected" in m:
                val_thresholds[m["label"]] = {k: v["threshold"]
                                              for k, v in m["thresholds_selected"].items()}
        base_label = src["models"][0]["label"]

    models = []
    base_records = None
    names: list[str] = []
    for i, mp in enumerate(args.models):
        records, names = load_or_predict(mp, args.data, args.split, imgsz=args.imgsz,
                                         device=device, cache_dir=args.cache_dir)
        entry = {"path": mp, "label": model_label(mp),
                 "bootstrap": bootstrap_ap(records, names, n_boot=args.bootstrap,
                                           seed=args.seed)}
        if i == 0:
            base_records = records
        else:
            entry["paired_vs_baseline"] = paired_bootstrap(
                records, base_records, names, n_boot=args.bootstrap, seed=args.seed)
        if args.split == "val":
            entry["thresholds_selected"] = f1_optimal_thresholds(records, names)
        if val_thresholds:
            own = val_thresholds.get(entry["label"])
            source = entry["label"] if own is not None else base_label
            entry["thresholds_source"] = source
            entry["thresholds_applied"] = apply_thresholds(
                records, names, own if own is not None else val_thresholds[base_label])
        models.append(entry)

    payload = {
        "name": args.name or f"uncertainty_{args.split}",
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "n_boot": args.bootstrap,
        "seed": args.seed,
        "class_names": names,
        "images": models[0]["bootstrap"]["images"],
        "thresholds_from": args.thresholds_from,
        "provenance": run_provenance(model_paths=args.models, data_yaml=args.data,
                                     split=args.split,
                                     config={"conf": yolo_io.AP_CONF,
                                             "nms_iou": yolo_io.AP_NMS_IOU,
                                             "imgsz": args.imgsz, "device": device}),
        "models": models,
    }
    os.makedirs(args.out, exist_ok=True)
    jp = os.path.join(args.out, payload["name"] + ".json")
    mp_ = os.path.join(args.out, payload["name"] + ".md")
    with open(jp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    with open(mp_, "w") as fh:
        fh.write(render_markdown(payload))
    print(f"wrote {mp_} and {jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
