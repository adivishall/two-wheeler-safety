"""Plate-OCR evaluation and preprocessing experiments (Phases 4-5).

Runs the OCR reader over a labelled set of plate images and reports the metrics
in `modules.ocr_eval` — exact / normalized / character accuracy, edit distance,
invalid-output rate, and confidence-vs-correctness — optionally broken down by a
`condition` column and optionally compared across the preprocessing pipelines in
`modules.ocr_preprocess`, so a preprocessing step is kept only if it *measurably*
helps.

Labels file: a CSV with a header and columns ``image_path,plate`` and an
optional ``condition`` (clean/blur/angle/lowlight/small/occluded). Paths are
resolved relative to the CSV's directory.

    python3 evaluate_ocr.py --labels ocr_eval.csv
    python3 evaluate_ocr.py --labels ocr_eval.csv --compare-preprocess

The scoring is model-free (`modules.ocr_eval`, unit-tested in CI); only the OCR
pass itself needs EasyOCR, which is imported lazily so `--help` and the metrics
import work without the heavy stack.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

from modules.detector import clean_plate
from modules.logging_setup import configure_logging, get_logger
from modules.ocr_eval import OcrObservation, evaluate_with_conditions
from modules.ocr_preprocess import PIPELINES, apply_pipeline

log = get_logger("evaluate_ocr")


def read_labels(path: str) -> list[dict]:
    """Read the labels CSV into ``[{image_path, plate, condition}, ...]`` with
    image paths resolved relative to the CSV location."""
    base = os.path.dirname(os.path.abspath(path))
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            img = row.get("image_path") or row.get("image") or row.get("path")
            plate = row.get("plate") or row.get("truth") or ""
            if not img:
                continue
            if not os.path.isabs(img):
                img = os.path.join(base, img)
            rows.append({
                "image_path": img,
                "plate": plate.strip(),
                "condition": (row.get("condition") or "all").strip() or "all",
            })
    return rows


def ocr_one(reader, image, pipeline: str) -> tuple[str, float]:
    """Run OCR on one already-loaded BGR image through a named preprocessing
    pipeline. Returns ``(cleaned_text, mean_confidence)``."""
    processed = apply_pipeline(image, pipeline)
    # detail=1 gives (box, text, confidence) per detected line.
    results = reader.readtext(processed, detail=1)
    text = clean_plate("".join(r[1] for r in results))
    confs = [float(r[2]) for r in results if len(r) > 2]
    return text, (sum(confs) / len(confs) if confs else 0.0)


def run(labels: list[dict], reader, pipeline: str) -> list[OcrObservation]:
    """OCR every labelled image through ``pipeline`` into observations."""
    import cv2

    observations = []
    for row in labels:
        img = cv2.imread(row["image_path"])
        if img is None:
            log.warning("could not read %s; skipping", row["image_path"])
            continue
        pred, conf = ocr_one(reader, img, pipeline)
        observations.append(OcrObservation(
            truth=row["plate"], pred=pred, confidence=conf,
            condition=row["condition"],
        ))
    return observations


def _print_report(title: str, report) -> None:
    o = report.overall.as_dict()
    print(f"\n=== {title} (n={o['n']}) ===")
    print(f"  exact match:      {o['exact_match']:.3f}")
    print(f"  normalized match: {o['normalized_match']:.3f}")
    print(f"  char accuracy:    {o['char_accuracy']:.3f}")
    print(f"  mean edit dist:   {o['mean_edit_distance']:.3f}")
    print(f"  invalid rate:     {o['invalid_rate']:.3f}")
    print(f"  conf separation:  {o['confidence_separation']:.3f} "
          f"(correct {o['mean_conf_correct']:.3f} / wrong {o['mean_conf_wrong']:.3f})")
    if len(report.by_condition) > 1:
        print("  by condition:")
        for cond, m in report.by_condition.items():
            md = m.as_dict()
            print(f"    {cond:12s} n={md['n']:<4d} "
                  f"norm={md['normalized_match']:.3f} char={md['char_accuracy']:.3f}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate plate OCR on a labelled set.")
    ap.add_argument("--labels", required=True, help="CSV: image_path,plate[,condition]")
    ap.add_argument("--pipeline", default="none", choices=sorted(PIPELINES),
                    help="preprocessing pipeline to use (default: none)")
    ap.add_argument("--compare-preprocess", action="store_true",
                    help="run every pipeline and rank them by normalized match")
    ap.add_argument("--out", default="reports", help="output directory")
    ap.add_argument("--name", default=None, help="report name")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)
    if not os.path.exists(args.labels):
        log.error("labels file not found: %s", args.labels)
        return 2

    labels = read_labels(args.labels)
    if not labels:
        log.error("no usable rows in %s", args.labels)
        return 2
    log.info("loaded %d labelled plates", len(labels))

    import easyocr

    reader = easyocr.Reader(["en"])

    pipelines = sorted(PIPELINES) if args.compare_preprocess else [args.pipeline]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels": os.path.relpath(args.labels),
        "count": len(labels),
        "results": {},
    }
    for pipeline in pipelines:
        obs = run(labels, reader, pipeline)
        report = evaluate_with_conditions(obs)
        payload["results"][pipeline] = report.as_dict()
        _print_report(f"pipeline={pipeline}", report)

    if args.compare_preprocess:
        ranked = sorted(
            payload["results"].items(),
            key=lambda kv: kv[1]["overall"]["normalized_match"], reverse=True,
        )
        print("\n=== ranking by normalized match ===")
        for name, r in ranked:
            print(f"  {name:16s} {r['overall']['normalized_match']:.3f}")
        payload["best_pipeline"] = ranked[0][0]

    os.makedirs(args.out, exist_ok=True)
    name = args.name or f"ocr_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_path = os.path.join(args.out, f"{name}.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\nOCR evaluation written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
