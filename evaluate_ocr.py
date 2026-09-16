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

Two further modes answer the question the runtime actually turns on — *is
temporal stabilization better than reading one frame?*:

    # no OCR engine, no data: simulated character noise, exactly reproducible
    python3 evaluate_ocr.py --simulate

    # the real-data version: CSV with a sequence_id column grouping frames of
    # one tracked plate, run through actual EasyOCR
    python3 evaluate_ocr.py --sequences ocr_sequences.csv

The simulation is a measurement of the *decision rule* under a stated noise
model, not of EasyOCR; the sequence mode is the one that would settle field
accuracy and needs a labelled sequence set, which is not shipped.

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
from modules.ocr_temporal_eval import (
    FrameRead,
    compare_policies,
    noise_sweep,
    run_simulation,
)

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
                # Frames of one tracked vehicle share a sequence_id; absent, each
                # row is its own single-frame sequence.
                "sequence_id": (row.get("sequence_id") or "").strip(),
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


def run_sequences(labels: list[dict], reader, pipeline: str) -> list:
    """OCR every frame of every sequence into ``(truth, [FrameRead, ...])`` pairs.

    Rows are grouped by ``sequence_id``; every row in a group must carry the same
    ground-truth plate (they are frames of one vehicle), and the first non-empty
    one is used. Ungrouped rows become single-frame sequences, which is a fair
    representation of "the pipeline only ever saw this plate once".
    """
    import cv2

    groups: dict[str, list] = {}
    order: list[str] = []
    for i, row in enumerate(labels):
        key = row["sequence_id"] or f"__row{i}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    pairs = []
    for key in order:
        rows = groups[key]
        truth = next((r["plate"] for r in rows if r["plate"]), "")
        reads = []
        for row in rows:
            img = cv2.imread(row["image_path"])
            if img is None:
                log.warning("could not read %s; counting as a missed frame",
                            row["image_path"])
                reads.append(FrameRead(None, 0.0))
                continue
            pred, conf = ocr_one(reader, img, pipeline)
            reads.append(FrameRead(pred or None, conf))
        pairs.append((truth, reads))
    return pairs


def _print_policies(title: str, report) -> None:
    d = report.as_dict()
    print(f"\n=== {title} (n={d['n_sequences']} sequences) ===")
    print(f"  {'policy':10s} {'coverage':>9} {'acc|answered':>13} "
          f"{'norm match':>11} {'char acc':>9} {'invalid':>8}")
    for pol, m in d["policies"].items():
        print(f"  {pol:10s} {m['coverage']:>9.3f} {m['answered_accuracy']:>13.3f} "
              f"{m['normalized_match']:>11.3f} {m['char_accuracy']:>9.3f} "
              f"{m['invalid_rate']:>8.3f}")
    print(f"  temporal gain vs best single-frame (normalized match): "
          f"{d['temporal_gain_vs_best_single_frame']:+.3f}")
    print("  NOTE: coverage is how often the policy names a plate at all; "
          "accuracy|answered\n        is how often it is right when it does. "
          "A wrong plate fines an\n        innocent rider; an abstention only "
          "misses a fine.")


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
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels", help="CSV: image_path,plate[,condition]")
    src.add_argument("--sequences",
                     help="CSV: sequence_id,image_path,plate — frames of one "
                          "tracked plate share a sequence_id; compares "
                          "single-frame vs temporal OCR on real images")
    src.add_argument("--simulate", action="store_true",
                     help="compare single-frame vs temporal OCR under simulated "
                          "character noise (no OCR engine or data needed)")
    ap.add_argument("--sweep", action="store_true",
                    help="with --simulate, sweep the character substitution rate")
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

    if args.simulate:
        report = run_simulation()
        _print_policies("simulated OCR noise (default model)", report)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "simulate",
            "default_noise": report.as_dict(),
        }
        if args.sweep:
            payload["noise_sweep"] = noise_sweep()
            print("\n=== substitution-rate sweep ===")
            print(f"  {'rate':>6} {'policy':10s} {'coverage':>9} {'acc|answered':>13}")
            for rate, rep in payload["noise_sweep"]["sweep"].items():
                for pol, m in rep["policies"].items():
                    print(f"  {rate:>6} {pol:10s} {m['coverage']:>9.3f} "
                          f"{m['answered_accuracy']:>13.3f}")
        return _write(args, payload, "ocr_policy_simulation")

    if args.sequences:
        if not os.path.exists(args.sequences):
            log.error("sequences file not found: %s", args.sequences)
            return 2
        labels = read_labels(args.sequences)
        if not labels:
            log.error("no usable rows in %s", args.sequences)
            return 2
        import easyocr

        reader = easyocr.Reader(["en"])
        pairs = run_sequences(labels, reader, args.pipeline)
        report = compare_policies(pairs, noise={"source": "real images"})
        _print_policies(f"real sequences ({os.path.basename(args.sequences)})", report)
        return _write(args, {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "sequences",
            "labels": os.path.relpath(args.sequences),
            "pipeline": args.pipeline,
            "result": report.as_dict(),
        }, "ocr_policy_sequences")

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

    return _write(args, payload, "ocr_eval")


def _write(args, payload: dict, default_stem: str) -> int:
    """Write ``payload`` as JSON under ``--out`` and report where it landed."""
    os.makedirs(args.out, exist_ok=True)
    name = args.name or f"{default_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_path = os.path.join(args.out, f"{name}.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\nOCR evaluation written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
