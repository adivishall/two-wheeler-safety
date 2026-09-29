"""Evaluate the shipped system on labelled field footage.

    python3 evaluate_field.py --dataset data/field --split held_out \\
        --model runs/detect/traffic_model-2/weights/best.pt --device mps

Measures, per condition (lighting, weather, view angle, occlusion, plate
visibility, blur, plate size, resolution, vehicles in frame, crossing, camera):
rider/plate detection recall, helmet/triple class accuracy, track
fragmentation, plate association accuracy, OCR, fine precision/recall, wrong
plates, missed and false fines — and the stage each error is charged to
(``modules/field_eval.py``). Plus the OCR policy comparison (last frame vs most
confident vs the temporal vote) on the labelled plate crops.

Evaluation splits only (validation / held_out / external); ``development`` is
refused unless ``--allow-development`` (numbers on data you tune on are not
evaluation). With no field dataset at ``--dataset`` this writes an explicit
NOT MEASURED report and exits 0 — the absence of evidence is the result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from modules.logging_setup import configure_logging, get_logger

log = get_logger("evaluate_field")

NOT_MEASURED = """# Field evaluation — NOT MEASURED

No labelled field dataset was found at `{path}`.

Everything the project reports past the detector (tracking, association, OCR,
fines, speed, calibration) is measured on synthetic inputs, and the detector
itself only on still images (`docs/FIELD_EVALUATION.md`). This command measures
all of it on real footage once footage exists:

1. Record sequences from fixed cameras; label them in the schema of
   `modules/field_data.py` (`dataset.json`, `vehicles.jsonl`, `frames.jsonl`),
   exhaustively on labelled frames, with annotator and reviewer per label.
2. `python3 field_dataset.py validate {path} --check-files`
3. `python3 field_dataset.py assign-splits {path} --external-camera <a camera never
   developed on>`
4. `python3 evaluate_field.py --dataset {path} --split held_out --model <weights>`

Until then there is no field accuracy to report, and none is claimed.
"""


def _render(report: dict) -> str:
    ocr, mx = report["ocr"], report["matrix"]
    ov = mx["overall"]
    lines = [f"# Field evaluation — `{report['dataset']}` v{report['dataset_version']}, "
             f"split **{report['split']}**", "",
             f"- Sequences {report['sequences']}, labelled vehicles {report['vehicles']}, "
             f"frames run {report['frames_run']}; lock `{report['lock_hash']}`.",
             f"- Model `{report['model']}`; pipeline {report['pipeline_version']}.",
             "- Scoring: IoU ≥ 0.5; a violation is expected to be fined only if its plate is "
             "visible; a fine on no labelled vehicle is a phantom.", "",
             "## End to end", "",
             f"Fine precision **{ov['fine_precision']}**, recall **{ov['fine_recall']}** — "
             f"{ov['correct']} correct, {ov['wrong_plate']} wrong plate, {ov['missed']} missed, "
             f"{ov['false_fines']} false, {ov['phantom_fines']} phantom; "
             f"{ov['correctly_withheld']} correctly withheld (plate not visible).",
             f"Largest error owner: **{ov['bottleneck']}** ({ov['bottleneck_errors']} errors).",
             "", "| condition | vehicles | fine P | fine R | rider recall | class acc | "
             "plate recall | assoc acc | bottleneck (errors) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for attr, values in mx["by_condition"].items():
        for value, r in values.items():
            lines.append(f"| {attr}: {value} | {r['vehicles']} | {r['fine_precision']} | "
                         f"{r['fine_recall']} | {r['rider_recall']} | {r['class_accuracy']} | "
                         f"{r['plate_recall']} | {r['association_accuracy']} | "
                         f"{r['bottleneck'] or '—'} ({r['bottleneck_errors']}) |")
    lines += ["", "## OCR on labelled plate crops", "",
              f"{ocr['vehicles_scored']} vehicles, {ocr['reads']} reads, latency "
              f"{ocr['latency_ms']} ms. {ocr['scoring']}.", "",
              "| policy | coverage | normalized match | wrong plate | exact | char acc | "
              "edit dist | invalid | abstain |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p, m in ocr["overall"].items():
        if m.get("vehicles"):
            lines.append(f"| {p} | {m['coverage']} | {m['normalized_match']} | "
                         f"{m['wrong_plate_rate']} | {m['exact_match']} | {m['char_accuracy']} | "
                         f"{m['mean_edit_distance']} | {m['invalid_rate']} | "
                         f"{m['abstention_rate']} |")
    for attr, values in ocr["by_condition"].items():
        lines += ["", f"### OCR by {attr}", "",
                  "| value | vehicles | last | best_conf | temporal (coverage) |",
                  "|---|---:|---:|---:|---|"]
        for value, block in values.items():
            t = block["temporal"]
            lines.append(f"| {value} | {t['vehicles']} | {block['last']['normalized_match']} | "
                         f"{block['best_conf']['normalized_match']} | "
                         f"{t['normalized_match']} ({t['coverage']}) |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    configure_logging()
    ap = argparse.ArgumentParser(description="Evaluate the system on labelled field data.")
    ap.add_argument("--dataset", default="data/field")
    ap.add_argument("--split", default="held_out",
                    choices=["validation", "held_out", "external", "development"])
    ap.add_argument("--allow-development", action="store_true")
    ap.add_argument("--model", default="runs/detect/traffic_model-2/weights/best.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--name", default="field_evaluation")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    md_path = os.path.join(args.out, f"{args.name}.md")

    from modules.provenance import portable_path, run_provenance

    if not os.path.exists(os.path.join(args.dataset, "dataset.json")):
        with open(md_path, "w") as fh:
            fh.write(NOT_MEASURED.format(path=portable_path(args.dataset)))
        with open(os.path.join(args.out, f"{args.name}.json"), "w") as fh:
            json.dump({"measured": False, "reason": "no labelled field dataset",
                       "dataset": portable_path(args.dataset),
                       "provenance": run_provenance()}, fh, indent=2, sort_keys=True)
        print(f"NOT MEASURED: no field dataset at {args.dataset}; wrote {md_path}")
        return 0
    if args.split == "development" and not args.allow_development:
        log.error("development is the split you tune on; pass --allow-development to "
                  "evaluate it anyway (the report will say so)")
        return 2

    from modules.field_data import (
        eval_fingerprint,
        frames_in,
        load_dataset,
        read_lock,
        split_of,
        vehicles_in,
    )
    from modules.field_eval import (
        collect_plate_reads,
        condition_matrix,
        iter_frames,
        ocr_report,
        run_sequence,
        score_sequence,
    )

    ds = load_dataset(args.dataset, check_files=True)
    lock = read_lock(args.dataset)
    if lock is None:
        log.error("no split lock: run `field_dataset.py assign-splits %s` first", args.dataset)
        return 2
    vehicles = vehicles_in(ds, lock, args.split)
    seq_ids = sorted({v.sequence_id for v in vehicles})
    if not vehicles:
        log.error("no labelled vehicles in split %s", args.split)
        return 2

    import cv2

    from modules.association import DetBox
    from modules.config import PIPELINE_VERSION, load_config
    from modules.detector import load_models
    from modules.pipeline import pipeline_config_from_detection
    from modules.video_detector import _read_plate_text
    from modules.yolo_io import resolve_device

    det_cfg = load_config().detection
    device = resolve_device(args.device)
    model, reader = load_models(args.model)

    def detect(image):
        res = model(image, conf=det_cfg.conf_threshold, verbose=False, device=device)[0]
        return [DetBox(model.names[int(b.cls[0])], tuple(int(v) for v in b.xyxy[0]),
                       float(b.conf[0])) for b in res.boxes]

    def read(image, box):
        x1, y1, x2, y2 = box
        return _read_plate_text(reader, image[max(0, y1):y2, max(0, x1):x2])

    reads = collect_plate_reads(ds, vehicles, read, lambda f: cv2.imread(ds.frame_file(f)))
    ocr = ocr_report(ds, vehicles, reads)
    config = pipeline_config_from_detection(det_cfg)
    scores, frames_run = [], 0
    for sid in seq_ids:
        run = run_sequence(ds, sid, iter_frames(ds, sid), detect, read, config=config)
        frames_run += run.frames_run
        scores.append(score_sequence(ds, run))
    matrix = condition_matrix(ds, scores)
    report = {
        "measured": True, "dataset": ds.name, "dataset_version": ds.version,
        "split": args.split, "sequences": len(seq_ids), "vehicles": len(vehicles),
        "labelled_frames": len(frames_in(ds, lock, args.split)), "frames_run": frames_run,
        "lock_hash": lock["lock_hash"], "eval_fingerprint": eval_fingerprint(ds, lock),
        "model": portable_path(args.model), "pipeline_version": PIPELINE_VERSION,
        "splits_seen": sorted({split_of(ds, lock, s) or "" for s in seq_ids}),
        "ocr": ocr, "matrix": matrix, "scores": scores,
        "provenance": run_provenance(model_paths=[args.model], config={"split": args.split}),
    }
    with open(os.path.join(args.out, f"{args.name}.json"), "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=list)
    with open(md_path, "w") as fh:
        fh.write(_render(report))
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
