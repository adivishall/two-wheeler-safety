"""Reproducible model evaluation and error analysis.

This is the evaluation story the runtime pipeline needs to be trusted: it turns
a trained YOLO checkpoint + a validation/test split into per-class metrics, a
confusion matrix, and a targeted error analysis (where helmet detection fails,
how often WithHelmet/WithoutHelmet are confused, representative false positives
and false negatives, and whether confidence correlates with correctness).

Nothing is fabricated: every number comes from running the given model on the
given data. The repo ships neither weights nor dataset (both are gitignored), so
this is *tooling* the owner runs locally:

    python3 evaluate_model.py \\
        --model runs/detect/traffic_model-2/weights/best.pt \\
        --data master_traffic_violation_dataset/data.yaml \\
        --split val --benchmark

Outputs land in ``reports/`` (a Markdown report + a JSON with the raw numbers +
Ultralytics' own confusion-matrix / PR-curve PNGs). Two layers of metrics:

* **Authoritative** — Ultralytics ``model.val()`` gives mAP@50, mAP@50-95, and
  per-class precision/recall exactly as the training run reports them.
* **Error analysis** — a second pass matches predictions to ground truth
  (``modules.evaluation``) to surface class confusions, FP/FN examples, and the
  confidence/correctness separation that ``val()`` does not expose per image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from modules.evaluation import (
    ConfidenceStats,
    ConfusionMatrix,
    build_class_reports,
    match_image,
)
from modules.logging_setup import configure_logging, get_logger

log = get_logger("evaluate")


def resolve_device(name: str) -> str:
    """Map ``auto`` to the best available backend; pass anything else through."""
    if name != "auto":
        return name
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - torch absent or probing failed
        pass
    return "cpu"


def load_data_yaml(path: str) -> dict:
    """Read a YOLO ``data.yaml`` into ``{names, root, splits}``."""
    import yaml  # pyyaml ships with ultralytics

    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    names = cfg.get("names")
    if isinstance(names, dict):  # {0: 'Plate', ...}
        names = [names[k] for k in sorted(names)]
    root = cfg.get("path", os.path.dirname(os.path.abspath(path)))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(path)), root)
    return {"names": list(names or []), "root": root, "cfg": cfg}


def _split_dir(data: dict, split: str) -> str | None:
    entry = data["cfg"].get(split)
    if not entry:
        return None
    path = entry if os.path.isabs(entry) else os.path.join(data["root"], entry)
    return path


def _iter_image_label_pairs(split_dir: str):
    """Yield ``(image_path, label_path)`` for a YOLO split directory.

    Handles the standard layout where a ``.../images/...`` tree mirrors a
    ``.../labels/...`` tree with matching stems.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_root = split_dir
    if not os.path.isdir(img_root):
        # data.yaml may point at a text file listing images; skip in that case.
        return
    for dirpath, _dirs, files in os.walk(img_root):
        for name in files:
            if os.path.splitext(name)[1].lower() not in exts:
                continue
            image_path = os.path.join(dirpath, name)
            label_path = (
                image_path.replace(os.sep + "images" + os.sep,
                                   os.sep + "labels" + os.sep)
            )
            label_path = os.path.splitext(label_path)[0] + ".txt"
            yield image_path, label_path


def _read_gt(label_path: str, w: int, h: int) -> list:
    """Parse a YOLO label file into ``[(cls, (x1,y1,x2,y2)), ...]`` pixels."""
    gt = []
    if not os.path.exists(label_path):
        return gt
    with open(label_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(p) for p in parts[1:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            gt.append((cls, (x1, y1, x2, y2)))
    return gt


def run_official_val(model, data_path, split, imgsz, conf, iou, device) -> dict:
    """Ultralytics ``val()`` — the authoritative per-class P/R + mAP."""
    log.info("running Ultralytics val() on split=%s", split)
    metrics = model.val(
        data=data_path, split=split, imgsz=imgsz, conf=conf, iou=iou,
        device=device, verbose=False,
    )
    box = metrics.box
    names = metrics.names if hasattr(metrics, "names") else {}
    per_class = {}
    # ap_class_index maps row order in the per-class arrays to class ids.
    for i, cls_id in enumerate(getattr(box, "ap_class_index", [])):
        cname = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(cls_id)
        per_class[cname] = {
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
            "map50": round(float(box.ap50[i]), 4),
            "map50_95": round(float(box.ap[i]), 4),
        }
    return {
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "mean_precision": round(float(box.mp), 4),
        "mean_recall": round(float(box.mr), 4),
        "per_class": per_class,
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }


def run_error_analysis(model, split_dir, class_names, conf, iou, max_images) -> dict:
    """Second pass: match predictions to GT for confusions + FP/FN + confidence."""
    import cv2

    cm = ConfusionMatrix(len(class_names))
    conf_stats = ConfidenceStats()
    fp_examples: list[dict] = []
    fn_examples: list[dict] = []
    n_images = 0

    pairs = list(_iter_image_label_pairs(split_dir))
    if max_images:
        pairs = pairs[:max_images]
    log.info("error-analysis pass over %d images", len(pairs))

    for image_path, label_path in pairs:
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = _read_gt(label_path, w, h)

        result = model.predict(source=image_path, conf=conf, verbose=False)[0]
        preds = []
        for b in result.boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            preds.append((cls, (x1, y1, x2, y2), float(b.conf[0])))

        matches = match_image(gt, preds, iou_threshold=iou)
        cm.add_matches(matches)
        for m in matches:
            if m.pred_class is not None and m.gt_class is not None:
                conf_stats.add(m.pred_class == m.gt_class, m.score)
            elif m.pred_class is not None:  # false positive
                conf_stats.add(False, m.score)
                if len(fp_examples) < 20:
                    fp_examples.append({
                        "image": os.path.relpath(image_path),
                        "predicted": class_names[m.pred_class],
                        "score": round(m.score, 3),
                    })
            elif m.gt_class is not None:  # false negative (missed)
                if len(fn_examples) < 20:
                    fn_examples.append({
                        "image": os.path.relpath(image_path),
                        "missed": class_names[m.gt_class],
                    })
        n_images += 1

    reports = build_class_reports(cm, class_names)
    # Helmet-specific confusion (the documented failure mode).
    helmet_confusion = None
    try:
        wi = class_names.index("WithHelmet")
        wo = class_names.index("WithoutHelmet")
        helmet_confusion = {
            "withhelmet_as_withouthelmet": cm.matrix[wi][wo],
            "withouthelmet_as_withhelmet": cm.matrix[wo][wi],
        }
    except ValueError:
        pass

    return {
        "images": n_images,
        "confusion_matrix": cm.as_list(),
        "class_report": [r.__dict__ for r in reports],
        "helmet_confusion": helmet_confusion,
        "confidence": conf_stats.summary(),
        "false_positive_examples": fp_examples,
        "false_negative_examples": fn_examples,
    }


def benchmark_inference(model, split_dir, n: int) -> dict:
    """Measure per-image inference latency over the first ``n`` split images."""
    pairs = list(_iter_image_label_pairs(split_dir))[:n]
    if not pairs:
        return {}
    # Warm up (first inference includes lazy CUDA/graph init).
    model.predict(source=pairs[0][0], verbose=False)
    times = []
    for image_path, _ in pairs:
        t0 = time.perf_counter()
        model.predict(source=image_path, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mean = sum(times) / len(times)
    return {
        "images": len(times),
        "mean_ms": round(mean, 2),
        "p50_ms": round(times[len(times) // 2], 2),
        "p90_ms": round(times[int(len(times) * 0.9)], 2),
        "fps": round(1000.0 / mean, 1) if mean else 0.0,
    }


def write_reports(out_dir: str, name: str, payload: dict) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    md_path = os.path.join(out_dir, f"{name}.md")
    with open(md_path, "w") as fh:
        fh.write(_render_markdown(payload))
    return md_path, json_path


def _render_markdown(p: dict) -> str:
    lines = [f"# Model evaluation — {p['name']}", ""]
    lines.append(f"- Generated: {p['generated_at']}")
    lines.append(f"- Model: `{p['model']}`")
    lines.append(f"- Data: `{p['data']}` (split: {p['split']})")
    lines.append(f"- conf={p['conf']}, iou={p['iou']}, imgsz={p['imgsz']}, device={p['device']}")
    lines.append("")

    official = p.get("official")
    if official:
        lines += ["## Headline metrics (Ultralytics val)", "",
                  f"- **mAP@50**: {official['map50']}",
                  f"- **mAP@50-95**: {official['map50_95']}",
                  f"- mean precision: {official['mean_precision']}",
                  f"- mean recall: {official['mean_recall']}", "",
                  "| Class | Precision | Recall | mAP@50 | mAP@50-95 |",
                  "|---|---|---|---|---|"]
        for cname, m in official["per_class"].items():
            lines.append(
                f"| {cname} | {m['precision']} | {m['recall']} | {m['map50']} | {m['map50_95']} |"
            )
        if official.get("save_dir"):
            lines += ["", f"Confusion matrix / PR curves: `{official['save_dir']}`"]
        lines.append("")

    ea = p.get("error_analysis")
    if ea:
        lines += ["## Error analysis", "",
                  f"Images analysed: {ea['images']}", "",
                  "| Class | TP | FP | FN | Precision | Recall | F1 |",
                  "|---|---|---|---|---|---|---|"]
        for r in ea["class_report"]:
            lines.append(
                f"| {r['name']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                f"{r['precision']} | {r['recall']} | {r['f1']} |"
            )
        lines.append("")
        if ea.get("helmet_confusion"):
            hc = ea["helmet_confusion"]
            lines += ["### Helmet confusion", "",
                      f"- WithHelmet predicted as WithoutHelmet: "
                      f"{hc['withhelmet_as_withouthelmet']}",
                      f"- WithoutHelmet predicted as WithHelmet: "
                      f"{hc['withouthelmet_as_withhelmet']}", ""]
        c = ea["confidence"]
        lines += ["### Confidence vs correctness", "",
                  f"- mean confidence when correct: {c['mean_conf_correct']} "
                  f"({c['n_correct']} preds)",
                  f"- mean confidence when wrong: {c['mean_conf_wrong']} "
                  f"({c['n_wrong']} preds)",
                  f"- separation (correct - wrong): **{c['separation']}** "
                  "(>0 is good; <=0 means confidently wrong)", ""]
        if ea["false_positive_examples"]:
            lines += ["### Representative false positives", ""]
            for ex in ea["false_positive_examples"][:10]:
                lines.append(f"- `{ex['image']}` predicted **{ex['predicted']}** ({ex['score']})")
            lines.append("")
        if ea["false_negative_examples"]:
            lines += ["### Representative false negatives (missed)", ""]
            for ex in ea["false_negative_examples"][:10]:
                lines.append(f"- `{ex['image']}` missed **{ex['missed']}**")
            lines.append("")

    bench = p.get("benchmark")
    if bench:
        lines += ["## Inference benchmark", "",
                  f"- {bench['images']} images, mean {bench['mean_ms']} ms "
                  f"(p50 {bench['p50_ms']}, p90 {bench['p90_ms']}) — {bench['fps']} FPS", ""]

    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Evaluate a trained YOLO model and produce an error-analysis report.",
    )
    ap.add_argument("--model", required=True, help="path to YOLO weights (.pt)")
    ap.add_argument("--data", required=True, help="path to the dataset data.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"],
                    help="dataset split to evaluate (default: val)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold for the error-analysis pass")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold for matching / NMS")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--out", default="reports", help="output directory for reports")
    ap.add_argument("--name", default=None, help="report name (default: eval_<timestamp>)")
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap images in the error-analysis pass (0 = all)")
    ap.add_argument("--benchmark", action="store_true",
                    help="also measure inference latency")
    ap.add_argument("--benchmark-images", type=int, default=50)
    ap.add_argument("--no-official", action="store_true",
                    help="skip Ultralytics val() (error analysis only)")
    ap.add_argument("--no-error-analysis", action="store_true",
                    help="skip the per-image matching pass (headline metrics only)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)

    if not os.path.exists(args.model):
        log.error("model weights not found: %s", args.model)
        return 2
    if not os.path.exists(args.data):
        log.error("data.yaml not found: %s", args.data)
        return 2

    device = resolve_device(args.device)
    data = load_data_yaml(args.data)
    class_names = data["names"]
    if not class_names:
        log.error("no class names found in %s", args.data)
        return 2

    from ultralytics import YOLO

    log.info("loading model %s on %s", args.model, device)
    model = YOLO(args.model)

    payload = {
        "name": args.name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": device,
        "class_names": class_names,
    }

    if not args.no_official:
        try:
            payload["official"] = run_official_val(
                model, args.data, args.split, args.imgsz, args.conf, args.iou, device
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("official val() failed: %s", exc)

    split_dir = _split_dir(data, args.split)
    need_predict = (not args.no_error_analysis) or args.benchmark
    if need_predict and payload.get("official") is not None:
        # torch>=2.6 (seen on 2.12 / MPS) raises "Inference tensors do not track
        # version counter" if .predict() runs on a YOLO object that already ran
        # .val(). val() leaves the fused weights as inference tensors, which the
        # predict forward pass then trips over. Reloading gives the predict
        # passes clean weights; val()'s authoritative numbers are already saved.
        log.info("reloading model for the predict-based passes (post-val())")
        model = YOLO(args.model)

    if not args.no_error_analysis:
        if split_dir and os.path.isdir(split_dir):
            payload["error_analysis"] = run_error_analysis(
                model, split_dir, class_names, args.conf, args.iou, args.max_images
            )
        else:
            log.warning("split dir not found (%s); skipping error analysis", split_dir)

    if args.benchmark and split_dir and os.path.isdir(split_dir):
        payload["benchmark"] = benchmark_inference(model, split_dir, args.benchmark_images)

    md_path, json_path = write_reports(args.out, payload["name"], payload)
    log.info("wrote %s and %s", md_path, json_path)
    print(f"\nEvaluation report: {md_path}")
    print(f"Raw metrics:       {json_path}")
    if payload.get("official"):
        o = payload["official"]
        print(f"mAP@50 = {o['map50']}   mAP@50-95 = {o['map50_95']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
