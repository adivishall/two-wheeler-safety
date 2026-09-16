"""A/B compare trained checkpoints on one held-out split — pick a model by evidence.

Four checkpoints accumulated in ``runs/detect/`` over this project's life
(``traffic_model-2``, ``traffic_model_r2``, ``traffic_model_helmetfix``,
``traffic_model_probe``) and the shipped one was chosen by memory, not by
measurement. This script settles it: every checkpoint is evaluated on the
**same** split with the **same** inference settings, and the winner is whichever
one the table says, on the criterion you name.

Reported per model: mAP@50, mAP@50-95, mean precision/recall, per-class mAP@50,
inference latency (mean ms over a fixed image count, after a warm-up), weights
size on disk, and the manifest version if one resolves.

    # compare everything under runs/detect on the de-leaked held-out test split
    python3 compare_models.py --data eval/clean_splits/data.yaml --split test

    # or name the checkpoints explicitly, and rank by helmet performance
    python3 compare_models.py --models a/best.pt b/best.pt \\
        --data ... --split test --rank-by WithoutHelmet

Two honesty rules are enforced rather than trusted:

* Every model is evaluated on the same ``--data``/``--split``/``--imgsz``/
  ``--conf``/``--iou``; the settings are recorded in the report.
* A model whose evaluation *fails* is reported as failed, not silently dropped,
  so a comparison can never quietly become a table of the ones that happened to
  work.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

from modules.logging_setup import configure_logging, get_logger

log = get_logger("compare_models")


def discover_models(root: str = "runs/detect") -> list[str]:
    """Every ``*/weights/best.pt`` under ``root``, sorted for determinism."""
    return sorted(glob.glob(os.path.join(root, "*", "weights", "best.pt")))


def model_label(path: str) -> str:
    """``runs/detect/<name>/weights/best.pt`` -> ``<name>``."""
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3 and parts[-2] == "weights":
        return parts[-3]
    return os.path.splitext(os.path.basename(path))[0]


def measure_latency(model, images: list[str], warmup: int = 2) -> dict:
    """Mean/p90 single-image inference latency in ms over ``images``.

    The warm-up matters: the first forward pass pays lazy device/graph
    initialisation and would otherwise dominate a small sample and make whichever
    model ran first look slowest.
    """
    if not images:
        return {}
    for path in images[:warmup]:
        model.predict(source=path, verbose=False)
    times = []
    for path in images:
        t0 = time.perf_counter()
        model.predict(source=path, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mean = sum(times) / len(times)
    return {
        "images": len(times),
        "mean_ms": round(mean, 2),
        "p90_ms": round(times[int(len(times) * 0.9)], 2),
        "fps": round(1000.0 / mean, 1) if mean else 0.0,
    }


def evaluate_one(
    weights: str, data: str, split: str, imgsz: int, conf: float, iou: float,
    device: str, latency_images: list[str],
) -> dict:
    """Run Ultralytics ``val()`` on one checkpoint plus a latency measurement."""
    from ultralytics import YOLO

    row: dict = {
        "label": model_label(weights),
        "weights": weights,
        "size_mb": round(os.path.getsize(weights) / (1024 * 1024), 2),
    }
    try:
        from modules.model_manifest import model_version_string, resolve_manifest

        row["model_version"] = model_version_string(resolve_manifest(weights))
    except Exception:  # noqa: BLE001 - provenance is nice-to-have, not required
        row["model_version"] = None

    model = YOLO(weights)
    metrics = model.val(
        data=data, split=split, imgsz=imgsz, conf=conf, iou=iou,
        device=device, verbose=False,
    )
    box = metrics.box
    names = metrics.names if isinstance(getattr(metrics, "names", None), dict) else {}
    per_class = {}
    for i, cls_id in enumerate(getattr(box, "ap_class_index", [])):
        cname = names.get(int(cls_id), str(cls_id))
        per_class[cname] = {
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
            "map50": round(float(box.ap50[i]), 4),
        }
    row.update({
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "mean_precision": round(float(box.mp), 4),
        "mean_recall": round(float(box.mr), 4),
        "per_class": per_class,
    })

    if latency_images:
        # val() leaves fused inference tensors behind; a fresh load gives the
        # predict pass clean weights (same reason evaluate_model.py reloads).
        row["latency"] = measure_latency(YOLO(weights), latency_images)
    return row


def _split_images(data: str, split: str, n: int) -> list[str]:
    """First ``n`` images of the split, for the latency measurement."""
    import yaml

    with open(data) as fh:
        cfg = yaml.safe_load(fh)
    root = cfg.get("path") or os.path.dirname(os.path.abspath(data))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(data)), root)
    entry = cfg.get(split)
    if not entry:
        return []
    d = entry if os.path.isabs(entry) else os.path.join(root, entry)
    if not os.path.isdir(d):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    for name in sorted(os.listdir(d)):
        if os.path.splitext(name)[1].lower() in exts:
            out.append(os.path.join(d, name))
        if len(out) >= n:
            break
    return out


def rank(rows: list[dict], key: str) -> list[dict]:
    """Sort successful rows best-first by ``key`` (a top-level metric or a class name)."""
    def score(r: dict) -> float:
        if r.get("error"):
            return -1.0
        if key in r:
            return float(r[key])
        cls = r.get("per_class", {}).get(key)
        return float(cls["map50"]) if cls else -1.0

    return sorted(rows, key=score, reverse=True)


def render_markdown(payload: dict) -> str:
    rows = payload["models"]
    ok = [r for r in rows if not r.get("error")]
    lines = ["# Model A/B comparison", "",
             f"- Generated: {payload['generated_at']}",
             f"- Data: `{payload['data']}` (split: **{payload['split']}**)",
             f"- conf={payload['conf']}, iou={payload['iou']}, "
             f"imgsz={payload['imgsz']}, device={payload['device']}",
             f"- Ranked by: **{payload['rank_by']}**", ""]
    if payload.get("winner"):
        lines += [f"**Winner: `{payload['winner']}`** "
                  f"(every model saw the identical split and settings).", ""]

    lines += ["| Model | version | mAP@50 | mAP@50-95 | precision | recall | "
              "latency (ms) | size (MB) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in ok:
        lat = r.get("latency", {}).get("mean_ms", "—")
        lines.append(
            f"| `{r['label']}` | {r.get('model_version') or '—'} | {r['map50']} | "
            f"{r['map50_95']} | {r['mean_precision']} | {r['mean_recall']} | "
            f"{lat} | {r['size_mb']} |"
        )
    lines.append("")

    classes = sorted({c for r in ok for c in r.get("per_class", {})})
    if classes:
        lines += ["## Per-class mAP@50", "",
                  "| Model | " + " | ".join(classes) + " |",
                  "|---|" + "---:|" * len(classes)]
        for r in ok:
            cells = [
                str(r["per_class"].get(c, {}).get("map50", "—")) for c in classes
            ]
            lines.append(f"| `{r['label']}` | " + " | ".join(cells) + " |")
        lines.append("")

    failed = [r for r in rows if r.get("error")]
    if failed:
        lines += ["## Failed to evaluate", ""]
        for r in failed:
            lines.append(f"- `{r['label']}` — {r['error']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Compare trained checkpoints on one held-out split.",
    )
    ap.add_argument("--models", nargs="*", default=None,
                    help="weights to compare (default: every runs/detect/*/weights/best.pt)")
    ap.add_argument("--runs-root", default="runs/detect",
                    help="where to auto-discover checkpoints (default: runs/detect)")
    ap.add_argument("--data", required=True, help="dataset data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--rank-by", default="map50",
                    help="metric to rank by: map50 | map50_95 | mean_precision | "
                         "mean_recall | a class name (ranks that class's mAP@50)")
    ap.add_argument("--latency-images", type=int, default=25,
                    help="images used for the latency measurement (0 = skip)")
    ap.add_argument("--out", default="eval/results", help="output directory")
    ap.add_argument("--name", default="model_comparison", help="report name")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)

    if not os.path.exists(args.data):
        log.error("data.yaml not found: %s", args.data)
        return 2
    models = args.models or discover_models(args.runs_root)
    models = [m for m in models if os.path.exists(m)]
    if not models:
        log.error("no checkpoints found (looked in %s)", args.runs_root)
        return 2

    from evaluate_model import resolve_device

    device = resolve_device(args.device)
    latency_images = (
        _split_images(args.data, args.split, args.latency_images)
        if args.latency_images else []
    )
    log.info("comparing %d checkpoint(s) on %s split=%s", len(models), args.data,
             args.split)

    rows = []
    for weights in models:
        label = model_label(weights)
        log.info("evaluating %s", label)
        try:
            rows.append(evaluate_one(
                weights, args.data, args.split, args.imgsz, args.conf, args.iou,
                device, latency_images,
            ))
        except Exception as exc:  # noqa: BLE001 - a broken checkpoint is a result
            log.exception("%s failed: %s", label, exc)
            rows.append({"label": label, "weights": weights, "error": str(exc)})

    ranked = rank(rows, args.rank_by)
    winner = next((r["label"] for r in ranked if not r.get("error")), None)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": args.data, "split": args.split, "imgsz": args.imgsz,
        "conf": args.conf, "iou": args.iou, "device": device,
        "rank_by": args.rank_by, "winner": winner, "models": ranked,
    }

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, f"{args.name}.json")
    md_path = os.path.join(args.out, f"{args.name}.md")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    with open(md_path, "w") as fh:
        fh.write(render_markdown(payload))

    print(f"\n{'model':28s} {'mAP@50':>8} {'mAP@50-95':>10} {'P':>7} {'R':>7} "
          f"{'ms':>7} {'MB':>6}")
    for r in ranked:
        if r.get("error"):
            print(f"{r['label']:28s}   FAILED: {r['error'][:60]}")
            continue
        lat = r.get("latency", {}).get("mean_ms", 0.0)
        print(f"{r['label']:28s} {r['map50']:>8.4f} {r['map50_95']:>10.4f} "
              f"{r['mean_precision']:>7.3f} {r['mean_recall']:>7.3f} "
              f"{lat:>7.1f} {r['size_mb']:>6.1f}")
    print(f"\nWinner by {args.rank_by}: {winner}")
    print(f"Wrote {md_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
