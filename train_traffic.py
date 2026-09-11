"""Reproducible training entry point for the 4-class traffic model.

The original script hard-coded every hyperparameter, so a different dataset path
or epoch count meant editing the file. This keeps the same defaults (so existing
usage is unchanged) but exposes them as CLI arguments and prints exactly where
the run's weights and metrics land.

    # unchanged default behaviour:
    python3 train_traffic.py

    # fully specified:
    python3 train_traffic.py \\
        --data master_traffic_violation_dataset/data.yaml \\
        --epochs 50 --imgsz 640 --batch 16 --workers 4 \\
        --device auto --model yolov8n.pt --name traffic_model

Outputs go to ``runs/detect/<name>/`` (Ultralytics' convention):
``weights/best.pt`` (used everywhere as MODEL_PATH), ``weights/last.pt``,
``results.csv`` + curves, and ``confusion_matrix.png``. After training it runs
validation once and prints the headline metrics so a run is self-documenting.
For a deeper error analysis of a finished checkpoint, use ``evaluate_model.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

from modules.config import CLASS_NAMES
from modules.logging_setup import configure_logging, get_logger

log = get_logger("train")

DEFAULT_DATA = "master_traffic_violation_dataset/data.yaml"


def resolve_device(name: str) -> str:
    """Map ``auto`` to cuda/mps/cpu; pass an explicit device straight through."""
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=f"Train the 4-class YOLO traffic model ({', '.join(CLASS_NAMES)}).",
    )
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help=f"dataset data.yaml (default: {DEFAULT_DATA})")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="starting checkpoint (default: yolov8n.pt, auto-downloaded)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--name", default="traffic_model", help="experiment name")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed for reproducible runs (default: 0)")
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stopping patience (epochs with no val improvement)")
    ap.add_argument("--conf", type=float, default=None,
                    help="confidence threshold for the post-training val")
    ap.add_argument("--iou", type=float, default=None,
                    help="IoU threshold for the post-training val / NMS")
    ap.add_argument("--resume", action="store_true",
                    help="resume the last run with this --name")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the post-training validation summary")
    ap.add_argument("--no-manifest", action="store_true",
                    help="skip writing the model provenance manifest")
    ap.add_argument("--model-version", default="0.1.0",
                    help="semantic version recorded in the model manifest")

    # Augmentation knobs. Left as None means "use the Ultralytics default", so
    # omitting all of these reproduces the historical training behaviour exactly;
    # setting any is passed straight through to model.train() for an experiment.
    aug = ap.add_argument_group("augmentation (unset = Ultralytics default)")
    aug.add_argument("--hsv-h", type=float, default=None, help="hue jitter fraction")
    aug.add_argument("--hsv-s", type=float, default=None, help="saturation jitter")
    aug.add_argument("--hsv-v", type=float, default=None, help="value/brightness jitter")
    aug.add_argument("--degrees", type=float, default=None, help="rotation degrees")
    aug.add_argument("--translate", type=float, default=None, help="translation fraction")
    aug.add_argument("--scale", type=float, default=None, help="scale gain")
    aug.add_argument("--fliplr", type=float, default=None, help="horizontal flip prob")
    aug.add_argument("--mosaic", type=float, default=None, help="mosaic prob")
    aug.add_argument("--mixup", type=float, default=None, help="mixup prob")
    return ap.parse_args(argv)


# CLI augmentation flag -> Ultralytics train() kwarg. Only forwarded when set.
_AUG_KEYS = (
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
    "fliplr", "mosaic", "mixup",
)


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)

    if not os.path.exists(args.data):
        log.error(
            "dataset config not found: %s (point --data at your data.yaml)",
            args.data,
        )
        return 2

    from ultralytics import YOLO

    device = resolve_device(args.device)
    log.info(
        "training %s on %s for %d epochs (imgsz=%d, batch=%d, device=%s)",
        args.model, args.data, args.epochs, args.imgsz, args.batch, device,
    )

    model = YOLO(args.model)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        name=args.name,
        seed=args.seed,
        deterministic=True,  # with a fixed seed this makes runs reproducible
        resume=args.resume,
    )
    if args.patience is not None:
        train_kwargs["patience"] = args.patience
    # Forward only the augmentation knobs the caller actually set.
    for key in _AUG_KEYS:
        val = getattr(args, key)
        if val is not None:
            train_kwargs[key] = val

    results = model.train(**train_kwargs)

    save_dir = getattr(results, "save_dir", None) or os.path.join(
        "runs", "detect", args.name
    )
    best = os.path.join(str(save_dir), "weights", "best.pt")
    print("\n=== Training complete ===")
    print(f"Run directory: {save_dir}")
    print(f"Best weights:  {best}")
    print("Point MODEL_PATH (or --model on the CLIs) at that best.pt.")

    metrics = None
    if not args.no_validate:
        log.info("running validation on the best checkpoint")
        val_kwargs = dict(data=args.data, imgsz=args.imgsz, device=device, verbose=False)
        if args.conf is not None:
            val_kwargs["conf"] = args.conf
        if args.iou is not None:
            val_kwargs["iou"] = args.iou
        val = model.val(**val_kwargs)
        metrics = {
            "split": "val",
            "map50": round(float(val.box.map50), 4),
            "map50_95": round(float(val.box.map), 4),
            "mean_precision": round(float(val.box.mp), 4),
            "mean_recall": round(float(val.box.mr), 4),
        }
        print(f"Validation mAP@50 = {metrics['map50']:.4f}  "
              f"mAP@50-95 = {metrics['map50_95']:.4f}")
        print("For per-class error analysis run: "
              f"python3 evaluate_model.py --model {best} --data {args.data}")

    if not args.no_manifest and os.path.exists(best):
        # Record provenance (training config, dataset version, git commit,
        # checksum, val metrics) so the checkpoint is auditable and the runtime
        # can stamp its version onto evidence. See modules/model_manifest.py.
        from modules.model_manifest import (
            build_manifest,
            model_version_string,
            write_manifest,
        )

        manifest = build_manifest(
            best, name=args.name, version=args.model_version,
            data_yaml=args.data, metrics=metrics,
        )
        beside, registry = write_manifest(manifest, best)
        print(f"Model manifest: {beside}")
        print(f"           and: {registry}")
        print(f"Model version:  {model_version_string(manifest)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
