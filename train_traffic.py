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
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stopping patience (epochs with no val improvement)")
    ap.add_argument("--resume", action="store_true",
                    help="resume the last run with this --name")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the post-training validation summary")
    return ap.parse_args(argv)


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
        resume=args.resume,
    )
    if args.patience is not None:
        train_kwargs["patience"] = args.patience

    results = model.train(**train_kwargs)

    save_dir = getattr(results, "save_dir", None) or os.path.join(
        "runs", "detect", args.name
    )
    best = os.path.join(str(save_dir), "weights", "best.pt")
    print("\n=== Training complete ===")
    print(f"Run directory: {save_dir}")
    print(f"Best weights:  {best}")
    print("Point MODEL_PATH (or --model on the CLIs) at that best.pt.")

    if not args.no_validate:
        log.info("running validation on the best checkpoint")
        val = model.val(data=args.data, imgsz=args.imgsz, device=device, verbose=False)
        print(f"Validation mAP@50 = {float(val.box.map50):.4f}  "
              f"mAP@50-95 = {float(val.box.map):.4f}")
        print("For per-class error analysis run: "
              f"python3 evaluate_model.py --model {best} --data {args.data}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
