"""Shared YOLO-dataset I/O and prediction helpers for the evaluation CLIs.

``evaluate_model.py``, ``compare_models.py``, ``benchmark.py``,
``train_traffic.py`` and the uncertainty/robustness tools all need the same
few things: pick a device, read a ``data.yaml``, walk a split's image/label
pairs, parse YOLO labels to pixel boxes, and run the model over a split. They
each carried a copy; this is the one copy.

Split iteration is **sorted**, so every tool sees images in the same order on
every machine (``os.walk`` order is filesystem-dependent) and a cached
prediction file lines up image-for-image with another model's.
"""

from __future__ import annotations

import hashlib
import json
import os

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
    """Read a YOLO ``data.yaml`` into ``{names, root, cfg}``."""
    import yaml

    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    names = cfg.get("names")
    if isinstance(names, dict):  # {0: 'Plate', ...}
        names = [names[k] for k in sorted(names)]
    root = cfg.get("path", os.path.dirname(os.path.abspath(path)))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(path)), root)
    return {"names": list(names or []), "root": root, "cfg": cfg}


def split_dir(data: dict, split: str) -> str | None:
    entry = data["cfg"].get(split)
    if not entry:
        return None
    return entry if os.path.isabs(entry) else os.path.join(data["root"], entry)


def iter_image_label_pairs(img_root: str):
    """Yield ``(image_path, label_path)`` for a split, in sorted order.

    Handles the standard layout where a ``.../images/...`` tree mirrors a
    ``.../labels/...`` tree with matching stems."""
    if not img_root or not os.path.isdir(img_root):
        return
    found = []
    for dirpath, _dirs, files in os.walk(img_root):
        for name in files:
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            image_path = os.path.join(dirpath, name)
            label_path = image_path.replace(
                os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
            found.append((image_path, os.path.splitext(label_path)[0] + ".txt"))
    yield from sorted(found)


def read_gt(label_path: str, w: int, h: int) -> list:
    """Parse a YOLO label file into ``[(cls, (x1,y1,x2,y2)), ...]`` pixels."""
    gt: list = []
    if not os.path.exists(label_path):
        return gt
    with open(label_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(p) for p in parts[1:5])
            gt.append((cls, ((cx - bw / 2) * w, (cy - bh / 2) * h,
                             (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return gt


# Ultralytics val() defaults: keep every box down to 0.001 so AP integrates the
# whole precision/recall curve, and NMS at IoU 0.7.
AP_CONF = 0.001
AP_NMS_IOU = 0.7


def predict_split(model, pairs, n_classes: int, *, conf: float = AP_CONF,
                  nms_iou: float = AP_NMS_IOU, imgsz: int = 640, corrupt=None,
                  device: str | None = None):
    """Run ``model`` over ``pairs`` and return AP-ready per-image records.

    ``corrupt(image, rng_key) -> image`` optionally transforms each image first
    (robustness testing); labels are left untouched, so a corruption can only
    make the task harder, never move the answer."""
    import cv2

    from modules.detection_stats import match_for_ap

    records = []
    for image_path, label_path in pairs:
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_gt(label_path, w, h)
        if corrupt is not None:
            img = corrupt(img, os.path.basename(image_path))
        kwargs: dict = dict(conf=conf, iou=nms_iou, imgsz=imgsz, verbose=False)
        if device:
            kwargs["device"] = device
        result = model.predict(source=img, **kwargs)[0]
        preds = [(int(b.cls[0]), tuple(float(v) for v in b.xyxy[0]), float(b.conf[0]))
                 for b in result.boxes]
        records.append(match_for_ap(gt, preds, n_classes))
    return records


def cache_key(**parts) -> str:
    """Stable short key for a prediction cache entry."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
