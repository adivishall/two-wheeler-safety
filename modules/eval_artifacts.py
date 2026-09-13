"""Turn evaluation failures into things you can actually look at.

A table saying `WithoutHelmet` has 37 false positives tells you the model is
wrong; it does not tell you *why*. This module writes each representative
failure to disk as an annotated crop plus a JSON sidecar, so the next question —
"is it mislabelled data, a hard scene, or a real model weakness?" — can be
answered by opening a folder.

Layout under the output root (Phase 4 of the evaluation spec):

    eval/
        false_positives/   predicted a box where the ground truth has none
        false_negatives/   ground-truth box the model missed entirely
        class_confusions/  box found, wrong class (the helmet failure mode)
        low_confidence/    correct, but only just above the threshold
        index.json         every saved case, machine-readable

Each image is the **context crop** around the box (padded, so the surroundings
that explain the error are visible) with the box drawn on: red for a prediction,
green for ground truth. The sidecar records image, class(es), score, box, IoU,
frame index when the source was a video, and the model version — everything
needed to cite a case in a report or feed it back into labelling.

Artifacts are written to a gitignored directory by design: they are derived from
(potentially unlicensed) dataset images and can be regenerated at any time, so
they must never be committed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

CATEGORIES = (
    "false_positives",
    "false_negatives",
    "class_confusions",
    "low_confidence",
)

# Box colours (BGR): predictions red, ground truth green.
PRED_COLOR = (0, 0, 255)
GT_COLOR = (0, 200, 0)


@dataclass
class FailureCase:
    """One saved evaluation failure."""

    category: str
    source_image: str
    predicted_class: str | None = None
    expected_class: str | None = None
    score: float | None = None
    box: tuple | None = None  # the prediction box, pixels (x1,y1,x2,y2)
    gt_box: tuple | None = None
    iou: float | None = None
    frame_index: int | None = None  # set when the source was a video
    model_version: str | None = None
    artifact: str | None = None  # relative path of the written crop

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("box", "gt_box"):
            if d[k] is not None:
                d[k] = [round(float(v), 1) for v in d[k]]
        if d["score"] is not None:
            d["score"] = round(float(d["score"]), 4)
        if d["iou"] is not None:
            d["iou"] = round(float(d["iou"]), 4)
        return d


def _crop_with_boxes(image, boxes, *, pad: float = 0.6, min_size: int = 96):
    """Crop the region around ``boxes`` (list of ``(box, colour)``) with context.

    ``pad`` is a fraction of the box size added on each side, and ``min_size``
    guarantees a small plate crop is still large enough to judge by eye.
    """
    import cv2

    h, w = image.shape[:2]
    xs = [c for box, _ in boxes for c in (box[0], box[2])]
    ys = [c for box, _ in boxes for c in (box[1], box[3])]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    px, py = bw * pad, bh * pad
    # Grow a tiny box up to min_size so the crop stays legible.
    px = max(px, (min_size - bw) / 2)
    py = max(py, (min_size - bh) / 2)
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = image[cy1:cy2, cx1:cx2].copy()
    for box, colour in boxes:
        cv2.rectangle(
            crop,
            (int(box[0] - cx1), int(box[1] - cy1)),
            (int(box[2] - cx1), int(box[3] - cy1)),
            colour, 2,
        )
    return crop


@dataclass
class ArtifactWriter:
    """Collects failure cases and writes crops + an index under ``root``.

    ``per_category`` caps how many cases each bucket keeps, so a bad model does
    not fill the disk; the cap is recorded in the index so a truncated bucket is
    never mistaken for a complete one.
    """

    root: str = "eval"
    per_category: int = 30
    model_version: str | None = None
    cases: list = field(default_factory=list)
    _counts: dict = field(default_factory=dict)

    def room_for(self, category: str) -> bool:
        return self._counts.get(category, 0) < self.per_category

    def add(
        self,
        category: str,
        image,
        case: FailureCase,
    ) -> FailureCase | None:
        """Write one case's crop and record it. Returns None when the bucket is
        full or the crop could not be produced."""
        import cv2

        if category not in CATEGORIES:
            raise ValueError(f"unknown category: {category}")
        if not self.room_for(category):
            return None

        boxes = []
        if case.box is not None:
            boxes.append((case.box, PRED_COLOR))
        if case.gt_box is not None:
            boxes.append((case.gt_box, GT_COLOR))
        if not boxes or image is None:
            return None
        crop = _crop_with_boxes(image, boxes)
        if crop is None or crop.size == 0:
            return None

        idx = self._counts.get(category, 0)
        out_dir = os.path.join(self.root, category)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(case.source_image))[0][:60]
        name = f"{idx:03d}_{stem}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), crop)

        case.category = category
        case.artifact = os.path.join(category, name)
        case.model_version = case.model_version or self.model_version
        self._counts[category] = idx + 1
        self.cases.append(case)
        return case

    def summary(self) -> dict:
        return {
            "root": self.root,
            "per_category_cap": self.per_category,
            "counts": {c: self._counts.get(c, 0) for c in CATEGORIES},
            "truncated": [
                c for c in CATEGORIES
                if self._counts.get(c, 0) >= self.per_category
            ],
        }

    def write_index(self) -> str:
        """Write ``index.json`` describing every saved case; return its path."""
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, "index.json")
        with open(path, "w") as fh:
            json.dump(
                {
                    "summary": self.summary(),
                    "model_version": self.model_version,
                    "legend": {
                        "red_box": "model prediction",
                        "green_box": "ground truth",
                    },
                    "cases": [c.as_dict() for c in self.cases],
                },
                fh, indent=2, sort_keys=True,
            )
        return path


def categorise(
    pred_class: int | None,
    gt_class: int | None,
    score: float | None,
    *,
    low_conf_threshold: float = 0.45,
) -> str | None:
    """Bucket one match. ``None`` means "a correct, confident detection —
    nothing to save"."""
    if pred_class is None and gt_class is not None:
        return "false_negatives"
    if pred_class is not None and gt_class is None:
        return "false_positives"
    if pred_class is not None and gt_class is not None:
        if pred_class != gt_class:
            return "class_confusions"
        if score is not None and score < low_conf_threshold:
            return "low_confidence"
    return None
