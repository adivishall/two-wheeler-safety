"""Label-quality and composition audit for a YOLO dataset (pure, model-free).

The leakage audit (``modules/dataset_audit.py``) answers "is held-out really
held out?". This answers the other questions a detector's numbers depend on:

* **Malformed labels** — wrong field count, class id out of range, coordinates
  outside [0, 1], zero/negative size. Any of these silently corrupts training
  or evaluation.
* **Duplicate boxes** — the same object annotated twice (same class, IoU >
  0.9): inflates instance counts and penalises a correct single prediction as
  a false negative.
* **Contradictory annotations** — a ``WithHelmet`` and a ``WithoutHelmet`` box
  on the same rider (IoU > 0.5). The model is trained to be confused exactly
  where the labels are.
* **Object size** — the share of tiny boxes per class. Plates and helmets are
  small; a class dominated by tiny boxes is resolution-limited, which no amount
  of threshold tuning fixes.
* **Source composition** — which upstream export each image came from (the
  filename prefix), per split and per class. If one class's held-out instances
  come mostly from one source, that class's metric measures that source.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict

TINY_AREA = (16 / 640) ** 2  # < 16 px square at the 640 training resolution
DUP_IOU = 0.9
CONTRADICTION_IOU = 0.5


def _iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def source_of(filename: str) -> str:
    """Upstream export an image came from: the filename's first token.

    ``ds1_58e91e_BikesHelmets558_png.rf.<hash>.jpg`` -> ``ds1``;
    ``aug_8a4368db.jpg`` -> ``aug`` (offline-augmented; origin not recoverable)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split("_", 1)[0] or "unknown"


def parse_label_file(text: str, n_classes: int) -> tuple[list, list]:
    """``(boxes, problems)`` where boxes are ``(cls, cx, cy, w, h)``."""
    boxes, problems = [], []
    for ln, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            problems.append(("field_count", ln))
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            problems.append(("not_numeric", ln))
            continue
        if not 0 <= cls < n_classes:
            problems.append(("class_out_of_range", ln))
            continue
        if w <= 0 or h <= 0:
            problems.append(("non_positive_size", ln))
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1) or w > 1.0001 or h > 1.0001:
            problems.append(("coords_out_of_range", ln))
        boxes.append((cls, cx, cy, w, h))
    return boxes, problems


def audit_split(pairs, class_names: list[str], *, check_images: bool = False) -> dict:
    """Audit one split. ``pairs`` yields ``(image_path, label_path)``."""
    n = len(class_names)
    try:
        h_idx = class_names.index("WithHelmet")
        nh_idx = class_names.index("WithoutHelmet")
    except ValueError:
        h_idx = nh_idx = -1
    problems: Counter = Counter()
    problem_examples: list = []
    instances: Counter = Counter()
    tiny: Counter = Counter()
    areas: dict[int, list] = defaultdict(list)
    duplicates = contradictions = 0
    contradiction_examples: list = []
    sources: Counter = Counter()
    source_class: dict[str, Counter] = defaultdict(Counter)
    cooc = [[0] * n for _ in range(n)]  # images containing class i AND class j
    images = background = unreadable = 0

    for image_path, label_path in pairs:
        images += 1
        src = source_of(image_path)
        sources[src] += 1
        if check_images:
            import cv2

            if cv2.imread(image_path) is None:
                unreadable += 1
        if not os.path.exists(label_path):
            background += 1
            continue
        with open(label_path) as fh:
            boxes, probs = parse_label_file(fh.read(), n)
        for kind, ln in probs:
            problems[kind] += 1
            if len(problem_examples) < 10:
                problem_examples.append({"label": os.path.relpath(label_path), "line": ln,
                                         "problem": kind})
        if not boxes:
            background += 1
        present = sorted({b[0] for b in boxes})
        for i in present:
            for j in present:
                cooc[i][j] += 1
        for cls, _cx, _cy, w, h in boxes:
            instances[cls] += 1
            source_class[src][cls] += 1
            areas[cls].append(w * h)
            if w * h < TINY_AREA:
                tiny[cls] += 1
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                v = _iou_xywh(a[1:], b[1:])
                if a[0] == b[0] and v > DUP_IOU:
                    duplicates += 1
                elif {a[0], b[0]} == {h_idx, nh_idx} and v > CONTRADICTION_IOU:
                    contradictions += 1
                    if len(contradiction_examples) < 10:
                        contradiction_examples.append(
                            {"label": os.path.relpath(label_path), "iou": round(v, 3)})

    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    per_class = {}
    for c, name in enumerate(class_names):
        k = instances[c]
        per_class[name] = {
            "instances": k,
            "tiny_boxes": tiny[c],
            "tiny_share": round(tiny[c] / k, 4) if k else None,
            "median_area_pct": round(100 * median(areas[c]), 3) if areas[c] else None,
        }
    return {
        "images": images,
        "background_images": background,
        "unreadable_images": unreadable if check_images else None,
        "malformed_labels": dict(problems),
        "malformed_examples": problem_examples,
        "duplicate_boxes": duplicates,
        "helmet_contradictions": contradictions,
        "contradiction_examples": contradiction_examples,
        "per_class": per_class,
        "sources": dict(sources.most_common()),
        "class_cooccurrence_images": {
            class_names[i]: {class_names[j]: cooc[i][j] for j in range(n)} for i in range(n)
        },
        "source_class_instances": {
            s: {class_names[c]: k for c, k in sorted(cnt.items())}
            for s, cnt in sorted(source_class.items())
        },
    }


def dominant_source_share(split_report: dict, class_name: str) -> tuple[str | None, float]:
    """Which source contributes most of a class's instances in a split, and how much."""
    per_source = {s: c.get(class_name, 0)
                  for s, c in split_report["source_class_instances"].items()}
    total = sum(per_source.values())
    if not total:
        return None, 0.0
    src = max(per_source, key=lambda s: per_source[s])
    return src, round(per_source[src] / total, 4)


def disjoint_label_groups(split_report: dict) -> list[list[str]]:
    """Groups of classes that NEVER share a labelled image in this split.

    Two groups means the split is really two datasets glued together, each
    annotating only its own classes — so an object of the other group in an
    image is unlabelled, i.e. taught (and scored) as background."""
    cooc = split_report["class_cooccurrence_images"]
    names = [c for c in cooc if cooc[c][c] > 0]
    groups: list[set] = []
    for c in names:
        linked = [g for g in groups if any(cooc[c][o] > 0 for o in g)]
        merged = {c}.union(*linked) if linked else {c}
        groups = [g for g in groups if g not in linked] + [merged]
    return [sorted(g) for g in groups]
