"""Small, dependency-free box-geometry helpers shared across the pipeline.

Boxes are ``(x1, y1, x2, y2)`` in pixel coordinates with ``x2 >= x1`` and
``y2 >= y1``. Everything here is pure and deterministic so it can be unit
tested with synthetic boxes and no model.

Before this module the same ``iou`` was copy-pasted into ``modules/detector``,
``main`` and ``modules/video_detector``; they now share one definition.
"""

from __future__ import annotations

import math

Box = tuple[int, int, int, int]


def area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def centroid(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def intersection_area(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two boxes; 0.0 when they don't overlap."""
    inter = intersection_area(a, b)
    if inter == 0.0:
        return 0.0
    return inter / (area(a) + area(b) - inter)


def containment(inner: Box, outer: Box) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer`` (0..1).

    High when a small box (e.g. a helmet) sits inside a big one (e.g. a
    triple-riding body box), even though their IoU is low.
    """
    a = area(inner)
    if a == 0.0:
        return 0.0
    return intersection_area(inner, outer) / a


def horizontal_overlap_ratio(a: Box, b: Box) -> float:
    """Overlap of the boxes' x-spans divided by the narrower span (0..1).

    On a two-wheeler the plate sits roughly under the rider, so their
    horizontal spans overlap strongly even when the boxes are vertically
    apart (low IoU). This separates "plate belongs to this rider" from a
    neighbouring bike's plate far better than centroid distance alone.
    """
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    narrower = min(ax2 - ax1, bx2 - bx1)
    if narrower <= 0:
        return 0.0
    return overlap / narrower


def centroid_distance(a: Box, b: Box) -> float:
    ax, ay = centroid(a)
    bx, by = centroid(b)
    return math.hypot(ax - bx, ay - by)


def union_box(a: Box, b: Box) -> Box:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))


def diagonal(box: Box) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(x2 - x1, y2 - y1)
