"""Stratified detector evaluation: condition measurements, COCO range matching,
unpaired gaps, and the rules that keep a per-stratum table from over-claiming."""

import cv2
import numpy as np

from modules.detection_stats import (
    match_for_ap,
    match_for_ap_in_range,
    recall_at,
    unpaired_gap,
)
from modules.image_conditions import (
    evaluate,
    fit_thresholds,
    image_properties,
    image_strata,
    notable_gaps,
)

NAMES = ["Plate", "WithHelmet", "WithoutHelmet", "TripleRiding"]


def test_size_range_matching_ignores_out_of_range_objects():
    small, large = (0, 0, 10, 10), (0, 50, 100, 150)
    gt = [(2, small), (2, large)]
    preds = [(2, large, 0.9),            # found the large object: ignored, not a FP
             (2, (200, 200, 205, 205), 0.8),  # unmatched, small -> in range -> FP
             (2, (300, 0, 400, 100), 0.7)]    # unmatched, large -> out of range -> dropped
    is_small = lambda c, b: (b[2] - b[0]) < 20  # noqa: E731
    rec = match_for_ap_in_range(gt, preds, 4, is_small)
    assert rec.gt_counts[2] == 1
    assert list(rec.pred_tp) == [0.0]  # only the small false positive remains
    full = match_for_ap(gt, preds, 4)
    assert full.gt_counts[2] == 2 and len(full.pred_tp) == 3


def _records(n, hit_rate, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        gt = [(2, (0, 0, 50, 50))]
        hit = rng.random() < hit_rate
        preds = [(2, (0, 0, 50, 50) if hit else (200, 200, 250, 250), float(rng.random()))]
        out.append(match_for_ap(gt, preds, 4))
    return out


def test_unpaired_gap_separates_a_real_difference_from_noise():
    good, bad = _records(60, 0.95, 1), _records(60, 0.3, 2)
    assert unpaired_gap(good, bad, NAMES, n_boot=300)["per_class"]["WithoutHelmet"]["significant"]
    same = unpaired_gap(_records(60, 0.7, 3), _records(60, 0.7, 4), NAMES, n_boot=300)
    assert not same["per_class"]["WithoutHelmet"]["significant"]


def test_recall_at_uses_each_classes_operating_threshold():
    recs = [match_for_ap([(2, (0, 0, 50, 50))], [(2, (0, 0, 50, 50), 0.3)], 4)]
    assert recall_at(recs, NAMES, {"WithoutHelmet": 0.25})["WithoutHelmet"]["recall"] == 1.0
    assert recall_at(recs, NAMES, {"WithoutHelmet": 0.375})["WithoutHelmet"]["recall"] == 0.0


def test_image_properties_measure_darkness_blur_and_glare():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    dark = (sharp * 0.1).astype(np.uint8)
    glare = sharp.copy()
    glare[:100] = 255
    assert image_properties(dark)["luminance"] < 60 < image_properties(sharp)["luminance"]
    assert image_properties(blurred)["sharpness"] < image_properties(sharp)["sharpness"]
    assert image_properties(glare)["glare_share"] > 0.05 > image_properties(sharp)["glare_share"]


def _synthetic_split(n=80, seed=0):
    """Rows where multi-rider images are harder for WithoutHelmet by design."""
    rng = np.random.default_rng(seed)
    rows, props, sources = [], [], []
    for i in range(n):
        riders = 1 if i % 2 else 3
        gt, preds = [], []
        for k in range(riders):
            box = [10 + 60 * k, 10, 60 + 60 * k, 110]
            gt.append([2, *box])
            hit = rng.random() < (0.95 if riders == 1 else 0.4)
            pbox = box if hit else [400, 400, 450, 450]
            preds.append([2, *pbox, float(rng.uniform(0.3, 1.0))])
        gt.append([0, 20, 120, 60, 135])
        preds.append([0, 20, 120, 60, 135, 0.9])
        rows.append({"image": f"img{i}.jpg", "w": 640, "h": 480, "gt": gt, "preds": preds})
        props.append({"luminance": 30.0 if i < 12 else 120.0, "sharpness": float(100 + i),
                      "glare_share": 0.0, "width": 640, "height": 480})
        sources.append("ds1")
    return rows, props, sources


def test_stratified_evaluation_finds_a_planted_difference_and_counts_comparisons():
    rows, props, sources = _synthetic_split()
    thr = fit_thresholds(rows, props, NAMES)
    rep = evaluate(rows, props, sources, NAMES, thr, op_thresholds={}, n_boot=200)
    strata = rep["image_strata"]["labelled rider boxes"]
    assert set(strata) == {"1", "2-3"}
    assert strata["1"]["ap"]["per_class"]["WithoutHelmet"]["ap50"] > \
        strata["2-3"]["ap"]["per_class"]["WithoutHelmet"]["ap50"]
    findings, tested = notable_gaps(rep, NAMES)
    assert tested > 0
    assert any(f["attribute"] == "labelled rider boxes" and f["class"] == "WithoutHelmet"
               for f in findings)
    # classes with fewer than 10 instances on either side are never "findings"
    assert all(f["class"] != "WithHelmet" for f in findings)


def test_strata_follow_the_fitted_cut_points():
    rows, props, sources = _synthetic_split()
    thr = fit_thresholds(rows, props, NAMES)
    s = image_strata(rows[0], {**props[0], "luminance": 20.0, "glare_share": 0.2}, thr, NAMES,
                     "dst")
    assert s["lighting (luminance proxy)"] == "dark" and s["glare"] == "glare"
    assert s["source"] == "dst" and s["labelled rider boxes"] == "2-3"


def test_range_matching_prefers_the_in_range_object_like_coco():
    small, large = (0, 0, 20, 20), (0, 0, 22, 22)  # overlapping: IoU ~0.83
    gt = [(2, large), (2, small)]
    pred = [(2, (0, 0, 21, 21), 0.9)]
    is_small = lambda c, b: (b[2] - b[0]) < 21  # noqa: E731
    rec = match_for_ap_in_range(gt, pred, 4, is_small)
    assert list(rec.pred_tp) == [1.0]  # counted for the small object, not dropped


def test_binary_conditions_are_tested_once_not_mirrored():
    rows, props, sources = _synthetic_split()
    for i, p in enumerate(props):
        p["glare_share"] = 0.2 if i % 2 else 0.0
    thr = fit_thresholds(rows, props, NAMES)
    rep = evaluate(rows, props, sources, NAMES, thr, op_thresholds={}, n_boot=100)
    findings, _ = notable_gaps(rep, NAMES)
    glare_values = {f["value"] for f in findings if f["attribute"] == "glare"}
    assert len(glare_values) <= 1
