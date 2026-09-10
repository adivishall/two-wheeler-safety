"""Tests for the model-free evaluation primitives (modules/evaluation.py)."""

from modules.evaluation import (
    ConfidenceStats,
    ConfusionMatrix,
    build_class_reports,
    iou,
    match_image,
    precision_recall_f1,
)

NAMES = ["Plate", "WithHelmet", "WithoutHelmet", "TripleRiding"]


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # half-overlap on x -> intersection 50, union 150
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-6


def test_match_true_positive():
    gt = [(1, (0, 0, 10, 10))]
    preds = [(1, (0, 0, 9, 9), 0.9)]
    m = match_image(gt, preds, iou_threshold=0.5)
    assert len(m) == 1
    assert m[0].gt_class == 1 and m[0].pred_class == 1


def test_match_false_positive_and_negative():
    gt = [(1, (0, 0, 10, 10))]
    preds = [(2, (100, 100, 110, 110), 0.8)]  # nowhere near the GT
    m = match_image(gt, preds, iou_threshold=0.5)
    # one FP (pred with no GT) and one FN (GT with no pred)
    fps = [x for x in m if x.gt_class is None and x.pred_class == 2]
    fns = [x for x in m if x.pred_class is None and x.gt_class == 1]
    assert len(fps) == 1 and len(fns) == 1


def test_match_class_confusion_is_one_pair_not_fp_plus_fn():
    # A WithHelmet box predicted over a WithoutHelmet GT is a confusion.
    gt = [(2, (0, 0, 10, 10))]
    preds = [(1, (0, 0, 10, 10), 0.7)]
    m = match_image(gt, preds, iou_threshold=0.5)
    assert len(m) == 1
    assert m[0].gt_class == 2 and m[0].pred_class == 1  # matched, but wrong class


def test_highest_confidence_pred_claims_gt_first():
    gt = [(1, (0, 0, 10, 10))]
    preds = [(1, (0, 0, 10, 10), 0.4), (1, (0, 0, 9, 9), 0.95)]
    m = match_image(gt, preds, iou_threshold=0.5)
    matched = [x for x in m if x.gt_class is not None and x.pred_class is not None]
    fps = [x for x in m if x.gt_class is None]
    assert len(matched) == 1 and matched[0].score == 0.95  # the confident one won
    assert len(fps) == 1  # the other became a false positive


def test_confusion_matrix_counts():
    cm = ConfusionMatrix(len(NAMES))
    gt = [(1, (0, 0, 10, 10)), (2, (20, 20, 30, 30))]
    preds = [
        (1, (0, 0, 10, 10), 0.9),        # correct WithHelmet
        (1, (20, 20, 30, 30), 0.8),      # WithoutHelmet GT predicted WithHelmet
        (3, (99, 99, 109, 109), 0.7),    # spurious TripleRiding -> FP
    ]
    cm.add_matches(match_image(gt, preds, 0.5))
    counts = cm.per_class_counts()
    assert counts[1]["tp"] == 1
    assert counts[2]["fn"] == 1  # WithoutHelmet missed as such
    assert counts[3]["fp"] == 1  # spurious TripleRiding


def test_precision_recall_f1_edges():
    assert precision_recall_f1(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    prf = precision_recall_f1(8, 2, 2)
    assert prf["precision"] == 0.8 and prf["recall"] == 0.8 and prf["f1"] == 0.8


def test_build_class_reports_names_align():
    cm = ConfusionMatrix(len(NAMES))
    reports = build_class_reports(cm, NAMES)
    assert [r.name for r in reports] == NAMES


def test_confidence_separation_direction():
    cs = ConfidenceStats()
    cs.add(True, 0.9)
    cs.add(True, 0.8)
    cs.add(False, 0.4)
    s = cs.summary()
    assert s["n_correct"] == 2 and s["n_wrong"] == 1
    assert s["separation"] > 0  # more confident when right
