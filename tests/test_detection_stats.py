"""AP, bootstrap CIs, paired comparisons and threshold selection — on inputs
small enough to check by hand. Model-free."""

import numpy as np
import pytest

from modules.detection_stats import (
    ImageRecord,
    apply_thresholds,
    bootstrap_ap,
    f1_optimal_thresholds,
    match_for_ap,
    paired_bootstrap,
    pr_curve,
    records_from_json,
    records_to_json,
)

NAMES = ["A", "B"]
BOX = (0, 0, 10, 10)
FAR = (100, 100, 110, 110)


def rec(gt, preds):
    return match_for_ap(gt, preds, n_classes=2)


def test_matching_is_class_aware_and_score_ordered():
    r = rec([(0, BOX)], [(1, BOX, 0.9), (0, BOX, 0.8), (0, BOX, 0.7)])
    # class 1 pred has no class-1 GT -> FP; best class-0 pred takes the GT; the
    # lower-scored duplicate is an FP.
    by = sorted(zip(r.pred_cls.tolist(), r.pred_score.tolist(), r.pred_tp.tolist()))
    assert by == [(0, 0.7, 0.0), (0, 0.8, 1.0), (1, 0.9, 0.0)]
    assert r.gt_counts.tolist() == [1, 0]


def test_iou_threshold_is_respected():
    r = rec([(0, BOX)], [(0, (5, 5, 15, 15), 0.9)])  # IoU = 25/175 < 0.5
    assert r.pred_tp.tolist() == [0.0]


def test_perfect_detector_has_ultralytics_max_ap_and_a_degenerate_ci():
    # 101-point interpolation tops out at 0.995, exactly as Ultralytics reports.
    recs = [rec([(0, BOX), (1, FAR)], [(0, BOX, 0.9), (1, FAR, 0.8)]) for _ in range(20)]
    out = bootstrap_ap(recs, NAMES, n_boot=200, seed=0)
    for n in NAMES:
        pc = out["per_class"][n]
        assert pc["ap50"] == pytest.approx(0.995, abs=1e-3)
        assert pc["ci_low"] == pytest.approx(0.995, abs=1e-3)
    assert out["map50"]["value"] == pytest.approx(0.995, abs=1e-3)


def test_ap_curve_matches_ultralytics_compute_ap():
    """Pinned against values from ultralytics.utils.metrics.compute_ap (8.4.71),
    including the capped-recall case an older formula got wrong (0.75)."""
    from modules.detection_stats import _ap_from_curve

    assert _ap_from_curve(np.linspace(0.05, 0.5, 10), np.ones(10)) == \
        pytest.approx(0.495, abs=1e-6)
    assert _ap_from_curve(np.linspace(0.1, 1.0, 10), np.ones(10)) == \
        pytest.approx(0.995, abs=1e-6)


def test_a_false_positive_ranked_first_lowers_ap():
    good = [rec([(0, BOX)], [(0, BOX, 0.9)]) for _ in range(5)]
    bad = [rec([(0, BOX)], [(0, FAR, 0.99), (0, BOX, 0.5)]) for _ in range(5)]
    ap_good = bootstrap_ap(good, NAMES, n_boot=10)["per_class"]["A"]["ap50"]
    ap_bad = bootstrap_ap(bad, NAMES, n_boot=10)["per_class"]["A"]["ap50"]
    assert ap_good > ap_bad


def test_missed_objects_cap_recall():
    # 1 of 2 GT found with no false positives: AP ~ 0.5
    recs = [rec([(0, BOX), (0, FAR)], [(0, BOX, 0.9)]) for _ in range(10)]
    ap = bootstrap_ap(recs, NAMES, n_boot=10)["per_class"]["A"]["ap50"]
    assert ap == pytest.approx(0.495, abs=0.01)


def test_class_without_gt_is_reported_as_none_not_zero():
    recs = [rec([(0, BOX)], [(0, BOX, 0.9)]) for _ in range(5)]
    out = bootstrap_ap(recs, NAMES, n_boot=10)
    assert out["per_class"]["B"]["ap50"] is None
    assert out["per_class"]["B"]["instances"] == 0


def test_bootstrap_is_deterministic_and_brackets_the_point():
    rng = np.random.default_rng(3)
    recs = []
    for _ in range(40):
        hit = rng.random() < 0.7
        preds = [(0, BOX if hit else FAR, float(rng.random()))]
        recs.append(rec([(0, BOX)], preds))
    a = bootstrap_ap(recs, NAMES, n_boot=300, seed=5)
    b = bootstrap_ap(recs, NAMES, n_boot=300, seed=5)
    assert a == b
    pc = a["per_class"]["A"]
    assert pc["ci_low"] <= pc["ap50"] <= pc["ci_high"]
    assert pc["ci_high"] - pc["ci_low"] > 0.05  # 40 images is genuinely uncertain


def test_paired_bootstrap_identical_models_are_indistinguishable():
    recs = [rec([(0, BOX)], [(0, BOX, 0.9), (0, FAR, 0.3)]) for _ in range(20)]
    out = paired_bootstrap(recs, recs, NAMES, n_boot=100)
    assert out["per_class"]["A"]["diff"] == 0.0
    assert out["per_class"]["A"]["significant"] is False


def test_paired_bootstrap_detects_a_clearly_better_model():
    good = [rec([(0, BOX)], [(0, BOX, 0.9)]) for _ in range(30)]
    bad = [rec([(0, BOX)], [(0, FAR, 0.9)]) for _ in range(30)]
    out = paired_bootstrap(good, bad, NAMES, n_boot=200)
    a = out["per_class"]["A"]
    assert a["diff"] > 0.9 and a["significant"] and a["p_a_better"] == 1.0


def test_paired_bootstrap_requires_the_same_images():
    with pytest.raises(ValueError):
        paired_bootstrap([rec([], [])], [], NAMES, n_boot=5)


def test_thresholds_chosen_on_one_split_are_applied_unchanged_to_another():
    val = [rec([(0, BOX)], [(0, BOX, 0.8), (0, FAR, 0.3)]) for _ in range(10)]
    chosen = f1_optimal_thresholds(val, NAMES)
    t = chosen["A"]["threshold"]
    assert 0.3 < t <= 0.8  # above the FP's score, at/below the TP's
    test = [rec([(0, BOX)], [(0, BOX, 0.6), (0, FAR, 0.35)]) for _ in range(10)]
    applied = apply_thresholds(test, NAMES, {"A": t, "B": 0.5})
    assert applied["A"]["threshold"] == t


def test_pr_curve_and_json_roundtrip():
    recs = [rec([(0, BOX)], [(0, BOX, 0.9)]) for _ in range(3)]
    curve = pr_curve(recs, NAMES, points=5)
    assert len(curve["A"]) == 5
    back = records_from_json(records_to_json(recs))
    assert isinstance(back[0], ImageRecord)
    assert back[0].pred_tp.tolist() == recs[0].pred_tp.tolist()
