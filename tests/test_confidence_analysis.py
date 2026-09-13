"""Confidence-curve maths: the module must say "the score ranks correctness"
only when it actually does, and must say the opposite when it doesn't."""

from __future__ import annotations

import pytest

from modules.confidence_analysis import (
    Bin,
    ConfidenceCurve,
    ScoredPrediction,
    analyse,
    build_curve,
    render_histogram,
)


def preds(spec):
    """spec: list of (score, correct, class_name)."""
    return [ScoredPrediction(s, c, n) for s, c, n in spec]


def test_bins_are_half_open_and_cover_score_one():
    """Score exactly 1.0 must land in the top bin, not fall off the end —
    YOLO does emit 1.0 and a dropped prediction would silently bias the curve."""
    curve = build_curve(preds([(1.0, True, "a"), (0.25, False, "a")]))
    assert curve.n == 2
    assert curve.bins[-1].n == 1
    assert curve.bins[0].n == 1


def test_scores_below_the_first_edge_are_excluded():
    """Predictions under the evaluation conf threshold aren't in the data; if one
    appears it must not be silently folded into the bottom bin."""
    curve = build_curve(preds([(0.10, False, "a")]))
    assert curve.n == 0


def test_perfectly_ordered_scores_give_positive_spearman_and_gap():
    curve = build_curve(preds(
        [(0.30, False, "a")] * 10
        + [(0.50, False, "a")] * 5 + [(0.50, True, "a")] * 5
        + [(0.95, True, "a")] * 10
    ))
    assert curve.spearman > 0.9
    assert curve.gap == pytest.approx(1.0)
    assert curve.monotonic_violations == 0
    assert "useful ranking signal" in curve.verdict()


def test_inverted_relationship_is_called_out():
    """A model that is *more* wrong when confident must not be reported as a
    useful ranking signal — that would be the most dangerous possible error."""
    curve = build_curve(preds(
        [(0.30, True, "a")] * 15 + [(0.95, False, "a")] * 15
    ))
    assert curve.gap < 0
    assert "does NOT separate" in curve.verdict()


def test_small_sample_refuses_to_judge():
    curve = build_curve(preds([(0.9, True, "a")] * 5))
    assert curve.n == 5
    assert curve.verdict() == "too few predictions to judge"


def test_monotonic_violations_counted():
    curve = ConfidenceCurve("x", bins=[
        Bin(0.25, 0.4, n=10, correct=2),   # 0.2
        Bin(0.4, 0.55, n=10, correct=8),   # 0.8  (up)
        Bin(0.55, 0.7, n=10, correct=4),   # 0.4  (down -> violation)
        Bin(0.7, 1.0001, n=10, correct=9),  # 0.9  (up)
    ])
    assert curve.monotonic_violations == 1


def test_empty_bins_are_ignored_by_spearman_and_gap():
    curve = ConfidenceCurve("x", bins=[
        Bin(0.25, 0.4, n=10, correct=1),
        Bin(0.4, 0.55, n=0, correct=0),  # empty: must not count as accuracy 0.0
        Bin(0.55, 1.0001, n=10, correct=9),
    ])
    assert len(curve.populated) == 2
    assert curve.gap == pytest.approx(0.8)
    assert curve.monotonic_violations == 0


def test_analyse_splits_per_class():
    out = analyse(preds(
        [(0.9, True, "Plate")] * 25
        + [(0.9, False, "WithHelmet")] * 25
    ))
    assert set(out["per_class"]) == {"Plate", "WithHelmet"}
    assert out["per_class"]["Plate"]["bins"][-1]["accuracy"] == 1.0
    assert out["per_class"]["WithHelmet"]["bins"][-1]["accuracy"] == 0.0
    assert out["overall"]["n"] == 50
    assert "not a calibrated probability" in out["note"]


def test_analyse_handles_no_predictions():
    out = analyse([])
    assert out["overall"]["n"] == 0
    assert out["per_class"] == {}


def test_render_histogram_marks_empty_bins():
    lines = render_histogram(build_curve(preds([(0.9, True, "a")] * 3)))
    assert lines[0].strip().startswith("score bin")
    # An empty bin shows a dash, never "0.000", which would read as 0% accuracy.
    assert any("  -  " in line for line in lines[1:])
    assert len(lines) == 6  # header + 5 default bins


def test_render_histogram_with_no_data_does_not_divide_by_zero():
    lines = render_histogram(build_curve([]))
    assert len(lines) == 6
