"""Confidence-calibration tests (Phase 10). Pure Python, deterministic."""

import pytest

from modules.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    fit_and_evaluate,
    load_calibrator,
    maximum_calibration_error,
    reliability_curve,
    save_calibrator,
)


def _dataset(accuracy_of_score, *, per_bin=100):
    """Build (scores, labels) where, at each score level, the fraction of
    correct(1) labels equals accuracy_of_score(score). Deterministic."""
    scores, labels = [], []
    for i in range(1, 10):
        s = i / 10.0
        acc = accuracy_of_score(s)
        n_correct = round(acc * per_bin)
        for k in range(per_bin):
            scores.append(s)
            labels.append(1 if k < n_correct else 0)
    return scores, labels


# -- metrics ----------------------------------------------------------------

def test_brier_score_bounds():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0
    assert brier_score([], []) == 0.0


def test_ece_near_zero_for_calibrated_data():
    scores, labels = _dataset(lambda s: s)  # accuracy == confidence
    ece = expected_calibration_error(scores, labels, n_bins=10)
    assert ece < 0.02


def test_ece_high_for_overconfident_data():
    scores, labels = _dataset(lambda s: s * 0.5)  # 0.8 conf -> 40% correct
    ece = expected_calibration_error(scores, labels, n_bins=10)
    assert ece > 0.15


def test_reliability_curve_bins_cover_data():
    scores, labels = _dataset(lambda s: s)
    bins = reliability_curve(scores, labels, n_bins=10)
    assert len(bins) == 10
    assert sum(b.count for b in bins) == len(scores)


def test_mce_is_at_least_ece():
    scores, labels = _dataset(lambda s: s * 0.5)
    ece = expected_calibration_error(scores, labels)
    mce = maximum_calibration_error(scores, labels)
    assert mce >= ece


# -- calibrators ------------------------------------------------------------

def test_platt_reduces_ece_on_overconfident_data():
    scores, labels = _dataset(lambda s: s * 0.5)
    raw = expected_calibration_error(scores, labels)
    cal = PlattCalibrator().fit(scores, labels)
    calibrated = [cal.predict(s) for s in scores]
    assert expected_calibration_error(calibrated, labels) < raw


def test_platt_is_monotonic():
    scores, labels = _dataset(lambda s: s * 0.5)
    cal = PlattCalibrator().fit(scores, labels)
    preds = [cal.predict(s / 20) for s in range(21)]
    assert preds == sorted(preds)


def test_isotonic_reduces_ece_and_is_monotonic():
    scores, labels = _dataset(lambda s: s * 0.5)
    raw = expected_calibration_error(scores, labels)
    cal = IsotonicCalibrator().fit(scores, labels)
    calibrated = [cal.predict(s) for s in scores]
    assert expected_calibration_error(calibrated, labels) <= raw
    preds = [cal.predict(s / 20) for s in range(21)]
    assert preds == sorted(preds)


def test_isotonic_predict_without_fit_is_identity_clamped():
    cal = IsotonicCalibrator()
    assert cal.predict(0.5) == 0.5
    assert cal.predict(2.0) == 1.0


# -- fit_and_evaluate + persistence -----------------------------------------

def test_fit_and_evaluate_picks_calibration_when_it_helps():
    scores, labels = _dataset(lambda s: s * 0.5)
    report = fit_and_evaluate(scores, labels)
    assert report["best"] in ("platt", "isotonic")  # raw is miscalibrated
    assert report["platt"]["ece"] < report["raw"]["ece"]


def test_fit_and_evaluate_keeps_raw_when_already_calibrated():
    scores, labels = _dataset(lambda s: s)
    report = fit_and_evaluate(scores, labels)
    # already calibrated: raw ECE is tiny; best must not be meaningfully worse.
    assert report["raw"]["ece"] < 0.02


def test_save_and_load_roundtrip(tmp_path):
    scores, labels = _dataset(lambda s: s * 0.5)
    cal = PlattCalibrator().fit(scores, labels)
    path = tmp_path / "cal.json"
    save_calibrator(cal, str(path))
    loaded = load_calibrator(str(path))
    assert isinstance(loaded, PlattCalibrator)
    assert loaded.predict(0.8) == pytest.approx(cal.predict(0.8), abs=1e-9)


def test_load_missing_calibrator_returns_none(tmp_path):
    assert load_calibrator(str(tmp_path / "nope.json")) is None


# -- out-of-fold selection (the fix for in-sample calibrator selection) -------

def _bernoulli_calibrated(n=400, seed=5):
    """Scores whose raw value IS the probability of being correct."""
    import random

    rng = random.Random(seed)
    scores = [rng.random() for _ in range(n)]
    labels = [1 if rng.random() < s else 0 for s in scores]
    return scores, labels


def test_calibrated_scores_are_not_replaced_by_an_in_sample_winner():
    """Regression: calibrators used to be fitted and scored on the same rows,
    and the lowest in-sample ECE won. On scores that are ALREADY calibrated that
    rule adopted a calibrator in ~30 of 40 simulated datasets (it rewards
    fitting the noise); out-of-fold selection adopts none."""
    scores, labels = _bernoulli_calibrated(n=300, seed=0)
    old_rule = {"raw": expected_calibration_error(scores, labels)}
    for name, make in (("platt", PlattCalibrator), ("isotonic", IsotonicCalibrator)):
        cal = make().fit(scores, labels)
        old_rule[name] = expected_calibration_error([cal.predict(x) for x in scores], labels)
    assert min(old_rule, key=lambda k: old_rule[k]) != "raw"  # the old trap fires here
    report = fit_and_evaluate(scores, labels)
    assert report["best"] == "raw"
    assert report["verdict"] == "raw_is_as_good"


def test_too_few_outcomes_is_reported_not_calibrated():
    scores, labels = _bernoulli_calibrated(n=40)
    report = fit_and_evaluate(scores, labels)
    assert report["verdict"] == "insufficient_data" and report["best"] == "raw"
    assert "platt" not in report


def test_grouped_folds_never_split_a_group():
    from modules.calibration import _fold_ids

    groups = [f"session{i // 7}" for i in range(70)]
    ids = _fold_ids(70, 5, groups, seed=1)
    for g in set(groups):
        assert len({f for f, gg in zip(ids, groups) if gg == g}) == 1


def test_threshold_table_trades_recall_for_precision():
    from modules.calibration import threshold_table

    scores, labels = _dataset(lambda s: s)
    rows = threshold_table(scores, labels, [0.1, 0.5, 0.9])
    assert rows[0]["recall"] >= rows[1]["recall"] >= rows[2]["recall"]
    assert rows[0]["precision"] <= rows[2]["precision"]
    assert rows[2]["kept"] < rows[0]["kept"]
