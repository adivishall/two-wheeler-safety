"""Per-violation pipeline decisions, edge cases, and the error budget."""

from __future__ import annotations

import random

import pytest

from modules.association import DetBox
from modules.pipeline_eval import (
    DEFAULT_INJECTIONS,
    DecisionMetrics,
    Injection,
    apply_faults,
    apply_injection,
    error_budget,
    evaluate_all,
    evaluate_violation,
    helmet_scenarios,
    run_pipeline_decisions,
    stage_tolerance,
    sweep_confirm_window,
    triple_riding_scenarios,
)
from modules.system_eval import GTVehicle, Scenario, VehicleFrame


def by_name(scenarios):
    return {s.name: s for s in scenarios}


# ---------------------------------------------------------------------------
# Decision metrics maths
# ---------------------------------------------------------------------------

def test_decision_metrics_precision_recall_f1():
    m = DecisionMetrics("no_helmet", true_positives=3, false_positives=1,
                        false_negatives=1, true_negatives=5)
    assert m.precision == pytest.approx(0.75)
    assert m.recall == pytest.approx(0.75)
    assert m.f1 == pytest.approx(0.75)


def test_decision_metrics_empty_is_not_zero_division():
    m = DecisionMetrics("no_helmet")
    assert m.precision == 1.0  # no predictions => no wrong predictions
    assert m.recall == 1.0
    assert m.as_dict()["f1"] == 1.0


# ---------------------------------------------------------------------------
# Helmet pipeline (Phase 10)
# ---------------------------------------------------------------------------

def test_helmet_pipeline_is_correct_on_the_edge_case_suite():
    m = evaluate_violation(helmet_scenarios(), "no_helmet")
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.precision == 1.0 and m.recall == 1.0


def test_single_frame_helmet_flicker_does_not_fine():
    """One bad frame on a helmeted rider must not produce a fine — the reason
    the temporal layer exists."""
    sc = by_name(helmet_scenarios())["helmet_flicker_only"]
    assert "no_helmet" not in run_pipeline_decisions(sc)[1]


def test_helmet_contradiction_is_never_fined():
    """Model asserts helmet AND no-helmet on one rider every frame: ambiguous,
    so nothing is fined. This is a reproduced real model bug."""
    sc = by_name(helmet_scenarios())["helmet_contradiction"]
    assert run_pipeline_decisions(sc)[1] == set()


def test_helmet_survives_temporary_detection_loss():
    sc = by_name(helmet_scenarios())["no_helmet_detection_loss"]
    assert "no_helmet" in run_pipeline_decisions(sc)[1]


# ---------------------------------------------------------------------------
# Triple riding (Phase 11)
# ---------------------------------------------------------------------------

def test_triple_pipeline_is_correct_on_the_edge_case_suite():
    m = evaluate_violation(triple_riding_scenarios(), "triple_riding")
    assert m.false_positives == 0
    assert m.false_negatives == 0


def test_exactly_three_and_four_or_more_both_fine():
    """The detector has one TripleRiding class, so both must produce the same
    verdict; the system detects the violation, it does not count riders."""
    sc = by_name(triple_riding_scenarios())
    assert "triple_riding" in run_pipeline_decisions(sc["exactly_three"])[1]
    assert "triple_riding" in run_pipeline_decisions(sc["four_or_more"])[1]


def test_two_riders_are_not_fined_for_triple_riding():
    sc = by_name(triple_riding_scenarios())["two_riders_legal"]
    assert "triple_riding" not in run_pipeline_decisions(sc)[1]


def test_triple_flicker_does_not_fine():
    sc = by_name(triple_riding_scenarios())["triple_flicker_only"]
    assert "triple_riding" not in run_pipeline_decisions(sc)[1]


def test_triple_survives_plate_occlusion_and_detection_loss():
    sc = by_name(triple_riding_scenarios())
    assert "triple_riding" in run_pipeline_decisions(sc["triple_plate_occluded"])[1]
    assert "triple_riding" in run_pipeline_decisions(sc["triple_detection_loss"])[1]


# ---------------------------------------------------------------------------
# Confirmation-window sweep
# ---------------------------------------------------------------------------

def test_single_frame_confirmation_is_measurably_worse():
    """confirm_window=1 is fining on one frame. It must score *worse* precision
    than the shipped window, or the temporal layer is buying nothing and the
    claim that it does would be unfounded."""
    sweep = sweep_confirm_window(helmet_scenarios(), "no_helmet")["windows"]
    assert sweep["1"]["precision"] < sweep["3"]["precision"]
    # ...and the extra frames must not cost recall on these scenarios.
    assert sweep["3"]["recall"] == sweep["1"]["recall"]


def test_sweep_covers_every_requested_window():
    sweep = sweep_confirm_window(helmet_scenarios(), "no_helmet",
                                 windows=(1, 4))["windows"]
    assert set(sweep) == {"1", "4"}
    assert sweep["4"]["confirm_window"] == 4


# ---------------------------------------------------------------------------
# Error budget (Phase 15)
# ---------------------------------------------------------------------------

def _clean_scenario() -> Scenario:
    frames = []
    for f in range(30):  # long enough for the shipped >=12-frame helmet gate
        body = (100 + 5 * f, 100, 160 + 5 * f, 200)
        frames.append([VehicleFrame(1, [
            DetBox("WithoutHelmet", body, 0.9),
            DetBox("Plate", (110 + 5 * f, 205, 150 + 5 * f, 235), 0.9),
        ], "MH12AB1234")])
    return Scenario("clean", [GTVehicle(1, "MH12AB1234", frozenset({"no_helmet"}))],
                    frames)


def test_injection_with_zero_rates_is_a_no_op():
    sc = _clean_scenario()
    out = apply_injection(sc, Injection("none", "nothing"), random.Random(0))
    assert [len(f) for f in out.frames] == [len(f) for f in sc.frames]
    assert out.frames[0][0].plate_text == "MH12AB1234"


def test_injection_never_changes_ground_truth():
    """Degrading an input must not change what the right answer is, or the
    measurement would be meaningless."""
    sc = _clean_scenario()
    out = apply_injection(sc, DEFAULT_INJECTIONS[0], random.Random(1))
    assert out.vehicles == sc.vehicles


def test_detection_drop_removes_boxes():
    sc = _clean_scenario()
    out = apply_injection(sc, Injection("d", "", detection_drop=1.0), random.Random(0))
    assert all(vf.dets == [] for frame in out.frames for vf in frame)


def test_class_flip_swaps_helmet_labels_only():
    sc = _clean_scenario()
    out = apply_injection(sc, Injection("c", "", class_flip=1.0), random.Random(0))
    labels = {d.label for frame in out.frames for vf in frame for d in vf.dets}
    assert "WithHelmet" in labels  # WithoutHelmet was flipped
    assert "Plate" in labels  # Plate is untouched


def test_plate_drop_removes_the_text_not_the_box():
    sc = _clean_scenario()
    out = apply_injection(sc, Injection("p", "", plate_drop=1.0), random.Random(0))
    assert all(vf.plate_text is None for frame in out.frames for vf in frame)
    assert any(d.label == "Plate" for frame in out.frames for vf in frame
               for d in vf.dets)


def test_ocr_noise_corrupts_text_but_keeps_its_length():
    sc = _clean_scenario()
    out = apply_injection(sc, Injection("o", "", ocr_noise=1.0), random.Random(0))
    text = out.frames[0][0].plate_text
    assert text != "MH12AB1234"
    assert len(text) == len("MH12AB1234")


def test_error_budget_is_an_oracle_ablation_and_is_deterministic():
    scenarios = [_clean_scenario()]
    a = error_budget(scenarios, trials=4, seed=3)
    b = error_budget(scenarios, trials=4, seed=3)
    assert a == b
    assert a["method"].startswith("oracle ablation")
    for block in a["by_ocr_rate"].values():
        rec = [s["recovered_f1"] for s in block["stages"]]
        assert rec == sorted(rec, reverse=True)  # largest owner first
        assert all(s["f1_if_perfect"] >= block["all_faults_f1"] - 1e-9
                   for s in block["stages"])


def test_error_budget_ocr_rate_is_swept_not_assumed():
    a = error_budget([_clean_scenario()], trials=2, seed=1, ocr_rates=(0.1, 0.5))
    assert set(a["by_ocr_rate"]) == {"0.10", "0.50"}
    assert "NOT a claim about how often" in a["interpretation"]
    assert isinstance(a["bottleneck_robust_to_ocr_assumption"], bool)


def test_faults_use_one_unit_and_never_touch_ground_truth():
    sc = _clean_scenario()
    all_off = apply_faults(sc, {"ocr_corrupt": 0.0}, random.Random(0))
    assert [len(f[0].dets) for f in all_off.frames] == [len(f[0].dets) for f in sc.frames]
    plate_gone = apply_faults(sc, {"plate_miss": 1.0}, random.Random(0))
    assert all(f[0].plate_text is None for f in plate_gone.frames)
    assert all(d.label != "Plate" for f in plate_gone.frames for d in f[0].dets)
    misread = apply_faults(sc, {"ocr_corrupt": 1.0}, random.Random(0))
    t = misread.frames[0][0].plate_text
    assert t != "MH12AB1234" and len(t) == len("MH12AB1234")
    assert sum(a != b for a, b in zip(t, "MH12AB1234")) == 1  # ONE glyph per read
    flipped = apply_faults(sc, {"class_flip": {"WithoutHelmet": ("WithHelmet", 1.0)}},
                           random.Random(0))
    assert flipped.frames[0][0].dets[0].label == "WithHelmet"
    assert flipped.vehicles == sc.vehicles


def test_stage_tolerance_reports_every_stage_at_every_rate():
    out = stage_tolerance([_clean_scenario()], rates=(0.1, 0.5), trials=2)
    assert set(out["f1_by_rate"]) == {"rider_recall", "helmet_class", "plate_recall", "ocr"}
    assert all(set(r) == {"0.10", "0.50"} for r in out["f1_by_rate"].values())


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def test_evaluate_all_is_deterministic_and_complete():
    a = evaluate_all(trials=3)
    b = evaluate_all(trials=3)
    assert a == b
    assert set(a) == {"helmet", "triple_riding", "error_budget"}
    assert a["triple_riding"]["rider_count_note"]
    assert "does not count riders" in a["triple_riding"]["rider_count_note"]


# ---------------------------------------------------------------------------
# Pipeline vs a single-frame detector (Phase 30 — the central claim)
# ---------------------------------------------------------------------------

def test_naive_policy_fines_on_any_single_frame():
    """The strawman must actually be the strawman: one violation frame in an
    otherwise clean clip must make it fine."""
    from modules.pipeline_eval import _flicker_scenario, naive_single_frame_decisions

    sc = _flicker_scenario("flicker")  # clean bike, one TripleRiding frame
    assert "triple_riding" in naive_single_frame_decisions(sc)[1]
    # ...and the real pipeline must not.
    assert "triple_riding" not in run_pipeline_decisions(sc)[1]


def test_naive_policy_ignores_the_contradiction_safeguard():
    from modules.pipeline_eval import helmet_scenarios, naive_single_frame_decisions

    sc = by_name(helmet_scenarios())["helmet_contradiction"]
    assert "no_helmet" in naive_single_frame_decisions(sc)[1]
    assert "no_helmet" not in run_pipeline_decisions(sc)[1]


def test_pipeline_is_more_precise_at_every_noise_level():
    """The project's central claim, stated as precisely as it holds: the
    pipeline is more PRECISE than single-frame fining at every noise level."""
    from modules.pipeline_eval import pipeline_vs_single_frame

    out = pipeline_vs_single_frame(trials=5)
    for rate, block in out["rates"].items():
        assert block["pipeline"]["precision"] > block["naive"]["precision"], rate


def test_pipeline_wins_f1_in_the_measured_noise_range_only():
    """F1 advantage holds where the detector actually operates (val-measured
    helmet flip rates are 8-15% per frame) — and is NOT claimed at extreme
    noise, where the pipeline trades recall for precision so hard that F1 falls
    below the naive policy. If that ever flips, the docs are out of date."""
    from modules.pipeline_eval import pipeline_vs_single_frame

    out = pipeline_vs_single_frame(rates=(0.0, 0.1, 0.2, 0.4), trials=5)
    for rate in ("0.00", "0.10", "0.20"):
        assert out["rates"][rate]["pipeline_f1_advantage"] > 0, rate
    assert out["rates"]["0.40"]["pipeline_f1_advantage"] < 0


def test_advantage_is_precision_not_recall():
    """The naive policy fines on anything, so it never misses — the entire
    difference must come from precision. If a future change made the pipeline
    win on recall instead, the framing in the docs would be wrong."""
    from modules.pipeline_eval import pipeline_vs_single_frame

    out = pipeline_vs_single_frame(rates=(0.0, 0.2), trials=5)
    for block in out["rates"].values():
        assert block["naive"]["recall"] == 1.0
        assert block["pipeline"]["recall"] <= block["naive"]["recall"]
        assert block["pipeline"]["mean_false_positives"] < \
            block["naive"]["mean_false_positives"]


def test_headroom_is_deterministic():
    from modules.pipeline_eval import pipeline_vs_single_frame

    assert pipeline_vs_single_frame(rates=(0.1,), trials=3) == \
        pipeline_vs_single_frame(rates=(0.1,), trials=3)


def test_headroom_states_that_the_naive_policy_is_given_free_association():
    """The comparison is only honest if it says where it favours the strawman."""
    from modules.pipeline_eval import pipeline_vs_single_frame

    out = pipeline_vs_single_frame(rates=(0.0,), trials=2)
    assert "lower bound" in out["note"]
