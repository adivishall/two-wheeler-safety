"""Per-violation pipeline decisions, edge cases, and the error budget."""

from __future__ import annotations

import random

import pytest

from modules.association import DetBox
from modules.pipeline_eval import (
    DEFAULT_INJECTIONS,
    DecisionMetrics,
    Injection,
    apply_injection,
    error_budget,
    evaluate_all,
    evaluate_violation,
    helmet_scenarios,
    run_pipeline_decisions,
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
    for f in range(10):
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


def test_error_budget_ranks_stages_and_is_deterministic():
    scenarios = [_clean_scenario()]
    a = error_budget(scenarios, trials=5, seed=3)
    b = error_budget(scenarios, trials=5, seed=3)
    assert a == b
    drops = [s["f1_drop"] for s in a["stages"]]
    assert drops == sorted(drops, reverse=True)  # worst first
    assert a["bottleneck"] == a["stages"][0]["stage"]


def test_error_budget_shares_sum_to_one_when_anything_dropped():
    a = error_budget([_clean_scenario()], trials=5, seed=3)
    total = sum(s["share_of_measured_sensitivity"] for s in a["stages"])
    if any(s["f1_drop"] > 0 for s in a["stages"]):
        assert total == pytest.approx(1.0, abs=0.01)


def test_error_budget_states_what_it_is_not():
    """The interpretation note is load-bearing: without it the numbers read as
    field failure rates, which they are not."""
    a = error_budget([_clean_scenario()], trials=3, seed=1)
    assert "NOT a claim about how often" in a["interpretation"]


def test_error_budget_baseline_is_the_undegraded_suite():
    a = error_budget([_clean_scenario()], trials=3, seed=1)
    assert a["baseline_f1"] == 1.0
    assert all(s["mean_f1"] <= a["baseline_f1"] + 1e-9 for s in a["stages"])


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


def test_pipeline_beats_single_frame_at_every_noise_level():
    """The project's central claim, as a regression test."""
    from modules.pipeline_eval import pipeline_vs_single_frame

    out = pipeline_vs_single_frame(trials=5)
    for rate, block in out["rates"].items():
        assert block["pipeline_f1_advantage"] > 0, f"no advantage at noise {rate}"
        assert block["pipeline"]["precision"] > block["naive"]["precision"]


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
