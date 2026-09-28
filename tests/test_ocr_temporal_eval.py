"""Single-frame vs temporal OCR comparison.

The headline claim this project makes about OCR is that temporal stabilization
beats reading one frame. These tests protect the *measurement* of that claim:
the comparison must be reproducible, must score every policy on identical
inputs, and must not quietly let the temporal policy's abstentions be counted as
wins.
"""

from __future__ import annotations

import random

import pytest

from modules.ocr_temporal_eval import (
    POLICIES,
    ComparisonReport,
    FrameRead,
    NoiseModel,
    apply_policy,
    compare_policies,
    corrupt,
    noise_sweep,
    run_simulation,
    simulate_sequence,
)

PLATE = "MH12AB1234"


def test_simulation_is_deterministic_for_a_seed():
    a = run_simulation(frames=6, repeats=2, seed=42).as_dict()
    b = run_simulation(frames=6, repeats=2, seed=42).as_dict()
    assert a == b


def test_different_seeds_give_different_draws():
    a = run_simulation(frames=6, repeats=2, seed=1).as_dict()
    b = run_simulation(frames=6, repeats=2, seed=2).as_dict()
    assert a != b


def test_zero_noise_leaves_the_plate_untouched():
    model = NoiseModel(substitute_rate=0, drop_char_rate=0, insert_char_rate=0)
    text, corrupted = corrupt(PLATE, model, random.Random(0))
    assert text == PLATE
    assert corrupted is False


def test_corruption_only_uses_plausible_confusions():
    """Substitutions must come from the look-alike table; random characters would
    make the simulation easier than reality in the way that matters (a wrong
    character that looks nothing like the right one is trivially outvoted)."""
    model = NoiseModel(substitute_rate=1.0, drop_char_rate=0, insert_char_rate=0)
    from modules.ocr_temporal_eval import CONFUSIONS

    text, corrupted = corrupt(PLATE, model, random.Random(3))
    assert corrupted is True
    assert len(text) == len(PLATE)
    for original, got in zip(PLATE, text):
        assert got == original or got in CONFUSIONS.get(original, ())


def test_missed_frames_produce_no_text():
    model = NoiseModel(miss_frame_rate=1.0)
    reads = simulate_sequence(PLATE, 5, model, random.Random(0))
    assert all(r.text is None for r in reads)
    assert all(r.confidence == 0.0 for r in reads)


def test_policy_last_takes_the_final_readable_frame():
    reads = [FrameRead("AAA", 0.9), FrameRead(None, 0.0), FrameRead("ZZZ", 0.1)]
    text, conf, abstained = apply_policy("last", reads)
    assert (text, conf, abstained) == ("ZZZ", 0.1, False)


def test_policy_best_conf_takes_the_highest_scoring_frame():
    reads = [FrameRead("AAA", 0.9), FrameRead("ZZZ", 0.1)]
    text, _conf, abstained = apply_policy("best_conf", reads)
    assert text == "AAA"
    assert abstained is False


def test_policy_temporal_elects_the_majority_plate():
    reads = [FrameRead(PLATE, 0.9)] * 4 + [FrameRead("XX99ZZ0000", 0.3)]
    text, _conf, abstained = apply_policy("temporal", reads)
    assert text == PLATE
    assert abstained is False


def test_policy_temporal_abstains_on_three_way_disagreement():
    """Real disagreement must produce an abstention, not a guess — the whole
    safety argument is that the pipeline declines rather than inventing a
    plate."""
    reads = [FrameRead("AB12CD3456", 0.4), FrameRead("ZZ99YY8888", 0.4),
             FrameRead("MH01XY1111", 0.4)]
    text, conf, abstained = apply_policy("temporal", reads)
    assert abstained is True
    assert text == ""
    assert conf < 0.35  # below PlateConfig.min_confidence, hence the abstention


def test_policy_temporal_abstains_on_structurally_invalid_readings():
    reads = [FrameRead("@@##", 0.4), FrameRead("!!!!", 0.4)]
    _text, _conf, abstained = apply_policy("temporal", reads)
    assert abstained is True


def test_two_way_tie_abstains():
    """An exact two-way split between structurally valid plates is no evidence
    for either. It used to elect one by string order (agreement 0.5 cleared the
    old 0.35 threshold) — a coin flip with a fine on it. It now abstains, both
    because 0.5 no longer clears the selected majority threshold and because a
    tie is refused outright (reason "contested")."""
    from modules.plate_recognizer import PlateConfig, PlateStabilizer

    reads = [FrameRead("AB12CD3456", 0.4), FrameRead("ZZ99YY8888", 0.4)] * 2
    text, conf, abstained = apply_policy("temporal", reads)
    assert abstained is True and text == ""
    assert conf == pytest.approx(0.5)
    # The tie rule holds on its own, even with the old permissive thresholds.
    stab = PlateStabilizer(PlateConfig(min_confidence=0.35, min_observations=2,
                                       min_support=1, min_margin=0.0))
    for r in reads:
        stab.add(r.text, r.confidence)
    res = stab.result()
    assert res.stable is None and res.abstain_reason == "contested"
    assert res.runner_up is not None and res.margin == 0.0


def test_a_clear_majority_beats_a_single_dissenter():
    reads = [FrameRead("AB12CD3456", 0.4)] * 3 + [FrameRead("ZZ99YY8888", 0.4)]
    text, conf, abstained = apply_policy("temporal", reads)
    assert (text, abstained) == ("AB12CD3456", False)
    assert conf > 0.6


def test_policy_temporal_abstains_on_a_single_observation():
    text, _conf, abstained = apply_policy("temporal", [FrameRead(PLATE, 0.99)])
    assert abstained is True
    assert text == ""


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        apply_policy("telepathy", [FrameRead(PLATE, 0.9)])


def test_all_policies_are_scored_on_identical_inputs():
    pairs = [(PLATE, [FrameRead(PLATE, 0.9)] * 4)]
    report = compare_policies(pairs)
    assert set(report.policies) == set(POLICIES)
    assert report.n_sequences == 1


def test_answered_accuracy_excludes_abstentions():
    """A policy that answers half the time and is right every time it answers
    must score 1.0 on answered accuracy and 0.5 on raw normalized match."""
    report = ComparisonReport(n_sequences=2, frames_per_sequence=1)
    from modules.ocr_temporal_eval import PolicyResult

    r = PolicyResult(
        policy="temporal", metrics={"normalized_match": 0.5}, abstain_rate=0.5,
    )
    assert r.answered_accuracy == pytest.approx(1.0)
    d = r.as_dict()
    assert d["coverage"] == 0.5
    assert d["answered_accuracy"] == 1.0
    assert report.temporal_gain == 0.0  # no policies recorded yet


def test_answered_accuracy_of_a_total_abstainer_is_zero():
    from modules.ocr_temporal_eval import PolicyResult

    r = PolicyResult("temporal", {"normalized_match": 0.0}, abstain_rate=1.0)
    assert r.answered_accuracy == 0.0


def test_temporal_is_more_precise_than_single_frame_under_noise():
    """The project's central OCR claim, as a regression test.

    Stated as precision *when the policy answers*, because that is the quantity
    that decides whether an innocent rider is fined; the coverage cost is
    asserted separately below so a regression in either direction is caught.
    """
    d = run_simulation(frames=10, repeats=5, seed=1234).as_dict()
    temporal = d["policies"]["temporal"]
    best_single = d["policies"]["best_conf"]
    last = d["policies"]["last"]

    assert temporal["answered_accuracy"] > best_single["answered_accuracy"]
    assert best_single["answered_accuracy"] > last["answered_accuracy"]
    # ...and it pays for that precision in coverage, which must stay visible.
    assert temporal["coverage"] < 1.0
    assert last["coverage"] == 1.0


def test_temporal_never_answers_with_a_structurally_invalid_plate():
    """``invalid_rate`` counts an abstention ("") as an invalid output, so a
    more cautious stabilizer raises it. The property that matters is that
    every temporal non-answer is an abstention — it never NAMES an invalid
    plate — while single-frame policies do."""
    d = run_simulation(frames=10, repeats=5, seed=1234).as_dict()
    t = d["policies"]["temporal"]
    assert t["invalid_rate"] == pytest.approx(t["abstain_rate"])
    assert d["policies"]["last"]["invalid_rate"] > 0.0


def test_noise_sweep_covers_every_rate_and_stays_honest():
    out = noise_sweep(rates=(0.05, 0.25), frames=6, repeats=2)
    assert set(out["sweep"]) == {"0.05", "0.25"}
    assert "not field data" in out["note"]
    # Higher noise must not make any policy *better* at answering correctly.
    low = out["sweep"]["0.05"]["policies"]["best_conf"]["answered_accuracy"]
    high = out["sweep"]["0.25"]["policies"]["best_conf"]["answered_accuracy"]
    assert high < low


def test_empty_sequence_set_does_not_divide_by_zero():
    report = compare_policies([])
    assert report.n_sequences == 0
    for policy in POLICIES:
        assert report.policies[policy]["abstain_rate"] == 0.0


# -- stabilizer threshold selection -------------------------------------------

def test_systematic_misread_creates_a_consistent_competitor():
    import random

    from modules.ocr_temporal_eval import NoiseModel, simulate_sequence

    model = NoiseModel(substitute_rate=0.0, drop_char_rate=0.0, insert_char_rate=0.0,
                       miss_frame_rate=0.0, systematic_rate=1.0, systematic_strength=1.0)
    reads = simulate_sequence("MH12AB1234", 6, model, random.Random(0))
    texts = {r.text for r in reads}
    assert len(texts) == 1 and texts != {"MH12AB1234"}  # the SAME wrong read


def test_independent_noise_model_is_unchanged_when_systematic_is_off():
    import random

    from modules.ocr_temporal_eval import NoiseModel, simulate_sequence

    a = simulate_sequence("MH12AB1234", 8, NoiseModel(), random.Random(3))
    b = simulate_sequence("MH12AB1234", 8, NoiseModel(systematic_rate=0.0), random.Random(3))
    assert [(r.text, r.confidence) for r in a] == [(r.text, r.confidence) for r in b]


def test_stabilizer_selection_protocol():
    from modules.ocr_temporal_eval import select_stabilizer, stabilizer_experiment

    exp = stabilizer_experiment(repeats=1, frames=6)
    assert exp["dev_seed"] != exp["test_seed"]
    sel = exp["selection"]
    assert sel["selected"] in {t["config"] for t in sel["table"]}
    # Every reported test-seed row is the selected or the previous default.
    for rows in exp["test"].values():
        assert set(rows) <= {sel["selected"], exp["previous_default"]}
    # Exact ties go to the more conservative config.
    res = {c: {n: {"coverage": 0.5, "wrong_plate_rate": 0.0}
               for n in ("support>=1, agreement>=0.5, margin>=0.0",
                         "support>=2, agreement>=0.5, margin>=0.2")}
           for c in ("sub=0.04,sys=0.0",)}
    sel = select_stabilizer(res, design_rates=(0.04,), design_systematic=(0.0,))
    assert sel["selected"] == "support>=2, agreement>=0.5, margin>=0.2"


def test_stabilizer_is_scored_at_the_moment_the_pipeline_commits():
    """A wrong plate elected early and corrected later still counts as wrong:
    the pipeline fines a held violation at the FIRST election."""
    from modules.ocr_temporal_eval import FrameRead, _stabilizer_outcomes
    from modules.plate_recognizer import PlateConfig

    cfg = PlateConfig(min_observations=2, min_support=1, min_confidence=0.35, min_margin=0.0)
    reads = [FrameRead("MH12AB1284", 0.9), FrameRead("MH12AB1284", 0.9)] + \
        [FrameRead("MH12AB1234", 0.9)] * 8
    out = _stabilizer_outcomes([("MH12AB1234", reads)], cfg)
    assert out["wrong_plate_rate"] == 1.0  # committed after 2 reads, then outvoted


def test_one_valid_read_among_junk_cannot_elect_a_plate():
    from modules.plate_recognizer import PlateStabilizer

    stab = PlateStabilizer()
    for text in ("MH12AB1284", "QQQQ", "12AB1Z3"):
        stab.add(text, 0.9)
    res = stab.result()
    assert res.stable is None and res.abstain_reason == "too_few_supporting_reads"
