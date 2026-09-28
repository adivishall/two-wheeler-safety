"""The temporal-confirmation experiment and the rules it chooses between."""

import random

import pytest

from modules.temporal_eval import (
    DetectorNoise,
    Rule,
    candidate_rules,
    noise_from_confusion,
    run_experiment,
    score_rule,
    select_rule,
    simulate_population,
    simulate_rider,
    sweep,
)
from modules.violation_state import (
    HelmetConfig,
    HelmetStateMachine,
    TripleConfig,
    TripleRidingStateMachine,
)

CLEAN = DetectorNoise("clean", 0.0, 0.0, 0.0, 0.0)


def _nh(sm, f, conf=0.9):
    return sm.update(has_helmet=False, has_no_helmet=True, no_helmet_conf=conf,
                     ambiguous=False, frame_idx=f)


def _h(sm, f):
    return sm.update(has_helmet=True, has_no_helmet=False, no_helmet_conf=0.0,
                     ambiguous=False, frame_idx=f)


def _blank(sm, f):
    return sm.update(has_helmet=False, has_no_helmet=False, no_helmet_conf=0.0,
                     ambiguous=False, frame_idx=f)


# -- the rules ----------------------------------------------------------------

def test_k_of_n_survives_a_flip_that_resets_a_streak():
    streak = HelmetStateMachine(HelmetConfig(confirm_window=3))
    vote = HelmetStateMachine(HelmetConfig(confirm_window=3, vote_window=5))
    for f, step in enumerate([_nh, _nh, _h, _nh]):
        step(streak, f)
        step(vote, f)
    assert not streak.confirmed  # the helmet frame reset it
    assert vote.confirmed  # 3 of the last 4 observed frames


def test_k_of_n_ignores_blank_frames_and_counts_ambiguous_as_no():
    sm = HelmetStateMachine(HelmetConfig(confirm_window=2, vote_window=4))
    _nh(sm, 0)
    for f in range(1, 10):
        _blank(sm, f)  # not evidence either way
    sm.update(has_helmet=True, has_no_helmet=True, no_helmet_conf=0.9,
              ambiguous=True, frame_idx=10)
    assert not sm.confirmed and sm.candidate_frames == 1
    _nh(sm, 11)
    assert sm.confirmed


def test_fraction_gate_blocks_a_lucky_streak_on_a_mostly_helmeted_rider():
    gated = HelmetStateMachine(HelmetConfig(confirm_window=3, min_observed=12,
                                            min_fraction=0.7))
    plain = HelmetStateMachine(HelmetConfig(confirm_window=3))
    seq = [_h] * 20 + [_nh] * 3  # 3 flips in a row after 20 helmet frames
    for f, step in enumerate(seq):
        step(gated, f)
        step(plain, f)
    assert plain.confirmed and not gated.confirmed
    assert gated.observed_frames == 23


def test_fraction_gate_needs_enough_observed_frames():
    sm = HelmetStateMachine(HelmetConfig(confirm_window=3, min_observed=12, min_fraction=0.7))
    for f in range(11):
        _nh(sm, f)
    assert not sm.confirmed  # 11 < 12 observed, however consistent
    _nh(sm, 11)
    assert sm.confirmed


def test_triple_k_of_n_uses_observed_frames():
    sm = TripleRidingStateMachine(TripleConfig(confirm_window=3, vote_window=5))
    sm.update(has_triple=True, conf=0.9, frame_idx=0, observed=True)
    sm.update(has_triple=False, conf=0.0, frame_idx=1, observed=True)
    sm.update(has_triple=True, conf=0.9, frame_idx=2, observed=True)
    sm.update(has_triple=False, conf=0.0, frame_idx=3, observed=False)  # blank
    assert not sm.confirmed
    sm.update(has_triple=True, conf=0.9, frame_idx=4, observed=True)
    assert sm.confirmed


# -- the experiment -----------------------------------------------------------

def test_noise_rates_come_from_the_confusion_matrix():
    names = ["Plate", "WithHelmet", "WithoutHelmet", "TripleRiding"]
    cm = [[10, 0, 0, 0, 0],
          [0, 18, 4, 0, 4],   # helmeted: 4/26 called no-helmet, 4/26 missed
          [0, 16, 150, 0, 42],
          [0, 0, 0, 9, 1],
          [0, 0, 0, 0, 0]]
    n = noise_from_confusion(cm, names, "t")
    assert n.false_violation == pytest.approx(4 / 26, abs=1e-4)
    assert n.miss_compliant == pytest.approx(4 / 26, abs=1e-4)
    assert n.false_compliant == pytest.approx(16 / 208, abs=1e-4)


def test_simulated_rider_keeps_ground_truth_fixed():
    sc = simulate_rider(1, True, 20, CLEAN, random.Random(0), plate="MH12AB1234")
    assert len(sc.frames) == 20
    assert all(any(d.label == "WithoutHelmet" for d in fr[0].dets) for fr in sc.frames)
    assert "no_helmet" in sc.vehicles[0].violations


def test_every_rule_is_perfect_without_noise():
    pop = simulate_population(CLEAN, dwell=25, n_per_class=5, seed=0)
    for rule in (Rule("c5", 5), Rule("c5+gate", 5, None, 12, 0.7)):
        m = score_rule(rule, pop)
        assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["false_flag_rate"] == 0.0


def test_a_rider_seen_for_fewer_frames_than_the_gate_is_never_flagged():
    pop = simulate_population(CLEAN, dwell=8, n_per_class=5, seed=0)
    m = score_rule(Rule("gate", 3, None, 12, 0.7), pop)
    assert m["recall"] == 0.0 and m["false_flag_rate"] == 0.0


def test_selection_prefers_feasible_rules_and_falls_back_honestly():
    rules = [Rule("r1", 1), Rule("r2", 2)]
    res = {"rho=0,dwell=10": {
        "r1": {"recall": 1.0, "false_flag_rate": 0.5},
        "r2": {"recall": 0.8, "false_flag_rate": 0.0}}}
    sel = select_rule(res, rules, design_rho=(0.0,))
    assert sel["selected"] == "r2" and sel["any_feasible"]
    res["rho=0,dwell=10"]["r2"]["false_flag_rate"] = 0.2
    sel = select_rule(res, rules, design_rho=(0.0,))
    assert sel["selected"] == "r2" and not sel["any_feasible"]
    assert "fallback" in sel["selected_by"]


def test_sweep_and_experiment_are_deterministic_and_separate_dev_from_test():
    rules = candidate_rules()[:2]
    a = sweep(rules, CLEAN, seed=1, n_per_class=2, dwells=(12,), stickiness=(0.0,))
    b = sweep(rules, CLEAN, seed=1, n_per_class=2, dwells=(12,), stickiness=(0.0,))
    assert a == b
    out = run_experiment(n_per_class=2, dev_seed=1, test_seed=2,
                         noise=DetectorNoise("x", 0.1, 0.1, 0.1, 0.05))
    assert out["dev_seed"] != out["test_seed"]
    assert out["selection"]["selected"] in out["test"][next(iter(out["test"]))]
    assert "unmeasured" in out["caveat"]
