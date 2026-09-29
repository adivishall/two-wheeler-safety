"""Recipe comparison: the seed term is part of the uncertainty, too few seeds
is never a verdict, and a class regression vetoes a promotion."""

import numpy as np

from modules.detection_stats import match_for_ap, recipe_comparison

NAMES = ["Plate", "WithHelmet", "WithoutHelmet", "TripleRiding"]


def _run(hit_rates, n=120, seed=0):
    """One checkpoint's records: per class, a hit with probability hit_rates[c]."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        gt, preds = [], []
        for c, p in enumerate(hit_rates):
            box = (10 + 60 * c, 10, 60 + 60 * c, 60)
            gt.append((c, box))
            hit = rng.random() < p
            preds.append((c, box if hit else (400, 400, 450, 450), float(rng.uniform(0.3, 1))))
        recs.append(match_for_ap(gt, preds, 4))
    return recs


def test_one_seed_per_recipe_is_never_a_verdict():
    rep = recipe_comparison({"base": [_run([0.8] * 4, seed=1)],
                             "cand": [_run([0.95] * 4, seed=2)]}, "base", NAMES, n_boot=200)
    assert rep["comparisons"]["cand"]["verdict"].startswith("insufficient seeds")
    assert rep["recipes"]["base"]["per_class"]["Plate"]["between_seed_sd"] is None


def test_a_clear_gain_across_seeds_is_promoted():
    base = [_run([0.6] * 4, seed=s) for s in range(3)]
    cand = [_run([0.95] * 4, seed=10 + s) for s in range(3)]
    rep = recipe_comparison({"base": base, "cand": cand}, "base", NAMES, n_boot=300)
    assert rep["comparisons"]["cand"]["verdict"] == "promote"
    assert rep["recipes"]["cand"]["seeds"] == 3


def test_a_class_regression_vetoes_a_better_average():
    base = [_run([0.6, 0.6, 0.6, 0.95], seed=s) for s in range(3)]
    cand = [_run([0.95, 0.95, 0.95, 0.4], seed=10 + s) for s in range(3)]
    rep = recipe_comparison({"base": base, "cand": cand}, "base", NAMES, n_boot=300)
    c = rep["comparisons"]["cand"]
    assert c["map50"]["diff"] > 0
    assert c["verdict"] == "reject: significantly worse on TripleRiding"


def test_seed_spread_widens_the_interval():
    """Same image-level evidence, but seeds that disagree: the recipe CI must
    be wider than when the seeds agree."""
    base = [_run([0.7] * 4, seed=s) for s in range(3)]
    agree = [_run([0.8] * 4, seed=20 + s) for s in range(3)]
    spread = [_run([p] * 4, seed=30 + s) for s, p in enumerate((0.55, 0.8, 0.98))]
    width = {}
    for name, cand in (("agree", agree), ("spread", spread)):
        m = recipe_comparison({"base": base, name: cand}, "base", NAMES,
                              n_boot=300)["comparisons"][name]["map50"]
        width[name] = m["ci_high"] - m["ci_low"]
    assert width["spread"] > width["agree"]


def test_t_quantiles_widen_for_few_seeds():
    from modules.detection_stats import _t_quantile

    assert _t_quantile(0.95, 1) == 12.706 and _t_quantile(0.95, 2) == 4.303
    assert abs(_t_quantile(0.95, 10) - 2.228) < 0.01
    assert abs(_t_quantile(0.95, float("inf")) - 1.96) < 0.001


def test_identical_recipes_are_not_promoted():
    """Three seeds each of the same recipe (seed-to-seed spread 0.06): the old
    seed-resampling interval promoted such pairs ~2x as often as nominal."""
    rng = np.random.default_rng(7)
    promoted = 0
    for t in range(6):
        runs = {name: [_run(np.clip(0.7 + rng.normal(0, 0.06, 4), 0.05, 0.99),
                            n=60, seed=100 * t + 10 * i + j)
                       for j in range(3)] for i, name in enumerate(("a", "b"))}
        rep = recipe_comparison(runs, "a", NAMES, n_boot=100, seed=t)
        promoted += rep["comparisons"]["b"]["verdict"] == "promote"
    assert promoted == 0
