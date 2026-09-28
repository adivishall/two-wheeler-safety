"""Temporal confirmation: which rule and which window — measured, not guessed.

A violation is only fined after it "persists". How it must persist is a real
decision with a real trade-off: a strict streak absorbs detector flicker (good
for precision) but one flipped frame resets it (bad for recall), and the size
of both effects depends on the detector's error rates and on how long a
vehicle stays in view. The old suite answered this with hand-written scenarios
whose flickers were exactly one frame long, so "window 2 fixes everything" was
true by construction.

This module replaces that with a small generative experiment:

* **Detector-noise model.** Each frame, a rider's box is missed, labelled
  correctly, or labelled as the *other* helmet class. The rates default to what
  the shipped detector actually does on the **validation** split (its
  confusion matrix), never the test split. ``stickiness`` makes errors
  temporally correlated — consecutive video frames share pose and lighting, so
  real errors come in runs — and is the one parameter no labelled video exists
  to measure, so it is swept rather than assumed.
* **Rules under test.** The shipped state machines, driven through the shipped
  :class:`~modules.pipeline.ViolationPipeline`: consecutive-k, and k-of-n
  voting over observed frames.
* **Protocol.** Every rule sees the *same* simulated detections (common random
  numbers). The rule is **selected on a development seed** under a stated
  objective — highest recall subject to a false-flag rate of at most 1% of
  compliant riders — and then **reported on a separate seed**, so the reported
  number is not the one the choice was tuned on.

What it cannot tell you: real temporal-error statistics. Those need labelled
video, which this project does not have. The ``stickiness`` sweep shows how
the conclusion depends on that unknown instead of hiding it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from modules.association import DetBox
from modules.pipeline import PipelineConfig
from modules.system_eval import GTVehicle, Scenario, VehicleFrame, run_scenario

VIOLATOR_LABEL = "WithoutHelmet"
COMPLIANT_LABEL = "WithHelmet"


@dataclass(frozen=True)
class DetectorNoise:
    """Per-frame helmet-classification error model for one rider."""

    name: str
    miss_violator: float  # P(no rider box) for a bare-headed rider
    miss_compliant: float  # P(no rider box) for a helmeted rider
    false_violation: float  # P(helmeted rider labelled WithoutHelmet)
    false_compliant: float  # P(bare-headed rider labelled WithHelmet)
    stickiness: float = 0.0  # P(a frame repeats the previous frame's outcome)

    def with_stickiness(self, rho: float) -> DetectorNoise:
        return replace(self, stickiness=rho, name=f"{self.name}, stickiness {rho:g}")

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def noise_from_confusion(confusion: list, class_names: list, name: str) -> DetectorNoise:
    """Derive helmet error rates from an evaluate_model confusion matrix
    (rows = ground truth, last column = missed)."""
    h = class_names.index(COMPLIANT_LABEL)
    v = class_names.index(VIOLATOR_LABEL)
    miss_col = len(confusion[0]) - 1

    def rate(row, col):
        n = sum(confusion[row])
        return confusion[row][col] / n if n else 0.0

    return DetectorNoise(
        name=name,
        miss_violator=round(rate(v, miss_col), 4),
        miss_compliant=round(rate(h, miss_col), 4),
        false_violation=round(rate(h, v), 4),
        false_compliant=round(rate(v, h), 4),
    )


# The shipped detector on the de-leaked VALIDATION split, at the operating
# threshold (eval/results/eval_traffic_model-2_val_clean.json, confusion rows
# WithHelmet / WithoutHelmet). evaluate_temporal.py re-derives it from that
# file (--noise-from); this constant is the same numbers for model-free callers.
MEASURED_VAL_NOISE = DetectorNoise(
    name="measured (val, de-leaked)",
    miss_violator=0.201,
    miss_compliant=0.1538,
    false_violation=0.1538,
    false_compliant=0.0766,
)


@dataclass(frozen=True)
class Rule:
    name: str
    confirm_window: int
    vote_window: int | None = None
    min_observed: int = 0
    min_fraction: float = 0.0
    family: str = "streak"

    def config(self) -> PipelineConfig:
        return PipelineConfig(helmet_confirm_window=self.confirm_window,
                              vote_window=self.vote_window,
                              helmet_min_observed=self.min_observed,
                              helmet_min_fraction=self.min_fraction)


SHIPPED_BEFORE = Rule("consecutive 5 (pre-experiment default)", 5)


def candidate_rules() -> list[Rule]:
    """Iteration 1: streaks and k-of-n windows. Iteration 2 (added after
    iteration 1 found no feasible rule): a streak plus a track-level gate."""
    rules = [Rule(f"consecutive {k}", k) for k in (1, 2, 3, 4, 5, 6, 8)]
    for k in (2, 3, 4, 5, 6):
        for n in sorted({k + 2, 2 * k}):
            rules.append(Rule(f"{k} of {n}", k, n, family="k_of_n"))
    for k in (3, 5):
        for m in (8, 12):
            for theta in (0.5, 0.6, 0.7):
                rules.append(Rule(f"consecutive {k} + >={theta:.0%} of >={m} observed",
                                  k, None, m, theta, family="streak+fraction"))
    return rules


def simulate_rider(
    gt_id: int, violator: bool, dwell: int, noise: DetectorNoise, rng: random.Random,
    plate: str,
) -> Scenario:
    """One rider crossing the frame for ``dwell`` frames. The plate is always
    detected and readable, so the only thing under test is the helmet decision."""
    miss = noise.miss_violator if violator else noise.miss_compliant
    flip = noise.false_compliant if violator else noise.false_violation
    right = VIOLATOR_LABEL if violator else COMPLIANT_LABEL
    wrong = COMPLIANT_LABEL if violator else VIOLATOR_LABEL
    frames = []
    prev = None
    for f in range(dwell):
        if prev is not None and rng.random() < noise.stickiness:
            outcome = prev
        else:
            r = rng.random()
            outcome = "miss" if r < miss else "flip" if r < miss + flip else "ok"
        prev = outcome
        x = 100 + 6 * f
        dets = [DetBox("Plate", (x + 10, 205, x + 50, 235), 0.9)]
        if outcome != "miss":
            dets.append(DetBox(right if outcome == "ok" else wrong, (x, 100, x + 60, 200), 0.8))
        frames.append([VehicleFrame(gt_id, dets, plate)])
    vios = frozenset({"no_helmet"}) if violator else frozenset()
    return Scenario(f"rider_{gt_id}", [GTVehicle(gt_id, plate, vios)], frames)


def simulate_population(
    noise: DetectorNoise, *, dwell: int, n_per_class: int, seed: int,
) -> list[Scenario]:
    rng = random.Random(seed)
    out = []
    for i in range(n_per_class * 2):
        out.append(simulate_rider(i, violator=i % 2 == 0, dwell=dwell, noise=noise,
                                  rng=rng, plate="MH12AB1234"))
    return out


def score_rule(rule: Rule, scenarios: list[Scenario]) -> dict:
    """Vehicle-level no-helmet decision quality for one rule."""
    tp = fp = fn = tn = 0
    cfg = rule.config()
    for sc in scenarios:
        pipe, _fines, votes = run_scenario(sc, cfg)
        decided = any("no_helmet" in pipe.confirmed_violations(t) for t in votes)
        truth = "no_helmet" in sc.vehicles[0].violations
        if truth and decided:
            tp += 1
        elif truth:
            fn += 1
        elif decided:
            fp += 1
        else:
            tn += 1
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        # Innocent riders flagged, per compliant rider: the number that matters
        # most for a system that fines people.
        "false_flag_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


DWELLS = (10, 25, 50)  # frames in view: 0.4 s / 1 s / 2 s at 25 fps
STICKINESS = (0.0, 0.5, 0.8)
MAX_FALSE_FLAG = 0.01


def sweep(
    rules: list[Rule], base_noise: DetectorNoise, *, seed: int, n_per_class: int,
    dwells=DWELLS, stickiness=STICKINESS,
) -> dict:
    """Score every rule on every (stickiness, dwell) condition for one seed."""
    out: dict = {}
    for rho in stickiness:
        noise = base_noise.with_stickiness(rho)
        for dwell in dwells:
            pop = simulate_population(noise, dwell=dwell, n_per_class=n_per_class,
                                      seed=seed + int(rho * 100) + dwell)
            key = f"rho={rho:g},dwell={dwell}"
            out[key] = {r.name: score_rule(r, pop) for r in rules}
    return out


def select_rule(results: dict, rules: list[Rule], *, design_rho=(0.0, 0.5),
                max_false_flag: float = MAX_FALSE_FLAG) -> dict:
    """Highest mean recall over the design conditions, subject to the false-flag
    rate staying <= ``max_false_flag`` in EVERY design condition."""
    design_keys = [k for k in results
                   if float(k.split(",")[0].split("=")[1]) in design_rho]
    table = []
    for r in rules:
        rows = [results[k][r.name] for k in design_keys]
        worst_ff = max(x["false_flag_rate"] for x in rows)
        mean_recall = sum(x["recall"] for x in rows) / len(rows)
        table.append({"rule": r.name, "mean_recall": round(mean_recall, 4),
                      "worst_false_flag_rate": round(worst_ff, 4),
                      "feasible": worst_ff <= max_false_flag})
    feasible = [t for t in table if t["feasible"]]
    # Ties on recall go to the rule needing fewer supporting frames, then the
    # smaller window — the less machinery, the better.
    order = {r.name: (r.confirm_window, r.vote_window or r.confirm_window) for r in rules}
    if feasible:
        best = max(feasible, key=lambda t: (t["mean_recall"], -order[t["rule"]][0],
                                            -order[t["rule"]][1]))
        how = "primary objective"
    else:
        # Fallback, added after iteration 1 found no feasible rule: when the
        # detector is too noisy for ANY rule to meet the target, protect the
        # innocent first — lowest worst-case false-flag rate, then recall.
        # (The candidate grid contains no degenerate "never fine" rule, so this
        # cannot select one.)
        best = min(table, key=lambda t: (t["worst_false_flag_rate"], -t["mean_recall"]))
        how = "fallback: no rule met the target; minimum worst-case false-flag rate"
    return {"design_conditions": design_keys, "objective": (
        f"max mean recall s.t. false-flag rate <= {max_false_flag:.0%} in every "
        "design condition; if none qualifies, minimise the worst-case false-flag "
        "rate (ties -> recall)"), "table": table, "selected": best["rule"],
        "selected_by": how, "any_feasible": bool(feasible)}


def run_experiment(*, dev_seed: int = 101, test_seed: int = 202, n_per_class: int = 150,
                   noise: DetectorNoise = MEASURED_VAL_NOISE) -> dict:
    rules = candidate_rules()
    dev = sweep(rules, noise, seed=dev_seed, n_per_class=n_per_class)
    choice = select_rule(dev, rules)
    by_name = {r.name: r for r in rules}
    report_rules = [SHIPPED_BEFORE, by_name["consecutive 1"]]
    if choice["selected"] not in {r.name for r in report_rules}:
        report_rules.append(by_name[choice["selected"]])
    test = sweep(report_rules, noise, seed=test_seed, n_per_class=n_per_class)
    return {
        "noise_model": noise.as_dict(),
        "dwells_frames": list(DWELLS),
        "stickiness": list(STICKINESS),
        "n_riders_per_condition": 2 * n_per_class,
        "dev_seed": dev_seed,
        "test_seed": test_seed,
        "selection": choice,
        "dev": dev,
        "test": test,
        "caveat": (
            "Simulated detector noise at rates measured on the validation split. "
            "Temporal correlation of real errors is unmeasured (no labelled "
            "video); stickiness is swept to show the dependence. Plates are "
            "always readable here, so this isolates the helmet decision."
        ),
    }
