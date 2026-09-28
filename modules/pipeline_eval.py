"""Per-violation pipeline evaluation and an error budget for the whole system.

`modules/system_eval.py` scores the *fines* the pipeline emits. This module asks
two narrower questions it cannot answer:

**Phases 10-11 — how good is each violation decision on its own?**
:func:`evaluate_violation` drives detections through the real tracker,
association and state machines and scores the final per-vehicle
``no_helmet`` / ``triple_riding`` decision as a binary classification
(precision / recall / F1 / FP / FN). This is deliberately *not* raw YOLO
accuracy: it answers "how often does the complete pipeline make the correct
helmet call for this vehicle?", which is the decision a rider is actually fined
on. :func:`sweep_confirm_window` then measures what the temporal confirmation
window buys, so ``confirm_window`` is a tuned parameter rather than a guess.

**Phase 15 — where do system failures actually come from?**
:func:`error_budget` runs the same scenario suite repeatedly, each time injecting
*one* stage's characteristic failure (detector misses, detector class flips, OCR
character noise, plate dropout) at a controlled rate, and measures the resulting
drop in end-to-end fines. The output is an attribution: which stage, when
degraded by a fixed amount, costs the system the most. That is a measured
sensitivity, not a claim about how often each stage fails in the field — the
distinction is stated in the report and must stay stated.

All inputs are deterministic synthetic scenarios and a seeded RNG, so every
number here is reproducible and needs no model weights.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from modules.association import DetBox
from modules.pipeline import DEFAULT_PIPELINE_CONFIG, PipelineConfig
from modules.system_eval import (
    DWELL,
    GTVehicle,
    Scenario,
    VehicleFrame,
    evaluate_system,
    majority_gt,
    run_scenario,
)

SHIPPED_WINDOW = DEFAULT_PIPELINE_CONFIG.confirm_window

VIOLATIONS = ("no_helmet", "triple_riding")


# ---------------------------------------------------------------------------
# Phases 10 & 11 — per-violation pipeline decision quality
# ---------------------------------------------------------------------------

@dataclass
class DecisionMetrics:
    """Binary classification metrics for one violation type over one suite."""

    violation: str
    confirm_window: int = SHIPPED_WINDOW
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 1.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d.update({
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        })
        return d


def run_pipeline_decisions(
    scenario: Scenario, *, confirm_window: int = SHIPPED_WINDOW,
    config: PipelineConfig | None = None,
) -> dict[int, set]:
    """Drive the shipped pipeline and return ``{gt_id: {confirmed violations}}``.

    This scores the *violation decision* (did the state machine confirm), not
    the fine, so a vehicle whose plate never stabilised still counts as a
    correct helmet call. Tracks are mapped back to ground-truth vehicles by
    majority overlap vote across the clip (the same attribution ``system_eval``
    uses), so an ID switch mid-clip does not by itself corrupt the decision
    being scored — which is the point: the pipeline is allowed to lose a track
    id and still be right about the vehicle.
    """
    pipe, _fines, track_gt_votes = run_scenario(
        scenario, config or PipelineConfig(confirm_window=confirm_window))
    decisions: dict[int, set] = {v.gt_id: set() for v in scenario.vehicles}
    for tid, votes in track_gt_votes.items():
        gt_id = majority_gt(votes)
        if gt_id is None or gt_id not in decisions:
            continue
        decisions[gt_id] |= pipe.confirmed_violations(tid)
    return decisions


def evaluate_violation(
    scenarios: list, violation: str, *, confirm_window: int = SHIPPED_WINDOW,
    config: PipelineConfig | None = None,
) -> DecisionMetrics:
    """Score the pipeline's final decision for one violation across scenarios."""
    m = DecisionMetrics(violation=violation, confirm_window=confirm_window)
    for sc in scenarios:
        decisions = run_pipeline_decisions(sc, confirm_window=confirm_window,
                                           config=config)
        for v in sc.vehicles:
            expected = violation in v.violations
            got = violation in decisions.get(v.gt_id, set())
            if expected and got:
                m.true_positives += 1
            elif expected and not got:
                m.false_negatives += 1
            elif not expected and got:
                m.false_positives += 1
            else:
                m.true_negatives += 1
    return m


def sweep_confirm_window(
    scenarios: list, violation: str, windows=(1, 2, 3, 5, 8),
) -> dict:
    """Precision/recall against the streak length ALONE (helmet dwell gate off).

    A window of 1 is "fine on a single frame" — the behaviour the temporal layer
    exists to replace. This isolates what the streak buys on the hand-built
    scenarios. It is a sanity check, not how the shipped rule was chosen: the
    scenarios' flickers are one frame long by construction, so "window 2 is
    enough" is built in. The rule was selected by ``evaluate_temporal.py``
    against measured detector error rates.
    """
    rows = {
        str(w): evaluate_violation(
            scenarios, violation, confirm_window=w,
            config=PipelineConfig(confirm_window=w, helmet_min_observed=0,
                                  helmet_min_fraction=0.0),
        ).as_dict()
        for w in windows
    }
    return {
        "violation": violation,
        "windows": rows,
        "note": (
            "Streak length alone (helmet dwell gate off). confirm_window=1 is "
            "single-frame fining. The scenarios contain one-frame flickers by "
            "construction; the shipped rule is chosen in "
            "eval/results/temporal_confirmation.md, not here."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 11 — triple-riding edge cases
# ---------------------------------------------------------------------------

def _rider_scenario(
    name: str, label: str, violations: set, *, n_frames: int = DWELL,
    plate: str = "MH12AB1234", drop_frames: tuple = (), occlude_frames: tuple = (),
) -> Scenario:
    """One bike carrying ``label`` for ``n_frames``, with optional detection loss.

    ``drop_frames`` removes *all* detections (the vehicle vanishes);
    ``occlude_frames`` keeps the body but removes the plate (a rider seen but
    not identifiable), which is a different and more common failure.
    """
    frames = []
    for f in range(n_frames):
        if f in drop_frames:
            frames.append([])
            continue
        body = (100 + 6 * f, 100, 160 + 6 * f, 200)
        dets = [DetBox(label, body, 0.9)]
        text = None
        if f not in occlude_frames:
            dets.append(DetBox("Plate", (110 + 6 * f, 205, 150 + 6 * f, 235), 0.9))
            text = plate
        frames.append([VehicleFrame(1, dets, text)])
    return Scenario(name, [GTVehicle(1, plate, frozenset(violations))], frames)


def triple_riding_scenarios() -> list:
    """The rider-count edge cases the spec asks for.

    An important honesty note encoded here: the detector has a single
    ``TripleRiding`` class, so "exactly three" and "four or more" are the *same*
    detection to this system. The four-up scenario is therefore scored as a
    ``triple_riding`` violation (which is correct enforcement behaviour — four-up
    is at least as illegal as three-up), not as a distinct rider count. The
    system cannot and does not claim to count riders.
    """
    return [
        # Two riders: legal. The detector emits WithoutHelmet (a helmet
        # violation) but never TripleRiding — the negative case that must not
        # produce a triple-riding fine.
        _rider_scenario("two_riders_legal", "WithoutHelmet", {"no_helmet"}),
        # Exactly three: the canonical positive.
        _rider_scenario("exactly_three", "TripleRiding", {"triple_riding"}),
        # Four-up: same class, must still fine.
        _rider_scenario("four_or_more", "TripleRiding", {"triple_riding"}),
        # Partial occlusion: plate hidden for 3 frames, rider still visible.
        _rider_scenario("triple_plate_occluded", "TripleRiding", {"triple_riding"},
                        occlude_frames=(10, 11, 12)),
        # Temporary total detection loss (the bike passes behind a bus).
        _rider_scenario("triple_detection_loss", "TripleRiding", {"triple_riding"},
                        drop_frames=(12, 13)),
        # A brief, non-persistent TripleRiding flicker must NOT confirm.
        _flicker_scenario("triple_flicker_only"),
    ]


def _flicker_scenario(name: str, *, n_frames: int = DWELL, flicker=(12,)) -> Scenario:
    """A clean two-up bike the detector briefly mislabels as TripleRiding.

    This is the single-frame false positive the temporal layer exists to absorb;
    a pipeline that fines here is fining on detector noise.
    """
    frames = []
    for f in range(n_frames):
        label = "TripleRiding" if f in flicker else "WithHelmet"
        body = (100 + 6 * f, 100, 160 + 6 * f, 200)
        frames.append([VehicleFrame(1, [
            DetBox(label, body, 0.9),
            DetBox("Plate", (110 + 6 * f, 205, 150 + 6 * f, 235), 0.9),
        ], "MH12AB1234")])
    return Scenario(name, [GTVehicle(1, "MH12AB1234", frozenset())], frames)


def helmet_scenarios() -> list:
    """Helmet-decision edge cases, including the contradiction safeguard."""
    return [
        _rider_scenario("no_helmet_clean", "WithoutHelmet", {"no_helmet"}),
        _rider_scenario("helmet_clean", "WithHelmet", set()),
        _rider_scenario("no_helmet_detection_loss", "WithoutHelmet", {"no_helmet"},
                        drop_frames=(12, 13)),
        _helmet_flicker_scenario("helmet_flicker_only"),
        _contradiction_scenario("helmet_contradiction"),
    ]


def _helmet_flicker_scenario(name: str, *, n_frames: int = DWELL, flicker=(12,)) -> Scenario:
    """A helmeted rider the detector briefly calls bare-headed: must not fine."""
    frames = []
    for f in range(n_frames):
        label = "WithoutHelmet" if f in flicker else "WithHelmet"
        body = (100 + 6 * f, 100, 160 + 6 * f, 200)
        frames.append([VehicleFrame(1, [
            DetBox(label, body, 0.9),
            DetBox("Plate", (110 + 6 * f, 205, 150 + 6 * f, 235), 0.9),
        ], "MH12AB1234")])
    return Scenario(name, [GTVehicle(1, "MH12AB1234", frozenset())], frames)


def _contradiction_scenario(name: str, *, n_frames: int = DWELL) -> Scenario:
    """Every frame asserts helmet AND no-helmet on one rider.

    The measured, reproduced model bug. The pipeline must treat it as ambiguous
    and fine nothing, so the expected violation set is empty.
    """
    frames = []
    for f in range(n_frames):
        body = (100 + 6 * f, 100, 160 + 6 * f, 200)
        frames.append([VehicleFrame(1, [
            DetBox("WithoutHelmet", body, 0.9),
            DetBox("WithHelmet", body, 0.85),
            DetBox("Plate", (110 + 6 * f, 205, 150 + 6 * f, 235), 0.9),
        ], "MH12AB1234")])
    return Scenario(name, [GTVehicle(1, "MH12AB1234", frozenset())], frames)


# ---------------------------------------------------------------------------
# Phase 15 — error budget via single-stage fault injection
# ---------------------------------------------------------------------------

@dataclass
class Injection:
    """One stage's characteristic failure, applied at a controlled rate."""

    stage: str
    description: str
    detection_drop: float = 0.0  # p(a detection box vanishes this frame)
    class_flip: float = 0.0  # p(a helmet/no-helmet box is flipped)
    plate_drop: float = 0.0  # p(the plate is not read this frame)
    ocr_noise: float = 0.0  # p(a character in the plate text is corrupted)


DEFAULT_INJECTIONS = (
    Injection("detection", "boxes vanish (detector recall failure)",
              detection_drop=0.30),
    Injection("detection_class", "helmet/no-helmet boxes flipped (class confusion)",
              class_flip=0.30),
    Injection("plate_recall", "plate not read on a frame (plate detection/OCR miss)",
              plate_drop=0.30),
    Injection("ocr", "characters corrupted in the plate text (OCR noise)",
              ocr_noise=0.30),
)

_FLIP = {"WithHelmet": "WithoutHelmet", "WithoutHelmet": "WithHelmet"}
_OCR_SWAP = {"0": "O", "O": "0", "1": "I", "I": "1", "5": "S", "S": "5",
             "8": "B", "B": "8", "2": "Z", "Z": "2", "6": "G", "G": "6"}


def apply_injection(scenario: Scenario, inj: Injection, rng: random.Random) -> Scenario:
    """Return a copy of ``scenario`` with one stage's failure injected.

    The ground-truth vehicles are untouched: degrading an input must never
    change what the *right* answer is, or the measurement would be meaningless.
    """
    frames = []
    for frame in scenario.frames:
        new_frame = []
        for vf in frame:
            dets = []
            for d in vf.dets:
                if rng.random() < inj.detection_drop:
                    continue
                label = d.label
                if label in _FLIP and rng.random() < inj.class_flip:
                    label = _FLIP[label]
                dets.append(DetBox(label, d.box, d.conf))
            text = vf.plate_text
            if text and rng.random() < inj.plate_drop:
                text = None
            elif text and inj.ocr_noise:
                text = "".join(
                    _OCR_SWAP.get(ch, ch) if rng.random() < inj.ocr_noise else ch
                    for ch in text
                )
            new_frame.append(VehicleFrame(vf.gt_id, dets, text))
        frames.append(new_frame)
    return Scenario(scenario.name, scenario.vehicles, frames)


# The shipped detector's per-frame error rates on the de-leaked VALIDATION
# split at the operating threshold (confusion matrix rows in
# eval/results/eval_traffic_model-2_val_clean.json). One unit for every stage:
# the probability that, on one frame, one vehicle's evidence from that stage is
# missing or wrong. OCR has no labelled field set, so its rate is SWEPT, never
# assumed.
MEASURED_OPERATING_POINT = {
    "rider_miss": {"WithHelmet": 0.1538, "WithoutHelmet": 0.201, "TripleRiding": 0.1034},
    "class_flip": {"WithHelmet": ("WithoutHelmet", 0.1538),
                   "WithoutHelmet": ("WithHelmet", 0.0766),
                   "TripleRiding": ("WithoutHelmet", 0.0517)},
    "plate_miss": 0.1822,
    "source": "eval/results/eval_traffic_model-2_val_clean.json (confusion matrix, conf 0.25)",
}
OCR_READ_ERROR_RATES = (0.1, 0.3, 0.5)  # P(a plate read has >= 1 wrong glyph); unmeasured

BUDGET_STAGES = (
    ("rider_recall", "detector misses the rider box"),
    ("helmet_class", "detector labels the rider with the wrong helmet/triple class"),
    ("plate_recall", "detector misses the plate box (nothing to OCR)"),
    ("ocr", "OCR reads the plate with a wrong glyph"),
)


def _misread(text: str, rng: random.Random) -> str:
    """One look-alike glyph error, the kind OCR actually makes on plates."""
    from modules.ocr_temporal_eval import CONFUSIONS

    idx = [i for i, ch in enumerate(text) if ch in CONFUSIONS]
    if not idx:
        return text
    i = rng.choice(idx)
    return text[:i] + rng.choice(CONFUSIONS[text[i]]) + text[i + 1:]


def apply_faults(scenario: Scenario, faults: dict, rng: random.Random) -> Scenario:
    """Degrade a scenario at per-vehicle, per-frame rates. Ground truth is
    untouched: degrading an input must never change the right answer."""
    miss = faults.get("rider_miss", {})
    flip = faults.get("class_flip", {})
    plate_miss = faults.get("plate_miss", 0.0)
    ocr = faults.get("ocr_corrupt", 0.0)
    frames = []
    for frame in scenario.frames:
        new_frame = []
        for vf in frame:
            dets = []
            text = vf.plate_text
            for d in vf.dets:
                if d.label == "Plate":
                    if rng.random() < plate_miss:
                        text = None
                        continue
                    dets.append(d)
                    continue
                if rng.random() < miss.get(d.label, 0.0):
                    continue
                label = d.label
                if label in flip and rng.random() < flip[label][1]:
                    label = flip[label][0]
                dets.append(DetBox(label, d.box, d.conf))
            if text and rng.random() < ocr:
                text = _misread(text, rng)
            new_frame.append(VehicleFrame(vf.gt_id, dets, text))
        frames.append(new_frame)
    return Scenario(scenario.name, scenario.vehicles, frames)


def _without(faults: dict, stage: str) -> dict:
    out = dict(faults)
    key = {"rider_recall": "rider_miss", "helmet_class": "class_flip",
           "plate_recall": "plate_miss", "ocr": "ocr_corrupt"}[stage]
    out[key] = {} if isinstance(faults.get(key), dict) else 0.0
    return out


def _mean_f1(scenarios, faults, trials, seed) -> float:
    rng = random.Random(seed)
    scores = [_suite_f1([apply_faults(sc, faults, rng) for sc in scenarios])
              for _ in range(trials)]
    return sum(scores) / len(scores)


def error_budget(
    scenarios: list, *, operating_point: dict | None = None,
    ocr_rates=OCR_READ_ERROR_RATES, trials: int = 20, seed: int = 7,
) -> dict:
    """Oracle ablation at the measured operating point.

    Every stage fails at its measured per-frame rate at once (OCR at each swept
    rate). Then each stage in turn is made perfect, and the end-to-end fine F1
    it recovers is that stage's share of the error. This answers "where would
    fixing things pay off, given how the system actually fails?" — which the
    old equal-rate injection could not: it compared 30% per-CHARACTER OCR
    corruption (a 10-glyph plate then almost never reads cleanly) with 30%
    per-BOX detector faults, so its "OCR is 77% of sensitivity" was a
    statement about units, not about the system.

    Stage shares are not strictly additive (stages interact); the report gives
    the ceiling gap alongside so the non-additivity is visible.
    """
    op = operating_point or MEASURED_OPERATING_POINT
    base = {k: v for k, v in op.items() if k in ("rider_miss", "class_flip", "plate_miss")}
    ceiling = _suite_f1(scenarios)
    by_rate: dict = {}
    for rate in ocr_rates:
        faults = {**base, "ocr_corrupt": rate}
        all_f1 = _mean_f1(scenarios, faults, trials, seed)
        rows = []
        for stage, desc in BUDGET_STAGES:
            fixed = _mean_f1(scenarios, _without(faults, stage), trials, seed)
            rows.append({"stage": stage, "description": desc,
                         "f1_if_perfect": round(fixed, 4),
                         "recovered_f1": round(max(0.0, fixed - all_f1), 4)})
        total = sum(r["recovered_f1"] for r in rows)
        for r in rows:
            r["share_of_recovered"] = round(r["recovered_f1"] / total, 4) if total else 0.0
        rows.sort(key=lambda r: r["recovered_f1"], reverse=True)
        by_rate[f"{rate:.2f}"] = {
            "ocr_read_error_rate": rate,
            "all_faults_f1": round(all_f1, 4),
            "ceiling_gap": round(ceiling - all_f1, 4),
            "stages": rows,
            "largest_owner": rows[0]["stage"] if rows and rows[0]["recovered_f1"] > 0 else None,
        }
    owners = {v["largest_owner"] for v in by_rate.values()}
    central = by_rate[f"{ocr_rates[len(ocr_rates) // 2]:.2f}"]
    return {
        "method": "oracle ablation at the measured operating point",
        "operating_point": {**{k: v for k, v in op.items() if k != "class_flip"},
                            "class_flip": {k: list(v) for k, v in op["class_flip"].items()}},
        "no_fault_f1": round(ceiling, 4),
        "trials": trials,
        "seed": seed,
        "by_ocr_rate": by_rate,
        # Kept for the dashboard: the owner at the central OCR assumption, and
        # whether that conclusion survives the whole OCR sweep.
        "bottleneck": central["largest_owner"],
        "bottleneck_robust_to_ocr_assumption": len(owners) == 1,
        "stages": central["stages"],
        "interpretation": (
            "Each stage fails at its per-frame rate measured on the validation "
            "split (OCR, unmeasured, is swept); making one stage perfect and "
            "measuring the fine-F1 it recovers attributes the error. Synthetic "
            "scenarios, so this ranks where effort pays off in THIS pipeline; it "
            "is NOT a claim about how often each stage fails in the field."
        ),
    }


def stage_tolerance(scenarios: list, *, rates=(0.1, 0.2, 0.3, 0.5), trials: int = 10,
                    seed: int = 11) -> dict:
    """Dose-response per stage in ONE unit (per vehicle-frame event
    probability): end-to-end F1 when only that stage fails at each rate."""
    out: dict = {}
    for stage, _desc in BUDGET_STAGES:
        row = {}
        for rate in rates:
            if stage == "rider_recall":
                f = {"rider_miss": dict.fromkeys(("WithHelmet", "WithoutHelmet",
                                                  "TripleRiding"), rate)}
            elif stage == "helmet_class":
                f = {"class_flip": {"WithHelmet": ("WithoutHelmet", rate),
                                    "WithoutHelmet": ("WithHelmet", rate)}}
            elif stage == "plate_recall":
                f = {"plate_miss": rate}
            else:
                f = {"ocr_corrupt": rate}
            row[f"{rate:.2f}"] = round(_mean_f1(scenarios, f, trials, seed), 4)
        out[stage] = row
    return {"unit": "probability per vehicle per frame", "f1_by_rate": out}


def _suite_f1(scenarios: list) -> float:
    """End-to-end fine F1 across a scenario suite (the metric the budget moves)."""
    tp = fp = fn = 0
    for sc in scenarios:
        m = evaluate_system(sc)
        tp += m.true_positives
        fp += m.false_positives
        fn += m.false_negatives
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------

@dataclass
class PipelineReport:
    helmet: dict = field(default_factory=dict)
    triple: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"helmet": self.helmet, "triple_riding": self.triple,
                "error_budget": self.budget}


def evaluate_all(*, trials: int = 20) -> dict:
    """Run every per-violation evaluation plus the error budget."""
    from modules.system_eval import builtin_scenarios

    helmet = helmet_scenarios()
    triple = triple_riding_scenarios()
    combined = builtin_scenarios() + helmet + triple

    return PipelineReport(
        helmet={
            "decision": evaluate_violation(helmet, "no_helmet").as_dict(),
            "confirm_window_sweep": sweep_confirm_window(helmet, "no_helmet"),
            "scenarios": [s.name for s in helmet],
        },
        triple={
            "decision": evaluate_violation(triple, "triple_riding").as_dict(),
            "confirm_window_sweep": sweep_confirm_window(triple, "triple_riding"),
            "scenarios": [s.name for s in triple],
            "rider_count_note": (
                "The detector has one TripleRiding class, so three-up and "
                "four-up are indistinguishable to this system; it detects the "
                "violation, it does not count riders."
            ),
        },
        budget=error_budget(combined, trials=trials),
    ).as_dict()


# ---------------------------------------------------------------------------
# Does the pipeline actually beat a single-frame detector? (Phase 30)
# ---------------------------------------------------------------------------

def naive_single_frame_decisions(scenario: Scenario) -> dict[int, set]:
    """The strawman this whole project exists to beat, implemented honestly.

    A single-frame detector policy: if *any* frame shows a violation box for a
    vehicle, fine it. No tracking, no temporal confirmation, no contradiction
    check, no OCR voting — just believe the detector.

    Attribution is by box overlap with the ground-truth vehicle, which is
    *generous* to the naive policy: it is handed perfect association for free,
    something a real single-frame system would not have. The comparison is
    therefore a lower bound on the pipeline's advantage, not an inflated one.
    """
    decisions: dict[int, set] = {v.gt_id: set() for v in scenario.vehicles}
    for frame in scenario.frames:
        for vf in frame:
            if vf.gt_id not in decisions:
                continue
            for d in vf.dets:
                if d.label == "WithoutHelmet":
                    decisions[vf.gt_id].add("no_helmet")
                elif d.label == "TripleRiding":
                    decisions[vf.gt_id].add("triple_riding")
    return decisions


def _score(scenarios, decide, violation) -> dict:
    tp = fp = fn = tn = 0
    for sc in scenarios:
        decisions = decide(sc)
        for v in sc.vehicles:
            expected = violation in v.violations
            got = violation in decisions.get(v.gt_id, set())
            if expected and got:
                tp += 1
            elif expected:
                fn += 1
            elif got:
                fp += 1
            else:
                tn += 1
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(p, 4), "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
    }


DETECTOR_NOISE_RATES = (0.0, 0.1, 0.2, 0.3, 0.4)


def pipeline_vs_single_frame(
    *, rates=DETECTOR_NOISE_RATES, trials: int = 20, seed: int = 11,
) -> dict:
    """Head-to-head: naive single-frame fining vs the full pipeline.

    Both policies see the **same** detections, degraded by the same detector
    class-confusion noise (the measured failure mode: WithHelmet <-> WithoutHelmet
    flips, see docs/MODEL_EVALUATION.md). Averaged over seeded trials, because a
    single draw at 30% noise is mostly luck.

    This is the project's central claim reduced to one table: a violation
    decision built from tracking + temporal confirmation should degrade more
    gracefully under detector noise than believing any single frame.
    """
    from modules.system_eval import builtin_scenarios

    scenarios = builtin_scenarios() + helmet_scenarios() + triple_riding_scenarios()
    out: dict = {}
    for rate in rates:
        inj = Injection("detector_noise", "helmet class flips", class_flip=rate)
        rng = random.Random(seed)
        acc: dict = {"naive": [], "pipeline": []}
        for _ in range(trials):
            degraded = [apply_injection(sc, inj, rng) for sc in scenarios]
            for violation in VIOLATIONS:
                acc["naive"].append(
                    _score(degraded, naive_single_frame_decisions, violation))
                acc["pipeline"].append(
                    _score(degraded, run_pipeline_decisions, violation))
        summary = {}
        for policy, rows in acc.items():
            summary[policy] = {
                key: round(sum(r[key] for r in rows) / len(rows), 4)
                for key in ("precision", "recall", "f1")
            }
            summary[policy]["mean_false_positives"] = round(
                sum(r["false_positives"] for r in rows) / len(rows), 3)
        summary["pipeline_f1_advantage"] = round(
            summary["pipeline"]["f1"] - summary["naive"]["f1"], 4)
        out[f"{rate:.2f}"] = summary
    return {
        "note": (
            "Both policies see identical detections. The naive policy is handed "
            "perfect plate-to-vehicle association for free, which a real "
            "single-frame system would not have, so the pipeline's measured "
            "advantage is a lower bound."
        ),
        "trials_per_rate": trials,
        "seed": seed,
        "rates": out,
    }
