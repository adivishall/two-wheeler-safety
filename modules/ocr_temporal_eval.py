"""Does temporal OCR stabilization actually beat single-frame OCR?

The pipeline spends real complexity on `PlateStabilizer` — cross-frame weighted
voting plus structural plate validation — on the premise that per-frame OCR is
too noisy to fine from. That premise deserves a measurement, not an assertion.

Measuring it needs *sequences*: the same plate read repeatedly, as a tracked
vehicle produces. There are two ways to get them, and this module supports both
because they answer different questions:

1. **Simulated noise** (:func:`run_simulation`, no OCR engine required).
   Ground-truth plate strings are corrupted with a character-level error model
   built from the confusions OCR genuinely makes on plate glyphs (``0/O``,
   ``1/I``, ``8/B``, ``5/S``, …) plus dropped characters, spurious characters,
   and whole missed frames. Deterministic given a seed, so it runs in CI and its
   result is exactly reproducible.

   What this can honestly claim: *"under the stated noise model at per-character
   substitution rate p, temporal voting lifts normalized plate accuracy from X
   to Y."* What it cannot claim: a field OCR accuracy. The noise model is a
   model, not a camera.

2. **Real images** (`evaluate_ocr.py --sequences`). Given a CSV of real plate
   crops grouped into sequences, the identical comparison runs through actual
   EasyOCR. That is the number that would settle it for field data; it needs a
   labelled sequence set, which is not shipped (see docs/DATASET.md).

Three policies are compared on identical observations, so the only variable is
the decision rule:

* ``last``      — the last frame's reading (what the pre-stabilizer pipeline did).
* ``best_conf`` — the single highest-confidence reading (a strong single-frame
  baseline; beating ``last`` is easy, so this is the honest bar).
* ``temporal``  — `PlateStabilizer`'s elected plate (what ships).

`temporal` may also *abstain*: it refuses to elect below ``min_observations`` /
``min_confidence``. Abstention is not a wrong answer — it is the pipeline
declining to fine — so it is reported separately as ``abstain_rate`` while still
being counted as not-correct in the accuracy columns, which keeps the comparison
conservative rather than flattering.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from modules.ocr_eval import OcrObservation, evaluate
from modules.plate_recognizer import PlateStabilizer

# Character confusions EasyOCR actually makes on plate glyphs, both directions.
CONFUSIONS: dict[str, tuple[str, ...]] = {
    "0": ("O", "D", "Q"), "O": ("0", "D", "Q"),
    "1": ("I", "L", "7"), "I": ("1", "L"), "L": ("1", "I"),
    "2": ("Z",), "Z": ("2",),
    "5": ("S", "6"), "S": ("5",),
    "6": ("G", "5"), "G": ("6",),
    "8": ("B", "3"), "B": ("8",),
    "4": ("A",), "A": ("4",),
    "7": ("1", "T"), "T": ("7",),
    "3": ("8", "9"), "9": ("3", "8"),
    "M": ("N", "H"), "N": ("M", "H"), "H": ("M", "N"),
    "C": ("G", "0"), "D": ("0", "O"),
}

ALPHANUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

POLICIES = ("last", "best_conf", "temporal")


@dataclass
class NoiseModel:
    """Per-frame OCR corruption parameters.

    Defaults describe a *moderately* noisy read: roughly one character in eight
    substituted with a look-alike, occasional dropped/spurious characters, and
    one frame in ten yielding nothing at all. Every rate is explicit so a report
    can state the exact conditions it measured under.
    """

    substitute_rate: float = 0.12  # per character, swap for a look-alike
    drop_char_rate: float = 0.04  # per character, delete it
    insert_char_rate: float = 0.03  # per character position, insert junk
    miss_frame_rate: float = 0.10  # whole frame returns nothing
    conf_correct: float = 0.82  # mean confidence of an uncorrupted read
    conf_wrong: float = 0.62  # mean confidence of a corrupted read
    conf_jitter: float = 0.08
    # Systematic (correlated) misreads: in this fraction of sequences one glyph
    # is consistently read as the same look-alike on `systematic_strength` of
    # frames — the same plate image misread the same way — which creates a
    # persistent COMPETING plate instead of scattered junk. 0 = independent
    # noise only (the original model).
    systematic_rate: float = 0.0
    systematic_strength: float = 0.5

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class FrameRead:
    """One per-frame OCR reading (simulated or real)."""

    text: str | None
    confidence: float


def corrupt(plate: str, model: NoiseModel, rng: random.Random) -> tuple[str, bool]:
    """Apply the character noise model once. Returns ``(text, was_corrupted)``."""
    out: list[str] = []
    corrupted = False
    for ch in plate:
        if rng.random() < model.drop_char_rate:
            corrupted = True
            continue
        if rng.random() < model.substitute_rate:
            alts = CONFUSIONS.get(ch)
            if alts:
                out.append(rng.choice(alts))
                corrupted = True
                continue
        out.append(ch)
        if rng.random() < model.insert_char_rate:
            out.append(rng.choice(ALPHANUM))
            corrupted = True
    return "".join(out), corrupted


def simulate_sequence(
    plate: str, n_frames: int, model: NoiseModel, rng: random.Random
) -> list[FrameRead]:
    """Simulate ``n_frames`` per-frame OCR reads of one tracked plate.

    A corrupted read gets the lower ``conf_wrong`` mean, which is the property
    the stabilizer's confidence weighting is supposed to exploit; if OCR
    confidence carried no signal at all, weighting could not help and the
    measurement would say so.
    """
    reads: list[FrameRead] = []
    sys_pos, sys_alt = None, None
    if model.systematic_rate > 0 and rng.random() < model.systematic_rate:
        positions = [i for i, ch in enumerate(plate) if ch in CONFUSIONS]
        if positions:
            sys_pos = rng.choice(positions)
            sys_alt = rng.choice(CONFUSIONS[plate[sys_pos]])
    for _ in range(n_frames):
        if rng.random() < model.miss_frame_rate:
            reads.append(FrameRead(None, 0.0))
            continue
        base_plate = plate
        systematic = False
        if sys_pos is not None and rng.random() < model.systematic_strength:
            base_plate = plate[:sys_pos] + sys_alt + plate[sys_pos + 1:]
            systematic = True
        text, corrupted = corrupt(base_plate, model, rng)
        corrupted = corrupted or systematic
        base = model.conf_wrong if corrupted else model.conf_correct
        conf = min(0.99, max(0.05, rng.gauss(base, model.conf_jitter)))
        reads.append(FrameRead(text, conf))
    return reads


def apply_policy(policy: str, reads: list[FrameRead]) -> tuple[str, float, bool]:
    """Reduce a sequence of per-frame reads to one decision.

    Returns ``(text, confidence, abstained)``. ``abstained`` is True only for the
    temporal policy, when the stabilizer declines to elect a plate.
    """
    usable = [r for r in reads if r.text]
    if policy == "last":
        r = usable[-1] if usable else FrameRead("", 0.0)
        return (r.text or ""), r.confidence, False
    if policy == "best_conf":
        r = max(usable, key=lambda x: x.confidence) if usable else FrameRead("", 0.0)
        return (r.text or ""), r.confidence, False
    if policy == "temporal":
        stab = PlateStabilizer()
        for r in reads:
            stab.add(r.text, conf=r.confidence)
        res = stab.result()
        if res.stable:
            return res.stable, res.confidence, False
        return "", res.confidence, True
    raise ValueError(f"unknown policy: {policy}")


@dataclass
class PolicyResult:
    policy: str
    metrics: dict
    abstain_rate: float

    @property
    def answered_accuracy(self) -> float:
        """Of the sequences this policy *answered*, the fraction it got right.

        This is the metric a fine-issuing system is actually judged on. Raw
        ``normalized_match`` counts an abstention the same as a wrong plate,
        which is backwards: refusing to name a vehicle costs a missed fine,
        naming the wrong one fines an innocent rider. Reporting coverage
        (1 - abstain_rate) alongside this precision keeps both costs visible
        instead of hiding one inside a single average.
        """
        answered = 1.0 - self.abstain_rate
        if answered <= 0:
            return 0.0
        return self.metrics["normalized_match"] / answered

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "abstain_rate": round(self.abstain_rate, 4),
            "coverage": round(1.0 - self.abstain_rate, 4),
            "answered_accuracy": round(min(1.0, self.answered_accuracy), 4),
            **self.metrics,
        }


@dataclass
class ComparisonReport:
    """Side-by-side policy comparison over one set of sequences."""

    n_sequences: int
    frames_per_sequence: int
    noise: dict = field(default_factory=dict)
    policies: dict = field(default_factory=dict)

    @property
    def temporal_gain(self) -> float:
        """Temporal minus the *better* single-frame baseline, on normalized match.

        Comparing against the better baseline (not the weaker ``last``) is the
        conservative framing: it is the gain that survives even if the
        single-frame pipeline were improved to pick its most confident read.
        """
        t = self.policies.get("temporal", {}).get("normalized_match", 0.0)
        singles = [
            self.policies[p]["normalized_match"]
            for p in ("last", "best_conf") if p in self.policies
        ]
        return round(t - max(singles), 4) if singles else 0.0

    def as_dict(self) -> dict:
        return {
            "n_sequences": self.n_sequences,
            "frames_per_sequence": self.frames_per_sequence,
            "noise_model": self.noise,
            "policies": self.policies,
            "temporal_gain_vs_best_single_frame": self.temporal_gain,
        }


def compare_policies(
    pairs: list[tuple[str, list[FrameRead]]],
    *,
    frames_per_sequence: int = 0,
    noise: dict | None = None,
) -> ComparisonReport:
    """Score every policy on the same ``(truth_plate, reads)`` sequences.

    A list of pairs rather than a dict, so the same plate can legitimately appear
    in several independent sequences (different noise draws, or several sightings
    of one vehicle) without collapsing into one key.
    """
    report = ComparisonReport(
        n_sequences=len(pairs),
        frames_per_sequence=frames_per_sequence,
        noise=noise or {},
    )
    for policy in POLICIES:
        observations: list[OcrObservation] = []
        abstains = 0
        for truth, reads in pairs:
            text, conf, abstained = apply_policy(policy, reads)
            abstains += 1 if abstained else 0
            observations.append(
                OcrObservation(truth=truth, pred=text, confidence=conf)
            )
        report.policies[policy] = PolicyResult(
            policy=policy,
            metrics=evaluate(observations).as_dict(),
            abstain_rate=abstains / len(pairs) if pairs else 0.0,
        ).as_dict()
    return report


# A fixed plate set spanning the structures the validator knows about, so the
# simulation exercises real plate shapes rather than random strings.
SAMPLE_PLATES = (
    "MH12AB1234", "KA05MN6789", "DL8CAF5031", "TN22BC4567", "GJ01AA0001",
    "UP32DK4321", "RJ14CV0002", "WB06Z9999", "AP09BR1111", "KL07CD8080",
    "MH02DL4596", "HR26DQ5551", "PB10CE1234", "TS08FA7777", "OD02AB3456",
    "BR01PA9090", "AS01AC1212", "CG04LM5656", "JH05AB1010", "MP09XY2468",
)


def run_simulation(
    *,
    plates=SAMPLE_PLATES,
    frames: int = 10,
    repeats: int = 5,
    model: NoiseModel | None = None,
    seed: int = 1234,
) -> ComparisonReport:
    """Simulate ``repeats`` independent sequences per plate and compare policies.

    ``repeats`` exists because one draw per plate would make the result a
    coin-flip on the seed; repeating gives each plate several independent noise
    realisations, so the comparison reflects the decision rules rather than luck.
    """
    model = model or NoiseModel()
    rng = random.Random(seed)
    pairs: list[tuple[str, list[FrameRead]]] = []
    for _rep in range(repeats):
        for plate in plates:
            pairs.append((plate, simulate_sequence(plate, frames, model, rng)))
    return compare_policies(
        pairs, frames_per_sequence=frames, noise=model.as_dict()
    )


DEFAULT_NOISE_SWEEP = (0.04, 0.08, 0.12, 0.20, 0.30)


def noise_sweep(
    rates=DEFAULT_NOISE_SWEEP,
    *,
    frames: int = 10,
    repeats: int = 5,
    seed: int = 1234,
) -> dict:
    """Repeat the comparison across per-character substitution rates.

    A single operating point can flatter either policy; the sweep shows *where*
    temporal voting helps and where it stops helping — at high enough noise every
    frame is wrong, so voting has nothing correct left to elect.
    """
    out = {}
    for rate in rates:
        model = NoiseModel(substitute_rate=rate)
        rep = run_simulation(frames=frames, repeats=repeats, model=model, seed=seed)
        out[f"{rate:.2f}"] = rep.as_dict()
    return {
        "note": (
            "Simulated OCR noise, not field data. These numbers describe the "
            "stated character-error model; they measure the decision rule, not "
            "EasyOCR's accuracy on real plates."
        ),
        "sweep": out,
    }


# ---------------------------------------------------------------------------
# Which stabilizer thresholds? (abstention vs wrong plates, measured)
# ---------------------------------------------------------------------------

def _stabilizer_outcomes(pairs, config) -> dict:
    """Coverage and wrong-plate rate of one PlateConfig over ``pairs``."""
    right = wrong = abstain = 0
    for truth, reads in pairs:
        stab = PlateStabilizer(config)
        for r in reads:
            stab.add(r.text, conf=r.confidence)
        res = stab.result()
        if res.stable is None:
            abstain += 1
        elif res.stable == truth:
            right += 1
        else:
            wrong += 1
    n = len(pairs) or 1
    return {"coverage": round((right + wrong) / n, 4),
            "correct_rate": round(right / n, 4),
            # Share of ALL vehicles that would be fined against the wrong plate:
            # the error that lands on an innocent owner.
            "wrong_plate_rate": round(wrong / n, 4),
            "abstain_rate": round(abstain / n, 4)}


def stabilizer_candidates() -> list:
    from modules.plate_recognizer import PlateConfig

    out = []
    for obs in (2, 3):
        for agree in (0.35, 0.5):
            for margin in (0.0, 0.1, 0.2, 0.3):
                out.append((f"obs>={obs}, agreement>={agree}, margin>={margin}",
                            PlateConfig(min_observations=obs, min_confidence=agree,
                                        min_margin=margin)))
    return out


DESIGN_RATES = (0.04, 0.08, 0.12)  # up to the default "moderate" noise model
SYSTEMATIC_RATES = (0.0, 0.3)  # independent noise only / 30% of plates misread consistently
MAX_WRONG_PLATE = 0.01


def _cond(rate: float, systematic: float) -> str:
    return f"sub={rate:.2f},sys={systematic:.1f}"


def stabilizer_sweep(*, rates=DEFAULT_NOISE_SWEEP, systematic=SYSTEMATIC_RATES,
                     frames: int = 10, repeats: int = 50, seed: int = 1234) -> dict:
    """Every candidate config at every noise condition, on identical sequences."""
    cands = stabilizer_candidates()
    out: dict = {}
    for sysr in systematic:
        for rate in rates:
            model = NoiseModel(substitute_rate=rate, systematic_rate=sysr)
            rng = random.Random(seed + int(rate * 1000) + int(sysr * 10))
            pairs = [(p, simulate_sequence(p, frames, model, rng))
                     for _ in range(repeats) for p in SAMPLE_PLATES]
            out[_cond(rate, sysr)] = {name: _stabilizer_outcomes(pairs, cfg)
                                      for name, cfg in cands}
    return out


def select_stabilizer(results: dict, *, design_rates=DESIGN_RATES,
                      design_systematic=SYSTEMATIC_RATES,
                      max_wrong: float = MAX_WRONG_PLATE) -> dict:
    """Max mean coverage over the design conditions s.t. wrong-plate rate <=
    max_wrong in every one; if none qualifies, minimum worst-case wrong rate."""
    keys = [_cond(r, s) for s in design_systematic for r in design_rates]
    cfgs = dict(stabilizer_candidates())
    names = list(next(iter(results.values())))
    table = []
    for n in names:
        rows = [results[k][n] for k in keys]
        table.append({"config": n,
                      "mean_coverage": round(sum(r["coverage"] for r in rows) / len(rows), 4),
                      "worst_wrong_plate_rate": max(r["wrong_plate_rate"] for r in rows)})
    for t in table:
        t["feasible"] = t["worst_wrong_plate_rate"] <= max_wrong

    def caution(t):
        # Exact ties on the measured objectives go to the MORE conservative
        # config (larger margin, then more observations): same measured cost,
        # plus protection in cases the noise model does not generate (e.g. a
        # plate read as two strings in strict alternation). Specified after the
        # first run produced exact ties between margins 0.0 / 0.1 / 0.2.
        c = cfgs[t["config"]]
        return (c.min_margin, c.min_observations)

    feasible = [t for t in table if t["feasible"]]
    if feasible:
        best = max(feasible, key=lambda t: (t["mean_coverage"], caution(t)))
        how = "primary objective"
    else:
        best = min(table, key=lambda t: (t["worst_wrong_plate_rate"], -t["mean_coverage"],
                                         tuple(-x for x in caution(t))))
        how = "fallback: minimum worst-case wrong-plate rate; exact ties -> more conservative"
    return {"objective": (f"max mean coverage s.t. wrong-plate rate <= {max_wrong:.0%} "
                          f"in every design condition ({', '.join(keys)})"),
            "table": table, "selected": best["config"], "selected_by": how}


def stabilizer_experiment(*, dev_seed: int = 1234, test_seed: int = 5678,
                          repeats: int = 50, frames: int = 10) -> dict:
    """Select on one seed, report on another (the reported numbers are not the
    ones the choice was made on).

    ``repeats=50`` (1,000 sequences per condition): the first run used 10
    (200 sequences), where the worst-case wrong-plate rates being compared
    differed by 2-4 sequences, and it picked a different, far more costly
    config (agreement >= 0.5). At 5x the sample that choice did not survive —
    the selection must be stable to sample size before it is shipped."""
    dev = stabilizer_sweep(seed=dev_seed, repeats=repeats, frames=frames)
    choice = select_stabilizer(dev)
    test = stabilizer_sweep(seed=test_seed, repeats=repeats, frames=frames)
    previous = "obs>=2, agreement>=0.35, margin>=0.0"
    return {
        "frames_per_sequence": frames,
        "sequences_per_rate": repeats * len(SAMPLE_PLATES),
        "dev_seed": dev_seed, "test_seed": test_seed,
        "selection": choice,
        "previous_default": previous,
        "test": {rate: {k: v for k, v in rows.items()
                        if k in (choice["selected"], previous)}
                 for rate, rows in test.items()},
        "caveat": ("Simulated character-level OCR noise (look-alike substitutions, "
                   "drops, insertions, missed frames). Measures the decision rule, "
                   "not EasyOCR's field accuracy."),
    }
