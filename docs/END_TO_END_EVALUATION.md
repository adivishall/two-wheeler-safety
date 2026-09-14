# End-to-end evaluation — the pipeline, not the model

[MODEL_EVALUATION.md](MODEL_EVALUATION.md) answers *"how good is the detector?"*.
This page answers the question that actually matters for a system that fines
people:

> **How often does the complete pipeline — detect → track → associate → OCR →
> temporal confirmation → confidence → violation → attribution — produce a
> correct, correctly-attributed fine?**

These are different questions with different answers and they are never
combined. A strong pipeline cannot exceed its detector on the things the
detector decides; it can be much better or much worse on the things the
*pipeline* decides. Measuring only one of them tells you almost nothing about
the other.

## How this is measured without a labelled video corpus

There is no labelled multi-vehicle video set for this project, and inventing one
would produce fiction. Instead, `modules/system_eval.py` and
`modules/pipeline_eval.py` feed **deterministic synthetic scenarios** —
hand-built multi-bike frame sequences with known ground truth — through the
**real runtime code**: `VehicleTracker`, `associate()`, `PlateStabilizer`, the
helmet and triple-riding state machines, `compute_confidence`. No
re-implementation, no mocks of the logic under test.

**What this can honestly claim:** the pipeline *logic* is correct, and the
temporal/voting layers buy a measurable amount over the naive alternatives.
**What it cannot claim:** field accuracy. Real footage brings motion blur,
crowding, and detector noise in combinations these scenarios do not contain.
Every number below is a logic-level result on constructed inputs.

Regenerate everything here with:

```bash
python3 evaluate_pipeline.py
```

---

## 1. The headline question: is the pipeline better than the detector alone?

If a multi-frame pipeline cannot beat "believe the detector", none of the rest of
this project is justified. So it is measured head-to-head.

Both policies see **identical** detections, degraded by the same helmet
class-confusion noise (the detector's measured failure mode). `naive` fines
whenever any single frame shows a violation box — no tracking, no temporal
confirmation, no contradiction check. Averaged over 20 seeded trials:

| detector noise | naive P | naive R | naive F1 | naive FPs | pipeline P | pipeline R | pipeline F1 | pipeline FPs | F1 gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.838 | 1.000 | 0.912 | **1.50** | **1.000** | 1.000 | **1.000** | **0.00** | +0.088 |
| 0.10 | 0.735 | 1.000 | 0.840 | 3.42 | 0.998 | 0.992 | 0.994 | 0.03 | **+0.154** |
| 0.20 | 0.708 | 1.000 | 0.820 | 4.08 | 0.976 | 0.953 | 0.962 | 0.23 | +0.142 |
| 0.30 | 0.697 | 1.000 | 0.810 | 4.40 | 0.962 | 0.883 | 0.913 | 0.30 | +0.102 |
| 0.40 | 0.694 | 1.000 | 0.808 | 4.47 | 0.903 | 0.819 | 0.849 | 0.72 | +0.040 |

**Yes, and the shape of the answer matters more than the size.**

- **Even at zero detector noise** the naive policy averages **1.50 false
  positives** and precision 0.838, because the scenario suite contains cases the
  detector genuinely gets wrong on single frames — flicker and the
  helmet-contradiction bug. The pipeline: **0.00 FPs**.
- **The naive policy has recall 1.000 at every noise level** — it fines on
  anything, so it never misses. **The entire difference is precision.** The
  pipeline is a precision machine, which is the right objective for a system that
  fines people: a missed violation costs a missed fine, a false one accuses an
  innocent rider.
- **The advantage peaks at moderate noise (+0.154 at 10%) and narrows at 40%
  (+0.040).** That is honest and expected: when the detector is broken enough,
  there is no consistent signal left for temporal confirmation to confirm. The
  pipeline degrades gracefully; it does not work miracles.
- The naive policy is handed **perfect plate-to-vehicle association for free**,
  which a real single-frame system would not have. So the measured advantage is
  a **lower bound**.

---

## 2. End-to-end fines

Across 8 multi-bike scenarios (adjacent bikes, crossing bikes, three bikes,
occlusion, late plate, clean rider, single violations):

**Precision 1.000 · Recall 1.000 — TP 8, FP 0, FN 0, wrong-vehicle 0,
wrong-plate 0, duplicates 0.**

| scenario | TP | FP | FN | ID switches | assoc. accuracy |
|---|---:|---:|---:|---:|---:|
| `single_no_helmet` | 1 | 0 | 0 | 0 | 1.00 |
| `single_triple` | 1 | 0 | 0 | 0 | 1.00 |
| `late_plate` | 1 | 0 | 0 | 0 | 1.00 |
| `clean_helmet` | 0 | 0 | 0 | 0 | 1.00 |
| `two_adjacent` | 1 | 0 | 0 | 0 | 1.00 |
| `crossing` | 2 | 0 | 0 | **0** | **1.00** |
| `three_bikes` | 1 | 0 | 0 | 0 | 1.00 |
| `occlusion` | 1 | 0 | 0 | 0 | 1.00 |

A perfect score on a suite you wrote yourself is weak evidence on its own, so the
interesting rows are the ones that *nearly* failed:

- **`crossing` used to be the weak point, and is now clean.** Heavy mid-cross
  overlap used to make association merge two bodies (accuracy 0.85, **4 ID
  switches**); even then both fines landed on the correct plate, because
  attribution runs on the *voted plate*, not the track id — the pipeline
  absorbing a failure. The merge fault has since been fixed at its root
  ([ERROR_BUDGET.md](ERROR_BUDGET.md) §3), so `crossing` now records **0 ID
  switches at association accuracy 1.00**, and the plate-voting defence remains
  as the safety net it was designed to be.
- **A 2-frame total occlusion is survived with no ID switch** (`max_age` holds
  the track LOST, then re-confirms the same id).
- **A late plate** (readable only from frame 4) still produces the fine, because
  recording happens on the *confirmed* state rather than the transition frame.
- A dedicated test injects a consistently-misread plate and asserts it is scored
  as `wrong_plate` + false positive, never a true positive — so the suite cannot
  score itself perfect by being lenient.

---

## 3. Per-violation decisions

Scoring the final per-vehicle verdict as a binary classification, over edge cases
including single-frame flicker, the helmet contradiction bug, plate occlusion,
total detection loss, two-up, exactly three-up and four-up:

| violation | precision | recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_helmet` | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 | 3 |
| `triple_riding` | 1.000 | 1.000 | 1.000 | 4 | 0 | 0 | 2 |

Edge cases that specifically must **not** fine, and don't:

- a helmeted rider the detector briefly calls bare-headed (single-frame flicker),
- a two-up bike the detector briefly calls `TripleRiding`,
- a rider the detector labels helmet **and** no-helmet every frame — the
  reproduced contradiction bug, resolved as ambiguous rather than guessed.

Edge cases that must still fine, and do: three-up, four-up, a bike whose plate is
occluded for 3 frames, and a bike that vanishes entirely for 2 frames.

**Rider counting, stated plainly:** the detector has a single `TripleRiding`
class, so three-up and four-up are the same detection. Four-up is at least as
illegal so fining is correct, but **this system detects a violation; it does not
count riders**, and does not claim to.

---

## 4. What temporal confirmation actually buys

`confirm_window=1` is fining on a single frame — the behaviour the temporal layer
replaces. Sweeping it:

| violation | window | precision | recall |
|---|---:|---:|---:|
| `no_helmet` | **1** | **0.67** | 1.00 |
| `no_helmet` | 2 | 1.00 | 1.00 |
| `no_helmet` | 3 (ships) | 1.00 | 1.00 |
| `no_helmet` | 5 | 1.00 | 1.00 |
| `triple_riding` | **1** | **0.80** | 1.00 |
| `triple_riding` | 2 | 1.00 | 1.00 |
| `triple_riding` | 3 (ships) | 1.00 | 1.00 |

So the temporal layer is not decoration: single-frame fining demonstrably
produces false positives on these scenarios — one wrongly-fined rider in three
for helmets — and two frames removes them **at no recall cost**. The shipped
window of 3+ keeps margin for noisier real footage.

---

## 5. OCR: single-frame vs temporal voting

The other central claim — that cross-frame plate voting beats reading one frame.
Measured by corrupting ground-truth plates with a character-level noise model
built from the confusions OCR genuinely makes on plate glyphs (`0/O`, `1/I`,
`8/B`, `5/S`), then scoring three decision rules on **identical** observations.

`PlateStabilizer` can *abstain* — refuse to name a plate. An abstention is not a
wrong answer; it costs a missed fine, whereas a wrong plate fines an innocent
rider. So the honest framing is coverage plus precision-when-it-answers:

| char substitution rate | `last` cov/acc | `best_conf` cov/acc | `temporal` cov/acc |
|---|---:|---:|---:|
| 0.04 | 1.00 / 0.320 | 1.00 / 0.930 | 0.78 / **0.987** |
| 0.08 | 1.00 / 0.190 | 1.00 / 0.840 | 0.64 / **1.000** |
| 0.12 | 1.00 / 0.150 | 1.00 / 0.700 | 0.51 / **0.980** |
| 0.20 | 1.00 / 0.060 | 1.00 / 0.450 | 0.23 / 0.913 |
| 0.30 | 1.00 / 0.050 | 1.00 / 0.230 | 0.11 / 0.909 |

`last` = the last frame's reading (the pre-stabilizer behaviour). `best_conf` =
the single highest-confidence reading — a strong single-frame baseline, chosen
because beating `last` is easy and would not be an honest comparison.

**Reading:** at 12% character noise the last-frame rule names the wrong plate
**85%** of the time and the best-confidence rule **30%**; the shipped stabilizer
**2%**. Temporal voting does not make OCR *more accurate* — it converts OCR
errors into **abstentions**. For a system that issues fines, that is the correct
trade, and the coverage cost (51% at that noise level) is stated rather than
averaged away.

> **This is simulated noise, not field data.** It measures the *decision rule*
> under a stated error model; it is not a claim about EasyOCR's accuracy on real
> plates. The real-data version exists —
> `python3 evaluate_ocr.py --sequences <csv>` runs the identical comparison
> through actual EasyOCR — and needs a labelled plate-sequence set, which does
> not ship. See [DATASET.md](DATASET.md).

---

## 6. Error budget — where failures come from

Each stage is degraded by an **equal 30%** and the end-to-end F1 drop measured
over 20 seeded trials. Clean baseline F1: **1.000**.

| stage | injected failure | mean F1 | F1 drop | share of measured sensitivity |
|---|---|---:|---:|---:|
| **`ocr`** | characters corrupted in the plate text | 0.547 | **−0.453** | **76.8%** |
| `detection_class` | helmet/no-helmet boxes flipped | 0.910 | −0.090 | 15.3% |
| `detection` | boxes vanish (recall failure) | 0.957 | −0.043 | 7.3% |
| `plate_recall` | plate not read on a frame | 0.997 | −0.003 | 0.6% |

**OCR is the bottleneck, by a factor of five over the next stage.**

The asymmetry is the interesting part and it is not obvious in advance:

- A **corrupted** plate is catastrophic (−0.453). The fine is attributed to a
  real but *wrong* vehicle — an innocent rider gets the ticket, and the system
  has no way to notice.
- A **missing** plate is nearly free (−0.003). Other frames recover it; the
  stabilizer simply waits.

That result is consistent with §4 and explains the design: the stabilizer's
willingness to abstain is defending against the expensive failure mode, and
paying for it in the cheap one.

**What this is not.** It ranks how much the system *depends on* each stage at
equal fault rates. It is **not** a claim about how often each stage fails in the
field — that needs field data nobody has here. A stage with high sensitivity is
where a real failure would hurt most; it is not necessarily where failures
actually occur.

---

## 7. Speed estimation

Synthetic constant-speed trajectories on a perspective ground plane (known ground
truth), with detection noise, through the real estimator:

| motion | calibration | MAE (km/h) | overspeed recall |
|---|---|---:|---:|
| toward camera | constant ppm | ~42 | **0.00** |
| toward camera | linear plane | ~40 | 0.00 |
| toward camera | **homography** | **11.3** | **1.00** |
| lateral | constant ppm | ~1.8 | 1.00 |
| lateral | **linear plane** | **0.87** | 1.00 |
| lateral | homography | ~3.2 | 1.00 |

For motion toward or away from the camera — the common enforcement geometry — a
single pixels-per-metre value is perspective-blind and **misses every
overspeeder**. The homography is the only usable option there. For purely lateral
motion at the calibrated depth the simpler calibrations are already accurate.

Even at its best the homography's MAE is **11.3 km/h**. That is why speed is
labelled an **estimate** everywhere and never presented as radar-grade
measurement, and why an uncalibrated camera reports no speed at all rather than a
meaningless number.

---

## 8. Known weaknesses of the pipeline itself

Distinct from the detector's weaknesses ([MODEL_EVALUATION.md](MODEL_EVALUATION.md) §8):

- **Two-way OCR ties elect rather than abstain.** With exactly two conflicting
  but structurally valid readings, the agreement score is 0.5, which clears
  `PlateConfig.min_confidence` (0.35), so the stabilizer picks one. Three-way
  disagreement correctly abstains. This is the weakest point of the voting rule.
  It is pinned by a test (`test_two_way_tie_elects_at_half_agreement`) so it
  cannot change silently, and is left as-is here because retuning it is a runtime
  behaviour change, not an evaluation one.
- **~~ID switches under heavy overlap~~ (fixed).** The `crossing` scenario used
  to log 4 ID switches; a frame-by-frame diagnosis attributed them to the
  *association* layer merging two vehicles' bodies before the tracker ran, not
  to the tracker. Raising the body-merge thresholds to 0.5 IoU / 0.8 containment
  removes every switch (13 → 0 across 8 targeted scenarios) with no regression —
  see [ERROR_BUDGET.md](ERROR_BUDGET.md) §2–3 and `eval/results/tracking_comparison.json`.
  No Kalman filter was added, because the tracker was never the bottleneck.
- **Synthetic scenarios are clean.** They contain no motion blur, no partial
  boxes, no detector confidence noise, no crowds of 10+ bikes. Real footage will
  be harder in ways this suite does not model.
- **Perfect scores are a ceiling, not a measurement.** P = R = 1.0 says the logic
  handles every case *the suite contains*. The suite was written by the same
  person as the pipeline; the negative cases (§2) are the partial defence against
  that, not a complete one.
- **Speed assumes constant velocity** through the estimation window and a
  correctly surveyed calibration.

---

## 9. Reproducing this page

```bash
python3 evaluate_pipeline.py                    # everything, writes eval/results/
python3 evaluate_pipeline.py --only ocr speed   # one section
python3 evaluate_system.py                      # just the scenario table
python3 evaluate_ocr.py --simulate --sweep      # just the OCR policy comparison
```

Outputs: `eval/results/pipeline_evaluation.{json,md,csv}` and `latest.json` (the
summary the dashboard reads). No weights, no dataset, no network — which is why
CI runs this on every push, and why the evaluation tooling cannot silently rot
between the rare full model evaluations.
