# Error analysis

Where this system is wrong, why, and what the pipeline does about it. Every
number here is measured (see [EVALUATION.md](EVALUATION.md) for how); this page
is the honest failure-mode companion to it.

## 1. Detector: the weakest link, and its specific failures

The detector bounds everything downstream. Measured on the val split
(`traffic-4class@1.0.0`): mAP@50 **0.697**, mAP@50-95 **0.502**.

### `WithHelmet` is the worst class (P 0.42, mAP@50 0.44)
Two compounding causes:
- **Data scarcity** — only **27** `WithHelmet` instances in the val split, so the
  metric is high-variance (a handful of boxes swing it).
- **Genuine confusion** — the model mixes up helmeted vs un-helmeted heads: in the
  error-analysis pass, **16** `WithoutHelmet`→`WithHelmet` confusions and **4** the
  other way. This is the single most consequential error mode, because a wrong
  helmet call is a wrong *fine*.

**Mitigation in the runtime (not the model):** a `no_helmet` fine is never raised
from a raw box. It requires (a) the **contradiction check** — if the same rider is
called both helmet and no-helmet in a frame, it's treated as ambiguous and
dropped, not fined; and (b) a **multi-frame temporal streak** (`HelmetStateMachine`)
before confirmation. A single confident-but-wrong frame cannot produce a fine.
The contradiction case is locked by a regression test (`test_main_helpers.py`).

### `Plate` localization is loose (mAP@50-95 0.43)
Boxes aren't pixel-tight. This is acceptable by design: OCR only needs a readable
crop, not a tight box, and `Plate` mAP@50 (0.81) — the "is there a plate here"
question — is strong.

### Confidence is discriminative but not calibrated
Mean confidence **0.809 when correct vs 0.561 when wrong** (+0.248 separation). So
the score is useful for **ranking** a review queue, but "0.8" does **not** mean
"80% correct". The UI labels it a score everywhere and never a probability;
calibration tooling exists but no calibrator is applied without a labelled
outcome set ([EVALUATION.md](EVALUATION.md) §5).

## 2. Tracking/association: the crossing case

The end-to-end synthetic suite (`evaluate_system.py`) is precision = recall = 1.0
across 8 scenarios **except** one clear weak point:

- **`crossing`** — when two bikes overlap heavily mid-cross, association merges
  their bodies (accuracy drops to **0.85**) and the tracker logs **4 ID switches**.

**Why it still produces correct fines:** temporal confirmation + OCR plate-voting
attribute each fine to the correct *plate* even when the *track id* churns — the
pipeline absorbs a tracker failure. A 2-frame occlusion is survived with no ID
switch (`max_age` keeps the track LOST then re-confirms the same id). This is the
whole reason the design confirms over a streak and votes the plate rather than
trusting any single frame or id.

## 3. OCR: the failure the stabilizer is built for

A single OCR frame flickers look-alikes (`0/O`, `1/I`, `8/B`, `5/S`), drops
readings, and can invent a plate. Left unchecked, a fine could be issued against
the wrong vehicle.

**Mitigation:** `PlateStabilizer` collects readings across frames, scores each by
OCR confidence **and** structural validity (must match a real Indian plate
structure), and elects a plate by weighted temporal voting — refusing to report
until it has seen enough observations. `correct_plate` only proposes a look-alike
fix if it makes the string match a valid structure within a small edit bound.
(No labelled OCR set ships with the repo, so no field OCR accuracy is asserted;
the voting/correction logic is unit-tested in `test_plate_recognizer.py`.)

A performance note that is *not* an accuracy trade-off: once the stabilizer has
locked a high-confidence plate, per-frame OCR is skipped (measured +172%
throughput) and the recorded fine is **byte-identical** — verified in
`test_pipeline_integration.py`.

## 4. Known systemic limitations (honest)

- **Component ≠ field accuracy.** §1 numbers are on a small, same-pool val set;
  they don't claim real-world accuracy. See the leakage/imbalance caveats in
  [DATASET.md](DATASET.md).
- **Speed is an estimate, not radar.** Only usable with a homography calibration;
  perspective-blind for motion toward the camera without it
  ([EVALUATION.md](EVALUATION.md) §4).
- **The model is the ceiling.** A strong pipeline around a mediocre detector is
  still bounded by the detector — which is exactly why every flag is a *candidate*
  for human review, never a verdict.

## 5. How each failure is contained (summary)

| Failure mode | Root cause | Containment |
|--------------|-----------|-------------|
| Wrong helmet call | detector class confusion, scarce `WithHelmet` data | contradiction check + temporal streak |
| Single-frame false positive | model flicker | confirm over N consecutive frames |
| Wrong plate on a fine | OCR character flicker | temporal voting + structure-valid correction |
| ID switch when bikes cross | association merge under heavy overlap | plate-voting attributes to the right plate anyway |
| Overconfident wrong flag | uncalibrated score | shown as a score; mandatory human review |
