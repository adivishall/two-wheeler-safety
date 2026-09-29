# Experiment register

One record per experiment, in the same shape: what was believed, what was run,
on which data and seed, what came out, and what was decided. Numbers are read
from the named result files. Model runs share the evaluation protocol in
[EVALUATION.md](EVALUATION.md) §1 (select on de-leaked val with paired bootstrap
CIs; report on test once).

Dataset fingerprints (`modules/provenance.py`): master `sha256:362a262721cd15ab`;
de-duplicated train `sha256:5ba9cee6da3943dd`; helmet-only subset
`sha256:ca24dc69bfb0bf37`; de-leaked held-out `sha256:e1be4f14028241d1`.

## Detector training

### M1 — de-duplicated training data (`traffic_model_v2_dedup`)

- **Hypothesis.** The training set's equal class counts (4,862 each) come from
  duplication; WithHelmet has the least unique data; training on unique images
  helps WithHelmet.
- **Config.** yolov8n.pt, 15 epochs, imgsz 640, batch 16, lr0 0.01, seed 0,
  Ultralytics defaults — identical to v1 except the data
  (`datasets/train_clean`, 5,772 images after removing 48% near-duplicates).
- **Result** (`uncertainty_val.md`). mAP@50 +0.021 [−0.022, +0.063];
  WithHelmet +0.028 [−0.107, +0.167]; WithoutHelmet **+0.046** [+0.005, +0.087];
  Plate **−0.022** [−0.044, −0.003].
- **Conclusion.** No significant overall gain; a Plate regression that the
  bootstrap calls significant but that is no larger than a seed change (M5) →
  not promoted, not rejected. The hypothesis is neither supported nor refuted;
  the WithoutHelmet gain is worth a multi-seed run.
  (First judged on test point estimates as a WithHelmet regression — that was
  noise; see [AUDIT.md](AUDIT.md) E4.)

### M2 — two more epochs from v1 (`traffic_model_probe`)

- **Hypothesis.** v1 is under-trained at 15 epochs.
- **Config.** v1 `last.pt` fine-tuned 2 epochs, same data and settings.
- **Result.** mAP@50 **−0.051** [−0.095, −0.006]; Plate −0.041, TripleRiding
  −0.074 (significant).
- **Conclusion.** A worse checkpoint; whether longer training hurts is not
  established (the deficit is close to M5's seed shift). An earlier test-split
  "promote it" recommendation is withdrawn.

### M3 — seven more epochs from v1 (`traffic_model_r2`)

- **Config.** `probe` fine-tuned 5 more epochs.
- **Result.** mAP@50 −0.030 [−0.067, +0.007]; Plate **−0.085** (significant).
- **Conclusion.** Not promoted. Its Plate drop is 3.5× M5's seed shift in that
  class — the one regression here likely to be real. With M2: no sign that longer
  training from v1 helps.

### M4 — helmet-only fine-tune (`traffic_model_helmetfix`)

- **Hypothesis.** Fine-tuning v1 on a cleaned helmet subset fixes WithHelmet.
- **Config.** v1 `best.pt`, 3 epochs, lr0 0.003, on a helmet-only subset
  (no TripleRiding labels).
- **Result** (historical `model_comparison.md`, 1.0.0). TripleRiding mAP@50
  **0.000** — catastrophic forgetting of a class absent from the fine-tune set.
- **Conclusion.** Rejected; the reason every comparison reads every class column.

### M5 — seed variance of the v1 recipe (`traffic_model_seed1`)

- **Question.** How much does the training seed alone move per-class AP — and
  are M1–M3's differences bigger than that?
- **Config.** v1's `args.yaml` with only `seed: 0 → 1`; Ultralytics 8.4.71, MPS,
  same `data.yaml` (v1's torch version and training-time data fingerprint were
  not recorded).
- **Result** (`uncertainty_val.md`, `uncertainty_test.md`, paired vs v1).
  mAP@50 −0.040 [−0.108, +0.025] val, **−0.076 [−0.155, −0.003] test**;
  WithHelmet −0.098 val, −0.203 test; Plate −0.024 val; TripleRiding −0.077 test
  (significant). Official `val()` mAP@50 0.766 → 0.718.
- **Conclusion.** The paired image bootstrap cannot see training randomness: a
  seed change alone is "significantly worse" on test. One replica is one draw,
  not a spread, but it is as large as M1's Plate regression and M2's deficit —
  so those verdicts are re-graded (MODEL_EVALUATION §6): M1's Plate regression
  is not evidence, its WithoutHelmet gain is the one lead worth testing; M2 is
  a worse checkpoint, not proof that more epochs hurt; M3's Plate −0.085 (3.5×
  the seed shift) stands. v1 is the better of two seeds, so the headline
  detector numbers are probably optimistic for the recipe.
- **Decision.** Promotion now needs ≥ 3 seeds per recipe
  ([RETRAINING_LOOP.md](RETRAINING_LOOP.md) §7). v1 stays.

## Evaluation protocol

### P1 — mAP protocol

- **Question.** How much did computing mAP at the operating threshold (conf 0.25)
  understate it?
- **Result.** Test mAP@50 0.727 → **0.769**, mAP@50-95 0.532 → **0.558** under the
  standard protocol (conf 0.001, NMS 0.7), same weights and split.
- **Conclusion.** Standard protocol for mAP; operating threshold only for the
  confusion matrix.

### P2 — per-class thresholds, val → test transfer

- **Question.** Do F1-optimal per-class thresholds chosen on val hold on test?
- **Result** (`uncertainty_val.md`, `uncertainty_test.md`). WithoutHelmet 0.375:
  yes (false boxes 48→42 val, 31→26 test, no recall lost). WithHelmet 0.675: no
  (F1 0.58 → 0.42). TripleRiding 0.55: worse than 0.25 on test.
- **Decision.** `HELMET_MIN_CONF` 0.3 → 0.375; no per-class threshold for the
  others.

## Decision rules (model-free; synthetic inputs at measured rates)

### D1 — temporal confirmation rule

- **Hypothesis.** A streak absorbs per-frame helmet flips; the window size is
  the only question.
- **Config** (`evaluate_temporal.py`). Riders simulated at the val-measured
  per-frame rates (helmeted→no-helmet 15.4%, bare→helmet 7.7%, missed 15–20%),
  stickiness 0 / 0.5 / 0.8, dwell 10 / 25 / 50 frames, 1,200 riders per
  condition; 28 rules (streaks, k-of-n, streak + fraction gate); dev seed 101,
  test seed 202; objective: max recall s.t. ≤ 1% false flags in every design
  condition.
- **Result** (`temporal_confirmation.md`). The hypothesis was wrong: a streak's
  false-flag rate grows with dwell under correlated errors (31.5% at 2 s,
  stickiness 0.5); k-of-n is worse. No rule met 1%; the fallback chose streak 5 +
  ≥70% of ≥12 observed frames: 1.2% worst case on the held-out seed. Same choice
  at 4× the sample.
- **Decision.** Shipped for the helmet decision. Cost: riders seen < 12 frames
  are never fined for no helmet.

### D2 — OCR vote thresholds

- **Question.** Which vote thresholds keep wrong plates ≤ 1% at the least
  coverage cost?
- **Config** (`evaluate_ocr.py --simulate --policy-sweep`). 24 configs (winner's
  own valid reads 1–3 × agreement 0.35/0.5 × margin 0–0.3) plus the 1.0.0 rule;
  character substitution 4–12%, with and without consistent misreads; 1,000
  sequences per condition; dev seed 1234, test seed 5678; objective: max
  coverage s.t. ≤ 1% wrong plates in every design condition; **scored at the
  first election**, the moment a held violation is fined.
- **Result** (`ocr_stabilizer_selection.md`). No config met 1% (lowest dev
  worst case 2.0%). Selected by the noise-aware fallback (most coverage within
  2 SE of that minimum): ≥ 3 valid reads, ≥ 35%, margin ≥ 0.3. Held-out: wrong
  plate 0.1% without consistent misreads, 0.9–1.8% with them, at 34–70%
  coverage; the 1.0.0 rule, 40–62% wrong at 100% coverage. The support
  requirement did more than the margin: every config allowing a single valid
  read had a worst case ≥ 7%.
- **History.** At 200 sequences the experiment picked a different, costlier
  config that did not survive 5× the sample. It then scored plates after all
  ten reads — later than the pipeline commits — and reported the old rule at
  3.2% and the selected one at 1.2%; an independent review found that
  (AUDIT R1), and those numbers are withdrawn.
- **Decision.** Shipped; exact ties abstain unconditionally.

### D3 — association merge thresholds

- **Config** (`evaluate_tracking.py`). 8 crossing/overlap scenarios, old
  (IoU 0.3 / containment 0.6) vs new (0.5 / 0.8), decision rule held fixed.
- **Result** (`tracking_comparison.json`). ID switches 13 → 0; association
  accuracy 0.760 → 0.989; fine false positives 5 → 0.
- **Decision.** New thresholds (1.0.0), retained.

## Robustness and performance

### R1 — synthetic corruptions

- **Config** (`evaluate_robustness.py`). 7 label-preserving transforms × 2
  severities on val, paired bootstrap vs clean.
- **Result** (`robustness_val.md`). Plate is the most fragile class (motion blur
  21 px −0.66 AP); glare hits WithHelmet (−0.16 mild); ¼ resolution barely matters
  except for plates.
- **Implication.** For a real camera, shutter speed matters more than resolution.

### R2 — OCR lock (performance)

- **Hypothesis.** Once the vote has settled a plate, further OCR is wasted work.
- **Config** (`benchmark.py --ocr-lock-ab`). `demo_traffic.mp4` (334×596, 120
  frames), MPS, lock on (shipped: lock at ≥ 0.90 after 5 reads, re-read every
  10 frames) vs off; 5 rounds per arm, interleaved A B / B A.
- **Result** (`benchmark.md`). 19.4 → 36.7 FPS median, **+88%** (per round +80%
  to +139%); OCR calls 120 → 30; same fine (MH02DL4596, no helmet) in all 10
  runs, also pinned on a stub video by `tests/test_pipeline_integration.py`.
- **Caveat.** Two earlier single-run benchmarks of the same config gave 32.6 and
  26.9 FPS; the repeat/interleave protocol exists because of that spread.

## Field-evaluation wave

### C1 — detector accuracy by measured image condition

- **Question.** Does the detector degrade in the conditions a field camera
  brings — dark, blurred, glare, dense, small — on the real held-out images?
- **Config** (`evaluate_conditions.py`). Conditions measured from pixels and
  boxes (`modules/image_conditions.py`), cut points fitted on val, reused on
  test; per class, stratum vs the rest (unpaired bootstrap), small vs large
  objects (paired); ≥ 10 instances on both sides; confirmed only if test agrees
  in sign with its own CI excluding zero.
- **Result** (`conditions.md`). 27 comparisons; 9 significant on val (~1
  expected by chance); 2 replicated. Dark images: Plate +0.10 / +0.12 — but they
  are close-up checkpoint photos with lit plates, not night CCTV. Single-rider
  images: WithoutHelmet +0.37 / +0.18 vs the rest — on val the 2–3-rider bucket
  is much worse (0.56 vs 0.84), on test it isn't (0.80 vs 0.80); test's
  replication rests on 3 images with 4+ riders. Small riders harder on val, not
  significant on test. Resolution, angle, weather: not measurable here.
- **Decision.** Leads for the field dataset's sampling (ROADMAP #1), not facts.

### D4 — calibrator selection out of fold

- **Question.** Does the calibration experiment pick a calibrator when none is
  needed?
- **Config.** Scores that are already calibrated (label ~ Bernoulli(score)),
  n = 80 / 150 / 300, 40 datasets each; old rule (lowest in-sample ECE) vs new
  (out-of-fold Brier, CI excluding zero).
- **Result.** Old rule adopted a calibrator in 33, 31 and 30 of 40; new rule
  in 0 of 40 at every n (AUDIT W1).
- **Decision.** New rule shipped; applied to the local database: 0 admissible
  outcomes, NOT MEASURED (`calibration.md`).

### A1 — first active-learning queue

- **Config** (`select_for_labeling.py`). Detector predictions on the 5,772-image
  de-duplicated training pool; signals and diversity as in
  `modules/active_learning.py`; budget 150; plus a 50-image uniform-random,
  model-blind audit of val/test.
- **Result** (`labeling_queue.md`). 320 candidates, 150 selected;
  contradictions / class disagreements / label gaps are 13% / 7% / 7% of the
  queue vs 7% / 4% / 3% of candidates. A first version selected 77 held-out
  images by model disagreement — withdrawn (DECISIONS #26).
- **Open.** Does it beat random selection? ROADMAP #7.

### M6 — v2 against the v1 recipe mean

- **Config** (`compare_recipes.py`). v1 recipe = {v1, seed 1}; v2 = {v2_dedup};
  val; images and seeds bootstrapped together.
- **Result** (`recipe_comparison_val.md`). mAP@50 +0.041 [−0.013, +0.107];
  WithoutHelmet +0.042 [−0.001, +0.081]; Plate −0.010 [−0.037, +0.013].
  Verdict: insufficient seeds (2 vs 1).
- **Decision.** No promotion; one more v1 seed and two more v2 seeds settle it
  (ROADMAP #5).
