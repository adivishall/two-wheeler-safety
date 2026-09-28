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
- **Conclusion.** No significant overall gain; significant Plate regression → not
  promoted. The hypothesis is neither supported nor refuted at this sample size.
  (First judged on test point estimates as a WithHelmet regression — that was
  noise; see [AUDIT.md](AUDIT.md) E4.)

### M2 — two more epochs from v1 (`traffic_model_probe`)

- **Hypothesis.** v1 is under-trained at 15 epochs.
- **Config.** v1 `last.pt` fine-tuned 2 epochs, same data and settings.
- **Result.** mAP@50 **−0.051** [−0.095, −0.006]; Plate −0.041, TripleRiding
  −0.074 (significant).
- **Conclusion.** Worse. An earlier test-split "promote it" recommendation is
  withdrawn.

### M3 — seven more epochs from v1 (`traffic_model_r2`)

- **Config.** `probe` fine-tuned 5 more epochs.
- **Result.** mAP@50 −0.030 [−0.067, +0.007]; Plate **−0.085** (significant).
- **Conclusion.** Not promoted. With M2, longer training from v1 does not help on
  this data.

### M4 — helmet-only fine-tune (`traffic_model_helmetfix`)

- **Hypothesis.** Fine-tuning v1 on a cleaned helmet subset fixes WithHelmet.
- **Config.** v1 `best.pt`, 3 epochs, lr0 0.003, on a helmet-only subset
  (no TripleRiding labels).
- **Result** (historical `model_comparison.md`, 1.0.0). TripleRiding mAP@50
  **0.000** — catastrophic forgetting of a class absent from the fine-tune set.
- **Conclusion.** Rejected; the reason every comparison reads every class column.

### M5 — seed variance of the v1 recipe (`traffic_model_seed1`)

<!-- SEED-EXPERIMENT -->

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
  are never fined.

### D2 — OCR vote thresholds

- **Config** (`evaluate_ocr.py --simulate --policy-sweep`). 16 configs
  (observations × agreement × margin); character substitution 4–12%, with and
  without consistent misreads; 1,000 sequences per condition; dev/test seeds.
- **Result** (`ocr_stabilizer_selection.md`). Old rule wrong-plate up to 3.2%;
  selected (≥3 reads, ≥35%, margin ≥0.3): ~1% (1.2% worst on the held-out seed),
  ~30% less coverage. At 200 sequences the experiment had picked a different,
  costlier config; it did not survive 5× the sample.
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
- **Result** (`benchmark.md`, same clip, same process, lock on vs off): see
  [EVALUATION.md](EVALUATION.md) §4. Recorded fine identical in both arms
  (`tests/test_pipeline_integration.py`).
