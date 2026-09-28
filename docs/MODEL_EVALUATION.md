# Model evaluation — the detector, on its own

*"What can the detector actually see?"* — measured separately from the pipeline
built around it ([END_TO_END_EVALUATION.md](END_TO_END_EVALUATION.md)).

Every number on this page is read from a generated file under `eval/results/`
(named next to each table) and can be regenerated with the commands in §10.
Each of those files carries a provenance block: weights SHA-256, dataset
content fingerprint, split fingerprint, git commit, library versions, hardware.

## 1. What was evaluated

| | |
|---|---|
| model | `traffic-4class@1.0.0` — YOLOv8n, 15 epochs, Ultralytics defaults, seed 0 |
| weights | `sha256:d49f7983…712cbd` (`models/manifests/traffic-4class.json`) |
| training data | `master_traffic_violation_dataset` — fingerprint `sha256:362a262721cd15ab` |
| held-out data | `eval/clean_splits` (val 352 / test 175 images after removing near-duplicates of training images) — fingerprint `sha256:e1be4f14028241d1` |
| hardware | Apple M4, MPS · torch 2.12.1 · ultralytics 8.4.71 |

## 2. Protocol — and what changed from earlier versions of this page

1. **Choose on validation, report on test.** Model comparisons and threshold
   choices are made on val. Test is evaluated once, for the model already
   chosen. (Earlier, models were compared and a candidate "rejected" or
   "recommended" on the test split — which is tuning on test.)
2. **Standard mAP protocol.** mAP comes from Ultralytics `val()` at conf 0.001,
   NMS IoU 0.7. It used to be computed at the *operating* threshold (conf 0.25,
   NMS 0.5), which truncates the precision/recall curve: same weights, same
   split, **0.727 → 0.769 mAP@50 on test**. The operating threshold is still
   used where it belongs — the confusion matrix and error analysis.
3. **Uncertainty.** Per-class AP@50 comes with a 95% image-bootstrap CI
   (2,000 resamples). Differences between models use a *paired* bootstrap on the
   same resampled images and are called real only if the CI excludes zero.
4. **Two AP flavours, both reported.** `val()` uses *multi-label* NMS (a box can
   carry a second class hypothesis); the pipeline calls `predict()`, which is
   single-label. Bootstrap AP is computed from `predict()` outputs with
   Ultralytics' own AP formula (pinned by a test), so it measures what the
   pipeline actually receives. The gap between the two is largest exactly where
   classes are confusable — WithHelmet.

## 3. Headline — de-leaked **test** split

`eval/results/eval_traffic_model-2_test_clean.json`, `eval/results/uncertainty_test.json`

| | mAP@50 | mAP@50-95 | P | R |
|---|---:|---:|---:|---:|
| `val()`, standard protocol | **0.769** | **0.558** | 0.794 | 0.744 |
| `predict()` AP@50, 95% CI | **0.750** [0.690, 0.840] | — | — | — |

| class | instances | AP@50 `val()` | AP@50 `predict()` [95% CI] | mAP@50-95 |
|---|---:|---:|---:|---:|
| `TripleRiding` | 30 | 0.971 | 0.963 [0.897, 0.995] | 0.770 |
| `Plate` | 131 | 0.887 | 0.889 [0.837, 0.938] | 0.516 |
| `WithoutHelmet` | 105 | 0.791 | 0.734 [0.655, 0.820] | 0.612 |
| `WithHelmet` | 27 | 0.426 | **0.415 [0.204, 0.729]** | 0.336 |

Validation, for comparison (`eval_traffic_model-2_val_clean.json`,
`uncertainty_val.json`): `val()` mAP@50 0.766 / mAP@50-95 0.542; `predict()`
AP@50 0.721 [0.668, 0.786].

**How to read it.** The CI on WithHelmet spans half the scale: with 27 instances
the class's AP is barely constrained. Any statement of the form "model X is
better on WithHelmet by 0.05" is noise unless a paired test says otherwise.
Plate's mAP@50-95 (0.52) is far below its mAP@50 (0.89): plates are *found* but
boxed loosely — which matters for OCR crops more than for detection counts.

## 4. At the operating threshold (conf 0.25)

| class | val P / R / F1 | test P / R / F1 |
|---|---|---|
| `Plate` | 0.906 / 0.818 / 0.860 | 0.898 / 0.870 / 0.884 |
| `WithHelmet` | 0.346 / 0.692 / 0.462 | 0.519 / 0.519 / 0.519 |
| `WithoutHelmet` | 0.721 / 0.718 / 0.719 | 0.690 / 0.743 / 0.716 |
| `TripleRiding` | 0.845 / 0.845 / 0.845 | 0.879 / 0.967 / 0.921 |

Helmet confusions at this threshold: WithoutHelmet → WithHelmet 16 (val) / 8
(test); WithHelmet → WithoutHelmet 4 / 3. Per rider-box on val: a helmeted rider
is labelled WithoutHelmet **15.4%** of the time, a bare-headed rider WithHelmet
7.7%, and 15–20% of rider boxes are missed. These are the rates the temporal
experiment and the error budget use.

## 5. Thresholds: chosen on val, checked once on test

`eval/results/uncertainty_val.md` picks each class's F1-optimal confidence on
val; `uncertainty_test.md` applies those thresholds unchanged.

- **WithoutHelmet: 0.375 transfers.** Versus the old 0.3 floor it removes 48 → 42
  false boxes on val and 31 → 26 on test for one lost true positive (val) and
  none (test). **Adopted** as `HELMET_MIN_CONF`.
- **WithHelmet: does not transfer.** Val-optimal 0.675 gives F1 0.58 on val and
  0.42 on test. With 26/27 instances the threshold itself is noise, so no
  per-class threshold is used for it.
- **TripleRiding**: val-optimal 0.55 is *worse* on test than 0.25 (0.90 vs
  0.92). Not adopted.

## 6. Choosing a model — re-analysed with CIs

`eval/results/uncertainty_val.md` (selection), `uncertainty_test.md` (report).
Paired differences vs `traffic-4class@1.0.0`, **val**:

| candidate | what it is | Δ mAP@50 [95% CI] | significant per-class differences |
|---|---|---|---|
| `traffic_model_v2_dedup` | same recipe, de-duplicated training split | +0.021 [−0.022, +0.063] | WithoutHelmet **+0.046**; Plate **−0.022** |
| `traffic_model_probe` | v1 fine-tuned 2 more epochs | **−0.051** [−0.095, −0.006] | Plate −0.041, TripleRiding −0.074 |
| `traffic_model_r2` | earlier retrain (different size) | −0.030 [−0.067, +0.007] | Plate −0.085 |

**Decision: keep v1.** Rule (stated before looking): promote only if the mAP@50
gain's CI excludes zero *and* no class regresses significantly. v2 fails both;
probe is significantly worse.

This reverses the *reasons* recorded earlier, not the outcome for v2:

- "v2 rejected because WithHelmet regressed 0.387 → 0.299" — on test, the paired
  difference is −0.039 [−0.210, +0.087]: **indistinguishable from noise**, and on
  val v2 is nominally *better* on WithHelmet. The rejection stands on Plate, not
  WithHelmet.
- "Promote `traffic_model_probe` (wins on both helmet classes)" — a test-split
  point estimate. On val it is significantly worse overall. That recommendation
  is withdrawn.

### Seed variance

<!-- SEED-VARIANCE -->

## 7. Robustness to image degradation (synthetic)

`eval/results/robustness_val.md` — label-preserving transforms of real val
images, paired against the same images clean. **Synthetic transforms; not a
measurement on real night / rain footage.** Δ AP@50 (★ = CI excludes 0):

| transform | Plate | WithHelmet | WithoutHelmet | TripleRiding | mAP@50 |
|---|---:|---:|---:|---:|---:|
| motion blur 9 px | −0.137★ | −0.089 | −0.073★ | −0.114★ | −0.103★ |
| motion blur 21 px | **−0.655★** | −0.415★ | −0.310★ | −0.531★ | −0.478★ |
| low light (severe) | −0.320★ | −0.245★ | −0.364★ | −0.353★ | −0.321★ |
| glare (mild) | −0.008 | **−0.158★** | −0.011 | −0.027★ | −0.051★ |
| JPEG q=8 | −0.151★ | −0.055 | −0.104★ | −0.063★ | −0.093★ |
| half resolution | −0.018★ | +0.002 | −0.001 | −0.002 | −0.005 |
| 25% occluded | −0.359★ | −0.163★ | −0.376★ | −0.653★ | −0.388★ |

- **Plate is the most fragile class** — first to fall under blur, compression
  and even halved resolution. Every fine depends on the plate, so for a real
  camera motion blur (shutter speed vs vehicle speed) is the parameter to design
  against, ahead of resolution.
- **Glare hits WithHelmet specifically** (−0.16 even when mild): a shiny helmet
  under glare stops looking like a helmet — the confusion that leads toward a
  false no-helmet call.
- Resolution loss down to ¼ barely matters except for plates.

## 8. What the errors actually are — manual review

`eval/results/manual_error_review.md`: the 16 highest-confidence false positives
and 16 highest-confidence class confusions on val, inspected one by one (crops
listed so anyone can re-check; judgements, not ground truth).

- **11 of 16 top "false positives" are real objects the labels omit** —
  bare-headed riders, plates (`TS 20 7607`, `MH34AJ 7050`), unlabelled riders.
  Measured precision *understates* the detector for Plate and WithoutHelmet.
- **At least 3 (probably 6) of 16 top class confusions are label errors** — a
  rider in a full-face helmet labelled WithoutHelmet, a number plate labelled
  WithoutHelmet, bare-headed riders labelled WithHelmet.
- **Genuine model errors, by kind:**
  - *Head coverings read as helmets* (4 of 16 confusions): women wearing a
    dupatta or scarf are called WithHelmet. The consequence is a **missed**
    violation (the safe direction for fines) — but it is a failure correlated
    with a demographic group, i.e. a fairness problem, not only an accuracy one.
  - *Context*: a man in a yellow helmet standing with a **bicycle**, and a
    pedestrian beside a parked motorbike, are both called WithoutHelmet. The
    model detects "person near two-wheeler", not "rider".
  - *Helmeted rider called no-helmet* (2 confusions, 1–2 false positives): the
    error that fines an innocent person. Rare in this sample, and it is what the
    temporal rule and human review exist for.

## 9. Dataset composition is the largest hidden limit

`eval/results/label_audit.md` (details in [DATASET.md](DATASET.md)): the dataset
is two exports glued together with **disjoint label sets**. {Plate, WithHelmet,
WithoutHelmet} and {TripleRiding} never appear in the same labelled image — in
train, val or test. Consequences:

- TripleRiding's excellent AP (0.96) is measured entirely on one source's
  images, and plates / riders in those images were never labelled, so the model
  learned them as background there. On val it predicts a plate on **65%** of
  helmet-source images but **6%** of triple-riding images.
- A fine needs a plate. On real footage, triple riding will therefore mostly be
  **confirmed but withheld** — and no held-out metric can show it, because the
  held-out labels share the gap.

## Confidence

Confidence ranks correctness well for WithoutHelmet (Spearman 1.00 on both val
and test over score bins) and Plate (0.9 / 0.6). For WithHelmet and
TripleRiding the rank correlation flips sign between val and test (0.95 → −0.15;
0.7 → −0.2): at 27–58 instances even *whether* confidence is informative cannot
be established. The system therefore never presents a score as a probability.

## Limits of everything above

- Held-out ≠ generalisation: val and test come from the same sources as training
  and share their cameras, cities and biases.
- WithHelmet has 26/27 instances per split; its numbers are wide intervals, not
  facts.
- Labels are noisy in both directions (§8); some "errors" are the model being
  right.
- The robustness transforms approximate real conditions; they do not reproduce
  them.
- The dataset's source and licence are unverified
  ([data/DATASET_MANIFEST.md](../data/DATASET_MANIFEST.md)).

## 10. Reproducing this page

```bash
python3 audit_dataset.py --data master_traffic_violation_dataset/data.yaml \
    --write-clean-split eval/clean_splits
python3 audit_labels.py --data master_traffic_violation_dataset/data.yaml \
    --model runs/detect/traffic_model-2/weights/best.pt
M=runs/detect/traffic_model-2/weights/best.pt
python3 evaluate_model.py --model $M --data eval/clean_splits/data.yaml --split val \
    --out eval/results --name eval_traffic_model-2_val_clean --save-artifacts eval/artifacts_val
python3 evaluate_model.py --model $M --data eval/clean_splits/data.yaml --split test \
    --out eval/results --name eval_traffic_model-2_test_clean
python3 evaluate_uncertainty.py --split val --name uncertainty_val --models $M <candidates…>
python3 evaluate_uncertainty.py --split test --name uncertainty_test \
    --thresholds-from eval/results/uncertainty_val.json --models $M <chosen…>
python3 evaluate_robustness.py
```
