# Model evaluation — the detector, on its own

This page is about **one question only**: how good is the YOLO detector at
finding and classifying boxes? It deliberately says nothing about tracking, OCR,
or whether the system issues correct fines — those are a different question with
a different answer, in [END_TO_END_EVALUATION.md](END_TO_END_EVALUATION.md).
Mixing the two is the most misleading thing this project could do, so they are
kept apart everywhere.

Every number here was produced by running the model on labelled data. Nothing is
estimated, rounded up, or carried over from a training log. The generating
commands are at the bottom and the raw output lives in `eval/results/`.

---

## 1. What was evaluated

| | |
|---|---|
| Model | `traffic-4class@1.0.0` (`runs/detect/traffic_model-2/weights/best.pt`) |
| Architecture | YOLOv8n, 3.0 M params, 8.1 GFLOPs, 73 fused layers |
| Trained | 15 epochs, imgsz 640, batch 16, seed 0, from `yolov8n.pt` |
| Dataset | `master_traffic_violation_dataset` (`sha256:bef022b6e9fd485e`) |
| Device | Apple M4 (MPS), torch 2.12.1, ultralytics 8.4.71 |
| Inference | conf 0.25, IoU 0.5, imgsz 640 |

Classes, in dataset order: `Plate`, `WithHelmet`, `WithoutHelmet`,
`TripleRiding`.

---

## 2. Splits, and which number to believe

Three splits exist. Which one a metric comes from changes what it means, so all
three are reported rather than the flattering one:

| split | images | what it means |
|---|---:|---|
| `train` | 11,195 | the model saw these. Never evaluated on. |
| `val` | 383 | used during training; **not** an unseen test set |
| `test` | 194 | genuinely held out — never used for training or model selection |
| `test` (de-leaked) | 175 | the above, minus images that duplicate training data |

The de-leaked split exists because the held-out data was not, in fact, entirely
held out.

### Leakage was measured, not assumed

The training images were re-hashed by an offline augmentation pass
(`aug_<hash>.jpg`), so a held-out image cannot be matched to its training copy
by filename — which is why [DATASET.md](DATASET.md) previously recorded leakage
as **UNVERIFIED**. `audit_dataset.py` compares *pixel content* with a 64-bit
difference hash instead:

| split | images | near-duplicates of a training image | rate |
|---|---:|---:|---:|
| `val` | 383 | 31 | **8.1%** |
| `test` | 194 | 19 | **9.8%** |

Most matched at Hamming distance 0. Spot-checking by direct pixel comparison
gave a mean absolute difference of 1–10 out of 255 — the *same photographs*,
JPEG-recompressed. The leakage is real.

**And it turned out not to matter much**, which is worth stating plainly because
the opposite was expected:

| split | mAP@50 | mAP@50-95 | precision | recall |
|---|---:|---:|---:|---:|
| `val` (contains leakage) | 0.6967 | 0.5021 | 0.7828 | 0.7398 |
| `test` (contains leakage) | 0.7202 | 0.5270 | 0.7582 | 0.7773 |
| **`test` de-leaked** | **0.7265** | **0.5320** | 0.7559 | 0.7846 |

Removing the leaked images moved mAP@50 by **+0.006** — in the *opposite*
direction to inflation. Two honest readings of that: the leaked images were not
systematically easier than the rest, and 19 images is too small a sample to shift
a mean much either way. The de-leaked number is the one quoted from here on,
because it is the one with a clean argument behind it, not because it is higher.

**Caveat that still stands:** dHash catches recompression, resize and mild
photometric duplicates. It will miss a heavy geometric augmentation (a large
rotation or crop). So the de-leaked split is *cleaner*, not *provably clean*.

---

## 3. Headline metrics — de-leaked held-out test split

**mAP@50 0.7265 · mAP@50-95 0.5320 · mean precision 0.7559 · mean recall 0.7846**
(175 images, 293 labelled instances)

| class | precision | recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| `Plate` | 0.894 | 0.901 | 0.863 | 0.503 |
| `WithHelmet` | 0.538 | 0.519 | **0.387** | 0.310 |
| `WithoutHelmet` | 0.738 | 0.752 | 0.705 | 0.554 |
| `TripleRiding` | 0.853 | 0.967 | **0.950** | 0.761 |

Matching predictions to ground truth per image (`modules/evaluation.py`) gives
the counts behind those rates:

| class | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `Plate` | 114 | 13 | 17 | 0.898 | 0.870 | 0.884 |
| `WithHelmet` | 14 | 13 | 13 | 0.519 | 0.519 | **0.519** |
| `WithoutHelmet` | 78 | 35 | 27 | 0.690 | 0.743 | 0.716 |
| `TripleRiding` | 29 | 4 | 1 | 0.879 | 0.967 | **0.921** |

---

## 4. Class-wise error analysis

### Best class: `TripleRiding` (mAP@50 0.950, F1 0.921)

Three-up riding changes the *silhouette* of the whole vehicle — a taller, wider,
denser blob — so it is a large, distinctive, well-supported target. One missed
instance in the whole split.

One thing it is not: a rider count. The detector has a single `TripleRiding`
class, so three-up and four-up are the same detection. Four-up is at least as
illegal, so fining on it is correct, but **the system detects a violation and
does not count riders** — and does not claim to anywhere.

### Worst class: `WithHelmet` (mAP@50 0.387, F1 0.519)

Three compounding causes, in order of how much they matter:

1. **Scarcity.** Only **27** `WithHelmet` instances in the test split (and 27 in
   val) against 4,862 in train. A handful of boxes swing the metric, so treat it
   as indicative, not precise.
2. **Genuine confusion with `WithoutHelmet`.** In the error-analysis pass: **8**
   `WithoutHelmet`→`WithHelmet` and **3** the other way. A helmeted and a
   bare head occupy the same few dozen pixels at these scales.
3. **It is the one class where a wrong call is a wrong fine.** `Plate` and
   `TripleRiding` errors cost a missed or duplicated detection; a helmet error
   accuses the wrong person.

The `class_confusions/` artifacts make this concrete rather than abstract. The
first saved case is a plainly bare-headed rider on a clear, well-lit, centred
frame, ground truth `WithoutHelmet`, predicted `WithHelmet` at **0.855
confidence**. Not a hard image — a confidently wrong one.

**What the runtime does about it** (this is why the pipeline exists): a
`no_helmet` fine is never raised from a raw box. It requires the contradiction
check — if the model asserts helmet *and* no-helmet on one rider, both are
discarded — plus a multi-frame temporal streak. The measured value of that
second requirement is in
[END_TO_END_EVALUATION.md](END_TO_END_EVALUATION.md) §3.

### `WithoutHelmet` (mAP@50 0.705): the false-positive class

35 false positives against 78 true positives — the highest FP count of any
class. Since this is the class that drives the primary violation, its FP rate is
the main reason the system is a review queue and not an enforcement authority.

### `Plate` (mAP@50 0.863, mAP@50-95 0.503)

Strong at "is there a plate here" (0.863), much weaker at pixel-tight boxes
(0.503). Acceptable by design: OCR needs a readable crop, not a tight box. Note
what this metric does **not** say — a detected `Plate` says nothing about whether
OCR can read it. That is measured separately, and OCR is the system's real
bottleneck ([END_TO_END_EVALUATION.md](END_TO_END_EVALUATION.md) §4).

---

## 5. Confidence: a score that ranks, not a probability

Pooled over all classes, confidence separates correct from wrong: mean **0.810**
when correct vs **0.669** when wrong (+0.142). Binned, the relationship is
clearly monotonic overall:

```
   score bin      n     acc
 0.25-0.40      16   0.312
 0.40-0.55      20   0.600
 0.55-0.70      19   0.474
 0.70-0.85     119   0.849
 0.85-1.00     126   0.857
```

Spearman(score rank, accuracy rank) = **0.90**, top-bin minus bottom-bin accuracy
= **+0.545**.

**But it is not uniform across classes**, and that is the finding the pooled
number hides:

| class | n | Spearman | top−bottom gap | reading |
|---|---:|---:|---:|---|
| `WithoutHelmet` | 113 | **1.00** | **+0.86** | strong ranking signal |
| `Plate` | 127 | 0.60 | +0.25 | usable |
| `WithHelmet` | 27 | −0.15 | +0.21 | weak, non-monotonic |
| `TripleRiding` | 33 | **−0.20** | **−0.04** | **no signal — do not threshold** |

So: ordering a review queue by confidence is well-founded for `WithoutHelmet`,
and **not** for `TripleRiding`, where a 0.9 detection is no more likely to be
right than a 0.5 one. (`TripleRiding` is also the most accurate class overall, so
there is simply little wrong-ness left for the score to sort.)

**This is a score-vs-accuracy curve, not calibration.** YOLO's objectness ×
class score is a model output, not a probability; "0.8" does not mean "80%
correct" and is never presented as such in the API, the UI or these docs. The
path to an actual probability is Platt/isotonic calibration against labelled
review outcomes (`modules/calibration.py`), which needs a labelled outcome set
that does not exist yet — so no calibrator is applied.

---

## 6. Inspectable failures

`evaluate_model.py --save-artifacts` writes representative failures as annotated
context crops (**red** = prediction, **green** = ground truth) plus
`eval/index.json` recording class, score, boxes, IoU and model version:

| category | saved | meaning |
|---|---:|---|
| `false_positives/` | 30 (capped) | predicted a box where there is no ground truth |
| `false_negatives/` | 30 (capped) | ground-truth box missed entirely |
| `class_confusions/` | 12 | box found, wrong class |
| `low_confidence/` | 12 | correct, but below 0.45 |

The artifacts are gitignored: they are derived from dataset imagery whose licence
is unverified, and they regenerate in seconds.

---

## 7. Choosing a model by evidence

Four checkpoints had accumulated in `runs/detect/`, and the shipped one was
chosen by memory. `compare_models.py` evaluates every checkpoint on the identical
de-leaked test split with identical inference settings:

| model | mAP@50 | mAP@50-95 | `Plate` | `WithHelmet` | `WithoutHelmet` | `TripleRiding` | ms | MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `traffic_model_probe` | **0.741** | **0.546** | 0.840 | **0.470** | **0.771** | 0.882 | 27.9 | 6.0 |
| `traffic_model_r2` | 0.728 | 0.519 | 0.836 | 0.422 | 0.752 | 0.902 | 27.4 | 23.4 |
| `traffic_model-2` *(ships)* | 0.727 | 0.532 | **0.863** | 0.387 | 0.705 | **0.950** | 27.9 | 6.0 |
| `traffic_model_helmetfix` | 0.500 | 0.339 | 0.871 | 0.417 | 0.711 | **0.000** | 28.0 | 6.0 |

Lineage: `traffic_model-2` (15 ep from `yolov8n.pt`) → `traffic_model_probe`
(+2 ep) → `traffic_model_r2` (+5 ep). `traffic_model_helmetfix` is
`traffic_model-2` fine-tuned 3 epochs on the narrower `dataset_helmet_clean`.

### The result worth reading twice

**`traffic_model_helmetfix` scores 0.000 on `TripleRiding`.** The checkpoint
named for fixing helmet detection forgot an entire violation class — textbook
catastrophic forgetting from fine-tuning on a narrower dataset — and it barely
moved helmet accuracy either (0.417 vs 0.387). Promoting it on the strength of
its name would have silently disabled triple-riding enforcement. Its own leakage
exposure against `dataset_helmet_clean` was not separately audited, but that only
*flatters* it here, and it still loses decisively.

### Recommendation, not an action

`traffic_model_probe` beats the shipped model on both helmet classes
(`WithHelmet` +0.083, `WithoutHelmet` +0.066) — the weak link and the driver of
the primary violation — while giving up `TripleRiding` (−0.068) and `Plate`
(−0.023). On a suite where helmet is the bottleneck class, that is the better
trade.

It is **not** promoted in this branch. Changing default weights changes runtime
behaviour, wants its own version bump, manifest and evidence-trail check, and
deserves a decision rather than a side effect of an evaluation commit. The
evidence is recorded here so the decision can be made on it.

---

## 8. Honest limits of everything above

- **Held out ≠ generalisation.** The `test` split comes from the same pool as
  `train`/`val` and shares its biases (camera types, cities, lighting). It
  measures held-out performance, not performance on a new deployment.
- **Small splits.** 175 images / 293 instances gives a ballpark, not tight
  confidence intervals. `WithHelmet`'s 27 instances especially.
- **De-leaked, not provably clean.** §2's caveat.
- **One machine, one device.** Latency numbers are Apple M4 / MPS.
- **`Plate` detection ≠ readable plate.**
- **These numbers bound the system; they do not describe it.** A pipeline can be
  no better than its detector on the things the detector decides — but it can be
  *worse or better* on the things the pipeline decides. That is the whole point
  of measuring both.

---

## 9. Reproducing this page

```bash
# 1. audit split hygiene and build a de-leaked test split
python3 audit_dataset.py --data master_traffic_violation_dataset/data.yaml \
    --write-clean-split eval/clean_splits

# 2. evaluate the detector on it, saving failure artifacts
python3 evaluate_model.py \
    --model runs/detect/traffic_model-2/weights/best.pt \
    --data eval/clean_splits/data.yaml --split test \
    --name eval_traffic_model-2_test_clean \
    --save-artifacts eval --benchmark

# 3. compare every checkpoint on the same split
python3 compare_models.py --data eval/clean_splits/data.yaml --split test

# 4. regenerate the dataset manifest
python3 dataset_manifest.py --data master_traffic_violation_dataset/data.yaml
```

Raw output: `eval/results/*.json` (machine-readable) and `eval/results/*.md`.
Weights and the dataset are gitignored, so these run locally; CI runs the
model-free evaluations instead ([TESTING.md](TESTING.md)).
