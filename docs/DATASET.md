# Dataset provenance

This documents the dataset behind the trained 4-class detector
(`traffic-4class`). Every count here is measured from the label files by
`modules.model_manifest.dataset_version` (run
`python3 -c "from modules.model_manifest import dataset_version, json; ..."`),
never estimated. Where something is **not** verifiable from the files on disk it
is marked **UNVERIFIED** rather than guessed — see *Known gaps* below.

The dataset itself is gitignored (it is large and its licensing is unverified;
see below). This file is the committed record of what it contains.

## Classes

The detector is a single YOLOv8n model with four classes, in dataset order:

| id | class | what it marks |
|----|-------|---------------|
| 0 | `Plate` | a licence-plate region (fed to OCR) |
| 1 | `WithHelmet` | a rider's head **with** a helmet |
| 2 | `WithoutHelmet` | a rider's head **without** a helmet (→ `no_helmet` violation) |
| 3 | `TripleRiding` | three-up riding on one two-wheeler (→ `triple_riding` violation) |

`Plate` is an object-localisation class; the other three are the behavioural
classes the violation logic keys off. `no_helmet` is only reported after the
helmet contradiction check in `modules/detector.py` /
`modules/violation_state.py` (a `WithoutHelmet` box overlapping a `WithHelmet`
box is treated as the model contradicting itself and is suppressed).

## Annotation format

Standard YOLO detection: one `.txt` per image under a parallel `labels/` tree,
each line `class cx cy w h` with box coordinates normalised to `[0, 1]`. An
image with no `.txt` (or an empty one) is a **background** image (no objects).

## Split, image and instance counts (measured)

Directory: `master_traffic_violation_dataset/` with `train/ valid/ test/`
`images/` + `labels/` subtrees.

| split | label files | background (empty) | instances |
|-------|------------:|-------------------:|----------:|
| train | 11,195 | 453 | 21,191 |
| valid |    383 |  76 |    572 |
| test  |    194 |  31 |    311 |
| **total** | **11,772** | **560** | **22,074** |

Per-class **instance** counts:

| class | train | valid | test |
|-------|------:|------:|-----:|
| `Plate` | 6,605 | 247 | 131 |
| `WithHelmet` | 4,862 | 27 | 27 |
| `WithoutHelmet` | 4,862 | 209 | 105 |
| `TripleRiding` | 4,862 | 89 | 48 |

## Class imbalance

* **Training set is deliberately balanced.** The three behavioural classes have
  *exactly* 4,862 instances each. That is not natural traffic — it is the result
  of **offline augmentation/balancing** (see below), which equalised the
  violation classes. `Plate` (6,605) is higher because most images contain a
  plate regardless of behaviour.
* **Validation/test are imbalanced and small.** `WithHelmet` has only **27**
  instances in each of valid and test. Any per-class metric for `WithHelmet` is
  therefore **high-variance** — this is the direct reason its measured mAP@50 is
  the weakest of the four classes (see [EVALUATION.md](EVALUATION.md)). Treat
  `WithHelmet` numbers as indicative, not precise.

## Augmentation

The **training images are pre-augmented offline**, not augmented on-the-fly:

* every `train/images` file is named `aug_<hash>.jpg` and is paired with a
  cached `aug_<hash>.npy` (11,195 of each), i.e. augmentation was baked into the
  stored dataset.
* Consequently the default `train_traffic.py` run keeps Ultralytics' *own*
  augmentation at defaults; the exposed `--hsv-* --degrees --translate --scale
  --fliplr --mosaic --mixup` flags let you experiment with additional on-the-fly
  augmentation on top, but doubling up on already-augmented data is usually not
  helpful and should be measured (see [EVALUATION.md](EVALUATION.md)).

## Known bias, gaps and risks

These are stated honestly because the spec (and good practice) forbids claiming
dataset quality without evidence:

* **Source & licence — UNVERIFIED.** Filenames (`*_jpg.rf.<hash>`,
  `BikesHelmets###`, `ds1_/dst_` prefixes) indicate the set was assembled from
  Roboflow-exported and merged public sources, but the original datasets, their
  authors, and their licences are **not** recorded with the data. **Do not
  redistribute the images or assert a licence until this is traced.** This is the
  single biggest provenance gap.
* **Leakage — MEASURED, and present.** Previously recorded here as UNVERIFIED.
  It has now been checked with perceptual hashing (`audit_dataset.py`,
  `modules/dataset_audit.py`), which compares pixel content instead of the
  re-hashed `aug_<hash>` filenames:

  | split | images | near-duplicates of a training image | rate |
  |-------|-------:|------------------------------------:|-----:|
  | valid |    383 |                                  31 | 8.1% |
  | test  |    194 |                                  19 | 9.8% |

  Most matched at Hamming distance 0; spot-checking by direct pixel comparison
  gave a mean absolute difference of 1–10/255 — the same photographs,
  JPEG-recompressed. **The leakage is real.** Measured impact: evaluating on a
  de-leaked test split moved mAP@50 by **+0.006** (0.7202 → 0.7265), i.e. the
  leakage was *not* inflating the headline metric. Quote the de-leaked number
  anyway, because it is the one with a clean argument behind it
  ([MODEL_EVALUATION.md](MODEL_EVALUATION.md) §2).

  Caveat: dHash catches recompression/resize/photometric duplicates, not heavy
  geometric augmentation, so the de-leaked split is *cleaner*, not provably
  clean.
* **Geographic/plate bias.** Plates and RTO decoding assume the Indian plate
  format (`modules/plate_info.py`); the imagery skews to Indian road scenes.
* **Small, imbalanced eval sets.** 383 val / 194 test images is enough for a
  ballpark but not for tight confidence intervals, especially for `WithHelmet`.
* **`Plate` ≠ readable plate.** A `Plate` box being detected says nothing about
  whether OCR can read it; OCR quality is evaluated separately (see
  `evaluate_ocr.py` and [EVALUATION.md](EVALUATION.md)).

## External test data

There is no separate, independently-sourced external test set yet. The `test/`
split above comes from the same pool as `train/valid` and therefore shares its
biases. A genuinely external clip set (different cameras/cities) is the right
next step for a trustworthy generalisation estimate and is listed as a known
gap, not claimed as done.

## Machine-readable manifest

[../data/DATASET_MANIFEST.md](../data/DATASET_MANIFEST.md) and
`data/dataset_manifest.json` carry the same facts per split as a generated,
diffable artifact (content-hash version, per-split class counts, background
counts, split-hygiene block, and the provenance fields still marked
`UNVERIFIED`). Regenerate with:

```bash
python3 dataset_manifest.py --data master_traffic_violation_dataset/data.yaml \
    --leakage eval/results/dataset_leakage.json
```

## Reproducing these counts

```bash
python3 - <<'PY'
import json
from modules.model_manifest import dataset_version
print(json.dumps(dataset_version("master_traffic_violation_dataset/data.yaml"), indent=2))
PY
```
