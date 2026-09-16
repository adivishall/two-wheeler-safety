# Model versioning

A trained `best.pt` is opaque: on its own it does not say which dataset produced
it, on what code, with what hyper-parameters, or how well it scored. That makes
a fine un-auditable and a model swap silent. `modules/model_manifest.py` fixes
this with a small, **torch-free** JSON manifest attached to each checkpoint.

## What a manifest records

| field | source | why |
|-------|--------|-----|
| `name`, `version` | you (`--name`, `--model-version`) | the model line's identity |
| `checksum` | SHA-256 of the weights | tamper-evident id; also used to match a manifest to weights |
| `weights` | path | where the `.pt` lives |
| `training_date` | run's `results.csv` mtime | when the weights were produced |
| `created_at` | now (UTC) | when the manifest was written |
| `git_commit` | `git rev-parse --short HEAD` | code version at manifest time |
| `classes` | dataset `data.yaml` | trained classes, in order |
| `dataset` | label files | path + content-hash `version` + per-class counts |
| `training_config` | run's `args.yaml` | epochs/imgsz/batch/seed/lr0/optimizer/… |
| `metrics` | `evaluate_model.py` JSON or a passed dict | real validation metrics (never fabricated) |

The manifest is written in two places: **beside the weights**
(`<run>/weights/model_manifest.json`, travels with the `.pt`) and into a
**committable registry** (`models/manifests/<name>.json`, so the provenance is
in version control even though the weights are gitignored).

## Generating a manifest

Automatically, at the end of every training run (unless `--no-manifest`):

```bash
python3 train_traffic.py --name traffic_model --model-version 1.1.0
# → runs/detect/traffic_model/weights/model_manifest.json
# → models/manifests/traffic_model.json
```

Or for an existing checkpoint, folding in a real evaluation:

```bash
python3 evaluate_model.py --model runs/detect/traffic_model-2/weights/best.pt \
    --data master_traffic_violation_dataset/data.yaml --split val \
    --name eval_run --out reports
python3 -m modules.model_manifest \
    --model runs/detect/traffic_model-2/weights/best.pt \
    --name traffic-4class --version 1.0.0 \
    --data master_traffic_violation_dataset/data.yaml \
    --metrics reports/eval_run.json
```

## The current shipped model

`models/manifests/traffic-4class.json` describes the checkpoint the app loads by
default (`runs/detect/traffic_model-2/weights/best.pt`):

* **version** `traffic-4class@1.0.0`
* YOLOv8n, trained 15 epochs, imgsz 640, batch 16, seed 0, from `yolov8n.pt`
* dataset `master_traffic_violation_dataset` (22,074 labelled instances)
* val mAP@50 **0.697**, mAP@50-95 **0.502** (see [EVALUATION.md](EVALUATION.md))

## How the runtime uses it

On startup `app.py` calls `resolve_manifest(MODEL_PATH)` — which finds the
manifest beside the weights, or falls back to a registry manifest whose
`checksum` matches the on-disk weights — and turns it into a `name@version`
string. That string is passed to `process_video(..., model_version=...)` and
stamped into every evidence package's metadata (`model_version`). So each fine
carries the exact model identity that raised it, and swapping weights changes the
recorded version automatically. If no manifest is found the runtime simply
records `model_version: null` rather than inventing one.

## v2 candidate (rejected): dedup + de-leak retrain

A second checkpoint, `traffic_model_v2_dedup`, was trained to test one
evidence-based hypothesis: the shipped training set's "perfect" 4,862 / 4,862 /
4,862 class balance is an artefact of duplication (47.7% of images are
perceptual duplicates), and `WithHelmet` — the weakest class — has the least
*unique* data (1,867 instances, 2.6× duplicated). The hypothesis was that
training on the de-duplicated, leakage-free split
([build_train_split.py](../build_train_split.py), 5,772 unique images) would
help `WithHelmet`.

It was trained to **match v1 exactly** — same `yolov8n.pt`, 15 epochs, imgsz
640, batch 16, seed 0, Ultralytics defaults — so the **only** variable is the
data. Both checkpoints were then evaluated on the **same** de-leaked test split
(`eval/clean_splits`, 175 images / 293 instances) with
[compare_models.py](../compare_models.py); full numbers in
[eval/results/model_comparison_v1_v2.json](../eval/results/model_comparison_v1_v2.json).

| metric | v1 `traffic_model-2` | v2 `dedup` | verdict |
|---|---:|---:|---|
| mAP@50 | **0.7265** | 0.7035 | v1 |
| mAP@50-95 | **0.5320** | 0.5155 | v1 |
| mean precision | 0.756 | **0.788** | v2 |
| mean recall | **0.785** | 0.732 | v1 |
| **`WithHelmet` mAP@50** | **0.387** | 0.299 | v1 |
| `WithHelmet` recall | **0.519** | 0.370 | v1 |
| `WithoutHelmet` mAP@50 | 0.705 | **0.734** | v2 |
| `TripleRiding` mAP@50 | **0.950** | 0.925 | v1 |
| `Plate` mAP@50 | **0.863** | 0.855 | v1 |
| latency (ms/img) | **27.0** | 29.2 | v1 |

**Decision: REJECTED — v1 remains the shipped model, no version bump.** The
retrain not only failed to improve the target class, it made `WithHelmet`
*worse* (mAP@50 −0.089, recall −0.149). The most likely cause is that removing
48% of the training images took real signal with the duplicates, and the small
`WithHelmet` class — which had the least unique data to begin with — could least
afford it. Cleaner data did not compensate for less of it at a matched epoch
budget.

What the experiment *did* deliver is the diagnosis itself: the duplication and
per-class unique-count imbalance are now measured, not assumed. The honest
conclusion is that **more unique `WithHelmet` data** (not a re-split of the
existing data) is the highest-value next step — see
[RETRAINING_LOOP.md](RETRAINING_LOOP.md). No v2 manifest is committed, because
no v2 model shipped.

## Versioning policy

`name` identifies the model *line* (`traffic-4class`); `version` is semantic:

* **patch** — retrain on the same dataset/config (bug-fix-level change),
* **minor** — new data or augmentation/config change that shifts metrics,
* **major** — class-set change or architecture change.

Bump the version whenever the weights change; the checksum guarantees a stale
manifest can't silently describe different weights (a mismatch simply fails to
resolve).
