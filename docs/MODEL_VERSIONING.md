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
| `dataset` | label files | path + content fingerprint `version` (label bytes + image names/sizes — the same id every evaluation report records) + per-class counts |
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
* dataset `master_traffic_violation_dataset` (22,074 labelled instances),
  fingerprint `sha256:362a262721cd15ab`
* de-leaked val, standard protocol: mAP@50 **0.766**, mAP@50-95 **0.542**; test:
  **0.769** / **0.558** ([MODEL_EVALUATION.md](MODEL_EVALUATION.md))

## Which exact model generated this result?

Every report in `eval/results/` carries a `provenance` block
(`modules/provenance.py`):

```json
"provenance": {
  "models": [{"path": "runs/detect/traffic_model-2/weights/best.pt",
              "sha256": "sha256:d49f7983…", "version": "traffic-4class@1.0.0"}],
  "dataset": {"version": "sha256:e1be4f14028241d1", "split": "test",
              "split_version": "sha256:fb92cb1c0c0e250f", "split_images": 175},
  "git": {"commit": "…", "dirty": false},
  "environment": {"chip": "Apple M4", "torch": "2.12.1", "ultralytics": "8.4.71", …},
  "config": {"ap_conf": 0.001, "ap_nms_iou": 0.7, "operating_conf": 0.25, …}
}
```

So a number can be traced to weights by hash (not by the ambiguous `best.pt`
path), to data by content, and to code by commit — and a report produced from
uncommitted code or data says so (`dirty: true`; generated results and `*.md`
docs don't count, untracked source files do). Evidence packages carry the same
model version plus the pipeline version and the thresholds actually applied.

## How the runtime uses it

On startup `app.py` calls `resolve_manifest(MODEL_PATH)` — which finds the
manifest beside the weights, or falls back to a registry manifest whose
`checksum` matches the on-disk weights — and turns it into a `name@version`
string. That string is passed to `process_video(..., model_version=...)` and
stamped into every evidence package's metadata (`model_version`). So each fine
carries the exact model identity that raised it, and swapping weights changes the
recorded version automatically. If no manifest is found the runtime simply
records `model_version: null` rather than inventing one.

## v2 candidate (not promoted): dedup + de-leak retrain

`traffic_model_v2_dedup` tested one hypothesis: the training set's "perfect"
4,862 / 4,862 / 4,862 class balance is an artefact of duplication (47.7% of
images are perceptual duplicates), and `WithHelmet` has the least *unique* data,
so training on the de-duplicated, leakage-free split
([build_train_split.py](../build_train_split.py), 5,772 unique images) should
help it. It matches v1 exactly — `yolov8n.pt`, 15 epochs, imgsz 640, batch 16,
seed 0, Ultralytics defaults — so the only variable is the data.

**How it was first judged, and why that was wrong.** The original comparison
used test-split point estimates (`eval/results/model_comparison_v1_v2.md`) and
rejected v2 because WithHelmet mAP@50 fell 0.387 → 0.299. That was a decision
made on test, on 27 instances, without an uncertainty estimate.

**Re-analysed** (`eval/results/uncertainty_val.md`, paired bootstrap on val, the
selection split):

| | Δ AP@50 v2 − v1 [95% CI] | verdict |
|---|---|---|
| mAP@50 | +0.021 [−0.022, +0.063] | not distinguishable |
| WithHelmet | +0.028 [−0.107, +0.167] | not distinguishable |
| WithoutHelmet | +0.046 [+0.005, +0.087] | **v2 better** |
| Plate | −0.022 [−0.044, −0.003] | **v2 worse** |
| TripleRiding | +0.032 [−0.039, +0.113] | not distinguishable |

On test (reported once, `uncertainty_test.md`) the WithHelmet difference is
−0.039 [−0.210, +0.087] — the "regression" was noise.

**Decision: not promoted** under the stated rule (promote only if the mAP gain's
CI excludes zero and no class regresses significantly): no significant overall
gain, and a significant Plate regression — the class every fine depends on. v1
remains `traffic-4class@1.0.0`; no v2 manifest is committed. What the experiment
did establish is that de-duplication by itself neither helps nor clearly hurts:
more *unique* WithHelmet data, and the label fixes in
[ERROR_ANALYSIS.md](ERROR_ANALYSIS.md), are the next steps.

## Versioning policy

`name` identifies the model *line* (`traffic-4class`); `version` is semantic:

* **patch** — retrain on the same dataset/config (bug-fix-level change),
* **minor** — new data or augmentation/config change that shifts metrics,
* **major** — class-set change or architecture change.

Bump the version whenever the weights change; the checksum guarantees a stale
manifest can't silently describe different weights (a mismatch simply fails to
resolve).
