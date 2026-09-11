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

## Versioning policy

`name` identifies the model *line* (`traffic-4class`); `version` is semantic:

* **patch** — retrain on the same dataset/config (bug-fix-level change),
* **minor** — new data or augmentation/config change that shifts metrics,
* **major** — class-set change or architecture change.

Bump the version whenever the weights change; the checksum guarantees a stale
manifest can't silently describe different weights (a mismatch simply fails to
resolve).
