# Error-driven retraining loop

How a model gets better here, as a repeatable process rather than a hunch. The
rule the loop exists to enforce: **a checkpoint is promoted because a held-out
measurement says so, never because it is newer or because its name suggests it
should be.**

That rule has already earned its keep. `traffic_model_helmetfix` was trained to
fix helmet detection; it scores **0.000 mAP@50 on `TripleRiding`** — it forgot an
entire violation class — and barely moved helmet accuracy. Nothing but a
comparison on a common split would have caught that
([MODEL_EVALUATION.md](MODEL_EVALUATION.md) §7).

---

## The loop

```
  ┌─ 0. verify split hygiene ──────────────────────────────┐
  │                                                        │
  ▼                                                        │
  1. evaluate on the held-out test split                   │
  ▼                                                        │
  2. inspect saved failure artifacts                       │
  ▼                                                        │
  3. decide: label bug, data gap, or model limit?          │
  ▼                                                        │
  4. fix labels / add hard cases                           │
  ▼                                                        │
  5. retrain (seeded, config recorded)                     │
  ▼                                                        │
  6. compare A/B on the SAME split ────────────────────────┘
  ▼
  7. promote (or don't) — and record why
```

---

## 0. Verify split hygiene first

Never skip this. Every later step compares numbers on a held-out split; if the
split is not held out, the whole loop measures nothing.

```bash
python3 audit_dataset.py --data master_traffic_violation_dataset/data.yaml \
    --write-clean-split eval/clean_splits
```

Exits **non-zero** on leakage, so it can gate a pipeline. Any newly-added
training images must be re-audited — adding hard cases in step 4 is exactly how
a clean split becomes a dirty one.

Current state: the shipped dataset **does** leak (9.8% of test, 8.1% of val), so
work from `eval/clean_splits/data.yaml`, not the raw split.

---

## 1. Evaluate on the held-out test split

```bash
python3 evaluate_model.py \
    --model runs/detect/<run>/weights/best.pt \
    --data eval/clean_splits/data.yaml --split test \
    --name eval_<run>_test_clean \
    --save-artifacts eval --benchmark
```

Use `--split test`, not `val`. Val guided training, so improving on it is
partly self-congratulation. Keep test for decisions only, and resist looking at
it often — every decision made on it erodes its independence a little.

---

## 2. Inspect the failure artifacts

This is the step that distinguishes error-driven retraining from cargo-culted
retraining. Open the folders:

```
eval/false_positives/    predicted a box where the ground truth has none
eval/false_negatives/    ground-truth box missed entirely
eval/class_confusions/   box found, wrong class
eval/low_confidence/     correct, but barely
eval/index.json          class, score, boxes, IoU, model version per case
```

Red box = prediction, green = ground truth. Look at **all** of them, not a
sample — 30 per category is a few minutes' work and the patterns only appear
when you see them together.

---

## 3. Classify each failure before changing anything

Every failure is one of four things, and they need opposite responses:

| what you're seeing | response |
|---|---|
| **Label bug** — ground truth is wrong | fix the label. Do **not** train on it. |
| **Data gap** — genuine case, under-represented | add examples of that case |
| **Model limit** — genuinely ambiguous even to you | accept it; tighten the runtime containment instead |
| **Pipeline containment** — detector wrong but the pipeline absorbs it | leave the model alone |

The last row matters most and is the one most often got wrong. A detector error
that temporal confirmation already suppresses costs nothing downstream — the
error budget ([END_TO_END_EVALUATION.md](END_TO_END_EVALUATION.md) §5) shows
detector class-flips contribute only 15.3% of measured system sensitivity while
OCR contributes 76.8%. **Retraining the detector is often not the highest-value
fix available.**

### What the current failures say

- `WithHelmet` is worst (mAP@50 0.387) and has only **27** test instances. The
  first move is *more `WithHelmet` data*, not more epochs — the metric is
  currently too noisy to steer by.
- The saved confusion crops include a plainly bare-headed rider called
  `WithHelmet` at 0.855 confidence on an easy frame. That is a model limit, not a
  label bug.
- `TripleRiding` confidence carries **no** ranking signal (Spearman −0.20), so
  don't tune its threshold; there is nothing there to tune.

---

## 4. Fix labels, add hard cases

- Correct wrong labels in place.
- Add new images to `train/`, never to `val/` or `test/`.
- Bias new data toward the failure categories from step 3 — hard cases teach more
  per image than easy ones.
- **Re-run step 0 afterwards.** New training images can duplicate held-out ones.

---

## 5. Retrain, reproducibly

```bash
python3 train_traffic.py \
    --data master_traffic_violation_dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 30 --imgsz 640 --batch 16 --seed 0 \
    --name traffic_model_v2 --model-version 1.1.0
```

`--seed` with `deterministic=True` makes the run repeatable; the full config,
dataset content-hash, git commit and weights checksum are written automatically
to a manifest ([MODEL_VERSIONING.md](MODEL_VERSIONING.md)).

**Train from `yolov8n.pt`, not from the previous best**, unless you have a
specific reason. Continuing to fine-tune on a narrower dataset is exactly how
`traffic_model_helmetfix` lost `TripleRiding` entirely. If you do fine-tune, the
A/B in step 6 must check *every* class, not the one you were trying to improve.

Version bumps: **patch** = same data and config; **minor** = new data or config
that shifts metrics; **major** = class-set or architecture change.

---

## 6. Compare on the same split

```bash
python3 compare_models.py --data eval/clean_splits/data.yaml --split test
python3 compare_models.py --data eval/clean_splits/data.yaml --split test \
    --rank-by WithoutHelmet     # rank by the class you actually care about
```

Identical split, identical `conf`/`iou`/`imgsz` for every checkpoint — enforced
by the tool, not by discipline. A checkpoint whose evaluation fails is reported
as failed rather than dropped, so the table can never quietly become "the ones
that happened to work".

Read **every class column**, and the latency and size columns. A model that wins
on mean mAP while dropping a class to zero is a regression, not an improvement.

---

## 7. Promote, or don't — and record why

Promote when the candidate wins on the metric that matters for the decision the
system makes, **and** loses nothing important elsewhere:

1. Bump the version and regenerate the manifest.
2. Point `MODEL_PATH` (or `modules/config.DEFAULT_MODEL_PATH`) at the new
   weights. The runtime stamps the new version onto every piece of evidence
   automatically, so fines remain auditable across the swap.
3. Re-run `python3 evaluate_pipeline.py` — the detector changed, so the
   end-to-end story should be re-stated even though the pipeline logic did not.
4. Record the decision and its evidence in [DECISIONS.md](DECISIONS.md), and
   update [MODEL_EVALUATION.md](MODEL_EVALUATION.md).

**Do not promote inside an evaluation change.** Changing default weights changes
runtime behaviour and deserves its own commit and its own review — which is why
`traffic_model_probe` is currently a *recommendation* in
[MODEL_EVALUATION.md](MODEL_EVALUATION.md) §7 and not the shipped default,
despite winning on both helmet classes.

---

## Guardrails

- **Never evaluate on `train`.** If a number looks surprisingly good, check the
  split before believing it.
- **Never tune on `test`.** Tune on `val`; use `test` to decide.
- **Never quote a metric you did not generate.** Every number in these docs has a
  command that reproduces it and a JSON file in `eval/results/` behind it.
- **Never promote on a class-average alone.** Read the per-class table.
- **Re-audit hygiene after any data change.**
- **Check the error budget before retraining at all.** If the bottleneck is OCR
  (it currently is, at 76.8%), a better detector is not the highest-value work
  available.
