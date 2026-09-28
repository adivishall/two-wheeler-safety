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
  1. evaluate on the validation split (+ CIs)              │
  ▼                                                        │
  2. inspect saved failure artifacts                       │
  ▼                                                        │
  3. decide: label bug, data gap, or model limit?          │
  ▼                                                        │
  4. fix labels / add hard cases                           │
  ▼                                                        │
  5. retrain (seeded, config recorded)                     │
  ▼                                                        │
  6. paired A/B on the SAME val split, with CIs ───────────┘
  ▼
  7. promote (or don't) by a rule stated in advance;
     then report the chosen model on test, once
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

## 1. Evaluate on the validation split

```bash
python3 evaluate_model.py \
    --model runs/detect/<run>/weights/best.pt \
    --data eval/clean_splits/data.yaml --split val \
    --name eval_<run>_val_clean --save-artifacts eval/artifacts_val
python3 evaluate_uncertainty.py --split val --name uncertainty_val \
    --models runs/detect/traffic_model-2/weights/best.pt runs/detect/<run>/weights/best.pt
```

Work on **val**. Test is reserved for reporting a model that has already been
chosen — every decision made by looking at test erodes its independence, and
this project's earlier "reject v2" / "promote probe" calls were exactly that
(`AUDIT.md` E4).

A subtlety to keep in mind: Ultralytics picks each run's `best.pt` by validation
fitness, so val numbers are mildly optimistic for *every* candidate alike. That
is a reason to report on test once at the end — not a reason to choose on test.
The clean fix is a third split used only for model selection, which this dataset
is too small to afford (WithHelmet has 26 val instances).

Failure crops come from val too (`--save-artifacts eval/artifacts_val`), so the
inspection in step 2 never leaks test into decisions.

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

The last row matters, but it cuts both ways. The temporal gate absorbs
*independent* per-frame helmet flips well; it does not absorb a rider the model
misreads consistently, and the error budget
([ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) §2) says that after the pipeline has done
its work, what remains is mostly the detector: missed rider boxes and helmet
class confusion own ~90% of the lost end-to-end F1.

### What the current failures say

From `eval/results/manual_error_review.md` (val):

- **Label bugs are common.** 11 of the 16 top-confidence false positives are
  real riders or plates the labels omit; at least 3 of the 16 top confusions are
  mislabelled. Fix these first — until then every comparison is partly measuring
  label noise.
- **Data gaps:** head coverings called helmets; cyclists and pedestrians near
  parked bikes called no-helmet; and, largest of all, plates and heads never
  labelled on the triple-riding source (`label_audit.md`).
- `WithHelmet` has 26–27 held-out instances; its AP CI is [0.20, 0.73]. More
  `WithHelmet` data comes before more epochs — the metric is too noisy to steer by.
- Whether confidence ranks correctness for `WithHelmet` / `TripleRiding` flips
  sign between val and test at these sample sizes — don't tune thresholds for
  them yet.

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

## 6. Paired A/B on the same val split

```bash
python3 evaluate_uncertainty.py --split val --name uncertainty_val \
    --models runs/detect/traffic_model-2/weights/best.pt runs/detect/<candidate>/weights/best.pt
python3 compare_models.py --data eval/clean_splits/data.yaml --split val   # point estimates + latency
```

The first model listed is the baseline; each other model gets a **paired
bootstrap** of its AP difference on the same resampled images, per class and for
mAP@50. A difference is real only if its 95% CI excludes zero. With 26
WithHelmet instances, a point-estimate "win" of 0.05 on that class is noise
until the paired test says otherwise.

Read every class, and the latency and size columns. A model that wins on mean
mAP while dropping a class to zero is a regression, not an improvement
(`traffic_model_helmetfix`: TripleRiding 0.000).

Also compare against **seed noise**: retraining the same recipe with another
seed moves per-class AP by itself ([MODEL_EVALUATION.md](MODEL_EVALUATION.md),
Seed variance). A candidate that differs from the baseline by less than two
seeds of the baseline differ from each other has not shown anything.

---

## 7. Promote, or don't — by a rule stated in advance

The current rule: **promote only if the mAP@50 gain's paired CI excludes zero
and no class regresses significantly.** Then:

1. Report the chosen model on test, once:
   `evaluate_uncertainty.py --split test --thresholds-from eval/results/uncertainty_val.json`.
2. Bump the version and regenerate the manifest (weights hash, dataset
   fingerprint).
3. Point `MODEL_PATH` (or `modules/config.DEFAULT_MODEL_PATH`) at the new
   weights. The runtime stamps the new version onto every evidence package.
4. Re-derive the operating-point error rates from the new model's val confusion
   matrix and re-run `evaluate_temporal.py` and `evaluate_pipeline.py`: the
   decision rules were chosen for the old detector's error rates.
5. Record the decision and its evidence in [DECISIONS.md](DECISIONS.md) and
   [MODEL_EVALUATION.md](MODEL_EVALUATION.md).

Applied so far: `traffic_model_v2_dedup` — not distinguishable overall,
significantly worse on Plate → not promoted. `traffic_model_probe` —
significantly worse overall → not promoted (an earlier test-split
"recommendation" to promote it is withdrawn).

---

## Guardrails

- **Never evaluate on `train`.** If a number looks surprisingly good, check the
  split before believing it.
- **Never tune or choose on `test`.** Choose on `val`; report on `test` once.
- **Never quote a metric you did not generate.** Every number in these docs has a
  command that reproduces it and a JSON file in `eval/results/` behind it.
- **Never promote on a class-average alone.** Read the per-class table.
- **Re-audit hygiene after any data change.**
- **Check the error analysis before retraining at all.** The current budget
  says the detector owns most lost end-to-end F1, but the analysis says the
  cause is *data* (disjoint labels, 27 WithHelmet instances, head coverings,
  label noise) — relabelling beats another training run
  ([ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) §3).
