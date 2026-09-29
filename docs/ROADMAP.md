# Engineering roadmap — the measurable improvement loop

Ten issues, ranked by: scientific uncertainty first, then safety/correctness,
value of the evidence, engineering impact, and difficulty. Every one names the
evidence it starts from and the measurement that closes it. Tooling for most of
them exists (docs/FIELD_EVALUATION.md); what is missing is labelled data and
compute, which is why the order is what it is.

---

## 1. First labelled field dataset, and the first field evaluation

- **Problem.** No stage past the detector has seen a labelled real frame;
  tracking, association, OCR, fines and speed are measured only on synthetic
  inputs. This is the largest uncertainty in the project.
- **Evidence.** `field_evaluation.md`: NOT MEASURED. `FIELD_EVALUATION.md` §1.
- **Hypothesis.** The synthetic ranking (detector owns the lost F1; OCR costs
  coverage, not wrong plates) survives on real footage — or it doesn't, and we
  learn which stage is the real bottleneck per condition.
- **Implementation.** Record from ≥ 3 fixed cameras (one designated EXTERNAL
  before any footage is looked at); label in the `field_data` schema; validate;
  `assign-splits`; run `evaluate_field.py` on `held_out` and `external`.
- **Dataset requirement.** ~30 sequences of 30–60 s; every labelled frame
  exhaustively boxed (every 5th frame); plate text read by a human; lighting
  and weather per sequence; 2 annotators on 20% of held-out for agreement.
  Oversample night, rain, dense traffic and multi-rider scenes (the §4 leads).
- **Metrics.** Per condition: fine precision/recall, wrong plates, phantom
  fines, stage attribution; OCR policy table; inter-annotator agreement (plate
  exact match, helmet-state κ).
- **Acceptance.** The dataset validates with 0 errors; every reported cell has
  ≥ 20 vehicles or is marked insufficient; the report names a bottleneck per
  condition with its error count.
- **Regression risks.** Labelling protocol drift between annotators; leaking
  evaluation footage into development (the lock + guard prevent the second).

## 2. Blind audit of the held-out labels

- **Problem.** Detector metrics are computed against labels known to be noisy,
  and the only estimate of that noise comes from a model-guided sample.
- **Evidence.** 11 of the 16 most confident val "false positives" were
  unlabelled real objects (`manual_error_review.md` — an AI-assisted review of a
  *model-selected* sample, so its rate is biased).
- **Hypothesis.** Held-out label noise is large enough (> 5% of boxes) to move
  per-class AP by more than its CI.
- **Implementation.** Label the 50 images in `labeling_queue_eval_audit.json`
  without seeing predictions; two annotators; adjudicate; re-report with both
  label versions side by side.
- **Dataset requirement.** Those 50 images (uniform-random, seed 0).
- **Metrics.** Missing-box, wrong-class and wrong-box rates per class with
  bootstrap CIs; AP under original vs audited labels.
- **Acceptance.** Noise rates with CIs published; if any class's AP moves by
  more than its CI, re-label the whole held-out split blind.
- **Regression risks.** Relabelling only where the model disagrees — which the
  selector now refuses to do.

## 3. Close the triple-riding label gap

- **Problem.** The triple-riding source never labels plates or heads, so the
  detector learned them as background there: a plate is found on 6% of those
  images vs 65% elsewhere. Triple riding will mostly be *withheld* in the field.
- **Evidence.** `label_audit.md`; ERROR_ANALYSIS F5; 10 `label_gap` items lead
  the current queue.
- **Hypothesis.** Labelling plates and heads on the triple-riding source raises
  plate recall on it to ≥ 50% without lowering any class elsewhere.
- **Implementation.** Label plates/heads on the dst-source training images;
  retrain with ≥ 3 seeds; `compare_recipes.py` against the v1 recipe.
- **Dataset requirement.** All dst-source training images (plus val/test dst
  images relabelled blind, for evaluation).
- **Metrics.** Plate AP/recall on dst; per-class AP overall; share of
  triple-riding vehicles with a fineable plate (field, once #1 exists).
- **Acceptance.** `compare_recipes` verdict `promote`; no class significantly
  worse.
- **Regression risks.** Plate false positives on crowded scenes; TripleRiding
  AP changes from the new co-occurring boxes.

## 4. Head coverings: a fairness audit, then targeted data

- **Problem.** Riders in a dupatta or scarf are read as helmeted — missed
  violations concentrated in one group of people.
- **Evidence.** 4 of the 16 top val class confusions (`manual_error_review.md`);
  ERROR_ANALYSIS F4. No per-group metric exists.
- **Hypothesis.** WithoutHelmet recall on head-covered riders is materially
  lower than on bare-headed riders.
- **Implementation.** Add a `head_covering` attribute to rider labels in the
  field schema; report recall by it; collect/label head-covered riders; retrain
  (≥ 3 seeds).
- **Dataset requirement.** ≥ 100 head-covered riders across held-out and
  external, labelled by people briefed on the distinction.
- **Metrics.** WithoutHelmet recall and helmeted-rider false-flag rate, by
  head covering, with CIs.
- **Acceptance.** The gap is measured with a CI; after retraining it shrinks
  without raising helmeted-rider false flags.
- **Regression risks.** Over-correcting into fining helmeted riders — the error
  that fines an innocent person.

## 5. Settle v2 (de-duplicated data) against the v1 recipe

- **Problem.** The one data-cleaning experiment is undecided: v2 leads the v1
  recipe mean but with 2 vs 1 seeds.
- **Evidence.** `recipe_comparison_val.md`: mAP@50 +0.041 [−0.013, +0.107],
  WithoutHelmet +0.042 [−0.001, +0.081]; v1's checkpoint is its better seed.
- **Hypothesis.** De-duplication improves WithoutHelmet without costing Plate.
- **Implementation.** Train v1 seed 2 and v2 seeds 1–2 (identical args except
  data and seed); `compare_recipes.py` on val; report the winner on test once.
- **Dataset requirement.** None new.
- **Metrics.** Recipe-mean AP per class with seed-and-image CIs.
- **Acceptance.** Verdict `promote` → v2 becomes 1.1.0 of the model with a
  manifest; otherwise v1 stays and the result is recorded.
- **Regression risks.** Promoting on val noise — the three-seed rule and the
  single test report guard against it. ~4 × 3 h of training.

## 6. Camera profiles with immutable calibration; speed ground truth

- **Problem.** The web job never gets calibration, so overspeed never runs in
  the app; speed accuracy is known only in simulation (MAE ~11 km/h, biased
  high, approaching bikes).
- **Evidence.** `app.py` passes no calibration; `pipeline_evaluation.md` § Speed.
- **Hypothesis.** A surveyed homography per camera brings real-world MAE under
  5 km/h at 40 km/h limits.
- **Implementation.** Versioned camera profiles (FIELD_EVALUATION §8) selected
  per job; the session and every sidecar record profile id/version/hash; a test
  that editing a profile leaves historical records and their hashes unchanged.
- **Dataset requirement.** Test rides with GPS-logged speed through each
  camera's view; surveyed ground markers.
- **Metrics.** Speed MAE/bias by camera and distance; overspeed precision.
- **Acceptance.** MAE and bias measured per camera with CIs; overspeed fines
  only on cameras that pass.
- **Regression risks.** A profile change silently re-interpreting old
  evidence; frame-resize scaling (fixed once already).

## 7. Measure whether active learning beats random selection

- **Problem.** The labelling queue is designed for information gain; that is a
  hypothesis.
- **Evidence.** `labeling_queue.md`: contradictions, class disagreements and
  label gaps are over-represented vs the pool — which says the selector does
  what it was built to do, not that it helps.
- **Hypothesis.** Per labelled image, the queue finds ≥ 1.5× the label errors
  that a random sample does, and yields more AP per labelled image.
- **Implementation.** Label the 150-item queue and a random 150 from the same
  pool, blind to which is which; compare error discovery; then retrain each
  (≥ 3 seeds) on the corrected pool.
- **Dataset requirement.** 300 labelled training images.
- **Metrics.** Errors found per item (paired bootstrap); recipe-mean AP.
- **Acceptance.** Queue beats random on errors found with a CI excluding 1.0;
  otherwise switch to stratified random sampling.
- **Regression risks.** Selection bias in *what* gets fixed (it is training
  data only; evaluation stays blind).

## 8. Calibration on real review outcomes, with structured reviews in the UI

- **Problem.** The confidence score is a ranking; there are 0 admissible review
  outcomes to test calibration.
- **Evidence.** `calibration.md`: NOT MEASURED. The API now takes a dismissal
  reason and corrected plate; the dashboard doesn't yet.
- **Hypothesis.** Per violation type, the raw score is monotone but
  miscalibrated; an isotonic map improves out-of-fold Brier.
- **Implementation.** Reason and corrected-plate controls in the review modal;
  double review on a 10% sample; `calibrate_confidence.py --db` per violation
  type once ≥ 30/30 outcomes exist.
- **Dataset requirement.** Reviewed pipeline output, ≥ 30 correct and 30
  incorrect per type.
- **Metrics.** OOF ECE/MCE/Brier, precision/recall by threshold, reviewer
  agreement.
- **Acceptance.** A calibrated probability is exposed — separately from the
  raw score — only if the Brier gain's CI excludes zero.
- **Regression risks.** Calibrating on one camera and applying everywhere;
  review drift.

## 9. Streaming: bounded latency, measured

- **Problem.** The video path processes files; nothing measures end-to-end
  delay, queue depth or frame dropping, so no real-time claim can be made.
- **Evidence.** `benchmark.md` is a 334×596 clip with one plate.
- **Hypothesis.** On 1080p at 25 fps, OCR load scales with plates in view and
  becomes the bottleneck before YOLO does.
- **Implementation.** A replay harness feeding frames at source rate into a
  bounded queue with an explicit drop policy; per-frame timestamps from capture
  to decision; OCR budget per frame.
- **Dataset requirement.** Field sequences from #1 at native resolution.
- **Metrics.** p50/p95 end-to-end delay, drop rate, OCR calls per frame vs
  vehicles in view, memory, decisions changed by dropping.
- **Acceptance.** A published latency/drop curve per hardware; fines under
  frame dropping compared with offline processing of the same footage.
- **Regression risks.** Dropping frames starving the temporal rules (fewer
  observed frames → more withheld).

## 10. Replace offline-augmented copies with online augmentation of originals

- **Problem.** 79% of the training pool is offline-augmented copies whose
  original can't be recovered: a label fixed on an original never reaches its
  copies, and the copies inflate apparent class balance.
- **Evidence.** `labeling_queue.md` § By upstream source; DATASET.md.
- **Hypothesis.** Training on originals with online augmentation matches or
  beats the current recipe, and makes label fixes propagate.
- **Implementation.** Drop `aug_*` images; train with Ultralytics' online
  augmentation; ≥ 3 seeds; `compare_recipes.py`.
- **Dataset requirement.** None new (fewer images).
- **Metrics.** Recipe-mean AP per class.
- **Acceptance.** `promote` or `not distinguishable` with no class worse → adopt
  (simpler data, fixes propagate).
- **Regression risks.** Fewer effective samples for rare classes (WithHelmet).
