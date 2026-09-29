# Field evaluation — from synthetic evidence to real traffic-camera evidence

The question this layer exists to answer: **does the system keep working as the
evaluation environment becomes more like real traffic cameras?** Today, the
honest answer for everything past the detector is *unknown*. This page says
exactly what is measured, what is only synthetic, what the new tooling measures
on real data now, and what it will measure once labelled field footage exists.

No number on this page is field accuracy. Where something can't be measured, it
says **NOT MEASURED**.

## 1. Where the evidence stands

| layer | ground truth available | measured on | status |
|---|---|---|---|
| Detector (boxes, classes) | yes — real still images, noisy labels | de-leaked val (352) / test (175) | measured; WithHelmet has 26–27 instances |
| Detector by condition | measured proxies (luminance, sharpness, glare, rider boxes, object size) | same real images | **new** — §4 |
| Tracking, association | no | synthetic scenarios | synthetic only |
| Temporal decision rule | no | simulated riders at val-measured error rates | synthetic only |
| OCR | no plate text on any real image | simulated noise; 8 synthetic renders | synthetic only; field OCR **NOT MEASURED** |
| End-to-end fines | no | synthetic scenarios | synthetic only; field **NOT MEASURED** |
| Speed | no | synthetic trajectories | synthetic only |
| Confidence calibration | no pipeline-produced review outcomes | — | **NOT MEASURED** (§5) |
| Resolution, camera angle, weather, time of day | no | — | NOT MEASURED |

Where errors originate, by the evidence that exists: the synthetic error budget
charges the lost F1 to the detector (a perfect rider detector alone would close
the whole gap; `ERROR_ANALYSIS.md` §2), and the dataset audit says the detector's
limits are mostly *data* — disjoint label sets, 27 held-out WithHelmet
instances, label noise. That ranking is itself an assumption about field
conditions. **The largest uncertainty is not any one stage: it is that no stage
after the detector has ever seen a labelled real frame.**

Modules ready to accept real labels today: `field_data` (the dataset),
`field_eval` (OCR + end to end, per condition), `calibration` via
`Database.review_labels()`, `active_learning` (queue), `detection_stats`
(recipe comparison), `image_conditions` (stratified detector).

## 2. The field dataset (`modules/field_data.py`, `field_dataset.py`)

A directory with `dataset.json` (cameras, sequences), `vehicles.jsonl`
(track-level truth) and `frames.jsonl` (per-frame boxes):

| level | fields |
|---|---|
| camera | id, view (rear/front/side/oblique/overhead), resolution, fps, mounting height, tilt, pixels-per-metre or 3×3 homography, speed limit |
| sequence | id, camera, video or frame directory, fps, start time, lighting (day/dusk_dawn/night), weather (clear/rain/fog) |
| vehicle | sequence, vehicle id, first/last frame, plate text, plate visibility (full/partial/none), rider count, helmet state per rider, violations, occlusion, optional true speed |
| frame | frame path, timestamp, blur (sharp/mild/severe), boxes (rider / plate per vehicle, helmet state, plate text, occlusion) |
| every label | annotator, reviewer, review decision (accepted/corrected/rejected/unreviewed), label confidence (certain/probable/uncertain) |

The loader rejects labels that can't be used: triple riding without three
riders, `no_helmet` without a bare-headed rider, plate text on an invisible
plate, paths outside the dataset, boxes outside the camera frame, unknown enum
values, missing annotators. A plate that doesn't look like an Indian plate is a
*warning* — the pipeline's structural check would reject it, which is itself
worth knowing.

**Splits.** `development`, `validation`, `held_out`, `external`, assigned to
*groups* — a sequence, or with `--group-by camera_day` a camera's whole day —
never to frames (frames of a sequence are near-duplicates). `external` is whole
cameras, named up front: viewpoints the system was never developed on. New
groups are placed by a salted hash, so adding data never reshuffles old data.
The lock (`splits.lock.json`, hash-checked — a hand edit is refused) records
every *sequence's* split: a locked sequence that is renamed, removed, or
re-grouped (its camera or date edited) stops everything rather than being
re-hashed into development. A camera can't be declared external after its
footage was used.

**Evaluation data cannot become training data.** Field training data leaves
only through `export_training`: development frames, label-confidence floor,
rejected labels dropped, frames with a rider of unknown helmet state skipped
whole (dropping just that box would teach it as background). First, every image
is checked against evaluation data, because each check alone has a hole:

- it is not an evaluation file — any labelled frame *or any image in an
  evaluation sequence's frame directory*;
- its bytes match no evaluation file the lock has ever recorded — the lock
  appends the SHA-256 of every evaluation file, so a renamed copy, or a copy of
  a since-deleted original, is still caught; an evaluation frame that is
  missing and was never hashed fails the check closed;
- its objects don't all reappear, at the same places with near-identical crops,
  in one evaluation frame (a re-encode or resize). The perceptual check works on
  object crops, not whole frames: whole-frame hashes match any two moments of a
  fixed camera through the shared background.

`train_traffic.py --field-dataset DIR` runs the same guard over every image a
YOLO config trains on *or selects checkpoints with* (`train` and `val`; image
directories, `.txt` image lists, or lists of either) before training starts, and
a guard that finds nothing to check fails (exit 3). Every export records the
lock hash and a fingerprint of the evaluation side. Frames extracted from an
evaluation *video* are covered only once labelled; exporting them some other way
is outside what the guard can see.

```bash
python3 field_dataset.py validate data/field --check-files
python3 field_dataset.py assign-splits data/field --external-camera cam_07
python3 field_dataset.py coverage data/field        # labelled vehicles per split x condition
python3 field_dataset.py export-train data/field out/field_train
```

## 3. OCR on real plates (`field_eval.ocr_report`) — NOT MEASURED

On the labelled plate boxes of an evaluation split: last-frame read vs most
confident read vs the shipped temporal vote (single reads get the stabilizer's
own cleaning and look-alike correction, so the comparison is fair), with exact
and normalized match, character accuracy, edit distance, invalid-format rate,
abstention, coverage and latency — overall and by lighting, blur, view angle,
plate size (distance proxy), occlusion and resolution.

`python3 evaluate_field.py --dataset data/field --split held_out` runs it.
There is no labelled field dataset, so `eval/results/field_evaluation.md` says
**NOT MEASURED**, and the only OCR evidence remains the simulation
(`ocr_stabilizer_selection.md`) and 8 synthetic renders.

## 4. The field robustness matrix

### What the harness measures (once footage exists) — `field_eval`

The shipped `ViolationPipeline` runs over every frame of each labelled
sequence; results are scored against vehicle-level truth per condition —
lighting, weather, view angle, occlusion, plate visibility, blur, plate size,
resolution, vehicles in frame (density), crossing, camera — for detection
(rider/plate recall, helmet/triple class accuracy), tracking (fragmentation),
association accuracy, OCR, fine precision and recall, wrong plates, missed
violations, false fines and phantom fines. **Every missed or wrong fine is
charged to the first stage that failed** (rider missed → wrong class → track
lost → decision rule not confirming → plate missed → plate mislinked → plate
misread), so each condition's bottleneck is a count, not an opinion. A
violation whose plate isn't visible is expected to be *withheld*; a fine on it
is counted separately, never as a success. Tests break one stage at a time and
check it takes the blame (`tests/test_field_eval.py`).

### What can be measured now — the detector on real images (`conditions.md`)

`evaluate_conditions.py` stratifies the detector on the real val/test images by
conditions *measured* from pixels and boxes (cut points fitted on val, reused on
test). 25 per-class comparisons had ≥ 10 instances on both sides (a binary
condition is tested once, not as two mirror images); 11 cleared 95% on val
where ~1 would by chance, and **2 replicated on test**:

| condition (proxy) | class | val ΔAP vs rest | test ΔAP vs rest | reading |
|---|---|---|---|---|
| single labelled rider box | WithoutHelmet | +0.366 [+0.220, +0.514] | +0.175 [+0.023, +0.337] | **suggestive, not established** — see below |
| dark (luminance < 60) | Plate | +0.102 [+0.025, +0.161] | +0.123 [+0.064, +0.182] | real, but not "night CCTV" — see below |

- **Multi-rider scenes and no-helmet.** On val, images with 2–3 rider boxes
  score WithoutHelmet AP 0.56 vs 0.84 for single-rider images (recall at the
  operating threshold 0.48 vs 0.81). On test, the 2–3 bucket is *not* worse
  (0.80 vs 0.80); test's replication of "single vs the rest" is carried by 3
  images with 4+ riders. Same direction, different buckets: a lead for field
  data, not a result.
- **Dark images and plates.** The darkest images are mostly after-dusk road
  scenes (11 of the 12 darkest val images by eye), but close-up checkpoint
  photos with lit, retro-reflective plates — not distant night CCTV. Plate AP
  there is 0.97 (22 val images) / 0.995 (13 test). Nothing here says the
  detector works at night on a camera.
- **Object size.** Small riders are harder on val (WithoutHelmet −0.156, TripleRiding
  −0.271, paired small vs large); same direction on test, not significant. Plate
  size makes no difference in this data.
- **Not measurable here:** resolution (every image was pre-resized to 640 or
  512 px), camera angle, weather, true time of day.

## 5. Human review → labels → calibration

A review is now a label (`Database.set_review`): a dismissal records **which
stage was wrong** (`wrong_violation`, `wrong_plate`, `wrong_vehicle`,
`duplicate`, `evidence_unusable`, `other`), a wrong plate records the **true
plate**, and the reviewer is stored. `Database.review_labels()` exports decided
reviews — **only violations the pipeline produced** (scored, with an evidence
sidecar). The pre-1.1.0 demo seeder's rows looked like real runs (camera-like
sources, a real model version, hand-picked scores, "reviews"); the first
version of this export admitted them. They are now excluded, with a test.

Two rules for these labels to mean anything: duplicate and unusable-evidence
dismissals are not correctness labels (excluded); and calibration must be fitted
on reviews of *uniformly sampled or exhaustively reviewed* sessions — reviews
the active-learning queue prioritised (low scores, contested plates) would bias
P(correct | score) toward the hard cases.

`calibrate_confidence.py --db traffic.db` judges calibration **out of fold**
(grouped by session), adopts a calibrator only if its Brier score beats the raw
score with a CI excluding zero, needs ≥ 30 correct and 30 incorrect outcomes,
and reports ECE, MCE, Brier, the reliability curve and precision/recall by
threshold. The previous version fitted and scored calibrators on the same rows;
on scores that were *already* calibrated it adopted a calibrator in ~30 of 40
simulated datasets (AUDIT W1). Result on the local database: **0 admissible
outcomes — NOT MEASURED** (`calibration.md`). The score stays a ranking score.

## 6. Active learning (`modules/active_learning.py`, `select_for_labeling.py`)

The labelling budget is spent on information, not volume. Signals (each named
in the queue): helmet/no-helmet contradiction on one rider, a decision score
within 0.1 of its threshold, a confident detection with no label under it, a
confident class that disagrees with the label, a confident class on a source
whose labels never include it (the label gap), and for pipeline output a plate
vote with a runner-up or small margin, weak association, low score, or a
withheld violation. Selection decays repeats of a pattern and skips perceptual
near-duplicates.

**Model signals choose from training data only.** Relabelling the held-out
images the model disagrees with would inflate measured accuracy (model-favoured
corrections get made; shared errors are never found) — the first run of the
selector did exactly that with 77 of 150 picks. Held-out labels are audited on
a seeded, uniform-random sample instead, labelled without seeing predictions.

First queue (`labeling_queue.md`): 150 of 320 candidate training images (from
5,772), plus a 50-image blind audit of val/test. Contradictions, class
disagreements and label-gap items make up 13%, 7% and 7% of the queue against
7%, 4% and 3% of the candidate pool. 79% of the
training pool is offline-augmented copies whose original can't be recovered —
a data-design problem the roadmap addresses. **Whether this queue beats random
selection is NOT MEASURED** until labels come back; the test is issue #4 in
[ROADMAP.md](ROADMAP.md).

## 7. Retraining — only against a measured weakness

No model was trained in this wave. The confirmed weaknesses need *new labels*
(plates on triple-riding images, head coverings, multi-rider scenes), and a
retrain without them would repeat M1–M3. What changed is the test a retrain must
pass: `compare_recipes.py` compares recipes by their **seed means**,
bootstrapping images and seeds together, and refuses a verdict below three seeds
per recipe; a class significantly worse vetoes a better average.

Applied to what exists (`recipe_comparison_val.md`): v2 (de-duplicated data)
against the v1 *recipe* mean rather than the shipped checkpoint is +0.041
[−0.035, +0.117] mAP@50, WithoutHelmet +0.042 [+0.005, +0.079], and v2's
earlier "Plate regression" is gone (−0.010 [−0.074, +0.053]) — with 2 vs 1
seeds, **insufficient seeds**, no verdict (v2's single seed contributes no
seed variance, which is exactly why no verdict is given). The shipped v1
checkpoint is the better of its two seeds. The interval adds image-bootstrap
and between-seed variance under a Welch t; on simulated identical recipes with
three seeds each it promotes 2.0% of the time (nominal 2.5%) and rejects 9.5% —
the per-class veto's deliberate cost.

## 8. Camera configuration and calibration — design

Today the web job never receives calibration, so overspeed never runs in the
app, and the homography calibration in `speed.py` is only used by evaluators.
The design ([ROADMAP.md](ROADMAP.md) #6):

- **Camera profiles** — id, version, resolution, fps, speed limit, zones, and
  pixels-per-metre or a homography with its survey points — stored immutably:
  editing a profile creates a new version; the content hash is the identity.
- **Every session records the profile id, version and hash**; every evidence
  sidecar records the calibration *actually applied* (already true for
  `pixels_per_meter`, scaled for resized frames) and the speed computed then.
- **Historical evidence is never re-interpreted**: speeds are computed at
  processing time and stored; nothing recomputes them from a current profile.
  A test must pin that changing a profile leaves stored records byte-identical
  and their sidecar hashes valid.

## 9. Streaming — assessed, not measured

The video path processes files, not streams: no bounded input queue, no frame
dropping, no end-to-end delay measurement. The benchmark (36.7 FPS on a
334×596 clip with one plate, Apple M4; `benchmark.md`) says nothing about a
1080p camera with a dozen vehicles, where OCR calls scale with plates in view.
**No real-time claim is made.** What a measurement needs is issue #9.

## 10. Reproducing

```bash
python3 evaluate_conditions.py --model $M --device mps                # §4, real images
python3 select_for_labeling.py --device mps --budget 150               # §6
python3 compare_recipes.py --baseline v1 --recipe v1=$M,$S1 --recipe v2_dedup=$V2   # §7
python3 calibrate_confidence.py --db traffic.db --out eval/results     # §5
python3 evaluate_field.py --dataset data/field                         # §3–4 (NOT MEASURED today)
```
