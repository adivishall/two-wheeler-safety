# Pipeline design decisions

The choices that shape what a recorded violation means, with the alternatives,
the trade-offs, where each lives in the code, and the evidence for it. Where an
alternative was never tried, this says so. Shorter records of the same and other
decisions (deployment, config, packaging) are in [DECISIONS.md](DECISIONS.md).

Pipeline, per video frame (`modules/video_detector.py:process_video`):

```
YOLOv8n boxes -> merge rider boxes -> Hungarian plate<->rider -> Hungarian track<->vehicle
   -> OCR the plate crop -> weighted vote across frames -> per-track state machines
   -> confidence score -> evidence package -> SQLite -> human review
```

## 1. One 4-class YOLOv8n detector

- **Choice.** A single YOLOv8n model with classes `Plate`, `WithHelmet`,
  `WithoutHelmet`, `TripleRiding`, trained 15 epochs at 640 px from
  `yolov8n.pt`, seed 0 (`train_traffic.py`; recorded in
  `models/manifests/traffic-4class.json`). Loaded once and cached
  (`modules/detector.py:load_models`, `app.py:get_models`).
- **Alternatives.** An earlier separate 2-class helmet model was retired once
  the 4-class model covered its job (`de821c1`). A larger YOLOv8 variant, or a
  two-stage design (detect riders, then classify head crops), was **not tried**;
  no experiment in the repository compares model sizes.
- **Why.** One forward pass gives every box the rest of the pipeline needs, and
  the nano model keeps inference cheap enough that the pipeline around it is
  the interesting part.
- **Trade-offs.** The model has no notion of a vehicle, so plates and riders
  must be associated afterwards (section 2). `WithHelmet` is weak (test AP@50
  0.387 on 27 instances) and `build_train_split.py` measured it as the most
  duplicated class (2.6x) with the least unique data; whether more capacity or
  more unique data would help more is untested.
- **Evidence.** `eval/results/eval_traffic_model-2_test_clean.json`,
  `eval/results/model_comparison.json` (four checkpoints of the same lineage;
  selected on test, see VERIFICATION.md section 2, row 8).

## 2. Plate-to-rider association: merge, then Hungarian

- **Choice.** First merge overlapping rider-class boxes into one body per
  vehicle (`modules/association.py:merge_bodies`: IoU > 0.5 or containment >
  0.8), then match plates to bodies one-to-one by minimum total cost
  (`hungarian`, `gated_min_cost_matching`, `associate`). Cost = normalised
  centroid distance + 1.5 x (1 - horizontal overlap) + 0.5 x "plate above the
  rider" penalty (`_plate_body_cost`); a plate with no horizontal overlap and
  more than 1.2 body-diagonals away is forbidden.
- **Alternatives.** The original code attached each violation box to the
  nearest tracked plate (`main.py:nearest_plate_id`, kept for its regression
  tests). Greedy cheapest-pair-first matching on the same cost. A learned
  association, or a detector class that boxes rider and plate together, was not
  tried.
- **Why.** Nearest-plate is local: two riders can claim one plate, or a
  neighbour's plate can be stolen. A global one-to-one assignment cannot do
  either, and at a handful of boxes per frame its O(n^3) cost is negligible.
- **Evidence.** On random multi-rider layouts with known ownership (both
  matchers given the identical cost matrix), greedy left at least one wrong
  pair in 18.7% of frames and Hungarian in 7.2%; wrong pairs 11.1% vs 3.5% of
  those emitted (`python3 evaluate_association.py` →
  `eval/results/association_baseline.md`; synthetic). On the shipped synthetic
  scenarios the two never disagree, so those suites do not test this choice.
  Constructed counter-examples:
  `tests/test_association.py::test_hungarian_beats_greedy_nearest`,
  `::test_nearest_plate_mis_assigns_but_association_is_correct`.
- **Trade-offs and open problems.** The weights were set by hand and never fit
  to labelled geometry. The distance gate never fires for a horizontally
  aligned plate (#19). The dataset's helmet classes mark heads
  (`docs/DATASET.md`), while the synthetic scenarios use rider-sized boxes, so
  the cost has not been checked against head-sized boxes. Merge thresholds were
  raised from 0.3 / 0.6 after they merged two crossing vehicles (`483c9ea`).

## 3. Tracking: Hungarian on IoU plus distance, no Kalman filter

- **Choice.** `modules/vehicle_tracker.py:VehicleTracker` predicts each track
  with an EMA velocity, matches tracks to this frame's vehicle instances with
  the same gated Hungarian routine on `(1 - IoU) + 0.5 x normalised distance`,
  and keeps a lifecycle (tentative → confirmed after 3 hits → lost → removed
  after 15 missed frames).
- **Alternatives.** A greedy centroid tracker (`utils/tracker.py`, now used
  only by its tests). A Kalman filter was considered for the crossing failure
  and rejected on evidence: the ID switches came from association merging two
  vehicles before the tracker ran (`483c9ea`, `modules/tracking_eval.py`
  docstring), which no motion model can undo. Appearance features (re-ID
  embeddings) were not tried.
- **Trade-offs.** No appearance features, so a vehicle that leaves view for
  more than 15 frames returns as a new track. On `main` that can produce a
  second fine (de-duplication is per track id); the branch de-duplicates per
  plate and violation (`c4b8c6d`).
- **Evidence.** `eval/results/tracking_comparison.json` (13 → 0 ID switches
  across 8 synthetic crossing scenarios after the merge fix);
  `tests/test_vehicle_tracker.py::test_two_crossing_bikes_keep_distinct_ids`,
  `::test_temporary_disappearance_reuses_id`.

## 4. Plate reading: vote across frames, validate structure, correct look-alikes

- **Choice.** EasyOCR on the tight plate crop each frame; per track, a
  `PlateStabilizer` (`modules/plate_recognizer.py`) normalises each reading,
  corrects look-alike characters only toward a valid Indian plate structure
  within 2 edits (`correct_plate`), weights each reading by OCR confidence
  (x 0.25 if structurally invalid), and elects the top-weighted valid plate if
  it holds at least 0.35 of the vote. Only the elected plate is ever fined.
- **Alternatives.** Use the last frame's read (the original behaviour), or the
  highest-confidence single read. Both were measured against the vote in
  simulation.
- **Why.** Per-frame OCR flickers between look-alikes and occasionally invents
  a plate; the plate is the record's primary key, so a wrong plate fines the
  wrong owner. Abstaining is the safe failure.
- **Trade-offs.** Coverage for accuracy: at 12% simulated character noise the
  vote names a plate for 51% of vehicles and is wrong on 2% of those, against
  85% wrong for last-frame and 30% for best-confidence, which always answer
  (`eval/results/ocr_policy_simulation.json`; simulated noise, not EasyOCR on
  real plates).
- **Known weaknesses on `main`.** One valid read plus one unreadable read
  elects a plate, because `min_observations` counts unreadable reads; an exact
  two-way tie elects by string order. The branch requires three agreeing reads
  and a 0.3 margin (`ae180c1`, `d1d59ce`).

## 5. OCR lock

- **Choice.** Once a track's plate is elected with agreement ≥ 0.90 over ≥ 5
  readings, stop running OCR on that track (`process_video`, `plate_locked`).
- **Why.** OCR is the second-largest per-frame cost after YOLO (profiling in
  `docs/EVALUATION.md` §6, no committed result file); re-reading a settled
  plate should not change the fine.
- **Evidence.** `tests/test_pipeline_integration.py::test_ocr_lock_skips_redundant_ocr_without_changing_the_fine`.
  The throughput gain (+74.9%, `docs/EVALUATION.md` §6) has no committed result
  file; an earlier +172% figure was wrong because the benchmark never turned
  the lock off (`ffb79ec`).
- **Trade-off.** A locked plate is never re-checked, so if the track switches
  to another vehicle the old plate stays. The branch re-reads every 10 frames
  and requires a recent agreeing read before fining (`ae180c1`).

## 6. Temporal confirmation: per-track state machines

- **Choice.** `modules/violation_state.py`: per track, a helmet machine
  (unknown / helmet / no-helmet candidate / confirmed / ambiguous) and a
  triple-riding machine (none / candidate / confirmed / cleared). A frame
  counts only if its box confidence is ≥ 0.3; confirmation needs
  `STREAK_THRESHOLD` (5) consecutive supporting frames; a helmet+no-helmet
  contradiction on one rider resets the count and never confirms. Once
  confirmed, a violation is latched and recorded once, as soon as the plate is
  readable. Overspeed uses its own consecutive-frame streak in `process_video`.
- **Alternatives.** Fining on any single frame (the "naive" policy in
  `modules/pipeline_eval.py`); k-of-n voting or a track-level share of frames
  (adopted on the branch after its temporal experiment); hysteresis that
  un-confirms on contrary evidence (not tried; latching makes de-duplication
  simple).
- **Why.** The detector flickers between helmet classes on one rider
  (observed on real footage in `d8cdee4`), and a single confidently wrong frame
  should not become a fine.
- **Trade-offs.** Short dwell is missed (a rider in view for fewer than 5
  frames is never confirmed). A *steadily* wrong prediction passes, which is why
  review is mandatory. The helmet machine tolerates one blank frame; the triple
  machine does not, and the two thresholds are not separately justified.
- **Evidence.** Synthetic only: window 1 gives helmet precision 0.67 and
  triple-riding 0.80 on the edge-case suites, window ≥ 2 gives 1.00
  (`eval/results/pipeline_evaluation.json`), but those suites have only one-frame
  flickers by construction and three to five decisions per cell. The branch's
  simulation at measured error rates found that a plain streak's false-flag
  rate grows with dwell time when errors are correlated, which is why it adds a
  track-level gate.
- **Photo route.** `/analyze` decides from one image; no temporal rule applies.

## 7. Confidence: a transparent weighted mean, not a probability

- **Choice.** `modules/confidence.py:compute_confidence` averages detection,
  temporal, association and OCR components with equal weights; the UI and API
  call it a score and never a probability.
- **Alternatives.** A product of components (punishes any weak component
  harder); a calibrated probability fitted on review outcomes
  (`modules/calibration.py` exists; there are no admissible outcomes yet, #15).
- **Trade-offs.** Easy to explain and to break down in the UI; but the
  temporal component is 1.0 for every recorded violation (#20), and nothing
  says 0.8 means 80%.

## 8. Speed: video time, calibration required

- **Choice.** `modules/speed.py:SpeedEstimator.estimate` uses
  `frame_index / fps`, median-smooths, drops jumps above 150 km/h and needs 3
  samples over 0.2 s before reporting. Without `pixels_per_meter` no speed is
  computed.
- **Alternatives.** Wall-clock time (the original; made speed depend on the
  machine). Row-dependent scale (`LinearPlaneCalibration`) and a road-plane
  homography (`HomographyPlaneCalibration`) are implemented and evaluated on
  synthetic trajectories.
- **Trade-offs.** The shipped CLI path uses a constant scale, which the
  synthetic evaluation shows misses every overspeeder approaching the camera
  (MAE 41.8 km/h vs 11.3 km/h with a homography,
  `eval/results/pipeline_evaluation.json`). The web app computes no speed at all
  (#13).

## 9. Evaluation design

- **Model-free tests with fakes.** The video pipeline is exercised end to end
  with a fake model, fake OCR reader and stubbed video I/O
  (`tests/test_pipeline_integration.py`), so CI needs no weights. The cost: no
  test covers the real model's output format beyond those fakes.
- **Synthetic scenarios for the pipeline, real images for the detector.** The
  two are reported separately and never combined (README, VERIFICATION.md).
  On `main` the pipeline evaluators re-implement the per-frame loop with
  `confirm_window=3` instead of calling the shipped code; the branch replaced
  them with one shared decision core (`c4b8c6d`).
- **De-leaked held-out split.** Measured with a perceptual hash and rebuilt
  without the duplicates (`5ec3bd7`); see FAILURE_ANALYSIS.md cases 5-7 and #23
  for what that check still misses.
