# Interview preparation

Answers derived from this repository — code, tests and generated results — with
the file that backs each claim. If a number is quoted, its source is named; if
something is not measured, the answer says so. Numbers are for
`traffic-4class@1.0.0` and pipeline 1.1.0.

---

## ML / computer vision

### Why YOLO?

A one-stage detector gives boxes for all four classes in one forward pass
(~20–23 ms warm for a 1906×1078 image on an Apple M4, CPU or MPS —
`benchmark.md`) — fast enough to run every
frame of a video, which the temporal logic depends on. YOLOv8n specifically because the dataset is small (the
held-out WithHelmet class has 27 instances): a bigger model would mostly
memorise. Ultralytics also gives a reproducible training/validation loop and the
standard mAP protocol, so the evaluation is comparable to published work. The
detector is deliberately treated as a replaceable component: nothing downstream
depends on YOLO beyond "boxes with a class and a score".

### Why these four classes?

`Plate`, `WithHelmet`, `WithoutHelmet`, `TripleRiding` are what the source
datasets annotate. The design consequence is that **the detector has no notion
of a vehicle**: a plate box and a rider box are independent predictions. That is
why the association layer exists — the system has to decide which plate belongs
to which rider (`modules/association.py`). Two class choices matter:
`WithHelmet` / `WithoutHelmet` are a head-level binary judgement, so glare,
caps and head coverings sit right on the boundary; and `TripleRiding` is one
class, so the system detects "three or more", it does not count riders
(`pipeline_eval.triple_riding_scenarios` encodes that).

### What are the dataset's limitations?

In order of consequence (`docs/DATASET.md`, `eval/results/label_audit.md`):

1. **It is two datasets with disjoint labels.** No labelled image contains a
   TripleRiding box and any other class — in train, val or test. Plates and
   heads on triple-riding images were never labelled, so the model learned them
   as background: it predicts a plate on 65% of helmet-source val images but 6%
   of triple-riding images. Since a fine needs a plate, triple riding will
   mostly be withheld on real footage — and the held-out metrics can't show it,
   because their labels share the gap.
2. **WithHelmet is tiny in held-out data** (26/27 instances): its test AP@50 CI
   is [0.20, 0.73].
3. **Labels are noisy both ways**: 11 of the 16 highest-confidence val "false
   positives" are real, unlabelled riders or plates; at least 3 of the 16 top
   class confusions are mislabelled (`manual_error_review.md` — an AI-assisted
   visual review; re-check the crops yourself before quoting it).
4. **80% of training images are offline-augmented copies**, and 8–10% of
   val/test were near-duplicates of training images before de-leaking.
5. Source and licence are unverified; everything is one region's roads.

### Why mAP@50 vs mAP@50-95?

mAP@50 asks "was the object found?"; mAP@50-95 averages over stricter IoU
thresholds and asks "was it boxed tightly?". Both matter differently here.
Plate has mAP@50 0.89 but mAP@50-95 0.52 on test: plates are found but boxed
loosely — which matters for the OCR crop more than any detection count shows.
For the violation classes, mAP@50 is the operative one: the pipeline needs the
rider box to overlap the rider, not to be pixel-tight. I also report which
protocol: the standard one is conf 0.001 / NMS 0.7. The repo used to compute
mAP at the operating threshold 0.25, which truncates the PR curve — same
weights, 0.727 → 0.769 mAP@50 on test once fixed (`docs/AUDIT.md` E5).

### Which class is weakest and why?

WithHelmet: AP@50 0.426 (`val()`) / 0.415 [0.204, 0.729] (`predict()`, bootstrap)
on test. Causes, from the evidence: very few held-out instances; heavy
augmentation of few originals in training; genuine visual ambiguity (caps,
hoods, dupattas and scarves, shiny helmets under glare — glare alone costs it
−0.16 AP, `robustness_val.md`); and label errors in exactly this class. There is
also a measurement subtlety: `val()` uses multi-label NMS, which lets a box carry
a second class hypothesis; the pipeline uses single-label `predict()`. On
confusable classes the gap is large (val: 0.517 vs 0.417), so I report both.

### How would you improve it?

Data before architecture, because the error analysis says so
(`docs/ERROR_ANALYSIS.md` §3):
1. Label plates and heads on the triple-riding images (or train with per-source
   label masks so unlabelled classes are ignored, not taught as background).
2. More WithHelmet and head-covering riders, plus hard negatives — pedestrians
   next to parked bikes and cyclists in helmets, which the model currently calls
   no-helmet.
3. Fix the held-out labels, then compare models only with paired CIs.
4. Glare and motion-blur augmentation (the two transforms that hurt most).
I tested the obvious data-cleaning idea: training on a de-duplicated split (v2).
With CIs it is statistically indistinguishable from v1 overall — better on
WithoutHelmet, worse on Plate — so it was not promoted. Then I retrained v1's
exact recipe with a different seed: the paired bootstrap called *that*
"significantly worse" on test (mAP@50 −0.076), and its shifts are as big as
v2's. The bootstrap resamples images with the weights fixed, so it can't see
training randomness — comparing recipes needs several seeds each. v2's
WithoutHelmet gain (+0.05, several times the seed shift) is the one lead worth
that run; its Plate "regression" isn't evidence of anything.

### How do you detect leakage?

Perceptual hashing (dHash, Hamming ≤ 5) of every held-out image against every
training image (`audit_dataset.py`). It found 9.8% of test and 8.1% of val were
the same photographs, recompressed. I built de-leaked splits and report on those.
Leakage turned out not to inflate mAP much (+0.006), but the clean number is the
one with an argument behind it. Limits: dHash misses heavy geometric
augmentation, so "de-leaked" means cleaner, not provably clean. The dataset
version is a content fingerprint, so a report says exactly which data it used.

### How should thresholds be chosen?

On validation, against the objective the system actually has, then checked once
on test. For the detector: per-class F1-optimal thresholds on val
(`uncertainty_val.md`). WithoutHelmet's 0.375 transferred to test (fewer false
boxes, no recall lost) and was adopted; WithHelmet's did not (F1 0.58 → 0.42),
so it wasn't — a threshold tuned on 26 instances is noise. For decisions, the
objective is not F1: it is "don't fine innocent people", so the temporal rule and
the OCR vote were chosen by *constrained* objectives (max recall/coverage subject
to ≤ 1% false flags / wrong plates), on a dev seed, reported on another seed.
Neither rule met 1% in every condition; the reports say so and name the
fallback used (lowest worst case — for OCR, anything within 2 standard errors of
it, added after a plain arg-min was decided by a single sequence).
And a choice must be stable to sample size: the first OCR selection flipped when
I re-ran it at 5× the sequences (`DECISIONS.md` #16).

---

## Systems

### Why temporal state?

Because single frames are wrong often and confidently. On val a helmeted
rider's box is called WithoutHelmet 15.4% of the time. A policy that fines on
any frame flags essentially every helmeted rider seen for a second. The state
machines (`modules/violation_state.py`) require persistence. The interesting
part is *which* rule: the experiment in `temporal_confirmation.md` showed a plain
"5 frames in a row" rule's false-flag rate **grows with time in view** when
errors are correlated — a helmeted rider seen for 2 s gets many chances at one
lucky run of 5 flips. k-of-n voting is worse. What fixes it is a track-level
condition that converges instead of accumulating chances: ≥ 70% no-helmet frames
over ≥ 12 observed frames, on top of the streak. The cost is explicit: a rider
seen for fewer than 12 frames is never fined for no helmet.

### Why Hungarian association?

Because the constraint is global. Each rider has one plate and each plate one
rider; greedy nearest-plate can give one plate to two riders or steal a
neighbour's. Hungarian finds the minimum-cost one-to-one matching over a cost of
normalised distance, horizontal overlap (a plate sits *under* its rider) and a
penalty for a plate above the rider, with a gate so an implausible pair stays
unmatched rather than forced. The same algorithm matches tracks to detections
across frames. It is O(n³) in boxes per frame — tiny. The thresholds for
merging rider boxes were raised (IoU 0.3 → 0.5, containment 0.6 → 0.8) after
measuring that two crossing riders reach IoU 0.43 and were being merged into one
vehicle (`tracking_eval.py`).

### Why SQLite?

Single-node, single-writer workload, zero operations, transactional, and the
whole store is one file you can hand to a reviewer. Foreign keys are on,
related inserts share a transaction, and the one race that mattered (payment
status) is a compare-and-set. It stops being right with multiple app instances
or heavy concurrent writes — then it's Postgres, and the in-process job manager
and rate limiter would have to move to shared services too.

### How are duplicate fines prevented?

Three layers. The state machines never un-confirm, so a (track, violation) is
confirmed once. The pipeline emits a decision once per (track, violation). And —
added after the audit found riders were fined twice when they left view longer
than the tracker's 15-frame `max_age` and came back as a new track — once per
(plate, violation) per run, with later sightings reported as suppressed
duplicates (`docs/AUDIT.md` P2). Not prevented: the same video uploaded twice
creates a second session; that is left to human review.

### How are jobs cancelled?

Cooperatively. `JobManager` gives each job a `threading.Event`; the worker's
`cancel_check()` is polled once per frame by `process_video`, which stops at the
next frame boundary, releases the capture and writer, and returns what it has.
Fines already recorded stay recorded (they were fully decided); the session is
persisted as `cancelled` — it used to say `completed`. There is also a
processing-time ceiling (`truncated`). Threads can't be killed safely in Python,
so cooperative is the correct design, not a shortcut.

### How is evidence generated?

At the moment a violation is decided, `build_evidence` writes the original frame,
the annotated frame, a violation crop, a plate crop (only if the plate is in that
frame), and a JSON sidecar: video time, frame index, track id, plate, the plate
vote (agreement, margin, runner-up, observation count), the confidence
breakdown, speed, model and pipeline versions, the thresholds actually applied,
and a SHA-256 of every image. `verify_evidence` re-hashes and reports tampering.
The database links the whole package, so from a violation id you can get to the
exact weights, thresholds and pixels that produced it.

### What happens if OCR fails?

Nothing is fined against a guess. The winning plate needs 3 valid readings of
its own, ≥ 35% of vote weight and a 0.3 margin over any competing valid plate;
an exact tie abstains. Once settled, the plate is still re-read every 10 frames,
and a fine needs a reading that agreed with it in the last 25 frames — so an
identity switch can't carry one rider's plate onto another.
A confirmed violation without a trustworthy plate is *withheld*: reported in the
job result with a reason (`contested`, `low_agreement`, …), written to the audit
log, and counted on the session. On a photo, an unreadable or malformed plate
returns the violation under `abstained` with the reason. The trade was measured
in simulation, scored at the moment a fine is committed: at 4–12% character
noise the 1.0.0 rule named the wrong plate for 40–62% of vehicles, the shipped
one for ≤ 1.8% — by naming a plate for only 34–70% of them. It still misses
the 1% target I set, and a *consistent* misread (the same glyph wrong on every
frame) is what's left, because no vote can out-vote it.

---

## Critical thinking

### What can this system NOT reliably do?

- Fine triple riding — on real footage the plate is usually not detected on those
  vehicles (dataset gap, `label_audit.md`).
- Judge helmets on riders wearing head coverings; it tends to call them helmeted.
- Tell a rider from a person standing next to a parked bike, or a cyclist.
- Read plates reliably: on clean *synthetic* renders EasyOCR reads 37.5% exactly
  (`eval/results/ocr_sanity_check.json`, n = 8); real accuracy is unmeasured.
- Fine riders in view for under ~0.5 s (12 frames) for no helmet.
- Measure speed like radar: MAE ~11 km/h (biased high) for approaching bikes even
  with a homography, in simulation.
- Anything about generalisation to another city or camera: there is no external
  test set.

### What is the largest source of error?

At the measured operating point, the **detector**: in the oracle-ablation error
budget (synthetic scenarios), a perfect rider detector alone would recover the
whole lost end-to-end F1, perfect helmet classification alone 72–80% of it,
perfect plate detection 7–8% — the stages overlap, so those don't add up — and
that holds across an OCR sweep up to half of all reads wrong, because the vote
withholds instead of guessing (`ERROR_ANALYSIS.md` §2). OCR's cost shows up as coverage, not F1. Upstream of
the detector, the largest source is the **data**: disjoint labels and 27
WithHelmet instances.

### What part is model-limited?

Everything the detector cannot see: missed riders, helmet/no-helmet confusions,
head coverings, context errors, plates on triple-riding vehicles, degradation
under motion blur and low light. The pipeline can refuse to act on weak
evidence; it cannot create evidence.

### What part is pipeline-limited?

The decisions made *given* the detections: how long a violation must persist
(and the short-dwell blind spot that creates), association when two riders fully
overlap, whether a parked vehicle should be eligible, speed calibration, and how
conservative the OCR vote is. These are where the audit found real bugs — stale
plate boxes, duplicate fines, unapplied thresholds, speed under downscaling, and
(in the second review) a plate that one valid read could elect or an identity
switch could carry to another rider — all fixed and pinned by tests.

### What did reviewing your own fixes find?

Eight more defects (`AUDIT.md` R1–R8), and the two serious ones were in the
layer I had just called safe: plate identity. `min_observations` counted
unreadable reads, so one valid read plus two junk ones elected a plate; and my
OCR experiment scored the plate after all ten reads, while the pipeline commits
at the *first* election — so it under-counted wrong plates several-fold. The review was
a separate AI agent given only the branch, not my reasoning — which is the
point: it had none of my assumptions. The lesson I took: score a decision rule
at the moment the system acts on it, and have something that didn't write the
fix try to break it.

### How would you know it works on a real traffic camera?

Today I wouldn't: nothing past the detector has seen a labelled real frame, and
the reports say NOT MEASURED rather than guess. What exists is the loop that
would find out. A field dataset schema with annotator provenance. Splits frozen
per sequence, with whole cameras held out, and an export that refuses any image
that is — or copies — evaluation data. A harness that runs the shipped pipeline
over labelled footage and charges every wrong fine to the first stage that
failed, per condition. On the real still images I could stratify the detector
by measured conditions: 27 comparisons, 9 significant on val, 2 replicated on
test — and one of those rests on 3 test images. That taught me to require
replication before calling anything a finding.

### How do you decide what to label next?

By what a label would change, not by volume: model contradictions, scores at
the decision threshold, confident detections with no label, label gaps — with
diminishing returns per pattern so the budget isn't spent on one failure. But
only on training data. My first version also queued held-out images the model
disagreed with; fixing only those would have made the model look better. Held-out
labels get a blind, uniform-random audit instead.

### What would be required before deployment?

1. A labelled, externally sourced **video** test set (other cameras and cities)
   with plate strings — to measure end-to-end fines, real OCR accuracy, and real
   temporal error correlation, none of which exist today.
2. The data fixes above, then a re-selection of every threshold on that data.
3. Calibration of the confidence score against reviewed outcomes, if it is ever
   to be read as a probability.
4. A fairness review (head coverings).
5. Legal review of evidence handling and retention (`docs/PRIVACY.md`), surveyed
   speed calibration per camera, and a human confirming every fine — which the
   system already enforces with `review_status`.
6. Operationally: Postgres, a real job queue and shared rate limiting for more
   than one instance; auth keys configured.
