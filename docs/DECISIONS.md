# Engineering decision records

Short records of the non-obvious choices, so the reasoning survives the code.

## 1. Vehicle-level tracking instead of tracking plates alone

**Context.** The detector emits plate boxes and violation boxes independently. A
fine needs to say *this rider on this plate committed this violation*.

**Decision.** Introduce `VehicleTracker`/`VehicleTrack`: group each frame's boxes
into per-vehicle instances (rider body + its plate) and follow those across
frames with stable IDs and a lifecycle.

**Why.** Tracking only plates and then attaching violations to the nearest plate
(the original approach) mis-attributes when bikes are close together. A vehicle
abstraction makes association explicit and de-duplication per vehicle correct.

## 2. Hungarian assignment for plate ↔ rider association

**Decision.** Match plates to riders one-to-one with the Hungarian algorithm over
a geometric cost, rather than greedy nearest-neighbour.

**Why.** Nearest-neighbour can assign two riders to one plate (or steal a plate
from the correct rider) in dense frames. A global optimal assignment respects the
one-plate-per-vehicle constraint. It's O(n³) on the number of boxes per frame,
which is tiny.

## 3. Temporal OCR stabilization

**Decision.** Never fine against a single frame's OCR. `PlateStabilizer` votes
across frames and validates plate structure before electing a stable plate.

**Why.** Frame-to-frame OCR is noisy (motion blur, angle, lighting). A single bad
read would otherwise fine the wrong plate — and the plate is the primary key of
the whole record. Voting + structural validation makes the elected plate robust,
and the agreement score feeds the confidence engine.

## 4. Video speed from video time, not wall-clock time

**Decision.** `SpeedEstimator.estimate` takes `frame_index / fps` (video
seconds), not `time.time()`.

**Why.** Speed = distance / time. If "time" is processing wall-clock, a slower
machine (fewer FPS) inflates the apparent speed of the same vehicle — the
estimate would depend on the computer, not the physics. Video time is a property
of the footage and is reproducible. (A legacy wall-clock method is retained only
for live-capture callers where processing keeps up with the source.)

## 5. Structured evidence packages

**Decision.** Each confirmed violation writes original + annotated frame, plate
and violation crops, and a JSON sidecar — not a single crop named
`{plate}_{violation}_{sec}`.

**Why.** A lone crop is weak, auditable evidence, and second-resolution names
collide when two fines land in the same second. A package with a full confidence
breakdown and frame metadata is defensible and debuggable; a random token in the
filename removes collisions.

## 6. Confidence is a score, not a probability

**Decision.** Combine four component scores into a 0–1 confidence, and everywhere
(code, API, UI) call it a *confidence score*, never a probability or "% likely".

**Why.** The number has **not** been calibrated against ground truth. Presenting
an uncalibrated score as a probability would imply a rigor we haven't
established, and could invite treating a fine as certain. Showing the raw score
plus its component breakdown is honest and still useful for triage/ranking.

## 7. Human-in-the-loop review

**Decision.** Every violation is `pending` until a person marks it `confirmed`
or `dismissed`; automated detection and human confirmation are separate concepts
in the schema and the UI.

**Why.** CV predictions are not ground truth (the model has a documented steady
wrong-call failure mode). A prototype must not imply legal certainty. Separating
detection from confirmation is the honest architecture and mirrors how real
enforcement review works.

## 8. In-process jobs instead of Celery/Redis

**Decision.** Run video analysis in a background thread managed by an in-process
`JobManager` (concurrency cap, cancellation, progress, expiry) — no broker.

**Why.** For a single-node prototype, a broker + worker fleet is operational
overhead with no payoff: it adds services to deploy, monitor, and secure. The
in-process manager delivers the properties that actually matter here (bounded
concurrency, cancellation, no unbounded growth) in ~130 lines and zero infra. The
trade-off — jobs live only as long as the process, and the design assumes a
single worker — is acceptable and documented; a distributed queue is the right
move only once there are multiple nodes.

## 9. Single gunicorn worker + threads in production

**Decision.** Serve with `gunicorn -w 1 --threads 8`.

**Why.** The cached models, the `JobManager`, and the rate limiter are
per-process state. Multiple workers would each hold their own copy, so
`/video_status/<id>` could land on a worker that never ran the job, and the rate
limiter would be per-worker. One worker with threads gives request concurrency
while keeping that state coherent. It's a direct consequence of decision #8.

## 10. Centralized typed config, read at import

**Decision.** One `modules/config.py` built from env via `from_env`, instantiated
at module load in each entry point.

**Why.** Scattered `os.environ` reads made the tunable surface invisible and let
the CLI and web app drift to different defaults. Reading at import (not caching a
singleton at first import) preserves the behaviour tests rely on: patch the env,
re-import the app, get fresh values.

## 11. Model-free tests and lean CI

**Decision.** Keep torch/ultralytics/easyocr behind lazy imports so the test
suite never loads them; CI installs a lean dependency set (`requirements-ci.txt`)
and runs the same suite.

**Why.** The trained weights aren't available in CI, and installing the full DL
stack is ~2 GB and slow/flaky. The engineering logic (tracking, association, OCR
voting, state machines, confidence, DB, API, evaluation math) is all testable
without a model, so CI stays fast, deterministic, and genuinely green — while the
model's *accuracy* is evaluated separately by `evaluate_model.py`.

## 12. Keep the legacy association helpers for regression tests

**Decision.** `main.py` now wraps the shared modern pipeline, but its old pure
helpers (`iou`, `is_contradicted`, `nearest_plate_id`, `clean_plate`) are kept.

**Why.** They encode the regression test for a real bug — a photo where the
higher-confidence helmet call was wrong — that must never silently regress. They
cost nothing to keep and document the contradiction safeguard's origin.

## 13. Keep the flat layout — do NOT adopt a `src/` package

**Decision.** Evaluated moving the code under `src/two_wheeler_safety/` (a
pip-installable package). **Rejected.** Keep the current layout: a `modules/`
package of importable logic plus top-level entry-point scripts (`app.py`,
`main.py`, `demo.py`, `benchmark.py`, `evaluate_*.py`, `train_traffic.py`,
`seed_demo.py`, `calibrate_confidence.py`).

**Why.** A `src/` layout earns its keep when a project is *distributed as a
library* — published to PyPI, imported by other packages, versioned as an API.
This project is an **application**: a web app plus a set of CLIs run in place. It
is never `pip install`-ed as a dependency. Against zero real benefit, the move
has real costs:

* **It breaks every working CLI.** 12 entry points import `from modules.…`;
  repackaging forces a rename (`from two_wheeler_safety.…`) across the tree and
  changes how each script is invoked — exactly the "preserve working CLIs"
  constraint we're told not to violate.
* **The stated wins are already covered another way.** Import hygiene is enforced
  by ruff (`I`/`F`); the accidental-`import config`-vs-`modules.config` ambiguity
  a `src/` layout prevents is instead handled by mypy's
  `explicit_package_bases`; reproducibility is handled by the constraints files.
* It would be complexity added for its own sake — against the project's rule of
  not adding buzzword structure without a measured benefit.

**When to revisit.** If the tracking/OCR/confidence logic in `modules/` is ever
genuinely reused by a *separate* project, extract just that into a package then —
not the whole application.

## 14. One decision core, shared by production and every evaluator

**Context.** `process_video` owned the per-frame decision loop, and
`system_eval` and `pipeline_eval` each carried a private copy. The copies had
drifted — confirm window 3 vs the shipped 5, a frame with no rider box skipped
instead of counted as a miss, OCR text handed over whether or not a plate box
existed — so "pipeline metrics" described a sibling of the shipped code.

**Decision.** `modules/pipeline.py :: ViolationPipeline` owns tracking,
association, OCR voting, state machines, speed, confidence and de-duplication.
`process_video` is the pixel/disk shell around it; the evaluators drive the same
object with synthetic boxes and a plate-box → text lookup.

**Why.** A number is only evidence about the code that produced it. Driving the
real loop immediately exposed four shipped bugs the copies had hidden (stale
plate box, speed under downscaling, duplicate fines on re-entry, a scorer that
counted a duplicate as a second true positive).

## 15. Helmet confirmation: a streak AND a track-level fraction, chosen by experiment

**Decision.** Confirm no-helmet after 5 consecutive supporting frames **and** ≥ 70%
supporting frames over ≥ 12 observed frames. Triple-riding and overspeed keep the
plain 5-frame streak.

**Why.** `evaluate_temporal.py` simulated riders at the detector's per-frame
error rates measured on the *validation* split and swept rules, choosing on one
seed and reporting on another. Finding: a plain streak's false-flag rate *grows
with time in view* under temporally correlated errors (a helmeted rider seen for
2 s gets many chances at one lucky run of flips: 31.5% flagged at moderate
correlation), and k-of-n voting is worse still. A fraction converges instead of
accumulating chances. No rule met the pre-stated ≤ 1% target, so the stated
fallback (minimise the worst case) chose this one: worst design case 1.2% on the
held-out seed, vs 31.5% for the old rule. Re-run at 4× the sample (1,200 riders
per condition) the same rule won; the runner-up is the same gate with a 3-frame
streak (1.67% vs 1.33% on dev), so the *gate* is the robust finding and 3 vs 5
frames is within noise. **Cost, accepted and stated:** a rider seen on fewer than
12 frames is never fined. Triple-riding was not part of the experiment, so it was
not changed.

## 16. OCR vote thresholds chosen by experiment, scored where the pipeline commits

**Decision.** The stabilizer elects a plate only when the winner has ≥ 3 valid
readings of its own, ≥ 35% of vote weight, and a ≥ 0.3 margin over the best
competing valid plate. An exact tie abstains regardless.

**Why.** A wrong plate fines an innocent owner; a withheld fine costs revenue.
`evaluate_ocr.py --simulate --policy-sweep` measures that trade on simulated
plate sequences with look-alike errors and, in half the design conditions,
*consistent* misreads (one glyph read the same wrong way on many frames — what a
real plate image does). It scores each config **at the moment the pipeline
commits** — the first read after which anything is elected, which is when a
held violation is fined — and chooses on one seed, reports on another.

**How the choice was made — including two corrections.**

1. The first run used 200 sequences per condition and picked `agreement ≥ 0.5`;
   the configs it separated differed by 2–4 sequences. At 1,000 sequences that
   pick did not survive (the experiment now runs at 1,000).
2. An independent review then found the experiment scored the plate after *all*
   reads, a later and better-informed moment than the pipeline's commit — and
   that `min_observations` counted junk reads, so one valid reading could elect
   a plate. Scored at the commit point, the 1.0.0 rule names the wrong plate for
   40–86% of vehicles under this noise model. `min_support` was added and the
   thresholds re-selected. No config met the pre-stated ≤ 1% worst case; the
   fallback is noise-aware: every config within 2 standard errors of the lowest
   worst-case wrong-plate rate counts as equally safe, and the most coverage
   wins — because a plain arg-min was, again, being decided by one sequence
   between configs six coverage points apart.

## 17. The photo path uses the same association, and is stricter than video

**Decision.** `/analyze` decides per vehicle through `associate()`; a violation
is recorded only if *that* vehicle's plate is structurally valid; a no-helmet box
overlapping any helmet box (IoU > `CONTRADICTION_IOU`) abstains. Abstentions are
returned with reasons.

**Why.** The old path fined every violation in the frame against whichever plate
was OCR'd last — with two bikes, possibly the wrong rider — and fined against raw
OCR text such as `0285`. A photo has no time axis to resolve contradictions, so it
must be more conservative than video, not less.

## 18. One fine per (plate, violation) per run

**Decision.** In addition to per-track de-duplication, a run fines each
(plate, violation) once; later sightings are reported as suppressed duplicates.

**Why.** A rider who leaves view for longer than the tracker's `max_age` returns
as a new track and used to be fined twice. If two different vehicles share an
OCR'd plate this under-fines — the safe direction.

## 19. Select models on validation; test is for reporting; differences need CIs

**Decision.** `compare_models.py` defaults to the val split; model and threshold
choices are made there. `evaluate_uncertainty.py` reports per-class AP@50 with
image-bootstrap CIs and a *paired* bootstrap for differences. A candidate is
promoted only if its mAP@50 gain has a CI excluding zero and no class has a
significant regression.

**Why.** Earlier decisions ("v2 rejected", "probe wins") compared test-split
point estimates on a class with 27 instances. On val with CIs, v2 vs v1 is not
distinguishable overall (and significantly worse on Plate); "probe" is
significantly *worse*. Choosing on test is tuning on test.

## 20. mAP uses the standard protocol; the operating threshold is separate

**Decision.** Reported mAP comes from `val()` at conf 0.001 / NMS 0.7 (the
standard protocol). The error analysis (confusions, FP/FN) runs at the operating
threshold 0.25. Bootstrap AP uses `predict()` (single-label NMS), the path the
pipeline actually runs, and is reported next to the official number.

**Why.** mAP at conf 0.25 truncates the PR curve; it understated mAP@50 by ~0.06.
`val()` uses multi-label NMS (a box can carry a second class hypothesis), which
flatters confusable classes relative to deployment — most of all WithHelmet —
so both numbers are shown and the gap is explained, not hidden.

## 21. Error budget by oracle ablation at the measured operating point

**Decision.** Inject every stage's fault at its measured per-frame rate (OCR,
unmeasured, is swept), make one stage perfect at a time, and attribute error by
the end-to-end F1 it recovers.

**Why.** The old equal-rate budget compared 30% per-*character* OCR corruption
with 30% per-*box* detector faults and concluded "OCR is 77% of sensitivity" — a
statement about units. Ablation at the real operating point answers the question
an engineer actually has: where would fixing things pay off?

## 22. Dataset version = content fingerprint

**Decision.** One scheme (`modules/provenance.py :: dataset_fingerprint`): hash
of label-file contents plus image names/sizes, independent of location; deep
pixel hashing opt-in. Every evaluation records it, with the weights' SHA-256, git
commit + dirty flag, library versions and hardware.

**Why.** Two schemes existed, both hashing per-class *counts* (a moved box kept
the version), one also hashing the absolute path (moving the repo changed it), so
one dataset had two ids.

## 23. Evidence records what was applied, not what was configured

**Decision.** `process_video` builds the evidence `config_snapshot` from the
arguments it actually ran with; the web job passes every env threshold through.

**Why.** The web job ignored `STREAK_THRESHOLD`, `DETECT_CONF_THRESHOLD`,
`SPEED_LIMIT_KMH` and more while stamping those env values into evidence — so
evidence could claim a threshold the run never used, which is the one thing an
audit trail must not do.

## 24. A locked plate keeps being checked, and a fine needs a recent agreeing read

**Decision.** After the vote locks a plate, OCR re-reads it every 10 visible
frames; a disagreeing valid read drops the agreement below the lock threshold,
so reading resumes. A fine is emitted only if a valid reading agreed with the
elected plate on that track within the last 25 frames (1 s at 25 fps).

**Why.** The tracker can switch identities (one bike passing directly behind
another). A lock that never re-reads keeps the first vehicle's plate on the
track forever, and the second vehicle's violation is fined under it — found by
review, reproduced in `tests/test_pipeline_core.py`. The re-check costs a
fraction of the lock's savings (one read in ten instead of none); the
freshness interlock is a safety rule rather than a tuned threshold.
