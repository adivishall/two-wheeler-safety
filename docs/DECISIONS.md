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
