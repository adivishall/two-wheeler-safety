# Resume notes (internal)

Material for a résumé / portfolio writeup. Everything below is grounded in code
that exists in this repo. **Do not state accuracy/latency numbers you have not
measured** — run `evaluate_model.py` and `benchmark.py` on your weights/data and
fill in the real figures.

## Strongest technical achievements

- Built an **end-to-end CV system**, not a demo detector: detection → multi-object
  tracking → cross-object association → temporal reasoning → OCR stabilization →
  confidence scoring → structured evidence → persistence → async processing →
  review dashboard.
- Solved **cross-object association** (which plate belongs to which rider) with
  **Hungarian assignment** over a geometric cost, replacing a nearest-neighbour
  heuristic that mis-attributed in dense frames.
- Made noisy signals reliable with **temporal methods**: per-vehicle tracking with
  a lifecycle, an **OCR stabilizer** (cross-frame voting + structural plate
  validation), and **state machines** that confirm a violation only after it
  persists — plus a contradiction safeguard for a real, reproduced model bug.
- Got **video speed estimation** right by using **video time** (`frame/fps`), not
  wall-clock time, with calibration, median smoothing, outlier rejection, and an
  uncertainty estimate — and skipping speed entirely when uncalibrated rather than
  reporting a meaningless number.
- Treated **confidence honestly**: a composite 0–1 score with a transparent
  component breakdown, explicitly *not* presented as a calibrated probability
  anywhere in the code, API, or UI.
- Designed for **auditability and correctness**: structured evidence packages,
  per-(vehicle, violation) de-duplication so one bike in view for 200 frames
  yields at most one fine, and a normalized schema with a safe legacy migration.
- Hardened the **web layer**: upload sniffing + size/time caps, traversal-proof
  evidence serving, optional API-key auth, per-IP rate limiting, and a bounded,
  cancellable in-process job manager for long video work.
- Built a **model-free, ~1s test suite** (heavy DL libs isolated behind lazy
  imports) and a **lean CI** that runs green without weights or a GPU — while
  model *accuracy* is evaluated by a separate reproducible tool.

## Suggested résumé bullets (pick 2–3)

- *Built an end-to-end two-wheeler violation-detection system (YOLOv8 + EasyOCR +
  Flask) that tracks vehicles across frames, associates plates to riders via
  Hungarian assignment, and confirms violations with temporal state machines
  before recording confidence-scored, evidence-backed fines.*
- *Eliminated frame-to-frame OCR and detector noise with cross-frame plate voting,
  temporal confirmation, and a contradiction safeguard; added a human-in-the-loop
  review workflow because CV predictions aren't ground truth.*
- *Engineered video speed estimation on video time (not wall-clock) with camera
  calibration, smoothing, and uncertainty — plus a bounded, cancellable
  background-job system for long video processing.*
- *Wrote a reproducible model-evaluation + error-analysis tool (per-class P/R,
  mAP, confusion matrix, helmet-confusion and confidence-vs-correctness analysis)
  and a model-free test suite (299 tests) with lean GitHub Actions CI.*

## Technologies actually used

Python, Ultralytics YOLOv8, EasyOCR, OpenCV, NumPy, Flask, SQLite, gunicorn,
Docker, pytest, ruff, GitHub Actions. Concepts: multi-object tracking, Hungarian
assignment, temporal state machines, OCR post-processing, confidence modelling,
camera calibration for speed, REST API design, background job management.

## Metrics you can legitimately collect

- Per-class precision/recall, mAP@50, mAP@50-95, confusion matrix
  (`evaluate_model.py`).
- WithHelmet↔WithoutHelmet confusion counts; FP/FN examples; confidence
  separation (correct vs wrong).
- Image inference latency, OCR latency, video throughput (FPS, ms/frame), peak
  memory (`benchmark.py`).
- Test count and CI status (299 tests; CI green on a model-free install).

## Limitations to state honestly (don't overclaim)

- Confidence is **not** calibrated to a probability.
- Helmet weights have a documented steady wrong-call failure mode on unfamiliar
  footage; human review is required.
- Speed is an estimate assuming perpendicular motion, needing calibration.
- Single-node design (in-process jobs / per-process state).
- It is a **prototype detection & review assistant**, not legal enforcement, and
  it never resolves vehicle-owner identity.
