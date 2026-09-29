# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses a single
`PIPELINE_VERSION` (`modules/config.py`) stamped onto every evidence package.

The trained weights, datasets, and generated evidence are **not** shipped in the
repo (all gitignored) — a release is the *code*, evaluated against locally-held
weights whose provenance is recorded in the model manifest
([docs/MODEL_VERSIONING.md](docs/MODEL_VERSIONING.md)).

## [1.1.0] — 2026-09-28 — audit, shared decision core, decisions by experiment

A hostile review of 1.0.0 ([docs/AUDIT.md](docs/AUDIT.md)) found 30 defects in
shipped behaviour and in published claims, and an independent review of the
fixes found 8 more. This release fixes them, makes the
evaluation measure the shipped code, and replaces hand-set decision thresholds
with ones chosen by stated experiments. Several 1.0.0 headline numbers are
withdrawn; the replacements are below.

### Changed — behaviour (evidence records `pipeline_version: 1.1.0`)

- **One decision core** (`modules/pipeline.py :: ViolationPipeline`) shared by
  the web job, the CLI, the benchmark and every evaluator.
- **Helmet confirmation**: 5-frame streak **and** ≥ 70% no-helmet over ≥ 12
  observed frames (was: streak only). Chosen by `evaluate_temporal.py`; riders
  seen < 12 frames are no longer fined.
- **OCR vote**: the winner needs ≥ 3 valid readings of its own and a ≥ 0.3
  margin over any competing valid plate; exact ties abstain (was: 2 readings of
  any kind — one valid read among unreadable ones could elect — and ties broken
  by string order).
- **Plate lock**: a locked plate is re-read every 10 frames, and a fine needs a
  reading that agreed with the plate within the last 25 frames (was: locked for
  the track's life, so after an identity switch a rider could be fined under
  another rider's plate).
- **No-helmet frame floor** `HELMET_MIN_CONF` 0.3 → 0.375 (val-selected).
- **Photo path** decides per vehicle through the association layer and abstains,
  with a reason, when a plate is not a valid plate.
- One fine per (plate, violation) per run; withheld confirmations are persisted
  (audit log + `sessions.violations_withheld`).
- `DETECT_DEVICE` (default auto): inference previously always ran on the CPU on
  Apple silicon.

### Fixed

- OCR and speed sampled a stale plate box after the plate vanished.
- Speed read at 2/3 of truth when frames were downscaled.
- Riders leaving view > 15 frames were fined twice.
- Documented thresholds were not applied by the web job but were stamped into
  evidence as if they were.
- Cancelled / time-limited jobs were recorded as completed; "processing FPS" was
  the source video's frame rate; video fines didn't link their evidence sidecar,
  and the sidecar couldn't be served, so the review modal's breakdown never
  loaded.
- Upload and cancel routes were not role-gated with auth on; payment status
  update raced; `recent=-1` returned the whole table; rate-limiter memory grew
  without bound; RIFF/video uploads were not content-checked.
- Demo data claimed a real model version and showed photos of other plates.
- `/detect` honoured only `DETECT_API_KEY`, so it was open when role keys were
  configured.
- A violation whose plate became readable only on frames without a rider box
  was never fined.

### Evaluation — corrected

- Pipeline metrics now drive the shipped code: end-to-end **precision 1.0 /
  recall 0.9** (the miss is the documented short-dwell cost) — was 1.0 / 1.0 from
  a private evaluator loop with different settings.
- mAP at the standard protocol: test mAP@50 **0.769** / mAP@50-95 **0.558** (was
  0.727 / 0.532 computed at conf 0.25).
- Model selection on val with paired bootstrap CIs (`evaluate_uncertainty.py`):
  v2 "rejected for a WithHelmet regression" was noise (not distinguishable; not
  promoted); "promote probe" withdrawn (significantly worse).
- Seed variance: v1's recipe retrained with seed 1 is "significantly worse" on
  test by the same bootstrap (mAP@50 −0.076), as large as the recipe differences
  above. Promotion now requires ≥ 3 seeds per recipe; the headline numbers are
  v1's checkpoint, the better of two seeds.
- Error budget by oracle ablation at the measured operating point, paired random
  streams: the detector owns essentially all lost end-to-end F1 — rider boxes
  54–56%, helmet class 40–42%, plate boxes 4% (was "OCR is 76.8%", a units
  artefact).
- OCR vote thresholds selected by simulation **scored at the moment a fine is
  committed** (the first election), not after all reads: there the 1.0.0 rule
  names the wrong plate for 40–62% of simulated vehicles at 4–12% character
  noise, the 1.1.0 rule for ≤ 1.8%, at 34–70% coverage.
- "Pipeline beats the detector at every noise level" restated precisely: more
  precise at every level; F1 advantage in the measured range, reversed at 40%.

### Added

- `evaluate_temporal.py`, `evaluate_uncertainty.py`, `evaluate_robustness.py`,
  `audit_labels.py`, `evaluate_ocr.py --policy-sweep`; provenance on every
  result (`modules/provenance.py`); one content-based dataset fingerprint.
- Label-coverage audit: the dataset's two sources have disjoint label sets.
- Visual review of the top-confidence val errors (`manual_error_review.md`;
  AI-assisted, not yet re-checked by a person).
- `benchmark.py`: explicit device, cold/warm, input hashes, micro-benchmarks
  (DB, decision core, API overhead), committed output.
- Review modal shows how each violation was decided; `GET
  /api/violations/<id>/verify` re-hashes its evidence.
- `tests/test_end_to_end.py` (real video/photo/API paths through multi-error
  scenarios) and `tests/test_hardening.py`; 466 → 600 tests.
- Docs: AUDIT, ERROR_ANALYSIS (taxonomy + budget; replaces ERROR_BUDGET),
  INTERVIEW; rewritten README, MODEL_EVALUATION, END_TO_END_EVALUATION,
  EVALUATION, RESUME.

## [1.0.0] — 2026-09-16 — CV evaluation & model quality

The application was already evaluated; the **vision system** was not. This
release makes every layer measurable and separates model quality from pipeline
quality so neither can stand in for the other.

### Added

- **Dataset split hygiene** (`audit_dataset.py`, `modules/dataset_audit.py`) —
  perceptual-hash leakage detection between train and held-out splits, and
  de-leaked split generation. Resolves a gap `docs/DATASET.md` had recorded as
  UNVERIFIED: **9.8% of the test split and 8.1% of val are near-duplicates of
  training images.** Measured impact on mAP@50: +0.006, i.e. the leakage was not
  inflating the headline metric.
- **Held-out test evaluation.** The `test` split existed but had never been
  used; every previously quoted number came from `val`. De-leaked test:
  **mAP@50 0.7265, mAP@50-95 0.5320.**
- **Failure artifacts** (`modules/eval_artifacts.py`) — representative false
  positives, false negatives, class confusions and low-confidence detections
  saved as annotated context crops with a JSON index.
- **Per-class confidence curves** (`modules/confidence_analysis.py`) — score vs
  accuracy, binned. Confidence ranks correctness for `WithoutHelmet`
  (Spearman 1.00) and **not at all** for `TripleRiding` (−0.20).
- **Model A/B comparison** (`compare_models.py`) — every checkpoint on one
  split with identical settings. Caught `traffic_model_helmetfix` scoring
  **0.000 mAP@50 on TripleRiding**: catastrophic forgetting that would have
  silently disabled a violation class.
- **OCR decision-policy benchmark** (`modules/ocr_temporal_eval.py`) —
  single-frame vs temporal voting under a stated character-noise model. At 12%
  noise the last-frame policy names the wrong plate 85% of the time,
  best-confidence 30%, the shipped stabilizer **2%** — temporal voting converts
  OCR errors into abstentions rather than raising accuracy.
- **Per-violation pipeline metrics and error budget**
  (`modules/pipeline_eval.py`) — helmet and triple-riding decisions scored as
  classifications over edge cases; a confirmation-window sweep showing
  single-frame fining drops helmet precision to **0.67**; and equal-rate fault
  injection identifying **OCR as 76.8% of system sensitivity**.
- **Pipeline vs single-frame detector benchmark** — on identical inputs the
  naive policy averages **1.50 false positives even with a perfect detector**
  versus **0.00** for the pipeline; the entire advantage is precision.
- **Unified evaluation CLI** (`evaluate_pipeline.py`) writing JSON + Markdown +
  CSV to `eval/results/`, plus a dashboard panel that reads those files and
  never computes or invents a metric.
- **Machine-readable dataset manifest** (`dataset_manifest.py`,
  `data/DATASET_MANIFEST.md`) with provenance gaps marked `UNVERIFIED`.
- **Docs**: `MODEL_EVALUATION.md` (detector only), `END_TO_END_EVALUATION.md`
  (pipeline only), `RETRAINING_LOOP.md`, `RESUME.md`.
- **CI** now smoke-tests every evaluation entry point on each push, with no GPU,
  weights, dataset or network.

### Fixed

- **`benchmark.py` never forwarded the OCR-lock setting to `process_video`**, so
  the documented `OCR_LOCK_CONFIDENCE=1.1` baseline silently did nothing and
  both arms of the published A/B ran with the lock on. Re-measured correctly:
  **+74.9%** (28.7 → 50.2 FPS, 120 → 5 OCR calls, identical recorded fine) —
  not the previously published +172%. `docs/EVALUATION.md` carries an explicit
  correction rather than a silently edited number.
- Generated clean-split `data.yaml` files lacked the `train:` key Ultralytics
  requires.
- `evaluate_model.py` reloaded the model after `val()` only when `val()`
  *succeeded*, so a failed `val()` took the error-analysis pass down with it.
- Error-budget stage `association_ocr` renamed to `plate_recall`; it injects
  plate dropout, not an association failure.

### Changed

- README separates **model**, **pipeline** and **application** performance into
  sections that are never combined.
- Tests: 299 → 466, still model-free and ~5 s. Coverage 94.4%.

## [1.0.0-rc1] — 2026-09-11

First release candidate. The system detects two-wheeler violations (no-helmet,
triple-riding, overspeed) in photos and video, reads and stabilizes the plate,
confirms violations over time, scores confidence, packages auditable evidence,
and surfaces everything in a review dashboard — with the engineering *around* the
model (association, temporal confirmation, OCR voting, calibration, evidence
integrity) as the substance.

Measured results (real runs, local weights `traffic-4class@1.0.0`): detector
mAP@50 **0.697** / mAP@50-95 **0.502**; end-to-end synthetic system precision =
recall = **1.0** across 8 scenarios. Full detail in
[docs/EVALUATION.md](docs/EVALUATION.md); failure modes in
[docs/ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md).

### Added
- **Detection & pipeline:** YOLOv8 4-class detection + EasyOCR; `VehicleTracker`
  with lifecycle + stable IDs; Hungarian plate↔rider association; temporal
  `PlateStabilizer` with structure-valid look-alike correction; helmet/triple
  state machines with a self-contradiction safeguard; video-time speed with
  homography calibration; four-component confidence score.
- **Web app & dashboard:** photo/video/plate flows; a review dashboard with
  overview stats, a filterable/paginated violations table, an evidence viewer,
  and a `pending → confirmed | dismissed` human-review workflow.
- **Dashboard analytics + per-session drill-down:** violations over time,
  confidence distribution, review outcomes (confirmation rate), plate-recognition
  rate, and processing throughput/model stats; a per-run session detail view
  (`/api/analytics`, enriched `/api/sessions/<id>`).
- **Evidence integrity:** SHA-256 over each evidence package + verification.
- **Persistence:** normalized SQLite (vehicles/violations/evidence/detections/
  sessions/audit_log/jobs) with backward-compatible migration of the legacy flat
  table; role-based auth (viewer/reviewer/admin, off until keys set); payment
  lifecycle; append-only audit log; durable job records.
- **Reproducibility & quality gates:** pinned constraints
  (`constraints-ci.txt` / `constraints-runtime.txt`); scoped mypy config;
  branch-coverage gate (93.6%, `fail_under=90`); CI runs `pip check` + ruff +
  mypy + coverage on Python 3.11/3.12.
- **Tooling:** `benchmark.py` per-stage profiler; a one-command
  dependency-light demo (`demo.py` / `make demo`) + `Makefile`.
- **Docs:** INSTALL, TESTING, SECURITY, ERROR_ANALYSIS added; API, PRIVACY,
  EVALUATION, DECISIONS expanded and audited against the code.

### Changed
- OCR is skipped once the stabilizer has locked a high-confidence plate, with a
  **byte-identical** recorded fine (disable with `OCR_LOCK_CONFIDENCE=1.1`).
  *(The throughput figure originally quoted for rc1 was a benchmark-harness
  artifact; the corrected measurement is +74.9% — 28.7 → 50.2 FPS — see the
  [1.0.0] Fixed section and `docs/EVALUATION.md`.)*
- `/detect` errors and failed video jobs now return generic, non-leaking
  messages; the API returns JSON (not HTML) for 404/405/413/500.

### Security
- Parameterized SQL with a whitelisted sort column; upload sniffing + size caps;
  traversal-proof evidence serving; per-IP rate limiting; header-based API keys
  (so CSRF is structurally N/A); debugger off by default. Reviewed and tested in
  [docs/SECURITY.md](docs/SECURITY.md) + `tests/test_security.py`.

### Notes
- No model weights, datasets, or evidence are shipped. Install per
  [docs/INSTALL.md](docs/INSTALL.md); tag `v1.0.0` on `main` once this RC is
  validated.
