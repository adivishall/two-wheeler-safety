# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses a single
`PIPELINE_VERSION` (`modules/config.py`) stamped onto every evidence package.

The trained weights, datasets, and generated evidence are **not** shipped in the
repo (all gitignored) — a release is the *code*, evaluated against locally-held
weights whose provenance is recorded in the model manifest
([docs/MODEL_VERSIONING.md](docs/MODEL_VERSIONING.md)).

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
- OCR is skipped once the stabilizer has locked a high-confidence plate — a
  **measured +172% video throughput** on the demo clip with a **byte-identical**
  recorded fine (disable with `OCR_LOCK_CONFIDENCE=1.1`).
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
