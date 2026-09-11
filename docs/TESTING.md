# Testing

The test strategy has one governing idea: **the engineering logic is testable
without the model, so it is tested exhaustively and deterministically; the
model's accuracy is measured separately** (see [EVALUATION.md](EVALUATION.md)).
This keeps the suite fast (~3 s), reproducible, and genuinely green in CI without
a GPU, weights, or the ~2 GB DL stack.

## Running

```bash
pip install -r requirements-ci.txt -c constraints-ci.txt   # pinned, model-free
pytest                        # 299 tests, ~3 s
pytest --cov --cov-report=term-missing   # with branch coverage
```

CI (`.github/workflows/ci.yml`) runs, on Python 3.11 and 3.12: `pip check`,
`ruff check`, `mypy`, then `pytest` under coverage. A local `make check` runs the
same gates.

## Why the suite is model-free

torch / ultralytics / easyocr are behind **lazy imports** — they load only when a
route actually runs inference, which no test does. The integration test drives
the *real* pipeline (tracker → association → OCR stabilizer → state machines →
confidence → evidence → record) with a **fake model and fake reader** and stubbed
OpenCV video I/O, so every line of orchestration is exercised with zero model
dependency. (Decision record: [DECISIONS.md](DECISIONS.md) #11.)

## What is covered (299 tests across 27 files)

| Area | Files | Focus |
|------|-------|-------|
| HTTP API | `test_app.py`, `test_app_auth.py`, `test_security.py` | routes, upload validation, API-key + role gating, analytics/filter/review/payment APIs, security headers, traversal, rate limit, non-leaking errors |
| Database | `test_db.py`, `test_db_lifecycle.py` | record/lookup, normalization, analytics, filtering/pagination, **sort-injection safety**, review + payment lifecycles, sessions, audit, migration |
| Tracking & association | `test_vehicle_tracker.py`, `test_tracker.py`, `test_association.py` | ID stability, occlusion survival, Hungarian assignment, contradiction safeguard |
| OCR | `test_plate_recognizer.py`, `test_plate_info.py`, `test_ocr_eval.py` | temporal voting, look-alike correction, structure validation, eval math |
| Decision logic | `test_violation_state.py`, `test_confidence.py` | helmet/triple state machines, confidence composition |
| Speed & calibration | `test_speed.py`, `test_speed_eval.py`, `test_calibration.py` | video-time speed, homography, ECE/Brier calibration |
| Evidence & manifest | `test_evidence.py`, `test_model_manifest.py` | tamper-evident packages, SHA-256 verify, model versioning |
| Pipeline (integration) | `test_pipeline_integration.py` | end-to-end with fakes, incl. the **OCR-lock** optimization giving identical fines |
| Jobs & perf | `test_jobs.py`, `test_profiling.py`, `test_benchmark.py` | job lifecycle/cancel/expiry, per-stage profiler math |
| Evaluation CLIs | `test_evaluation.py`, `test_evaluate_cli.py`, `test_system_eval.py`, `test_main_helpers.py` | eval primitives, CLI helpers, synthetic system scenarios, the helmet-contradiction **bug regression** |

## Branch coverage

Coverage is measured with **branch** tracking (`pytest --cov`), gated at
`fail_under = 90` in `pyproject.toml`. Measured: **93.6%** overall, with the
core-logic modules strong — confidence & jobs 100%, vehicle_tracker /
violation_state / plate_recognizer 98%, db 97%, speed 96%, association 95%,
validation 100%.

The two model-only modules (`detector.py` loads YOLO/EasyOCR; `plate_ocr.py`
wraps them) are **omitted** from the headline: the model-free suite structurally
can't reach them, and padding the count with model-dependent tests would be
dishonest. Their behaviour is covered by the offline evaluators and the
fake-model integration test.

## Conventions

- Each HTTP test gets a fresh temp `TRAFFIC_DB_PATH` and reloads `app.py`, so
  tests never touch the real `traffic.db` and are order-independent.
- Tests assert on **behaviour and honest numbers**, not on log text.
- New behaviour ships with a test; a fixed bug ships with a regression test
  (the helmet-contradiction case is the canonical example).
- Don't inflate the count — add a test where it buys real confidence.
