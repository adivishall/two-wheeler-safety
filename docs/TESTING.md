# Testing

The test strategy has one governing idea: **the engineering logic is testable
without the model, so it is tested exhaustively and deterministically; the
model's accuracy is measured separately** (see [EVALUATION.md](EVALUATION.md)).
This keeps the suite reproducible and genuinely green in CI without a GPU,
weights, or the ~2 GB DL stack; it runs in well under a minute on a laptop.

## Running

```bash
pip install -r requirements-ci.txt -c constraints-ci.txt   # pinned, model-free
pytest                        # 602 tests
pytest --cov --cov-report=term-missing   # with branch coverage
```

CI (`.github/workflows/ci.yml`) runs, on Python 3.11 and 3.12: `pip check`,
`ruff check`, `mypy`, then `pytest` under coverage. A local `make check` runs the
same gates.

## Why the suite is model-free

torch / ultralytics / easyocr are behind **lazy imports** — they load only when a
route actually runs inference, which no test does. The decision core
(`modules/pipeline.py`) never touches pixels, so it is tested directly; and the
end-to-end tests drive the *real* `process_video`, `analyze_image` and Flask
routes with a fake detector, a fake OCR and scripted video, so every line of
orchestration is exercised with zero model dependency. (Decision record:
[DECISIONS.md](DECISIONS.md) #11.)

The fake OCR in `tests/test_end_to_end.py` reads *pixels*: every plate is painted
into the scripted frame with a unique grey level, and the reader decodes which
plate a crop shows. An OCR call on a crop that is not a plate — a stale box —
reads nothing and is counted. That is how the stale-plate-box bug was caught and
how it stays fixed.

## What is covered (602 tests across 44 files)

| Area | Files | Focus |
|------|-------|-------|
| End-to-end, multi-error | `test_end_to_end.py` | the real video/photo/API paths: side-by-side bikes, plate occlusion (no stale-box OCR), competing OCR reads (withheld), re-entry (one fine), simultaneous violations, noisy detector, intermittent frames, no calibration, downscaled speed, cancellation, persisted record + evidence hashes, withheld violations persisted, photo attribution and abstention |
| Hardening regressions | `test_hardening.py` | role-gated uploads/cancel, payment compare-and-set (deterministic race + threads), clamped limits, rate-limiter memory, RIFF/video sniffing, env thresholds reaching the pipeline |
| HTTP API | `test_app.py`, `test_app_auth.py`, `test_security.py` | routes, upload validation, API-key + role gating, analytics/filter/review/payment APIs, security headers, traversal, rate limit, non-leaking errors |
| Database | `test_db.py`, `test_db_lifecycle.py` | record/lookup, normalization, analytics, filtering/pagination, sort-injection safety, review + payment lifecycles, sessions, audit, migration |
| Tracking & association | `test_vehicle_tracker.py`, `test_tracker.py`, `test_association.py`, `test_tracking_eval.py` | ID stability, occlusion, Hungarian assignment, merge thresholds, contradiction safeguard |
| OCR | `test_plate_recognizer.py`, `test_plate_info.py`, `test_ocr_eval.py`, `test_ocr_temporal_eval.py` | voting, margin, tie abstention, look-alike correction, structure validation, policy/threshold experiments |
| Decision logic | `test_violation_state.py`, `test_temporal_eval.py`, `test_confidence.py` | streak, k-of-n and fraction-gate rules, the rule-selection protocol, confidence composition |
| Pipeline evaluation | `test_system_eval.py`, `test_pipeline_eval.py`, `test_pipeline_integration.py` | scenarios through the shipped core, identity-split scoring, naive-vs-pipeline claim as stated, error-budget ablation (paired streams), OCR-lock equivalence |
| Detector evaluation | `test_detection_stats.py`, `test_evaluation.py`, `test_eval_artifacts.py`, `test_confidence_analysis.py` | AP pinned to Ultralytics `compute_ap`, bootstrap/paired CIs, threshold transfer, matching |
| Data & provenance | `test_dataset_audit.py`, `test_label_audit.py`, `test_provenance.py`, `test_model_manifest.py` | leakage, label integrity, disjoint-label detection, corruption determinism, content fingerprint (location-independent, box-sensitive) |
| Speed & calibration | `test_speed.py`, `test_speed_eval.py`, `test_calibration.py` | video-time speed, homography, ECE/Brier |
| Evidence, jobs, perf, CLIs | `test_evidence.py`, `test_jobs.py`, `test_profiling.py`, `test_benchmark.py`, `test_evaluate_cli.py`, `test_evaluation_cli_smoke.py`, `test_main_helpers.py`, `test_make_ocr_sanity_set.py`, `test_eval_report.py` | tamper-evident packages, job lifecycle, profiler math, every CLI parses and produces well-formed reports |

## Branch coverage

Coverage is measured with **branch** tracking (`pytest --cov`), gated at
`fail_under = 90` in `pyproject.toml`. Measured: **95%** overall; the decision
core `pipeline.py` 91%, `violation_state` 98%, `plate_recognizer` 98%,
`association` 97%, `db` 97%, `video_detector` 93%.

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
