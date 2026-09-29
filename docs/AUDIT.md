# Audit register — v1.1.0

A hostile review of the v1.0.0 system (30 findings, plus one found later by a
seed-variance run, E12), then an independent review of the fixes themselves
(8 more, §"Found by an independent review"). Both were done with an AI coding
assistant: the first taking four reviewer perspectives in turn (an ML engineer,
a backend engineer, a computer-vision interviewer and a skeptical hiring
manager), the second a separate AI agent given only the branch. No human
reviewer was involved; every finding was confirmed by running code, and each
row names its evidence. Each finding is a defect in shipped behaviour or in a published claim,
with the evidence that it was real, the commit that fixed it, and the test or
result file that keeps it fixed. Findings that are *limitations* rather than
defects live in [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md).

Severity: **H** = could fine the wrong person, corrupt the record, or falsify a
headline claim; **M** = wrong behaviour or misleading number with bounded
impact; **L** = hardening.

## Pipeline correctness

| # | sev | finding | evidence | fix | pinned by |
|---|---|---|---|---|---|
| P1 | H | **Photo path fined every violation against the last plate OCR'd.** With two bikes in a photo, the compliant rider's plate could be fined for the other rider's violation; raw OCR text (`0285`) was accepted as a plate. | code read; reproduced with a scripted two-bike photo | `7feb4f0` — per-vehicle decision via `associate()`, structural plate required, abstentions returned | `test_end_to_end.py::test_photo_with_two_bikes_fines_the_violators_own_plate`, `…_without_a_valid_plate_abstains` |
| P2 | H | **Duplicate fines on re-entry.** De-dup was per track id; a rider out of view > 15 frames came back as a new track and was fined again. The evaluator scored the duplicate as a second true positive. | `reappears_after_exit` scenario: 2 fines before the fix | `b497384` — one fine per (plate, violation) per run; scorer counts identity splits as duplicates | `test_system_eval.py::test_rider_who_returns_under_a_new_track_is_fined_once`, `test_end_to_end.py::test_reappearing_vehicle_is_fined_once` |
| P3 | H | **Stale plate box.** After a plate vanished its last box was still cropped and OCR'd every frame (feeding the vote pixels from where the plate *used* to be) and fed to the speed estimator (a zero-motion sample, then a catch-up spike). | e2e harness counts OCR calls on non-plate pixels: 12 per occlusion before the fix | `b497384` — `VehicleTrack.plate_visible`; OCR, speed and evidence crops require it | `test_end_to_end.py::test_plate_occlusion_never_ocrs_a_stale_box` |
| P4 | H | **Speed under downscaling.** `pixels_per_meter` measured on source frames was applied to resized frames: a 1920-px video processed at 1280 reported 2/3 of the true speed. | e2e: 22.5 km/h reported for a true 45 km/h | `b497384` — calibration scaled by the resize factor; evidence records both values | `test_end_to_end.py::test_speed_calibration_is_scaled_with_the_frame` |
| P5 | H | **Documented thresholds not applied; evidence claimed they were.** The web job ignored `STREAK_THRESHOLD`, `DETECT_CONF_THRESHOLD`, `HELMET_MIN_CONF`, `TRIPLE_MIN_CONF`, `SPEED_LIMIT_KMH`, `MAX_VIDEO_WIDTH` while stamping the env values into evidence. | code read | `7feb4f0` — `pipeline_config_from_detection`; snapshot built from applied values | `test_hardening.py::test_detection_env_vars_reach_the_pipeline`, `test_end_to_end.py::test_applied_thresholds_reach_the_detector_and_the_evidence` |
| P6 | M | **OCR tie broken by string order.** An exact two-way split elected whichever plate sorted last. | `test_ocr_temporal_eval.py` pinned it as known behaviour | `b497384`, `5131a03` — ties abstain; margin required | `test_ocr_temporal_eval.py::test_two_way_tie_abstains` |
| P7 | M | **Confirmed violations without a plate vanished silently.** | code read | `b497384`, `5131a03` — reported as `unfined_confirmations`, persisted to the audit log and `sessions.violations_withheld` | `test_end_to_end.py::test_withheld_violations_are_persisted_not_just_logged` |

## Evaluation validity

| # | sev | finding | evidence | fix | result |
|---|---|---|---|---|---|
| E1 | H | **The pipeline metrics did not measure the shipped pipeline.** Two evaluators re-implemented the per-frame loop with a different confirm window (3 vs 5), different handling of frames with no rider box, and OCR text supplied whether or not a plate box existed. | code diff between the three loops | `b497384` — `ViolationPipeline` shared by production and evaluators | `pipeline_evaluation.md` now reports precision 1.0 / recall 0.9 (the miss is a documented short-dwell case) instead of 1.0 / 1.0 |
| E2 | H | **"Pipeline beats a single-frame detector at every noise level" held only for the evaluator's private window.** With the shipped window it lost at 30% noise. | `test_pipeline_eval.py` failed once the evaluator used the shipped config | claim restated precisely: more *precise* at every level; F1 advantage in the measured noise range, reversed at extreme noise | `test_pipeline_is_more_precise_at_every_noise_level`, `test_pipeline_wins_f1_in_the_measured_noise_range_only` |
| E3 | H | **Error budget compared incomparable units.** 30% per-*character* OCR corruption vs 30% per-*box* detector faults produced "OCR is 76.8% of system sensitivity". | a 10-glyph plate at 30%/char reads cleanly 2.8% of the time | `54cae20`, `9d9556b` — oracle ablation at the val-measured operating point, OCR swept, paired random streams | the detector owns ~90% of lost end-to-end F1 (rider recall, then helmet class) across the whole OCR sweep |
| E4 | H | **Models chosen on the test split, from point estimates.** "v2 rejected", "promote probe" were test-split comparisons on a class with 27 instances. | `model_comparison*.md` headers | `9ec9813` — select on val; paired bootstrap; promotion rule stated in advance | v2 vs v1: not distinguishable (Δ mAP +0.021 [−0.022, +0.063]); probe significantly *worse* (−0.051) — recommendation withdrawn |
| E5 | H | **Headline mAP computed at the operating threshold.** `val()` ran at conf 0.25, truncating the PR curve. | same weights: 0.727 → 0.769 test mAP@50 under the standard protocol | `9ec9813` | `eval_traffic_model-2_test_clean.json` |
| E6 | M | **Temporal window "chosen" on scenarios whose flickers were one frame by construction** — "window 2 fixes it" was built in. | scenario code | `b497384` — `evaluate_temporal.py`: simulated riders at measured error rates, dev/test seeds | found the streak's false-flag rate *grows with dwell* under correlated errors; shipped rule adds a track-level gate |
| E7 | M | **An OCR selection that did not survive a larger sample.** The first sweep (200 sequences) picked a config separated from its rivals by 2–4 sequences. | rerun at 1,000 sequences chose differently | `5131a03` — larger default; reversal recorded (DECISIONS #16) | `ocr_stabilizer_selection.md` |
| E8 | M | **Tracking metric used "last writer wins"** when two tracks overlapped one rider, inflating ID switches in crossings. | `system_eval.evaluate_tracking` comment: "scenarios avoid ties" | `b497384` — one-to-one Hungarian track↔GT matching | crossing: 0 switches, association 0.95 |
| E9 | M | **Two "dataset versions" for one dataset, neither content-based.** Both hashed per-class *counts*; one also hashed the absolute path. | `sha256:8d09…` (model manifest) vs `sha256:bef0…` (dataset manifest) for the same data | `9ec9813` — one content fingerprint | `test_provenance.py::test_fingerprint_changes_when_a_box_moves_but_counts_do_not` |
| E11 | M | **Benchmark labelled CPU numbers "mps".** Ultralytics only auto-selects CUDA; on Apple silicon every predict ran on the CPU while `benchmark.py` recorded the requested device. The app and CLIs never used the GPU either. | `model.predictor.device` after a default predict: `cpu` | `5e08614` — `DETECT_DEVICE` resolved and passed everywhere; results record the device used | `benchmark.md` measures CPU and MPS side by side |
| E10 | M | **AP replica bug caught before use.** An older compute_ap formula credits a spurious triangle above max recall (0.75 vs 0.495). | pinned against `ultralytics.utils.metrics.compute_ap` | `9ec9813` | `test_detection_stats.py::test_ap_curve_matches_ultralytics_compute_ap` |
| E12 | M | **A checkpoint-level test used as a recipe comparison.** The paired image bootstrap holds the weights fixed; M1–M3 verdicts ("v2 significantly worse on Plate") treated it as evidence about training changes. | v1's recipe retrained with only the seed changed is "significantly worse" on test (mAP@50 −0.076 [−0.155, −0.003]); its Plate shift (−0.024) matches v2's "regression" | promotion needs ≥ 3 seeds per recipe; M1–M3 re-graded; headline numbers labelled as the better of two seeds | `uncertainty_val.md`, `uncertainty_test.md` (`traffic_model_seed1`) |

## Application and data integrity

| # | sev | finding | fix | pinned by |
|---|---|---|---|---|
| A1 | H | Upload routes that create fines (`/analyze`, `/analyze_video`) and `/video_cancel` were open even with role keys configured. | `7feb4f0` — reviewer role required when auth is on | `test_hardening.py::test_upload_routes_that_record_fines_need_the_reviewer_role` |
| A2 | M | Cancelled or time-limited jobs were persisted as `completed`. | `7feb4f0` — `cancelled` / `truncated` | `test_end_to_end.py::test_cancelled_video_job_is_not_recorded_as_completed` |
| A3 | M | Dashboard "processing throughput" showed the source video's frame rate. | `7feb4f0` — measured frames / wall second | `test_end_to_end.py::test_video_job_persists_a_traceable_record` |
| A4 | M | Video fines stored one image path; the sidecar (hashes, model version, confidence breakdown) was unreachable from the DB. | `7feb4f0` — whole package linked | same test verifies the sidecar's hashes through the API |
| A5 | M | Payment status: validate-then-write outside the transaction; concurrent paid/cancelled could overwrite a terminal state. | `7feb4f0` — compare-and-set | `test_hardening.py::test_payment_update_is_compare_and_set`, `…_concurrent_payment_updates…` |
| A6 | L | `/api/stats?recent=-1` → `LIMIT -1` → the whole table. | clamp | `test_stats_recent_limit_is_clamped` |
| A7 | L | Rate limiter kept a deque per IP forever. | idle-key sweep | `test_rate_limiter_forgets_idle_clients` |
| A8 | L | Image sniffer accepted any RIFF (AVI/WAV) as WEBP; video uploads not content-checked. | signatures | `test_image_sniffer_…`, `test_video_sniffer`, `test_renamed_non_video_is_rejected_before_a_worker_starts` |
| A9 | L | Photo evidence written CWD-relative with second-resolution names (collisions, 404s when not run from the repo root). | `7feb4f0` — `build_evidence` in `EVIDENCE_DIR` | e2e photo tests |

## Honesty of presentation

| # | sev | finding | fix |
|---|---|---|---|
| H1 | M | `demo.py` stamped illustrative sessions with the real model version, and attached a photo of plate MH02DL4596 as evidence for "KA05CD9876". | `0dc248f` — demo version string; photos only for the plate they show; labelled placeholders otherwise |
| H2 | M | `seed_demo.py` records hand-chosen violation types (overspeed from a photo) indistinguishably from pipeline output. | `0dc248f` — labelled "[curated demo seed] not pipeline output" |
| H3 | M | Published numbers that did not survive the audit: OCR "76.8%" bottleneck, "precision/recall 1.00", mAP 0.727, "probe wins", "1.50 false positives even with a perfect detector" (the suite contains designed flickers). | README, RESUME, evaluation docs rewritten from regenerated results |

## Found by an independent review of the 1.1.0 changes

After the fixes above, a separate AI review agent — given the branch, not the
reasoning behind it — audited *this* branch's code and found eight more
defects, each confirmed by running code. They are recorded
here because the most serious ones sat exactly where a fine lands on the wrong
owner — the plate-identity layer — and two survived the first round of fixes.
All fixed in `3902441`, each with a regression test.

| # | sev | finding | fix | pinned by |
|---|---|---|---|---|
| R1 | H | **One valid OCR reading could elect a plate**: `min_observations` counted unreadable reads, so 1 valid + 2 junk reads elected (margin 0.67). And the OCR-threshold experiment scored the plate after all reads, while the pipeline commits at the *first* election — under-counting wrong plates several-fold (4.6% vs 0.2% in one condition). | winner must have 3 supporting valid reads; experiment scores at the commit point; thresholds re-selected | `test_one_valid_read_among_junk_cannot_elect_a_plate`, `test_stabilizer_is_scored_at_the_moment_the_pipeline_commits` |
| R2 | H | **The OCR lock froze a track's plate**: after an identity switch the second rider was fined under the first rider's plate. | locked plates re-read every 10 frames; a fine needs an agreeing read within 25 frames | `test_pipeline_core.py::test_identity_switch_under_the_ocr_lock_does_not_fine_the_first_plate`, `…_stale_plate_is_not_used_for_a_late_confirmation` |
| R3 | M | `/detect` honoured only `DETECT_API_KEY`: open with role keys configured. | role keys honoured too | `test_detect_honours_role_keys_too` |
| R4 | M | A violation whose plate stabilised only on frames without a rider box was never fined; its withheld reason was frozen. | emission retried every frame with the last rider box | `test_violation_fined_when_plate_stabilises_only_without_a_rider_box` |
| R5 | M | The test report applied the baseline's val thresholds to every model. | per-model thresholds | `test_each_model_is_reported_at_its_own_val_thresholds` |
| R6 | L | OCR time profiled per frame, not per call (benchmark under-counted calls with several plates). | per-call accounting | `test_profiler_counts_every_ocr_call_not_every_frame` |
| R7 | L | Provenance `dirty` ignored untracked source files — and (found while regenerating) counted edits to docs, so results regenerated during a docs edit were stamped dirty. | untracked code counts; `*.md` doesn't | `test_dirty_flag_ignores_generated_results` |
| R8 | L | Error-budget OCR stream took a variable number of draws, so one oracle shifted another stage's outcomes. | constant draws per opportunity | — (noise, not bias) |

## What the review did not find

No SQL injection path (parameterised queries; the one interpolated column is
whitelisted), no path traversal in evidence serving, no secret leakage in
errors or `/health`, no malformed labels, duplicate boxes or same-rider
contradictory annotations in the dataset files.
