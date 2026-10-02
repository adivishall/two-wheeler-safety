# Verification map

For each claim the project makes: where it is implemented, which test fails if
it breaks, and how to run that test. Then every headline ML number with its
dataset, split, metric, model, sample size and source file. Then the claims on
`main` that do **not** hold, with their status.

Scope: branch `main` as of `2ef6991` plus the fixes in this branch. Newer work
on `feature/flagship-hardening` and PR #18 (`feature/field-evaluation-loop`,
stacked on it) changes several rows; where it does, the row says so and names
the commit. Nothing here is about field accuracy: there is no labelled field
data yet (#8).

## How to run everything below

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11-3.13
pip install -r requirements-ci.txt -c constraints-ci.txt
make check            # ruff + mypy + pytest with the 90% branch-coverage gate (what CI runs)
pytest tests/test_association.py::test_hungarian_beats_greedy_nearest   # any single test
```

Measured on the review machine (macOS 26.5, Apple Silicon, Python 3.13.7): a
clean clone to a green `pytest` took 27 s; the suite (481 tests) runs in about
5-10 s; branch coverage of `modules/` is 94.5%. CI runs the same gates on Ubuntu
with Python 3.11 and 3.12 (`.github/workflows/ci.yml`).

No test loads YOLO or EasyOCR. `modules/detector.py` and `modules/plate_ocr.py`
are excluded from coverage for that reason (`pyproject.toml`).

## 1. Invariants and where they are enforced

| # | Claim | Implemented in | Test that fails if it breaks | Notes |
|---|---|---|---|---|
| 1 | Plate-to-rider association is one-to-one and globally minimum-cost, not greedy | `modules/association.py`: `hungarian`, `gated_min_cost_matching`, `associate` | `tests/test_association.py::test_hungarian_beats_greedy_nearest`, `::test_one_violation_never_assigned_to_two_vehicles`, `::test_nearest_plate_mis_assigns_but_association_is_correct`, `::test_hungarian_rectangular_more_rows_than_cols` | The evidence for Hungarian over greedy is constructed counter-examples, not a measured rate on data. |
| 2 | A plate with no plausible rider stays unmatched | `modules/association.py`: `_plate_body_cost` | `tests/test_association.py::test_plate_with_no_plausible_body_stays_unmatched` | **Holds only for diagonally displaced plates.** A horizontally aligned plate is paired at any distance (12 body-diagonals in the probe in #19). |
| 3 | Two crossing vehicles are not merged into one; a helmet inside its rider is | `modules/association.py`: `merge_bodies`, `AssociationConfig.body_merge_iou` / `body_merge_containment` | `tests/test_tracking_eval.py::test_two_crossing_riders_no_longer_merge`, `::test_helmet_inside_a_triple_body_still_merges`, `::test_new_config_removes_every_id_switch_in_the_suite`; `tests/test_system_eval.py::test_crossing_no_longer_causes_id_switches` | Fixed in `483c9ea`; see FAILURE_ANALYSIS. |
| 4 | Helmet and no-helmet on one rider is never fined (contradiction safeguard) | `modules/association.py`: `VehicleBody.ambiguous_helmet`; `modules/violation_state.py`: `HelmetStateMachine.update`; photo path `modules/detector.py`: `analyze_image` (IoU > 0.1) | `tests/test_violation_state.py::test_ambiguous_frame_blocks_confirmation`, `tests/test_pipeline_integration.py::test_contradiction_is_not_recorded`, `tests/test_main_helpers.py::test_iou_real_contradiction_case`, `tests/test_pipeline_eval.py::test_helmet_contradiction_is_never_fined` | Origin: a real photo, `3ab41c8`. |
| 5 | In video, a violation must hold for `STREAK_THRESHOLD` (default 5) frames before it is recorded | `modules/violation_state.py`: `HelmetStateMachine`, `TripleRidingStateMachine`; `modules/video_detector.py`: `process_video` (`confirm` for overspeed) | `tests/test_violation_state.py::test_helmet_confirms_only_after_window`, `::test_triple_single_frame_does_not_confirm`, `::test_helmet_below_confidence_threshold_does_not_count`; `tests/test_pipeline_eval.py::test_single_frame_helmet_flicker_does_not_fine` | **Video only.** The photo route (`/analyze`) decides from one image. Once confirmed, a violation never un-confirms (no hysteresis on the way down). The helmet machine tolerates one blank frame (`miss_tolerance=1`); the triple machine resets on any miss. |
| 6 | At most one fine per (vehicle, violation) | `modules/video_detector.py`: `process_video` (`reported` set) | `tests/test_pipeline_integration.py::test_video_pipeline_records_confirmed_no_helmet` (one record over 10 frames) | **Per track id on `main`.** A rider who leaves view for more than `max_age` (15) frames returns as a new track and can be fined again. Fixed on `feature/flagship-hardening` in `c4b8c6d` (one fine per plate and violation per run). |
| 7 | A fine is written only against the temporally voted plate, never one frame's read | `modules/plate_recognizer.py`: `PlateStabilizer.result`; `modules/video_detector.py` uses `track.stable_plate` | `tests/test_plate_recognizer.py::test_single_observation_never_elects_a_plate`, `::test_only_invalid_readings_elect_nothing`, `::test_majority_valid_reading_wins` | **Does not hold on `main`:** one valid read plus one unreadable read elects (agreement 0.8), because `min_observations` counts junk reads. An exact two-way tie also elects, by string order (`tests/test_ocr_temporal_eval.py::test_two_way_tie_elects_at_half_agreement` pins it). Both fixed on the branch in `ae180c1` (`test_one_valid_read_among_junk_cannot_elect_a_plate`) and `c4b8c6d`/`d1d59ce` (ties abstain). |
| 8 | Look-alike correction only moves toward a valid Indian plate structure, within a bounded number of edits | `modules/plate_recognizer.py`: `correct_plate`, `_coerce_standard`, `_coerce_bh` | `tests/test_plate_recognizer.py::test_correction_is_bounded_and_refuses_to_hallucinate`, `::test_digit_lookalike_corrected_in_number_position` | |
| 9 | Skipping OCR once a plate is locked does not change the fine | `modules/video_detector.py`: `process_video` (`plate_locked`) | `tests/test_pipeline_integration.py::test_ocr_lock_skips_redundant_ocr_without_changing_the_fine`; `tests/test_benchmark.py::test_benchmark_forwards_the_ocr_lock_setting` | On `main` the lock never re-reads, so an identity switch keeps the first plate; the branch re-reads every 10 frames (`ae180c1`). |
| 10 | On `main`, OCR may run on a stale plate box | `modules/vehicle_tracker.py`: `_update_track` keeps the last `plate_box` when the plate is missed; `process_video` crops it | none on `main` | Known defect, fixed on the branch in `c4b8c6d` (`VehicleTrack.plate_visible`, `test_plate_occlusion_never_ocrs_a_stale_box`). It also feeds zero-motion samples to the speed estimator. |
| 11 | Speed uses video time (`frame_index / fps`), not wall-clock | `modules/speed.py`: `SpeedEstimator.estimate` | `tests/test_speed.py::test_processing_wall_clock_does_not_affect_estimate`, `::test_estimate_matches_known_ground_truth_speed`, `::test_two_tracked_objects_dont_interfere`, `::test_impossible_jump_is_rejected` | |
| 12 | No speed is reported without calibration | `modules/video_detector.py` (`speed_estimator` is `None` without `pixels_per_meter`); `app.py`: `analyze_video` passes none | none | The web app therefore never computes speed; only `main.py --pixels-per-meter` does, with a constant scale. `HomographyPlaneCalibration` exists in `modules/speed.py` but no pipeline uses it (#13). |
| 13 | Confidence is a weighted mean in [0, 1], never called a probability | `modules/confidence.py`: `compute_confidence` | `tests/test_confidence.py::test_inputs_are_clamped`, `::test_equal_weights_is_the_mean` | The temporal component is 1.0 on every recorded violation, so it cannot rank (#20). |
| 14 | Each evidence package records SHA-256 per artifact, and tampering is detected | `modules/evidence.py`: `build_evidence`, `verify_evidence` | `tests/test_evidence.py::test_verify_evidence_detects_a_modified_artifact`, `::test_verify_evidence_detects_a_deleted_artifact`, `::test_evidence_ids_are_unique_within_the_same_second`, `::test_unsafe_plate_is_sanitized_in_filenames` | Hashes live next to the files they cover, so they detect accidental change, not a deliberate edit of both. |
| 15 | Evidence files are served only as a bare basename with a media extension | `app.py`: `evidence` | `tests/test_security.py::test_evidence_route_blocks_traversal_and_bad_types`, `tests/test_app.py::test_evidence_rejects_non_media_and_subpaths` | |
| 16 | A caller-supplied `image_path` is reduced to a name with no directory, no traversal and no markup | `modules/validation.py`: `safe_evidence_name` | `tests/test_validation.py::test_safe_evidence_name_blocks_traversal_and_bad_types`, `::test_safe_evidence_name_rejects_markup_and_quote_characters`, `tests/test_security.py::test_detect_rejects_image_path_that_could_break_out_of_an_attribute`, `tests/test_app.py::test_detect_rejects_traversal_image_path` | The markup rule is new in this branch (stored XSS, see FAILURE_ANALYSIS). |
| 17 | The dashboard escapes API strings it puts into `src` / `href` / `alt` attributes | `templates/frontend.html` (`escapeHtml`) | `tests/test_security.py::test_dashboard_escapes_every_url_and_text_attribute_it_interpolates` | Static check of the template. |
| 18 | SQL is parameterised; the sort column is whitelisted; page sizes are clamped | `modules/db.py`: `Database.list_violations` and the other queries | `tests/test_db.py::test_list_sort_whitelist_rejects_injection`, `::test_list_limit_is_clamped`; `tests/test_db_lifecycle.py::test_update_job_whitelists_columns` | `/api/stats?recent=-1` returns every row on `main` (SQLite `LIMIT -1`); clamped on the branch (`faa4438`). |
| 19 | Uploads are checked by extension and magic bytes, with server-chosen temp names and a size cap | `modules/validation.py`: `sniff_image`, `safe_extension`; `app.py`: `analyze`, `analyze_video` | `tests/test_app.py::test_analyze_rejects_non_image_content`, `::test_analyze_rejects_unsupported_extension`, `tests/test_validation.py::test_sniff_image_by_magic_bytes` | On `main` any RIFF file passes as an image and videos are not sniffed (branch adds both). |
| 20 | Writes can require an API key or a role | `app.py`: `detect`, `_require_role`; `modules/config.py`: `ServerConfig.role_for_key` | `tests/test_app.py::test_detect_rejects_wrong_or_missing_api_key_when_configured`, `tests/test_app_auth.py::test_review_requires_reviewer_role_when_enabled`, `::test_audit_endpoint_requires_admin` | Open gaps on `main` and on the branch: key guessing is not rate-limited, keys are compared with `==`, multipart uploads accept cross-origin POSTs, read routes ignore roles (#22). On `main`, `/analyze` and `/analyze_video` record fines with no role check (fixed on the branch). |
| 21 | Debugger off by default; errors are generic JSON; `/health` exposes no secrets | `modules/config.py`; `app.py`: `_err_*`, `health` | `tests/test_security.py::test_health_does_not_leak_secrets`, `::test_error_responses_are_json_and_do_not_leak`, `::test_video_status_error_is_generic` | `/analyze` returns the server temp path when an image cannot be decoded (#21). |
| 22 | Write and upload routes are rate-limited per client | `modules/validation.py`: `RateLimiter`; `app.py`: `_rate_limited` | `tests/test_validation.py::test_rate_limiter_sliding_window`, `tests/test_security.py::test_rate_limit_trips` | Keyed on `remote_addr`: behind a reverse proxy all clients share one bucket unless a proxy hop is trusted (#22). |
| 23 | Video jobs are bounded, cancellable and expire | `modules/jobs.py`: `JobManager` | `tests/test_jobs.py::test_max_concurrent_rejects_further_submissions`, `::test_cancellation_is_cooperative`, `::test_cleanup_expires_finished_jobs` | Fines recorded before a cancel stay recorded. |
| 24 | Held-out splits are audited against train **and against each other**; a rebuilt clean split contains exactly the kept images | `modules/dataset_audit.py`: `audit_splits`, `find_leaks`, `build_clean_split`, `prepare_link_dir`; `build_train_split.py`: `materialise` | `tests/test_dataset_audit.py::test_audit_splits_flags_leakage`, `::test_audit_flags_an_image_shared_by_val_and_test`, `::test_rebuilding_a_clean_split_removes_images_that_are_now_leaked`, `::test_clean_split_refuses_to_mix_with_real_files`; `tests/test_build_train_split.py::test_materialise_rerun_removes_images_no_longer_kept` | The val-vs-test comparison and the stale-link fix are new in this branch. dHash misses flips and crops, and nothing groups frames by source video (#23). |
| 25 | The model-free dependency set is pinned and consistent | `requirements-ci.txt`, `constraints-ci.txt` | CI step `pip check` | The full runtime stack (`constraints-runtime.txt`) is a record, not enforced by CI. |
| 26 | Training is seeded and its configuration recorded | `train_traffic.py` (`--seed 0`, `deterministic=True`); `modules/model_manifest.py`: `build_manifest`; `models/manifests/traffic-4class.json` | `tests/test_model_manifest.py::test_build_manifest_records_full_provenance` | The manifest records seed, epochs, device and a SHA-256 of the weights, but not the Ultralytics or torch version; bit-identical reruns on MPS are not established. The weights themselves are not in the repository. |
| 27 | The evaluation CLIs keep working without weights | `evaluate_pipeline.py`, `evaluate_system.py`, `evaluate_ocr.py` | CI step "Evaluation tooling smoke test (model-free)"; `tests/test_evaluation_cli_smoke.py` | |

## 2. ML claims

Every number below is read from a committed file. "Model" is
`traffic-4class@1.0.0` = `runs/detect/traffic_model-2/weights/best.pt`
(YOLOv8n, 15 epochs from `yolov8n.pt`, seed 0, MPS; SHA-256 in
`models/manifests/traffic-4class.json`). The dataset is
`master_traffic_violation_dataset` (assembled from public exports; source and
licence **unverified**, `docs/DATASET.md`). Neither the weights nor the dataset
are in the repository, so rows 1-8 can be re-run only by someone who has them.

| # | Claim (where stated) | Dataset / split | Metric | Model | n | Source file (key) | Caveat |
|---|---|---|---|---|---|---|---|
| 1 | Detector mAP@50 **0.727**, mAP@50-95 0.532, P 0.756, R 0.785 (README) | de-leaked **test** (`eval/clean_splits/test`) | Ultralytics `val()` at conf 0.25, IoU 0.5, imgsz 640 | 1.0.0 | 175 images, 293 instances | `eval/results/eval_traffic_model-2_test_clean.json` (`official.map50` 0.7265) | Computed at the operating confidence (0.25), which truncates the PR curve; under the standard protocol the branch's copy of the same file reads 0.7688 (protocol fixed in `d4da223`). Same source pool as train. Single run, no interval on `main`. |
| 2 | Val mAP@50 **0.697**, mAP@50-95 0.502 (EVALUATION, ERROR_ANALYSIS, MODEL_VERSIONING, manifest) | **val** (original, not de-leaked) | same | 1.0.0 | 383 images, 572 instances | `eval/results/eval_traffic_model-2_val.json` (`official.map50` 0.6967) | 8.1% of val images are near-duplicates of training images. Val also picks the checkpoint (`best.pt`), so it is not an independent estimate. The manifest's `metrics.source` says `reports/…`; the file is in `eval/results/`. |
| 3 | Per class, test: WithHelmet mAP@50 0.387, Plate 0.863, WithoutHelmet 0.705, TripleRiding 0.950 | de-leaked test | per-class AP@50 | 1.0.0 | 27 / 131 / 105 / 30 instances | `eval_traffic_model-2_test_clean.json` (`official.per_class`) | WithHelmet rests on 27 instances; the branch's bootstrap 95% CI for its AP@50 is [0.204, 0.729] (README on `feature/flagship-hardening`; bootstrap added in `d4da223`). |
| 4 | Per-class F1 (WithHelmet 0.519 …) | de-leaked test | F1 from a second matching pass | 1.0.0 | 300 matched predictions | same file (`error_analysis.class_report`) | `modules/evaluation.py:match_image` is greedy and class-agnostic at IoU 0.5, so it differs from Ultralytics' class-aware matching when both helmet classes hit one head. |
| 5 | "Confidence ranks correctness for WithoutHelmet (Spearman 1.00) and not for TripleRiding (−0.20)" (README) | de-leaked test | Spearman between **bin midpoint and bin accuracy**, over at most 5 score bins | 1.0.0 | WithoutHelmet 113 predictions; TripleRiding 33, with bins of 0/1/1/7/24 | same file (`error_analysis.confidence_curve.per_class`) | `modules/confidence_analysis.py:ConfidenceCurve.spearman` is unweighted by bin size; the TripleRiding value is driven by two single-prediction bins. Read both as "not established". |
| 6 | 9.8% of test and 8.1% of val were near-duplicates of training images | test 194, val 383 vs 11,195 train | dHash 64-bit, Hamming ≤ 5 | — | 19 / 31 images | `eval/results/dataset_leakage.json` (`splits.*.leak_rate`) | Misses flips (0/14 bundled samples caught) and crops; val vs test was not compared when this file was generated; frames of the same video span all three splits (#23). |
| 7 | De-leaking moved test mAP@50 by +0.006 (0.7202 → 0.7265) | test 194 vs de-leaked test 175 | as row 1 | 1.0.0 | 194 vs 175 images | `eval_traffic_model-2_test.json`, `eval_traffic_model-2_test_clean.json` | Two point estimates on different image sets; the difference is well inside sampling noise, so "not inflating" means "no detectable effect". |
| 8 | Checkpoint A/B: `traffic_model_probe` 0.741 vs shipped 0.727; "promote probe" (README Future improvements, MODEL_EVALUATION) | de-leaked **test** | mAP@50 | four local checkpoints | 175 images | `eval/results/model_comparison.json` | **Selected on the test split.** Re-run on val with a paired bootstrap on the branch (method `d4da223`, results regenerated in `aeb865b`): probe is significantly worse (−0.051) and the recommendation was withdrawn. |
| 9 | End-to-end fines precision 1.0, recall 1.0 | 8 hand-built synthetic scenarios | fine-level P/R | none (synthetic detections) | 8 expected fines | `eval/results/pipeline_evaluation.json` (`system.totals`) | Synthetic. The evaluator re-implements the per-frame loop with `confirm_window=3` (`modules/system_eval.py:evaluate_system`); the app ships 5. Through the shipped loop the branch measures P 1.0 / R 0.9 (`c4b8c6d`). |
| 10 | Pipeline vs single-frame policy: F1 0.994 vs 0.840 at 10% helmet-label noise; naive averages 1.50 false positives at 0% injected noise | 19 synthetic scenarios × 20 trials, seed 11 | F1, mean FPs | none | 380 runs per noise level | `pipeline_evaluation.json` (`headroom.rates["0.10"]`, `["0.00"]`) | "0% injected noise" still includes the scenarios' designed one-frame flickers; window 3 as in row 9. The F1 gain shrinks to +0.043 at 40% noise and, with the shipped window, reverses at high noise (branch, AUDIT E2). |
| 11 | Temporal window: window 1 gives helmet precision 0.67, triple-riding 0.80; window 2 restores 1.00 | synthetic edge-case suites | decision precision | none | helmet 2 TP + 1 FP; triple 4 TP + 1 FP | `pipeline_evaluation.json` (`violations.*.confirm_window_sweep`) | Three to five decisions per cell, and the flickers are one frame long by construction, so "window 2 is enough" is built in. |
| 12 | OCR policy at 12% character noise: last-frame read wrong 85%, best-confidence 30%, temporal vote 2% while answering 51% | 100 simulated sequences × 10 frames | accuracy when answering, coverage | none (simulated EasyOCR-like noise) | 100 sequences | `eval/results/ocr_policy_simulation.json` (`default_noise`) | Simulated noise model, not EasyOCR on real plates; no field OCR accuracy exists. |
| 13 | Error budget: OCR is 76.8% of system sensitivity | 8 synthetic scenarios, 30% injection, 20 trials, seed 7 | share of summed F1 drop | none | 20 trials per stage | `eval/results/latest.json` (`error_budget`) | The OCR rate is per **character**, the detector rates per **box**: at 30% per character a 10-character plate survives intact 0.7^10 ≈ 2.8% of the time, so the shares compare different units. The branch replaces this with an oracle ablation and the ranking reverses (the detector owns the lost F1; `a4ae630`, `6bee6b8`). |
| 14 | Speed MAE 11.3 km/h toward the camera with a homography; constant scale misses every overspeeder there | synthetic constant-velocity trajectories | MAE, overspeed recall | none | synthetic tracks | `pipeline_evaluation.json` (`speed`) | No surveyed ground truth. The homography is not used by the app or CLI (row 12 of section 1). |
| 15 | Association fix: ID switches 13 → 0, wrong fines 5 → 0 | 8 synthetic crossing scenarios | ID switches, fines | none | 8 scenarios | `eval/results/tracking_comparison.json` (`totals`) | Synthetic geometry with rider-sized boxes; the dataset's helmet classes mark heads (`docs/DATASET.md`). |
| 16 | Single-image inference 24.2 ms (≈41 FPS), EasyOCR 14.2 ms, peak RSS 987 MB, video 50.2 vs 28.7 FPS (+74.9%) with the OCR lock, "88% YOLO, 9% OCR" (README, EVALUATION §6) | `samples/test.jpg`, a 120-frame local clip | latency, throughput | 1.0.0 | single runs | **no committed result file** (prose in `docs/EVALUATION.md` §6) | The only committed latency is 28.57 ms mean over 50 images (`eval_traffic_model-2_test_clean.json`, `benchmark`). The branch found that runs labelled "mps" executed on the CPU (`7fb9fd1`) and re-measured with repeated, interleaved runs (`2865ac4`). |

### Reproduced in this review

The three model-free result files regenerate **identically** (apart from
timestamps) from the committed code on the review machine:

```bash
python3 evaluate_pipeline.py --out /tmp/eval          # == eval/results/pipeline_evaluation.{json,csv}, latest.json
python3 evaluate_ocr.py --simulate --sweep --out /tmp/ocr --name ocr_policy_simulation
python3 evaluate_tracking.py --json /tmp/tracking_comparison.json
```

Rows 1-8 and 16 need the weights and the dataset and were not re-run.

## 3. Claims on `main` that do not match their source

| Claim | Where | What the source says | Status |
|---|---|---|---|
| "A fine is written only against this temporally-stable plate, never a single frame's read" | README "OCR pipeline", `PlateConfig.min_observations` comment | one valid + one junk read elects (section 1, row 7) | fixed on branch (`ae180c1`); not on `main` |
| "Temporal confirmation — a violation must persist over several frames" | README "What it does" | true for video only; `/analyze` fines from one photo | README corrected in this branch |
| "Overspeed detection" | README "What it does" | the app never computes speed; CLI only, constant scale | README corrected in this branch |
| "No duplicate fines … at most one fine per violation" | README | per track id; re-entry fines twice | README caveat in this branch; fixed on branch (`c4b8c6d`) |
| Inference/throughput figures | README "Application performance", `docs/EVALUATION.md` §6, `docs/RESUME.md` | no committed result file | README labelled in this branch |
| "everything the project wrote around them … is under 2% combined" | README, `docs/EVALUATION.md` | the prose table lists 1.1% plus 1.3% "unaccounted (draw, state machines, glue)" = 2.4% | README corrected in this branch |
| "88% YOLO and 9% OCR … <2% everything else" | `docs/RESUME.md` | 100 − 87.5 − 9.1 = 3.4% | open (file rewritten on the branch) |
| 0.30 and 0.40 noise rows (recall 0.883 / F1 0.913; F1 0.849, gain +0.040) | `docs/END_TO_END_EVALUATION.md` §1 | 0.8889 / 0.9174; 0.8511, +0.0427 | open (file rewritten on the branch) |
| "1.50 false positives even with a perfect detector" | `CHANGELOG.md`, `docs/RESUME.md`, `docs/INTERVIEW.md` | 1.5 at 0% *injected* noise; the scenarios contain designed flickers | README wording corrected in this branch; others open |
| window "3 (ships)" | `docs/END_TO_END_EVALUATION.md` | the app ships `STREAK_THRESHOLD=5` | open |
| "test was never used for model selection" | `docs/MODEL_EVALUATION.md` §1 | `compare_models.py` defaults to `--split test` and the README recommends the test-split winner | README caveat in this branch; Makefile fixed on branch (`e710249`) |
| "the leakage was not inflating the headline metric" | README, `docs/DATASET.md` | +0.006 between two point estimates on different image sets | reworded in README in this branch |
| `--epochs 50` training command | README "Model training" | the shipped model was trained 15 epochs (manifest) | README corrected in this branch |
| test-suite timing "~3 s" | `docs/TESTING.md` | 5-10 s measured | open (minor) |

Rows marked "open" are in documents that `feature/flagship-hardening`
rewrites; editing them on `main` would only create merge conflicts. They are
listed so a reader of `main` is not misled.
