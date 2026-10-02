# Failure analysis

Real defects from this repository's history: what was observed, why it
happened, how it was found, the commit that fixed it, and the test that keeps it
fixed. Every fix SHA below resolves with `git show <sha>` (the last section
deliberately lists some that do not), and every test name exists, checked with
`git grep`. Cases where no regression test exists say so.

Limitations that are not defects (a weak class, synthetic-only evaluation) live
in [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) and [VERIFICATION.md](VERIFICATION.md).

## Fixed on `main` (or in this branch)

### 1. Stored XSS through `/detect` and the plate lookup
- **Symptom.** `POST /detect` with `image_path` `x" onerror="alert(document.domain)" a=".jpg` returned 200; looking the plate up in the dashboard's Plate tab rendered `<img src="/evidence/x" onerror="alert(...)">` and ran the script.
- **Root cause.** `modules/validation.py:safe_evidence_name` removed directories but kept quotes, angle brackets and spaces; `app.py:get_fines` returned the stored name; the plate-lookup row in `templates/frontend.html` interpolated `f.image` without `escapeHtml` (the dashboard modal did escape). `/detect` is open unless `DETECT_API_KEY` is set.
- **Detection.** Security review of every `innerHTML` sink against the API fields that reach it; reproduced with the Flask test client on `main` and on `feature/field-evaluation-loop`.
- **Fix.** `3b158c8`: names restricted to `[A-Za-z0-9._-]` (`fullmatch`), and the row escapes its URL, alt text, status and label (rows stored before the fix are escaped on render).
- **Tests.** `tests/test_validation.py::test_safe_evidence_name_rejects_markup_and_quote_characters`, `tests/test_security.py::test_detect_rejects_image_path_that_could_break_out_of_an_attribute`, `tests/test_security.py::test_dashboard_escapes_every_url_and_text_attribute_it_interpolates` (all three failed before the fix).

### 2. A helmeted rider would have been fined: the model contradicted itself
- **Symptom.** On `samples/plate6.jpeg` (rider wearing a helmet) the model returned `WithHelmet` 0.392 and `WithoutHelmet` 0.604 on almost the same box (IoU 0.95). The photo path reported `no_helmet` whenever a `WithoutHelmet` box existed, so the higher-confidence, wrong box decided.
- **Root cause.** `main_ocr.py` and `main.py` had no notion of two boxes describing one rider.
- **Detection.** Manual run on the bundled sample photos.
- **Fix.** `3ab41c8`: an overlapping `WithHelmet` box makes the frame ambiguous; neither label is trusted. The same rule became `VehicleBody.ambiguous_helmet` and the `AMBIGUOUS` state of `HelmetStateMachine`.
- **Tests.** `tests/test_main_helpers.py::test_iou_real_contradiction_case`, `tests/test_main_helpers.py::test_is_contradicted_true_for_overlapping_boxes`, `tests/test_violation_state.py::test_ambiguous_frame_blocks_confirmation`, `tests/test_pipeline_integration.py::test_contradiction_is_not_recorded`.
- **Not fixed by this.** `plate9.jpeg`: a helmeted rider called `WithoutHelmet` at 0.887 with no competing box. A steadily wrong prediction passes every rule; that is why every record stays `pending` until a person reviews it.

### 3. Two crossing vehicles were merged into one before tracking
- **Symptom.** In the synthetic crossing suite: 13 ID switches, 5 false-positive fines and 2 missed fines across 8 scenarios.
- **Root cause.** `modules/association.py:merge_bodies` merged boxes at IoU > 0.3 or containment > 0.6; two crossing riders reach IoU 0.429 / containment 0.600, so two vehicles became one body and the tracker was handed one anchor for two vehicles. The tracker was not at fault.
- **Detection.** Frame-by-frame diagnosis (`python3 evaluate_tracking.py --diagnose`). The earlier test asserted `id_switches >= 1`, i.e. it pinned the bug as expected behaviour.
- **Fix.** `483c9ea`: merge at IoU > 0.5 or containment > 0.8, values chosen from the geometry of the cases that must and must not merge, not fitted to one scenario.
- **Tests.** `tests/test_tracking_eval.py::test_old_config_collapses_two_vehicles_into_one_body`, `tests/test_tracking_eval.py::test_new_config_removes_the_wrong_fines`, `tests/test_system_eval.py::test_crossing_no_longer_causes_id_switches`.

### 4. The OCR-lock benchmark compared the optimisation with itself
- **Symptom.** A published "+172% (18.5 → 50.2 FPS)" speed-up for skipping OCR on a locked plate.
- **Root cause.** `benchmark.py:benchmark_video` never passed `ocr_lock_confidence` to `process_video`, so the documented baseline setting had no effect and both arms ran with the lock on.
- **Detection.** Re-running the documented performance reproduction during final validation. Nothing in the output distinguished the two arms, which is why it went unnoticed.
- **Fix.** `ffb79ec`: the setting is forwarded and each arm reports its OCR call count. Re-measured at +92.6% (later +74.9%, `ea39043`); the recorded fine was identical in both arms.
- **Tests.** `tests/test_benchmark.py::test_benchmark_forwards_the_ocr_lock_setting`, `tests/test_benchmark.py::test_benchmark_reports_ocr_call_count`.

### 5. Held-out images were near-copies of training images
- **Symptom.** 19 of 194 test images (9.8%) and 31 of 383 val images (8.1%) were near-duplicates of training images; every number quoted until then came from val.
- **Root cause.** An offline augmentation pass renamed training files (`aug_<hash>.jpg`), so name-based checks could not see the copies.
- **Detection.** dHash audit of pixel content (`audit_dataset.py`, `modules/dataset_audit.py`), added in `5ec3bd7`.
- **Fix.** `5ec3bd7` builds a de-leaked test split; `e7a94cf` builds a training split without held-out copies instead. The de-leaked test mAP@50 differed from the original by +0.006 (two point estimates on different image sets).
- **Tests.** `tests/test_dataset_audit.py::test_find_leaks_flags_shared_images`, `tests/test_dataset_audit.py::test_build_clean_split_drops_leaked_and_keeps_labels`.

### 6. A rebuilt "clean" split still contained images it had dropped
- **Symptom.** Re-running `audit_dataset.py --write-clean-split` (or `build_train_split.py`) after the leak set grew left the previous run's symlink for a now-leaked image in place. The returned count said it was gone; Ultralytics, which reads the directory, still evaluated (or trained) on it.
- **Root cause.** `modules/dataset_audit.py:build_clean_split` and `build_train_split.py:materialise` only added links. The existing idempotency test re-ran with the same inputs, so it could not see this.
- **Detection.** Review of the split writers.
- **Fix.** `137f36c`: `prepare_link_dir` removes links this run did not keep and refuses (without deleting anything) when the directory holds a real file.
- **Tests.** `tests/test_dataset_audit.py::test_rebuilding_a_clean_split_removes_images_that_are_now_leaked`, `tests/test_dataset_audit.py::test_clean_split_refuses_to_mix_with_real_files`, `tests/test_build_train_split.py::test_materialise_rerun_removes_images_no_longer_kept`, `tests/test_build_train_split.py::test_materialise_refuses_to_delete_a_real_file`.

### 7. The split audit never compared val with test
- **Symptom.** A test image identical to a val image was reported as clean.
- **Root cause.** `modules/dataset_audit.py:audit_splits` compared each held-out split with train only. Val selects the checkpoint, so such a test image is not independent of model selection.
- **Detection.** Review of the audit's comparisons.
- **Fix.** `f6feebd`: every pair of held-out splits is compared and reported under `held_out_overlap`. Not yet re-run on the dataset, which is not in the repository.
- **Tests.** `tests/test_dataset_audit.py::test_audit_flags_an_image_shared_by_val_and_test`.

### 8. Speeds were meaningless with more than one bike
- **Symptom.** With two or more bikes in frame, reported speeds compared one bike's position with another's.
- **Root cause.** `modules/speed.py:SpeedEstimator` kept one global previous position.
- **Detection.** Found while wiring the tracker into the video pipeline.
- **Fix.** `8f33c81`: state per track id.
- **Tests.** `tests/test_speed.py::test_two_tracked_objects_dont_interfere`.

### 9. Plate lookups missed stored fines; malformed requests crashed `/detect`
- **Symptom.** A lookup with different case or spacing missed the stored fine; a payload without `plate` raised a `KeyError` (500).
- **Root cause.** `app.py` stored and queried plates without normalising them and indexed the JSON body without validation.
- **Detection.** Not stated in the commit.
- **Fix.** `1c8c433` (the same commit also fixed a triple-riding overlap check that compared x-ranges only, in code since removed).
- **Tests.** `tests/test_app.py::test_plate_lookup_is_case_and_whitespace_insensitive`, `tests/test_app.py::test_detect_requires_all_fields`.

### 10. Re-detecting a plate overwrote the earlier fine's evidence; `/detect` had no auth
- **Symptom.** Evidence was saved as `evidence/{plate}.jpg`, so the second fine for a plate replaced the first fine's photo; anyone who could reach the port could write fines.
- **Fix.** `bf18cba`: unique evidence names (later random ids in `modules/evidence.py`) and an optional `DETECT_API_KEY`.
- **Tests.** `tests/test_evidence.py::test_evidence_ids_are_unique_within_the_same_second` (pins the later id scheme; `bf18cba`'s per-second timestamps could still collide), `tests/test_app.py::test_detect_rejects_wrong_or_missing_api_key_when_configured`.

### 11. The video pipeline could not run without a display
- **Symptom.** `main.py` hung or failed on any headless machine.
- **Root cause.** An unconditional `cv2.imshow` preview.
- **Detection.** The first run on real footage. The same run showed the model flickering between `WithHelmet` and `WithoutHelmet` on one rider, which is where the 5-frame streak (`STREAK_THRESHOLD`) came from.
- **Fix.** `d8cdee4`: headless by default (`--display` opt-in).
- **Tests.** None.

### 12. Error analysis crashed after `val()`
- **Symptom.** `evaluate_model.py` raised "Inference tensors do not track version counter" in the error-analysis and benchmark passes (torch 2.12, Apple MPS).
- **Root cause.** Ultralytics `model.val()` leaves fused inference tensors that a following `predict()` on the same object trips over.
- **Fix.** `1f22c46`: reload the checkpoint before the predict-based passes.
- **Tests.** None (needs torch and the weights).

### 13. The review modal said "loading" forever
- **Symptom.** For records without a metadata sidecar (demo data, photo fines) the confidence panel never left "Component breakdown loading…".
- **Root cause.** `templates/frontend.html:openDetail` had no branch for a missing sidecar.
- **Detection.** Taking the README screenshots.
- **Fix.** `2ef6991`.
- **Tests.** None (template behaviour). Related open defect on `main`: for video records that do have a sidecar, `/evidence/<name>.json` is refused by the evidence route's media allowlist, so the panel shows "No per-component breakdown (single-image detection)". Fixed on the branch (`faa4438`).

### 14. Broken imports and a crash on photos without a plate
- **Symptom.** `main.py` failed on import; `main_ocr.py` raised `NameError` when no plate was found.
- **Fix.** `f1027e9`.
- **Tests.** None (code since rewritten).

## Fixed on the unmerged branches

`feature/flagship-hardening` (and PR #18, stacked on it) fixes defects that are
still present on `main`. Tests listed here exist on those branches only.

| Commit | Defect on `main` | Regression test (branch) |
|---|---|---|
| `faa4438` | The photo route fined every violation in an image against the **last** plate OCR'd (two bikes → the compliant rider's plate fined) and accepted non-plate OCR text such as `0285`. | `tests/test_end_to_end.py::test_photo_with_two_bikes_fines_the_violators_own_plate` |
| `faa4438` | `/analyze` and `/analyze_video` record fines with no role check even when role keys are configured; the web job ignored the detection thresholds it wrote into evidence. | `tests/test_hardening.py::test_upload_routes_that_record_fines_need_the_reviewer_role`, `tests/test_hardening.py::test_detection_env_vars_reach_the_pipeline` |
| `c4b8c6d` | The evaluators re-implemented the per-frame loop (confirm window 3, not the shipped 5), so pipeline metrics did not describe the shipped code. | `tests/test_pipeline_core.py` (shared `ViolationPipeline`) |
| `c4b8c6d` | A rider out of view for more than 15 frames returned under a new track id and was fined again; the scorer counted it as a second true positive. | `tests/test_system_eval.py::test_rider_who_returns_under_a_new_track_is_fined_once` |
| `c4b8c6d` | After a plate was missed, its old box was still cropped and OCR'd every frame and fed to the speed estimator. | `tests/test_end_to_end.py::test_plate_occlusion_never_ocrs_a_stale_box` |
| `ae180c1` | One valid OCR read plus junk reads elected a plate (agreement 0.8 with one junk read on `main`). | `tests/test_ocr_temporal_eval.py::test_one_valid_read_among_junk_cannot_elect_a_plate` |
| `ae180c1` | The OCR lock froze a track's plate, so after an identity switch the second rider was fined under the first rider's plate. | `tests/test_pipeline_core.py::test_identity_switch_under_the_ocr_lock_does_not_fine_the_first_plate` |
| `a4ae630`, `6bee6b8` | The error budget compared per-character OCR corruption with per-box detector faults ("OCR is 76.8% of sensitivity"), then divided by the sum of overlapping recoveries. | `tests/test_pipeline_eval.py::test_error_budget_reports_each_stage_against_the_gap_not_the_sum` |
| `7fb9fd1` | Benchmarks labelled "mps" ran on the CPU: Ultralytics only auto-selects CUDA. | none |
| `e710249` | `make eval-compare` selected models on the test split. | none (Makefile) |
| `4e77701` | A confidence calibrator was chosen by in-sample ECE and adopted on ~30 of 40 already-calibrated simulated datasets. | `tests/test_calibration.py::test_calibrated_scores_are_not_replaced_by_an_in_sample_winner` |

## What the cases have in common

- **Most were measurement defects that flattered the result**: leakage (5),
  a benchmark comparing a setting with itself (4), evaluators running a
  different loop from the app (`c4b8c6d`), an error budget mixing units
  (`a4ae630`). Each was found by re-running a documented reproduction or by
  reading the evaluator next to the code it claims to measure.
- **Two tests pinned the bug as expected behaviour** (the crossing test in 3,
  the two-way OCR tie in `tests/test_ocr_temporal_eval.py`). A test that asserts
  current behaviour is not evidence that the behaviour is right.
- **Real images found what synthetic scenarios could not** (2, 11): synthetic
  suites only contain the failure modes someone thought to write down.

## Note on `docs/AUDIT.md` (branch `feature/field-evaluation-loop`)

That register cites fix commits that do not exist in this repository (the
branch history was rewritten after they were recorded). Matched by content:
`7feb4f0` → `faa4438`, `b497384` → `c4b8c6d`, `5131a03` → `d1d59ce`,
`54cae20` → `a4ae630`, `9d9556b` → `b826d9c`, `9ec9813` → `d4da223`,
`5e08614` → `7fb9fd1`, `8507c87` → `6bee6b8`, `3902441` → `ae180c1`,
`32b7060` → `4e77701`, `9084003` → `6a572c5`, `89fba50` → `e710249`,
`0dc248f` → `c7a2785`. The three V-series fixes (`19a2152`, `8898775`,
`28b4a65`) are `88fcb3c`, `d92fa82` and `41ff86d` as a set; the one-to-one
pairing is by order only.
