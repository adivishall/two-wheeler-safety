# Reviewer guide: ten hard questions

The questions a skeptical ML/CV reviewer should ask of this project, the answer
the repository actually supports, and where to check it. Where the honest answer
is "not measured" or "broken on `main`", it says so. Claim-by-claim detail is in
[VERIFICATION.md](VERIFICATION.md); history in
[FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md); reasoning in
[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md).

## 1. How do you know test images are not in training?

A perceptual-hash audit (64-bit dHash, Hamming ≤ 5) found that 9.8% of test
(19/194) and 8.1% of val (31/383) were near-copies of training images, most at
distance 0. The headline detector numbers use a test split rebuilt without
them (175 images).

- Code: `modules/dataset_audit.py` (`find_leaks`, `audit_splits`, `build_clean_split`), `audit_dataset.py`, `build_train_split.py`.
- Result: `eval/results/dataset_leakage.json`.
- Tests: `tests/test_dataset_audit.py::test_find_leaks_flags_shared_images`, `::test_audit_flags_an_image_shared_by_val_and_test`, `::test_rebuilding_a_clean_split_removes_images_that_are_now_leaked`.
- Limits: dHash misses flips (0 of 14 bundled samples caught) and crops; frames of one video appear in train, val and test (`00000000188000000_mp4`), and nothing groups by source (#23). Val vs test was not compared until this branch (`f6feebd`), and the committed audit predates that. "Cleaner", not "clean".

## 2. Why Hungarian matching and not nearest-plate?

Nearest-plate lets two riders claim one plate. Hungarian gives a one-to-one,
minimum-total-cost assignment. Measured on random multi-rider layouts with known
ownership, using the same cost for both: greedy left a wrong pair in 18.7% of
frames, Hungarian in 7.2%.

- Code: `modules/association.py` (`hungarian`, `associate`, `_plate_body_cost`).
- Result: `eval/results/association_baseline.md` (`python3 evaluate_association.py`).
- Tests: `tests/test_association.py::test_hungarian_beats_greedy_nearest`, `tests/test_evaluate_association.py::test_hungarian_attributes_fewer_plates_wrongly_than_greedy`.
- Limits: synthetic layouts and hand-set weights; the shipped synthetic scenarios never separate the two matchers; the distance gate never rejects a horizontally aligned plate (#19). Gap closed in this branch: before it, the only evidence was constructed counter-examples.

## 3. How does OCR fail, and what stops a wrong plate from being fined?

Per frame, EasyOCR confuses look-alikes (0/O, 1/I, 8/B, 5/S), drops reads and
invents strings. A per-track vote weights each read by confidence and plate
structure and abstains below 0.35 agreement; corrections must produce a valid
Indian plate within 2 edits.

- Code: `modules/plate_recognizer.py` (`correct_plate`, `PlateStabilizer`); `modules/plate_info.py`.
- Result: `eval/results/ocr_policy_simulation.json`: at 12% simulated character noise, last-frame reads are wrong 85% of the time, best-confidence 30%, the vote 2% while answering for 51% of vehicles.
- Tests: `tests/test_plate_recognizer.py::test_correction_is_bounded_and_refuses_to_hallucinate`, `tests/test_ocr_temporal_eval.py::test_temporal_is_more_precise_than_single_frame_under_noise`.
- Limits: the noise is simulated; **no OCR accuracy on real plates exists**. On `main`, one valid read plus one junk read elects a plate, and a two-way tie elects by string order (fixed on the branch, `ae180c1`). The photo route reads one image.

## 4. Where do temporal confirmation's false positives come from?

A no-helmet or triple-riding call must hold for 5 consecutive frames at box
confidence ≥ 0.3; a frame where the model says helmet and no-helmet on the same
rider counts as ambiguous and never confirms. What still gets through: a
*steadily* wrong prediction (on `samples/plate9.jpeg` the model calls a
helmeted rider `WithoutHelmet` at 0.887 with no competing box; repeated on every
frame, that confirms), correlated errors over a long dwell, and anything on the photo
route, which has no temporal rule. That is why every record is `pending` until
reviewed.

- Code: `modules/violation_state.py`; `modules/video_detector.py:process_video`.
- Tests: `tests/test_violation_state.py::test_helmet_confirms_only_after_window`, `::test_ambiguous_frame_blocks_confirmation`, `tests/test_pipeline_eval.py::test_single_frame_helmet_flicker_does_not_fine`.
- Limits: the window sweep is on synthetic one-frame flickers (3-5 decisions per cell); the evaluators ran window 3 while the app ships 5. Re-entry after 15 missed frames can fine twice on `main` (fixed on the branch, `c4b8c6d`).

## 5. What does "mAP@50 0.727" mean, and what does it not?

Mean over the four classes of average precision, with a prediction counted
correct at IoU ≥ 0.5 with a same-class box, computed by Ultralytics `val()` on
the de-leaked test split (175 images, 293 instances) for `traffic-4class@1.0.0`.

- Source: `eval/results/eval_traffic_model-2_test_clean.json` (`official.map50` = 0.7265).
- It is computed at the operating confidence 0.25, which truncates the PR curve; the branch's standard-protocol number for the same weights is 0.769. The val number, 0.697, is on a split with 8.1% leakage that also selected the checkpoint.
- It does not measure: whether a plate is readable, whether a violation is attributed to the right vehicle, behaviour on other cameras or cities (the test split shares the training pool), or any class reliably at n = 27 (`WithHelmet`, AP@50 0.387).

## 6. What is measured on real data, and what is synthetic?

| Real images | Synthetic | Not measured |
|---|---|---|
| Detector AP / P / R per class on the dataset's held-out split; class confusions; confidence vs correctness; leakage | Every pipeline number: end-to-end fine precision/recall, naive-vs-pipeline, window sweep, OCR policy, error budget, speed MAE, tracking ID switches, Hungarian vs greedy | OCR on real plates, fines on real footage, tracking/association on real footage, speed ground truth, confidence calibration, streaming latency |

- The synthetic results regenerate identically from the code: `python3 evaluate_pipeline.py`, `python3 evaluate_ocr.py --simulate --sweep`, `python3 evaluate_tracking.py`, `python3 evaluate_association.py` (checked in this review).
- The detector results need the weights and the dataset, neither of which is in the repository.

## 7. How fast is it, and is it real-time?

The only committed latency is 28.57 ms mean per image (p50 27.57 ms, 50 test
images, `eval_traffic_model-2_test_clean.json`, `benchmark`), recorded as MPS;
the branch later found that runs labelled MPS actually ran on the CPU
(`7fb9fd1`). The README's 24.2 ms / 50.2 FPS / +74.9% OCR-lock figures come from
`docs/EVALUATION.md` §6 with no committed result file. Video is processed from
files; nothing measures end-to-end delay or frame dropping on a live stream, so
no real-time claim is made (#16). Vehicle speed estimation is a different
question: see VERIFICATION.md section 2, row 14.

## 8. What happens to plates, faces and evidence?

The app stores plate strings, timestamps, confidence and evidence images (which
can include faces and other people), decodes only the public registration
region from the plate, and never looks up owners.

- Docs: `docs/PRIVACY.md` (stored-field inventory, recommended retention), `SECURITY.md` (reporting).
- Code: `modules/evidence.py` (`build_evidence` records SHA-256 per artifact; `verify_evidence`), `app.py:evidence` (basename + media-type allowlist).
- Tests: `tests/test_evidence.py::test_verify_evidence_detects_a_modified_artifact`, `tests/test_security.py::test_evidence_route_blocks_traversal_and_bad_types`.
- Limits: retention is recommended, not enforced; when auth is on, evidence and read routes are still served without a key (#22); the hashes sit next to the files, so they detect accidents, not a deliberate edit of both. A stored-XSS path through `/detect` and the plate lookup was fixed in this branch (`3b158c8`).

## 9. What was the hardest bug?

Two vehicles crossing produced ID switches and wrong fines (13 switches, 5
false-positive fines, 2 missed fines over 8 synthetic crossing scenarios). The
natural fix was a better motion model. A frame-by-frame diagnosis showed the
tracker matched correctly whenever it was given two vehicles; the association
layer had merged the two riders into one body (IoU 0.429 > 0.3) before tracking.
The fix was to raise the merge thresholds to IoU 0.5 / containment 0.8, values
read off the geometry of cases that must and must not merge. The earlier test
asserted `id_switches >= 1`, i.e. it had pinned the bug.

- Commit `483c9ea`; `python3 evaluate_tracking.py --diagnose`; `eval/results/tracking_comparison.json`.
- Tests: `tests/test_tracking_eval.py::test_old_config_collapses_two_vehicles_into_one_body`, `::test_new_config_removes_the_wrong_fines`.
- Other real cases: FAILURE_ANALYSIS.md.

## 10. What is the biggest limitation?

Nothing after the detector has been measured on real footage. Tracking,
association, OCR, temporal confirmation, speed and the fines themselves are
validated on synthetic inputs only, and the detector itself only on held-out
images from the same pool it was trained on, with a dataset whose source and
licence are unverified and which cannot be redistributed. Second: the class that
decides most fines, helmet vs no helmet, is the weakest (WithHelmet AP@50 0.387,
n = 27). The README section "How This Could Be Validated Externally" lists the
next steps; roadmap issues #8 and #9 cover the field dataset and a blind label
audit.
