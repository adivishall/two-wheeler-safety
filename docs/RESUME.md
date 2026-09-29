# Résumé material

Every number here comes from a file committed under `eval/results/` (or from the
test suite), produced by the command listed with it. Nothing is estimated or
rounded in the flattering direction. **If you cannot point at the file, do not
use the number.** Pipeline numbers are on synthetic inputs and must be described
that way; none of them is field accuracy.

---

## Three bullets

> **Built an end-to-end traffic-violation evidence pipeline** around a YOLOv8
> detector — Hungarian rider↔plate association, multi-object tracking, temporal
> OCR voting, violation state machines, tamper-evident evidence (SHA-256) and a
> Flask review dashboard — with one model-free decision core shared by the
> production job and every evaluator, covered by 601 model-free tests at ~95%
> branch coverage of the core modules.

> **Chose decision thresholds by experiment instead of by hand**: simulated riders
> at the detector's validation-measured error rates showed a consecutive-frame
> rule flags **31.5%** of helmeted riders seen for 2 s under assumed correlated
> errors; a track-level fraction gate cut it to **1.2%** on a held-out seed (no
> rule met the 1% target everywhere; this was the lowest worst case). Selected OCR
> vote thresholds the same way, scored at the moment a fine is committed:
> simulated wrong-plate fines **40–62% → ≤ 1.8%**, paid for by withholding
> 30–66% of plates for human review.

> **Ran a hostile audit of v1.0 and fixed 30 defects**, including
> evaluators that did not run the shipped code, OCR on stale plate boxes,
> duplicate fines on re-entry, speed read at 2/3 of true on downscaled video, and
> model decisions made on test-set noise — a retrain "rejected" for a test-split
> WithHelmet regression (ΔAP −0.039, 95% CI [−0.210, +0.087]) was noise, and
> selection moved to validation with paired bootstrap CIs; an independent review
> of the fixes found 8 more (7 pinned by regression tests).

## More bullet candidates

- **Dataset audit.** Perceptual-hash leakage audit found **9.8%** of the test
  split duplicated training images; a label-coverage audit found the dataset is
  two merged sources with **disjoint label sets**, so the detector finds a plate
  on **6%** of triple-riding images vs **65%** elsewhere — a failure no held-out
  metric could reveal.
- **Evaluation protocol.** Corrected mAP to the standard protocol (test mAP@50
  **0.727 → 0.769**, same weights) and replaced point-estimate model comparisons
  with a paired image bootstrap; a retrain previously "rejected" for a WithHelmet
  regression was statistically indistinguishable.
- **Seed variance.** Retrained the shipped recipe with only the seed changed: the
  paired bootstrap called it "significantly worse" (test mAP@50 **−0.076**
  [−0.155, −0.003]) — a checkpoint-level test cannot compare training recipes.
  Re-graded three model verdicts and made promotion require ≥ 3 seeds per
  recipe.
- **Error budget (synthetic scenarios).** Oracle ablation at the measured
  operating point: fixing rider detection alone closes the whole 1.5-point F1
  gap, helmet classification alone **72–80%** of it, plate detection 7–8%, OCR
  ~0 — replacing an earlier budget whose "OCR is 77%" came from comparing
  per-character with per-box noise.
- **Tracking (synthetic scenarios).** Root-caused ID switches in crossing
  traffic to rider-box merge thresholds, not the tracker: **13 → 0** ID switches
  and association accuracy **0.760 → 0.989** across 8 crossing/overlap
  scenarios.
- **Robustness.** Synthetic-corruption suite with paired CIs: 21-px motion blur
  costs plate AP **−0.66**; mild glare costs WithHelmet **−0.16**.
- **Precision over recall, measured (synthetic scenarios).** Against a
  fine-on-any-frame baseline on identical detections, the pipeline keeps
  precision ~**1.0** vs **0.72–0.85** and gains **+0.13–0.14 F1** at 10–20%
  simulated helmet flips (measured: 8–15%).
- **Performance, measured as an A/B.** Profiled the video job per stage (YOLO
  65%, OCR 24% of frame time) and benchmarked the OCR lock with interleaved
  repeated runs: **19.4 → 36.7 FPS (+88%)** on the same clip, with identical
  fines in all 10 runs — after finding single runs too noisy to publish (the
  unlocked arm alone spans 15.3–20.5 FPS over 5 runs).

## Evidence behind each number

| number | file | command |
|---|---|---|
| 601 tests, ~95% branch coverage of `modules/` (model-loading modules excluded) | test run | `pytest --cov` |
| 31.5% → 1.2% helmeted riders flagged | `temporal_confirmation.md` (held-out seed, stickiness 0.5, 2 s) | `python3 evaluate_temporal.py` |
| wrong plate 40–62% → ≤ 1.8%, 34–70% coverage | `ocr_stabilizer_selection.md` (held-out seed, 4–12% character noise, scored at first election) | `python3 evaluate_ocr.py --simulate --sweep --policy-sweep --out eval/results --name ocr_policy_simulation` |
| 30 defects + 8 from the independent review | `docs/AUDIT.md` | — (each row names its fix and test) |
| WithHelmet ΔAP −0.039 [−0.210, +0.087] | `uncertainty_test.md` | `python3 evaluate_uncertainty.py --split test …` |
| 9.8% test leakage | `dataset_leakage.json` | `python3 audit_dataset.py --data …` |
| plate on 6% vs 65% of images | `label_audit.md` (model probe, val) | `python3 audit_labels.py --data … --model …` |
| seed replica −0.076 [−0.155, −0.003] | `uncertainty_test.md` (`traffic_model_seed1` vs `traffic_model-2`) | `python3 evaluate_uncertainty.py --split test …` |
| mAP@50 0.727 → 0.769 | `git show fc1679d:eval/results/eval_traffic_model-2_test_clean.json` (conf 0.25) vs the current file (conf 0.001) | `python3 evaluate_model.py … --split test` |
| 19.4 → 36.7 FPS, +88%, identical fines | `benchmark.md` (Apple M4, 5 interleaved rounds per arm) | `python3 benchmark.py --model … --video demo_traffic.mp4 --ocr-lock-ab --micro` |
| rider detection 100% / helmet class 72–80% / plate 7–8% of the F1 gap, each alone | `pipeline_evaluation.md` § Error budget | `python3 evaluate_pipeline.py` |
| 13 → 0 ID switches, 0.760 → 0.989 | `tracking_comparison.json` | `python3 evaluate_tracking.py --json …` |
| −0.66 / −0.16 AP | `robustness_val.md` | `python3 evaluate_robustness.py` |
| precision ~1.0 vs 0.72–0.85, +0.13–0.14 F1 (10–20% simulated flips) | `pipeline_evaluation.md` § single-frame | `python3 evaluate_pipeline.py` |

## Detector numbers, if asked

`traffic-4class@1.0.0`, de-leaked test split (175 images), standard protocol:
mAP@50 **0.769**, mAP@50-95 **0.558** (`eval_traffic_model-2_test_clean.json`).
Per class AP@50 (`predict()`, 95% CI, `uncertainty_test.md`): TripleRiding 0.963
[0.897, 0.995], Plate 0.889 [0.837, 0.938], WithoutHelmet 0.734 [0.655, 0.820],
**WithHelmet 0.415 [0.204, 0.729]**. Say the WithHelmet interval out loud — it is
the honest headline. And say these are one checkpoint, the better of two seeds
(the other: WithHelmet 0.212).

## How to talk about it

- Lead with the separation: *what the model can see* vs *what the software does
  with it*, measured independently.
- Every "improvement" has a cost that is stated: the helmet gate never fines a
  rider seen < 12 frames for no helmet; the OCR vote withholds more plates.
- The strongest stories are the reversals: the evaluator that wasn't running the
  shipped code, the OCR choice that didn't survive 5× the sample, the OCR
  experiment that scored plates later than the pipeline commits them, the
  "rejected" model that was noise, the "significant" difference that a new seed
  produces on its own, the benchmark labelled "mps" that ran on the CPU.
- If the 40–62% sounds implausible, say why: the 1.0.0 rule could elect a plate
  from its first readable read, so at the commit point it was about as good as
  trusting one or two reads.
- Don't say "accurate", "robust" or "production-ready". Say what was measured,
  on what, and what wasn't.
- Be straightforward that the audit, the independent review and the error
  review were done with an AI coding assistant (AUDIT.md says so). What you
  should be able to do unaided is explain every finding, why it mattered, and
  how the fix is tested — interviewers will probe exactly that.
