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
> production job and every evaluator, covered by 567 model-free tests at 95%
> branch coverage.

> **Chose decision thresholds by experiment instead of by hand**: simulated riders
> at the detector's validation-measured error rates showed a consecutive-frame
> rule flags **31.5%** of helmeted riders seen for 2 s under correlated errors; a
> track-level fraction gate cut it to **1.2%** on a held-out seed. Selected OCR
> vote thresholds the same way, cutting simulated wrong-plate fines from **3.0% to
> 1.2%**.

> **Audited my own v1.0 as a hostile reviewer and fixed 30 defects**, including
> evaluators that did not run the shipped code, OCR on stale plate boxes,
> duplicate fines on re-entry, speed read at 2/3 of true on downscaled video, and
> model decisions made on test-set noise — re-running selection on validation
> with paired bootstrap CIs (WithHelmet ΔAP −0.039, 95% CI [−0.210, +0.087]).

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
- **Error budget.** Oracle-ablation error budget at the measured operating point:
  the detector (missed rider boxes, helmet confusion) owns ~**90%** of lost
  end-to-end F1 — replacing an earlier budget whose "OCR is 77%" came from
  comparing per-character with per-box noise.
- **Tracking.** Root-caused ID switches in crossing traffic to rider-box merge
  thresholds, not the tracker: **13 → 0** ID switches and association accuracy
  **0.760 → 0.989** across 8 crossing/overlap scenarios.
- **Robustness.** Synthetic-corruption suite with paired CIs: 21-px motion blur
  costs plate AP **−0.66**; mild glare costs WithHelmet **−0.16**.
- **Precision over recall, measured.** Against a fine-on-any-frame baseline on
  identical detections, the pipeline keeps precision ~**1.0** vs **0.72–0.85** and
  gains **+0.13–0.14 F1** at the detector's measured noise rates.
<!-- RESUME-PERF -->

## Evidence behind each number

| number | file | command |
|---|---|---|
| 567 tests, 95% branch coverage | test run | `pytest --cov` |
| 31.5% → 1.2% helmeted riders flagged | `temporal_confirmation.md` (held-out seed, stickiness 0.5, 2 s) | `python3 evaluate_temporal.py` |
| wrong plate 3.0% → 1.2% | `ocr_stabilizer_selection.md` (held-out seed, worst case) | `python3 evaluate_ocr.py --simulate --sweep --policy-sweep --out eval/results --name ocr_policy_simulation` |
| 30 defects | `docs/AUDIT.md` | — (each row names its fix and test) |
| WithHelmet ΔAP −0.039 [−0.210, +0.087] | `uncertainty_test.md` | `python3 evaluate_uncertainty.py --split test …` |
| 9.8% test leakage | `dataset_leakage.json` | `python3 audit_dataset.py --data …` |
| plate on 6% vs 65% of images | `label_audit.md` (model probe, val) | `python3 audit_labels.py --data … --model …` |
| mAP@50 0.727 → 0.769 | `git show fc1679d:eval/results/eval_traffic_model-2_test_clean.json` (conf 0.25) vs the current file (conf 0.001) | `python3 evaluate_model.py … --split test` |
| ~90% of lost F1 owned by the detector | `pipeline_evaluation.md` § Error budget | `python3 evaluate_pipeline.py` |
| 13 → 0 ID switches, 0.760 → 0.989 | `tracking_comparison.json` | `python3 evaluate_tracking.py --json …` |
| −0.66 / −0.16 AP | `robustness_val.md` | `python3 evaluate_robustness.py` |
| precision ~1.0 vs 0.72–0.85, +0.13–0.14 F1 | `pipeline_evaluation.md` § single-frame | `python3 evaluate_pipeline.py` |

## Detector numbers, if asked

`traffic-4class@1.0.0`, de-leaked test split (175 images), standard protocol:
mAP@50 **0.769**, mAP@50-95 **0.558** (`eval_traffic_model-2_test_clean.json`).
Per class AP@50 (`predict()`, 95% CI, `uncertainty_test.md`): TripleRiding 0.963
[0.897, 0.995], Plate 0.889 [0.837, 0.938], WithoutHelmet 0.734 [0.655, 0.820],
**WithHelmet 0.415 [0.204, 0.729]**. Say the WithHelmet interval out loud — it is
the honest headline.

## How to talk about it

- Lead with the separation: *what the model can see* vs *what the software does
  with it*, measured independently.
- Every "improvement" has a cost that is stated: the helmet gate never fines a
  rider seen < 12 frames; the OCR vote withholds more plates.
- The strongest stories are the reversals: the evaluator that wasn't running the
  shipped code, the OCR choice that didn't survive 5× the sample, the "rejected"
  model that was noise, the benchmark labelled "mps" that ran on the CPU.
- Don't say "accurate", "robust" or "production-ready". Say what was measured,
  on what, and what wasn't.
