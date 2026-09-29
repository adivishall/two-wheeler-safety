# Evaluation — methodology and reproducibility

Three questions get called "performance" in a CV system. They are measured by
different tools, on different inputs, and are never combined into one number.

| question | measured on | where |
|---|---|---|
| **Model** — what can the detector see? | real, de-leaked held-out images | [MODEL_EVALUATION.md](MODEL_EVALUATION.md) |
| **Pipeline** — how reliably do noisy detections become a correct record? | synthetic scenarios through the shipped decision core; detector noise at *measured* rates | [END_TO_END_EVALUATION.md](END_TO_END_EVALUATION.md) |
| **Application** — how fast, how much memory? | a local image and video on stated hardware | §4 below |

Failure analysis and the error budget that connects them:
[ERROR_ANALYSIS.md](ERROR_ANALYSIS.md). Defects found by auditing the previous
version of all of this: [AUDIT.md](AUDIT.md).

## 1. Rules every measurement follows

1. **Choose on validation, report on test once.** Model choice, per-class
   thresholds and the operating-point error rates that feed the decision
   experiments all come from val. Test is evaluated for the chosen model only.
2. **Uncertainty or it didn't happen.** Detector differences use a paired image
   bootstrap; a gap is real only if its 95% CI excludes zero.
3. **Decision rules are chosen by a stated objective**, on one random seed, and
   reported on another. If the choice changes with a larger sample, the larger
   sample wins and the reversal is recorded (DECISIONS #16).
4. **Measure the shipped code.** Evaluators drive the same `ViolationPipeline`
   the web job runs, with its defaults.
5. **Label synthetic as synthetic.** Every simulated or transformed input is
   stamped as such in its result file, and no synthetic number is presented as
   field accuracy.
6. **Provenance on every result.** Weights SHA-256, dataset content fingerprint,
   split fingerprint, git commit + dirty flag, library versions, hardware
   (`modules/provenance.py`).

## 2. The tools

| tool | produces (`eval/results/`) | needs |
|---|---|---|
| `audit_dataset.py` | `dataset_leakage.json`, de-leaked `eval/clean_splits/` | dataset |
| `audit_labels.py` | `label_audit.{json,md}` — label integrity, source mix, co-occurrence, per-source model probe | dataset (+ weights for the probe) |
| `dataset_manifest.py` | `data/DATASET_MANIFEST.md`, `data/dataset_manifest.json` | dataset |
| `evaluate_model.py` | `eval_<model>_<split>_clean.{json,md}` — standard-protocol mAP, operating-point confusions, failure crops | weights + dataset |
| `evaluate_uncertainty.py` | `uncertainty_{val,test}.{json,md}` — AP with CIs, paired model differences, threshold selection/transfer | weights + dataset |
| `evaluate_robustness.py` | `robustness_val.{json,md}` — AP change under synthetic corruptions | weights + dataset |
| `compare_models.py` | `model_comparison*.md` — point estimates + latency (selection split: val) | weights + dataset |
| `evaluate_pipeline.py` | `pipeline_evaluation.{json,md,csv}`, `latest.json` (dashboard) | nothing |
| `evaluate_temporal.py` | `temporal_confirmation.{json,md}` | nothing (rates from the val report) |
| `evaluate_ocr.py --simulate --sweep --policy-sweep` | `ocr_policy_simulation.json`, `ocr_stabilizer_selection.md` | nothing |
| `evaluate_ocr.py --labels data/ocr_sanity/labels.csv` | EasyOCR on synthetic plate renders (a tooling check, not accuracy) | easyocr |
| `benchmark.py` | `benchmark.{json,md}` | weights + local image/video |
| `calibrate_confidence.py` | reliability / ECE / Brier, Platt and isotonic fits | reviewed outcomes (none yet) |

`manual_error_review.{json,md}` is the one hand-made result: a visual review of
the top-confidence val errors, listed crop by crop so anyone can re-check it.
The reviewer was the AI coding assistant viewing the crops, not a human
annotator; until a person re-checks it, cite it as that.

## 3. Confidence calibration — deliberately not applied

`modules/calibration.py` can fit Platt or isotonic calibration and report ECE,
MCE and Brier; it is validated on synthetic data (an overconfident set's ECE
≈ 0.25 is materially reduced; a calibrated set stays < 0.02). It is **not
applied**, because there is nothing honest to fit it on: calibration needs each
stored score joined to a verified outcome, which would come from the human-review
workflow (`review_status` confirmed / dismissed). Until then the confidence is a
ranking score. Even as a ranking it is only established for WithoutHelmet and
Plate; for WithHelmet and TripleRiding the score's rank correlation with
correctness flips sign between val and test ([MODEL_EVALUATION.md](MODEL_EVALUATION.md)).

## 4. Application performance

`eval/results/benchmark.md` (+ `.json`), one invocation of `benchmark.py` at
commit `8a68089` on a clean tree. Protocol:

| | |
|---|---|
| hardware | Apple M4, macOS 26.5 |
| software | Python 3.13.7 · torch 2.12.1 · Ultralytics 8.4.71 · OpenCV 4.13.0 |
| model | `traffic-4class@1.0.0` (YOLOv8n, sha256 `d49f7983…`), imgsz 640 |
| inputs | `samples/test.jpg` 1906×1078; `demo_traffic.mp4` 334×596, 120 frames @ 25 fps, one plate — each identified by sha256 in the JSON |
| batch | 1 (the pipeline is frame-by-frame) |
| cold vs warm | model load and the first call reported separately; everything else warm |
| repeats | image: 50 calls per device; video: 5 interleaved rounds per arm (A B, B A, …), median run reported with the range |

| measurement | result |
|---|---|
| model load, cold (YOLO + EasyOCR) | 3.6 s |
| YOLO, one image, warm p50 | MPS 23.2 ms · CPU 20.5 ms (first MPS call 402 ms; one run per device) |
| EasyOCR, one plate crop | 13.5 ms mean |
| **video, shipped settings (MPS)** | **36.7 FPS** (range 35.0–37.2); YOLO 65% of wall, OCR 24% |
| video, OCR lock off | 19.4 FPS (range 15.3–20.5); OCR 56% of wall |
| OCR lock A/B | **+88%** (per round +80% to +139%); 120 → 30 OCR calls; same fine in all 10 runs |
| decision core, one frame, three bikes | 0.04 ms |
| recording a fine (SQLite) | 0.42 ms |
| `/analyze` HTTP + validation + DB overhead | 1.4 ms median of 30 paired calls, p10–p90 −7 to +8 ms: below the call's own noise |
| peak RSS | 942 MB |

What these numbers do and don't say:

- **The OCR lock is the only optimization measured as an A/B**, and the
  benchmark checks it changed nothing: every run of both arms fined the same
  plate for the same violation. Locked plates are still re-read every 10 frames
  (AUDIT R2), which is where the 30 remaining calls come from.
- **No claim that MPS is faster for one YOLOv8n pass on an M4.** The committed
  run has the CPU slightly ahead, from one 50-call run per device, not
  interleaved; earlier uncommitted runs had MPS ahead. Nothing here supports
  "the GPU is faster".
  Before 1.1.0 inference silently ran on the CPU (AUDIT E11); the fix was about
  reporting the device honestly, not about speed.
- **Separate invocations varied more than rounds within one.** Two earlier
  single-round runs on the same machine gave 32.6 and 26.9 FPS for the shipped
  arm — which is why the benchmark now repeats and interleaves. Treat
  throughput as roughly ±15% between sessions.
- **One small clip with one plate.** OCR cost scales with plates in view; this
  is not a throughput claim for a busy junction or a full-HD camera.


## 5. Regenerating everything

```bash
# model-free (this is what CI runs, at reduced trial counts)
python3 evaluate_pipeline.py
python3 evaluate_temporal.py
python3 evaluate_ocr.py --simulate --sweep --policy-sweep --out eval/results --name ocr_policy_simulation

# needs the dataset and weights (not shipped)
python3 audit_dataset.py --data master_traffic_violation_dataset/data.yaml --write-clean-split eval/clean_splits
python3 audit_labels.py --data master_traffic_violation_dataset/data.yaml --model runs/detect/traffic_model-2/weights/best.pt
M=runs/detect/traffic_model-2/weights/best.pt
python3 evaluate_model.py --model $M --data eval/clean_splits/data.yaml --split val \
    --out eval/results --name eval_traffic_model-2_val_clean --save-artifacts eval/artifacts_val
python3 evaluate_model.py --model $M --data eval/clean_splits/data.yaml --split test \
    --out eval/results --name eval_traffic_model-2_test_clean
python3 evaluate_uncertainty.py --split val --name uncertainty_val --models $M <candidates…>
python3 evaluate_uncertainty.py --split test --name uncertainty_test \
    --thresholds-from eval/results/uncertainty_val.json --models $M <chosen…>
python3 evaluate_robustness.py
python3 benchmark.py --model $M --image samples/test.jpg --video demo_traffic.mp4 \
    --max-frames 120 --ocr-lock-ab --devices cpu mps --micro
```

Performance numbers are only comparable on an otherwise idle machine; the
benchmark records the hardware and software it ran on, not what else was
running — so don't run it next to a training job. `--repeats` (default 5) sets
the interleaved rounds per video arm.
