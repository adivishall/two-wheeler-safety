# Pipeline evaluation (model-free)

- Generated: 2026-09-14T09:22:14.416725+00:00
- Inputs: deterministic synthetic scenarios driven through the **real** tracker, association, state machines, OCR stabilizer and speed estimator.
- No model weights, no dataset, no network. Every number is reproducible with `python3 evaluate_pipeline.py`.

> These measure **pipeline logic**, not detector quality. Detector metrics live in `docs/MODEL_EVALUATION.md` and are a separate question — see the note on mixing them there.

## End-to-end fines

- precision **1.0**, recall **1.0**
- TP 8 · FP 0 · FN 0 · wrong-vehicle 0 · wrong-plate 0 · duplicates 0

| scenario | TP | FP | FN | ID switches | assoc acc |
|---|---:|---:|---:|---:|---:|
| `single_no_helmet` | 1 | 0 | 0 | 0 | 1.0 |
| `single_triple` | 1 | 0 | 0 | 0 | 1.0 |
| `late_plate` | 1 | 0 | 0 | 0 | 1.0 |
| `clean_helmet` | 0 | 0 | 0 | 0 | 1.0 |
| `two_adjacent` | 1 | 0 | 0 | 0 | 1.0 |
| `crossing` | 2 | 0 | 0 | 0 | 1.0 |
| `three_bikes` | 1 | 0 | 0 | 0 | 1.0 |
| `occlusion` | 1 | 0 | 0 | 0 | 1.0 |

## Per-violation pipeline decision

How often the *complete* pipeline reaches the correct verdict for a vehicle — not raw detector accuracy.

| violation | P | R | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `helmet` | 1.0 | 1.0 | 1.0 | 2 | 0 | 0 | 3 |
| `triple_riding` | 1.0 | 1.0 | 1.0 | 4 | 0 | 0 | 2 |

### What the temporal confirmation window buys

`confirm_window=1` is single-frame fining — the behaviour the temporal layer replaces.

| violation | window | precision | recall |
|---|---:|---:|---:|
| `helmet` | 1 | 0.6667 | 1.0 |
| `helmet` | 2 | 1.0 | 1.0 |
| `helmet` | 3 | 1.0 | 1.0 |
| `helmet` | 5 | 1.0 | 1.0 |
| `helmet` | 8 | 1.0 | 1.0 |
| `triple_riding` | 1 | 0.8 | 1.0 |
| `triple_riding` | 2 | 1.0 | 1.0 |
| `triple_riding` | 3 | 1.0 | 1.0 |
| `triple_riding` | 5 | 1.0 | 1.0 |
| `triple_riding` | 8 | 1.0 | 1.0 |

## Is the pipeline better than a single-frame detector?

Both policies see **identical** detections, degraded by the same helmet class-confusion noise. `naive` fines whenever any single frame shows a violation box — no tracking, no temporal confirmation, no contradiction check.

Both policies see identical detections. The naive policy is handed perfect plate-to-vehicle association for free, which a real single-frame system would not have, so the pipeline's measured advantage is a lower bound.

| detector noise | naive P | naive R | naive F1 | naive FPs | pipeline P | pipeline R | pipeline F1 | pipeline FPs | F1 gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.8377 | 1.0 | 0.9116 | 1.5 | 1.0 | 1.0 | 1.0 | 0.0 | **+0.088** |
| 0.10 | 0.7345 | 1.0 | 0.8402 | 3.425 | 0.9975 | 0.9917 | 0.9943 | 0.025 | **+0.154** |
| 0.20 | 0.708 | 1.0 | 0.8198 | 4.075 | 0.976 | 0.9528 | 0.9619 | 0.225 | **+0.142** |
| 0.30 | 0.6966 | 1.0 | 0.8105 | 4.4 | 0.9624 | 0.8889 | 0.9174 | 0.3 | **+0.107** |
| 0.40 | 0.6941 | 1.0 | 0.8084 | 4.475 | 0.903 | 0.8222 | 0.8511 | 0.725 | **+0.043** |

The naive policy always scores recall 1.000 because it fines on anything — so the whole difference is precision. The pipeline is a precision machine, and that is the right objective for a system that fines people.

## Error budget (equal-rate fault injection)

Equal-rate fault injection: each stage is degraded by the same amount and the end-to-end F1 drop is measured. This ranks how much the system DEPENDS on each stage. It is NOT a claim about how often each stage fails in the field — that needs field data.

Clean baseline end-to-end F1: **1.0** (20 trials per injection, seed 7).

| stage | injected rate | mean F1 | F1 drop | share of measured sensitivity |
|---|---:|---:|---:|---:|
| `ocr` | 0.3 | 0.5467 | **0.4533** | 76.8% |
| `detection_class` | 0.3 | 0.9097 | **0.0903** | 15.3% |
| `detection` | 0.3 | 0.9569 | **0.0431** | 7.3% |
| `plate_recall` | 0.3 | 0.9966 | **0.0034** | 0.6% |

**Most sensitive stage: `ocr`.**

## OCR decision policy — single-frame vs temporal

Simulated OCR noise, not field data. These numbers describe the stated character-error model; they measure the decision rule, not EasyOCR's accuracy on real plates.

`coverage` is the share of vehicles the policy names at all; `accuracy|answered` is how often it is right **when it answers**. A wrong plate fines an innocent rider; an abstention only misses a fine, so both are reported.

| substitution rate | policy | coverage | accuracy\|answered |
|---|---|---:|---:|
| 0.04 | `last` | 1.0 | 0.32 |
| 0.04 | `best_conf` | 1.0 | 0.93 |
| 0.04 | `temporal` | 0.78 | 0.9872 |
| 0.08 | `last` | 1.0 | 0.19 |
| 0.08 | `best_conf` | 1.0 | 0.84 |
| 0.08 | `temporal` | 0.64 | 1.0 |
| 0.12 | `last` | 1.0 | 0.15 |
| 0.12 | `best_conf` | 1.0 | 0.7 |
| 0.12 | `temporal` | 0.51 | 0.9804 |
| 0.20 | `last` | 1.0 | 0.06 |
| 0.20 | `best_conf` | 1.0 | 0.45 |
| 0.20 | `temporal` | 0.23 | 0.913 |
| 0.30 | `last` | 1.0 | 0.05 |
| 0.30 | `best_conf` | 1.0 | 0.23 |
| 0.30 | `temporal` | 0.11 | 0.9091 |

## Speed estimation

| motion | calibration | MAE (km/h) | RMSE | bias | overspeed P | overspeed R |
|---|---|---:|---:|---:|---:|---:|
| approach | `constant` | 41.767 | 44.783 | -41.767 | 1.0 | 0.0 |
| approach | `linear` | 40.05 | 43.063 | -40.05 | 1.0 | 0.0 |
| approach | `homography` | 11.283 | 13.764 | 8.383 | 0.6 | 1.0 |
| approach | **best: `homography`** | | | | | |
| lateral | `constant` | 1.85 | 2.282 | -1.817 | 1.0 | 1.0 |
| lateral | `linear` | 0.867 | 0.983 | -0.2 | 1.0 | 1.0 |
| lateral | `homography` | 3.183 | 4.671 | 3.183 | 0.6 | 1.0 |
| lateral | **best: `linear`** | | | | | |

