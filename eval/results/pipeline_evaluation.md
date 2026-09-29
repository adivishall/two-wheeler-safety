# Pipeline evaluation (model-free)

- Generated: 2026-09-29T00:13:14.337811+00:00
- Inputs: deterministic synthetic scenarios driven through the **shipped** `ViolationPipeline` (the object the video job runs per frame, with its defaults) and the real speed estimator.
- No model weights, no dataset, no network. Every number is reproducible with `python3 evaluate_pipeline.py`.

> These measure **pipeline logic**, not detector quality. Detector metrics live in `docs/MODEL_EVALUATION.md` and are a separate question — see the note on mixing them there.

## End-to-end fines

- precision **1.0**, recall **0.9**
- TP 9 · FP 0 · FN 1 · wrong-vehicle 0 · wrong-plate 0 · duplicates 0

| scenario | TP | FP | FN | ID switches | assoc acc |
|---|---:|---:|---:|---:|---:|
| `single_no_helmet` | 1 | 0 | 0 | 0 | 1.0 |
| `single_triple` | 1 | 0 | 0 | 0 | 1.0 |
| `late_plate` | 1 | 0 | 0 | 0 | 1.0 |
| `clean_helmet` | 0 | 0 | 0 | 0 | 1.0 |
| `two_adjacent` | 1 | 0 | 0 | 0 | 1.0 |
| `crossing` | 2 | 0 | 0 | 0 | 0.95 |
| `three_bikes` | 1 | 0 | 0 | 0 | 1.0 |
| `occlusion` | 1 | 0 | 0 | 0 | 1.0 |
| `brief_pass` | 0 | 0 | 1 | 0 | 1.0 |
| `reappears_after_exit` | 1 | 0 | 0 | 1 | 1.0 |

## Per-violation pipeline decision

How often the *complete* pipeline reaches the correct verdict for a vehicle — not raw detector accuracy.

| violation | P | R | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `helmet` | 1.0 | 1.0 | 1.0 | 2 | 0 | 0 | 3 |
| `triple_riding` | 1.0 | 1.0 | 1.0 | 4 | 0 | 0 | 2 |

### What the streak length alone buys (sanity check)

`confirm_window=1` is single-frame fining. Helmet dwell gate off here; these scenarios' flickers are one frame long by construction, so this cannot choose a rule — `eval/results/temporal_confirmation.md` does.

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
| 0.00 | 0.8516 | 1.0 | 0.9199 | 1.5 | 1.0 | 0.9546 | 0.9762 | 0.0 | **+0.056** |
| 0.10 | 0.722 | 1.0 | 0.8313 | 4.383 | 1.0 | 0.953 | 0.9753 | 0.0 | **+0.144** |
| 0.20 | 0.718 | 1.0 | 0.8282 | 4.5 | 1.0 | 0.9212 | 0.9567 | 0.0 | **+0.129** |
| 0.30 | 0.718 | 1.0 | 0.8282 | 4.5 | 1.0 | 0.7879 | 0.8611 | 0.0 | **+0.033** |
| 0.40 | 0.718 | 1.0 | 0.8282 | 4.5 | 0.9925 | 0.606 | 0.6649 | 0.033 | **-0.163** |

The naive policy always scores recall 1.000 because it fines on anything — so the whole difference is precision. The pipeline keeps precision near 1 at every noise level; its F1 advantage holds in the range the detector actually operates in (val-measured helmet flips 8–15% per frame) and reverses at extreme symmetric noise, where it trades recall for precision. That is the right trade for a system that fines people, and it is a trade.

## Error budget — oracle ablation at the measured operating point

Each stage fails at its per-frame rate measured on the validation split (OCR, unmeasured, is swept); making one stage perfect and measuring the fine-F1 it recovers attributes the error. Synthetic scenarios, so this ranks where effort pays off in THIS pipeline; it is NOT a claim about how often each stage fails in the field.

- Detector rates per vehicle per frame, from `eval/results/eval_traffic_model-2_val_clean.json (confusion matrix, conf 0.25)`: rider box missed WithHelmet 15.4%, WithoutHelmet 20.1%, TripleRiding 10.3%; wrong class WithHelmet→WithoutHelmet 15.4%, WithoutHelmet→WithHelmet 7.7%, TripleRiding→WithoutHelmet 5.2%; plate missed 18.2%.
- OCR read-error rate (≥1 wrong glyph per read) is **unmeasured** — no labelled plate sequences exist — so it is swept.
- No-fault F1 (the suite's by-design misses only): **0.9697**; 30 trials, seed 7.

**OCR read-error rate 10%** — all faults on: F1 0.9547 (gap to no-fault 0.015)

| stage made perfect | F1 | recovered | share |
|---|---:|---:|---:|
| `rider_recall` — detector misses the rider box | 0.9697 | **0.015** | 54% |
| `helmet_class` — detector labels the rider with the wrong helmet/triple class | 0.9666 | **0.0119** | 42% |
| `plate_recall` — detector misses the plate box (nothing to OCR) | 0.9558 | **0.0011** | 4% |
| `ocr` — OCR reads the plate with a wrong glyph | 0.9547 | **0.0** | 0% |

**OCR read-error rate 30%** — all faults on: F1 0.9557 (gap to no-fault 0.014)

| stage made perfect | F1 | recovered | share |
|---|---:|---:|---:|
| `rider_recall` — detector misses the rider box | 0.9697 | **0.014** | 54% |
| `helmet_class` — detector labels the rider with the wrong helmet/triple class | 0.9666 | **0.0109** | 42% |
| `plate_recall` — detector misses the plate box (nothing to OCR) | 0.9567 | **0.0011** | 4% |
| `ocr` — OCR reads the plate with a wrong glyph | 0.9547 | **0.0** | 0% |

**OCR read-error rate 50%** — all faults on: F1 0.9546 (gap to no-fault 0.0151)

| stage made perfect | F1 | recovered | share |
|---|---:|---:|---:|
| `rider_recall` — detector misses the rider box | 0.9697 | **0.0151** | 56% |
| `helmet_class` — detector labels the rider with the wrong helmet/triple class | 0.9655 | **0.0109** | 40% |
| `plate_recall` — detector misses the plate box (nothing to OCR) | 0.9557 | **0.0011** | 4% |
| `ocr` — OCR reads the plate with a wrong glyph | 0.9547 | **0.0001** | 0% |

Largest owner at the central OCR assumption: **`rider_recall`** (holds across the whole OCR sweep).

### Tolerance — one stage failing alone, same unit for all

End-to-end fine F1 when only that stage fails (probability per vehicle per frame).

| stage | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---:|---:|---:|---:|
| `rider_recall` | 0.9619 | 0.9576 | 0.9142 | 0.6111 |
| `helmet_class` | 0.9697 | 0.9502 | 0.8304 | 0.5483 |
| `plate_recall` | 0.9697 | 0.9697 | 0.9697 | 0.9603 |
| `ocr` | 0.9697 | 0.9697 | 0.9697 | 0.9697 |

## OCR decision policy — single-frame vs temporal

Simulated OCR noise, not field data. These numbers describe the stated character-error model; they measure the decision rule, not EasyOCR's accuracy on real plates.

`coverage` is the share of vehicles the policy names at all; `accuracy|answered` is how often it is right **when it answers**. A wrong plate fines an innocent rider; an abstention only misses a fine, so both are reported.

| substitution rate | policy | coverage | accuracy\|answered |
|---|---|---:|---:|
| 0.04 | `last` | 1.0 | 0.32 |
| 0.04 | `best_conf` | 1.0 | 0.93 |
| 0.04 | `temporal` | 0.64 | 1.0 |
| 0.08 | `last` | 1.0 | 0.19 |
| 0.08 | `best_conf` | 1.0 | 0.84 |
| 0.08 | `temporal` | 0.44 | 1.0 |
| 0.12 | `last` | 1.0 | 0.15 |
| 0.12 | `best_conf` | 1.0 | 0.7 |
| 0.12 | `temporal` | 0.28 | 1.0 |
| 0.20 | `last` | 1.0 | 0.06 |
| 0.20 | `best_conf` | 1.0 | 0.45 |
| 0.20 | `temporal` | 0.12 | 1.0 |
| 0.30 | `last` | 1.0 | 0.05 |
| 0.30 | `best_conf` | 1.0 | 0.23 |
| 0.30 | `temporal` | 0.02 | 1.0 |

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

