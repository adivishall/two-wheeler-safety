# Visual review of the highest-confidence validation errors

> Visual review of the 16 highest-confidence false positives and 16 highest-confidence class confusions on the de-leaked VALIDATION split (traffic-4class@1.0.0, conf 0.25). Reviewer: the AI coding assistant (Claude), viewing each saved crop — not a human annotator, and not yet re-checked by one. Judgements made at crop resolution: they are an error-analysis aid, NOT ground truth, and several are marked ambiguous. Re-check any row with the listed artifact (evaluate_model.py --save-artifacts eval/artifacts_val).

## Tally

| set | missing label (model right) | label error | label error (probable) | model error | ambiguous |
|---|---:|---:|---:|---:|---:|
| top-16 false positives | 11 | 0 | 0 | 4 | 1 |
| top-16 class confusions | 0 | 3 | 3 | 7 | 3 |

## False positives

| # | predicted | labelled | score | judgement |
|---:|---|---|---:|---|
| 0 | WithoutHelmet | — | 0.90 | model error:helmeted rider called no helmet(likely) |
| 1 | WithHelmet | — | 0.90 | missing label:unlabelled rider |
| 2 | WithoutHelmet | — | 0.88 | missing label:bareheaded rider |
| 3 | WithoutHelmet | — | 0.81 | missing label:bareheaded rider(night) |
| 4 | WithoutHelmet | — | 0.77 | missing label:bareheaded rider |
| 5 | Plate | — | 0.75 | missing label:plate |
| 6 | Plate | — | 0.69 | missing label:plate |
| 7 | WithoutHelmet | — | 0.69 | missing label:rider in cap |
| 8 | WithoutHelmet | — | 0.69 | model error:helmeted person with bicycle called no helmet |
| 9 | WithoutHelmet | — | 0.67 | model error:pedestrian beside parked motorbike |
| 10 | WithoutHelmet | — | 0.67 | missing label:bareheaded rider |
| 11 | WithoutHelmet | — | 0.59 | model error:rotated image non rider |
| 12 | WithHelmet | — | 0.58 | missing label:helmeted rider |
| 13 | WithoutHelmet | — | 0.52 | missing label:bareheaded rider |
| 14 | Plate | — | 0.50 | missing label:plate |
| 15 | WithoutHelmet | — | 0.49 | ambiguous:truncated rider at edge |

## Class confusions

| # | predicted | labelled | score | judgement |
|---:|---|---|---:|---|
| 0 | WithHelmet | WithoutHelmet | 0.95 | ambiguous:light headwear |
| 1 | WithHelmet | WithoutHelmet | 0.93 | label error:full face helmet labelled WithoutHelmet |
| 2 | WithHelmet | WithoutHelmet | 0.93 | label error probable:white helmet or cap |
| 3 | WithHelmet | WithoutHelmet | 0.92 | label error probable:dark helmet |
| 4 | WithoutHelmet | WithHelmet | 0.90 | label error probable:no helmet visible labelled WithHelmet |
| 5 | WithHelmet | WithoutHelmet | 0.90 | model error:head covering(dupatta) called helmet |
| 6 | WithHelmet | WithoutHelmet | 0.89 | model error:head covering(dupatta) called helmet |
| 7 | WithHelmet | WithoutHelmet | 0.88 | model error:head covering(scarf) called helmet |
| 8 | WithoutHelmet | WithHelmet | 0.85 | model error:helmeted rider called no helmet |
| 9 | Plate | WithoutHelmet | 0.81 | label error:plate labelled WithoutHelmet |
| 10 | WithoutHelmet | WithHelmet | 0.81 | label error:bareheaded riders labelled WithHelmet |
| 11 | WithHelmet | WithoutHelmet | 0.80 | model error:bareheaded rider called helmet |
| 12 | WithHelmet | WithoutHelmet | 0.76 | ambiguous:resolution |
| 13 | WithoutHelmet | WithHelmet | 0.74 | model error:helmeted rider called no helmet |
| 14 | WithHelmet | WithoutHelmet | 0.74 | ambiguous:hood |
| 15 | WithHelmet | WithoutHelmet | 0.65 | model error:head covering(scarf) called helmet |
