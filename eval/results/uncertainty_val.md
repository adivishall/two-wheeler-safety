# Detector metrics with uncertainty — `val` split

- Generated: 2026-09-29T00:11:54.757140+00:00
- Data: `eval/clean_splits/data.yaml` split **val** (352 images, fingerprint `sha256:032b9391d7cd2dc2`)
- Bootstrap: 2000 image resamples, 95% percentile intervals, seed 0
- AP@50 recomputed from cached predictions (conf ≥ 0.001, NMS IoU 0.7, imgsz 640); 101-point interpolation as in Ultralytics.

## AP@50 per class, with 95% CI

| model | Plate | WithHelmet | WithoutHelmet | TripleRiding | mAP@50 |
|---|---:|---:|---:|---:|---:|
| `traffic_model-2` | 0.873 [0.830, 0.912] | 0.417 [0.247, 0.625] | 0.735 [0.674, 0.796] | 0.861 [0.763, 0.948] | 0.721 [0.668, 0.786] |
| `traffic_model_seed1` | 0.849 [0.800, 0.891] | 0.318 [0.153, 0.525] | 0.745 [0.685, 0.806] | 0.814 [0.711, 0.917] | 0.681 [0.625, 0.748] |
| `traffic_model_v2_dedup` | 0.850 [0.802, 0.894] | 0.444 [0.265, 0.660] | 0.782 [0.725, 0.835] | 0.893 [0.812, 0.961] | 0.742 [0.686, 0.809] |
| `traffic_model_probe` | 0.832 [0.783, 0.879] | 0.312 [0.169, 0.506] | 0.750 [0.690, 0.803] | 0.786 [0.672, 0.893] | 0.670 [0.620, 0.736] |
| `traffic_model_r2` | 0.787 [0.729, 0.846] | 0.395 [0.234, 0.600] | 0.761 [0.704, 0.812] | 0.821 [0.712, 0.914] | 0.691 [0.637, 0.756] |

Instances: Plate 247, WithHelmet 26, WithoutHelmet 209, TripleRiding 58

## Paired differences vs `traffic_model-2`

AP(model) − AP(baseline) on the same resampled images. **Significant** only when the 95% CI excludes 0.

| model | class | Δ AP@50 | 95% CI | P(model better) | verdict |
|---|---|---:|---|---:|---|
| `traffic_model_seed1` | Plate | -0.024 | [-0.048, +0.002] | 0.04 | not distinguishable |
| `traffic_model_seed1` | WithHelmet | -0.098 | [-0.347, +0.129] | 0.20 | not distinguishable |
| `traffic_model_seed1` | WithoutHelmet | +0.009 | [-0.035, +0.059] | 0.69 | not distinguishable |
| `traffic_model_seed1` | TripleRiding | -0.047 | [-0.114, +0.009] | 0.06 | not distinguishable |
| `traffic_model_seed1` | mAP@50 | -0.040 | [-0.108, +0.025] | 0.12 | not distinguishable |
| `traffic_model_v2_dedup` | Plate | -0.022 | [-0.044, -0.003] | 0.01 | **worse** |
| `traffic_model_v2_dedup` | WithHelmet | +0.028 | [-0.107, +0.167] | 0.65 | not distinguishable |
| `traffic_model_v2_dedup` | WithoutHelmet | +0.046 | [+0.005, +0.087] | 0.99 | **better** |
| `traffic_model_v2_dedup` | TripleRiding | +0.032 | [-0.039, +0.113] | 0.76 | not distinguishable |
| `traffic_model_v2_dedup` | mAP@50 | +0.021 | [-0.022, +0.063] | 0.83 | not distinguishable |
| `traffic_model_probe` | Plate | -0.041 | [-0.069, -0.013] | 0.00 | **worse** |
| `traffic_model_probe` | WithHelmet | -0.104 | [-0.238, +0.042] | 0.07 | not distinguishable |
| `traffic_model_probe` | WithoutHelmet | +0.014 | [-0.019, +0.048] | 0.80 | not distinguishable |
| `traffic_model_probe` | TripleRiding | -0.074 | [-0.149, -0.010] | 0.01 | **worse** |
| `traffic_model_probe` | mAP@50 | -0.051 | [-0.095, -0.006] | 0.01 | **worse** |
| `traffic_model_r2` | Plate | -0.085 | [-0.125, -0.044] | 0.00 | **worse** |
| `traffic_model_r2` | WithHelmet | -0.022 | [-0.147, +0.104] | 0.40 | not distinguishable |
| `traffic_model_r2` | WithoutHelmet | +0.025 | [-0.016, +0.063] | 0.89 | not distinguishable |
| `traffic_model_r2` | TripleRiding | -0.039 | [-0.104, +0.012] | 0.07 | not distinguishable |
| `traffic_model_r2` | mAP@50 | -0.030 | [-0.067, +0.007] | 0.05 | not distinguishable |

## F1-optimal confidence thresholds (chosen on this split) — `traffic_model-2`

| class | threshold | precision | recall | F1 |
|---|---:|---:|---:|---:|
| Plate | 0.25 | 0.9058 | 0.8178 | 0.8596 |
| WithHelmet | 0.675 | 0.5 | 0.6923 | 0.5806 |
| WithoutHelmet | 0.375 | 0.7812 | 0.7177 | 0.7481 |
| TripleRiding | 0.55 | 0.9057 | 0.8276 | 0.8649 |

## F1-optimal confidence thresholds (chosen on this split) — `traffic_model_seed1`

| class | threshold | precision | recall | F1 |
|---|---:|---:|---:|---:|
| Plate | 0.575 | 0.9347 | 0.753 | 0.8341 |
| WithHelmet | 0.525 | 0.5 | 0.4231 | 0.4583 |
| WithoutHelmet | 0.55 | 0.7418 | 0.756 | 0.7488 |
| TripleRiding | 0.775 | 0.8475 | 0.8621 | 0.8547 |

## F1-optimal confidence thresholds (chosen on this split) — `traffic_model_v2_dedup`

| class | threshold | precision | recall | F1 |
|---|---:|---:|---:|---:|
| Plate | 0.4 | 0.8969 | 0.8097 | 0.8511 |
| WithHelmet | 0.525 | 0.5517 | 0.6154 | 0.5818 |
| WithoutHelmet | 0.45 | 0.8022 | 0.6986 | 0.7468 |
| TripleRiding | 0.45 | 0.9608 | 0.8448 | 0.8991 |

## F1-optimal confidence thresholds (chosen on this split) — `traffic_model_probe`

| class | threshold | precision | recall | F1 |
|---|---:|---:|---:|---:|
| Plate | 0.325 | 0.8565 | 0.7976 | 0.826 |
| WithHelmet | 0.55 | 0.3333 | 0.6154 | 0.4324 |
| WithoutHelmet | 0.6 | 0.8085 | 0.7273 | 0.7657 |
| TripleRiding | 0.175 | 0.8824 | 0.7759 | 0.8257 |

## F1-optimal confidence thresholds (chosen on this split) — `traffic_model_r2`

| class | threshold | precision | recall | F1 |
|---|---:|---:|---:|---:|
| Plate | 0.425 | 0.8578 | 0.7328 | 0.7904 |
| WithHelmet | 0.7 | 0.5185 | 0.5385 | 0.5283 |
| WithoutHelmet | 0.4 | 0.7512 | 0.7512 | 0.7512 |
| TripleRiding | 0.375 | 0.8679 | 0.7931 | 0.8288 |

