# Detector metrics with uncertainty — `test` split

- Generated: 2026-09-29T00:12:04.530947+00:00
- Data: `eval/clean_splits/data.yaml` split **test** (175 images, fingerprint `sha256:fb92cb1c0c0e250f`)
- Bootstrap: 2000 image resamples, 95% percentile intervals, seed 0
- AP@50 recomputed from cached predictions (conf ≥ 0.001, NMS IoU 0.7, imgsz 640); 101-point interpolation as in Ultralytics.

## AP@50 per class, with 95% CI

| model | Plate | WithHelmet | WithoutHelmet | TripleRiding | mAP@50 |
|---|---:|---:|---:|---:|---:|
| `traffic_model-2` | 0.889 [0.837, 0.938] | 0.415 [0.204, 0.729] | 0.734 [0.655, 0.820] | 0.963 [0.897, 0.995] | 0.750 [0.690, 0.840] |
| `traffic_model_seed1` | 0.885 [0.828, 0.939] | 0.212 [0.087, 0.455] | 0.716 [0.628, 0.817] | 0.885 [0.760, 0.977] | 0.675 [0.623, 0.759] |
| `traffic_model_v2_dedup` | 0.881 [0.825, 0.938] | 0.377 [0.201, 0.668] | 0.793 [0.719, 0.867] | 0.952 [0.880, 0.991] | 0.751 [0.699, 0.835] |

Instances: Plate 131, WithHelmet 27, WithoutHelmet 105, TripleRiding 30

## Paired differences vs `traffic_model-2`

AP(model) − AP(baseline) on the same resampled images. **Significant** only when the 95% CI excludes 0.

| model | class | Δ AP@50 | 95% CI | P(model better) | verdict |
|---|---|---:|---|---:|---|
| `traffic_model_seed1` | Plate | -0.004 | [-0.033, +0.026] | 0.39 | not distinguishable |
| `traffic_model_seed1` | WithHelmet | -0.203 | [-0.483, +0.041] | 0.07 | not distinguishable |
| `traffic_model_seed1` | WithoutHelmet | -0.018 | [-0.091, +0.057] | 0.37 | not distinguishable |
| `traffic_model_seed1` | TripleRiding | -0.077 | [-0.158, -0.015] | 0.00 | **worse** |
| `traffic_model_seed1` | mAP@50 | -0.076 | [-0.155, -0.003] | 0.02 | **worse** |
| `traffic_model_v2_dedup` | Plate | -0.007 | [-0.041, +0.026] | 0.34 | not distinguishable |
| `traffic_model_v2_dedup` | WithHelmet | -0.039 | [-0.210, +0.087] | 0.37 | not distinguishable |
| `traffic_model_v2_dedup` | WithoutHelmet | +0.059 | [+0.018, +0.098] | 1.00 | **better** |
| `traffic_model_v2_dedup` | TripleRiding | -0.011 | [-0.054, +0.020] | 0.22 | not distinguishable |
| `traffic_model_v2_dedup` | mAP@50 | +0.001 | [-0.041, +0.036] | 0.58 | not distinguishable |

## `traffic_model-2` at its own val-selected thresholds from `eval/results/uncertainty_val.json`

| class | threshold | precision | recall | F1 |
|---|---:|---:|---:|---:|
| Plate | 0.25 | 0.8976 | 0.8702 | 0.8837 |
| WithHelmet | 0.675 | 0.4762 | 0.3704 | 0.4167 |
| WithoutHelmet | 0.375 | 0.7547 | 0.7619 | 0.7583 |
| TripleRiding | 0.55 | 0.875 | 0.9333 | 0.9032 |

## `traffic_model_seed1` at its own val-selected thresholds from `eval/results/uncertainty_val.json`

| class | threshold | precision | recall | F1 |
|---|---:|---:|---:|---:|
| Plate | 0.575 | 0.9224 | 0.8168 | 0.8664 |
| WithHelmet | 0.525 | 0.4545 | 0.1852 | 0.2632 |
| WithoutHelmet | 0.55 | 0.7453 | 0.7524 | 0.7488 |
| TripleRiding | 0.775 | 0.8 | 0.9333 | 0.8615 |

## `traffic_model_v2_dedup` at its own val-selected thresholds from `eval/results/uncertainty_val.json`

| class | threshold | precision | recall | F1 |
|---|---:|---:|---:|---:|
| Plate | 0.4 | 0.8976 | 0.8702 | 0.8837 |
| WithHelmet | 0.525 | 0.5 | 0.2963 | 0.3721 |
| WithoutHelmet | 0.45 | 0.7818 | 0.819 | 0.8 |
| TripleRiding | 0.45 | 0.9 | 0.9 | 0.9 |

