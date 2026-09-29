# Detector robustness under synthetic corruptions — `traffic_model-2`

> **Synthetic transforms of real held-out images.** Labels are untouched; only pixels change. This measures sensitivity to each transform, not performance on real night / rain / glare footage.

- Split **val** of `eval/clean_splits/data.yaml` (352 images); predict-mode AP@50 (the path the pipeline runs); 1000 paired bootstrap resamples.
- Clean mAP@50: **0.721** [0.668, 0.785]

## Change in AP@50 vs the same images uncorrupted

Cells: Δ AP@50 (★ = 95% CI excludes 0).

| corruption | Plate | WithHelmet | WithoutHelmet | TripleRiding | mAP@50 |
|---|---:|---:|---:|---:|---:|
| `gaussian_blur_s2` | -0.034 ★ | +0.031 | -0.015 | -0.026 | -0.011 |
| `gaussian_blur_s4` | -0.087 ★ | -0.025 | -0.056 ★ | -0.073 ★ | -0.060 ★ |
| `motion_blur_9` | -0.137 ★ | -0.089 | -0.073 ★ | -0.114 ★ | -0.103 ★ |
| `motion_blur_21` | -0.655 ★ | -0.415 ★ | -0.310 ★ | -0.531 ★ | -0.478 ★ |
| `low_light_mild` | -0.073 ★ | -0.045 | -0.054 ★ | -0.049 | -0.055 ★ |
| `low_light_severe` | -0.320 ★ | -0.245 ★ | -0.364 ★ | -0.353 ★ | -0.321 ★ |
| `glare_mild` | -0.008 | -0.158 ★ | -0.011 | -0.027 ★ | -0.051 ★ |
| `glare_severe` | -0.065 ★ | -0.234 ★ | -0.056 ★ | -0.125 ★ | -0.120 ★ |
| `jpeg_q20` | -0.059 ★ | -0.016 | -0.017 | +0.007 | -0.021 |
| `jpeg_q8` | -0.151 ★ | -0.055 | -0.104 ★ | -0.063 ★ | -0.093 ★ |
| `low_res_x0.5` | -0.018 ★ | +0.002 | -0.001 | -0.002 | -0.005 |
| `low_res_x0.25` | -0.059 ★ | -0.005 | -0.004 | -0.023 | -0.023 |
| `occlusion_10pct` | -0.129 ★ | -0.087 | -0.114 ★ | -0.098 | -0.107 ★ |
| `occlusion_25pct` | -0.359 ★ | -0.163 ★ | -0.376 ★ | -0.653 ★ | -0.388 ★ |

## Worst-hit class per corruption

- `gaussian_blur_s2`: Plate (-0.034)
- `gaussian_blur_s4`: Plate (-0.087)
- `motion_blur_9`: Plate (-0.137)
- `motion_blur_21`: Plate (-0.655)
- `low_light_mild`: Plate (-0.073)
- `low_light_severe`: WithoutHelmet (-0.364)
- `glare_mild`: WithHelmet (-0.158)
- `glare_severe`: WithHelmet (-0.234)
- `jpeg_q20`: Plate (-0.059)
- `jpeg_q8`: Plate (-0.151)
- `low_res_x0.5`: Plate (-0.018)
- `low_res_x0.25`: Plate (-0.059)
- `occlusion_10pct`: Plate (-0.129)
- `occlusion_25pct`: TripleRiding (-0.653)

