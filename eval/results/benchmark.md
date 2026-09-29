# Performance benchmark

- Generated: 2026-09-29T00:26:25.709908+00:00
- Hardware: Apple M4 · macOS-26.5.2-arm64-arm-64bit-Mach-O
- Software: python 3.13.7 · torch 2.12.1 · ultralytics 8.4.71 · opencv 4.13.0
- Model: `traffic-4class@1.0.0` (`sha256:d49f79833cb86eda…`)
- Batch size 1 throughout (the pipeline processes frames one at a time).
- Model load (cold, YOLO + EasyOCR): **3.19 s**

- Input image: `samples/test.jpg` 1906×1078, `sha256:f82add8e4a199b52…`
- Input video: `demo_traffic.mp4` 334×596, 120 frames @ 25.0 fps, `sha256:b095924e1882ad2d…`

## Single image — YOLO inference

| device | first call (cold) | warm mean | p50 | p90 | FPS |
|---|---:|---:|---:|---:|---:|
| mps | 401.56 ms | 23.0 ms | 23.17 | 24.42 | 43.5 |
| cpu | 53.99 ms | 20.75 ms | 20.53 | 22.43 | 48.2 |

EasyOCR on the plate crop: 13.53 ms mean (p90 19.32).

## Video — OCR lock on (shipped) — device mps

120 frames in 3.27 s → **36.65 FPS** (27.29 ms/frame), 30 OCR calls. Median of 5 interleaved runs (range 34.95–37.16 FPS); stage profile from that run.

| stage | % of wall | ms/call | calls |
|---|---:|---:|---:|
| yolo | 64.7% | 17.566 | 120 |
| ocr | 23.9% | 25.979 | 30 |
| read | 0.7% | 0.178 | 121 |
| encode | 0.5% | 0.149 | 120 |
| track | 0.2% | 0.061 | 120 |
| evidence | 0.1% | 2.205 | 1 |
| db | 0.0% | 0.002 | 1 |
| unaccounted (draw, glue) | 9.9% | | |

## Video — OCR lock off — device mps

120 frames in 6.17 s → **19.44 FPS** (51.43 ms/frame), 120 OCR calls. Median of 5 interleaved runs (range 15.33–20.49 FPS); stage profile from that run.

| stage | % of wall | ms/call | calls |
|---|---:|---:|---:|
| ocr | 55.7% | 28.584 | 120 |
| yolo | 36.5% | 18.744 | 120 |
| encode | 0.5% | 0.235 | 120 |
| read | 0.4% | 0.226 | 121 |
| track | 0.3% | 0.141 | 120 |
| evidence | 0.1% | 3.888 | 1 |
| db | 0.0% | 0.004 | 1 |
| unaccounted (draw, glue) | 6.5% | | |

OCR lock A/B (same clip, same process, medians): 19.44 → **36.65 FPS (+88.5%)**, 120 → 30 OCR calls; per-round speed-up +79.8% to +138.9% over 5 rounds.
Same decisions in every run of both arms: **yes** (fined: MH02DL4596 no_helmet; withheld: 0).

## Micro-benchmarks

| operation | mean | p90 |
|---|---:|---:|
| db_record_fine | 0.42 ms | 0.45 ms |
| decision_core_step_3_bikes | 0.04 ms | 0.06 ms |
| analyze_direct | 51.53 ms | 72.3 ms |
| analyze_via_api | 54.67 ms | 71.04 ms |

HTTP + Flask + validation + DB overhead of `/analyze` over a direct call, median of 30 paired calls: **1.39 ms** (p10–p90 -7.12 to 8.2 ms) — the range spans zero: below the run-to-run noise of the call.

Peak RSS: 941.6 MB.
