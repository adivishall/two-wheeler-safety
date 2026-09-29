# Model A/B comparison

- Generated: 2026-09-29T00:24:12.530766+00:00
- Data: `eval/clean_splits/data.yaml` (split: **val**)
- conf=0.001, iou=0.7, imgsz=640, device=mps
- Ranked by: **map50**

**Highest point estimate: `traffic_model-2`** on the selection split (every model saw the identical split and settings).

> A point-estimate ranking is not a significance test. Whether a gap is real is answered by `evaluate_uncertainty.py` (paired image bootstrap).

| Model | version | mAP@50 | mAP@50-95 | precision | recall | latency p50 (ms) | file (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| `traffic_model-2` | traffic-4class@1.0.0 | 0.7661 | 0.542 | 0.7773 | 0.7486 | 19.52 | 5.96 |
| `traffic_model_v2_dedup` | traffic_model_v2_dedup@2.0.0 | 0.7444 | 0.5268 | 0.7624 | 0.772 | 13.57 | 5.96 |
| `traffic_model_probe` | — | 0.7411 | 0.5162 | 0.696 | 0.7235 | 18.29 | 5.96 |
| `traffic_model_r2` | — | 0.7379 | 0.5192 | 0.7064 | 0.7359 | 19.48 | 23.35 |
| `traffic_model_seed1` | — | 0.7175 | 0.5071 | 0.6811 | 0.6961 | 19.56 | 5.96 |

Latency: median of single-image `predict()` calls (load + pre/post-processing) after a warm-up, on the device above — a sanity check, not a benchmark (`benchmark.md` is). Between checkpoints of the same architecture a latency gap is measurement noise. File size includes any optimizer state left in the checkpoint.

## Per-class mAP@50

| Model | Plate | TripleRiding | WithHelmet | WithoutHelmet |
|---|---:|---:|---:|---:|
| `traffic_model-2` | 0.8727 | 0.9075 | 0.5171 | 0.767 |
| `traffic_model_v2_dedup` | 0.8566 | 0.8729 | 0.4495 | 0.7985 |
| `traffic_model_probe` | 0.8413 | 0.9037 | 0.4294 | 0.7898 |
| `traffic_model_r2` | 0.7905 | 0.8822 | 0.4994 | 0.7797 |
| `traffic_model_seed1` | 0.8455 | 0.8391 | 0.4175 | 0.7679 |

