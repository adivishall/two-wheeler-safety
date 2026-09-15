# Model A/B comparison

- Generated: 2026-09-15T16:04:24.227704+00:00
- Data: `eval/clean_splits/data.yaml` (split: **test**)
- conf=0.25, iou=0.5, imgsz=640, device=mps
- Ranked by: **WithHelmet**

**Winner: `traffic_model-2`** (every model saw the identical split and settings).

| Model | version | mAP@50 | mAP@50-95 | precision | recall | latency (ms) | size (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| `traffic_model-2` | traffic-4class@1.0.0 | 0.7265 | 0.532 | 0.7559 | 0.7846 | 27.03 | 5.96 |
| `traffic_model_v2_dedup` | traffic_model_v2_dedup@2.0.0 | 0.7035 | 0.5155 | 0.788 | 0.7317 | 29.22 | 5.96 |

## Per-class mAP@50

| Model | Plate | TripleRiding | WithHelmet | WithoutHelmet |
|---|---:|---:|---:|---:|
| `traffic_model-2` | 0.863 | 0.9502 | 0.3873 | 0.7054 |
| `traffic_model_v2_dedup` | 0.8554 | 0.9253 | 0.2988 | 0.7344 |

