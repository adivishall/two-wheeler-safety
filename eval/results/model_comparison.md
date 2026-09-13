# Model A/B comparison

- Generated: 2026-09-13T12:53:59.991229+00:00
- Data: `eval/clean_splits/data.yaml` (split: **test**)
- conf=0.25, iou=0.5, imgsz=640, device=mps
- Ranked by: **map50**

**Winner: `traffic_model_probe`** (every model saw the identical split and settings).

| Model | version | mAP@50 | mAP@50-95 | precision | recall | latency (ms) | size (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| `traffic_model_probe` | — | 0.7408 | 0.5455 | 0.7316 | 0.8021 | 27.88 | 5.96 |
| `traffic_model_r2` | — | 0.7279 | 0.5188 | 0.8014 | 0.7684 | 27.4 | 23.35 |
| `traffic_model-2` | traffic-4class@1.0.0 | 0.7265 | 0.532 | 0.7559 | 0.7846 | 27.91 | 5.96 |
| `traffic_model_helmetfix` | — | 0.4996 | 0.3385 | 0.7996 | 0.5195 | 27.97 | 5.96 |

## Per-class mAP@50

| Model | Plate | TripleRiding | WithHelmet | WithoutHelmet |
|---|---:|---:|---:|---:|
| `traffic_model_probe` | 0.8403 | 0.8817 | 0.4699 | 0.7713 |
| `traffic_model_r2` | 0.8355 | 0.902 | 0.4223 | 0.7519 |
| `traffic_model-2` | 0.863 | 0.9502 | 0.3873 | 0.7054 |
| `traffic_model_helmetfix` | 0.8708 | 0.0 | 0.4166 | 0.711 |

