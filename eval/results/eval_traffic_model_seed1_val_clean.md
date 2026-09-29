# Model evaluation — eval_traffic_model_seed1_val_clean

- Generated: 2026-09-29T00:11:21.855068+00:00
- Model: `runs/detect/traffic_model_seed1/weights/best.pt` (no manifest found — version not recorded)
- Data: `eval/clean_splits/data.yaml` (split: val)
- conf=0.25, iou=0.5, imgsz=640, device=mps

## Headline metrics (Ultralytics val)

- **mAP@50**: 0.7175
- **mAP@50-95**: 0.5071
- mean precision: 0.6811
- mean recall: 0.6961

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Plate | 0.9218 | 0.749 | 0.8455 | 0.4441 |
| WithHelmet | 0.4151 | 0.3846 | 0.4175 | 0.3386 |
| WithoutHelmet | 0.7592 | 0.7368 | 0.7679 | 0.6057 |
| TripleRiding | 0.6285 | 0.9138 | 0.8391 | 0.6401 |

Confusion matrix / PR curves: `/Users/adivishal/Projects/Two Wheeler Safety ASEP 2/runs/detect/val-26`

