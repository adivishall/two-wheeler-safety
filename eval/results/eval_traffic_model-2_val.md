# Model evaluation — eval_traffic_model-2_val

- Generated: 2026-09-11T07:54:31.619591+00:00
- Model: `runs/detect/traffic_model-2/weights/best.pt`
- Data: `master_traffic_violation_dataset/data.yaml` (split: val)
- conf=0.25, iou=0.5, imgsz=640, device=mps

## Headline metrics (Ultralytics val)

- **mAP@50**: 0.6967
- **mAP@50-95**: 0.5021
- mean precision: 0.7828
- mean recall: 0.7398

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Plate | 0.9414 | 0.7801 | 0.8106 | 0.4284 |
| WithHelmet | 0.4196 | 0.6296 | 0.4427 | 0.3606 |
| WithoutHelmet | 0.8483 | 0.6957 | 0.6878 | 0.5649 |
| TripleRiding | 0.9218 | 0.8539 | 0.8458 | 0.6545 |

Confusion matrix / PR curves: `runs/detect/val-6`

## Error analysis

Images analysed: 383

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Plate | 202 | 23 | 45 | 0.8978 | 0.8178 | 0.8559 |
| WithHelmet | 18 | 34 | 9 | 0.3462 | 0.6667 | 0.4557 |
| WithoutHelmet | 150 | 85 | 59 | 0.6383 | 0.7177 | 0.6757 |
| TripleRiding | 75 | 9 | 14 | 0.8929 | 0.8427 | 0.8671 |

### Helmet confusion

- WithHelmet predicted as WithoutHelmet: 4
- WithoutHelmet predicted as WithHelmet: 16

### Confidence vs correctness

- mean confidence when correct: 0.8094 (445 preds)
- mean confidence when wrong: 0.5612 (151 preds)
- separation (correct - wrong): **0.2482** (>0 is good; <=0 means confidently wrong)

### Representative false positives

- `master_traffic_violation_dataset/valid/images/ds1_427ad3_CYB00EC198314520_jpg.rf.8d6684e67b97f08f0579e4ac3b820df2.jpg` predicted **WithoutHelmet** (0.693)
- `master_traffic_violation_dataset/valid/images/ds1_4fa36f_KBA01EC190721559_jpg.rf.f6cd20a200528535b7b875cc7fd079d8.jpg` predicted **WithoutHelmet** (0.432)
- `master_traffic_violation_dataset/valid/images/ds1_f5ea0e_BikesHelmets757_png.rf.6ff3a2ccd410cbf10816e7c293f83478.jpg` predicted **WithHelmet** (0.917)
- `master_traffic_violation_dataset/valid/images/ds1_f5ea0e_BikesHelmets757_png.rf.6ff3a2ccd410cbf10816e7c293f83478.jpg` predicted **WithHelmet** (0.648)
- `master_traffic_violation_dataset/valid/images/ds1_f5ea0e_BikesHelmets757_png.rf.6ff3a2ccd410cbf10816e7c293f83478.jpg` predicted **WithHelmet** (0.558)
- `master_traffic_violation_dataset/valid/images/ds1_f5ea0e_BikesHelmets757_png.rf.6ff3a2ccd410cbf10816e7c293f83478.jpg` predicted **WithoutHelmet** (0.557)
- `master_traffic_violation_dataset/valid/images/ds1_db0e8a_BikesHelmets468_png.rf.ad14d1a0dc6dcee330508d3c7c0f1c64.jpg` predicted **WithoutHelmet** (0.265)
- `master_traffic_violation_dataset/valid/images/ds1_368e76_CYB00EC198311408_jpg.rf.1daf164fd13b7874491d9c2dfbddb7a4.jpg` predicted **WithHelmet** (0.897)
- `master_traffic_violation_dataset/valid/images/dst_75a704_127_jpg.rf.14c56400771aaac3fd9c880b99576cfb.jpg` predicted **WithoutHelmet** (0.885)
- `master_traffic_violation_dataset/valid/images/dst_75a704_127_jpg.rf.14c56400771aaac3fd9c880b99576cfb.jpg` predicted **Plate** (0.676)

### Representative false negatives (missed)

- `master_traffic_violation_dataset/valid/images/dst_0cd780_101_jpg.rf.12f5147824e95c34d2b9fc54b29f98a7.jpg` missed **TripleRiding**
- `master_traffic_violation_dataset/valid/images/ds1_c6b612_ADB03TE193708176_jpg.rf.c899daa23bca0462ea17b2a5ef9eadd0.jpg` missed **Plate**
- `master_traffic_violation_dataset/valid/images/ds1_cd1325_CYB00EC198360788_jpg.rf.9e85ad2a23789e881998398dc6302c96.jpg` missed **Plate**
- `master_traffic_violation_dataset/valid/images/ds1_cd1325_CYB00EC198360788_jpg.rf.9e85ad2a23789e881998398dc6302c96.jpg` missed **WithoutHelmet**
- `master_traffic_violation_dataset/valid/images/ds1_cd1325_CYB00EC198360788_jpg.rf.9e85ad2a23789e881998398dc6302c96.jpg` missed **WithHelmet**
- `master_traffic_violation_dataset/valid/images/ds1_831588_ADB03TE193752779_jpg.rf.ffe597180bc10eab6c67e2e44a13d8bd.jpg` missed **Plate**
- `master_traffic_violation_dataset/valid/images/ds1_82ceb1_ADB04EC198490718_jpg.rf.ebcc1c4036447bd26866264f2acfd197.jpg` missed **Plate**
- `master_traffic_violation_dataset/valid/images/ds1_82ceb1_ADB04EC198490718_jpg.rf.ebcc1c4036447bd26866264f2acfd197.jpg` missed **WithoutHelmet**
- `master_traffic_violation_dataset/valid/images/ds1_ae1e63_ADB03EC198521343_jpg.rf.fe37ede8340a8cfbdcb6bf7fc58afef2.jpg` missed **Plate**
- `master_traffic_violation_dataset/valid/images/ds1_afcc20_ADB04EC198530154_jpg.rf.6fb159d01b493e36b925ff90bdb2f63f.jpg` missed **Plate**

## Inference benchmark

- 50 images, mean 27.61 ms (p50 27.52, p90 28.33) — 36.2 FPS

