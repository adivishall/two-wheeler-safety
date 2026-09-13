# Model evaluation — eval_traffic_model-2_test

- Generated: 2026-09-13T12:44:10.358209+00:00
- Model: `runs/detect/traffic_model-2/weights/best.pt`
- Data: `master_traffic_violation_dataset/data.yaml` (split: test)
- conf=0.25, iou=0.5, imgsz=640, device=mps

## Headline metrics (Ultralytics val)

- **mAP@50**: 0.7202
- **mAP@50-95**: 0.527
- mean precision: 0.7582
- mean recall: 0.7773

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Plate | 0.8806 | 0.9008 | 0.8618 | 0.5022 |
| WithHelmet | 0.5385 | 0.5185 | 0.3873 | 0.3103 |
| WithoutHelmet | 0.7315 | 0.7524 | 0.7054 | 0.5535 |
| TripleRiding | 0.8824 | 0.9375 | 0.9264 | 0.7422 |

Confusion matrix / PR curves: `/Users/adivishal/Projects/Two Wheeler Safety ASEP 2/runs/detect/val-7`

## Error analysis

Images analysed: 194

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Plate | 114 | 14 | 17 | 0.8906 | 0.8702 | 0.8803 |
| WithHelmet | 14 | 13 | 13 | 0.5185 | 0.5185 | 0.5185 |
| WithoutHelmet | 78 | 37 | 27 | 0.6783 | 0.7429 | 0.7091 |
| TripleRiding | 44 | 5 | 4 | 0.898 | 0.9167 | 0.9072 |

### Helmet confusion

- WithHelmet predicted as WithoutHelmet: 3
- WithoutHelmet predicted as WithHelmet: 8

### Confidence vs correctness

- mean confidence when correct: 0.8164 (250 preds)
- mean confidence when wrong: 0.6594 (69 preds)
- separation (correct - wrong): **0.157** (>0 is good; <=0 means confidently wrong)

### Representative false positives

- `master_traffic_violation_dataset/test/images/ds1_f12fcf_BikesHelmets86_png.rf.ec8064687b5bda317ebb66c85676cb04.jpg` predicted **WithoutHelmet** (0.305)
- `master_traffic_violation_dataset/test/images/ds1_f5fa68_CYB01EC198319216_jpg.rf.ab001350384b9bb0b9a0c3666bc34259.jpg` predicted **WithoutHelmet** (0.494)
- `master_traffic_violation_dataset/test/images/dst_764a4b_overload-351-_jpg.rf.9ff80276f5d729894bbb0801f05f648b.jpg` predicted **TripleRiding** (0.781)
- `master_traffic_violation_dataset/test/images/ds1_0d7e7c_BikesHelmets721_png.rf.93913ec914f27a6fc3f7253e794a3cb8.jpg` predicted **WithoutHelmet** (0.777)
- `master_traffic_violation_dataset/test/images/ds1_8bb1e2_KBA01EC191005985_jpg.rf.78e45253a8040209277b569e1fa6f7d5.jpg` predicted **Plate** (0.731)
- `master_traffic_violation_dataset/test/images/ds1_4a28ca_ADB03EC199045927_jpg.rf.491f3badcf4c052f3a75f1392293f118.jpg` predicted **Plate** (0.64)
- `master_traffic_violation_dataset/test/images/ds1_f6d995_ADB03EC198699975_jpg.rf.5fa1cb8b7a3432a775ace22a43f20043.jpg` predicted **WithoutHelmet** (0.91)
- `master_traffic_violation_dataset/test/images/ds1_a573d7_KBA01EC190474543_jpg.rf.01b09353842122cea9526c555e3cd6e7.jpg` predicted **WithoutHelmet** (0.784)
- `master_traffic_violation_dataset/test/images/ds1_2fa4a3_CYB00EC198322616_jpg.rf.8d5cb931f905864efd7cd8137a68be71.jpg` predicted **WithoutHelmet** (0.764)
- `master_traffic_violation_dataset/test/images/ds1_0ebec2_ADB03TE193655459_jpg.rf.6ead2afb1fb4adfcc4d05d50f55a8502.jpg` predicted **WithoutHelmet** (0.348)

### Representative false negatives (missed)

- `master_traffic_violation_dataset/test/images/ds1_4e4168_KBA01EC190872581_jpg.rf.fdf8bc3219354d7e38e68972a43973f5.jpg` missed **WithoutHelmet**
- `master_traffic_violation_dataset/test/images/ds1_8bb1e2_KBA01EC191005985_jpg.rf.78e45253a8040209277b569e1fa6f7d5.jpg` missed **Plate**
- `master_traffic_violation_dataset/test/images/ds1_f6d995_ADB03EC198699975_jpg.rf.5fa1cb8b7a3432a775ace22a43f20043.jpg` missed **WithoutHelmet**
- `master_traffic_violation_dataset/test/images/ds1_901eec_ADB03TE193734082_jpg.rf.146cd3aa6845772b6ab5dac845ccdb93.jpg` missed **Plate**
- `master_traffic_violation_dataset/test/images/ds1_ce1f50_CYB00EC198312056_jpg.rf.3be9650ef15ea1d92c6b0561e3bd67ad.jpg` missed **Plate**
- `master_traffic_violation_dataset/test/images/ds1_0ba112_RMD02EC198316235_jpg.rf.217218cf883dba2d21b23c2dfbbe5a54.jpg` missed **Plate**
- `master_traffic_violation_dataset/test/images/ds1_0ebec2_ADB03TE193655459_jpg.rf.6ead2afb1fb4adfcc4d05d50f55a8502.jpg` missed **Plate**
- `master_traffic_violation_dataset/test/images/ds1_0ebec2_ADB03TE193655459_jpg.rf.6ead2afb1fb4adfcc4d05d50f55a8502.jpg` missed **WithoutHelmet**
- `master_traffic_violation_dataset/test/images/ds1_5afc54_ADB03TE193612965_jpg.rf.df8b72cca1b26b98cc235dea72140796.jpg` missed **Plate**
- `master_traffic_violation_dataset/test/images/ds1_5afc54_ADB03TE193612965_jpg.rf.df8b72cca1b26b98cc235dea72140796.jpg` missed **WithoutHelmet**

## Inference benchmark

- 50 images, mean 28.9 ms (p50 28.93, p90 29.7) — 34.6 FPS

