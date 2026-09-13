# Model evaluation — eval_traffic_model-2_test_clean

- Generated: 2026-09-13T12:52:11.746889+00:00
- Model: `runs/detect/traffic_model-2/weights/best.pt` (version **traffic-4class@1.0.0**)
- Data: `eval/clean_splits/data.yaml` (split: test)
- conf=0.25, iou=0.5, imgsz=640, device=mps

## Headline metrics (Ultralytics val)

- **mAP@50**: 0.7265
- **mAP@50-95**: 0.532
- mean precision: 0.7559
- mean recall: 0.7846

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Plate | 0.8939 | 0.9008 | 0.863 | 0.5028 |
| WithHelmet | 0.5385 | 0.5185 | 0.3873 | 0.3103 |
| WithoutHelmet | 0.7383 | 0.7524 | 0.7054 | 0.5535 |
| TripleRiding | 0.8529 | 0.9667 | 0.9502 | 0.7613 |

Confusion matrix / PR curves: `/Users/adivishal/Projects/Two Wheeler Safety ASEP 2/runs/detect/val-9`

## Error analysis

Images analysed: 175

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Plate | 114 | 13 | 17 | 0.8976 | 0.8702 | 0.8837 |
| WithHelmet | 14 | 13 | 13 | 0.5185 | 0.5185 | 0.5185 |
| WithoutHelmet | 78 | 35 | 27 | 0.6903 | 0.7429 | 0.7156 |
| TripleRiding | 29 | 4 | 1 | 0.8788 | 0.9667 | 0.9206 |

### Helmet confusion

- WithHelmet predicted as WithoutHelmet: 3
- WithoutHelmet predicted as WithHelmet: 8

### Confidence vs correctness (binned)

YOLO confidence is a model score, not a calibrated probability. This measures whether accuracy increases with the score (ranking usefulness), not whether 0.8 means 80% correct.

```
   score bin      n     acc  histogram
 0.25-0.40      16   0.312  #####
 0.40-0.55      20   0.600  ######
 0.55-0.70      19   0.474  ######
 0.70-0.85     119   0.849  #####################################
 0.85-1.00     126   0.857  ########################################
```

- Spearman(score rank, accuracy rank) = **0.9**
- top-bin minus bottom-bin accuracy = **0.5446**
- monotonicity violations: 1
- verdict: *higher score => more likely correct (useful ranking signal)*

| Class | n | Spearman | top-bottom gap | verdict |
|---|---:|---:|---:|---|
| Plate | 127 | 0.6 | 0.25 | higher score => more likely correct (useful ranking signal) |
| TripleRiding | 33 | -0.2 | -0.0417 | score does NOT separate correct from wrong (do not threshold on it) |
| WithHelmet | 27 | -0.1539 | 0.2051 | weak/non-monotonic relationship; threshold with care |
| WithoutHelmet | 113 | 1.0 | 0.8615 | higher score => more likely correct (useful ranking signal) |

### Saved failure artifacts

Annotated crops under `eval/` (index: `eval/index.json`), red box = prediction, green box = ground truth.

| Category | Saved |
|---|---:|
| `false_positives` | 30 (capped) |
| `false_negatives` | 30 (capped) |
| `class_confusions` | 12 |
| `low_confidence` | 12 |

### Confidence vs correctness

- mean confidence when correct: 0.8104 (235 preds)
- mean confidence when wrong: 0.6685 (65 preds)
- separation (correct - wrong): **0.1419** (>0 is good; <=0 means confidently wrong)

### Representative false positives

- `eval/clean_splits/test/images/ds1_f12fcf_BikesHelmets86_png.rf.ec8064687b5bda317ebb66c85676cb04.jpg` predicted **WithoutHelmet** (0.305)
- `eval/clean_splits/test/images/ds1_f5fa68_CYB01EC198319216_jpg.rf.ab001350384b9bb0b9a0c3666bc34259.jpg` predicted **WithoutHelmet** (0.494)
- `eval/clean_splits/test/images/dst_764a4b_overload-351-_jpg.rf.9ff80276f5d729894bbb0801f05f648b.jpg` predicted **TripleRiding** (0.781)
- `eval/clean_splits/test/images/ds1_0d7e7c_BikesHelmets721_png.rf.93913ec914f27a6fc3f7253e794a3cb8.jpg` predicted **WithoutHelmet** (0.777)
- `eval/clean_splits/test/images/ds1_8bb1e2_KBA01EC191005985_jpg.rf.78e45253a8040209277b569e1fa6f7d5.jpg` predicted **Plate** (0.731)
- `eval/clean_splits/test/images/ds1_4a28ca_ADB03EC199045927_jpg.rf.491f3badcf4c052f3a75f1392293f118.jpg` predicted **Plate** (0.64)
- `eval/clean_splits/test/images/ds1_f6d995_ADB03EC198699975_jpg.rf.5fa1cb8b7a3432a775ace22a43f20043.jpg` predicted **WithoutHelmet** (0.91)
- `eval/clean_splits/test/images/ds1_a573d7_KBA01EC190474543_jpg.rf.01b09353842122cea9526c555e3cd6e7.jpg` predicted **WithoutHelmet** (0.784)
- `eval/clean_splits/test/images/ds1_2fa4a3_CYB00EC198322616_jpg.rf.8d5cb931f905864efd7cd8137a68be71.jpg` predicted **WithoutHelmet** (0.764)
- `eval/clean_splits/test/images/ds1_0ebec2_ADB03TE193655459_jpg.rf.6ead2afb1fb4adfcc4d05d50f55a8502.jpg` predicted **WithoutHelmet** (0.348)

### Representative false negatives (missed)

- `eval/clean_splits/test/images/ds1_4e4168_KBA01EC190872581_jpg.rf.fdf8bc3219354d7e38e68972a43973f5.jpg` missed **WithoutHelmet**
- `eval/clean_splits/test/images/ds1_8bb1e2_KBA01EC191005985_jpg.rf.78e45253a8040209277b569e1fa6f7d5.jpg` missed **Plate**
- `eval/clean_splits/test/images/ds1_f6d995_ADB03EC198699975_jpg.rf.5fa1cb8b7a3432a775ace22a43f20043.jpg` missed **WithoutHelmet**
- `eval/clean_splits/test/images/ds1_901eec_ADB03TE193734082_jpg.rf.146cd3aa6845772b6ab5dac845ccdb93.jpg` missed **Plate**
- `eval/clean_splits/test/images/ds1_ce1f50_CYB00EC198312056_jpg.rf.3be9650ef15ea1d92c6b0561e3bd67ad.jpg` missed **Plate**
- `eval/clean_splits/test/images/ds1_0ba112_RMD02EC198316235_jpg.rf.217218cf883dba2d21b23c2dfbbe5a54.jpg` missed **Plate**
- `eval/clean_splits/test/images/ds1_0ebec2_ADB03TE193655459_jpg.rf.6ead2afb1fb4adfcc4d05d50f55a8502.jpg` missed **Plate**
- `eval/clean_splits/test/images/ds1_0ebec2_ADB03TE193655459_jpg.rf.6ead2afb1fb4adfcc4d05d50f55a8502.jpg` missed **WithoutHelmet**
- `eval/clean_splits/test/images/ds1_5afc54_ADB03TE193612965_jpg.rf.df8b72cca1b26b98cc235dea72140796.jpg` missed **Plate**
- `eval/clean_splits/test/images/ds1_5afc54_ADB03TE193612965_jpg.rf.df8b72cca1b26b98cc235dea72140796.jpg` missed **WithoutHelmet**

## Inference benchmark

- 50 images, mean 28.57 ms (p50 27.57, p90 28.61) — 35.0 FPS

