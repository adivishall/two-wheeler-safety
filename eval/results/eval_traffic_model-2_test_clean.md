# Model evaluation — eval_traffic_model-2_test_clean

- Generated: 2026-09-29T00:11:01.873011+00:00
- Model: `runs/detect/traffic_model-2/weights/best.pt` (version **traffic-4class@1.0.0**)
- Data: `eval/clean_splits/data.yaml` (split: test)
- conf=0.25, iou=0.5, imgsz=640, device=mps

## Headline metrics (Ultralytics val)

- **mAP@50**: 0.7688
- **mAP@50-95**: 0.5583
- mean precision: 0.7936
- mean recall: 0.7441

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Plate | 0.9093 | 0.8321 | 0.8867 | 0.5156 |
| WithHelmet | 0.5701 | 0.4444 | 0.4264 | 0.3359 |
| WithoutHelmet | 0.7966 | 0.7333 | 0.7913 | 0.612 |
| TripleRiding | 0.8984 | 0.9667 | 0.9707 | 0.7695 |

Confusion matrix / PR curves: `/Users/adivishal/Projects/Two Wheeler Safety ASEP 2/runs/detect/val-25`

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

### Confidence vs correctness

- mean confidence when correct: 0.8104 (235 preds)
- mean confidence when wrong: 0.6685 (65 preds)
- separation (correct - wrong): **0.1419** (>0 is good; <=0 means confidently wrong)

### Representative false positives

- `eval/clean_splits/test/images/ds1_012628_CYB01EC198313881_jpg.rf.a05516b96c576a5ba9fb70ad59bdce42.jpg` predicted **Plate** (0.85)
- `eval/clean_splits/test/images/ds1_0268f8_CYB03EC198353931_jpg.rf.da94f82fba01c660febd1a592d296701.jpg` predicted **WithoutHelmet** (0.413)
- `eval/clean_splits/test/images/ds1_06480b_CYB00EC198320847_jpg.rf.7d9dd6a4a12231041118ae913279b2d0.jpg` predicted **WithoutHelmet** (0.839)
- `eval/clean_splits/test/images/ds1_0c157e_BikesHelmets560_png.rf.faa07ae27c23724588d5a64014092aaf.jpg` predicted **WithoutHelmet** (0.864)
- `eval/clean_splits/test/images/ds1_0c157e_BikesHelmets560_png.rf.faa07ae27c23724588d5a64014092aaf.jpg` predicted **WithoutHelmet** (0.706)
- `eval/clean_splits/test/images/ds1_0c157e_BikesHelmets560_png.rf.faa07ae27c23724588d5a64014092aaf.jpg` predicted **Plate** (0.642)
- `eval/clean_splits/test/images/ds1_0c157e_BikesHelmets560_png.rf.faa07ae27c23724588d5a64014092aaf.jpg` predicted **Plate** (0.281)
- `eval/clean_splits/test/images/ds1_0d7e7c_BikesHelmets721_png.rf.93913ec914f27a6fc3f7253e794a3cb8.jpg` predicted **WithoutHelmet** (0.777)
- `eval/clean_splits/test/images/ds1_0ebec2_ADB03TE193655459_jpg.rf.6ead2afb1fb4adfcc4d05d50f55a8502.jpg` predicted **WithoutHelmet** (0.348)
- `eval/clean_splits/test/images/ds1_155b35_KBA01EC190474679_jpg.rf.0f408b6905169f5223d614230af611c9.jpg` predicted **WithoutHelmet** (0.878)

### Representative false negatives (missed)

- `eval/clean_splits/test/images/ds1_06480b_CYB00EC198320847_jpg.rf.7d9dd6a4a12231041118ae913279b2d0.jpg` missed **WithHelmet**
- `eval/clean_splits/test/images/ds1_0ba112_RMD02EC198316235_jpg.rf.217218cf883dba2d21b23c2dfbbe5a54.jpg` missed **Plate**
- `eval/clean_splits/test/images/ds1_0d998d_BikesHelmets362_png.rf.29f5cb82f4fff580de2c44c489a9f558.jpg` missed **WithHelmet**
- `eval/clean_splits/test/images/ds1_0ebec2_ADB03TE193655459_jpg.rf.6ead2afb1fb4adfcc4d05d50f55a8502.jpg` missed **Plate**
- `eval/clean_splits/test/images/ds1_0ebec2_ADB03TE193655459_jpg.rf.6ead2afb1fb4adfcc4d05d50f55a8502.jpg` missed **WithoutHelmet**
- `eval/clean_splits/test/images/ds1_18e1a6_BikesHelmets354_png.rf.cfab8538ae357eec28493a5cb37e2298.jpg` missed **WithoutHelmet**
- `eval/clean_splits/test/images/ds1_18e1a6_BikesHelmets354_png.rf.cfab8538ae357eec28493a5cb37e2298.jpg` missed **WithoutHelmet**
- `eval/clean_splits/test/images/ds1_1ee7e8_BikesHelmets351_png.rf.462f154c654dec84a092a6a426d9bf75.jpg` missed **Plate**
- `eval/clean_splits/test/images/ds1_1ee7e8_BikesHelmets351_png.rf.462f154c654dec84a092a6a426d9bf75.jpg` missed **Plate**
- `eval/clean_splits/test/images/ds1_1ee7e8_BikesHelmets351_png.rf.462f154c654dec84a092a6a426d9bf75.jpg` missed **WithHelmet**

