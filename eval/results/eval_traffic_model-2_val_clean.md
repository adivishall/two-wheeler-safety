# Model evaluation — eval_traffic_model-2_val_clean

- Generated: 2026-09-29T04:49:24.763804+00:00
- Model: `runs/detect/traffic_model-2/weights/best.pt` (version **traffic-4class@1.0.0**)
- Data: `eval/clean_splits/data.yaml` (split: val)
- conf=0.25, iou=0.5, imgsz=640, device=mps

## Headline metrics (Ultralytics val)

- **mAP@50**: 0.7661
- **mAP@50-95**: 0.542
- mean precision: 0.7773
- mean recall: 0.7486

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Plate | 0.9414 | 0.7801 | 0.8727 | 0.4601 |
| WithHelmet | 0.4193 | 0.6538 | 0.5171 | 0.421 |
| WithoutHelmet | 0.8638 | 0.6986 | 0.767 | 0.6182 |
| TripleRiding | 0.8847 | 0.8621 | 0.9075 | 0.6687 |

Confusion matrix / PR curves: `runs/detect/val-37`

## Error analysis

Images analysed: 352

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Plate | 202 | 21 | 45 | 0.9058 | 0.8178 | 0.8596 |
| WithHelmet | 18 | 34 | 8 | 0.3462 | 0.6923 | 0.4615 |
| WithoutHelmet | 150 | 58 | 59 | 0.7212 | 0.7177 | 0.7194 |
| TripleRiding | 49 | 9 | 9 | 0.8448 | 0.8448 | 0.8448 |

### Helmet confusion

- WithHelmet predicted as WithoutHelmet: 4
- WithoutHelmet predicted as WithHelmet: 16

### Confidence vs correctness (binned)

YOLO confidence is a model score, not a calibrated probability. This measures whether accuracy increases with the score (ranking usefulness), not whether 0.8 means 80% correct.

```
   score bin      n     acc  histogram
 0.25-0.40      42   0.333  ########
 0.40-0.55      39   0.308  #######
 0.55-0.70      48   0.542  #########
 0.70-0.85     207   0.889  ########################################
 0.85-1.00     205   0.893  #######################################
```

- Spearman(score rank, accuracy rank) = **0.9**
- top-bin minus bottom-bin accuracy = **0.5593**
- monotonicity violations: 1
- verdict: *higher score => more likely correct (useful ranking signal)*

| Class | n | Spearman | top-bottom gap | verdict |
|---|---:|---:|---:|---|
| Plate | 223 | 0.9 | 0.3333 | higher score => more likely correct (useful ranking signal) |
| TripleRiding | 58 | 0.7 | 0.9487 | higher score => more likely correct (useful ranking signal) |
| WithHelmet | 52 | 0.9487 | 0.5 | higher score => more likely correct (useful ranking signal) |
| WithoutHelmet | 208 | 1.0 | 0.7005 | higher score => more likely correct (useful ranking signal) |

### Saved failure artifacts

Annotated crops under `eval/artifacts_val/` (index: `eval/artifacts_val/index.json`), red box = prediction, green box = ground truth.

| Category | Saved |
|---|---:|
| `false_positives` | 30 (capped) |
| `false_negatives` | 30 (capped) |
| `class_confusions` | 24 |
| `low_confidence` | 18 |

### Confidence vs correctness

- mean confidence when correct: 0.8032 (419 preds)
- mean confidence when wrong: 0.6019 (122 preds)
- separation (correct - wrong): **0.2013** (>0 is good; <=0 means confidently wrong)

### Representative false positives

- `eval/clean_splits/val/images/ds1_00b3d6_CYB00EC198321128_jpg.rf.73e29c8c820ae6059a575a025a4a7c6a.jpg` predicted **WithoutHelmet** (0.257)
- `eval/clean_splits/val/images/ds1_062724_KBA01EC190628383_jpg.rf.9ea1b8b7e87d9899e478bc94525ed8be.jpg` predicted **Plate** (0.499)
- `eval/clean_splits/val/images/ds1_0d7c11_CYB00EC198319987_jpg.rf.46e5a012d3abae98c5b394c048f92114.jpg` predicted **WithoutHelmet** (0.463)
- `eval/clean_splits/val/images/ds1_101784_KBA01EC191108356_jpg.rf.73bd74bb8ce8a24a1de21e57002ab820.jpg` predicted **Plate** (0.748)
- `eval/clean_splits/val/images/ds1_19e97f_BikesHelmets677_png.rf.cfba5b9f26307590078051a246c9dc54.jpg` predicted **WithHelmet** (0.386)
- `eval/clean_splits/val/images/ds1_1e24d2_CYB03EC198353903_jpg.rf.947e09f18a1e82d382598f250f99c8da.jpg` predicted **WithoutHelmet** (0.771)
- `eval/clean_splits/val/images/ds1_260986_ADB03EC198442158_jpg.rf.e687673877b13b1b3e365399cecdffd9.jpg` predicted **WithoutHelmet** (0.339)
- `eval/clean_splits/val/images/ds1_29a63c_BikesHelmets301_png.rf.f4d291bda8c239728aee7c76d63dd281.jpg` predicted **WithoutHelmet** (0.688)
- `eval/clean_splits/val/images/ds1_2a2a43_CYB00EC198323712_jpg.rf.5157b2c1d6c04eab4f83cf725bc686d9.jpg` predicted **WithoutHelmet** (0.667)
- `eval/clean_splits/val/images/ds1_2a8054_ADB03TE193616150_jpg.rf.46803358e6b9b7299e2e28fb00cdf169.jpg` predicted **WithoutHelmet** (0.401)

### Representative false negatives (missed)

- `eval/clean_splits/val/images/ds1_101784_KBA01EC191108356_jpg.rf.73bd74bb8ce8a24a1de21e57002ab820.jpg` missed **Plate**
- `eval/clean_splits/val/images/ds1_11ae6a_KBA01EC190769810_jpg.rf.a1a3251a9b73692f18b824254b3fd006.jpg` missed **WithoutHelmet**
- `eval/clean_splits/val/images/ds1_128cb1_KBA01EC190575078_jpg.rf.116d8760330aa84c3c20942cb7212b3a.jpg` missed **WithoutHelmet**
- `eval/clean_splits/val/images/ds1_1a33f6_KBA01EC191104962_jpg.rf.a52c80dd3465b74111b6ffbe6881ffd8.jpg` missed **Plate**
- `eval/clean_splits/val/images/ds1_1a33f6_KBA01EC191104962_jpg.rf.a52c80dd3465b74111b6ffbe6881ffd8.jpg` missed **WithoutHelmet**
- `eval/clean_splits/val/images/ds1_260986_ADB03EC198442158_jpg.rf.e687673877b13b1b3e365399cecdffd9.jpg` missed **Plate**
- `eval/clean_splits/val/images/ds1_2a8054_ADB03TE193616150_jpg.rf.46803358e6b9b7299e2e28fb00cdf169.jpg` missed **Plate**
- `eval/clean_splits/val/images/ds1_2a8054_ADB03TE193616150_jpg.rf.46803358e6b9b7299e2e28fb00cdf169.jpg` missed **WithoutHelmet**
- `eval/clean_splits/val/images/ds1_2f82eb_RMD01EC198351922_jpg.rf.ce88871f04919811f2e86206120da6db.jpg` missed **WithoutHelmet**
- `eval/clean_splits/val/images/ds1_3b5a93_CYB00EC198313544_jpg.rf.3b03f040e9bf76e69652c4fce8245615.jpg` missed **WithoutHelmet**

