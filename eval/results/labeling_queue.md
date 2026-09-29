# Labelling queue — active-learning selection

- Budget 150; selected **150** from the 5772-image training pool and 5 pending reviews (candidates with any signal: 320); val/test (527 images) feed only the blind audit below.
- Priority ranks informativeness (`1 - Π(1 - signal)`), not a probability; selection decays repeats of a pattern and skips near-duplicate images.
- Evaluation images are **not** model-selected: they get a separate, seeded uniform-random audit sample, labelled blind to predictions (`labeling_queue_eval_audit.json`). Model-guided relabelling of held-out data would inflate measured accuracy.
- Blind evaluation audit: **50** val/test images.
- Whether this queue beats random selection is **not yet measured** — that needs the labels back (docs/FIELD_EVALUATION.md §5).

## Signal share: queue vs pool

| signal | pool candidates | queue |
|---|---:|---:|
| class_disagreement | 4% | 7% |
| helmet_contradiction | 7% | 13% |
| known_weak_stratum | 70% | 84% |
| label_gap | 3% | 7% |
| near_threshold | 50% | 48% |
| possible_missing_label | 46% | 45% |

## Queue by pattern — what a label would resolve

| pattern | items | what it resolves | measured by |
|---|---:|---|---|
| `training_relabel:near_threshold` | 61 | Decisions at the operating threshold flip with small changes. | Precision/recall of the 0.375 no-helmet floor on the labelled slice. |
| `training_relabel:possible_missing_label` | 61 | Confident detections with no label are mostly real, unlabelled objects on validation. | Measured precision after relabelling; label-noise rate per class. |
| `training_relabel:helmet_contradiction` | 14 | The model asserts both helmet and no helmet on one rider. | Contradiction rate; no-helmet precision at the operating point. |
| `training_relabel:label_gap` | 10 | Plates/heads are unlabelled on the triple-riding source, so the detector learned them as background there. | Plate recall on the dst source; share of triple riding that becomes fineable. |
| `training_relabel:class_disagreement` | 4 | A confident class that disagrees with the label is a model error or a label error — the label decides which. | WithHelmet/WithoutHelmet confusion rate on corrected labels. |

## By upstream source

`aug` images are offline-augmented copies whose original is not recoverable (`label_audit.source_of`): a label fixed on one fixes that copy only — as it would on an original, whose copies can't be found either. Dropping offline augmentation for online augmentation of cleaned originals is the structural fix (docs/ROADMAP.md).

| source | pool candidates | queue |
|---|---:|---:|
| aug | 196 | 88 |
| ds1 | 98 | 43 |
| dsv | 26 | 19 |

## Top 25

| # | item | purpose | priority | signals |
|---:|---|---|---:|---|
| 1 | `train:aug_46d52e44.jpg` | training_relabel | 0.9995 | near_threshold 0.98, possible_missing_label 0.88, helmet_contradiction 0.69, known_weak_stratum 0.30 |
| 2 | `train:aug_b43a2829.jpg` | training_relabel | 0.9965 | helmet_contradiction 0.97, class_disagreement 0.85, known_weak_stratum 0.30 |
| 3 | `train:dsv_dc876d_img316_jpg.rf.1rV9OHmQE49odcbG0XMe.jpg` | training_relabel | 0.9818 | label_gap 0.89, near_threshold 0.77, known_weak_stratum 0.30 |
| 4 | `train:aug_1a2d105e.jpg` | training_relabel | 0.9796 | class_disagreement 0.86, helmet_contradiction 0.80, known_weak_stratum 0.30 |
| 5 | `train:ds1_f5f095_BikesHelmets450_png.rf.67797e10f94ea74c1546988ad80616d7.jpg` | training_relabel | 0.9738 | possible_missing_label 0.89, near_threshold 0.67, known_weak_stratum 0.30 |
| 6 | `train:aug_02d6ed8d.jpg` | training_relabel | 0.9952 | helmet_contradiction 0.96, class_disagreement 0.82, known_weak_stratum 0.30 |
| 7 | `train:aug_8cb35c94.jpg` | training_relabel | 0.994 | near_threshold 0.99, known_weak_stratum 0.30 |
| 8 | `train:aug_0e1ba9e7.jpg` | training_relabel | 0.9656 | possible_missing_label 0.95, known_weak_stratum 0.30 |
| 9 | `train:ds1_a9427d_BikesHelmets200_png.rf.f00cecae7983a42077077529d592b9f1.jpg` | training_relabel | 0.9561 | label_gap 0.86, possible_missing_label 0.69 |
| 10 | `train:aug_93814316.jpg` | training_relabel | 0.956 | class_disagreement 0.66, helmet_contradiction 0.65, near_threshold 0.47, known_weak_stratum 0.30 |
| 11 | `train:dsv_fd3879_img226_jpg.rf.vvXWTMITNsmV2EJqlgzs.jpg` | training_relabel | 0.993 | near_threshold 0.95, possible_missing_label 0.79, known_weak_stratum 0.30 |
| 12 | `train:aug_9be953e4.jpg` | training_relabel | 0.9819 | helmet_contradiction 0.93, possible_missing_label 0.65, known_weak_stratum 0.30 |
| 13 | `train:aug_11aea479.jpg` | training_relabel | 0.9641 | possible_missing_label 0.82, class_disagreement 0.72, known_weak_stratum 0.30 |
| 14 | `train:ds1_6fe401_BikesHelmets600_png.rf.287e5a46270d62b0a51d265a4d65a0e1.jpg` | training_relabel | 0.8972 | label_gap 0.90 |
| 15 | `train:aug_4dcdc6ab.jpg` | training_relabel | 0.8534 | class_disagreement 0.79, known_weak_stratum 0.30 |
| 16 | `train:aug_86a1e11f.jpg` | training_relabel | 0.9915 | near_threshold 0.93, possible_missing_label 0.84, known_weak_stratum 0.30 |
| 17 | `train:aug_d289f455.jpg` | training_relabel | 0.9641 | helmet_contradiction 0.66, class_disagreement 0.63, near_threshold 0.59, known_weak_stratum 0.30 |
| 18 | `train:aug_e19480c5.jpg` | training_relabel | 0.9581 | possible_missing_label 0.86, near_threshold 0.57, known_weak_stratum 0.30 |
| 19 | `train:dsv_5c0ca0_img704_jpg.rf.PxoqlRYSFPl0ATbllMnO.jpg` | training_relabel | 0.8969 | label_gap 0.85, known_weak_stratum 0.30 |
| 20 | `train:aug_a6cc25e2.jpg` | training_relabel | 0.8519 | class_disagreement 0.85 |
| 21 | `train:ds1_7c9b07_CYB00EC198323079_jpg.rf.74eed0f030093afa2af6e3f370630a02.jpg` | training_relabel | 0.991 | near_threshold 0.85, class_disagreement 0.85, helmet_contradiction 0.43, known_weak_stratum 0.30 |
| 22 | `train:aug_a7697ae8.jpg` | training_relabel | 0.9633 | helmet_contradiction 0.82, class_disagreement 0.71, known_weak_stratum 0.30 |
| 23 | `train:aug_407f26bc.jpg` | training_relabel | 0.9431 | possible_missing_label 0.92, known_weak_stratum 0.30 |
| 24 | `train:dsv_c4c83d_img522_jpg.rf.9dQApQTBJa6NDWTLHjiR.jpg` | training_relabel | 0.8763 | label_gap 0.82, known_weak_stratum 0.30 |
| 25 | `train:dsv_d47b96_img245_jpg.rf.QSXNI3IYmgckYFts82KE.jpg` | training_relabel | 0.9866 | near_threshold 0.94, possible_missing_label 0.67, known_weak_stratum 0.30 |
