# Recipe comparison — seed means on `val` (352 images)

- Baseline **v1**. A recipe's AP = mean over its seeds; CIs resample images and seeds together. Verdicts need ≥ 3 seeds per recipe.

| recipe | seeds | Plate | WithHelmet | WithoutHelmet | TripleRiding | mAP@50 |
|---|---:|---|---|---|---|---:|
| v1 | 2 | 0.8606 ± 0.0171 | 0.3675 ± 0.0695 | 0.74 ± 0.0066 | 0.8372 ± 0.0333 | 0.7013 |
| v2_dedup | 1 | 0.8503 | 0.4441 | 0.7818 | 0.8927 | 0.7422 |

± = between-seed SD (needs ≥ 2 seeds).

| candidate − baseline | Plate | WithHelmet | WithoutHelmet | TripleRiding | mAP@50 | verdict |
|---|---|---|---|---|---|---|
| v2_dedup | -0.010 [-0.074, +0.053] | +0.077 [-0.125, +0.279] | +0.042 [+0.005, +0.079] | +0.055 [-0.044, +0.155] | +0.041 [-0.035, +0.117] | **insufficient seeds (2 vs 1; need 3 each)** |
