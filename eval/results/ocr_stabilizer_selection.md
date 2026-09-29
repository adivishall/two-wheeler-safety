# OCR stabilizer thresholds — selection experiment

> Simulated character-level OCR noise (look-alike substitutions, drops, insertions, missed frames). Measures the decision rule, not EasyOCR's field accuracy.

- 1000 simulated plate sequences per condition, 10 frames each; selected on seed 1234, reported on seed 5678.
- **Scored at the moment the pipeline commits**: the first read after which the vote elects anything (when a held violation is fined) — not after all reads.
- Objective: max mean coverage s.t. wrong-plate rate <= 1% in every design condition (sub=0.04,sys=0.0, sub=0.08,sys=0.0, sub=0.12,sys=0.0, sub=0.04,sys=0.3, sub=0.08,sys=0.3, sub=0.12,sys=0.3). If none qualifies: every config within 2 standard errors of the lowest worst-case wrong-plate rate is treated as equally safe and the most coverage wins (then the more conservative).
- `sys` = share of plates whose sequences contain a *consistent* look-alike misread (the same glyph misread the same way on half the frames).
- wrong-plate rate = share of ALL vehicles fined against a plate that is not theirs. coverage = share for which any plate is elected.

**Selected: `support>=3, agreement>=0.35, margin>=0.3`** (fallback: no config met the target; most coverage among configs within 2 SE (0.89%) of the lowest worst-case wrong-plate rate (2.0%)); previous default `1.0.0 rule: obs>=2, agreement>=0.35, no margin`.

## Selection (dev seed)

| config | mean coverage | worst wrong-plate rate | meets target |
|---|---:|---:|---|
| `support>=3, agreement>=0.5, margin>=0.3` | 0.440 | 2.0% | no |
| `support>=3, agreement>=0.35, margin>=0.3` | 0.500 | 2.1% | no |
| `support>=3, agreement>=0.5, margin>=0.0` | 0.445 | 2.1% | no |
| `support>=3, agreement>=0.5, margin>=0.1` | 0.445 | 2.1% | no |
| `support>=3, agreement>=0.5, margin>=0.2` | 0.444 | 2.1% | no |
| `support>=3, agreement>=0.35, margin>=0.2` | 0.570 | 3.0% | no |
| `support>=2, agreement>=0.35, margin>=0.3` | 0.579 | 3.4% | no |
| `support>=2, agreement>=0.5, margin>=0.3` | 0.532 | 3.4% | no |
| `support>=3, agreement>=0.35, margin>=0.1` | 0.585 | 3.7% | no |
| `support>=3, agreement>=0.35, margin>=0.0` | 0.587 | 4.0% | no |
| `support>=2, agreement>=0.5, margin>=0.2` | 0.565 | 4.5% | no |
| `support>=2, agreement>=0.5, margin>=0.1` | 0.569 | 4.9% | no |
| `support>=2, agreement>=0.5, margin>=0.0` | 0.569 | 5.0% | no |
| `support>=2, agreement>=0.35, margin>=0.2` | 0.719 | 6.1% | no |
| `support>=1, agreement>=0.35, margin>=0.3` | 0.609 | 7.1% | no |
| `support>=1, agreement>=0.5, margin>=0.3` | 0.563 | 7.1% | no |
| `support>=1, agreement>=0.5, margin>=0.2` | 0.601 | 8.4% | no |
| `support>=2, agreement>=0.35, margin>=0.1` | 0.764 | 8.9% | no |
| `support>=2, agreement>=0.35, margin>=0.0` | 0.765 | 9.2% | no |
| `support>=1, agreement>=0.35, margin>=0.2` | 0.744 | 9.9% | no |
| `support>=1, agreement>=0.5, margin>=0.1` | 0.635 | 10.8% | no |
| `support>=1, agreement>=0.5, margin>=0.0` | 0.638 | 11.0% | no |
| `support>=1, agreement>=0.35, margin>=0.1` | 0.822 | 16.4% | no |
| `support>=1, agreement>=0.35, margin>=0.0` | 0.983 | 48.5% | no |
| `1.0.0 rule: obs>=2, agreement>=0.35, no margin` | 1.000 | 59.4% | no |

## Held-out seed

| condition | config | coverage | correct | wrong plate | abstain |
|---|---|---:|---:|---:|---:|
| sub=0.04,sys=0.0 | `1.0.0 rule: obs>=2, agreement>=0.35, no margin` | 1.000 | 0.598 | 40.2% | 0.000 |
| sub=0.04,sys=0.0 | `support>=3, agreement>=0.35, margin>=0.3` | 0.700 | 0.699 | 0.1% | 0.300 |
| sub=0.08,sys=0.0 | `1.0.0 rule: obs>=2, agreement>=0.35, no margin` | 1.000 | 0.478 | 52.2% | 0.000 |
| sub=0.08,sys=0.0 | `support>=3, agreement>=0.35, margin>=0.3` | 0.497 | 0.496 | 0.1% | 0.503 |
| sub=0.12,sys=0.0 | `1.0.0 rule: obs>=2, agreement>=0.35, no margin` | 1.000 | 0.378 | 62.2% | 0.000 |
| sub=0.12,sys=0.0 | `support>=3, agreement>=0.35, margin>=0.3` | 0.344 | 0.343 | 0.1% | 0.656 |
| sub=0.20,sys=0.0 | `1.0.0 rule: obs>=2, agreement>=0.35, no margin` | 1.000 | 0.285 | 71.5% | 0.000 |
| sub=0.20,sys=0.0 | `support>=3, agreement>=0.35, margin>=0.3` | 0.151 | 0.147 | 0.4% | 0.849 |
| sub=0.30,sys=0.0 | `1.0.0 rule: obs>=2, agreement>=0.35, no margin` | 0.999 | 0.140 | 85.9% | 0.001 |
| sub=0.30,sys=0.0 | `support>=3, agreement>=0.35, margin>=0.3` | 0.030 | 0.028 | 0.2% | 0.970 |
| sub=0.04,sys=0.3 | `1.0.0 rule: obs>=2, agreement>=0.35, no margin` | 1.000 | 0.564 | 43.6% | 0.000 |
| sub=0.04,sys=0.3 | `support>=3, agreement>=0.35, margin>=0.3` | 0.625 | 0.607 | 1.8% | 0.375 |
| sub=0.08,sys=0.3 | `1.0.0 rule: obs>=2, agreement>=0.35, no margin` | 1.000 | 0.466 | 53.4% | 0.000 |
| sub=0.08,sys=0.3 | `support>=3, agreement>=0.35, margin>=0.3` | 0.485 | 0.471 | 1.4% | 0.515 |
| sub=0.12,sys=0.3 | `1.0.0 rule: obs>=2, agreement>=0.35, no margin` | 1.000 | 0.396 | 60.4% | 0.000 |
| sub=0.12,sys=0.3 | `support>=3, agreement>=0.35, margin>=0.3` | 0.343 | 0.334 | 0.9% | 0.657 |
| sub=0.20,sys=0.3 | `1.0.0 rule: obs>=2, agreement>=0.35, no margin` | 0.999 | 0.259 | 74.0% | 0.001 |
| sub=0.20,sys=0.3 | `support>=3, agreement>=0.35, margin>=0.3` | 0.121 | 0.116 | 0.5% | 0.879 |
| sub=0.30,sys=0.3 | `1.0.0 rule: obs>=2, agreement>=0.35, no margin` | 1.000 | 0.138 | 86.2% | 0.000 |
| sub=0.30,sys=0.3 | `support>=3, agreement>=0.35, margin>=0.3` | 0.027 | 0.025 | 0.2% | 0.973 |
