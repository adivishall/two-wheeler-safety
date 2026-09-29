# Temporal confirmation — rule selection experiment

> Simulated detector noise at rates measured on the validation split. Temporal correlation of real errors is unmeasured (no labelled video); stickiness is swept to show the dependence. Plates are always readable here, so this isolates the helmet decision.

- Noise model: `measured (val, de-leaked)` — missed box 20.1% (violator) / 15.4% (compliant); helmeted rider labelled WithoutHelmet **15.4%**; bare-headed rider labelled WithHelmet 7.7% (per frame). Source: `eval/results/eval_traffic_model-2_val_clean.json`.
- 1200 riders per condition (half violators), dwell 10, 25, 50 frames, stickiness 0.0, 0.5, 0.8 (P a frame repeats the last frame's outcome — real errors come in runs; unmeasured, so swept).
- Selected on dev seed 101, reported on test seed 202.
- Objective: max mean recall s.t. false-flag rate <= 1% in every design condition; if none qualifies, minimise the worst-case false-flag rate (ties -> recall). Design conditions: rho=0,dwell=10, rho=0,dwell=25, rho=0,dwell=50, rho=0.5,dwell=10, rho=0.5,dwell=25, rho=0.5,dwell=50.

**Selected rule: consecutive 5 + >=70% of >=12 observed** (fallback: no rule met the target; minimum worst-case false-flag rate).

## Selection (dev seed) — every candidate

| rule | mean recall | worst false-flag rate | meets ≤1%? |
|---|---:|---:|---|
| consecutive 1 | 0.999 | 100.0% | no |
| 2 of 4 | 0.997 | 93.2% | no |
| consecutive 2 | 0.995 | 88.8% | no |
| 3 of 6 | 0.992 | 76.7% | no |
| 3 of 5 | 0.991 | 75.0% | no |
| 4 of 8 | 0.981 | 64.3% | no |
| 4 of 6 | 0.979 | 59.5% | no |
| consecutive 3 | 0.978 | 69.2% | no |
| 5 of 10 | 0.969 | 50.5% | no |
| 5 of 7 | 0.966 | 41.0% | no |
| 6 of 12 | 0.944 | 41.8% | no |
| consecutive 4 | 0.944 | 51.7% | no |
| 6 of 8 | 0.941 | 30.0% | no |
| consecutive 5 | 0.897 | 33.2% | no |
| consecutive 3 + >=50% of >=8 observed | 0.844 | 14.2% | no |
| consecutive 3 + >=60% of >=8 observed | 0.843 | 8.2% | no |
| consecutive 6 | 0.839 | 21.2% | no |
| consecutive 3 + >=70% of >=8 observed | 0.837 | 3.8% | no |
| consecutive 5 + >=50% of >=8 observed | 0.804 | 10.3% | no |
| consecutive 5 + >=60% of >=8 observed | 0.802 | 6.7% | no |
| consecutive 5 + >=70% of >=8 observed | 0.798 | 3.0% | no |
| consecutive 8 | 0.709 | 8.2% | no |
| consecutive 3 + >=50% of >=12 observed | 0.660 | 9.5% | no |
| consecutive 3 + >=60% of >=12 observed | 0.658 | 5.2% | no |
| consecutive 3 + >=70% of >=12 observed | 0.654 | 1.7% | no |
| consecutive 5 + >=50% of >=12 observed | 0.634 | 6.5% | no |
| consecutive 5 + >=60% of >=12 observed | 0.633 | 3.7% | no |
| consecutive 5 + >=70% of >=12 observed | 0.629 | 1.3% | no |

## Held-out seed — selected vs previous default vs single frame

| condition | rule | precision | recall | false-flag rate |
|---|---|---:|---:|---:|
| rho=0,dwell=10 | consecutive 5 (pre-experiment default) | 0.998 | 0.712 | 0.2% |
| rho=0,dwell=10 | consecutive 1 | 0.545 | 1.000 | 83.5% |
| rho=0,dwell=10 | consecutive 5 + >=70% of >=12 observed | 1.000 | 0.000 | 0.0% |
| rho=0,dwell=25 | consecutive 5 (pre-experiment default) | 0.998 | 0.987 | 0.2% |
| rho=0,dwell=25 | consecutive 1 | 0.503 | 1.000 | 98.7% |
| rho=0,dwell=25 | consecutive 5 + >=70% of >=12 observed | 1.000 | 0.898 | 0.0% |
| rho=0,dwell=50 | consecutive 5 (pre-experiment default) | 0.992 | 1.000 | 0.8% |
| rho=0,dwell=50 | consecutive 1 | 0.500 | 1.000 | 100.0% |
| rho=0,dwell=50 | consecutive 5 + >=70% of >=12 observed | 1.000 | 0.998 | 0.0% |
| rho=0.5,dwell=10 | consecutive 5 (pre-experiment default) | 0.931 | 0.680 | 5.0% |
| rho=0.5,dwell=10 | consecutive 1 | 0.622 | 0.990 | 60.2% |
| rho=0.5,dwell=10 | consecutive 5 + >=70% of >=12 observed | 1.000 | 0.000 | 0.0% |
| rho=0.5,dwell=25 | consecutive 5 (pre-experiment default) | 0.871 | 0.975 | 14.5% |
| rho=0.5,dwell=25 | consecutive 1 | 0.542 | 1.000 | 84.3% |
| rho=0.5,dwell=25 | consecutive 5 + >=70% of >=12 observed | 0.994 | 0.853 | 0.5% |
| rho=0.5,dwell=50 | consecutive 5 (pre-experiment default) | 0.760 | 1.000 | 31.5% |
| rho=0.5,dwell=50 | consecutive 1 | 0.504 | 1.000 | 98.5% |
| rho=0.5,dwell=50 | consecutive 5 + >=70% of >=12 observed | 0.988 | 0.992 | 1.2% |
| rho=0.8,dwell=10 | consecutive 5 (pre-experiment default) | 0.842 | 0.727 | 13.7% |
| rho=0.8,dwell=10 | consecutive 1 | 0.726 | 0.945 | 35.7% |
| rho=0.8,dwell=10 | consecutive 5 + >=70% of >=12 observed | 1.000 | 0.000 | 0.0% |
| rho=0.8,dwell=25 | consecutive 5 (pre-experiment default) | 0.758 | 0.950 | 30.3% |
| rho=0.8,dwell=25 | consecutive 1 | 0.617 | 0.992 | 61.5% |
| rho=0.8,dwell=25 | consecutive 5 + >=70% of >=12 observed | 0.930 | 0.798 | 6.0% |
| rho=0.8,dwell=50 | consecutive 5 (pre-experiment default) | 0.650 | 0.998 | 53.8% |
| rho=0.8,dwell=50 | consecutive 1 | 0.550 | 1.000 | 81.7% |
| rho=0.8,dwell=50 | consecutive 5 + >=70% of >=12 observed | 0.928 | 0.950 | 7.3% |

