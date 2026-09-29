# End-to-end evaluation — the pipeline, not the model

*"How reliably does the software turn noisy detections into a correct final
violation record?"* — measured separately from *"what can the detector see?"*
([MODEL_EVALUATION.md](MODEL_EVALUATION.md)).

> **Everything on this page runs on synthetic inputs.** The numbers are
> statements about the decision logic under stated input conditions, **not**
> field accuracy on real roads. Real-footage end-to-end accuracy is unmeasured:
> no labelled video exists for this project.

## How it is measured

- **The shipped code, not a copy.** Every evaluator drives
  `modules/pipeline.py :: ViolationPipeline` — the object the web job runs per
  frame, with its defaults. The only substitutions are the inputs: synthetic
  boxes instead of YOLO, and a plate-box → text lookup instead of EasyOCR, called
  only for a plate box detected that frame. (Before v1.1.0 two evaluators
  carried their own copies of the loop, with a different confirm window and
  different handling of missing boxes — `docs/AUDIT.md` E1.)
- **Detector noise at measured rates.** Where noise is injected, its rates come
  from the detector's confusion matrix on the de-leaked *validation* split, not
  from guesses and never from test.
- **Selection vs report.** Every rule on this page was chosen on one random seed
  and is reported on another, under an objective stated before looking.
- **Real-code integration tests** (`tests/test_end_to_end.py`) run the actual
  `process_video`, `analyze_image` and Flask routes on scripted video with fake
  YOLO/OCR, through the multi-error cases below.

Generated sources: `eval/results/pipeline_evaluation.md`,
`temporal_confirmation.md`, `ocr_stabilizer_selection.md`. Regenerate in §8.

## 1. End-to-end fines — deterministic scenarios

`pipeline_evaluation.md` § End-to-end fines. Ten scenarios, ~1.2 s of video each
unless stated.

| scenario | what it tests | result |
|---|---|---|
| `single_no_helmet`, `single_triple` | the basic cases | fined, right plate |
| `late_plate` | plate readable only after the violation confirms | held, then fined once the plate stabilises |
| `clean_helmet` | compliant rider | not fined |
| `two_adjacent` | violator next to a compliant rider | only the violator, with *its* plate |
| `crossing` | two violators cross and fully overlap for 2 frames | both fined correctly; identity survives (0 ID switches); association 0.95 (the merged frames) |
| `three_bikes` | middle bike of three | only the middle one |
| `occlusion` | rider vanishes for 2 frames | same identity, fined once |
| `reappears_after_exit` | rider gone 20 frames (> tracker `max_age`), returns as a new track | **fined once** (was twice before v1.1.0) |
| `brief_pass` | violator in view for 8 frames | **not fined — by design** (needs ≥ 12 observed frames) |

**Totals: precision 1.0, recall 0.9** (TP 9, FP 0, FN 1; 0 wrong-vehicle, 0
wrong-plate, 0 duplicates). The one miss is the documented short-dwell cost.

`tests/test_end_to_end.py` adds, on the real `process_video` / app: plate
occlusion (no OCR ever runs on a stale box), competing OCR readings (withheld,
reason `contested`), one bad OCR frame in ten (plate unchanged), simultaneous
violations on two bikes, a noisy detector at the measured 15% flip rate (helmeted
rider not fined), intermittent frames (violator still fined), no speed
calibration (no speed, no overspeed), downscaled speed (45 km/h measured
correctly), cancellation (stops, persisted as `cancelled`), and the persisted
record behind the API (sidecar hashes verify).

## 2. Is the pipeline better than a single-frame detector?

`pipeline_evaluation.md` § Is the pipeline better… Both policies see identical
detections, degraded by symmetric helmet class flips. `naive` fines if any single
frame shows a violation box — and is handed perfect plate association for free,
so the pipeline's advantage is a lower bound.

| flip rate | naive P | naive F1 | pipeline P | pipeline R | pipeline F1 | F1 gain |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.852 | 0.920 | 1.000 | 0.955 | 0.976 | **+0.056** |
| 0.10 | 0.722 | 0.831 | 1.000 | 0.953 | 0.975 | **+0.144** |
| 0.20 | 0.718 | 0.828 | 1.000 | 0.921 | 0.957 | **+0.129** |
| 0.30 | 0.718 | 0.828 | 1.000 | 0.788 | 0.861 | +0.033 |
| 0.40 | 0.718 | 0.828 | 0.993 | 0.606 | 0.665 | **−0.163** |

- The pipeline is **more precise at every noise level** — near 1.0 throughout.
- Its **F1** advantage holds where the detector actually operates (val-measured
  helmet flips: 7.7–15.4% per frame) and **reverses at 40% symmetric flips**,
  where it trades recall for precision. For a system that fines people that is
  the right trade — but it is a trade, and the earlier claim that the pipeline
  "beats the detector at every noise level" only held under an evaluator setting
  that never shipped (`AUDIT.md` E2).
- At 0.00 injected noise the naive policy still makes 1.5 false positives: the
  suite contains designed one-frame flickers and a contradiction case. Those are
  detector errors by construction — "zero injected noise" is not "a perfect
  detector".

## 3. Choosing the temporal confirmation rule

`temporal_confirmation.md`. Simulated riders whose rider boxes are missed or
mislabelled per frame at the detector's val-measured rates; `stickiness` makes
errors come in runs (real frames are correlated — how much is unmeasured, so it
is swept). 1,200 riders per condition; dwell 10 / 25 / 50 frames (0.4 / 1 / 2 s).
Objective: maximise recall s.t. ≤ 1% of helmeted riders flagged in every design
condition (stickiness 0 and 0.5).

Held-out seed, false-flag rate (share of **helmeted** riders flagged):

| stickiness, dwell | consecutive 5 (old) | **shipped: consecutive 5 + ≥70% of ≥12 observed** | recall, shipped |
|---|---:|---:|---:|
| 0, 1 s | 0.2% | 0.0% | 0.90 |
| 0, 2 s | 0.8% | 0.0% | 1.00 |
| 0.5, 1 s | 14.5% | 0.5% | 0.85 |
| 0.5, 2 s | **31.5%** | **1.2%** | 0.99 |
| 0.8, 2 s (stress) | 53.8% | 7.3% | 0.95 |
| any, 0.4 s | 0.2–13.7% | 0.0% | **0.00** |

- **A streak's false-flag rate grows with time in view** under correlated errors:
  a helmeted rider seen longer gets more chances at one lucky run of flips.
  k-of-n voting (e.g. 3 of 6) is worse still — it scans more windows.
- A **track-level fraction** converges instead of accumulating chances; adding
  it is what cut the worst case from 31.5% to 1.2%.
- No rule met the 1% target in every condition, so the pre-stated fallback
  (minimise the worst case) selected it. Re-run at 4× the riders, the same rule
  won; the runner-up differs only in streak length (3 vs 5), so the gate is the
  finding and 3 vs 5 frames is within noise.
- **Price:** a rider in view for fewer than 12 frames is never fined for no
  helmet. (Triple riding and overspeed keep the 5-frame streak.)

## 4. Choosing the OCR vote

`ocr_stabilizer_selection.md`. Simulated plate sequences (10 reads each) with
look-alike substitutions, dropped/extra characters, missed frames — and, in half
the design conditions, a *consistent* misread (one glyph read the same wrong way
on half the frames, as a real plate image does). 1,000 sequences per condition;
selected on one seed, reported on another. Objective: maximise coverage s.t.
≤ 1% of vehicles get a wrong plate in every design condition (4–12% character
noise, with and without consistent misreads).

**Scored at the moment the pipeline commits** — the first read after which the
vote elects anything, which is when a violation held for want of a plate is
fined. An earlier version scored the plate after all ten reads, a later and
better-informed moment than the pipeline ever uses, and so under-counted wrong
plates ([AUDIT.md](AUDIT.md) R1).

Held-out seed:

| condition | 1.0.0 rule coverage | 1.0.0 rule **wrong plate** | shipped coverage | shipped **wrong plate** |
|---|---:|---:|---:|---:|
| 4% char noise | 1.00 | 40.2% | 0.70 | 0.1% |
| 8% | 1.00 | 52.2% | 0.50 | 0.1% |
| 12% | 1.00 | 62.2% | 0.34 | 0.1% |
| 4% + consistent misreads | 1.00 | 43.6% | 0.63 | **1.8%** |
| 8% + consistent misreads | 1.00 | 53.4% | 0.49 | 1.4% |
| 12% + consistent misreads | 1.00 | 60.4% | 0.34 | 0.9% |

1.0.0 rule: ≥ 2 reads *of any kind*, ≥ 35% agreement, no margin — so it could
elect from its first readable read. Shipped: ≥ 3 valid reads for the winner,
≥ 35% agreement, margin ≥ 0.3; exact ties abstain.

- **No config met the 1% target** on the dev seed (lowest worst case 2.0%). The
  fallback treats every config within 2 standard errors of that minimum as
  equally safe and takes the most coverage — a plain arg-min was being decided
  by one sequence between configs six coverage points apart.
- **The price is coverage**: the shipped vote names a plate for 34–70% of
  vehicles and withholds the rest for review. The worst case left is the
  consistent misread at low noise — a stable wrong glyph is exactly what a vote
  cannot out-vote.
- The first run of this experiment (200 sequences) picked a different config
  whose advantage was 2–4 sequences; at 5× the sample it did not survive
  (DECISIONS #16).

Versus single-frame policies (`pipeline_evaluation.md` § OCR; 12% substitution,
no systematic misreads, **scored after all ten reads** over 100 sequences — so
its coverage is not the 34% above): the last frame's read is right 15% of the
time, the most confident read 70%, the temporal vote answers 28% of vehicles and
is right on 100% of those. Every non-answer is an abstention, never an invalid
plate.

## 5. Error budget

Oracle ablation at the measured operating point — see
[ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) §2. Summary: at the val-measured detector
rates the pipeline loses ~1.5 F1 points on these scenarios, and the detector
owns them: making rider detection perfect alone recovers the whole gap, helmet
classification alone 72–80%, plate detection 7–8% (stages overlap, so these
don't sum) — across the whole OCR sweep (10–50% of reads with a wrong glyph). OCR errors cost ~0 F1 here because the vote withholds rather than
guessing, and these scenarios keep each plate in view long enough to collect
agreeing reads; the OCR cost that remains is coverage and consistent misreads
(§4), which F1 on this suite does not see.

## 6. Speed

`pipeline_evaluation.md` § Speed. Synthetic constant-velocity trajectories on a
perspective ground plane; MAE in km/h.

| motion | constant px/m | linear-plane | homography |
|---|---:|---:|---:|
| lateral (across the frame) | 1.9 | **0.9** | 3.2 |
| approach (toward the camera) | 41.8 | 40.1 | **11.3** |

A constant pixels-per-metre calibration misses **every** approaching
overspeeder. Even the best calibration has MAE 11.3 km/h (biased +8.4) for
approaching bikes in simulation, with no
surveyed ground truth — an estimate, not a radar reading. v1.1.0 also fixed
calibration under frame downscaling (speeds read at 2/3 of truth on a 1920-px
video processed at 1280).

## 7. What this page does not show

- Field accuracy of anything. No labelled video; synthetic inputs throughout.
- Real temporal error correlation (swept), real OCR error statistics (modelled),
  real plate coverage.
- Behaviour when two vehicles fully occlude each other for long, or in dense
  traffic beyond three bikes.
- Triple riding end-to-end on real footage: the detector rarely finds plates on
  those vehicles (dataset gap, [DATASET.md](DATASET.md)), which synthetic
  scenarios with a plate always present cannot reveal.

## 8. Reproducing this page

```bash
python3 evaluate_pipeline.py                         # scenarios, naive vs pipeline, budget, OCR, speed
python3 evaluate_temporal.py                         # temporal rule (rates from the val report)
python3 evaluate_ocr.py --simulate --sweep --policy-sweep --out eval/results \
    --name ocr_policy_simulation                     # OCR vote thresholds
pytest tests/test_end_to_end.py                      # real-code multi-error scenarios
```

All model-free: no weights, no dataset, no network.
