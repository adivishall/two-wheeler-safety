# Error analysis and error budget

Where the system fails, which layer owns each failure, and whether it is fixed
by better data, better pipeline logic, or not at all yet. Every row points at
the generated evidence behind it. Detector metrics are in
[MODEL_EVALUATION.md](MODEL_EVALUATION.md); pipeline metrics in
[END_TO_END_EVALUATION.md](END_TO_END_EVALUATION.md).

## 1. Failure taxonomy

| # | failure | evidence | owner | fix by training / data? | pipeline mitigation | still unresolved |
|---|---|---|---|---|---|---|
| F1 | **Helmeted rider called no-helmet** — the error that fines an innocent person | 15.4% of helmeted rider boxes on val (`eval_traffic_model-2_val_clean.json`); examples in `manual_error_review.md`; glare costs WithHelmet −0.16 AP (`robustness_val.md`) | detector | more WithHelmet data (27 instances per held-out split is the root problem); glare augmentation | streak + ≥70%-of-≥12-frames gate (worst-case false-flag 31.5% → 1.2%, `temporal_confirmation.md`); contradiction safeguard; per-frame floor 0.375; human review | a rider misclassified *consistently* across frames (highly correlated errors: 6–7% false flags in the stress case); photos get no temporal defence |
| F2 | **Person near a two-wheeler ≠ rider** — cyclist in a helmet, pedestrian by a parked bike, called WithoutHelmet | `manual_error_review.md` FP #8, #9 | detector / data | hard negatives: pedestrians, cyclists, parked bikes | video: needs ≥12 observed frames; a plate must associate | a person standing by a parked bike for 12+ frames can be confirmed, and a photo of one is fined outright if the plate reads. A "vehicle is moving" gate would close this — and would also exempt bare-headed riders waiting at a red light, so it is a policy choice, not a free fix |
| F3 | **Rider box missed** | 15–20% of rider boxes on val | detector | more data; resolution matters less than blur (`robustness_val.md`) | one blank frame tolerated; gate counts *observed* frames | the largest owner of end-to-end F1 at the measured operating point (§2) |
| F4 | **Head covering read as a helmet** — dupatta/scarf → WithHelmet | 4 of the 16 top val confusions (`manual_error_review.md`) | data | add head-covering riders to training | none possible downstream | missed violations concentrated in one group of riders — a fairness problem |
| F5 | **Triple riding without a plate** | disjoint label sets; plate predicted on 6% of triple-riding images vs 65% elsewhere (`label_audit.md`) | data | label plates/heads on the triple-riding source, or mask unlabelled classes | confirmed triple riding without a plate is withheld and logged, never fined against a guess | most triple riding on real footage will be withheld; held-out metrics cannot show it |
| F6 | **OCR misreads** — wrong glyph, consistent misread, invalid format | clean *synthetic* renders: 37.5% read exactly, n = 8 per condition (`ocr_sanity_check.json`); `ocr_stabilizer_selection.md` | OCR | a plate-specific recogniser; a labelled plate-sequence set to measure it | structure-aware look-alike correction; winner needs 3 valid reads of its own, ≥35% agreement, ≥0.3 margin; ties abstain; locked plates re-read and must be recently re-confirmed to fine; wrong-plate ≤ 1.8% in simulation, scored when the fine is committed | coverage: at 12% character noise only 34% of vehicles get a plate (70% at 4%); a *consistent* misread is the case the vote cannot out-vote. Field accuracy is **unmeasured** |
| F7 | **Correlated detector errors defeat temporal voting** | a plain streak flags 31.5% of helmeted riders at 2 s dwell under moderate correlation; k-of-n is worse (`temporal_confirmation.md`) | decision rule (given the detector) | a better detector | track-level fraction gate (this is why it exists) | real temporal correlation is unmeasured (no labelled video) — swept, not known |
| F8 | **Short dwell** — rider in view < 12 frames | `brief_pass` scenario is a miss by design | decision rule | — | — | never fined: the stated price of F1/F7 protection |
| F9 | **Association under full overlap** — one bike directly behind another | `crossing` scenario: association accuracy 0.95, 2 frames merged (`pipeline_evaluation.md`) | association | — | merge thresholds (IoU 0.5 / containment 0.8) chosen by `tracking_eval.py`; identity survives the crossing | from boxes alone, fully overlapping riders cannot be separated |
| F10 | **Identity split on re-entry** — rider leaves view > 15 frames | `reappears_after_exit` scenario | tracking | — | one fine per (plate, violation) per run (**fixed**; used to fine twice) | two different vehicles with the same OCR'd plate in one run are fined once (the safe direction) |
| F11 | **Stale plate box** — OCR/speed on where the plate *was* | e2e test `test_plate_occlusion_never_ocrs_a_stale_box` | pipeline | — | **fixed**: OCR, speed and evidence crops require the plate in the current frame | — |
| F12 | **Speed geometry** — motion toward the camera | constant pixels-per-metre misses every overspeeder (MAE 41.8 km/h); homography MAE 11.3 (`pipeline_evaluation.md`, synthetic) | speed | — | homography calibration; speed skipped without calibration; calibration scaled with frame resize (**fixed**: read 2/3 speed on downscaled video) | no surveyed ground truth; ±11 km/h is not enforcement grade |
| F13 | **Label noise** in held-out data | 11 of the 16 top val FPs are unlabelled real objects; ≥3 of 16 top confusions mislabelled (`manual_error_review.md`) | data | relabel val/test | — | measured precision understates the detector; WithHelmet numbers are noisy on top of small |

## 2. Error budget — oracle ablation at the measured operating point

`eval/results/pipeline_evaluation.md` § Error budget. Every stage fails at its
per-frame rate measured on val (rider missed 15–20%, helmet class flipped
7.7–15.4%, plate missed 18%); OCR, which has no field measurement, is swept.
Each stage is then made perfect in turn; the end-to-end fine F1 it recovers is
its share. Synthetic scenarios, independent per-frame faults.

| OCR read-error rate | F1, all faults | rider recall | helmet class | plate recall | OCR |
|---|---:|---:|---:|---:|---:|
| 10% | 0.955 | **54%** | 42% | 4% | 0% |
| 30% | 0.956 | **54%** | 42% | 4% | 0% |
| 50% | 0.955 | **56%** | 40% | 4% | 0% |

Faults are drawn with common random numbers — one independent stream per stage
— so making one stage perfect changes only that stage's outcomes; an earlier
version shared one stream and an "oracle" run could score below the faulty run
by chance.

Reading it:

- At the measured detector rates **the pipeline absorbs most detector noise**:
  end-to-end F1 falls from 0.970 (no faults; the gap is the by-design short-dwell
  miss) to ~0.95.
- What remains is owned by the **detector** — missed rider boxes first (54–56%),
  helmet class flips second (40–42%), missed plate boxes a distant third (4%) —
  and that ranking holds across the whole OCR sweep.
- Independent single-glyph OCR errors cost ~0 F1, even at 50% of reads: the vote
  needs three agreeing valid reads, and these scenarios keep a plate in view
  long enough to collect them (when they don't agree, it abstains rather than
  electing a wrong plate). OCR's real risks are not F1 but **coverage** and
  **wrong plates under consistent misreads** — measured separately in
  `ocr_stabilizer_selection.md`, and the reason the vote needs a margin.
- This replaces an earlier budget that called OCR "76.8% of system sensitivity".
  That figure compared 30% per-*character* OCR corruption (a 10-glyph plate then
  almost never reads cleanly) with 30% per-*box* detector faults — a statement
  about units, not about the system.

## 3. What the analysis says to do next, in order

1. **Label plates and heads on the triple-riding images** (F5). The single
   change that turns a whole violation type from "withheld" into "fineable" —
   and the precondition for measuring it at all.
2. **More WithHelmet and head-covering riders, plus hard negatives** (F1, F2,
   F4). WithHelmet's AP CI is [0.20, 0.73]: nothing about that class can be
   tuned until it is measured on more than 27 instances.
3. **Decide the parked-vehicle policy** (F2). A motion gate is a few lines in
   `ViolationPipeline`, but it trades the pedestrian-by-a-parked-bike false
   positive for missing riders stopped at a signal; better hard negatives (item 2)
   attack the same error without that trade.
4. **A labelled plate-sequence set** (F6) — the only way to replace the
   simulated OCR numbers with real ones, and to know how much coverage the
   conservative vote actually costs.
5. **Fix the held-out labels** (F13) before any further model comparison.

What training can fix: F1–F5. What only the pipeline can fix: F7–F11 (done where
marked). What neither can fix without new data: the size of the WithHelmet
uncertainty, real OCR accuracy, real temporal error correlation.
