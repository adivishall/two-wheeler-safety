# Interview guide

This is the *spoken* companion to [RESUME.md](RESUME.md). Same rule applies:
every number here is produced by a named command in this repo, and the honest
version of each story is the one worth telling. If you can't run the command,
don't say the number.

The one sentence to anchor on:

> **A strong pipeline around an imperfect detector makes the detector's output
> *defensible* — associated, confirmed over time, de-duplicated, confidence-scored
> and auditable — but it cannot exceed what the model can see. So I measured both
> layers separately.**

---

## 30-second version

"It's a computer-vision system that flags two-wheeler traffic violations —
no-helmet, triple-riding, overspeed — from a photo or video, reads the number
plate, and packages auditable evidence into a review dashboard where a human
confirms or dismisses each one. The interesting part isn't 'YOLO detects
helmets' — it's the engineering *around* the detector that turns a noisy box
into a defensible fine: tracking, plate-to-rider association, temporal
confirmation, OCR voting, confidence scoring. And I measured the model and the
system separately so neither number is mistaken for the other."

## 60-second version

Add the honest thesis and one concrete example:

"The detector alone is middling and I say so — mAP@50 0.727 on a de-leaked
held-out split, with a genuinely weak helmet class. A raw detector at that
accuracy can't write a fine you'd defend. So the project is the layer on top: a
video pipeline that tracks each vehicle with stable IDs, matches plate boxes to
riders with Hungarian assignment instead of nearest-neighbour, requires a
violation to persist several frames before recording it, and votes OCR readings
across frames so one noisy frame can't name the wrong plate. Every recorded
violation carries a confidence built from four signals and a structured evidence
package. I benchmarked that pipeline against a naive single-frame detector on
identical inputs: even with a *perfect* detector, the naive policy averages 1.5
false positives per scenario and the pipeline has zero. The whole difference is
precision — which is exactly the axis that matters when the output is a fine."

## 3-minute version

Structure: problem → why it's hard → what I built → how I proved it → what I
won't claim.

1. **Problem.** Object detection on traffic footage is easy to demo and hard to
   trust. A detector produces independent boxes per frame; it's confidently wrong
   sometimes; plate text jitters; and several bikes can be in frame at once. None
   of that is safe to turn directly into an enforcement action.

2. **The hard parts** (the four I'd lead with):
   - **Cross-object association.** Plate boxes and rider/violation boxes are
     detected independently. With multiple bikes close together, deciding which
     plate belongs to which rider is a matching problem — I solve it with
     Hungarian assignment over a per-vehicle cost, not a nearest-neighbour guess.
   - **Noisy OCR.** A single frame's plate read is unreliable. A temporal
     stabilizer normalizes readings, votes across frames, and validates plate
     *structure* before trusting a result — and abstains rather than guessing when
     agreement is low.
   - **Confidently-wrong CV.** Temporal state machines require a violation to
     persist before it's recorded, and a contradiction safeguard refuses to
     advance when the model asserts both "helmet" and "no-helmet" on one rider.
   - **Speed must use video time.** Speed is `frame_index / fps`, not wall-clock,
     so a slower machine doesn't change the reported km/h.

3. **What I built.** The full path: detection → tracking → association → OCR
   stabilization → violation state machines → confidence → evidence → normalized
   SQLite → Flask review dashboard, with a bounded, cancellable background job for
   video. One video pipeline shared by the CLI and the web app so there's no
   drift.

4. **How I proved it.** Three separate evaluations — the detector on real
   de-leaked images, the pipeline on deterministic synthetic scenarios through
   the *real* runtime code, and application latency/throughput — plus a
   fault-injection error budget that says OCR carries 76.8% of system
   sensitivity, so that's where the next effort should go, not the detector.

5. **What I won't claim.** The synthetic 1.00s are a ceiling on what the suite
   tests, not field accuracy; confidence is a score, not a calibrated
   probability; and it's a review assistant with a human in the loop, not legal
   enforcement.

---

## 5 strongest technical decisions

1. **Measure the model and the system separately, and never combine them.** The
   most common way to mislead about a CV project is to present a system-level
   number as a computer-vision result. Detector metrics come from Ultralytics
   `val()` on real de-leaked images; pipeline metrics come from synthetic
   scenarios through the real tracker/association/stabilizer. The README has a
   side-by-side table whose entire purpose is to stop a reader conflating them.
   *(docs/EVALUATION.md, docs/MODEL_EVALUATION.md, docs/END_TO_END_EVALUATION.md)*

2. **Hungarian assignment for plate↔rider, replacing nearest-neighbour.** The
   original "attribute each violation to the nearest tracked plate" heuristic
   mis-assigns when bikes are close. Framing it as an optimal one-to-one
   assignment over a per-vehicle cost fixed the misattribution and is the kind of
   decision that's easy to defend. *(modules/association.py)*

3. **Temporal OCR voting that abstains.** At 12% simulated character noise,
   last-frame OCR names the wrong plate 85% of the time and best-confidence 30%;
   the stabilizer is wrong 2% — by answering only 51% of the time. For a system
   that fines people, converting errors into abstentions is the correct trade.
   *(modules/plate_recognizer.py, `evaluate_ocr.py --simulate --sweep`)*

4. **Confidence is a score in [0,1], deliberately never a probability.** It has
   never been calibrated against ground truth, so the code, the API, and the UI
   never render it as "97% likely". Choosing *not* to overstate is a decision.
   *(modules/confidence.py)*

5. **A model-free test suite.** Heavy inference is isolated behind lazy imports
   so 466 tests run in ~4s with no GPU, weights, or dataset — which is what makes
   CI and a fresh-clone demo possible. *(conftest.py, tests/, requirements-ci.txt)*

## 5 hardest engineering problems (and the resolution)

1. **An ID-switch bug in the tracker.** Two vehicles crossing under heavy overlap
   were merged into one track. Root-caused to the association merge thresholds;
   the fix took ID switches 13 → 0 and association accuracy 0.760 → 0.989 across
   eight crossing/overlap scenarios. *(commit `eca0ee5`, evaluate_tracking.py)*

2. **A benchmark that measured nothing.** The OCR-lock A/B reported a "+172%"
   speedup that didn't reproduce — the benchmark never forwarded the lock setting,
   so both arms ran with it on. Fixed, the real, honest number is +74.9%
   (28.7 → 50.2 FPS) with a byte-identical recorded fine. Finding this is the
   better story than the wrong number was. *(commit `2ba500b`, benchmark.py)*

3. **Dataset leakage I had to disprove, not assume.** A perceptual-hash audit
   found 9.8% of the test split were near-duplicates of training images. The
   tempting narrative is "leakage inflated our metrics" — but de-leaking moved
   mAP@50 by only +0.006. The data's answer was more interesting than the
   assumption. *(audit_dataset.py)*

4. **A "helmet fix" model that silently forgot a class.** An A/B checkpoint
   comparison caught a retrain that improved helmets but dropped `TripleRiding`
   mAP@50 to 0.000. Without the A/B harness this ships. *(compare_models.py)*

5. **Speed that's independent of the machine running it.** Wall-clock timing
   makes a slower host report a slower vehicle. Speed is computed from video time
   (`frame_index / fps`) with a required pixels-per-meter calibration; without
   calibration, speed is skipped rather than reported meaninglessly.
   *(modules/speed.py)*

## 5 likely interviewer questions

**Q: Your pipeline shows precision and recall of 1.0 — isn't that overfit /
too good to be true?**
Yes, and I say so in the README. That 1.0 is on *deterministic synthetic
scenarios* with no motion blur, no crowds, no detector noise, and the suite was
written by the same person as the pipeline. It's a statement that the plumbing —
tracking, association, temporal confirmation, de-duplication — is correct, not a
claim about real roads. The real-image number is the detector's mAP@50 0.727,
which is middling and honestly so.

**Q: Why Hungarian assignment and not just nearest plate?**
Nearest-neighbour is a greedy per-object choice that mis-assigns when two bikes
are close: the same plate can end up "nearest" to two riders, or a rider grabs a
neighbour's plate. Hungarian gives the globally optimal one-to-one matching over
a cost matrix, so the whole frame is assigned consistently. The measurable effect
is in the association-accuracy jump on crossing scenarios.

**Q: How do you handle a detector that's confidently wrong?**
Two mechanisms. Temporal confirmation: a violation must persist `STREAK_THRESHOLD`
frames before it's recorded, which filters single-frame misfires — fining on one
frame drops helmet precision to 0.67, two frames restores 1.00 at no recall cost.
And a contradiction safeguard: if the model asserts both helmet and no-helmet on
the same rider (IoU overlap), the state machine advances neither. Neither of
these fixes a *steadily* wrong prediction — that's what human review is for.

**Q: What's the bottleneck, and how do you know?**
OCR, and I know because I ran a fault-injection error budget — degrade each stage
by an equal 30% and measure the end-to-end F1 drop. OCR accounts for 76.8% of
system sensitivity, ahead of detector class-confusion (15.3%) and recall (7.3%).
The actionable insight: a *corrupted* plate is catastrophic (it fines a real but
wrong vehicle), a *missing* plate is nearly free because other frames recover it.
So the highest-value missing thing is a labelled plate-sequence set, not a bigger
detector.

**Q: Is this deployable as traffic enforcement?**
No, and the README leads with that. It's a detection-and-review assistant: a
human confirms every flag, it never resolves owner identity (it only decodes the
public registration *region* from the plate), confidence is a score not a
probability, and the accuracy numbers are held-out-on-the-same-pool, not
generalisation to a new city or camera.

## Honest limitations (state these unprompted)

- Held-out ≠ generalisation: the test split shares the training pool's biases.
- `WithHelmet` is genuinely weak (mAP@50 0.387) on only 27 test instances.
- Confidence ranks correctness well for `WithoutHelmet` (Spearman 1.00) and not
  at all for `TripleRiding` (−0.20); "0.8" never means "80% correct".
- No *field* OCR accuracy is measured — the comparison uses simulated noise.
- Two-way OCR ties elect rather than abstain (agreement 0.5 clears threshold).
- Speed is an estimate (best MAE 11.3 km/h), needs a surveyed homography.
- Single-process design; not a distributed service.

See [docs/MODEL_EVALUATION.md](MODEL_EVALUATION.md) §8 and
[docs/END_TO_END_EVALUATION.md](END_TO_END_EVALUATION.md) §7 for the full list.
