# Demo walkthrough

A five-minute tour that exercises the whole system end to end. Everything runs
in one browser window; no terminal needed after startup.

## One command (no model, no weights)

```bash
make demo         # or: python3 demo.py
```

This seeds a small, self-contained **demo dataset** and starts the app at
<http://127.0.0.1:5000>. It needs only the model-free deps
(`pip install -r requirements-ci.txt -c constraints-ci.txt`) — **no torch, no
weights** — so it works on a fresh clone in seconds. It populates every panel so
there's something real to click:

- **5 vehicles**, ~14 violations across the three types, spread over 10 days;
- confidence scores spanning ~0.44–0.97 (fills the distribution histogram);
- a mix of **pending / confirmed / dismissed** reviews and a few **paid** fines;
- **2 processing sessions** with throughput stats (so Analytics + the per-session
  drill-down are meaningful);
- real bundled photos as evidence **only for the plates they actually show**
  (`MH02DL4596` = `samples/plate4.jpeg`, `MH12HS8818` = `samples/plate8.jpeg`);
  the other demo plates get a grey "DEMO RECORD — illustrative" card instead of a
  photo of someone else's bike.

The seeded data is **synthetic demo data**, labelled as such on stdout and in the
dashboard (sessions are named `[demo] …` with model version "demo (synthetic
records, no model run)") — the confidence scores and sessions are illustrative,
not real detections. For a
genuine end-to-end run with the actual detector, use `make demo-real`
(= `python3 demo.py --real`), which needs the weights + full stack and delegates
to `seed_demo.py`. Flags: `--no-serve` (seed only), `--keep` (don't wipe),
`--port N`.

The rest of this page is the **manual** walkthrough with the real model.

## 0. Prerequisites

- Python 3.11–3.13
- Trained weights at `runs/detect/traffic_model-2/weights/best.pt` (or set
  `MODEL_PATH`). Train with `train_traffic.py` if you don't have them.

## 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Start the application

```bash
python3 app.py            # http://127.0.0.1:5000
```

Open the URL. You land on the **Dashboard** tab (empty until you add data).

## 3. Seed a realistic demo set

In a second terminal (same venv):

```bash
python3 seed_demo.py
```

It runs the real detector on bundled sample photos to get genuine plates and
evidence images, then records a **curated** mix of violation types (a plate with
two unpaid fines, a triple-riding fine, an already-paid one) — hand-chosen for the
demo, not detected (a photo cannot show overspeed), so they sit in a session
labelled `[curated demo seed] not pipeline output`. It prints which plates to look
up (`MH02DL4596`, `MH12HS8818`).

## 4. Upload a photo — and watch it decline

**Photo** tab → choose an image → **Analyze**. The app runs detection + OCR
server-side, pairs each rider with the plate under *that* rider, and shows the
plate, what was recorded (with a confidence score), and the annotated evidence,
then looks the plate up below.

Try these bundled photos to see the decision layer, not just the detector:

| photo | what happens | why |
|---|---|---|
| `samples/plate4.jpeg` | no-helmet recorded against `MH02DL4596` | valid plate under the violating rider |
| `samples/plate9.jpeg` | plate read as `MHI2HS 8 8 18`, recorded as `MH12HS8818` | look-alike correction toward a valid plate structure |
| `samples/plate7.jpeg` | **seen but not fined** — plate text `~0294MAZXP` is not a valid registration | a malformed OCR read is never fined against |
| `samples/plate6.jpeg` | **seen but not fined** — contradiction | the model called the same rider helmeted *and* bare-headed |

A single photo has no temporal evidence, so its score is deliberately lower than
a video's and every record still waits for human review.

## 5. Upload a video

**Video** tab → choose a short clip → **Analyze**. Watch the live progress bar
(you can **Cancel** mid-run — the run is then recorded as `cancelled`, not
`completed`). When it finishes, the annotated video plays back with a summary:
frames, vehicles tracked, fines recorded, processing FPS, violations **confirmed
but withheld** because no plate was trustworthy (with the reason), and repeat
sightings of an already-fined plate that were not fined again.

## 6. Inspect a violation & its evidence

**Dashboard** tab → **Review** on any row. The detail modal shows:

- the evidence images (annotated / original / plate crop),
- the **confidence breakdown** (detection / temporal / association / OCR / final)
  — shown as scores, with a plain reminder that it's not a probability or proof,
- **how it was decided**: supporting frames vs the streak and helmet gate that
  applied, video time, vehicle track,
- the **plate vote**: agreement, margin over the runner-up reading, number of
  readings and disagreements,
- **provenance**: model, pipeline version, device — and a live **integrity
  check** that re-hashes the evidence files against the SHA-256 recorded when the
  violation was created (`GET /api/violations/<id>/verify`),
- the decoded registration region.

## 7. Review a fine record

In the same modal, click **Confirm** or **Dismiss** (or **Reset**). The row's
review badge and the "pending review" stat update immediately — this is the
human-in-the-loop step that separates a detection from a confirmed violation.

Filter the table by plate, type, review state, or payment status to find specific
records.

## 8. Look up a plate

**Plate** tab → type a plate (e.g. one `seed_demo.py` printed) → **Check**. You
get its fines, unpaid total, and the registration region decoded from the plate
itself.

---

### Command-line alternative

```bash
python3 main_ocr.py --image samples/plate8.jpeg          # single image → API
python3 main.py --source your_clip.mp4           # video (shared pipeline) → API
```

### Notes

- No weights, datasets, or videos ship in the repo (all gitignored) — use your
  own footage / model.
- Confidence numbers are honest CV confidence scores, not calibrated
  probabilities; treat flags as needing review.
