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
- real bundled plate photos as evidence (`MH02DL4596`, `MH12HS8818`, …).

The seeded data is **synthetic demo data**, clearly labelled as such on stdout —
the confidence scores and sessions are illustrative, not real detections. For a
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

It runs the real detector on bundled sample photos, records a curated mix (a
plate with two unpaid fines, a triple-riding fine, an already-paid one), and
prints exactly which plates to look up. Refresh the dashboard — the stat cards,
type breakdown, and violations table now have data.

## 4. Upload a photo

**Photo** tab → choose an image with a visible plate → **Analyze**. The app runs
detection + OCR server-side and shows the detected plate, any violations, and the
annotated evidence image, then auto-looks-up the plate's full record below. The
Dashboard updates to include the new violation.

## 5. Upload a video

**Video** tab → choose a short clip → **Analyze**. Watch the live progress bar
(you can **Cancel** mid-run). When it finishes, the annotated video plays back
with a summary (frames, vehicles tracked, fines recorded). Fines it confirmed now
appear on the Dashboard.

## 6. Inspect a violation & its evidence

**Dashboard** tab → **Review** on any row. The detail modal shows:

- the evidence images (annotated / original / plate crop),
- the **confidence breakdown** (detection / temporal / association / OCR / final)
  — shown as scores, with a plain reminder that it's not a probability or proof,
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
python3 main_ocr.py --image plate8.jpeg          # single image → API
python3 main.py --source your_clip.mp4           # video (shared pipeline) → API
```

### Notes

- No weights, datasets, or videos ship in the repo (all gitignored) — use your
  own footage / model.
- Confidence numbers are honest CV confidence scores, not calibrated
  probabilities; treat flags as needing review.
