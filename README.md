# Two Wheeler Safety — Traffic Violation Detection

An AI system that detects two-wheeler traffic violations (no-helmet, triple-riding,
speeding), reads the number plate via OCR, and records fines in a web-based
"Traffic Fine Checker" portal.

## Architecture

The project has three loosely-coupled parts:

### 1. Detection & OCR pipeline (offline / batch)
Both entry points use the same custom-trained YOLOv8 model
(`runs/detect/traffic_model-2/weights/best.pt`, classes `Plate`, `WithHelmet`,
`WithoutHelmet`, `TripleRiding`) rather than a generic pretrained model, so plate
OCR runs on a tight plate crop and violations come from the model's own trained
classes instead of hand-written heuristics.
- **`main_ocr.py`** — single-image pipeline. Detects violations, runs EasyOCR on the
  plate crop, saves a timestamped evidence image, and POSTs each violation to the
  Flask API.
- **`main.py`** — video pipeline. Tracks detected plates across frames with a small
  centroid tracker (`utils/tracker.py`) so each plate gets a stable ID, estimates
  per-vehicle speed, and associates `WithoutHelmet`/`TripleRiding` detections with
  the nearest tracked plate. That association is a best-effort nearest-neighbor
  heuristic — reliable with one vehicle in frame, but it can mis-attribute a
  violation when several vehicles are close together, since the model detects plate
  and violation boxes independently rather than as one linked vehicle instance. A
  violation must hold for `STREAK_THRESHOLD` (5) consecutive frames before it's
  trusted, and each (track, violation) pair is only reported once while that
  vehicle stays in view — both exist because single-frame model misfires are
  common (see "Known limitation" below).
  Runs headless by default (writes annotated output via `--output`, no window) —
  pass `--display` for a live preview if you have one. See `python main.py --help`.
- **`modules/plate_ocr.py`** — EasyOCR plate reader helper.
- **`modules/speed.py`** — `SpeedEstimator`, per-tracked-object frame-to-frame speed estimate.
- **`utils/tracker.py`** — `CentroidTracker`, a minimal greedy nearest-centroid multi-object tracker.
- **`plate_reader.py`**, `main_ocr` variants, and `test_*.py` — standalone detection/OCR scripts.

### 2. Model training
- **`train_traffic.py`** — trains YOLOv8n on `master_traffic_violation_dataset/` (4 classes).
- **`train_helmet.py`** — trains YOLOv8n on `data/` (helmet / no_helmet, 2 classes).
- Training outputs land in `runs/detect/<name>/` (metrics, curves, weights).

### 3. Web app — Traffic Fine Checker (Flask + SQLite)
- **`app.py`** — Flask server with a SQLite (`traffic.db`) `fines` table.
  - `GET  /` — renders the fine-checker UI (`templates/frontend.html`)
  - `POST /detect` — records a violation `{plate, violation, image_path}` and assigns a fine
    (no_helmet ₹500, triple_riding ₹1000, default ₹300). Plate matching is
    case/whitespace-insensitive on both insert and lookup. If `DETECT_API_KEY` is
    set, this endpoint requires a matching `X-API-Key` header (see below) — without
    it, anyone who can reach the server can write arbitrary fines for any plate.
  - `GET  /get_fines/<plate>` — returns fines + unpaid total for a plate
  - `GET  /fines` — lists all fines
  - `GET  /evidence/<file>` — serves saved evidence images
- **`templates/frontend.html`** — single-page UI to look up fines by plate.

### Data flow
```
image/video ──▶ YOLOv8 detection ──▶ EasyOCR (plate) ──▶ evidence image
                                                        └▶ POST /detect ──▶ SQLite (traffic.db)
                                                                                    ▲
                                            user looks up plate ──▶ GET /get_fines ─┘
```

## Tech stack
Python, Ultralytics YOLOv8, EasyOCR, OpenCV, Flask, SQLite, pandas.

## Running

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the web app:
```bash
python app.py          # http://127.0.0.1:5000
```

Run detection on an image (writes evidence + posts violations to the running app):
```bash
python main_ocr.py     # edit IMAGE_PATH / MODEL_PATH at the top
```

Run detection on a video:
```bash
python main.py --source your_video.mp4 --output annotated.mp4   # headless, saves an annotated copy
python main.py --source your_video.mp4 --display                # live preview window instead
python main.py --source your_video.mp4 --max-frames 300          # quick test on the first N frames
```
No video ships in the repo (`*.mp4`/`*.mov`/`*.avi`/`*.mkv` are gitignored — too large to track) —
point `--source` at your own footage, or a webcam index like `--source 0`.

### Known limitation: helmet-classification noise
Testing against real video surfaced a real accuracy problem in the trained model
weights themselves (`traffic_model-2`), not a code bug: on footage stylistically
different from its training set, it can flicker between `WithHelmet` and
`WithoutHelmet` for the same continuously-helmeted rider frame to frame, and the
`WithoutHelmet` box is sometimes poorly localized (covering the whole rider instead
of just the head). The `STREAK_THRESHOLD` consecutive-frame check in `main.py`
filters out most single-frame flicker, but it doesn't fix misclassifications the
model holds steadily for several frames — only retraining with more/better-annotated
data would. Treat `no_helmet` fines from unfamiliar footage as needing review, not
as ground truth.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | Port `app.py` listens on. |
| `FLASK_DEBUG` | `1` (on) | Set to `0` before any real deployment — the Werkzeug debugger allows arbitrary code execution if the server is exposed with it on. |
| `DETECT_API_KEY` | unset (no auth) | If set, `/detect` requires this value in an `X-API-Key` header. `main.py` and `main_ocr.py` read the same variable and send it automatically. |

## Not tracked in git
Datasets, trained model weights (`*.pt`), `runs/` training outputs, the source
`.zip`, `traffic.db`, and generated `evidence/` are excluded via `.gitignore`
(they are large and/or regenerable). Retrain with `train_traffic.py` /
`train_helmet.py`, and let Ultralytics download `yolov8n.pt` on first run.
