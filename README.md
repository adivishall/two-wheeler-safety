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
- **`modules/detector.py`** — the shared single-image pipeline (`analyze_image`):
  runs the model, OCRs the plate crop, applies the helmet-contradiction
  suppression, draws labeled detection boxes onto the evidence image, and saves
  that annotated copy. Both `main_ocr.py` and the web app's `/analyze` upload route
  call it, so they can't drift apart.
- **`main_ocr.py`** — single-image CLI. Wraps `modules/detector.analyze_image` and
  POSTs each detected violation to the Flask API.
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
- **`train_traffic.py`** — trains YOLOv8n on `master_traffic_violation_dataset/` (4 classes:
  `Plate`, `WithHelmet`, `WithoutHelmet`, `TripleRiding`).
- Training outputs land in `runs/detect/<name>/` (metrics, curves, weights).
- An earlier 2-class helmet/no_helmet model (`train_helmet.py`, its own `data/`
  dataset, `test_model.py`) was retired — nothing in the live app used it once the
  4-class model above covered the same ground plus plate detection and triple-riding.

### 3. Web app — Traffic Fine Checker (Flask + SQLite)
- **`app.py`** — Flask server with a SQLite (`traffic.db`) `fines` table.
  - `GET  /` — renders the fine-checker UI (`templates/frontend.html`)
  - `POST /analyze` — accepts an uploaded photo (multipart `image`), runs the same
    detection pipeline server-side, records any violations, and returns the detected
    plate + violations + annotated evidence. This is what powers the in-browser
    "upload a photo" flow, so the whole demo can happen in one window with no
    terminal. The model + OCR reader load lazily on the first upload (a few seconds),
    then stay cached.
  - `POST /detect` — records a violation `{plate, violation, image_path}` and assigns a fine
    (no_helmet ₹500, triple_riding ₹1000, overspeed ₹700, default ₹300). Plate matching is
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
Python, Ultralytics YOLOv8, EasyOCR, OpenCV, Flask, SQLite.

## Running

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the web app:
```bash
python app.py          # http://127.0.0.1:5000
```

The web UI has two ways in: **upload a photo** (it runs detection server-side and
shows the detected plate, violations, and annotated evidence, then looks the plate
up automatically), or **type a plate** to look it up directly. The upload path keeps
the whole demo in one browser window — no terminal, and no retyping an OCR-mangled
plate.

Pre-seed a few realistic records for a demo (a plate with two unpaid fines, a
triple-riding fine, and an already-paid one). It prints exactly which plates to look
up:
```bash
python seed_demo.py          # wipes existing fines, then seeds
python seed_demo.py --keep   # keeps existing fines, just adds the demo set
```

Run detection on an image from the command line instead (writes evidence + posts
violations to the running app):
```bash
python main_ocr.py --image your_photo.jpg                                    # defaults otherwise
python main_ocr.py --image your_photo.jpg --model path/to/best.pt --api-url http://127.0.0.1:5000/detect
```

Run detection on a video:
```bash
python main.py --source your_video.mp4 --output annotated.mp4   # headless, saves an annotated copy
python main.py --source your_video.mp4 --display                # live preview window instead
python main.py --source your_video.mp4 --max-frames 300          # quick test on the first N frames
python main.py --source your_video.mp4 --pixels-per-meter 68     # enables speed/overspeed detection
```
No video ships in the repo (`*.mp4`/`*.mov`/`*.avi`/`*.mkv` are gitignored — too large to track) —
point `--source` at your own footage, or a webcam index like `--source 0`.

### Known limitations

**Helmet-classification noise.** Testing against real video and real sample images
surfaced accuracy problems in the trained model weights themselves
(`traffic_model-2`), not code bugs:
- On footage stylistically different from its training set, it can flicker between
  `WithHelmet` and `WithoutHelmet` for the same continuously-helmeted rider frame to
  frame, and the `WithoutHelmet` box is sometimes poorly localized (covering the
  whole rider instead of just the head).
- When the model outputs *both* classes for the same rider (a same-frame
  contradiction), the higher-confidence one isn't necessarily correct — verified
  against a real photo where `WithoutHelmet` (0.60 confidence) was wrong and
  `WithHelmet` (0.39 confidence) was right. `main.py`/`main_ocr.py` now detect this
  overlap and report neither rather than guess (see `is_contradicted`/`iou`).
- That fix only catches contradictions. A single, confidently-wrong prediction with
  nothing to contradict it (a helmeted rider called `WithoutHelmet` alone at 0.887
  confidence, tested on a real sample image) isn't caught by any code-level check —
  fixing that needs retraining on more/better-annotated data, which is a separate,
  ongoing effort. The `STREAK_THRESHOLD` consecutive-frame check in `main.py`
  filters single-frame flicker but not a misclassification the model holds
  steadily. Treat `no_helmet` fines from unfamiliar footage as needing review, not
  as ground truth.

**Speed/overspeed requires camera calibration.** `SpeedEstimator` converts pixel
displacement to km/h using `pixels_per_meter` — there is no default, and `main.py`
skips speed estimation entirely if `--pixels-per-meter` isn't given, rather than
report a number with no real physical meaning. To calibrate: mark two points in
your camera's frame that are a known real-world distance apart (e.g. two road
markings), measure the pixel distance between them in a frame, and compute
`pixel_distance / real_world_meters` (see `modules/speed.calibrate_pixels_per_meter`).
This also assumes motion roughly perpendicular to the camera — a vehicle moving
toward/away from the camera will read artificially slow.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | Port `app.py` listens on. |
| `FLASK_DEBUG` | `1` (on) | Set to `0` before any real deployment — the Werkzeug debugger allows arbitrary code execution if the server is exposed with it on. |
| `DETECT_API_KEY` | unset (no auth) | If set, `/detect` requires this value in an `X-API-Key` header. `main.py` and `main_ocr.py` read the same variable and send it automatically. |
| `TRAFFIC_DB_PATH` | `traffic.db` | SQLite file `app.py` reads/writes. Override for a different deployment path, or to point tests at a throwaway DB instead of the real one. |
| `MODEL_PATH` | `runs/detect/traffic_model-2/weights/best.pt` | YOLO weights the `/analyze` upload route (and `seed_demo.py`) run detection with. |

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Covers `CentroidTracker`, `SpeedEstimator`/calibration, the pure detection-logic
helpers in `main.py` (`iou`, `is_contradicted`, `nearest_plate_id`, `clean_plate`)
— including regression tests built from the real photos that exposed the
helmet-contradiction bug — and `app.py`'s Flask routes (plate normalization, fine
amounts, `DETECT_API_KEY` gating, response shapes) against a temporary SQLite DB,
never the real `traffic.db`. Model inference itself (does the trained model detect
correctly) isn't covered here — that's evaluated via the training run's own
validation metrics and confusion matrix, not unit tests.

## Not tracked in git
Datasets, trained model weights (`*.pt`), `runs/` training outputs, the source
`.zip`, `traffic.db`, and generated `evidence/` are excluded via `.gitignore`
(they are large and/or regenerable). Retrain with `train_traffic.py`, and let
Ultralytics download `yolov8n.pt` on first run.
