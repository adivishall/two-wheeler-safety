# Two Wheeler Safety — Traffic Violation Detection

An AI system that detects two-wheeler traffic violations (no-helmet, triple-riding,
speeding), reads the number plate via OCR, and records fines in a web-based
"Traffic Fine Checker" portal.

## Architecture

The project has three loosely-coupled parts:

### 1. Detection & OCR pipeline (offline / batch)
- **`main_ocr.py`** — the primary pipeline. Runs a custom-trained YOLOv8 model on an
  image, detects `Plate`, `WithHelmet`, `WithoutHelmet`, `TripleRiding`, runs EasyOCR
  on the plate crop, saves an evidence image, and POSTs each violation to the Flask API.
- **`main.py`** — video pipeline. Runs YOLO frame-by-frame for triple-riding (person/bike
  overlap heuristic), speed estimation, and live plate OCR overlays.
- **`modules/plate_ocr.py`** — EasyOCR plate reader helper.
- **`modules/speed.py`** — `SpeedEstimator`, frame-to-frame pixel-displacement speed estimate.
- **`plate_reader.py`**, `main_ocr` variants, and `test_*.py` — standalone detection/OCR scripts.

### 2. Model training
- **`train_traffic.py`** — trains YOLOv8n on `master_traffic_violation_dataset/` (4 classes).
- **`train_helmet.py`** — trains YOLOv8n on `data/` (helmet / no_helmet, 2 classes).
- Training outputs land in `runs/detect/<name>/` (metrics, curves, weights).

### 3. Web app — Traffic Fine Checker (Flask + SQLite)
- **`app.py`** — Flask server with a SQLite (`traffic.db`) `fines` table.
  - `GET  /` — renders the fine-checker UI (`templates/frontend.html`)
  - `POST /detect` — records a violation `{plate, violation, image_path}` and assigns a fine
    (no_helmet ₹500, triple_riding ₹1000, default ₹300)
  - `GET  /get_fines/<plate>` — returns fines + unpaid total for a plate
  - `GET  /fines` — dumps all fines
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

## Not tracked in git
Datasets, trained model weights (`*.pt`), `runs/` training outputs, the source
`.zip`, `traffic.db`, and generated `evidence/` are excluded via `.gitignore`
(they are large and/or regenerable). Retrain with `train_traffic.py` /
`train_helmet.py`, and let Ultralytics download `yolov8n.pt` on first run.
