# Baseline Audit

Snapshot of the repository **before** the `feature/two-wheeler-safety-upgrade`
work began. Every number here is measured on the machine described below, not
estimated. Later phases are graded against this document.

## Environment

| Item | Value |
|---|---|
| Date | 2026-09-08 |
| Platform | macOS (Darwin 25.5.0), Apple silicon |
| Python | 3.13.7 |
| Inference device | CPU (no CUDA; MPS not used by the pipeline) |
| flask | 3.1.3 |
| opencv-python (cv2) | 4.13.0 |
| numpy | 2.4.6 |
| ultralytics | 8.4.71 |
| easyocr | 1.7.2 |
| torch | 2.12.1 |
| pytest | 9.1.1 |
| ruff | installed |
| black | installed |
| Model weights | `runs/detect/traffic_model-2/weights/best.pt` present locally (6.2 MB), git-ignored |

## Repository shape

Tracked Python: **2116 lines** across the files below. Model weights, datasets,
`runs/`, `traffic.db`, `evidence/`, and all video/zip files are git-ignored.

Absent scaffolding at baseline: `docs/`, `config.py`, `.env.example`,
`Dockerfile`, `.github/` (no CI), `pyproject.toml`, `conftest.py`.

### Code map (what each file is and whether it's live)

| File | Role | Live? |
|---|---|---|
| `app.py` | Flask server: routes, SQLite `fines` table, lazy model cache, in-memory video jobs | **live** |
| `modules/detector.py` | Shared single-image detect + OCR + annotate (`analyze_image`, `load_models`, `iou`, `clean_plate`) | **live** |
| `modules/video_detector.py` | `process_video()` — per-call video pipeline the `/analyze_video` route runs | **live** |
| `modules/plate_ocr.py` | `read_plate()` used by `main.py`; **module-level** `easyocr.Reader(['en'])` at import | **live (CLI)** |
| `modules/plate_info.py` | Decode plate → state / RTO district / BH-series | **live** |
| `modules/speed.py` | `SpeedEstimator` (frame-to-frame pixel displacement), `calibrate_pixels_per_meter` | **live** |
| `utils/tracker.py` | `CentroidTracker` — greedy nearest-centroid multi-object tracker | **live** |
| `main.py` | CLI video pipeline (own copy of association/streak logic) | **live (CLI)** |
| `main_ocr.py` | CLI single-image pipeline | **live (CLI)** |
| `seed_demo.py` | Seeds `traffic.db` with demo records from real sample photos | **live (demo)** |
| `train_traffic.py` | YOLOv8n training entry point | tooling |
| `plate_reader.py` | One-off detect+OCR probe on `test.jpg` | **dead / experimental** |
| `ocr_test.py` | 6-line EasyOCR probe on `plate3.jpeg` | **dead / experimental** |
| `test_traffic.py` | Root-level probe: `model.predict(..., save=True)` at import time | **dead / footgun** (see below) |
| `_helmet_fix_train.py`, `_helmet_fix_compare.py`, `_retrain_probe.py` | Untracked local retraining experiments | untracked |

## Code paths actually used (traced)

- **Image upload** (`POST /analyze`): `app.get_models()` → `modules.detector.analyze_image` → `record_fine` → SQLite. Model + OCR loaded lazily on first call, cached under a lock.
- **Video upload** (`POST /analyze_video`): spawns a daemon `threading.Thread` running `_run_video_job` → `modules.video_detector.process_video` with `record_fn=record_fine`; browser polls `GET /video_status/<job_id>`.
- **CLI image** (`main_ocr.py`): `load_models` → `analyze_image` → HTTP `POST /detect`.
- **CLI video** (`main.py`): its **own** loop (does not import `video_detector`) — `YOLO`, `CentroidTracker`, `SpeedEstimator`, `nearest_plate_id`, streak logic → HTTP `POST /detect`.
- **DB recording**: `record_fine(plate, violation, image_path)` in `app.py` — one `INSERT` into `fines`; a fresh `sqlite3.connect` per call.
- **Fine lookup** (`GET /get_fines/<plate>`): reads `fines` by normalized plate + `modules.plate_info.decode_plate`.

## Build / run / test results (measured)

### Tests — `python3 -m pytest tests/ -q`
```
46 passed in 3.58s
```
All 46 pass. Coverage: `CentroidTracker` lifecycle, `SpeedEstimator` math/calibration,
`main.py` pure helpers (`iou`, `is_contradicted`, `nearest_plate_id`, `clean_plate`,
including regression cases from real photos), `plate_info.decode_plate`, and the Flask
routes (`/detect` auth + validation, fine amounts, plate normalization, response shapes)
against a temp SQLite DB.

### Lint — `ruff check .`
**15 findings**, almost all cosmetic import-ordering (`I001`) across `app.py`,
`main.py`, `main_ocr.py`, `seed_demo.py`, `modules/*`, `tests/test_app.py`; a couple
of unused/other minor items. No formatter config committed (`ruff`/`black` present but
unconfigured). No `pyproject.toml`.

### Single-image pipeline (measured, CPU)
`analyze_image('plate4.jpeg')`:
- `load_models`: **2.72 s** (one-time YOLO + EasyOCR load)
- `analyze_image`: **0.32 s**
- Result: plate `MH02DL4596`, violations `['no_helmet']`, 2 detections (`Plate`, `WithoutHelmet`), annotated evidence written.

### Video pipeline (measured, CPU)
`process_video('demo_traffic.mp4', max_frames=40)`:
- Wall: **2.37 s** for 40 frames → **~16.9 processing FPS** (CPU)
- Source FPS: 25.0; 1 plate tracked; 1 violation recorded.

### Flask
Routes exercised via `app.test_client()` in `tests/test_app.py` (all pass). There is
**no** `/health` / readiness endpoint at baseline. `FLASK_DEBUG` defaults to **on**
(`1`) — the Werkzeug debugger allows RCE if exposed; documented in README but on by default.

### Existing database
`traffic.db` (git-ignored, present locally): single table.
```sql
CREATE TABLE fines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT, violation TEXT, amount INTEGER, image_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'unpaid'
);
```
4 rows present. No indexes beyond the implicit primary key.

## Known weaknesses identified (targets for phases 1–10)

1. **Vehicle association is nearest-centroid only.** `nearest_plate_id` (in both `main.py` and `video_detector.py`) attributes a `WithoutHelmet`/`TripleRiding` box to the geometrically nearest tracked *plate*. There is no vehicle object linking plate + rider + violation; with several bikes close together it can mis-attribute. No IoU/overlap, no gating, no one-to-one constraint (one violation box can map to a plate that already owns another). **[Phase 1]**
2. **OCR trusts a single frame.** `read_plate`/`analyze_image` take one OCR pass per plate crop and `clean_plate` it. In video, `plate_text_by_id[track_id]` is overwritten every frame — the *last* frame's reading wins; a fine can be issued from one noisy frame. No temporal voting, no confidence aggregation, no format validation before storing. **[Phase 2]**
3. **No unified confidence model.** YOLO `conf` is available in `analyze_image` detections but is discarded for the fine decision; the recorded fine carries no confidence at all. The DB has no confidence column. **[Phase 3]**
4. **Helmet logic is stateless per frame** apart from the streak counter. The contradiction suppression (overlapping `WithHelmet`/`WithoutHelmet` ⇒ trust neither) is preserved and correct, but there are no helmet states (unknown / candidate / confirmed / ambiguous) carried on a vehicle. **[Phase 4]**
5. **Triple-riding uses the same generic streak** as other violations (threshold 5) with no dedicated candidate/confirmed/cleared model or evidence-frame selection. **[Phase 5]**
6. **Speed uses wall-clock time.** `SpeedEstimator.calculate_speed` uses `time.time()` between calls — a slower CPU makes vehicles appear *slower/faster* depending on processing rate, not video time. No smoothing, no impossible-jump rejection, no minimum track duration, no uncertainty. **[Phase 6]**
7. **Evidence identity relies on `time.time()`** (seconds) in `main.py`/`analyze_image` and milliseconds in `video_detector.py`. Collisions possible; filenames embed the plate but there is no structured metadata sidecar, no original+annotated+crop package. **[Phase 7]**
8. **Single denormalized `fines` table**, one connection per call, no transactions spanning related writes, no indexes, no migration path. **[Phase 8]**
9. **Weak input validation / security.** `/detect` accepts an arbitrary caller-supplied `image_path` string and stores it (later served via `/evidence/<path:filename>` — path-traversal surface). No upload size limit, no extension/MIME validation, uploads saved with `os.path.splitext(upload.filename)[1]` suffix from user input. `FLASK_DEBUG` on by default. No `/health`. **[Phase 9]**
10. **Video jobs live in a process-global dict** (`_video_jobs`) that grows unbounded (never cleaned up/expired), with no concurrency cap, no cancellation, no max-size/duration guard. **[Phase 10]**

### Footgun: bare `pytest`
`test_traffic.py` at the repo root matches pytest's `test_*.py` collection glob and
runs `YOLO(...).predict(source="plate10.jpeg", save=True)` **at import time**. Running
`pytest` with no path argument (instead of `pytest tests/`) would load the model and
run inference — writing into `runs/` — during collection. The project's documented
command is `pytest tests/`, which scopes collection and avoids this. Flagged for
cleanup.

## What is intentionally correct and must be preserved

- **Helmet-contradiction suppression** (overlapping `WithHelmet`/`WithoutHelmet` ⇒ report neither) — verified against a real photo where the higher-confidence box was wrong.
- **Streak confirmation** before recording a violation (filters single-frame flicker).
- **Speed disabled without calibration** — no `--pixels-per-meter` ⇒ no speed number, rather than a physically meaningless one.
- **Plate normalization** (case/whitespace-insensitive) on both insert and lookup.
- **Registration decode is offline & honest** — state/RTO only, never claims owner/maker.
- **Lazy, cached, lock-guarded model loading** in the web app.
