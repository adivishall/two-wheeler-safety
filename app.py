import os
import tempfile
import threading
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from modules.db import Database
from modules.validation import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    RateLimiter,
    file_extension,
    is_allowed_extension,
    safe_evidence_name,
    safe_extension,
    sniff_image,
    validate_plate,
    validate_violation,
)

app = Flask(__name__)

EVIDENCE_DIR = os.path.join(app.root_path, "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# --- Upload / rate-limit config (Phase 9) ---
# Global hard cap on request size (Flask returns 413 above it). Generous enough
# for a video upload; images are far smaller.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Simple in-process per-IP rate limit on the write/upload endpoints.
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))
_rate_limiter = RateLimiter(max_requests=RATE_LIMIT_PER_MIN, window_s=60.0)


def _rate_limited() -> bool:
    return not _rate_limiter.allow(request.remote_addr or "unknown")

# If set, /detect requires this value in the X-API-Key header — without
# it, anyone who can reach the server can write arbitrary fines for any
# plate. Unset by default so local dev/demo usage is unaffected.
DETECT_API_KEY = os.environ.get("DETECT_API_KEY")

DB_PATH = os.environ.get("TRAFFIC_DB_PATH", "traffic.db")

# Weights the /analyze upload route runs detection with. Same default as
# the CLI (main_ocr.py); override with MODEL_PATH for a different model.
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "runs/detect/traffic_model-2/weights/best.pt"
)

# Normalized SQLite store (vehicles / violations / evidence / jobs). Creating it
# initializes the schema and migrates any legacy flat `fines` table in place.
db = Database(DB_PATH)

# YOLO + EasyOCR are heavy to load (a few seconds) and not needed unless
# someone actually uploads a photo, so they're loaded once on the first
# /analyze call and cached. The lock keeps two simultaneous first-uploads
# from loading the models twice.
_models = None
_models_lock = threading.Lock()


def get_models():
    global _models
    if _models is None:
        with _models_lock:
            if _models is None:
                from modules.detector import load_models
                _models = load_models(MODEL_PATH)
    return _models


def record_fine(plate, violation, image_path):
    """Insert one fine and return the amount charged. Kept as a thin wrapper so
    the CLIs and video pipeline can pass it as a ``record_fn`` callback."""
    return db.record_fine(plate, violation, image_path)

# =====================================
# HOME PAGE
# =====================================

@app.route("/")
def home():
    return render_template("frontend.html")

# =====================================
# SERVE EVIDENCE IMAGES
# =====================================

@app.route('/evidence/<path:filename>')
def evidence(filename):
    # Only ever serve a plain basename with an allowed media extension from the
    # evidence directory — no subpaths, no traversal. (send_from_directory is
    # itself traversal-safe; this is defense in depth + a media-type allowlist.)
    name = os.path.basename(filename)
    if name != filename or file_extension(name) not in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS):
        abort(404)
    return send_from_directory(EVIDENCE_DIR, name)


@app.route("/health")
def health():
    """Liveness/readiness without exposing secrets: reports whether the models
    are loaded (they load lazily on first upload)."""
    return jsonify({
        "status": "ok",
        "models_loaded": _models is not None,
        "debug": app.debug,
    })

# =====================================
# SAVE DETECTED VIOLATION
# =====================================

@app.route("/detect", methods=["POST"])
def detect():

    if DETECT_API_KEY and request.headers.get("X-API-Key") != DETECT_API_KEY:
        return jsonify({"error": "invalid or missing API key"}), 401

    if _rate_limited():
        return jsonify({"error": "rate limit exceeded"}), 429

    data = request.get_json(silent=True) or {}

    if not all(k in data for k in ("plate", "violation", "image_path")):
        return jsonify({
            "error": "plate, violation, and image_path are required"
        }), 400

    plate = validate_plate(data["plate"])
    if plate is None:
        return jsonify({"error": "invalid plate"}), 400
    if not validate_violation(data["violation"]):
        return jsonify({"error": "invalid violation type"}), 400

    # Never trust a caller-supplied path: reduce it to a safe evidence basename
    # (no directory, must be an allowed image type) rather than storing it raw.
    image_name = safe_evidence_name(data["image_path"])
    if data["image_path"] and image_name is None:
        return jsonify({"error": "invalid image_path"}), 400
    stored_path = f"evidence/{image_name}" if image_name else None

    record_fine(plate, data["violation"], stored_path)

    return jsonify({
        "message": "Violation Recorded"
    })

# =====================================
# ANALYZE AN UPLOADED PHOTO (one-window demo)
# =====================================

@app.route("/analyze", methods=["POST"])
def analyze():
    """Run detection + OCR on an uploaded image, record any violations, and
    return what was found so the web UI can show it without the terminal."""

    from modules.detector import analyze_image

    if _rate_limited():
        return jsonify({"error": "rate limit exceeded"}), 429

    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "no image uploaded"}), 400

    upload = request.files["image"]
    if not is_allowed_extension(upload.filename, IMAGE_EXTENSIONS):
        return jsonify({"error": "unsupported image type"}), 400

    # Server-controlled temp name (never the user's filename); validate the file
    # is really an image by its magic bytes before handing it to OpenCV.
    suffix = safe_extension(upload.filename, IMAGE_EXTENSIONS, ".jpg")
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        upload.save(tmp_path)

        with open(tmp_path, "rb") as fh:
            if not sniff_image(fh.read(16)):
                return jsonify({"error": "file is not a valid image"}), 400

        model, reader = get_models()
        result = analyze_image(tmp_path, model, reader)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if result.get("error"):
        return jsonify({"error": result["error"]}), 400

    plate = result["plate"]

    if not plate:
        return jsonify({
            "plate": None,
            "message": "Couldn't read a number plate in that photo. Try a clearer image."
        })

    evidence_url = None
    if result["evidence_file"]:
        evidence_url = "/evidence/" + os.path.basename(result["evidence_file"])

    recorded = []
    for violation in result["violations"]:
        amount = record_fine(plate, violation, result["evidence_file"])
        recorded.append({"type": violation, "amount": amount})

    return jsonify({
        "plate": plate,
        "evidence": evidence_url,
        "violations": recorded,
    })

# =====================================
# ANALYZE AN UPLOADED VIDEO (background job + progress)
# =====================================

# job_id -> {status, done, total, result, error}. Video processing takes far
# longer than one request, so /analyze_video kicks off a background thread and
# the browser polls /video_status for progress.
_video_jobs = {}
_video_jobs_lock = threading.Lock()


def _set_job(job_id, **fields):
    with _video_jobs_lock:
        _video_jobs.setdefault(job_id, {}).update(fields)


def _run_video_job(job_id, video_path):
    from modules.video_detector import process_video

    try:
        model, reader = get_models()
        output_path = os.path.join("evidence", f"annotated_{job_id}.mp4")

        def progress(done, total):
            _set_job(job_id, done=done, total=total)

        # pixels_per_meter left unset: speed/overspeed needs calibration, so
        # it's skipped rather than reporting a meaningless number (same as CLI).
        result = process_video(
            video_path, model, reader, output_path,
            record_fn=record_fine, progress_cb=progress,
        )
        _set_job(job_id, status="done", result=result)
    except Exception as exc:  # surface the failure to the poller
        _set_job(job_id, status="error", error=str(exc))
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    if _rate_limited():
        return jsonify({"error": "rate limit exceeded"}), 429

    if "video" not in request.files or request.files["video"].filename == "":
        return jsonify({"error": "no video uploaded"}), 400

    upload = request.files["video"]
    if not is_allowed_extension(upload.filename, VIDEO_EXTENSIONS):
        return jsonify({"error": "unsupported video type"}), 400

    suffix = safe_extension(upload.filename, VIDEO_EXTENSIONS, ".mp4")
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    upload.save(tmp_path)

    job_id = uuid.uuid4().hex
    _set_job(job_id, status="processing", done=0, total=0, result=None, error=None)

    threading.Thread(
        target=_run_video_job, args=(job_id, tmp_path), daemon=True
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/video_status/<job_id>")
def video_status(job_id):
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
        job = dict(job) if job else None
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)

# =====================================
# GET FINES BY PLATE
# =====================================

@app.route("/get_fines/<plate>")
def get_fines(plate):

    from modules.plate_info import decode_plate

    result = db.get_fines(plate)
    result["vehicle"] = decode_plate(plate)
    return jsonify(result)

# =====================================
# SHOW ALL FINES
# =====================================

@app.route("/fines")
def all_fines():
    return jsonify(db.all_fines())

# =====================================
# RUN
# =====================================

if __name__ == "__main__":
    # Debug is OFF by default now (the Werkzeug debugger allows arbitrary code
    # execution if the server is exposed). Opt in explicitly with FLASK_DEBUG=1
    # for local development only.
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(debug=debug_mode, host=host, port=port)