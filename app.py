import os
import tempfile
import threading
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from modules.config import PIPELINE_VERSION, load_config
from modules.db import Database
from modules.jobs import JobManager
from modules.logging_setup import configure_logging, get_logger
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

# All runtime knobs come from one typed config built from the environment.
# Read here (at import) so a test that patches the env and re-imports app.py
# still picks up fresh values, exactly as the old inline os.environ reads did.
config = load_config()
configure_logging(config.log_level)
log = get_logger("app")

app = Flask(__name__)

EVIDENCE_DIR = os.path.join(app.root_path, config.evidence_dir)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Global hard cap on request size (Flask returns 413 above it). Generous enough
# for a video upload; images are far smaller.
app.config["MAX_CONTENT_LENGTH"] = config.server.max_upload_mb * 1024 * 1024

# Simple in-process per-IP rate limit on the write/upload endpoints.
_rate_limiter = RateLimiter(max_requests=config.server.rate_limit_per_min, window_s=60.0)


def _rate_limited() -> bool:
    return not _rate_limiter.allow(request.remote_addr or "unknown")

# If set, /detect requires this value in the X-API-Key header — without
# it, anyone who can reach the server can write arbitrary fines for any
# plate. Unset by default so local dev/demo usage is unaffected.
DETECT_API_KEY = config.server.detect_api_key

# Role-based access (Phase 16). viewer < reviewer < admin. When no role keys are
# configured the app is open (unchanged local/demo behaviour); once configured,
# mutation endpoints require at least the reviewer role so review data is not
# publicly writable in a deployment.
_ROLE_LEVELS = {"viewer": 1, "reviewer": 2, "admin": 3}


def _current_actor():
    """Return (role, actor_label) for the request's API key.

    When auth is disabled, every request is treated as an anonymous admin so
    local/demo usage keeps working. When enabled, the role comes from the key
    (or None if the key is missing/unknown)."""
    if not config.server.auth_enabled:
        return "admin", "anonymous"
    role = config.server.role_for_key(request.headers.get("X-API-Key"))
    return role, (role or "unknown")


def _require_role(min_role: str):
    """Return None if the caller meets ``min_role``, else a Flask (json, code)
    error response to return from the endpoint."""
    role, _actor = _current_actor()
    if role is None or _ROLE_LEVELS[role] < _ROLE_LEVELS[min_role]:
        return jsonify({"error": f"{min_role} role required"}), 403
    return None

DB_PATH = config.db_path

# Weights the /analyze upload route runs detection with; override with MODEL_PATH.
MODEL_PATH = config.model_path

# Best-effort model version (name@version) resolved from the checkpoint's
# manifest (modules/model_manifest.py). Stamped onto every evidence package so a
# fine is traceable to the exact weights that raised it; None if no manifest.
from modules.model_manifest import model_version_string, resolve_manifest  # noqa: E402

MODEL_VERSION = model_version_string(resolve_manifest(MODEL_PATH))
if MODEL_VERSION:
    log.info("detection model version: %s", MODEL_VERSION)

# Normalized SQLite store (vehicles / violations / evidence / jobs). Creating it
# initializes the schema and migrates any legacy flat `fines` table in place.
db = Database(DB_PATH)
log.info("database ready at %s", DB_PATH)

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
                log.info("loading YOLO + OCR models from %s", MODEL_PATH)
                _models = load_models(MODEL_PATH)
                log.info("models loaded")
    return _models


def record_fine(plate, violation, image_path, **extra):
    """Insert one fine and return the amount charged. Kept as a thin wrapper so
    the CLIs and video pipeline can pass it as a ``record_fn`` callback. Accepts
    keyword-only extras (confidence, track_id, session_id, detection_trace) that
    the video pipeline supplies; the photo/detect routes call it with three
    positional args and no extras, unchanged."""
    return db.record_fine(
        plate, violation, image_path,
        confidence=extra.get("confidence"),
        track_id=extra.get("track_id"),
        session_id=extra.get("session_id"),
        detection_trace=extra.get("detection_trace"),
    )

@app.after_request
def _security_headers(response):
    """Baseline hardening headers on every response. Kept modest for a
    single-page same-origin app: block MIME sniffing, disallow framing
    (clickjacking), and don't leak referrers. No strict CSP because the
    dashboard intentionally uses inline styles/scripts."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response

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

# Video processing takes far longer than one request, so /analyze_video hands
# the work to an in-process JobManager (bounded concurrency, cancellation,
# expiry) and the browser polls /video_status for progress.
MAX_VIDEO_MB = config.server.max_video_mb
MAX_VIDEO_SECONDS = config.server.max_video_seconds

job_manager = JobManager(
    max_concurrent=config.server.max_concurrent_video_jobs,
    max_age_s=config.server.job_max_age_s,
)


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
    source_id = os.path.basename(upload.filename or "upload")
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    upload.save(tmp_path)

    if os.path.getsize(tmp_path) > MAX_VIDEO_MB * 1024 * 1024:
        os.remove(tmp_path)
        return jsonify({"error": f"video exceeds {MAX_VIDEO_MB} MB limit"}), 400

    out_name = f"annotated_{uuid.uuid4().hex}.mp4"

    # A processing session (Phase 14) + durable job record (Phase 15) tie this
    # run to the model, config, metrics, and violations it produces. Keyed by a
    # session id so the record survives independently of the in-process worker.
    session_id = uuid.uuid4().hex
    db.create_session(session_id, source=source_id, model_version=MODEL_VERSION,
                      pipeline_version=PIPELINE_VERSION)
    db.create_job(session_id, "video", source=source_id)

    def target(progress_cb, cancel_check):
        from datetime import datetime, timezone

        from modules.video_detector import process_video
        try:
            model, reader = get_models()
            # pixels_per_meter unset: speed/overspeed needs calibration, so it's
            # skipped rather than reporting a meaningless number (same as CLI).
            summary = process_video(
                tmp_path, model, reader, os.path.join(EVIDENCE_DIR, out_name),
                record_fn=record_fine, progress_cb=progress_cb,
                cancel_check=cancel_check, max_seconds=MAX_VIDEO_SECONDS,
                model_version=MODEL_VERSION,
                pipeline_version=PIPELINE_VERSION,
                source_id=source_id,
                config_snapshot={
                    "conf_threshold": config.detection.conf_threshold,
                    "streak_threshold": config.detection.streak_threshold,
                    "speed_limit_kmh": config.detection.speed_limit_kmh,
                },
                session_id=session_id,
                trace_enabled=config.detection.trace_enabled,
                trace_max_frames=config.detection.trace_max_frames,
                ocr_lock_confidence=config.detection.ocr_lock_confidence,
                ocr_lock_min_observations=config.detection.ocr_lock_min_observations,
            )
            now = datetime.now(timezone.utc).isoformat()
            db.update_session(
                session_id, ended_at=now, status="completed",
                frames_processed=summary.get("frames", 0),
                vehicles_tracked=summary.get("plates_tracked", 0),
                violations_detected=len(summary.get("violations", [])),
                processing_fps=summary.get("fps"),
                output_path=summary.get("output"),
            )
            db.update_job(session_id, status="completed", progress=1.0,
                          completed_at=now, output=summary.get("output"))
            return summary
        except Exception as exc:  # record the failure durably, then re-raise
            db.update_session(session_id, status="failed", error=str(exc))
            db.update_job(session_id, status="failed", error=str(exc))
            raise
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    job_id = job_manager.submit("video", target)
    if job_id is None:
        os.remove(tmp_path)
        db.update_session(session_id, status="rejected",
                          error="concurrency cap reached")
        db.update_job(session_id, status="rejected")
        log.warning("video job rejected: concurrency cap reached")
        return jsonify({"error": "server busy: too many concurrent video jobs"}), 503

    log.info("video job %s accepted (session %s)", job_id, session_id)
    return jsonify({"job_id": job_id, "session_id": session_id})


@app.route("/video_status/<job_id>")
def video_status(job_id):
    job = job_manager.status(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)


@app.route("/video_cancel/<job_id>", methods=["POST"])
def video_cancel(job_id):
    if job_manager.cancel(job_id):
        return jsonify({"message": "cancellation requested"})
    return jsonify({"error": "job not found or not cancellable"}), 404

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
# DASHBOARD ANALYTICS + FILTERED LISTING (JSON API)
# =====================================

def _arg_float(name):
    raw = request.args.get(name)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _arg_int(name, default):
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@app.route("/api/stats")
def api_stats():
    """Aggregate counts for the dashboard overview (all real data)."""
    return jsonify(db.stats(recent_limit=_arg_int("recent", 5)))


@app.route("/api/analytics")
def api_analytics():
    """Deeper dashboard analytics: violations over time, breakdowns, review
    outcomes, confidence distribution, plate-recognition rate, and processing
    throughput/model stats. Every number is computed from stored rows."""
    return jsonify(db.analytics(
        days=_arg_int("days", 30),
        conf_buckets=_arg_int("conf_buckets", 10),
    ))


@app.route("/api/violations")
def api_violations():
    """Filtered, paginated violation list. All filters are optional query
    params: plate, type, status, review_status, min_confidence, max_confidence,
    date_from, date_to, sort, order (asc|desc), limit, offset."""
    result = db.list_violations(
        plate=request.args.get("plate"),
        violation_type=request.args.get("type"),
        status=request.args.get("status"),
        review_status=request.args.get("review_status"),
        session_id=request.args.get("session_id"),
        min_confidence=_arg_float("min_confidence"),
        max_confidence=_arg_float("max_confidence"),
        date_from=request.args.get("date_from"),
        date_to=request.args.get("date_to"),
        sort=request.args.get("sort", "id"),
        descending=request.args.get("order", "desc").lower() != "asc",
        limit=_arg_int("limit", 50),
        offset=_arg_int("offset", 0),
    )
    return jsonify(result)


@app.route("/api/violations/<int:violation_id>")
def api_violation_detail(violation_id):
    """Full detail for one violation, including its evidence package paths."""
    detail = db.get_violation(violation_id)
    if detail is None:
        return jsonify({"error": "violation not found"}), 404
    return jsonify(detail)


@app.route("/api/violations/<int:violation_id>/review", methods=["POST"])
def api_review(violation_id):
    """Record a human review decision. Body: {review_status, decision?, notes?}.

    Separates automated detection from human confirmation — CV predictions are
    not ground truth, so a violation stays 'pending' until a person confirms or
    dismisses it."""
    denied = _require_role("reviewer")
    if denied:
        return denied
    if _rate_limited():
        return jsonify({"error": "rate limit exceeded"}), 429

    data = request.get_json(silent=True) or {}
    review_status = data.get("review_status")
    if not review_status:
        return jsonify({"error": "review_status is required"}), 400

    _role, actor = _current_actor()
    try:
        updated = db.set_review(
            violation_id,
            review_status,
            reviewer_decision=data.get("decision"),
            notes=data.get("notes"),
            actor=actor,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not updated:
        return jsonify({"error": "violation not found"}), 404

    log.info("violation %s reviewed: %s by %s", violation_id, review_status, actor)
    return jsonify({"message": "review recorded", "review_status": review_status})


@app.route("/api/violations/<int:violation_id>/payment", methods=["POST"])
def api_payment(violation_id):
    """Update a violation's PAYMENT status (independent of review). Body:
    {status: unpaid|paid|cancelled}. Reviewer role required when auth is on."""
    denied = _require_role("reviewer")
    if denied:
        return denied
    if _rate_limited():
        return jsonify({"error": "rate limit exceeded"}), 429

    data = request.get_json(silent=True) or {}
    status = data.get("status")
    if not status:
        return jsonify({"error": "status is required"}), 400

    _role, actor = _current_actor()
    try:
        updated = db.set_payment_status(violation_id, status, actor=actor)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not updated:
        return jsonify({"error": "violation not found"}), 404
    return jsonify({"message": "payment status updated", "status": status})


@app.route("/api/audit")
def api_audit():
    """Recent audit-log events (admin role required when auth is on)."""
    denied = _require_role("admin")
    if denied:
        return denied
    return jsonify({"events": db.list_audit(
        limit=_arg_int("limit", 100),
        record_id=request.args.get("record_id"),
    )})


@app.route("/api/sessions")
def api_sessions():
    """Processing sessions (most recent first)."""
    return jsonify({"sessions": db.list_sessions(limit=_arg_int("limit", 50))})


@app.route("/api/sessions/<session_id>")
def api_session_detail(session_id):
    """One processing session plus the violations it produced (most recent
    first, bounded), so the dashboard can show a per-run drill-down."""
    s = db.get_session(session_id)
    if s is None:
        return jsonify({"error": "session not found"}), 404
    s["violations"] = db.list_violations(
        session_id=session_id, limit=_arg_int("limit", 100)
    )
    return jsonify(s)


@app.route("/api/violations/<int:violation_id>/trace")
def api_violation_trace(violation_id):
    """The supporting detection trace for a confirmed violation (Phase 12)."""
    return jsonify({"trace": db.get_detection_trace(violation_id)})

# =====================================
# RUN
# =====================================

if __name__ == "__main__":
    # Debug is OFF by default (the Werkzeug debugger allows arbitrary code
    # execution if the server is exposed). Opt in explicitly with FLASK_DEBUG=1
    # for local development only. For production use a WSGI server (gunicorn) —
    # see docs/DEPLOYMENT.md.
    log.info(
        "starting dev server on %s:%s (debug=%s)",
        config.server.host, config.server.port, config.server.debug,
    )
    app.run(
        debug=config.server.debug,
        host=config.server.host,
        port=config.server.port,
    )
