import os
import tempfile
import threading
from flask import Flask, request, jsonify, render_template, send_from_directory
import sqlite3

app = Flask(__name__)

os.makedirs("evidence", exist_ok=True)

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

# Fine amount (₹) per violation type; anything unlisted falls back to 300.
FINE_AMOUNTS = {
    "no_helmet": 500,
    "triple_riding": 1000,
    "overspeed": 700,
}

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
    """Insert one fine and return the amount charged."""
    amount = FINE_AMOUNTS.get(violation, 300)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO fines (plate, violation, amount, image_path)
        VALUES (?, ?, ?, ?)
        """,
        (plate.strip().upper(), violation, amount, image_path),
    )
    conn.commit()
    conn.close()

    return amount

# =====================================
# DATABASE SETUP
# =====================================

def init_db():

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS fines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT,
        violation TEXT,
        amount INTEGER,
        image_path TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'unpaid'
    )
    """)

    conn.commit()
    conn.close()

init_db()

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

    evidence_folder = os.path.join(
        app.root_path,
        "evidence"
    )

    return send_from_directory(
        evidence_folder,
        filename
    )

# =====================================
# SAVE DETECTED VIOLATION
# =====================================

@app.route("/detect", methods=["POST"])
def detect():

    if DETECT_API_KEY and request.headers.get("X-API-Key") != DETECT_API_KEY:
        return jsonify({"error": "invalid or missing API key"}), 401

    data = request.get_json(silent=True) or {}

    if not all(k in data for k in ("plate", "violation", "image_path")):
        return jsonify({
            "error": "plate, violation, and image_path are required"
        }), 400

    record_fine(data["plate"], data["violation"], data["image_path"])

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

    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "no image uploaded"}), 400

    upload = request.files["image"]

    # Persist the upload to a temp file so OpenCV can read it by path, then
    # remove it — analyze_image writes its own annotated copy into evidence/.
    suffix = os.path.splitext(upload.filename)[1] or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        upload.save(tmp_path)

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
# GET FINES BY PLATE
# =====================================

@app.route("/get_fines/<plate>")
def get_fines(plate):

    plate = plate.strip().upper()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    SELECT
        violation,
        amount,
        image_path,
        timestamp,
        status
    FROM fines
    WHERE plate=?
    """, (plate,))

    rows = c.fetchall()

    conn.close()

    fines = []
    total = 0

    for row in rows:

        image_file = row[2].split("/")[-1]

        fines.append({
            "violation": row[0],
            "amount": row[1],
            "status": row[4],
            "time": row[3],
            "image": f"/evidence/{image_file}"
        })

        if row[4] == "unpaid":
            total += row[1]

    return jsonify({
        "fines": fines,
        "total": total
    })

# =====================================
# SHOW ALL FINES
# =====================================

@app.route("/fines")
def all_fines():

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    SELECT
        id,
        plate,
        violation,
        amount,
        image_path,
        timestamp,
        status
    FROM fines
    """)

    rows = c.fetchall()

    conn.close()

    fines = [
        {
            "id": row[0],
            "plate": row[1],
            "violation": row[2],
            "amount": row[3],
            "image_path": row[4],
            "timestamp": row[5],
            "status": row[6]
        }
        for row in rows
    ]

    return jsonify(fines)

# =====================================
# RUN
# =====================================

if __name__ == "__main__":
    # Set FLASK_DEBUG=0 before any real deployment — the debugger
    # allows arbitrary code execution if the server is exposed.
    debug_mode = os.environ.get("FLASK_DEBUG", "1") != "0"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug_mode, port=port)