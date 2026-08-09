import os
import time
import math
import cv2
import requests
from ultralytics import YOLO
from modules.plate_ocr import read_plate
from modules.speed import SpeedEstimator
from utils.tracker import CentroidTracker

# =====================================
# SETTINGS
# =====================================

MODEL_PATH = "runs/detect/traffic_model-2/weights/best.pt"
API_URL = "http://127.0.0.1:5000/detect"
SPEED_LIMIT_KMH = 40

# Must match DETECT_API_KEY on the server if it has one set; empty is
# fine when the server has no key configured.
API_KEY = os.environ.get("DETECT_API_KEY", "")

os.makedirs("evidence", exist_ok=True)

# =====================================
# LOAD MODEL + TRACKING
# =====================================

model = YOLO(MODEL_PATH)

tracker = CentroidTracker()
speed_estimator = SpeedEstimator()

# (track_id, violation) pairs already reported this run, so the same
# vehicle isn't fined again on every single frame it stays in view.
reported = set()

# =====================================
# HELPERS
# =====================================

def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def nearest_plate_id(box, tracked_plates):
    """Best-effort association: attributes a violation to the nearest
    tracked plate. Works reliably with one vehicle in frame; with several
    vehicles close together it can mis-attribute a violation, since the
    model detects violation and plate boxes independently rather than as
    a single linked vehicle instance."""

    if not tracked_plates:
        return None

    cx, cy = centroid(box)

    return min(
        tracked_plates,
        key=lambda tid: math.hypot(
            cx - centroid(tracked_plates[tid])[0],
            cy - centroid(tracked_plates[tid])[1]
        )
    )


def clean_plate(text):
    if not text:
        return None
    return "".join(ch for ch in text if ch.isalnum()).upper()


def report_violation(track_id, violation, plate, frame, box):
    key = (track_id, violation)

    if key in reported or not plate:
        return

    reported.add(key)

    x1, y1, x2, y2 = box
    # timestamped so re-detecting the same plate/violation later doesn't
    # silently overwrite an earlier fine's evidence photo
    evidence_file = f"evidence/{plate}_{violation}_{int(time.time())}.jpg"
    cv2.imwrite(evidence_file, frame[y1:y2, x1:x2])

    try:
        response = requests.post(
            API_URL,
            json={
                "plate": plate,
                "violation": violation,
                "image_path": evidence_file
            },
            headers={"X-API-Key": API_KEY}
        )
        print("Sent:", violation, "for", plate, "-", response.text)

    except requests.exceptions.ConnectionError:
        print(f"Could not reach {API_URL} — is app.py running?")

# =====================================
# MAIN LOOP
# =====================================

cap = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    plates = []
    without_helmet = []
    triple_riding = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "Plate":
            plates.append((x1, y1, x2, y2))
            color = (0, 255, 0)
        elif label == "WithoutHelmet":
            without_helmet.append((x1, y1, x2, y2))
            color = (0, 0, 255)
        elif label == "TripleRiding":
            triple_riding.append((x1, y1, x2, y2))
            color = (0, 0, 255)
        else:  # WithHelmet
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    tracked_plates = tracker.update(plates)
    plate_text_by_id = {}

    # ------------------------
    # PER-PLATE: OCR + SPEED
    # ------------------------
    for track_id, plate_box in tracked_plates.items():
        px1, py1, px2, py2 = plate_box

        crop = frame[py1:py2, px1:px2]
        plate_text = clean_plate(read_plate(crop))
        plate_text_by_id[track_id] = plate_text

        if plate_text:
            cv2.putText(frame, plate_text, (px1, py2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        speed = speed_estimator.calculate_speed(track_id, plate_box)

        if speed > SPEED_LIMIT_KMH:
            cv2.putText(frame, f"Speed: {speed} km/h", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            report_violation(track_id, "overspeed", plate_text, frame, plate_box)

    # ------------------------
    # NO HELMET / TRIPLE RIDING: associate to nearest tracked plate
    # ------------------------
    for box in without_helmet:
        pid = nearest_plate_id(box, tracked_plates)

        if pid is not None:
            report_violation(pid, "no_helmet", plate_text_by_id.get(pid), frame, tracked_plates[pid])

    for box in triple_riding:
        cv2.putText(frame, "Triple Riding!", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        pid = nearest_plate_id(box, tracked_plates)

        if pid is not None:
            report_violation(pid, "triple_riding", plate_text_by_id.get(pid), frame, tracked_plates[pid])

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
