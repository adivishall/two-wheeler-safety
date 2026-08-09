import os
import cv2
import requests
from ultralytics import YOLO
from modules.plate_ocr import read_plate
from modules.speed import SpeedEstimator
from utils.tracker import CentroidTracker

# =====================================
# SETTINGS
# =====================================

API_URL = "http://127.0.0.1:5000/detect"
SPEED_LIMIT_KMH = 40

os.makedirs("evidence", exist_ok=True)

# =====================================
# LOAD MODEL + TRACKING
# =====================================

model = YOLO("yolov8n.pt")

tracker = CentroidTracker()
speed_estimator = SpeedEstimator()

# (track_id, violation) pairs already reported this run, so the same
# bike isn't fined again on every single frame it stays in view.
reported = set()

# =====================================
# REPORT A VIOLATION
# =====================================

def report_violation(track_id, violation, plate, frame, box):
    key = (track_id, violation)

    if key in reported or not plate:
        return

    reported.add(key)

    x1, y1, x2, y2 = box
    evidence_file = f"evidence/{plate}_{violation}.jpg"
    cv2.imwrite(evidence_file, frame[y1:y2, x1:x2])

    try:
        response = requests.post(
            API_URL,
            json={
                "plate": plate,
                "violation": violation,
                "image_path": evidence_file
            }
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

    persons = []
    bikes = []

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:  # person
            persons.append((x1, y1, x2, y2))
        elif cls == 3:  # motorcycle
            bikes.append((x1, y1, x2, y2))

        # Draw boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    tracked_bikes = tracker.update(bikes)

    for track_id, bike in tracked_bikes.items():
        bx1, by1, bx2, by2 = bike

        # ------------------------
        # TRIPLE RIDING DETECTION
        # ------------------------
        rider_count = 0

        for person in persons:
            px1, py1, px2, py2 = person

            # bounding-box overlap check (both axes, not just x)
            if px1 < bx2 and px2 > bx1 and py1 < by2 and py2 > by1:
                rider_count += 1

        is_triple_riding = rider_count > 2

        if is_triple_riding:
            cv2.putText(frame, "Triple Riding!", (bx1, by1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # ------------------------
        # SPEED ESTIMATION
        # ------------------------
        speed = speed_estimator.calculate_speed(track_id, bike)
        is_overspeed = speed > SPEED_LIMIT_KMH

        if is_overspeed:
            cv2.putText(frame, f"Speed: {speed} km/h", (bx1, by1-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # ------------------------
        # PLATE OCR
        # ------------------------
        crop = frame[by1:by2, bx1:bx2]
        plate_text = read_plate(crop)

        if plate_text:
            plate_text = "".join(ch for ch in plate_text if ch.isalnum()).upper()
            cv2.putText(frame, plate_text, (bx1, by2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # ------------------------
        # REPORT VIOLATIONS (once per tracked bike)
        # ------------------------
        if is_triple_riding:
            report_violation(track_id, "triple_riding", plate_text, frame, bike)

        if is_overspeed:
            report_violation(track_id, "overspeed", plate_text, frame, bike)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
