import re
import os
import time
import cv2
import requests
import easyocr
from ultralytics import YOLO

# ---------------------------------
# SETTINGS
# ---------------------------------

IMAGE_PATH = "plate8.jpeg"

MODEL_PATH = "runs/detect/traffic_model-2/weights/best.pt"

API_URL = "http://127.0.0.1:5000/detect"

# Must match DETECT_API_KEY on the server if it has one set; empty is
# fine when the server has no key configured.
API_KEY = os.environ.get("DETECT_API_KEY", "")

# ---------------------------------
# LOAD MODEL
# ---------------------------------

model = YOLO(MODEL_PATH)

reader = easyocr.Reader(['en'])

# ---------------------------------
# HELPERS
# ---------------------------------

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    return inter_area / float(area_a + area_b - inter_area)

# ---------------------------------
# CREATE EVIDENCE FOLDER
# ---------------------------------

os.makedirs("evidence", exist_ok=True)

# ---------------------------------
# RUN DETECTION
# ---------------------------------

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25
)

img = cv2.imread(IMAGE_PATH)

detected_classes = []
without_helmet_boxes = []
with_helmet_boxes = []

plate_number = None
plate_text = []

# ---------------------------------
# PROCESS DETECTIONS
# ---------------------------------

for r in results:

    for box in r.boxes:

        cls = int(box.cls[0])

        label = model.names[cls]

        print("Detected:", label)

        detected_classes.append(label)

        if label == "WithoutHelmet":
            without_helmet_boxes.append(tuple(map(int, box.xyxy[0])))
        elif label == "WithHelmet":
            with_helmet_boxes.append(tuple(map(int, box.xyxy[0])))

        # -------------------------
        # OCR ON PLATE
        # -------------------------

        if label == "Plate":

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            plate_crop = img[
                y1:y2,
                x1:x2
            ]

            plate_text = reader.readtext(
                plate_crop,
                detail=0
            )

            print(
                "Plate Number:",
                plate_text
            )

if len(plate_text) > 0:

    plate_number = "".join(plate_text)

    plate_number = re.sub(
        r'[^A-Za-z0-9]',
        '',
        plate_number
    ).upper()

    print(
        "Clean Plate:",
        plate_number
    )

# ---------------------------------
# BUILD VIOLATION LIST
# ---------------------------------

violations = []

# A WithoutHelmet box that overlaps a WithHelmet box is the model
# contradicting itself on the same rider — tested against a real photo
# where the *higher-confidence* box was the wrong one, so the safe move
# is to trust neither rather than guess. Only report no_helmet if at
# least one WithoutHelmet detection has no such contradiction.
no_helmet_confirmed = any(
    not any(iou(whb, wb) > 0.1 for wb in with_helmet_boxes)
    for whb in without_helmet_boxes
)

if no_helmet_confirmed:
    violations.append(
        "no_helmet"
    )

if "TripleRiding" in detected_classes:
    violations.append(
        "triple_riding"
    )

# ---------------------------------
# SAVE EVIDENCE IMAGE
# ---------------------------------

if plate_number:

    # timestamped so re-scanning the same plate later doesn't silently
    # overwrite an earlier fine's evidence photo
    evidence_file = (
        f"evidence/{plate_number}_{int(time.time())}.jpg"
    )

    cv2.imwrite(
        evidence_file,
        img
    )

    print(
        "Evidence Saved:",
        evidence_file
    )

# ---------------------------------
# SEND TO WEBSITE
# ---------------------------------

if plate_number and len(violations) > 0:

    for violation in violations:

        try:
            response = requests.post(
                API_URL,
                json={
                    "plate": plate_number,
                    "violation": violation,
                    "image_path": evidence_file
                },
                headers={"X-API-Key": API_KEY}
            )

            print(
                "Sent:",
                violation
            )

            print(
                response.text
            )

        except requests.exceptions.ConnectionError:
            print(
                f"Could not reach {API_URL} — is app.py running?"
            )

else:

    print(
        "No valid violation found."
    )