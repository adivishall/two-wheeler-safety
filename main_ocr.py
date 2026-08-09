import re
import os
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

# ---------------------------------
# LOAD MODEL
# ---------------------------------

model = YOLO(MODEL_PATH)

reader = easyocr.Reader(['en'])

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
    )

    print(
        "Clean Plate:",
        plate_number
    )

# ---------------------------------
# BUILD VIOLATION LIST
# ---------------------------------

violations = []

if "WithoutHelmet" in detected_classes:
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

    evidence_file = (
        f"evidence/{plate_number}.jpg"
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

        response = requests.post(
            API_URL,
            json={
                "plate": plate_number,
                "violation": violation,
                "image_path": evidence_file
            }
        )

        print(
            "Sent:",
            violation
        )

        print(
            response.text
        )

else:

    print(
        "No valid violation found."
    )