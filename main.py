import cv2
import time
import pandas as pd
from ultralytics import YOLO
from modules.plate_ocr import read_plate
from modules.speed import SpeedEstimator

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize speed estimator
speed_estimator = SpeedEstimator()

# Open video
cap = cv2.VideoCapture("test_video.mp4")

# Log file
log_file = "output/logs.csv"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    persons = []
    bikes = []
    plates = []

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:  # person
            persons.append((x1, y1, x2, y2))
        elif cls == 3:  # motorcycle
            bikes.append((x1, y1, x2, y2))

        # Draw boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # ------------------------
    # TRIPLE RIDING DETECTION
    # ------------------------
    for bike in bikes:
        bx1, by1, bx2, by2 = bike
        rider_count = 0

        for person in persons:
            px1, py1, px2, py2 = person

            # simple overlap check
            if px1 < bx2 and px2 > bx1:
                rider_count += 1

        if rider_count > 2:
            cv2.putText(frame, "Triple Riding!", (bx1, by1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # ------------------------
    # SPEED ESTIMATION
    # ------------------------
    speed = speed_estimator.calculate_speed(bikes)

    if speed > 40:
        cv2.putText(frame, f"Speed: {speed} km/h", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # ------------------------
    # PLATE OCR
    # ------------------------
    for bike in bikes:
        x1, y1, x2, y2 = bike
        crop = frame[y1:y2, x1:x2]

        plate_text = read_plate(crop)

        if plate_text:
            cv2.putText(frame, plate_text, (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()