from ultralytics import YOLO
import easyocr
import cv2

model = YOLO("runs/detect/traffic_model-2/weights/best.pt")

reader = easyocr.Reader(['en'])

results = model.predict(
    source="test.jpg",
    conf=0.25
)

img = cv2.imread("test.jpg")

for r in results:
    for box in r.boxes:

        cls = int(box.cls[0])

        label = model.names[cls]

        if label == "Plate":

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            plate_crop = img[y1:y2, x1:x2]

            text = reader.readtext(
                plate_crop,
                detail=0
            )

            print("Plate Number:", text)