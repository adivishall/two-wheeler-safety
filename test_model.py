from ultralytics import YOLO

model = YOLO("runs/detect/helmet_model/weights/best.pt")

results = model.predict(
    source="data/test/images",
    save=True,
    conf=0.25
)