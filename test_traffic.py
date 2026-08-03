from ultralytics import YOLO

model = YOLO("runs/detect/traffic_model-2/weights/best.pt")

results = model.predict(
    source="plate10.jpeg",
    conf=0.25,
    save=True
)

print("Done!")