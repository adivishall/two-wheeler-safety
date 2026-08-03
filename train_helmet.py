from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="mps",
    workers=4,
    name="helmet_model"
)