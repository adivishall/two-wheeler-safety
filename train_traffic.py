from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="master_traffic_violation_dataset/data.yaml",
    epochs=15,
    imgsz=640,
    batch=16,
    device="mps",
    workers=4,
    name="traffic_model"
)