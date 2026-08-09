import torch
from ultralytics import YOLO

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = YOLO("yolov8n.pt")

model.train(
    data="master_traffic_violation_dataset/data.yaml",
    epochs=15,
    imgsz=640,
    batch=16,
    device=device,
    workers=4,
    name="traffic_model"
)