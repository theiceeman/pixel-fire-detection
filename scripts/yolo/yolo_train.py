# python3 ./train.py
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
model.train(
    data="dataset",
    epochs=20,
    imgsz=640,
    batch=16,
    project=".",
    name="runs/classify/train",
)
