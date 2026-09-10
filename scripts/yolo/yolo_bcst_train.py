# python3 ./scripts/yolo/yolo_bcst_train.py
# Expects dataset/bcst_aug already built (train/val with fire/non_fire)
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
model.train(
    data="dataset/bcst_aug",
    epochs=20,
    imgsz=640,
    batch=16,
    project=".",
    name="runs/classify/yolo_bcst",
)

print("Done. Weights: runs/classify/yolo_bcst/weights/best.pt")
