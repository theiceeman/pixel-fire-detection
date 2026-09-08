# python3 ./scripts/yolo/yolo_transfer_finetune.py
import os
import random
import shutil
from ultralytics import YOLO

MODEL_A = "runs/classify/train/weights/best.pt"
SOURCE_DIR = "dataset/forest_train"
SPLIT_DIR = "dataset/forest_train_split"
CLASSES = ["fire", "non_fire"]
VAL_RATIO = 0.2
SEED = 42

EPOCHS = 5
BATCH = 8
LR0 = 0.0001
IMGSZ = 640

random.seed(SEED)

# build train/val split for YOLO classify
if os.path.exists(SPLIT_DIR):
    shutil.rmtree(SPLIT_DIR)

for split in ("train", "val"):
    for cls in CLASSES:
        os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)

for cls in CLASSES:
    src = os.path.join(SOURCE_DIR, cls)
    files = [
        f for f in os.listdir(src)
        if not f.startswith(".") and os.path.isfile(os.path.join(src, f))
    ]
    random.shuffle(files)
    n_val = max(1, int(len(files) * VAL_RATIO)) if len(files) > 1 else 1
    val_files = set(files[:n_val])
    for f in files:
        split = "val" if f in val_files else "train"
        shutil.copy2(os.path.join(src, f), os.path.join(SPLIT_DIR, split, cls, f))
    print(f"{cls}: train={len(files) - len(val_files)} val={len(val_files)}")

# Model B: continue from Model A on forest
model = YOLO(MODEL_A)
model.train(
    data=SPLIT_DIR,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    lr0=LR0,
    project=".",
    name="runs/classify/transfer_forest",
)

print("Done. Model B weights: runs/classify/transfer_forest/weights/best.pt")
