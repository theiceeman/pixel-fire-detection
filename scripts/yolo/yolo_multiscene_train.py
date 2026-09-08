# python3 ./scripts/yolo/yolo_multiscene_train.py
import os
import shutil
from ultralytics import YOLO

OUT_DIR = "dataset/multiscene"
CLASSES = ["fire", "non_fire"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def copy_dir_images(src_dir, dst_dir, prefix=""):
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    if not os.path.isdir(src_dir):
        print(f"Warning: missing {src_dir}")
        return 0
    for name in os.listdir(src_dir):
        if name.startswith("."):
            continue
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in VALID_EXT:
            continue
        dst_name = f"{prefix}{name}" if prefix else name
        dst = os.path.join(dst_dir, dst_name)
        if os.path.exists(dst):
            stem, e = os.path.splitext(dst_name)
            dst = os.path.join(dst_dir, f"{stem}_dup{e}")
        shutil.copy2(src, dst)
        n += 1
    return n


# rebuild multiscene dataset
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

for split in ("train", "val"):
    for cls in CLASSES:
        os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

# val = original val only
for cls in CLASSES:
    n = copy_dir_images(f"dataset/val/{cls}", os.path.join(OUT_DIR, "val", cls))
    print(f"val/{cls}: {n} from dataset/val")

# train = original + forest + cctv(fire only)
for cls in CLASSES:
    n1 = copy_dir_images(f"dataset/train/{cls}", os.path.join(OUT_DIR, "train", cls))
    n2 = copy_dir_images(f"dataset/forest_train/{cls}", os.path.join(OUT_DIR, "train", cls), prefix="forest_")
    print(f"train/{cls}: {n1} original + {n2} forest")

n_cctv = copy_dir_images("images/cctv", os.path.join(OUT_DIR, "train", "fire"), prefix="cctv_")
print(f"train/fire: +{n_cctv} cctv")

for split in ("train", "val"):
    for cls in CLASSES:
        count = len([
            f for f in os.listdir(os.path.join(OUT_DIR, split, cls))
            if not f.startswith(".")
        ])
        print(f"{OUT_DIR}/{split}/{cls}: {count}")

# train YOLO from pretrained (same recipe as baseline)
model = YOLO("yolo11n-cls.pt")
model.train(
    data=OUT_DIR,
    epochs=20,
    imgsz=640,
    batch=16,
    project=".",
    name="runs/classify/yolo_multiscene",
)

print("Done. Weights: runs/classify/yolo_multiscene/weights/best.pt")
