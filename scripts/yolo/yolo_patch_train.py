# python3 ./scripts/yolo/yolo_patch_train.py
# Crops 8/16/32 patches from images/cctv + images/forest using patches_index.csv,
# then trains YOLO classify. No FLAME eval.
import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "images" / "patches_index.csv"
IMAGES_DIR = ROOT / "images"
OUT_ROOT = ROOT / "dataset" / "patches"
RESULTS_DIR = ROOT / "results" / "yolo"
SIZES = [8, 16, 32]
SEED = 42
TRAIN_CAP = 5000
VAL_CAP = 1000
EPOCHS = 20
IMGSZ = 640
BATCH = 16


def load_rows(size: int):
    rows = []
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if int(row["size"]) != size:
                continue
            rows.append({
                "image": row["image"],
                "x": int(row["x"]),
                "y": int(row["y"]),
                "size": size,
                "label": int(float(row["label"])),
            })
    return rows


def split_images(images, rng):
    images = sorted(images)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * 0.2))
    val = set(images[:n_val])
    train = set(images[n_val:])
    return train, val


def sample_balanced(rows, image_set, cap, rng):
    fire = [r for r in rows if r["image"] in image_set and r["label"] == 1]
    non_fire = [r for r in rows if r["image"] in image_set and r["label"] == 0]
    rng.shuffle(fire)
    rng.shuffle(non_fire)
    n = min(len(fire), len(non_fire), cap)
    return fire[:n], non_fire[:n]


def crop_rows(picked, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    opened = {}
    saved = 0
    for i, row in enumerate(picked):
        src = IMAGES_DIR / row["image"]
        if not src.exists():
            raise SystemExit(f"Missing image: {src}")
        if src not in opened:
            opened[src] = Image.open(src).convert("RGB")
        img = opened[src]
        x, y, s = row["x"], row["y"], row["size"]
        if x + s > img.width or y + s > img.height:
            continue
        crop = img.crop((x, y, x + s, y + s))
        name = f"{src.stem}_{x}_{y}_{i}.png"
        crop.save(dest_dir / name)
        saved += 1
    for img in opened.values():
        img.close()
    return saved


def build_size(size: int, train_imgs, val_imgs, rng):
    rows = load_rows(size)
    if not rows:
        raise SystemExit(f"No CSV rows for size={size}")

    train_fire, train_nf = sample_balanced(rows, train_imgs, TRAIN_CAP, rng)
    val_fire, val_nf = sample_balanced(rows, val_imgs, VAL_CAP, rng)
    if not train_fire or not val_fire:
        raise SystemExit(f"size={size}: empty fire split (train={len(train_fire)} val={len(val_fire)})")

    out = OUT_ROOT / f"{size}"
    if out.exists():
        shutil.rmtree(out)

    counts = {
        "train_fire": crop_rows(train_fire, out / "train" / "fire"),
        "train_non_fire": crop_rows(train_nf, out / "train" / "non_fire"),
        "val_fire": crop_rows(val_fire, out / "val" / "fire"),
        "val_non_fire": crop_rows(val_nf, out / "val" / "non_fire"),
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
    }
    return out, counts


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    all_images = sorted({r["image"] for r in load_rows(SIZES[0])})
    train_imgs, val_imgs = split_images(all_images, rng)
    manifest = {
        "seed": SEED,
        "train_cap": TRAIN_CAP,
        "val_cap": VAL_CAP,
        "train_images": sorted(train_imgs),
        "val_images": sorted(val_imgs),
        "sizes": {},
    }

    for size in SIZES:
        print(f"\n=== patch size {size} ===")
        data_dir, counts = build_size(size, train_imgs, val_imgs, rng)
        manifest["sizes"][str(size)] = counts
        print(f"Built {data_dir}: {counts}")

        run_name = f"runs/classify/yolo_patch_{size}"
        model = YOLO("yolo11n-cls.pt")
        model.train(
            data=str(data_dir),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            project=str(ROOT),
            name=run_name,
            exist_ok=True,
        )
        print(f"Done. Weights: {run_name}/weights/best.pt")

    with open(RESULTS_DIR / "patch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {RESULTS_DIR / 'patch_manifest.json'}")


if __name__ == "__main__":
    main()
