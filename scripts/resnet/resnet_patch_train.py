# python3 ./scripts/resnet/resnet_patch_train.py
# Crops 8px patches from images/cctv + images/forest using patches_index.csv,
# then trains ResNet50 classify. No FLAME eval.
import csv
import json
import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "images" / "patches_index.csv"
IMAGES_DIR = ROOT / "images"
OUT_ROOT = ROOT / "dataset" / "patches"
RESULTS_DIR = ROOT / "results" / "resnet"
SAVE_DIR = ROOT / "runs" / "classify" / "resnet50_patch_8"

SIZE = 8
SEED = 42
TRAIN_CAP = 600
VAL_CAP = 100
EPOCHS = 20
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 0.001


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
        folder, name = row["image"].split("/", 1)
        src = IMAGES_DIR / folder.lower() / name
        if not src.exists():
            raise SystemExit(f"Missing image: {src}")
        if src not in opened:
            opened[src] = Image.open(src).convert("RGB")
        img = opened[src]
        x, y, s = row["x"], row["y"], row["size"]
        if x + s > img.width or y + s > img.height:
            continue
        crop = img.crop((x, y, x + s, y + s))
        if s < 10:
            crop = crop.resize((32, 32), Image.NEAREST)
        out_name = f"{src.stem}_{x}_{y}_{i}.png"
        crop.save(dest_dir / out_name)
        saved += 1
    for img in opened.values():
        img.close()
    return saved


def build_patches(rng):
    rows = load_rows(SIZE)
    if not rows:
        raise SystemExit(f"No CSV rows for size={SIZE}")

    all_images = sorted({r["image"] for r in rows})
    train_imgs, val_imgs = split_images(all_images, rng)
    train_fire, train_nf = sample_balanced(rows, train_imgs, TRAIN_CAP, rng)
    val_fire, val_nf = sample_balanced(rows, val_imgs, VAL_CAP, rng)
    if not train_fire or not val_fire:
        raise SystemExit(
            f"size={SIZE}: empty fire split "
            f"(train={len(train_fire)} val={len(val_fire)})"
        )

    out = OUT_ROOT / str(SIZE)
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
    manifest = {
        "seed": SEED,
        "size": SIZE,
        "train_cap": TRAIN_CAP,
        "val_cap": VAL_CAP,
        "train_images": sorted(train_imgs),
        "val_images": sorted(val_imgs),
        "counts": counts,
    }
    return out, manifest


def train(data_dir: Path):
    os.makedirs(SAVE_DIR / "weights", exist_ok=True)

    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(str(data_dir / "train"), transform=transform_train)
    val_dataset = datasets.ImageFolder(str(data_dir / "val"), transform=transform_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = timm.create_model(
        "resnet50.a1_in1k",
        pretrained=True,
        num_classes=2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Classes: {train_dataset.classes}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total if total else 0.0
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"loss: {running_loss / max(len(train_loader), 1):.4f} - "
            f"val_acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "class_names": train_dataset.classes},
                SAVE_DIR / "weights" / "best.pt",
            )

    torch.save(
        {"model": model.state_dict(), "class_names": train_dataset.classes},
        SAVE_DIR / "weights" / "last.pt",
    )
    print(f"Best val_acc: {best_acc:.4f}")
    print(f"Done. Weights: {SAVE_DIR}/weights/best.pt")


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print(f"=== building patch size {SIZE} ===")
    data_dir, manifest = build_patches(rng)
    print(f"Built {data_dir}: {manifest['counts']}")

    with open(RESULTS_DIR / "patch_8_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {RESULTS_DIR / 'patch_8_manifest.json'}")

    print(f"\n=== training ResNet50 on patch-{SIZE} ===")
    train(data_dir)


if __name__ == "__main__":
    main()
