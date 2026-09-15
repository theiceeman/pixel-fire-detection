# python3 ./scripts/mobilenet/mobilenet_bcst_dose.py
# Uses dataset/bcst_aug with train/{fire,non_fire,firelike} + val/{fire,non_fire}.
# For each dose N: mix N firelike into non_fire (600/600), train only.
import json
import os
import random
import shutil
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "bcst_aug"
TMP = ROOT / "dataset" / "bcst_dose_tmp"
RESULTS_DIR = ROOT / "results" / "mobilenet"
DOSES = [30, 60, 90]
SEED = 42
EPOCHS = 20
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 0.001
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path):
    return sorted(
        p.name
        for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VALID_EXT
    )


def link_all(src_dir: Path, dst_dir: Path, names=None):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names if names is not None else list_images(src_dir):
        dst = dst_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src_dir / name, dst)


def build_dose(n: int, firelike_pick: list[str], non_fire_pick: list[str]) -> Path:
    out = TMP / f"n{n}"
    if out.exists():
        shutil.rmtree(out)

    link_all(SRC / "train" / "fire", out / "train" / "fire")
    link_all(SRC / "train" / "non_fire", out / "train" / "non_fire", non_fire_pick)
    link_all(SRC / "train" / "firelike", out / "train" / "non_fire", firelike_pick)
    link_all(SRC / "val" / "fire", out / "val" / "fire")
    link_all(SRC / "val" / "non_fire", out / "val" / "non_fire")
    return out


def train_dose(data_dir: Path, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

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
    if len(train_dataset.classes) != 2:
        raise SystemExit(f"Expected 2 classes, got {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=True, num_classes=2)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
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

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {running_loss/len(train_loader):.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "class_names": train_dataset.classes},
                save_dir / "best.pt",
            )

    torch.save(
        {"model": model.state_dict(), "class_names": train_dataset.classes},
        save_dir / "last.pt",
    )
    print(f"Best val_acc: {best_acc:.4f}")
    print(f"Done. Weights: {save_dir / 'best.pt'}")


def main():
    firelike = list_images(SRC / "train" / "firelike")
    non_fire = list_images(SRC / "train" / "non_fire")
    if len(firelike) < max(DOSES) or len(non_fire) < 600:
        raise SystemExit(f"Need >= {max(DOSES)} firelike and 600 non_fire; got {len(firelike)}, {len(non_fire)}")

    rng = random.Random(SEED)
    firelike_shuffled = firelike[:]
    non_fire_shuffled = non_fire[:]
    rng.shuffle(firelike_shuffled)
    rng.shuffle(non_fire_shuffled)

    manifest = {"seed": SEED, "doses": {}}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for n in DOSES:
        fl_pick = firelike_shuffled[:n]
        nf_pick = non_fire_shuffled[: 600 - n]
        manifest["doses"][str(n)] = {"firelike": fl_pick, "non_fire": nf_pick}

        print(f"\n=== dose N={n} ===")
        data_dir = build_dose(n, fl_pick, nf_pick)
        save_dir = ROOT / "runs" / "classify" / f"mobilenetv4_bcst_{n}" / "weights"
        train_dose(data_dir, save_dir)

    with open(RESULTS_DIR / "bcst_dose_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {RESULTS_DIR / 'bcst_dose_manifest.json'}")


if __name__ == "__main__":
    main()
