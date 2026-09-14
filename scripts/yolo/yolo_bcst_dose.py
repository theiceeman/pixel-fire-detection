# python3 ./scripts/yolo/yolo_bcst_dose.py
# Uses dataset/bcst_aug with train/{fire,non_fire,firelike} + val/{fire,non_fire}.
# For each dose N: mix N firelike into non_fire (600/600), train only.
import json
import os
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "bcst_aug"
TMP = ROOT / "dataset" / "bcst_dose_tmp"
RESULTS_DIR = ROOT / "results" / "yolo"
DOSES = [30, 60, 90]
SEED = 42
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
        run_name = f"runs/classify/yolo_bcst_{n}"

        model = YOLO("yolo11n-cls.pt")
        model.train(
            data=str(data_dir),
            epochs=20,
            imgsz=640,
            batch=16,
            project=str(ROOT),
            name=run_name,
            exist_ok=True,
        )
        print(f"Done. Weights: {run_name}/weights/best.pt")

    with open(RESULTS_DIR / "bcst_dose_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {RESULTS_DIR / 'bcst_dose_manifest.json'}")


if __name__ == "__main__":
    main()
