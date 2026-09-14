# python3 ./scripts/yolo/yolo_bcst_dose.py
# Uses dataset/bcst_aug with train/{fire,non_fire,firelike} + val/{fire,non_fire}.
# For each dose N: mix N firelike into non_fire (600/600), train, eval flame.
import json
import os
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "bcst_aug"
TMP = ROOT / "dataset" / "bcst_dose_tmp"
SCALE_DIR = ROOT / "dataset" / "flame_scale_eval"
RESULTS_DIR = ROOT / "results" / "yolo"
DOSES = [30, 60, 90]
SEED = 42
FIRE_CLASS = "fire"
SCALES = ["tiny", "small", "medium", "large"]
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


def evaluate(weights: Path, output_path: Path):
    model = YOLO(str(weights))
    results = {}

    for scale in SCALES:
        scale_dir = SCALE_DIR / scale
        detected = missed = 0
        conf_det, conf_miss = [], []

        if not scale_dir.is_dir():
            results[scale] = {
                "total": 0,
                "detected": 0,
                "missed": 0,
                "fire_detection_rate": 0,
                "avg_confidence": 0,
                "avg_confidence_detected": 0,
                "avg_confidence_missed": 0,
            }
            continue

        for filename in os.listdir(scale_dir):
            if filename.startswith(".") or Path(filename).suffix.lower() not in VALID_EXT:
                continue
            preds = model.predict(str(scale_dir / filename), verbose=False)
            predicted = model.names[preds[0].probs.top1]
            conf = float(preds[0].probs.top1conf)
            if predicted == FIRE_CLASS:
                detected += 1
                conf_det.append(conf)
            else:
                missed += 1
                conf_miss.append(conf)

        total = detected + missed
        all_conf = conf_det + conf_miss
        results[scale] = {
            "total": total,
            "detected": detected,
            "missed": missed,
            "fire_detection_rate": detected / total if total else 0,
            "avg_confidence": sum(all_conf) / len(all_conf) if all_conf else 0,
            "avg_confidence_detected": sum(conf_det) / len(conf_det) if conf_det else 0,
            "avg_confidence_missed": sum(conf_miss) / len(conf_miss) if conf_miss else 0,
        }
        print(f"  {scale}: {detected}/{total} ({results[scale]['fire_detection_rate']:.3f})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved {output_path}")


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

        weights = ROOT / run_name / "weights" / "best.pt"
        evaluate(weights, RESULTS_DIR / f"flame_bcst_{n}_result.json")

    with open(RESULTS_DIR / "bcst_dose_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {RESULTS_DIR / 'bcst_dose_manifest.json'}")


if __name__ == "__main__":
    main()
