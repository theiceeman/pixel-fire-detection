# python3 ./scripts/yolo/yolo_patch_evaluate.py
# Tile FLAME images at 8/16/32, score each crop, image = fire if any crop is fire.
# Images are resized so the long side is 640 first (native 4K 8x8 tiling is too slow).
import json
import random
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
# weights = ROOT / "runs" / "classify" / f"yolo_patch_{size}" / "weights" / "best.pt"
MODEL_PATH = weights = ROOT / "runs" / "classify" / "train" / "weights" / "best.pt"
SCALE_DIR = ROOT / "dataset" / "flame_scale_eval"
RESULTS_DIR = ROOT / "results" / "yolo"
SIZES = [8, 16, 32]
MAX_SIDE = 640
BATCH = 64
SAMPLE = 20
SEED = 42
FIRE_CLASS = "fire"
SCALES = ["tiny", "small", "medium", "large"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return img


def make_tiles(img: Image.Image, size: int):
    w, h = img.size
    if w < size or h < size:
        crop = img.resize((max(size, 32), max(size, 32)), Image.NEAREST)
        return [crop]
    tiles = []
    for y in range(0, h - size + 1, size):
        for x in range(0, w - size + 1, size):
            crop = img.crop((x, y, x + size, y + size))
            if size < 10:
                crop = crop.resize((32, 32), Image.NEAREST)
            tiles.append(crop)
    return tiles


def image_is_fire(model, tiles, fire_idx):
    max_fire = 0.0
    hit = False
    for i in range(0, len(tiles), BATCH):
        preds = model.predict(tiles[i : i + BATCH], verbose=False)
        for pred in preds:
            fire_p = float(pred.probs.data[fire_idx])
            if fire_p > max_fire:
                max_fire = fire_p
            if model.names[pred.probs.top1] == FIRE_CLASS:
                hit = True
        if hit:
            break
    return hit, max_fire


def evaluate(size: int):
    # weights = ROOT / "runs" / "classify" / f"yolo_patch_{size}" / "weights" / "best.pt"
    weights = MODEL_PATH

    if not weights.exists():
        raise SystemExit(f"Missing {weights}")
    model = YOLO(str(weights))
    fire_idx = next(i for i, name in model.names.items() if name == FIRE_CLASS)
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

        files = sorted(
            p for p in scale_dir.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VALID_EXT
        )
        if SAMPLE and len(files) > SAMPLE:
            files = random.Random(SEED).sample(files, SAMPLE)
            files = sorted(files)
        for n, path in enumerate(files, 1):
            tiles = make_tiles(load_image(path), size)
            hit, conf = image_is_fire(model, tiles, fire_idx)
            if hit:
                detected += 1
                conf_det.append(conf)
            else:
                missed += 1
                conf_miss.append(conf)
            if n % 50 == 0:
                print(f"  {scale} {n}/{len(files)}")

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
        print(f"{scale}: {detected}/{total} ({results[scale]['fire_detection_rate']:.3f})")

    # out = RESULTS_DIR / f"yolo_flame_patch_{size}_tiled_result.json"
    out = RESULTS_DIR / f"yolo_flame_baseline_tiled_{size}_result.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")


def main():
    for size in SIZES:
        print(f"\n=== eval patch size {size} ===")
        evaluate(size)


if __name__ == "__main__":
    main()
