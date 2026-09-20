# python3 ./yolo_tile_scale_evaluate.py
# Loops models/<name>/weights/best.pt. Same tile rule for all strategies.
# Fire if >= MIN_HITS tiles have fire score >= STRONG.
# Sample SAMPLE images per fire scale (fixed SEED); use all none images.
import json
import random
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

MODELS_DIR = Path("models")
SCALE_DIR = Path("dataset/flame_scale_eval")
RESULTS_DIR = Path("results/yolo")
PATCH_SIZE = 8
MAX_SIDE = 640
BATCH = 64
STRONG = 0.7
MIN_HITS = 2
SAMPLE = 20
SEED = 42
FIRE_CLASS = "fire"
FIRE_SCALES = ["tiny", "small", "medium", "large"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_weights():
    # models/<run_name>/weights/best.pt
    return sorted(MODELS_DIR.glob("*/weights/best.pt"))


def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return img


def make_tiles(img, size):
    w, h = img.size
    tiles = []
    for y in range(0, h - size + 1, size):
        for x in range(0, w - size + 1, size):
            crop = img.crop((x, y, x + size, y + size))
            if size < 10:
                crop = crop.resize((32, 32), Image.NEAREST)
            tiles.append(crop)
    return tiles or [img]


def list_images(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VALID_EXT
    )


def sample_files(files, n, seed):
    if n <= 0 or len(files) <= n:
        return files
    return sorted(random.Random(seed).sample(files, n))


def score_image(model, fire_idx, path):
    tiles = make_tiles(load_image(path), PATCH_SIZE)
    scores = []
    for i in range(0, len(tiles), BATCH):
        for pred in model.predict(tiles[i : i + BATCH], verbose=False):
            scores.append(float(pred.probs.data[fire_idx]))
    hits = sum(1 for s in scores if s >= STRONG)
    mx = max(scores) if scores else 0.0
    avg = sum(scores) / len(scores) if scores else 0.0
    is_fire = hits >= MIN_HITS
    return is_fire, hits, mx, avg, len(scores)


def build_file_lists():
    lists = {}
    for scale in FIRE_SCALES:
        folder = SCALE_DIR / scale
        if not folder.is_dir():
            lists[scale] = []
            continue
        files = list_images(folder)
        lists[scale] = sample_files(files, SAMPLE, SEED)
    none_dir = SCALE_DIR / "none"
    lists["none"] = list_images(none_dir) if none_dir.is_dir() else []
    return lists


def evaluate_model(weights: Path, file_lists):
    model = YOLO(str(weights))
    fire_idx = next(i for i, n in model.names.items() if n == FIRE_CLASS)
    results = {
        "model": weights.name,
        "patch_size": PATCH_SIZE,
        "strong": STRONG,
        "min_hits": MIN_HITS,
        "sample": SAMPLE,
        "seed": SEED,
    }

    for scale in FIRE_SCALES:
        files = file_lists[scale]
        detected = missed = 0
        hits_list, max_list, avg_list = [], [], []
        for n, path in enumerate(files, 1):
            is_fire, hits, mx, avg, _ = score_image(model, fire_idx, path)
            hits_list.append(hits)
            max_list.append(mx)
            avg_list.append(avg)
            if is_fire:
                detected += 1
            else:
                missed += 1
            if n % 5 == 0 or n == len(files):
                print(f"  {scale} {n}/{len(files)}")
        total = detected + missed
        results[scale] = {
            "total": total,
            "detected": detected,
            "missed": missed,
            "fire_detection_rate": detected / total if total else 0,
            "avg_hits": sum(hits_list) / len(hits_list) if hits_list else 0,
            "avg_max": sum(max_list) / len(max_list) if max_list else 0,
            "avg_tile_score": sum(avg_list) / len(avg_list) if avg_list else 0,
        }
        print(
            f"  {scale}: {detected}/{total} "
            f"({results[scale]['fire_detection_rate']:.3f}) "
            f"avg_hits={results[scale]['avg_hits']:.1f}"
        )

    files = file_lists["none"]
    false_alarms = correct = 0
    hits_list, max_list, avg_list = [], [], []
    for n, path in enumerate(files, 1):
        is_fire, hits, mx, avg, _ = score_image(model, fire_idx, path)
        hits_list.append(hits)
        max_list.append(mx)
        avg_list.append(avg)
        if is_fire:
            false_alarms += 1
        else:
            correct += 1
        print(f"  none {path.name}: {'FALSE ALARM' if is_fire else 'ok'} hits={hits}")
    total = false_alarms + correct
    results["none"] = {
        "total": total,
        "false_alarms": false_alarms,
        "correct": correct,
        "false_alarm_rate": false_alarms / total if total else 0,
        "avg_hits": sum(hits_list) / len(hits_list) if hits_list else 0,
        "avg_max": sum(max_list) / len(max_list) if max_list else 0,
        "avg_tile_score": sum(avg_list) / len(avg_list) if avg_list else 0,
    }
    print(
        f"  none: false_alarms={false_alarms}/{total} "
        f"({results['none']['false_alarm_rate']:.3f})"
    )
    return results


def main():
    if not MODELS_DIR.is_dir():
        raise SystemExit(f"Missing {MODELS_DIR}/")

    weights_list = find_weights()
    if not weights_list:
        raise SystemExit(f"No models/*/weights/best.pt under {MODELS_DIR}/")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_lists = build_file_lists()
    print(
        f"patch={PATCH_SIZE} strong>={STRONG} min_hits={MIN_HITS} "
        f"sample={SAMPLE} seed={SEED}"
    )
    for scale in FIRE_SCALES + ["none"]:
        print(f"  {scale}: {len(file_lists[scale])} images")
    print("models:")
    for w in weights_list:
        print(f"  {w.parent.parent.name} -> {w}")

    for weights in weights_list:
        name = weights.parent.parent.name  # e.g. yolo_patch_8, train
        out = RESULTS_DIR / f"{name}_tile{PATCH_SIZE}_result.json"
        if out.exists():
            print(f"\n=== skip {name} (exists {out}) ===")
            continue
        print(f"\n=== {name} ===")
        results = evaluate_model(weights, file_lists)
        results["model"] = name
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
