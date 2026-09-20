# python3 ./yolo_tile_probe.py
# Quick check: 2 tiny + 2 none. Full tile scan, fire if enough strong squares.
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "./runs/classify/train/weights/best.pt"
PATCH_SIZE = 8
MAX_SIDE = 640
BATCH = 64
STRONG = 0.7  # square counts if fire score >= this
MIN_HITS = 2  # photo = fire if this many strong squares
FIRE_CLASS = "fire"

PATHS = [
    Path("dataset/flame_scale_eval/tiny/image_335_jpg.rf.8711be2e5380deb66977ac3f4d8b9ece.jpg"),
    Path("dataset/flame_scale_eval/tiny/image_337_jpg.rf.4dab2563d64d22e6115fed1ba9dc134d.jpg"),
    Path("dataset/flame_scale_eval/none/Test8_1.jpg"),
    Path("dataset/flame_scale_eval/none/Vedio9002.jpg"),
]


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


model = YOLO(MODEL_PATH)
fire_idx = next(i for i, n in model.names.items() if n == FIRE_CLASS)

print(f"model={MODEL_PATH}  patch={PATCH_SIZE}  strong>={STRONG}  min_hits={MIN_HITS}\n")

for path in PATHS:
    if not path.exists():
        print(f"MISSING {path}")
        continue
    tiles = make_tiles(load_image(path), PATCH_SIZE)
    scores = []
    for i in range(0, len(tiles), BATCH):
        for pred in model.predict(tiles[i : i + BATCH], verbose=False):
            scores.append(float(pred.probs.data[fire_idx]))
    avg = sum(scores) / len(scores) if scores else 0.0
    mx = max(scores) if scores else 0.0
    hits = sum(1 for s in scores if s >= STRONG)
    fire = hits >= MIN_HITS
    label = "FIRE" if fire else "no fire"
    print(
        f"{path.parent.name}/{path.name}: {label}  "
        f"hits>={STRONG}: {hits}/{len(scores)}  avg={avg:.3f}  max={mx:.3f}"
    )
