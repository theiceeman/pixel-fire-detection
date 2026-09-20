# python3 ./yolo_none_evaluate.py
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

NONE_DIR = Path("dataset/flame_scale_eval/none")
MODEL_PATH = "./runs/classify/yolo_patch_32/weights/best.pt"
PATCH_SIZE = 8  # 8, 16, or 32 — match how you tiled that JSON
MAX_SIDE = 640
BATCH = 64
THRESHOLD = 0.5
FIRE_CLASS = "fire"
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

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

model = YOLO(str(MODEL_PATH))
fire_idx = next(i for i, n in model.names.items() if n == FIRE_CLASS)
files = sorted(p for p in NONE_DIR.iterdir() if p.suffix.lower() in VALID_EXT)
false_alarms, correct = 0, 0
conf_fa, conf_ok = [], []

for path in files:
    tiles = make_tiles(load_image(path), PATCH_SIZE)
    scores = []
    for i in range(0, len(tiles), BATCH):
        for pred in model.predict(tiles[i:i+BATCH], verbose=False):
            scores.append(float(pred.probs.data[fire_idx]))
    avg = sum(scores) / len(scores) if scores else 0.0
    hit = avg >= THRESHOLD
    print(f"{path.name}: {'FALSE ALARM' if hit else 'ok'}  avg={avg:.3f}  tiles={len(scores)}")
    if hit:
        false_alarms += 1
        conf_fa.append(avg)
    else:
        correct += 1
        conf_ok.append(avg)

n = false_alarms + correct
print("\nPaste into JSON under \"none\":")
print({
    "total": n,
    "false_alarms": false_alarms,
    "correct": correct,
    "false_alarm_rate": false_alarms / n if n else 0,
    "avg_confidence": sum(conf_fa + conf_ok) / n if n else 0,
    "avg_confidence_false_alarm": sum(conf_fa) / len(conf_fa) if conf_fa else 0,
    "avg_confidence_correct": sum(conf_ok) / len(conf_ok) if conf_ok else 0,
})