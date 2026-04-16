# python3 ./scale_evaluate.py
import json
import os
from ultralytics import YOLO

WEIGHTS_PATH = "./runs/classify/train/weights/best.pt"
SCALE_DIR = "dataset/camp_scale_eval"
OUTPUT_PATH = "scale_result.json"
FIRE_CLASS = "fire"
SCALES = ["tiny", "small", "medium", "large"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

model = YOLO(WEIGHTS_PATH)
results = {}

for scale in SCALES:
    scale_dir = os.path.join(SCALE_DIR, scale)
    detected = 0
    missed = 0
    confidence_detected = []
    confidence_missed = []
    
    if not os.path.isdir(scale_dir):
        results[scale] = {"total": 0, "detected": 0, "missed": 0, "fire_detection_rate": 0, "avg_confidence": 0, "avg_confidence_detected": 0, "avg_confidence_missed": 0}
        continue

    # loop through files in each scale dir
    for filename in os.listdir(scale_dir):
        if filename.startswith(".") or os.path.splitext(filename)[1].lower() not in VALID_EXT:
            continue

        filepath = os.path.join(scale_dir, filename)
        preds = model.predict(filepath, verbose=False)
        predicted_class = model.names[preds[0].probs.top1]
        confidence = float(preds[0].probs.top1conf)

        if predicted_class == FIRE_CLASS:
            detected += 1
            confidence_detected.append(confidence)
        else:
            missed += 1
            confidence_missed.append(confidence)

    total = detected + missed
    all_conf = confidence_detected + confidence_missed

    results[scale] = {
        "total": total,
        "detected": detected,
        "missed": missed,
        "fire_detection_rate": detected / total if total else 0,
        "avg_confidence": sum(all_conf) / len(all_conf) if all_conf else 0,
        "avg_confidence_detected": sum(confidence_detected) / len(confidence_detected) if confidence_detected else 0,
        "avg_confidence_missed": sum(confidence_missed) / len(confidence_missed) if confidence_missed else 0,
    }

    print(f"{scale}: {detected}/{total} detected ({results[scale]['fire_detection_rate']:.3f}), avg conf: {results[scale]['avg_confidence']:.3f}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_PATH}")
