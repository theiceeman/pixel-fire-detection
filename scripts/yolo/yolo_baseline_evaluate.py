# python3 ./evaluate.py
import json
import os
from ultralytics import YOLO

WEIGHTS_PATH = "./runs/classify/train/weights/best.pt"
VAL_DIR = "dataset/val"
OUTPUT_PATH = "result.json"
FIRE_CLASS = "fire"
NON_FIRE_CLASS = "non_fire"

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
model = YOLO(WEIGHTS_PATH)
tp = tn = fp = fn = 0

for actual_class in [FIRE_CLASS, NON_FIRE_CLASS]:
    class_dir = os.path.join(VAL_DIR, actual_class)

    for filename in os.listdir(class_dir):
        if filename.startswith(".") or os.path.splitext(filename)[1].lower() not in valid_ext:
            continue
        filepath = os.path.join(class_dir, filename)

        results = model.predict(filepath, verbose=False)
        predicted_class = model.names[results[0].probs.top1]

        if actual_class == FIRE_CLASS:
            if predicted_class == FIRE_CLASS:
                tp += 1
            else:
                fn += 1
        else:
            if predicted_class == FIRE_CLASS:
                fp += 1
            else:
                tn += 1

total = tp + tn + fp + fn
accuracy = (tp + tn) / total if total else 0
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
fire_detection_rate = recall
error_warning_rate = fp / (fp + tn) if (fp + tn) else 0

result = {
    "fire_detection_rate": fire_detection_rate,
    "error_warning_rate": error_warning_rate,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
