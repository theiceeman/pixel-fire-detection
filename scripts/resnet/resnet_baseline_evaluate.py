# python3 ./scripts/resnet_baseline_evaluate.py
import json
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

WEIGHTS_PATH = "./runs/classify/resnet50/weights/best.pt"
VAL_DIR = "dataset/val"
OUTPUT_PATH = "result.json"
IMG_SIZE = 224
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
class_names = checkpoint["class_names"]
FIRE_CLASS = "fire"

model = timm.create_model("resnet50.a1_in1k", pretrained=False, num_classes=len(class_names))
model.load_state_dict(checkpoint["model"])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tp = tn = fp = fn = 0

for actual_class in [FIRE_CLASS, "non_fire"]:
    class_dir = os.path.join(VAL_DIR, actual_class)
    if not os.path.isdir(class_dir):
        print(f"Warning: {class_dir} not found, skipping")
        continue

    for filename in os.listdir(class_dir):
        if filename.startswith(".") or os.path.splitext(filename)[1].lower() not in VALID_EXT:
            continue

        filepath = os.path.join(class_dir, filename)
        image = Image.open(filepath).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        predicted_class = class_names[predicted_idx.item()]

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
