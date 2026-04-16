# python3 ./scripts/mobilenet_scale_evaluate.py
import json
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

WEIGHTS_PATH = "./runs/classify/mobilenetv4/weights/best.pt"
SCALE_DIR = "dataset/snow_scale_eval"
OUTPUT_PATH = "scale_result.json"
IMG_SIZE = 224
SCALES = ["tiny", "small", "medium", "large"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
class_names = checkpoint["class_names"]
FIRE_CLASS = "fire"

model = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=False, num_classes=len(class_names))
model.load_state_dict(checkpoint["model"])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

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

    for filename in os.listdir(scale_dir):
        if filename.startswith(".") or os.path.splitext(filename)[1].lower() not in VALID_EXT:
            continue

        filepath = os.path.join(scale_dir, filename)
        image = Image.open(filepath).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            conf, predicted_idx = torch.max(probs, 1)

        predicted_class = class_names[predicted_idx.item()]
        confidence = float(conf.item())

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
