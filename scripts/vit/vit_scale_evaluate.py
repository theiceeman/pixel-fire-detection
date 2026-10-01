# python3 ./scripts/vit/vit_scale_evaluate.py
# Track A: whole-image FDR on flame_scale_eval for all strategies.
import json
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_LIST = [
    ROOT / "runs" / "classify" / "vit_small" / "weights" / "best.pt",
    ROOT / "runs" / "classify" / "vit_small_transfer_forest_to_flame" / "weights" / "best.pt",
    ROOT / "runs" / "classify" / "vit_small_multiscene" / "weights" / "best.pt",
    ROOT / "runs" / "classify" / "vit_small_bcst_30" / "weights" / "best.pt",
    ROOT / "runs" / "classify" / "vit_small_bcst_60" / "weights" / "best.pt",
    ROOT / "runs" / "classify" / "vit_small_bcst_90" / "weights" / "best.pt",
    ROOT / "runs" / "classify" / "vit_small_patch_8" / "weights" / "best.pt",
]
SCALE_DIR = ROOT / "dataset" / "flame_scale_eval"
RESULTS_DIR = ROOT / "results" / "vit"
MODEL_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"
IMG_SIZE = 224
SCALES = ["tiny", "small", "medium", "large"]
FIRE_CLASS = "fire"
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def evaluate_one(weights: Path, device: torch.device) -> dict:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    results = {}
    for scale in SCALES:
        scale_dir = SCALE_DIR / scale
        detected = missed = 0
        confidence_detected, confidence_missed = [], []

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

        for path in sorted(scale_dir.iterdir()):
            if path.name.startswith(".") or path.suffix.lower() not in VALID_EXT:
                continue
            image = Image.open(path).convert("RGB")
            x = TRANSFORM(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)
                conf, pred = torch.max(probs, 1)
            confidence = float(conf.item())
            if class_names[pred.item()] == FIRE_CLASS:
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
            "avg_confidence_detected": (
                sum(confidence_detected) / len(confidence_detected)
                if confidence_detected else 0
            ),
            "avg_confidence_missed": (
                sum(confidence_missed) / len(confidence_missed)
                if confidence_missed else 0
            ),
        }
        print(
            f"  {scale}: {detected}/{total} "
            f"({results[scale]['fire_detection_rate']:.3f}), "
            f"avg conf: {results[scale]['avg_confidence']:.3f}"
        )
    return results


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    present = [w for w in WEIGHTS_LIST if w.exists()]
    missing = [w for w in WEIGHTS_LIST if not w.exists()]
    if missing:
        print("Missing weights (will skip):")
        for w in missing:
            print(f"  {w}")
    if not present:
        raise SystemExit("No ViT weights found under runs/classify/")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for weights in present:
        name = weights.parent.parent.name
        out = RESULTS_DIR / f"{name}_flame_result.json"
        if out.exists():
            print(f"\n=== skip {name} (exists {out}) ===")
            continue
        print(f"\n=== {name} ===")
        results = evaluate_one(weights, device)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
