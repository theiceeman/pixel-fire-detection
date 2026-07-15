import json
import numpy as np
from scipy import stats

DATASETS = {
    "Amazon": {
        "ResNet50": "results/resnet/amazon_scale_result.json",
        "EfficientNetV2": "scripts/efficientnet/amazon_scale_result.json",
        "MobileNetV3": "results/mobilenet/amazon_scale_result.json",
        "YOLOv8": "results/yolo/amazon_scale_result.json",
    },
    "Camp": {
        "ResNet50": "results/resnet/camp_scale_result.json",
        "EfficientNetV2": "scripts/efficientnet/camp_scale_result.json",
        "MobileNetV3": "results/mobilenet/camp_scale_result.json",
        "YOLOv8": "results/yolo/camp_scale_result.json",
    },
    "Snow": {
        "ResNet50": "results/resnet/snow_scale_result.json",
        "EfficientNetV2": "scripts/efficientnet/snow_scale_result.json",
        "MobileNetV3": "results/mobilenet/snow_scale_result.json",
        "YOLOv8": "results/yolo/snow_scale_result.json",
    },
}

SCALE_MAP = {"tiny": 1, "small": 2, "medium": 3, "large": 4}

output = {}

print("Pearson Correlation: Fire Detection Rate vs. Object Scale")
print("r = Σ(xi−x̄)(yi−ȳ) / √[Σ(xi−x̄)² · Σ(yi−ȳ)²]\n")

for env_name, models in DATASETS.items():
    scales = ["tiny", "small", "medium"] if env_name == "Snow" else ["tiny", "small", "medium", "large"]
    x = np.array([SCALE_MAP[s] for s in scales])

    print(f"{'='*58}")
    print(f"  {env_name}  (n={len(scales)} scale levels)")
    print(f"{'='*58}")
    print(f"  {'Model':<18} {'Pearson r':>10} {'p-value':>10}  {'Sig.':>5}")
    print(f"  {'-'*45}")

    output[env_name] = {}

    for model_name, path in models.items():
        with open(path) as f:
            data = json.load(f)
        y = np.array([data[s]["fire_detection_rate"] for s in scales])
        r, p = stats.pearsonr(x, y)
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
        print(f"  {model_name:<18} {r:>+10.4f} {p:>10.4f}  {sig:>5}")

        output[env_name][model_name] = {
            "pearson_r": round(r, 4),
            "p_value": round(p, 4),
            "n": len(scales),
            "detection_rates": {s: round(data[s]["fire_detection_rate"], 4) for s in scales},
        }

    print()

print("Significance: *** p<0.01, ** p<0.05, * p<0.1, n.s. = not significant\n")

with open("results/pearson_correlation.json", "w") as f:
    json.dump(output, f, indent=2)

print("Results saved to results/pearson_correlation.json")
