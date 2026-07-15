import json
import matplotlib.pyplot as plt

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

SCALES = ["tiny", "small", "medium", "large"]
SNOW_SCALES = ["tiny", "small", "medium"]

COLORS = {
    "ResNet50": "#e74c3c",
    "EfficientNetV2": "#2ecc71",
    "MobileNetV3": "#3498db",
    "YOLOv8": "#f39c12",
}
MARKERS = {"ResNet50": "o", "EfficientNetV2": "s", "MobileNetV3": "^", "YOLOv8": "D"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

for ax, (env_name, models) in zip(axes, DATASETS.items()):
    scales = SNOW_SCALES if env_name == "Snow" else SCALES

    for model_name, path in models.items():
        with open(path) as f:
            data = json.load(f)
        rates = [data[s]["fire_detection_rate"] for s in scales]
        ax.plot(
            scales, rates,
            marker=MARKERS[model_name],
            label=model_name,
            color=COLORS[model_name],
            linewidth=2.2,
            markersize=8,
        )

    ax.set_title(env_name, fontsize=14, fontweight="bold")
    ax.set_xlabel("Fire Object Scale", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3, linestyle="--")

axes[0].set_ylabel("Fire Detection Rate", fontsize=11)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Fire Detection Rate vs. Object Scale Across Environments",
    fontsize=15, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("results/scale_degradation_trend.png", dpi=200, bbox_inches="tight")
print("Saved to results/scale_degradation_trend.png")
