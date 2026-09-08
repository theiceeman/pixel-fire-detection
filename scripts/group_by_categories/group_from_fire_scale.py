# python3 ./scripts/group_from_fire_scale.py
import csv
import os
import shutil

csv_path = "images/fire_scale.csv"
images_root = "images"

# CSV uses CCTV/FOREST; folders on disk are lowercase
dataset_map = {
    "CCTV": ("cctv", "dataset/cctv_scale_eval"),
    "FOREST": ("forest", "dataset/forest_scale_eval"),
}

scales = ["tiny", "small", "medium", "large"]
for _, output_dir in dataset_map.values():
    for s in scales:
        os.makedirs(os.path.join(output_dir, s), exist_ok=True)

counts = {name: {s: 0 for s in scales} for name in dataset_map}
skipped = 0

with open(csv_path) as f:
    for row in csv.DictReader(f):
        image = row["image"]  # e.g. CCTV/Vedio1001.jpg
        dataset_key, filename = image.split("/", 1)

        if dataset_key not in dataset_map:
            print(f"Warning: unknown dataset {dataset_key} for {image}")
            continue

        fire_pixels = int(row["fire_pixels"])
        if fire_pixels == 0:
            skipped += 1
            continue

        fire_ratio = float(row["fire_fraction"]) * 100

        if fire_ratio < 0.5:
            scale = "tiny"
        elif 0.5 <= fire_ratio < 2:
            scale = "small"
        elif 2 <= fire_ratio < 10:
            scale = "medium"
        else:
            scale = "large"

        folder, output_dir = dataset_map[dataset_key]
        src_path = os.path.join(images_root, folder, filename)
        dst_path = os.path.join(output_dir, scale, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            counts[dataset_key][scale] += 1
        else:
            print(f"Warning: {src_path} not found")

print(f"Skipped zero-fire: {skipped}")
for name, scale_counts in counts.items():
    print(f"{name}: {scale_counts}")
