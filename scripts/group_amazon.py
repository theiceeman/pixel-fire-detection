# python3 ./group_amazon.py
import json
import os
import shutil

coco_json_path = "images/amazon/_annotations.coco.json"
images_dir = "images/amazon"
output_dir = "dataset/amazon_scale_eval"

with open(coco_json_path) as f:
    coco = json.load(f)

image_info = {img['id']: img for img in coco['images']}

# keep largest fire-only annotation per image (category_id=1 is fire, 2 is smoke)
FIRE_CATEGORY_ID = 1
image_dataset = {}
for annotation in coco['annotations']:
    if annotation['category_id'] != FIRE_CATEGORY_ID:
        continue
    img_id = annotation['image_id']
    area = annotation['area']
    if img_id not in image_dataset or area > image_dataset[img_id]['area']:
        image_dataset[img_id] = annotation

for annotation in image_dataset.values():
    img_id = annotation['image_id']
    img = image_info[img_id]
    filename = img['file_name']
    width = img['width']
    height = img['height']
    image_area = width * height

    fire_area = annotation['area']
    fire_ratio = (fire_area / image_area) * 100

    if fire_ratio < 0.5:
        scale = "tiny"
    elif 0.5 <= fire_ratio < 2:
        scale = "small"
    elif 2 <= fire_ratio < 10:
        scale = "medium"
    else:
        scale = "large"

    src_path = os.path.join(images_dir, filename)
    dst_path = os.path.join(output_dir, scale, filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"Warning: {src_path} not found")
