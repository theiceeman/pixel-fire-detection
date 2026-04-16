import json
import os
import shutil

coco_json_path = "images/_annotations.coco.json"     
images_dir = "images"              
output_dir = "dataset/eval"         

# load JSON
with open(coco_json_path) as f:
    coco = json.load(f)

# create a dictionary
image_info = {img['id']: img for img in coco['images']}

# store images by their largest bounding box
image_dataset = {}
for annotation in coco['annotations']:
  img_id = annotation['image_id']
  area = annotation['area']

  if img_id not in image_dataset or area > image_dataset[img_id]['area']:
      image_dataset[img_id] = annotation

# loop through annotations 
for annotation in image_dataset.values():
    img_id = annotation['image_id']
    img = image_info[img_id]
    filename = img['file_name']
    width = img['width']
    height = img['height']
    image_area = width * height

    fire_area = annotation['area']

    # compute percentage of fire in image
    fire_ratio = (fire_area / image_area) * 100

    # determine scale
    if fire_ratio < 0.5:
        scale = "tiny"
    elif 0.5 <= fire_ratio < 2:
        scale = "small"
    elif 2 <= fire_ratio < 10:
        scale = "medium"
    else:
        scale = "large"

    # copy image to the appropriate folder
    src_path = os.path.join(images_dir, filename)
    dst_path = os.path.join(output_dir, scale, filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"Warning: {src_path} not found")