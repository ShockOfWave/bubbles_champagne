import os
from glob import glob
from tqdm import tqdm
import shutil 

classes = glob("frames/*")

for cls in classes:
    cls_name = cls.split("/")[-1]
    os.makedirs(os.path.join("data/train", cls_name), exist_ok=True)
    os.makedirs(os.path.join("data/val", cls_name), exist_ok=True)

for cls in tqdm(classes, desc="Splitting images on train/val: "):
    images = glob(f"{cls}/*")
    names = set([img.split('/')[-1].split('.')[0].split("_")[0] for img in images])
    for i, image_name in enumerate(names):
        if i % 3 != 0:
            images = glob(f"{cls}/{image_name}*")
            for image in images:
                shutil.copy(image, os.path.join("data", "train", cls.split('/')[1], os.path.basename(image)))
        else:
            images = glob(f"{cls}/{image_name}*.jpg")
            for image in images:
                shutil.copy(image, os.path.join("data", "val", cls.split('/')[1], os.path.basename(image)))