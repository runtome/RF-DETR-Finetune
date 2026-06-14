"""
Prepare dataset for RF-DETR training.
- Train images are resized to the model's target resolution.
- Valid/test images are copied as-is.
- All labels are copied unchanged.
- data.yaml is copied for class name reference.

Usage:
    python dataset_prep.py [--model rfdetr-nano] [--source /path/to/dataset]
"""

import argparse
import os
import shutil

import cv2
from tqdm import tqdm

from utils.config import SOURCE_DATASET, WORKING_DIR, DEFAULT_MODEL
from utils.model_config import get_resolution


def prepare_dataset(model_name: str = DEFAULT_MODEL, source: str = SOURCE_DATASET) -> str:
    resolution = get_resolution(model_name)
    dst = os.path.join(WORKING_DIR, f"dataset_{resolution}")

    if os.path.exists(os.path.join(dst, "data.yaml")):
        print(f"[dataset_prep] Already prepared at {dst}, skipping.")
        return dst

    print(f"[dataset_prep] Preparing dataset → {dst}  (model={model_name}, resolution={resolution})")
    os.makedirs(dst, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png"}

    for split in ["train", "valid", "test"]:
        img_src = os.path.join(source, split, "images")
        lbl_src = os.path.join(source, split, "labels")
        img_dst = os.path.join(dst, split, "images")
        lbl_dst = os.path.join(dst, split, "labels")

        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        if not os.path.isdir(img_src):
            print(f"[dataset_prep] Source not found: {img_src}, skipping split.")
            continue

        img_files = [f for f in os.listdir(img_src) if os.path.splitext(f)[1].lower() in image_exts]
        print(f"  {split}: {len(img_files)} images")

        for img_file in tqdm(img_files, desc=f"  {split}", leave=False):
            src_path = os.path.join(img_src, img_file)
            dst_path = os.path.join(img_dst, img_file)

            if split == "train":
                img = cv2.imread(src_path)
                if img is None:
                    print(f"[WARN] Cannot read {src_path}, skipping.")
                    continue
                img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
                cv2.imwrite(dst_path, img)
            else:
                shutil.copy2(src_path, dst_path)

        if os.path.isdir(lbl_src):
            for lbl_file in os.listdir(lbl_src):
                shutil.copy2(
                    os.path.join(lbl_src, lbl_file),
                    os.path.join(lbl_dst, lbl_file),
                )

    yaml_src = os.path.join(source, "data.yaml")
    if os.path.exists(yaml_src):
        shutil.copy2(yaml_src, os.path.join(dst, "data.yaml"))

    print(f"[dataset_prep] Done → {dst}")
    return dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RF-DETR dataset")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name (determines resize resolution)")
    parser.add_argument("--source", default=SOURCE_DATASET, help="Source dataset directory")
    args = parser.parse_args()

    path = prepare_dataset(args.model, args.source)
    print(f"Dataset ready at: {path}")
