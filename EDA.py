"""
Exploratory Data Analysis — display sample images with bounding boxes.

Usage:
    python EDA.py --split train --sample 5
    python EDA.py --split test  --sample 5 --class Car
"""

import argparse
import ast
import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from utils.config import SOURCE_DATASET


def load_class_names(dataset_dir: str):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return names


def draw_yolo_boxes(image, label_path: str, class_names):
    h, w = image.shape[:2]
    try:
        with open(label_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return image

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    ]

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = colors[cls_id % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        cv2.putText(image, label, (x1, max(20, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return image


def run_eda(split: str, n_samples: int, class_filter: str | None, dataset_dir: str):
    class_names = load_class_names(dataset_dir)

    manifest_path = os.path.join(dataset_dir, "images_manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"[EDA] Manifest not found at {manifest_path}. Falling back to directory scan.")
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        exts = {".jpg", ".jpeg", ".png"}
        all_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
        import random
        sampled = random.sample(all_files, min(n_samples, len(all_files)))
        rows = [
            {"image_filename": f, "label_filename": os.path.splitext(f)[0] + ".txt"}
            for f in sampled
        ]
    else:
        df = pd.read_csv(manifest_path)
        mask = df["split"] == split
        if class_filter:
            mask &= df["class_names_present"].str.contains(class_filter, case=False, na=False)
        filtered = df[mask]
        if filtered.empty:
            print(f"[EDA] No images found for split='{split}'"
                  + (f", class='{class_filter}'" if class_filter else "") + ".")
            return
        sampled_df = filtered.sample(min(n_samples, len(filtered)), random_state=None)
        rows = sampled_df.to_dict("records")

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        img_path = os.path.join(dataset_dir, split, "images", row["image_filename"])
        lbl_path = os.path.join(dataset_dir, split, "labels", row["label_filename"])

        img = cv2.imread(img_path)
        if img is None:
            ax.set_title("Cannot read image")
            ax.axis("off")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = draw_yolo_boxes(img, lbl_path, class_names)
        ax.imshow(img)
        ax.set_title(row["image_filename"], fontsize=8)
        ax.axis("off")

    title = f"Split: {split}"
    if class_filter:
        title += f"  |  Class: {class_filter}"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    save_name = f"EDA_{split}" + (f"_{class_filter}" if class_filter else "") + f"_n{n}.png"
    plt.savefig(save_name, bbox_inches="tight", dpi=150)
    print(f"Saved: {save_name}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA: visualise dataset samples with bounding boxes")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"],
                        help="Dataset split to sample from")
    parser.add_argument("--sample", type=int, default=5, help="Number of images to display")
    parser.add_argument("--class", dest="class_filter", default=None,
                        help="Filter images containing this class name")
    parser.add_argument("--source", default=SOURCE_DATASET, help="Source dataset directory")
    args = parser.parse_args()

    run_eda(args.split, args.sample, args.class_filter, args.source)
