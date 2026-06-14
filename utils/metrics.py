import json
import os
from pathlib import Path
from typing import List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import supervision as sv


def _yolo_label_to_detections(label_path: str, img_w: int, img_h: int) -> sv.Detections:
    boxes, class_ids = [], []
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (xc - bw / 2) * img_w
                y1 = (yc - bh / 2) * img_h
                x2 = (xc + bw / 2) * img_w
                y2 = (yc + bh / 2) * img_h
                boxes.append([x1, y1, x2, y2])
                class_ids.append(cls)
    except FileNotFoundError:
        pass
    if boxes:
        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )
    return sv.Detections.empty()


def compute_confusion_matrix(
    model,
    image_dir: str,
    label_dir: str,
    class_names: List[str],
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> sv.ConfusionMatrix:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])

    predictions_list, targets_list = [], []
    for img_path in image_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Cannot read {img_path}, skipping.")
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            preds = model.predict(img_rgb, threshold=conf_threshold)
        except Exception as e:
            print(f"[WARN] predict failed on {img_path}: {e}")
            preds = sv.Detections.empty()

        label_path = label_dir / (img_path.stem + ".txt")
        gt = _yolo_label_to_detections(str(label_path), w, h)

        predictions_list.append(preds)
        targets_list.append(gt)

    return sv.ConfusionMatrix.from_detections(
        predictions=predictions_list,
        targets=targets_list,
        classes=class_names,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


def save_confusion_matrix(cm: sv.ConfusionMatrix, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"confusion_matrix_{name}.json")
    with open(json_path, "w") as f:
        json.dump({"matrix": cm.matrix.tolist()}, f, indent=2)
    print(f"Confusion matrix JSON saved: {json_path}")

    png_path = os.path.join(output_dir, f"confusion_matrix_{name}.png")
    fig = cm.plot(show_in_notebook=False)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Confusion matrix PNG saved: {png_path}")


def plot_metrics_from_csv(csv_path: str, output_dir: str, name: str) -> None:
    if not os.path.exists(csv_path):
        print(f"[WARN] metrics.csv not found at {csv_path}, skipping plots.")
        return
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        print("[WARN] 'epoch' column missing in metrics.csv, skipping plots.")
        return

    by_epoch = df.groupby("epoch").mean(numeric_only=True)

    # Loss curve
    loss_cols = [c for c in ["train/loss", "val/loss"] if c in by_epoch.columns]
    if loss_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in loss_cols:
            ax.plot(by_epoch.index, by_epoch[col], label=col)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss — {name}")
        ax.legend()
        ax.grid(True)
        path = os.path.join(output_dir, f"loss_{name}.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Loss curve saved: {path}")

    # mAP curve
    map_cols = [c for c in ["val/mAP_50_95", "val/ema_mAP_50_95", "val/mAP_50"] if c in by_epoch.columns]
    if map_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in map_cols:
            ax.plot(by_epoch.index, by_epoch[col], label=col)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.set_title(f"mAP — {name}")
        ax.legend()
        ax.grid(True)
        path = os.path.join(output_dir, f"mAP_{name}.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"mAP curve saved: {path}")
