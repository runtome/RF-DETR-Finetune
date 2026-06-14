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


def yolo_label_to_detections(label_path: str, img_w: int, img_h: int) -> sv.Detections:
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
        gt = yolo_label_to_detections(str(label_path), w, h)

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
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"confusion_matrix_{name}.json")
    with open(json_path, "w") as f:
        json.dump({"matrix": cm.matrix.tolist()}, f, indent=2)
    print(f"Confusion matrix JSON saved: {json_path}")

    png_path = os.path.join(output_dir, f"confusion_matrix_{name}.png")
    matrix = cm.matrix.astype(int)
    # supervision stores class names in cm.classes (n_classes entries, no background)
    class_labels = list(getattr(cm, "classes", []))
    if class_labels:
        labels = class_labels + ["Background"]
    else:
        labels = [str(i) for i in range(matrix.shape[0])]

    n = matrix.shape[0]
    fig_w = max(10, n // 2 + 2)
    fig_h = max(8, n // 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        ax=ax,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
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


# ── Inference-time metrics ────────────────────────────────────────────────────

def _det_to_tm_pred(det: sv.Detections) -> dict:
    import torch
    if len(det) == 0:
        return {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}
    confs = det.confidence if det.confidence is not None else np.ones(len(det))
    ids   = det.class_id   if det.class_id   is not None else np.zeros(len(det), dtype=int)
    return {
        "boxes":  torch.as_tensor(det.xyxy,  dtype=torch.float32),
        "scores": torch.as_tensor(confs,     dtype=torch.float32),
        "labels": torch.as_tensor(ids,       dtype=torch.long),
    }


def _det_to_tm_target(det: sv.Detections) -> dict:
    import torch
    if len(det) == 0:
        return {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}
    ids = det.class_id if det.class_id is not None else np.zeros(len(det), dtype=int)
    return {
        "boxes":  torch.as_tensor(det.xyxy, dtype=torch.float32),
        "labels": torch.as_tensor(ids,      dtype=torch.long),
    }


def compute_map_metrics(preds_tm: list, targets_tm: list, class_names: List[str]) -> dict | None:
    """Compute mAP / mAR per class using torchmetrics (installed with rfdetr)."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        print("[WARN] torchmetrics not available — mAP metrics skipped.")
        return None

    metric = MeanAveragePrecision(class_metrics=True, iou_type="bbox")
    metric.update(preds_tm, targets_tm)
    result = metric.compute()

    overall = {
        "mAP_50_95": float(result.get("map",     0)),
        "mAP_50":    float(result.get("map_50",  0)),
        "mAP_75":    float(result.get("map_75",  0)),
        "mAR_100":   float(result.get("mar_100", 0)),
    }

    map_per = result.get("map_per_class",     [])
    mar_per = result.get("mar_100_per_class", [])

    per_class = {}
    for i, cls in enumerate(class_names):
        ap = float(map_per[i]) if i < len(map_per) else -1.0
        ar = float(mar_per[i]) if i < len(mar_per) else -1.0
        per_class[cls] = {
            "AP_50_95": max(0.0, ap),
            "AR":       max(0.0, ar),
        }

    return {"overall": overall, "per_class": per_class}


def print_metrics_table(map_result: dict, cm: sv.ConfusionMatrix, class_names: List[str]) -> None:
    """Print rfdetr-style Val — Overall and Per-class metrics tables."""
    matrix = cm.matrix

    # Per-class F1 / Precision / Recall from confusion matrix
    cls_stats = {}
    tot_tp = tot_fp = tot_fn = 0.0
    for i, cls in enumerate(class_names):
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - tp)
        fn = float(matrix[i, :].sum() - tp)
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        cls_stats[cls] = {"prec": prec, "recall": recall, "f1": f1}
        tot_tp += tp; tot_fp += fp; tot_fn += fn

    ov_prec   = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    ov_recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    ov_f1     = 2 * ov_prec * ov_recall / (ov_prec + ov_recall) if (ov_prec + ov_recall) > 0 else 0.0

    ov = map_result["overall"]

    W = 72
    print("\n" + "─" * W)
    print("Val — Overall Metrics".center(W))
    print("─" * W)
    print(f"  {'mAP 50:95':^10}  {'mAP 50':^8}  {'mAP 75':^8}  {'mAR @100':^10}  {'F1':^8}  {'Prec':^8}  {'Recall':^8}")
    print("─" * W)
    print(f"  {ov['mAP_50_95']:^10.4f}  {ov['mAP_50']:^8.4f}  {ov['mAP_75']:^8.4f}"
          f"  {ov['mAR_100']:^10.4f}  {ov_f1:^8.4f}  {ov_prec:^8.4f}  {ov_recall:^8.4f}")
    print("─" * W)

    col_w = max((len(c) for c in class_names), default=8)
    PW = W
    print(f"\nVal — Per-class Metrics")
    print("─" * PW)
    print(f"  {'Class':<{col_w}}  {'AP 50:95':^10}  {'AR':^8}  {'F1':^8}  {'Precision':^10}  {'Recall':^8}")
    print("─" * PW)
    for cls in class_names:
        ap = map_result["per_class"].get(cls, {}).get("AP_50_95", 0.0)
        ar = map_result["per_class"].get(cls, {}).get("AR",       0.0)
        st = cls_stats[cls]
        print(f"  {cls:<{col_w}}  {ap:^10.4f}  {ar:^8.4f}  {st['f1']:^8.4f}"
              f"  {st['prec']:^10.4f}  {st['recall']:^8.4f}")
    print("─" * PW)
