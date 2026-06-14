"""
RF-DETR inference script.

Usage:
    python inference.py --file image.jpg
    python inference.py --file /path/to/folder --name my_run
    python inference.py --validate valid
    python inference.py --validate test --model outputs/trial_01/save_model/checkpoint_best_total.pth
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml
from tqdm import tqdm

from utils.config import (
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD,
    SOURCE_DATASET, OUTPUTS_DIR,
)
from utils.trial_utils import get_next_trial_name, find_latest_checkpoint


class TeeLogger:
    def __init__(self, filepath, original_stream):
        self.terminal = original_stream
        self.log = open(filepath, "a", buffering=1, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    def isatty(self):
        return False


def load_class_names(dataset_dir: str):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return names


def load_model(ckpt_path: str, model_name: str | None = None):
    print(f"Loading checkpoint: {ckpt_path}")
    try:
        import rfdetr
        model = rfdetr.RFDETR.from_pretrained(ckpt_path)
        return model
    except Exception:
        pass

    if model_name:
        from utils.model_config import get_model_class
        ModelClass = get_model_class(model_name)
        return ModelClass(pretrain_weights=ckpt_path)

    from utils import model_config
    for cfg in model_config.MODEL_CONFIGS.values():
        try:
            import rfdetr as _r
            cls = getattr(_r, cfg["class"])
            return cls(pretrain_weights=ckpt_path)
        except Exception:
            continue
    raise RuntimeError(
        f"Could not load checkpoint {ckpt_path}. "
        "Pass --model-name to specify the model variant."
    )


def annotate_image(img_bgr, detections: sv.Detections, class_names):
    box_ann   = sv.BoxAnnotator()
    label_ann = sv.LabelAnnotator()
    labels = []
    if detections.class_id is not None and detections.confidence is not None:
        for cls_id, conf in zip(detections.class_id, detections.confidence):
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            labels.append(f"{name} {conf:.2f}")
    annotated = box_ann.annotate(scene=img_bgr.copy(), detections=detections)
    annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)
    return annotated


def collect_images(file_arg: str | None, validate_split: str | None, source: str):
    exts = {".jpg", ".jpeg", ".png"}
    if validate_split:
        img_dir = Path(source) / validate_split / "images"
        lbl_dir = Path(source) / validate_split / "labels"
        files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
        return files, lbl_dir
    path = Path(file_arg)
    if path.is_file():
        return [path], None
    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in exts])
        return files, None
    raise FileNotFoundError(f"--file path not found: {file_arg}")


def _print_timing_summary(time_records: list) -> None:
    if not time_records:
        return
    all_ms = [float(r["inference_time_ms"]) for r in time_records]
    print(f"\nInference timing ({len(all_ms)} images):")
    print(f"  1st image (warmup) : {all_ms[0]:.2f} ms")
    if len(all_ms) > 1:
        rest = all_ms[1:]
        avg  = sum(rest) / len(rest)
        print(f"  Avg (excl. warmup) : {avg:.2f} ms/image  ({len(rest)} images)")
        print(f"  FPS (avg)          : {1000 / avg:.1f} fps")
    else:
        print("  (only 1 image — warmup not excluded)")


def run_inference(args=None):
    parser = argparse.ArgumentParser(description="RF-DETR inference")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", default=None, help="Image file or folder path")
    group.add_argument("--validate", default=None, choices=["train", "valid", "test"],
                       help="Evaluate on a dataset split")
    parser.add_argument("--split", default=None, dest="validate",
                        choices=["train", "valid", "test"],
                        help="Alias for --validate")
    parser.add_argument("--name",       default=None, help="Output folder name (auto if omitted)")
    parser.add_argument("--model",      default=None, help="Checkpoint path (auto-finds latest if omitted)")
    parser.add_argument("--model-name", default=None, help="rfdetr model variant (e.g. rfdetr-nano)")
    parser.add_argument("--threshold",     type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--source", default=SOURCE_DATASET, help="Source dataset directory")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if args.file is None and args.validate is None:
        parser.error("Provide either --file or --validate.")

    name        = args.name or get_next_trial_name(OUTPUTS_DIR, "inference")
    out_dir     = os.path.join(OUTPUTS_DIR, name)
    img_out_dir = os.path.join(out_dir, "inference_image")
    log_path    = os.path.join(out_dir, f"outputs_log_{name}.txt")
    csv_path    = os.path.join(out_dir, f"inference_time_{name}.csv")

    os.makedirs(img_out_dir, exist_ok=True)

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    tee_out = TeeLogger(log_path, orig_stdout)
    tee_err = TeeLogger(log_path, orig_stderr)
    sys.stdout = tee_out
    sys.stderr = tee_err

    try:
        print("=" * 60)
        print(f"Inference run: {name}")
        print(f"  file       : {args.file}")
        print(f"  validate   : {args.validate}")
        print(f"  model ckpt : {args.model or '(auto-find latest)'}")
        print(f"  threshold  : {args.threshold}")
        print(f"  iou_thresh : {args.iou_threshold}")
        print(f"  outputs    : {out_dir}")
        print("=" * 60)

        import torch
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            print(f"GPU: {n_gpu}x {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}")
        else:
            print("GPU: not available, using CPU")

        # Load model
        ckpt_path = args.model or find_latest_checkpoint(OUTPUTS_DIR)
        if ckpt_path is None:
            raise FileNotFoundError(
                "No checkpoint found in outputs/trial_*/save_model/. "
                "Pass --model /path/to/checkpoint.pth explicitly."
            )
        model = load_model(ckpt_path, args.model_name)

        # Class names
        try:
            class_names = load_class_names(args.source)
        except Exception:
            class_names = getattr(model, "class_names", [])
        print(f"Classes: {len(class_names)}")

        # Collect image list
        image_files, label_dir = collect_images(args.file, args.validate, args.source)
        print(f"Images to process: {len(image_files)}")

        # Import label parser for --validate mode
        do_metrics = args.validate is not None and label_dir is not None
        if do_metrics:
            from utils.metrics import yolo_label_to_detections, _det_to_tm_pred, _det_to_tm_target
            preds_sv:   list = []
            targets_sv: list = []
            preds_tm:   list = []
            targets_tm: list = []

        time_records: list = []

        # ── Main inference loop ───────────────────────────────────────────────
        for img_path in tqdm(image_files, desc="Inference", unit="img"):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"[WARN] Cannot read {img_path}, skipping.")
                continue
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            detections = model.predict(img_rgb, threshold=args.threshold)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            annotated = annotate_image(img_bgr, detections, class_names)
            cv2.imwrite(os.path.join(img_out_dir, img_path.name), annotated)
            time_records.append({"filename": img_path.name, "inference_time_ms": f"{elapsed_ms:.2f}"})

            # Collect for metrics (--validate only)
            if do_metrics:
                gt = yolo_label_to_detections(
                    str(label_dir / (img_path.stem + ".txt")), w, h
                )
                preds_sv.append(detections)
                targets_sv.append(gt)
                preds_tm.append(_det_to_tm_pred(detections))
                targets_tm.append(_det_to_tm_target(gt))

        # ── Timing summary (always printed) ──────────────────────────────────
        _print_timing_summary(time_records)

        # Write CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "inference_time_ms"])
            writer.writeheader()
            writer.writerows(time_records)
        print(f"Inference time CSV: {csv_path}")

        # ── Validation metrics (--validate only) ─────────────────────────────
        if do_metrics and preds_sv:
            print(f"\nComputing metrics on split '{args.validate}' ...")

            # Confusion matrix (no second inference pass — uses collected lists)
            cm = sv.ConfusionMatrix.from_detections(
                predictions=preds_sv,
                targets=targets_sv,
                classes=class_names,
                conf_threshold=args.threshold,
                iou_threshold=args.iou_threshold,
            )
            from utils.metrics import save_confusion_matrix, compute_map_metrics, print_metrics_table
            save_confusion_matrix(cm, out_dir, name)

            # mAP table
            map_result = compute_map_metrics(preds_tm, targets_tm, class_names)
            if map_result:
                print_metrics_table(map_result, cm, class_names)

                # Build combined metrics JSON
                matrix = cm.matrix
                metrics_data: dict = {"overall": map_result["overall"], "per_class": {}}
                for i, cls in enumerate(class_names):
                    tp = float(matrix[i, i])
                    fp = float(matrix[:, i].sum() - tp)
                    fn = float(matrix[i, :].sum() - tp)
                    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
                    metrics_data["per_class"][cls] = {
                        "AP_50_95":  map_result["per_class"].get(cls, {}).get("AP_50_95", 0.0),
                        "AR":        map_result["per_class"].get(cls, {}).get("AR",       0.0),
                        "F1":        round(f1,     4),
                        "precision": round(prec,   4),
                        "recall":    round(recall, 4),
                    }
                metrics_path = os.path.join(out_dir, f"metrics_{name}.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics_data, f, indent=2)
                print(f"Metrics JSON: {metrics_path}")

        print("\n" + "=" * 60)
        print(f"Done. Outputs saved to: {out_dir}")
        print(f"  Annotated images : {img_out_dir}")
        print(f"  Timing CSV       : {csv_path}")
        print(f"  Log              : {log_path}")
        print("=" * 60)

    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        tee_out.close()
        tee_err.close()

    return name


if __name__ == "__main__":
    run_inference()
