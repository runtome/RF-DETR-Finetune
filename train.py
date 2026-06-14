"""
RF-DETR fine-tuning training script.

Usage:
    python train.py
    python train.py --model rfdetr-small --epochs 20 --trial-name my_run
    python train.py --model rfdetr-medium --batch-size 2 --lr 5e-5
"""

import argparse
import glob
import os
import shutil
import sys

import torch
import yaml

from dataset_prep import prepare_dataset
from utils.config import (
    DEFAULT_MODEL, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR,
    DEFAULT_GRAD_ACCUM, DEFAULT_GRADIENT_CHECKPOINTING,
    SOURCE_DATASET, WORKING_DIR, OUTPUTS_DIR,
)
from utils.model_config import get_model_class, get_resolution
from utils.trial_utils import get_next_trial_name


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


def load_model_for_eval(model_name: str, best_path: str):
    try:
        import rfdetr
        model = rfdetr.RFDETR.from_pretrained(best_path)
        return model
    except Exception:
        pass
    ModelClass = get_model_class(model_name)
    try:
        return ModelClass(pretrain_weights=best_path)
    except Exception as e:
        print(f"[WARN] Could not load checkpoint for evaluation: {e}")
        return None


def train_main(args=None):
    parser = argparse.ArgumentParser(description="Train RF-DETR model")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--trial-name", default=None)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false", default=DEFAULT_GRADIENT_CHECKPOINTING)
    parser.add_argument("--dataset-dir", default=None,
                        help="Path to prepared dataset. If omitted, runs dataset_prep automatically.")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # ── DDP worker detection ──────────────────────────────────────────────────
    # PTL's subprocess_script launcher re-runs this entire script for each GPU
    # rank with LOCAL_RANK set in the environment. The original user-invoked
    # process has no LOCAL_RANK and acts as the orchestrator/post-trainer.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_ddp_worker = "LOCAL_RANK" in os.environ   # True for PTL-spawned subprocesses
    is_original   = not is_ddp_worker             # True only for the user-invoked process

    # ── Trial name ────────────────────────────────────────────────────────────
    # Workers read the name the original process stored in an env var so all
    # ranks use the same output directory.
    if is_ddp_worker:
        trial = os.environ.get("RF_DETR_TRIAL_NAME") or args.trial_name or "trial_unknown"
    else:
        trial = args.trial_name or get_next_trial_name(OUTPUTS_DIR, "trial")
        os.environ["RF_DETR_TRIAL_NAME"] = trial   # inherited by subprocess workers

    trial_dir      = os.path.join(OUTPUTS_DIR, trial)
    save_model_dir = os.path.join(trial_dir, "save_model")
    tensorboard_dir = os.path.join(trial_dir, "tensorboard")
    log_path       = os.path.join(trial_dir, "runninglog.txt")

    # Dirs only need creating once (original process)
    if is_original:
        os.makedirs(save_model_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)

    # TeeLogger only on original process (workers would duplicate the log)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    tee_out = tee_err = None
    if is_original:
        tee_out = TeeLogger(log_path, orig_stdout)
        tee_err = TeeLogger(log_path, orig_stderr)
        sys.stdout = tee_out
        sys.stderr = tee_err

    try:
        if is_original:
            print("=" * 60)
            print(f"Trial: {trial}")
            print(f"  model                  : {args.model}")
            print(f"  epochs                 : {args.epochs}")
            print(f"  batch_size             : {args.batch_size}")
            print(f"  lr                     : {args.lr}")
            print(f"  grad_accum             : {args.grad_accum}")
            print(f"  gradient_checkpointing : {args.gradient_checkpointing}")
            print(f"  dataset_dir            : {args.dataset_dir or '(auto)'}")
            print(f"  outputs                : {trial_dir}")
            print("=" * 60)

        # GPU info
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if is_original:
                gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
                print(f"GPU: {n_gpus}x {gpu_names}")
        else:
            n_gpus = 0
            if is_original:
                print("GPU: not available, using CPU")

        # Dataset preparation — original process only; workers compute the path
        if is_original:
            dataset_dir = args.dataset_dir or prepare_dataset(args.model, SOURCE_DATASET)
            class_names = load_class_names(dataset_dir)
            print(f"Classes ({len(class_names)}): {class_names}")
        else:
            resolution = get_resolution(args.model)
            dataset_dir = args.dataset_dir or os.path.join(WORKING_DIR, f"dataset_{resolution}")
            class_names = load_class_names(dataset_dir)

        # Model — all processes must load it (DDP requires each rank to have its own copy)
        if is_original:
            print(f"\nLoading model: {args.model}")
        ModelClass = get_model_class(args.model)
        model = ModelClass()

        # Training — all processes call this; PTL coordinates via DDP
        rfdetr_out = os.path.join(trial_dir, "rfdetr_raw")
        resolution = get_resolution(args.model)
        if is_original:
            if n_gpus > 1:
                print(f"\nStarting training with {n_gpus} GPUs  (resolution={resolution}x{resolution}) ...")
            else:
                print(f"\nStarting training  (resolution={resolution}x{resolution}) ...")

        train_kwargs = dict(
            dataset_dir=dataset_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            resolution=resolution,
            gradient_checkpointing=args.gradient_checkpointing,
            lr=args.lr,
            output_dir=rfdetr_out,
        )
        if n_gpus == 1:
            train_kwargs["devices"] = 1
        elif n_gpus > 1:
            train_kwargs["devices"] = n_gpus
            # RF-DETR has params not active in every forward pass — required for multi-GPU DDP
            train_kwargs["strategy"] = "ddp_find_unused_parameters_true"

        model.train(**train_kwargs)

        # ── Post-training ─────────────────────────────────────────────────────
        # Runs ONLY on the original process, after all DDP workers have finished.
        # Workers exit after model.train() returns; only the original process
        # reaches this block.
        if is_original:
            print("\nTraining complete.")

            # Copy checkpoints
            for ckpt_name in [
                "checkpoint_best_total.pth",
                "checkpoint_best_regular.pth",
                "checkpoint_best_ema.pth",
            ]:
                src = os.path.join(rfdetr_out, ckpt_name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(save_model_dir, ckpt_name))
                    print(f"Checkpoint saved: {os.path.join(save_model_dir, ckpt_name)}")

            # Copy TensorBoard events
            for evt in glob.glob(os.path.join(rfdetr_out, "events.out.tfevents.*")):
                shutil.copy2(evt, tensorboard_dir)
                print(f"TensorBoard event copied: {os.path.basename(evt)}")
            for evt in glob.glob(os.path.join(rfdetr_out, "**", "events.out.tfevents.*"),
                                  recursive=True):
                shutil.copy2(evt, tensorboard_dir)

            # Plot loss / mAP curves from CSV
            metrics_csv = os.path.join(rfdetr_out, "metrics.csv")
            from utils.metrics import plot_metrics_from_csv
            plot_metrics_from_csv(metrics_csv, trial_dir, trial)

            # Confusion matrix on validation split
            best_path = os.path.join(save_model_dir, "checkpoint_best_total.pth")
            if os.path.exists(best_path):
                print("\nRunning confusion matrix on validation set ...")
                eval_model = load_model_for_eval(args.model, best_path)
                if eval_model is not None:
                    from utils.metrics import compute_confusion_matrix, save_confusion_matrix
                    cm = compute_confusion_matrix(
                        eval_model,
                        image_dir=os.path.join(dataset_dir, "valid", "images"),
                        label_dir=os.path.join(dataset_dir, "valid", "labels"),
                        class_names=class_names,
                    )
                    save_confusion_matrix(cm, trial_dir, trial)
            else:
                print(f"[WARN] Best checkpoint not found at {best_path}; skipping confusion matrix.")

            print("\n" + "=" * 60)
            print(f"All outputs saved to: {trial_dir}")
            print(f"  Weights    : {save_model_dir}")
            print(f"  TensorBoard: {tensorboard_dir}  →  tensorboard --logdir {tensorboard_dir}")
            print(f"  Log        : {log_path}")
            print("=" * 60)

    finally:
        if is_original and tee_out:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            tee_out.close()
            tee_err.close()

    return trial


if __name__ == "__main__":
    train_main()
