"""
RF-DETR full pipeline orchestrator.

Runs: dataset_prep → train → inference --validate valid

Usage:
    python main.py
    python main.py --model rfdetr-small --epochs 20 --trial-name my_exp
    python main.py --model rfdetr-nano  --batch-size 2
"""

import argparse
import sys

from dataset_prep import prepare_dataset
from utils.config import (
    DEFAULT_MODEL, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR,
    DEFAULT_GRAD_ACCUM, DEFAULT_GRADIENT_CHECKPOINTING,
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD,
    SOURCE_DATASET, OUTPUTS_DIR,
)


def main():
    parser = argparse.ArgumentParser(
        description="RF-DETR full pipeline: dataset_prep → train → inference (validate)"
    )
    # Training args
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--trial-name", default=None)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false", default=DEFAULT_GRADIENT_CHECKPOINTING)
    parser.add_argument("--dataset-dir", default=None,
                        help="Skip dataset_prep and use this directory directly")
    # Inference args
    parser.add_argument("--validate", default="valid", choices=["train", "valid", "test"],
                        help="Split to evaluate after training (default: valid)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--source", default=SOURCE_DATASET)

    args = parser.parse_args()

    print("=" * 60)
    print("RF-DETR Pipeline")
    print(f"  Step 1: Dataset Preparation")
    print(f"  Step 2: Training  ({args.model}, {args.epochs} epochs)")
    print(f"  Step 3: Inference / Evaluation  (--validate {args.validate})")
    print("=" * 60)

    # Step 1: Dataset prep
    dataset_dir = args.dataset_dir or prepare_dataset(args.model, args.source)

    # Step 2: Train
    from train import train_main

    train_argv = [
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--grad-accum", str(args.grad_accum),
        "--dataset-dir", dataset_dir,
    ]
    if args.trial_name:
        train_argv += ["--trial-name", args.trial_name]
    if not args.gradient_checkpointing:
        train_argv.append("--no-gradient-checkpointing")

    trial = train_main(train_argv)

    # Step 3: Inference / Evaluation
    from inference import run_inference

    inf_argv = [
        "--validate", args.validate,
        "--threshold", str(args.threshold),
        "--iou-threshold", str(args.iou_threshold),
        "--source", args.source,
        "--name", f"{trial}_eval",
    ]

    inf_name = run_inference(inf_argv)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Training outputs : {OUTPUTS_DIR}/{trial}/")
    print(f"  Eval outputs     : {OUTPUTS_DIR}/{inf_name}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
