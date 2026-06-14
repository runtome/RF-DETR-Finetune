# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

CLI codebase for fine-tuning RF-DETR object detection models, designed to run in Kaggle notebooks (T4 GPU, `/kaggle/working` as working dir). Target dataset: 25-class obstacle detection, YOLO format.

## Common Commands

```bash
# Full pipeline (dataset prep → train → validate)
python main.py
python main.py --model rfdetr-large --epochs 30 --batch-size 4 --grad-accum 4

# Individual steps
python dataset_prep.py --model rfdetr-nano
python train.py --model rfdetr-nano --epochs 2 --batch-size 8 --grad-accum 2
python inference.py --validate test
python inference.py --file /path/to/image.jpg
python EDA.py --split test --sample 5 --class Car

# Install (Kaggle: torch/torchvision are pre-installed — do not add them to requirements.txt)
pip install -r requirements.txt
# For xlarge / 2xlarge models only:
pip install rfdetr[plus]
```

## Architecture

All scripts are independently runnable AND importable as modules. `main.py` calls `prepare_dataset()`, `train_main()`, and `run_inference()` by importing from the other scripts.

**Shared config** (`utils/config.py`) — single source of truth for all defaults. Every script reads defaults from here; CLI args override at runtime.

**Model registry** (`utils/model_config.py`) — maps CLI model names (e.g. `rfdetr-nano`) to rfdetr class names and input resolutions. `get_model_class(name)` imports rfdetr lazily and returns the class.

**Trial naming** (`utils/trial_utils.py`) — `get_next_trial_name("outputs", "trial")` scans `outputs/` for `trial_NN` dirs and returns the next zero-padded name. `find_latest_checkpoint("outputs")` globs `outputs/trial_*/save_model/*.pth` and returns the most recently modified.

**Metrics** (`utils/metrics.py`) — contains: confusion matrix (seaborn heatmap, `save_confusion_matrix`), loss/mAP curve plots (`plot_metrics_from_csv`), torchmetrics mAP computation (`compute_map_metrics`), and metrics table printer (`print_metrics_table`). Public helpers `yolo_label_to_detections`, `_det_to_tm_pred`, `_det_to_tm_target` convert between YOLO labels and supervision/torchmetrics formats.

## Key Behaviours

**DDP (multi-GPU):** `train.py` detects `"LOCAL_RANK" in os.environ` to distinguish PTL-spawned subprocess workers from the original process. The original process stores the trial name in `os.environ["RF_DETR_TRIAL_NAME"]` before `model.train()` so all worker subprocesses inherit the same output path. Post-training file operations (checkpoint copy, plots, confusion matrix) are guarded by `is_original = not is_ddp_worker`. Strategy is `"ddp_find_unused_parameters_true"` (required — RF-DETR transformers have params inactive in some forward passes).

**TeeLogger:** Both `train.py` and `inference.py` redirect `sys.stdout`/`sys.stderr` to a `TeeLogger` that writes to both terminal and a `runninglog.txt` / `outputs_log_*.txt` file. Always restored in a `finally` block.

**Inference metrics pass:** `inference.py` collects `sv.Detections` predictions and ground-truth in the main loop (no second inference pass). After the loop it calls `sv.ConfusionMatrix.from_detections()` from the collected lists. Timing summary always prints; mAP table and confusion matrix only print in `--validate` mode.

**Checkpoint file name:** rfdetr saves `checkpoint_best_total.pth` (not `best.ckpt`). `train.py` copies this to `outputs/{trial}/save_model/`.

**rfdetr quirks:**
- `model.evaluate()` does not exist — use `model.predict()` loop + supervision
- `model.predict(img_rgb, threshold=...)` returns `sv.Detections` directly (no `.from_inference()` needed)
- `callbacks={}` is silently discarded — do not pass it
- rfdetr ignores `path:` entries in `data.yaml`; it only uses the file for class names

## Output Layout

```
outputs/
├── trial_01/
│   ├── runninglog.txt
│   ├── save_model/checkpoint_best_total.pth
│   ├── rfdetr_raw/metrics.csv + events.out.tfevents.*
│   ├── tensorboard/events.out.tfevents.*
│   ├── confusion_matrix_trial_01.{json,png}
│   ├── loss_trial_01.png
│   └── mAP_trial_01.png
├── inference_01/
│   ├── outputs_log_inference_01.txt
│   ├── inference_time_inference_01.csv
│   ├── inference_image/*.jpg
│   ├── confusion_matrix_inference_01.{json,png}   # --validate only
│   └── metrics_inference_01.json                  # --validate only
└── EDA_01/
    └── EDA_{split}_{class}_n{N}.png
```
