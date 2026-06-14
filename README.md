# RF-DETR Finetune

Fine-tune [RF-DETR](https://github.com/roboflow/rf-detr) object detection models on custom datasets in Kaggle notebooks.

---

## Quick Start (Kaggle)

```python
%cd /kaggle/working
!git clone https://github.com/runtome/RF-DETR-Finetune.git
%cd RF-DETR-Finetune
!pip install -r requirements.txt

# Full pipeline: dataset prep → train → evaluate
!python main.py
```

---

## Dataset

| Field | Value |
|---|---|
| Source path | `/kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset` |
| Format | YOLO (`.txt` labels) |
| Classes | 25 obstacle categories |
| Total images | 24,326 (train 19,186 / valid 3,511 / test 1,629) |

---

## Supported Models

| Model name | Class | Input size |
|---|---|---|
| `rfdetr-nano` | RFDETRNano | 384×384 |
| `rfdetr-small` | RFDETRSmall | 512×512 |
| `rfdetr-medium` | RFDETRMedium | 576×576 |
| `rfdetr-large` | RFDETRLarge | 704×704 |
| `rfdetr-xlarge` | RFDETRXLarge | 700×700 |
| `rfdetr-2xlarge` | RFDETR2XLarge | 880×880 |

Default is **`rfdetr-nano`** (384×384). All defaults live in `utils/config.py`.

---

## Scripts

### `main.py` — Full Pipeline

Runs dataset preparation → training → validation inference in one command.

```bash
python main.py
python main.py --model rfdetr-small --epochs 20 --trial-name exp01
python main.py --model rfdetr-medium --batch-size 2 --lr 5e-5
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `rfdetr-nano` | Model variant |
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `4` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--trial-name` | auto (`trial_01`, …) | Output folder name |
| `--grad-accum` | `2` | Gradient accumulation steps |
| `--no-gradient-checkpointing` | — | Disable gradient checkpointing |
| `--dataset-dir` | auto | Skip prep, use existing dataset dir |
| `--validate` | `valid` | Split to evaluate after training |

---

### `dataset_prep.py` — Dataset Preparation

Resizes **train** images to the model's target resolution. Copies valid/test images as-is. Labels are always copied unchanged.

```bash
python dataset_prep.py
python dataset_prep.py --model rfdetr-medium
```

Output: `/kaggle/working/dataset_{resolution}/`

---

### `EDA.py` — Exploratory Data Analysis

Visualise random samples with bounding boxes drawn.

```bash
python EDA.py --split train --sample 5
python EDA.py --split test  --sample 5 --class Car
python EDA.py --split valid --sample 3 --class Person
```

| Argument | Default | Description |
|---|---|---|
| `--split` | `train` | Dataset split |
| `--sample` | `5` | Number of images to show |
| `--class` | — | Filter by class name (case-insensitive) |

Saves a PNG to the working directory and displays it inline.

---

### `train.py` — Training Only

```bash
python train.py
python train.py --model rfdetr-small --epochs 30 --trial-name run02
```

Accepts the same arguments as `main.py` (training args only).

**Output layout:**
```
outputs/trial_01/
├── runninglog.txt                   # full stdout/stderr log
├── save_model/
│   ├── checkpoint_best_total.pth    # best combined checkpoint
│   ├── checkpoint_best_regular.pth  # best regular checkpoint (if saved)
│   └── checkpoint_best_ema.pth      # best EMA checkpoint (if saved)
├── rfdetr_raw/                      # rfdetr internal output dir
│   ├── metrics.csv                  # per-step training metrics
│   └── events.out.tfevents.*        # TensorBoard events
├── tensorboard/                     # copied TensorBoard events
│   └── events.out.tfevents.*
├── confusion_matrix_trial_01.json
├── confusion_matrix_trial_01.png
├── loss_trial_01.png
└── mAP_trial_01.png
```

**TensorBoard:**
```bash
tensorboard --logdir outputs/trial_01/tensorboard
```

---

### `inference.py` — Inference & Evaluation

```bash
# Single image
python inference.py --file image.jpg

# Folder of images
python inference.py --file /path/to/images/ --name my_run

# Evaluate on a dataset split (computes confusion matrix + metrics)
python inference.py --validate valid
python inference.py --validate test  --model outputs/trial_01/save_model/checkpoint_best_total.pth
```

If `--model` is not specified, the script **auto-finds the most recently trained checkpoint** in `outputs/trial_*/save_model/`.

| Argument | Default | Description |
|---|---|---|
| `--file` | — | Image file or folder (mutually exclusive with `--validate`) |
| `--validate` | — | Evaluate on `train`/`valid`/`test` split |
| `--name` | auto (`inference_01`, …) | Output folder name |
| `--model` | auto-find latest | Path to `.pth` checkpoint |
| `--threshold` | `0.5` | Confidence threshold |
| `--iou-threshold` | `0.5` | IoU threshold for confusion matrix |

**Output layout:**
```
outputs/inference_01/
├── outputs_log_inference_01.txt
├── inference_time_inference_01.csv      # columns: filename, inference_time_ms
├── inference_image/
│   └── *.jpg                            # annotated copies
├── confusion_matrix_inference_01.json   # --validate only
├── confusion_matrix_inference_01.png    # --validate only
└── metrics_inference_01.json            # --validate only (per-class precision/recall)
```

---

## Configuration

All defaults are in `utils/config.py`. Edit this file to change dataset paths or training defaults without touching CLI args.

```python
SOURCE_DATASET = "/kaggle/input/datasets/abtinzandi/obstacle-detection-dataset/ROD-Dataset/dataset"
WORKING_DIR    = "/kaggle/working"

DEFAULT_MODEL        = "rfdetr-nano"
DEFAULT_EPOCHS       = 10
DEFAULT_BATCH_SIZE   = 4
DEFAULT_LR           = 1e-4
DEFAULT_GRAD_ACCUM   = 2
DEFAULT_CONF_THRESHOLD = 0.5
```

---

## Project Structure

```
RF-DETR-Finetune/
├── requirements.txt
├── main.py            # full pipeline orchestrator
├── train.py           # training only
├── inference.py       # inference / evaluation
├── dataset_prep.py    # dataset preparation
├── EDA.py             # exploratory data analysis
└── utils/
    ├── config.py      # shared defaults
    ├── model_config.py
    ├── trial_utils.py
    └── metrics.py
```

---

## GPU

GPU is auto-detected at startup. If no GPU is available, training falls back to CPU (slow — not recommended for full training runs).
