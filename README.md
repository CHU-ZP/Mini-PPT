# miniPPT

`miniPPT` is a minimal, runnable course-project prototype inspired by the paper *Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training (PPT)*.

This version uses a **decoupled two-head setup**:

- one shared lightweight PointNet-style encoder
- one dataset embedding used as a domain embedding
- one optional Prompt-Driven Normalization (PDNorm) module
- one classifier head for `ModelNet40`
- one classifier head for `ScanObjectNN`

The goal is to study whether **domain-conditioned normalization** helps joint training under a real dataset gap, without reproducing the full PPT system.

## What This Project Implements

- 3D point cloud classification
- a shared PointNet-style encoder
- dataset-conditioned PDNorm
- decoupled dataset-specific classifier heads
- two-domain joint training with:
  - `ModelNet40`
  - `ScanObjectNN`

## What This Project Does Not Implement

- CLIP
- language-guided categorical alignment
- semantic segmentation
- large research frameworks such as Pointcept
- full paper-level reproduction

## Core Experiment

The repo treats the two datasets as two domains:

- `domain_a = ModelNet40`
- `domain_b = ScanObjectNN`

This keeps the experiment simple while introducing a meaningful synthetic-vs-real domain gap.

Unlike the earlier shared-class prototype, this version keeps the **full label space of each dataset** and uses **decoupled heads**:

- `ModelNet40 head`: 40-way classification
- `ScanObjectNN head`: 15-way classification

During joint training, each sample uses the head that matches its dataset. This avoids hand-crafted shared-category alignment and is much closer to the paper's `Decoupled` ablation setting.

## Training Modes

- `train_modelnet_only`
- `train_scanobjectnn_only`
- `train_joint_naive`
- `train_joint_pdnorm`

For backward compatibility, the old aliases still work:

- `train_a_only` -> `train_modelnet_only`
- `train_b_only` -> `train_scanobjectnn_only`

## Project Layout

```text
miniPPT/
├── README.md
├── pyproject.toml
├── config.py
├── dataset.py
├── model.py
├── prepare_data.py
├── train.py
├── eval.py
└── utils.py
```

## Data Expectations

### ModelNet40

Supported inputs:

- the common `modelnet40_normal_resampled` txt layout
- a preprocessed `npy` layout with:
  - `train_points.npy`
  - `train_labels.npy`
  - `test_points.npy`
  - `test_labels.npy`
  - `modelnet40_shape_names.txt`

### ScanObjectNN

The training code expects a preprocessed `npy` layout:

```text
data/scanobjectnn_npy/
├── train_points.npy
├── train_labels.npy
├── test_points.npy
├── test_labels.npy
└── scanobjectnn_shape_names.txt
```

`prepare_data.py` can convert the official ScanObjectNN `h5_files.zip` archive into this layout.

## Installation

The recommended workflow uses `uv` and `pyproject.toml`.

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

If you want to preprocess ScanObjectNN from its official `h5` archive, install the optional preprocessing dependency:

```bash
uv pip install -e .[preprocess]
```

Available commands:

```bash
minippt-train --help
minippt-eval --help
minippt-prepare-data --help
```

The training CLI intentionally keeps only the parameters you are likely to change during coursework:

- dataset roots
- training mode
- epochs
- batch size
- number of points
- experiment name

Learning rate, workers, dropout, AMP default, and output directory are fixed in [config.py](/home/zepeng/Obsidian/ComputerScience/2MVA/NPM3d/miniPPT/config.py#L23).

## Dataset Preparation

### 1. Prepare ModelNet40 from a Local Archive

If you have Princeton `ModelNet40.zip`:

```bash
uv run minippt-prepare-data modelnet40 \
  --archive_path /path/to/ModelNet40.zip \
  --data_root data/modelnet40_princeton_npy
```

If you already have `modelnet40_normal_resampled.zip`:

```bash
uv run minippt-prepare-data modelnet40 \
  --archive_path /path/to/modelnet40_normal_resampled.zip \
  --data_root data/modelnet40_normal_resampled
```

### 2. Prepare ScanObjectNN from a Local Archive

If you have the official `h5_files.zip` archive:

```bash
uv run minippt-prepare-data scanobjectnn \
  --archive_path /path/to/h5_files.zip \
  --data_root data/scanobjectnn_npy \
  --variant PB_T50_RS \
  --split split1
```

Supported ScanObjectNN variants:

- `OBJ_ONLY`
- `OBJ_BG`
- `PB_T25`
- `PB_T25_R`
- `PB_T50_RS`

The recommended variant is `PB_T50_RS`.
The recommended split is `split1`.

## Recommended Default Configuration

The current defaults are tuned for **quick iteration** on a single RTX 4060 laptop GPU:

- `batch_size=64`
- `epochs=20`
- `num_points=512`
- `num_workers=4`
- `AMP enabled`
- shared lightweight encoder
- small domain embedding

If you need exact full-precision behavior for debugging, add `--no_amp`.

If you want a slightly stronger but slower setting, a simple upgrade is:

- `epochs=40`
- `num_points=1024`

## Usage

### Train on ModelNet40 Only

```bash
uv run minippt-train \
  --mode train_modelnet_only
```

### Train on ScanObjectNN Only

```bash
uv run minippt-train \
  --mode train_scanobjectnn_only
```

### Run Naive Joint Training

```bash
uv run minippt-train \
  --mode train_joint_naive
```

### Run Joint Training with PDNorm

```bash
uv run minippt-train \
  --mode train_joint_pdnorm
```

Override only what you need, for example:

```bash
uv run minippt-train \
  --mode train_joint_pdnorm \
  --epochs 40 \
  --batch_size 96 \
  --num_points 1024
```

## Evaluation

Evaluate both validation domains:

```bash
uv run minippt-eval \
  --checkpoint runs/train_joint_pdnorm_pts512_bs64_ep20_seed42/best.pt \
  --domains both
```

Evaluate only one validation domain:

```bash
uv run minippt-eval \
  --checkpoint runs/train_joint_naive_pts512_bs64_ep20_seed42/best.pt \
  --domains modelnet
```

## Outputs

Each run creates a folder inside `runs/` containing:

- `config.json`
- `history.json`
- `summary.json`
- `curves.png`
- `last.pt`
- `best.pt`

For joint training, the logs report:

- train loss
- mean validation accuracy
- `ModelNet40` validation accuracy
- `ScanObjectNN` validation accuracy
- best checkpoint and best epoch

## Recommended Experiment Order

1. `train_modelnet_only`
2. `train_scanobjectnn_only`
3. `train_joint_naive`
4. `train_joint_pdnorm`

This gives a clean story for a course presentation:

- first show the difficulty gap between the two datasets
- then show the negative transfer of naive joint training
- finally show whether PDNorm helps the shared encoder adapt to both domains
