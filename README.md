# miniPPT

`miniPPT` is a minimal, runnable course-project prototype inspired by the paper *Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training (PPT)*.

This version uses a **shared encoder with switchable prediction heads**:

- one shared lightweight PointNet-style encoder
- one dataset embedding used as a domain embedding
- one optional Prompt-Driven Normalization (PDNorm) module
- one `decoupled` prediction head option with separate classifiers for `ModelNet40` and `ScanObjectNN`
- one `language_guided` prediction head option that projects 3D features into a frozen text space and classifies by text-prototype similarity

The goal is to study whether **domain-conditioned normalization** helps joint training under a real dataset gap, without reproducing the full PPT system.

## What This Project Implements

- 3D point cloud classification
- a shared PointNet-style encoder
- dataset-conditioned PDNorm
- switchable `decoupled` or `language_guided` prediction heads
- frozen text prototypes with dataset-restricted InfoNCE-style classification for the language-guided head
- two-domain joint training with:
  - `ModelNet40`
  - `ScanObjectNN`

## What This Project Does Not Implement

- CLIP image encoder
- semantic segmentation
- large research frameworks such as Pointcept
- full paper-level reproduction

## Core Experiment

The repo treats the two datasets as two domains:

- `domain_a = ModelNet40`
- `domain_b = ScanObjectNN`

This keeps the experiment simple while introducing a meaningful synthetic-vs-real domain gap.

Unlike the earlier shared-class prototype, this version keeps the **full label space of each dataset** and supports two prediction-space strategies:

- `decoupled`: `ModelNet40` uses a 40-way head and `ScanObjectNN` uses a 15-way head
- `language_guided`: a single lightweight projection head maps 3D features into a frozen text embedding space, and classification is performed by similarity to text prototypes from the current dataset

The `language_guided` option is intentionally lightweight: it does not reproduce the full paper, but it does replace the decoupled prediction heads with a text-prototype classifier and uses **dataset-restricted negatives**.

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

If you want to use the language-guided head or the standalone pretrained text embedding module, install:

```bash
uv pip install -e .[text]
```

Available commands:

```bash
minippt-train --help
minippt-eval --help
minippt-prepare-data --help
```

Optional module:

- `text_encoder.py`: a standalone frozen text embedding wrapper used by the language-guided head and by the text-label visualization utility

The training CLI intentionally keeps only the parameters you are likely to change during coursework:

- dataset roots
- training mode
- prediction head type
- epochs
- batch size
- number of points
- experiment name

Learning rate, workers, dropout, AMP default, and output directory are fixed in [config.py](/home/zepeng/Obsidian/ComputerScience/2MVA/NPM3d/miniPPT/config.py#L23).

## Optional Text Embeddings

The repo also includes a standalone pretrained text embedding wrapper in [text_encoder.py](/home/zepeng/Obsidian/ComputerScience/2MVA/NPM3d/miniPPT/text_encoder.py#L1). The same encoder is reused by the optional `language_guided` head.

Example:

```python
from text_encoder import FrozenTextEmbedder

encoder = FrozenTextEmbedder()
embeddings = encoder.encode(["chair", "table", "display"])
print(embeddings.shape)
```

You can also encode all dataset labels at once:

```python
from dataset import build_datasets
from text_encoder import FrozenTextEmbedder

datasets = build_datasets("data/modelnet40_princeton_npy", "data/scanobjectnn_npy", num_points=1024)
encoder = FrozenTextEmbedder()
label_embeddings = encoder.encode_domains(datasets["domain_class_names"])
```

To quickly visualize the encoded class names with UMAP:

```bash
uv run python text_encoder.py \
  --device cpu \
  --output artifacts/text_embeddings_umap.png
```

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

The current defaults are tuned for a **stronger single-GPU run** on a single RTX 4060 laptop GPU:

- `batch_size=128`
- `epochs=40`
- `num_points=1024`
- `learning_rate=2e-3`
- `num_workers=4`
- `AMP enabled`
- shared lightweight encoder
- small domain embedding

If you need exact full-precision behavior for debugging, add `--no_amp`.

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

### Run the True Lightweight Language-Guided Head

```bash
uv run minippt-train \
  --mode train_joint_pdnorm \
  --head_type language_guided
```

Override only what you need, for example:

```bash
uv run minippt-train \
  --mode train_joint_pdnorm \
  --head_type language_guided \
  --batch_size 256
```

## Evaluation

Evaluate both validation domains:

```bash
uv run minippt-eval \
  --checkpoint runs/train_joint_pdnorm_pts1024_bs128_ep40_seed42/best.pt \
  --domains both
```

Evaluate only one validation domain:

```bash
uv run minippt-eval \
  --checkpoint runs/train_joint_naive_pts1024_bs128_ep40_seed42/best.pt \
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
3. `train_joint_naive --head_type decoupled`
4. `train_joint_pdnorm --head_type decoupled`
5. `train_joint_naive --head_type language_guided`
6. `train_joint_pdnorm --head_type language_guided`

This gives a clean story for a course presentation:

- first show the difficulty gap between the two datasets
- then compare `decoupled` against a true lightweight `language_guided` prediction head
- finally show whether PDNorm helps under each prediction-space choice
