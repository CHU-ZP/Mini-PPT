# miniPPT

`miniPPT` is a lightweight course project for **MVA NPM3D**.

It is inspired by:
- the official large-scale 3D framework **Pointcept**: <https://github.com/Pointcept/Pointcept>
- the paper **Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training (PPT)**: <https://arxiv.org/abs/2308.09718>

This repository is **not** a reproduction of the full paper or the full Pointcept codebase. It implements a much smaller and more controllable experiment for coursework.

## What This Repo Implements

- 3D point cloud classification
- two datasets as two domains:
  - `ModelNet40`
  - `ScanObjectNN`
- two backbone choices:
  - `pointnet`
  - `dgcnn`
- optional `PDNorm`
- two prediction-space choices:
  - `decoupled`
  - `language_guided`

## What This Repo Does Not Implement

- the full Pointcept framework
- large-scale segmentation training
- the full PPT system
- CLIP image encoder

## Project Goal

The goal is to test whether the following choices help multi-dataset joint training:

- `PDNorm`
- a lightweight `language_guided` head

The project keeps the code short and runnable on a single RTX 4060 laptop GPU.

## Technical Design

### Backbone

The repo now supports two backbone choices.

`pointnet`
- lightweight PointNet-style shared encoder
- input: `xyz` only
- three point-wise blocks: `3 -> 64 -> 128 -> 256`
- `1x1 Conv + Norm + ReLU`
- global max pooling for the final shape feature

`dgcnn`
- a lightweight DGCNN-style alternative
- dynamic graph feature extraction with kNN-based EdgeConv blocks
- channel progression: `3 -> 64 -> 64 -> 128 -> 256`
- multi-scale feature concatenation followed by a fusion block and global max pooling

The default remains `pointnet`. `dgcnn` is provided as a stronger alternative backbone.

### Domain Prompt / PDNorm

`PDNorm` is implemented as a lightweight domain-conditioned normalization layer:

- first normalize the intermediate feature with `InstanceNorm1d(affine=False)`
- then use a learned dataset embedding as a domain embedding
- predict channel-wise affine `scale` and `bias` from the domain embedding
- modulate the normalized feature with the predicted affine parameters

In this repo, the domain embedding is simply the dataset id:

- `ModelNet40`
- `ScanObjectNN`

The same idea is reused for both backbones:

- in `pointnet`, PDNorm is applied after each point-wise block
- in `dgcnn`, PDNorm is applied after each EdgeConv block and after the fusion block

### Prediction Heads

The repo supports two prediction-space strategies.

`decoupled`
- one classifier head for `ModelNet40`
- one classifier head for `ScanObjectNN`
- each sample is routed to the head of its own dataset

`language_guided`
- no dataset-specific classifier heads
- a shared projection head maps 3D features into a frozen text embedding space
- classification is performed by similarity to class text prototypes from the current dataset

### Language Encoder

The lightweight language-guided implementation uses a **frozen pretrained text encoder**:

- model: `sentence-transformers/all-MiniLM-L6-v2`
- text prompts are built from class names with the template:
  - `a 3d point cloud of a {}`
- text embeddings are cached locally and reused during training and evaluation

This is a lightweight substitute for the full language-guided alignment idea in the paper.

### Loss

For `decoupled`, training uses standard classification cross-entropy on dataset-specific logits.

For `language_guided`, logits are built from:

- normalized 3D projected features
- normalized text prototypes
- temperature-scaled dot-product similarity

The loss is implemented as **InfoNCE-style cross-entropy** over these similarities.
Negative classes are restricted to the **current dataset label space only**, which matches the intended multi-dataset setting.

### Methods Compared

The main comparison in this repo is:

- single-dataset training
- joint training with `decoupled` heads
- joint training with `decoupled + PDNorm`
- joint training with `language_guided`
- joint training with `language_guided + PDNorm`

So in this project:

- `train_joint_naive` vs `train_joint_pdnorm` controls whether `PDNorm` is used
- `--head_type decoupled` vs `--head_type language_guided` controls the prediction-space strategy

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

If you need dataset preprocessing:

```bash
uv pip install -e .[preprocess]
```

If you want the `language_guided` head or the standalone text embedding tool:

```bash
uv pip install -e .[text]
```

## Expected Data Layout

```text
data/
├── modelnet40_princeton_npy/
│   ├── train_points.npy
│   ├── train_labels.npy
│   ├── test_points.npy
│   ├── test_labels.npy
│   └── modelnet40_shape_names.txt
└── scanobjectnn_npy/
    ├── train_points.npy
    ├── train_labels.npy
    ├── test_points.npy
    ├── test_labels.npy
    └── scanobjectnn_shape_names.txt
```

You can prepare local archives with:

```bash
uv run minippt-prepare-data modelnet40 --archive_path /path/to/ModelNet40.zip --data_root data/modelnet40_princeton_npy
uv run minippt-prepare-data scanobjectnn --archive_path /path/to/h5_files.zip --data_root data/scanobjectnn_npy --variant PB_T50_RS --split split1
```

## Main Commands

Single-dataset training:

```bash
uv run minippt-train --mode train_modelnet_only
uv run minippt-train --mode train_scanobjectnn_only
```

Joint training with `decoupled` heads:

```bash
uv run minippt-train --mode train_joint_naive --head_type decoupled
uv run minippt-train --mode train_joint_pdnorm --head_type decoupled
```

Use DGCNN instead of the default PointNet backbone:

```bash
uv run minippt-train --mode train_joint_pdnorm --backbone_type dgcnn --head_type decoupled
```

Joint training with the lightweight `language_guided` head:

```bash
uv run minippt-train --mode train_joint_naive --head_type language_guided
uv run minippt-train --mode train_joint_pdnorm --head_type language_guided
```

Evaluation:

```bash
uv run minippt-eval --checkpoint runs/<run_name>/best.pt --domains both
```

Run the full benchmark suite:

```bash
./run_all_experiments.sh
```

Run the same benchmark with DGCNN:

```bash
BACKBONE_TYPE=dgcnn ./run_all_experiments.sh
```

## Default Training Setup

- `backbone_type=pointnet`
- `epochs=50`
- `batch_size=128`
- `num_points=1024`
- `learning_rate=2e-3`

For `dgcnn`, a smaller batch size is usually more realistic than the default PointNet setting.

## Outputs

Each run writes to `runs/<run_name>/`:

- `config.json`
- `history.json`
- `summary.json`
- `curves.png`
- `last.pt`
- `best.pt`

## References

- Pointcept: <https://github.com/Pointcept/Pointcept>
- PPT paper: <https://arxiv.org/abs/2308.09718>
