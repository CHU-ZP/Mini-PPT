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
- a lightweight PointNet-style shared encoder
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

## Default Training Setup

- `epochs=50`
- `batch_size=128`
- `num_points=1024`
- `learning_rate=2e-3`

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
