# miniPPT

`miniPPT` is a lightweight **MVA NPM3D** course project inspired by:
- **Pointcept**: <https://github.com/Pointcept/Pointcept>
- **Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training (PPT)**: <https://arxiv.org/abs/2308.09718>

This repo is **not** a reproduction of the full Pointcept/PPT system. It implements a compact two-dataset classification setting for studying domain-conditioned normalization and language-guided prediction on a single GPU.

## Scope

Implemented:
- object-level point cloud classification
- two datasets as two domains: `ModelNet40`, `ScanObjectNN`
- two backbones: `pointnet`, `dgcnn`
- optional `PDNorm`
- two prediction-space designs: `decoupled`, `language_guided`

Not implemented:
- the full Pointcept framework
- large-scale segmentation
- the complete PPT training setup
- CLIP image encoder

## Method

### Backbone

- `pointnet`: lightweight PointNet-style encoder with three point-wise blocks `3 -> 64 -> 128 -> 256`, followed by global max pooling
- `dgcnn`: lightweight DGCNN-style encoder with kNN-based EdgeConv blocks `3 -> 64 -> 64 -> 128 -> 256`, feature fusion, and global max pooling

### PDNorm

`PDNorm` is inserted inside the shared encoder as:

1. normalize intermediate features with `InstanceNorm1d(affine=False)`
2. map dataset id to a learned domain embedding
3. predict channel-wise affine scale and bias
4. modulate the normalized feature

It is applied:
- after each point-wise block in `pointnet`
- after each EdgeConv block and the fusion block in `dgcnn`

### Prediction Space

- `decoupled`: one classifier head per dataset
- `language_guided`: a shared projection head maps 3D features to a frozen text embedding space, and prediction is made by similarity to text class prototypes

### Language Encoder and Loss

- text encoder: `sentence-transformers/all-MiniLM-L6-v2`
- prompt template: `a 3d point cloud of a {}`
- text embeddings are cached locally

For `decoupled`, training uses standard cross-entropy.  
For `language_guided`, logits are temperature-scaled similarities between projected 3D features and text prototypes, trained with an **InfoNCE-style cross-entropy**. Negative classes are restricted to the current dataset label space.

## Repo Layout

```text
miniPPT/
├── config.py
├── dataset.py
├── model.py
├── train.py
├── eval.py
├── prepare_data.py
├── text_encoder.py
├── utils.py
├── scripts/
│   ├── run_all_experiments.sh
│   ├── run_semantic_alignment_suite.sh
│   ├── plot_benchmark_results.py
│   ├── analyze_semantic_alignment.py
│   └── analyze_language_projection.py
└── references/
    └── ppt-paper.pdf
```

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Optional extras:

```bash
uv pip install -e .[preprocess]
uv pip install -e .[text]
```

## Data Layout

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

Prepare local archives with:

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

Joint training:

```bash
uv run minippt-train --mode train_joint_naive --head_type decoupled
uv run minippt-train --mode train_joint_pdnorm --head_type decoupled
uv run minippt-train --mode train_joint_naive --head_type language_guided
uv run minippt-train --mode train_joint_pdnorm --head_type language_guided
```

Use DGCNN:

```bash
uv run minippt-train --mode train_joint_pdnorm --backbone_type dgcnn --head_type decoupled
```

Evaluation:

```bash
uv run minippt-eval --checkpoint runs/<run_name>/best.pt --domains both
```

Benchmark scripts:

```bash
./scripts/run_all_experiments.sh
BACKBONE_TYPE=dgcnn ./scripts/run_all_experiments.sh
./scripts/run_semantic_alignment_suite.sh
uv run python scripts/plot_benchmark_results.py
```

## Defaults

- `backbone_type=pointnet`
- `epochs=50`
- `batch_size=128`
- `num_points=1024`
- `learning_rate=2e-3`

For `dgcnn`, a smaller batch size is often more practical.

## Outputs

Training runs are written to `runs/<run_name>/` and typically contain:
- `config.json`
- `history.json`
- `summary.json`
- `curves.png`
- `last.pt`
- `best.pt`
