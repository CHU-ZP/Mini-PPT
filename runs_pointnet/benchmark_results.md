# Experiment Results

- Total experiments: 6

| Experiment | Mode | Head | Best Epoch | Best Mean Acc | ModelNet40 Acc | ScanObjectNN Acc |
| --- | --- | --- | --- | --- | --- | --- |
| Single-dataset training on ModelNet40 | train_modelnet_only | decoupled | 43 | 0.8075 | 0.8075 | - |
| Single-dataset training on ScanObjectNN | train_scanobjectnn_only | decoupled | 49 | 0.6617 | - | 0.6617 |
| Multi-dataset joint training, Decoupled | train_joint_naive | decoupled | 48 | 0.7346 | 0.8055 | 0.6638 |
| Multi-dataset joint training, Decoupled + PDNorm | train_joint_pdnorm | decoupled | 42 | 0.7452 | 0.8213 | 0.6690 |
| Multi-dataset joint training, Lightweight Language-guided Categorical Alignment | train_joint_naive | language_guided | 44 | 0.7280 | 0.7909 | 0.6652 |
| Multi-dataset joint training, Lightweight Language-guided Categorical Alignment + PDNorm | train_joint_pdnorm | language_guided | 49 | 0.7446 | 0.8229 | 0.6662 |

## Checkpoints

- `Single-dataset training on ModelNet40`: `/home/zepeng/Obsidian/ComputerScience/2MVA/NPM3d/miniPPT/runs/benchmark_single_modelnet40/best.pt`
- `Single-dataset training on ScanObjectNN`: `/home/zepeng/Obsidian/ComputerScience/2MVA/NPM3d/miniPPT/runs/benchmark_single_scanobjectnn/best.pt`
- `Multi-dataset joint training, Decoupled`: `/home/zepeng/Obsidian/ComputerScience/2MVA/NPM3d/miniPPT/runs/benchmark_joint_decoupled/best.pt`
- `Multi-dataset joint training, Decoupled + PDNorm`: `/home/zepeng/Obsidian/ComputerScience/2MVA/NPM3d/miniPPT/runs/benchmark_joint_decoupled_pdnorm/best.pt`
- `Multi-dataset joint training, Lightweight Language-guided Categorical Alignment`: `/home/zepeng/Obsidian/ComputerScience/2MVA/NPM3d/miniPPT/runs/benchmark_joint_language_guided/best.pt`
- `Multi-dataset joint training, Lightweight Language-guided Categorical Alignment + PDNorm`: `/home/zepeng/Obsidian/ComputerScience/2MVA/NPM3d/miniPPT/runs/benchmark_joint_language_guided_pdnorm/best.pt`
