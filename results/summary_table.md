# Benchmark Summary

| Method | Backbone | Mode | Head | Best Epoch | Mean Acc | ModelNet40 Acc | ScanObjectNN Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Single ModelNet40 | dgcnn | train_modelnet_only | decoupled | 46 | 0.8545 | 0.8545 | - |
| Single ScanObjectNN | dgcnn | train_scanobjectnn_only | decoupled | 46 | 0.7707 | - | 0.7707 |
| Joint Decoupled | dgcnn | train_joint_naive | decoupled | 48 | 0.8118 | 0.8505 | 0.7732 |
| Joint Decoupled + PDNorm | dgcnn | train_joint_pdnorm | decoupled | 46 | 0.7943 | 0.8614 | 0.7272 |
| Joint Language-guided | dgcnn | train_joint_naive | language_guided | 45 | 0.8055 | 0.8416 | 0.7693 |
| Joint Language-guided + PDNorm | dgcnn | train_joint_pdnorm | language_guided | 48 | 0.7986 | 0.8513 | 0.7460 |
| Single ModelNet40 | pointnet | train_modelnet_only | decoupled | 43 | 0.8075 | 0.8075 | - |
| Single ScanObjectNN | pointnet | train_scanobjectnn_only | decoupled | 49 | 0.6617 | - | 0.6617 |
| Joint Decoupled | pointnet | train_joint_naive | decoupled | 48 | 0.7346 | 0.8055 | 0.6638 |
| Joint Decoupled + PDNorm | pointnet | train_joint_pdnorm | decoupled | 42 | 0.7452 | 0.8213 | 0.6690 |
| Joint Language-guided | pointnet | train_joint_naive | language_guided | 44 | 0.7280 | 0.7909 | 0.6652 |
| Joint Language-guided + PDNorm | pointnet | train_joint_pdnorm | language_guided | 49 | 0.7446 | 0.8229 | 0.6662 |
