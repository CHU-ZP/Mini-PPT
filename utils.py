import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def prepare_output_dir(output_root: str, run_name: str) -> Path:
    out_dir = Path(output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_history(history: dict, save_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train loss", color="#1f77b4")
    if "semantic_loss" in history and len(history["semantic_loss"]) == len(epochs):
        axes[0].plot(epochs, history["semantic_loss"], label="semantic loss", color="#9467bd")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="val acc", color="#2ca02c")
    if "modelnet_acc" in history and len(history["modelnet_acc"]) == len(epochs):
        axes[1].plot(epochs, history["modelnet_acc"], label="modelnet", color="#ff7f0e")
    if "scanobjectnn_acc" in history and len(history["scanobjectnn_acc"]) == len(epochs):
        axes[1].plot(epochs, history["scanobjectnn_acc"], label="scanobjectnn", color="#d62728")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
