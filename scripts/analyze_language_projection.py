import argparse
from pathlib import Path
import sys

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ExperimentConfig, canonical_mode
from dataset import DOMAIN_TO_ID, build_datasets, collate_point_cloud_batch
from model import PointNetClassifier
from train import build_text_prototypes
from utils import choose_device, save_json


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze the projected semantic space of language-guided checkpoints."
    )
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="*", default=[])
    parser.add_argument("--modelnet_root", type=str, default="")
    parser.add_argument("--scanobjectnn_root", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/language_projection")
    return parser


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx].astype(np.float64)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a * b) / denom)


def load_language_guided_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = ExperimentConfig.from_dict(checkpoint["config"])
    cfg.mode = canonical_mode(cfg.mode)
    if cfg.head_type != "language_guided":
        raise ValueError(f"{checkpoint_path} is not a language-guided checkpoint.")

    domain_num_classes = checkpoint["domain_num_classes"]
    model = PointNetClassifier(
        num_classes_by_domain=[
            domain_num_classes["modelnet"],
            domain_num_classes["scanobjectnn"],
        ],
        emb_dim=cfg.emb_dim,
        use_pdnorm=(cfg.mode == "train_joint_pdnorm"),
        dropout=cfg.dropout,
        num_domains=2,
        head_type=cfg.head_type,
        text_embedding_dim=cfg.text_embedding_dim,
        backbone_type=cfg.backbone_type,
        dgcnn_k=cfg.dgcnn_k,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, cfg, model


def resolve_text_embeddings(checkpoint, cfg, domain_class_names):
    text_embeddings_by_domain = checkpoint.get("text_embeddings_by_domain")
    if text_embeddings_by_domain is not None:
        return {
            int(domain_idx): embeddings.float().cpu()
            for domain_idx, embeddings in text_embeddings_by_domain.items()
        }

    rebuilt, _ = build_text_prototypes(cfg, domain_class_names)
    return {
        int(domain_idx): embeddings.float().cpu()
        for domain_idx, embeddings in rebuilt.items()
    }


@torch.no_grad()
def extract_projected_features(model, dataset, device, batch_size: int, num_workers: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_point_cloud_batch,
    )

    projected_list = []
    label_list = []
    for points, labels, domain_ids in loader:
        points = points.to(device, non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)
        features = model.forward_features(points, domain_ids)
        projected = model.language_guided_head(features)
        projected_list.append(projected.cpu())
        label_list.append(labels)

    projected = torch.cat(projected_list, dim=0).numpy()
    labels = torch.cat(label_list, dim=0).numpy()
    return projected, labels


def compute_class_prototypes(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    normalized_features = l2_normalize(features.astype(np.float32))
    prototypes = []
    for class_idx in range(num_classes):
        class_features = normalized_features[labels == class_idx]
        if class_features.shape[0] == 0:
            raise ValueError(f"Missing samples for class index {class_idx}.")
        prototypes.append(class_features.mean(axis=0))
    return l2_normalize(np.stack(prototypes, axis=0))


def analyze_domain(projected_features, labels, text_prototypes):
    feature_prototypes = compute_class_prototypes(projected_features, labels, text_prototypes.shape[0])
    text_prototypes = l2_normalize(text_prototypes.astype(np.float32))

    matched_cosine = float(np.mean(np.sum(feature_prototypes * text_prototypes, axis=1)))
    feature_similarity = feature_prototypes @ feature_prototypes.T
    text_similarity = text_prototypes @ text_prototypes.T
    matrix_pearson = pearson_corr(
        upper_triangle_values(feature_similarity),
        upper_triangle_values(text_similarity),
    )
    return {
        "matched_text_cosine": matched_cosine,
        "matrix_pearson": matrix_pearson,
    }


def summarize_checkpoint(name: str, checkpoint_path: str, args, device: torch.device):
    checkpoint, cfg, model = load_language_guided_model(checkpoint_path, device)
    if args.modelnet_root:
        cfg.modelnet_root = args.modelnet_root
    if args.scanobjectnn_root:
        cfg.scanobjectnn_root = args.scanobjectnn_root

    datasets = build_datasets(
        modelnet_root=cfg.modelnet_root,
        scanobjectnn_root=cfg.scanobjectnn_root,
        num_points=cfg.num_points,
        use_cache=False,
    )
    domain_class_names = datasets["domain_class_names"]
    text_embeddings_by_domain = resolve_text_embeddings(checkpoint, cfg, domain_class_names)

    results = {}
    for domain_name, dataset_key in (("modelnet", "val_modelnet"), ("scanobjectnn", "val_scanobjectnn")):
        projected_features, labels = extract_projected_features(
            model,
            datasets[dataset_key],
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        results[domain_name] = analyze_domain(
            projected_features,
            labels,
            text_embeddings_by_domain[DOMAIN_TO_ID[domain_name]].numpy(),
        )

    results["macro_matched_text_cosine"] = float(
        np.mean([results["modelnet"]["matched_text_cosine"], results["scanobjectnn"]["matched_text_cosine"]])
    )
    results["macro_matrix_pearson"] = float(
        np.mean([results["modelnet"]["matrix_pearson"], results["scanobjectnn"]["matrix_pearson"]])
    )

    return {
        "name": name,
        "checkpoint": checkpoint_path,
        "config": cfg.to_dict(),
        "results": results,
    }


def export_results(summaries, output_dir: Path):
    csv_lines = [
        "name,backbone,mode,modelnet_matched_text_cosine,scanobjectnn_matched_text_cosine,macro_matched_text_cosine,modelnet_matrix_pearson,scanobjectnn_matrix_pearson,macro_matrix_pearson"
    ]
    md_lines = [
        "| Name | Backbone | Mode | ModelNet text cosine | ScanObjectNN text cosine | Macro text cosine | ModelNet matrix Pearson | ScanObjectNN matrix Pearson | Macro matrix Pearson |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in summaries:
        cfg = summary["config"]
        results = summary["results"]
        row = [
            summary["name"],
            str(cfg["backbone_type"]),
            str(cfg["mode"]),
            f"{results['modelnet']['matched_text_cosine']:.6f}",
            f"{results['scanobjectnn']['matched_text_cosine']:.6f}",
            f"{results['macro_matched_text_cosine']:.6f}",
            f"{results['modelnet']['matrix_pearson']:.6f}",
            f"{results['scanobjectnn']['matrix_pearson']:.6f}",
            f"{results['macro_matrix_pearson']:.6f}",
        ]
        csv_lines.append(",".join(row))
        md_lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} |"
        )

    (output_dir / "language_projection_results.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (output_dir / "language_projection_results.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def plot_results(summaries, output_dir: Path):
    names = [summary["name"] for summary in summaries]
    x = np.arange(len(names))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    color_map = {
        "modelnet": "#1f77b4",
        "scanobjectnn": "#d62728",
        "macro": "#2ca02c",
    }

    text_cosine_specs = [
        ("modelnet", "matched_text_cosine", "ModelNet40"),
        ("scanobjectnn", "matched_text_cosine", "ScanObjectNN"),
        ("macro", "macro_matched_text_cosine", "Macro"),
    ]
    for idx, (domain_name, key, label) in enumerate(text_cosine_specs):
        offset = (idx - 1) * width
        if domain_name == "macro":
            values = [summary["results"][key] for summary in summaries]
        else:
            values = [summary["results"][domain_name][key] for summary in summaries]
        axes[0].bar(x + offset, values, width=width, color=color_map[domain_name], label=label, alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=18, ha="right")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_title("Projected Prototype vs Text Prototype")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)
    axes[0].legend()

    pearson_specs = [
        ("modelnet", "matrix_pearson", "ModelNet40"),
        ("scanobjectnn", "matrix_pearson", "ScanObjectNN"),
        ("macro", "macro_matrix_pearson", "Macro"),
    ]
    for idx, (domain_name, key, label) in enumerate(pearson_specs):
        offset = (idx - 1) * width
        if domain_name == "macro":
            values = [summary["results"][key] for summary in summaries]
        else:
            values = [summary["results"][domain_name][key] for summary in summaries]
        axes[1].bar(x + offset, values, width=width, color=color_map[domain_name], label=label, alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=18, ha="right")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_title("Projected Semantic Geometry vs Text Geometry")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "language_projection_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.names and len(args.names) != len(args.checkpoints):
        raise ValueError("--names must either be omitted or match the number of checkpoints.")

    device = choose_device("cuda")
    summaries = []
    for idx, checkpoint_path in enumerate(args.checkpoints):
        name = args.names[idx] if args.names else Path(checkpoint_path).parent.name
        summaries.append(summarize_checkpoint(name, checkpoint_path, args, device))

    payload = {
        "summaries": summaries,
    }
    save_json(payload, output_dir / "language_projection_summary.json")
    export_results(summaries, output_dir)
    plot_results(summaries, output_dir)
    print(f"Saved language-projection analysis to: {output_dir}")


if __name__ == "__main__":
    main()
