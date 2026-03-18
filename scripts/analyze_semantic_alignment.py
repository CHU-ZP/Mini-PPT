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
from dataset import build_datasets, collate_point_cloud_batch
from model import PointNetClassifier
from text_encoder import FrozenTextEmbedder
from utils import choose_device, save_json


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare feature-vs-text semantic geometry for two checkpoints."
    )
    parser.add_argument("--checkpoint_a", type=str, required=True)
    parser.add_argument("--checkpoint_b", type=str, required=True)
    parser.add_argument("--name_a", type=str, default="")
    parser.add_argument("--name_b", type=str, default="")
    parser.add_argument("--modelnet_root", type=str, default="")
    parser.add_argument("--scanobjectnn_root", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_permutations", type=int, default=500)
    parser.add_argument("--text_style", type=str, choices=["raw", "prompted"], default="prompted")
    parser.add_argument("--output_dir", type=str, default="results/semantic_alignment")
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


def average_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start
        while end + 1 < values.shape[0] and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end) + 1.0
        ranks[order[start : end + 1]] = rank
        start = end + 1
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_corr(average_rank(a), average_rank(b))


def mantel_test(
    similarity_a: np.ndarray,
    similarity_b: np.ndarray,
    num_permutations: int = 500,
    seed: int = 42,
):
    distance_a = 1.0 - similarity_a
    distance_b = 1.0 - similarity_b
    observed = pearson_corr(upper_triangle_values(distance_a), upper_triangle_values(distance_b))

    rng = np.random.default_rng(seed)
    permuted_scores = []
    for _ in range(num_permutations):
        perm = rng.permutation(distance_b.shape[0])
        permuted = distance_b[perm][:, perm]
        score = pearson_corr(upper_triangle_values(distance_a), upper_triangle_values(permuted))
        permuted_scores.append(score)

    permuted_scores = np.asarray(permuted_scores, dtype=np.float64)
    p_value = float((np.sum(np.abs(permuted_scores) >= abs(observed)) + 1) / (num_permutations + 1))
    return float(observed), p_value


def matrix_correlations(
    feature_similarity: np.ndarray,
    text_similarity: np.ndarray,
    num_permutations: int,
):
    feature_values = upper_triangle_values(feature_similarity)
    text_values = upper_triangle_values(text_similarity)
    mantel_r, mantel_p = mantel_test(
        feature_similarity,
        text_similarity,
        num_permutations=num_permutations,
    )
    return {
        "pearson": pearson_corr(feature_values, text_values),
        "spearman": spearman_corr(feature_values, text_values),
        "mantel_r": mantel_r,
        "mantel_p": mantel_p,
    }


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = ExperimentConfig.from_dict(checkpoint["config"])
    cfg.mode = canonical_mode(cfg.mode)

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


@torch.no_grad()
def extract_features(model, dataset, device, batch_size: int, num_workers: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_point_cloud_batch,
    )

    feature_list = []
    label_list = []
    for points, labels, domain_ids in loader:
        points = points.to(device, non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)
        features = model.forward_features(points, domain_ids)
        feature_list.append(features.cpu())
        label_list.append(labels)

    features = torch.cat(feature_list, dim=0).numpy()
    labels = torch.cat(label_list, dim=0).numpy()
    return features, labels


def compute_class_prototypes(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    normalized_features = l2_normalize(features.astype(np.float32))
    prototypes = []
    for class_idx in range(num_classes):
        class_features = normalized_features[labels == class_idx]
        if class_features.shape[0] == 0:
            raise ValueError(f"No samples found for class index {class_idx}.")
        prototypes.append(class_features.mean(axis=0))
    return l2_normalize(np.stack(prototypes, axis=0))


def build_text_prototypes(
    cfg: ExperimentConfig,
    class_names_by_domain: dict[str, list[str]],
    text_style: str,
):
    encoder = FrozenTextEmbedder(model_name=cfg.text_model_name, device="cpu")
    text_prototypes = {}
    text_labels = {}
    for domain_name, class_names in class_names_by_domain.items():
        if text_style == "prompted":
            texts = [cfg.text_prompt_template.format(class_name.replace("_", " ")) for class_name in class_names]
        else:
            texts = [class_name.replace("_", " ") for class_name in class_names]
        cache_path = encoder.default_cache_path(texts, cfg.text_cache_dir, prefix=f"{domain_name}_{text_style}_analysis")
        embeddings = encoder.encode_with_cache(texts, cache_path)
        text_prototypes[domain_name] = l2_normalize(embeddings.cpu().numpy())
        text_labels[domain_name] = texts
    return text_prototypes, text_labels


def compute_similarity_matrix(prototypes: np.ndarray) -> np.ndarray:
    return prototypes @ prototypes.T


def set_heatmap_ticks(ax, labels):
    step = max(1, int(np.ceil(len(labels) / 12)))
    tick_positions = list(range(0, len(labels), step))
    if tick_positions[-1] != len(labels) - 1:
        tick_positions.append(len(labels) - 1)
    tick_labels = [labels[idx] for idx in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=7)


def plot_similarity_heatmaps(domain_name, summaries, save_path: Path):
    fig, axes = plt.subplots(len(summaries), 3, figsize=(14, 4.8 * len(summaries)))
    if len(summaries) == 1:
        axes = np.array([axes])

    for row_idx, summary in enumerate(summaries):
        labels = summary["class_names_by_domain"][domain_name]
        feat_sim = summary["domain_results"][domain_name]["feature_similarity"]
        text_sim = summary["domain_results"][domain_name]["text_similarity"]
        diff_sim = feat_sim - text_sim

        panels = [
            (feat_sim, "Feature similarity", "coolwarm", -1.0, 1.0),
            (text_sim, "Text similarity", "coolwarm", -1.0, 1.0),
            (diff_sim, "Feature - Text", "coolwarm", -1.0, 1.0),
        ]
        for col_idx, (matrix, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            set_heatmap_ticks(ax, labels)
            ax.set_title(f"{summary['name']} | {title}")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{domain_name.capitalize()} Similarity Matrices", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(summaries, save_path: Path):
    metric_keys = ["pearson", "spearman", "mantel_r"]
    metric_titles = ["Pearson", "Spearman", "Mantel r"]
    domains = ["modelnet", "scanobjectnn"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(metric_keys))
    width = 0.35 if len(summaries) == 2 else 0.8 / max(len(summaries), 1)
    all_values = []
    for summary in summaries:
        for domain_name in domains:
            for key in metric_keys:
                all_values.append(summary["domain_results"][domain_name]["metrics"][key])
    lower_bound = min(-0.1, float(min(all_values)) - 0.05)

    for domain_idx, domain_name in enumerate(domains):
        ax = axes[domain_idx]
        for idx, summary in enumerate(summaries):
            values = [summary["domain_results"][domain_name]["metrics"][key] for key in metric_keys]
            offset = (idx - (len(summaries) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=summary["name"],
                color=colors[idx % len(colors)],
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_titles, rotation=15, ha="right")
        ax.set_ylim(lower_bound, 1.0)
        ax.set_title("ModelNet40" if domain_name == "modelnet" else "ScanObjectNN")
        ax.set_ylabel("Correlation")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend()

    fig.suptitle("Semantic Geometry Alignment Comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_results_table(summaries, csv_path: Path, md_path: Path):
    header = [
        "name",
        "backbone",
        "head_type",
        "use_pdnorm",
        "domain",
        "pearson",
        "spearman",
        "mantel_r",
        "mantel_p",
    ]
    rows = []
    for summary in summaries:
        cfg = summary["config"]
        for domain_name in ("modelnet", "scanobjectnn"):
            metrics = summary["domain_results"][domain_name]["metrics"]
            rows.append(
                [
                    summary["name"],
                    cfg["backbone_type"],
                    cfg["head_type"],
                    str(cfg["mode"] == "train_joint_pdnorm"),
                    domain_name,
                    f"{metrics['pearson']:.6f}",
                    f"{metrics['spearman']:.6f}",
                    f"{metrics['mantel_r']:.6f}",
                    f"{metrics['mantel_p']:.6f}",
                ]
            )

    csv_lines = [",".join(header)]
    csv_lines.extend(",".join(row) for row in rows)
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md_lines = [
        "| Name | Backbone | Head | PDNorm | Domain | Pearson | Spearman | Mantel r | Mantel p |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def summarize_checkpoint(name: str, checkpoint_path: str, args, device: torch.device):
    checkpoint, cfg, model = load_model_from_checkpoint(checkpoint_path, device)
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
    class_names_by_domain = datasets["domain_class_names"]
    text_prototypes, text_labels = build_text_prototypes(cfg, class_names_by_domain, args.text_style)

    domain_results = {}
    for domain_name, dataset_key in (("modelnet", "val_modelnet"), ("scanobjectnn", "val_scanobjectnn")):
        features, labels = extract_features(
            model,
            datasets[dataset_key],
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        feature_prototypes = compute_class_prototypes(features, labels, len(class_names_by_domain[domain_name]))
        feature_similarity = compute_similarity_matrix(feature_prototypes)
        text_similarity = compute_similarity_matrix(text_prototypes[domain_name])
        metrics = matrix_correlations(
            feature_similarity,
            text_similarity,
            num_permutations=args.num_permutations,
        )
        domain_results[domain_name] = {
            "feature_similarity": feature_similarity,
            "text_similarity": text_similarity,
            "metrics": metrics,
        }

    return {
        "name": name,
        "checkpoint": checkpoint_path,
        "config": cfg.to_dict(),
        "best_metrics": checkpoint.get("metrics", {}),
        "class_names_by_domain": class_names_by_domain,
        "text_labels_by_domain": text_labels,
        "domain_results": domain_results,
    }


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device("cuda")
    name_a = args.name_a or Path(args.checkpoint_a).parent.name
    name_b = args.name_b or Path(args.checkpoint_b).parent.name

    summaries = [
        summarize_checkpoint(name_a, args.checkpoint_a, args, device),
        summarize_checkpoint(name_b, args.checkpoint_b, args, device),
    ]

    payload = {
        "text_style": args.text_style,
        "num_permutations": args.num_permutations,
        "summaries": [
            {
                "name": summary["name"],
                "checkpoint": summary["checkpoint"],
                "config": summary["config"],
                "best_metrics": summary["best_metrics"],
                "text_labels_by_domain": summary["text_labels_by_domain"],
                "domain_results": {
                    domain_name: {"metrics": domain_payload["metrics"]}
                    for domain_name, domain_payload in summary["domain_results"].items()
                },
            }
            for summary in summaries
        ],
    }
    save_json(payload, output_dir / "semantic_alignment_summary.json")
    export_results_table(
        summaries,
        output_dir / "semantic_alignment_results.csv",
        output_dir / "semantic_alignment_results.md",
    )
    plot_metric_comparison(summaries, output_dir / "semantic_alignment_comparison.png")
    plot_similarity_heatmaps("modelnet", summaries, output_dir / "modelnet_similarity_heatmaps.png")
    plot_similarity_heatmaps("scanobjectnn", summaries, output_dir / "scanobjectnn_similarity_heatmaps.png")
    print(f"Saved semantic-alignment analysis to: {output_dir}")


if __name__ == "__main__":
    main()
