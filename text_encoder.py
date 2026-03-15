import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class FrozenTextEmbedder(nn.Module):
    """
    A thin wrapper around a pretrained sentence embedding model.

    This module is intentionally standalone: it does not change the current
    training or evaluation pipeline unless another file imports and uses it.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        device: str = "cpu",
        normalize: bool = True,
    ):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "FrozenTextEmbedder requires the optional dependency "
                "`sentence-transformers`.\n"
                "Install it with: uv pip install -e .[text]"
            ) from exc

        self.model_name = model_name
        self.normalize = normalize
        self.device_name = device
        self.encoder = SentenceTransformer(model_name, device=device)
        self.encoder.eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return int(self.encoder.get_sentence_embedding_dimension())

    def train(self, mode: bool = True):
        # Keep the pretrained text model frozen even if parent modules call train().
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode(self, texts, convert_to_tensor: bool = True) -> torch.Tensor:
        embeddings = self.encoder.encode(
            list(texts),
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings)
        return embeddings.float()

    @torch.no_grad()
    def encode_domains(self, class_names_by_domain: dict[str, list[str]]) -> dict[str, torch.Tensor]:
        return {
            domain_name: self.encode(class_names)
            for domain_name, class_names in class_names_by_domain.items()
        }

    def default_cache_path(self, texts, cache_dir: str | Path, prefix: str = "") -> Path:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "normalize": self.normalize,
            "texts": list(texts),
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        stem = prefix or "text_embeddings"
        return cache_root / f"{stem}_{digest}.pt"

    @torch.no_grad()
    def encode_with_cache(self, texts, cache_path: str | Path) -> torch.Tensor:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu")
        embeddings = self.encode(texts)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings.cpu(), cache_path)
        return embeddings


def umap_project_2d(embeddings: torch.Tensor) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP visualization requires the optional dependency `umap-learn`.\n"
            "Install it with: uv pip install -e .[text]"
        ) from exc

    x = embeddings.detach().cpu().numpy().astype(np.float32)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(10, max(2, x.shape[0] - 1)),
        min_dist=0.15,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(x)


def plot_domain_embeddings(points_2d: np.ndarray, labels, domains, save_path: str | Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "modelnet": "#1f77b4",
        "scanobjectnn": "#d62728",
    }
    markers = {
        "modelnet": "o",
        "scanobjectnn": "s",
    }

    fig, ax = plt.subplots(figsize=(11, 8))
    unique_domains = sorted(set(domains))
    for domain in unique_domains:
        mask = np.array([item == domain for item in domains], dtype=bool)
        ax.scatter(
            points_2d[mask, 0],
            points_2d[mask, 1],
            label=domain,
            s=60,
            marker=markers.get(domain, "o"),
            color=colors.get(domain, "#333333"),
            alpha=0.85,
        )

    for (x, y), label, domain in zip(points_2d, labels, domains):
        ax.text(x + 0.01, y + 0.01, f"{label}", fontsize=8, alpha=0.9)

    ax.set_title("Text Embedding UMAP of Dataset Class Names")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    from config import ExperimentConfig

    cfg = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Encode dataset class names with a pretrained text encoder")
    parser.add_argument("--modelnet_root", type=str, default=cfg.modelnet_root)
    parser.add_argument("--scanobjectnn_root", type=str, default=cfg.scanobjectnn_root)
    parser.add_argument("--num_points", type=int, default=cfg.num_points)
    parser.add_argument("--model_name", type=str, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cache_dir", type=str, default="artifacts/text_cache")
    parser.add_argument("--output", type=str, default="artifacts/text_embeddings_umap.png")
    return parser


def main():
    from dataset import build_datasets

    args = build_parser().parse_args()
    datasets = build_datasets(
        modelnet_root=args.modelnet_root,
        scanobjectnn_root=args.scanobjectnn_root,
        num_points=args.num_points,
        use_cache=False,
    )
    class_names_by_domain = datasets["domain_class_names"]

    encoder = FrozenTextEmbedder(model_name=args.model_name, device=args.device)

    all_embeddings = []
    all_labels = []
    all_domains = []
    for domain_name, class_names in class_names_by_domain.items():
        cache_path = encoder.default_cache_path(class_names, args.cache_dir, prefix=domain_name)
        embeddings = encoder.encode_with_cache(class_names, cache_path)
        all_embeddings.append(embeddings)
        all_labels.extend(class_names)
        all_domains.extend([domain_name] * len(class_names))
        print(f"{domain_name}: {len(class_names)} labels -> {tuple(embeddings.shape)}")

    merged_embeddings = torch.cat(all_embeddings, dim=0)
    projected = umap_project_2d(merged_embeddings)
    plot_domain_embeddings(projected, all_labels, all_domains, args.output)
    print(f"Saved UMAP plot to: {args.output}")


if __name__ == "__main__":
    main()
