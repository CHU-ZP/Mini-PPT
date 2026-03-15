import argparse

import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig, canonical_mode, use_pdnorm, uses_language_guided_head
from dataset import build_datasets, collate_point_cloud_batch
from model import PointNetClassifier
from train import build_text_prototypes, evaluate_loader
from utils import choose_device


def build_parser():
    cfg = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--modelnet_root", type=str, default=cfg.modelnet_root)
    parser.add_argument("--scanobjectnn_root", type=str, default=cfg.scanobjectnn_root)
    parser.add_argument(
        "--domains",
        type=str,
        choices=["auto", "modelnet", "scanobjectnn", "both", "a", "b"],
        default="auto",
    )
    return parser


def resolve_domains(mode: str, domains_flag: str):
    if domains_flag in {"modelnet", "a"}:
        return ["modelnet"]
    if domains_flag in {"scanobjectnn", "b"}:
        return ["scanobjectnn"]
    if domains_flag == "both":
        return ["modelnet", "scanobjectnn"]
    if mode == "train_modelnet_only":
        return ["modelnet"]
    if mode == "train_scanobjectnn_only":
        return ["scanobjectnn"]
    return ["modelnet", "scanobjectnn"]


def main():
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = checkpoint["config"]
    cfg = ExperimentConfig.from_dict(cfg_dict)
    cfg.mode = canonical_mode(cfg.mode)
    cfg.modelnet_root = args.modelnet_root
    cfg.scanobjectnn_root = args.scanobjectnn_root

    device = choose_device(cfg.device)
    domain_num_classes = checkpoint.get("domain_num_classes")
    if domain_num_classes is None:
        raise KeyError("Checkpoint is missing `domain_num_classes`. Please re-train with the current multi-dataset format.")

    model = PointNetClassifier(
        num_classes_by_domain=[
            domain_num_classes["modelnet"],
            domain_num_classes["scanobjectnn"],
        ],
        emb_dim=cfg.emb_dim,
        use_pdnorm=use_pdnorm(cfg.mode),
        dropout=cfg.dropout,
        head_type=cfg.head_type,
        text_embedding_dim=cfg.text_embedding_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    text_embeddings_by_domain = checkpoint.get("text_embeddings_by_domain")
    if text_embeddings_by_domain is not None:
        text_embeddings_by_domain = {
            int(domain_idx): embeddings.float() for domain_idx, embeddings in text_embeddings_by_domain.items()
        }
    elif uses_language_guided_head(cfg.head_type):
        domain_class_names = checkpoint.get("domain_class_names")
        if domain_class_names is None:
            raise KeyError(
                "Checkpoint is missing both `text_embeddings_by_domain` and `domain_class_names`, "
                "so the language-guided head cannot be evaluated."
            )
        text_embeddings_by_domain, _ = build_text_prototypes(cfg, domain_class_names)

    datasets = build_datasets(
        modelnet_root=cfg.modelnet_root,
        scanobjectnn_root=cfg.scanobjectnn_root,
        num_points=cfg.num_points,
        use_cache=False,
    )
    name_to_dataset = {
        "modelnet": datasets["val_modelnet"],
        "scanobjectnn": datasets["val_scanobjectnn"],
    }
    chosen = resolve_domains(cfg.mode, args.domains)

    metrics = {}
    for name in chosen:
        loader = DataLoader(
            name_to_dataset[name],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_point_cloud_batch,
        )
        metrics[name] = evaluate_loader(
            model,
            loader,
            device,
            use_amp=bool(getattr(cfg, "amp", False) and device.type == "cuda"),
            head_type=cfg.head_type,
            text_embeddings_by_domain=text_embeddings_by_domain,
            language_guided_temperature=cfg.language_guided_temperature,
        )

    if len(metrics) > 1:
        metrics["mean_acc"] = sum(metrics.values()) / len(metrics)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {cfg.mode}")
    print(f"Head type: {cfg.head_type}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
