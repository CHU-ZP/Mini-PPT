import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BACKBONE_TYPES, HEAD_TYPES, ExperimentConfig, MODES, canonical_mode, default_run_name, use_pdnorm, uses_language_guided_head
from dataset import DOMAIN_TO_ID, build_datasets, collate_point_cloud_batch
from model import PointNetClassifier
from utils import AverageMeter, choose_device, plot_history, prepare_output_dir, save_json, seed_everything


def build_parser():
    cfg = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Train the minimal PPT-inspired classifier")
    parser.add_argument("--modelnet_root", type=str, default=cfg.modelnet_root)
    parser.add_argument("--scanobjectnn_root", type=str, default=cfg.scanobjectnn_root)
    parser.add_argument("--mode", type=str, choices=MODES, default=cfg.mode)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--num_points", type=int, default=cfg.num_points)
    parser.add_argument("--backbone_type", type=str, choices=BACKBONE_TYPES, default=cfg.backbone_type)
    parser.add_argument("--head_type", type=str, choices=HEAD_TYPES, default=cfg.head_type)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--exp_name", type=str, default=cfg.exp_name)
    parser.set_defaults(amp=cfg.amp)
    return parser


def args_to_config(args):
    cfg = ExperimentConfig()
    return ExperimentConfig(
        modelnet_root=args.modelnet_root,
        scanobjectnn_root=args.scanobjectnn_root,
        mode=canonical_mode(args.mode),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_points=args.num_points,
        output_root=cfg.output_root,
        num_workers=cfg.num_workers,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        emb_dim=cfg.emb_dim,
        dropout=cfg.dropout,
        seed=cfg.seed,
        cache_data=cfg.cache_data,
        device=cfg.device,
        amp=args.amp,
        backbone_type=args.backbone_type,
        dgcnn_k=cfg.dgcnn_k,
        head_type=args.head_type,
        text_embedding_dim=cfg.text_embedding_dim,
        language_guided_temperature=cfg.language_guided_temperature,
        text_model_name=cfg.text_model_name,
        text_prompt_template=cfg.text_prompt_template,
        text_cache_dir=cfg.text_cache_dir,
        exp_name=args.exp_name,
    )


def make_loaders(cfg: ExperimentConfig):
    datasets = build_datasets(
        modelnet_root=cfg.modelnet_root,
        scanobjectnn_root=cfg.scanobjectnn_root,
        num_points=cfg.num_points,
        use_cache=cfg.cache_data,
    )

    if cfg.mode == "train_modelnet_only":
        train_set = datasets["train_modelnet"]
        val_sets = {"modelnet": datasets["val_modelnet"]}
    elif cfg.mode == "train_scanobjectnn_only":
        train_set = datasets["train_scanobjectnn"]
        val_sets = {"scanobjectnn": datasets["val_scanobjectnn"]}
    else:
        train_set = datasets["joint_train"]
        val_sets = {
            "modelnet": datasets["val_modelnet"],
            "scanobjectnn": datasets["val_scanobjectnn"],
        }

    loader_kwargs = {}
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_point_cloud_batch,
        **loader_kwargs,
    )
    val_loaders = {
        name: DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_point_cloud_batch,
            **loader_kwargs,
        )
        for name, dataset in val_sets.items()
    }
    return train_loader, val_loaders, datasets


def format_class_prompts(domain_class_names: dict[str, list[str]], template: str) -> dict[str, list[str]]:
    prompts = {}
    for domain_name, class_names in domain_class_names.items():
        prompts[domain_name] = [template.format(class_name.replace("_", " ")) for class_name in class_names]
    return prompts


def build_text_prototypes(cfg: ExperimentConfig, domain_class_names: dict[str, list[str]]):
    if not uses_language_guided_head(cfg.head_type):
        return None, None

    from text_encoder import FrozenTextEmbedder

    prompted_class_names = format_class_prompts(domain_class_names, cfg.text_prompt_template)
    encoder = FrozenTextEmbedder(model_name=cfg.text_model_name, device="cpu")

    text_embeddings_by_domain = {}
    for domain_name, prompts in prompted_class_names.items():
        cache_path = encoder.default_cache_path(prompts, cfg.text_cache_dir, prefix=f"{domain_name}_language_guided")
        embeddings = encoder.encode_with_cache(prompts, cache_path)
        text_embeddings_by_domain[DOMAIN_TO_ID[domain_name]] = F.normalize(embeddings.float(), dim=1)

    first_embedding = next(iter(text_embeddings_by_domain.values()))
    cfg.text_embedding_dim = int(first_embedding.shape[1])
    return text_embeddings_by_domain, prompted_class_names


def classification_cross_entropy(logits_by_domain, labels, domain_ids, criterion):
    total_count = labels.numel()
    total_loss = labels.new_zeros((), dtype=torch.float32)
    for domain_idx, logits in logits_by_domain.items():
        mask = domain_ids == int(domain_idx)
        domain_labels = labels[mask]
        if domain_labels.numel() == 0:
            continue
        total_loss = total_loss + criterion(logits, domain_labels) * domain_labels.numel()
    return total_loss / max(total_count, 1)


def get_autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    use_amp: bool,
    head_type: str,
    text_embeddings_by_domain=None,
    language_guided_temperature: float = 0.07,
):
    model.train()
    loss_meter = AverageMeter()

    for points, labels, domain_ids in tqdm(loader, desc="train", leave=False):
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(device, use_amp):
            logits_by_domain = model(
                points,
                domain_ids,
                text_embeddings_by_domain=text_embeddings_by_domain if uses_language_guided_head(head_type) else None,
                temperature=language_guided_temperature,
            )
            cls_loss = classification_cross_entropy(logits_by_domain, labels, domain_ids, criterion)
            loss = cls_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), points.size(0))
    return {"train_loss": loss_meter.avg}


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    device,
    use_amp: bool = False,
    head_type: str = "decoupled",
    text_embeddings_by_domain=None,
    language_guided_temperature: float = 0.07,
):
    model.eval()
    correct = 0
    total = 0
    for points, labels, domain_ids in tqdm(loader, desc="eval", leave=False):
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)

        with get_autocast_context(device, use_amp):
            logits_by_domain = model(
                points,
                domain_ids,
                text_embeddings_by_domain=text_embeddings_by_domain if uses_language_guided_head(head_type) else None,
                temperature=language_guided_temperature,
            )
        for domain_idx, logits in logits_by_domain.items():
            mask = domain_ids == int(domain_idx)
            domain_labels = labels[mask]
            preds = logits.argmax(dim=1)
            correct += (preds == domain_labels).sum().item()
            total += domain_labels.numel()
    return correct / max(total, 1)


def evaluate_all(
    model,
    val_loaders,
    device,
    use_amp: bool = False,
    head_type: str = "decoupled",
    text_embeddings_by_domain=None,
    language_guided_temperature: float = 0.07,
):
    metrics = {}
    for name, loader in val_loaders.items():
        metrics[name] = evaluate_loader(
            model,
            loader,
            device,
            use_amp=use_amp,
            head_type=head_type,
            text_embeddings_by_domain=text_embeddings_by_domain,
            language_guided_temperature=language_guided_temperature,
        )
    metrics["val_acc"] = sum(metrics.values()) / len(metrics)
    return metrics


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    cfg: ExperimentConfig,
    epoch: int,
    metrics: dict,
    domain_class_names,
    domain_num_classes,
    prompted_class_names=None,
    text_embeddings_by_domain=None,
):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg.to_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "domain_class_names": domain_class_names,
            "domain_num_classes": domain_num_classes,
            "prompted_class_names": prompted_class_names,
            "text_embeddings_by_domain": {
                int(domain_idx): embeddings.cpu() for domain_idx, embeddings in (text_embeddings_by_domain or {}).items()
            },
        },
        path,
    )


def main():
    args = build_parser().parse_args()
    cfg = args_to_config(args)
    seed_everything(cfg.seed)
    device = choose_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    run_name = default_run_name(cfg)
    out_dir = prepare_output_dir(cfg.output_root, run_name)

    train_loader, val_loaders, datasets = make_loaders(cfg)
    domain_num_classes = datasets["domain_num_classes"]
    domain_class_names = datasets["domain_class_names"]
    text_embeddings_by_domain, prompted_class_names = build_text_prototypes(cfg, domain_class_names)

    num_classes_by_domain = [
        domain_num_classes["modelnet"],
        domain_num_classes["scanobjectnn"],
    ]
    model = PointNetClassifier(
        num_classes_by_domain=num_classes_by_domain,
        emb_dim=cfg.emb_dim,
        use_pdnorm=use_pdnorm(cfg.mode),
        dropout=cfg.dropout,
        num_domains=len(DOMAIN_TO_ID),
        backbone_type=cfg.backbone_type,
        dgcnn_k=cfg.dgcnn_k,
        head_type=cfg.head_type,
        text_embedding_dim=cfg.text_embedding_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "train_loss": [],
        "val_acc": [],
        "lr": [],
        "modelnet_acc": [],
        "scanobjectnn_acc": [],
    }

    best_acc = -1.0
    best_epoch = -1
    start_time = time.time()

    save_json(
        {
            **cfg.to_dict(),
            "domain_class_names": domain_class_names,
            "domain_num_classes": domain_num_classes,
            "prompted_class_names": prompted_class_names,
        },
        out_dir / "config.json",
    )

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp,
            head_type=cfg.head_type,
            text_embeddings_by_domain=text_embeddings_by_domain,
            language_guided_temperature=cfg.language_guided_temperature,
        )
        metrics = evaluate_all(
            model,
            val_loaders,
            device,
            use_amp=use_amp,
            head_type=cfg.head_type,
            text_embeddings_by_domain=text_embeddings_by_domain,
            language_guided_temperature=cfg.language_guided_temperature,
        )
        scheduler.step()

        history["train_loss"].append(train_stats["train_loss"])
        history["val_acc"].append(metrics["val_acc"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if "modelnet" in metrics:
            history["modelnet_acc"].append(metrics["modelnet"])
        if "scanobjectnn" in metrics:
            history["scanobjectnn_acc"].append(metrics["scanobjectnn"])

        log_line = (
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_stats['train_loss']:.4f} | val_acc={metrics['val_acc']:.4f}"
        )
        if "modelnet" in metrics:
            log_line += f" | modelnet={metrics['modelnet']:.4f}"
        if "scanobjectnn" in metrics:
            log_line += f" | scanobjectnn={metrics['scanobjectnn']:.4f}"
        print(log_line)

        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            cfg,
            epoch,
            metrics,
            domain_class_names,
            domain_num_classes,
            prompted_class_names=prompted_class_names,
            text_embeddings_by_domain=text_embeddings_by_domain,
        )
        if metrics["val_acc"] > best_acc:
            best_acc = metrics["val_acc"]
            best_epoch = epoch
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                cfg,
                epoch,
                metrics,
                domain_class_names,
                domain_num_classes,
                prompted_class_names=prompted_class_names,
                text_embeddings_by_domain=text_embeddings_by_domain,
            )

        save_json(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_acc": best_acc,
                "latest_metrics": metrics,
                "latest_train_stats": train_stats,
                "elapsed_seconds": time.time() - start_time,
            },
            out_dir / "summary.json",
        )
        plot_history(history, out_dir / "curves.png")
        save_json(history, out_dir / "history.json")

    final_metrics = evaluate_all(
        model,
        val_loaders,
        device,
        use_amp=use_amp,
        head_type=cfg.head_type,
        text_embeddings_by_domain=text_embeddings_by_domain,
        language_guided_temperature=cfg.language_guided_temperature,
    )
    best_checkpoint = torch.load(out_dir / "best.pt", map_location="cpu")
    best_metrics = best_checkpoint.get("metrics", {})
    print("\nTraining finished")
    print(f"Best accuracy: {best_acc:.4f} at epoch {best_epoch}")
    if "modelnet" in best_metrics:
        print(f"Best ModelNet accuracy: {best_metrics['modelnet']:.4f}")
    if "scanobjectnn" in best_metrics:
        print(f"Best ScanObjectNN accuracy: {best_metrics['scanobjectnn']:.4f}")
    print(f"Final accuracy: {final_metrics['val_acc']:.4f}")
    if "modelnet" in final_metrics:
        print(f"Final ModelNet accuracy: {final_metrics['modelnet']:.4f}")
    if "scanobjectnn" in final_metrics:
        print(f"Final ScanObjectNN accuracy: {final_metrics['scanobjectnn']:.4f}")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
