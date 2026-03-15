import torch
import torch.nn as nn
import torch.nn.functional as F


class PDNorm1d(nn.Module):
    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=False)
        self.to_gamma = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, channels),
        )
        self.to_beta = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, channels),
        )
        nn.init.zeros_(self.to_gamma[-1].weight)
        nn.init.zeros_(self.to_gamma[-1].bias)
        nn.init.zeros_(self.to_beta[-1].weight)
        nn.init.zeros_(self.to_beta[-1].bias)

    def forward(self, x: torch.Tensor, domain_emb: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gamma = 1.0 + self.to_gamma(domain_emb).unsqueeze(-1)
        beta = self.to_beta(domain_emb).unsqueeze(-1)
        return x * gamma + beta


class PointBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_pdnorm: bool, emb_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.use_pdnorm = use_pdnorm
        if use_pdnorm:
            self.norm = PDNorm1d(out_channels, emb_dim)
        else:
            self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, domain_emb: torch.Tensor = None) -> torch.Tensor:
        x = self.conv(x)
        if self.use_pdnorm:
            x = self.norm(x, domain_emb)
        else:
            x = self.norm(x)
        return F.relu(x, inplace=True)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class SemanticProjectionHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class PointNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes_by_domain,
        emb_dim: int = 16,
        use_pdnorm: bool = False,
        dropout: float = 0.3,
        num_domains: int = 2,
        head_type: str = "decoupled",
        text_embedding_dim: int = 384,
    ):
        super().__init__()
        self.use_pdnorm = use_pdnorm
        self.num_domains = num_domains
        self.num_classes_by_domain = list(num_classes_by_domain)
        self.head_type = head_type
        self.text_embedding_dim = text_embedding_dim

        if use_pdnorm:
            self.domain_embedding = nn.Embedding(num_domains, emb_dim)
        else:
            self.domain_embedding = None

        self.block1 = PointBlock(3, 64, use_pdnorm, emb_dim)
        self.block2 = PointBlock(64, 128, use_pdnorm, emb_dim)
        self.block3 = PointBlock(128, 256, use_pdnorm, emb_dim)
        if head_type == "decoupled":
            self.heads = nn.ModuleList(
                [ClassificationHead(256, num_classes, dropout=dropout) for num_classes in self.num_classes_by_domain]
            )
            self.language_guided_head = None
        elif head_type == "language_guided":
            self.heads = None
            self.language_guided_head = SemanticProjectionHead(256, text_embedding_dim)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward_features(self, points: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        domain_emb = None
        if self.use_pdnorm:
            if domain_ids is None:
                raise ValueError("domain_ids are required when PDNorm is enabled.")
            domain_emb = self.domain_embedding(domain_ids)

        x = self.block1(points, domain_emb)
        x = self.block2(x, domain_emb)
        x = self.block3(x, domain_emb)
        return torch.max(x, dim=2).values

    def build_language_guided_logits(
        self,
        language_features: torch.Tensor,
        domain_ids: torch.Tensor,
        text_embeddings_by_domain: dict[int, torch.Tensor],
        temperature: float = 1.0,
    ) -> dict[int, torch.Tensor]:
        if text_embeddings_by_domain is None:
            raise ValueError("text_embeddings_by_domain are required for the language-guided head.")

        normalized_features = F.normalize(language_features.float(), dim=1)
        logits_by_domain = {}
        for domain_idx in torch.unique(domain_ids).tolist():
            mask = domain_ids == int(domain_idx)
            domain_text_embeddings = text_embeddings_by_domain[int(domain_idx)].to(
                normalized_features.device,
                non_blocking=True,
            )
            logits = normalized_features[mask] @ domain_text_embeddings.T
            logits_by_domain[int(domain_idx)] = logits / temperature
        return logits_by_domain

    def forward_outputs(
        self,
        points: torch.Tensor,
        domain_ids: torch.Tensor,
        text_embeddings_by_domain: dict[int, torch.Tensor] | None = None,
        temperature: float = 1.0,
    ):
        if domain_ids is None:
            raise ValueError("domain_ids are required for dataset-aware prediction heads.")

        features = self.forward_features(points, domain_ids)
        outputs = {
            "features": features,
        }

        if self.head_type == "decoupled":
            logits_by_domain = {}
            for domain_idx in torch.unique(domain_ids).tolist():
                mask = domain_ids == int(domain_idx)
                logits_by_domain[int(domain_idx)] = self.heads[int(domain_idx)](features[mask])
            outputs["logits_by_domain"] = logits_by_domain
        else:
            language_features = self.language_guided_head(features)
            outputs["language_features"] = language_features
            if text_embeddings_by_domain is not None:
                outputs["logits_by_domain"] = self.build_language_guided_logits(
                    language_features,
                    domain_ids,
                    text_embeddings_by_domain=text_embeddings_by_domain,
                    temperature=temperature,
                )
        return outputs

    def forward(
        self,
        points: torch.Tensor,
        domain_ids: torch.Tensor,
        text_embeddings_by_domain: dict[int, torch.Tensor] | None = None,
        temperature: float = 1.0,
    ):
        return self.forward_outputs(
            points,
            domain_ids,
            text_embeddings_by_domain=text_embeddings_by_domain,
            temperature=temperature,
        )
