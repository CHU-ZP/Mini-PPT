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


class FeatureNorm1d(nn.Module):
    def __init__(self, channels: int, use_pdnorm: bool, emb_dim: int):
        super().__init__()
        self.use_pdnorm = use_pdnorm
        if use_pdnorm:
            self.norm = PDNorm1d(channels, emb_dim)
        else:
            self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor, domain_emb: torch.Tensor = None) -> torch.Tensor:
        if self.use_pdnorm:
            return self.norm(x, domain_emb)
        return self.norm(x)


class PointBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_pdnorm: bool, emb_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = FeatureNorm1d(out_channels, use_pdnorm, emb_dim)

    def forward(self, x: torch.Tensor, domain_emb: torch.Tensor = None) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x, domain_emb)
        return F.relu(x, inplace=True)


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    points = x.transpose(2, 1).contiguous()
    num_points = points.size(1)
    effective_k = min(k + 1, num_points)
    distances = torch.cdist(points, points)
    idx = distances.topk(k=effective_k, dim=-1, largest=False)[1]
    if effective_k > 1:
        idx = idx[:, :, 1:]
    if idx.size(-1) == 0:
        idx = idx.new_zeros(points.size(0), num_points, 1)
    return idx


def get_graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    batch_size, channels, num_points = x.shape
    idx = knn(x.detach(), k)
    k_eff = idx.size(-1)

    idx_base = torch.arange(batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).reshape(-1)

    x_transposed = x.transpose(2, 1).contiguous()
    neighbors = x_transposed.reshape(batch_size * num_points, channels)[idx]
    neighbors = neighbors.reshape(batch_size, num_points, k_eff, channels)
    centers = x_transposed.unsqueeze(2).expand(-1, -1, k_eff, -1)

    feature = torch.cat((neighbors - centers, centers), dim=-1)
    return feature.permute(0, 3, 1, 2).contiguous()


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int, use_pdnorm: bool, emb_dim: int):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.norm = FeatureNorm1d(out_channels, use_pdnorm, emb_dim)

    def forward(self, x: torch.Tensor, domain_emb: torch.Tensor = None) -> torch.Tensor:
        edge_feature = get_graph_feature(x, self.k)
        x = self.conv(edge_feature)
        x = x.max(dim=-1).values
        x = self.norm(x, domain_emb)
        return F.relu(x, inplace=True)


class PointNetBackbone(nn.Module):
    def __init__(self, use_pdnorm: bool, emb_dim: int):
        super().__init__()
        self.block1 = PointBlock(3, 64, use_pdnorm, emb_dim)
        self.block2 = PointBlock(64, 128, use_pdnorm, emb_dim)
        self.block3 = PointBlock(128, 256, use_pdnorm, emb_dim)

    def forward(self, points: torch.Tensor, domain_emb: torch.Tensor = None) -> torch.Tensor:
        x = self.block1(points, domain_emb)
        x = self.block2(x, domain_emb)
        x = self.block3(x, domain_emb)
        return torch.max(x, dim=2).values


class DGCNNBackbone(nn.Module):
    def __init__(self, use_pdnorm: bool, emb_dim: int, k: int = 20):
        super().__init__()
        self.edge1 = EdgeConvBlock(3, 64, k=k, use_pdnorm=use_pdnorm, emb_dim=emb_dim)
        self.edge2 = EdgeConvBlock(64, 64, k=k, use_pdnorm=use_pdnorm, emb_dim=emb_dim)
        self.edge3 = EdgeConvBlock(64, 128, k=k, use_pdnorm=use_pdnorm, emb_dim=emb_dim)
        self.edge4 = EdgeConvBlock(128, 256, k=k, use_pdnorm=use_pdnorm, emb_dim=emb_dim)
        self.fuse = PointBlock(64 + 64 + 128 + 256, 256, use_pdnorm, emb_dim)

    def forward(self, points: torch.Tensor, domain_emb: torch.Tensor = None) -> torch.Tensor:
        x1 = self.edge1(points, domain_emb)
        x2 = self.edge2(x1, domain_emb)
        x3 = self.edge3(x2, domain_emb)
        x4 = self.edge4(x3, domain_emb)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.fuse(x, domain_emb)
        return torch.max(x, dim=2).values


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


class PointCloudClassifier(nn.Module):
    def __init__(
        self,
        num_classes_by_domain,
        emb_dim: int = 16,
        use_pdnorm: bool = False,
        dropout: float = 0.3,
        num_domains: int = 2,
        head_type: str = "decoupled",
        text_embedding_dim: int = 384,
        backbone_type: str = "pointnet",
        dgcnn_k: int = 20,
    ):
        super().__init__()
        self.use_pdnorm = use_pdnorm
        self.num_domains = num_domains
        self.num_classes_by_domain = list(num_classes_by_domain)
        self.head_type = head_type
        self.text_embedding_dim = text_embedding_dim
        self.backbone_type = backbone_type

        if use_pdnorm:
            self.domain_embedding = nn.Embedding(num_domains, emb_dim)
        else:
            self.domain_embedding = None

        if backbone_type == "pointnet":
            self.backbone = PointNetBackbone(use_pdnorm=use_pdnorm, emb_dim=emb_dim)
        elif backbone_type == "dgcnn":
            self.backbone = DGCNNBackbone(use_pdnorm=use_pdnorm, emb_dim=emb_dim, k=dgcnn_k)
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

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
        return self.backbone(points, domain_emb)

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

    def forward(
        self,
        points: torch.Tensor,
        domain_ids: torch.Tensor,
        text_embeddings_by_domain: dict[int, torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> dict[int, torch.Tensor]:
        if domain_ids is None:
            raise ValueError("domain_ids are required for dataset-aware prediction heads.")

        features = self.forward_features(points, domain_ids)
        if self.head_type == "decoupled":
            logits_by_domain = {}
            for domain_idx in torch.unique(domain_ids).tolist():
                mask = domain_ids == int(domain_idx)
                logits_by_domain[int(domain_idx)] = self.heads[int(domain_idx)](features[mask])
            return logits_by_domain

        if text_embeddings_by_domain is None:
            raise ValueError("text_embeddings_by_domain are required when using the language-guided head.")

        language_features = self.language_guided_head(features)
        return self.build_language_guided_logits(
            language_features,
            domain_ids,
            text_embeddings_by_domain=text_embeddings_by_domain,
            temperature=temperature,
        )


PointNetClassifier = PointCloudClassifier
