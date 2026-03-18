from dataclasses import asdict, dataclass, fields


MODE_ALIASES = {
    "train_a_only": "train_modelnet_only",
    "train_b_only": "train_scanobjectnn_only",
}
HEAD_TYPES = (
    "decoupled",
    "language_guided",
)
BACKBONE_TYPES = (
    "pointnet",
    "dgcnn",
)
MODES = (
    "train_modelnet_only",
    "train_scanobjectnn_only",
    "train_joint_naive",
    "train_joint_pdnorm",
    "train_a_only",
    "train_b_only",
)


@dataclass
class ExperimentConfig:
    modelnet_root: str = "data/modelnet40_princeton_npy"
    scanobjectnn_root: str = "data/scanobjectnn_npy"
    output_root: str = "runs"
    mode: str = "train_joint_pdnorm"
    epochs: int = 50
    batch_size: int = 128
    num_workers: int = 4
    num_points: int = 1024
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    emb_dim: int = 16
    dropout: float = 0.3
    seed: int = 42
    cache_data: bool = False
    device: str = "cuda"
    amp: bool = True
    backbone_type: str = "pointnet"
    dgcnn_k: int = 20
    head_type: str = "decoupled"
    text_embedding_dim: int = 384
    language_guided_temperature: float = 0.07
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_prompt_template: str = "a 3d point cloud of a {}"
    text_cache_dir: str = "artifacts/text_cache"
    exp_name: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**filtered)


def canonical_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def use_pdnorm(mode: str) -> bool:
    return canonical_mode(mode) == "train_joint_pdnorm"


def uses_language_guided_head(head_type: str) -> bool:
    return head_type == "language_guided"


def default_run_name(cfg: ExperimentConfig) -> str:
    mode = canonical_mode(cfg.mode)
    if cfg.exp_name:
        return cfg.exp_name
    if cfg.backbone_type != "pointnet":
        mode = f"{mode}_{cfg.backbone_type}"
    if cfg.head_type != "decoupled":
        mode = f"{mode}_{cfg.head_type}"
    return f"{mode}_pts{cfg.num_points}_bs{cfg.batch_size}_ep{cfg.epochs}_seed{cfg.seed}"
