from dataclasses import asdict, dataclass


DOMAIN_NUM_CLASSES = {
    "modelnet": 40,
    "scanobjectnn": 15,
}
MODE_ALIASES = {
    "train_a_only": "train_modelnet_only",
    "train_b_only": "train_scanobjectnn_only",
}
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
    scanobjectnn_variant: str = "PB_T50_RS"
    output_root: str = "runs"
    mode: str = "train_joint_pdnorm"
    epochs: int = 40 
    batch_size: int = 64
    num_workers: int = 4
    num_points: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    emb_dim: int = 16
    dropout: float = 0.3
    seed: int = 42
    cache_data: bool = False
    device: str = "cuda"
    amp: bool = True
    exp_name: str = ""

    def to_dict(self):
        return asdict(self)


def canonical_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def use_pdnorm(mode: str) -> bool:
    return canonical_mode(mode) == "train_joint_pdnorm"


def is_joint_mode(mode: str) -> bool:
    return canonical_mode(mode) in {"train_joint_naive", "train_joint_pdnorm"}


def default_run_name(cfg: ExperimentConfig) -> str:
    mode = canonical_mode(cfg.mode)
    if cfg.exp_name:
        return cfg.exp_name
    return f"{mode}_pts{cfg.num_points}_bs{cfg.batch_size}_ep{cfg.epochs}_seed{cfg.seed}"
