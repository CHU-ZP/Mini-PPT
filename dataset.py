from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


DOMAIN_NAMES = ("modelnet", "scanobjectnn")
DOMAIN_TO_ID = {name: idx for idx, name in enumerate(DOMAIN_NAMES)}
SCANOBJECTNN_CLASS_NAMES = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    points = points - np.mean(points, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 1e-8:
        points = points / scale
    return points


def sample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    if points.shape[0] >= num_points:
        choice = np.random.choice(points.shape[0], num_points, replace=False)
    else:
        extra = np.random.choice(points.shape[0], num_points - points.shape[0], replace=True)
        choice = np.concatenate([np.arange(points.shape[0]), extra], axis=0)
    return points[choice]


def rotation_matrix_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def jitter_points(points: np.ndarray, sigma: float = 0.01, clip: float = 0.03) -> np.ndarray:
    noise = np.clip(np.random.normal(0.0, sigma, size=points.shape), -clip, clip)
    return points + noise.astype(np.float32)


def augment_train_points(points: np.ndarray) -> np.ndarray:
    points = points @ rotation_matrix_y(np.random.uniform(0.0, 2.0 * np.pi)).T
    points = jitter_points(points, sigma=0.01, clip=0.03)
    return points.astype(np.float32)


class ModelNet40Base:
    def __init__(self, root: str, split: str = "train", use_cache: bool = False):
        self.root = Path(root)
        self.split = split
        self.use_cache = use_cache
        self.cache = {}
        self.samples, self.class_names = self._discover_samples()

    def _discover_samples(self):
        txt_index = self.root / f"modelnet40_{self.split}.txt"
        shape_names = self.root / "modelnet40_shape_names.txt"
        if txt_index.exists() and shape_names.exists():
            return self._load_txt_format(txt_index, shape_names)

        points_path = self.root / f"{self.split}_points.npy"
        labels_path = self.root / f"{self.split}_labels.npy"
        if points_path.exists() and labels_path.exists():
            return self._load_npy_format(points_path, labels_path)

        raise FileNotFoundError(
            f"Cannot find ModelNet40 data under {self.root}. "
            "Expected txt layout or train/test npy files."
        )

    def _load_txt_format(self, txt_index: Path, shape_names: Path):
        class_names = [line.strip() for line in shape_names.read_text().splitlines() if line.strip()]
        samples = []
        name_to_label = {name: idx for idx, name in enumerate(class_names)}
        for line in txt_index.read_text().splitlines():
            shape_id = line.strip()
            if not shape_id:
                continue
            class_name = "_".join(shape_id.split("_")[:-1])
            file_path = self.root / class_name / f"{shape_id}.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing point file: {file_path}")
            samples.append((file_path, name_to_label[class_name]))
        return samples, class_names

    def _load_npy_format(self, points_path: Path, labels_path: Path):
        self.npy_points = np.load(points_path).astype(np.float32)
        labels = np.load(labels_path).reshape(-1)
        class_names_path = self.root / "modelnet40_shape_names.txt"
        if class_names_path.exists():
            class_names = [line.strip() for line in class_names_path.read_text().splitlines() if line.strip()]
        else:
            num_classes = int(labels.max()) + 1
            class_names = [f"class_{idx:02d}" for idx in range(num_classes)]
        samples = [(idx, int(label)) for idx, label in enumerate(labels)]
        return samples, class_names

    def _load_txt_points(self, path: Path) -> np.ndarray:
        try:
            points = np.loadtxt(path, delimiter=",", dtype=np.float32)
        except ValueError:
            points = np.loadtxt(path, dtype=np.float32)
        return points[:, :3]

    def load_points(self, key):
        if self.use_cache and key in self.cache:
            return self.cache[key].copy()

        if isinstance(key, int):
            points = self.npy_points[key][:, :3]
        else:
            points = self._load_txt_points(key)

        if self.use_cache:
            self.cache[key] = points.copy()
        return points

    def __len__(self):
        return len(self.samples)


class ScanObjectNNBase:
    def __init__(self, root: str, split: str = "train", use_cache: bool = False):
        self.root = Path(root)
        self.split = split
        self.use_cache = use_cache
        self.cache = {}
        self.samples, self.class_names = self._discover_samples()

    def _discover_samples(self):
        points_path = self.root / f"{self.split}_points.npy"
        labels_path = self.root / f"{self.split}_labels.npy"
        if not points_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Cannot find ScanObjectNN npy data under {self.root}. "
                "Expected train_points.npy / train_labels.npy / test_points.npy / test_labels.npy"
            )
        self.npy_points = np.load(points_path).astype(np.float32)
        labels = np.load(labels_path).reshape(-1)

        class_names_candidates = [
            self.root / "scanobjectnn_shape_names.txt",
            self.root / "class_names.txt",
        ]
        class_names = None
        for candidate in class_names_candidates:
            if candidate.exists():
                class_names = [line.strip() for line in candidate.read_text().splitlines() if line.strip()]
                break
        if class_names is None:
            class_names = SCANOBJECTNN_CLASS_NAMES

        samples = [(idx, int(label)) for idx, label in enumerate(labels)]
        return samples, class_names

    def load_points(self, key):
        if self.use_cache and key in self.cache:
            return self.cache[key].copy()
        points = self.npy_points[key][:, :3]
        if self.use_cache:
            self.cache[key] = points.copy()
        return points

    def __len__(self):
        return len(self.samples)


class PointCloudDataset(Dataset):
    def __init__(self, base_dataset, domain_id: int, num_points: int, split: str):
        self.base_dataset = base_dataset
        self.domain_id = domain_id
        self.num_points = num_points
        self.split = split
        self.is_train = split == "train"
        self.class_names = list(base_dataset.class_names)
        self.samples = list(base_dataset.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        key, label = self.samples[index]
        points = self.base_dataset.load_points(key)
        points = sample_points(points, self.num_points)
        points = normalize_points(points)
        if self.is_train:
            points = augment_train_points(points)

        points = torch.from_numpy(points.T.copy()).float()
        label = torch.tensor(label, dtype=torch.long)
        domain_id = torch.tensor(self.domain_id, dtype=torch.long)
        return points, label, domain_id


def build_datasets(
    modelnet_root: str,
    scanobjectnn_root: str,
    num_points: int,
    use_cache: bool = False,
):
    modelnet_train_base = ModelNet40Base(modelnet_root, split="train", use_cache=use_cache)
    modelnet_val_base = ModelNet40Base(modelnet_root, split="test", use_cache=use_cache)
    scan_train_base = ScanObjectNNBase(scanobjectnn_root, split="train", use_cache=use_cache)
    scan_val_base = ScanObjectNNBase(scanobjectnn_root, split="test", use_cache=use_cache)

    train_modelnet = PointCloudDataset(
        modelnet_train_base,
        domain_id=DOMAIN_TO_ID["modelnet"],
        num_points=num_points,
        split="train",
    )
    val_modelnet = PointCloudDataset(
        modelnet_val_base,
        domain_id=DOMAIN_TO_ID["modelnet"],
        num_points=num_points,
        split="test",
    )
    train_scanobjectnn = PointCloudDataset(
        scan_train_base,
        domain_id=DOMAIN_TO_ID["scanobjectnn"],
        num_points=num_points,
        split="train",
    )
    val_scanobjectnn = PointCloudDataset(
        scan_val_base,
        domain_id=DOMAIN_TO_ID["scanobjectnn"],
        num_points=num_points,
        split="test",
    )

    domain_class_names = {
        "modelnet": train_modelnet.class_names,
        "scanobjectnn": train_scanobjectnn.class_names,
    }
    domain_num_classes = {name: len(class_names) for name, class_names in domain_class_names.items()}

    return {
        "train_modelnet": train_modelnet,
        "train_scanobjectnn": train_scanobjectnn,
        "val_modelnet": val_modelnet,
        "val_scanobjectnn": val_scanobjectnn,
        "joint_train": ConcatDataset([train_modelnet, train_scanobjectnn]),
        "domain_class_names": domain_class_names,
        "domain_num_classes": domain_num_classes,
    }


def collate_point_cloud_batch(batch):
    points_list, labels_list, domain_ids_list = zip(*batch)
    points = torch.stack(points_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    domain_ids = torch.stack(domain_ids_list, dim=0)
    return points, labels, domain_ids
