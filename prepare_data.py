import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


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
SCANOBJECTNN_VARIANTS = {
    "OBJ_ONLY": "objectdataset.h5",
    "OBJ_BG": "objectdataset_withbg.h5",
    "PB_T25": "objectdataset_augmented25_norot.h5",
    "PB_T25_R": "objectdataset_augmented25rot.h5",
    "PB_T50_RS": "objectdataset_augmentedrot_scale75.h5",
}
SCANOBJECTNN_SPLITS = ("split1", "split2", "split4", "split1_nobg", "split2_nobg", "split4_nobg")


def parse_off(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not lines:
        raise ValueError(f"Empty OFF file: {path}")

    first = lines[0]
    if first == "OFF":
        header = lines[1].split()
        start_idx = 2
    elif first.startswith("OFF"):
        header = first[3:].strip().split()
        start_idx = 1
    else:
        raise ValueError(f"Invalid OFF header in: {path}")

    num_vertices, num_faces = int(header[0]), int(header[1])
    vertices = np.array(
        [[float(x) for x in lines[start_idx + i].split()[:3]] for i in range(num_vertices)],
        dtype=np.float64,
    )

    faces = []
    face_start = start_idx + num_vertices
    for i in range(num_faces):
        parts = [int(x) for x in lines[face_start + i].split()]
        degree = parts[0]
        indices = parts[1 : degree + 1]
        if degree < 3:
            continue
        if degree == 3:
            faces.append(indices)
        else:
            for j in range(1, degree - 1):
                faces.append([indices[0], indices[j], indices[j + 1]])

    if not faces:
        raise ValueError(f"No valid faces in OFF file: {path}")
    return vertices, np.asarray(faces, dtype=np.int32)


def sample_points_from_mesh(vertices: np.ndarray, faces: np.ndarray, num_points: int) -> np.ndarray:
    finite_vertex_mask = np.isfinite(vertices).all(axis=1)
    if not np.any(finite_vertex_mask):
        raise ValueError("Mesh has no finite vertices.")

    valid_faces = finite_vertex_mask[faces].all(axis=1)
    faces = faces[valid_faces]
    if len(faces) == 0:
        clean_vertices = vertices[finite_vertex_mask]
        ids = np.random.choice(len(clean_vertices), size=num_points, replace=True)
        return clean_vertices[ids].astype(np.float32)

    triangles = vertices[faces]
    vec1 = triangles[:, 1] - triangles[:, 0]
    vec2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(vec1, vec2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    valid_area_mask = np.isfinite(areas) & (areas > 1e-12)
    triangles = triangles[valid_area_mask]
    areas = areas[valid_area_mask]
    if len(triangles) == 0:
        clean_vertices = vertices[finite_vertex_mask]
        ids = np.random.choice(len(clean_vertices), size=num_points, replace=True)
        return clean_vertices[ids].astype(np.float32)

    area_sum = areas.sum()
    if not np.isfinite(area_sum) or area_sum <= 0:
        clean_vertices = vertices[finite_vertex_mask]
        ids = np.random.choice(len(clean_vertices), size=num_points, replace=True)
        return clean_vertices[ids].astype(np.float32)

    probs = areas / area_sum
    if not np.isfinite(probs).all():
        clean_vertices = vertices[finite_vertex_mask]
        ids = np.random.choice(len(clean_vertices), size=num_points, replace=True)
        return clean_vertices[ids].astype(np.float32)

    triangle_ids = np.random.choice(len(triangles), size=num_points, p=probs)
    chosen = triangles[triangle_ids]

    r1 = np.sqrt(np.random.rand(num_points, 1)).astype(np.float64)
    r2 = np.random.rand(num_points, 1).astype(np.float64)
    samples = (
        (1.0 - r1) * chosen[:, 0]
        + r1 * (1.0 - r2) * chosen[:, 1]
        + r1 * r2 * chosen[:, 2]
    )
    return samples.astype(np.float32)


def has_resampled_layout(root: Path) -> bool:
    return (root / "modelnet40_train.txt").exists() and (root / "modelnet40_test.txt").exists()


def has_princeton_off_layout(root: Path) -> bool:
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    return any((class_dir / "train").exists() or (class_dir / "test").exists() for class_dir in class_dirs)


def find_named_root(extract_parent: Path, predicate, expected_name: str) -> Path:
    if predicate(extract_parent):
        return extract_parent
    candidates = [p for p in extract_parent.rglob("*") if p.is_dir() and predicate(p)]
    if len(candidates) == 1:
        return candidates[0]
    exact = extract_parent / expected_name
    if exact.exists() and predicate(exact):
        return exact
    raise FileNotFoundError(f"Could not locate extracted dataset directory inside: {extract_parent}")


def convert_princeton_modelnet40_to_npy(raw_root: Path, output_root: Path, sampled_points: int = 2048):
    output_root.mkdir(parents=True, exist_ok=True)
    class_names = sorted([p.name for p in raw_root.iterdir() if p.is_dir()])
    class_to_label = {name: idx for idx, name in enumerate(class_names)}

    split_points = {"train": [], "test": []}
    split_labels = {"train": [], "test": []}
    split_skipped = {"train": 0, "test": 0}

    for class_name in tqdm(class_names, desc="classes"):
        class_dir = raw_root / class_name
        for split in ("train", "test"):
            split_dir = class_dir / split
            if not split_dir.exists():
                continue
            off_files = sorted(split_dir.rglob("*.off"))
            for off_path in tqdm(off_files, desc=f"{class_name}/{split}", leave=False):
                try:
                    vertices, faces = parse_off(off_path)
                    points = sample_points_from_mesh(vertices, faces, sampled_points)
                except Exception as exc:
                    split_skipped[split] += 1
                    print(f"Skipping invalid mesh {off_path}: {exc}")
                    continue
                split_points[split].append(points)
                split_labels[split].append(class_to_label[class_name])

    for split in ("train", "test"):
        if not split_points[split]:
            raise FileNotFoundError(f"No {split} samples were converted from Princeton ModelNet40.")
        np.save(output_root / f"{split}_points.npy", np.stack(split_points[split]).astype(np.float32))
        np.save(output_root / f"{split}_labels.npy", np.asarray(split_labels[split], dtype=np.int64))

    (output_root / "modelnet40_shape_names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    with (output_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_format": "princeton_off",
                "sampled_points_per_shape": sampled_points,
                "num_classes": len(class_names),
                "num_train": len(split_labels["train"]),
                "num_test": len(split_labels["test"]),
                "skipped_train": split_skipped["train"],
                "skipped_test": split_skipped["test"],
            },
            f,
            indent=2,
        )


def extract_archive(archive_path: Path, destination: Path):
    print(f"Extracting archive to: {destination}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(destination)


def ensure_clean_target(target: Path, force: bool):
    if target.exists() and not force:
        print(f"Dataset already exists: {target}")
        return False
    if target.exists() and force:
        print(f"Removing existing dataset directory: {target}")
        shutil.rmtree(target)
    return True


def ensure_modelnet40(data_root: str, archive_path: str, force: bool = False, sampled_points: int = 2048, keep_raw: bool = False):
    target = Path(data_root)
    archive = Path(archive_path)
    extract_dir = target.parent / f"_{archive.stem}_raw"

    if not ensure_clean_target(target, force):
        return target
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    print(f"Using local archive: {archive}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_archive(archive, extract_dir)

    extracted_root = find_named_root(
        extract_dir,
        lambda root: has_resampled_layout(root) or has_princeton_off_layout(root),
        "ModelNet40",
    )

    if has_resampled_layout(extracted_root):
        shutil.move(str(extracted_root), str(target))
    elif has_princeton_off_layout(extracted_root):
        print("Detected Princeton OFF mesh layout. Converting meshes to point-cloud numpy files...")
        convert_princeton_modelnet40_to_npy(extracted_root, target, sampled_points=sampled_points)
    else:
        raise FileNotFoundError(f"Unsupported extracted ModelNet40 layout under: {extracted_root}")

    if not keep_raw and extract_dir.exists():
        shutil.rmtree(extract_dir)

    print(f"ModelNet40 is ready at: {target}")
    return target


def convert_scanobjectnn_h5_to_npy(raw_root: Path, output_root: Path, variant: str, split_name: str):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Preparing ScanObjectNN from official h5 files requires the optional dependency `h5py`.\n"
            "Install it with: uv pip install -e .[preprocess]"
        ) from exc

    suffix = SCANOBJECTNN_VARIANTS[variant]
    split_root = raw_root / split_name
    train_path = split_root / f"training_{suffix}"
    test_path = split_root / f"test_{suffix}"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Could not find ScanObjectNN split files for variant {variant} under {split_root}."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for split, path in {"train": train_path, "test": test_path}.items():
        with h5py.File(path, "r") as f:
            points = np.asarray(f["data"], dtype=np.float32)
            labels = np.asarray(f["label"], dtype=np.int64).reshape(-1)
        np.save(output_root / f"{split}_points.npy", points)
        np.save(output_root / f"{split}_labels.npy", labels)

    (output_root / "scanobjectnn_shape_names.txt").write_text(
        "\n".join(SCANOBJECTNN_CLASS_NAMES) + "\n",
        encoding="utf-8",
    )
    with (output_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_format": "scanobjectnn_h5",
                "variant": variant,
                "split": split_name,
                "num_classes": len(SCANOBJECTNN_CLASS_NAMES),
            },
            f,
            indent=2,
        )


def ensure_scanobjectnn(
    data_root: str,
    archive_path: str,
    force: bool = False,
    variant: str = "PB_T50_RS",
    split_name: str = "split1",
    keep_raw: bool = False,
):
    target = Path(data_root)
    archive = Path(archive_path)
    extract_dir = target.parent / f"_{archive.stem}_raw"

    if variant not in SCANOBJECTNN_VARIANTS:
        raise ValueError(f"Unsupported ScanObjectNN variant: {variant}")
    if split_name not in SCANOBJECTNN_SPLITS:
        raise ValueError(f"Unsupported ScanObjectNN split: {split_name}")
    if not ensure_clean_target(target, force):
        return target
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    print(f"Using local archive: {archive}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_archive(archive, extract_dir)

    extracted_root = find_named_root(
        extract_dir,
        lambda root: any((root / split).exists() for split in SCANOBJECTNN_SPLITS),
        "h5_files",
    )
    convert_scanobjectnn_h5_to_npy(extracted_root, target, variant=variant, split_name=split_name)

    if not keep_raw and extract_dir.exists():
        shutil.rmtree(extract_dir)

    print(f"ScanObjectNN is ready at: {target}")
    return target


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare local datasets for miniPPT")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    modelnet = subparsers.add_parser("modelnet40", help="Prepare ModelNet40 from a local archive")
    modelnet.add_argument("--archive_path", type=str, required=True)
    modelnet.add_argument("--data_root", type=str, default="data/modelnet40_normal_resampled")
    modelnet.add_argument("--force", action="store_true")
    modelnet.add_argument("--sampled_points", type=int, default=2048)
    modelnet.add_argument("--keep_raw", action="store_true")

    scan = subparsers.add_parser("scanobjectnn", help="Prepare ScanObjectNN from a local archive")
    scan.add_argument("--archive_path", type=str, required=True)
    scan.add_argument("--data_root", type=str, default="data/scanobjectnn_npy")
    scan.add_argument("--variant", type=str, choices=sorted(SCANOBJECTNN_VARIANTS), default="PB_T50_RS")
    scan.add_argument("--split", type=str, choices=SCANOBJECTNN_SPLITS, default="split1")
    scan.add_argument("--force", action="store_true")
    scan.add_argument("--keep_raw", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.dataset == "modelnet40":
        ensure_modelnet40(
            data_root=args.data_root,
            archive_path=args.archive_path,
            force=args.force,
            sampled_points=args.sampled_points,
            keep_raw=args.keep_raw,
        )
    elif args.dataset == "scanobjectnn":
        ensure_scanobjectnn(
            data_root=args.data_root,
            archive_path=args.archive_path,
            force=args.force,
            variant=args.variant,
            split_name=args.split,
            keep_raw=args.keep_raw,
        )


if __name__ == "__main__":
    main()
