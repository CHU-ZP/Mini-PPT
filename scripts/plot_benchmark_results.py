import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT_DIR / "runs"
RESULTS_DIR = ROOT_DIR / "results"


def to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def canonical_method_name(report_name: str) -> str:
    mapping = {
        "Single-dataset training on ModelNet40": "single_modelnet40",
        "Single-dataset training on ScanObjectNN": "single_scanobjectnn",
        "Multi-dataset joint training, Decoupled": "joint_decoupled",
        "Multi-dataset joint training, Decoupled + PDNorm": "joint_decoupled_pdnorm",
        "Multi-dataset joint training, Lightweight Language-guided Categorical Alignment": "joint_language_guided",
        "Multi-dataset joint training, Lightweight Language-guided Categorical Alignment + PDNorm": "joint_language_guided_pdnorm",
    }
    return mapping.get(report_name, report_name.lower().replace(" ", "_"))


def display_method_name(method_name: str) -> str:
    mapping = {
        "single_modelnet40": "Single ModelNet40",
        "single_scanobjectnn": "Single ScanObjectNN",
        "joint_decoupled": "Decoupled + Naive",
        "joint_decoupled_pdnorm": "Decoupled + PDNorm",
        "joint_language_guided": "Language-guided + Naive",
        "joint_language_guided_pdnorm": "Language-guided + PDNorm",
    }
    return mapping[method_name]


def load_result_rows() -> list[dict]:
    rows = []
    for csv_path in sorted(RUNS_DIR.glob("*results.csv")):
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["best_acc"] = to_float(row["best_acc"])
                row["modelnet_acc"] = to_float(row["modelnet_acc"])
                row["scanobjectnn_acc"] = to_float(row["scanobjectnn_acc"])
                row["best_epoch"] = int(row["best_epoch"])
                row["method_name"] = canonical_method_name(row["report_name"])
                row["history_path"] = str(RUNS_DIR / row["exp_name"] / "history.json")
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No benchmark result CSV files found under: {RUNS_DIR}")
    return rows


def write_summary_tables(rows: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "summary_table.csv"
    md_path = RESULTS_DIR / "summary_table.md"

    fieldnames = [
        "method_name",
        "report_name",
        "backbone_type",
        "mode",
        "head_type",
        "best_epoch",
        "best_acc",
        "modelnet_acc",
        "scanobjectnn_acc",
        "checkpoint",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    header = [
        "Method",
        "Backbone",
        "Mode",
        "Head",
        "Best Epoch",
        "Mean Acc",
        "ModelNet40 Acc",
        "ScanObjectNN Acc",
    ]
    lines = [
        "# Benchmark Summary",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    display_method_name(row["method_name"]),
                    row["backbone_type"],
                    row["mode"],
                    row["head_type"],
                    str(row["best_epoch"]),
                    f"{row['best_acc']:.4f}",
                    "-" if row["modelnet_acc"] is None else f"{row['modelnet_acc']:.4f}",
                    "-" if row["scanobjectnn_acc"] is None else f"{row['scanobjectnn_acc']:.4f}",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row_lookup(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["backbone_type"], row["method_name"]): row for row in rows}


def plot_grouped_dataset_bars(rows: list[dict]):
    lookup = row_lookup(rows)
    backbones = ["pointnet", "dgcnn"]
    joint_methods = [
        "joint_decoupled",
        "joint_decoupled_pdnorm",
        "joint_language_guided",
        "joint_language_guided_pdnorm",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    bar_width = 0.36
    x = np.arange(len(joint_methods))

    for ax, backbone in zip(axes, backbones):
        modelnet_values = [lookup[(backbone, method)]["modelnet_acc"] for method in joint_methods]
        scan_values = [lookup[(backbone, method)]["scanobjectnn_acc"] for method in joint_methods]

        ax.bar(x - bar_width / 2, modelnet_values, width=bar_width, label="ModelNet40", color="#1f77b4")
        ax.bar(x + bar_width / 2, scan_values, width=bar_width, label="ScanObjectNN", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                "Decoupled + Naive",
                "Decoupled + PDNorm",
                "Language-guided + Naive",
                "Language-guided + PDNorm",
            ],
            rotation=18,
            ha="right",
        )
        ax.set_ylim(0.55, 0.9)
        ax.set_title(backbone.upper())
        ax.set_ylabel("Accuracy")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend()

    fig.suptitle("Joint Training Accuracy by Dataset")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "joint_dataset_accuracy_bars.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mean_accuracy(rows: list[dict]):
    lookup = row_lookup(rows)
    methods = [
        "joint_decoupled",
        "joint_decoupled_pdnorm",
        "joint_language_guided",
        "joint_language_guided_pdnorm",
    ]
    labels = [
        "Decoupled + Naive",
        "Decoupled + PDNorm",
        "Language-guided + Naive",
        "Language-guided + PDNorm",
    ]

    pointnet_values = [lookup[("pointnet", method)]["best_acc"] for method in methods]
    dgcnn_values = [lookup[("dgcnn", method)]["best_acc"] for method in methods]

    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, pointnet_values, width=width, label="PointNet", color="#2ca02c")
    ax.bar(x + width / 2, dgcnn_values, width=width, label="DGCNN", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Mean Accuracy")
    ax.set_ylim(0.68, 0.84)
    ax.set_title("Mean Accuracy Across Joint Methods")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mean_accuracy_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gap_to_single(rows: list[dict]):
    lookup = row_lookup(rows)
    methods = [
        "joint_decoupled",
        "joint_decoupled_pdnorm",
        "joint_language_guided",
        "joint_language_guided_pdnorm",
    ]
    labels = [
        "Decoupled + Naive",
        "Decoupled + PDNorm",
        "Language-guided + Naive",
        "Language-guided + PDNorm",
    ]
    backbones = ["pointnet", "dgcnn"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    width = 0.36
    x = np.arange(len(methods))

    for ax, backbone in zip(axes, backbones):
        single_modelnet = lookup[(backbone, "single_modelnet40")]["modelnet_acc"]
        single_scan = lookup[(backbone, "single_scanobjectnn")]["scanobjectnn_acc"]

        modelnet_gaps = [lookup[(backbone, method)]["modelnet_acc"] - single_modelnet for method in methods]
        scan_gaps = [lookup[(backbone, method)]["scanobjectnn_acc"] - single_scan for method in methods]

        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.bar(x - width / 2, modelnet_gaps, width=width, label="ModelNet40 gap", color="#ff7f0e")
        ax.bar(x + width / 2, scan_gaps, width=width, label="ScanObjectNN gap", color="#17becf")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_title(backbone.upper())
        ax.set_ylabel("Joint Acc - Single-dataset Acc")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend()

    fig.suptitle("Gap to Single-dataset Training")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "gap_to_single.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_history(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_joint_curves(rows: list[dict], backbone: str):
    method_order = [
        "joint_decoupled",
        "joint_decoupled_pdnorm",
        "joint_language_guided",
        "joint_language_guided_pdnorm",
    ]
    colors = {
        "joint_decoupled": "#1f77b4",
        "joint_decoupled_pdnorm": "#2ca02c",
        "joint_language_guided": "#d62728",
        "joint_language_guided_pdnorm": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for method_name in method_order:
        row = next(row for row in rows if row["backbone_type"] == backbone and row["method_name"] == method_name)
        history = load_history(row["history_path"])
        epochs = np.arange(1, len(history["val_acc"]) + 1)
        ax.plot(
            epochs,
            history["val_acc"],
            label=display_method_name(method_name),
            color=colors[method_name],
            linewidth=2.0,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Validation Accuracy")
    ax.set_title(f"Joint Training Curves ({backbone.upper()})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{backbone}_joint_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load_result_rows()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_summary_tables(rows)
    plot_grouped_dataset_bars(rows)
    plot_mean_accuracy(rows)
    plot_gap_to_single(rows)
    for backbone in ["pointnet", "dgcnn"]:
        plot_joint_curves(rows, backbone)
    print(f"Saved summary tables and figures to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
