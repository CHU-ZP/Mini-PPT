#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODELNET_ROOT="${MODELNET_ROOT:-$ROOT_DIR/data/modelnet40_princeton_npy}"
SCANOBJECTNN_ROOT="${SCANOBJECTNN_ROOT:-$ROOT_DIR/data/scanobjectnn_npy}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_POINTS="${NUM_POINTS:-1024}"
BACKBONE_TYPE="${BACKBONE_TYPE:-pointnet}"
RUN_PREFIX="${RUN_PREFIX:-benchmark}"
FORCE="${FORCE:-0}"

if [[ ! -d "$MODELNET_ROOT" ]]; then
  echo "ModelNet40 directory not found: $MODELNET_ROOT" >&2
  exit 1
fi

if [[ ! -d "$SCANOBJECTNN_ROOT" ]]; then
  echo "ScanObjectNN directory not found: $SCANOBJECTNN_ROOT" >&2
  exit 1
fi

if ! uv run python - <<'PY' >/dev/null 2>&1
import sentence_transformers  # noqa: F401
PY
then
  echo "Missing optional text dependency. Install it with: uv pip install -e .[text]" >&2
  exit 1
fi

RESULTS_TSV="$(mktemp)"
trap 'rm -f "$RESULTS_TSV"' EXIT

printf "exp_name\treport_name\tmode\tbackbone_type\thead_type\tbest_epoch\tbest_acc\tmodelnet_acc\tscanobjectnn_acc\tcheckpoint\n" > "$RESULTS_TSV"

EXPERIMENTS=(
  "single_modelnet40|Single-dataset training on ModelNet40|train_modelnet_only|decoupled"
  "single_scanobjectnn|Single-dataset training on ScanObjectNN|train_scanobjectnn_only|decoupled"
  "joint_decoupled|Multi-dataset joint training, Decoupled|train_joint_naive|decoupled"
  "joint_decoupled_pdnorm|Multi-dataset joint training, Decoupled + PDNorm|train_joint_pdnorm|decoupled"
  "joint_language_guided|Multi-dataset joint training, Lightweight Language-guided Categorical Alignment|train_joint_naive|language_guided"
  "joint_language_guided_pdnorm|Multi-dataset joint training, Lightweight Language-guided Categorical Alignment + PDNorm|train_joint_pdnorm|language_guided"
)

run_experiment() {
  local exp_key="$1"
  local report_name="$2"
  local mode="$3"
  local head_type="$4"

  local exp_name="${RUN_PREFIX}_${BACKBONE_TYPE}_${exp_key}"
  local run_dir="$ROOT_DIR/runs/$exp_name"
  local checkpoint_path="$run_dir/best.pt"

  echo
  echo "==> $report_name"
  echo "    mode=$mode | backbone_type=$BACKBONE_TYPE | head_type=$head_type | exp_name=$exp_name"

  if [[ "$FORCE" == "1" || ! -f "$checkpoint_path" ]]; then
    uv run minippt-train \
      --modelnet_root "$MODELNET_ROOT" \
      --scanobjectnn_root "$SCANOBJECTNN_ROOT" \
      --mode "$mode" \
      --backbone_type "$BACKBONE_TYPE" \
      --head_type "$head_type" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_points "$NUM_POINTS" \
      --exp_name "$exp_name"
  else
    echo "    skipping training because checkpoint already exists"
  fi

  uv run python - "$checkpoint_path" "$exp_name" "$report_name" "$mode" "$BACKBONE_TYPE" "$head_type" >> "$RESULTS_TSV" <<'PY'
import math
import sys
import torch

checkpoint_path, exp_name, report_name, mode, backbone_type, head_type = sys.argv[1:7]
checkpoint = torch.load(checkpoint_path, map_location="cpu")
metrics = checkpoint.get("metrics", {})
epoch = checkpoint.get("epoch", -1)

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)

print(
    "\t".join(
        [
            exp_name,
            report_name,
            mode,
            backbone_type,
            head_type,
            str(epoch),
            fmt(metrics.get("val_acc")),
            fmt(metrics.get("modelnet")),
            fmt(metrics.get("scanobjectnn")),
            checkpoint_path,
        ]
    )
)
PY
}

for spec in "${EXPERIMENTS[@]}"; do
  IFS="|" read -r exp_key report_name mode head_type <<<"$spec"
  run_experiment "$exp_key" "$report_name" "$mode" "$head_type"
done

CSV_PATH="$ROOT_DIR/runs/${RUN_PREFIX}_${BACKBONE_TYPE}_results.csv"
MD_PATH="$ROOT_DIR/runs/${RUN_PREFIX}_${BACKBONE_TYPE}_results.md"

uv run python - "$RESULTS_TSV" "$CSV_PATH" "$MD_PATH" <<'PY'
import csv
import sys
from pathlib import Path

tsv_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
md_path = Path(sys.argv[3])

with tsv_path.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

csv_path.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "exp_name",
    "report_name",
    "mode",
    "backbone_type",
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
    writer.writerows(rows)

header = [
    "Experiment",
    "Mode",
    "Backbone",
    "Head",
    "Best Epoch",
    "Best Mean Acc",
    "ModelNet40 Acc",
    "ScanObjectNN Acc",
]

lines = [
    "# Experiment Results",
    "",
    f"- Total experiments: {len(rows)}",
    "",
    "| " + " | ".join(header) + " |",
    "| " + " | ".join(["---"] * len(header)) + " |",
]

for row in rows:
    lines.append(
        "| "
        + " | ".join(
            [
                row["report_name"],
                row["mode"],
                row["backbone_type"],
                row["head_type"],
                row["best_epoch"],
                row["best_acc"],
                row["modelnet_acc"] or "-",
                row["scanobjectnn_acc"] or "-",
            ]
        )
        + " |"
    )

lines.extend(
    [
        "",
        "## Checkpoints",
        "",
    ]
)

for row in rows:
    lines.append(f"- `{row['report_name']}`: `{row['checkpoint']}`")

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo
echo "Finished all experiments."
echo "CSV summary: $CSV_PATH"
echo "Markdown summary: $MD_PATH"
