#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FORCE="${FORCE:-0}"
NUM_PERMUTATIONS="${NUM_PERMUTATIONS:-500}"

run_analysis() {
  local output_dir="$1"
  shift

  if [[ "$FORCE" != "1" && -f "$output_dir/semantic_alignment_results.csv" ]]; then
    echo "Skipping existing analysis: $output_dir"
    return
  fi

  echo "Running semantic alignment analysis: $output_dir"
  uv run python scripts/analyze_semantic_alignment.py "$@" --num_permutations "$NUM_PERMUTATIONS" --output_dir "$output_dir"
}

run_analysis \
  "results/semantic_alignment_pointnet_naive" \
  --checkpoint_a runs/benchmark_pointnet_joint_decoupled/best.pt \
  --checkpoint_b runs/benchmark_pointnet_joint_language_guided/best.pt \
  --name_a "Decoupled + Naive" \
  --name_b "Language-guided + Naive"

run_analysis \
  "results/semantic_alignment_pointnet_pdnorm" \
  --checkpoint_a runs/benchmark_pointnet_joint_decoupled_pdnorm/best.pt \
  --checkpoint_b runs/benchmark_pointnet_joint_language_guided_pdnorm/best.pt \
  --name_a "Decoupled + PDNorm" \
  --name_b "Language-guided + PDNorm"

run_analysis \
  "results/semantic_alignment_dgcnn_naive" \
  --checkpoint_a runs/benchmark_dgcnn_joint_decoupled/best.pt \
  --checkpoint_b runs/benchmark_dgcnn_joint_language_guided/best.pt \
  --name_a "Decoupled + Naive" \
  --name_b "Language-guided + Naive"

run_analysis \
  "results/semantic_alignment_dgcnn_pdnorm" \
  --checkpoint_a runs/benchmark_dgcnn_joint_decoupled_pdnorm/best.pt \
  --checkpoint_b runs/benchmark_dgcnn_joint_language_guided_pdnorm/best.pt \
  --name_a "Decoupled + PDNorm" \
  --name_b "Language-guided + PDNorm"

echo "Semantic alignment suite finished."
