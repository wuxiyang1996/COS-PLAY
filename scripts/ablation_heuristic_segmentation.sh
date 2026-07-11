#!/bin/bash
# =============================================================================
# Ablation: Heuristic-Only Segmentation
# =============================================================================
# Replaces learned segmentation (Viterbi DP + LLM preference teacher)
# with heuristic-only boundary detection.  Boundaries come from tag
# changes, embedding changepoints, and event spikes.  Skills are
# assigned by majority intention-tag match (no learned scorer).
#
# Purpose: answer whether rollout-to-skill segmentation is a core
# contribution vs. the SkillRL weak baseline with simple boundaries.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python scripts/run_coevolution.py \
    --heuristic-segmentation \
    --run-dir "${STORAGE_PATH:-runs}/ablation_heuristic_segmentation" \
    --wandb-run-name "ablation-heuristic-segmentation" \
    "$@"
