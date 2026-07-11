#!/bin/bash
# =============================================================================
# Ablation: Raw Delta Contract
# =============================================================================
# Replaces multi-instance consensus contracts with raw per-instance
# predicate deltas (B_end − B_start).  No frequency thresholds, no
# multi-instance verification, no contract refinement step.
#
# Purpose: answer whether the contract is just predicate counting,
# or if multi-instance consensus actually matters.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python scripts/run_coevolution.py \
    --raw-delta-contract \
    --run-dir "${STORAGE_PATH:-runs}/ablation_raw_delta_contract" \
    --wandb-run-name "ablation-raw-delta-contract" \
    "$@"
