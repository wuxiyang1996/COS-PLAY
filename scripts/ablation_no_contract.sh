#!/bin/bash
# =============================================================================
# Ablation: w/o Effect Contract
# =============================================================================
# Removes predicate-level effect contracts while keeping NL skill protocols.
# Stage 3 (contract learn/verify/refine) is skipped.
# Contract compatibility scorer weight is forced to 0.
# Reward follow_predicate_bonus and follow_completion_bonus are zeroed.
#
# Purpose: answer whether the predicate-level contract machinery is
# necessary, or if the NL protocol alone suffices.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Zero out contract-related reward terms
export COSPLAY_REWARD_OVERRIDES='{"follow_predicate_bonus": 0.0, "follow_completion_bonus": 0.0}'

# Disable the contract LoRA adapter (keep SFT weights frozen)
export COSPLAY_DISABLE_ADAPTERS="contract"

python scripts/run_coevolution.py \
    --no-contract \
    --run-dir "${STORAGE_PATH:-runs}/ablation_no_contract" \
    --wandb-run-name "ablation-no-contract" \
    "$@"
