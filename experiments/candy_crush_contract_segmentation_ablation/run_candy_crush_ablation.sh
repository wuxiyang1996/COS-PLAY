#!/usr/bin/env bash
# Launch or print the Candy Crush contract / segmentation ablation commands.
#
# This wrapper intentionally centralizes the rebuttal experiment contract.
# Variants marked `requires_wiring` in experiment_matrix.yaml export explicit
# COSPLAY_* environment variables so the small follow-up implementation patch
# has a stable interface to read from.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VARIANT="full"
TOTAL_STEPS=60
EPISODES_PER_GAME=8
SEED=0
RUN_ROOT="runs/candy_crush_contract_segmentation_ablation"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
    sed -n '1,80p' "$0"
    cat <<'EOF'

Usage:
  bash experiments/candy_crush_contract_segmentation_ablation/run_candy_crush_ablation.sh \
      --variant full

Variants:
  full
  no_effect_contract
  raw_delta_contract
  heuristic_only_segmentation

Common flags:
  --total-steps N
  --episodes-per-game N
  --seed N
  --run-root DIR
  --dry-run
  --extra "additional scripts/run_coevolution.py args"

EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
        --episodes-per-game) EPISODES_PER_GAME="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --extra)
            # shellcheck disable=SC2206
            EXTRA_ARGS=($2)
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ablation] unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

case "$VARIANT" in
    full)
        export COSPLAY_CC_EFFECT_CONTRACT_MODE="consensus_verified"
        export COSPLAY_CC_CONTRACT_MATCHING="1"
        export COSPLAY_CC_CONTRACT_COMPLETION_REWARD="1"
        export COSPLAY_CC_SEGMENTATION_MODE="learned_decode"
        ;;
    no_effect_contract)
        export COSPLAY_CC_EFFECT_CONTRACT_MODE="none"
        export COSPLAY_CC_CONTRACT_MATCHING="0"
        export COSPLAY_CC_CONTRACT_COMPLETION_REWARD="0"
        export COSPLAY_CC_SEGMENTATION_MODE="learned_decode"
        ;;
    raw_delta_contract)
        export COSPLAY_CC_EFFECT_CONTRACT_MODE="raw_delta"
        export COSPLAY_CC_CONTRACT_MATCHING="raw_delta_only"
        export COSPLAY_CC_CONTRACT_COMPLETION_REWARD="0"
        export COSPLAY_CC_SEGMENTATION_MODE="learned_decode"
        ;;
    heuristic_only_segmentation)
        export COSPLAY_CC_EFFECT_CONTRACT_MODE="consensus_verified"
        export COSPLAY_CC_CONTRACT_MATCHING="1"
        export COSPLAY_CC_CONTRACT_COMPLETION_REWARD="1"
        export COSPLAY_CC_SEGMENTATION_MODE="heuristic_only"
        ;;
    *)
        echo "[ablation] unknown variant: $VARIANT" >&2
        usage
        exit 2
        ;;
esac

RUN_DIR="${RUN_ROOT}/${VARIANT}/seed_${SEED}"
mkdir -p "$RUN_DIR"

CMD=(
    python scripts/run_coevolution.py
    --games candy_crush
    --curriculum none
    --total-steps "$TOTAL_STEPS"
    --episodes-per-game "$EPISODES_PER_GAME"
    --episodes-per-game-overrides '{}'
    --run-dir "$RUN_DIR"
    --harness-enabled
    --harness-mode full
    --wandb-run-name "cc_${VARIANT}_seed_${SEED}"
    "${EXTRA_ARGS[@]}"
)

echo "[ablation] variant=$VARIANT"
echo "[ablation] run_dir=$RUN_DIR"
echo "[ablation] COSPLAY_CC_EFFECT_CONTRACT_MODE=$COSPLAY_CC_EFFECT_CONTRACT_MODE"
echo "[ablation] COSPLAY_CC_CONTRACT_MATCHING=$COSPLAY_CC_CONTRACT_MATCHING"
echo "[ablation] COSPLAY_CC_CONTRACT_COMPLETION_REWARD=$COSPLAY_CC_CONTRACT_COMPLETION_REWARD"
echo "[ablation] COSPLAY_CC_SEGMENTATION_MODE=$COSPLAY_CC_SEGMENTATION_MODE"
printf '[ablation] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

exec "${CMD[@]}"

