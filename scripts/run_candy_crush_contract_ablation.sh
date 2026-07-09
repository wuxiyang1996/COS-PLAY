#!/usr/bin/env bash
# ======================================================================
#  Candy Crush contract / segmentation ablations.
#
#  Paper-matched setting from EMNLP_2026_Co_evolving_Agent__Copy_.pdf
#  Table 4:
#    - total_steps = 10
#    - episodes_per_step = 8
#    - checkpoint_interval = 1 (local 4xA100 run override)
#    - max_steps_per_episode = 50 (trainer game config)
#    - GRPO defaults: lr=5e-5, kl=0.05, clip=0.20, max_epochs=4
#
#  Default local GPU layout for a 4xA100 machine:
#    - vLLM rollout servers: GPUs 0 1
#    - GRPO/FSDP trainer:   GPUs 2 3
#
#  Variants:
#    full
#    no_effect_contract
#    raw_delta_contract
#    heuristic_only_segmentation
#
#  Usage:
#    bash scripts/run_candy_crush_contract_ablation.sh
#
#    # Background the whole sweep. Jobs still run sequentially by default,
#    # which is important on a single 4xA100 node:
#    NOHUP=1 bash scripts/run_candy_crush_contract_ablation.sh
#
#    # Run a subset:
#    VARIANTS="full raw_delta_contract" SEEDS="0" \
#      bash scripts/run_candy_crush_contract_ablation.sh
#
#    # Print commands without training:
#    DRY_RUN=1 bash scripts/run_candy_crush_contract_ablation.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

LAUNCHER="experiments/candy_crush_contract_segmentation_ablation/run_candy_crush_ablation.sh"

VARIANTS="${VARIANTS:-full no_effect_contract raw_delta_contract heuristic_only_segmentation}"
SEEDS="${SEEDS:-0 1 2}"
LOG_DIR="${LOG_DIR:-logs/candy_crush_contract_segmentation_ablation}"
RUN_ROOT="${RUN_ROOT:-runs/candy_crush_contract_segmentation_ablation}"
NOHUP="${NOHUP:-0}"
DRY_RUN="${DRY_RUN:-0}"
VLLM_GPUS="${VLLM_GPUS:-0 1}"
GRPO_GPUS="${GRPO_GPUS:-2 3}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${NOHUP}" == "1" && "${_COSPLAY_ABLATION_BG:-0}" != "1" ]]; then
    mkdir -p "${LOG_DIR}"
    sweep_log="${LOG_DIR}/sweep_$(date +%Y%m%d_%H%M%S).log"
    echo "[candy-crush-ablation] launching sequential sweep in background"
    echo "[candy-crush-ablation] log=${sweep_log}"
    _COSPLAY_ABLATION_BG=1 NOHUP=0 nohup "$0" > "${sweep_log}" 2>&1 &
    echo "[candy-crush-ablation] pid=$!"
    exit 0
fi

mkdir -p "${LOG_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  Candy Crush Contract / Segmentation Ablations"
echo "══════════════════════════════════════════════════════════════"
echo "  Variants:       ${VARIANTS}"
echo "  Seeds:          ${SEEDS}"
echo "  Run root:       ${RUN_ROOT}"
echo "  Log dir:        ${LOG_DIR}"
echo "  Paper setting:  total_steps=10 episodes=8 ckpt=1 max_steps=50"
echo "  4xA100 layout:  vLLM GPUs=${VLLM_GPUS}; GRPO GPUs=${GRPO_GPUS}"
echo "  Background:     ${NOHUP}"
echo "  Dry run:        ${DRY_RUN}"
echo "══════════════════════════════════════════════════════════════"

for variant in ${VARIANTS}; do
    for seed in ${SEEDS}; do
        log_file="${LOG_DIR}/${variant}_seed${seed}.log"
        cmd=(
            bash "${LAUNCHER}"
            --variant "${variant}"
            --seed "${seed}"
            --run-root "${RUN_ROOT}"
            --extra "--vllm-gpus ${VLLM_GPUS} --grpo-devices ${GRPO_GPUS} --vllm-gpu-util ${VLLM_GPU_UTIL} ${EXTRA_ARGS}"
        )

        if [[ "${DRY_RUN}" == "1" ]]; then
            cmd+=(--dry-run)
        fi

        echo ""
        echo "[candy-crush-ablation] variant=${variant} seed=${seed}"
        echo "[candy-crush-ablation] log=${log_file}"

        "${cmd[@]}" 2>&1 | tee "${log_file}"
    done
done
