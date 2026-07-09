#!/usr/bin/env bash
# ======================================================================
#  Candy Crush contract / segmentation ablations.
#
#  Paper-matched setting from EMNLP_2026_Co_evolving_Agent__Copy_.pdf
#  Table 4:
#    - total_steps = 10
#    - episodes_per_step = 8
#    - checkpoint_interval = 3
#    - max_steps_per_episode = 50 (trainer game config)
#    - GRPO defaults: lr=5e-5, kl=0.05, clip=0.20, max_epochs=4
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
#    # Background jobs with one log per variant/seed:
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

mkdir -p "${LOG_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  Candy Crush Contract / Segmentation Ablations"
echo "══════════════════════════════════════════════════════════════"
echo "  Variants:       ${VARIANTS}"
echo "  Seeds:          ${SEEDS}"
echo "  Run root:       ${RUN_ROOT}"
echo "  Log dir:        ${LOG_DIR}"
echo "  Paper setting:  total_steps=10 episodes=8 ckpt=3 max_steps=50"
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
        )

        if [[ "${DRY_RUN}" == "1" ]]; then
            cmd+=(--dry-run)
        fi

        echo ""
        echo "[candy-crush-ablation] variant=${variant} seed=${seed}"
        echo "[candy-crush-ablation] log=${log_file}"

        if [[ "${NOHUP}" == "1" ]]; then
            nohup "${cmd[@]}" > "${log_file}" 2>&1 &
            echo "[candy-crush-ablation] launched pid=$!"
        else
            "${cmd[@]}" 2>&1 | tee "${log_file}"
        fi
    done
done

if [[ "${NOHUP}" == "1" ]]; then
    echo ""
    echo "[candy-crush-ablation] all jobs launched"
    echo "[candy-crush-ablation] monitor: tail -f ${LOG_DIR}/full_seed0.log"
fi
