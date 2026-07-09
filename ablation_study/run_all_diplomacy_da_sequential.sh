#!/usr/bin/env bash
# Run all Diplomacy DA vs GPT-5.4 ablation scripts sequentially on one GPU.
#
# Run: Qwen3-8B_20260327_062035
#   Best checkpoint: step_0017 (mean_reward = 4.935, 62 skills)
#   First checkpoint: step_0000 (cold-start, 26 skills)
#
# Usage:
#   EVAL_GPUS=6 bash ablation_study/run_all_diplomacy_da_sequential.sh
#   LOG_DIR=/path/to/logs EVAL_GPUS=6 bash ablation_study/run_all_diplomacy_da_sequential.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GAME_AI_AGENT_ENV_BIN="${GAME_AI_AGENT_ENV_BIN:-/workspace/miniconda3/envs/game-ai-agent/bin}"
if [ -x "${GAME_AI_AGENT_ENV_BIN}/python" ]; then
  export PATH="${GAME_AI_AGENT_ENV_BIN}:${PATH}"
fi

EVAL_GPUS="${EVAL_GPUS:-6}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/ablation_study/output/diplomacy_da_sequential_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  run_base_model_diplomacy_da.sh
  run_sft_no_bank_diplomacy_da.sh
  run_no_bank_diplomacy_da.sh
  run_sft_best_bank_diplomacy_da.sh
  run_sft_first_bank_diplomacy_da.sh
  run_with_bank_diplomacy_da.sh
)

echo "══════════════════════════════════════════════════════════════"
echo "  Sequential Diplomacy DA ablations on GPU(s): ${EVAL_GPUS}"
echo "  Run: Qwen3-8B_20260327_062035"
echo "  Best step: step_0017 | First step: step_0000"
echo "  Opponent: gpt-5.4"
echo "  Logs: ${LOG_DIR}"
echo "══════════════════════════════════════════════════════════════"

FAILED=()
for s in "${SCRIPTS[@]}"; do
  log="${LOG_DIR}/${s%.sh}.log"
  echo ""
  echo "[$(date -Iseconds)] START ${s} → ${log}"
  if EVAL_GPUS="${EVAL_GPUS}" bash "${SCRIPT_DIR}/${s}" 2>&1 | tee "${log}"; then
    echo "[$(date -Iseconds)] OK   ${s}"
  else
    echo "[$(date -Iseconds)] FAIL ${s} (exit $?)"
    FAILED+=("${s}")
  fi
done

echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "  All ${#SCRIPTS[@]} runs finished successfully."
else
  echo "  Failed (${#FAILED[@]}): ${FAILED[*]}"
  exit 1
fi
echo "  Logs: ${LOG_DIR}"
echo "══════════════════════════════════════════════════════════════"
