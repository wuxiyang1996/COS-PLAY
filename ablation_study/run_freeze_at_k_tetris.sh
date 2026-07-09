#!/usr/bin/env bash
# ======================================================================
#  Tetris (macro) freeze-at-iter-k cell
#
#  Mirror of run_freeze_at_k_candy_crush.sh for tetris. Loads
#  decision + skillbank adapters and the per-game skill bank from a
#  specific A0 checkpoint of the tetris sweep, then continues with
#  --freeze-skillbank for NCONT more steps.
#
#  Usage:
#    bash run_freeze_at_k_tetris.sh <K> [NCONT]
#
#  Env:
#    A0_RUN        default: latest ablation_study/output/tetris_sweep_*/A0/run
#    OUTPUT_BASE   default: ablation_study/output/freeze_at_k_tetris_<ts>
#    NCONT         default: 5
#    VLLM_GPUS     default: "0 1 2 3"
#    GRPO_GPUS     default: "4 5 6 7"
# ======================================================================
set -euo pipefail

K="${1:-}"
NCONT="${2:-${NCONT:-5}}"
if [ -z "${K}" ]; then
    echo "Usage: bash run_freeze_at_k_tetris.sh <K> [NCONT]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -z "${A0_RUN:-}" ]; then
    A0_RUN="$(ls -d "${PROJECT_ROOT}"/ablation_study/output/tetris_sweep_*/A0/run 2>/dev/null | sort | tail -1)"
fi
if [ ! -d "${A0_RUN}" ]; then
    echo "[freeze-at-k:tetris] ERROR: A0 run not found. Set A0_RUN explicitly."
    exit 1
fi

K_PAD=$(printf "%04d" "${K}")
CKPT="${A0_RUN}/checkpoints/step_${K_PAD}"
DECISION_SRC="${CKPT}/adapters/decision"
SKILLBANK_SRC="${CKPT}/adapters/skillbank"
SEED_BANK_SRC="${CKPT}/banks"

if [ ! -f "${SEED_BANK_SRC}/tetris/skill_bank.jsonl" ]; then
    echo "[freeze-at-k:tetris] ERROR: missing seed bank at ${SEED_BANK_SRC}/tetris/skill_bank.jsonl"
    echo "  Available checkpoints:"
    ls "${A0_RUN}/checkpoints/" || true
    exit 1
fi
if [ ! -d "${DECISION_SRC}" ] || [ ! -d "${SKILLBANK_SRC}" ]; then
    echo "[freeze-at-k:tetris] ERROR: missing adapter dirs under ${CKPT}/adapters/"
    exit 1
fi

CKPT_REWARD="$(python3 -c "
import json
try:
    d = json.load(open('${CKPT}/metadata.json'))
    print(f\"{d.get('mean_reward', 0):.2f} | n_skills={d.get('n_skills','?')}\")
except Exception:
    print('?')
" 2>/dev/null)"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/ablation_study/output/freeze_at_k_tetris_$(date +%Y%m%d_%H%M%S)}"
CELL_DIR="${OUTPUT_BASE}/F${K}"
RUN_DIR="${CELL_DIR}/run"
mkdir -p "${RUN_DIR}"

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
EPISODES="${EPISODES:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-2}"

cat > "${CELL_DIR}/ablation_meta.json" <<EOF
{
  "cell": "F${K}",
  "experiment": "freeze_at_iter_k",
  "freeze_at_step": ${K},
  "continuation_steps": ${NCONT},
  "source_a0_run": "${A0_RUN}",
  "source_checkpoint": "${CKPT}",
  "source_reward_at_freeze": "${CKPT_REWARD}",
  "decision_adapters_src": "${DECISION_SRC}",
  "skillbank_adapters_src": "${SKILLBANK_SRC}",
  "seed_bank_src": "${SEED_BANK_SRC}",
  "freeze_skillbank": true,
  "game": "tetris",
  "macro_actions": true,
  "model": "${MODEL}",
  "episodes_per_step": ${EPISODES},
  "timestamp": "$(date -u +%FT%TZ)"
}
EOF

echo "══════════════════════════════════════════════════════════════"
echo "  Tetris (macro) Freeze-at-iter-k Cell: F${K}"
echo "══════════════════════════════════════════════════════════════"
echo "  A0 source run:       ${A0_RUN}"
echo "  Freeze at step:      ${K}   (reward=${CKPT_REWARD})"
echo "  Continue for:        ${NCONT} steps"
echo "  Decision adapters:   ${DECISION_SRC}"
echo "  Skill bank adapters: ${SKILLBANK_SRC} (FROZEN)"
echo "  Seed bank:           ${SEED_BANK_SRC}/tetris/skill_bank.jsonl"
echo "  Run dir:             ${RUN_DIR}"
echo "  vLLM GPUs:           ${VLLM_GPUS}"
echo "  GRPO GPUs:           ${GRPO_GPUS}"
echo "══════════════════════════════════════════════════════════════"

python scripts/run_coevolution.py \
    --games tetris \
    --total-steps "${NCONT}" \
    --curriculum none \
    --episodes-per-game "${EPISODES}" \
    --checkpoint-interval "${CKPT_INTERVAL}" \
    --model "${MODEL}" \
    --run-dir "${RUN_DIR}" \
    --load-decision-adapters "${DECISION_SRC}" \
    --load-skillbank-adapters "${SKILLBANK_SRC}" \
    --seed-bank-dir "${SEED_BANK_SRC}" \
    --freeze-skillbank \
    --vllm-gpus ${VLLM_GPUS} \
    --grpo-devices ${GRPO_GPUS} \
    --vllm-base-port "${PORT}" \
    --vllm-gpu-util "${GPU_UTIL}" \
    --speculative-model "Qwen/Qwen3-0.6B" \
    --num-speculative-tokens 5 \
    --wandb-project "game-ai-freeze-at-k" \
    --wandb-run-name "freeze_at_k_tetris_F${K}_cont${NCONT}" \
    --no-wandb \
    --debug-io \
    2>&1 | tee "${CELL_DIR}/train.log"
