#!/usr/bin/env bash
# ======================================================================
#  Ablation cell: freeze skill bank at iteration k, continue training
#  the decision agent for N more steps.
#
#  Implements the "freeze-at-iter-k" experiment requested by reviewer
#  vJ13 [Q2] and reviewer 65jm [Q3] (skill collapse / cyclic refinement
#  check).  Loads ALL 5 LoRA adapters + the per-game skill bank state
#  from a specific A0 checkpoint (full co-evolution control), then
#  continues with --freeze-skillbank for NCONT additional steps.
#
#  Source of checkpoints: the R2 sweep's A0 cell
#    ablation_study/output/candy_crush_sweep_<TS>/A0/run/checkpoints/step_<K>/
#    {adapters/{decision,skillbank}, banks/candy_crush/skill_bank.jsonl}
#
#  Usage:
#    bash run_freeze_at_k_candy_crush.sh <K> [NCONT]
#
#  Environment variables:
#    A0_RUN          (default: latest R2 sweep's A0/run)
#    OUTPUT_BASE     (default: ablation_study/output/freeze_at_k_<ts>)
#    NCONT           (default: 5)
#    VLLM_GPUS       (default: "0 1 2 3")
#    GRPO_GPUS       (default: "4 5 6 7")
# ======================================================================
set -euo pipefail

K="${1:-}"
NCONT="${2:-${NCONT:-5}}"
if [ -z "${K}" ]; then
    echo "Usage: bash run_freeze_at_k_candy_crush.sh <K> [NCONT]"
    echo "  K     = freeze point (checkpoint step in A0, e.g. 0/2/5/8/9)"
    echo "  NCONT = continuation steps (default 5)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Locate the A0 run (control = full co-evolution, 10 steps) ─────────
if [ -z "${A0_RUN:-}" ]; then
    A0_RUN="$(ls -d "${PROJECT_ROOT}"/ablation_study/output/candy_crush_sweep_*/A0/run 2>/dev/null | sort | tail -1)"
fi
if [ ! -d "${A0_RUN}" ]; then
    echo "[freeze-at-k] ERROR: A0 run not found. Set A0_RUN explicitly."
    exit 1
fi

K_PAD=$(printf "%04d" "${K}")
CKPT="${A0_RUN}/checkpoints/step_${K_PAD}"
DECISION_SRC="${CKPT}/adapters/decision"
SKILLBANK_SRC="${CKPT}/adapters/skillbank"
SEED_BANK_SRC="${CKPT}/banks"

if [ ! -f "${SEED_BANK_SRC}/candy_crush/skill_bank.jsonl" ]; then
    echo "[freeze-at-k] ERROR: missing seed bank at ${SEED_BANK_SRC}/candy_crush/skill_bank.jsonl"
    echo "  Available checkpoints:"
    ls "${A0_RUN}/checkpoints/" || true
    exit 1
fi
if [ ! -d "${DECISION_SRC}" ] || [ ! -d "${SKILLBANK_SRC}" ]; then
    echo "[freeze-at-k] ERROR: missing adapter dirs under ${CKPT}/adapters/"
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

# ── Output dir (one per freeze cell) ──────────────────────────────────
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/ablation_study/output/freeze_at_k_$(date +%Y%m%d_%H%M%S)}"
CELL_DIR="${OUTPUT_BASE}/F${K}"
RUN_DIR="${CELL_DIR}/run"
mkdir -p "${RUN_DIR}"

# ── Environment ───────────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Hyperparameters (paper Appendix C Table 3, candy_crush row) ──────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
EPISODES="${EPISODES:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-2}"

# ── Save cell metadata ────────────────────────────────────────────────
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
  "game": "candy_crush",
  "model": "${MODEL}",
  "episodes_per_step": ${EPISODES},
  "timestamp": "$(date -u +%FT%TZ)"
}
EOF

echo "══════════════════════════════════════════════════════════════"
echo "  Freeze-at-iter-k Ablation Cell: F${K}"
echo "══════════════════════════════════════════════════════════════"
echo "  A0 source run:      ${A0_RUN}"
echo "  Freeze at step:     ${K}   (reward=${CKPT_REWARD})"
echo "  Continue for:       ${NCONT} steps"
echo "  Decision adapters:  ${DECISION_SRC}"
echo "  Skill bank adapters: ${SKILLBANK_SRC} (FROZEN)"
echo "  Seed bank:          ${SEED_BANK_SRC}/candy_crush/skill_bank.jsonl"
echo "  Run dir:            ${RUN_DIR}"
echo "  vLLM GPUs:          ${VLLM_GPUS}"
echo "  GRPO GPUs:          ${GRPO_GPUS}"
echo "══════════════════════════════════════════════════════════════"

python scripts/run_coevolution.py \
    --games candy_crush \
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
    --wandb-run-name "freeze_at_k_F${K}_cont${NCONT}" \
    --no-wandb \
    --debug-io \
    2>&1 | tee "${CELL_DIR}/train.log"
