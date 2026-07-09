#!/usr/bin/env bash
# ======================================================================
#  Train Candy Crush with LoRA adapters warm-started from SFT cold-start.
#
#  Main COS-PLAY Candy Crush training entry point, mirroring run_2048.sh.
#  Hyperparameters follow paper Appendix C (Table 3):
#      Total Steps     = 10
#      Episodes/Step   = 8
#      Ckpt Interval   = 3
#      GRPO defaults: LR 5e-5, KL 0.05, Clip 0.20, MaxEpochs 4, no Adv Clip
#
#  Loads pre-trained LoRA adapters (skill_selection, action_taking,
#  segment, contract, curator) from the SFT cold-start output and
#  begins co-evolution on candy_crush.  The Candy Crush skill bank
#  starts empty and bootstraps from scratch.
#
#  Usage:
#    conda activate game-ai-agent
#    bash scripts/run_candy_crush.sh
#
#    # Or with overrides:
#    TOTAL_STEPS=15 EPISODES=12 bash scripts/run_candy_crush.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── Candy Crush segmentation tuning ───────────────────────────────────
# Candy Crush episodes are short (50 steps) with text-rendered 8x8 board
# and dynamic swap-action lists.  Medium token budget; tight timeout.
export SKILLBANK_LLM_TEACHER_MAX_TOKENS="${SKILLBANK_LLM_TEACHER_MAX_TOKENS:-400}"
export SKILLBANK_SEGMENT_TIMEOUT_S="${SKILLBANK_SEGMENT_TIMEOUT_S:-120}"

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Configurable parameters (paper Appendix C defaults) ──────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"

TOTAL_STEPS="${TOTAL_STEPS:-10}"
EPISODES="${EPISODES:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

# ── SFT cold-start adapter paths ─────────────────────────────────────
SFT_DIR="${SFT_DIR:-${PROJECT_ROOT}/runs/sft_coldstart}"
DECISION_ADAPTERS="${SFT_DIR}/decision"
SKILLBANK_ADAPTERS="${SFT_DIR}/skillbank"

if [ ! -d "${DECISION_ADAPTERS}" ]; then
    echo "ERROR: Decision adapters not found: ${DECISION_ADAPTERS}"
    echo "Run SFT cold-start first:  bash scripts/run_sft_coldstart.sh"
    exit 1
fi
if [ ! -d "${SKILLBANK_ADAPTERS}" ]; then
    echo "ERROR: Skill-bank adapters not found: ${SKILLBANK_ADAPTERS}"
    echo "Run SFT cold-start first:  bash scripts/run_sft_coldstart.sh"
    exit 1
fi

# ── New run directory for Candy Crush ────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-/workspace/game_agent/Game-AI-Agent/runs/Qwen3-8B_candy_crush_${TIMESTAMP}}"
mkdir -p "${RUN_DIR}/checkpoints"

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[candy_crush] Shutting down..."
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    echo "[candy_crush] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Candy Crush: Warm-start from SFT cold-start LoRA adapters"
echo "  (Hyperparameters: paper Appendix C, Table 3)"
echo "══════════════════════════════════════════════════════════════"
echo "  SFT dir:        ${SFT_DIR}"
echo "  Decision LoRA:  ${DECISION_ADAPTERS}"
echo "  SkillBank LoRA: ${SKILLBANK_ADAPTERS}"
echo "  New run dir:    ${RUN_DIR}"
echo "  Model:          ${MODEL}"
echo "  Total steps:    ${TOTAL_STEPS}"
echo "  Episodes/step:  ${EPISODES}"
echo "  Checkpoint:     every ${CKPT_INTERVAL} steps"
echo "  vLLM GPUs:      ${VLLM_GPUS}"
echo "  GRPO GPUs:      ${GRPO_GPUS}"
echo "  Spec decode:    ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo ""
echo "  Candy Crush game profile:"
echo "    - 8x8 match-3 board with 4 candy colors"
echo "    - Dynamic action space: valid coordinate-pair swaps each step"
echo "    - 50 max steps/episode"
echo "    - Reward: match score + cascade bonuses"
echo "    - Skill bank starts empty; LoRA adapters warm-started from SFT"
echo ""
echo "  GRPO hyperparameters (defaults, per paper Appendix C):"
echo "    - LR:           5e-5"
echo "    - KL coeff:     0.05"
echo "    - Clip ratio:   0.20"
echo "    - Max epochs:   4"
echo "    - Adv clip:     none"
echo "══════════════════════════════════════════════════════════════"
echo ""

# Show SFT adapter info
echo "[candy_crush] SFT cold-start adapters:"
for adapter_dir in "${DECISION_ADAPTERS}"/* "${SKILLBANK_ADAPTERS}"/*; do
    if [ -d "${adapter_dir}" ]; then
        name="$(basename "${adapter_dir}")"
        if [ -f "${adapter_dir}/adapter_config.json" ]; then
            echo "  ✓ ${name}"
        else
            echo "  ✗ ${name} (missing adapter_config.json)"
        fi
    fi
done
echo ""

# ── Build training command ───────────────────────────────────────────
TRAIN_ARGS=(
    --games candy_crush
    --total-steps "${TOTAL_STEPS}"
    --curriculum none
    --episodes-per-game "${EPISODES}"
    --checkpoint-interval "${CKPT_INTERVAL}"
    --model "${MODEL}"
    --wandb-project "${WANDB_PROJECT}"
    --run-dir "${RUN_DIR}"
    --load-decision-adapters "${DECISION_ADAPTERS}"
    --load-skillbank-adapters "${SKILLBANK_ADAPTERS}"
    --debug-io
    # shellcheck disable=SC2086
    --vllm-gpus ${VLLM_GPUS}
    --grpo-devices ${GRPO_GPUS}
    --vllm-base-port "${PORT}"
    --vllm-gpu-util "${GPU_UTIL}"
    --speculative-model "${SPEC_MODEL}"
    --num-speculative-tokens "${SPEC_TOKENS}"
)

echo "[candy_crush] Command:"
echo "  python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

# ── Run training ─────────────────────────────────────────────────────
python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Candy Crush training COMPLETE"

    echo ""
    echo "  Step log:"
    if [ -f "${RUN_DIR}/step_log.jsonl" ]; then
        python -c "
import json
with open('${RUN_DIR}/step_log.jsonl') as f:
    rows = [json.loads(l) for l in f if l.strip()]
for r in rows[-5:]:
    step = r['step']
    mr = r['mean_reward']
    ns = r.get('n_skills', '?')
    wt = r['wall_time_s'] / 60
    print(f'  Step {step:2d}: mean_reward={mr:6.1f}  skills={ns}  time={wt:.1f}m')
"
    fi
else
    echo "  Candy Crush training FAILED (exit code ${EXIT_CODE})"
    echo "  Check logs: ${RUN_DIR}/coevolution.log"
fi
echo ""
echo "  Run dir: ${RUN_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
