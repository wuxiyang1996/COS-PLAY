#!/usr/bin/env bash
# ======================================================================
#  Train Sokoban with LoRA adapters warm-started from SFT cold-start.
#
#  Loads pre-trained LoRA adapters (skill_selection, action_taking,
#  segment, contract, curator) from the SFT cold-start output and
#  begins co-evolution on Sokoban.  The Sokoban skill bank starts
#  empty and bootstraps from scratch.
#
#  Sokoban game profile:
#    - 9 actions (push up/down/left/right, move up/down/left/right, noop)
#    - 200 max steps/episode
#    - Reward: +1 box on target, -1 box off target, +10 puzzle solved,
#              -0.02/step penalty, distance shaping ±0.2, deadlock -0.5
#    - Episodes are shorter than Tetris → faster iteration, more
#      sensitive to strategic mistakes (deadlocks, oscillation)
#
#  Usage:
#    conda activate game-ai-agent
#    bash scripts/run_sokoban.sh
#
#    # With overrides:
#    TOTAL_STEPS=30 EPISODES=12 bash scripts/run_sokoban.sh
#
#    # Resume from a checkpoint:
#    RESUME_FROM_STEP=10 TOTAL_STEPS=40 \
#      RUN_DIR=runs/Qwen3-8B_sokoban_20260322_200000 \
#      bash scripts/run_sokoban.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── Sokoban-specific segmentation tuning ──────────────────────────────
# Sokoban episodes are shorter (200 steps max) with denser reward
# signal.  Moderate token budget; 120s timeout is enough.
export SKILLBANK_LLM_TEACHER_MAX_TOKENS="${SKILLBANK_LLM_TEACHER_MAX_TOKENS:-350}"
export SKILLBANK_SEGMENT_TIMEOUT_S="${SKILLBANK_SEGMENT_TIMEOUT_S:-120}"

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"

TOTAL_STEPS="${TOTAL_STEPS:-20}"
EPISODES="${EPISODES:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-}"

# ── Sokoban GRPO stability overrides ─────────────────────────────────
# Sokoban has smaller rewards (mostly -0.02 to +1.0 per step) and
# shorter episodes than Tetris.  The reward variance is lower so we
# can use a slightly higher LR than tetris-tuned but still below the
# default 5e-5.  Tighter clip + advantage clipping to prevent
# oscillation-prone strategies from being reinforced.
GRPO_LR="${GRPO_LR:-3e-5}"
GRPO_KL_COEFF="${GRPO_KL_COEFF:-0.06}"
GRPO_CLIP_RATIO="${GRPO_CLIP_RATIO:-0.15}"
GRPO_MAX_EPOCHS="${GRPO_MAX_EPOCHS:-2}"
GRPO_ADV_CLIP="${GRPO_ADV_CLIP:-3.0}"

# ── SFT cold-start adapter paths ─────────────────────────────────────
SFT_DIR="${SFT_DIR:-${PROJECT_ROOT}/runs/sft_coldstart}"
DECISION_ADAPTERS="${SFT_DIR}/decision"
SKILLBANK_ADAPTERS="${SFT_DIR}/skillbank"

if [ -z "${RESUME_FROM_STEP}" ]; then
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
fi

# ── Run directory for Sokoban ─────────────────────────────────────────
if [ -n "${RESUME_FROM_STEP}" ] && [ -z "${RUN_DIR:-}" ]; then
    echo "ERROR: RESUME_FROM_STEP requires RUN_DIR to be set"
    exit 1
fi
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-/workspace/game_agent/Game-AI-Agent/runs/Qwen3-8B_sokoban_${TIMESTAMP}}"
mkdir -p "${RUN_DIR}/checkpoints"

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[sokoban] Shutting down..."
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    echo "[sokoban] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
if [ -n "${RESUME_FROM_STEP}" ]; then
    echo "  Sokoban: RESUME from step ${RESUME_FROM_STEP}"
else
    echo "  Sokoban: Warm-start from SFT cold-start LoRA adapters"
fi
echo "══════════════════════════════════════════════════════════════"
if [ -n "${RESUME_FROM_STEP}" ]; then
    echo "  Resume step:    ${RESUME_FROM_STEP}"
else
    echo "  SFT dir:        ${SFT_DIR}"
    echo "  Decision LoRA:  ${DECISION_ADAPTERS}"
    echo "  SkillBank LoRA: ${SKILLBANK_ADAPTERS}"
fi
echo "  Run dir:        ${RUN_DIR}"
echo "  Model:          ${MODEL}"
echo "  Total steps:    ${TOTAL_STEPS}"
echo "  Episodes/step:  ${EPISODES}"
echo "  Checkpoint:     every ${CKPT_INTERVAL} steps"
echo "  vLLM GPUs:      ${VLLM_GPUS}"
echo "  GRPO GPUs:      ${GRPO_GPUS}"
echo "  Spec decode:    ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo ""
echo "  Sokoban game profile:"
echo "    - 9 actions (push/move up/down/left/right + noop)"
echo "    - 200 max steps/episode"
echo "    - Reward: +1 box→target, -1 box off target, +10 solved"
echo "    - Shaping: -0.02/step, ±0.2 distance, -0.5 deadlock"
echo "    - Skill bank starts empty; LoRA adapters warm-started from SFT"
echo ""
echo "  Sokoban segmentation tuning:"
echo "    - LLM teacher max_tokens: ${SKILLBANK_LLM_TEACHER_MAX_TOKENS}"
echo "    - Segment timeout: ${SKILLBANK_SEGMENT_TIMEOUT_S}s"
echo ""
echo "  GRPO stability overrides (sokoban-tuned):"
echo "    - LR:           ${GRPO_LR} (default 5e-5)"
echo "    - KL coeff:     ${GRPO_KL_COEFF} (default 0.05)"
echo "    - Clip ratio:   ${GRPO_CLIP_RATIO} (default 0.2)"
echo "    - Max epochs:   ${GRPO_MAX_EPOCHS} (default 4)"
echo "    - Adv clip:     ${GRPO_ADV_CLIP} (default: none)"
echo "══════════════════════════════════════════════════════════════"
echo ""

if [ -n "${RESUME_FROM_STEP}" ]; then
    echo "[sokoban] Resuming from checkpoint step ${RESUME_FROM_STEP}"
    echo "  Checkpoint: ${RUN_DIR}/checkpoints/step_$(printf '%04d' "${RESUME_FROM_STEP}")"
    echo ""
else
    echo "[sokoban] SFT cold-start adapters:"
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
fi

# ── Build training command ───────────────────────────────────────────
TRAIN_ARGS=(
    --games sokoban
    --total-steps "${TOTAL_STEPS}"
    --curriculum none
    --episodes-per-game "${EPISODES}"
    --checkpoint-interval "${CKPT_INTERVAL}"
    --model "${MODEL}"
    --wandb-project "${WANDB_PROJECT}"
    --run-dir "${RUN_DIR}"
    --debug-io
    # shellcheck disable=SC2086
    --vllm-gpus ${VLLM_GPUS}
    --grpo-devices ${GRPO_GPUS}
    --vllm-base-port "${PORT}"
    --vllm-gpu-util "${GPU_UTIL}"
    --speculative-model "${SPEC_MODEL}"
    --num-speculative-tokens "${SPEC_TOKENS}"
    --grpo-lr "${GRPO_LR}"
    --grpo-kl-coeff "${GRPO_KL_COEFF}"
    --grpo-clip-ratio "${GRPO_CLIP_RATIO}"
    --grpo-max-epochs "${GRPO_MAX_EPOCHS}"
    --grpo-adv-clip "${GRPO_ADV_CLIP}"
)

if [ -n "${RESUME_FROM_STEP}" ]; then
    TRAIN_ARGS+=(--resume --resume-from-step "${RESUME_FROM_STEP}")
else
    TRAIN_ARGS+=(
        --load-decision-adapters "${DECISION_ADAPTERS}"
        --load-skillbank-adapters "${SKILLBANK_ADAPTERS}"
    )
fi

echo "[sokoban] Command:"
echo "  python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

# ── Run training ─────────────────────────────────────────────────────
python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Sokoban training COMPLETE"

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
    echo "  Sokoban training FAILED (exit code ${EXIT_CODE})"
    echo "  Check logs: ${RUN_DIR}/coevolution.log"
fi
echo ""
echo "  Run dir: ${RUN_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
