#!/usr/bin/env bash
# ======================================================================
#  Train Diplomacy with LoRA adapters warm-started from SFT cold-start.
#
#  Loads pre-trained LoRA adapters (skill_selection, action_taking,
#  segment, contract, curator) from the SFT cold-start output and
#  begins co-evolution on Diplomacy.  The Diplomacy skill bank starts
#  empty and bootstraps from scratch.
#
#  Diplomacy game profile:
#    - 7-player strategic board game (classic map)
#    - Powers: Austria, England, France, Germany, Italy, Russia, Turkey
#    - Phase cycle: Spring Move → Spring Retreat → Fall Move →
#                   Fall Retreat → Fall Adjustment → next year
#    - 20 max phases/episode
#    - Reward: supply_centers/18 + potential-based shaping (+0.5/gained center)
#    - Multi-role: agent plays all 7 powers, per-power skill banks
#    - Negotiation support: message exchange before order submission
#    - LLM-based partner policy for non-controlled powers
#    - Episodes are long-horizon with continuous strategic reward
#
#  Usage:
#    conda activate game-ai-agent
#    bash scripts/run_diplomacy.sh
#
#    # With overrides:
#    TOTAL_STEPS=30 EPISODES=21 bash scripts/run_diplomacy.sh
#
#    # Resume from a checkpoint:
#    RESUME_FROM_STEP=8 TOTAL_STEPS=30 \
#      RUN_DIR=runs/Qwen3-8B_diplomacy_20260322_200000 \
#      bash scripts/run_diplomacy.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── Diplomacy-specific segmentation tuning ────────────────────────────
# Diplomacy episodes have 20 phases with multi-power orders and
# negotiation messages — extremely text-heavy.  Higher token budget
# for diplomatic context; longer timeout for complex multi-phase
# segment ranking calls.
export SKILLBANK_LLM_TEACHER_MAX_TOKENS="${SKILLBANK_LLM_TEACHER_MAX_TOKENS:-768}"
export SKILLBANK_SEGMENT_TIMEOUT_S="${SKILLBANK_SEGMENT_TIMEOUT_S:-180}"

# ── CUDA memory management ───────────────────────────────────────────
# Diplomacy prompts are 2-4× longer than single-player games, causing
# FSDP GRPO OOM at the default batch size.  Reduce FSDP micro-batch
# from 32→8 and ref micro-batch from 8→4.  Enable expandable segments
# to reduce fragmentation.  Raise NCCL timeout to survive long
# all-reduce stalls after OOM recovery.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GRPO_FSDP_BATCH_SIZE="${GRPO_FSDP_BATCH_SIZE:-8}"
export GRPO_REF_MICRO_BATCH="${GRPO_REF_MICRO_BATCH:-4}"
export GRPO_NCCL_TIMEOUT_S="${GRPO_NCCL_TIMEOUT_S:-900}"

# ── Skillbank GRPO accumulation threshold ────────────────────────────
# Default threshold of 32 is too high for Diplomacy: 28 episodes only
# produce ~2-7 new segment/curator samples per step, so the buffer
# never fires.  Lower to 12 so skillbank adapters actually train.
export SKILLBANK_TRAIN_THRESHOLD="${SKILLBANK_TRAIN_THRESHOLD:-12}"

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"

TOTAL_STEPS="${TOTAL_STEPS:-25}"
EPISODES="${EPISODES:-28}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-}"

# ── Diplomacy GRPO stability overrides ────────────────────────────────
# Diplomacy is a 7-player long-horizon strategic game with continuous
# reward signal (supply center counts).  7 powers × negotiation ×
# multi-phase structure creates high variance across rollouts.
# Very conservative LR prevents destabilization from the complex
# multi-agent dynamics; stronger KL keeps the policy anchored while
# still allowing diplomatic strategy exploration.
GRPO_LR="${GRPO_LR:-1e-5}"
GRPO_KL_COEFF="${GRPO_KL_COEFF:-0.08}"
GRPO_CLIP_RATIO="${GRPO_CLIP_RATIO:-0.12}"
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

# ── Run directory for Diplomacy ──────────────────────────────────────
if [ -n "${RESUME_FROM_STEP}" ] && [ -z "${RUN_DIR:-}" ]; then
    echo "ERROR: RESUME_FROM_STEP requires RUN_DIR to be set"
    exit 1
fi
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-/workspace/game_agent/Game-AI-Agent/runs/Qwen3-8B_diplomacy_${TIMESTAMP}}"
mkdir -p "${RUN_DIR}/checkpoints"

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[diplomacy] Shutting down..."
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    echo "[diplomacy] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
if [ -n "${RESUME_FROM_STEP}" ]; then
    echo "  Diplomacy: RESUME from step ${RESUME_FROM_STEP}"
else
    echo "  Diplomacy: Warm-start from SFT cold-start LoRA adapters"
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
echo "  Diplomacy game profile:"
echo "    - 7-player strategic board game (classic map)"
echo "    - Powers: Austria, England, France, Germany, Italy, Russia, Turkey"
echo "    - Phases: Spring/Fall Movement + Retreat + Fall Adjustment"
echo "    - 20 max phases/episode"
echo "    - Reward: supply_centers/18 + potential shaping (+0.5/center gained)"
echo "    - Negotiation: message exchange before order submission"
echo "    - LLM-based partner policy for non-controlled powers"
echo "    - Unified role rollouts: ON (deterministic power cycling)"
echo "    - ${EPISODES} eps/step → 4 eps per power (7 powers)"
echo "    - Skill banks split by power: diplomacy/AUSTRIA .. diplomacy/TURKEY"
echo "    - Skill bank starts empty; LoRA adapters warm-started from SFT"
echo ""
echo "  Diplomacy segmentation tuning:"
echo "    - LLM teacher max_tokens: ${SKILLBANK_LLM_TEACHER_MAX_TOKENS}"
echo "    - Segment timeout: ${SKILLBANK_SEGMENT_TIMEOUT_S}s"
echo ""
echo "  GRPO stability overrides (diplomacy-tuned):"
echo "    - LR:           ${GRPO_LR} (default 5e-5)"
echo "    - KL coeff:     ${GRPO_KL_COEFF} (default 0.05)"
echo "    - Clip ratio:   ${GRPO_CLIP_RATIO} (default 0.2)"
echo "    - Max epochs:   ${GRPO_MAX_EPOCHS} (default 4)"
echo "    - Adv clip:     ${GRPO_ADV_CLIP} (default: none)"
echo "    - FSDP batch:   ${GRPO_FSDP_BATCH_SIZE} (default 32)"
echo ""
echo "  Memory & skillbank tuning:"
echo "    - vLLM GPU util:      ${GPU_UTIL} (default 0.90)"
echo "    - CUDA alloc conf:    ${PYTORCH_CUDA_ALLOC_CONF}"
echo "    - Skillbank threshold: ${SKILLBANK_TRAIN_THRESHOLD} (default 32)"
echo "══════════════════════════════════════════════════════════════"
echo ""

if [ -n "${RESUME_FROM_STEP}" ]; then
    echo "[diplomacy] Resuming from checkpoint step ${RESUME_FROM_STEP}"
    echo "  Checkpoint: ${RUN_DIR}/checkpoints/step_$(printf '%04d' "${RESUME_FROM_STEP}")"
    echo ""
else
    echo "[diplomacy] SFT cold-start adapters:"
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
    --games diplomacy
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
    --unified-roles
)

if [ -n "${RESUME_FROM_STEP}" ]; then
    TRAIN_ARGS+=(--resume --resume-from-step "${RESUME_FROM_STEP}")
else
    TRAIN_ARGS+=(
        --load-decision-adapters "${DECISION_ADAPTERS}"
        --load-skillbank-adapters "${SKILLBANK_ADAPTERS}"
    )
fi

echo "[diplomacy] Command:"
echo "  python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

# ── Run training ─────────────────────────────────────────────────────
python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Diplomacy training COMPLETE"

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
    echo "  Diplomacy training FAILED (exit code ${EXIT_CODE})"
    echo "  Check logs: ${RUN_DIR}/coevolution.log"
fi
echo ""
echo "  Run dir: ${RUN_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
