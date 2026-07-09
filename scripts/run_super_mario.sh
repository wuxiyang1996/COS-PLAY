#!/usr/bin/env bash
# ======================================================================
#  Train Super Mario with LoRA adapters warm-started from SFT cold-start.
#
#  Loads pre-trained LoRA adapters (skill_selection, action_taking,
#  segment, contract, curator) from the SFT cold-start output and
#  begins co-evolution on Super Mario.  The Super Mario skill bank
#  starts empty and bootstraps from scratch.
#
#  Super Mario game profile:
#    - Single-player NES platformer via subprocess env (orak-mario)
#    - 7 discrete actions (noop, right, right+A, A, left, left+A, down)
#    - 500 max steps/episode (~8 min of play)
#    - Reward: distance traveled + coins + time bonus
#    - NES emulator needs NumPy 1.x → runs in separate orak-mario env
#    - SubprocessEnv bridges main env (NumPy 2.x) and mario env over JSON pipes
#    - Xvfb required for headless NES rendering
#
#  Usage:
#    conda activate game-ai-agent
#    bash scripts/run_super_mario.sh
#
#    # With overrides:
#    TOTAL_STEPS=30 EPISODES=12 bash scripts/run_super_mario.sh
#
#    # Resume from a checkpoint:
#    RESUME_FROM_STEP=8 TOTAL_STEPS=30 \
#      RUN_DIR=runs/Qwen3-8B_super_mario_20260322_200000 \
#      bash scripts/run_super_mario.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── CUDA memory management ────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# Super Mario (NES via stable-retro) needs a framebuffer.
if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb &>/dev/null; then
        XVFB_DISPLAY=":99"
        if ! pgrep -f "Xvfb ${XVFB_DISPLAY}" &>/dev/null; then
            echo "[super_mario] Starting Xvfb on ${XVFB_DISPLAY}..."
            Xvfb "${XVFB_DISPLAY}" -screen 0 1024x768x24 &>/dev/null &
            sleep 1
        fi
        export DISPLAY="${XVFB_DISPLAY}"
    else
        echo "[WARN] No DISPLAY set and Xvfb not found — NES rendering may fail."
    fi
fi

# ── Subprocess env (orak-mario) ──────────────────────────────────────
# The Mario NES emulator runs in a separate conda env to avoid the
# NumPy 2.x vs 1.x conflict.  ORAK_PYTHON tells SubprocessEnv which
# interpreter to use for the game subprocess.
export ORAK_PYTHON="${ORAK_PYTHON:-/workspace/miniconda3/envs/orak-mario/bin/python}"

if [ ! -x "${ORAK_PYTHON}" ]; then
    echo "[ERROR] orak-mario Python not found at: ${ORAK_PYTHON}"
    echo "  Create it with:  conda create -n orak-mario python=3.11 && conda run -n orak-mario pip install gym-super-mario-bros nes-py"
    exit 1
fi
echo "[super_mario] Subprocess env Python: ${ORAK_PYTHON}"

# ── Super Mario segmentation tuning ──────────────────────────────────
# Super Mario episodes run ~500 steps with screen-state observations.
# Medium token budget for platformer context (position, enemies, items);
# generous timeout for segment ranking over medium-length episodes.
export SKILLBANK_LLM_TEACHER_MAX_TOKENS="${SKILLBANK_LLM_TEACHER_MAX_TOKENS:-400}"
export SKILLBANK_SEGMENT_TIMEOUT_S="${SKILLBANK_SEGMENT_TIMEOUT_S:-150}"

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

# ── Super Mario GRPO tuned overrides ─────────────────────────────────
# Tuned from the baseline 20-step run analysis:
#   - LR 2e-5→3e-5: slightly more aggressive to make progress in each step
#   - KL 0.08→0.04: previous run regressed in second half due to KL pulling
#     the policy back too hard; halving it lets the policy diverge further
#   - Clip 0.1→0.15: allows larger per-step policy updates (was too tight)
#   - Epochs 2→3: only ~70 fresh samples/step; 2 epochs was underfitting
#   - Adv clip 3.0→5.0: less aggressive advantage clipping to preserve
#     learning signal from high-reward outlier episodes (e.g. 1500+ runs)
export GRPO_FSDP_BATCH_SIZE="${GRPO_FSDP_BATCH_SIZE:-16}"
export GRPO_MAX_SEQ_LEN="${GRPO_MAX_SEQ_LEN:-2048}"
GRPO_LR="${GRPO_LR:-3e-5}"
GRPO_KL_COEFF="${GRPO_KL_COEFF:-0.04}"
GRPO_CLIP_RATIO="${GRPO_CLIP_RATIO:-0.15}"
GRPO_MAX_EPOCHS="${GRPO_MAX_EPOCHS:-3}"
GRPO_ADV_CLIP="${GRPO_ADV_CLIP:-5.0}"

# ── Training schedule overrides ──────────────────────────────────────
# Warmup 15 of 20 steps (75%): gives longer ramp so the schedule doesn't
# reach its most aggressive KL/temperature right before the run ends.
# Initial KL 0.005 (was 0.01): gentler start lets policy explore more.
# Temperature 1.0→0.6: slightly lower steady temp for more exploitation.
WARMUP_STEPS="${WARMUP_STEPS:-15}"
INITIAL_KL="${INITIAL_KL:-0.005}"
INITIAL_TEMP="${INITIAL_TEMP:-1.0}"
STEADY_TEMP="${STEADY_TEMP:-0.6}"

# ── Stuck detection overrides ────────────────────────────────────────
# Previous run had mean 14.3 steps/episode (vs 50.7 for GPT-5.4 data).
# Relax stuck detection so episodes have more time to make progress.
STUCK_WINDOW="${STUCK_WINDOW:-10}"
MIN_STEPS_BEFORE_STUCK="${MIN_STEPS_BEFORE_STUCK:-30}"

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

# ── Run directory for Super Mario ─────────────────────────────────────
if [ -n "${RESUME_FROM_STEP}" ] && [ -z "${RUN_DIR:-}" ]; then
    echo "ERROR: RESUME_FROM_STEP requires RUN_DIR to be set"
    exit 1
fi
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-/workspace/game_agent/Game-AI-Agent/runs/Qwen3-8B_super_mario_${TIMESTAMP}}"
mkdir -p "${RUN_DIR}/checkpoints"

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[super_mario] Shutting down..."
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    echo "[super_mario] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
if [ -n "${RESUME_FROM_STEP}" ]; then
    echo "  Super Mario: RESUME from step ${RESUME_FROM_STEP}"
else
    echo "  Super Mario: Warm-start from SFT cold-start LoRA adapters"
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
echo "  DISPLAY:        ${DISPLAY:-<unset>}"
echo "  ORAK_PYTHON:    ${ORAK_PYTHON}"
echo ""
echo "  Super Mario game profile:"
echo "    - Single-player NES platformer via subprocess env (orak-mario)"
echo "    - 7 discrete actions (noop/right/right+A/A/left/left+A/down)"
echo "    - 500 max steps/episode"
echo "    - Reward: distance traveled + coins + time bonus"
echo "    - Subprocess env bridges NumPy 2.x (main) ↔ NumPy 1.x (NES)"
echo "    - Skill bank starts empty; LoRA adapters warm-started from SFT"
echo ""
echo "  Super Mario segmentation tuning:"
echo "    - LLM teacher max_tokens: ${SKILLBANK_LLM_TEACHER_MAX_TOKENS}"
echo "    - Segment timeout: ${SKILLBANK_SEGMENT_TIMEOUT_S}s"
echo ""
echo "  GRPO tuned overrides:"
echo "    - LR:           ${GRPO_LR} (baseline was 2e-5)"
echo "    - KL coeff:     ${GRPO_KL_COEFF} (baseline was 0.08)"
echo "    - Clip ratio:   ${GRPO_CLIP_RATIO} (baseline was 0.1)"
echo "    - Max epochs:   ${GRPO_MAX_EPOCHS} (baseline was 2)"
echo "    - Adv clip:     ${GRPO_ADV_CLIP} (baseline was 3.0)"
echo ""
echo "    - FSDP batch:   ${GRPO_FSDP_BATCH_SIZE}"
echo "    - Max seq len:  ${GRPO_MAX_SEQ_LEN}"
echo ""
echo "  Training schedule:"
echo "    - Warmup steps: ${WARMUP_STEPS} (75% of total)"
echo "    - Initial KL:   ${INITIAL_KL} (baseline was 0.01)"
echo "    - Temperature:  ${INITIAL_TEMP} -> ${STEADY_TEMP} (baseline was 1.0 -> 0.7)"
echo ""
echo "  Episode control:"
echo "    - Stuck window: ${STUCK_WINDOW} (baseline was 15)"
echo "    - Min steps before stuck check: ${MIN_STEPS_BEFORE_STUCK} (baseline was 20)"
echo "══════════════════════════════════════════════════════════════"
echo ""

if [ -n "${RESUME_FROM_STEP}" ]; then
    echo "[super_mario] Resuming from checkpoint step ${RESUME_FROM_STEP}"
    echo "  Checkpoint: ${RUN_DIR}/checkpoints/step_$(printf '%04d' "${RESUME_FROM_STEP}")"
    echo ""
else
    echo "[super_mario] SFT cold-start adapters:"
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
    --games super_mario
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
    --warmup-steps "${WARMUP_STEPS}"
    --initial-kl-coeff "${INITIAL_KL}"
    --initial-temperature "${INITIAL_TEMP}"
    --steady-temperature "${STEADY_TEMP}"
    --stuck-window "${STUCK_WINDOW}"
    --min-steps-before-stuck "${MIN_STEPS_BEFORE_STUCK}"
)

if [ -n "${RESUME_FROM_STEP}" ]; then
    TRAIN_ARGS+=(--resume --resume-from-step "${RESUME_FROM_STEP}")
else
    TRAIN_ARGS+=(
        --load-decision-adapters "${DECISION_ADAPTERS}"
        --load-skillbank-adapters "${SKILLBANK_ADAPTERS}"
    )
fi

echo "[super_mario] Command:"
echo "  python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

# ── Run training ─────────────────────────────────────────────────────
python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Super Mario training COMPLETE"

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
    echo "  Super Mario training FAILED (exit code ${EXIT_CODE})"
    echo "  Check logs: ${RUN_DIR}/coevolution.log"
fi
echo ""
echo "  Run dir: ${RUN_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
