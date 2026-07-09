n#!/usr/bin/env bash
# ======================================================================
#  Resume Candy Crush training from step-4 checkpoint.
#
#  Context: The original Qwen3-8B_20260321_213813 run peaked at step 6
#  (mean reward 655) then regressed due to action hallucination spiral.
#  Root-cause fixes applied to the codebase:
#
#    1. Format penalty (episode_runner.py)
#       OOB / unparseable actions now get reward=0 in GRPO records,
#       preventing the model from being rewarded for garbage outputs.
#
#    2. LR cosine decay (config.py)
#       Non-from-scratch runs now decay LR from 5e-5 → 5e-6 over the
#       training horizon (was flat 5e-5). KL penalty ramps up 50% to
#       keep the policy closer to reference.
#
#    3. Replay buffer cap (grpo_training.py)
#       Replay samples are capped at 1:1 ratio with fresh on-policy
#       data (was unbounded, reaching 4:1 stale-to-fresh).
#
#    4. Contract adapter stabilization (grpo_training.py)
#       SkillBank train threshold raised 16→32, contract LR halved,
#       KL raised 50%, epochs reduced 2→1.  Prevents contract updates
#       from destabilizing the action_taking adapter.
#
#  The step-4 checkpoint (mean_reward=623, OOB rate=3.2%) is the last
#  clean state before the contract co-training at step 4 caused the
#  cascade that worsened after step 8.
#
#  Usage:
#    conda activate game-ai-agent
#    bash scripts/run_resume_candy_crush.sh
#
#    # Or with overrides:
#    TOTAL_STEPS=20 EPISODES=12 bash scripts/run_resume_candy_crush.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

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

TOTAL_STEPS="${TOTAL_STEPS:-10}"
EPISODES="${EPISODES:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
RESUME_STEP="${RESUME_STEP:-5}"

RUN_DIR="/workspace/game_agent/Game-AI-Agent/runs/Qwen3-8B_20260321_213813"

# ── Validate checkpoint ──────────────────────────────────────────────
CKPT_DIR="${RUN_DIR}/checkpoints/step_$(printf '%04d' "${RESUME_STEP}")"
if [ ! -d "${CKPT_DIR}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT_DIR}"
    echo "Available checkpoints:"
    ls -d "${RUN_DIR}/checkpoints"/step_* 2>/dev/null || echo "  (none)"
    exit 1
fi

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[resume] Shutting down..."
    # Kill any child processes
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    echo "[resume] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  RESUME: Candy Crush from step ${RESUME_STEP}"
echo "══════════════════════════════════════════════════════════════"
echo "  Run dir:        ${RUN_DIR}"
echo "  Checkpoint:     ${CKPT_DIR}"
echo "  Model:          ${MODEL}"
echo "  Total steps:    ${TOTAL_STEPS}"
echo "  Episodes/step:  ${EPISODES}"
echo "  Checkpoint:     every ${CKPT_INTERVAL} steps"
echo "  vLLM GPUs:      ${VLLM_GPUS}"
echo "  GRPO GPUs:      ${GRPO_GPUS}"
echo "  Spec decode:    ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo ""
echo "  Fixes applied (round 1):"
echo "    - OOB action format penalty (reward=0)"
echo "    - LR cosine decay (5e-5 → 5e-6)"
echo "    - Replay buffer capped at 1:1 fresh:stale"
echo "    - Contract adapter: threshold=32, lr×0.5, kl×1.5, 1 epoch"
echo ""
echo "  Fixes applied (round 2):"
echo "    - Raw model output in GRPO records (not reconstructed)"
echo "    - Format bonus (+1.0) for valid action outputs"
echo "    - Constrained decoding: max_tokens=48, extra stop seq"
echo "    - Best-checkpoint rollback after 3 consecutive declines"
echo "    - Stronger KL ramp: ×(1+progress) instead of ×(1+0.5*progress)"
echo "    - Reduced GRPO epochs: max 1 (was max 2)"
echo "══════════════════════════════════════════════════════════════"
echo ""

# Show checkpoint metadata
echo "[resume] Checkpoint metadata:"
python -c "
import json
with open('${CKPT_DIR}/metadata.json') as f:
    m = json.load(f)
print(f'  Step:        {m[\"step\"]}')
print(f'  Mean reward: {m[\"mean_reward\"]}')
print(f'  Skills:      {m[\"n_skills\"]}')
print(f'  Mode:        {m[\"mode\"]}')
for g, stats in m.get('reward_per_game', {}).items():
    print(f'  {g}: mean={stats[\"mean_reward\"]:.1f}, max={stats[\"max_reward\"]:.0f}, min={stats[\"min_reward\"]:.0f}')
"
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
    --resume-from-step "${RESUME_STEP}"
    --debug-io
    # shellcheck disable=SC2086
    --vllm-gpus ${VLLM_GPUS}
    --grpo-devices ${GRPO_GPUS}
    --vllm-base-port "${PORT}"
    --vllm-gpu-util "${GPU_UTIL}"
    --speculative-model "${SPEC_MODEL}"
    --num-speculative-tokens "${SPEC_TOKENS}"
)

echo "[resume] Command:"
echo "  python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

# ── Run training ─────────────────────────────────────────────────────
python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Resume training COMPLETE"

    echo ""
    echo "  Step log:"
    if [ -f "${RUN_DIR}/step_log.jsonl" ]; then
        python -c "
import json
with open('${RUN_DIR}/step_log.jsonl') as f:
    rows = [json.loads(l) for l in f if l.strip()]
# Show last 5 steps
for r in rows[-5:]:
    step = r['step']
    mr = r['mean_reward']
    ns = r.get('n_skills', '?')
    wt = r['wall_time_s'] / 60
    print(f'  Step {step:2d}: mean_reward={mr:6.1f}  skills={ns}  time={wt:.1f}m')
"
    fi
else
    echo "  Resume training FAILED (exit code ${EXIT_CODE})"
    echo "  Check logs: ${RUN_DIR}/coevolution.log"
fi
echo ""
echo "  Run dir: ${RUN_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
