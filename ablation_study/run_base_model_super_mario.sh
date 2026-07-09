#!/usr/bin/env bash
# ======================================================================
#  Ablation: Base Qwen3-8B on Super Mario (No LoRA, No Skill Bank)
#
#  Super Mario requires the orak-mario conda env (gym-super-mario-bros /
#  nes_py need NumPy 1.x).  The vLLM server runs in the main env; the
#  episode runner runs in orak-mario and connects over HTTP.
#
#  Usage:
#    conda activate game-ai-agent
#    bash ablation_study/run_base_model_super_mario.sh
#
#    # With overrides:
#    EVAL_GPUS=0    bash ablation_study/run_base_model_super_mario.sh
#    EPISODES=16    bash ablation_study/run_base_model_super_mario.sh
#    NO_SERVER=1 VLLM_BASE_URL=http://localhost:8020/v1 \
#                   bash ablation_study/run_base_model_super_mario.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── Xvfb for NES rendering ───────────────────────────────────────────
if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb &>/dev/null; then
        XVFB_DISPLAY=":99"
        if ! pgrep -f "Xvfb ${XVFB_DISPLAY}" &>/dev/null; then
            echo "[base-mario] Starting Xvfb on ${XVFB_DISPLAY}..."
            Xvfb "${XVFB_DISPLAY}" -screen 0 1024x768x24 &>/dev/null &
            sleep 1
        fi
        export DISPLAY="${XVFB_DISPLAY}"
    else
        echo "[WARN] No DISPLAY set and Xvfb not found — NES rendering may fail."
    fi
fi

# ── Subprocess env (orak-mario) ───────────────────────────────────────
export ORAK_PYTHON="${ORAK_PYTHON:-/workspace/miniconda3/envs/orak-mario/bin/python}"
if [ ! -x "${ORAK_PYTHON}" ]; then
    echo "[ERROR] orak-mario Python not found at: ${ORAK_PYTHON}"
    echo "  Create it with:  conda create -n orak-mario python=3.11 && conda run -n orak-mario pip install gym-super-mario-bros nes-py"
    exit 1
fi

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PROJECT_ROOT}/../Orak/src:${PYTHONPATH:-}"

# ── Configurable parameters ──────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
EVAL_GPUS="${EVAL_GPUS:-0}"
VLLM_PORT="${VLLM_PORT:-8020}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
NO_SERVER="${NO_SERVER:-0}"
SEED="${SEED:-42}"
EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-500}"
TEMPERATURE="${TEMPERATURE:-0.3}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Python for vLLM server (must have vllm installed; not orak-mario) ─
VLLM_PYTHON="${VLLM_PYTHON:-/workspace/miniconda3/bin/python}"
if ! "${VLLM_PYTHON}" -c "import vllm" 2>/dev/null; then
    echo "[WARN] ${VLLM_PYTHON} has no vllm — trying conda envs..."
    for candidate in /workspace/miniconda3/envs/game-ai-agent/bin/python /workspace/miniconda3/bin/python; do
        if [ -x "${candidate}" ] && "${candidate}" -c "import vllm" 2>/dev/null; then
            VLLM_PYTHON="${candidate}"; break
        fi
    done
fi

# ── Output directory ─────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_DIR:-${PROJECT_ROOT}/ablation_study/output/base_model_super_mario_${TIMESTAMP}}"
mkdir -p "${OUTPUT_BASE}"

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    set +e
    echo ""
    echo "[base-mario] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[base-mario] Done."
}
trap cleanup EXIT INT TERM

# ── Banner ────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Ablation: Base Qwen3-8B on Super Mario (no LoRA, no bank)"
echo "══════════════════════════════════════════════════════════════"
echo "  Base model:     ${BASE_MODEL}"
echo "  Adapter:        NONE (vanilla base model)"
echo "  Skill bank:     NONE (--no-bank)"
echo "  Episodes:       ${EPISODES}"
echo "  Max steps:      ${MAX_STEPS}"
echo "  Temperature:    ${TEMPERATURE}"
echo "  GPU(s):         ${EVAL_GPUS}"
echo "  DISPLAY:        ${DISPLAY:-<unset>}"
echo "  ORAK_PYTHON:    ${ORAK_PYTHON}"
echo "  Output:         ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server (base model, no LoRA) ─────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    echo "[base-mario] Launching vLLM server on ${VLLM_HOST}:${VLLM_PORT}..."
    VLLM_LOG="${OUTPUT_BASE}/vllm_server.log"

    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
        "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
            --model "${BASE_MODEL}" \
            --host "${VLLM_HOST}" \
            --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL}" \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.90 \
            --dtype auto \
            --trust-remote-code \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!

    echo "[base-mario] vLLM PID=${VLLM_PID}, log at ${VLLM_LOG}"

    MAX_WAIT=600
    WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[base-mario] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[base-mario] ERROR: vLLM server exited unexpectedly. Check ${VLLM_LOG}"
            tail -30 "${VLLM_LOG}" 2>/dev/null || true
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[base-mario] ERROR: vLLM server did not start within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "[base-mario] Using existing vLLM server at ${VLLM_BASE_URL}"
fi

# ── Run inference ────────────────────────────────────────────────────
GAME_OUTPUT="${OUTPUT_BASE}/super_mario"
mkdir -p "${GAME_OUTPUT}"

EVAL_ARGS=(
    --games super_mario
    --episodes "${EPISODES}"
    --max_steps "${MAX_STEPS}"
    --temperature "${TEMPERATURE}"
    --model "${BASE_MODEL}"
    --seed "${SEED}"
    --output_dir "${GAME_OUTPUT}"
    --no-bank
)

echo "[base-mario] Command:"
echo "  ${ORAK_PYTHON} -m scripts.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

EXIT_CODE=0
${ORAK_PYTHON} -m scripts.run_qwen3_8b_eval "${EVAL_ARGS[@]}" || EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"

python3 - "${OUTPUT_BASE}" <<'PYEOF'
import json, sys, math
from pathlib import Path

output_base = Path(sys.argv[1])
coevo_reward = 930.00

summaries = list(output_base.rglob("rollout_summary.json"))
if not summaries:
    print("  No rollout_summary.json found.")
    sys.exit(0)

with open(summaries[0]) as f:
    data = json.load(f)

rewards = [ep["total_reward"] for ep in data["episode_stats"]]
n = len(rewards)
mean = sum(rewards) / n
std = math.sqrt(sum((x - mean)**2 for x in rewards) / (n - 1)) if n > 1 else 0
se = std / math.sqrt(n)
ci = 1.96 * se
delta = mean - coevo_reward
pct = (delta / coevo_reward * 100) if coevo_reward != 0 else 0

print(f"  Super Mario — Base Qwen3-8B (no LoRA, no bank)")
print(f"    Episodes:    {n}")
print(f"    Mean±CI:     {mean:.2f} ± {ci:.2f}")
print(f"    Std:         {std:.2f}")
print(f"    Min/Max:     {min(rewards):.0f} / {max(rewards):.0f}")
print(f"    vs CoEvo:    {coevo_reward:.2f}  Δ={delta:+.2f} ({pct:+.1f}%)")

summary_path = output_base / "ablation_summary.json"
with open(summary_path, "w") as f:
    json.dump({
        "experiment": "base_model_super_mario",
        "description": "Base Qwen3-8B (no LoRA, no skill bank) on Super Mario",
        "results": [{
            "game": "super_mario",
            "base_model_reward": mean,
            "ci_95": ci,
            "std": std,
            "coevo_full_reward": coevo_reward,
            "delta": delta,
            "delta_pct": pct,
            "episodes": n,
        }],
    }, f, indent=2)
print(f"\n  Summary: {summary_path}")
PYEOF

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "  Super Mario base-model inference FAILED (exit code ${EXIT_CODE})"
fi
echo "  Output: ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
