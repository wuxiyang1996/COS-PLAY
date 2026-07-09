#!/usr/bin/env bash
# ======================================================================
#  Super Mario ablation: adapter evaluation (no skill bank variants)
#
#  Adapter options:   base — vanilla Qwen/Qwen3.5-9B, no LoRA
#                     sft  — SFT cold-start LoRA
#
#  Requires orak-mario conda env (gym-super-mario-bros / nes_py need
#  NumPy 1.x). The vLLM server runs in the main env; the episode
#  runner runs in orak-mario.
#
#  Usage:
#    bash ablation_study/run_super_mario_ablation.sh --adapter base
#    bash ablation_study/run_super_mario_ablation.sh --adapter sft
#    EVAL_GPUS=0 EPISODES=16 bash ablation_study/run_super_mario_ablation.sh --adapter base
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Parse arguments ──────────────────────────────────────────────────
ADAPTER_TYPE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapter) ADAPTER_TYPE="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done
if [ -z "${ADAPTER_TYPE}" ]; then
    echo "Usage: $0 --adapter {base|sft}"
    exit 1
fi

# ── Environment ──────────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PROJECT_ROOT}/../Orak/src:${PYTHONPATH:-}"

# ── Xvfb for NES rendering ──────────────────────────────────────────
if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb &>/dev/null; then
        XVFB_DISPLAY=":99"
        if ! pgrep -f "Xvfb ${XVFB_DISPLAY}" &>/dev/null; then
            echo "[mario-ablation] Starting Xvfb on ${XVFB_DISPLAY}..."
            Xvfb "${XVFB_DISPLAY}" -screen 0 1024x768x24 &>/dev/null &
            sleep 1
        fi
        export DISPLAY="${XVFB_DISPLAY}"
    else
        echo "[WARN] No DISPLAY set and Xvfb not found — NES rendering may fail."
    fi
fi

# ── Subprocess env (orak-mario) ──────────────────────────────────────
export ORAK_PYTHON="${ORAK_PYTHON:-/workspace/miniconda3/envs/orak-mario/bin/python}"
if [ ! -x "${ORAK_PYTHON}" ]; then
    echo "[ERROR] orak-mario Python not found: ${ORAK_PYTHON}"
    echo "  Create: conda create -n orak-mario python=3.11 && conda run -n orak-mario pip install gym-super-mario-bros nes-py"
    exit 1
fi

VLLM_PYTHON="${VLLM_PYTHON:-/workspace/miniconda3/bin/python}"
if ! "${VLLM_PYTHON}" -c "import vllm" 2>/dev/null; then
    for candidate in /workspace/miniconda3/envs/game-ai-agent/bin/python /workspace/miniconda3/bin/python; do
        [ -x "${candidate}" ] && "${candidate}" -c "import vllm" 2>/dev/null && VLLM_PYTHON="${candidate}" && break
    done
fi

# ── Configurable parameters ─────────────────────────────────────────
RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
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

# ── Resolve adapter ──────────────────────────────────────────────────
ENABLE_LORA=0
ADAPTER_PATH=""
LORA_NAME=""
MODEL_FOR_EVAL="${BASE_MODEL}"

case "${ADAPTER_TYPE}" in
    base)
        ADAPTER_DESC="NONE (vanilla base model)"
        ;;
    sft)
        ADAPTER_PATH="${RUNS_DIR}/sft_coldstart/decision/action_taking/action_taking"
        LORA_NAME="qwen3-8b-mario-sft"
        ENABLE_LORA=1
        MODEL_FOR_EVAL="${LORA_NAME}"
        ADAPTER_DESC="SFT cold-start → ${ADAPTER_PATH}"
        [ ! -f "${ADAPTER_PATH}/adapter_config.json" ] && echo "[ERROR] SFT adapter not found: ${ADAPTER_PATH}" && exit 1
        ;;
    *)  echo "[ERROR] Unknown adapter: ${ADAPTER_TYPE}. Use: base, sft"; exit 1 ;;
esac

# ── Output directory ─────────────────────────────────────────────────
TAG="${ADAPTER_TYPE}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_DIR:-${PROJECT_ROOT}/ablation_study/output/${TAG}_super_mario_${TIMESTAMP}}"
mkdir -p "${OUTPUT_BASE}"

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    set +e
    echo ""
    echo "[${TAG}-mario] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do kill -0 "${VLLM_PID}" 2>/dev/null || break; sleep 1; done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[${TAG}-mario] Done."
}
trap cleanup EXIT INT TERM

# ── Banner ───────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Super Mario Ablation: adapter=${ADAPTER_TYPE} (no bank)"
echo "══════════════════════════════════════════════════════════════"
echo "  Base model:     ${BASE_MODEL}"
echo "  Adapter:        ${ADAPTER_DESC}"
echo "  Episodes:       ${EPISODES}    Max steps: ${MAX_STEPS}"
echo "  GPU(s):         ${EVAL_GPUS}"
echo "  DISPLAY:        ${DISPLAY:-<unset>}"
echo "  ORAK_PYTHON:    ${ORAK_PYTHON}"
echo "  Output:         ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server ───────────────────────────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    VLLM_LOG="${OUTPUT_BASE}/vllm_server.log"
    VLLM_ARGS=(
        --model "${BASE_MODEL}" --host "${VLLM_HOST}" --port "${VLLM_PORT}"
        --tensor-parallel-size "${TENSOR_PARALLEL}" --max-model-len 4096
        --gpu-memory-utilization 0.90 --dtype auto --trust-remote-code
    )
    if [ "${ENABLE_LORA}" = "1" ]; then
        VLLM_ARGS+=(--enable-lora --max-lora-rank 16)
        VLLM_ARGS+=(--lora-modules "${LORA_NAME}=${ADAPTER_PATH}")
    fi

    echo "[${TAG}-mario] Launching vLLM server..."
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
        "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!

    MAX_WAIT=600; WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1 && break
        kill -0 "${VLLM_PID}" 2>/dev/null || { echo "[ERROR] vLLM exited. See ${VLLM_LOG}"; exit 1; }
        sleep 5; WAITED=$((WAITED + 5))
    done
    [ ${WAITED} -ge ${MAX_WAIT} ] && echo "[ERROR] vLLM did not start within ${MAX_WAIT}s." && exit 1
    echo "[${TAG}-mario] vLLM ready (${WAITED}s)."
else
    echo "[${TAG}-mario] Using existing server at ${VLLM_BASE_URL}"
fi

# ── Run inference ────────────────────────────────────────────────────
GAME_OUTPUT="${OUTPUT_BASE}/super_mario"
mkdir -p "${GAME_OUTPUT}"

EVAL_ARGS=(
    --games super_mario --episodes "${EPISODES}" --max_steps "${MAX_STEPS}"
    --temperature "${TEMPERATURE}" --model "${MODEL_FOR_EVAL}"
    --seed "${SEED}" --output_dir "${GAME_OUTPUT}" --no-bank
)

echo "[${TAG}-mario] ${ORAK_PYTHON} -m inference.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

EXIT_CODE=0
${ORAK_PYTHON} -m inference.run_qwen3_8b_eval "${EVAL_ARGS[@]}" || EXIT_CODE=$?

# ── Post-run comparison ──────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"

python3 - "${OUTPUT_BASE}" "${ADAPTER_TYPE}" <<'PYEOF'
import json, sys, math
from pathlib import Path

output_base, adapter_type = Path(sys.argv[1]), sys.argv[2]
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

labels = {"base": "Base Qwen/Qwen3.5-9B (no LoRA, no bank)", "sft": "SFT cold-start (no bank)"}
print(f"  Super Mario — {labels.get(adapter_type, adapter_type)}")
print(f"    Episodes:    {n}")
print(f"    Mean±CI:     {mean:.2f} ± {ci:.2f}")
print(f"    Std:         {std:.2f}")
print(f"    Min/Max:     {min(rewards):.0f} / {max(rewards):.0f}")
print(f"    vs CoEvo:    {coevo_reward:.2f}  Δ={delta:+.2f} ({pct:+.1f}%)")

summary_path = output_base / "ablation_summary.json"
with open(summary_path, "w") as f:
    json.dump({
        "experiment": f"{adapter_type}_super_mario",
        "results": [{"game": "super_mario", "reward": mean, "ci_95": ci,
                      "std": std, "coevo_reward": coevo_reward,
                      "delta": delta, "delta_pct": pct, "episodes": n}],
    }, f, indent=2)
print(f"\n  Summary: {summary_path}")
PYEOF

[ ${EXIT_CODE} -ne 0 ] && echo "  Super Mario ablation FAILED (exit code ${EXIT_CODE})"
echo "  Output: ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"
exit ${EXIT_CODE}
