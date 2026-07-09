#!/usr/bin/env bash
# ======================================================================
#  R10 — Multi-LoRA State-Summary Noise Sweep (candy_crush)
#
#  Same cells as run_candy_crush_noise_sweep.sh, but uses the SAME
#  multi-LoRA inference path as training rollouts:
#     trainer/coevolution/episode_runner.run_episode_async
#  routes calls to {skill_selection, action_taking, segment, contract,
#  curator, base} LoRAs via AsyncVLLMClient.
#
#  Noise hook: decision_agents/agent_helper.py:_apply_summary_noise,
#  gated by env var COSPLAY_SUMMARY_NOISE (consumed inside
#  build_rag_summary which the training rollout path calls each step).
#
#  Cells:
#    N0       control                        ""
#    N2       dropout p=0.25                 "dropout:p=0.25"
#    N3       dropout p=0.50                 "dropout:p=0.5"
#    F-board  drop board= field              "drop:board"
#    NUM      ±20% perturb numerics          "num_noise:0.2"
#
#  Env overrides:
#    GPU=4
#    VLLM_PORT=8070
#    EPISODES=8
#    MAX_STEPS=200
#    SEED=42
#    CONCURRENCY=4
#    CELLS_TO_RUN="N0 N2 N3 F-board NUM"
#    OUTPUT_ROOT=ablation_study/output/r10_multilora_<ts>
# ======================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate game-ai-agent

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${GPU:-4}"
VLLM_PORT="${VLLM_PORT:-8070}"
EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-200}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.3}"
CONCURRENCY="${CONCURRENCY:-4}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_20260321_213813_(Candy_crush)"
# BEST_DIR can be overridden, e.g.
#   BEST_DIR=${RUN_DIR}/checkpoints/step_99999   ← peak-reward snapshot (mean_reward=657.75)
#   BEST_DIR=${RUN_DIR}/best                     ← default "best" save (mean_reward=528.375)
BEST_DIR="${BEST_DIR:-${RUN_DIR}/best}"

# All five LoRA adapters used during training rollouts.
ADAPTER_ACTION="${BEST_DIR}/adapters/decision/action_taking"
ADAPTER_SKILL="${BEST_DIR}/adapters/decision/skill_selection"
ADAPTER_SEGMENT="${BEST_DIR}/adapters/skillbank/segment"
ADAPTER_CONTRACT="${BEST_DIR}/adapters/skillbank/contract"
ADAPTER_CURATOR="${BEST_DIR}/adapters/skillbank/curator"

BANK="${BANK:-${BEST_DIR}/banks/candy_crush/skill_bank.jsonl}"

for D in "${ADAPTER_ACTION}" "${ADAPTER_SKILL}" "${ADAPTER_SEGMENT}" "${ADAPTER_CONTRACT}" "${ADAPTER_CURATOR}"; do
    if [ ! -f "${D}/adapter_config.json" ]; then
        echo "[r10-mlora] ERROR: missing adapter at ${D}" >&2
        exit 1
    fi
done
if [ ! -f "${BANK}" ]; then
    echo "[r10-mlora] ERROR: bank not found at ${BANK}" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/ablation_study/output/r10_multilora_${TS}}"
mkdir -p "${OUTPUT_ROOT}"
SWEEP_LOG="${OUTPUT_ROOT}/sweep.log"
exec > >(tee -a "${SWEEP_LOG}") 2>&1

echo "══════════════════════════════════════════════════════════════"
echo "  R10 Multi-LoRA Candy Crush Noise Sweep — ${TS}"
echo "══════════════════════════════════════════════════════════════"
echo "  GPU:           ${GPU}"
echo "  vLLM port:     ${VLLM_PORT}"
echo "  Adapters:"
echo "    action_taking   = ${ADAPTER_ACTION}"
echo "    skill_selection = ${ADAPTER_SKILL}"
echo "    segment         = ${ADAPTER_SEGMENT}"
echo "    contract        = ${ADAPTER_CONTRACT}"
echo "    curator         = ${ADAPTER_CURATOR}"
echo "  Bank:          ${BANK}"
echo "  Episodes:      ${EPISODES}, max_steps=${MAX_STEPS}, T=${TEMPERATURE}, seed=${SEED}, conc=${CONCURRENCY}"
echo "  Output:        ${OUTPUT_ROOT}"
echo "══════════════════════════════════════════════════════════════"

VLLM_PID=""
cleanup() {
    set +e
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[r10-mlora] tearing down vLLM (PID ${VLLM_PID})..."
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[r10-mlora] launching vLLM on GPU ${GPU} port ${VLLM_PORT} with 5 LoRAs..."
CUDA_VISIBLE_DEVICES="${GPU}" \
    python -m vllm.entrypoints.openai.api_server \
        --model "${BASE_MODEL}" \
        --host 127.0.0.1 \
        --port "${VLLM_PORT}" \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --dtype auto \
        --trust-remote-code \
        --enable-lora \
        --max-loras 5 \
        --max-lora-rank 16 \
        --lora-modules \
            "action_taking=${ADAPTER_ACTION}" \
            "skill_selection=${ADAPTER_SKILL}" \
            "segment=${ADAPTER_SEGMENT}" \
            "contract=${ADAPTER_CONTRACT}" \
            "curator=${ADAPTER_CURATOR}" \
        > "${OUTPUT_ROOT}/vllm.log" 2>&1 &
VLLM_PID=$!

MAX_WAIT=600
WAITED=0
while [ ${WAITED} -lt ${MAX_WAIT} ]; do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "[r10-mlora] vLLM ready (waited ${WAITED}s)"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[r10-mlora] ERROR: vLLM exited unexpectedly"
        tail -40 "${OUTPUT_ROOT}/vllm.log"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
if [ ${WAITED} -ge ${MAX_WAIT} ]; then
    echo "[r10-mlora] ERROR: vLLM did not become healthy in ${MAX_WAIT}s"
    exit 1
fi

# Confirm all 5 LoRAs are visible
echo "[r10-mlora] /v1/models registered:"
curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); [print('   -',m['id']) for m in d['data']]" \
    || true

CELLS_TO_RUN="${CELLS_TO_RUN:-N0 N2 N3 F-board NUM}"

echo
echo "══════════════════════════════════════════════════════════════"
echo "  Running cells: ${CELLS_TO_RUN}"
echo "══════════════════════════════════════════════════════════════"

python -m ablation_study.r10_noise_sweep_multilora \
    --vllm-url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --bank "${BANK}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --concurrency "${CONCURRENCY}" \
    --cells ${CELLS_TO_RUN} \
    --output "${OUTPUT_ROOT}"

echo
echo "[r10-mlora] all done. Results: ${OUTPUT_ROOT}/summary.{json,md}"
