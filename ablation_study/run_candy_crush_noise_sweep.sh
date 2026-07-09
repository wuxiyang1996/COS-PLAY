#!/usr/bin/env bash
# ======================================================================
#  R10 — State-Summary Noise Robustness Sweep (candy_crush)
#
#  Eval the paper's best candy_crush adapter under perturbed state
#  summaries.  No retraining — purely an inference-time robustness test.
#  Hook lives in decision_agents/agent_helper.py:_apply_summary_noise,
#  gated by COSPLAY_SUMMARY_NOISE.
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
#    VLLM_PORT=8060
#    EPISODES=16
#    SEED=42
#    OUTPUT_ROOT=ablation_study/output/candy_crush_noise_sweep_<ts>
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
VLLM_PORT="${VLLM_PORT:-8060}"
EPISODES="${EPISODES:-16}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MAX_STEPS="${MAX_STEPS:-50}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_20260321_213813_(Candy_crush)"
ADAPTER_PATH="${RUN_DIR}/best/adapters/decision/action_taking"
LORA_NAME="qwen3-8b-candy-crush-best"
# Cold-start bank is fine (same as paper's reference run with bank loaded)
BANK="${BANK:-${PROJECT_ROOT}/runs/fixed_skillbank_seeds/candy_crush/skill_bank.jsonl}"

if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "[r10] ERROR: adapter not found at ${ADAPTER_PATH}" >&2
    exit 1
fi
if [ ! -f "${BANK}" ]; then
    echo "[r10] ERROR: bank not found at ${BANK}" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/ablation_study/output/candy_crush_noise_sweep_${TS}}"
mkdir -p "${OUTPUT_ROOT}"
SWEEP_LOG="${OUTPUT_ROOT}/sweep.log"
exec > >(tee -a "${SWEEP_LOG}") 2>&1

echo "══════════════════════════════════════════════════════════════"
echo "  R10 Candy Crush Noise Sweep — ${TS}"
echo "══════════════════════════════════════════════════════════════"
echo "  GPU:        ${GPU}"
echo "  vLLM port:  ${VLLM_PORT}"
echo "  Adapter:    ${ADAPTER_PATH}"
echo "  Bank:       ${BANK}"
echo "  Episodes:   ${EPISODES}, max_steps=${MAX_STEPS}, T=${TEMPERATURE}, seed=${SEED}"
echo "  Output:     ${OUTPUT_ROOT}"
echo "══════════════════════════════════════════════════════════════"

VLLM_PID=""
cleanup() {
    set +e
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[r10] tearing down vLLM (PID ${VLLM_PID})..."
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

echo "[r10] launching vLLM on GPU ${GPU} port ${VLLM_PORT}..."
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
        --lora-modules "${LORA_NAME}=${ADAPTER_PATH}" \
        --max-lora-rank 16 \
        > "${OUTPUT_ROOT}/vllm.log" 2>&1 &
VLLM_PID=$!

MAX_WAIT=600
WAITED=0
while [ ${WAITED} -lt ${MAX_WAIT} ]; do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "[r10] vLLM ready (waited ${WAITED}s)"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[r10] ERROR: vLLM exited unexpectedly"
        tail -20 "${OUTPUT_ROOT}/vllm.log"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
if [ ${WAITED} -ge ${MAX_WAIT} ]; then
    echo "[r10] ERROR: vLLM did not become healthy in ${MAX_WAIT}s"
    exit 1
fi

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="EMPTY"

# Cell definitions: NAME|NOISE_SPEC
CELLS=(
    "N0|"
    "N2|dropout:p=0.25"
    "N3|dropout:p=0.5"
    "F-board|drop:board"
    "NUM|num_noise:0.2"
)

CELLS_TO_RUN="${CELLS_TO_RUN:-N0 N2 N3 F-board NUM}"

for spec in "${CELLS[@]}"; do
    NAME="${spec%%|*}"
    NOISE="${spec#*|}"
    if ! echo " ${CELLS_TO_RUN} " | grep -q " ${NAME} "; then
        echo "[r10] skipping ${NAME} (not in CELLS_TO_RUN='${CELLS_TO_RUN}')"
        continue
    fi
    CELL_DIR="${OUTPUT_ROOT}/${NAME}"
    mkdir -p "${CELL_DIR}"
    CELL_LOG="${CELL_DIR}/cell.log"
    EVAL_OUT="${CELL_DIR}/eval_out"
    echo
    echo "══════════════════════════════════════════════════════════════"
    echo "  R10 Cell: ${NAME}  | noise='${NOISE}'"
    echo "══════════════════════════════════════════════════════════════"
    CELL_START=$(date +%s)
    EXIT_CODE=0
    COSPLAY_SUMMARY_NOISE="${NOISE}" \
    COSPLAY_SUMMARY_NOISE_SEED="${SEED}" \
        python -m scripts.run_qwen3_8b_eval \
            --games candy_crush \
            --episodes "${EPISODES}" \
            --max_steps "${MAX_STEPS}" \
            --temperature "${TEMPERATURE}" \
            --model "${LORA_NAME}" \
            --seed "${SEED}" \
            --output_dir "${EVAL_OUT}" \
            --bank "${BANK}" \
            > "${CELL_LOG}" 2>&1 || EXIT_CODE=$?
    CELL_WALL=$(( $(date +%s) - CELL_START ))
    cat > "${CELL_DIR}/cell_meta.json" <<EOF
{
  "experiment": "R10_summary_noise",
  "cell": "${NAME}",
  "noise_spec": "${NOISE}",
  "noise_seed": ${SEED},
  "adapter_path": "${ADAPTER_PATH}",
  "bank": "${BANK}",
  "episodes": ${EPISODES},
  "max_steps": ${MAX_STEPS},
  "temperature": ${TEMPERATURE},
  "seed": ${SEED},
  "cell_wall_s": ${CELL_WALL},
  "exit_code": ${EXIT_CODE},
  "timestamp": "$(date -u +%FT%TZ)"
}
EOF
    if [ ${EXIT_CODE} -eq 0 ]; then
        ROLLOUT_JSON=$(find "${EVAL_OUT}" -name "rollout_summary.json" | head -1)
        if [ -n "${ROLLOUT_JSON}" ]; then
            MR=$(python3 -c "import json; d=json.load(open('${ROLLOUT_JSON}')); print(f\"mean={d['mean_reward']:.2f} max={d['max_reward']:.2f} min={d['min_reward']:.2f}\")")
            echo "[r10] ✓ ${NAME} COMPLETE (${CELL_WALL}s)  ${MR}"
        else
            echo "[r10] ⚠ ${NAME} done but no rollout_summary.json found"
        fi
    else
        echo "[r10] ✗ ${NAME} FAILED (exit ${EXIT_CODE}, ${CELL_WALL}s)"
    fi
done

echo
echo "══════════════════════════════════════════════════════════════"
echo "  R10 sweep complete.  Aggregating..."
echo "══════════════════════════════════════════════════════════════"
python3 "${SCRIPT_DIR}/aggregate_candy_crush_noise.py" \
    --output-root "${OUTPUT_ROOT}" \
    || echo "[r10] aggregate script not found / failed (will hand-aggregate)"

echo "[r10] all done. Results: ${OUTPUT_ROOT}/summary.{json,md}"
