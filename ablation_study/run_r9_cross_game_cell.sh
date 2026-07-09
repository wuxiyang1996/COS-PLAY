#!/usr/bin/env bash
# ======================================================================
#  R9 — Cross-game skill transfer (one cell = one target game)
#
#  Launches vLLM with TARGET game's best action_taking LoRA, then runs
#  the matching evaluator with a DONOR bank (= union of OTHER 5 games'
#  skill banks built by build_donor_banks.py).
#
#  Usage:
#    bash run_r9_cross_game_cell.sh <target_game> <donor_bank_path> [output_root]
#
#  target_game ∈ {twenty_forty_eight, candy_crush, tetris, super_mario,
#                  avalon, diplomacy}
#
#  Env overrides:
#    EPISODES_SP / EPISODES_MP : per-cell episode count for single/multi-player
#                                (default: 16 / 10)
#    EVAL_GPUS                 : default 0  (single vLLM instance per cell)
#    OPPONENT_MODEL            : default gpt-5.4  (multi-player only)
#    TEMPERATURE_SP / TEMPERATURE_MP : sampling temperature (default 0.3 / 0.4)
#    SEED                      : default 42
# ======================================================================
set -uo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <target_game> <donor_bank_path> [output_root]" >&2
    exit 2
fi

TARGET="$1"
DONOR_BANK="$2"
OUTPUT_ROOT="${3:-${OUTPUT_ROOT:-}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ ! -f "${DONOR_BANK}" ]; then
    echo "[r9-cell:${TARGET}] ERROR: donor bank not found: ${DONOR_BANK}" >&2
    exit 1
fi

# Source conda
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate game-ai-agent

# Common env
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Load OpenRouter key if multi-player (gpt-5.4 opponent)
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f /workspace/keys.py ]; then
    KEY=$(python3 -c "
import sys; sys.path.insert(0, '/workspace')
import keys
for k in ['openrouter_api_key','open_router_api_key','OPENROUTER_API_KEY']:
    v = getattr(keys, k, None)
    if v:
        print(v); break
" 2>/dev/null)
    [ -n "${KEY}" ] && export OPENROUTER_API_KEY="${KEY}"
fi

# Defaults — auto-pick first GPU with > 60 GiB free if EVAL_GPUS not set,
# otherwise use what caller passed.
if [ -z "${EVAL_GPUS:-}" ]; then
    EVAL_GPUS=$(python3 - <<'PY' 2>/dev/null
import subprocess
try:
    out = subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
        text=True, timeout=5)
    for line in out.strip().splitlines():
        idx, free = [p.strip() for p in line.split(',')]
        if int(free) >= 60000:
            print(idx); break
    else:
        print('0')
except Exception:
    print('0')
PY
)
fi
EVAL_GPUS="${EVAL_GPUS:-0}"
EPISODES_SP="${EPISODES_SP:-16}"
EPISODES_MP="${EPISODES_MP:-10}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
TEMPERATURE_SP="${TEMPERATURE_SP:-0.3}"
TEMPERATURE_MP="${TEMPERATURE_MP:-0.4}"
SEED="${SEED:-42}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8030}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

# Resolve per-target paths / config
case "${TARGET}" in
    twenty_forty_eight)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_2048_20260322_071227"
        LORA_NAME="qwen3-8b-2048-best"
        EVAL_MODULE="scripts.run_qwen3_8b_eval"
        EPISODES="${EPISODES_SP}"
        TEMP="${TEMPERATURE_SP}"
        IS_MULTIPLAYER=0
        MAX_STEPS=200
        ;;
    candy_crush)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_20260321_213813_(Candy_crush)"
        LORA_NAME="qwen3-8b-candy-crush-best"
        EVAL_MODULE="scripts.run_qwen3_8b_eval"
        EPISODES="${EPISODES_SP}"
        TEMP="${TEMPERATURE_SP}"
        IS_MULTIPLAYER=0
        MAX_STEPS=200
        ;;
    tetris)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_tetris_20260322_170438"
        LORA_NAME="qwen3-8b-tetris-best"
        EVAL_MODULE="scripts.run_qwen3_8b_eval"
        EPISODES="${EPISODES_SP}"
        TEMP="${TEMPERATURE_SP}"
        IS_MULTIPLAYER=0
        # Tetris training (qwen3_decision_agent.py:1584) auto-wraps env with
        # TetrisMacroActionWrapper (placement-level actions). The eval must
        # match the training action space, otherwise the action_taking LoRA
        # outputs primitive-vocab actions into a macro-vocab env and rewards
        # are meaningless. MAX_STEPS is also smaller since each macro action
        # consumes ~5-10 primitive frames.
        MAX_STEPS=50
        TETRIS_MACRO_ACTIONS=1
        ;;
    super_mario)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_super_mario_20260323_030839"
        LORA_NAME="qwen3-8b-mario-best"
        EVAL_MODULE="scripts.run_qwen3_8b_eval"
        EPISODES="${EPISODES_SP}"
        TEMP="${TEMPERATURE_SP}"
        IS_MULTIPLAYER=0
        # super_mario action set is already "Jump Level: 0..6" (7-bucket macro);
        # MAX_STEPS counts macro decisions, not raw frames.
        MAX_STEPS=200
        # NES rendering needs Xvfb + orak-mario env (game-ai-agent lacks
        # `gym` / `gym-super-mario-bros` because of NumPy 2.x incompat).
        # The orak-mario env DOES have openai + gym + gym_super_mario_bros
        # + Game-AI-Agent's local modules (via PYTHONPATH), so we can run
        # the entire eval script using that env's Python — only the eval
        # process moves; vLLM keeps running in game-ai-agent.
        SUPER_MARIO_USE_ORAK_PY=1
        export ORAK_PYTHON="${ORAK_PYTHON:-/workspace/miniconda3/envs/orak-mario/bin/python}"
        if [ ! -x "${ORAK_PYTHON}" ]; then
            echo "[r9-cell:${TARGET}] ERROR: ORAK_PYTHON not executable: ${ORAK_PYTHON}" >&2
            exit 1
        fi
        if [ -z "${DISPLAY:-}" ] && command -v Xvfb &>/dev/null; then
            XVFB_DISPLAY=":99"
            if ! pgrep -f "Xvfb ${XVFB_DISPLAY}" &>/dev/null; then
                Xvfb "${XVFB_DISPLAY}" -screen 0 1024x768x24 &>/dev/null &
                sleep 1
            fi
            export DISPLAY="${XVFB_DISPLAY}"
        fi
        ;;
    avalon)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_avalon_20260322_200424"
        LORA_NAME="qwen3-8b-avalon-best"
        EVAL_MODULE="scripts.run_qwen3_avalon_matched"
        EPISODES="${EPISODES_MP}"
        TEMP="${TEMPERATURE_MP}"
        IS_MULTIPLAYER=1
        ;;
    diplomacy)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_diplomacy_20260322_234548"
        LORA_NAME="qwen3-8b-diplomacy-best"
        EVAL_MODULE="scripts.run_diplomacy_discrete_eval"
        EPISODES="${EPISODES_MP}"
        TEMP="${TEMPERATURE_MP}"
        IS_MULTIPLAYER=1
        ;;
    *)
        echo "[r9-cell] ERROR: unknown target game '${TARGET}'" >&2
        exit 2 ;;
esac

ADAPTER_PATH="${RUN_DIR}/best/adapters/decision/action_taking"
if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "[r9-cell:${TARGET}] ERROR: adapter not found: ${ADAPTER_PATH}" >&2
    exit 1
fi

if [ -z "${OUTPUT_ROOT}" ]; then
    OUTPUT_ROOT="${PROJECT_ROOT}/ablation_study/output/r9_cross_game/runs_$(date +%Y%m%d_%H%M%S)"
fi
CELL_DIR="${OUTPUT_ROOT}/${TARGET}_donor"
mkdir -p "${CELL_DIR}"
LOG_FILE="${CELL_DIR}/cell.log"
EVAL_OUT_DIR="${CELL_DIR}/eval_out"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

echo "══════════════════════════════════════════════════════════════" | tee "${LOG_FILE}"
echo "  R9 Cross-Game Cell: target=${TARGET}" | tee -a "${LOG_FILE}"
echo "══════════════════════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "  RUN_DIR:        ${RUN_DIR}" | tee -a "${LOG_FILE}"
echo "  Adapter:        ${ADAPTER_PATH}" | tee -a "${LOG_FILE}"
echo "  Donor bank:     ${DONOR_BANK}  ($(wc -l < "${DONOR_BANK}") skills)" | tee -a "${LOG_FILE}"
echo "  Eval module:    ${EVAL_MODULE}" | tee -a "${LOG_FILE}"
echo "  Episodes:       ${EPISODES} (multiplayer=${IS_MULTIPLAYER})" | tee -a "${LOG_FILE}"
echo "  Temperature:    ${TEMP}" | tee -a "${LOG_FILE}"
echo "  GPU:            ${EVAL_GPUS}, vLLM port ${VLLM_PORT}" | tee -a "${LOG_FILE}"
echo "  Output:         ${EVAL_OUT_DIR}" | tee -a "${LOG_FILE}"
[ ${IS_MULTIPLAYER} -eq 1 ] && echo "  Opponent:       ${OPPONENT_MODEL}  (OPENROUTER_API_KEY=${OPENROUTER_API_KEY:+set}${OPENROUTER_API_KEY:-UNSET})" | tee -a "${LOG_FILE}"
echo "══════════════════════════════════════════════════════════════" | tee -a "${LOG_FILE}"

# ── Launch vLLM ───────────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    set +e
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[r9-cell:${TARGET}] tearing down vLLM (PID ${VLLM_PID})..." | tee -a "${LOG_FILE}"
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

echo "[r9-cell:${TARGET}] launching vLLM..." | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
    python -m vllm.entrypoints.openai.api_server \
        --model "${BASE_MODEL}" \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        --tensor-parallel-size "${TENSOR_PARALLEL}" \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --dtype auto \
        --trust-remote-code \
        --enable-lora \
        --lora-modules "${LORA_NAME}=${ADAPTER_PATH}" \
        --max-lora-rank 16 \
        >> "${CELL_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!

MAX_WAIT=600
WAITED=0
while [ ${WAITED} -lt ${MAX_WAIT} ]; do
    if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "[r9-cell:${TARGET}] vLLM ready (waited ${WAITED}s)" | tee -a "${LOG_FILE}"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[r9-cell:${TARGET}] ERROR: vLLM exited unexpectedly" | tee -a "${LOG_FILE}"
        tail -20 "${CELL_DIR}/vllm.log" | tee -a "${LOG_FILE}"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
if [ ${WAITED} -ge ${MAX_WAIT} ]; then
    echo "[r9-cell:${TARGET}] ERROR: vLLM did not become healthy in ${MAX_WAIT}s" | tee -a "${LOG_FILE}"
    exit 1
fi

# ── Build eval command ────────────────────────────────────────────────
EVAL_START=$(date +%s)
mkdir -p "${EVAL_OUT_DIR}"

EXIT_CODE=0
if [ ${IS_MULTIPLAYER} -eq 0 ]; then
    EVAL_ARGS=(
        --games "${TARGET}"
        --episodes "${EPISODES}"
        --max_steps "${MAX_STEPS}"
        --temperature "${TEMP}"
        --model "${LORA_NAME}"
        --seed "${SEED}"
        --output_dir "${EVAL_OUT_DIR}"
        --bank "${DONOR_BANK}"
    )
    if [ "${TETRIS_MACRO_ACTIONS:-0}" -eq 1 ]; then
        EVAL_ARGS+=(--macro-actions)
    fi
    # super_mario needs orak-mario env (game-ai-agent lacks gym + gym_super_mario_bros)
    if [ "${SUPER_MARIO_USE_ORAK_PY:-0}" -eq 1 ]; then
        EVAL_PYTHON="${ORAK_PYTHON}"
    else
        EVAL_PYTHON="python"
    fi
    echo "[r9-cell:${TARGET}] running: ${EVAL_PYTHON} -m ${EVAL_MODULE} ${EVAL_ARGS[*]}" | tee -a "${LOG_FILE}"
    "${EVAL_PYTHON}" -m "${EVAL_MODULE}" "${EVAL_ARGS[@]}" >> "${LOG_FILE}" 2>&1 || EXIT_CODE=$?
elif [ "${TARGET}" = "avalon" ]; then
    EVAL_ARGS=(
        --model "${LORA_NAME}"
        --episodes "${EPISODES}"
        --temperature "${TEMP}"
        --seed "${SEED}"
        --opponent_model "${OPPONENT_MODEL}"
        --bank "${DONOR_BANK}"
        --output_dir "${EVAL_OUT_DIR}"
        --verbose
    )
    echo "[r9-cell:${TARGET}] running: python -m ${EVAL_MODULE} ${EVAL_ARGS[*]}" | tee -a "${LOG_FILE}"
    python -m "${EVAL_MODULE}" "${EVAL_ARGS[@]}" >> "${LOG_FILE}" 2>&1 || EXIT_CODE=$?
elif [ "${TARGET}" = "diplomacy" ]; then
    EVAL_ARGS=(
        --model "${LORA_NAME}"
        --opponent_model "${OPPONENT_MODEL}"
        --episodes "${EPISODES}"
        --temperature "${TEMP}"
        --seed "${SEED}"
        --bank "${DONOR_BANK}"
        --unchosen_strategy hold
        --output_dir "${EVAL_OUT_DIR}"
        --verbose
    )
    echo "[r9-cell:${TARGET}] running: python -m ${EVAL_MODULE} ${EVAL_ARGS[*]}" | tee -a "${LOG_FILE}"
    python -m "${EVAL_MODULE}" "${EVAL_ARGS[@]}" >> "${LOG_FILE}" 2>&1 || EXIT_CODE=$?
fi
EVAL_WALL=$(( $(date +%s) - EVAL_START ))

# ── Write metadata ────────────────────────────────────────────────────
cat > "${CELL_DIR}/cell_meta.json" <<EOF
{
  "experiment": "R9_cross_game_transfer",
  "target_game": "${TARGET}",
  "donor_bank": "${DONOR_BANK}",
  "donor_n_skills": $(wc -l < "${DONOR_BANK}"),
  "adapter_path": "${ADAPTER_PATH}",
  "eval_module": "${EVAL_MODULE}",
  "episodes": ${EPISODES},
  "is_multiplayer": ${IS_MULTIPLAYER},
  "opponent_model": "$( [ ${IS_MULTIPLAYER} -eq 1 ] && echo "${OPPONENT_MODEL}" || echo "n/a" )",
  "temperature": ${TEMP},
  "seed": ${SEED},
  "eval_wall_s": ${EVAL_WALL},
  "exit_code": ${EXIT_CODE},
  "timestamp": "$(date -u +%FT%TZ)"
}
EOF

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[r9-cell:${TARGET}] ✓ COMPLETE (eval took ${EVAL_WALL}s)" | tee -a "${LOG_FILE}"
else
    echo "[r9-cell:${TARGET}] ✗ FAILED (exit ${EXIT_CODE}, eval took ${EVAL_WALL}s)" | tee -a "${LOG_FILE}"
fi

exit ${EXIT_CODE}
