#!/usr/bin/env bash
# ======================================================================
#  Parallel evaluation: 2 GRPO models vs gpt-5.4 on Avalon + Diplomacy.
#
#  Primary agent uses GRPO best-step LoRA adapters served by a local vLLM
#  instance on VLLM_PORT (default 8050).  Opponents call gpt-5.4 via
#  OpenRouter.
#
#  Adapters (must be pre-loaded into the vLLM server):
#    qwen3-8b-avalon-nb-best     (no-bank GRPO, step 7)
#    qwen3-8b-avalon-fb-best     (fixed-bank GRPO, step 3)
#    qwen3-8b-diplomacy-nb-best  (no-bank GRPO, step 7)
#    qwen3-8b-diplomacy-fb-best  (fixed-bank GRPO, step 7)
#
#  Defaults to 4 runs in parallel, 8 episodes each.
# ======================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV:-game-ai-agent}"
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]; then
    if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
        source /workspace/miniconda3/etc/profile.d/conda.sh
        conda activate "${CONDA_ENV}"
    fi
fi

load_key() {
    local var="$1" attr="$2"
    if [ -z "${!var:-}" ] && [ -f /workspace/keys.py ]; then
        local v
        v="$(python - <<PY
import importlib.util
spec = importlib.util.spec_from_file_location("keys", "/workspace/keys.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(getattr(m, "${attr}", ""))
PY
)"
        if [ -n "${v}" ]; then export "${var}=${v}"; fi
    fi
}
load_key OPENROUTER_API_KEY  openrouter_api_key
load_key OPENAI_API_KEY      openai_api_key

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[grpo-vs-gpt54] ERROR: OPENROUTER_API_KEY not set." >&2
    exit 1
fi

VLLM_PORT="${VLLM_PORT:-8050}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
export RAG_EMBEDDER_DEVICE="${RAG_EMBEDDER_DEVICE:-cpu}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PYTHONPATH:-}"

EPISODES="${EPISODES:-8}"
OPP="${OPPONENT_MODEL:-gpt-5.4}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs/grpo_vs_gpt54_eval/parallel_${TIMESTAMP}}"
mkdir -p "${LOG_ROOT}"

PIDS_FILE="${LOG_ROOT}/PIDS.txt"
SUMMARY="${LOG_ROOT}/SUMMARY.txt"
: > "${PIDS_FILE}"
: > "${SUMMARY}"

echo "════════════════════════════════════════════════════════════════"
echo "  GRPO × {no-bank, fixed-bank} × {Avalon, Diplomacy} vs ${OPP}"
echo "  Episodes per run:  ${EPISODES}"
echo "  VLLM base URL:     ${VLLM_BASE_URL}"
echo "  Log root:          ${LOG_ROOT}"
echo "════════════════════════════════════════════════════════════════"

declare -A GAME_PID
launch_run() {
    local label="$1"; shift
    local logfile="${LOG_ROOT}/${label}.log"
    local outdir="${LOG_ROOT}/${label}/output"
    mkdir -p "${outdir}"
    : > "${logfile}"

    echo "[grpo-vs-gpt54] launching ${label}"
    echo "  cmd: $*" | tee -a "${logfile}"

    setsid nohup bash -c "$* --output_dir ${outdir}" </dev/null >>"${logfile}" 2>&1 &
    local pid=$!
    GAME_PID[${label}]=${pid}
    echo "${label} ${pid}" >> "${PIDS_FILE}"
    disown ${pid} 2>/dev/null || true
}

# Avalon — no-bank GRPO vs gpt-5.4
# --per_role rotates the GRPO-controlled player slot across episodes so opponents
# are gpt-5.4.  Without it, mixed_model is disabled and ALL 5 players use --model.
launch_run avalon_no_bank \
    python -m scripts.run_qwen3_avalon_matched \
        --model qwen3-8b-avalon-nb-best \
        --opponent_model "${OPP}" \
        --episodes "${EPISODES}" \
        --temperature 0.3 \
        --per_role \
        --no-bank

# Avalon — fixed-bank GRPO vs gpt-5.4
launch_run avalon_fixed_bank \
    python -m scripts.run_qwen3_avalon_matched \
        --model qwen3-8b-avalon-fb-best \
        --opponent_model "${OPP}" \
        --episodes "${EPISODES}" \
        --temperature 0.3 \
        --per_role \
        --bank "${PROJECT_ROOT}/runs/fixed_skillbank_seeds/avalon/combined_skill_bank.jsonl"

# Diplomacy — no-bank GRPO vs gpt-5.4 (--per_power cycles AUSTRIA..TURKEY)
launch_run diplomacy_no_bank \
    python -m scripts.run_diplomacy_discrete_eval \
        --model qwen3-8b-diplomacy-nb-best \
        --opponent_model "${OPP}" \
        --episodes "${EPISODES}" \
        --per_power \
        --no-bank

# Diplomacy — fixed-bank GRPO vs gpt-5.4
launch_run diplomacy_fixed_bank \
    python -m scripts.run_diplomacy_discrete_eval \
        --model qwen3-8b-diplomacy-fb-best \
        --opponent_model "${OPP}" \
        --episodes "${EPISODES}" \
        --per_power \
        --bank "${PROJECT_ROOT}/runs/fixed_skillbank_seeds/diplomacy/combined_skill_bank.jsonl"

echo ""
echo "── Worker PIDs ──"
for l in "${!GAME_PID[@]}"; do
    printf "  %-22s  PID=%s\n" "${l}" "${GAME_PID[${l}]}"
done | sort
echo ""

# Wait for all
OVERALL_RC=0
for l in "${!GAME_PID[@]}"; do
    pid=${GAME_PID[${l}]}
    start_ts=$(stat -c %Y "${LOG_ROOT}/${l}.log" 2>/dev/null || date +%s)
    wait ${pid}
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    H=$(( elapsed / 3600 )); M=$(( (elapsed % 3600) / 60 )); S=$(( elapsed % 60 ))
    if [ ${rc} -eq 0 ]; then status="OK"; else status="FAIL(${rc})"; OVERALL_RC=${rc}; fi
    printf "%-22s  %-10s  %02d:%02d:%02d  PID=%s\n" \
        "${l}" "${status}" "${H}" "${M}" "${S}" "${pid}" | tee -a "${SUMMARY}"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Summary: ${SUMMARY}"
echo "════════════════════════════════════════════════════════════════"
cat "${SUMMARY}"
exit ${OVERALL_RC}
