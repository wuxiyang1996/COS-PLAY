#!/usr/bin/env bash
# ======================================================================
#  Parallel evaluation: GPT-5.4 (focal) vs gpt-5-mini (opponents)
#  on Avalon (50 ep, per_role) and Diplomacy (70 ep, per_power).
#
#  Goal: verify that paper Table 1's GPT-5.4 baseline benefits from
#  the in-distribution (weaker) opponent setting — quantifies the
#  opponent-distribution shift highlighted by vJ13 Q6/Reject 7 and
#  Fs6X #2 in the rebuttal.
#
#  All API calls go through OpenRouter — no vLLM, no GPU required.
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
    echo "[gpt54-vs-gpt5mini] ERROR: OPENROUTER_API_KEY not set." >&2
    exit 1
fi

export RAG_EMBEDDER_DEVICE="${RAG_EMBEDDER_DEVICE:-cpu}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PYTHONPATH:-}"

FOCAL="${FOCAL_MODEL:-gpt-5.4}"
OPP="${OPPONENT_MODEL:-gpt-5-mini}"
AVALON_EPISODES="${AVALON_EPISODES:-50}"
DIPLO_EPISODES="${DIPLO_EPISODES:-70}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs/gpt54_vs_gpt5mini_eval/parallel_${TIMESTAMP}}"
mkdir -p "${LOG_ROOT}"

PIDS_FILE="${LOG_ROOT}/PIDS.txt"
SUMMARY="${LOG_ROOT}/SUMMARY.txt"
: > "${PIDS_FILE}"
: > "${SUMMARY}"

echo "════════════════════════════════════════════════════════════════"
echo "  ${FOCAL} (focal) vs ${OPP} (opponents)"
echo "  Avalon episodes:    ${AVALON_EPISODES}  (per-role rotation)"
echo "  Diplomacy episodes: ${DIPLO_EPISODES}  (per-power rotation)"
echo "  Log root:           ${LOG_ROOT}"
echo "════════════════════════════════════════════════════════════════"

declare -A GAME_PID
launch_run() {
    local label="$1"; shift
    local logfile="${LOG_ROOT}/${label}.log"
    local outdir="${LOG_ROOT}/${label}/output"
    mkdir -p "${outdir}"
    : > "${logfile}"

    echo "[gpt54-vs-gpt5mini] launching ${label}"
    echo "  cmd: $*" | tee -a "${logfile}"

    setsid nohup bash -c "$* --output_dir ${outdir}" </dev/null >>"${logfile}" 2>&1 &
    local pid=$!
    GAME_PID[${label}]=${pid}
    echo "${label} ${pid}" >> "${PIDS_FILE}"
    disown ${pid} 2>/dev/null || true
}

# Avalon — gpt-5.4 (focal) vs gpt-5-mini (opp), no skill bank
launch_run avalon \
    python -m scripts.run_qwen3_avalon_matched \
        --model "${FOCAL}" \
        --opponent_model "${OPP}" \
        --episodes "${AVALON_EPISODES}" \
        --temperature 0.3 \
        --per_role \
        --no-bank

# Diplomacy — gpt-5.4 (focal) vs gpt-5-mini (opp), no skill bank
launch_run diplomacy \
    python -m scripts.run_diplomacy_discrete_eval \
        --model "${FOCAL}" \
        --opponent_model "${OPP}" \
        --episodes "${DIPLO_EPISODES}" \
        --per_power \
        --no-bank

echo ""
echo "── Worker PIDs ──"
for l in "${!GAME_PID[@]}"; do
    printf "  %-22s  PID=%s\n" "${l}" "${GAME_PID[${l}]}"
done | sort
echo ""

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
