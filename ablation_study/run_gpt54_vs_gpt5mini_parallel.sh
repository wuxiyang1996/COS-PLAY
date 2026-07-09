#!/usr/bin/env bash
# ======================================================================
#  PARALLEL evaluation: GPT-5.4 (focal) vs gpt-5-mini (opponents)
#  on Avalon and Diplomacy, sharding by episodes across workers.
#
#  Why parallel:
#    The single-process run was ~5.4 min/ep Avalon and ~13.5 min/ep
#    Diplomacy (50ep → 4.5hr, 70ep → 16hr) because focal is now an
#    API call (gpt-5.4) rather than local 8B vLLM.
#  Strategy:
#    - Split episodes across N workers (different --seed per worker).
#    - Each worker writes to its own output dir; aggregator merges.
#    - Avalon's --per_role still rotates focal through 5 players per
#      worker; total role coverage = (eps/worker) per role per worker.
#    - Diplomacy's --per_power rotates focal through 7 powers per worker.
#
#  Env knobs:
#    AVALON_WORKERS (default 5)   AVALON_EPS_PER_WORKER (default 10)
#    DIPLO_WORKERS  (default 7)   DIPLO_EPS_PER_WORKER  (default 10)
#    FOCAL_MODEL     (default gpt-5.4)
#    OPPONENT_MODEL  (default gpt-5-mini)
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
    echo "[parallel-gpt54-vs-gpt5mini] ERROR: OPENROUTER_API_KEY not set." >&2
    exit 1
fi

export RAG_EMBEDDER_DEVICE="${RAG_EMBEDDER_DEVICE:-cpu}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PYTHONPATH:-}"
# Make python stdout/stderr unbuffered so we get live progress in the worker logs
export PYTHONUNBUFFERED=1

FOCAL="${FOCAL_MODEL:-gpt-5.4}"
OPP="${OPPONENT_MODEL:-gpt-5-mini}"

AVALON_WORKERS="${AVALON_WORKERS:-5}"
AVALON_EPS_PER_WORKER="${AVALON_EPS_PER_WORKER:-10}"
DIPLO_WORKERS="${DIPLO_WORKERS:-7}"
DIPLO_EPS_PER_WORKER="${DIPLO_EPS_PER_WORKER:-10}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs/gpt54_vs_gpt5mini_eval/parallel_${TIMESTAMP}}"
mkdir -p "${LOG_ROOT}"

PIDS_FILE="${LOG_ROOT}/PIDS.txt"
SUMMARY="${LOG_ROOT}/SUMMARY.txt"
: > "${PIDS_FILE}"
: > "${SUMMARY}"

echo "════════════════════════════════════════════════════════════════"
echo "  PARALLEL ${FOCAL} (focal) vs ${OPP} (opponents)"
echo "  Avalon:    ${AVALON_WORKERS} workers × ${AVALON_EPS_PER_WORKER} eps = $((AVALON_WORKERS * AVALON_EPS_PER_WORKER)) eps"
echo "  Diplomacy: ${DIPLO_WORKERS} workers × ${DIPLO_EPS_PER_WORKER} eps = $((DIPLO_WORKERS * DIPLO_EPS_PER_WORKER)) eps"
echo "  Log root:           ${LOG_ROOT}"
echo "════════════════════════════════════════════════════════════════"

launch_worker() {
    local label="$1"; shift
    local logfile="${LOG_ROOT}/${label}.log"
    local outdir="${LOG_ROOT}/${label}/output"
    mkdir -p "${outdir}"
    : > "${logfile}"

    echo "  cmd: $*" | tee -a "${logfile}"
    setsid nohup bash -c "$* --output_dir ${outdir}" </dev/null >>"${logfile}" 2>&1 &
    local pid=$!
    echo "${label} ${pid}" >> "${PIDS_FILE}"
    disown ${pid} 2>/dev/null || true
}

# Avalon workers
for w in $(seq 0 $((AVALON_WORKERS - 1))); do
    SEED=$((42 + w * 1000))
    launch_worker "avalon_w${w}" \
        python -m scripts.run_qwen3_avalon_matched \
            --model "${FOCAL}" \
            --opponent_model "${OPP}" \
            --episodes "${AVALON_EPS_PER_WORKER}" \
            --temperature 0.3 \
            --per_role \
            --no-bank \
            --seed "${SEED}"
done

# Diplomacy workers
for w in $(seq 0 $((DIPLO_WORKERS - 1))); do
    SEED=$((42 + w * 1000))
    launch_worker "diplomacy_w${w}" \
        python -m scripts.run_diplomacy_discrete_eval \
            --model "${FOCAL}" \
            --opponent_model "${OPP}" \
            --episodes "${DIPLO_EPS_PER_WORKER}" \
            --per_power \
            --no-bank \
            --seed "${SEED}"
done

echo ""
echo "── Worker PIDs (${LOG_ROOT}/PIDS.txt) ──"
cat "${PIDS_FILE}"
echo ""
echo "Workers detached (setsid). To monitor:"
echo "  tail -f ${LOG_ROOT}/*.log"
echo "  ls ${LOG_ROOT}/*/output/{avalon,diplomacy}/*/episode_*.json | wc -l"
