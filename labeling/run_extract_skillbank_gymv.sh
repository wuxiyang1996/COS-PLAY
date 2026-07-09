#!/usr/bin/env bash
# Parallel dispatcher for the gym-v Skill Bank cold-start extractor.
#
# Each Temporal_<env>-v0 folder under --input_dir is processed by its own
# extract_skillbank_gymv_gpt54.py invocation (with --envs <one>) so the
# per-env pipelines run concurrently. Stdout/stderr for each env streams
# to a dedicated logfile and `wait_for_next_env` keeps at most $PARALLEL
# workers alive at any time. The outer script blocks until every env has
# finished.
#
# Usage (defaults pick up the completed gpt-5.4 stream run):
#
#   bash labeling/run_extract_skillbank_gymv.sh
#
#   bash labeling/run_extract_skillbank_gymv.sh \
#        --input_dir Cold-start-out-gymv/<run_dir> \
#        --output_dir skill_bank_sft \
#        --model gpt-5.4 \
#        --parallel 6
#
#   # Resume a partially-finished run
#   bash labeling/run_extract_skillbank_gymv.sh --resume
#
#   # Smoke-test on a single env / single episode
#   bash labeling/run_extract_skillbank_gymv.sh \
#        --envs Temporal_Airstriker-v0 --max_episodes 1 -v

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
INPUT_DIR="${REPO_ROOT}/Cold-start-out-gymv/sft_gpt5p4_e20_s100_stream_20260429_080127"
OUTPUT_DIR="${REPO_ROOT}/skill_bank_sft"
MODEL="gpt-5.4"
PARALLEL="${PARALLEL:-4}"
MAX_EPISODES=""
RESUME=""
DRY_RUN=""
VERBOSE=""
CACHE_NORM="--cache_normalized"
ENVS=()

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir) INPUT_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --max_episodes) MAX_EPISODES="$2"; shift 2 ;;
        --envs)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ENVS+=("$1"); shift
            done
            ;;
        --resume) RESUME="--resume"; shift ;;
        --no-cache-norm) CACHE_NORM=""; shift ;;
        --dry_run|--dry-run) DRY_RUN="--dry_run"; shift ;;
        -v|--verbose) VERBOSE="-v"; shift ;;
        -h|--help)
            sed -n '1,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "[ERROR] Input dir not found: ${INPUT_DIR}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/_dispatch_logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date '+%Y%m%d_%H%M%S')"
DISPATCH_LOG="${OUTPUT_DIR}/_dispatch_${STAMP}.log"

# ---------------------------------------------------------------------------
# Env / paths
# ---------------------------------------------------------------------------
# PYTHONPATH for the Python workers + the inline api-key bootstrap below.
# WORKSPACE_ROOT is included so ``import api_keys`` resolves to the
# repo-level ``/workspace/api_keys.py``.
export PYTHONPATH="${REPO_ROOT}:${WORKSPACE_ROOT}:${WORKSPACE_ROOT}/GamingAgent:${PYTHONPATH:-}"

# Pull api keys from /workspace/api_keys.py if not already set.
if [[ -z "${OPENROUTER_API_KEY:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
    if [[ -f "${WORKSPACE_ROOT}/api_keys.py" ]]; then
        eval "$(WORKSPACE_ROOT="${WORKSPACE_ROOT}" python3 - <<'PY'
import os, sys, shlex
sys.path.insert(0, os.environ["WORKSPACE_ROOT"])
import api_keys as a
or_k = getattr(a, "openrouter_api_key", "") or ""
op_k = getattr(a, "openai_api_key", "") or ""
if or_k and not os.environ.get("OPENROUTER_API_KEY"):
    print(f"export OPENROUTER_API_KEY={shlex.quote(or_k)}")
if op_k and not os.environ.get("OPENAI_API_KEY"):
    print(f"export OPENAI_API_KEY={shlex.quote(op_k)}")
PY
        )"
    fi
fi

if [[ -z "${OPENROUTER_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[ERROR] No OPENAI_API_KEY or OPENROUTER_API_KEY available." >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# Discover envs
# ---------------------------------------------------------------------------
if [[ ${#ENVS[@]} -eq 0 ]]; then
    while IFS= read -r d; do
        ENVS+=("$(basename "$d")")
    done < <(find "${INPUT_DIR}" -maxdepth 1 -type d -name 'Temporal_*' | sort)
fi

if [[ ${#ENVS[@]} -eq 0 ]]; then
    echo "[ERROR] No Temporal_* env folders under ${INPUT_DIR}" >&2
    exit 2
fi

echo "============================================================"  | tee -a "${DISPATCH_LOG}"
echo "  labeling/run_extract_skillbank_gymv: dispatcher"             | tee -a "${DISPATCH_LOG}"
echo "============================================================"  | tee -a "${DISPATCH_LOG}"
echo "  Input dir   : ${INPUT_DIR}"                                  | tee -a "${DISPATCH_LOG}"
echo "  Output dir  : ${OUTPUT_DIR}"                                 | tee -a "${DISPATCH_LOG}"
echo "  Model       : ${MODEL}"                                      | tee -a "${DISPATCH_LOG}"
echo "  Parallel    : ${PARALLEL}"                                   | tee -a "${DISPATCH_LOG}"
echo "  Resume      : ${RESUME:-(off)}"                              | tee -a "${DISPATCH_LOG}"
echo "  Max eps     : ${MAX_EPISODES:-(all)}"                        | tee -a "${DISPATCH_LOG}"
echo "  Envs        : ${#ENVS[@]}"                                   | tee -a "${DISPATCH_LOG}"
for e in "${ENVS[@]}"; do echo "    - ${e}" | tee -a "${DISPATCH_LOG}"; done
echo                                                                 | tee -a "${DISPATCH_LOG}"

if [[ -n "${DRY_RUN}" ]]; then
    echo "[DRY RUN] not launching workers." | tee -a "${DISPATCH_LOG}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Concurrency-capped launch
# ---------------------------------------------------------------------------
declare -a PIDS=()
declare -a NAMES=()
FAILED=0

wait_for_next_env() {
    if [[ ${#PIDS[@]} -eq 0 ]]; then return; fi
    local pid="${PIDS[0]}"
    local name="${NAMES[0]}"
    PIDS=("${PIDS[@]:1}")
    NAMES=("${NAMES[@]:1}")
    if wait "${pid}"; then
        echo "  [OK]   ${name}" | tee -a "${DISPATCH_LOG}"
    else
        local rc=$?
        FAILED=$((FAILED + 1))
        echo "  [FAIL] ${name}  (rc=${rc})  log=${LOG_DIR}/${name}.log" | tee -a "${DISPATCH_LOG}"
    fi
}

launch_env() {
    local env="$1"
    local log="${LOG_DIR}/${env}.log"
    local cmd=(
        python3 -u "${SCRIPT_DIR}/extract_skillbank_gymv_gpt54.py"
        --input_dir "${INPUT_DIR}"
        --output_dir "${OUTPUT_DIR}"
        --model "${MODEL}"
        --envs "${env}"
    )
    [[ -n "${MAX_EPISODES}" ]] && cmd+=(--max_episodes "${MAX_EPISODES}")
    [[ -n "${RESUME}" ]] && cmd+=("${RESUME}")
    [[ -n "${VERBOSE}" ]] && cmd+=("${VERBOSE}")
    [[ -n "${CACHE_NORM}" ]] && cmd+=("${CACHE_NORM}")

    {
        echo "===== ${env} | $(date) ====="
        echo "CMD: ${cmd[*]}"
        echo
    } > "${log}"
    "${cmd[@]}" >> "${log}" 2>&1 &
    PIDS+=("$!")
    NAMES+=("${env}")
    echo "  [START] ${env}  pid=$!  log=${log}" | tee -a "${DISPATCH_LOG}"
}

T0=$(date +%s)
for env in "${ENVS[@]}"; do
    while [[ ${#PIDS[@]} -ge ${PARALLEL} ]]; do
        wait_for_next_env
    done
    launch_env "${env}"
done

while [[ ${#PIDS[@]} -gt 0 ]]; do
    wait_for_next_env
done

# ---------------------------------------------------------------------------
# Cross-env unified index. Each per-env worker only sees its own slice of
# OUTPUT_DIR, so the union step runs once here after every worker exits.
# This produces ${OUTPUT_DIR}/_unified/ with skill_index.jsonl,
# skill_catalog_all.json, and skill_rag_index.json.
# ---------------------------------------------------------------------------
echo                                                                 | tee -a "${DISPATCH_LOG}"
echo "  [unify] aggregating cross-env skill index ..."               | tee -a "${DISPATCH_LOG}"
if python3 -u "${SCRIPT_DIR}/unify_skill_index.py" \
        --root "${OUTPUT_DIR}" --output_dir "${OUTPUT_DIR}" \
        ${VERBOSE} >> "${DISPATCH_LOG}" 2>&1; then
    echo "  [unify] OK  -> ${OUTPUT_DIR}/_unified/"                  | tee -a "${DISPATCH_LOG}"
else
    echo "  [unify] FAILED  (see ${DISPATCH_LOG})"                   | tee -a "${DISPATCH_LOG}"
    FAILED=$((FAILED + 1))
fi

T1=$(date +%s)
echo                                                                 | tee -a "${DISPATCH_LOG}"
echo "============================================================" | tee -a "${DISPATCH_LOG}"
echo "  Skill Bank dispatcher — DONE"                                | tee -a "${DISPATCH_LOG}"
echo "============================================================" | tee -a "${DISPATCH_LOG}"
echo "  Envs        : ${#ENVS[@]}"                                   | tee -a "${DISPATCH_LOG}"
echo "  Failed      : ${FAILED}"                                     | tee -a "${DISPATCH_LOG}"
echo "  Elapsed     : $((T1 - T0))s"                                 | tee -a "${DISPATCH_LOG}"
echo "  Output      : ${OUTPUT_DIR}"                                 | tee -a "${DISPATCH_LOG}"
echo "  Dispatch log: ${DISPATCH_LOG}"                               | tee -a "${DISPATCH_LOG}"

exit $(( FAILED > 0 ? 1 : 0 ))
