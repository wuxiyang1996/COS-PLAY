#!/usr/bin/env bash
# Parallel dispatcher for labeling/label_intentions_gpt54.py.
#
# Fans out one Python worker per (corpus, env_or_game) bucket — gym-v
# Temporal_*-v0 envs go to the gym_v vocabulary, env_wrappers games go
# to the legacy SUBGOAL_TAGS vocabulary.  Each worker writes its
# labelled episodes under
# ``${OUTPUT_DIR}/<corpus>/<bucket>/episode_*.json`` and a per-bucket
# ``_intentions_summary.json``.  After every worker finishes the
# script prints a roll-up of LLM / rule / fallback counts per bucket.
#
# Usage (defaults pick up the most recent gpt-5.4 cold-start runs):
#
#   bash labeling/run_label_intentions.sh
#
#   bash labeling/run_label_intentions.sh \
#        --gymv_input  Cold-start-out-gymv/<run_dir> \
#        --envw_input  Cold-start-out/<run_dir> \
#        --output_dir  labeling/intentions_out/<my_run> \
#        --model gpt-5.4 \
#        --parallel 6 --workers 8
#
#   # Single env smoke test
#   bash labeling/run_label_intentions.sh \
#        --envs Temporal_Airstriker-v0 \
#        --max_episodes 1 --workers 4 -v

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
GYMV_INPUT="${REPO_ROOT}/Cold-start-out-gymv/sft_gpt5p4_e20_s100_stream_20260429_080127"
ENVW_INPUT="${REPO_ROOT}/Cold-start-out/sft_envw_e20_gpt5p4_20260429_080916"
OUTPUT_DIR="${REPO_ROOT}/labeling/intentions_out/run_$(date '+%Y%m%d_%H%M%S')"
MODEL="gpt-5.4"
PARALLEL="${PARALLEL:-4}"
WORKERS="${WORKERS:-8}"
MAX_EPISODES=""
RESUME=""
DRY_RUN=""
VERBOSE=""
ENVS=()
GAMES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gymv_input)  GYMV_INPUT="$2"; shift 2 ;;
        --envw_input)  ENVW_INPUT="$2"; shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --parallel)    PARALLEL="$2"; shift 2 ;;
        --workers)     WORKERS="$2"; shift 2 ;;
        --max_episodes) MAX_EPISODES="$2"; shift 2 ;;
        --envs)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ENVS+=("$1"); shift
            done
            ;;
        --games)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                GAMES+=("$1"); shift
            done
            ;;
        --resume)      RESUME="--resume"; shift ;;
        --dry_run|--dry-run) DRY_RUN="--dry_run"; shift ;;
        -v|--verbose)  VERBOSE="-v"; shift ;;
        --no_gymv|--no-gymv) GYMV_INPUT=""; shift ;;
        --no_envw|--no-envw) ENVW_INPUT=""; shift ;;
        -h|--help)     sed -n '1,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/_dispatch_logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date '+%Y%m%d_%H%M%S')"
DISPATCH_LOG="${OUTPUT_DIR}/_dispatch_${STAMP}.log"

# ---------------------------------------------------------------------------
# Env / paths
# ---------------------------------------------------------------------------
export PYTHONPATH="${REPO_ROOT}:${WORKSPACE_ROOT}:${WORKSPACE_ROOT}/GamingAgent:${PYTHONPATH:-}"

# Pull api keys from /workspace/api_keys.py if not already in the environment.
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
# Discover buckets
# ---------------------------------------------------------------------------
declare -a BUCKETS=()       # "gym_v|<env>|<input_root>" or "env_wrappers|<game>|<input_root>|<group>"

if [[ -n "${GYMV_INPUT}" && -d "${GYMV_INPUT}" ]]; then
    if [[ ${#ENVS[@]} -gt 0 ]]; then
        for e in "${ENVS[@]}"; do
            if [[ -d "${GYMV_INPUT}/${e}" ]]; then
                BUCKETS+=("gym_v|${e}|${GYMV_INPUT}")
            fi
        done
    else
        while IFS= read -r d; do
            BUCKETS+=("gym_v|$(basename "$d")|${GYMV_INPUT}")
        done < <(find "${GYMV_INPUT}" -maxdepth 1 -type d -name 'Temporal_*' | sort)
    fi
fi

if [[ -n "${ENVW_INPUT}" && -d "${ENVW_INPUT}" ]]; then
    while IFS= read -r game_dir; do
        local_game="$(basename "${game_dir}")"
        if [[ ${#GAMES[@]} -gt 0 ]]; then
            keep=0
            for g in "${GAMES[@]}"; do
                [[ "${g}" == "${local_game}" ]] && keep=1 && break
            done
            [[ ${keep} -eq 0 ]] && continue
        fi
        BUCKETS+=("env_wrappers|${local_game}|${ENVW_INPUT}|${game_dir}")
    done < <(
        find "${ENVW_INPUT}" -mindepth 2 -maxdepth 2 -type d \
            -not -path '*/_logs*' \
            | sort
    )
fi

if [[ ${#BUCKETS[@]} -eq 0 ]]; then
    echo "[ERROR] No buckets to label.  Check --gymv_input / --envw_input." >&2
    exit 2
fi

echo "============================================================" | tee -a "${DISPATCH_LOG}"
echo "  labeling/run_label_intentions: dispatcher"                  | tee -a "${DISPATCH_LOG}"
echo "============================================================" | tee -a "${DISPATCH_LOG}"
echo "  GYM-V input  : ${GYMV_INPUT:-(skipped)}"                    | tee -a "${DISPATCH_LOG}"
echo "  ENV-W input  : ${ENVW_INPUT:-(skipped)}"                    | tee -a "${DISPATCH_LOG}"
echo "  Output dir   : ${OUTPUT_DIR}"                               | tee -a "${DISPATCH_LOG}"
echo "  Model        : ${MODEL}"                                    | tee -a "${DISPATCH_LOG}"
echo "  Parallel     : ${PARALLEL}"                                 | tee -a "${DISPATCH_LOG}"
echo "  Workers/ep   : ${WORKERS}"                                  | tee -a "${DISPATCH_LOG}"
echo "  Resume       : ${RESUME:-(off)}"                            | tee -a "${DISPATCH_LOG}"
echo "  Max eps      : ${MAX_EPISODES:-(all)}"                      | tee -a "${DISPATCH_LOG}"
echo "  Buckets      : ${#BUCKETS[@]}"                              | tee -a "${DISPATCH_LOG}"
for b in "${BUCKETS[@]}"; do echo "    - ${b}" | tee -a "${DISPATCH_LOG}"; done
echo                                                                 | tee -a "${DISPATCH_LOG}"

if [[ -n "${DRY_RUN}" ]]; then
    echo "[DRY RUN] launching workers with --dry_run." | tee -a "${DISPATCH_LOG}"
fi

# ---------------------------------------------------------------------------
# Concurrency-capped launch
# ---------------------------------------------------------------------------
declare -a PIDS=()
declare -a NAMES=()
FAILED=0

wait_for_next_bucket() {
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
        echo "  [FAIL] ${name}  (rc=${rc})  log=${LOG_DIR}/${name//\//_}.log" | tee -a "${DISPATCH_LOG}"
    fi
}

launch_bucket() {
    local spec="$1"
    local IFS='|'
    read -ra parts <<< "${spec}"
    local corpus="${parts[0]}"
    local bucket="${parts[1]}"
    local input_root="${parts[2]}"

    local name="${corpus}_${bucket}"
    local log="${LOG_DIR}/${name}.log"

    local cmd=(
        python3 -u "${SCRIPT_DIR}/label_intentions_gpt54.py"
        --output_dir "${OUTPUT_DIR}"
        --model "${MODEL}"
        --workers "${WORKERS}"
    )
    if [[ "${corpus}" == "gym_v" ]]; then
        cmd+=(--gymv_input "${input_root}" --envs "${bucket}")
    else
        # env_wrappers — point at the run root, then filter to this game.
        cmd+=(--envw_input "${input_root}" --games "${bucket}")
    fi
    [[ -n "${MAX_EPISODES}" ]] && cmd+=(--max_episodes "${MAX_EPISODES}")
    [[ -n "${RESUME}" ]] && cmd+=("${RESUME}")
    [[ -n "${DRY_RUN}" ]] && cmd+=("${DRY_RUN}")
    [[ -n "${VERBOSE}" ]] && cmd+=("${VERBOSE}")

    {
        echo "===== ${name} | $(date) ====="
        echo "CMD: ${cmd[*]}"
        echo
    } > "${log}"
    "${cmd[@]}" >> "${log}" 2>&1 &
    PIDS+=("$!")
    NAMES+=("${name}")
    echo "  [START] ${name}  pid=$!  log=${log}" | tee -a "${DISPATCH_LOG}"
}

T0=$(date +%s)
for spec in "${BUCKETS[@]}"; do
    while [[ ${#PIDS[@]} -ge ${PARALLEL} ]]; do
        wait_for_next_bucket
    done
    launch_bucket "${spec}"
done

while [[ ${#PIDS[@]} -gt 0 ]]; do
    wait_for_next_bucket
done

T1=$(date +%s)

# ---------------------------------------------------------------------------
# Roll-up summary
# ---------------------------------------------------------------------------
echo                                                                 | tee -a "${DISPATCH_LOG}"
echo "============================================================" | tee -a "${DISPATCH_LOG}"
echo "  Intention labelling — DONE"                                  | tee -a "${DISPATCH_LOG}"
echo "============================================================" | tee -a "${DISPATCH_LOG}"
echo "  Buckets      : ${#BUCKETS[@]}"                               | tee -a "${DISPATCH_LOG}"
echo "  Failed       : ${FAILED}"                                    | tee -a "${DISPATCH_LOG}"
echo "  Elapsed      : $((T1 - T0))s"                                | tee -a "${DISPATCH_LOG}"
echo "  Output       : ${OUTPUT_DIR}"                                | tee -a "${DISPATCH_LOG}"
echo "  Dispatch log : ${DISPATCH_LOG}"                              | tee -a "${DISPATCH_LOG}"

if command -v python3 >/dev/null 2>&1; then
    python3 - "${OUTPUT_DIR}" <<'PY' | tee -a "${DISPATCH_LOG}" || true
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
print()
print("  Per-bucket roll-up:")
for f in sorted(root.glob("*/*/_intentions_summary.json")):
    s = json.load(open(f))
    counts = s.get("source_counts", {})
    dist = s.get("tag_distribution", {})
    top = ",".join(f"{k}:{v}" for k, v in list(dist.items())[:4])
    print(f"    {s['corpus']:>11} / {s['bucket']:<28} "
          f"steps={s['step_count_total']:>5}  "
          f"llm={counts.get('llm', 0):>4}  "
          f"rule={counts.get('rule_classifier', 0):>4}  "
          f"fallback={counts.get('fallback_default', 0):>4}  "
          f"top={top}")
PY
fi

exit $(( FAILED > 0 ? 1 : 0 ))
