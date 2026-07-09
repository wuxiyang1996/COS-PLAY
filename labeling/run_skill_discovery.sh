#!/usr/bin/env bash
# Unified parallel Skill-Discovery dispatcher.
#
# Runs the SkillBankAgent pipeline (SEGMENT / CONTRACT / CURATOR LoRAs)
# concurrently across:
#   - all 13 gym-v Temporal_*-v0 envs            (extract_skillbank_gymv_gpt54.py)
#   - all 4 env_wrappers games                  (extract_skillbank_gpt54.py)
#
# Inputs come from a *labelled* corpus produced by `run_label_intentions.sh`
# (i.e. ``labeling/intentions_out/run_dualaxis_<ts>/{gym_v,env_wrappers}/``)
# so each Experience already carries dual-axis ``[OPERATOR/SUBGOAL] note``
# tags and the segmenter sees a real categorical signal.
#
# Outputs are organised under a fresh per-run directory:
#
#   labeling/skill_bank_out/run_<ts>/
#   ├── _run_meta.json
#   ├── _dispatch_<ts>.log
#   ├── _dispatch_logs/<corpus>_<source>.log
#   ├── gym_v/<env>/...
#   ├── env_wrappers/<game>/...
#   ├── _decorator_summary.json   (added by _decorate_skill_records.py)
#   └── _unified/                  (added by unify_skill_index.py)
#
# Each env subfolder includes the standard SkillBankAgent traces:
#   * skill_bank.jsonl                — final per-env bank (decorated below)
#   * skill_catalog.json              — canonical catalog (decorated below)
#   * stage_io_log.json               — per-LoRA inputs/outputs (SEGMENT / CONTRACT / CURATOR)
#   * episode_snapshots/episode_<i>/  — per-episode bank snapshots
#   * coldstart_io_all.jsonl          — Stage-3/4 module I/O
#   * teacher_io_coldstart.jsonl      — Stage-2 SEGMENT teacher rankings
#   * llm_calls_log.json              — every gpt-5.4 prompt+response
#   * per_episode_bank_management/    — Stage-4 split/merge/refine logs
#   * reports/                        — verification reports
#
# After all per-env extractors finish, the dispatcher:
#   1. Decorates every skill_bank.jsonl + skill_catalog.json with the
#      SkillRecord-shape fields (status=draft, source_type=mined_from_trace,
#      applicable_domains=["gymv"], evidence_role from operator, ...) so a
#      future `lifecycle.ingest_draft` pass can promote them into
#      `draft_store/` without a destructive migration.
#   2. Aggregates skills cross-corpus via unify_skill_index.py.
#
# Usage:
#
#   bash labeling/run_skill_discovery.sh
#
#   # Custom inputs / output
#   bash labeling/run_skill_discovery.sh \
#        --intentions_dir labeling/intentions_out/run_dualaxis_20260429_224917 \
#        --output_root    labeling/skill_bank_out \
#        --parallel       8 \
#        --model          gpt-5.4
#
#   # Smoke test — one env per corpus, one episode each
#   bash labeling/run_skill_discovery.sh --smoke
#
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
INTENTIONS_DIR_DEFAULT="${REPO_ROOT}/labeling/intentions_out/run_dualaxis_20260429_224917"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/labeling/skill_bank_out"
COLD_START_GYMV_DEFAULT="${REPO_ROOT}/Cold-start-out-gymv/sft_gpt5p4_e20_s100_stream_20260429_080127"
COLD_START_ENVW_DEFAULT="${REPO_ROOT}/Cold-start-out/sft_envw_e20_gpt5p4_20260429_080916"

INTENTIONS_DIR="${INTENTIONS_DIR_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
COLD_START_GYMV="${COLD_START_GYMV_DEFAULT}"
COLD_START_ENVW="${COLD_START_ENVW_DEFAULT}"
MODEL="gpt-5.4"
PARALLEL="${PARALLEL:-8}"
MAX_EPISODES=""
ENVS_GYMV=()
GAMES_ENVW=()
SMOKE=""
SKIP_GYMV=""
SKIP_ENVW=""

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --intentions_dir) INTENTIONS_DIR="$2"; shift 2 ;;
        --output_root)    OUTPUT_ROOT="$2"; shift 2 ;;
        --cold_start_gymv) COLD_START_GYMV="$2"; shift 2 ;;
        --cold_start_envw) COLD_START_ENVW="$2"; shift 2 ;;
        --model)          MODEL="$2"; shift 2 ;;
        --parallel)       PARALLEL="$2"; shift 2 ;;
        --max_episodes)   MAX_EPISODES="$2"; shift 2 ;;
        --envs)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ENVS_GYMV+=("$1"); shift
            done ;;
        --games)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                GAMES_ENVW+=("$1"); shift
            done ;;
        --smoke)          SMOKE="1"; shift ;;
        --skip_gymv)      SKIP_GYMV="1"; shift ;;
        --skip_envw)      SKIP_ENVW="1"; shift ;;
        -h|--help)        sed -n '1,55p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -n "${SMOKE}" ]]; then
    PARALLEL=2
    MAX_EPISODES="1"
    if [[ ${#ENVS_GYMV[@]} -eq 0 ]]; then
        ENVS_GYMV=("Temporal_Airstriker-v0")
    fi
    if [[ ${#GAMES_ENVW[@]} -eq 0 ]]; then
        GAMES_ENVW=("tetris")
    fi
fi

if [[ ! -d "${INTENTIONS_DIR}" ]]; then
    echo "[ERROR] Intentions dir not found: ${INTENTIONS_DIR}" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${OUTPUT_ROOT}/run_${STAMP}"
mkdir -p "${RUN_DIR}"
LOG_DIR="${RUN_DIR}/_dispatch_logs"
mkdir -p "${LOG_DIR}"
DISPATCH_LOG="${RUN_DIR}/_dispatch_${STAMP}.log"

GYMV_OUT="${RUN_DIR}/gym_v"
ENVW_OUT="${RUN_DIR}/env_wrappers"
[[ -z "${SKIP_GYMV}" ]] && mkdir -p "${GYMV_OUT}"
[[ -z "${SKIP_ENVW}" ]] && mkdir -p "${ENVW_OUT}"

# ---------------------------------------------------------------------------
# Env / paths
# ---------------------------------------------------------------------------
export PYTHONPATH="${REPO_ROOT}:${WORKSPACE_ROOT}:${WORKSPACE_ROOT}/GamingAgent:${PYTHONPATH:-}"

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
# Discover envs / games
# ---------------------------------------------------------------------------
GYMV_INPUT="${INTENTIONS_DIR}/gym_v"
ENVW_INPUT="${INTENTIONS_DIR}/env_wrappers"

if [[ -z "${SKIP_GYMV}" ]]; then
    if [[ ${#ENVS_GYMV[@]} -eq 0 ]]; then
        if [[ -d "${GYMV_INPUT}" ]]; then
            while IFS= read -r d; do
                ENVS_GYMV+=("$(basename "$d")")
            done < <(find "${GYMV_INPUT}" -maxdepth 1 -type d -name 'Temporal_*' | sort)
        fi
    fi
fi

if [[ -z "${SKIP_ENVW}" ]]; then
    if [[ ${#GAMES_ENVW[@]} -eq 0 ]]; then
        if [[ -d "${ENVW_INPUT}" ]]; then
            while IFS= read -r d; do
                GAMES_ENVW+=("$(basename "$d")")
            done < <(find "${ENVW_INPUT}" -maxdepth 1 -type d ! -path "${ENVW_INPUT}" | sort)
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
{
    echo "============================================================"
    echo "  labeling/run_skill_discovery: dispatcher"
    echo "============================================================"
    echo "  Intentions dir : ${INTENTIONS_DIR}"
    echo "  Output run dir : ${RUN_DIR}"
    echo "  Model          : ${MODEL}"
    echo "  Parallel       : ${PARALLEL}"
    echo "  Max episodes   : ${MAX_EPISODES:-(all)}"
    echo "  gym_v envs     : ${#ENVS_GYMV[@]}"
    for e in "${ENVS_GYMV[@]:-}"; do echo "    - ${e}"; done
    echo "  env_wrappers   : ${#GAMES_ENVW[@]}"
    for g in "${GAMES_ENVW[@]:-}"; do echo "    - ${g}"; done
    echo
} | tee -a "${DISPATCH_LOG}"

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
declare -a PIDS=()
declare -a NAMES=()
FAILED=0

wait_for_next() {
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

# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------
launch_gymv() {
    local env="$1"
    local key="gym_v_${env}"
    local log="${LOG_DIR}/${key}.log"
    local cmd=(
        python3 -u "${SCRIPT_DIR}/extract_skillbank_gymv_gpt54.py"
        --input_dir  "${GYMV_INPUT}"
        --output_dir "${GYMV_OUT}"
        --model      "${MODEL}"
        --envs       "${env}"
        --cache_normalized
    )
    [[ -n "${MAX_EPISODES}" ]] && cmd+=(--max_episodes "${MAX_EPISODES}")
    {
        echo "===== ${key} | $(date) ====="
        echo "CMD: ${cmd[*]}"
        echo
    } > "${log}"
    "${cmd[@]}" >> "${log}" 2>&1 &
    PIDS+=("$!")
    NAMES+=("${key}")
    echo "  [START] ${key}  pid=$!  log=${log}" | tee -a "${DISPATCH_LOG}"
}

launch_envw() {
    local game="$1"
    local key="env_wrappers_${game}"
    local log="${LOG_DIR}/${key}.log"
    local game_out="${ENVW_OUT}"
    mkdir -p "${game_out}"
    local cmd=(
        python3 -u "${SCRIPT_DIR}/extract_skillbank_gpt54.py"
        --input_dir  "${ENVW_INPUT}"
        --output_dir "${game_out}"
        --model      "${MODEL}"
        --games      "${game}"
        --skip_archetypes
    )
    [[ -n "${MAX_EPISODES}" ]] && cmd+=(--max_episodes "${MAX_EPISODES}")
    {
        echo "===== ${key} | $(date) ====="
        echo "CMD: ${cmd[*]}"
        echo
    } > "${log}"
    "${cmd[@]}" >> "${log}" 2>&1 &
    PIDS+=("$!")
    NAMES+=("${key}")
    echo "  [START] ${key}  pid=$!  log=${log}" | tee -a "${DISPATCH_LOG}"
}

# ---------------------------------------------------------------------------
# Phase 1 — interleave gym_v and env_wrappers workers under one budget
# ---------------------------------------------------------------------------
T0=$(date +%s)
i=0
j=0
N_GYMV=${#ENVS_GYMV[@]}
N_ENVW=${#GAMES_ENVW[@]}
[[ -n "${SKIP_GYMV}" ]] && N_GYMV=0
[[ -n "${SKIP_ENVW}" ]] && N_ENVW=0

while [[ $i -lt ${N_GYMV} || $j -lt ${N_ENVW} ]]; do
    while [[ ${#PIDS[@]} -ge ${PARALLEL} ]]; do
        wait_for_next
    done
    if [[ $i -lt ${N_GYMV} ]]; then
        launch_gymv "${ENVS_GYMV[$i]}"
        i=$((i + 1))
    elif [[ $j -lt ${N_ENVW} ]]; then
        launch_envw "${GAMES_ENVW[$j]}"
        j=$((j + 1))
    fi
    if [[ ${#PIDS[@]} -ge ${PARALLEL} ]]; then continue; fi
    if [[ $j -lt ${N_ENVW} && ${#PIDS[@]} -lt ${PARALLEL} ]]; then
        launch_envw "${GAMES_ENVW[$j]}"
        j=$((j + 1))
    fi
done

while [[ ${#PIDS[@]} -gt 0 ]]; do
    wait_for_next
done

T1=$(date +%s)
echo                                                                 | tee -a "${DISPATCH_LOG}"
echo "  Phase 1 (extraction) done. Elapsed: $((T1 - T0))s"           | tee -a "${DISPATCH_LOG}"
echo                                                                 | tee -a "${DISPATCH_LOG}"

# ---------------------------------------------------------------------------
# Phase 2 — decorate with SkillRecord-shape fields
# ---------------------------------------------------------------------------
INTENTIONS_RUN_NAME="$(basename "${INTENTIONS_DIR}")"
echo "  Phase 2: decorating SkillRecord-shape fields ..."             | tee -a "${DISPATCH_LOG}"
if python3 -u "${SCRIPT_DIR}/_decorate_skill_records.py" \
        --root           "${RUN_DIR}" \
        --intentions_run "${INTENTIONS_RUN_NAME}" \
        --cold_start_run "$(basename "${COLD_START_GYMV}")" \
        --model          "${MODEL}" >> "${DISPATCH_LOG}" 2>&1; then
    echo "  [OK]   decorator"                                          | tee -a "${DISPATCH_LOG}"
else
    echo "  [FAIL] decorator (see ${DISPATCH_LOG})"                    | tee -a "${DISPATCH_LOG}"
    FAILED=$((FAILED + 1))
fi

# ---------------------------------------------------------------------------
# Phase 3 — cross-corpus aggregation
# ---------------------------------------------------------------------------
echo                                                                  | tee -a "${DISPATCH_LOG}"
echo "  Phase 3: aggregating cross-corpus skill index ..."             | tee -a "${DISPATCH_LOG}"
if python3 -u "${SCRIPT_DIR}/unify_skill_index.py" \
        --root       "${RUN_DIR}" \
        --output_dir "${RUN_DIR}" >> "${DISPATCH_LOG}" 2>&1; then
    echo "  [OK]   unify  -> ${RUN_DIR}/_unified/"                     | tee -a "${DISPATCH_LOG}"
else
    echo "  [FAIL] unify (see ${DISPATCH_LOG})"                        | tee -a "${DISPATCH_LOG}"
    FAILED=$((FAILED + 1))
fi

T2=$(date +%s)

# ---------------------------------------------------------------------------
# Run meta
# ---------------------------------------------------------------------------
python3 - <<PY >> "${DISPATCH_LOG}" 2>&1
import json, os, time
meta = {
    "run_dir": "${RUN_DIR}",
    "intentions_dir": "${INTENTIONS_DIR}",
    "intentions_run": "${INTENTIONS_RUN_NAME}",
    "cold_start_gymv": "${COLD_START_GYMV}",
    "cold_start_envw": "${COLD_START_ENVW}",
    "model": "${MODEL}",
    "parallel": ${PARALLEL},
    "max_episodes": "${MAX_EPISODES}",
    "envs_gymv": ${#ENVS_GYMV[@]},
    "games_envw": ${#GAMES_ENVW[@]},
    "elapsed_extraction_seconds": $((T1 - T0)),
    "elapsed_total_seconds": $((T2 - T0)),
    "failed_workers": ${FAILED},
    "phases": ["extraction", "decoration", "unify"],
    "next_step_hint": (
        "Outputs are SkillRecord-shape but un-gated (status=draft, "
        "verified_domains=[]). To promote into draft_store/ later: "
        "call skill_bank.lifecycle.SkillLifecycleManager.ingest_draft "
        "with the rows in <run_dir>/<corpus>/<env>/skill_bank.jsonl, "
        "then run orchestrator.GateService.evaluate(...) with "
        "stages={STATIC, REPLAY}. See PLAN-UNIFIED-SKILL-GATE §6."
    ),
}
with open(os.path.join("${RUN_DIR}", "_run_meta.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print("Wrote _run_meta.json")
PY

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
{
    echo
    echo "============================================================"
    echo "  Skill-discovery dispatcher — DONE"
    echo "============================================================"
    echo "  Workers       : $((N_GYMV + N_ENVW))   (gym_v=${N_GYMV}  env_wrappers=${N_ENVW})"
    echo "  Failed        : ${FAILED}"
    echo "  Elapsed (ext) : $((T1 - T0))s"
    echo "  Elapsed (tot) : $((T2 - T0))s"
    echo "  Run dir       : ${RUN_DIR}"
    echo "  Dispatch log  : ${DISPATCH_LOG}"
} | tee -a "${DISPATCH_LOG}"

exit $(( FAILED > 0 ? 1 : 0 ))
