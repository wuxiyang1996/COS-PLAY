#!/usr/bin/env bash
#
# run_coldstart_actor_browsergym_shard.sh — N-way parallel BrowserGym
# cold-start actor over the lean-plan task pool.
#
# The single-process launcher (``generate_cold_start_actor_browsergym.py``)
# loops tasks serially because each task spins up its own headless
# Chromium / Playwright session — those sessions are *process-coupled*
# and would race under threading.  This wrapper dispatches the task
# list across N independent Python processes (= N independent Chromium
# instances), each handed a disjoint round-robin slice of tasks.
#
# The wrapper also auto-sources the WebArena env file
# (``webarena_env.sh``) when that suite appears in the task list —
# exactly mirroring the single-shot launcher. (VisualWebArena was
# dropped 2026-05-03 — see legacy/visualwebarena/README.md.)
#
# Usage:
#
#   # Default: lean-plan pools (MiniWoB 125 + AssistantBench 180),
#   # 8 shards, gpt-5.4 + reasoning_effort=minimal
#   bash cold_start/run_coldstart_actor_browsergym_shard.sh
#
#   # 16 shards, custom tasks file, save frames
#   bash cold_start/run_coldstart_actor_browsergym_shard.sh \
#       --num_shards 16 \
#       --tasks_file cold_start/task_samples/browsergym_assistantbench_200.txt \
#       -- --save_frames -v
#
#   # Pass any remaining flags after `--` straight to the Python launcher
#   bash cold_start/run_coldstart_actor_browsergym_shard.sh \
#       --num_shards 8 -- --max_steps 16 --reasoning_effort medium -v
#
# Flags (handled by this wrapper):
#   --num_shards N            number of parallel processes (default 8)
#   --tasks_file PATH         add a tasks file (repeatable; lines starting
#                             with `#` are skipped)
#   --output_dir PATH         shared output dir for all shards
#                             (default: Cold-start-out-browsergym)
#   --model NAME              model passed to the Python launcher
#                             (default: gpt-5.4)
#   --reasoning_effort EFFORT one of {minimal,low,medium,high}
#                             (default: minimal)
#   --                        end of wrapper flags; everything after is
#                             forwarded to generate_cold_start_actor_browsergym.py
#
# Bottlenecks (be honest):
#   - RAM: each Chromium ≈ 500 MB, so 8 shards ≈ 4 GB RAM
#   - WebArena / VisualWebArena self-hosted sites have per-host QPS limits;
#     16+ shards may saturate them.
#   - OpenAI rate limits: tier-4 ≈ 10 k RPM (plenty for 8-16 shards),
#     tier-1 ≈ 500 RPM (drop to 2-4 shards).
#
# Output layout:
#   <output_dir>/
#       <safe_task_id>/episode_000.json     ← per-task artifacts (shared)
#       _shard_logs/shard_NN.log            ← per-shard stdout/stderr
#       _shard_logs/shard_NN.tasks          ← per-shard task list (audit)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODEBASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${CODEBASE_ROOT}/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────
NUM_SHARDS=8
TASKS_FILES=()
OUTPUT_DIR=""
MODEL="gpt-5.4"
REASONING_EFFORT="minimal"
CONDA_ENV="${BROWSERGYM_CONDA_ENV:-browsergym}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_shards|-N)        shift; NUM_SHARDS="$1"; shift ;;
        --tasks_file|-T)        shift; TASKS_FILES+=("$1"); shift ;;
        --output_dir|-O)        shift; OUTPUT_DIR="$1"; shift ;;
        --model)                shift; MODEL="$1"; shift ;;
        --reasoning_effort)     shift; REASONING_EFFORT="$1"; shift ;;
        --conda_env|--conda-env) shift; CONDA_ENV="$1"; shift ;;
        --help|-h)
            sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        # Eat the literal `--` separator so we don't forward it verbatim
        # — argparse rejects bare `--` followed by `--flag value`.
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *)  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Default to the lean-plan duo if no tasks files given.
# (Was a trio that included visualwebarena_200; VWA dropped 2026-05-03 —
# see legacy/visualwebarena/README.md.)
if [[ ${#TASKS_FILES[@]} -eq 0 ]]; then
    TASKS_FILES=(
        "${CODEBASE_ROOT}/cold_start/task_samples/browsergym_miniwob_200.txt"
        "${CODEBASE_ROOT}/cold_start/task_samples/browsergym_assistantbench_200.txt"
    )
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${CODEBASE_ROOT}/Cold-start-out-browsergym"
fi

# ── Validate + count tasks ────────────────────────────────────────────────
if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || [[ "$NUM_SHARDS" -lt 1 ]]; then
    echo "[ERROR] --num_shards must be a positive integer (got: $NUM_SHARDS)" >&2
    exit 2
fi

ALL_TASKS=()
for f in "${TASKS_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] tasks file not found: $f" >&2
        exit 2
    fi
    # `|| [[ -n "$line" ]]` so we still capture the final line when
    # the file lacks a trailing newline (common for hand-edited lists).
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -n "$line" ]] && ALL_TASKS+=("$line")
    done < "$f"
done
# de-dupe while preserving order
declare -A SEEN
TASKS_UNIQUE=()
for t in "${ALL_TASKS[@]}"; do
    [[ -n "${SEEN[$t]:-}" ]] && continue
    SEEN[$t]=1
    TASKS_UNIQUE+=("$t")
done

N_TASKS=${#TASKS_UNIQUE[@]}
if [[ $N_TASKS -eq 0 ]]; then
    echo "[ERROR] no tasks loaded from: ${TASKS_FILES[*]}" >&2
    exit 2
fi
if [[ $NUM_SHARDS -gt $N_TASKS ]]; then
    echo "[INFO] capping --num_shards from $NUM_SHARDS to $N_TASKS (= one task per shard)"
    NUM_SHARDS=$N_TASKS
fi

# ── Auto-source self-hosted-site env files when relevant ──────────────────
# (VisualWebArena env auto-source removed 2026-05-03 — see
#  legacy/visualwebarena/README.md.)
NEEDS_WA=0
NEEDS_VWA=0
for t in "${TASKS_UNIQUE[@]}"; do
    case "$t" in
        *webarena.*) NEEDS_WA=1 ;;
        *visualwebarena.*) NEEDS_VWA=1 ;;
    esac
done
if [[ $NEEDS_WA -eq 1 ]] && [[ -f "${CODEBASE_ROOT}/cold_start/webarena_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CODEBASE_ROOT}/cold_start/webarena_env.sh"
fi
if [[ $NEEDS_VWA -eq 1 ]]; then
    echo "[WARN] visualwebarena.* tasks detected, but VWA support was dropped" >&2
    echo "       on 2026-05-03. See legacy/visualwebarena/README.md." >&2
fi

# ── Auto-wire MINIWOB_URL if any task is browsergym/miniwob.* ─────────────
# (mirrors run_coldstart_actor_browsergym.sh; HTML pages ship from a
# separate clone of Farama-Foundation/miniwob-plusplus, NOT the pip pkg.)
_NEED_MINIWOB=0
for t in "${TASKS_UNIQUE[@]}"; do
    case "$t" in
        browsergym/miniwob.*) _NEED_MINIWOB=1; break ;;
    esac
done
if [[ $_NEED_MINIWOB -eq 1 ]] && [[ -z "${MINIWOB_URL:-}" ]]; then
    _candidates=()
    [[ -n "${MINIWOB_HTML_DIR:-}" ]] && _candidates+=("${MINIWOB_HTML_DIR}")
    _candidates+=(
        "/workspace/BrowserGym/miniwob-plusplus/miniwob/html/miniwob"
        "/workspace/miniwob-plusplus/miniwob/html/miniwob"
        "${HOME}/miniwob-plusplus/miniwob/html/miniwob"
    )
    for d in "${_candidates[@]}"; do
        if [[ -d "$d" ]] && [[ -f "$d/click-button.html" ]]; then
            export MINIWOB_URL="file://${d}/"
            echo "[INFO] MINIWOB_URL auto-set to ${MINIWOB_URL}"
            break
        fi
    done
    if [[ -z "${MINIWOB_URL:-}" ]]; then
        echo "[ERROR] miniwob.* tasks requested but MINIWOB_URL is unset and" >&2
        echo "        miniwob-plusplus HTML pages were not found in any of:" >&2
        for d in "${_candidates[@]}"; do echo "          - $d" >&2; done
        echo "        Install with:" >&2
        echo "          git clone https://github.com/Farama-Foundation/miniwob-plusplus.git \\" >&2
        echo "              /workspace/BrowserGym/miniwob-plusplus" >&2
        echo "          git -C /workspace/BrowserGym/miniwob-plusplus reset --hard \\" >&2
        echo "              7fd85d71a4b60325c6585396ec4f48377d049838" >&2
        exit 2
    fi
fi

# ── Headless display + PYTHONPATH ─────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
if [[ -z "${DISPLAY:-}" ]]; then
    if command -v Xvfb >/dev/null 2>&1 && ! pgrep -x Xvfb >/dev/null 2>&1; then
        Xvfb :99 -screen 0 1280x1024x24 &>/dev/null &
        sleep 0.5
    fi
    if command -v Xvfb >/dev/null 2>&1; then
        export DISPLAY="${DISPLAY:-:99}"
    fi
fi
# Build PYTHONPATH cleanly so we never inject empty entries (which would
# silently put the cwd on sys.path on POSIX bash).
PYPATH_ADD=("${CODEBASE_ROOT}" "${WORKSPACE_ROOT}")
JOINED_PYPATH="$(IFS=:; echo "${PYPATH_ADD[*]}")"
export PYTHONPATH="${JOINED_PYPATH}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Conda env (auto-activate `browsergym` so the shards can `import browsergym`)
# Override with `--conda_env <name>` or `BROWSERGYM_CONDA_ENV=<name>`.
# Pass `--conda_env none` to skip activation (e.g. when running from a
# pre-activated env that already has BrowserGym installed).
if [[ "${CONDA_ENV}" != "none" && "${CONDA_ENV}" != "" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "[ERROR] conda not on PATH but --conda_env=${CONDA_ENV} requested." >&2
        echo "        Pass --conda_env none if your active env already has browsergym." >&2
        exit 2
    fi
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
        echo "[ERROR] conda env '${CONDA_ENV}' not found. Available:" >&2
        conda env list | awk '$1 !~ /^#/ && $1 != "" {print "  - "$1}' >&2
        echo "        Install via: bash install/install_browsergym.sh" >&2
        exit 2
    fi
    conda activate "${CONDA_ENV}"
fi
# Preflight: is `browsergym` actually importable in the current env?
if ! python3 -c "import browsergym" 2>/dev/null; then
    echo "[ERROR] 'import browsergym' fails in the active env ($(python3 -c 'import sys;print(sys.executable)'))." >&2
    echo "        Activate the right env or run with --conda_env <env>." >&2
    exit 2
fi

# ── Banner ────────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}/_shard_logs"
echo "================================================================"
echo "  Cold-Start Actor — BrowserGym SHARD wrapper"
echo "================================================================"
echo "  Tasks files: ${TASKS_FILES[*]}"
echo "  Total tasks: ${N_TASKS}  (deduped)"
echo "  Shards:      ${NUM_SHARDS}  (~ $((N_TASKS / NUM_SHARDS)) tasks/shard)"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Model:       ${MODEL}    reasoning_effort=${REASONING_EFFORT}"
echo "  Extra args:  ${EXTRA_ARGS[*]:-(none)}"
echo "  Logs:        ${OUTPUT_DIR}/_shard_logs/shard_NN.log"
echo "================================================================"

# ── Dispatch ──────────────────────────────────────────────────────────────
PIDS=()
SHARD_OUT_PATHS=()
for ((i = 0; i < NUM_SHARDS; i++)); do
    SHARD_TASKS_FILE="${OUTPUT_DIR}/_shard_logs/shard_$(printf '%02d' "$i").tasks"
    : > "${SHARD_TASKS_FILE}"
    SHARD_TASKS=()
    for ((j = i; j < N_TASKS; j += NUM_SHARDS)); do
        SHARD_TASKS+=("${TASKS_UNIQUE[$j]}")
        echo "${TASKS_UNIQUE[$j]}" >> "${SHARD_TASKS_FILE}"
    done
    if [[ ${#SHARD_TASKS[@]} -eq 0 ]]; then
        continue
    fi

    LOG="${OUTPUT_DIR}/_shard_logs/shard_$(printf '%02d' "$i").log"
    echo "[shard $i] dispatching ${#SHARD_TASKS[@]} tasks  -> ${LOG}"
    (
        python3 "${CODEBASE_ROOT}/cold_start/generate_cold_start_actor_browsergym.py" \
            --tasks "${SHARD_TASKS[@]}" \
            --model "${MODEL}" \
            --reasoning_effort "${REASONING_EFFORT}" \
            --output_dir "${OUTPUT_DIR}" \
            "${EXTRA_ARGS[@]}" \
            >"${LOG}" 2>&1
    ) &
    PIDS+=($!)
    SHARD_OUT_PATHS+=("${LOG}")
done

# ── Wait + collect ────────────────────────────────────────────────────────
#
# NOTE on the global summary file: the python launcher writes one
# ``<output_dir>/batch_rollout_summary.json`` at the END of its run, so
# concurrent shards race and the last-completing shard wins.  Per-task
# artifacts under ``<output_dir>/<safe_task_id>/`` are disjoint across
# shards (each shard owns a distinct task slice), so the per-task
# rollouts are *not* corrupted — only the global roll-up is overwritten.
# We aggregate a race-free roll-up below by walking the per-task
# rollout_summary.json files (race-free: shards own disjoint tasks).
FAILS=0
for ((k = 0; k < ${#PIDS[@]}; k++)); do
    pid="${PIDS[$k]}"
    if ! wait "$pid"; then
        FAILS=$((FAILS + 1))
        echo "[shard $k] FAILED  (see ${SHARD_OUT_PATHS[$k]})"
    else
        echo "[shard $k] done"
    fi
done

# Race-free post-aggregation: walk per-task rollout_summary.json files
# (each shard owns a disjoint set of tasks, so per-task files do NOT race).
AGG_PATH="${OUTPUT_DIR}/batch_rollout_summary_aggregated.json"
python3 - "$OUTPUT_DIR" "$AGG_PATH" <<'PY' || echo "[WARN] aggregation skipped"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
out  = Path(sys.argv[2])
per_task = []
for p in sorted(root.rglob("rollout_summary.json")):
    try:
        per_task.append({"path": str(p.relative_to(root)),
                         **json.loads(p.read_text(encoding="utf-8"))})
    except Exception as exc:
        per_task.append({"path": str(p.relative_to(root)), "error": repr(exc)})
out.write_text(json.dumps({
    "schema": "browsergym_shard_aggregated_v1",
    "n_tasks": len(per_task),
    "per_task": per_task,
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[OK] wrote {out}  ({len(per_task)} tasks aggregated)")
PY

if [[ $FAILS -gt 0 ]]; then
    echo "================================================================"
    echo "  ${FAILS}/${NUM_SHARDS} shard(s) FAILED — inspect logs in"
    echo "  ${OUTPUT_DIR}/_shard_logs/ for details."
    echo "================================================================"
    exit 1
fi

echo "================================================================"
echo "  All ${NUM_SHARDS} shards completed."
echo "  Per-task artifacts under: ${OUTPUT_DIR}/<safe_task_id>/"
echo "================================================================"
