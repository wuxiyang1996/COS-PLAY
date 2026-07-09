#!/usr/bin/env bash
#
# run_coldstart_actor_gymv_all.sh — drive ALL gym-v Temporal envs through
# the gpt-5.5 actor cold-start, sequentially or in parallel, with per-env
# logs grouped under a timestamped run dir.
#
# Output layout (default):
#
#   <codebase_root>/Cold-start-out-gymv/
#   ├── <run_id>/                         # YYYY-MM-DD_HH-MM-SS
#   │   ├── Temporal_Airstriker-v0/       # one dir per env (sanitised id)
#   │   │   ├── episode_NNN.json
#   │   │   ├── episode_buffer.json
#   │   │   ├── rollouts.jsonl
#   │   │   ├── rollout_summary.json
#   │   │   └── frames/ep_NNN/step_NNN.png
#   │   ├── ... (one per env) ...
#   │   ├── _logs/<env_id_safe>.log       # per-env stdout/stderr
#   │   └── _run_meta.json                # run config + per-env rc
#   └── latest -> <run_id>                # symlink to the most recent run
#
# Each env gets its own python3 process so that one env's failure (ROM
# missing, retro glitch, OOM, etc.) cannot take down the others.
#
# Usage:
#
#   # PARALLEL (default), 8 retained Temporal envs, full vision pipeline
#   # (5 games dropped 2026-05-03; see baselines/README.md §
#   # "Gym-V benchmark scope").
#   bash cold_start/run_coldstart_actor_gymv_all.sh \
#       --episodes 1 --max_steps 20 --save_frames -v
#
#   # SEQUENTIAL dispatch (one env at a time)
#   bash cold_start/run_coldstart_actor_gymv_all.sh --sequential \
#       --episodes 1 --max_steps 20 -v
#
#   # Cap concurrency to 4 (rate-limit-friendly)
#   bash cold_start/run_coldstart_actor_gymv_all.sh --max_parallel 4 \
#       --episodes 1 --max_steps 20 -v
#
#   # Cheap dry-run (no API spend) on all envs
#   bash cold_start/run_coldstart_actor_gymv_all.sh --no_vision \
#       --episodes 1 --max_steps 20
#
#   # Restrict to a subset
#   bash cold_start/run_coldstart_actor_gymv_all.sh \
#       --envs Temporal/Airstriker-v0 Temporal/Columns-v0 \
#       --episodes 1 --max_steps 20 -v
#
#   # Custom run id (so a follow-up call lands in the same folder)
#   bash cold_start/run_coldstart_actor_gymv_all.sh \
#       --run_id smoke_2026 --episodes 1 --max_steps 20 --resume -v
#
# Wrapper-only flags (consumed here, NOT forwarded to the python launcher):
#   --parallel | -P             dispatch envs concurrently (default)
#   --sequential                dispatch envs one at a time
#   --max_parallel N            cap concurrency (default: unlimited)
#   --envs <id>...              restrict to a subset (default: 8 retained
#                               Gym-V envs; see baselines/README.md §
#                               "Gym-V benchmark scope")
#   --run_id <id>               override auto-timestamped run id
#   --output_dir <path>         override base dir
#                               (default: <codebase_root>/Cold-start-out-gymv)
#   --conda_env <name>          conda env to run inside (default: game-ai-agent)
#
# All other arguments are forwarded VERBATIM to the python launcher for
# every env. The wrapper itself injects ``--envs <env>`` and
# ``--output_dir <run_dir>`` per dispatch.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${CODEBASE_ROOT}/.." && pwd)"
GYMV_ROOT="$(cd "${WORKSPACE_ROOT}/gym-v" 2>/dev/null && pwd || echo "")"
PY_LAUNCHER="${SCRIPT_DIR}/generate_cold_start_actor_gymv.py"
DEFAULT_BASE_DIR="${CODEBASE_ROOT}/Cold-start-out-gymv"
DEFAULT_CONDA_ENV="game-ai-agent"

# Default Gym-V benchmark suite: 8 of the 13 registered Temporal envs.
#
# We dropped 5 games (CastleOfIllusion, CastlevaniaBloodlines, GoldenAxe,
# KidChameleon, MortalKombatII) after the 2026-05-03 frame_skip=8 sweep
# showed they remain at ≤8 % per-episode success rate across all six
# tested backbones (GPT-5.4, Claude-4.6, Gemini-3.1-Pro, Qwen3-VL-235B,
# Qwen3.5-9B, Qwen3.5-35B-A3B). Their reward functions and save states
# are sound — the failure mode is task design (precise platforming /
# combat-block timing / multi-step combo input) that simply does not
# emit reward density compatible with single-shot LLM rollouts at the
# current ~640-frame budget. See `baselines/README.md` § "Gym-V
# benchmark scope" for the full decision log + per-game numbers.
#
# To run the full 13-game registry pass `--envs Temporal/Airstriker-v0
# Temporal/AlteredBeast-v0 ...` explicitly. The dropped games are still
# available via `TEMPORAL_GAME_SPECS` in
# `gymv_wrapper/temporal_visual_grounding.py`.
DEFAULT_ENVS=(
    Temporal/Airstriker-v0
    Temporal/AlteredBeast-v0
    Temporal/Columns-v0
    Temporal/DynamiteHeaddy-v0
    Temporal/SpaceHarrierII-v0
    Temporal/StreetsOfRage2-v0
    Temporal/Strider-v0
    Temporal/ThunderForceIII-v0
)

# ── Parse wrapper-only flags; forward the rest ───────────────────────────
PARALLEL=1
MAX_PARALLEL=0       # 0 = unlimited
ENVS=()
USER_ARGS=()
RUN_ID=""
BASE_DIR="$DEFAULT_BASE_DIR"
CONDA_ENV="$DEFAULT_CONDA_ENV"

while [ $# -gt 0 ]; do
    case "$1" in
        --parallel|-P)
            PARALLEL=1; shift ;;
        --sequential)
            PARALLEL=0; shift ;;
        --max_parallel|--max-parallel)
            shift; MAX_PARALLEL="${1:-0}"; shift ;;
        --run_id|--run-id)
            shift; RUN_ID="${1:-}"; shift ;;
        --output_dir|--output-dir)
            shift; BASE_DIR="${1:-$DEFAULT_BASE_DIR}"; shift ;;
        --conda_env|--conda-env)
            shift; CONDA_ENV="${1:-$DEFAULT_CONDA_ENV}"; shift ;;
        --envs)
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                ENVS+=("$1"); shift
            done ;;
        *)
            USER_ARGS+=("$1"); shift ;;
    esac
done

[ ${#ENVS[@]} -eq 0 ] && ENVS=("${DEFAULT_ENVS[@]}")
[ -z "$RUN_ID" ] && RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"

RUN_DIR="${BASE_DIR}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/_logs"
META_FILE="${RUN_DIR}/_run_meta.json"

# ── Make conda usable ────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is not on PATH. Cannot dispatch envs." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

ENV_LIST="$(conda env list | awk '$1 !~ /^#/ {print $1}')"
has_env() { printf '%s\n' "${ENV_LIST}" | grep -qx "$1"; }
if ! has_env "$CONDA_ENV"; then
    echo "[ERROR] conda env '$CONDA_ENV' not found. Available:"
    printf '%s\n' "${ENV_LIST}" | sed 's/^/  - /'
    exit 1
fi

mkdir -p "$LOG_DIR"

_sanitize_env_id() {
    printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

# ── Headless rendering (stable-retro often needs SDL) ────────────────────
export SDL_VIDEODRIVER=dummy
export PYGLET_HEADLESS=1
if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb >/dev/null 2>&1 && ! pgrep -x Xvfb >/dev/null 2>&1; then
        Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
        sleep 0.5
    fi
    if command -v Xvfb >/dev/null 2>&1; then
        export DISPLAY="${DISPLAY:-:99}"
    fi
fi

# ── PYTHONPATH ────────────────────────────────────────────────────────────
PYPATH_ADD=("${CODEBASE_ROOT}")
[ -n "${GYMV_ROOT}" ] && [ -d "${GYMV_ROOT}" ] && PYPATH_ADD+=("${GYMV_ROOT}")
JOINED_PYPATH="$(IFS=:; echo "${PYPATH_ADD[*]}")"
export PYTHONPATH="${JOINED_PYPATH}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Print plan ───────────────────────────────────────────────────────────
echo "============================================================"
if [ "$PARALLEL" -eq 1 ]; then
    if [ "$MAX_PARALLEL" -gt 0 ]; then
        echo "  COS-PLAY Actor (gym-v) — PARALLEL dispatch (cap=$MAX_PARALLEL)"
    else
        echo "  COS-PLAY Actor (gym-v) — PARALLEL dispatch (unlimited)"
    fi
else
    echo "  COS-PLAY Actor (gym-v) — SEQUENTIAL dispatch"
fi
echo "============================================================"
echo "  Run id:        $RUN_ID"
echo "  Run dir:       $RUN_DIR"
echo "  Per-env logs:  $LOG_DIR/<env_id_safe>.log"
echo "  Conda env:     $CONDA_ENV"
echo "  Forwarded args: ${USER_ARGS[*]:-<none>}"
echo
for env_id in "${ENVS[@]}"; do
    safe="$(_sanitize_env_id "$env_id")"
    printf "  %-34s -> %s/%s\n" "$env_id" "$RUN_DIR" "$safe"
done
echo "============================================================"

# ── Per-env runner ───────────────────────────────────────────────────────
run_env() {
    local env_id=$1
    local safe; safe="$(_sanitize_env_id "$env_id")"
    local logfile="${LOG_DIR}/${safe}.log"

    PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGLET_HEADLESS=1 \
    conda run -n "$CONDA_ENV" --no-capture-output \
        python3 "$PY_LAUNCHER" \
            --envs "$env_id" \
            --output_dir "$RUN_DIR" \
            "${USER_ARGS[@]}" \
        > "$logfile" 2>&1
}

declare -A RC

# ── Dispatch ─────────────────────────────────────────────────────────────
START_TS="$(date +%s)"
if [ "$PARALLEL" -eq 1 ]; then
    declare -A PIDS
    declare -a INFLIGHT_ENVS
    INFLIGHT_ENVS=()
    wait_for_next_env() {
        local done_env="${INFLIGHT_ENVS[0]}"
        INFLIGHT_ENVS=("${INFLIGHT_ENVS[@]:1}")
        wait "${PIDS[$done_env]}"
        local rc=$?
        RC[$done_env]=$rc
        printf "  [DONE]   %-34s rc=%d\n" "$done_env" "$rc"
    }
    echo
    echo "Dispatching ${#ENVS[@]} env(s) in parallel:"
    for env_id in "${ENVS[@]}"; do
        # Concurrency cap: wait for one slot to free up before launching.
        if [ "$MAX_PARALLEL" -gt 0 ]; then
            while [ "${#INFLIGHT_ENVS[@]}" -ge "$MAX_PARALLEL" ]; do
                wait_for_next_env
            done
        fi

        run_env "$env_id" &
        PIDS[$env_id]=$!
        INFLIGHT_ENVS+=("$env_id")
        safe="$(_sanitize_env_id "$env_id")"
        printf "  [START]  %-34s pid=%-8s log=%s\n" \
            "$env_id" "${PIDS[$env_id]}" "${LOG_DIR}/${safe}.log"
    done
    echo
    echo "Live tail (any of these in another terminal):"
    for env_id in "${ENVS[@]}"; do
        safe="$(_sanitize_env_id "$env_id")"
        echo "  tail -f ${LOG_DIR}/${safe}.log"
    done
    echo
    echo "Waiting for completion ..."
    while [ "${#INFLIGHT_ENVS[@]}" -gt 0 ]; do
        wait_for_next_env
    done
else
    for env_id in "${ENVS[@]}"; do
        echo
        echo ">>> [SEQUENTIAL] $env_id"
        run_env "$env_id"
        rc=$?
        RC[$env_id]=$rc
        safe="$(_sanitize_env_id "$env_id")"
        tail -n 12 "${LOG_DIR}/${safe}.log" 2>/dev/null | sed 's/^/    /'
        echo "    rc=$rc"
    done
fi
END_TS="$(date +%s)"
ELAPSED=$((END_TS - START_TS))

# ── Update `latest` symlink ──────────────────────────────────────────────
ln -sfn "$RUN_ID" "${BASE_DIR}/latest" 2>/dev/null || true

# ── Write meta ───────────────────────────────────────────────────────────
{
    printf '{\n'
    printf '  "run_id": "%s",\n' "$RUN_ID"
    printf '  "started_at_unix": %s,\n' "$START_TS"
    printf '  "ended_at_unix": %s,\n' "$END_TS"
    printf '  "elapsed_seconds": %s,\n' "$ELAPSED"
    printf '  "mode": "%s",\n' "$([ "$PARALLEL" -eq 1 ] && echo parallel || echo sequential)"
    printf '  "max_parallel": %s,\n' "${MAX_PARALLEL:-0}"
    printf '  "conda_env": "%s",\n' "$CONDA_ENV"
    printf '  "forwarded_args": "%s",\n' "${USER_ARGS[*]:-}"
    printf '  "envs": {\n'
    first=1
    for env_id in "${ENVS[@]}"; do
        rc=${RC[$env_id]:-null}
        safe="$(_sanitize_env_id "$env_id")"
        [ $first -eq 1 ] || printf ',\n'
        printf '    "%s": {"rc": %s, "out": "%s"}' "$env_id" "$rc" "$safe"
        first=0
    done
    printf '\n  }\n'
    printf '}\n'
} > "$META_FILE"

# ── Summary ──────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "  Combined run finished"
echo "============================================================"
echo "  Run id:        $RUN_ID"
echo "  Elapsed:       ${ELAPSED}s"
ANY_OK=0
for env_id in "${ENVS[@]}"; do
    rc=${RC[$env_id]:-?}
    safe="$(_sanitize_env_id "$env_id")"
    count=0
    if [ -d "$RUN_DIR/$safe" ]; then
        count=$(find "$RUN_DIR/$safe" -maxdepth 1 -name 'episode_*.json' \
            ! -name 'episode_buffer.json' 2>/dev/null | wc -l)
    fi
    printf "  %-34s rc=%-3s episodes=%-3s out=%s/\n" \
        "$env_id" "$rc" "$count" "$safe"
    [ "$rc" = "0" ] && ANY_OK=1
done
echo "  Run dir:       $RUN_DIR/"
echo "  Latest:        $BASE_DIR/latest -> $RUN_ID"
echo "  Meta:          $META_FILE"
echo "============================================================"

# Non-zero only if NO env succeeded.
if [ "$ANY_OK" -eq 0 ]; then
    exit 1
fi
exit 0
