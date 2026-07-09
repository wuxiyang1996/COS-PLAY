#!/usr/bin/env bash
#
# run_coldstart_actor_all_games.sh — drive ALL 4 games across the two
# required conda envs, sequentially or in parallel, with per-run
# timestamped output directories grouped by conda env.
#
# Output layout (default):
#
#   <codebase_root>/Cold-start-out/
#   ├── <run_id>/                        # YYYY-MM-DD_HH-MM-SS
#   │   ├── game-ai-agent/               # env subdir
#   │   │   ├── twenty_forty_eight/      # <env>/<game>/episode_NNN.json + rollouts.jsonl
#   │   │   ├── candy_crush/
#   │   │   └── tetris/
#   │   ├── orak-mario/
#   │   │   └── super_mario/
#   │   ├── _logs/<game>.log             # per-game stdout/stderr
#   │   └── _run_meta.json               # run config + per-game rc
#   └── latest -> <run_id>               # symlink updated to the most recent run
#
# Per-game env mapping:
#   twenty_forty_eight, candy_crush, tetris  -> game-ai-agent
#   super_mario                               -> orak-mario
#
# Usage:
#
#   # SEQUENTIAL (default), auto-timestamped run id
#   bash cold_start/run_coldstart_actor_all_games.sh \
#       --episodes 1 --model gpt-5.5 --save_frames -v
#
#   # PARALLEL: all 4 games dispatched concurrently
#   bash cold_start/run_coldstart_actor_all_games.sh --parallel \
#       --episodes 1 --model gpt-5.5 --save_frames -v
#
#   # Custom run id (e.g. to resume into the same folder later)
#   bash cold_start/run_coldstart_actor_all_games.sh \
#       --run_id smoke_2025q4 --episodes 2 --resume -v
#
#   # Skip mario (orak-mario env not installed yet)
#   bash cold_start/run_coldstart_actor_all_games.sh --no_mario \
#       --episodes 2 -v
#
# Wrapper-only flags (consumed here, NOT forwarded to the launcher):
#   --parallel | -P         dispatch games concurrently
#   --no_mario              skip super_mario
#   --games <g>...          restrict to a subset (default: all 4)
#   --run_id <id>           override auto-timestamped run id
#   --output_dir <path>     override base dir (default: <repo>/Cold-start-out)
#
# All other arguments are forwarded VERBATIM to ``run_coldstart_actor.sh``
# for every game.  The wrapper itself injects ``--games <game>`` and
# ``--output_dir <run_dir>/<env>`` per dispatch.
#
# Pre-reqs (one-time):
#   bash install/install_main_env.sh                 # game-ai-agent
#   pip install tile_match_gym                       # candy_crush
#   git clone https://github.com/krafton-ai/Orak.git <parent>/Orak
#   bash install/install_orak_mario.sh               # orak-mario

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_coldstart_actor.sh"
DEFAULT_BASE_DIR="${CODEBASE_ROOT}/Cold-start-out"

# Per-game conda env mapping.  When you add a new game, register it here.
declare -A GAME_ENV=(
    [twenty_forty_eight]=game-ai-agent
    [candy_crush]=game-ai-agent
    [tetris]=game-ai-agent
    [super_mario]=orak-mario
)
DEFAULT_GAMES=(twenty_forty_eight candy_crush tetris super_mario)

# ── Parse wrapper-only flags; forward the rest ───────────────────────────
PARALLEL=0
NO_MARIO=0
GAMES=()
USER_ARGS=()
RUN_ID=""
BASE_DIR="$DEFAULT_BASE_DIR"

while [ $# -gt 0 ]; do
    case "$1" in
        --parallel|-P)
            PARALLEL=1; shift ;;
        --no_mario|--no-mario)
            NO_MARIO=1; shift ;;
        --run_id|--run-id)
            shift; RUN_ID="${1:-}"; shift ;;
        --output_dir|--output-dir)
            shift; BASE_DIR="${1:-$DEFAULT_BASE_DIR}"; shift ;;
        --games)
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                GAMES+=("$1"); shift
            done ;;
        *)
            USER_ARGS+=("$1"); shift ;;
    esac
done

if [ ${#GAMES[@]} -eq 0 ]; then
    GAMES=("${DEFAULT_GAMES[@]}")
fi
if [ "$NO_MARIO" -eq 1 ]; then
    FILTERED=()
    for g in "${GAMES[@]}"; do
        [ "$g" != "super_mario" ] && FILTERED+=("$g")
    done
    GAMES=("${FILTERED[@]}")
fi
if [ -z "$RUN_ID" ]; then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
fi

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

mkdir -p "$LOG_DIR"

# ── Print plan ───────────────────────────────────────────────────────────
echo "============================================================"
if [ "$PARALLEL" -eq 1 ]; then
    echo "  COS-PLAY Actor — all games (PARALLEL dispatch)"
else
    echo "  COS-PLAY Actor — all games (SEQUENTIAL dispatch)"
fi
echo "============================================================"
echo "  Run id:        $RUN_ID"
echo "  Run dir:       $RUN_DIR"
echo "  Per-game logs: $LOG_DIR/<game>.log"
echo "  Forwarded args: ${USER_ARGS[*]:-<none>}"
echo
for g in "${GAMES[@]}"; do
    env="${GAME_ENV[$g]:-?}"
    out="${RUN_DIR}/${env}/"
    if has_env "$env"; then
        printf "  %-22s -> env=%-14s out=%s\n" "$g" "$env" "$out"
    else
        printf "  %-22s -> SKIP (env '%s' not installed)\n" "$g" "$env"
    fi
done
echo "============================================================"

# ── Per-game runner ──────────────────────────────────────────────────────
run_game() {
    local game=$1
    local env="${GAME_ENV[$game]:-}"
    local out_dir="${RUN_DIR}/${env}"
    local logfile="${LOG_DIR}/${game}.log"

    if [ -z "$env" ]; then
        echo "[ERROR] no env mapping for game '$game'" | tee "$logfile" >&2
        return 2
    fi
    if ! has_env "$env"; then
        echo "[SKIP] $game — conda env '$env' missing" | tee "$logfile" >&2
        return 127
    fi
    mkdir -p "$out_dir"

    # PYTHONUNBUFFERED so per-game logfile streams live (no block buffering
    # when stdout isn't a TTY).  SDL/PYGLET headless flags belt-and-braces.
    PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGLET_HEADLESS=1 \
    conda run -n "$env" --no-capture-output \
        bash "$LAUNCHER" \
            --games "$game" \
            --output_dir "$out_dir" \
            "${USER_ARGS[@]}" \
        > "$logfile" 2>&1
}

declare -A RC

# ── Dispatch ─────────────────────────────────────────────────────────────
START_TS="$(date +%s)"
if [ "$PARALLEL" -eq 1 ]; then
    declare -A PIDS
    echo
    echo "Dispatching ${#GAMES[@]} game(s) in parallel:"
    for g in "${GAMES[@]}"; do
        run_game "$g" &
        PIDS[$g]=$!
        printf "  [START]  %-22s pid=%-8s log=%s\n" \
            "$g" "${PIDS[$g]}" "${LOG_DIR}/${g}.log"
    done
    echo
    echo "Live tail (any of these in another terminal):"
    for g in "${GAMES[@]}"; do
        echo "  tail -f ${LOG_DIR}/${g}.log"
    done
    echo
    echo "Waiting for completion ..."
    for g in "${GAMES[@]}"; do
        wait "${PIDS[$g]}"
        rc=$?
        RC[$g]=$rc
        printf "  [DONE]   %-22s rc=%d\n" "$g" "$rc"
    done
else
    for g in "${GAMES[@]}"; do
        echo
        echo ">>> [SEQUENTIAL] $g  (env=${GAME_ENV[$g]:-?})"
        run_game "$g"
        rc=$?
        RC[$g]=$rc
        # Mirror the tail of per-game log to stdout so sequential runs feel live.
        tail -n 12 "${LOG_DIR}/${g}.log" 2>/dev/null | sed 's/^/    /'
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
    printf '  "forwarded_args": "%s",\n' "${USER_ARGS[*]:-}"
    printf '  "games": {\n'
    first=1
    for g in "${GAMES[@]}"; do
        rc=${RC[$g]:-null}
        env="${GAME_ENV[$g]:-unknown}"
        [ $first -eq 1 ] || printf ',\n'
        printf '    "%s": {"env": "%s", "rc": %s, "out": "%s/%s"}' \
            "$g" "$env" "$rc" "$env" "$g"
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
for g in "${GAMES[@]}"; do
    rc=${RC[$g]:-?}
    printf "  %-22s rc=%s    out=%s/%s/\n" \
        "$g" "$rc" "${GAME_ENV[$g]:-?}" "$g"
    [ "$rc" = "0" ] && ANY_OK=1
done
echo "  Run dir:       $RUN_DIR/"
echo "  Latest:        $BASE_DIR/latest -> $RUN_ID"
echo "  Meta:          $META_FILE"
echo "============================================================"

# Non-zero only if NO game succeeded.
if [ "$ANY_OK" -eq 0 ]; then
    exit 1
fi
exit 0
