#!/usr/bin/env bash
#
# run_coldstart_actor_gymv.sh — gpt-5.5 actor cold-start over gym-v Temporal envs
#
# Runs ``cold_start/generate_cold_start_actor_gymv.py`` to drive the COS-PLAY
# actor pipeline against gym-v ``Temporal/<Title>-v0`` environments
# (stable-retro / Genesis ROMs). At each step the agent:
#
#   1. Pulls the multimodal Observation (PIL frame, text, metadata) from gym-v.
#   2. Calls gpt-5.5 (vision) to produce a canonical <state>...</state>
#      schema following ``vlm_wrapper.schema``.
#   3. Calls gpt-5.5 with the schema + the env's
#      ``obs.metadata['available_actions']`` list and function-calling to
#      choose ONE action verbatim.
#   4. ``env.step({agent_id: action})``, saves the Experience (with schema
#      + frame + heuristic grounding) into
#      ``<codebase_root>/Cold-start-out-gymv/<env_id_safe>/``.
#
# Default env: ``Temporal/Airstriker-v0`` (stable-retro ships its ROM).
#
# Usage:
#
#   # Default: 1 episode of Airstriker, 60 steps, with frames + verbose
#   bash cold_start/run_coldstart_actor_gymv.sh
#
#   # Two envs, 3 episodes each, 80 steps
#   bash cold_start/run_coldstart_actor_gymv.sh \
#       --envs Temporal/Airstriker-v0 Temporal/SpaceHarrierII-v0 \
#       --episodes 3 --max_steps 80 -v
#
#   # Resume an interrupted run, save frames to disk for debugging
#   bash cold_start/run_coldstart_actor_gymv.sh --resume --save_frames -v
#
#   # Skip the vision call (cheap heuristic-schema baseline)
#   bash cold_start/run_coldstart_actor_gymv.sh --no_vision --episodes 3
#
#   # Show all options:
#   bash cold_start/run_coldstart_actor_gymv.sh --help
#
# Optional environment variables:
#   OPENAI_API_KEY / OPENROUTER_API_KEY  one of these must be set (or pass
#                                         ``--api_key`` on the CLI).
#   PYTHONPATH                           extra paths to prepend (the codebase
#                                         root and gym-v are added automatically
#                                         when found alongside this repo).

set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$CODEBASE_ROOT/.." && pwd)"
GYMV_ROOT="$(cd "$WORKSPACE_ROOT/gym-v" 2>/dev/null && pwd || echo "")"

# ── Parse user args (forwarded to the python launcher) ────────────────────
EXTRA_ARGS=("$@")

# ── Headless rendering (stable-retro spawns SDL/pyglet in some games) ─────
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

# ── API key check ─────────────────────────────────────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[WARNING] Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set."
    echo "          Both VLM and actor calls will fail unless --api_key is passed."
fi

# ── Banner ────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Cold-Start Actor Agent — gym-v Temporal + gpt-5.5"
echo "================================================================"
echo "  Codebase:    ${CODEBASE_ROOT}"
[ -n "${GYMV_ROOT}" ] && echo "  Gym-V:       ${GYMV_ROOT}"
echo "  DISPLAY:     ${DISPLAY:-unset}"
echo "  Python:      $(python3 --version 2>&1)"

# ── Dependency probe (don't auto-install) ─────────────────────────────────
if python3 -c "import openai" >/dev/null 2>&1; then
    :
else
    echo "[ERROR] 'openai' package missing. Install: pip install openai"
    exit 1
fi

SMOKE_OUTPUT="$(python3 - <<'PYEOF' 2>&1 || true
import warnings, sys
warnings.filterwarnings("ignore")
ok = []
try:
    import gym_v  # noqa: F401
    ok.append("gym_v")
except Exception as exc:
    print(f"FAIL gym_v: {exc}", file=sys.stderr)
try:
    from gymv_wrapper.temporal_visual_grounding import (
        TEMPORAL_GAME_SPECS,
        build_temporal_visual_schema,
    )  # noqa: F401
    ok.append("gymv_wrapper.temporal_visual_grounding")
except Exception as exc:
    print(f"FAIL gymv_wrapper.temporal_visual_grounding: {exc}", file=sys.stderr)
try:
    from vlm_wrapper.schema import build_system_prompt, build_user_message  # noqa: F401
    ok.append("vlm_wrapper.schema")
except Exception as exc:
    print(f"FAIL vlm_wrapper.schema: {exc}", file=sys.stderr)
try:
    import stable_retro  # noqa: F401
    ok.append("stable_retro")
except Exception as exc:
    print(f"WARN stable_retro unavailable (Temporal envs will fail to load): {exc}", file=sys.stderr)
print("imports_ok=" + ",".join(ok))
PYEOF
)"
echo "  ${SMOKE_OUTPUT}" | sed 's/^/  /'

[ -n "${OPENROUTER_API_KEY:-}" ] && echo "  API key:     ${OPENROUTER_API_KEY:0:12}... (OpenRouter)"
[ -z "${OPENROUTER_API_KEY:-}" ] && [ -n "${OPENAI_API_KEY:-}" ] && echo "  API key:     ${OPENAI_API_KEY:0:12}... (OpenAI)"
echo "================================================================"
echo ""

# ── Defaults: when the user passes nothing, run the safe single-env smoke
#               test with frames saved and verbose stepping.
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--save_frames -v)
fi

python3 "${SCRIPT_DIR}/generate_cold_start_actor_gymv.py" "${EXTRA_ARGS[@]}"
EXIT_CODE=$?

# ── Post-run summary ──────────────────────────────────────────────────────
# Honor an explicit ``--output_dir`` in the forwarded args; fall back to the
# default Cold-start-out-gymv at the repo root.
OUTPUT_DIR="${CODEBASE_ROOT}/Cold-start-out-gymv"
prev_was_outdir=0
for arg in "${EXTRA_ARGS[@]}"; do
    if [ "$prev_was_outdir" -eq 1 ]; then
        OUTPUT_DIR="$arg"
        prev_was_outdir=0
        continue
    fi
    case "$arg" in
        --output_dir|--output-dir) prev_was_outdir=1 ;;
        --output_dir=*) OUTPUT_DIR="${arg#--output_dir=}" ;;
        --output-dir=*) OUTPUT_DIR="${arg#--output-dir=}" ;;
    esac
done

echo ""
echo "================================================================"
echo "  Cold-Start Actor (gym-v) — Post-Run Summary"
echo "================================================================"

if [ -d "$OUTPUT_DIR" ]; then
    TOTAL=0
    for env_dir in "$OUTPUT_DIR"/*/; do
        [ -d "$env_dir" ] || continue
        env_id="$(basename "$env_dir")"
        count=$(find "$env_dir" -maxdepth 1 -name 'episode_*.json' ! -name 'episode_buffer.json' 2>/dev/null | wc -l)
        TOTAL=$((TOTAL + count))
        has_buffer="no"; [ -f "$env_dir/episode_buffer.json" ] && has_buffer="yes"
        has_jsonl="no";  [ -f "$env_dir/rollouts.jsonl" ]      && has_jsonl="yes"
        printf "  %-32s %3d episodes  buffer=%s  jsonl=%s\n" "$env_id" "$count" "$has_buffer" "$has_jsonl"
    done
    echo ""
    echo "  Total episodes: $TOTAL"
    echo "  Output dir:     $OUTPUT_DIR"
    [ -f "$OUTPUT_DIR/batch_rollout_summary.json" ] && \
        echo "  Master summary: $OUTPUT_DIR/batch_rollout_summary.json"
else
    echo "  (no output produced — exit code ${EXIT_CODE})"
fi

echo ""
echo "  Load into trainer:"
echo "    from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records"
echo "    eps = load_episodes_from_jsonl('${OUTPUT_DIR}/<env_id_safe>/rollouts.jsonl')"
echo "================================================================"

exit ${EXIT_CODE}
