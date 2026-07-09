#!/usr/bin/env bash
#
# run_coldstart_actor.sh — gpt-5.5 actor cold-start over env_wrappers
#
# Runs ``cold_start/generate_cold_start_actor.py`` to drive the COS-PLAY
# actor pipeline against the four supported env_wrappers games:
#
#   - twenty_forty_eight  (LM-Game Bench / GamingAgent)
#   - candy_crush         (LM-Game Bench / GamingAgent)
#   - tetris              (LM-Game Bench / GamingAgent, macro action wrapper)
#   - super_mario         (Orak — needs the orak-mario conda env + Xvfb)
#
# For each step the agent:
#   1. Renders the wrapper's frame (PIL) via env_wrappers.visual_utils.
#   2. Calls gpt-5.5 (vision) to produce a canonical <state>...</state>
#      schema following vlm_wrapper.schema.
#   3. Calls gpt-5.5 with the schema + the wrapper's valid-action list and
#      function-calling to choose ONE action verbatim.
#   4. env.step(action), saves the Experience (with schema+frame) into
#      <codebase_root>/Cold-start-out/<game>/.
#
# Usage:
#
#   # All four games, defaults
#   bash cold_start/run_coldstart_actor.sh
#
#   # Just 2048 + tetris, 10 episodes each, max 50 steps
#   bash cold_start/run_coldstart_actor.sh \
#       --games twenty_forty_eight tetris --episodes 10 --max_steps 50
#
#   # Resume an interrupted run, save frames to disk for debugging
#   bash cold_start/run_coldstart_actor.sh --resume --save_frames -v
#
#   # Skip the vision call (cheap canonical-schema baseline)
#   bash cold_start/run_coldstart_actor.sh --no_vision --episodes 3
#
#   # Show all options:
#   bash cold_start/run_coldstart_actor.sh --help
#
# Optional environment variables:
#   USE_ORAK_MARIO_CONDA=1   activate the orak-mario conda env (needed when
#                            running ``super_mario`` so nes-py / pyglet are
#                            available); default 0 / off.
#   OPENAI_API_KEY / OPENROUTER_API_KEY  one of these must be set.
#   PYTHONPATH               extra paths to prepend (Game-AI-Agent /
#                            GamingAgent / Orak/src are added automatically
#                            when found alongside this repo).

set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAMINGAGENT_ROOT="$(cd "$CODEBASE_ROOT/../GamingAgent" 2>/dev/null && pwd || echo "")"
ORAK_SRC="$(cd "$CODEBASE_ROOT/../Orak/src" 2>/dev/null && pwd || echo "")"

# ── Parse user args ───────────────────────────────────────────────────────
EXTRA_ARGS=("$@")

# Detect whether super_mario is in the requested games (so we know to set
# up Xvfb / the orak-mario conda env).
NEEDS_MARIO=0
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    NEEDS_MARIO=1
else
    if printf '%s\n' "${EXTRA_ARGS[@]}" | grep -E -q '(^|=)(super_mario|mario|supermario)( |$)'; then
        NEEDS_MARIO=1
    fi
fi

# ── Optional: activate orak-mario conda env for Mario runs ────────────────
if [ "${USE_ORAK_MARIO_CONDA:-0}" -eq 1 ] || [ "${NEEDS_MARIO}" -eq 1 ]; then
    if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "orak-mario"; then
        eval "$(conda shell.bash hook)"
        conda activate orak-mario || true
    elif [ "${NEEDS_MARIO}" -eq 1 ]; then
        echo "[INFO] orak-mario conda env not found; running super_mario in current env"
        echo "       (will fail if nes-py / gym_super_mario_bros are missing)"
    fi
fi

# ── Headless rendering ────────────────────────────────────────────────────
export SDL_VIDEODRIVER=dummy
export PYGLET_HEADLESS=1

if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb >/dev/null 2>&1 && ! pgrep -x Xvfb >/dev/null 2>&1; then
        Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
        sleep 0.5
    fi
    if command -v Xvfb >/dev/null 2>&1; then
        export DISPLAY=:99
    fi
fi

# ── PYTHONPATH ────────────────────────────────────────────────────────────
PYPATH_ADD=("${CODEBASE_ROOT}")
[ -n "${GAMINGAGENT_ROOT}" ] && [ -d "${GAMINGAGENT_ROOT}" ] && PYPATH_ADD+=("${GAMINGAGENT_ROOT}")
[ -n "${ORAK_SRC}" ]         && [ -d "${ORAK_SRC}" ]         && PYPATH_ADD+=("${ORAK_SRC}")
JOINED_PYPATH="$(IFS=:; echo "${PYPATH_ADD[*]}")"
export PYTHONPATH="${JOINED_PYPATH}${PYTHONPATH:+:${PYTHONPATH}}"

# ── API key check ─────────────────────────────────────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[WARNING] Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set."
    echo "          Both VLM and actor calls will fail. See .env.example."
fi

# ── Dependency probe (don't auto-install — keep the launcher hermetic) ────
echo "================================================================"
echo "  Cold-Start Actor Agent — env_wrappers + gpt-5.5"
echo "================================================================"
echo "  Codebase:    ${CODEBASE_ROOT}"
[ -n "${GAMINGAGENT_ROOT}" ] && echo "  GamingAgent: ${GAMINGAGENT_ROOT}"
[ -n "${ORAK_SRC}" ]         && echo "  Orak src:    ${ORAK_SRC}"
echo "  DISPLAY:     ${DISPLAY:-unset}"
echo "  Python:      $(python3 --version 2>&1)"

if python3 -c "import openai" >/dev/null 2>&1; then
    :
else
    echo "[ERROR] 'openai' package missing. Install: pip install openai"
    exit 1
fi

# Smoke-test that env_wrappers and vlm_wrapper.schema import cleanly.
SMOKE_OUTPUT="$(python3 - <<'PYEOF' 2>&1 || true
import warnings, sys
warnings.filterwarnings("ignore")
ok = []
try:
    from env_wrappers.gym_like import make_gaming_env
    ok.append("env_wrappers.gym_like")
except Exception as exc:
    print(f"FAIL env_wrappers.gym_like: {exc}", file=sys.stderr)
try:
    from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
    ok.append("env_wrappers.gamingagent_nl_wrapper")
except Exception as exc:
    print(f"FAIL env_wrappers.gamingagent_nl_wrapper: {exc}", file=sys.stderr)
try:
    from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper
    ok.append("env_wrappers.tetris_macro_wrapper")
except Exception as exc:
    print(f"FAIL env_wrappers.tetris_macro_wrapper: {exc}", file=sys.stderr)
try:
    from env_wrappers.orak_nl_wrapper import make_orak_env
    ok.append("env_wrappers.orak_nl_wrapper")
except Exception as exc:
    print(f"WARN env_wrappers.orak_nl_wrapper unavailable (super_mario will fail): {exc}", file=sys.stderr)
try:
    from vlm_wrapper.schema import build_system_prompt, build_user_message
    ok.append("vlm_wrapper.schema")
except Exception as exc:
    print(f"FAIL vlm_wrapper.schema: {exc}", file=sys.stderr)
print("imports_ok=" + ",".join(ok))
PYEOF
)"
echo "  ${SMOKE_OUTPUT}" | sed 's/^/  /'

[ -n "${OPENROUTER_API_KEY:-}" ] && echo "  API key:     ${OPENROUTER_API_KEY:0:12}... (OpenRouter)"
[ -z "${OPENROUTER_API_KEY:-}" ] && [ -n "${OPENAI_API_KEY:-}" ] && echo "  API key:     ${OPENAI_API_KEY:0:12}... (OpenAI)"
echo "================================================================"
echo ""

# ── Defaults: when the user passes nothing, run all four games with sane
#               per-game caps and resume any in-progress runs.
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--resume --save_frames -v)
fi

python3 "${SCRIPT_DIR}/generate_cold_start_actor.py" "${EXTRA_ARGS[@]}"
EXIT_CODE=$?

# ── Post-run summary ──────────────────────────────────────────────────────
# Honor an explicit ``--output_dir`` in the forwarded args (the all-games
# wrapper passes a per-run/per-env path); fall back to the default
# Cold-start-out at the repo root.
OUTPUT_DIR="${CODEBASE_ROOT}/Cold-start-out"
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
echo "  Cold-Start Actor — Post-Run Summary"
echo "================================================================"

if [ -d "$OUTPUT_DIR" ]; then
    TOTAL=0
    for game_dir in "$OUTPUT_DIR"/*/; do
        [ -d "$game_dir" ] || continue
        game="$(basename "$game_dir")"
        count=$(find "$game_dir" -maxdepth 1 -name 'episode_*.json' ! -name 'episode_buffer.json' 2>/dev/null | wc -l)
        TOTAL=$((TOTAL + count))
        has_buffer="no"; [ -f "$game_dir/episode_buffer.json" ] && has_buffer="yes"
        has_jsonl="no";  [ -f "$game_dir/rollouts.jsonl" ]      && has_jsonl="yes"
        printf "  %-22s %3d episodes  buffer=%s  jsonl=%s\n" "$game" "$count" "$has_buffer" "$has_jsonl"
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
echo "    eps = load_episodes_from_jsonl('${OUTPUT_DIR}/<game>/rollouts.jsonl')"
echo "================================================================"

exit ${EXIT_CODE}
