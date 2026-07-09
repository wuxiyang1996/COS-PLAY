#!/usr/bin/env bash
#
# run_coldstart_actor_browsergym.sh — gpt-5.5 actor cold-start over BrowserGym.
#
# Runs ``cold_start/generate_cold_start_actor_browsergym.py`` to drive the
# COS-PLAY actor pipeline against the live BrowserGym env (Playwright +
# headless Chromium — there is NO synthetic / offline mode). Two target
# modes are supported:
#
#   * --tasks <env_id>    Pre-registered BrowserGym task ids
#                          (e.g. browsergym/miniwob.click-button,
#                          browsergym/webarena.42,
#                          browsergym/assistantbench.test.0).
#                          Each task ships with its own goal + reward.
#   * --urls <url>        Open-ended browsing on top of
#                          ``browsergym/openended`` (reward is always 0).
#
# At each step the agent:
#
#   1. Captures a multimodal observation from the live BrowserGym env
#      (PIL screenshot + AXTree + extra_element_properties + last action).
#   2. Computes the deterministic AXTree-walked schema via
#      ``browsergym_wrapper.heuristic.obs_to_schema`` (the fast/free baseline,
#      always emitted as a canonical fallback).
#   3. Calls gpt-5.5 (vision) on the screenshot — with the AXTree as
#      grounding context — to produce the canonical
#      ``<state>...</state>`` schema following ``vlm_wrapper.schema``.
#   4. Calls gpt-5.5 with the schema + the candidate-action list (from
#      ``browsergym_wrapper.tools.list_valid_actions`` plus standard
#      navigation actions) and OpenAI function-calling to choose ONE
#      BrowserGym action (``click(bid)`` / ``fill(bid, "...")`` / scroll /
#      go_back / noop).
#   5. ``env.step(action_string)``, saves the Episode/Experience trail
#      (with schema, frame path, candidate actions, reasoning) into
#      ``<codebase_root>/Cold-start-out-browsergym/<safe_id>/``.
#
# Usage:
#
#   # Default: 1 episode each on Google + Wikipedia (openended), real Chromium
#   bash cold_start/run_coldstart_actor_browsergym.sh
#
#   # List the registered BrowserGym task ids per suite (no rollout)
#   bash cold_start/run_coldstart_actor_browsergym.sh --list_tasks
#
#   # Real benchmark tasks (1 episode each, 30 steps — VWA literature
#   # default; bumped from 12 on 2026-05-03 after the multi-constraint
#   # search-task diagnostic).
#   bash cold_start/run_coldstart_actor_browsergym.sh \
#       --tasks browsergym/miniwob.click-button \
#               browsergym/miniwob.enter-text \
#               browsergym/assistantbench.test.0 \
#       --episodes 1 --max_steps 30 --save_frames -v
#
#   # Custom URL on browsergym/openended, 2 episodes, 30 steps
#   bash cold_start/run_coldstart_actor_browsergym.sh \
#       --urls https://en.wikipedia.org/wiki/Reinforcement_learning \
#       --episodes 2 --max_steps 30 --save_frames -v
#
#   # Cheap baseline: skip the vision call (heuristic schema only) but
#   # still render real Chromium pages
#   bash cold_start/run_coldstart_actor_browsergym.sh --no_vision --episodes 1
#
#   # Visible (non-headless) Chromium for debugging
#   bash cold_start/run_coldstart_actor_browsergym.sh --no_headless -v
#
#   # Resume an interrupted run
#   bash cold_start/run_coldstart_actor_browsergym.sh --resume --save_frames -v
#
#   # Show all options:
#   bash cold_start/run_coldstart_actor_browsergym.sh --help
#
# Pre-reqs:
#   - browsergym + Playwright + Chromium binary installed:
#       bash install/install_browsergym.sh
#       conda activate browsergym
#   - OpenAI/OpenRouter API key (unless ``--no_vision`` is used):
#       OPENAI_API_KEY / OPENROUTER_API_KEY env, or ``api_keys.py`` next to
#       the repo root, or pass ``--api_key`` on the CLI.
#
# Suite-specific infra (only required if those suites are in --tasks):
#   miniwob          MINIWOB_URL=file:///path/to/miniwob-plusplus/html/miniwob/
#   webarena         WA_HOMEPAGE / WA_SHOPPING / WA_REDDIT / WA_GITLAB / ...
#                    (see github.com/web-arena-x/webarena)
#   assistantbench   no extra infra (loads from HuggingFace dataset cache)
#   openended        no extra infra (any live URL works)
#
# NOTE: visualwebarena was DROPPED on 2026-05-03 — see
# legacy/visualwebarena/README.md for the rationale and resurrection
# instructions. The Python package and gym registrations remain
# importable (the conda env still has them) but no driver-side
# convenience scaffolding is wired up here.
#
# Optional environment variables:
#   PYTHONPATH                            extra paths to prepend (the
#                                         codebase root and workspace root
#                                         are added automatically).
#   BROWSERGYM_CONDA_ENV                  conda env to auto-activate when
#                                         browsergym/playwright are missing
#                                         (default: ``browsergym``).

set -uo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$CODEBASE_ROOT/.." && pwd)"

# ── Parse user args (forwarded to the python launcher) ────────────────────
EXTRA_ARGS=()
CONDA_ENV="${BROWSERGYM_CONDA_ENV:-browsergym}"
while [ $# -gt 0 ]; do
    case "$1" in
        --conda_env|--conda-env)
            shift; CONDA_ENV="${1:-$CONDA_ENV}"; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Auto-activate the browsergym conda env if available ───────────────────
# BrowserGym + Playwright + Chromium binary are heavy deps that typically
# live in their own conda env (created by ``install/install_browsergym.sh``).
# If the user is already in a python that can import browsergym.core, we
# leave them alone; otherwise we try to switch to ``$CONDA_ENV`` (default
# ``browsergym``). Override with --conda_env or BROWSERGYM_CONDA_ENV=<name>.
_python_has_browsergym() {
    python3 -c "import browsergym.core, playwright" >/dev/null 2>&1
}

if ! _python_has_browsergym; then
    if command -v conda >/dev/null 2>&1; then
        if conda env list | awk '$1 !~ /^#/ {print $1}' | grep -qx "$CONDA_ENV"; then
            eval "$(conda shell.bash hook)"
            conda activate "$CONDA_ENV"
            echo "[INFO] Activated conda env: $CONDA_ENV"
        else
            echo "[NOTE] conda env '$CONDA_ENV' not found; staying in current python."
            echo "       If imports fail, install via:"
            echo "         bash install/install_browsergym.sh"
        fi
    fi
fi

# ── Auto-source self-hosted-site env files (WebArena) ────────────────────
# install/install_webarena_sites.sh writes ``cold_start/webarena_env.sh``.
# If the file exists and the user requested tasks from that suite, source
# it so WA_* are exported.
#
# (Historical: a parallel ``visualwebarena_env.sh`` branch was here until
# 2026-05-03. VWA was dropped; see legacy/visualwebarena/README.md.)
_NEED_WEBARENA=0
_NEED_VISUALWEBARENA=0
for arg in "${EXTRA_ARGS[@]:-}"; do
    case "$arg" in
        browsergym/webarena.*)        _NEED_WEBARENA=1 ;;
        browsergym/visualwebarena.*)  _NEED_VISUALWEBARENA=1 ;;
    esac
done
if [ "$_NEED_WEBARENA" -eq 1 ] && [ -f "${SCRIPT_DIR}/webarena_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/webarena_env.sh"
    echo "[INFO] Sourced ${SCRIPT_DIR}/webarena_env.sh (WA_* exported)"
fi
if [ "$_NEED_VISUALWEBARENA" -eq 1 ]; then
    echo "[WARN] VisualWebArena tasks requested, but VWA support was dropped"
    echo "       on 2026-05-03. See legacy/visualwebarena/README.md for the"
    echo "       rationale and resurrection steps. Tasks may still run if"
    echo "       you manually source legacy/visualwebarena/visualwebarena_env.sh,"
    echo "       but expect the 10 known infra issues documented there."
fi

# ── Auto-wire MINIWOB_URL if any --tasks is browsergym/miniwob.* ──────────
# MiniWoB++ HTML pages are NOT shipped with the pip package; they live in a
# separate clone of Farama-Foundation/miniwob-plusplus that the launcher
# resolves automatically. Override with MINIWOB_URL=... or MINIWOB_HTML_DIR=...
_NEED_MINIWOB=0
for arg in "${EXTRA_ARGS[@]:-}"; do
    case "$arg" in
        browsergym/miniwob.*) _NEED_MINIWOB=1 ;;
    esac
done

if [ "$_NEED_MINIWOB" -eq 1 ] && [ -z "${MINIWOB_URL:-}" ]; then
    _candidates=()
    if [ -n "${MINIWOB_HTML_DIR:-}" ]; then
        _candidates+=("${MINIWOB_HTML_DIR}")
    fi
    _candidates+=(
        "/workspace/BrowserGym/miniwob-plusplus/miniwob/html/miniwob"
        "/workspace/miniwob-plusplus/miniwob/html/miniwob"
        "${HOME}/miniwob-plusplus/miniwob/html/miniwob"
    )
    for d in "${_candidates[@]}"; do
        if [ -d "$d" ] && [ -f "$d/click-button.html" ]; then
            export MINIWOB_URL="file://${d}/"
            echo "[INFO] MINIWOB_URL auto-set to ${MINIWOB_URL}"
            break
        fi
    done
    if [ -z "${MINIWOB_URL:-}" ]; then
        echo "[ERROR] You requested browsergym/miniwob.* tasks but MINIWOB_URL is unset"
        echo "        and miniwob-plusplus HTML pages were not found in any of:"
        for d in "${_candidates[@]}"; do echo "          - $d"; done
        echo "        Install with:"
        echo "          git clone https://github.com/Farama-Foundation/miniwob-plusplus.git \\"
        echo "              /workspace/BrowserGym/miniwob-plusplus"
        echo "          git -C /workspace/BrowserGym/miniwob-plusplus reset --hard \\"
        echo "              7fd85d71a4b60325c6585396ec4f48377d049838"
        echo "        Or export MINIWOB_URL=file:///path/to/miniwob/html/miniwob/"
        exit 1
    fi
fi

# ── Headless display (Playwright/Chromium needs an X server on Linux) ─────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb >/dev/null 2>&1 && ! pgrep -x Xvfb >/dev/null 2>&1; then
        Xvfb :99 -screen 0 1280x1024x24 &>/dev/null &
        sleep 0.5
    fi
    if command -v Xvfb >/dev/null 2>&1; then
        export DISPLAY="${DISPLAY:-:99}"
    fi
fi

# ── PYTHONPATH ────────────────────────────────────────────────────────────
PYPATH_ADD=("${CODEBASE_ROOT}" "${WORKSPACE_ROOT}")
JOINED_PYPATH="$(IFS=:; echo "${PYPATH_ADD[*]}")"
export PYTHONPATH="${JOINED_PYPATH}${PYTHONPATH:+:${PYTHONPATH}}"

# ── API key check ─────────────────────────────────────────────────────────
SKIP_VISION=0
for arg in "${EXTRA_ARGS[@]:-}"; do
    case "$arg" in
        --no_vision|--no-vision) SKIP_VISION=1 ;;
    esac
done
if [ "$SKIP_VISION" -eq 0 ] \
    && [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[NOTE] Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set."
    echo "       The launcher will try api_keys.py next to the repo root;"
    echo "       otherwise pass --api_key or --no_vision."
fi

# ── Banner ────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Cold-Start Actor Agent — BrowserGym + gpt-5.5"
echo "================================================================"
echo "  Codebase:    ${CODEBASE_ROOT}"
echo "  Workspace:   ${WORKSPACE_ROOT}"
echo "  DISPLAY:     ${DISPLAY:-unset}"
echo "  Python:      $(python3 --version 2>&1)"

# ── Dependency probe (don't auto-install — keep the launcher hermetic) ────
if python3 -c "import openai" >/dev/null 2>&1; then
    :
else
    echo "[ERROR] 'openai' package missing. Install: pip install openai"
    exit 1
fi

SMOKE_OUTPUT="$(python3 - <<PYEOF 2>&1 || true
import warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, "${CODEBASE_ROOT}")
sys.path.insert(0, "${WORKSPACE_ROOT}")
ok = []
fail = []
try:
    from browsergym_wrapper.heuristic import obs_to_schema  # noqa: F401
    ok.append("browsergym_wrapper.heuristic")
except Exception as exc:
    fail.append(f"browsergym_wrapper.heuristic: {exc}")
try:
    from browsergym_wrapper.tools import build_browser_registry  # noqa: F401
    ok.append("browsergym_wrapper.tools")
except Exception as exc:
    print(f"WARN browsergym_wrapper.tools unavailable: {exc}", file=sys.stderr)
try:
    from vlm_wrapper.schema import build_system_prompt, build_user_message  # noqa: F401
    ok.append("vlm_wrapper.schema")
except Exception as exc:
    fail.append(f"vlm_wrapper.schema: {exc}")
try:
    import gymnasium  # noqa: F401
    import browsergym.core  # noqa: F401
    ok.append("browsergym.core")
except Exception as exc:
    fail.append(f"browsergym.core (REQUIRED — install via install/install_browsergym.sh): {exc}")
try:
    import playwright  # noqa: F401
    ok.append("playwright")
except Exception as exc:
    fail.append(f"playwright (REQUIRED — install via install/install_browsergym.sh): {exc}")

# Optional task suites — best-effort import for visibility.
suite_ok, suite_fail = [], []
for mod in ("browsergym.miniwob", "browsergym.webarena",
            "browsergym.visualwebarena", "browsergym.assistantbench",
            "browsergym.workarena"):
    try:
        __import__(mod)
        suite_ok.append(mod.split(".",1)[1])
    except Exception as exc:
        suite_fail.append(f"{mod.split('.',1)[1]}({type(exc).__name__})")

# Count registered task ids per suite for the banner.
import gymnasium as _gym
counts = {}
for k in _gym.envs.registry.keys():
    if not k.startswith("browsergym/"):
        continue
    p = k.split("/",1)[1].split(".")[0] or "openended"
    counts[p] = counts.get(p, 0) + 1
counts_str = ", ".join(f"{p}={counts[p]}" for p in sorted(counts)) or "(none)"

print("imports_ok=" + ",".join(ok))
print("suites_ok=" + ",".join(suite_ok))
if suite_fail:
    print("suites_unavailable=" + ",".join(suite_fail))
print("task_counts=" + counts_str)
for f in fail:
    print(f"FAIL {f}", file=sys.stderr)
PYEOF
)"
echo "  ${SMOKE_OUTPUT}" | sed 's/^/  /'

if echo "${SMOKE_OUTPUT}" | grep -q '^FAIL '; then
    echo ""
    echo "[ERROR] Required BrowserGym/Playwright dependencies are missing."
    echo "        Install with: bash install/install_browsergym.sh"
    echo "        Then activate the env:  conda activate browsergym"
    exit 1
fi

[ -n "${OPENROUTER_API_KEY:-}" ] && echo "  API key:     ${OPENROUTER_API_KEY:0:12}... (OpenRouter)"
[ -z "${OPENROUTER_API_KEY:-}" ] && [ -n "${OPENAI_API_KEY:-}" ] && echo "  API key:     ${OPENAI_API_KEY:0:12}... (OpenAI)"
echo "================================================================"
echo ""

# ── Defaults: when the user passes nothing, run the default URLs with
#               --resume + verbose stepping. Headless is always the default
#               (set in the python launcher); pass --save_frames explicitly
#               if you want PNGs + per-step JSON sidecars on disk.
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--resume -v)
fi

# Surface frame-saving status in the banner before spawning the python agent.
SAVE_FRAMES=0
NO_HEADLESS=0
for arg in "${EXTRA_ARGS[@]:-}"; do
    case "$arg" in
        --save_frames|--save-frames) SAVE_FRAMES=1 ;;
        --no_headless|--no-headless) NO_HEADLESS=1 ;;
    esac
done
echo "  Headless:    $([ $NO_HEADLESS -eq 0 ] && echo 'YES (default)' || echo 'NO (--no_headless)')"
echo "  Save frames: $([ $SAVE_FRAMES -eq 1 ] && echo 'YES (PNG + step_NNN.json sidecar)' \
                                              || echo 'NO  (pass --save_frames to enable)')"
echo "================================================================"
echo ""

python3 "${SCRIPT_DIR}/generate_cold_start_actor_browsergym.py" "${EXTRA_ARGS[@]}"
EXIT_CODE=$?

# ── Post-run summary ──────────────────────────────────────────────────────
# Honor an explicit ``--output_dir`` in the forwarded args; fall back to
# Cold-start-out-browsergym at the repo root.
OUTPUT_DIR="${CODEBASE_ROOT}/Cold-start-out-browsergym"
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
echo "  Cold-Start Actor (BrowserGym) — Post-Run Summary"
echo "================================================================"

if [ -d "$OUTPUT_DIR" ]; then
    TOTAL=0
    for tgt_dir in "$OUTPUT_DIR"/*/; do
        [ -d "$tgt_dir" ] || continue
        safe_id="$(basename "$tgt_dir")"
        count=$(find "$tgt_dir" -maxdepth 1 -name 'episode_*.json' ! -name 'episode_buffer.json' 2>/dev/null | wc -l)
        TOTAL=$((TOTAL + count))
        has_buffer="no"; [ -f "$tgt_dir/episode_buffer.json" ] && has_buffer="yes"
        has_jsonl="no";  [ -f "$tgt_dir/rollouts.jsonl" ]      && has_jsonl="yes"
        printf "  %-44s %3d episodes  buffer=%s  jsonl=%s\n" \
            "$safe_id" "$count" "$has_buffer" "$has_jsonl"
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
echo "    eps = load_episodes_from_jsonl('${OUTPUT_DIR}/<safe_id>/rollouts.jsonl')"
echo "================================================================"

exit ${EXIT_CODE}
