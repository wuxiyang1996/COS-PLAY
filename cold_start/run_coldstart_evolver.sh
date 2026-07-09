#!/usr/bin/env bash
#
# run_coldstart_evolver.sh — GPT-5.4 cold-start rollouts for Avalon & Diplomacy
#
# Generates decision-making agent trajectories for the multi-agent evolver
# games (Avalon, Diplomacy) using GPT-5.4 as the backbone model.  Handles
# environment setup, dependency checks, and rollout generation.
#
# Output lands in cold_start/output/gpt54_evolver/<game_name>/ with
# Episode/Experience format ready for the co-evolution framework.
#
# Usage:
#   bash cold_start/run_coldstart_evolver.sh                        # Both games, 20 eps each
#   bash cold_start/run_coldstart_evolver.sh --episodes 5           # Quick test
#   bash cold_start/run_coldstart_evolver.sh --games avalon         # Avalon only
#   bash cold_start/run_coldstart_evolver.sh --games diplomacy      # Diplomacy only
#   bash cold_start/run_coldstart_evolver.sh --resume               # Resume interrupted run
#   bash cold_start/run_coldstart_evolver.sh --help                 # Show all options

set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$CODEBASE_ROOT/.." && pwd)"

# AgentEvolver: sibling of Game-AI-Agent or child
AGENTEVOLVER_ROOT=""
for candidate in "$WORKSPACE_ROOT/AgentEvolver" "$CODEBASE_ROOT/AgentEvolver"; do
    if [ -d "$candidate" ]; then
        AGENTEVOLVER_ROOT="$candidate"
        break
    fi
done

if [ -z "$AGENTEVOLVER_ROOT" ]; then
    echo "[ERROR] AgentEvolver repo not found as a sibling or child of Game-AI-Agent."
    echo "        Expected at: $WORKSPACE_ROOT/AgentEvolver or $CODEBASE_ROOT/AgentEvolver"
    exit 1
fi

# AI_Diplomacy: sibling of Game-AI-Agent or child (needed for diplomacy)
AI_DIPLOMACY_ROOT=""
for candidate in "$WORKSPACE_ROOT/AI_Diplomacy" "$CODEBASE_ROOT/AI_Diplomacy"; do
    if [ -d "$candidate" ]; then
        AI_DIPLOMACY_ROOT="$candidate"
        break
    fi
done

# ── Environment ────────────────────────────────────────────────────────────
export PYTHONPATH="${CODEBASE_ROOT}:${AGENTEVOLVER_ROOT}:${PYTHONPATH:-}"
[ -n "$AI_DIPLOMACY_ROOT" ] && export PYTHONPATH="${AI_DIPLOMACY_ROOT}:${PYTHONPATH}"

# Prefer OpenRouter key from api_keys.py; fallback to OPENAI_API_KEY
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    OPENROUTER_API_KEY="$(python3 -c "
import sys; sys.path.insert(0, '${CODEBASE_ROOT}')
from api_keys import open_router_api_key; print(open_router_api_key or '')
" 2>/dev/null || echo "")"
    export OPENROUTER_API_KEY
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    OPENAI_API_KEY="${OPENROUTER_API_KEY:-$(python3 -c "
import sys; sys.path.insert(0, '${CODEBASE_ROOT}')
from api_keys import openai_api_key; print(openai_api_key or '')
" 2>/dev/null || echo "")}"
    export OPENAI_API_KEY
fi

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] No API key found. Set OPENROUTER_API_KEY or open_router_api_key in api_keys.py (preferred), or OPENAI_API_KEY"
    exit 1
fi

# ── Dependency check ───────────────────────────────────────────────────────
echo "================================================================"
echo "  GPT-5.4 Avalon & Diplomacy — Cold-Start Environment Check"
echo "================================================================"

MISSING_DEPS=()
for pkg in openai tiktoken; do
    python3 -c "import $pkg" 2>/dev/null || MISSING_DEPS+=("$pkg")
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "[INFO] Installing missing Python packages: ${MISSING_DEPS[*]}"
    pip install --quiet openai tiktoken 2>/dev/null || true
fi

# Smoke-test: can we import the evolver env wrappers?
AVAIL_GAMES="$(python3 -c "
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '${CODEBASE_ROOT}')
sys.path.insert(0, '${AGENTEVOLVER_ROOT}')
${AI_DIPLOMACY_ROOT:+sys.path.insert(0, '${AI_DIPLOMACY_ROOT}')}
avail = []
try:
    from env_wrappers.avalon_nl_wrapper import AvalonNLWrapper
    if AvalonNLWrapper is not None:
        avail.append('avalon')
except Exception:
    pass
try:
    from env_wrappers.diplomacy_nl_wrapper import DiplomacyNLWrapper
    if DiplomacyNLWrapper is not None:
        avail.append('diplomacy')
except Exception:
    pass
print(' '.join(avail))
" 2>/dev/null)"

if [ -z "$AVAIL_GAMES" ]; then
    echo "[ERROR] Neither Avalon nor Diplomacy env wrappers could be imported."
    echo "        Check AgentEvolver / AI_Diplomacy installation and dependencies."
    exit 1
fi

echo "  Codebase root:   $CODEBASE_ROOT"
echo "  AgentEvolver:    $AGENTEVOLVER_ROOT"
[ -n "$AI_DIPLOMACY_ROOT" ] && echo "  AI_Diplomacy:    $AI_DIPLOMACY_ROOT" || echo "  AI_Diplomacy:    (not found — diplomacy will be skipped)"
echo "  Available games: $AVAIL_GAMES"
echo "  Model:           gpt-5.4"
[ -n "${OPENROUTER_API_KEY:-}" ] && echo "  API key:         ${OPENROUTER_API_KEY:0:12}... (OpenRouter)" || echo "  API key:         ${OPENAI_API_KEY:0:12}..."
echo "================================================================"
echo ""

# ── Run rollouts ───────────────────────────────────────────────────────────
# Defaults: --episodes 20 --model gpt-5.4 --no_label --resume
# Both games use their natural end conditions (no artificial max_steps cap):
#   avalon:    engine.done (3 quest fails or assassination resolves)
#   diplomacy: game.is_game_done or phases >= 20 (DiplomacyConfig.max_phases)
EXTRA_ARGS=("$@")

if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--episodes 20 --model gpt-5.4 --no_label --resume)
fi

python3 "${SCRIPT_DIR}/generate_cold_start_evolver.py" "${EXTRA_ARGS[@]}"

# ── Post-run verification ─────────────────────────────────────────────────
OUTPUT_DIR="${SCRIPT_DIR}/output/gpt54_evolver"
echo ""
echo "================================================================"
echo "  GPT-5.4 Avalon & Diplomacy — Post-Run Summary"
echo "================================================================"

TOTAL_EPISODES=0
if [ -d "$OUTPUT_DIR" ]; then
    for game_dir in "$OUTPUT_DIR"/*/; do
        [ -d "$game_dir" ] || continue
        game="$(basename "$game_dir")"
        count=$(find "$game_dir" -maxdepth 1 -name 'episode_*.json' ! -name 'episode_buffer.json' | wc -l)
        TOTAL_EPISODES=$((TOTAL_EPISODES + count))
        has_buffer="no"; [ -f "$game_dir/episode_buffer.json" ] && has_buffer="yes"
        has_jsonl="no";  [ -f "$game_dir/rollouts.jsonl" ]      && has_jsonl="yes"
        printf "  %-25s %3d episodes  buffer=%s  jsonl=%s\n" "$game" "$count" "$has_buffer" "$has_jsonl"
    done
fi

echo ""
echo "  Total episodes: $TOTAL_EPISODES"
echo "  Output dir:     $OUTPUT_DIR"
echo ""
echo "  Load into skill pipeline:"
echo "    from cold_start.load_rollouts import load_all_game_rollouts"
echo "    rollouts = load_all_game_rollouts('cold_start/output/gpt54_evolver')"
echo ""
echo "  Load into trainer:"
echo "    from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records"
echo "    eps = load_episodes_from_jsonl('cold_start/output/gpt54_evolver/<game>/rollouts.jsonl')"
echo "    records = episodes_to_rollout_records(eps)"
echo "================================================================"
