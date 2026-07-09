#!/usr/bin/env bash
#
# run_coldstart_gpt54.sh — GPT-5.4 base agent cold-start rollout generation
#
# Generates decision-making agent trajectories for LM-Game Bench using
# GPT-5.4 as the backbone model. Handles environment setup, dependency
# checks, and rollout generation.
#
# Output lands in cold_start/output/gpt54/<game_name>/ with Episode/Experience
# format ready for the co-evolution framework (skill pipeline + trainer).
#
# Usage:
#   bash cold_start/run_coldstart_gpt54.sh                     # All games, 100 eps each
#   bash cold_start/run_coldstart_gpt54.sh --episodes 5        # Quick test (5 eps)
#   bash cold_start/run_coldstart_gpt54.sh --resume            # Resume interrupted run
#   bash cold_start/run_coldstart_gpt54.sh --games tetris sokoban
#   bash cold_start/run_coldstart_gpt54.sh --help              # Show all options

set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAMINGAGENT_ROOT="$(cd "$CODEBASE_ROOT/../GamingAgent" 2>/dev/null && pwd || echo "")"

if [ -z "$GAMINGAGENT_ROOT" ] || [ ! -d "$GAMINGAGENT_ROOT" ]; then
    echo "[ERROR] GamingAgent repo not found at $CODEBASE_ROOT/../GamingAgent"
    echo "        Clone it as a sibling: git clone <url> alongside Game-AI-Agent"
    exit 1
fi

# ── Environment ────────────────────────────────────────────────────────────
export PYTHONPATH="${CODEBASE_ROOT}:${GAMINGAGENT_ROOT}:${PYTHONPATH:-}"

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
echo "  GPT-5.4 Base Agent — Cold-Start Environment Check"
echo "================================================================"

MISSING_DEPS=()
for pkg in openai anthropic google.genai tiktoken gymnasium pygame; do
    python3 -c "import $pkg" 2>/dev/null || MISSING_DEPS+=("$pkg")
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "[INFO] Installing missing Python packages: ${MISSING_DEPS[*]}"
    pip install --quiet openai anthropic google-genai tiktoken gymnasium pygame \
        opencv-python-headless imageio natsort psutil aiohttp termcolor \
        pettingzoo tile-match-gym rlcard 2>/dev/null || true
fi

# Smoke-test: can we import the registry?
AVAIL_GAMES="$(python3 -c "
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '${CODEBASE_ROOT}'); sys.path.insert(0, '${GAMINGAGENT_ROOT}')
from cold_start.generate_cold_start import GAME_REGISTRY
avail = [n for n, r in GAME_REGISTRY.items() if r['env_class'] is not None]
print(' '.join(avail))
" 2>/dev/null)"

if [ -z "$AVAIL_GAMES" ]; then
    echo "[ERROR] No game environments could be imported."
    echo "        Check GamingAgent installation and dependencies."
    exit 1
fi

echo "  Codebase root:   $CODEBASE_ROOT"
echo "  GamingAgent:     $GAMINGAGENT_ROOT"
echo "  Available games: $AVAIL_GAMES"
echo "  Model:           gpt-5.4"
[ -n "${OPENROUTER_API_KEY:-}" ] && echo "  API key:         ${OPENROUTER_API_KEY:0:12}... (OpenRouter)" || echo "  API key:         ${OPENAI_API_KEY:0:12}..."
echo "================================================================"
echo ""

# ── Run rollouts ───────────────────────────────────────────────────────────
# Defaults: --episodes 100 (no --max_steps: run until natural end per game) --model gpt-5.4 --no_label --resume
EXTRA_ARGS=("$@")

if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--episodes 100 --model gpt-5.4 --no_label --resume)
fi

python3 "${SCRIPT_DIR}/generate_cold_start_gpt54.py" "${EXTRA_ARGS[@]}"

# ── Post-run verification ─────────────────────────────────────────────────
OUTPUT_DIR="${SCRIPT_DIR}/output/gpt54"
echo ""
echo "================================================================"
echo "  GPT-5.4 Base Agent — Post-Run Summary"
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
echo "    rollouts = load_all_game_rollouts('cold_start/output/gpt54')"
echo ""
echo "  Load into trainer:"
echo "    from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records"
echo "    eps = load_episodes_from_jsonl('cold_start/output/gpt54/<game>/rollouts.jsonl')"
echo "    records = episodes_to_rollout_records(eps)"
echo "================================================================"
