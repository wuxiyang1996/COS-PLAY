#!/usr/bin/env bash
#
# run_gpt54_diplomacy.sh — GPT-5.4 baseline on Diplomacy
#
# Uses the SAME environment wrapper and game settings as the training
# script (scripts/run_diplomacy.sh → run_coevolution.py → episode_runner.py):
#
#   DiplomacyNLWrapper(max_phases=20)
#     → 7-player strategic board game (classic map)
#     → Powers: Austria, England, France, Germany, Italy, Russia, Turkey
#     → Phase cycle: Spring Move → Spring Retreat → Fall Move →
#                    Fall Retreat → Fall Adjustment → next year
#     → 20 max phases/episode
#     → End condition: solo victory or max phases reached
#
# GPT-5.4 controls all 7 powers independently with structured
# chain-of-thought reasoning via function calling (submit_orders).
# Per-power reasoning traces are stored in the Episode's intentions field.
#
# Key settings (from scripts/run_diplomacy.sh & trainer/coevolution/config.py):
#   - 7 powers on classic Diplomacy map
#   - 20 max phases/episode (DiplomacyConfig.max_phases)
#   - Reward: supply_centers/18 + potential-based shaping (+0.5/gained centre)
#   - Negotiation support: message exchange before order submission
#   - End condition: game.is_game_done or phases >= 20
#
# Output: baselines/output/gpt54_diplomacy_<timestamp>/diplomacy/
#   - episode_NNN.json        Individual episode data
#   - episode_buffer.json     All episodes in Episode_Buffer format
#   - rollouts.jsonl          Append-friendly JSONL
#   - rollout_summary.json    Aggregate run stats
#
# Usage:
#   bash baselines/run_gpt54_diplomacy.sh                # 20 episodes
#   EPISODES=5 bash baselines/run_gpt54_diplomacy.sh     # Quick test
#   bash baselines/run_gpt54_diplomacy.sh --resume       # Resume interrupted run

set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

# AgentEvolver: sibling of Game-AI-Agent or child
AGENTEVOLVER_ROOT=""
for candidate in "$WORKSPACE_ROOT/AgentEvolver" "$PROJECT_ROOT/AgentEvolver"; do
    if [ -d "$candidate" ]; then
        AGENTEVOLVER_ROOT="$candidate"
        break
    fi
done

if [ -z "$AGENTEVOLVER_ROOT" ]; then
    echo "[ERROR] AgentEvolver repo not found as a sibling or child of Game-AI-Agent."
    echo "        Expected at: $WORKSPACE_ROOT/AgentEvolver or $PROJECT_ROOT/AgentEvolver"
    exit 1
fi

# AI_Diplomacy: sibling of Game-AI-Agent or child
AI_DIPLOMACY_ROOT=""
for candidate in "$WORKSPACE_ROOT/AI_Diplomacy" "$PROJECT_ROOT/AI_Diplomacy"; do
    if [ -d "$candidate" ]; then
        AI_DIPLOMACY_ROOT="$candidate"
        break
    fi
done

if [ -z "$AI_DIPLOMACY_ROOT" ]; then
    echo "[WARN] AI_Diplomacy repo not found. DiplomacyNLWrapper may fail to import."
    echo "       Expected at: $WORKSPACE_ROOT/AI_Diplomacy or $PROJECT_ROOT/AI_Diplomacy"
fi

# ── Settings ──────────────────────────────────────────────────────────────────
NUM_POWERS=7
EPISODES_PER_POWER="${EPISODES_PER_POWER:-8}"
EPISODES="${EPISODES:-$((NUM_POWERS * EPISODES_PER_POWER))}"
TEMPERATURE="${TEMPERATURE:-0.4}"
MODEL="${MODEL:-gpt-5.4}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
SEED="${SEED:-42}"

# ── Headless rendering (from scripts/run_diplomacy.sh) ─────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── PYTHONPATH (matches scripts/run_diplomacy.sh) ──────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${AGENTEVOLVER_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"
[ -n "$AI_DIPLOMACY_ROOT" ] && export PYTHONPATH="${AI_DIPLOMACY_ROOT}:${PYTHONPATH}"

# ── API key resolution (same as run_gpt54_tetris.sh / run_coldstart_evolver.sh)
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    OPENROUTER_API_KEY="$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from api_keys import open_router_api_key; print(open_router_api_key or '')
" 2>/dev/null || echo "")"
    export OPENROUTER_API_KEY
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    OPENAI_API_KEY="${OPENROUTER_API_KEY:-$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from api_keys import openai_api_key; print(openai_api_key or '')
" 2>/dev/null || echo "")}"
    export OPENAI_API_KEY
fi

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] No API key found. Set OPENROUTER_API_KEY or open_router_api_key in api_keys.py"
    exit 1
fi

# ── Dependency check ───────────────────────────────────────────────────────
MISSING_DEPS=()
for pkg in openai tiktoken; do
    python3 -c "import $pkg" 2>/dev/null || MISSING_DEPS+=("$pkg")
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "[INFO] Installing missing Python packages: ${MISSING_DEPS[*]}"
    pip install --quiet openai tiktoken 2>/dev/null || true
fi

# ── Output directory ───────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/gpt54_diplomacy_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

# ── Banner ─────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  GPT-5.4 Baseline — Diplomacy (${EPISODES} episodes = ${EPISODES_PER_POWER}/power x ${NUM_POWERS})"
echo "══════════════════════════════════════════════════════════════"
echo "  Project root:   ${PROJECT_ROOT}"
echo "  AgentEvolver:   ${AGENTEVOLVER_ROOT}"
[ -n "$AI_DIPLOMACY_ROOT" ] && echo "  AI_Diplomacy:   ${AI_DIPLOMACY_ROOT}" || echo "  AI_Diplomacy:   (not found)"
echo "  Model:          ${MODEL}  (controlled power)"
echo "  Opponents:      ${OPPONENT_MODEL}"
echo "  Episodes:       ${EPISODES}"
echo "  Mode:           per-power (cycle through 7 powers)"
echo "  Temperature:    ${TEMPERATURE}"
echo "  Seed:           ${SEED}"
echo "  Output:         ${OUTPUT_DIR}"
[ -n "${OPENROUTER_API_KEY:-}" ] && echo "  API key:        ${OPENROUTER_API_KEY:0:12}... (OpenRouter)" || echo "  API key:        ${OPENAI_API_KEY:0:12}..."
echo ""
echo "  Diplomacy game profile:"
echo "    - 7-player strategic board game (classic map)"
echo "    - Powers: Austria, England, France, Germany, Italy, Russia, Turkey"
echo "    - 20 max phases/episode"
echo "    - Controlled power = ${MODEL}, opponents = ${OPPONENT_MODEL}"
echo "    - Chain-of-thought reasoning via function calling"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Collect extra args passed on the command line ──────────────────────────
EXTRA_CLI_ARGS=()
for arg in "$@"; do
    EXTRA_CLI_ARGS+=("$arg")
done

# ── Run rollouts via generate_cold_start_evolver.py ────────────────────────
python3 "${PROJECT_ROOT}/cold_start/generate_cold_start_evolver.py" \
    --games diplomacy \
    --episodes "${EPISODES}" \
    --model "${MODEL}" \
    --opponent_model "${OPPONENT_MODEL}" \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --per_power \
    --no_label \
    --verbose \
    --output_dir "${OUTPUT_DIR}" \
    "${EXTRA_CLI_ARGS[@]+"${EXTRA_CLI_ARGS[@]}"}"

# ── Final summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  GPT-5.4 Diplomacy Baseline COMPLETE"
echo "══════════════════════════════════════════════════════════════"
echo "  Output:         ${OUTPUT_DIR}"
echo "  Episodes:       ${OUTPUT_DIR}/diplomacy/episode_*.json"
echo "  Episode buffer: ${OUTPUT_DIR}/diplomacy/episode_buffer.json"
echo "  Rollouts JSONL: ${OUTPUT_DIR}/diplomacy/rollouts.jsonl"
echo "  Summary:        ${OUTPUT_DIR}/diplomacy/rollout_summary.json"

if [ -f "${OUTPUT_DIR}/diplomacy/rollout_summary.json" ]; then
    echo ""
    python3 -c "
import json
with open('${OUTPUT_DIR}/diplomacy/rollout_summary.json') as f:
    s = json.load(f)
print(f'  Total episodes:  {s.get(\"total_episodes\", \"?\")}')
print(f'  Mean reward:     {s.get(\"mean_reward\", 0):.2f}')
print(f'  Mean steps:      {s.get(\"mean_steps\", 0):.1f}')
print(f'  Elapsed:         {s.get(\"elapsed_seconds\", 0):.1f}s')
" 2>/dev/null || true
fi

echo "══════════════════════════════════════════════════════════════"
