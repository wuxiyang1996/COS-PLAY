#!/usr/bin/env bash
# Claude Sonnet 4.6 baseline on Avalon (40 episodes = 8/player x 5) via OpenRouter
# Controlled player = Claude 4.6, all opponents = GPT-5.4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

AGENTEVOLVER_ROOT=""
for candidate in "$WORKSPACE_ROOT/AgentEvolver" "$PROJECT_ROOT/AgentEvolver"; do
    [ -d "$candidate" ] && AGENTEVOLVER_ROOT="$candidate" && break
done
[ -z "$AGENTEVOLVER_ROOT" ] && echo "[ERROR] AgentEvolver repo not found." && exit 1

NUM_PLAYERS=5
EPISODES_PER_PLAYER="${EPISODES_PER_PLAYER:-8}"
EPISODES=$((NUM_PLAYERS * EPISODES_PER_PLAYER))
TEMPERATURE="${TEMPERATURE:-0.4}"
MODEL="${MODEL:-anthropic/claude-4.6-sonnet-20260217}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
SEED="${SEED:-42}"

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTHONPATH="${PROJECT_ROOT}:${AGENTEVOLVER_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── API key ───────────────────────────────────────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    OPENROUTER_API_KEY="$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from api_keys import open_router_api_key; print(open_router_api_key or '')
" 2>/dev/null || echo "")"
    export OPENROUTER_API_KEY
fi
[ -z "${OPENAI_API_KEY:-}" ] && export OPENAI_API_KEY="${OPENROUTER_API_KEY:-}"
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] No API key found."; exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/claude46_avalon_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  Claude Sonnet 4.6 Baseline — Avalon (${EPISODES} episodes = ${EPISODES_PER_PLAYER}/player x ${NUM_PLAYERS})"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:     ${MODEL}  (controlled player)"
echo "  Opponents: ${OPPONENT_MODEL}"
echo "  Episodes:  ${EPISODES}    Temperature: ${TEMPERATURE}"
echo "  Mode:      per-role (cycle through player positions 0-4)"
echo "  Output:    ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 "${PROJECT_ROOT}/cold_start/generate_cold_start_evolver.py" \
    --games avalon \
    --episodes "${EPISODES}" \
    --model "${MODEL}" \
    --opponent_model "${OPPONENT_MODEL}" \
    --temperature "${TEMPERATURE}" \
    --num_players "${NUM_PLAYERS}" \
    --seed "${SEED}" \
    --per_role \
    --no_label \
    --verbose \
    --output_dir "${OUTPUT_DIR}" \
    "$@"

echo ""
echo "  Claude Sonnet 4.6 Avalon Baseline COMPLETE — ${OUTPUT_DIR}"
