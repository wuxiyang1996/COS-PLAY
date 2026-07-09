#!/usr/bin/env bash
# Gemini 3.1 Pro baseline on Tetris (8 episodes) via OpenRouter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAMINGAGENT_ROOT="$(cd "$PROJECT_ROOT/../GamingAgent" 2>/dev/null && pwd || echo "")"

if [ -z "$GAMINGAGENT_ROOT" ] || [ ! -d "$GAMINGAGENT_ROOT" ]; then
    echo "[ERROR] GamingAgent repo not found at $PROJECT_ROOT/../GamingAgent"; exit 1
fi

EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-200}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MODEL="${MODEL:-google/gemini-3.1-pro-preview}"
SEED="${SEED:-42}"

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTHONPATH="${PROJECT_ROOT}:${GAMINGAGENT_ROOT}:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

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
    echo "[ERROR] No API key found. Set OPENROUTER_API_KEY or open_router_api_key in api_keys.py"; exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/gemini31pro_tetris_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  Gemini 3.1 Pro Baseline — Tetris (${EPISODES} episodes)"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:     ${MODEL}"
echo "  Episodes:  ${EPISODES}    Max steps: ${MAX_STEPS}"
echo "  Output:    ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 "${SCRIPT_DIR}/run_gpt54_tetris_macro.py" \
    --episodes "${EPISODES}" \
    --max_steps "${MAX_STEPS}" \
    --model "${MODEL}" \
    --temperature "${TEMPERATURE}" \
    --verbose \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "  Gemini 3.1 Pro Tetris Baseline COMPLETE — ${OUTPUT_DIR}"
