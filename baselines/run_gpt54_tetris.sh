#!/usr/bin/env bash
#
# run_gpt54_tetris.sh — GPT-5.4 baseline on Tetris (8 episodes)
#
# Uses the SAME environment wrapper chain as the training script
# (scripts/run_tetris.sh → run_coevolution.py → episode_runner.py):
#
#   make_gaming_env("tetris", max_steps=200)
#     → GamingAgentNLWrapper
#       → TetrisMacroActionWrapper
#
# Each GPT-5.4 decision places one entire piece (rotation + column),
# matching the macro-action semantics of training exactly.
#
# Key settings (from trainer/coevolution/config.py & episode_runner.py):
#   - GAME_MAX_STEPS["tetris"] = 200
#   - Action: placement-level macro actions via TetrisMacroActionWrapper
#   - Board: 10×20, gravity=True
#   - Reward: +1 per piece placed, +10 per line cleared
#
# Output: baselines/output/gpt54_tetris_<timestamp>/
#   - episode_NNN.json      Individual episode data
#   - reward_summary.json   Aggregate reward stats
#   - reward_report.txt     Human-readable reward table
#
# Usage:
#   bash baselines/run_gpt54_tetris.sh              # 8 episodes, 200 steps
#   EPISODES=4 bash baselines/run_gpt54_tetris.sh   # Override episode count
#   MAX_STEPS=100 bash baselines/run_gpt54_tetris.sh

set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAMINGAGENT_ROOT="$(cd "$PROJECT_ROOT/../GamingAgent" 2>/dev/null && pwd || echo "")"

if [ -z "$GAMINGAGENT_ROOT" ] || [ ! -d "$GAMINGAGENT_ROOT" ]; then
    echo "[ERROR] GamingAgent repo not found at $PROJECT_ROOT/../GamingAgent"
    echo "        Clone it as a sibling: git clone <url> alongside Game-AI-Agent"
    exit 1
fi

# ── Settings matching training (trainer/coevolution/config.py) ─────────────
EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-200}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MODEL="${MODEL:-gpt-5.4}"

# ── Headless rendering (from scripts/run_tetris.sh) ───────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── PYTHONPATH (matches scripts/run_tetris.sh) ────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${GAMINGAGENT_ROOT}:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── API key resolution ────────────────────────────────────────────────────
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
for pkg in openai tiktoken gymnasium pygame; do
    python3 -c "import $pkg" 2>/dev/null || MISSING_DEPS+=("$pkg")
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "[INFO] Installing missing Python packages: ${MISSING_DEPS[*]}"
    pip install --quiet openai tiktoken gymnasium pygame \
        opencv-python-headless imageio natsort psutil aiohttp termcolor 2>/dev/null || true
fi

# ── Output directory ───────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SCRIPT_DIR}/output/gpt54_tetris_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# ── Banner ─────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  GPT-5.4 Baseline — Tetris (${EPISODES} episodes)"
echo "══════════════════════════════════════════════════════════════"
echo "  Project root:   ${PROJECT_ROOT}"
echo "  GamingAgent:    ${GAMINGAGENT_ROOT}"
echo "  Model:          ${MODEL}"
echo "  Episodes:       ${EPISODES}"
echo "  Max steps:      ${MAX_STEPS}"
echo "  Temperature:    ${TEMPERATURE}"
echo "  Output:         ${OUTPUT_DIR}"
[ -n "${OPENROUTER_API_KEY:-}" ] && echo "  API key:        ${OPENROUTER_API_KEY:0:12}... (OpenRouter)" || echo "  API key:        ${OPENAI_API_KEY:0:12}..."
echo ""
echo "  Env chain (same as training):"
echo "    make_gaming_env('tetris', max_steps=${MAX_STEPS})"
echo "      → GamingAgentNLWrapper"
echo "        → TetrisMacroActionWrapper  (placement-level actions)"
echo ""
echo "  Settings (from trainer/coevolution/config.py):"
echo "    - GAME_MAX_STEPS['tetris'] = ${MAX_STEPS}"
echo "    - 10×20 board, gravity=True"
echo "    - Reward: +1 per piece placed, +10 per line cleared"
echo "    - Macro actions: each decision places one entire piece"
echo "    - Headless: PYGLET_HEADLESS=1, SDL_VIDEODRIVER=dummy"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Run rollouts ───────────────────────────────────────────────────────────
python3 "${SCRIPT_DIR}/run_gpt54_tetris_macro.py" \
    --episodes "${EPISODES}" \
    --max_steps "${MAX_STEPS}" \
    --model "${MODEL}" \
    --temperature "${TEMPERATURE}" \
    --verbose \
    --output_dir "${OUTPUT_DIR}"

# ── Final summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  GPT-5.4 Tetris Baseline COMPLETE"
echo "══════════════════════════════════════════════════════════════"
echo "  Output:         ${OUTPUT_DIR}"
echo "  Episodes:       ${OUTPUT_DIR}/episode_*.json"
echo "  Reward report:  ${OUTPUT_DIR}/reward_report.txt"
echo "  Reward JSON:    ${OUTPUT_DIR}/reward_summary.json"
echo "══════════════════════════════════════════════════════════════"
