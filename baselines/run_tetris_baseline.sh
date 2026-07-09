#!/usr/bin/env bash
# ======================================================================
#  Tetris baseline: LLM API model plays Tetris (macro-action wrapper)
#
#  Uses the SAME wrapper chain as training:
#    make_gaming_env("tetris") → GamingAgentNLWrapper → TetrisMacroActionWrapper
#
#  Supported models (via OpenRouter or direct API):
#    gpt-5.4, openai/gpt-oss-120b, google/gemini-3.1-pro-preview,
#    anthropic/claude-4.6-sonnet-20260217
#
#  Usage:
#    bash baselines/run_tetris_baseline.sh
#    bash baselines/run_tetris_baseline.sh --model openai/gpt-oss-120b
#    EPISODES=16 bash baselines/run_tetris_baseline.sh --model gpt-5.4
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="gpt-5.4"
while [[ $# -gt 0 ]]; do
    case "$1" in --model) MODEL="$2"; shift 2 ;; *) break ;; esac
done

EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-200}"
TEMPERATURE="${TEMPERATURE:-0.3}"
SEED="${SEED:-42}"

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

[ -z "${OPENROUTER_API_KEY:-}" ] && echo "Warning: OPENROUTER_API_KEY not set. See .env.example."
[ -z "${OPENAI_API_KEY:-}" ] && echo "Warning: OPENAI_API_KEY not set. See .env.example."

MODEL_TAG="${MODEL//\//_}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/${MODEL_TAG}_tetris_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  ${MODEL} Baseline — Tetris (${EPISODES} episodes)"
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
echo "  ${MODEL} Tetris Baseline COMPLETE — ${OUTPUT_DIR}"
