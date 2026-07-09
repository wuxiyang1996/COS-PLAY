#!/usr/bin/env bash
# ======================================================================
#  Candy Crush baseline: LLM API model plays Candy Crush
#
#  Supported models:
#    openai/gpt-oss-120b (default), google/gemini-3.1-pro-preview,
#    anthropic/claude-4.6-sonnet-20260217
#
#  Usage:
#    bash baselines/run_candy_crush_baseline.sh
#    bash baselines/run_candy_crush_baseline.sh --model google/gemini-3.1-pro-preview
#    EPISODES=16 bash baselines/run_candy_crush_baseline.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="openai/gpt-oss-120b"
while [[ $# -gt 0 ]]; do
    case "$1" in --model) MODEL="$2"; shift 2 ;; *) break ;; esac
done

EPISODES="${EPISODES:-8}"
TEMPERATURE="${TEMPERATURE:-0.3}"
SEED="${SEED:-42}"

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

[ -z "${OPENROUTER_API_KEY:-}" ] && echo "Warning: OPENROUTER_API_KEY not set. See .env.example."
[ -z "${OPENAI_API_KEY:-}" ] && echo "Warning: OPENAI_API_KEY not set. See .env.example."

MODEL_TAG="${MODEL//\//_}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/${MODEL_TAG}_candy_crush_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  ${MODEL} Baseline — Candy Crush (${EPISODES} episodes)"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:     ${MODEL}"
echo "  Episodes:  ${EPISODES}    Temperature: ${TEMPERATURE}"
echo "  Output:    ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 "${PROJECT_ROOT}/cold_start/generate_cold_start_gpt54.py" \
    --games candy_crush \
    --episodes "${EPISODES}" \
    --model "${MODEL}" \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --no_label \
    --verbose \
    --output_dir "${OUTPUT_DIR}" \
    "$@"

echo ""
echo "  ${MODEL} Candy Crush Baseline COMPLETE — ${OUTPUT_DIR}"
