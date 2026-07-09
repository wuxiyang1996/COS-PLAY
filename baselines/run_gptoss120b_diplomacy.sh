#!/usr/bin/env bash
# GPT-OSS 120B baseline on Diplomacy (56 episodes = 8/power x 7) via OpenRouter
# Controlled power = GPT-OSS 120B, all opponents = GPT-5.4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

AGENTEVOLVER_ROOT=""
for candidate in "$WORKSPACE_ROOT/AgentEvolver" "$PROJECT_ROOT/AgentEvolver"; do
    [ -d "$candidate" ] && AGENTEVOLVER_ROOT="$candidate" && break
done
[ -z "$AGENTEVOLVER_ROOT" ] && echo "[ERROR] AgentEvolver repo not found." && exit 1

AI_DIPLOMACY_ROOT=""
for candidate in "$WORKSPACE_ROOT/AI_Diplomacy" "$PROJECT_ROOT/AI_Diplomacy"; do
    [ -d "$candidate" ] && AI_DIPLOMACY_ROOT="$candidate" && break
done

NUM_POWERS=7
EPISODES_PER_POWER="${EPISODES_PER_POWER:-8}"
EPISODES=$((NUM_POWERS * EPISODES_PER_POWER))
TEMPERATURE="${TEMPERATURE:-0.4}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
SEED="${SEED:-42}"

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTHONPATH="${PROJECT_ROOT}:${AGENTEVOLVER_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"
[ -n "$AI_DIPLOMACY_ROOT" ] && export PYTHONPATH="${AI_DIPLOMACY_ROOT}:${PYTHONPATH}"

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
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/gptoss120b_diplomacy_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  GPT-OSS 120B Baseline — Diplomacy (${EPISODES} episodes = ${EPISODES_PER_POWER}/power x ${NUM_POWERS})"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:     ${MODEL}  (controlled power)"
echo "  Opponents: ${OPPONENT_MODEL}"
echo "  Episodes:  ${EPISODES}    Temperature: ${TEMPERATURE}"
echo "  Mode:      per-power (cycle through 7 powers)"
echo "  Output:    ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

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
    "$@"

echo ""
echo "  GPT-OSS 120B Diplomacy Baseline COMPLETE — ${OUTPUT_DIR}"
