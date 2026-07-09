#!/usr/bin/env bash
#
# run_gpt54_avalon_single_agent.sh — GPT-5.4 single-agent Avalon baseline
#
# Runs GPT-5.4 controlling ONE player per episode (cycling through player
# positions 0-4), with random partner policy for other players.
# This produces per-role reward data directly comparable with Qwen3-8B
# training runs.
#
# Default: 40 episodes (8 per player position), yielding ~8 Merlin,
# ~16 Servant, ~8 Minion, ~8 Assassin (role shuffle depends on seed).
#
# Usage:
#   bash baselines/run_gpt54_avalon_single_agent.sh
#   EPISODES=20 bash baselines/run_gpt54_avalon_single_agent.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

AGENTEVOLVER_ROOT=""
for candidate in "$WORKSPACE_ROOT/AgentEvolver" "$PROJECT_ROOT/AgentEvolver"; do
    if [ -d "$candidate" ]; then
        AGENTEVOLVER_ROOT="$candidate"
        break
    fi
done

if [ -z "$AGENTEVOLVER_ROOT" ]; then
    echo "[ERROR] AgentEvolver repo not found."
    exit 1
fi

EPISODES="${EPISODES:-40}"
NUM_PLAYERS="${NUM_PLAYERS:-5}"
TEMPERATURE="${TEMPERATURE:-0.4}"
MODEL="${MODEL:-gpt-5.4}"
SEED="${SEED:-42}"

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTHONPATH="${PROJECT_ROOT}:${AGENTEVOLVER_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/gpt54_avalon_single_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  GPT-5.4 Single-Agent Baseline — Avalon (${EPISODES} episodes)"
echo "══════════════════════════════════════════════════════════════"
echo "  Mode:          SINGLE-AGENT (--per_role, cycling player 0-4)"
echo "  Partner policy: random"
echo "  Model:          ${MODEL}"
echo "  Episodes:       ${EPISODES}"
echo "  Seed:           ${SEED}"
echo "  Output:         ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 "${PROJECT_ROOT}/cold_start/generate_cold_start_evolver.py" \
    --games avalon \
    --episodes "${EPISODES}" \
    --model "${MODEL}" \
    --temperature "${TEMPERATURE}" \
    --num_players "${NUM_PLAYERS}" \
    --seed "${SEED}" \
    --no_label \
    --verbose \
    --per_role \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  GPT-5.4 Single-Agent Avalon COMPLETE"
echo "══════════════════════════════════════════════════════════════"
echo "  Output: ${OUTPUT_DIR}"

if [ -f "${OUTPUT_DIR}/avalon/rollout_summary.json" ]; then
    echo ""
    python3 -c "
import json, math
from collections import defaultdict

with open('${OUTPUT_DIR}/avalon/rollout_summary.json') as f:
    s = json.load(f)

print(f'  Total episodes:  {s.get(\"total_episodes\", \"?\")}')
print(f'  Mean reward:     {s.get(\"mean_reward\", 0):.3f}')
print(f'  Mean steps:      {s.get(\"mean_steps\", 0):.1f}')
print(f'  Elapsed:         {s.get(\"elapsed_seconds\", 0):.1f}s')
print()

role_rewards = defaultdict(list)
for ep in s.get('episode_stats', []):
    rn = ep.get('role_name', '?')
    role_rewards[rn].append(ep['total_reward'])

print('  Per-role breakdown:')
print(f'  {\"Role\":>10s} | {\"n\":>3s} | {\"Mean\":>7s} | {\"Std\":>6s} | {\"Min\":>6s} | {\"Max\":>6s}')
print('  ' + '-' * 50)
for role in ['Merlin', 'Servant', 'Assassin', 'Minion']:
    rews = role_rewards.get(role, [])
    if not rews:
        continue
    m = sum(rews)/len(rews)
    v = sum((r-m)**2 for r in rews)/(len(rews)-1) if len(rews)>1 else 0
    std = math.sqrt(v)
    print(f'  {role:>10s} | {len(rews):3d} | {m:+7.3f} | {std:6.3f} | {min(rews):+6.2f} | {max(rews):+6.2f}')
" 2>/dev/null || true
fi

echo "══════════════════════════════════════════════════════════════"
