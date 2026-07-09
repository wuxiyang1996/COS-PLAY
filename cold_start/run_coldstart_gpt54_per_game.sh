#!/usr/bin/env bash
#
# Run GPT-5.4 cold-start per game for evaluate_gamingagent only.
# Supports only the 4 LMGame-Bench games used by evaluate_gamingagent:
#   2048, Sokoban, Candy Crush, Tetris.
# (Orak, Doom, Super Mario, etc. are not supported by this script.)
#
# Usage:
#   bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 50
#       # All 4 evaluate_gamingagent games, 50 episodes each.
#
#   bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 20 twenty_forty_eight tetris
#       # Only 2048 and Tetris, 20 episodes each.
#
#   bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 100 --resume --no_label
#
# Default: 100 episodes per game if --episodes is omitted.
# Games (if omitted): twenty_forty_eight sokoban candy_crush tetris

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAMINGAGENT_ROOT="$(cd "$CODEBASE_ROOT/../GamingAgent" 2>/dev/null && pwd || echo "")"

# evaluate_gamingagent: only these 4 games (see evaluate_gamingagent/game_configs.py)
DEFAULT_EPISODES=100
EVAL_GAMES=(twenty_forty_eight sokoban candy_crush tetris)

EPISODES="$DEFAULT_EPISODES"
EXTRA_ARGS=()
GAMES_ARG=()

while [ $# -gt 0 ]; do
    case "$1" in
        --episodes)
            if [ -n "${2:-}" ] && [[ "$2" =~ ^[0-9]+$ ]]; then
                EPISODES="$2"
                shift 2
            else
                echo "[ERROR] --episodes requires a number"
                exit 1
            fi
            ;;
        twenty_forty_eight|sokoban|candy_crush|tetris)
            GAMES_ARG+=("$1")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ${#GAMES_ARG[@]} -eq 0 ]; then
    GAMES_ARG=("${EVAL_GAMES[@]}")
fi

# Reuse env from main script
export PYTHONPATH="${CODEBASE_ROOT}:${GAMINGAGENT_ROOT}:${PYTHONPATH:-}"
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    OPENROUTER_API_KEY="$(python3 -c "
import sys; sys.path.insert(0, '${CODEBASE_ROOT}')
from api_keys import open_router_api_key; print(open_router_api_key or '')
" 2>/dev/null || echo "")"
    export OPENROUTER_API_KEY
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    OPENAI_API_KEY="${OPENROUTER_API_KEY:-}"
    export OPENAI_API_KEY
fi

echo "================================================================"
echo "  GPT-5.4 Cold-Start — evaluate_gamingagent (4 games only)"
echo "================================================================"
echo "  Episodes per game: $EPISODES"
echo "  Games:             ${GAMES_ARG[*]}"
echo "  Extra args:        ${EXTRA_ARGS[*]:-none}"
echo "================================================================"

for game in "${GAMES_ARG[@]}"; do
    echo ""
    echo ">>> Running: $game (--episodes $EPISODES)"
    python3 "${SCRIPT_DIR}/generate_cold_start_gpt54.py" \
        --games "$game" \
        --episodes "$EPISODES" \
        "${EXTRA_ARGS[@]}"
done

echo ""
echo "================================================================"
echo "  Done. Output: cold_start/output/gpt54/<game>/"
echo "  (evaluate_gamingagent games only: 2048, Sokoban, Candy Crush, Tetris)"
echo "================================================================"
