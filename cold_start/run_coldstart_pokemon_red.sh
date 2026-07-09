#!/usr/bin/env bash
#
# Run Pokemon Red cold-start rollouts using Orak env + toolset.
#
# This uses the Orak PokemonRedEnv (PyBoyRunner) directly with high-level
# tools: continue_dialog, select_move_in_battle, move_to, interact_with_object, etc.
# One tool call = many button presses, making LLM usage vastly more efficient.
#
# Prerequisites:
#   - PyBoy installed:  pip install pyboy
#   - ROM placed at:    GamingAgent/gamingagent/configs/custom_06_pokemon_red/rom/pokemon.gb
#   - API key set:      OPENROUTER_API_KEY or in api_keys.py
#
# Usage:
#   bash cold_start/run_coldstart_pokemon_red.sh
#   bash cold_start/run_coldstart_pokemon_red.sh --episodes 5 --verbose
#   bash cold_start/run_coldstart_pokemon_red.sh --episodes 20 --resume
#   bash cold_start/run_coldstart_pokemon_red.sh --episodes 3 --verbose
#   bash cold_start/run_coldstart_pokemon_red.sh --episodes 3 --label   # enable labeling
#   bash cold_start/run_coldstart_pokemon_red.sh --episodes 60 --fast   # max speed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAMINGAGENT_ROOT="$(cd "$CODEBASE_ROOT/../GamingAgent" 2>/dev/null && pwd || echo "")"
ORAK_SRC="$(cd "$CODEBASE_ROOT/../Orak/src" 2>/dev/null && pwd || echo "")"

if [ -z "$ORAK_SRC" ] || [ ! -d "$ORAK_SRC" ]; then
    echo "ERROR: Orak source not found at:"
    echo "  $CODEBASE_ROOT/../Orak/src"
    exit 1
fi

export PYTHONPATH="${CODEBASE_ROOT}:${GAMINGAGENT_ROOT}:${ORAK_SRC}:${PYTHONPATH:-}"

# Auto-detect API key from api_keys.py if not already set
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

# Resolve ROM path
ROM_PATH="${GAMINGAGENT_ROOT}/gamingagent/configs/custom_06_pokemon_red/rom/pokemon.gb"
ROM_ALT="$CODEBASE_ROOT/../ROMs/Pokemon - Red Version (USA, Europe).gb"
if [ ! -f "$ROM_PATH" ] && [ -f "$ROM_ALT" ]; then
    ROM_PATH="$ROM_ALT"
fi
if [ ! -f "$ROM_PATH" ]; then
    echo "ERROR: Pokemon Red ROM not found. Place it at one of:"
    echo "  ${GAMINGAGENT_ROOT}/gamingagent/configs/custom_06_pokemon_red/rom/pokemon.gb"
    echo "  ${ROM_ALT}"
    exit 1
fi

# Verify pyboy is installed
if ! python3 -c "import pyboy" 2>/dev/null; then
    echo "ERROR: PyBoy not installed. Install it with:"
    echo "  pip install pyboy"
    exit 1
fi

echo "================================================================"
echo "  Pokemon Red Cold-Start (Orak Env + Toolset)"
echo "================================================================"
echo "  Backend: Orak PyBoyRunner + PokemonToolset (high-level tools)"
echo "  ROM:     $ROM_PATH"
echo "  Args:    $*"
echo "================================================================"
echo ""

python3 "${SCRIPT_DIR}/generate_cold_start_pokemon_red.py" --rom_path "$ROM_PATH" "$@"

echo ""
echo "================================================================"
echo "  Done. Output: cold_start/output/gpt54/pokemon_red/"
echo "================================================================"
