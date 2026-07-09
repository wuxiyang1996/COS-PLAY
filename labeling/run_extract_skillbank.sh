#!/usr/bin/env bash
#
# Extract skills from labeled GPT-5.4 rollouts and build a Skill Bank.
#
# Reads from:  labeling/output/gpt54/<game>/episode_*.json
# Writes to:   labeling/output/gpt54_skillbank/<game>/
#
# Prerequisites:
#   - Labeled rollouts must exist in the input directory
#   - OPENROUTER_API_KEY (or OPENAI_API_KEY) must be set
#
# Usage:
#   bash labeling/run_extract_skillbank.sh                    # all games
#   bash labeling/run_extract_skillbank.sh --games tetris     # specific game
#   bash labeling/run_extract_skillbank.sh --one_per_game -v  # quick test
#   bash labeling/run_extract_skillbank.sh --dry_run          # preview only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAMINGAGENT_ROOT="$(cd "$CODEBASE_ROOT/../GamingAgent" 2>/dev/null && pwd || echo "")"

export PYTHONPATH="${CODEBASE_ROOT}${GAMINGAGENT_ROOT:+:$GAMINGAGENT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "WARNING: Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set."
    echo "         LLM calls for skill naming/description will fail."
    echo ""
fi

INPUT_DIR="${SCRIPT_DIR}/output/gpt54"
OUTPUT_DIR="${SCRIPT_DIR}/output/gpt54_skillbank"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    echo "       Run the labeling pipeline first, or specify --input_dir."
    exit 1
fi

N_GAMES=$(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
N_EPISODES=$(find "$INPUT_DIR" -name "episode_*.json" ! -name "episode_buffer.json" | wc -l)

echo "=============================================================="
echo "  Skill Bank Extraction Pipeline"
echo "=============================================================="
echo "  Input:    $INPUT_DIR"
echo "  Output:   $OUTPUT_DIR"
echo "  Games:    $N_GAMES"
echo "  Episodes: $N_EPISODES"
echo "  Args:     $*"
echo "=============================================================="
echo ""

python "$SCRIPT_DIR/extract_skillbank_gpt54.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
