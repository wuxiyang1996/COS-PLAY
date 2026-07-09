#!/usr/bin/env bash
#
# Extract skills from labeled GPT-5.4 rollouts and build a Skill Bank
# using the skill_agents_grpo pipeline.
#
# Reads from:  labeling/output/gpt54/<game>/episode_*.json
# Writes to:   skill_agents_grpo/extract_skillbank/output/gpt54_skillbank_grpo/<game>/
#
# Per-game outputs include:
#   - skill_bank.jsonl, skill_catalog.json      — extracted skills
#   - stage_io_log.json                         — per-stage I/O recordings
#   - llm_calls_log.json                        — direct GPT-5.4 call log
#   - teacher_io_coldstart.jsonl                — Stage 2 LLM teacher I/O
#       (segment rankings, transition rankings, pairwise choices, skill naming)
#       Use this as supervised fine-tuning data for Qwen3-8B cold-start.
#
# Prerequisites:
#   - Labeled rollouts must exist in the input directory
#   - OPENROUTER_API_KEY (or OPENAI_API_KEY) must be set
#
# Usage:
#   bash skill_agents_grpo/extract_skillbank/run_extract_skillbank_grpo.sh                    # all games
#   bash skill_agents_grpo/extract_skillbank/run_extract_skillbank_grpo.sh --games tetris     # specific game
#   bash skill_agents_grpo/extract_skillbank/run_extract_skillbank_grpo.sh --one_per_game -v  # quick test
#   bash skill_agents_grpo/extract_skillbank/run_extract_skillbank_grpo.sh --dry_run          # preview only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODEBASE_ROOT="$(cd "$GRPO_ROOT/.." && pwd)"
GAMINGAGENT_ROOT="$(cd "$CODEBASE_ROOT/../GamingAgent" 2>/dev/null && pwd || echo "")"

export PYTHONPATH="${CODEBASE_ROOT}${GAMINGAGENT_ROOT:+:$GAMINGAGENT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "WARNING: Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set."
    echo "         LLM calls for skill naming/description will fail."
    echo ""
fi

INPUT_DIR="${CODEBASE_ROOT}/labeling/output/gpt54"
OUTPUT_DIR="${SCRIPT_DIR}/output/gpt54_skillbank_grpo"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    echo "       Run the labeling pipeline first, or specify --input_dir."
    exit 1
fi

N_GAMES=$(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
N_EPISODES=$(find "$INPUT_DIR" -name "episode_*.json" ! -name "episode_buffer.json" | wc -l)

echo "=============================================================="
echo "  Skill Bank Extraction Pipeline (skill_agents_grpo)"
echo "=============================================================="
echo "  Pipeline: skill_agents_grpo"
echo "  Input:    $INPUT_DIR"
echo "  Output:   $OUTPUT_DIR"
echo "  Games:    $N_GAMES"
echo "  Episodes: $N_EPISODES"
echo "  Args:     $*"
echo "=============================================================="
echo ""

cd "$CODEBASE_ROOT"

python -m skill_agents_grpo.extract_skillbank.extract_skillbank_grpo_gpt54 \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
