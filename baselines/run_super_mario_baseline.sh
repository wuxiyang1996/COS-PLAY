#!/usr/bin/env bash
# ======================================================================
#  Super Mario baseline: LLM API model plays Super Mario Bros
#
#  Requires orak-mario conda env for NES emulator (NumPy 1.x).
#
#  Supported models:
#    openai/gpt-oss-120b (default), google/gemini-3.1-pro-preview,
#    anthropic/claude-4.6-sonnet-20260217
#
#  Usage:
#    bash baselines/run_super_mario_baseline.sh
#    bash baselines/run_super_mario_baseline.sh --model google/gemini-3.1-pro-preview
#    EPISODES=16 bash baselines/run_super_mario_baseline.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

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
ORAK_SRC="${WORKSPACE_ROOT}/Orak/src"
[ -d "$ORAK_SRC" ] && export PYTHONPATH="${ORAK_SRC}:${PYTHONPATH}"

# Super Mario requires the orak-mario conda env (nes-py needs NumPy 1.x).
# Install:  bash env_wrappers/envs/orak-mario/install.sh
# Activate: source env_wrappers/setup_orak_mario.sh
ORAK_PYTHON="${ORAK_PYTHON:-}"
if [ -z "${ORAK_PYTHON}" ]; then
    for candidate in \
        /workspace/miniconda3/envs/orak-mario/bin/python \
        "${CONDA_PREFIX:-/nonexistent}/../orak-mario/bin/python" \
        "$(command -v python3)"; do
        if [ -x "${candidate}" ] && "${candidate}" -c "import nes_py" 2>/dev/null; then
            ORAK_PYTHON="${candidate}"; break
        fi
    done
fi
if [ -z "${ORAK_PYTHON}" ]; then
    echo "[WARN] orak-mario env not detected. Install with:"
    echo "         bash env_wrappers/envs/orak-mario/install.sh"
    echo "       Using current python3 — may fail if nes_py/numpy<2 not available."
    ORAK_PYTHON="python3"
fi

# Xvfb for NES rendering
if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb &>/dev/null; then
        if ! pgrep -f "Xvfb :99" &>/dev/null; then
            Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
            sleep 1
        fi
        export DISPLAY=":99"
    fi
fi

[ -z "${OPENROUTER_API_KEY:-}" ] && echo "Warning: OPENROUTER_API_KEY not set. See .env.example."
[ -z "${OPENAI_API_KEY:-}" ] && echo "Warning: OPENAI_API_KEY not set. See .env.example."

MODEL_TAG="${MODEL//\//_}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/${MODEL_TAG}_super_mario_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  ${MODEL} Baseline — Super Mario (${EPISODES} episodes)"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:     ${MODEL}"
echo "  Python:    ${ORAK_PYTHON}"
echo "  Episodes:  ${EPISODES}    Temperature: ${TEMPERATURE}"
echo "  Output:    ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

${ORAK_PYTHON} "${PROJECT_ROOT}/env_wrappers/test_orak_mario_gpt54.py" \
    --game super_mario \
    --episodes "${EPISODES}" \
    --model "${MODEL}" \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "  ${MODEL} Super Mario Baseline COMPLETE — ${OUTPUT_DIR}"
