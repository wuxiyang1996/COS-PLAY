#!/usr/bin/env bash
# =============================================================================
# install_game_ai_agent_env.sh
#
# Creates the "game-ai-agent" conda environment with all dependencies for:
#   - Game-AI-Agent trainer (GRPO, SkillBank, co-evolution)
#   - legacy/evaluate_videogamebench (DOS games via Playwright)
#   - evaluate_gamingagent (2048, Sokoban, Tetris, VizDoom, etc.)
#   - legacy/evaluate_overcooked (multi-agent cooking)
#   - evaluation_evolver (Avalon + Diplomacy via AgentEvolver)
#
# Prerequisites:
#   - miniconda3 or anaconda installed
#   - CUDA 12.x drivers on the host (for GPU training/inference)
#   - The following repos cloned as siblings under the same parent directory:
#       Game-AI-Agent/     (this repo)
#       videogamebench/    (https://github.com/alexzhang13/videogamebench)
#       GamingAgent/       (https://github.com/lmgame-org/GamingAgent)
#       overcooked_ai/     (https://github.com/HumanCompatibleAI/overcooked_ai)
#       AgentEvolver/      (https://github.com/modelscope/AgentEvolver)
#
# Usage:
#   cd /path/to/parent          # directory containing all repos above
#   bash Game-AI-Agent/install_game_ai_agent_env.sh [CONDA_PATH]
#
#   CONDA_PATH: optional path to conda binary (default: auto-detect)
#
# After install:
#   conda activate game-ai-agent
#   export PYTHONPATH=$(pwd)/Game-AI-Agent:$(pwd)/overcooked_ai/src:$(pwd)/AgentEvolver:$(pwd)/videogamebench:$(pwd)/GamingAgent:$PYTHONPATH
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_NAME="game-ai-agent"
PYTHON_VERSION="3.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Parent of Game-AI-Agent = directory containing all sibling repos
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Locate conda
if [[ -n "${1:-}" ]]; then
    CONDA="$1"
elif command -v conda &>/dev/null; then
    CONDA="$(command -v conda)"
elif [[ -x "$PARENT_DIR/miniconda3/bin/conda" ]]; then
    CONDA="$PARENT_DIR/miniconda3/bin/conda"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA="$HOME/miniconda3/bin/conda"
else
    echo "ERROR: conda not found. Pass the path as an argument or install miniconda first."
    exit 1
fi

CONDA_DIR="$(dirname "$(dirname "$CONDA")")"
PIP="$CONDA_DIR/envs/$ENV_NAME/bin/pip"
PYTHON="$CONDA_DIR/envs/$ENV_NAME/bin/python"

echo "============================================================"
echo "  Game-AI-Agent environment installer"
echo "============================================================"
echo "  conda:       $CONDA"
echo "  env name:    $ENV_NAME"
echo "  python:      $PYTHON_VERSION"
echo "  parent dir:  $PARENT_DIR"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# Repo checks
# ---------------------------------------------------------------------------
REPOS=(
    "Game-AI-Agent"
    "videogamebench"
    "GamingAgent"
    "overcooked_ai"
    "AgentEvolver"
)

MISSING=()
for repo in "${REPOS[@]}"; do
    if [[ ! -d "$PARENT_DIR/$repo" ]]; then
        MISSING+=("$repo")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "WARNING: The following repos are not found under $PARENT_DIR:"
    for m in "${MISSING[@]}"; do
        echo "  - $m"
    done
    echo
    echo "The installer will skip editable installs for missing repos."
    echo "Clone them later and run the relevant pip install -e commands."
    echo
fi

# ---------------------------------------------------------------------------
# Step 1: Create conda environment
# ---------------------------------------------------------------------------
if "$CONDA" env list | grep -q "^${ENV_NAME} "; then
    echo "[1/7] Conda env '$ENV_NAME' already exists, skipping creation."
else
    echo "[1/7] Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION ..."
    "$CONDA" create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi
echo

# ---------------------------------------------------------------------------
# Step 2: Trainer core dependencies
# ---------------------------------------------------------------------------
echo "[2/7] Installing trainer dependencies (torch, transformers, sentence-transformers, etc.) ..."
"$PIP" install --quiet \
    "numpy==1.26.4" \
    "pyyaml>=6.0" \
    "sentence-transformers>=2.7.0" \
    "transformers>=4.51.0" \
    "omegaconf>=2.3.0" \
    "hydra-core>=1.3.0"
echo

# ---------------------------------------------------------------------------
# Step 3: VideoGameBench
# ---------------------------------------------------------------------------
echo "[3/7] Installing VideoGameBench dependencies ..."
"$PIP" install --quiet \
    "playwright>=1.40.0" \
    "openai" \
    "anthropic" \
    "ImageHash" \
    "litellm>=1.0.0" \
    "python-dotenv>=1.0.0" \
    "opencv-python"

if [[ -d "$PARENT_DIR/videogamebench" ]]; then
    "$PIP" install --quiet -e "$PARENT_DIR/videogamebench"
    echo "  Installed videogamebench (editable)"
fi

# Install Playwright browsers
echo "  Installing Playwright Chromium ..."
"$PYTHON" -m playwright install chromium 2>/dev/null || true
echo

# ---------------------------------------------------------------------------
# Step 4: GamingAgent
# ---------------------------------------------------------------------------
echo "[4/7] Installing GamingAgent ..."
if [[ -d "$PARENT_DIR/GamingAgent" ]]; then
    "$PIP" install --quiet -e "$PARENT_DIR/GamingAgent"
    echo "  Installed GamingAgent (editable)"
    # GamingAgent pins numpy==1.24.4 but works fine with 1.26.4; restore it
    "$PIP" install --quiet "numpy==1.26.4"
    echo "  Restored numpy==1.26.4 (GamingAgent pin is overly strict)"
fi
echo

# ---------------------------------------------------------------------------
# Step 5: Overcooked
# ---------------------------------------------------------------------------
echo "[5/7] Installing Overcooked AI ..."
if [[ -d "$PARENT_DIR/overcooked_ai" ]]; then
    # Relax Python version constraint if needed (upstream pins <3.11 but works on 3.11)
    PYPROJECT="$PARENT_DIR/overcooked_ai/pyproject.toml"
    if [[ -f "$PYPROJECT" ]] && grep -q '">=3.10,<3.11"' "$PYPROJECT"; then
        sed -i 's/">=3.10,<3.11"/">=3.10,<3.12"/' "$PYPROJECT"
        echo "  Relaxed overcooked_ai Python version constraint to <3.12"
    fi
    "$PIP" install --quiet -e "$PARENT_DIR/overcooked_ai"
    echo "  Installed overcooked_ai (editable)"
fi
echo

# ---------------------------------------------------------------------------
# Step 6: AgentEvolver (Avalon + Diplomacy)
# ---------------------------------------------------------------------------
echo "[6/7] Installing AgentEvolver eval dependencies (diplomacy, coloredlogs, loguru) ..."
"$PIP" install --quiet \
    "diplomacy>=1.1.2" \
    "coloredlogs" \
    "loguru>=0.7.0"

# AgentEvolver itself is added via PYTHONPATH (no pip install — its full
# requirements.txt would conflict with torch/transformers versions).
if [[ -d "$PARENT_DIR/AgentEvolver" ]]; then
    echo "  AgentEvolver found at $PARENT_DIR/AgentEvolver (use PYTHONPATH, not pip install)"
fi
echo

# ---------------------------------------------------------------------------
# Step 7: Final numpy pin & verification
# ---------------------------------------------------------------------------
echo "[7/7] Pinning numpy==1.26.4 and verifying ..."
"$PIP" install --quiet "numpy==1.26.4"

echo
echo "Running import checks ..."
PYTHONPATH="$PARENT_DIR/Game-AI-Agent:$PARENT_DIR/overcooked_ai/src:$PARENT_DIR/AgentEvolver:$PARENT_DIR/videogamebench:$PARENT_DIR/GamingAgent" \
"$PYTHON" -c "
import sys

failures = []
def check(label, fn):
    try:
        fn()
        print(f'  [OK]  {label}')
    except Exception as e:
        failures.append((label, str(e)))
        print(f'  [FAIL] {label}: {e}')

print(f'Python {sys.version}')
print()

# Trainer
check('numpy',                lambda: __import__('numpy'))
check('torch',                lambda: __import__('torch'))
check('transformers',         lambda: __import__('transformers'))
check('sentence_transformers',lambda: __import__('sentence_transformers'))
check('omegaconf',            lambda: __import__('omegaconf'))
check('hydra',                lambda: __import__('hydra'))

# VideoGameBench
check('playwright',           lambda: __import__('playwright'))
check('litellm',              lambda: __import__('litellm'))
check('pydantic',             lambda: __import__('pydantic'))

# GamingAgent
check('gymnasium',            lambda: __import__('gymnasium'))
check('vizdoom',              lambda: __import__('vizdoom'))
check('stable_retro',         lambda: __import__('stable_retro'))
check('xai_sdk',              lambda: __import__('xai_sdk'))

# Overcooked
check('overcooked_ai_py',     lambda: __import__('overcooked_ai_py'))

# AgentEvolver (Avalon)
check('games.games.avalon',   lambda: __import__('games.games.avalon.engine'))

# Diplomacy
check('diplomacy',            lambda: __import__('diplomacy'))

# Trainer modules
check('trainer.common.metrics',         lambda: __import__('trainer.common.metrics'))
check('trainer.decision.grpo_trainer',  lambda: __import__('trainer.decision.grpo_trainer'))
check('trainer.skillbank.em_trainer',   lambda: __import__('trainer.skillbank.em_trainer'))

print()
if failures:
    print(f'{len(failures)} check(s) failed:')
    for label, err in failures:
        print(f'  - {label}: {err}')
    sys.exit(1)
else:
    print('All checks passed.')
"

echo
echo "============================================================"
echo "  Installation complete!"
echo "============================================================"
echo
echo "  Activate:"
echo "    conda activate $ENV_NAME"
echo
echo "  Set PYTHONPATH (from the parent directory of all repos):"
echo "    export PYTHONPATH=\$(pwd)/Game-AI-Agent:\$(pwd)/overcooked_ai/src:\$(pwd)/AgentEvolver:\$(pwd)/videogamebench:\$(pwd)/GamingAgent:\$PYTHONPATH"
echo
echo "  Known nominal warning:"
echo "    gamingagent 0.1.0 requires numpy==1.24.4 (we use 1.26.4 — works fine)"
echo
echo "============================================================"
