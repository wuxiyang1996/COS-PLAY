#!/usr/bin/env bash
# =============================================================================
# install_main_env.sh
#
# Creates the "game-ai-agent" conda environment with ALL dependencies for:
#   - GRPO + FSDP + LoRA training on Qwen3.5-9B
#   - vLLM 0.20 inference (Qwen3.5-9B + Qwen3.5-35B-A3B)
#   - Skill bank pipeline + RAG (Qwen3-Embedding-0.6B)
#   - Cold-start data generation & labeling
#   - Baselines (OpenRouter, OpenAI, Anthropic, Google, Together, Z.AI, xAI)
#   - GamingAgent / LMGame-Bench (2048, plain Tetris)
#   - gym-v (179 visual envs incl. 13 Temporal/* Sega Genesis games)
#   - Avalon (PYTHONPATH), Diplomacy
#
# NOT installed here (separate env required):
#   - Super Mario Bros: install_orak_mario.sh   (nes-py needs numpy<2)
#   - BrowserGym:       install_browsergym.sh   (playwright==1.44 hard-pin)
#   - OSWorld:          install_osworld.sh      (gymnasium~=0.28, transformers~=4.35)
#   - Candy Crush:      tile_match_gym pulls numba → numpy<2 conflict
#
# Prerequisites:
#   - Miniconda3 or Anaconda installed
#   - CUDA 12.8+ or 13.x driver on the host (for GPU training / vLLM)
#   - The following repos cloned as siblings under the same parent directory
#     (the script will offer to clone gym-v if missing):
#       Multi-hop-Reasoning-VLM-Agent/   (this repo)
#       GamingAgent/                     (https://github.com/lmgame-org/GamingAgent)
#       AgentEvolver/                    (https://github.com/modelscope/AgentEvolver)
#       gym-v/                           (https://github.com/ModalMinds/gym-v)
#
# Usage:
#   cd /path/to/parent           # directory containing all repos
#   bash Multi-hop-Reasoning-VLM-Agent/install/install_main_env.sh \
#        [CONDA_PATH] [ROM_ZIP]
#
#   CONDA_PATH (optional) — path to the conda binary. Auto-detected if blank.
#   ROM_ZIP    (optional) — path to a Sega Genesis ROM zip. If provided, the
#                           script applies install/gymv_temporal_patch/ and
#                           imports the 13 Temporal/* envs.
#
# After install:
#   conda activate game-ai-agent
#   export PYTHONPATH=$(pwd)/Multi-hop-Reasoning-VLM-Agent:$(pwd)/AgentEvolver:$(pwd)/GamingAgent:$PYTHONPATH
#   cp Multi-hop-Reasoning-VLM-Agent/.env.example Multi-hop-Reasoning-VLM-Agent/.env
#   set -a && source Multi-hop-Reasoning-VLM-Agent/.env && set +a
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_NAME="game-ai-agent"
PYTHON_VERSION="3.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"          # Multi-hop-Reasoning-VLM-Agent/
PARENT_DIR="$(dirname "$REPO_DIR")"          # parent of all sibling repos
REQS="${SCRIPT_DIR}/requirements.txt"
GYMV_DIR="${PARENT_DIR}/gym-v"
GYMV_PATCH="${SCRIPT_DIR}/gymv_temporal_patch/apply_patch.sh"

ROM_ZIP="${2:-}"                              # optional 2nd positional arg

# ---------------------------------------------------------------------------
# Locate conda
# ---------------------------------------------------------------------------
if [[ -n "${1:-}" ]]; then
    CONDA="$1"
elif command -v conda &>/dev/null; then
    CONDA="$(command -v conda)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA="$HOME/miniconda3/bin/conda"
elif [[ -x "/workspace/miniconda3/bin/conda" ]]; then
    CONDA="/workspace/miniconda3/bin/conda"
else
    echo "ERROR: conda not found. Pass the conda path as an argument or install Miniconda first."
    echo "  curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash"
    exit 1
fi

CONDA_DIR="$(dirname "$(dirname "$CONDA")")"
PIP="$CONDA_DIR/envs/$ENV_NAME/bin/pip"
PYTHON="$CONDA_DIR/envs/$ENV_NAME/bin/python"

echo "============================================================"
echo "  Multi-hop-Reasoning-VLM-Agent — Main Environment Installer"
echo "============================================================"
echo "  conda:       $CONDA"
echo "  env name:    $ENV_NAME"
echo "  python:      $PYTHON_VERSION"
echo "  repo dir:    $REPO_DIR"
echo "  parent dir:  $PARENT_DIR"
echo "  gym-v dir:   $GYMV_DIR"
if [[ -n "$ROM_ZIP" ]]; then
    echo "  ROM zip:     $ROM_ZIP   (Temporal/* envs will be enabled)"
else
    echo "  ROM zip:     (none)     (gym-v installed without retro games)"
fi
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# Step 1: Create conda environment
# ---------------------------------------------------------------------------
if "$CONDA" env list | grep -q "^${ENV_NAME} "; then
    echo "[1/7] Conda env '$ENV_NAME' already exists — skipping creation."
else
    echo "[1/7] Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION ..."
    "$CONDA" create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi
echo

# ---------------------------------------------------------------------------
# Step 2: PyTorch is pulled in transitively by vLLM 0.20 (torch==2.11.0+cu130).
#         The wheels on PyPI ship with CUDA runtime libraries, which work on
#         any host with a CUDA 12.8+ / 13.x driver (covers H100/H200/B200).
#         Skipping an explicit `torch` install here avoids pinning a stale
#         CUDA 12.4 wheel that would then be re-resolved to torch==2.11.0
#         below and trigger an avoidable re-download.
# ---------------------------------------------------------------------------
echo "[2/7] Skipping explicit torch install — vLLM will pull torch==2.11.0+cu130 ..."
echo

# ---------------------------------------------------------------------------
# Step 3: Install all pip requirements
# ---------------------------------------------------------------------------
echo "[3/7] Installing pip requirements from $REQS ..."
"$PIP" install -r "$REQS"
echo

# ---------------------------------------------------------------------------
# Step 4: Install GamingAgent (editable). It pins numpy==1.24.4 nominally;
#         we keep numpy 2.x because vLLM 0.20 / torch 2.11 need it and the
#         GamingAgent runtime (2048, plain Tetris) is happy on numpy 2.x.
# ---------------------------------------------------------------------------
echo "[4/7] Installing GamingAgent ..."
if [[ -d "$PARENT_DIR/GamingAgent" ]]; then
    "$PIP" install --quiet --no-deps -e "$PARENT_DIR/GamingAgent"
    echo "  ✓ Installed GamingAgent (editable, --no-deps to avoid numpy downgrade)"
else
    echo "  ⚠ GamingAgent not found at $PARENT_DIR/GamingAgent"
    echo "    Clone it:  git clone https://github.com/lmgame-org/GamingAgent.git $PARENT_DIR/GamingAgent"
    echo "    Then run:  $PIP install --no-deps -e $PARENT_DIR/GamingAgent"
fi
echo

# ---------------------------------------------------------------------------
# Step 5: Install gym-v (editable). Clone if missing; install with
#         [games,spatial] extras to enable Games/* + Spatial/* envs.
# ---------------------------------------------------------------------------
echo "[5/7] Installing gym-v ..."
if [[ ! -d "$GYMV_DIR" ]]; then
    echo "  ⓘ gym-v not found at $GYMV_DIR — cloning ..."
    git clone --depth=1 https://github.com/ModalMinds/gym-v.git "$GYMV_DIR"
fi
"$PIP" install --quiet -e "${GYMV_DIR}[games,spatial]"
echo "  ✓ Installed gym-v editable from $GYMV_DIR (extras: games,spatial)"

if [[ -n "$ROM_ZIP" ]]; then
    if [[ ! -f "$ROM_ZIP" ]]; then
        echo "  ⚠ ROM zip not found at $ROM_ZIP — skipping Temporal/* setup."
    elif [[ ! -x "$GYMV_PATCH" ]]; then
        echo "  ⚠ gymv_temporal_patch/apply_patch.sh not executable — chmod-ing and retrying ..."
        chmod +x "$GYMV_PATCH" || true
    fi
    if [[ -f "$ROM_ZIP" && -x "$GYMV_PATCH" ]]; then
        echo "  ⓘ Applying Temporal/* multimodal patch + ROM import ..."
        # The patch script itself activates the right conda env via PYTHON / PIP it picks up.
        # We invoke it with the gym-v dir + ROM zip as positional args.
        bash "$GYMV_PATCH" "$GYMV_DIR" "$ROM_ZIP" || {
            echo "  ⚠ Temporal/* patch failed — gym-v Games/* and Spatial/* still work."
        }
    fi
else
    echo "  ⓘ Skipping Temporal/* (no ROM zip provided). To enable later:"
    echo "      bash $GYMV_PATCH $GYMV_DIR /path/to/Mega_Drive_Mini_Full_Set.zip"
fi
echo

# ---------------------------------------------------------------------------
# Step 6: Check AgentEvolver
# ---------------------------------------------------------------------------
echo "[6/7] Checking AgentEvolver ..."
if [[ -d "$PARENT_DIR/AgentEvolver" ]]; then
    echo "  ✓ AgentEvolver found at $PARENT_DIR/AgentEvolver (added via PYTHONPATH, not pip)"
else
    echo "  ⚠ AgentEvolver not found at $PARENT_DIR/AgentEvolver"
    echo "    Clone it:  git clone https://github.com/modelscope/AgentEvolver.git $PARENT_DIR/AgentEvolver"
fi
echo

# ---------------------------------------------------------------------------
# Step 7: Verify installation
# ---------------------------------------------------------------------------
echo "[7/7] Verifying installation ..."
echo

PYTHONPATH="${REPO_DIR}:${PARENT_DIR}/AgentEvolver:${PARENT_DIR}/GamingAgent:${PYTHONPATH:-}" \
"$PYTHON" -c "
import sys, warnings
warnings.filterwarnings('ignore')

failures = []
warns = []

def check(label, fn, required=True):
    try:
        fn()
        print(f'  [OK]   {label}')
    except Exception as e:
        if required:
            failures.append((label, str(e)))
            print(f'  [FAIL] {label}: {e}')
        else:
            warns.append((label, str(e)))
            print(f'  [WARN] {label}: {e}  (optional)')

print(f'Python {sys.version}')
print()

# --- Core ML ---
print('Core ML:')
check('numpy',                 lambda: __import__('numpy'))
check('torch',                 lambda: __import__('torch'))
check('torch.cuda',            lambda: (t:=__import__('torch'), print(f'           CUDA available: {t.cuda.is_available()}, devices: {t.cuda.device_count()}, version: {t.version.cuda}')))
check('transformers',          lambda: __import__('transformers'))
check('peft',                  lambda: __import__('peft'))
check('safetensors',           lambda: __import__('safetensors'))
check('datasets',              lambda: __import__('datasets'))
check('accelerate',            lambda: __import__('accelerate'))
print()

# --- Inference ---
print('Inference:')
check('vllm',                  lambda: __import__('vllm'))
check('httpx',                 lambda: __import__('httpx'))
print()

# --- RAG ---
print('RAG:')
check('sentence_transformers', lambda: __import__('sentence_transformers'))
check('PIL (Pillow)',          lambda: __import__('PIL'))
print()

# --- API clients ---
print('API Clients:')
check('openai',                lambda: __import__('openai'))
check('anthropic',             lambda: __import__('anthropic'))
check('google.genai',          lambda: __import__('google.genai'))
check('google.generativeai',   lambda: __import__('google.generativeai'))
check('together',              lambda: __import__('together'))
check('zai',                   lambda: __import__('zai'))
check('xai_sdk',               lambda: __import__('xai_sdk'))
print()

# --- Configuration ---
print('Configuration:')
check('omegaconf',             lambda: __import__('omegaconf'))
check('hydra',                 lambda: __import__('hydra'))
check('yaml (pyyaml)',         lambda: __import__('yaml'))
print()

# --- Skill bank ---
print('Skill Bank:')
check('sklearn',               lambda: __import__('sklearn'))
print()

# --- Game environments ---
print('Game Environments:')
check('gymnasium',             lambda: __import__('gymnasium'))
check('pygame',                lambda: __import__('pygame'))
check('gym_v',                 lambda: __import__('gym_v'))
check('gym_v.envs',            lambda: __import__('gym_v.envs'))
def _temporal():
    import gym_v, gym_v.envs
    n = sum(1 for k in gym_v.registry if k.startswith('Temporal/'))
    if n == 0:
        raise RuntimeError('0 Temporal/* envs registered (skip ROM_ZIP step? gym-v Games/Spatial still work)')
    print(f'           {n} Temporal/* envs registered')
check('gym_v Temporal/*',      _temporal,                                          required=False)
check('stable_retro',          lambda: __import__('stable_retro'),                  required=False)
check('diplomacy',             lambda: __import__('diplomacy'),                     required=False)
check('games.games.avalon',    lambda: __import__('games.games.avalon.engine'),     required=False)
print()

# --- env_wrappers (this repo) ---
print('env_wrappers:')
check('env_wrappers',          lambda: __import__('env_wrappers'),                  required=False)
check('env_wrappers.gym_like', lambda: __import__('env_wrappers.gym_like'),         required=False)
check('env_wrappers.osworld_wrapper', lambda: __import__('env_wrappers.osworld_wrapper'), required=False)
print()

# --- Logging ---
print('Logging & Testing:')
check('loguru',                lambda: __import__('loguru'))
check('tensorboard',           lambda: __import__('tensorboard'))
check('pytest',                lambda: __import__('pytest'))
check('wandb',                 lambda: __import__('wandb'),                         required=False)
print()

# --- Internal modules ---
print('Internal Modules:')
check('trainer.coevolution.config',  lambda: __import__('trainer.coevolution.config'),  required=False)
check('trainer.common.metrics',      lambda: __import__('trainer.common.metrics'),      required=False)
check('skill_agents.grpo',           lambda: __import__('skill_agents.grpo'),           required=False)
check('skill_agents.lora',           lambda: __import__('skill_agents.lora'),           required=False)
check('rag.retrieval',               lambda: __import__('rag.retrieval'),               required=False)
check('decision_agents',             lambda: __import__('decision_agents'),             required=False)
check('data_structure',              lambda: __import__('data_structure'),              required=False)
check('API_func',                    lambda: __import__('API_func'),                    required=False)
print()

# --- Summary ---
print('=' * 50)
if failures:
    print(f'{len(failures)} REQUIRED check(s) FAILED:')
    for label, err in failures:
        print(f'  ✗ {label}: {err}')
    sys.exit(1)
else:
    print('All required checks passed.')
if warns:
    print(f'{len(warns)} optional check(s) skipped (install sibling repos / ROMs to fix):')
    for label, err in warns:
        print(f'  ⚠ {label}')
print('=' * 50)
"

echo
echo "============================================================"
echo "  Installation complete!"
echo "============================================================"
echo
echo "  Activate:"
echo "    conda activate $ENV_NAME"
echo
echo "  Set PYTHONPATH (run from the parent directory of all repos):"
echo "    export PYTHONPATH=\$(pwd)/Multi-hop-Reasoning-VLM-Agent:\$(pwd)/AgentEvolver:\$(pwd)/GamingAgent:\$PYTHONPATH"
echo
echo "  Set API keys:"
echo "    cp $REPO_DIR/.env.example $REPO_DIR/.env"
echo "    # Edit .env with your API keys"
echo "    set -a && source $REPO_DIR/.env && set +a"
echo
echo "  Quick smoke tests:"
echo "    python -c \"from API_func import api_call; print('API_func OK')\""
echo "    python -c \"import gym_v, gym_v.envs; print('Temporal/*:', sum(1 for k in gym_v.registry if k.startswith('Temporal/')))\""
echo "    python -c \"from env_wrappers.gym_like import make_gaming_env; e=make_gaming_env('twenty_forty_eight'); o=e.reset(); print('2048 OK')\""
echo "    pytest $REPO_DIR/tests/ -q"
echo
echo "  For Super Mario, install the orak-mario env separately:"
echo "    bash $REPO_DIR/install/install_orak_mario.sh"
echo
echo "  For BrowserGym + OSWorld benchmarks (own envs):"
echo "    bash $REPO_DIR/install/install_browsergym.sh"
echo "    bash $REPO_DIR/install/install_osworld.sh"
echo
echo "  Known nominal warning:"
echo "    'gamingagent 0.1.0 requires numpy==1.24.4' — benign; we run numpy 2.x."
echo
echo "============================================================"
