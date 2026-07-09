# Installation

For the full install guide (troubleshooting, orak-mario, environment diagrams),
see [install/README.md](install/README.md).

## Requirements

- **Python**: 3.9–3.11
- **OS**: Linux (Ubuntu 20.04+ recommended). macOS for development only (no vLLM / FSDP).
- **GPU**: 8× A100-80GB recommended for full co-evolution training. 1× GPU (24+ GB)
  sufficient for inference and baselines.
- **CUDA**: 12.x drivers installed on the host.

## Quick Start

```bash
# 1. Clone all repos into the same parent directory
mkdir -p ~/cos-play && cd ~/cos-play

git clone https://github.com/wuxiyang1996/cos-play.git Game-AI-Agent
git clone https://github.com/lmgame-org/GamingAgent.git
git clone https://github.com/modelscope/AgentEvolver.git
git clone https://github.com/nicholascpark/orak.git Orak    # optional, for Super Mario

# 2. Install the main environment (creates conda env + all deps)
bash Game-AI-Agent/install/install_main_env.sh

# 3. Install the orak-mario environment (if running Super Mario)
bash Game-AI-Agent/install/install_orak_mario.sh

# 4. Configure API keys
cp Game-AI-Agent/.env.example Game-AI-Agent/.env
# Edit .env with your keys (OpenAI, Anthropic, Google, OpenRouter)

# 5. Activate
conda activate game-ai-agent
export PYTHONPATH=$(pwd)/Game-AI-Agent:$(pwd)/AgentEvolver:$(pwd)/GamingAgent:$PYTHONPATH
set -a && source Game-AI-Agent/.env && set +a
```

## Alternative Install Methods

If you prefer not to use the automated install script:

```bash
cd Game-AI-Agent

# Option A: pip install (editable mode — registers all packages)
pip install -e .

# Option B: pip install from requirements file
pip install -r requirements.txt
```

Both options require you to install PyTorch with CUDA separately:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Optional Dependencies

- **RAG (embeddings)**: `pip install -r rag/requirements.txt`
  See [rag/README.md](rag/README.md) for text and multimodal embedding setup.

- **Skill agents / boundary proposal**: `pip install -r skill_agents/boundary_proposal/requirements.txt`
  **Infer segmentation**: `pip install -r skill_agents/infer_segmentation/requirements.txt`

- **Game environments**: See [install/README.md](install/README.md),
  [env_wrappers/setup_gamingagent_eval_env.md](env_wrappers/setup_gamingagent_eval_env.md),
  [env_wrappers/README.md](env_wrappers/README.md).
