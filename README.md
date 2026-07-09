# COS-PLAY

**Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Tasks**

Official clean implementation of [COS-PLAY](https://github.com/wuxiyang1996/COS-PLAY) (arXiv:2604.20987).

An LLM **decision agent** retrieves skills from a learnable **skill bank** to act in long-horizon games. After each episode, a **skill bank agent** segments trajectories, learns contracts, and curates the bank. Training uses **GRPO + FSDP** with five function-specific **LoRA** adapters on Qwen3-8B.

## Repository layout

```
cos-play/
├── decision_agents/     # LLM decision loop (summary → skill → action → reward)
├── skill_agents_grpo/   # Skill bank pipeline + GRPO trainers (canonical)
├── skill_agents/        # → symlink to skill_agents_grpo (import compatibility)
├── trainer/
│   ├── coevolution/     # Main co-evolution orchestrator
│   └── SFT/             # Cold-start SFT for 5 LoRA adapters
├── data_structure/      # Episode, Experience, buffers
├── env_wrappers/        # NL wrappers for game environments
├── rag/                 # Qwen3-Embedding retrieval
├── cold_start/          # Teacher rollout generation (GPT-5.4)
├── labeling/            # Episode labeling + skill extraction
├── inference/           # Evaluation runners
├── scripts/             # Training / eval entry points
├── configs/             # YAML configs
├── install/             # Conda env installer + requirements
└── tests/
```

**Excluded from this repo** (generated at runtime): `runs/`, `output/`, `wandb/`, cold-start caches, checkpoints.

## Quick start

### 1. Clone sibling environments

```bash
mkdir -p cos-play-workspace && cd cos-play-workspace

git clone https://github.com/wuxiyang1996/COS-PLAY.git cos-play
git clone https://github.com/lmgame-org/GamingAgent.git
git clone https://github.com/modelscope/AgentEvolver.git
# optional: git clone https://github.com/ModalMinds/gym-v.git
```

### 2. Install environment

```bash
bash cos-play/install/install_main_env.sh
conda activate game-ai-agent
```

### 3. Configure API keys

```bash
cp cos-play/.env.example cos-play/.env
# edit keys, then:
set -a && source cos-play/.env && set +a
```

### 4. PYTHONPATH

```bash
export PYTHONPATH=$(pwd)/cos-play:$(pwd)/AgentEvolver:$(pwd)/GamingAgent:$PYTHONPATH
```

## Pipeline (5 steps)

| Step | What | Command |
|------|------|---------|
| 0 | Download pre-generated cold-start (optional) | `python labeling/download_cold_start.py` |
| 1 | Generate teacher rollouts | `bash cold_start/run_coldstart_gpt54.sh` |
| 2 | Label + extract skills | `bash labeling/run_label_with_skills.sh` |
| 3 | SFT cold-start (5 LoRA adapters) | `bash scripts/run_sft_coldstart.sh` |
| 4 | Co-evolution training | `python scripts/run_coevolution.py --load-decision-adapters runs/sft_coldstart/decision --load-skillbank-adapters runs/sft_coldstart/skillbank` |
| 5 | Inference | `python scripts/qwen3_decision_agent.py --games tetris` |

Pre-generated data: [HuggingFace `IntelligenceLab/Cos-Play-Cold-Start`](https://huggingface.co/datasets/IntelligenceLab/Cos-Play-Cold-Start) (~538 MB, 8 games).

## Games

| Game | Environment |
|------|-------------|
| 2048, Tetris, Candy Crush | [GamingAgent](https://github.com/lmgame-org/GamingAgent) |
| Avalon, Diplomacy | [AgentEvolver](https://github.com/modelscope/AgentEvolver) |
| Super Mario | [Orak](https://github.com/krafton-ai/Orak) (`install/install_orak_mario.sh`) |

## Hardware

| Use case | GPUs |
|----------|------|
| Full co-evolution | 8× A100 80GB |
| Single-game training | 1–2× A100 |
| Inference (vLLM) | 1× GPU 24GB+ |

## Citation

```bibtex
@misc{wu2026coevolvingllmdecisionskill,
  title={Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Tasks},
  author={Xiyang Wu and Zongxia Li and Guangyao Shi and Alexander Duffy and Tyler Marques and Matthew Lyle Olson and Tianyi Zhou and Dinesh Manocha},
  year={2026},
  eprint={2604.20987},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
}
```

## License

MIT — see [LICENSE](LICENSE).
