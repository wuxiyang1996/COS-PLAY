# Installation

## Requirements

- Python 3.10–3.11
- CUDA 12.8+ (Blackwell GPUs need cu128+ PyTorch wheels)
- NVIDIA GPU with 24GB+ VRAM for inference; 8×80GB for full co-evolution

## Recommended: automated install

From the parent directory that contains `cos-play/`, `GamingAgent/`, and `AgentEvolver/`:

```bash
bash cos-play/install/install_main_env.sh
conda activate game-ai-agent
export PYTHONPATH=$(pwd)/cos-play:$(pwd)/AgentEvolver:$(pwd)/GamingAgent:$PYTHONPATH
```

See [install/README.md](install/README.md) for CUDA driver notes, gym-v ROM setup, and Orak Mario env.

## pip editable install (development)

```bash
cd cos-play
conda create -n game-ai-agent python=3.11 -y
conda activate game-ai-agent
pip install -e ".[all]"
pip install -r install/requirements.txt   # full training stack
```

## Super Mario (separate env)

```bash
bash install/install_orak_mario.sh
```

## WebShop + ALFWorld (text environments)

These are text-based interactive environments for evaluating agents on
shopping and household tasks.

```bash
bash scripts/install_text_envs.sh
```

**Prerequisites:**
- WebShop repo cloned to `/workspace/WebShop` (or set `WEBSHOP_ROOT`)
- Internet access for `alfworld-download` (~320 MB of game data)

After installation, both environments are available as COS-PLAY games:
`webshop` and `alfworld`. Add them to `SKILL_BANK_GAMES` in
`trainer/coevolution/config.py` or pass `--games webshop,alfworld`
to `scripts/run_coevolution.py`.

Full usage — serving SkillRL checkpoints, paired inference-time-control
evaluations (including the official Lucene WebShop protocol and the
official ALFWorld ID protocol), and reference results — is documented in
[docs/ALFWORLD_WEBSHOP.md](docs/ALFWORLD_WEBSHOP.md).

## Data

Download pre-labeled cold-start data (skip Steps 1–2 of the pipeline):

```bash
python labeling/download_cold_start.py
```

Dataset: [IntelligenceLab/Cos-Play-Cold-Start](https://huggingface.co/datasets/IntelligenceLab/Cos-Play-Cold-Start)
