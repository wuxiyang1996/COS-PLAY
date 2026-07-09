# Setting up the GamingAgent evaluation environment

**GamingAgent is not included in this repo.** You need the external [GamingAgent](https://github.com/lmgame-org/GamingAgent) repository. Follow these steps so the `evaluate_gamingagent` scripts can run with the GamingAgent (LMGame-Bench) environments.

## 1. GamingAgent repository (external)

Clone and install GamingAgent as a sibling of Game-AI-Agent (or elsewhere and add to `PYTHONPATH`):

```powershell
cd D:\ICML2026
git clone https://github.com/lmgame-org/GamingAgent.git
cd GamingAgent
pip install -e .
```

Or if GamingAgent is already cloned as a sibling folder:

```powershell
cd D:\ICML2026\GamingAgent
pip install -e .
```

## 2. Create conda environment (optional)

```powershell
conda create -n gamingagent_eval python=3.11 -y
conda activate gamingagent_eval
pip install -e D:\ICML2026\GamingAgent
pip install gymnasium numpy
```

## 3. Environment variables for LLM APIs

```powershell
$env:OPENAI_API_KEY = "your_api_key"
# Or if using api_keys module:
# Ensure api_keys.py exists with openai_api_key
```

## 4. Run evaluation tests

From the Game-AI-Agent codebase root with GamingAgent on PYTHONPATH:

```powershell
cd D:\ICML2026\Game-AI-Agent
$env:PYTHONPATH = (Get-Location).Path + ";" + (Get-Location).Path + "\..\GamingAgent"
python evaluate_gamingagent/test_gamingagent_dummy.py --game twenty_forty_eight --max_steps 20
python evaluate_gamingagent/test_gamingagent_dummy.py --game sokoban --max_steps 30 --mode random_nl
```

Available games (run with `--list-games`): twenty_forty_eight, sokoban, candy_crush, tetris, pokemon_red (requires ROM).
