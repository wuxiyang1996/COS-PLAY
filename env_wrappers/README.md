# env_wrappers -- Game Environment Package

Unified package for game environment NL wrappers, Gymnasium adapters, game
configurations, and benchmark runners used by Game-AI-Agent.

## Overview

This package provides:

1. **NL wrappers** that convert raw game observations to/from natural language
   so LLM-based decision agents can interact with game environments using text.
2. **Gymnasium-compatible adapters** that expose native game APIs as standard
   `reset()`/`step()` interfaces.
3. **Game configurations** (per-game defaults for episodes, max steps, etc.).
4. **Benchmark runners** (CLI tools for automated evaluation).

Supported games:

| Game | Wrapper | Source |
|------|---------|--------|
| 2048 | `GamingAgentNLWrapper` | GamingAgent (LMGame-Bench) |
| Candy Crush | `GamingAgentNLWrapper` | GamingAgent (LMGame-Bench) |
| Tetris | `GamingAgentNLWrapper` + `TetrisMacroWrapper` | GamingAgent (LMGame-Bench) |
| Super Mario | `OrakNLWrapper` | Orak (krafton-ai/Orak) |
| Avalon | `AvalonNLWrapper` | AgentEvolver |
| Diplomacy | `DiplomacyNLWrapper` | AgentEvolver / AI-Diplomacy |

---

## Module Structure

```
env_wrappers/
  __init__.py                 Public API surface
  avalon_nl_wrapper.py        Avalon hidden-role deduction (5-10 agents)
  diplomacy_nl_wrapper.py     Diplomacy strategic negotiation (7 powers)
  gamingagent_nl_wrapper.py   GamingAgent / LMGame-Bench NL wrapper
  orak_nl_wrapper.py          Orak NL wrapper (Super Mario, etc.)
  tetris_macro_wrapper.py     Tetris macro-action wrapper (placement-level)
  subprocess_env.py           Subprocess isolation for dependency conflicts
  subprocess_env_worker.py    Worker process for SubprocessEnv
  game_configs.py             Per-game default configs (GameConfig dataclass)
  gym_like.py                 Gymnasium adapter for GamingAgent envs
  run_benchmark.py            LMGame-Bench CLI benchmark suite
  run_orak_benchmark.py       Orak CLI benchmark runner
  test_gamingagent_dummy.py   GamingAgent smoke test (single-game)
  test_orak_mario_gpt54.py    GPT-5.4 Super Mario evaluator
  run_gpt54_mario.sh          Shell wrapper for Mario evaluation
  setup_orak_mario.sh         Activate orak-mario conda env + PYTHONPATH
  setup_gamingagent_eval_env.md  GamingAgent setup instructions
  envs/orak-mario/            Orak-Mario conda env install (requirements, install.sh)
```

---

## Quick Start

### GamingAgent games (2048, Candy Crush, Tetris)

```bash
conda activate game-ai-agent
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

# List all games
python env_wrappers/run_benchmark.py --list

# Run full benchmark with GPT-5.4
python env_wrappers/run_benchmark.py --model gpt-5.4

# Run specific games
python env_wrappers/run_benchmark.py --games candy_crush tetris --episodes 5

# Save results to JSON
python env_wrappers/run_benchmark.py --model gpt-5.4 --output results/baseline.json

# Single-game smoke test
python env_wrappers/test_gamingagent_dummy.py \
    --game twenty_forty_eight --episodes 3 --mode llm --model gpt-5.4
```

### Orak games (Super Mario)

Super Mario requires the `orak-mario` conda env (nes-py needs NumPy < 2):

```bash
# Install (one-time)
bash env_wrappers/envs/orak-mario/install.sh

# Activate
source env_wrappers/setup_orak_mario.sh

# Run benchmark
python env_wrappers/run_orak_benchmark.py \
    --game super_mario --episodes 3 --max_steps 100

# GPT-5.4 evaluation
bash env_wrappers/run_gpt54_mario.sh --episodes 5
```

### Python API

```python
# GamingAgent
from env_wrappers.gym_like import make_gaming_env
from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper

env = GamingAgentNLWrapper(make_gaming_env("twenty_forty_eight", max_steps=200))
obs, info = env.reset()
obs, reward, term, trunc, info = env.step("up")
env.close()

# Orak (Super Mario)
from env_wrappers.orak_nl_wrapper import make_orak_env

env = make_orak_env("super_mario", max_steps=100)
obs, info = env.reset()
obs, reward, term, trunc, info = env.step("Jump Level : 3")
env.close()

# Benchmark API
from env_wrappers.run_benchmark import run_benchmark
results = run_benchmark(games=["tetris", "candy_crush"], model="gpt-5.4")
```

---

## Architecture

```
Decision Agent  (dummy_agent / VLMDecisionAgent / Qwen3 agent)
       |  NL action string
       v
NL Wrapper  (GamingAgentNLWrapper / OrakNLWrapper / AvalonNLWrapper / ...)
       |  obs2text() / text2action()
       v
Gymnasium Adapter  (gym_like._GymLikeWrapper / OrakNLWrapper internal)
       |  reset() / step()
       v
Native Game Env  (GamingAgent / Orak BaseEnv / AgentEvolver)
```

`SubprocessEnv` enables running environments in isolated processes when the
game engine has incompatible dependency versions (e.g. Orak's numpy<2).

---

## Game Details

### GamingAgent (LMGame-Bench) Games

| Game | Actions | Notes |
|------|---------|-------|
| **2048** | `up`, `down`, `left`, `right` | Pure Python, no external deps |
| **Candy Crush** | coordinate swaps, e.g. `((0,5),(1,5))` | Dynamic action space |
| **Tetris** | `move_left`, `move_right`, `rotate_cw`, `rotate_ccw`, `hard_drop`, `soft_drop` | Use `TetrisMacroWrapper` for placement-level actions |

### Orak Games

| Game | Genre | Actions | Scoring |
|------|-------|---------|---------|
| **Super Mario** | Action | `Jump Level : 0`..`6` | x_pos / 3161 * 100 |
| **2048 (Orak)** | Puzzle | `up`/`down`/`left`/`right` | min(score/20000*100, 100) |
| **Street Fighter III** | Action | Character move names | Stages cleared |

See `game_configs.py` for the full catalog of all supported games and their
default configurations.

---

## Agent Modes

| Mode | Description |
|------|-------------|
| `llm` | Uses `language_agent_action()` with the specified model via OpenRouter |
| `random_nl` | Picks a random valid action each step |
| `fallback` | Always picks the first valid action |
