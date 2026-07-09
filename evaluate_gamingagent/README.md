# LMGame-Bench Evaluation Suite

Evaluation harness for [GamingAgent (LMGame-Bench)](https://github.com/GamingAgent/LMGame-Bench) environments, adapted for use with the Game-AI-Agent codebase.

## Overview

This module provides a Gymnasium-compatible wrapper around native GamingAgent environments and a single-command benchmark runner. All code lives in `evaluate_gamingagent/`; the external `GamingAgent` repo is used as a read-only reference and is never modified.

## Game Inventory

LMGame-Bench contains **11 games** across 3 categories. **7 are included** in the training benchmark:

| # | Game | Category | Actions | Status |
|---|------|----------|---------|--------|
| 1 | **2048** | custom | `up`, `down`, `left`, `right` | Available |
| 2 | **Sokoban** | custom | `up`, `down`, `left`, `right` | Available |
| 3 | **Candy Crush** | custom | coordinate swaps, e.g. `((0,5),(1,5))` | Available |
| 4 | **Tetris** | custom | `move_left`, `move_right`, `rotate_cw`, `rotate_ccw`, `hard_drop`, `soft_drop` | Available |
| 5 | **Pokemon Red** | custom | `a`, `b`, `start`, `select`, `up`, `down`, `left`, `right` | Available (ROM) |
| 6 | **Tic-Tac-Toe** | zoo | `place 0` .. `place 8` | Available |
| 7 | **Texas Hold'em** | zoo | `call`, `raise`, `fold`, `check` | Available |

### Excluded Games

| Game | Reason |
|------|--------|
| Doom (Basic) | Text-only mode too limited for meaningful training |
| Super Mario Bros | Requires NES ROM via `stable-retro` |
| Ace Attorney | Requires GBA ROM via `stable-retro` |
| 1942 | Requires NES ROM via `stable-retro` |

## Scope of This Wrapper

This wrapper targets the **5 custom-category games** and 2 zoo games:

| # | Game | Actions | Requirements |
|---|------|---------|-------------|
| 1 | **2048** | `up`, `down`, `left`, `right` | None |
| 2 | **Sokoban** | `up`, `down`, `left`, `right` | None |
| 3 | **Candy Crush** | coordinate swaps, e.g. `((0,5),(1,5))` | None |
| 4 | **Tetris** | `move_left`, `move_right`, `rotate_cw`, `rotate_ccw`, `hard_drop`, `soft_drop` | None |
| 5 | **Pokemon Red** | `a`, `b`, `start`, `select`, `up`, `down`, `left`, `right` | PyBoy + `.gb` ROM |

The first four games are fully self-contained. Pokemon Red requires `pyboy` and a Game Boy ROM placed at `GamingAgent/gamingagent/configs/custom_06_pokemon_red/rom/pokemon.gb`. The remaining games are either excluded entirely (ROM dependencies for retro games) or deferred (PettingZoo multi-agent environments) and may be added in future iterations.

## Module Structure

```
evaluate_gamingagent/
  __init__.py             # Package marker
  README.md               # This file
  game_configs.py         # Per-game default configs (GameConfig dataclass)
  gym_like.py             # Gymnasium-compatible env adapter
  run_benchmark.py        # Single-command evaluation suite
  test_gamingagent_dummy.py  # Low-level single-game test harness
```

## Quick Start

### Prerequisites

```bash
# Activate the conda environment
conda activate game-ai-agent

# Ensure GamingAgent is cloned alongside Game-AI-Agent:
#   game_agent/
#     Game-AI-Agent/      <- this repo
#     GamingAgent/        <- reference repo (read-only)
```

### Pokemon Red Setup (optional)

```bash
# Install PyBoy emulator
pip install pyboy==2.5.2

# Place the ROM (symlink or copy)
mkdir -p ../GamingAgent/gamingagent/configs/custom_06_pokemon_red/rom
ln -s /path/to/your/pokemon_red.gb \
    ../GamingAgent/gamingagent/configs/custom_06_pokemon_red/rom/pokemon.gb
```

### List All Games

```bash
python evaluate_gamingagent/run_benchmark.py --list
```

### Run the Full Benchmark

```bash
# All 7 available games with GPT-5.4
python evaluate_gamingagent/run_benchmark.py --model gpt-5.4

# Save results to JSON
python evaluate_gamingagent/run_benchmark.py --model gpt-5.4 --output results/baseline.json
```

### Run Specific Games

```bash
# By name
python evaluate_gamingagent/run_benchmark.py --games tictactoe candy_crush tetris

# By category (custom / zoo)
python evaluate_gamingagent/run_benchmark.py --category zoo
```

### Override Settings

```bash
# Override episodes and max steps for all games
python evaluate_gamingagent/run_benchmark.py --episodes 5 --max-steps 100

# Dry run (preview what would execute)
python evaluate_gamingagent/run_benchmark.py --dry-run
```

### Run a Single Game (Low-Level)

```bash
python evaluate_gamingagent/test_gamingagent_dummy.py \
    --game twenty_forty_eight \
    --max_steps 200 \
    --episodes 3 \
    --mode llm \
    --model gpt-5.4 \
    --verbose
```

### Programmatic API

```python
from evaluate_gamingagent.run_benchmark import run_benchmark

results = run_benchmark(
    games=["tictactoe", "candy_crush"],
    model="gpt-5.4",
    episodes_override=10,
    output_path="results/run1.json",
)

for r in results:
    print(f"{r['display_name']}: mean_reward={r['mean_reward']:.2f}")
```

## Baseline Results (GPT-5.4)

Results from 3 episodes per game using the LLM agent mode:

| Game | Mean Reward | Mean Steps | Notes |
|------|------------|------------|-------|
| 2048 | 214.67 | 53 | High variance (48-528); actively merging tiles |
| Candy Crush | 80.33 | 50 | Good match-finding from effective swaps |
| Tetris | 10.33 | 10 | ~10 pieces placed; tends to spam `hard_drop` |
| Sokoban | -0.60 | 6 | Spatial reasoning is hard in text; stuck early |
| Tic-Tac-Toe | 1.00 | 3 | 100% win rate vs random opponent |
| Texas Hold'em | -1.00 | 2 | Overly conservative (folds too much) |

## Architecture

```
User Code
    |
    v
run_benchmark.py          # orchestrates multi-game evaluation
    |
    v
test_gamingagent_dummy.py  # per-game episode runner
    |
    v
GamingAgentNLWrapper       # converts obs to natural language
    |
    v
gym_like._GymLikeWrapper   # Gymnasium API adapter
    |
    v
Native GamingAgent Env     # e.g. TwentyFortyEightEnv, CandyCrushEnv
```

The `gym_like.py` adapter handles the mismatch between the native GamingAgent environment interface (custom `Observation` objects, 6-tuple step returns, `episode_id` reset args) and the standard Gymnasium API (dict observations, 5-tuple step returns).

## Agent Modes

| Mode | Description |
|------|-------------|
| `llm` | Uses `language_agent_action()` with the specified model via OpenRouter |
| `random_nl` | Picks a random valid action each step |
| `fallback` | Always picks the first valid action |

## Output Format

When `--output` is specified, results are saved as JSON:

```json
{
  "benchmark": "LMGame-Bench",
  "timestamp": "2026-03-11T02:08:35.424030",
  "model": "gpt-5.4",
  "mode": "llm",
  "total_elapsed_s": 455.2,
  "total_games_in_benchmark": 11,
  "games_run": 7,
  "results": [
    {
      "game": "candy_crush",
      "display_name": "Candy Crush",
      "category": "custom",
      "status": "OK",
      "episodes_run": 3,
      "mean_reward": 80.33,
      "min_reward": 50.0,
      "max_reward": 123.0,
      "mean_steps": 50.0,
      "elapsed_s": 151.7,
      "episode_details": [...]
    }
  ]
}
```
