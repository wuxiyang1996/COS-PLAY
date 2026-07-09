# evaluate_orak -- Orak Benchmark Integration

Integration of the [Orak](https://github.com/krafton-ai/Orak) game benchmark
(krafton-ai/Orak) into the Game-AI-Agent framework.

Orak is a foundational benchmark for evaluating LLM agents across **12 popular
video games** spanning 6 genres. This module provides NL wrappers, Gymnasium
interfaces, and a benchmark runner so that all Orak games can be driven by
Game-AI-Agent's decision agents.

Paper: [arXiv:2506.03610](https://arxiv.org/pdf/2506.03610)

---

## Game List

### Free Games (no purchase required)

| Game | Genre | Setup | Action Format | Scoring |
|------|-------|-------|---------------|---------|
| **2048** | Puzzle | `pip install pygame` | `up` / `down` / `left` / `right` | min(score/20000*100, 100) |
| **Super Mario** | Action | `pip install gym-super-mario-bros` (ROM bundled) | `Jump Level : 0`..`6` | x_pos / 3161 * 100 |
| **Street Fighter III** | Action | Diambra (free) + Docker + ROM | Character move names (e.g. `Fireball`) | Stages cleared |
| **StarCraft II** | Strategy | Battle.net (free) + SC2 client + maps | 5 macro actions per step (e.g. `TRAIN ZEALOT`) | Victory / Defeat |
| **StarCraft II Multi** | Strategy | Same as StarCraft II | 5 macro actions per step | Win / Loss |
| **Pokemon Red** | RPG | `pyboy` + `.gb` ROM + map data | Directional + tool calls (e.g. `move_to(x,y)`, `warp_with_warp_point`) | Milestone flags (0-12) |

### Paid Games ($6-$25 Steam purchase each)

| Game | Genre | Price | Action Format | Scoring |
|------|-------|-------|---------------|---------|
| **Baba Is You** | Puzzle | ~$15 | `idle` / `left` / `right` / `up` / `down` [N] | 0/20/40/100 |
| **Slay the Spire** | Strategy | ~$25 | `PLAY <idx>` / `END` / `CHOOSE <idx>` / `SKIP` | Floor reached (max 50) |
| **Darkest Dungeon** | RPG | ~$25 | `attack target X using skill slot Y` | 0.4*combat + 0.3*survival + 0.3*(1-stress) |
| **Ace Attorney** | Adventure | ~$20 | `Ok` / `Press` / `Present evidence <idx>` | Milestone rewards |
| **Her Story** | Adventure | ~$6 | `Search <keyword>` / `Play Video <idx>` | Videos viewed / 272 |
| **Stardew Valley** | Simulation | ~$15 | Python skill list (e.g. `["till_soil", "plant_seeds"]`) | Task-dependent |

> **Minecraft** is listed as free by Orak but requires the Minecraft Java
> client (~$30) + Voyager + Node.js setup. Its action space is JavaScript
> async functions.

---

## Directory Structure

```
evaluate_orak/
  __init__.py              Package init
  orak_nl_wrapper.py       NL wrapper (OrakNLWrapper, make_orak_env)
  orak_gym_like.py         Gymnasium-compatible wrapper (make_orak_gaming_env)
  run_orak_benchmark.py    CLI benchmark runner
  README.md                This file
```

---

## Prerequisites

1. **Clone the Orak repo** (already done at `game_agent/Orak/`):
   ```bash
   cd game_agent
   git clone --branch release https://github.com/krafton-ai/Orak.git
   ```

2. **Install base dependencies** (in the base env for 2048):
   ```bash
   pip install pygame omegaconf gymnasium dataclass_wizard dill dacite tenacity
   ```

---

## Dedicated Conda Environments

### Super Mario (`orak-mario`)

`nes-py` requires Python 3.10-3.12 and NumPy < 2. A dedicated conda env avoids
conflicts with the base Python 3.13 environment.

```bash
# Create env
conda create -n orak-mario python=3.11 -y

# Install all deps
/workspace/miniconda3/envs/orak-mario/bin/pip install \
  "gym==0.26.2" "gym-super-mario-bros==7.4.0" "nes-py==8.2.1" "numpy<2" \
  torch torchvision opencv-python-headless scikit-image Pillow requests \
  omegaconf gymnasium dacite dataclass-wizard dill tenacity pygame

# Activate and run
source evaluate_orak/setup_orak_mario.sh
python -c "from evaluate_orak.orak_nl_wrapper import make_orak_env; \
           e=make_orak_env('super_mario'); o,i=e.reset(); print(o[:200]); e.close()"
```

### StarCraft II (`orak-sc2`)

Requires the SC2 headless Linux client (free, no Battle.net account needed on
Linux) and `burnysc2`.

```bash
# 1. Create env
conda create -n orak-sc2 python=3.11 -y

# 2. Install Python deps
/workspace/miniconda3/envs/orak-sc2/bin/pip install \
  "burnysc2>=6.5" nest_asyncio dacite numpy Pillow omegaconf \
  gymnasium dataclass-wizard dill tenacity pygame requests mss screeninfo

# 3. Download SC2 headless Linux client (4.10, ~3.9GB)
mkdir -p /workspace/game_agent/StarCraftII && cd /workspace/game_agent/StarCraftII
wget "https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip" -O SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip
ln -s StarCraftII/Maps StarCraftII/maps  # burnysc2 expects lowercase
rm SC2.4.10.zip

# 4. Download ladder maps (2019 season, compatible with SC2 4.10)
# Maps bundled in the SC2 zip already include Ladder2019Season1:
#   AutomatonLE, CyberForestLE, KairosJunctionLE, KingsCoveLE,
#   NewRepugnancyLE, PortAleksanderLE, YearZeroLE

# 5. Activate and run
source evaluate_orak/setup_orak_sc2.sh
python -c "from evaluate_orak.orak_nl_wrapper import make_orak_env; \
           e=make_orak_env('star_craft'); o,i=e.reset(); print(o[:200]); e.close()"
```

**Note:** SC2 4.10 is the latest headless Linux build from Blizzard. The 2023
ladder maps in Orak's default config require newer SC2 versions only available
on Windows/Mac. The env auto-detects the SC2 version and falls back to
compatible 2019 ladder maps on Linux.

### Per-game dependencies (other games)

```bash
# Street Fighter III
pip install diambra-arena diambra ultralytics

# Pokemon Red
pip install pyboy==2.5.2
```

---

## Pokemon Red — Full Setup (text-only, no screens)

Pokemon Red in this repo is **text-only**: state is derived from PyBoy memory reads (map, position, party, dialog, battle), not from screenshots. Cold-start and evaluation use the Orak Pokemon Red env and toolset.

### Requirements

1. **ROM**: A `.gb` ROM (e.g. `Pokemon - Red Version (USA, Europe).gb`). Place or symlink at `GamingAgent/gamingagent/configs/custom_06_pokemon_red/rom/pokemon.gb` (or pass `--rom_path` to the cold-start script).
2. **PyBoy**: `pip install pyboy==2.5.2`
3. **Map data** (for navigation tools): Clone the [pret/pokered](https://github.com/pret/pokered) disassembly into `Orak/src/mcp_game_servers/pokemon_red/game/pokered`, then from the Orak repo root run:
   ```bash
   cd game_agent/Orak
   python3 src/mcp_game_servers/pokemon_red/game/utils/map_preprocess.py
   ```
   This generates `processed_map/` with collision and warp data. On Linux, map names from memory (e.g. `RedsHouse2f`) may differ in case from generated files (`RedsHouse2F.py`); symlinks can be added so both resolve.

### Cold-start (trajectory generation)

From `Game-AI-Agent`:

```bash
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$(pwd)/../Orak/src:$PYTHONPATH"
bash cold_start/run_coldstart_pokemon_red.sh --episodes 3 --max_steps 200 --verbose --no_label
```

Output: `cold_start/output/gpt54_pokemon_red/pokemon_red/`. See `cold_start/readme.md` for full options and natural termination conditions.

### Observation and actions

- **State**: Field/Dialog/Battle/Title; map name, dimensions, position; full map grid (walkable, warp points, objects); party, badges, bag, money; dialog text when in Dialog state.
- **Actions**: High-level tools (`move_to`, `warp_with_warp_point`, `continue_dialog`, `select_move_in_battle`, `interact_with_object`, etc.) when map data exists; raw buttons (`up`/`down`/`left`/`right`/`a`/`b`/`start`/`select`) for menu/dialog choices and fallback movement.

---

## Quick Start

### Python API

```python
from evaluate_orak.orak_nl_wrapper import make_orak_env

# Create a 2048 environment (works out of the box)
env = make_orak_env("twenty_fourty_eight", max_steps=1000)
obs, info = env.reset()
print(obs)
# "Board of 2048 Games: ..."

obs, reward, terminated, truncated, info = env.step("down")
env.close()
```

### Gymnasium Interface

```python
from evaluate_orak.orak_gym_like import make_orak_gaming_env

env = make_orak_gaming_env("orak_twenty_fourty_eight", max_steps=500)
obs_dict, info = env.reset()       # obs_dict["text"] = "Board of 2048..."
obs_dict, reward, term, trunc, info = env.step("left")
env.close()
```

### CLI Benchmark

```bash
cd Game-AI-Agent
export PYTHONPATH="$(pwd):$(pwd)/../Orak/src:$PYTHONPATH"

# Single game
python evaluate_orak/run_orak_benchmark.py \
    --game twenty_fourty_eight --episodes 3 --model gpt-5-mini

# All games
python evaluate_orak/run_orak_benchmark.py --all_games --episodes 1

# With VLM decision agent
python evaluate_orak/run_orak_benchmark.py \
    --game star_craft --agent_type vlm --episodes 1 --max_steps 100
```

Output is saved to `orak_benchmark_output/<game>/episode_XXX.json`.

---

## Architecture

```
                  ┌─────────────────┐
                  │  Decision Agent  │  (dummy_agent / VLMDecisionAgent)
                  │  (NL actions)    │
                  └────────┬────────┘
                           │ string action
                  ┌────────▼────────┐
                  │  OrakNLWrapper   │  evaluate_orak/orak_nl_wrapper.py
                  │  reset() / step()│
                  └────────┬────────┘
                           │ text2action() / obs2text()
                  ┌────────▼────────┐
                  │  Orak BaseEnv    │  Orak/src/mcp_game_servers/<game>/
                  │  (per-game impl) │
                  └─────────────────┘
```

The `OrakNLWrapper` bridges Orak's `BaseEnv` interface (which uses
`initial_obs()`, `obs2text()`, `text2action()`, `step()`, `evaluate()`) to
the standard `reset()`/`step()` Gymnasium-style loop that Game-AI-Agent
decision agents expect.

---

## Game Details

### 2048 (Puzzle)
- **Env class**: `TwentyFourtyEightEnv`
- **Obs**: 4x4 board grid + score
- **Actions**: `up`, `down`, `left`, `right`
- **Termination**: 5 consecutive no-change steps, or win/lose
- **Score**: Raw tile-merge score

### Super Mario (Action)
- **Env class**: `SuperMarioEnv`
- **Obs**: Mario position + nearby objects (bricks, enemies, pipes, pits)
- **Actions**: `Jump Level : N` where N = 0 (no jump) to 6 (max jump)
- **Score**: x_pos / 3161 * 100

### Street Fighter III (Action)
- **Env class**: `StreetFighterEnv`
- **Obs**: Character names, distance, health, super bar, stun, timer
- **Actions**: Character-specific moves (Ken/Q/Chun-Li): Move Closer, Fireball, etc.
- **Score**: Stages cleared

### StarCraft II (Strategy)
- **Env class**: `StarCraftEnv`
- **Obs**: Resources, Buildings, Units, Research, In Progress, Enemy intel
- **Actions**: 5 sequential macro commands per step (TRAIN/BUILD/RESEARCH/SCOUT/ATTACK/RETREAT)
- **Score**: Victory or Defeat

### Slay the Spire (Strategy)
- **Env class**: `SlayTheSpireEnv`
- **Obs**: Player HP/energy, cards in hand, monsters, relics
- **Actions**: `PLAY <card_idx> [target_idx]`, `END`, `CHOOSE <idx>`, `SKIP`
- **Score**: Floor reached (max 50 = boss defeated)

### Pokemon Red (RPG)
- **Env class**: `PokemonRedEnv` (Orak)
- **Obs**: Text-only: map name, (x,y), full map grid (O/X/WarpPoint/objects), party, badges, bag, money, dialog text. No screens.
- **Actions**: High-level tools (`move_to(x,y)`, `warp_with_warp_point`, `continue_dialog`, `select_move_in_battle`, `interact_with_object`, etc.) when `processed_map` data exists; raw buttons (`up`/`down`/`left`/`right`/`a`/`b`/`start`/`select`) for choices and fallback.
- **Score**: Orak storyline milestone flags (0-12). Cold-start: `cold_start/run_coldstart_pokemon_red.sh`; see `cold_start/readme.md`.

### Darkest Dungeon (RPG)
- **Env class**: `DarkestDungeonEnv`
- **Obs**: Hero stats (HP, stress, rank, skills), enemy formation
- **Actions**: `attack target X using skill slot Y`, `heal`, `swap`, `idle`
- **Score**: 0.4*combat + 0.3*survival + 0.3*(1-stress/800)

### Ace Attorney (Adventure)
- **Env class**: `PwaatEnv`
- **Obs**: Conversation dialog, evidence/profile records
- **Actions**: `Ok`, `Press`, `Present evidence <idx>`, option index
- **Score**: Milestone rewards from REWARD_CHECKER

### Her Story (Adventure)
- **Env class**: `HerStoryEnv`
- **Obs**: Search results with video metadata and scripts
- **Actions**: `Search <keyword>`, `Play Video <idx>`
- **Score**: Unique videos viewed (272 total)

### Baba Is You (Puzzle)
- **Env class**: `BabaIsYouEnv`
- **Obs**: Level grid with objects, active rules (X IS Y)
- **Actions**: `idle`, `left`, `right`, `up`, `down` (optional step count)
- **Score**: 100=win, 40=WIN rule exists, 20=WALL IS STOP broken, 0=fail

### Minecraft (Simulation)
- **Env class**: `MinecraftEnv`
- **Obs**: Biome, time, nearby blocks/entities, inventory, health
- **Actions**: JavaScript async functions with `bot` parameter
- **Score**: 1 if success_condition in inventory, 0 otherwise

### Stardew Valley (Simulation)
- **Env class**: `StardewValleyEnv`
- **Obs**: Player, obstacles, toolbar, crops, world state
- **Actions**: Python skill list (e.g. `["till_soil", "plant_seeds"]`)
- **Score**: Task-dependent (farm cleanup, cultivation, shopping, earn money)
