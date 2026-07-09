# Cold-Start Data Generation

Generate initial trajectory data and skill seeds for the Game-AI-Agent system.

## Scope: Environments and Games We Use

We have **four** cold-start generators covering **8 games** across three environment stacks:

### 1. LMGame-Bench (`generate_cold_start_gpt54.py`)

| # | Game | Registry Key | Actions |
|---|------|--------------|---------|
| 1 | **2048** | `twenty_forty_eight` | `up`, `down`, `left`, `right` |
| 2 | **Sokoban** | `sokoban` | `up`, `down`, `left`, `right` |
| 3 | **Candy Crush** | `candy_crush` | coordinate swaps, e.g. `((0,5),(1,5))` |
| 4 | **Tetris** | `tetris` | `move_left`, `move_right`, `rotate_cw`, `rotate_ccw`, `hard_drop`, `soft_drop` |

### 2. AgentEvolver (`generate_cold_start_evolver.py`)

| # | Game | Registry Key | Actions |
|---|------|--------------|---------|
| 5 | **Avalon** | `avalon` | team proposals, approve/reject votes, pass/fail, assassination target |
| 6 | **Diplomacy** | `diplomacy` | unit orders (move, hold, support, convoy, retreat, build, disband) |

### 3. Orak (`generate_cold_start_orak.py`)

| # | Game | Registry Key | Actions |
|---|------|--------------|---------|
| 7 | **Super Mario** | `super_mario` | `Jump Level : 0` … `6` |

### 4. Pokemon Red — Orak (`generate_cold_start_pokemon_red.py`)

| # | Game | Registry Key | Actions |
|---|------|--------------|---------|
| 8 | **Pokemon Red** | `pokemon_red` | High-level tools: `move_to`, `warp_with_warp_point`, `continue_dialog`, `select_move_in_battle`, etc.; raw buttons: `up`/`down`/`left`/`right`/`a`/`b` |

Uses the **Orak** Pokemon Red environment and toolset (PyBoy + `pokered` map data). Text-only (no screens). See [Pokemon Red cold-start](#pokemon-red-rollouts--orak) below.

Other envs and games (e.g. Doom, Ace Attorney, 1942, Tic-Tac-Toe, Texas Hold'em from the full LMGame-Bench set) are **not available** in our setup.

### End conditions (how episodes terminate)

All cold-start generators use the **natural end condition** of each game engine. Episodes are never cut short by an artificial step cap.

| Game | Natural end condition | Source |
|------|----------------------|--------|
| **2048** | No valid moves (game over), or reach 2048 tile; also terminates after 10 steps with no board change. | GamingAgent env |
| **Sokoban** | All boxes on targets (win); or env `max_steps_episode` (200); or 5 steps with no change. | GamingAgent env |
| **Candy Crush** | Run out of moves (`num_moves` = 50 in env config). | GamingAgent env |
| **Tetris** | Stack reaches top (game over); or 30 steps with no change. | GamingAgent env |
| **Avalon** | 3 quest failures (Evil wins) or assassination resolves after 3 quest successes. Always finite. | `AvalonGameEnvironment.done` |
| **Diplomacy** | Solo victory (`game.is_game_done`) or 20 phases elapsed (`DiplomacyConfig.max_phases = 20`). | `DiplomacyNLWrapper.done` |
| **Super Mario** | Level complete or game over; capped at 100 steps. | Orak env |
| **Pokemon Red** | Whiteout (all party HP=0); no progress (80 steps same location); Orak 12-milestone completion; or max_steps. | Orak env + cold-start script |

## Goal

1. **Prompt decision agents** powered by GPT-5.4 (or GPT-5-mini) to generate trajectories from game environments.
2. **Labeling is off by default**; use the separate `labeling/` folder for that. You can opt in with `--label` to label trajectories with GPT-5-mini (summaries, intentions, sub-task labels).

## Setup

```bash
# 1. Activate the cold-start conda environment
conda activate cold-start-agent

# 2. Set your API key (prefer OpenRouter; used for cold-start and ask_model)
# Option A: open_router_api_key in api_keys.py (recommended)
# Option B: export OPENROUTER_API_KEY="sk-or-..."
# Option C: export OPENAI_API_KEY="sk-..."

# 3. Set PYTHONPATH (from Game-AI-Agent root)
cd /path/to/Game-AI-Agent
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
```

## LMGame-Bench Rollouts (games 1-4)

### Batch rollouts (100 per game)

```bash
# All 4 games, 100 episodes each (default)
python cold_start/run_100_rollouts.py

# Specific games only
python cold_start/run_100_rollouts.py --games twenty_forty_eight sokoban tetris candy_crush

# Fewer episodes for testing
python cold_start/run_100_rollouts.py --episodes 5 --max_steps 30

# Use VLM decision agent (richer Experience fields)
python cold_start/run_100_rollouts.py --agent_type vlm

# Resume interrupted run (skips completed episodes)
python cold_start/run_100_rollouts.py --resume

# Labeling is off by default; use labeling/ for that. To enable here:
python cold_start/run_100_rollouts.py --label
```

### Per-game GPT-5.4 rollouts

```bash
# All 4 games, 50 episodes each (run until natural end per game)
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 50

# Only 2048 and Tetris, 20 episodes each
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 20 twenty_forty_eight tetris

# One game only
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 30 candy_crush

# With resume (labeling off by default)
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 100 --resume
```

Output: `cold_start/output/gpt54/<game_name>/`

### Single-game generation

```bash
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight --episodes 3 --max_steps 50 --model gpt-5-mini

# VLM decision agent
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight --agent_type vlm --episodes 3 --max_steps 50

# All games
python cold_start/generate_cold_start.py --all_games --episodes 3 --max_steps 50
```

Output: `cold_start/data/<game_name>/`

## AgentEvolver Rollouts — Avalon & Diplomacy (games 5-6)

Both games always run to their **natural end condition**. There is no `--max_steps` flag; the engines themselves decide when a game is over:

- **Avalon**: the `AvalonGameEnvironment` sets `done=True` when 3 quests fail (Evil wins) or the assassination phase resolves (after 3 quest successes). This matches `AvalonGame.run()` which loops on `while not self.env.done`.
- **Diplomacy**: the `DiplomacyNLWrapper` sets `done=True` when `game.is_game_done` (solo victory / draw) or `phases_processed >= 20` (matching `DiplomacyConfig.max_phases`). This matches `DiplomacyGame.run()` which loops on `while not self.game.is_game_done and phases_processed < self.config.max_phases`.

Per-power/per-player API calls within each phase are parallelized for speed.

```bash
# Both games, 20 episodes each (default)
bash cold_start/run_coldstart_evolver.sh

# Quick test
bash cold_start/run_coldstart_evolver.sh --episodes 5

# Avalon only
bash cold_start/run_coldstart_evolver.sh --games avalon

# Diplomacy only, 60 episodes
bash cold_start/run_coldstart_evolver.sh --episodes 60 --games diplomacy

# Resume interrupted run
bash cold_start/run_coldstart_evolver.sh --resume

# With labeling (opt-in)
bash cold_start/run_coldstart_evolver.sh --episodes 20 --label
```

Or call the Python script directly:

```bash
python cold_start/generate_cold_start_evolver.py --games avalon --episodes 10
python cold_start/generate_cold_start_evolver.py --games diplomacy --episodes 5 -v
python cold_start/generate_cold_start_evolver.py --resume
```

Output: `cold_start/output/gpt54_evolver/<game_name>/`

## Orak Rollouts — Super Mario (game 7)

Each Orak game needs its own conda environment.

```bash
# --- Super Mario ---
source evaluate_orak/setup_orak_mario.sh
python cold_start/generate_cold_start_orak.py --games super_mario --episodes 10
# or:
bash cold_start/run_coldstart_orak_mario.sh --episodes 10
```

Output: `cold_start/output/gpt54_orak/<game_name>/`

## Pokemon Red Rollouts — Orak

Pokemon Red cold-start uses the **Orak** environment and toolset (PyBoy emulator, text-only state). It requires:

1. **ROM**: A `.gb` ROM (e.g. `Pokemon - Red Version (USA, Europe).gb`). A symlink is typically used at `GamingAgent/gamingagent/configs/custom_06_pokemon_red/rom/pokemon.gb`.
2. **PyBoy**: `pip install pyboy==2.5.2`
3. **Map data**: The `pokered` disassembly must be cloned and processed so navigation tools (`move_to`, `warp_with_warp_point`, etc.) work:
   - Clone: `game_agent/Orak/src/mcp_game_servers/pokemon_red/game/pokered` from [pret/pokered](https://github.com/pret/pokered).
   - From Orak root: `python3 src/mcp_game_servers/pokemon_red/game/utils/map_preprocess.py` to generate `processed_map/` (and fix case symlinks if needed for Linux).

```bash
# From Game-AI-Agent root (labeling off by default; use labeling/ for that)
bash cold_start/run_coldstart_pokemon_red.sh --episodes 3 --max_steps 200 --verbose

# With custom ROM path
bash cold_start/run_coldstart_pokemon_red.sh --episodes 5 --max_steps 200 --rom_path /path/to/pokemon.gb

# Resume interrupted run
bash cold_start/run_coldstart_pokemon_red.sh --episodes 10 --resume

# Enable labeling (optional)
bash cold_start/run_coldstart_pokemon_red.sh --episodes 3 --label
```

Output: `cold_start/output/gpt54_pokemon_red/pokemon_red/`

Natural termination: whiteout, no-progress cap (80 steps same location), Orak 12-milestone score completion, or `--max_steps`.

## Output Structure

All generators produce the same directory layout per game:

```
cold_start/output/<suite>/<game_name>/
├── episode_000.json ... episode_NNN.json   # Individual episodes
├── episode_buffer.json                      # Episode_Buffer (loadable)
├── rollouts.jsonl                           # JSONL: one Episode per line
└── rollout_summary.json                     # Per-game stats
```

Suites: `gpt54/` (LMGame-Bench), `gpt54_evolver/` (Avalon/Diplomacy), `gpt54_orak/` (Mario), `gpt54_pokemon_red/` (Pokemon Red via Orak).

A `batch_rollout_summary.json` sits at the suite root with cross-game stats.

### Loading Rollouts into the Co-Evolution Framework

```python
from cold_start.load_rollouts import (
    load_episodes_from_jsonl,
    load_episode_buffer,
    episodes_to_rollout_records,
    load_all_game_rollouts,
)

# --- Skill pipeline ingestion ---
from skill_agents.pipeline import SkillBankAgent

episodes = load_episodes_from_jsonl("cold_start/output/gpt54/tetris/rollouts.jsonl")
agent = SkillBankAgent(bank_path="skills/bank.jsonl")
agent.ingest_episodes(episodes)
agent.run_until_stable(max_iterations=3)

# --- Trainer ingestion (Episode → RolloutRecord) ---
from trainer.skillbank.ingest_rollouts import ingest_rollouts

records = episodes_to_rollout_records(episodes)
trajectories = ingest_rollouts(records)

# --- Load all games at once ---
all_rollouts = load_all_game_rollouts("cold_start/output/gpt54")
for game_name, episodes in all_rollouts.items():
    print(f"{game_name}: {len(episodes)} episodes")
```

## Agent Types

- **`dummy`** (default for LMGame-Bench): Uses `language_agent_action` with GPT function calling. Single-turn action selection per step.
- **`vlm`** (LMGame-Bench only): Uses `run_episode_vlm_agent()` which returns `Episode` objects with fully-populated Experience fields (`summary_state`, `intentions`, `sub_tasks`, `reward_details`, `action_type`).
- **GPT-5.4 function-calling** (Evolver & Orak): Each active player/power is queried with a system prompt and structured tool call (`choose_action` / `submit_orders`). Chain-of-thought reasoning is stored in the `intentions` field.

## Design Notes

### Natural end conditions (Evolver games)

Both Avalon and Diplomacy always run to their engine's natural end condition. No
artificial `--max_steps` cap is applied in the episode loop (`while not env.done`).
This ensures cold-start data matches how the games actually play in AgentEvolver's
`AvalonGame.run()` (`while not self.env.done`) and `DiplomacyGame.run()`
(`while not self.game.is_game_done and phases_processed < self.config.max_phases`).

### Diplomacy 20-phase limit

The base `diplomacy` library (from
[AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy)) has no phase
limit -- games run until solo victory (18 supply centers) or draw, which rarely
happens with LLM agents. AI_Diplomacy itself uses `--max_year` (typically 1910,
~50 phases). AgentEvolver adds `max_phases=20` in `DiplomacyConfig` for
tractable rollouts. We use 20 for cold-start, training, and evaluation:

- Phases 1-20 (years 1901-1904) contain the richest strategic diversity:
  openings, alliance formation, first betrayals, early expansion.
- More episodes at 20 phases produces better seed skills than fewer episodes at
  50 phases for the same API budget -- breadth of experience beats depth.
- Training and evaluation both use 20 phases; matching this avoids distribution
  shift where the agent learns late-game stalemate tactics it never encounters
  during eval.

### Parallelized API calls

Per-power (Diplomacy, 7 concurrent) and per-player (Avalon, up to 5 concurrent)
LLM calls are parallelized within each phase via `ThreadPoolExecutor`. A shared
OpenAI client singleton avoids redundant TCP/TLS handshakes. This yields ~7x
speedup for Diplomacy episodes.

### StringComparator serialization fix

The diplomacy engine uses `StringComparator` objects as dict keys internally
(for locations, power names, etc.). Python's `json.dump` rejects non-`str` keys
even with `default=str` (which only handles values). A `_sanitize_keys()` helper
recursively converts all dict keys to plain `str` before serialization.

## Change Log (generate_cold_start_evolver.py)

Changes from the original codebase:

- Removed artificial `--max_steps` loop guard from both `run_avalon_episode()`
  and `run_diplomacy_episode()`; loops now use `while not env.done` only.
- Added `DIPLOMACY_MAX_PHASES = 20` constant, passed to
  `DiplomacyNLWrapper(max_phases=...)` to match `DiplomacyConfig.max_phases`.
- Parallelized per-agent API calls in both episode runners using
  `ThreadPoolExecutor` + `as_completed`.
- Replaced per-call `_make_client()` with shared singleton `_get_client()`.
- Added `_sanitize_keys()` to fix `StringComparator` JSON serialization bug in
  Diplomacy episodes.
- Removed `--max_steps` CLI argument and `default_max_steps` from
  `EVOLVER_GAMES` registry.
- Updated `run_coldstart_evolver.sh` to remove hardcoded `--max_steps 50`
  default.

## Output Format

Each episode JSON contains:
- `episode_id` — Unique UUID
- `env_name` — Platform name (`"gamingagent"`, `"avalon"`, `"diplomacy"`, etc.)
- `game_name` — Specific game (e.g. `"tetris"`, `"avalon"`, `"super_mario"`)
- `experiences` — List of Experience objects:
  - `state`, `action`, `reward`, `next_state`, `done`
  - `intentions`, `tasks`, `sub_tasks`
  - `summary`, `summary_state`
  - `reward_details` (r_env, r_follow, r_cost, r_total)
  - `action_type` (primitive, QUERY_MEM, QUERY_SKILL, CALL_SKILL)
  - `idx` (step index)
- `task` — Task description
- `outcome` — Episode outcome
- `summary` — Episode summary
- `metadata` — Rollout stats (steps, total_reward, model, agent_type, etc.)

This format is directly compatible with:
- `Episode.from_dict()` / `Episode_Buffer.load_from_json()` for data loading
- `SkillBankAgent.ingest_episodes()` for skill pipeline ingestion
- `episodes_to_rollout_records()` → `ingest_rollouts()` for trainer ingestion

## Orak End-Condition & Recording Audit (2026-03-11)

### What was verified

We traced the full end-condition flow for both Orak games against the
`orak-2025-starter-kit` and the `Orak` source repo to confirm that the
cold-start generator uses the **natural termination conditions** of each
underlying game engine -- not an artificial step cap.

| Game | Natural end condition | How it propagates |
|------|----------------------|-------------------|
| **Super Mario** | `gym_super_mario_bros` sets `done=True` on death, timer expiry, or flag reached. | `gym env.step()` → `SuperMarioEnv.step()` → `evaluate()` → `OrakNLWrapper.step()` returns `terminated=True`. |

The wrapper's `max_steps` (Mario=100, matching the starter kit's
`MAX_STEPS`) acts only as a safety-net truncation. It sets `truncated=True`
(never `terminated`) and only fires when the game hasn't already ended
naturally (line 316-317 of `orak_nl_wrapper.py`).

### What was changed in `generate_cold_start_orak.py`

Each `Experience.interface` dict now records the termination details
separately so downstream training code can distinguish natural game-over
from step-limit truncation:

```python
exp.interface = {
    "env_name": "orak",
    "game_name": game_name,
    "step": step_count,
    "terminated": terminated,   # natural game-end (death, victory, flag, etc.)
    "truncated": truncated,     # hit max_steps safety limit
    "score": step_info.get("score"),  # evaluate() result (e.g. x_pos for Mario)
    "cumulative_reward": total_reward,
}
```

Previously `Experience.done` was `terminated or truncated` with no way to
tell them apart. `done` is still set to `terminated or truncated` for
backward compatibility, but the interface dict now carries the full picture.

The verbose output also prints `TERM` or `TRUNC` labels per step for easier
debugging.

### What was already correct (no change needed)

- **State recording**: `exp.state` / `exp.raw_state` = NL observation from `obs2text()`.
- **Available actions**: `exp.available_actions` = full action name list from the env.
- **Agent action taken**: `exp.action` = exact string chosen by GPT-5.4.
- **Agent reasoning**: `exp.intentions` = chain-of-thought from the GPT tool call.
- **Shell scripts**: `run_coldstart_orak_mario.sh` (max_steps=100) matches the starter kit's
  `MAX_STEPS` dict.

### Gathering 60 rollouts

```bash
# Super Mario (from Game-AI-Agent dir, orak-mario conda env)
bash cold_start/run_coldstart_orak_mario.sh --episodes 60 --max_steps 100 --resume -v
```

`--resume` makes interrupted runs idempotent.
