# Cold-Start Data Generation

Cold-start data generation for the **COS-PLAY** co-evolution framework
(COLM 2026, Section 5).  The SFT teacher (``gpt-5.5`` by default — pulled
from ``common.models.BACKBONE_SFT_TEACHER_MODEL``) generates seed
trajectories per game, bootstrapping the co-evolution training loop
between the Decision Agent and the Skill Bank Agent.

> **Model mapping (current phase):** SFT teacher = ``gpt-5.5``; trained
> actor + skill-bank backbone = ``Qwen/Qwen3.5-9B``; frozen control plane
> = ``Qwen/Qwen3.5-35B-A3B``.  See [`../common/models.py`](../common/models.py).

## Directory Contents

| File | Purpose |
|------|---------|
| `generate_cold_start.py` | Core module: game registry, env wrapper, episode runners, labeling |
| `generate_cold_start_gpt54.py` | SFT-teacher (``gpt-5.5``) agent for LMGame-Bench (2048, Candy Crush, Tetris).  Filename retained for path compatibility. |
| `generate_cold_start_evolver.py` | SFT-teacher (``gpt-5.5``) agent for Avalon & Diplomacy |
| `generate_cold_start_orak.py` | SFT-teacher (``gpt-5.5``) agent for Super Mario (Orak env) |
| `generate_cold_start_actor.py` | **Actor-agent** rollouts via ``env_wrappers``: gpt-5.5 vision converts each rendered frame → ``<state>`` schema (per `vlm_wrapper.schema`), then gpt-5.5 picks one action from the schema with function calling.  Covers 2048 / Candy Crush / Tetris (macro) / Super Mario. |
| `load_rollouts.py` | Utility: load rollout outputs into Episode / RolloutRecord |
| `run_coldstart_gpt54.sh` | Shell launcher for LMGame-Bench rollouts |
| `run_coldstart_evolver.sh` | Shell launcher for Avalon & Diplomacy rollouts |
| `run_coldstart_orak_mario.sh` | Shell launcher for Super Mario (conda + Xvfb) |
| `run_coldstart_actor.sh` | Shell launcher for the actor-agent rollouts (env_wrappers + gpt-5.5 visual schema → action). |
| `run_coldstart_actor_all_games.sh` | **Multi-env wrapper** — runs `run_coldstart_actor.sh` twice in one command: 2048 / Candy Crush / Tetris under `game-ai-agent`, then Super Mario under `orak-mario` (with Xvfb).  Use this when you want all 4 games in one shot. |
| `generate_cold_start_actor_gymv.py` | Actor-agent rollouts on the **`gymv` source domain** (13 retro/Temporal envs: Airstriker, Columns, …). |
| `run_coldstart_actor_gymv.sh` / `run_coldstart_actor_gymv_all.sh` | Single-env / parallel-all-envs launchers for `gymv`. |
| `generate_cold_start_actor_browsergym.py` | Actor-agent rollouts on **BrowserGym** (MiniWoB / WebArena / AssistantBench). Auto-sources `webarena_env.sh` per task. (VisualWebArena was dropped 2026-05-03 — see `legacy/visualwebarena/README.md`.) |
| `run_coldstart_actor_browsergym.sh` | BrowserGym launcher; supports `--tasks`, `--tasks_file`, `--urls`. |
| `generate_cold_start_actor_osworld.py` | Actor-agent rollouts on **OSWorld** desktop tasks (KVM guest, AT-SPI accessibility tree). |
| `run_coldstart_actor_osworld.sh` / `run_coldstart_actor_osworld_all.sh` | Single-guest / parallel-multi-guest launchers for OSWorld. |
| `generate_cold_start_actor_visual_reasoning.py` | Actor-agent rollouts on **visual reasoning** benchmarks (VisualToolBench, TIR-Bench, Video-Holmes, SIV-Bench). Supports `--sample_ids_dir` for held-out splits. |
| `run_coldstart_actor_visual_reasoning.sh` | Visual-reasoning launcher (image + video MCQ). |
| `task_samples/` | Stratified pool / held-out manifests + sampler scripts (`build_browsergym_diverse_200.py`, `build_visual_reasoning_diverse_1000.py`). |
| `webarena_env.sh` | Auto-generated `WA_*` exports from `install/install_webarena_sites.sh`; auto-sourced by the BrowserGym launcher. |

## Games Covered (6 total)

### LMGame-Bench (`generate_cold_start_gpt54.py`)

| Game | Registry Key | Actions |
|------|--------------|---------|
| **2048** | `twenty_forty_eight` | `up`, `down`, `left`, `right` |
| **Candy Crush** | `candy_crush` | coordinate swaps, e.g. `((0,5),(1,5))` |
| **Tetris** | `tetris` | `move_left`, `move_right`, `rotate_cw`, `rotate_ccw`, `hard_drop`, `soft_drop` |

### AgentEvolver (`generate_cold_start_evolver.py`)

| Game | Registry Key | Actions |
|------|--------------|---------|
| **Avalon** | `avalon` | team proposals, votes, pass/fail, assassination |
| **Diplomacy** | `diplomacy` | unit orders (move, hold, support, convoy, retreat, build, disband) |

### Orak (`generate_cold_start_orak.py`)

| Game | Registry Key | Actions |
|------|--------------|---------|
| **Super Mario** | `super_mario` | `Jump Level : 0` ... `6` |

## Setup

```bash
conda activate game-ai-agent
export OPENROUTER_API_KEY="sk-or-..."   # or OPENAI_API_KEY
cd /path/to/Game-AI-Agent
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
```

## Quick Start

### LMGame-Bench (2048, Candy Crush, Tetris)

```bash
# All 3 games, 60 episodes each
bash cold_start/run_coldstart_gpt54.sh --episodes 60

# Specific games
bash cold_start/run_coldstart_gpt54.sh --games tetris candy_crush --episodes 60

# Resume interrupted run
bash cold_start/run_coldstart_gpt54.sh --episodes 60 --resume

# Python directly
python cold_start/generate_cold_start_gpt54.py --games tetris --episodes 5 --resume
```

Output: `cold_start/output/gpt54/<game_name>/`

### Avalon & Diplomacy

```bash
# Both games, 20 episodes each (default)
bash cold_start/run_coldstart_evolver.sh

# Avalon only, 60 episodes
bash cold_start/run_coldstart_evolver.sh --games avalon --episodes 60

# Resume + verbose
bash cold_start/run_coldstart_evolver.sh --resume -v

# Python directly
python cold_start/generate_cold_start_evolver.py --games diplomacy --episodes 10 -v
```

Output: `cold_start/output/gpt54_evolver/<game_name>/`

### Super Mario

Requires the `orak-mario` conda environment and Xvfb for headless rendering.

```bash
bash cold_start/run_coldstart_orak_mario.sh --episodes 10

# Or manually
source env_wrappers/setup_orak_mario.sh
python cold_start/generate_cold_start_orak.py --games super_mario --episodes 10
```

Output: `cold_start/output/gpt54_orak/<game_name>/`

### Actor agent (env_wrappers, gpt-5.5 visual schema → action)

The actor pipeline can run any subset of the four supported games
(2048 / Candy Crush / Tetris / Super Mario).  Choose **one** of the two
launchers depending on which games you need.

#### Single-env launcher — `run_coldstart_actor.sh`

Use this when you want to drive the cold-start from **one already-active
conda env**.  Best for 2048 + Candy Crush + Tetris (run from
`game-ai-agent`), or Super Mario alone (run from `orak-mario`).

```bash
conda activate game-ai-agent  # or orak-mario for super_mario only

# All games supported by the active env (defaults: --resume --save_frames -v)
bash cold_start/run_coldstart_actor.sh

# Just 2048 + tetris, 10 episodes each, max 50 steps, save VLM frames
bash cold_start/run_coldstart_actor.sh \
    --games twenty_forty_eight tetris --episodes 10 --max_steps 50 --save_frames

# Skip the vision call (cheaper canonical-schema baseline; deterministic)
bash cold_start/run_coldstart_actor.sh --no_vision --episodes 3

# Python directly (no Xvfb / conda env activation)
python cold_start/generate_cold_start_actor.py \
    --games twenty_forty_eight candy_crush --episodes 5 -v
```

When `super_mario` is in the requested games AND the `orak-mario` conda
env exists, the launcher auto-activates it and starts Xvfb on `:99` so
nes-py / pyglet can render headlessly.

#### Multi-env wrapper — `run_coldstart_actor_all_games.sh`

Use this when you want **all 4 games in a single command**.  The
wrapper dispatches each game via `conda run -n <env>` so the right
conda env is used per game (no manual `conda activate` swaps):

| Game | Env |
|---|---|
| `twenty_forty_eight`, `candy_crush`, `tetris` | `game-ai-agent` |
| `super_mario` | `orak-mario` |

All games share the same `Cold-start-out/` directory at the repo root
(`<codebase_root>/Cold-start-out/`), all extra CLI args are forwarded
verbatim, and vision is on by default.

```bash
# Sequential (default): one game at a time, ~20-30 min for --episodes 1
bash cold_start/run_coldstart_actor_all_games.sh \
    --episodes 1 --model gpt-5.5 --save_frames -v

# PARALLEL: all 4 games dispatched concurrently as separate processes.
# Bottleneck is the longest game; ~3-4x speedup for --episodes 1.
# Per-game stdout streams to Cold-start-out/_logs/<game>.log so output
# isn't interleaved.  Tail any game live with:
#   tail -f Cold-start-out/_logs/tetris.log
bash cold_start/run_coldstart_actor_all_games.sh --parallel \
    --episodes 1 --model gpt-5.5 --save_frames -v

# Skip super_mario (e.g. orak-mario env not installed yet)
bash cold_start/run_coldstart_actor_all_games.sh --no_mario \
    --episodes 2 --save_frames -v

# Cheap baseline (skip vision call; canonical schema only)
bash cold_start/run_coldstart_actor_all_games.sh --parallel \
    --episodes 3 --no_vision
```

Wrapper-only flags (consumed by the wrapper, NOT forwarded):

| Flag | Effect |
|---|---|
| `--parallel`, `-P` | Dispatch all games concurrently as background processes |
| `--no_mario` | Skip `super_mario` (and orak-mario env activation) |
| `--games <g>...` | Restrict to a subset of games (default: all 4) |
| `--run_id <id>` | Override auto-generated `YYYY-MM-DD_HH-MM-SS` run id (useful for `--resume` into an existing run) |
| `--output_dir <path>` | Override base dir (default: `<repo>/Cold-start-out`) |

Pre-reqs the wrapper expects:

```bash
# game-ai-agent — already required for everything else
bash install/install_main_env.sh
conda activate game-ai-agent
pip install tile_match_gym                # adds Candy Crush backend
                                          # (downgrades numpy to 1.26.x;
                                          #  fine for actor cold-start,
                                          #  conflicts with vLLM stack)

# orak-mario — required for Super Mario
git clone https://github.com/krafton-ai/Orak.git /path/to/Orak
bash install/install_orak_mario.sh
```

#### Outputs

The `run_coldstart_actor_all_games.sh` wrapper writes timestamped runs
grouped by conda env:

```
<codebase_root>/Cold-start-out/
├── <run_id>/                              # YYYY-MM-DD_HH-MM-SS
│   ├── game-ai-agent/                     # env subdir
│   │   ├── twenty_forty_eight/
│   │   │   ├── episode_NNN.json
│   │   │   ├── episode_buffer.json
│   │   │   ├── rollouts.jsonl
│   │   │   ├── rollout_summary.json
│   │   │   └── frames/ep_NNN/step_NNN.png   # only with --save_frames
│   │   ├── candy_crush/...
│   │   └── tetris/...
│   ├── orak-mario/
│   │   └── super_mario/...
│   ├── _logs/<game>.log                  # per-game stdout/stderr
│   └── _run_meta.json                    # run config + per-game rc
└── latest -> <run_id>                     # symlink to most recent run
```

The single-env launcher `run_coldstart_actor.sh` writes a flat layout
when invoked directly (no `<run_id>/<env>` wrapping):

```
<codebase_root>/Cold-start-out/
└── <game>/
    ├── episode_NNN.json
    ├── episode_buffer.json
    ├── rollouts.jsonl
    ├── rollout_summary.json
    └── frames/ep_NNN/step_NNN.png
```

`Cold-start-out/` lives at the top of the `Multi-hop-Reasoning-VLM-Agent`
repo and is git-ignored.  Override with `--output_dir <path>` if you
want to write somewhere else.

Each step persists the visual schema, the raw VLM output, the action +
reasoning, the canonical (deterministic) schema fallback, and (with
`--save_frames`) the actual PNG frame sent to the VLM.

Tetris is **always** wrapped with `TetrisMacroActionWrapper` so each
LLM call commits one full piece placement (rotation + column +
landing) instead of a primitive key press; this keeps every step
informative for skill mining / GRPO.

API keys are auto-loaded from `<workspace>/api_keys.py` (or
`$COSPLAY_API_KEYS_FILE`) at import time — no `export` needed. See
`generate_cold_start_actor.py::_bootstrap_api_keys_from_file` for the
lookup order.

## Multi-domain Cold-Start (Lean Plan)

The cold-start pipeline also seeds trajectories across the four
**transfer-target domains** of the project (see [`../readme.md`](../readme.md)
for the full architecture). Per the project README, games (`gymv`)
are the *source* domain and the four others are *transfer probes*
consumed at gate Stage 3a (`harness/few_shot_adapter.py`,
`k_shot_default=5`, `k_shot_max=16` per skill per domain). Volume is
sized for a **diverse pool**, not a full benchmark sweep.

### Recommended pool sizes

Stratified manifests live under `task_samples/` (built by the
`build_*` scripts). Sizes are tuned for: (1) supplying K∈[5,16]
diverse few-shot demos per skill at gate Stage 3a, (2) covering the
SFT cold-start corpus for the actor on each target schema, (3)
leaving a disjoint held-out slice for the E0/E1/E2 scoreboard.

| Domain | Bucket | Pool (used in this run) | Held-out (reserved for eval) | Sampler |
|---|---|---:|---:|---|
| `gymv` (**source**) | 13 retro envs × ~10 ep × 20 steps | ~130 episodes | n/a | `run_coldstart_actor_gymv_all.sh --episodes 10` |
| `browser` | **AssistantBench** (open web, no infra; multimodal-friendly) | 180 | + 30 | `task_samples/build_browsergym_diverse_200.py` |
| `browser` | MiniWoB++ (atomic primitives) | 125 | + 25 | same |
| `browser` | WebArena | *deferred* — overlaps AssistantBench coverage at much higher infra cost | — | — |
| `browser` | ~~VisualWebArena~~ | *dropped 2026-05-03* — see `legacy/visualwebarena/README.md` | — | — |
| `osworld` | OSWorld desktop tasks | 120 stratified | + 30 | `evaluation_dataset/build_pool_and_holdout.py` |
| `visual_reasoning` | VisualToolBench (image) | 300 stratified | + 100 | `task_samples/build_visual_reasoning_diverse_1000.py` |
| `visual_reasoning` | TIR-Bench (image, tool-use) | 300 stratified | + 100 | same |
| `video` (**headline**) | Video-Holmes | 1,000 | + 200 | same |
| `video` | SIV-Bench | 400 stratified | + 100 | same |

Total pool: ~3,200 step-equivalents from env trajectories + ~2,000
visual-reasoning samples, all stratified to keep diversity per
slot-binding axis (site / question-type / dimension / …) high
enough that any candidate skill probed at Stage 3a finds K relevant
demos. The reasoning is documented in `task_samples/build_*` scripts.

### Pool vs. held-out separation

Few-shot demos at gate Stage 3a are read-only probes
(`FewShotAdapter` is stateless), but the pool they're drawn from
**must be disjoint** from the eval slice or the E0 scoreboard is
contaminated. The `task_samples/build_*` scripts emit both files in
one pass — generate cold-start data on `<benchmark>_pool.txt`,
freeze `<benchmark>_holdout.txt` for end-of-phase eval.

### `reasoning_effort` per pipeline

The actor pipeline auto-detects `gpt-5.x` / `o1` / `o3` and routes
to `max_completion_tokens` with a 12 k cap (see
[Token budgets and reasoning models](#token-budgets-and-reasoning-models)),
but **does not currently set `reasoning_effort`**, so OpenAI's
default `medium` applies. For cold-start data generation this is
wasted budget: the SFT student (`Qwen3.5-9B`) only learns from the
visible `<state>` block and action JSON — hidden thinking tokens are
never transferred to the trained policy.

| Pipeline | Recommended `reasoning_effort` | Why |
|---|---|---|
| `gymv` (source-domain trajectories) | `minimal` | structured extraction; student can't use thinking |
| BrowserGym (all suites) | `minimal` | schema + constrained action; structured |
| OSWorld | `minimal` | same — schema is the planning surface, not hidden thought |
| Visual reasoning (image + video MCQ) | `medium` | teacher answer correctness is the bottleneck on multi-hop QA |

Switching to `minimal` on the structured pipelines drops per-step
cost from ~$0.045 to ~$0.013 (≈ 3.5× cheaper) and per-call latency
from ~15 s to ~5 s. Empirical visible output from
`Cold-start-out-browsergym/` confirms this: schema responses median
~2.9 KB / ~740 tokens, action responses median ~370 chars / ~95
tokens — the rest of today's bill is hidden thinking.

#### Smoke-test calibration (`gpt-5.4`, n = 5 paired MiniWoB tasks)

To replace extrapolation with a measurement we ran a paired smoke
test on five mid-difficulty MiniWoB tasks (`use-spinner`, `guess-number`,
`simple-arithmetic`, `click-checkboxes`, `simple-algebra`) at
`reasoning_effort=minimal` and `medium`, same seed, same task list,
8-step cap, `gpt-5.4` (saved under `Cold-start-out-smoke-effort/`):

| Metric | `minimal` | `medium` | Delta |
|---|---|---|---|
| Task success | **5/5 (100 %)** | 4/5 (80 %) | `medium` failed `guess-number` |
| Schema parse rate | 100 % | 100 % | tied |
| Action LLM ok | 100 % | 100 % | tied |
| Avg steps / episode | 4.6 | 4.4 | tied |
| Wall per step | 14.8 s | 31.5 s | **`medium` 2.1× slower** |
| Total wall (5 tasks) | 339 s | 693 s | `medium` 2.0× slower |

Concrete failure mode: on `guess-number` (binary search 0–9), `minimal`
correctly picked the midpoint first (`5 → 7 → 8 → 9`, classic binary
search) and won at step 8. `medium` started at 0 and incremented
linearly (`0 → 1 → 5 → 6`), exhausting the 8-step budget on the same
feedback signal. With more hidden thinking the model converged on a
*worse* policy on this task — concrete evidence that more reasoning ≠
better behavior on structured-action tasks.

Caveat: n = 5 is small. The wall-clock 2× delta is robust; the +20 pp
success-rate delta is suggestive but not statistically significant on
its own. The structural argument (`Qwen3.5-9B` cannot consume hidden
thinking, regardless) does not depend on the success-rate result.

Reproduce:

```bash
bash cold_start/run_coldstart_actor_browsergym.sh \
    --tasks browsergym/miniwob.use-spinner browsergym/miniwob.guess-number \
            browsergym/miniwob.simple-arithmetic browsergym/miniwob.click-checkboxes \
            browsergym/miniwob.simple-algebra \
    --episodes 1 --max_steps 8 --model gpt-5.4 \
    --reasoning_effort minimal \
    --output_dir Cold-start-out-smoke-effort/minimal --resume

# Then again with --reasoning_effort medium → Cold-start-out-smoke-effort/medium/
```

### Cost & wall-clock for one full pass

GPT-5 reasoning class pricing ($1.25 / M input, $10 / M output incl.
thinking):

| Setting | Total cost | Wall-clock @ realistic per-bucket parallelism |
|---|---:|---:|
| Original full plan (706 BG + full OSWorld + 1 k each visual), `medium` everywhere | ~$1,500 – $1,800 | ~12 – 15 h |
| **Lean plan, `minimal` for env / `medium` for visual reasoning** ← recommended | **~$260 – $280** | **~3 – 6 h** (set by OSWorld KVM concurrency) |
| Lean plan, `gpt-5.4-mini` everywhere except Video-Holmes | ~$70 – $100 | ~3 – 6 h |

Where the lean-plan budget goes (≈ $260 baseline):

| Bucket | Share | Cost |
|---|---:|---:|
| OSWorld | 45 % | ~$110 |
| gymv | 14 % | ~$36 |
| BrowserGym (MiniWoB + AssistantBench) | 24 % | ~$62 |
| Visual reasoning (4 benchmarks @ `medium`) | 17 % | ~$45 |

The two dominant levers:

1. **`reasoning_effort`** — `medium` → `minimal` on env pipelines saves ~70 % of the bill.
2. **OSWorld KVM concurrency** — 2 → 4 → 8 guests collapses wall-clock from ~6 h to ~1.6 h.

### Quick start (lean plan)

```bash
# 1. Build stratified manifests (idempotent; emits *_pool.txt + *_holdout.txt)
python cold_start/task_samples/build_browsergym_diverse_200.py
python cold_start/task_samples/build_visual_reasoning_diverse_1000.py

# 2. gymv (source) — 13 envs in parallel
bash cold_start/run_coldstart_actor_gymv_all.sh \
    --episodes 10 --max_steps 20 --save_frames -v

# 3. BrowserGym (MiniWoB + AssistantBench from the manifest)
bash cold_start/run_coldstart_actor_browsergym.sh \
    --tasks_file cold_start/task_samples/browsergym_all_diverse.txt \
    --episodes 1 --max_steps 12 --save_frames -v

# 4. OSWorld (120 stratified — bump KVM guests to taste)
bash cold_start/run_coldstart_actor_osworld_all.sh \
    --task_catalog cold_start/evaluation_dataset/pool/osworld_catalog.json \
    --max_steps 30 --num_guests 4 -v

# 5. Visual reasoning (use --sample_ids_dir to point at the pool manifests)
bash cold_start/run_coldstart_actor_visual_reasoning.sh \
    --benchmarks visual_toolbench tir_bench video_holmes siv_bench \
    --sample_ids_dir cold_start/task_samples/ \
    --num_test_cases 1000 --num_frames 6 -v
```

The `task_samples/<benchmark>_holdout.txt` files are *not* consumed
by these commands — they stay frozen for the eval driver
(`evaluation/`, Phase E) so the SFT-trained policy can be scored on
unseen tasks.

## Output Structure

All generators produce the same per-game layout:

```
<output_root>/<game_name>/
├── episode_000.json ... episode_NNN.json   # Individual episodes
├── episode_buffer.json                      # Episode_Buffer (loadable)
├── rollouts.jsonl                           # JSONL: one Episode per line
└── rollout_summary.json                     # Per-game stats
```

Default `<output_root>` per launcher:

| Launcher | `<output_root>` | Notes |
|---|---|---|
| `run_coldstart_gpt54.sh` | `cold_start/output/gpt54/` | LMGame-Bench (2048 / Candy Crush / Tetris) |
| `run_coldstart_evolver.sh` | `cold_start/output/gpt54_evolver/` | Avalon / Diplomacy |
| `run_coldstart_orak_mario.sh` | `cold_start/output/gpt54_orak/super_mario/` | Super Mario |
| `run_coldstart_actor.sh` & `run_coldstart_actor_all_games.sh` | **`Cold-start-out/`** at the repo root (git-ignored) | env_wrappers actor agent — visual-schema-driven action selection across all four games |

Override the actor pipeline's output dir with `--output_dir <path>`.

### Per-step extras (actor pipeline)

Each `Experience.extras` dict captured by `generate_cold_start_actor.py`
carries the schema metadata so trajectories can be replayed and parse
failures can be debugged offline:

| key | meaning |
|---|---|
| `schema` | The `<state>...</state>` block fed to the action LLM (parsed VLM output, or canonical fallback). |
| `schema_source` | `vlm` &nbsp;— parsed from the visual call.<br>`fallback_canonical` &nbsp;— parser failed, used the deterministic canonical schema.<br>`canonical` &nbsp;— vision was disabled (`--no_vision`).<br>`text_only` &nbsp;— no canonical helper available either. |
| `schema_recovery` | When the strict parser fails, how the schema was salvaged: `strict` / `fenced` / `truncated` / `untagged`.  Absent on `fallback_canonical`. |
| `schema_finish_reason` | OpenAI/OpenRouter `finish_reason` for the visual call (`stop`, `length`, …).  `length` ⇒ the response was truncated by the token cap; raise `_SCHEMA_MAX_TOKENS_REASONING` if you see this often. |
| `schema_raw_excerpt` | First 4 KB of the raw VLM response (only present when the call succeeded).  Use this to diagnose `fallback_canonical`. |
| `schema_raw_full_len` | Full length of the raw response (so you can detect truncation even after the excerpt is capped). |
| `schema_canonical` | The deterministic canonical schema for this step (the fallback that *would* be used). |
| `schema_error` | Exception repr if the visual API call itself raised. |
| `action_raw`, `action_error` | Raw output / exception from the action call. |
| `frame_path` | Path to the saved PNG frame fed to the VLM (when `--save_frames`). |
| `valid_actions`, `is_noop` | Action list at this step + whether the action was a no-op. |

### Token budgets and reasoning models

`generate_cold_start_actor.py` recognises reasoning models (`gpt-5.x`,
`o1`, `o3`, `o4`) via `_is_reasoning_model` and routes them straight to
`max_completion_tokens` with a generous cap (`_SCHEMA_MAX_TOKENS_REASONING
= 12000`).  This is necessary because:

- Reasoning models charge **internal thinking tokens** against the same
  budget as the visible response.
- Dense games (candy_crush 8×8, tetris 20×10) emit long schemas: 30+
  entities, attributes, affordances, relations, targets, actions.
- With the previous 1500-token cap the response was truncated **before**
  the closing `</state>` tag, so the strict regex parser always failed
  and every step fell back to canonical.  After the bump, parse rates
  for these games match 2048 (≥80 % `schema=vlm`).

The lenient parser (`_lenient_parse_schema`) is a defence-in-depth
backstop: it strips markdown code fences, closes a missing `</state>`
tag (`recovery=truncated`), and wraps loose `<entities>` blocks
(`recovery=untagged`).

**Note on `reasoning_effort`.** The 12 k cap accommodates *thinking +
visible schema*; it does **not** mean every call has to *use* that
much thinking. The pipeline currently passes no `reasoning_effort`
flag, so OpenAI's default `medium` applies — silently billing
~2–4 k thinking tokens per call. For cold-start data generation
those thinking tokens are wasted (the SFT student never sees them);
prefer `reasoning_effort="minimal"` on `gymv` / browser / OSWorld
pipelines and reserve `medium` for the visual-reasoning benchmarks
where teacher answer correctness matters. See
[Multi-domain Cold-Start (Lean Plan) → `reasoning_effort` per pipeline](#reasoning_effort-per-pipeline)
for the full table and cost impact.

## Loading Rollouts into the Training Pipeline

```python
from cold_start.load_rollouts import (
    load_episodes_from_jsonl,
    load_episode_buffer,
    episodes_to_rollout_records,
    load_all_game_rollouts,
)

# Load episodes from a single game
episodes = load_episodes_from_jsonl("cold_start/output/gpt54/tetris/rollouts.jsonl")

# Convert to RolloutRecord for the trainer
records = episodes_to_rollout_records(episodes)

# Load all games at once
all_rollouts = load_all_game_rollouts("cold_start/output/gpt54")
for game_name, eps in all_rollouts.items():
    print(f"{game_name}: {len(eps)} episodes")
```

## End Conditions

Episodes terminate at each game's **natural end condition**.  Step caps
mirror upstream COS-PLAY (`generate_cold_start.py::COLD_START_MAX_STEPS_NATURAL_END`
plus the `--max_steps 100` in `run_coldstart_orak_mario.sh`) and are
also used as the per-game defaults in `generate_cold_start_actor.py::DEFAULT_MAX_STEPS`:

| Game | Natural end condition | Default step cap |
|------|---------------------|------------------|
| **2048** | No valid moves, or reach 2048 tile; also 10 steps with no board change | 200 |
| **Candy Crush** | Run out of moves (50 in env config) | 50 |
| **Tetris** | Stack reaches top; or 30 steps with no change | 200 (primitive) / ≈60 macro placements is equivalent in the actor pipeline |
| **Avalon** | 3 quest failures or assassination resolves after 3 quest successes | (per-env) |
| **Diplomacy** | Solo victory or 20 phases elapsed | (per-env) |
| **Super Mario** | Level complete or game over | 100 |

## Design Notes

- **Diplomacy 20-phase limit**: Matches AgentEvolver's `DiplomacyConfig.max_phases`.
  Phases 1-20 contain the richest strategic diversity; more episodes at 20
  phases produces better seed skills than fewer at 50 phases.
- **Parallelized API calls**: Per-power (Diplomacy, 7 concurrent) and
  per-player (Avalon, up to 5 concurrent) calls are parallelized via
  `ThreadPoolExecutor` for ~7x speedup.
- **Labeling**: Off by default. Use the separate `labeling/` folder, or
  opt in with `--label` on the shell scripts.
