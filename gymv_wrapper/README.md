# `gymv_wrapper`

Gym-V–specific visual grounding, heuristics, VLM adapter, and grounding tools.

This package centralises everything that depends on
[ModalMinds/gym-v](https://github.com/ModalMinds/gym-v) (and, for the
`Temporal/*` games, [stable-retro](https://github.com/Farama-Foundation/stable-retro))
so the cross-domain code in `vlm_wrapper/` can stay environment-agnostic.

> **TL;DR** — wrap any of the 13 `Temporal/*` Genesis envs to get a
> JSON visual-grounding schema attached to every `Observation`, and/or call
> a vision LLM (default **GPT-5.5**) to produce the canonical
> `<state>…</state>` schema from the screenshot.

---

## What's inside

| File | What it does |
|------|--------------|
| `__init__.py`                   | Re-exports the public API (`generate_label`, `text_to_schema`, `build_gymv_registry`, all `Temporal*VisualGroundingWrapper` classes, `TEMPORAL_GAME_SPECS`, …). |
| `adapter.py`                    | Vision head: `generate_label(image, …)` calls a VLM and returns a `<state>` schema. Routing reuses `API_func.make_openai_client` / `effective_openai_model`. Also defines `GymVSchemaWrapper` (online VLM wrapper). |
| `heuristic.py`                  | Text head: `text_to_schema(obs_text, description, …)` — fast, no-API schema generator. Includes `parse_retro_pipe_text` for the pipe-separated `obs.text` emitted by `RetroGymVEnv`. |
| `tools.py`                      | Tool-calling registry for multi-turn grounding (entity queries, deadlock checks, spatial analysis, merge counting). Build with `build_gymv_registry(obs_text, description, step)`. |
| `temporal_visual_grounding.py`  | Per-game `GameVisualSpec` table + `build_temporal_visual_schema` (image-size + RAM-watch fuse) + `TemporalVisualGroundingWrapper` and 13 per-game subclasses. |
| `tests/test_temporal_visual_schema.py` | Unit tests for the schema fuser. |

`vlm_wrapper.gymv_adapter` / `vlm_wrapper.gymv_heuristic` / `vlm_wrapper.tools_gymv`
are now thin compatibility shims that re-export from this package, so
existing imports keep working.

---

## Game coverage

13 stable-retro Sega Genesis games are registered in
[`gym_v/envs/__init__.py`](https://github.com/ModalMinds/gym-v) and mirrored
in `TEMPORAL_GAME_SPECS`. **Default benchmark scope is the 8 games marked
"benchmark"** below; the other 5 (marked "registered") are still callable
via `gym_v.make(...)` but are excluded from the canonical leaderboards
(`run_coldstart_actor_gymv_all.sh`, `run_openrouter_baselines.sh`,
`run_qwen_vllm_baselines.sh`). See `baselines/README.md` § "Gym-V
benchmark scope" for the rationale and per-game data.

All envs are multimodal (image + text) in `RetroGymVEnv`:

| # | `gym_v.make(…)` | `stable-retro` integration | Status | Genre | Grounding focus |
|---|---|---|---|---|---|
| 1  | `Temporal/Airstriker-v0`            | `Airstriker-Genesis-v0`            | benchmark | shmup       | ships, bullets, scrolling playfield |
| 2  | `Temporal/AlteredBeast-v0`          | `AlteredBeast-Genesis-v0`          | benchmark | beatemup    | player avatar, beasts, power orbs |
| 3  | `Temporal/CastleOfIllusion-v0`      | `CastleOfIllusion-Genesis-v0`      | registered | platformer  | mickey, platforms, collectibles |
| 4  | `Temporal/CastlevaniaBloodlines-v0` | `CastlevaniaBloodlines-Genesis-v0` | registered | action      | whip combat, enemies, verticality |
| 5  | `Temporal/Columns-v0`               | `Columns-Genesis-v0`               | benchmark | puzzle      | jewel stacks, falling column, match clears |
| 6  | `Temporal/DynamiteHeaddy-v0`        | `DynamiteHeaddy-Genesis-v0`        | benchmark | platformer  | detachable head, bosses, platforms |
| 7  | `Temporal/GoldenAxe-v0`             | `GoldenAxe-Genesis-v0`             | registered | beatemup    | melee spacing, mounts, magic |
| 8  | `Temporal/KidChameleon-v0`          | `KidChameleon-Genesis-v0`          | registered | platformer  | helmets, transformations, hazards |
| 9  | `Temporal/MortalKombatII-v0`        | `MortalKombatII-Genesis-v0`        | registered | fighting    | two fighters, health, special moves |
| 10 | `Temporal/SpaceHarrierII-v0`        | `SpaceHarrierII-Genesis-v0`        | benchmark | shmup       | pseudo-3d rail, obstacles, projectiles |
| 11 | `Temporal/StreetsOfRage2-v0`        | `StreetsOfRage2-Genesis-v0`        | benchmark | beatemup    | combo chains, throws, crowd control |
| 12 | `Temporal/Strider-v0`               | `Strider-Genesis-v0`               | benchmark | action      | slash mobility, large sprites, bosses |
| 13 | `Temporal/ThunderForceIII-v0`       | `ThunderForceIII-Genesis-v0`       | benchmark | shmup       | weapon pickups, speed, bullet patterns |

Every env emits both modalities on every step (`Observation.image: PIL.Image`,
`Observation.text: str`). The text channel always carries
`Game | Frame | StepReward | EpReward | LastAction | Recent`; richer ground-truth
fields (e.g. `Score`, `Lives`, `Health`, `Level`, `Gameover`, plus other RAM
watches) come from each ROM's `data.json`.

A pre-flight modality probe lives at
[`visual_grounding_tests/check_io_modalities.py`](../visual_grounding_tests/check_io_modalities.py):

```bash
# static check (no stable_retro / gym_v / ROMs needed)
python visual_grounding_tests/check_io_modalities.py

# also instantiate each env and call reset()
python visual_grounding_tests/check_io_modalities.py --runtime
```

---

## Setup / dependencies

The package is **import-safe without** `gym_v` / `stable_retro`: only the
runtime classes that wrap an `Env` (`GymVSchemaWrapper`,
`Temporal*VisualGroundingWrapper`) actually require them. Everything else
(schema building, heuristics, tools) imports cleanly with just `PIL` and
`openai`.

For full functionality install:

```bash
# 1. base agent deps
pip install -e .

# 2. gym_v + stable-retro + Mega Drive ROMs (idempotent)
bash install/install_gymv.sh
bash install/gymv_temporal_patch/apply_patch.sh \
     /fs/gamma-projects/vlm-robot/gym-v \
     /fs/gamma-projects/vlm-robot/ROMs/Mega_Drive_Mini_Full_Set.zip
```

API keys (only needed for the vision head):

| Env var               | Used by | Notes |
|-----------------------|---------|-------|
| `OPENROUTER_API_KEY`  | `API_func.make_openai_client` (preferred when set) | Single key for cold-start, grounding, and gymv labels. |
| `OPENAI_API_KEY`      | `API_func.make_openai_client` (fallback)           | Direct OpenAI route. |
| `VLM_LABEL_MODEL`     | `gymv_wrapper.adapter`                              | Default model id for `generate_label`. Defaults to `gpt-5.5`. |
| `VLM_LABEL_MAX_TOKENS`| `gymv_wrapper.adapter`                              | Default `1200`. |
| `VLM_LABEL_TEMPERATURE`| `gymv_wrapper.adapter`                             | Default `0.2`. |

---

## Public API quick reference

```python
from gymv_wrapper import (
    # Vision head — image → <state> schema via VLM
    generate_label,
    GymVSchemaWrapper,

    # Heuristic head — text → <state> schema (no API)
    text_to_schema,

    # Per-game visual grounding for stable-retro envs
    TEMPORAL_GAME_SPECS,
    TEMPORAL_WRAPPER_BY_ENV_ID,
    TemporalVisualGroundingWrapper,
    build_temporal_visual_schema,

    # Multi-turn grounding tools
    build_gymv_registry,
)
```

### 1) Heuristic head (no API call, deterministic)

```python
from gymv_wrapper import text_to_schema

obs_text = "Game: Game2048-v0 | Score: 124 | Frame: 5 | StepReward: 0.0 | LastAction: UP"
schema = text_to_schema(
    obs_text=obs_text,
    description="Slide tiles to reach 2048.",
    task_id="Game2048-v0",
    step=5,
    max_entities=20,
)
print(schema)  # <state>…</state>
```

### 2) Vision head (image → schema via GPT-5.5 by default)

```python
from PIL import Image
from gymv_wrapper import generate_label

result = generate_label(
    Image.open("frame.png"),
    goal="Survive and rack up score in Airstriker.",
    task_id="Temporal/Airstriker-v0",
    step=12,
    obs_text="Game: Airstriker-Genesis-v0 | Score: 320 | Lives: 3 | …",
    valid_actions=["NOOP", "LEFT", "RIGHT", "B"],
    # Optional overrides:
    # model="gpt-5.5",                  # default
    # api_key=…, base_url=…,            # else env vars rule
    # temperature=0.2, max_tokens=1200,
)
print(result["schema"])        # <state>…</state>
print(result["model_routed"])  # e.g. "openai/gpt-5.5" via OpenRouter
print(result["validation"])    # validate_schema output
```

Routing rules (delegated to `API_func.py`):

* `OPENROUTER_API_KEY` set → `https://openrouter.ai/api/v1` with `openai/<model>` prefix.
* Otherwise `OPENAI_API_KEY` set → direct OpenAI.
* Explicit `api_key=` / `base_url=` arguments win.

### 3) Online VLM wrapper (replaces `obs.text` with the schema each step)

```python
import gym_v
from gymv_wrapper import GymVSchemaWrapper

env = gym_v.make("Games/Game2048-v0")
env = GymVSchemaWrapper(
    env,
    model="gpt-5.5",
    goal_override="Reach 2048 efficiently.",
)
obs_dict, info = env.reset()
# obs_dict["agent_0"].text is now the full <state>…</state>
```

> Calling a VLM every step is **expensive**. Use this for offline label
> generation, not real-time play.

### 4) Per-game visual-grounding wrappers (Temporal/*)

These attach a JSON `visual_grounding` dict to `obs.metadata` (no API call):

```python
import gym_v
from gymv_wrapper import (
    TEMPORAL_WRAPPER_BY_ENV_ID,
    TemporalAirstrikerVisualGroundingWrapper,
)

env = gym_v.make("Temporal/Airstriker-v0")
env = TemporalAirstrikerVisualGroundingWrapper(env, append_text_summary=True)

# Or, generic by env id:
env_id = "Temporal/StreetsOfRage2-v0"
env = TEMPORAL_WRAPPER_BY_ENV_ID[env_id](gym_v.make(env_id))

obs_dict, info = env.reset()
vg = obs_dict["agent_0"].metadata["visual_grounding"]
print(vg["display_name"], vg["genre"], vg["visual"])  # 'Streets of Rage 2', 'beatemup', {'width': 320, 'height': 224, 'channels': 3}
```

### 5) Schema shape (`build_temporal_visual_schema`)

```python
from gymv_wrapper import build_temporal_visual_schema
schema = build_temporal_visual_schema("Temporal/Columns-v0", obs)
```

Returns:

```jsonc
{
  "schema_kind": "gymv.temporal_visual_grounding",
  "schema_version": "1.0",
  "gym_env_id": "Temporal/Columns-v0",
  "retro_integration": "Columns-Genesis-v0",
  "display_name": "Columns",
  "genre": "puzzle",
  "grounding_focus": "jewel stacks, falling column, match clears",
  "visual": {"width": 320, "height": 224, "channels": 3},
  "simulation": {
    "frame_index": 7, "episode_reward": 1.0, "step_reward": 0.0,
    "last_action": "DOWN", "action_history": ["NOOP", "DOWN", "..."]
  },
  "parsed_text": {"game": "Columns-Genesis-v0", "score": "120", "...": "..."},
  "ram_watch": {"score": 120, "level": 1, "...": "..."},
  "control": {"available_actions": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "C", "START"]},
  "entities": [
    {"eid": "e1", "type": "region", "label": "genesis_viewport", "...": "..."},
    {"eid": "e2", "type": "text",   "label": "hud_score", "value": "120"}
  ]
}
```

### 6) Tool-calling registry (multi-turn grounding)

```python
from gymv_wrapper import build_gymv_registry
from vlm_wrapper.tool_loop import run_tool_loop

registry = build_gymv_registry(
    obs_text=obs.text or "",
    description=env.description,
    step=step,
)
result = run_tool_loop(
    image=obs.image,
    domain="gymv",
    registry=registry,
    goal="Reach 2048",
    task_id="Games/Game2048-v0",
)
print(result["schema"])
print(result["tool_trace"])  # SFT data: list of (call, result)
```

Available gymv-specific tools: `query_entity_pos`, `list_entities`,
`check_relation`, `get_state_flags`, `list_valid_actions`, `get_grid_state`,
`check_deadlock`, `spatial_analysis`, `count_merge_candidates`.

---

## Driver script: visual-grounding rollouts (unified image head)

End-to-end batch driver. The gymv-specific
`generate_gymv_visual_schema.py` was retired in favour of the cross-domain
visual grounding head `generate_gymv_image_schema.py`, which is the
unified entry point for every visual grounding task (gymv, env wrappers,
benchmark image-QA, …). The old script still exists as a thin
deprecation shim that forwards to the new one with identical CLI flags.

```bash
# rollout 1 episode of 3 NOOP frames, GPT-5.5 by default
export OPENROUTER_API_KEY=...   # or OPENAI_API_KEY
python visual_grounding_tests/generate_gymv_image_schema.py \
    --envs Temporal/Airstriker-v0 --episodes 1 --max_steps 3 -v

# dry run: no API; only heuristic grounding + saved frames
python visual_grounding_tests/generate_gymv_image_schema.py --dry_run \
    --envs Temporal/Airstriker-v0 --episodes 1 --max_steps 2

# all 13 games
python visual_grounding_tests/generate_gymv_image_schema.py \
    --envs Temporal/Airstriker-v0 Temporal/AlteredBeast-v0 \
           Temporal/CastleOfIllusion-v0 Temporal/CastlevaniaBloodlines-v0 \
           Temporal/Columns-v0 Temporal/DynamiteHeaddy-v0 \
           Temporal/GoldenAxe-v0 Temporal/KidChameleon-v0 \
           Temporal/MortalKombatII-v0 Temporal/SpaceHarrierII-v0 \
           Temporal/StreetsOfRage2-v0 Temporal/Strider-v0 \
           Temporal/ThunderForceIII-v0
```

Outputs land in
`visual_grounding_tests/output/gymv_image/<env_id_sanitized>/<run_id>_ep<NNN>/`:

* `images/step_NNN.png` — the rendered frame the VLM was grounded against.
* `steps.jsonl` — one JSON object per step (heuristic grounding + VLM
  schema under `schema_image_llm`, with `head: "image"` tagging).
* `run_summary.json` — model id, timing, `image_schema_ok` /
  `image_schema_fail` counts.

> **Migration note:** the legacy output tag was `gpt55_gymv` and the
> per-step VLM key was `vlm_label` / `vlm_schema_parsed_ok`. Update any
> downstream readers to the unified keys above.

A batch summary `batch_<run_id>.json` is written at the output root.

---

## Tests / smoke checks

```bash
# unit tests for the temporal visual schema
pytest gymv_wrapper/tests/test_temporal_visual_schema.py -v

# pre-flight modality + ROM coverage probe
python visual_grounding_tests/check_io_modalities.py [--runtime] [--json]

# full env smoke test (requires ROMs imported)
python install/gymv_temporal_patch/examples/temporal_smoketest.py
```

The smoketest enforces both modalities at runtime via:

```python
if first_frame is None or not first_text:
    raise RuntimeError("reset produced incomplete observation")
```

---

## Design notes / non-goals

* **No pixel-level object detection.** `build_temporal_visual_schema`
  fuses the visual frame *dimensions* with simulator ground truth
  (RAM watch, scores) into one JSON contract. For pixel-grounded
  detection wire in `vlm_wrapper.grounding` (OmniParser) instead.
* **Cross-env code stays in `vlm_wrapper/`.** Anything Gym-V–specific
  belongs here so other domains (BrowserGym, OS-World, video, etc.) can
  reuse `vlm_wrapper.schema` without dragging in stable-retro.
* **Backward compatibility.** `vlm_wrapper.gymv_adapter`,
  `vlm_wrapper.gymv_heuristic`, `vlm_wrapper.tools_gymv` re-export from
  this package. Importing `gymv_wrapper.adapter` first is now safe even
  when `vlm_wrapper.__init__` would otherwise create a circular import
  (the gymv re-exports are lazy via PEP 562 `__getattr__`).
* **Default model is `gpt-5.5`** (was `gpt-4o`); override with
  `VLM_LABEL_MODEL` or the `model=` argument. Routing goes through
  `API_func.make_openai_client` / `effective_openai_model` so OpenRouter
  is auto-prefixed (`openai/gpt-5.5`) when its key is set.
