# `gymv_temporal_patch` — multimodal upgrades for `Temporal/*` (stable-retro) envs

This folder is a self-contained patch that brings the `Temporal/*` (Sega
Genesis / stable-retro) environments in [ModalMinds/gym-v](https://github.com/ModalMinds/gym-v)
up to the same multimodal standard as the rest of `gym-v`: every step
returns both a **textual** representation (game state, action history,
step / episode bookkeeping) and a **visual** representation (raw RGB
frame, plus optional `Resize → Grayscale → FrameStack` wrappers).

It is shipped under `install/` so it slots into the `Multi-hop-Reasoning-VLM-Agent`
install flow alongside [`install_gymv.sh`](../install_gymv.sh) /
[`gymv.environment.yml`](../gymv.environment.yml). It does **not** modify
the agent's own [`env_wrappers/`](../../env_wrappers/) — those are the
benchmark-side adapters and live in a separate layer.

---

## TL;DR

> **Recommended path (unified `game-ai-agent` env):** the patch is applied
> automatically when you pass a ROM zip to `install_main_env.sh`:
>
> ```bash
> bash install/install_main_env.sh "" /path/to/Mega_Drive_Mini_Full_Set.zip
> ```
>
> The standalone `gymv` env path below still works — use it for
> VLM-eval-only setups that don't need the full training stack.

```bash
# Standalone path (env: gymv)
bash install/install_gymv.sh
bash install/gymv_temporal_patch/apply_patch.sh \
     /fs/gamma-projects/vlm-robot/gym-v \
     /fs/gamma-projects/vlm-robot/ROMs/Mega_Drive_Mini_Full_Set.zip

# Unified path (env: game-ai-agent) — manually re-running the patch later
conda activate game-ai-agent
bash install/gymv_temporal_patch/apply_patch.sh \
     /path/to/parent/gym-v \
     /path/to/Mega_Drive_Mini_Full_Set.zip
```

The script will: install `stable-retro` if missing, extract+import the 12
commercial Mega Drive Mini ROMs (plus the bundled `Airstriker-Genesis-v0`),
copy four files into the `gym-v` source tree, fix the
`Airstriker-Genesis` ↔ `Airstriker-Genesis-v0` mismatch, and run a smoke
test on all 13 retro envs.

---

## What's in this folder

```
gymv_temporal_patch/
├── README.md                                       (this file)
├── apply_patch.sh                                  (idempotent installer)
├── gym_v/
│   ├── wrappers/
│   │   ├── __init__.py                             (re-exports the new wrappers)
│   │   └── observation.py                          (NEW: 4 multimodal wrappers)
│   └── envs/multi_turn/temporal/
│       └── retro_env.py                            (UPGRADED RetroGymVEnv)
└── examples/
    └── temporal_smoketest.py                       (verifies all 13 envs)
```

Each file is a **drop-in replacement** for the path of the same name
inside `gym-v/`. There is no third-party packaging glue — `apply_patch.sh`
just copies them in.

---

## What each change does

### `gym_v/envs/multi_turn/temporal/retro_env.py` *(replaces upstream)*

Upgrades the `RetroGymVEnv` adapter that wraps stable-retro:

* **Visual representation (unchanged on the way in, instrumented on the way out)**
  * `Observation.image` is still a `PIL.Image` of the raw 320×224 RGB frame.
  * `Observation.metadata` now exposes a structured dict
    `{game, frame_index, step_reward, episode_reward, last_action,
    action_history, ram_watch, available_actions}` so downstream
    wrappers don't have to re-parse the text.

* **Textual representation (richer, single-line, stable layout)**
  ```
  Game: GoldenAxe-Genesis-v0 | Score: 0 | Lives: 3 | Frame: 6 |
  StepReward: 0.000 | EpReward: 0.000 | LastAction: UP+A |
  Recent: [NOOP,RIGHT,A,B+RIGHT,START,UP+A]
  ```
  * **Per-game RAM variables** (e.g. `Health`, `enemy_x_position`,
    `levelHi`, `gameover`) are auto-discovered from
    `<stable_retro>/data/stable/<GAME>/data.json` at construction
    time; well-known keys (`score`, `lives`, `health`, `level`,
    `gameover`) come first in canonical order, followed by the
    game-specific keys.
  * **Bookkeeping** added every step: `frame_index`, `episode_reward`,
    last action, sliding `action_history` window
    (`action_history_len=8` by default).

* **Misc**
  * Robust game-name resolution — if `Foo-Genesis` doesn't resolve in
    stable-retro's index, `Foo-Genesis-v0` is tried automatically and
    a `WARN` is emitted (this also covers people running with the
    pre-patch `__init__.py`).
  * `terminated` / `truncated` cast to native `bool` (some
    stable-retro versions return `np.bool_`).
  * `_action_to_mask` no longer crashes on `None` / empty action.

### `gym_v/wrappers/observation.py` *(new file)*

Four composable wrappers, all `RecordConstructorArgs`-aware so
`env.spec.additional_wrappers` round-trips them:

| Wrapper | Effect |
|---|---|
| `GrayscaleObservation(env, keep_dim=False)` | `Image.convert("L")`. With `keep_dim=True` the result is re-expanded to RGB so downstream code expecting 3-channel images keeps working. |
| `ResizeObservation(env, size, resample=BILINEAR)` | Resize to fixed `(W, H)` (or square `int`). |
| `FrameStack(env, num_stack=4)` | Replaces `Observation.image` with `list[PIL.Image.Image]` of the last *k* frames (oldest → newest). On reset, buffer is filled with the initial frame. Per-agent buffers (multi-agent friendly). |
| `TextStateAugmenter(env, include_fields=...)` | Appends bookkeeping from `info` to `Observation.text` so language-model agents see it without parsing `info`. Default fields: `frame_index, episode_reward, last_action, action_history`. |

All four operate transparently on either `Image.Image` or
`list[Image.Image]`, so they compose cleanly:

```python
env = gym_v.make("Temporal/GoldenAxe-v0")
env = StochasticFrameSkip(env, n=4, stickprob=0.25)   # already in gym-v
env = ResizeObservation(env, size=(160, 112))
env = GrayscaleObservation(env, keep_dim=True)
env = FrameStack(env, num_stack=4)
env = HistoryRecorder(env, max_turns=64)              # already in gym-v
env = TextStateAugmenter(env)
```

### `gym_v/wrappers/__init__.py` *(replaces upstream)*

Adds the four new symbols to the public re-export list. No behaviour
change for anything that was already there (`PassiveEnvChecker`,
`OrderEnforcing`, `StochasticFrameSkip`, `HistoryRecorder`,
`ToolWrapper`).

### Airstriker game-name typo *(in-place sed)*

Upstream `gym_v/envs/__init__.py` registers `Temporal/Airstriker-v0`
with `game="Airstriker-Genesis"`, but stable-retro 1.0 stores the ROM
under `Airstriker-Genesis-v0`. Result: `gym_v.make("Temporal/Airstriker-v0")`
raises `FileNotFoundError`. `apply_patch.sh` fixes both occurrences:

```diff
-        game="Airstriker-Genesis",
+        game="Airstriker-Genesis-v0",
```

```diff
- TEST_GAME = "Airstriker-Genesis"
+ TEST_GAME = "Airstriker-Genesis-v0"
```

(The replacement `retro_env.py` also resolves this at runtime as a
defence-in-depth, but the sed makes the registration self-explanatory
for anyone reading the source.)

### `examples/temporal_smoketest.py` *(new file)*

Iterates every registered `Temporal/*` env, runs a bare reset+step
pass, then a fully-wrapped reset+step pass, asserts both modalities are
populated, and (with `--save-frames`) writes
`00_reset.png`, `01_after_steps.png`, `02_wrapped_framestack.png` per
env to `examples/temporal_smoketest_out/`.

---

## Future installation workflow

There are two clean paths on a brand-new machine, depending on whether
you want the full training stack or a slim env for VLM-eval only.

### A. Unified path — recommended (env: `game-ai-agent`)

```bash
cd /path/to/parent

# install_main_env.sh installs gym-v + clones it + applies this patch in one shot.
bash Multi-hop-Reasoning-VLM-Agent/install/install_main_env.sh \
     ""                                                                \
     /path/to/Mega_Drive_Mini_Full_Set.zip
```

This is what we ship for the COS-PLAY pipeline because everything (GRPO,
vLLM, GamingAgent, gym-v Games/Spatial/Temporal) ends up in one env.

### B. Standalone path (env: `gymv`)

```bash
cd /fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent

# 1. Install the gymv conda env + clone gym-v editable.
bash install/install_gymv.sh
#    (creates conda env `gymv`, clones gym-v, pip install -e .[games,spatial],
#     runs install/gymv_smoke.py)

# 2. Drop the Temporal/* upgrades on top.
bash install/gymv_temporal_patch/apply_patch.sh
#    Default args:
#      $1 = /fs/gamma-projects/vlm-robot/gym-v
#      $2 = /fs/gamma-projects/vlm-robot/ROMs/Mega_Drive_Mini_Full_Set.zip
```

After either path, `python -c "import gym_v, gym_v.envs; print(sorted(e for e in gym_v.registry if e.startswith('Temporal/')))"` lists all 13 retro envs and they are usable end-to-end.

### One-shot integration into `install_gymv.sh` (optional)

If you only use the standalone env and want a single command, append a
call to the patcher at the bottom of
[`install_gymv.sh`](../install_gymv.sh):

```bash
echo "[5/4] Applying Temporal/* multimodal patch ..."
bash "${SCRIPT_DIR}/gymv_temporal_patch/apply_patch.sh" "$GYMV_DIR"
```

`apply_patch.sh` is idempotent (it greps before sed-ing, checks ROM
presence before importing, etc.), so re-running `install_gymv.sh` stays
safe.

> The unified `install_main_env.sh` already does this when invoked with a
> ROM zip — no manual patch step needed.

---

## Editing this patch

If `gym-v` upstream changes `RetroGymVEnv` or the wrappers package and
you need to refresh this patch:

1. Apply your changes inside the live tree at
   `/fs/gamma-projects/vlm-robot/gym-v` and verify with
   `python examples/temporal_smoketest.py`.
2. Sync the four files back here:
   ```bash
   GYMV=/fs/gamma-projects/vlm-robot/gym-v
   PATCH=Multi-hop-Reasoning-VLM-Agent/install/gymv_temporal_patch
   cp "$GYMV/gym_v/wrappers/__init__.py"                          "$PATCH/gym_v/wrappers/__init__.py"
   cp "$GYMV/gym_v/wrappers/observation.py"                       "$PATCH/gym_v/wrappers/observation.py"
   cp "$GYMV/gym_v/envs/multi_turn/temporal/retro_env.py"         "$PATCH/gym_v/envs/multi_turn/temporal/retro_env.py"
   cp "$GYMV/examples/temporal_smoketest.py"                      "$PATCH/examples/temporal_smoketest.py"
   ```
3. If you also fixed something with sed inside `apply_patch.sh`
   (e.g. another upstream typo), document it in the **What each change
   does** section above so future readers know what the script
   touches.
4. Re-run `bash install/gymv_temporal_patch/apply_patch.sh` against a
   fresh `gym-v` checkout to confirm the patch still applies cleanly.

### Manual / dry-run application

If you'd rather not run the script, the patch is just four files plus
two one-line typo fixes. Manually:

```bash
GYMV=/fs/gamma-projects/vlm-robot/gym-v
PATCH=$(pwd)/install/gymv_temporal_patch

# 1. Copy files
install -m 0644 "$PATCH/gym_v/wrappers/__init__.py"                  "$GYMV/gym_v/wrappers/__init__.py"
install -m 0644 "$PATCH/gym_v/wrappers/observation.py"               "$GYMV/gym_v/wrappers/observation.py"
install -m 0644 "$PATCH/gym_v/envs/multi_turn/temporal/retro_env.py" "$GYMV/gym_v/envs/multi_turn/temporal/retro_env.py"
install -m 0644 "$PATCH/examples/temporal_smoketest.py"              "$GYMV/examples/temporal_smoketest.py"

# 2. Fix the Airstriker typo (registration + tests)
sed -i 's/"Airstriker-Genesis"/"Airstriker-Genesis-v0"/' "$GYMV/gym_v/envs/__init__.py"
sed -i 's/TEST_GAME = "Airstriker-Genesis"$/TEST_GAME = "Airstriker-Genesis-v0"/' "$GYMV/tests/test_retro_integration.py"

# 3. ROM import (only the first time)
python -m retro.import /path/to/extracted/genesis_roms_flat/

# 4. Verify
( cd "$GYMV" && python examples/temporal_smoketest.py )
```

---

## Verification expectations

After `apply_patch.sh` finishes, the smoke test prints one block per env
and ends with `All envs passed.`. A representative line for
`MortalKombatII-Genesis-v0` after the wrapped pipeline:

```
=== Temporal/MortalKombatII-v0 ===
  image_size:   (320, 224)
  text_first:   Game: MortalKombatII-Genesis-v0 | Frame: 0 | StepReward: 0.000 | EpReward: 0.000
  text_last:    Game: MortalKombatII-Genesis-v0 | Health: 120 | enemy_health: 120 |
                enemy_rounds_won: 0 | rounds_won: 0 | wins: 0 | x_position: 602 |
                enemy_x_position: 750 | y_position: 0 | enemy_y_position: 0 |
                Frame: 6 | StepReward: 0.000 | EpReward: 0.000 | LastAction: UP+A |
                Recent: [NOOP,RIGHT,A,B+RIGHT,START,UP+A]
  ep_reward:    0.0
  frame_index:  6
  wrapped img:  list[Image] x4
  wrapped text: ... (same fields, plus the TextStateAugmenter block)
```

`pytest tests/test_retro_integration.py -q` should report `12 passed`
(all 12 stop being skipped because the Airstriker name is now correct).

---

## Compatibility notes

* Tested with `stable-retro==1.0.0`, `gymnasium 1.3.0` (matches `game-ai-agent`'s pin) or `gymnasium>=1.2.2` (standalone `gymv` env), Python 3.11. Works on `numpy 1.26` *and* `numpy 2.x`.
* `RetroGymVEnv.reset(seed=...)` forwards seeds to stable-retro,
  which is deterministic at the emulator level. The text representation
  is identical for identical action sequences across Python sessions.
* `FrameStack` produces `list[Image.Image]`. The gym-v `Observation`
  pydantic model already permits this (`image: Image | list[Image] | None`),
  so existing renderers / agents that explicitly type-check for
  `Image.Image` may need to handle the list case (the smoke test shows
  the standard pattern: render the most recent / tile horizontally).
