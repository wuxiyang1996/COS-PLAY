# Inference

Run the **decision agent** ([decision_agents/](../decision_agents/)) and store rollouts in the **data_structure** format ([data_structure/experience.py](../data_structure/experience.py)): `Experience` list and `Episode`.

## VERL-based inference (recommended for vLLM/sglang)

Inference using [VERL](https://github.com/verl-project/verl) and [verl-agent](https://github.com/verl-project/verl-agent) runs the same env and reward as training, in evaluation-only mode (no PPO updates):

```bash
# From repo root; requires verl-agent at ../verl-agent
python -m inference.run_verl_inference
# With overrides (Hydra)
python -m inference.run_verl_inference trainer.val_before_train=True data.val_batch_size=8
```

Or via the scripts runner:

```bash
python -m scripts.run_inference --verl
python -m scripts.run_inference --verl data.val_batch_size=8
```

## Local inference (single episode, no VERL)

```python
from inference import run_inference, rollout_to_episode
from data_structure.experience import Episode_Buffer, Experience_Replay_Buffer

# Run one episode and get an Episode (list of Experience)
episode = run_inference(
    env,
    task="Complete level 1",
    max_steps=500,
    verbose=True,
)

# Optional: add to buffers and/or append to a JSONL file
ep_buffer = Episode_Buffer(buffer_size=100)
exp_buffer = Experience_Replay_Buffer(buffer_size=10_000)
episode = run_inference(
    env,
    task="Complete level 1",
    episode_buffer=ep_buffer,
    experience_buffer=exp_buffer,
    save_path="rollouts/episodes.jsonl",
    verbose=True,
)
```

## Direct Episode return from `run_episode_vlm_agent`

`run_episode_vlm_agent()` now returns an `Episode` directly — no conversion step is needed:

```python
from decision_agents import run_episode_vlm_agent

episode = run_episode_vlm_agent(env, task="My task", max_steps=500)
# episode.experiences: list of Experience with rich fields:
#   - summary_state, intentions, sub_tasks (from agent internal state)
#   - reward_details (per-step reward breakdown: r_env, r_follow, r_cost, r_total)
# episode.metadata: flat rollout data for backward compatibility
#   - observations, actions, rewards, cumulative_reward, agent_state, done, steps
# episode.to_dict(): for JSON save/load
```

## Converting a legacy flat dict (backward compatibility)

`rollout_to_episode()` still accepts legacy flat dicts (with `observations`, `actions`, `rewards` keys) and converts them to `Episode`. If passed an `Episode` object, it passes through unchanged (optionally updating `task`):

```python
from inference import rollout_to_episode

# Legacy dict from older code
legacy = {"observations": [...], "actions": [...], "rewards": [...], "done": True}
episode = rollout_to_episode(legacy, task="My task")
```

## Storage format

- **Episode**: `experiences` (list of `Experience`), `task`, `outcome`, `summary`, `metadata`. See `Episode.to_dict()`.
- **Experience**: `state`, `action`, `reward`, `next_state`, `done`, optional `intentions`, `tasks`, `sub_tasks`, `summary`, `summary_state`, `idx`, `reward_details`, etc. All `from_dict()`/`to_dict()` methods are classmethods and handle all optional fields.
- **save_path**: If provided, each episode is appended as one JSON object per line (JSONL), so multiple runs can append to the same file.

## Dependencies

- **Local inference:** `decision_agents` (`run_episode_vlm_agent`, `VLMDecisionAgent`), `data_structure.experience` (`Experience`, `Episode`, optionally `Episode_Buffer`, `Experience_Replay_Buffer`)
- **VERL inference:** `verl-agent` at `../verl-agent` (install with `pip install -e .`); uses same stack as VERL training (Ray, vLLM/sglang)
