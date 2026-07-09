# Run VLM decision agent and store rollouts as Episode (data_structure format).
# Converts run_episode_vlm_agent result into Experience list + Episode; optional buffer/store.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from decision_agents import run_episode_vlm_agent, VLMDecisionAgent

try:
    from data_structure.experience import Experience, Episode, Episode_Buffer, Experience_Replay_Buffer
except ImportError:
    Experience = None
    Episode = None
    Episode_Buffer = None
    Experience_Replay_Buffer = None


def rollout_to_episode(
    rollout,
    task: str = "",
):
    """Convert a decision-agent rollout into an Episode (data_structure format).

    Accepts either:
      - An Episode object (returned directly by run_episode_vlm_agent) — passed through.
      - A legacy flat dict with keys: observations, actions, rewards, done.
    """
    if Episode is None or Experience is None:
        raise ImportError(
            "Experience and Episode are required from data_structure.experience. "
            "Install or add the data_structure package."
        )

    # If already an Episode, just return it (optionally update the task).
    if isinstance(rollout, Episode):
        if task and (not rollout.task or rollout.task == "Unspecified task"):
            rollout.task = task
        return rollout

    # Legacy path: flat dict from older callers.
    observations: List[str] = list(rollout.get("observations", []))
    actions: List[Any] = list(rollout.get("actions", []))
    rewards: List[float] = list(rollout.get("rewards", []))
    reward_details_list: List[dict] = list(rollout.get("reward_details", []))
    done_flag: bool = bool(rollout.get("done", False))

    experiences: List[Experience] = []
    n = len(actions)
    if n == 0:
        ep = Episode(experiences=experiences, task=task or "Unspecified task")
        ep.set_outcome()
        return ep

    for i in range(n):
        state = observations[i] if i < len(observations) else ""
        next_state = observations[i + 1] if i + 1 < len(observations) else ""
        action = actions[i] if i < len(actions) else None
        reward = float(rewards[i]) if i < len(rewards) else 0.0
        done = (i == n - 1) and done_flag

        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            intentions=None,
            tasks=task if task else None,
            sub_tasks=None,
        )
        exp.idx = i
        if i < len(reward_details_list):
            exp.reward_details = reward_details_list[i]
        experiences.append(exp)

    episode = Episode(
        experiences=experiences,
        task=task or "Unspecified task",
        env_name=rollout.get("env_name") or rollout.get("game") or "",
        game_name=rollout.get("game_name") or "",
        metadata={k: v for k, v in rollout.items()
                  if k not in ("observations", "actions", "rewards", "reward_details")},
    )
    episode.set_outcome()
    return episode


def run_inference(
    env,
    agent: Optional[VLMDecisionAgent] = None,
    *,
    task: str = "",
    model: Optional[str] = None,
    skill_bank: Any = None,
    memory: Any = None,
    reward_config: Any = None,
    max_steps: int = 1000,
    verbose: bool = False,
    episode_buffer: Optional[Episode_Buffer] = None,
    experience_buffer: Optional[Experience_Replay_Buffer] = None,
    save_path: Optional[str] = None,
):
    """
    Run one episode with the VLM decision agent and return/store the rollout
    in data_structure format (Episode with Experience list).

    - Runs run_episode_vlm_agent(env, agent=..., max_steps=..., verbose=...).
    - Converts the result to Episode via rollout_to_episode(..., task=task).
    - If episode_buffer is provided, adds the episode to it.
    - If experience_buffer is provided, adds the episode's experiences to it.
    - If save_path is provided, appends the episode to a JSON file (one JSON object
      per line, or a single JSON list if the file is new and you want one episode
      per file; here we write episode.to_dict() as one JSON object per line for
      append-friendly storage).

    Returns:
        Episode instance (data_structure.experience.Episode).
    """
    if Episode is None or Experience is None:
        raise ImportError(
            "Experience and Episode are required from data_structure.experience."
        )

    episode = run_episode_vlm_agent(
        env,
        agent=agent,
        model=model,
        skill_bank=skill_bank,
        memory=memory,
        reward_config=reward_config,
        task=task,
        max_steps=max_steps,
        verbose=verbose,
    )

    if episode_buffer is not None:
        episode_buffer.add_episode(episode)
        if verbose:
            print(f"Added episode to episode buffer (size: {len(episode_buffer)})")

    if experience_buffer is not None:
        experience_buffer.add_experience(episode)
        if verbose:
            print(f"Added {len(episode.experiences)} experiences to experience buffer")

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            line = json.dumps(episode.to_dict(), ensure_ascii=False) + "\n"
            f.write(line)
        if verbose:
            print(f"Appended episode to {path}")

    return episode
