#!/usr/bin/env python3
"""Convert RL model rollout JSONs to the episode format expected by extract_skillbank_gpt54.py.

Reads: labeling/output/rl_rollouts/alfworld/episode_XXXX.json
Writes three views:
  - rl_rollouts_labeled: all episodes
  - rl_rollouts_success: successful episodes for reusable-skill extraction
  - rl_rollouts_failure: failed episodes for failure-lesson extraction

The extraction pipeline expects:
  - experiences[]: state, action, reward, next_state, done, intentions, summary_state
  - task, game_name, episode_id, env_name
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_agents.agent_helper import build_rag_summary


def convert_episode(src: Path, dst: Path) -> dict:
    with open(src) as f:
        data = json.load(f)

    traj = data["trajectory"]
    task = data["task"]
    experiences = []

    for i, step in enumerate(traj):
        obs = step["observation"]
        action = step["action"]
        reward = step["reward"]

        # next_state is the next step's observation, or the last obs
        if i + 1 < len(traj):
            next_obs = traj[i + 1]["observation"]
        else:
            next_obs = obs

        done = step.get("terminated", False) or step.get("truncated", False)
        reasoning = step.get("reasoning", "")

        # Build summary_state from observation using our fact extractor
        summary_state = build_rag_summary(obs, game_name="alfworld", max_chars=400)

        # Infer intention from reasoning
        intention = ""
        if reasoning:
            first_line = reasoning.split("\n")[0].strip()
            if len(first_line) > 10:
                intention = first_line[:200]

        experiences.append({
            "idx": i,
            "state": obs,
            "raw_state": obs,
            "action": action,
            "reward": reward,
            "next_state": next_obs,
            "raw_next_state": next_obs,
            "done": done,
            "intentions": intention,
            "summary_state": summary_state,
            "summary": f"Step {i}: {action} (r={reward})",
        })

    episode = {
        "episode_id": data["episode_id"],
        "game_name": "alfworld",
        "env_name": "gamingagent",
        "task": task,
        "task_type": data.get("task_type", ""),
        "success": data["success"],
        "total_reward": data["total_reward"],
        "experiences": experiences,
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(episode, f, indent=1)

    return {"success": data["success"], "steps": len(experiences)}


def main():
    src_dir = PROJECT_ROOT / "labeling" / "output" / "rl_rollouts" / "alfworld"
    dst_dir = PROJECT_ROOT / "labeling" / "output" / "rl_rollouts_labeled" / "alfworld"
    success_dir = PROJECT_ROOT / "labeling" / "output" / "rl_rollouts_success" / "alfworld"
    failure_dir = PROJECT_ROOT / "labeling" / "output" / "rl_rollouts_failure" / "alfworld"

    for split_dir in (success_dir, failure_dir):
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)

    files = sorted(src_dir.glob("episode_*.json"))
    print(f"Found {len(files)} rollout files")

    n_success = 0
    n_total = 0
    for f in files:
        dst = dst_dir / f.name
        info = convert_episode(f, dst)
        n_total += 1
        if info["success"]:
            n_success += 1
            shutil.copy2(dst, success_dir / f.name)
        else:
            shutil.copy2(dst, failure_dir / f.name)

    print(f"Converted {n_total} episodes ({n_success} successful)")
    print(f"All:      {dst_dir}")
    print(f"Success:  {success_dir} ({n_success})")
    print(f"Failure:  {failure_dir} ({n_total - n_success})")


if __name__ == "__main__":
    main()
