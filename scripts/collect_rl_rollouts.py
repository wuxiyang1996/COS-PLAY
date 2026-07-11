#!/usr/bin/env python3
"""Collect rollout trajectories from the SkillRL RL model on ALFWorld.

Saves per-episode JSON files compatible with our skill extraction pipeline.
Each file contains the full trajectory (observations, actions, rewards)
plus metadata (task, success, steps).

Usage:
    python scripts/collect_rl_rollouts.py \
        --episodes 120 --split train --out-dir labeling/output/rl_rollouts/alfworld
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_skillrl_native import (
    TEMPLATE_NO_HIS,
    TEMPLATE_WITH_MEMORY,
    build_prompt,
    format_admissible,
    parse_action,
    pick_rl_section,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--model", default="skillrl-alfworld")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--history-window", type=int, default=2)
    ap.add_argument("--out-dir", default="labeling/output/rl_rollouts/alfworld")
    args = ap.parse_args()

    import openai

    from env_wrappers.alfworld_nl_wrapper import ALFWorldNLWrapper

    blocks_path = (
        PROJECT_ROOT
        / "labeling"
        / "output"
        / "skillrl_native"
        / "rl_skill_blocks.json"
    )
    blocks = json.loads(blocks_path.read_text())

    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)
    env = ALFWorldNLWrapper(
        split=args.split, max_steps=args.max_steps, shuffle=True
    )

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_success = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        task = env._goal
        section = pick_rl_section(task)
        exp_block = blocks.get(section) or blocks["pick_and_place"]

        trajectory = []
        history = []
        cur_obs = obs
        success = False
        raw_reward = 0.0
        t0 = time.time()

        for step in range(args.max_steps):
            admissible = info.get("action_names", [])
            if not admissible:
                break

            prompt = build_prompt(
                task, exp_block, history, cur_obs, admissible,
                window=args.history_window,
            )
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    max_tokens=512,
                )
                reply = resp.choices[0].message.content or ""
            except Exception as exc:
                print(f"  [ep {ep}] LLM error: {exc}")
                reply = ""

            action = parse_action(reply, admissible)
            obs_new, reward, terminated, truncated, info = env.step(action)
            raw = float(info.get("raw_env_reward", reward))
            raw_reward += raw

            # Extract reasoning from <think>...</think>
            think_m = re.search(r"<think>(.*?)</think>", reply, re.DOTALL)
            reasoning = think_m.group(1).strip() if think_m else ""

            trajectory.append(
                {
                    "step": step,
                    "observation": cur_obs,
                    "action": action,
                    "reward": raw,
                    "cumulative_reward": raw_reward,
                    "admissible_actions": admissible,
                    "reasoning": reasoning,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

            history.append((cur_obs, action))
            cur_obs = obs_new

            if terminated or truncated:
                success = raw_reward >= 1.0
                break

        episode_data = {
            "episode_id": ep,
            "game": "alfworld",
            "task": task,
            "task_type": section,
            "success": success,
            "total_reward": raw_reward,
            "steps": len(trajectory),
            "wall_seconds": round(time.time() - t0, 1),
            "trajectory": trajectory,
        }

        fname = out_dir / f"episode_{ep:04d}.json"
        with open(fname, "w") as f:
            json.dump(episode_data, f, indent=1)

        if success:
            n_success += 1
        print(
            f"[ep {ep}] {'SUCCESS' if success else 'fail'} | "
            f"type={section} | task={task[:50]} | "
            f"steps={len(trajectory)} | running={n_success}/{ep+1} "
            f"({n_success/(ep+1)*100:.0f}%)",
            flush=True,
        )

    print(
        f"\nDONE: {n_success}/{args.episodes} = "
        f"{n_success/args.episodes*100:.1f}% success"
    )
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
