#!/usr/bin/env python
"""ALFWorld text-protocol ablation runner.

This runner intentionally stays small: it evaluates the wrapper/runtime
protocol by rolling out a random admissible-action policy. It gives us a real
ALFWorld output artifact without pretending that a model-backed ALFWorld actor
already exists in the repo.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from env_wrappers.alfworld_nl_wrapper import make_alfworld_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["with_admissible", "without_admissible"], required=True)
    parser.add_argument("--split", default="eval_out_of_distribution")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _safe_action(actions: List[str]) -> str:
    if actions:
        return random.choice(actions)
    return "look"


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)

    include_admissible = args.variant == "with_admissible"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    episodes: List[Dict[str, Any]] = []
    t0 = time.time()
    env = make_alfworld_env(
        split=args.split,
        max_steps=args.max_steps,
        include_admissible=include_admissible,
    )
    try:
        for ep_idx in range(args.episodes):
            obs, info = env.reset()
            del obs
            total_reward = 0.0
            steps = 0
            done = False
            trace = []
            for step_idx in range(args.max_steps):
                actions = list(getattr(env, "action_names", []) or info.get("admissible_actions", []) or [])
                action = _safe_action(actions)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps = step_idx + 1
                trace.append({
                    "step": step_idx,
                    "action": action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "obs_preview": str(obs).replace("\n", " ")[:240],
                })
                if terminated or truncated:
                    done = bool(terminated)
                    break
            episodes.append({
                "episode": ep_idx,
                "reward": total_reward,
                "steps": steps,
                "success": done,
                "trace": trace,
            })
    finally:
        env.close()

    summary = {
        "domain": "alfworld",
        "variant": args.variant,
        "split": args.split,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "wall_time_s": time.time() - t0,
        "mean_reward": sum(ep["reward"] for ep in episodes) / len(episodes) if episodes else 0.0,
        "success_rate": sum(1 for ep in episodes if ep["success"]) / len(episodes) if episodes else 0.0,
        "mean_steps": sum(ep["steps"] for ep in episodes) / len(episodes) if episodes else 0.0,
        "episodes_detail": episodes,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in [
        "domain", "variant", "episodes", "mean_reward", "success_rate", "mean_steps",
    ]}, indent=2))
    print(f"[alfworld-ablation] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

