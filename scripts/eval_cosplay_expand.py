#!/usr/bin/env python3
"""Expanded ALFWorld COS-PLAY evaluation: full OOD set and/or pick_two only.

Reuses the fast controller. Supports:
  --all-ood     : run every solvable game in eval_out_of_distribution (no shuffle)
  --pick-two    : filter to put/find two object tasks only
  --episodes N  : random sample of N (default) with --seed
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import controller pieces by running the module's helpers
from scripts.eval_cosplay_fast_controller import (  # noqa: E402
    Controller,
    load_skills,
    build_prompt,
    parse_action,
)


def is_pick_two(task: str) -> bool:
    return bool(re.search(r"\b(?:find|put)\s+two\b", task.lower()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--model", default="skillrl-alfworld")
    ap.add_argument("--episodes", type=int, default=0,
                    help="0 = all matching games")
    ap.add_argument("--pick-two", action="store_true")
    ap.add_argument("--all-ood", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out", default="runs/eval_cosplay_expand.jsonl")
    args = ap.parse_args()

    import openai
    import time
    from env_wrappers.alfworld_nl_wrapper import ALFWorldNLWrapper

    random.seed(args.seed)
    bank = (
        PROJECT_ROOT / "labeling" / "output" / "skillrl_seed_bank"
        / "alfworld" / "skill_bank.jsonl"
    )
    skills = load_skills(bank)
    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)

    # Deterministic order for full OOD; shuffle for sampled runs
    shuffle = not args.all_ood
    env = ALFWorldNLWrapper(
        split="eval_out_of_distribution",
        max_steps=args.max_steps,
        shuffle=shuffle,
        task_types=[6] if args.pick_two else None,
    )
    n_games = len(env._game_files)
    n = args.episodes if args.episodes > 0 else n_games
    if args.all_ood and not args.pick_two:
        n = n_games
    print(f"games_available={n_games} will_run={n} pick_two={args.pick_two} "
          f"all_ood={args.all_ood}", flush=True)

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    results = []
    ep = 0
    attempts = 0
    while len(results) < n and attempts < n_games * 3:
        attempts += 1
        obs, info = env.reset()
        task = env._goal
        if args.pick_two and not is_pick_two(task):
            continue
        ctl = Controller(task, skills)
        history = []
        cur_obs = obs
        raw_reward = 0.0
        success = False
        repairs = 0
        t0 = time.time()
        from collections import Counter
        selected = Counter()

        for _ in range(args.max_steps):
            admissible = info.get("action_names", [])
            if not admissible:
                break
            sid, block = ctl.skill_block(admissible)
            selected[sid] += 1
            prompt = build_prompt(task, block, history, cur_obs, admissible, window=2)
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=512,
            )
            candidate = parse_action(
                resp.choices[0].message.content or "", admissible
            )
            # parse_action in eval_skillrl_native takes (reply, admissible)
            # but our controller's repair needs the raw string — check signature
            action, repaired = ctl.repair(candidate, admissible)
            repairs += int(repaired)
            next_obs, reward, terminated, truncated, info = env.step(action)
            raw_reward += float(info.get("raw_env_reward", reward))
            ctl.update(action, next_obs)
            history.append((cur_obs, action))
            cur_obs = next_obs
            if terminated or truncated:
                success = raw_reward >= 1.0
                break

        rec = {
            "episode": ep,
            "task": task,
            "success": success,
            "steps": len(history),
            "repairs": repairs,
            "selected_skills": dict(selected),
            "wall_s": round(time.time() - t0, 1),
        }
        results.append(rec)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        wins = sum(r["success"] for r in results)
        print(
            f"[{ep}] {'SUCCESS' if success else 'fail'} "
            f"steps={len(history)} repairs={repairs} "
            f"running={wins}/{len(results)} ({wins/len(results):.0%}) "
            f"task={task[:55]}",
            flush=True,
        )
        ep += 1

    wins = sum(r["success"] for r in results)
    print(f"FINAL: {wins}/{len(results)} = {wins/len(results):.1%}")


if __name__ == "__main__":
    main()
