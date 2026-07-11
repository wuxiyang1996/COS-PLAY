#!/usr/bin/env python3
"""Evaluate Jianwen/Alfworld-7B-SFT in OUR ALFWorld env with THEIR exact rollout format.

Faithful reproduction of SkillRL's agent_system/environments (verl-agent):
  * Step 0 uses ALFWORLD_TEMPLATE_NO_HIS — no task line, no experience block.
  * Later steps use ALFWORLD_TEMPLATE_WITH_MEMORY with the experience block.
  * history_length = 2 (ppo_trainer.yaml default), absolute step numbering:
      [Observation 3: '...', Action 3: '...']
  * Admissible actions are single-quoted, newline-separated, 'help' excluded.
  * Observations passed raw (banner + task line included), exactly as their
    env returns them.
  * Validation sampling: temperature 0.4, do_sample (their val_kwargs).

Usage:
    python scripts/eval_skillrl_native.py --episodes 20 --split eval_out_of_distribution
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

EXPERIENCE_BLOCKS_PATH = (
    PROJECT_ROOT / "labeling" / "output" / "skillrl_native" / "experience_blocks.json"
)

TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

TEMPLATE_WITH_MEMORY = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

## Retrieved Relevant Experience

{retrieved_memories}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def pick_section(task: str) -> str:
    t = task.lower()
    if t.startswith("examine") or "examine" in t.split(" and ")[0]:
        return "Examine"
    if t.startswith("look at") or "under the" in t:
        return "Look At Obj In Light"
    if "clean" in t:
        return "Clean"
    if "heat" in t or "hot " in t:
        return "Heat"
    if "cool" in t or "cold " in t:
        return "Cool"
    return "Pick And Place"


def pick_rl_section(task: str) -> str:
    """SkillRL SkillsOnlyMemory._detect_task_type (ALFWorld branch)."""
    goal = task.lower()
    if "look at" in goal and "under" in goal:
        return "look_at_obj_in_light"
    if "clean" in goal:
        return "clean"
    if "heat" in goal:
        return "heat"
    if "cool" in goal:
        return "cool"
    if "examine" in goal or "find" in goal:
        return "examine"
    return "pick_and_place"


def format_admissible(admissible: list) -> str:
    return "\n ".join(f"'{s}'" for s in admissible if s != "help")


def format_history(history: list, window: int) -> tuple:
    """SkillRL memory.fetch: absolute step numbers, window of `window`."""
    recent = history[-window:]
    start_idx = len(history) - len(recent)
    lines = [
        f"[Observation {start_idx + j + 1}: '{o}', Action {start_idx + j + 1}: '{a}']"
        for j, (o, a) in enumerate(recent)
    ]
    return "\n".join(lines), len(recent)


def build_prompt(task, exp_block, history, cur_obs, admissible, window=2):
    if not history:
        return TEMPLATE_NO_HIS.format(
            current_observation=cur_obs,
            admissible_actions=format_admissible(admissible),
        )
    action_history, valid_len = format_history(history, window)
    # exp_block includes the "## Retrieved Relevant Experience" heading; strip it
    mem = re.sub(r"^## Retrieved Relevant Experience\s*", "", exp_block.strip())
    return TEMPLATE_WITH_MEMORY.format(
        task_description=task,
        retrieved_memories=mem,
        step_count=len(history),
        history_length=valid_len,
        action_history=action_history,
        current_step=len(history) + 1,
        current_observation=cur_obs,
        admissible_actions=format_admissible(admissible),
    )


def parse_action(reply: str, admissible: list) -> str:
    """SkillRL's projection: extract <action> content and send it RAW.

    Their alfworld_projection does no admissible-matching — TextWorld
    itself answers "Nothing happens." for unparseable commands and the
    agent self-corrects from history. Force-mapping to an admissible
    action (the previous behaviour here) corrupts the loop.
    """
    m = re.search(r"<action>\s*(.*?)\s*</action>", reply, re.DOTALL)
    if m:
        return m.group(1).strip().lower()
    return reply[-30:].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--model", default="skillrl-alfworld")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--split", default="eval_out_of_distribution")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--history-window", type=int, default=2)
    ap.add_argument("--no-experience", action="store_true",
                    help="Use plain ALFWORLD_TEMPLATE without the experience block")
    ap.add_argument("--rl-blocks", action="store_true",
                    help="Use the RL-time skill blocks (claude_style_skills.json "
                         "via SkillsOnlyMemory) instead of the SFT-data blocks")
    ap.add_argument("--out", default="runs/skillrl_native_eval.jsonl")
    args = ap.parse_args()

    import openai

    from env_wrappers.alfworld_nl_wrapper import ALFWorldNLWrapper

    if args.rl_blocks:
        blocks_path = EXPERIENCE_BLOCKS_PATH.parent / "rl_skill_blocks.json"
    else:
        blocks_path = EXPERIENCE_BLOCKS_PATH
    blocks = json.loads(blocks_path.read_text())
    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)

    env = ALFWorldNLWrapper(split=args.split, max_steps=args.max_steps, shuffle=True)

    results = []
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        obs, info = env.reset()
        task = env._goal
        if args.rl_blocks:
            section = pick_rl_section(task)
            exp_block = blocks.get(section) or blocks["pick_and_place"]
        else:
            section = pick_section(task)
            exp_block = blocks.get(section) or blocks["Pick And Place"]
        if args.no_experience:
            exp_block = ""
        history = []
        cur_obs = obs  # raw, as their env passes it
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
                break
            action = parse_action(reply, admissible)
            obs, reward, terminated, truncated, info = env.step(action)
            raw = info.get("raw_env_reward", reward)
            raw_reward += float(raw)
            history.append((cur_obs, action))
            cur_obs = obs
            if terminated or truncated:
                success = raw_reward >= 1.0
                break

        rec = {
            "episode": ep,
            "task": task,
            "section": section,
            "steps": len(history),
            "success": success,
            "raw_reward": raw_reward,
            "wall_s": round(time.time() - t0, 1),
        }
        results.append(rec)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n_s = sum(r["success"] for r in results)
        print(
            f"[ep {ep}] {'SUCCESS' if success else 'fail'} | task={task[:50]} | "
            f"steps={len(history)} | running={n_s}/{len(results)} "
            f"({n_s/len(results)*100:.0f}%)",
            flush=True,
        )

    n_s = sum(r["success"] for r in results)
    print(f"\nFINAL: {n_s}/{len(results)} = {n_s/len(results)*100:.1f}% success")


if __name__ == "__main__":
    main()
