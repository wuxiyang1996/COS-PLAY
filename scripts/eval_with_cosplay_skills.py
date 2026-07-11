#!/usr/bin/env python3
"""Evaluate the RL model using COS-PLAY extracted skills vs their original skills.

Converts our skill_bank.jsonl into the "## Retrieved Relevant Experience"
block format matching SkillRL's ALFWORLD_TEMPLATE_WITH_MEMORY.

Usage:
    # With COS-PLAY skills
    python scripts/eval_with_cosplay_skills.py \
        --skill-bank labeling/output/cosplay_skillbank_v1/alfworld/skill_bank.jsonl \
        --episodes 30 --split eval_out_of_distribution

    # With their original skills (baseline)
    python scripts/eval_with_cosplay_skills.py \
        --their-skills --episodes 30 --split eval_out_of_distribution
"""

from __future__ import annotations

import argparse
import json
import random
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


def load_cosplay_skills(bank_path: str) -> dict:
    """Load COS-PLAY skill bank and format per task-type blocks.

    Maps our skill phases (search/acquire/transform/deliver) to ALFWorld
    task types using a heuristic: all skills are general enough that we
    include them all, but we can prefix with task-type-specific ordering.
    """
    skills = []
    with open(bank_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sk = entry.get("skill", entry)
            skills.append(sk)

    # Build the "## Retrieved Relevant Experience" block
    # Group by phase prefix for better organization
    phases = {}
    for sk in skills:
        sid = sk.get("skill_id", "")
        phase = sid.split(":")[0] if ":" in sid else "general"
        phases.setdefault(phase, []).append(sk)

    # Format in the same style as SkillRL's format_for_prompt
    def format_block(skill_list: list) -> str:
        sections = []

        # General principles (all skills)
        lines = ["### General Principles"]
        for sk in skill_list:
            name = sk.get("name", sk.get("skill_id", ""))
            desc = sk.get("strategic_description", sk.get("description", ""))
            lines.append(f"- **{name}**: {desc}")
        sections.append("\n".join(lines))

        # Execution hints as "Task-Specific Skills"
        task_lines = ["### Task-Specific Skills"]
        for sk in skill_list:
            name = sk.get("name", sk.get("skill_id", ""))
            hint = ""
            eh = sk.get("execution_hint", {})
            if isinstance(eh, dict):
                hint = eh.get("execution_description", "")
            if hint:
                task_lines.append(f"- **{name}**: {hint}")
        if len(task_lines) > 1:
            sections.append("\n".join(task_lines))

        # Failure modes as "Mistakes to Avoid"
        mistakes = ["### Mistakes to Avoid"]
        for sk in skill_list:
            eh = sk.get("execution_hint", {})
            if isinstance(eh, dict):
                for fm in eh.get("common_failure_modes", []):
                    mistakes.append(f"- **Don't**: {fm}")
        if len(mistakes) > 1:
            sections.append("\n".join(mistakes))

        return "\n\n".join(sections)

    # For each task type, include all skills (they're all relevant for ALFWorld)
    block = format_block(skills)

    # Return same block for all task types
    return {
        "pick_and_place": block,
        "clean": block,
        "heat": block,
        "cool": block,
        "look_at_obj_in_light": block,
        "examine": block,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--model", default="skillrl-alfworld")
    ap.add_argument("--skill-bank", default=None,
                    help="Path to COS-PLAY skill_bank.jsonl")
    ap.add_argument("--their-skills", action="store_true",
                    help="Use SkillRL's original skill blocks (baseline)")
    ap.add_argument("--no-skills", action="store_true",
                    help="No skill block at all (ablation)")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--split", default="eval_out_of_distribution")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--history-window", type=int, default=2)
    ap.add_argument(
        "--deterministic-order", action="store_true",
        help="Disable game-file shuffling for paired evaluations",
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import openai
    from env_wrappers.alfworld_nl_wrapper import ALFWorldNLWrapper

    # Load skill blocks
    if args.their_skills:
        blocks_path = (
            PROJECT_ROOT / "labeling" / "output" / "skillrl_native"
            / "rl_skill_blocks.json"
        )
        blocks = json.loads(blocks_path.read_text())
        skill_label = "skillrl_original"
    elif args.no_skills:
        blocks = {}
        skill_label = "no_skills"
    elif args.skill_bank:
        blocks = load_cosplay_skills(args.skill_bank)
        skill_label = "cosplay"
    else:
        print("ERROR: specify --skill-bank, --their-skills, or --no-skills")
        sys.exit(1)

    out_path = args.out or f"runs/eval_{skill_label}_{args.split}.jsonl"
    out_path = PROJECT_ROOT / out_path

    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)
    random.seed(args.seed)
    env = ALFWorldNLWrapper(
        split=args.split,
        max_steps=args.max_steps,
        shuffle=not args.deterministic_order,
    )

    print(f"Skill mode: {skill_label}")
    if blocks:
        sample_key = list(blocks.keys())[0]
        print(f"Sample block ({sample_key}): {len(blocks[sample_key])} chars")
    print(f"Evaluating {args.episodes} episodes on {args.split}")
    print()

    results = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    for ep in range(args.episodes):
        obs, info = env.reset()
        task = env._goal
        section = pick_rl_section(task)
        exp_block = blocks.get(section, "")
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
            "skill_mode": skill_label,
        }
        results.append(rec)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n_s = sum(r["success"] for r in results)
        print(
            f"[ep {ep}] {'SUCCESS' if success else 'fail'} | "
            f"type={section} | task={task[:50]} | "
            f"steps={len(history)} | running={n_s}/{len(results)} "
            f"({n_s/len(results)*100:.0f}%)",
            flush=True,
        )

    n_s = sum(r["success"] for r in results)
    print(f"\nFINAL [{skill_label}]: {n_s}/{len(results)} = "
          f"{n_s/len(results)*100:.1f}% success")


if __name__ == "__main__":
    main()
