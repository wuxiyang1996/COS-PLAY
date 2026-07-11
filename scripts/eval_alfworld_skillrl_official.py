#!/usr/bin/env python3
"""Run SkillRL's native 64-worker ALFWorld ID validation protocol sequentially."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLRL_ROOT = Path("/workspace/SkillRL")
ALFWORLD_PACKAGE = (
    SKILLRL_ROOT / "agent_system/environments/env_package/alfworld"
)
CONFIG_PATH = ALFWORLD_PACKAGE / "configs/config_tw.yaml"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SKILLRL_ROOT))
sys.path.insert(0, str(ALFWORLD_PACKAGE))

from agent_system.environments.prompts.alfworld import (  # noqa: E402
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE_WITH_MEMORY,
)
from agent_system.memory.skills_only_memory import SkillsOnlyMemory  # noqa: E402
from scripts.eval_skillrl_native import (  # noqa: E402
    format_admissible,
    format_history,
    parse_action,
)
from scripts.eval_with_cosplay_skills import load_cosplay_skills  # noqa: E402


def extract_task(obs: str) -> str:
    marker = "Your task is to: "
    start = obs.find(marker)
    if start < 0:
        raise ValueError(f"Task not found in observation: {obs[:300]}")
    return obs[start + len(marker) :].strip()


def build_prompt(
    task: str,
    memory_block: str,
    history: list[tuple[str, str]],
    obs: str,
    admissible: list[str],
) -> str:
    if not history:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=obs,
            admissible_actions=format_admissible(admissible),
        )
    action_history, valid_length = format_history(history, 2)
    return ALFWORLD_TEMPLATE_WITH_MEMORY.format(
        task_description=task,
        retrieved_memories=memory_block,
        step_count=len(history),
        history_length=valid_length,
        action_history=action_history,
        current_step=len(history) + 1,
        current_observation=obs,
        admissible_actions=format_admissible(admissible),
    )


def run_episode(
    *,
    base_env,
    client,
    model: str,
    mode: str,
    worker_seed: int,
    max_steps: int,
    temperature: float,
    native_memory: SkillsOnlyMemory,
    cosplay_blocks: dict[str, str],
) -> dict[str, Any]:
    env = base_env.init_env(batch_size=1)
    env.seed(worker_seed)
    observations, infos = env.reset()
    obs = observations[0]
    task = extract_task(obs)
    native_retrieved = native_memory.retrieve(task_description=task, top_k=6)
    native_block = native_memory.format_for_prompt(native_retrieved)
    if mode == "cosplay":
        task_type = native_retrieved["task_type"]
        cosplay_block = cosplay_blocks.get(task_type, "")
        memory_block = (
            native_block
            + "\n\n### COS-PLAY Active Controller\n"
            + cosplay_block
        )
    else:
        memory_block = native_block

    history: list[tuple[str, str]] = []
    trajectory: list[dict[str, Any]] = []
    won = False
    gamefile = infos.get("extra.gamefile", [None])[0]
    started = time.time()

    for step in range(max_steps):
        admissible = infos["admissible_commands"][0]
        prompt = build_prompt(task, memory_block, history, obs, admissible)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=512,
        )
        reply = response.choices[0].message.content or ""
        action = parse_action(reply, admissible)
        next_observations, _, dones, next_infos = env.step([action])
        next_obs = next_observations[0]
        done = bool(dones[0])
        won = bool(next_infos["won"][0])
        trajectory.append(
            {
                "step": step + 1,
                "action": action,
                "done": done,
                "won": won,
            }
        )
        history.append((obs, action))
        obs = next_obs
        infos = next_infos
        if done:
            break

    record = {
        "worker_seed": worker_seed,
        "task": task,
        "mode": mode,
        "won": won,
        "success": won,
        "reward": 10.0 if won else 0.0,
        "steps": len(trajectory),
        "task_type": native_retrieved["task_type"],
        "gamefile": gamefile,
        "wall_s": round(time.time() - started, 2),
        "trajectory": trajectory,
    }
    env.close()
    del env
    gc.collect()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["skillrl", "cosplay"], required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--validation-seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", default="skillrl-alfworld")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ.setdefault("PYGLET_HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import openai
    from alfworld.agents.environment import get_environment

    with CONFIG_PATH.open() as file:
        config = yaml.safe_load(file)
    base_env = get_environment(config["env"]["type"])(
        config,
        train_eval="eval_in_distribution",
    )
    native_memory = SkillsOnlyMemory(
        skills_json_path=str(
            SKILLRL_ROOT / "memory_data/alfworld/claude_style_skills.json"
        ),
        retrieval_mode="template",
    )
    cosplay_blocks = load_cosplay_skills(
        str(
            PROJECT_ROOT
            / "labeling/output/skillrl_seed_bank/alfworld/skill_bank.jsonl"
        )
    )
    client = openai.OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
        timeout=300,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    rows: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        row = run_episode(
            base_env=base_env,
            client=client,
            model=args.model,
            mode=args.mode,
            worker_seed=args.validation_seed + episode,
            max_steps=args.max_steps,
            temperature=args.temperature,
            native_memory=native_memory,
            cosplay_blocks=cosplay_blocks,
        )
        row["episode"] = episode
        rows.append(row)
        with out.open("a") as file:
            file.write(json.dumps(row) + "\n")
        wins = sum(record["won"] for record in rows)
        print(
            f"[{episode:02d}] {'WIN' if row['won'] else 'fail'} "
            f"steps={row['steps']} running={wins / len(rows):.1%} "
            f"type={row['task_type']} task={row['task'][:60]}",
            flush=True,
        )

    wins = sum(record["won"] for record in rows)
    task_counts = {
        task_type: {
            "n": sum(row["task_type"] == task_type for row in rows),
            "won": sum(
                row["won"] for row in rows if row["task_type"] == task_type
            ),
        }
        for task_type in sorted({row["task_type"] for row in rows})
    }
    print(
        f"FINAL [{args.mode}] n={len(rows)} "
        f"succ={wins / len(rows):.1%} ({wins}/{len(rows)}) "
        f"by_type={task_counts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
