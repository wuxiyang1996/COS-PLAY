#!/usr/bin/env python3
"""Evaluate SkillRL and SkillRL+COS-PLAY on SkillRL's native WebShop protocol.

This harness intentionally matches ``examples/grpo_trainer/run_webshop_skills.sh``:

* SkillRL's bundled WebShop environment and Pyserini/Lucene search index;
* 1,000-product catalog with synthetic goals (``use_small=True``);
* validation seed 1000, 64 workers/tasks sampled without replacement from
  validation goal indices 0..499;
* native ``SkillsOnlyMemory`` template retrieval (top_k=6);
* native prompt templates and action projection;
* 15-step limit, temperature 0.4, and official ``won`` / ``task_score`` metrics.

The environments are replayed sequentially. Episode ``i`` uses the same worker
seed and goal index as SkillRL's 64-worker validation vector environment:
worker_seed=1000+i and goal_idx=RandomState(1000).choice(0..499)[i].
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLRL_ROOT = Path("/workspace/SkillRL")
WEBSHOP_PACKAGE = (
    SKILLRL_ROOT
    / "agent_system"
    / "environments"
    / "env_package"
    / "webshop"
    / "webshop"
)
PRODUCT_FILE = Path("/workspace/WebShop/data/items_shuffle_1000.json")
ATTR_FILE = Path("/workspace/WebShop/data/items_ins_v2_1000.json")
HUMAN_ATTR_FILE = Path("/workspace/WebShop/data/items_human_ins.json")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SKILLRL_ROOT))
sys.path.insert(0, str(WEBSHOP_PACKAGE))

from agent_system.environments.prompts.webshop import (  # noqa: E402
    WEBSHOP_TEMPLATE_NO_HIS,
    WEBSHOP_TEMPLATE_WITH_MEMORY,
)
from agent_system.memory.skills_only_memory import SkillsOnlyMemory  # noqa: E402
from scripts.eval_webshop_cosplay_fast import (  # noqa: E402
    Controller,
    load_skills,
)

# Import the official projection file directly. Importing its parent package
# executes ``webshop/__init__.py``, which unnecessarily requires Ray even for
# this sequential evaluator.
_projection_path = (
    SKILLRL_ROOT
    / "agent_system/environments/env_package/webshop/projection.py"
)
_projection_spec = importlib.util.spec_from_file_location(
    "skillrl_webshop_projection",
    _projection_path,
)
if _projection_spec is None or _projection_spec.loader is None:
    raise ImportError(f"Cannot load SkillRL projection from {_projection_path}")
_projection_module = importlib.util.module_from_spec(_projection_spec)
_projection_spec.loader.exec_module(_projection_module)
webshop_projection = _projection_module.webshop_projection


def validation_schedule(episodes: int, seed: int = 1000) -> list[tuple[int, int]]:
    """Return exact ``(worker_seed, goal_idx)`` pairs used by SkillRL val envs."""
    if episodes > 500:
        raise ValueError("Official validation pool contains only 500 goal indices")
    idxs = np.random.RandomState(seed).choice(
        np.arange(500), size=episodes, replace=False
    )
    return [(seed + i, int(idx)) for i, idx in enumerate(idxs)]


def make_env(worker_seed: int):
    """Construct one native SkillRL WebShop environment with Lucene search."""
    import web_agent_site.engine.engine as engine
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    # The released SkillRL checkout omits data/. Point its module-level human
    # attribute constant at the matching file in the installed WebShop copy.
    engine.HUMAN_ATTR_PATH = str(HUMAN_ATTR_FILE)
    return WebAgentTextEnv(
        observation_mode="text",
        file_path=str(PRODUCT_FILE),
        attr_path=str(ATTR_FILE),
        num_products=None,  # exact run_webshop_skills.sh / use_small=True behavior
        human_goals=False,
        seed=worker_seed,
    )


def clean_task(raw_obs: str) -> str:
    parts = raw_obs.split(" [SEP] ")
    if len(parts) >= 3 and parts[1] == "Instruction:":
        return parts[2]
    raise ValueError(f"Unexpected native WebShop reset observation: {raw_obs[:200]}")


def format_obs(raw_obs: str, task: str) -> str:
    """Match ``WebshopEnvironmentManager.format_obs`` exactly."""
    parts = raw_obs.split(" [SEP] ")
    try:
        index = parts.index(task)
        return " [SEP] ".join(f"'{part}'" for part in parts[index + 1 :])
    except ValueError:
        return raw_obs


def action_names(env) -> list[str]:
    """Match ``WebshopEnvironmentManager.format_avail_actions``."""
    available = env.get_available_actions()
    actions: list[str] = []
    if available["has_search_bar"]:
        actions.append("search[<your query>]")
    actions.extend(f"click[{text}]" for text in available["clickables"])
    return actions


def format_actions(actions: list[str]) -> str:
    return "\n".join(f"'{action}'," for action in actions)


def format_history(history: list[tuple[str, str]], length: int = 2) -> tuple[str, int]:
    recent = history[-length:]
    start = len(history) - len(recent)
    lines = [
        f"[Observation {start + j + 1}: '{obs}', "
        f"Action {start + j + 1}: '{action}']"
        for j, (obs, action) in enumerate(recent)
    ]
    return "\n".join(lines), len(recent)


def build_prompt(
    task: str,
    current_obs: str,
    actions: list[str],
    history: list[tuple[str, str]],
    memory_block: str,
) -> str:
    formatted_actions = format_actions(actions)
    if not history:
        return WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task,
            current_observation=current_obs,
            available_actions=formatted_actions,
        )

    action_history, valid_length = format_history(history, length=2)
    return WEBSHOP_TEMPLATE_WITH_MEMORY.format(
        task_description=task,
        retrieved_memories=memory_block,
        step_count=len(history),
        history_length=valid_length,
        action_history=action_history,
        current_step=len(history) + 1,
        current_observation=current_obs,
        available_actions=formatted_actions,
    )


def native_action(reply: str) -> tuple[str, bool]:
    actions, valids = webshop_projection([reply])
    return actions[0], bool(valids[0])


class GuardController:
    """Minimal action-level guardrails over the frozen SkillRL policy.

    Unlike the full COS-PLAY controller, this never rewrites queries and never
    touches an admissible, non-looping action. It only intervenes on the two
    failure modes observed in the official-protocol baseline:
      1. no-progress loops (re-issuing an action that changed nothing);
      2. clicking Buy Now while a task-required option value is visibly
         available on the current product page but not yet selected.
    """

    # WebShop ASINs always start with "B0"; a bare 10-char alphanumeric match
    # would also catch option values like "cargoo5209".
    _ASIN_RE = re.compile(r"^click\[(b0[a-z0-9]{8})\]$", re.IGNORECASE)

    def __init__(self, task: str):
        self.required_values = self._parse_required_values(task)
        self.actions: list[str] = []
        self.no_effect: set[str] = set()
        self.clicked_values: set[str] = set()
        self.visited_asins: set[str] = set()
        self.current_asin: str | None = None
        self.last_obs: str | None = None

    @staticmethod
    def _parse_required_values(task: str) -> list[str]:
        values = []
        for match in re.finditer(
            r"(?:color|size|fit type|item shape|style name|scent|flavor name|"
            r"pattern name|team name):\s*([^,]+?)(?=, and |, And |$)",
            task,
            re.IGNORECASE,
        ):
            value = match.group(1).strip().lower().rstrip(".")
            if value:
                values.append(value)
        return values

    def update(self, action: str, obs: str) -> None:
        if self.last_obs is not None and obs.strip() == self.last_obs.strip():
            self.no_effect.add(action)
        self.last_obs = obs
        self.actions.append(action)

        asin_match = self._ASIN_RE.match(action)
        if asin_match:
            self.current_asin = asin_match.group(1).lower()
            self.visited_asins.add(self.current_asin)
            # Option selections do not carry over between product pages.
            self.clicked_values = set()
        elif action.startswith("search[") or "back to search" in action.lower():
            self.current_asin = None
            self.clicked_values = set()
        elif action.startswith("click["):
            self.clicked_values.add(action[6:-1].strip().lower())

    def _untried_product(self, admissible: list[str]) -> str | None:
        for action in admissible:
            asin_match = self._ASIN_RE.match(action)
            if asin_match and asin_match.group(1).lower() not in self.visited_asins:
                return action
        return None

    def repair(self, candidate: str, admissible: list[str]) -> tuple[str, bool]:
        lowered = candidate.lower().strip()

        # Guard 3: before Buy Now, select any required option that is shown
        # on the page but has not been clicked yet.
        if "buy now" in lowered:
            for value in self.required_values:
                option_action = f"click[{value}]"
                if value not in self.clicked_values and any(
                    a.lower() == option_action for a in admissible
                ):
                    return option_action, True
            return candidate, False

        # Guard 1: break no-progress loops (exact same action that previously
        # changed nothing, or a third consecutive repeat).
        recent = self.actions[-4:]
        looping = candidate in self.no_effect or recent.count(candidate) >= 2
        if looping:
            # If every required option is already selected, finishing the
            # purchase beats any form of wandering.
            buy = next((a for a in admissible if "buy now" in a.lower()), None)
            if buy and all(v in self.clicked_values for v in self.required_values):
                return buy, True
            alternative = self._untried_product(admissible)
            if alternative:
                return alternative, True
            back = next(
                (a for a in admissible if "back to search" in a.lower()), None
            )
            if back and candidate != back and back not in self.no_effect:
                return back, True

        return candidate, False


def run_episode(
    *,
    client,
    model: str,
    mode: str,
    worker_seed: int,
    goal_idx: int,
    max_steps: int,
    temperature: float,
    native_memory: SkillsOnlyMemory,
    cosplay_skills: dict[str, dict],
) -> dict[str, Any]:
    env = make_env(worker_seed)
    raw_obs, _ = env.reset(session=goal_idx)
    task = clean_task(raw_obs)
    current_obs = format_obs(raw_obs, task)
    history: list[tuple[str, str]] = []
    trajectory: list[dict[str, Any]] = []
    valid_actions = 0
    repairs = 0
    controller = Controller(task, cosplay_skills) if mode == "cosplay" else None
    guard = GuardController(task) if mode == "guard" else None

    retrieved = native_memory.retrieve(task_description=task, top_k=6)
    native_block = native_memory.format_for_prompt(retrieved)
    done = False
    won = False
    task_score = 0.0
    started = time.time()

    for step in range(max_steps):
        available = action_names(env)
        if mode == "cosplay":
            _, cosplay_block = controller.skill_block()
            # COS-PLAY augments the released SkillRL policy and its native
            # memory; it must not replace the memory that the RL policy was
            # trained with.
            memory_block = (
                native_block
                + "\n\n### COS-PLAY Active Controller\n"
                + cosplay_block
            )
        else:
            memory_block = native_block

        prompt = build_prompt(
            task=task,
            current_obs=current_obs,
            actions=available,
            history=history,
            memory_block=memory_block,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=768,
        )
        reply = response.choices[0].message.content or ""
        candidate, valid = native_action(reply)
        valid_actions += int(valid)

        if mode == "cosplay":
            action, repaired = controller.repair(candidate, available)
            repairs += int(repaired)
        elif mode == "guard":
            # Prompt stays fully native; guardrails act on actions only.
            action, repaired = guard.repair(candidate, available)
            repairs += int(repaired)
        else:
            # Native SkillRL executes the projected text directly. Invalid or
            # unavailable actions are handled by WebAgentTextEnv as no-ops.
            action = candidate

        next_raw_obs, reward, done, info = env.step(action)
        task_score = float(reward)
        won = bool(done and reward == 1.0)
        next_obs = format_obs(next_raw_obs, task)
        next_available = action_names(env) if not done else []

        trajectory.append(
            {
                "step": step + 1,
                "action": action,
                "projected_valid": valid,
                "repaired": mode in ("cosplay", "guard") and action != candidate,
                "reward": float(reward),
                "done": bool(done),
            }
        )
        history.append((current_obs, action))
        current_obs = next_obs
        if controller is not None:
            controller.update(action, next_obs, next_available)
        if guard is not None:
            guard.update(action, next_obs)
        if done:
            break

    record = {
        "worker_seed": worker_seed,
        "goal_idx": goal_idx,
        "task": task,
        "mode": mode,
        "won": won,
        "task_score": task_score,
        "success": won,
        "reward": task_score,
        "steps": len(trajectory),
        "valid_action_rate": valid_actions / max(len(trajectory), 1),
        "repairs": repairs,
        "retrieved_task_type": retrieved.get("task_type"),
        "wall_s": round(time.time() - started, 2),
        "trajectory": trajectory,
    }
    env.close()
    del env
    gc.collect()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["skillrl", "cosplay", "guard"], required=True
    )
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--validation-seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--model", default="skillrl-webshop")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if os.environ.get("JAVA_HOME") is None:
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

    import openai

    native_memory = SkillsOnlyMemory(
        skills_json_path=str(
            SKILLRL_ROOT / "memory_data/webshop/claude_style_skills.json"
        ),
        retrieval_mode="template",
    )
    cosplay_skills = load_skills(
        PROJECT_ROOT
        / "labeling/output/skillrl_seed_bank/webshop/skill_bank.jsonl"
    )
    client = openai.OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
        timeout=300,
    )
    schedule = validation_schedule(args.episodes, args.validation_seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    rows: list[dict[str, Any]] = []
    for episode, (worker_seed, goal_idx) in enumerate(schedule):
        row = run_episode(
            client=client,
            model=args.model,
            mode=args.mode,
            worker_seed=worker_seed,
            goal_idx=goal_idx,
            max_steps=args.max_steps,
            temperature=args.temperature,
            native_memory=native_memory,
            cosplay_skills=cosplay_skills,
        )
        row["episode"] = episode
        rows.append(row)
        with out.open("a") as file:
            file.write(json.dumps(row) + "\n")

        wins = sum(r["won"] for r in rows)
        mean_score = sum(r["task_score"] for r in rows) / len(rows)
        print(
            f"[{episode:02d}] {'WIN' if row['won'] else 'fail'} "
            f"score={row['task_score']:.3f} steps={row['steps']} "
            f"running_succ={wins / len(rows):.1%} "
            f"mean_score={mean_score:.3f} "
            f"goal={row['goal_idx']} task={row['task'][:55]}",
            flush=True,
        )

    wins = sum(r["won"] for r in rows)
    mean_score = sum(r["task_score"] for r in rows) / len(rows)
    print(
        f"FINAL [{args.mode}] n={len(rows)} "
        f"succ={wins / len(rows):.1%} ({wins}/{len(rows)}) "
        f"score={mean_score:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
