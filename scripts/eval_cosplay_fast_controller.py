#!/usr/bin/env python3
"""Fast COS-PLAY evaluation on ALFWorld with a frozen SkillRL RL policy.

This is an inference-time COS-PLAY controller:
  * selects one task/phase-relevant skill from the six-skill seed bank;
  * injects only that protocol plus a failure-derived recovery skill;
  * keeps a persistent searched/failed-action ledger;
  * grounds generated actions to the admissible set and vetoes no-effect loops.

No evaluation trajectory is used to update the skill bank or model weights.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_skillrl_native import build_prompt, parse_action


RECOVERY_SKILL = """\
### Active Recovery Skill
- **escape_no_effect_loop**: If an action returns "Nothing happens", mark that
  exact action invalid at the current state and never repeat it there. Choose a
  different admissible action that advances the active skill. After two
  repeated actions without progress, switch to an untried location or
  interaction immediately."""


def load_skills(path: Path) -> dict[str, dict]:
    skills = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        skill = entry.get("skill", entry)
        skills[skill["skill_id"]] = skill
    return skills


def task_spec(task: str) -> dict:
    """Parse only task semantics needed for skill routing."""
    low = task.lower().rstrip(".")
    transform = None
    if re.search(r"\b(?:heat|hot)\b", low):
        transform = "heat"
    elif re.search(r"\b(?:cool|cold)\b", low):
        transform = "cool"
    elif re.search(r"\bclean\b", low):
        transform = "clean"

    multiple = bool(re.search(r"\b(?:find|put)\s+two\b", low))
    lamp = "desklamp" in low or ("look at" in low and "under" in low)

    destination = ""
    matches = re.findall(r"\b(?:in|on|under|with)\s+(?:the\s+)?([a-z]+)", low)
    if matches:
        destination = matches[-1]

    # Match the object noun against take-actions later; keep candidate nouns.
    stop = {
        "put", "some", "a", "an", "the", "clean", "cool", "cold", "hot",
        "heat", "find", "two", "look", "at", "examine", "and", "it", "them",
        "in", "on", "under", "with", destination, "desklamp",
    }
    nouns = [w for w in re.findall(r"[a-z]+", low) if w not in stop]
    target = nouns[0] if nouns else ""
    return {
        "transform": transform,
        "multiple": multiple,
        "lamp": lamp,
        "destination": destination,
        "target": target,
    }


def format_skill(skill: dict, progress: str) -> str:
    protocol = skill.get("protocol", {})
    steps = protocol.get("steps", [])
    success = protocol.get("success_criteria", [])
    lines = [
        "### Active COS-PLAY Skill",
        f"- **{skill.get('name', skill.get('skill_id'))}**: "
        f"{skill.get('strategic_description', '')}",
    ]
    if steps:
        lines.append("  Protocol: " + " → ".join(str(x) for x in steps))
    if success:
        lines.append("  Success: " + "; ".join(str(x) for x in success))
    lines.extend(["", RECOVERY_SKILL, "", "### Persistent Progress Ledger", progress])
    return "\n".join(lines)


class Controller:
    def __init__(self, task: str, skills: dict[str, dict]):
        self.task = task
        self.spec = task_spec(task)
        self.skills = skills
        self.actions: list[str] = []
        self.failed: set[str] = set()
        self.visited: set[str] = set()
        self.holding = False
        self.transformed = False
        self.delivered = 0

    def update(self, action: str, obs: str) -> None:
        low = obs.lower()
        self.actions.append(action)
        if "nothing happens" in low:
            self.failed.add(action)
        if action.startswith("go to ") and "nothing happens" not in low:
            self.visited.add(action.removeprefix("go to "))
        if action.startswith(("take ", "pick up ")) and (
            "you pick up" in low or "you take" in low
        ):
            self.holding = True
        if action.startswith(("clean ", "heat ", "cool ")) and (
            "nothing happens" not in low
        ):
            self.transformed = True
        if action.startswith("put ") and "you put" in low:
            self.holding = False
            self.delivered += 1

    def skill_id(self, admissible: list[str]) -> str:
        target = self.spec["target"]
        if not self.holding and any(
            a.startswith("take ") and target in a for a in admissible
        ):
            return "acquire:COLLECT"
        if self.holding:
            if self.spec["lamp"]:
                return "deliver:EXECUTE"
            if self.spec["transform"] and not self.transformed:
                return "transform:EXECUTE"
            return "deliver:POSITION"
        if self.spec["multiple"] and self.delivered:
            return "search:NAVIGATE"
        return "search:EXPLORE"

    def progress(self) -> str:
        recent = self.actions[-6:]
        return (
            f"- holding_target={self.holding}; transformed={self.transformed}; "
            f"delivered_count={self.delivered}\n"
            f"- visited_locations={sorted(self.visited)}\n"
            f"- failed_actions_do_not_repeat={sorted(self.failed)}\n"
            f"- recent_actions={recent}"
        )

    def skill_block(self, admissible: list[str]) -> tuple[str, str]:
        sid = self.skill_id(admissible)
        return sid, format_skill(self.skills[sid], self.progress())

    def _phase_choice(self, admissible: list[str]) -> str | None:
        available = [a for a in admissible if a not in self.failed]
        if not available:
            available = list(admissible)
        target = self.spec["target"]
        destination = self.spec["destination"]

        take = [a for a in available if a.startswith("take ") and target in a]
        if not self.holding and take:
            return take[0]

        if self.holding and self.spec["lamp"]:
            use = [a for a in available if "desklamp" in a and a.startswith(("use ", "toggle "))]
            if use:
                return use[0]
            nav = [a for a in available if a.startswith("go to ") and any(
                x in a for x in ("desk", "sidetable")
            )]
            if nav:
                return nav[0]

        if self.holding and self.spec["transform"] and not self.transformed:
            verb = self.spec["transform"]
            direct = [a for a in available if a.startswith(f"{verb} ")]
            if direct:
                return direct[0]
            appliance = {"heat": "microwave", "cool": "fridge", "clean": "sinkbasin"}[verb]
            interact = [a for a in available if appliance in a and a.startswith("open ")]
            if interact:
                return interact[0]
            nav = [a for a in available if appliance in a and a.startswith("go to ")]
            if nav:
                return nav[0]

        ready = self.holding and (
            not self.spec["transform"] or self.transformed
        )
        if ready:
            put = [a for a in available if a.startswith("put ") and destination in a]
            if put:
                return put[0]
            open_dest = [a for a in available if a.startswith("open ") and destination in a]
            if open_dest:
                return open_dest[0]
            nav = [a for a in available if a.startswith("go to ") and destination in a]
            if nav:
                return nav[0]

        # Recovery search: prefer unseen locations, then unopened containers.
        nav = [
            a for a in available
            if a.startswith("go to ") and a.removeprefix("go to ") not in self.visited
        ]
        if nav:
            return nav[0]
        interact = [a for a in available if a.startswith(("open ", "examine "))]
        if interact:
            return interact[0]
        return available[0] if available else None

    def repair(self, candidate: str, admissible: list[str]) -> tuple[str, bool]:
        counts = Counter(self.actions[-6:])
        valid = candidate in admissible
        repeated = counts[candidate] >= 2
        blocked = candidate in self.failed
        if valid and not repeated and not blocked:
            return candidate, False

        # Preserve a near-exact generated command when only casing/spacing differs.
        if not blocked and not repeated and candidate:
            best = max(
                admissible,
                key=lambda a: SequenceMatcher(None, candidate.lower(), a.lower()).ratio(),
                default="",
            )
            if best and SequenceMatcher(None, candidate.lower(), best.lower()).ratio() >= 0.9:
                return best, best != candidate

        fallback = self._phase_choice(admissible)
        return (fallback or candidate), True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--model", default="skillrl-alfworld")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--split", default="eval_out_of_distribution")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="runs/eval_cosplay_fast.jsonl")
    args = ap.parse_args()

    import openai
    from env_wrappers.alfworld_nl_wrapper import ALFWorldNLWrapper

    bank = (
        PROJECT_ROOT / "labeling" / "output" / "skillrl_seed_bank"
        / "alfworld" / "skill_bank.jsonl"
    )
    skills = load_skills(bank)
    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)
    random.seed(args.seed)
    env = ALFWorldNLWrapper(
        split=args.split, max_steps=args.max_steps, shuffle=args.shuffle
    )
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    results = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        task = env._goal
        ctl = Controller(task, skills)
        history = []
        cur_obs = obs
        raw_reward = 0.0
        success = False
        repairs = 0
        selected = Counter()
        t0 = time.time()

        for _step in range(args.max_steps):
            admissible = info.get("action_names", [])
            if not admissible:
                break
            sid, block = ctl.skill_block(admissible)
            selected[sid] += 1
            prompt = build_prompt(task, block, history, cur_obs, admissible, window=2)
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=512,
            )
            candidate = parse_action(response.choices[0].message.content or "", admissible)
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

    wins = sum(r["success"] for r in results)
    print(f"FINAL: {wins}/{len(results)} = {wins/len(results):.1%}")


if __name__ == "__main__":
    main()
