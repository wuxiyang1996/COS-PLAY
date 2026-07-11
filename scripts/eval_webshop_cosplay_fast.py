#!/usr/bin/env python3
"""Fast COS-PLAY evaluation on WebShop with a frozen SkillRL RL policy.

Mirrors the ALFWorld fast controller:
  * selects one phase-relevant skill from the WebShop seed bank;
  * injects only that protocol plus a failure-derived recovery skill;
  * keeps a persistent failed-action / visited-product ledger;
  * grounds generated actions and vetoes no-progress loops.

Usage:
    python scripts/eval_webshop_cosplay_fast.py \
        --base-url http://localhost:8011/v1 --model skillrl-webshop \
        --episodes 30 --out runs/eval_webshop_cosplay_30.jsonl
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

TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

TEMPLATE_WITH_MEMORY = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.

## Retrieved Relevant Experience

{retrieved_memories}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

RECOVERY_SKILL = """\
### Active Recovery Skills (failure-derived)
- **escape_no_progress_loop**: Never repeat an action that produced no page
  change. Do not re-issue the same search query. Prefer an untried product
  code, option, or Back to Search over looping.
- **veto_partial_buy**: NEVER click Buy Now if the price exceeds the budget
  OR a required color/size/fit option is still unselected. Leave the product
  (Back to Search / results) instead of buying a partial match.
- **refine_query_after_mismatch**: After two unsuitable products, go Back to
  Search and issue a NEW query that adds unused attributes (color, size,
  material, fit). Do not page endlessly."""


def load_skills(path: Path) -> dict[str, dict]:
    skills = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        skill = entry.get("skill", entry)
        skills[skill["skill_id"]] = skill
    return skills


def parse_goal(goal: str) -> dict:
    """Parse human and synthetic WebShop instructions.

    Synthetic form:
      Find me <product> with <attrs> with color: C, and size: S, and price lower than P
    Human form:
      i am looking for ... with ..., and price lower than P
    """
    g = goal.strip()
    g = re.sub(r"^instruction:\s*", "", g, flags=re.I).strip()

    price = None
    m = re.search(r"price lower than\s*([0-9.]+)", g, re.I)
    if m:
        price = float(m.group(1))

    color = None
    size = None
    fit = None
    m = re.search(r"\bcolor:\s*([^,]+?)(?:,|\band\b|$)", g, re.I)
    if m:
        color = m.group(1).strip()
    m = re.search(r"\bsize:\s*([^,]+?)(?:,|\band\b|price|$)", g, re.I)
    if m:
        size = m.group(1).strip()
    m = re.search(r"\bfit type:\s*([^,]+?)(?:,|\band\b|price|$)", g, re.I)
    if m:
        fit = m.group(1).strip()

    # Strip trailing color/size/price clause before extracting free attrs.
    body = re.split(
        r"\bwith color:|\band size:|\band fit type:|\band price lower than\b",
        g,
        maxsplit=1,
        flags=re.I,
    )[0]

    attrs: list[str] = []
    m = re.search(r"\bwith\b(.+)$", body, re.I)
    if m:
        attrs = [
            a.strip(" .")
            for a in re.split(r",|\band\b", m.group(1))
            if a.strip(" .")
        ]

    head = re.split(r"\bwith\b|\band price\b", body, maxsplit=1, flags=re.I)[0]
    head = re.sub(
        r"^(?:find me|i(?:'m| am) looking for|i want(?: to buy)?|i need|"
        r"place order for|show me)\s+",
        "",
        head,
        flags=re.I,
    ).strip(" .")

    required_options = [x for x in (color, size, fit) if x]
    return {
        "raw": g,
        "product": head,
        "attrs": attrs,
        "price_cap": price,
        "color": color,
        "size": size,
        "fit": fit,
        "required_options": required_options,
    }


def _obs_price(obs: str) -> float | None:
    """Best-effort price parse from a product-page observation."""
    prices = [
        float(x)
        for x in re.findall(r"\$\s*([0-9]+(?:\.[0-9]+)?)", obs or "")
    ]
    return min(prices) if prices else None


def _option_matches(need: str, clickable: str) -> bool:
    """Fuzzy match a required color/size string to a click[option] label."""
    n = re.sub(r"[^a-z0-9]+", " ", need.lower()).strip()
    c = re.sub(r"[^a-z0-9]+", " ", clickable.lower()).strip()
    if not n or not c:
        return False
    if n == c or n in c or c in n:
        return True
    nt, ct = set(n.split()), set(c.split())
    if not nt:
        return False
    return len(nt & ct) / len(nt) >= 0.6


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


def format_admissible(actions: list[str]) -> str:
    return "\n".join(f"'{a}'" for a in actions)


def format_history(history: list, window: int = 2) -> tuple[str, int]:
    recent = history[-window:]
    start = len(history) - len(recent)
    lines = [
        f"[Observation {start + j + 1}: '{o[:400]}', Action {start + j + 1}: '{a}']"
        for j, (o, a) in enumerate(recent)
    ]
    return "\n".join(lines), len(recent)


def parse_action(reply: str) -> str:
    m = re.search(r"<action>\s*(.*?)\s*</action>", reply, re.DOTALL | re.I)
    if m:
        return m.group(1).strip()
    # fallback: search[...] or click[...]
    m = re.search(r"(search\[[^\]]+\]|click\[[^\]]+\])", reply, re.I)
    return m.group(1).strip() if m else reply[-40:].strip()


def detect_page(obs: str, admissible: list[str]) -> str:
    low = obs.lower()
    if any(a.startswith("search[") for a in admissible) and not any(
        a.startswith("click[") and a not in ("click[search]",) for a in admissible
    ):
        # only search bar
        if "back to search" not in low and "buy now" not in low:
            return "search"
    if any("buy now" in a.lower() for a in admissible):
        return "product"
    if "back to search" in low or any(re.match(r"click\[[a-z0-9]{8,}\]$", a, re.I) for a in admissible):
        return "results"
    if "buy now" in low or "price" in low:
        return "product"
    return "search"


class Controller:
    def __init__(self, goal: str, skills: dict[str, dict]):
        self.goal = goal
        self.spec = parse_goal(goal)
        self.skills = skills
        self.actions: list[str] = []
        self.failed: set[str] = set()
        self.tried_products: set[str] = set()
        self.rejected_products: set[str] = set()
        self.selected_options: set[str] = set()
        self.queries: list[str] = []
        self.query_round = 0
        self.page = "search"
        self.last_obs = ""
        self.current_asin: str | None = None

    def _options_satisfied(self, admissible: list[str] | None = None) -> bool:
        """True when each required color/size/fit is selected (fuzzy)."""
        needed = self.spec.get("required_options") or []
        if not needed:
            return True
        selected = list(self.selected_options)
        if admissible:
            # also treat currently highlighted clickables that match as selectable
            for a in admissible:
                if a.startswith("click["):
                    selected.append(a[6:-1])
        for need in needed:
            if not any(_option_matches(need, s) for s in selected):
                return False
        return True

    def _missing_options(self, admissible: list[str]) -> list[str]:
        missing = []
        for need in self.spec.get("required_options") or []:
            # already selected?
            if any(_option_matches(need, s) for s in self.selected_options):
                continue
            # find a clickable
            hits = [
                a for a in admissible
                if a.startswith("click[")
                and a not in self.failed
                and _option_matches(need, a[6:-1])
            ]
            if hits:
                missing.append(hits[0])
        return missing

    def _price_ok(self, obs: str | None = None) -> bool:
        cap = self.spec.get("price_cap")
        if cap is None:
            return True
        price = _obs_price(obs if obs is not None else self.last_obs)
        if price is None:
            return True  # unknown — don't block solely on parse miss
        return price <= cap + 1e-6

    def _can_buy(self, admissible: list[str], obs: str | None = None) -> bool:
        return self._options_satisfied() and self._price_ok(obs)

    def update(self, action: str, obs: str, admissible: list[str]) -> None:
        prev = self.last_obs
        self.actions.append(action)
        self.last_obs = obs
        self.page = detect_page(obs, admissible)
        if obs.strip() == prev.strip() or "nothing happens" in obs.lower():
            self.failed.add(action)
        if action.startswith("search["):
            q = action[7:-1]
            self.queries.append(q)
            if len(self.queries) >= 2 and self.queries[-1] == self.queries[-2]:
                self.failed.add(action)
        m = re.match(r"click\[([a-z0-9]{8,})\]$", action, re.I)
        if m:
            asin = m.group(1).lower()
            self.tried_products.add(asin)
            self.current_asin = asin
            self.selected_options.clear()
        if action.startswith("click[") and self.page == "product":
            opt = action[6:-1].strip().lower()
            if opt not in ("buy now", "description", "features", "reviews",
                           "attributes", "back to search", "< prev", "next >"):
                self.selected_options.add(opt)
        # Leaving a product without buying → mark rejected if options/price bad
        if (
            self.current_asin
            and ("back to search" in action.lower() or action.lower() == "click[< prev]")
        ):
            self.rejected_products.add(self.current_asin)
            self.current_asin = None

    def skill_id(self) -> str:
        # After enough mismatches, force refine-query skill when on search/results
        if (
            len(self.rejected_products) + len(self.tried_products) >= 2
            and self.page in ("search", "results")
            and "refine:EXPLORE" in self.skills
        ):
            if self.page == "search" or len(self.tried_products) >= 2:
                return "refine:EXPLORE"
        if self.page == "search":
            return "query:EXPLORE"
        if self.page == "results":
            return "browse:NAVIGATE"
        if self.page == "product":
            if not self._options_satisfied():
                return "verify:EXECUTE"
            if not self._price_ok():
                return "refine:EXPLORE" if "refine:EXPLORE" in self.skills else "browse:NAVIGATE"
            return "purchase:EXECUTE"
        return "browse:NAVIGATE"

    def progress(self) -> str:
        return (
            f"- page={self.page}; price_cap={self.spec['price_cap']}; "
            f"color={self.spec.get('color')}; size={self.spec.get('size')}; "
            f"required_options={self.spec.get('required_options')}; "
            f"attrs={self.spec['attrs'][:6]}\n"
            f"- options_satisfied={self._options_satisfied()}; "
            f"price_ok={self._price_ok()}; query_round={self.query_round}\n"
            f"- queries={self.queries[-3:]}\n"
            f"- tried_products={sorted(self.tried_products)}\n"
            f"- rejected_products={sorted(self.rejected_products)}\n"
            f"- selected_options={sorted(self.selected_options)}\n"
            f"- failed_actions_do_not_repeat={sorted(self.failed)[:12]}\n"
            f"- recent_actions={self.actions[-6:]}"
        )

    def skill_block(self) -> tuple[str, str]:
        sid = self.skill_id()
        skill = self.skills.get(sid) or next(iter(self.skills.values()))
        return sid, format_skill(skill, self.progress())

    def compose_query(self, refine: bool = False) -> str:
        """Build search keywords; on refine, rotate in unused attrs/color/size."""
        parts: list[str] = []
        prod = self.spec["product"]
        # Keep product short: first ~6 tokens
        if prod:
            parts.extend(prod.split()[:6])
        pool = list(self.spec["attrs"])
        for extra in (self.spec.get("color"), self.spec.get("size"), self.spec.get("fit")):
            if extra:
                pool.append(extra)
        # Drop ultra-generic tokens
        stop = {"for", "with", "and", "the", "a", "an", "men", "women", "mens", "womens"}
        pool = [p for p in pool if p and p.lower() not in stop]

        if refine or self.query_round > 0:
            # Rotate: skip attrs already used in prior queries
            used = " ".join(self.queries).lower()
            fresh = [p for p in pool if p.lower() not in used]
            chosen = (fresh or pool)[:4]
            self.query_round += 1
        else:
            # Prefer distinctive attrs: material/fit + color + size
            preferred = []
            for key in (self.spec.get("color"), self.spec.get("size")):
                if key:
                    preferred.append(key)
            preferred.extend(pool[:3])
            # dedupe
            seen = set()
            chosen = []
            for p in preferred:
                pl = p.lower()
                if pl not in seen:
                    seen.add(pl)
                    chosen.append(p)
                if len(chosen) >= 4:
                    break
        q = " ".join(parts + chosen).strip()
        return q[:90] or "product"

    def _phase_choice(self, admissible: list[str]) -> str | None:
        available = [a for a in admissible if a not in self.failed]
        if not available:
            available = list(admissible)

        # Force refine after enough bad products
        must_refine = len(self.tried_products) >= 2 or len(self.rejected_products) >= 1

        if self.page == "search" or (
            any(a.startswith("search[") for a in available)
            and not any(a.startswith("click[") for a in available if "search" not in a.lower())
        ):
            return f"search[{self.compose_query(refine=must_refine or self.query_round > 0)}]"

        if self.page == "results":
            if must_refine and self.query_round == 0:
                back = [a for a in available if "back to search" in a.lower()]
                if back:
                    return back[0]
            codes = [
                a for a in available
                if re.match(r"click\[[a-z0-9]{8,}\]$", a, re.I)
                and a[6:-1].lower() not in self.tried_products
                and a[6:-1].lower() not in self.rejected_products
            ]
            if codes:
                return codes[0]
            back = [a for a in available if "back to search" in a.lower()]
            if back:
                return back[0]
            nxt = [a for a in available if "next" in a.lower()]
            if nxt and len(self.tried_products) < 2:
                return nxt[0]
            return available[0]

        if self.page == "product":
            missing = self._missing_options(available)
            if missing:
                return missing[0]
            if self._can_buy(available):
                buy = [a for a in available if "buy now" in a.lower()]
                if buy:
                    return buy[0]
            # Cannot buy safely — leave
            if self.current_asin:
                self.rejected_products.add(self.current_asin)
            back = [a for a in available if "back to search" in a.lower()]
            if back:
                return back[0]
            prev = [a for a in available if "prev" in a.lower()]
            if prev:
                return prev[0]
        return available[0] if available else None

    def repair(self, candidate: str, admissible: list[str]) -> tuple[str, bool]:
        counts = Counter(self.actions[-6:])

        # Do not abandon a viable product while a required option is visibly
        # available but not yet selected. This is the main short-horizon
        # failure mode under SkillRL's official 15-step protocol.
        if self.page == "product" and (
            "back to search" in candidate.lower()
            or candidate.lower().strip() == "click[< prev]"
        ):
            missing = self._missing_options(admissible)
            if missing:
                return missing[0], True

        # Hard veto: Buy Now when options/price not satisfied (partial-match fails)
        if candidate.lower().strip() in ("click[buy now]", "buy now") or (
            candidate.startswith("click[") and "buy now" in candidate.lower()
        ):
            if not self._can_buy(admissible):
                fallback = self._phase_choice(admissible)
                return (fallback or candidate), True

        if candidate.startswith("search["):
            if candidate in self.failed or counts[candidate] >= 2:
                return f"search[{self.compose_query(refine=True)}]", True
            return candidate, False

        # Veto re-clicking rejected / already-tried ASINs when alternatives exist
        m = re.match(r"click\[([a-z0-9]{8,})\]$", candidate, re.I)
        if m and m.group(1).lower() in self.rejected_products:
            fallback = self._phase_choice(admissible)
            return (fallback or candidate), True

        valid = candidate in admissible
        repeated = counts[candidate] >= 2
        blocked = candidate in self.failed
        if valid and not repeated and not blocked:
            return candidate, False

        if candidate and not blocked and not repeated:
            best = max(
                admissible,
                key=lambda a: SequenceMatcher(None, candidate.lower(), a.lower()).ratio(),
                default="",
            )
            if best and SequenceMatcher(None, candidate.lower(), best.lower()).ratio() >= 0.85:
                # still veto buy if grounded to buy now unsafely
                if "buy now" in best.lower() and not self._can_buy(admissible):
                    fallback = self._phase_choice(admissible)
                    return (fallback or best), True
                return best, best != candidate

        fallback = self._phase_choice(admissible)
        return (fallback or candidate), True


def build_prompt(task, block, history, obs, admissible, window=2) -> str:
    acts = format_admissible(admissible)
    if not history:
        return TEMPLATE_NO_HIS.format(
            task_description=task,
            current_observation=obs[:3000],
            available_actions=acts,
        )
    hist, n = format_history(history, window)
    return TEMPLATE_WITH_MEMORY.format(
        task_description=task,
        retrieved_memories=block,
        step_count=len(history),
        history_length=n,
        action_history=hist,
        current_step=len(history) + 1,
        current_observation=obs[:3000],
        available_actions=acts,
    )


def load_their_blocks() -> dict:
    path = (
        PROJECT_ROOT / "SkillRL" / "memory_data" / "webshop" / "claude_style_skills.json"
    )
    if not path.exists():
        path = Path("/workspace/SkillRL/memory_data/webshop/claude_style_skills.json")
    data = json.loads(path.read_text())
    # format like SkillsOnlyMemory.format_for_prompt — use all general + other
    lines = ["### General Principles"]
    for sk in data.get("general_skills", [])[:6]:
        lines.append(f"- **{sk.get('title','')}**: {sk.get('principle','')}")
    lines.append("\n### Task-Relevant Skills")
    for cat, skills in data.get("task_specific_skills", {}).items():
        for sk in skills[:2]:
            lines.append(f"- **{sk.get('title','')}**: {sk.get('principle','')}")
            if sk.get("when_to_apply"):
                lines.append(f"  _Apply when: {sk['when_to_apply']}_")
    lines.append("\n### Mistakes to Avoid")
    for m in data.get("common_mistakes", [])[:5]:
        lines.append(f"- **Don't**: {m.get('description','')}")
        if m.get("how_to_avoid"):
            lines.append(f"  Fix: {m['how_to_avoid']}")
    return "\n".join(lines)


def resolve_goal_indices(
    n_goals: int,
    episodes: int,
    seed: int,
    goal_indices_path: Path | None,
    write_indices_path: Path | None,
) -> list[int]:
    """Deterministic unique goal indices for paired COS-PLAY vs SkillRL runs."""
    if goal_indices_path is not None:
        raw = json.loads(goal_indices_path.read_text())
        if isinstance(raw, dict):
            idxs = list(raw["indices"])
        else:
            idxs = list(raw)
        if len(idxs) < episodes:
            raise ValueError(
                f"goal indices file has {len(idxs)} entries, need {episodes}"
            )
        idxs = idxs[:episodes]
    else:
        rng = random.Random(seed)
        pool = list(range(n_goals))
        rng.shuffle(pool)
        if len(pool) >= episodes:
            idxs = pool[:episodes]
        else:
            # Human-goal overlap on the 1k catalog is tiny (~13 goals).
            # Allow repeats so larger paired-N still runs on that setting.
            idxs = [pool[i % len(pool)] for i in range(episodes)]
            rng.shuffle(idxs)
    if write_indices_path is not None:
        write_indices_path.parent.mkdir(parents=True, exist_ok=True)
        write_indices_path.write_text(
            json.dumps(
                {"seed": seed, "n_goals": n_goals, "indices": idxs},
                indent=2,
            )
        )
    return idxs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8011/v1")
    ap.add_argument("--model", default="skillrl-webshop")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.4)
    # Match SkillRL small-env training: 1000 products + synthetic goals.
    ap.add_argument("--num-products", type=int, default=1000)
    ap.add_argument(
        "--human-goals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use human goals (only ~13 overlap the 1k catalog). Default: synthetic.",
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mode", choices=["cosplay", "their_skills", "no_skills"],
                    default="cosplay")
    ap.add_argument("--out", default="runs/eval_webshop_cosplay.jsonl")
    ap.add_argument(
        "--goal-indices",
        type=str,
        default=None,
        help="JSON file of goal indices (or {indices:[...]}) for paired evals.",
    )
    ap.add_argument(
        "--write-goal-indices",
        type=str,
        default=None,
        help="If set, write the sampled goal indices here for the paired run.",
    )
    args = ap.parse_args()

    import openai
    from env_wrappers.webshop_nl_wrapper import WebShopNLWrapper

    # Seed before env construction so synthetic goal price text is reproducible.
    random.seed(args.seed)
    bank = (
        PROJECT_ROOT / "labeling" / "output" / "skillrl_seed_bank"
        / "webshop" / "skill_bank.jsonl"
    )
    skills = load_skills(bank)
    their_block = load_their_blocks() if args.mode == "their_skills" else ""

    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)
    env = WebShopNLWrapper(
        max_steps=args.max_steps,
        num_products=args.num_products,
        human_goals=args.human_goals,
    )
    print(
        f"WebShop ready: products={args.num_products} "
        f"human_goals={args.human_goals} n_goals={env.num_goals}",
        flush=True,
    )

    goal_indices = resolve_goal_indices(
        n_goals=env.num_goals,
        episodes=args.episodes,
        seed=args.seed,
        goal_indices_path=(
            Path(args.goal_indices) if args.goal_indices else None
        ),
        write_indices_path=(
            Path(args.write_goal_indices) if args.write_goal_indices else None
        ),
    )

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    results = []
    for ep in range(args.episodes):
        gidx = goal_indices[ep]
        obs, info = env.reset(goal_idx=gidx)
        goal = env._goal
        ctl = Controller(goal, skills)
        history = []
        cur_obs = obs
        total_reward = 0.0
        success = False
        repairs = 0
        selected = Counter()
        t0 = time.time()

        for _ in range(args.max_steps):
            admissible = info.get("action_names", [])
            if not admissible:
                break
            if args.mode == "cosplay":
                sid, block = ctl.skill_block()
                selected[sid] += 1
            elif args.mode == "their_skills":
                sid, block = "their", their_block
            else:
                sid, block = "none", ""

            prompt = build_prompt(goal, block, history, cur_obs, admissible)
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=512,
            )
            candidate = parse_action(resp.choices[0].message.content or "")
            if args.mode == "cosplay":
                action, repaired = ctl.repair(candidate, admissible)
                repairs += int(repaired)
            else:
                # soft ground for baseline
                if candidate.startswith("search[") or candidate in admissible:
                    action = candidate
                else:
                    best = max(
                        admissible,
                        key=lambda a: SequenceMatcher(
                            None, candidate.lower(), a.lower()
                        ).ratio(),
                        default=admissible[0],
                    )
                    action = best

            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(info.get("raw_env_reward", reward))
            if args.mode == "cosplay":
                ctl.update(action, next_obs, info.get("action_names", []))
            history.append((cur_obs, action))
            cur_obs = next_obs
            if terminated or truncated:
                # WebShop success: reward typically in [0,1], often 1.0 on buy match
                success = total_reward >= 0.5
                break

        rec = {
            "episode": ep,
            "goal_idx": gidx,
            "task": goal,
            "success": success,
            "reward": total_reward,
            "steps": len(history),
            "repairs": repairs,
            "selected_skills": dict(selected),
            "mode": args.mode,
            "wall_s": round(time.time() - t0, 1),
        }
        results.append(rec)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        wins = sum(r["success"] for r in results)
        print(
            f"[{ep}] {'SUCCESS' if success else 'fail'} "
            f"r={total_reward:.2f} steps={len(history)} repairs={repairs} "
            f"running={wins}/{len(results)} ({wins/len(results):.0%}) "
            f"gidx={gidx} task={goal[:60]}",
            flush=True,
        )

    wins = sum(r["success"] for r in results)
    mean_r = sum(r["reward"] for r in results) / max(len(results), 1)
    n_unique = len({r["task"] for r in results})
    print(
        f"FINAL [{args.mode}]: {wins}/{len(results)} = {wins/len(results):.1%} "
        f"mean_reward={mean_r:.3f} unique_goals={n_unique}/{len(results)}"
    )


if __name__ == "__main__":
    main()
