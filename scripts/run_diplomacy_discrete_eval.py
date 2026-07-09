#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate the Diplomacy Decision Agent using the TRAINING-STYLE discrete action
format.  This script bridges the gap between how the LoRA adapter was trained
(pick ONE numbered unit order, fill the rest automatically) and how inference
was previously run (generate free-form orders for all units at once).

For the DA-controlled power:
  1. Build a flat list of candidate unit orders (same as _DiplomacyAdapter).
  2. Present them as "Available actions (pick ONE by number):" with the
     same system prompt and format the LoRA was trained on.
  3. Send to vLLM (LoRA adapter).
  4. Parse "ACTION: <number>".
  5. Fill remaining units with the BEST heuristic order from the game engine's
     legal moves (hold for units without a chosen order).

For opponent powers:
  Use the opponent model (e.g. gpt-5.4) via OpenRouter with free-form
  order generation, same as run_qwen3_8b_eval.py.

Output format is identical to run_qwen3_8b_eval.py (Episode/Experience JSON)
so downstream analysis scripts work unchanged.

Usage:
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$(pwd)/../AgentEvolver:$(pwd)/../AI_Diplomacy:$PYTHONPATH"
    export VLLM_BASE_URL="http://localhost:8025/v1"

    python -m scripts.run_diplomacy_discrete_eval \\
        --model qwen3-8b-diplomacy-best \\
        --opponent_model gpt-5.4 \\
        --episodes 70 --per_power \\
        --bank path/to/combined_skill_bank.jsonl \\
        --temperature 0.4 --verbose
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"

for p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

from data_structure.experience import Experience, Episode, Episode_Buffer

try:
    from env_wrappers.diplomacy_nl_wrapper import (
        DiplomacyNLWrapper,
        parse_orders,
        build_structured_state_summary as _diplo_structured_summary,
    )
except ImportError:
    print("[FATAL] Cannot import DiplomacyNLWrapper. Install AI_Diplomacy deps.")
    sys.exit(1)

from decision_agents.agent_helper import (
    get_state_summary,
    infer_intention,
    strip_think_tags,
    compact_text_observation,
    extract_game_facts,
    HARD_SUMMARY_CHAR_LIMIT,
)

try:
    from API_func import ask_model, ask_vllm
except ImportError:
    ask_model = None
    ask_vllm = None

from skill_agents.skill_bank.bank import SkillBankMVP
try:
    from skill_agents.query import SkillQueryEngine
except ImportError:
    SkillQueryEngine = None

from decision_agents.agent_helper import select_skill_from_bank


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIPLOMACY_POWERS = [
    "AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY",
]
DIPLOMACY_MAX_PHASES = 20

SYSTEM_PROMPT = (
    "You are an expert game-playing agent. "
    "You receive a game state and must choose exactly one action by its NUMBER.\n\n"
    "Rules:\n"
    "- Study the state carefully before choosing.\n"
    "- Consider which action makes the most progress toward winning.\n"
    "- NEVER repeat the same action more than 2 times in a row — try something different.\n"
    "- If recent actions got zero reward, change strategy.\n\n"
    "Output format (strict):\n"
    "ACTION: <number>\n"
)

OPPONENT_SYSTEM = (
    "You are an expert Diplomacy player.\n"
    "You control one power and must issue orders for your units this phase.\n\n"
    "Order formats:\n"
    "  Hold:         A PAR H\n"
    "  Move:         A PAR - BUR\n"
    "  Support hold: A MAR S A PAR\n"
    "  Support move: A MAR S A PAR - BUR\n"
    "  Convoy:       F ENG C A LON - BRE\n"
    "  Retreat:      A PAR R MAR\n"
    "  Build:        A PAR B  or  F BRE B\n"
    "  Disband:      A PAR D\n\n"
    "Output format (strict):\n"
    "REASONING: <1-3 sentences>\n"
    "ORDERS: <order1>; <order2>; ..."
)

MAX_DA_CANDIDATES = 20
MAX_ORDERS_PER_LOC = 8
MAX_REPEAT_ACTIONS = 2


# ---------------------------------------------------------------------------
# Skill bank helpers (reused from run_qwen3_8b_eval)
# ---------------------------------------------------------------------------

def load_skill_bank(
    bank_path: str,
    *,
    use_query_engine: bool = True,
) -> Tuple[Any, Any]:
    bp = Path(bank_path)
    if bp.is_dir():
        candidates = ["bank.jsonl", "skill_bank.jsonl"]
        jsonl = None
        for c in candidates:
            if (bp / c).exists():
                jsonl = bp / c
                break
        if jsonl is None:
            jsonls = sorted(bp.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
            jsonl = jsonls[0] if jsonls else None
        if jsonl is None:
            print(f"[load_skill_bank] WARNING: no .jsonl found in {bp}, using empty bank.")
            return None, None
        bp = jsonl

    bank = SkillBankMVP(path=str(bp))
    bank.load()
    print(f"[load_skill_bank] Loaded {len(bank)} skills from {bp}")

    engine = None
    if use_query_engine and SkillQueryEngine is not None and len(bank) > 0:
        try:
            engine = SkillQueryEngine(bank)
            print(f"[load_skill_bank] SkillQueryEngine initialised")
        except Exception as exc:
            print(f"[load_skill_bank] SkillQueryEngine init failed: {exc}")
    return bank, engine


def get_skill_guidance(
    skill_bank: Any,
    state_text: str,
    game_name: str = "",
) -> Optional[Dict[str, Any]]:
    if skill_bank is None:
        return None
    key = state_text[:500]
    try:
        result = select_skill_from_bank(skill_bank, key, top_k=1)
        if result and result.get("skill_id"):
            if not result.get("skill_name"):
                underlying = (
                    getattr(skill_bank, "_bank", None)
                    or getattr(skill_bank, "bank", None)
                    or skill_bank
                )
                if hasattr(underlying, "get_skill"):
                    skill_obj = underlying.get_skill(result["skill_id"])
                    if skill_obj:
                        result["skill_name"] = skill_obj.name or result["skill_id"]
                        if not result.get("execution_hint"):
                            result["execution_hint"] = skill_obj.strategic_description or ""
            return result
    except Exception:
        pass
    return None


def format_skill_guidance_for_prompt(guidance: Optional[Dict[str, Any]]) -> str:
    if guidance is None or not guidance.get("skill_id"):
        return ""
    parts = [f"\n--- Active Skill: {guidance.get('skill_name', guidance['skill_id'])} ---"]
    if guidance.get("execution_hint"):
        parts.append(f"  Strategy: {guidance['execution_hint'][:120]}")
    protocol = guidance.get("protocol", {})
    steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
    if steps:
        parts.append("  Steps: " + " → ".join(s[:40] for s in steps[:5]))
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# DA action helpers (mirrors training _DiplomacyAdapter)
# ---------------------------------------------------------------------------

def build_da_candidates(game, power_name: str) -> List[str]:
    """Build the flat candidate list exactly as training does."""
    possible = game.get_all_possible_orders()
    orderable = game.get_orderable_locations(power_name) or []
    flat: List[str] = []
    for loc in orderable:
        orders = possible.get(loc, [])
        flat.extend(orders[:MAX_ORDERS_PER_LOC])
    if not flat:
        return ["hold"]
    return flat[:MAX_DA_CANDIDATES]


def make_full_orders(
    chosen_order: str,
    game,
    power_name: str,
    strategy: str = "hold",
) -> List[str]:
    """Given the DA's single chosen order, build orders for ALL units.

    *strategy* controls what unchosen units do:
      - "hold":   unchosen units hold (deterministic, conservative).
      - "random": unchosen units pick random valid orders (matches training).
    """
    possible = game.get_all_possible_orders()
    orderable = game.get_orderable_locations(power_name) or []
    orders: List[str] = []
    used = False
    for loc in orderable:
        loc_orders = possible.get(loc, [])
        if not loc_orders:
            continue
        if not used and chosen_order in loc_orders:
            orders.append(chosen_order)
            used = True
        else:
            if strategy == "random":
                orders.append(random.choice(loc_orders))
            else:
                hold = next((o for o in loc_orders if o.endswith(" H")), None)
                orders.append(hold if hold else loc_orders[0])
    if not used and chosen_order != "hold":
        pass
    return orders


def format_numbered_actions(action_names: List[str]) -> str:
    return "\n".join(f"  {i+1}. {a}" for i, a in enumerate(action_names))


def build_da_state_summary(
    game,
    power_name: str,
    step_idx: int = 0,
    total_steps: int = 20,
    total_reward: float = 0.0,
    note: str = "",
) -> str:
    """Build the compact state summary matching training's build_rag_summary format."""
    power = game.powers.get(power_name)
    if power is None:
        return f"game=diplomacy | power={power_name}"
    phase = game.get_current_phase()
    centers = len(power.centers) if power.centers else 0
    units_list = [str(u) for u in power.units]
    units_str = ", ".join(units_list) if units_list else "none"

    parts = [
        f"game=diplomacy",
        f"step={step_idx}/{total_steps}",
        f"phase={phase}",
        f"power={power_name}",
        f"centers={centers}",
        f"units={units_str}",
    ]
    if total_reward and total_reward != 0:
        parts.append(f"reward={total_reward:+g}")
    if note:
        parts.append(f"note={note}")
    return " | ".join(parts)


def build_recent_context(recent_actions: List[str], recent_rewards: List[float]) -> str:
    """Matches training's _build_recent_context exactly."""
    if not recent_actions:
        return ""
    lines = ["Recent actions and rewards:"]
    for a, r in zip(recent_actions[-5:], recent_rewards[-5:]):
        lines.append(f"  {a} -> reward {r:.1f}")
    total = sum(recent_rewards[-5:])
    if total == 0 and len(recent_actions) >= 3:
        lines.append("WARNING: Recent actions got 0 reward. Try a DIFFERENT action!")
    lines.append("")
    return "\n".join(lines) + "\n"


def apply_anti_repetition(
    action: str,
    candidates: List[str],
    recent_actions: List[str],
    recent_rewards: List[float],
) -> str:
    """Matches training's _apply_anti_repetition for Diplomacy."""
    if len(recent_actions) < MAX_REPEAT_ACTIONS:
        return action
    tail = recent_actions[-MAX_REPEAT_ACTIONS:]
    tail_rewards = recent_rewards[-MAX_REPEAT_ACTIONS:]
    if all(a == action for a in tail) and sum(tail_rewards) <= 0:
        alternatives = [a for a in candidates if a != action]
        if alternatives:
            return random.choice(alternatives)
    return action


def parse_action_number(text: str, candidates: List[str]) -> str:
    """Parse 'ACTION: N' from model output and return the corresponding candidate."""
    cleaned = strip_think_tags(text) or text
    match = re.search(r"ACTION\s*:\s*(\d+)", cleaned, re.IGNORECASE)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    num_match = re.search(r"\b(\d+)\b", cleaned)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0] if candidates else "hold"


# ---------------------------------------------------------------------------
# Opponent action (free-form LLM orders, same as run_qwen3_8b_eval)
# ---------------------------------------------------------------------------

def _parse_opponent_orders(reply: str) -> List[str]:
    cleaned = strip_think_tags(reply) or reply
    orders_m = re.search(r"ORDERS\s*:\s*(.+)", cleaned, re.DOTALL | re.IGNORECASE)
    raw = orders_m.group(1).strip() if orders_m else cleaned
    parts = re.split(r"[;\n]+", raw)
    orders = []
    for p in parts:
        p = p.strip().strip("'\",-")
        if re.match(r"^[A-Z]\s+[A-Z]{3}", p):
            orders.append(p)
    return orders


def opponent_action(
    state_nl: str,
    model: str,
    temperature: float = 0.3,
) -> Tuple[List[str], Optional[str]]:
    if ask_model is None:
        return [], None
    prompt = OPPONENT_SYSTEM + "\n\nCurrent game state:\n\n" + state_nl + "\n\nSubmit your orders using the format above."
    try:
        reply = ask_model(prompt, model=model, temperature=temperature, max_tokens=1200)
        if not reply or reply.startswith("Error"):
            return [], None
    except Exception as exc:
        print(f"    [WARN] Opponent call failed ({exc})")
        return [], None

    reasoning = None
    cleaned = strip_think_tags(reply) or reply
    m = re.search(r"REASONING\s*:\s*(.+?)(?=\nORDERS|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        reasoning = m.group(1).strip()
    orders = _parse_opponent_orders(reply)
    return orders, reasoning


# ---------------------------------------------------------------------------
# DA action (training-style discrete pick via vLLM LoRA)
# ---------------------------------------------------------------------------

def da_discrete_action(
    state_summary: str,
    candidates: List[str],
    model: str,
    temperature: float = 0.4,
    skill_guidance: Optional[Dict[str, Any]] = None,
    subgoal: str = "",
    recent_actions: Optional[List[str]] = None,
    recent_rewards: Optional[List[float]] = None,
) -> Tuple[str, Optional[str]]:
    """Query the LoRA model with the training-style discrete action prompt."""
    if ask_model is None:
        return candidates[0] if candidates else "hold", None

    skill_text = format_skill_guidance_for_prompt(skill_guidance)
    subgoal_line = f"Subgoal: {subgoal}\n" if subgoal else ""
    recent_context = build_recent_context(
        recent_actions or [], recent_rewards or [],
    )
    action_user = (
        f"Game state:\n\n{state_summary}\n\n"
        f"{subgoal_line}"
        f"{recent_context}"
        f"Available actions (pick ONE by number):\n{format_numbered_actions(candidates)}\n\n"
        f"Output ACTION number."
    )
    prompt = SYSTEM_PROMPT + skill_text + "\n" + action_user

    try:
        reply = ask_model(prompt, model=model, temperature=temperature, max_tokens=128)
        if not reply or reply.startswith("Error"):
            return candidates[0] if candidates else "hold", None
    except Exception as exc:
        print(f"    [WARN] DA vLLM call failed ({exc})")
        return candidates[0] if candidates else "hold", None

    chosen = parse_action_number(reply, candidates)

    reasoning = None
    cleaned = strip_think_tags(reply) or reply
    m = re.search(r"REASONING\s*:\s*(.+?)(?=\nACTION|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        reasoning = m.group(1).strip()

    return chosen, reasoning


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _sanitize_keys(d: Any) -> Any:
    if isinstance(d, dict):
        return {str(k): _sanitize_keys(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_sanitize_keys(v) for v in d]
    return d


def run_episode(
    model: str,
    controlled_power: str,
    opponent_model: str,
    temperature: float = 0.4,
    seed: int = 42,
    skill_bank: Any = None,
    verbose: bool = False,
    unchosen_strategy: str = "hold",
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one Diplomacy episode with DA (discrete) vs opponent LLM."""
    task = "Gain the most supply centres in Diplomacy through strategic orders and alliances."

    env = DiplomacyNLWrapper(seed=seed, max_phases=DIPLOMACY_MAX_PHASES)
    obs, info = env.reset()

    all_powers = list(obs.keys()) if isinstance(obs, dict) else DIPLOMACY_POWERS
    resolved_cp = controlled_power.upper()
    if resolved_cp not in all_powers:
        resolved_cp = all_powers[0]
        print(f"    [WARN] '{controlled_power}' not in {all_powers}, using {resolved_cp}")

    if verbose:
        print(f"    DA={resolved_cp} ({model}), opponents={opponent_model}, strategy={unchosen_strategy}")

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    prev_sc_counts: Dict[str, int] = {}
    if env.game is not None:
        prev_sc_counts = {
            pn: len(po.centers) for pn, po in env.game.powers.items()
            if not po.is_eliminated()
        }

    recent_da_actions: List[str] = []
    recent_da_rewards: List[float] = []
    last_note: str = ""

    while not env.done:
        actions: Dict[str, Union[List[str], str]] = {}
        step_reasonings: Dict[str, str] = {}

        active_powers = info.get("active_powers", {})
        powers_to_query = [
            pname for pname in obs
            if obs.get(pname) and pname in active_powers
        ]

        # --- DA controlled power: discrete action pick ---
        if resolved_cp in powers_to_query:
            # Generate strategic note (matches training's base model note)
            note = last_note
            if ask_model is not None:
                try:
                    obs_text = obs.get(resolved_cp, "")[:1000] if isinstance(obs, dict) else ""
                    note_compact = compact_text_observation(obs_text, max_chars=200) or obs_text[:200]
                    note_prompt = (
                        f"diplomacy: {note_compact}\n"
                        f"Key strategic note about the current threat or opportunity "
                        f"(max 10 words, be specific to what changed).\nNote:"
                    )
                    raw_note = ask_model(note_prompt, model=model, temperature=0.2, max_tokens=64)
                    if raw_note and not raw_note.startswith("Error"):
                        note = strip_think_tags(raw_note) or raw_note
                        note = note.split("\n")[0].strip().strip('"').strip("'")[:80]
                        last_note = note
                except Exception:
                    pass

            state_summary = build_da_state_summary(
                env.game, resolved_cp,
                step_idx=step_count, total_steps=DIPLOMACY_MAX_PHASES,
                total_reward=total_reward,
                note=note,
            )
            candidates = build_da_candidates(env.game, resolved_cp)

            guidance = get_skill_guidance(skill_bank, state_summary, game_name="diplomacy")

            subgoal = ""
            try:
                subgoal = infer_intention(
                    state_summary, game="diplomacy", model=model,
                    context={"power_name": resolved_cp},
                ) or ""
            except Exception:
                pass

            if len(candidates) > 1 or (candidates and candidates[0] != "hold"):
                chosen_order, da_reasoning = da_discrete_action(
                    state_summary, candidates, model,
                    temperature=temperature,
                    skill_guidance=guidance,
                    subgoal=subgoal,
                    recent_actions=recent_da_actions,
                    recent_rewards=recent_da_rewards,
                )
                chosen_order = apply_anti_repetition(
                    chosen_order, candidates,
                    recent_da_actions, recent_da_rewards,
                )
                full_orders = make_full_orders(
                    chosen_order, env.game, resolved_cp,
                    strategy=unchosen_strategy,
                )
                actions[resolved_cp] = full_orders
                if da_reasoning:
                    step_reasonings[resolved_cp] = da_reasoning
                recent_da_actions.append(chosen_order)
                if verbose:
                    print(f"    DA chose: {chosen_order} (from {len(candidates)} candidates)")
                    print(f"    Full orders: {full_orders}")
            else:
                actions[resolved_cp] = []
                recent_da_actions.append("hold")

        # --- Opponent powers: free-form LLM orders (in parallel) ---
        opp_powers = [p for p in powers_to_query if p != resolved_cp]
        if opp_powers:
            with ThreadPoolExecutor(max_workers=max(len(opp_powers), 1)) as pool:
                futures = {}
                for pname in opp_powers:
                    f = pool.submit(opponent_action, obs[pname], opponent_model, temperature)
                    futures[f] = pname

                for future in as_completed(futures):
                    pname = futures[future]
                    try:
                        opp_orders, opp_reasoning = future.result()
                    except Exception as exc:
                        print(f"    [WARN] {pname} failed ({exc})")
                        opp_orders, opp_reasoning = [], None
                    if parse_orders is not None and env.game is not None:
                        opp_orders = parse_orders(opp_orders, env.game, pname)
                    actions[pname] = opp_orders
                    if opp_reasoning:
                        step_reasonings[pname] = opp_reasoning
                    if verbose:
                        print(f"    {pname} [{opponent_model}]: {len(opp_orders)} orders")

        # --- Step the environment ---
        phase_before = info.get("phase", "")
        rewards: Any = {}
        next_obs, rewards, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated
        reward_val = sum(rewards.values()) if isinstance(rewards, dict) else 0.0
        total_reward += reward_val

        # Track DA per-step reward for recent context
        da_step_reward = float(rewards.get(resolved_cp, 0.0)) if isinstance(rewards, dict) else 0.0
        recent_da_rewards.append(da_step_reward)

        # SC tracking
        cur_sc_counts: Dict[str, int] = {}
        sc_delta_parts: List[str] = []
        if env.game is not None:
            for pn, po in env.game.powers.items():
                cur = len(po.centers) if not po.is_eliminated() else 0
                cur_sc_counts[pn] = cur
                diff = cur - prev_sc_counts.get(pn, cur)
                if diff != 0:
                    sc_delta_parts.append(f"{pn}{'+' if diff > 0 else ''}{diff}")
        sc_delta_str = ", ".join(sc_delta_parts)

        # Build experience record
        structured = None
        if env.game is not None and _diplo_structured_summary is not None:
            structured = _diplo_structured_summary(env.game, resolved_cp, prev_sc_counts=prev_sc_counts)
        state_summary_str = get_state_summary("", structured_state=structured) if structured else ""

        primary_obs_text = obs.get(resolved_cp, "")[:1500] if isinstance(obs, dict) else str(obs)[:1500]
        intention = infer_intention(
            state_summary_str or primary_obs_text,
            game="diplomacy", model=model,
            context={
                "last_actions": [e.action for e in experiences[-5:]],
                "task": task,
                "power_name": resolved_cp,
                "phase": phase_before,
                "sc_delta": sc_delta_str,
            },
        )

        combined_reasoning = "\n".join(f"{pn}: {r}" for pn, r in step_reasonings.items())
        effective_intention = intention or step_reasonings.get(resolved_cp) or combined_reasoning

        exp = Experience(
            state=json.dumps(_sanitize_keys(dict(obs)), ensure_ascii=False, default=str),
            action=json.dumps(_sanitize_keys(dict(actions)), ensure_ascii=False, default=str),
            reward=float(reward_val),
            next_state=json.dumps(
                _sanitize_keys(dict(next_obs)), ensure_ascii=False, default=str
            ) if isinstance(next_obs, dict) else str(next_obs),
            done=done,
            intentions=effective_intention,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.summary_state = state_summary_str if state_summary_str else None
        exp.interface = {"env_name": "diplomacy", "game_name": "diplomacy"}
        exp.summary = state_summary_str or None

        experiences.append(exp)

        if verbose:
            phase = next_info.get("phase", "")
            da_sc = cur_sc_counts.get(resolved_cp, "?")
            print(
                f"  step {step_count}: reward={reward_val:.2f}, cum={total_reward:.2f}, "
                f"phase={phase}, DA_SC={da_sc}, sc_delta=[{sc_delta_str}]"
            )

        prev_sc_counts = cur_sc_counts
        obs = next_obs
        info = next_info
        step_count += 1
        if done:
            break

    # Final metadata
    final_rewards: Dict[str, float] = {}
    if isinstance(rewards, dict):
        final_rewards = {str(k): float(v) for k, v in rewards.items()}

    meta = {
        "game": "diplomacy",
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "decision_agent_discrete",
        "max_phases": DIPLOMACY_MAX_PHASES,
        "final_sc_rewards": final_rewards,
        "controlled_power": resolved_cp,
        "controlled_power_reward": float(rewards.get(resolved_cp, 0.0)) if isinstance(rewards, dict) else 0.0,
        "opponent_model": opponent_model,
        "unchosen_strategy": unchosen_strategy,
    }

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="diplomacy",
        game_name="diplomacy",
        metadata=meta,
    )
    episode.set_outcome()

    return episode, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diplomacy Decision Agent eval with training-style discrete actions",
    )
    parser.add_argument("--model", type=str, required=True, help="vLLM model/LoRA name for DA")
    parser.add_argument("--opponent_model", type=str, required=True, help="Opponent model (e.g. gpt-5.4)")
    parser.add_argument("--episodes", type=int, default=70, help="Total episodes (default: 70)")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature (default: 0.4)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--per_power", action="store_true", help="Cycle controlled power through 7 powers")
    parser.add_argument("--bank", type=str, default=None, help="Path to skill bank JSONL")
    parser.add_argument("--no-bank", action="store_true", help="Run without skill bank")
    parser.add_argument("--no-query-engine", action="store_true", help="Disable SkillQueryEngine")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--unchosen_strategy",
        type=str,
        default="hold",
        choices=["hold", "random"],
        help="How to handle units the DA didn't pick an order for: "
             "'hold' (deterministic) or 'random' (matches training). Default: hold.",
    )

    args = parser.parse_args()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = CODEBASE_ROOT / "output" / f"infer_diplomacy_discrete_{timestamp}"
    game_dir = base_dir / "diplomacy" / timestamp
    game_dir.mkdir(parents=True, exist_ok=True)

    # Load skill bank
    skill_bank_obj = None
    if not args.no_bank and args.bank:
        bank, engine = load_skill_bank(
            args.bank,
            use_query_engine=not args.no_query_engine,
        )
        skill_bank_obj = engine if engine is not None else bank

    print("=" * 70)
    print("  Diplomacy Decision Agent (Discrete Actions) Evaluation")
    print("=" * 70)
    print(f"  DA model:          {args.model}")
    print(f"  Opponent model:    {args.opponent_model}")
    print(f"  Episodes:          {args.episodes}")
    print(f"  Per-power cycling: {args.per_power}")
    print(f"  Temperature:       {args.temperature}")
    print(f"  Unchosen strategy: {args.unchosen_strategy}")
    bank_desc = "none"
    if skill_bank_obj is not None:
        bank_size = len(skill_bank_obj) if hasattr(skill_bank_obj, "__len__") else "?"
        bank_desc = f"{args.bank} ({bank_size} skills)"
    print(f"  Skill bank:        {bank_desc}")
    print(f"  Output:            {game_dir}")
    print("=" * 70)

    all_stats: List[Dict[str, Any]] = []
    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    jsonl_path = game_dir / "rollouts.jsonl"
    t0 = time.time()

    for ep_idx in range(args.episodes):
        cp_power = DIPLOMACY_POWERS[ep_idx % len(DIPLOMACY_POWERS)] if args.per_power else "AUSTRIA"
        print(f"\n  Episode {ep_idx + 1}/{args.episodes}  (DA={cp_power})")

        try:
            episode, stats = run_episode(
                model=args.model,
                controlled_power=cp_power,
                opponent_model=args.opponent_model,
                temperature=args.temperature,
                seed=args.seed + ep_idx,
                skill_bank=skill_bank_obj,
                verbose=args.verbose,
                unchosen_strategy=args.unchosen_strategy,
            )

            stats["episode_index"] = ep_idx
            print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

            episode_buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = _sanitize_keys(episode.to_dict())
            ep_data["metadata"] = stats
            ep_path = game_dir / f"episode_{ep_idx:03d}.json"
            with open(ep_path, "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

            record = _sanitize_keys(episode.to_dict())
            record["rollout_metadata"] = stats
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        except Exception as e:
            print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
            traceback.print_exc()
            all_stats.append({
                "game": "diplomacy",
                "episode_index": ep_idx,
                "error": str(e),
                "steps": 0,
                "total_reward": 0.0,
            })

    elapsed = time.time() - t0

    # Save episode buffer
    buffer_path = game_dir / "episode_buffer.json"
    episode_buffer.save_to_json(str(buffer_path))

    # Save summary
    valid_stats = [s for s in all_stats if "error" not in s]
    summary: Dict[str, Any] = {
        "game": "diplomacy",
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "opponent_model": args.opponent_model,
        "agent_type": "decision_agent_discrete",
        "unchosen_strategy": args.unchosen_strategy,
        "total_episodes": len(all_stats),
        "successful_episodes": len(valid_stats),
        "elapsed_seconds": elapsed,
        "per_power": args.per_power,
        "episode_stats": all_stats,
    }
    if valid_stats:
        rewards = [s["total_reward"] for s in valid_stats]
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)

    summary_path = game_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Master summary (matches run_qwen3_8b_eval output structure)
    master_dir = base_dir / timestamp
    master_dir.mkdir(parents=True, exist_ok=True)
    master_summary = {
        "benchmark": "AgentEvolver",
        "run_started": timestamp,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "opponent_model": args.opponent_model,
        "agent_type": "decision_agent_discrete",
        "episodes_per_game": args.episodes,
        "temperature": args.temperature,
        "unchosen_strategy": args.unchosen_strategy,
        "total_elapsed_seconds": elapsed,
        "games_completed": ["diplomacy"],
        "results": [summary],
    }
    master_path = master_dir / "eval_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 70}")
    print(f"  DIPLOMACY DISCRETE-ACTION EVAL COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Episodes: {len(valid_stats)}/{len(all_stats)} successful")
    print(f"  Elapsed:  {elapsed:.1f}s")
    if valid_stats:
        print(f"  Mean reward: {summary['mean_reward']:.2f}")
    print(f"  Output:   {game_dir}")
    print(f"  Summary:  {master_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
