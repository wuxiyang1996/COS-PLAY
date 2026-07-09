#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Avalon evaluation with prompt format matched to the co-evolution training loop.

Key alignment with trainer/coevolution/episode_runner.py:
  - SYSTEM_PROMPT: identical to training (choose by NUMBER, ACTION: <number>)
  - Numbered discrete actions: built the same way as _AvalonAdapter._build_actions
  - Compact state: build_rag_summary() instead of raw NL observation
  - Skill guidance: richer format with protocol step highlighting
  - Recent actions / rewards injected into prompt
  - Response parsing: SUBGOAL + REASONING + ACTION:<number>

Opponents still use the free-text GPT prompt (no LoRA, no numbered actions).

Usage (from Game-AI-Agent root):
    python -m scripts.run_qwen3_avalon_matched \\
        --model qwen3-8b-avalon-best \\
        --episodes 50 --per_role \\
        --opponent_model gpt-5.4 \\
        --bank /path/to/combined_skill_bank.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"

for p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from data_structure.experience import Experience, Episode, Episode_Buffer

from decision_agents.agent_helper import (
    get_state_summary,
    infer_intention,
    strip_think_tags,
    build_rag_summary,
    extract_game_facts,
    HARD_SUMMARY_CHAR_LIMIT,
)
from decision_agents.agent_helper import (
    select_skill_from_bank,
    skill_bank_to_text,
)

try:
    from API_func import ask_model, ask_vllm
except ImportError:
    ask_model = None
    ask_vllm = None

try:
    from env_wrappers.avalon_nl_wrapper import AvalonNLWrapper
except ImportError:
    AvalonNLWrapper = None

try:
    from skill_library.skill_bank_mvp import SkillBankMVP
except ImportError:
    try:
        # Fallback to the canonical location used by the rest of the project.
        from skill_agents.skill_bank.bank import SkillBankMVP
    except ImportError:
        SkillBankMVP = None

try:
    from skill_library.skill_query_engine import SkillQueryEngine
except ImportError:
    try:
        from skill_agents.query import SkillQueryEngine
    except ImportError:
        SkillQueryEngine = None

# ---------------------------------------------------------------------------
# Constants — exact copies from training (episode_runner.py)
# ---------------------------------------------------------------------------

TRAINING_SYSTEM_PROMPT = (
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

OPPONENT_SYSTEM_PROMPT = (
    "You are an expert Avalon player powered by Qwen3-8B.\n"
    "You receive the current game state for a specific player and must choose an action.\n\n"
    "Before choosing, briefly reason about:\n"
    "1. What you know about other players' roles based on observations so far.\n"
    "2. Your immediate [TAG] sub-goal (e.g. [ATTACK] Sabotage quest, [DEFEND] Protect team, [EXPLORE] Identify evil).\n"
    "3. What information your action reveals and whether that helps or hurts your team.\n\n"
    "Phase actions:\n"
    "- Team Selection (leader): comma-separated player IDs, e.g. '0, 2, 3'\n"
    "- Team Voting: 'approve' or 'reject'\n"
    "- Quest Voting (on team): 'pass' or 'fail'\n"
    "- Assassination (Assassin only): a player ID, e.g. '2'\n"
    "- Not your turn: 'wait'\n\n"
    "Output format (strict):\n"
    "REASONING: <1-3 sentences>\n"
    "ACTION: <your action>"
)

OPPONENT_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "Choose your action using the format above."
)

EVOLVER_GAME_INFO = {
    "avalon": {
        "task": "Win the game of Avalon for your team by completing quests (good) or sabotaging them (evil).",
    },
}


# ---------------------------------------------------------------------------
# Skill bank loading (same as run_qwen3_8b_eval.py)
# ---------------------------------------------------------------------------

def load_skill_bank(bank_path: str, use_query_engine: bool = True):
    if SkillBankMVP is None:
        print("[load_skill_bank] SkillBankMVP not available.")
        return None, None

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
            print(f"[load_skill_bank] WARNING: no .jsonl found in {bp}")
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
    skill_bank: Any, state_text: str, game_name: str = "",
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


# ---------------------------------------------------------------------------
# Action list builder — mirrors _AvalonAdapter._build_actions exactly
# ---------------------------------------------------------------------------

def build_avalon_actions(info: dict, num_players: int = 5) -> List[str]:
    """Build discrete action list from env info, same as training."""
    phase = info.get("phase", -1)
    if phase == 1:
        return ["approve", "reject"]
    if phase == 2:
        return ["pass", "fail"]
    if phase == 0:
        team_size = info.get("team_size", 2)
        combos = list(combinations(range(num_players), team_size))
        return [",".join(str(p) for p in c) for c in combos[:15]]
    if phase == 3:
        return [str(i) for i in range(num_players)]
    return ["wait"]


def _format_numbered_actions(action_names: List[str]) -> str:
    """Same as training: numbered list with 2-space indent."""
    return "\n".join(f"  {i+1}. {a}" for i, a in enumerate(action_names))


# ---------------------------------------------------------------------------
# Skill guidance formatting — richer version matching training
# ---------------------------------------------------------------------------

def format_skill_guidance_for_prompt(
    guidance: Optional[Dict[str, Any]],
    protocol_step_idx: int = 0,
    progress_summary: str = "",
) -> str:
    """Match episode_runner._format_skill_guidance_for_prompt exactly."""
    if guidance is None or not guidance.get("skill_id"):
        return ""
    parts = [f"\n--- Active Skill: {guidance.get('skill_name', guidance['skill_id'])} ---"]
    if guidance.get("execution_hint"):
        parts.append(f"  Strategy: {guidance['execution_hint'][:200]}")
    if progress_summary:
        parts.append(f"  Progress: {progress_summary}")
    protocol = guidance.get("protocol", {})
    steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
    if steps:
        parts.append(f"  Plan ({len(steps)} steps):")
        for i, step in enumerate(steps[:7], 1):
            marker = ">>" if (i - 1) == protocol_step_idx else "  "
            parts.append(f"  {marker} {i}. {step}")
    preconditions = protocol.get("preconditions", []) if isinstance(protocol, dict) else []
    if preconditions:
        parts.append(f"  Preconditions: {'; '.join(preconditions[:3])}")
    success = protocol.get("success_criteria", []) if isinstance(protocol, dict) else []
    if success:
        parts.append(f"  Done when: {'; '.join(success[:2])}")
    abort = protocol.get("abort_criteria", []) if isinstance(protocol, dict) else []
    if abort:
        parts.append(f"  Abort if: {'; '.join(abort[:2])}")
    parts.append("--- end skill ---\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Recent actions context — same as training _build_recent_context
# ---------------------------------------------------------------------------

def _build_recent_context(recent_actions: List[str], recent_rewards: List[float]) -> str:
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


# ---------------------------------------------------------------------------
# Response parsing — same as training _parse_action_response
# ---------------------------------------------------------------------------

def _fuzzy_match_action(raw: str, valid_actions: List[str]) -> Optional[str]:
    if not raw or not valid_actions:
        return None
    raw_lower = raw.lower().rstrip(".").strip()
    lower_map = {a.lower(): a for a in valid_actions}
    if raw_lower in lower_map:
        return lower_map[raw_lower]
    for va in valid_actions:
        if va.lower() in raw_lower or raw_lower in va.lower():
            return va
    try:
        idx = int(raw_lower) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx]
    except ValueError:
        pass
    return None


def parse_training_response(
    reply: str, valid_actions: List[str],
) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse model output in training format: SUBGOAL/REASONING/ACTION:<number>.

    Returns (action_text, reasoning, intention).
    """
    fallback = valid_actions[0] if valid_actions else "wait"

    if not reply:
        return fallback, None, None

    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply

    intention = None
    subgoal_m = re.search(
        r"SUBGOAL\s*:\s*(.+?)(?=\nREASONING|\nACTION|\Z)",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if subgoal_m:
        raw_sg = subgoal_m.group(1).strip().split("\n")[0].strip()
        intention = raw_sg[:150] if raw_sg else None

    reasoning = None
    reasoning_m = re.search(
        r"REASONING\s*:\s*(.+?)(?=\nACTION|\Z)",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    action_m = re.search(r"ACTION\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if action_m:
        raw = action_m.group(1).strip()
        matched = _fuzzy_match_action(raw, valid_actions)
        if matched:
            return matched, reasoning, intention

    return fallback, reasoning, intention


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_controlled_player_prompt(
    obs_nl: str,
    info: dict,
    num_players: int,
    skill_guidance: Optional[Dict[str, Any]],
    recent_actions: List[str],
    recent_rewards: List[float],
    intention: str = "",
    step_idx: int = 0,
    total_steps: int = 100,
    reward: float = 0.0,
) -> Tuple[str, List[str]]:
    """Build the prompt for the controlled player matching training format.

    Returns (full_prompt, action_names_list).
    """
    action_names = build_avalon_actions(info, num_players)

    summary_state = build_rag_summary(
        obs_nl, "avalon",
        step_idx=step_idx, total_steps=total_steps, reward=reward,
    )

    skill_text = format_skill_guidance_for_prompt(skill_guidance)
    recent_context = _build_recent_context(recent_actions, recent_rewards)

    skill_context = ""
    if skill_guidance and skill_guidance.get("skill_id"):
        sk_name = skill_guidance.get("skill_name", skill_guidance["skill_id"])
        sk_hint = skill_guidance.get("execution_hint", "")
        skill_context = f"Active skill: {sk_name}"
        if sk_hint:
            skill_context += f" — {sk_hint[:100]}"
        skill_context += "\n"

    subgoal_line = f"Subgoal: {intention}\n" if intention else ""

    action_user = (
        f"Game state:\n\n{summary_state}\n\n"
        f"{subgoal_line}"
        f"{skill_context}{recent_context}"
        f"Available actions (pick ONE by number):\n{_format_numbered_actions(action_names)}\n\n"
        f"Output ACTION number."
    )
    full_prompt = TRAINING_SYSTEM_PROMPT + skill_text + "\n" + action_user
    return full_prompt, action_names


def build_opponent_prompt(obs_nl: str) -> str:
    """Build the free-text prompt for GPT opponents (no numbered actions)."""
    return (
        OPPONENT_SYSTEM_PROMPT + "\n\n"
        + OPPONENT_USER_TEMPLATE.format(state=obs_nl)
    )


# ---------------------------------------------------------------------------
# Action callers
# ---------------------------------------------------------------------------

def controlled_player_action(
    obs_nl: str,
    info: dict,
    num_players: int,
    model: str,
    temperature: float,
    skill_guidance: Optional[Dict[str, Any]],
    recent_actions: List[str],
    recent_rewards: List[float],
    intention: str = "",
    step_idx: int = 0,
    total_steps: int = 100,
    reward: float = 0.0,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Call the LoRA model with training-format prompt. Returns (action, reasoning, intention)."""
    if ask_model is None:
        return "wait", None, None

    prompt, action_names = build_controlled_player_prompt(
        obs_nl, info, num_players, skill_guidance,
        recent_actions, recent_rewards, intention,
        step_idx, total_steps, reward,
    )

    try:
        reply = ask_model(prompt, model=model, temperature=temperature, max_tokens=128)
        if not reply or reply.startswith("Error"):
            return action_names[0] if action_names else "wait", None, None
    except Exception as exc:
        print(f"    [WARN] Controlled player call failed ({exc})")
        return action_names[0] if action_names else "wait", None, None

    action, reasoning, parsed_intention = parse_training_response(reply, action_names)
    return action, reasoning, parsed_intention


def opponent_action(
    obs_nl: str,
    model: str,
    temperature: float,
) -> Tuple[str, Optional[str]]:
    """Call the opponent model with free-text prompt. Returns (action, reasoning)."""
    if ask_model is None:
        return "wait", None

    prompt = build_opponent_prompt(obs_nl)
    try:
        reply = ask_model(prompt, model=model, temperature=temperature, max_tokens=1024)
        if not reply or reply.startswith("Error"):
            return "wait", None
    except Exception as exc:
        print(f"    [WARN] Opponent call failed ({exc})")
        return "wait", None

    cleaned = strip_think_tags(reply) or reply
    reasoning = None
    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nACTION|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    action_m = re.search(r"ACTION\s*:\s*(.+)", cleaned, re.IGNORECASE)
    action = action_m.group(1).strip() if action_m else cleaned.strip().split("\n")[-1].strip()
    return action, reasoning


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_avalon_episode(
    model: str,
    temperature: float = 0.4,
    verbose: bool = False,
    num_players: int = 5,
    seed: int = 42,
    skill_bank: Any = None,
    controlled_player: Optional[int] = None,
    opponent_model: Optional[str] = None,
    max_steps: int = 100,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one Avalon episode with training-matched prompts for the controlled player.

    The controlled player uses the same prompt format as the co-evolution
    training loop (SYSTEM_PROMPT + compact state + numbered actions).
    Opponent players use the free-text GPT format.
    """
    if AvalonNLWrapper is None:
        raise ImportError("AvalonNLWrapper not available.")

    mixed_model = controlled_player is not None and opponent_model is not None
    task = EVOLVER_GAME_INFO["avalon"]["task"]
    env = AvalonNLWrapper(num_players=num_players, seed=seed)
    obs, info = env.reset()

    cp_role_name = cp_side = None
    if controlled_player is not None:
        roles = env.roles
        cp_role_id, cp_role_name, cp_is_good = roles[controlled_player]
        cp_side = "good" if cp_is_good else "evil"
        if verbose:
            opp_tag = f", opponents={opponent_model}" if mixed_model else ""
            print(f"    controlled player {controlled_player} = {cp_role_name} ({cp_side}){opp_tag}")

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    recent_actions: List[str] = []
    recent_rewards: List[float] = []
    current_intention = ""

    while not env.done and step_count < max_steps:
        active = info.get("active_players", [])
        actions: Dict[int, Any] = {}
        step_reasonings: List[str] = []

        players_to_query = [
            (pid, obs.get(pid, "")) for pid in active if obs.get(pid, "")
        ]

        guidance = get_skill_guidance(skill_bank, next((s for _, s in players_to_query), ""), game_name="avalon")

        with ThreadPoolExecutor(max_workers=max(len(players_to_query), 1)) as pool:
            futures = {}
            for pid, state_nl in players_to_query:
                if pid == controlled_player:
                    f = pool.submit(
                        controlled_player_action,
                        state_nl, info, num_players,
                        model, temperature, guidance,
                        recent_actions, recent_rewards,
                        current_intention,
                        step_count, max_steps, total_reward,
                    )
                elif mixed_model:
                    f = pool.submit(opponent_action, state_nl, opponent_model, temperature)
                else:
                    f = pool.submit(
                        controlled_player_action,
                        state_nl, info, num_players,
                        model, temperature, guidance,
                        recent_actions, recent_rewards,
                        current_intention,
                        step_count, max_steps, total_reward,
                    )
                futures[f] = pid

            for future in as_completed(futures):
                pid = futures[future]
                try:
                    result = future.result()
                    if pid == controlled_player or not mixed_model:
                        action, reasoning, parsed_intention = result
                        if parsed_intention:
                            current_intention = parsed_intention
                    else:
                        action, reasoning = result
                except Exception as exc:
                    print(f"    [WARN] Player {pid} call failed ({exc})")
                    action, reasoning = "wait", None
                actions[pid] = action
                if reasoning:
                    step_reasonings.append(f"Player {pid}: {reasoning}")
                if verbose:
                    tag = ""
                    if mixed_model:
                        tag = f" [{model}]" if pid == controlled_player else f" [{opponent_model}]"
                    short = (reasoning[:80] + "...") if reasoning and len(reasoning) > 80 else reasoning
                    print(f"  Player {pid}{tag} action={action!r}  reason={short}")

        next_obs, rewards, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated
        reward_val = sum(rewards.values()) if isinstance(rewards, dict) else 0.0
        total_reward += reward_val

        cp_action = actions.get(controlled_player, "wait")
        recent_actions.append(cp_action)
        recent_rewards.append(reward_val)

        combined_reasoning = "\n".join(step_reasonings) if step_reasonings else None

        structured = info.get("structured_state")
        state_summary = get_state_summary("", structured_state=structured) if structured else ""

        exp = Experience(
            state=json.dumps({str(k): v for k, v in obs.items()}, ensure_ascii=False, default=str),
            action=json.dumps({str(k): v for k, v in actions.items()}, ensure_ascii=False, default=str),
            reward=float(reward_val),
            next_state=json.dumps(
                {str(k): v for k, v in next_obs.items()}, ensure_ascii=False, default=str
            ) if isinstance(next_obs, dict) else str(next_obs),
            done=done,
            intentions=current_intention or combined_reasoning,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.summary_state = state_summary if state_summary else None
        exp.interface = {"env_name": "avalon", "game_name": "avalon", "num_players": num_players}
        exp.summary = state_summary or None

        experiences.append(exp)

        if verbose:
            phase = next_info.get("phase_name", next_info.get("phase", ""))
            print(f"  step {step_count}: reward={reward_val:.2f}, cum={total_reward:.2f}, phase={phase}")

        obs = next_obs
        info = next_info
        step_count += 1
        if done:
            break

    agent_type = "decision_agent_matched" if mixed_model else "qwen3_8b_matched"
    meta = {
        "done": True,
        "steps": step_count,
        "total_reward": total_reward,
        "model": model,
        "agent_type": agent_type,
        "prompt_format": "training_matched",
        "good_victory": info.get("good_victory"),
    }
    if controlled_player is not None:
        meta["controlled_player"] = controlled_player
        meta["role_name"] = cp_role_name
        meta["role_side"] = cp_side
    if opponent_model:
        meta["opponent_model"] = opponent_model

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="avalon",
        game_name="avalon",
        metadata=meta,
    )
    episode.set_outcome()

    stats = {
        "game": "avalon",
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": env.done,
        "truncated": False,
        "model": model,
        "agent_type": agent_type,
        "prompt_format": "training_matched",
        "good_victory": info.get("good_victory"),
    }
    if controlled_player is not None:
        stats["controlled_player"] = controlled_player
        stats["role_name"] = cp_role_name
        stats["role_side"] = cp_side
    if opponent_model:
        stats["opponent_model"] = opponent_model
    return episode, stats


# ---------------------------------------------------------------------------
# Evaluation loop with per-role rotation and stats aggregation
# ---------------------------------------------------------------------------

def run_evaluation(args) -> None:
    print(f"\n{'='*60}")
    print(f"  Avalon Evaluation (training-matched prompts)")
    print(f"{'='*60}")
    print(f"  Model:          {args.model}")
    print(f"  Opponent:       {args.opponent_model}")
    print(f"  Episodes:       {args.episodes}")
    print(f"  Per-role:       {args.per_role}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Seed:           {args.seed}")
    print(f"{'='*60}\n")

    skill_bank = None
    if args.bank and not args.no_bank:
        bank, _ = load_skill_bank(args.bank)
        skill_bank = bank

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) if args.output_dir else (CODEBASE_ROOT / "output" / f"avalon_matched_{ts}")
    game_dir = output_base / "avalon" / ts
    game_dir.mkdir(parents=True, exist_ok=True)

    all_stats: List[Dict[str, Any]] = []
    all_episodes: List[Episode] = []

    for ep_idx in range(args.episodes):
        seed = args.seed + ep_idx

        cp = None
        if args.per_role:
            cp = ep_idx % args.num_players

        opp_model = args.opponent_model

        t0 = time.time()
        try:
            episode, stats = run_avalon_episode(
                model=args.model,
                temperature=args.temperature,
                verbose=args.verbose,
                num_players=args.num_players,
                seed=seed,
                skill_bank=skill_bank,
                controlled_player=cp,
                opponent_model=opp_model,
            )
        except Exception:
            traceback.print_exc()
            print(f"  [ERROR] Episode {ep_idx} failed, skipping.")
            continue
        elapsed = time.time() - t0

        stats["episode_idx"] = ep_idx
        stats["elapsed_s"] = round(elapsed, 1)
        all_stats.append(stats)
        all_episodes.append(episode)

        ep_file = game_dir / f"episode_{ep_idx:03d}.json"
        ep_file.write_text(json.dumps(episode.to_dict(), indent=2, default=str), encoding="utf-8")

        gv = stats.get("good_victory")
        side = stats.get("role_side", "?")
        role = stats.get("role_name", "?")
        won = (gv is True and side == "good") or (gv is False and side == "evil")
        tag = "WIN" if won else "LOSS" if gv is not None else "?"
        print(
            f"  ep {ep_idx:3d}: player={cp} role={role}({side}) "
            f"gv={gv} {tag}  reward={stats['total_reward']:.2f}  "
            f"steps={stats['steps']}  {elapsed:.1f}s"
        )

    # --- Summary stats ---
    if not all_stats:
        print("\n  No episodes completed.")
        return

    n = len(all_stats)
    avg_reward = sum(s["total_reward"] for s in all_stats) / n
    avg_steps = sum(s["steps"] for s in all_stats) / n

    good_eps = [s for s in all_stats if s.get("role_side") == "good"]
    evil_eps = [s for s in all_stats if s.get("role_side") == "evil"]

    def win_rate(eps):
        if not eps:
            return 0.0, 0
        wins = sum(1 for s in eps if
                   (s.get("good_victory") is True and s.get("role_side") == "good") or
                   (s.get("good_victory") is False and s.get("role_side") == "evil"))
        return wins / len(eps), len(eps)

    good_wr, good_n = win_rate(good_eps)
    evil_wr, evil_n = win_rate(evil_eps)
    overall_wr, _ = win_rate(all_stats)

    gv_count = sum(1 for s in all_stats if s.get("good_victory") is True)
    ev_count = sum(1 for s in all_stats if s.get("good_victory") is False)

    role_stats: Dict[str, List[Dict]] = {}
    for s in all_stats:
        rn = s.get("role_name", "?")
        role_stats.setdefault(rn, []).append(s)

    summary = {
        "model": args.model,
        "opponent_model": args.opponent_model,
        "prompt_format": "training_matched",
        "num_episodes": n,
        "avg_reward": round(avg_reward, 4),
        "avg_steps": round(avg_steps, 1),
        "overall_win_rate": round(overall_wr, 4),
        "good_victories": gv_count,
        "evil_victories": ev_count,
        "good_win_rate": round(good_wr, 4),
        "good_episodes": good_n,
        "evil_win_rate": round(evil_wr, 4),
        "evil_episodes": evil_n,
        "per_role": {},
        "per_episode": all_stats,
    }

    for rn, reps in sorted(role_stats.items()):
        wr, rn_count = win_rate(reps)
        summary["per_role"][rn] = {
            "episodes": rn_count,
            "win_rate": round(wr, 4),
            "avg_reward": round(sum(s["total_reward"] for s in reps) / rn_count, 4),
        }

    summary_file = output_base / ts / "eval_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    buf = Episode_Buffer(buffer_size=max(len(all_episodes), 1))
    for ep in all_episodes:
        buf.add_episode(ep)
    buf_file = game_dir / "episode_buffer.json"
    buf_file.write_text(json.dumps(buf.to_dict(), indent=2, default=str), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  RESULTS ({n} episodes, training-matched prompts)")
    print(f"{'='*60}")
    print(f"  Average Reward:   {avg_reward:.4f}")
    print(f"  Average Steps:    {avg_steps:.1f}")
    print(f"  Overall Win Rate: {overall_wr:.1%}")
    print(f"  Good Victories:   {gv_count}/{n}  ({gv_count/n:.1%})")
    print(f"  Evil Victories:   {ev_count}/{n}  ({ev_count/n:.1%})")
    print(f"  Good Win Rate:    {good_wr:.1%}  ({good_n} eps)")
    print(f"  Evil Win Rate:    {evil_wr:.1%}  ({evil_n} eps)")
    print(f"  Per Role:")
    for rn, rd in sorted(summary["per_role"].items()):
        print(f"    {rn:12s}: {rd['win_rate']:.1%}  ({rd['episodes']} eps, avg_r={rd['avg_reward']:.3f})")
    print(f"\n  Output:  {output_base}")
    print(f"  Summary: {summary_file}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Avalon evaluation with training-matched prompt format",
    )
    parser.add_argument("--model", type=str, default="qwen3-8b-avalon-best",
                        help="Model name served by vLLM (default: qwen3-8b-avalon-best)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Total episodes to run (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature (default: 0.4)")
    parser.add_argument("--num_players", type=int, default=5,
                        help="Number of Avalon players (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--opponent_model", type=str, default="gpt-5.4",
                        help="Model for opponent players (default: gpt-5.4)")
    parser.add_argument("--per_role", action="store_true",
                        help="Cycle controlled player through 0..N-1")
    parser.add_argument("--bank", type=str, default=None,
                        help="Path to skill bank .jsonl file or directory")
    parser.add_argument("--no-bank", "--no_bank", action="store_true",
                        help="Disable skill bank even if --bank is provided")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step details")
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
