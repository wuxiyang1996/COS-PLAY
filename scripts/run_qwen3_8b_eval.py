#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate Qwen3-8B as a decision agent on GamingAgent (LMGame-Bench) games.

Uses the GamingAgentNLWrapper + gym_like env from evaluate_gamingagent,
and calls get_state_summary / infer_intention from decision_agents.agent_helper
at every step so that rollouts contain rich intention + summary fields.

Output is saved in Episode / Experience format (data_structure.experience) to:
    Game-AI-Agent/output/<model_name>/<game_name>/<YYYYMMDD_HHMMSS>/
        episode_NNN.json        Per-episode JSON
        episode_buffer.json     All episodes in Episode_Buffer format
        rollouts.jsonl          Append-friendly JSONL
        rollout_summary.json    Per-game run stats

Storage layout is model/game/timestamp (timestamp = run start), e.g.:
    output/Qwen3-8B/twenty_forty_eight/20260312_143025/episode_000.json

Requirements:
    - A vLLM server serving Qwen/Qwen3-8B on an OpenAI-compatible endpoint.
      Set VLLM_BASE_URL env var (default: http://localhost:8000/v1).
    - GamingAgent repo on PYTHONPATH.

Usage (from Game-AI-Agent root):

    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
    export VLLM_BASE_URL="http://localhost:8000/v1"

    # Single game, 3 episodes
    python -m scripts.run_qwen3_8b_eval --games twenty_forty_eight --episodes 3

    # All available games, 10 episodes each
    python -m scripts.run_qwen3_8b_eval --episodes 10

    # Resume an interrupted run
    python -m scripts.run_qwen3_8b_eval --resume

    # With labeling (generates summary/intentions via LLM for each experience)
    python -m scripts.run_qwen3_8b_eval --label --label_model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
from evaluate_gamingagent.gym_like import make_gaming_env, list_games
from evaluate_gamingagent.game_configs import GAME_CONFIGS

from decision_agents.agent_helper import (
    get_state_summary,
    infer_intention,
    strip_think_tags,
    build_rag_summary,
    HARD_SUMMARY_CHAR_LIMIT,
)
from decision_agents.dummy_agent import (
    detect_game,
    extract_action,
    _default_action,
    _parse_valid_actions_from_state,
    GAME_GAMINGAGENT,
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

# Skill bank loading — reuse the same function as run_inference.py
from skill_agents.skill_bank.bank import SkillBankMVP

try:
    from skill_agents.query import SkillQueryEngine
except ImportError:
    SkillQueryEngine = None

# --- Additional environment wrappers (Avalon, Diplomacy, Orak, Sokoban) ---
try:
    from env_wrappers.sokoban_nl_wrapper import SokobanNLWrapper
except ImportError:
    SokobanNLWrapper = None  # type: ignore[assignment,misc]

try:
    from env_wrappers.avalon_nl_wrapper import AvalonNLWrapper
except ImportError:
    AvalonNLWrapper = None  # type: ignore[assignment,misc]

try:
    from env_wrappers.diplomacy_nl_wrapper import (
        DiplomacyNLWrapper,
        parse_orders,
        build_structured_state_summary as _diplo_structured_summary,
    )
except ImportError:
    DiplomacyNLWrapper = None  # type: ignore[assignment,misc]
    parse_orders = None  # type: ignore[assignment]
    _diplo_structured_summary = None  # type: ignore[assignment]

try:
    from evaluate_orak.orak_nl_wrapper import make_orak_env, ORAK_GAMES
except ImportError:
    make_orak_env = None  # type: ignore[assignment]
    ORAK_GAMES = {}  # type: ignore[assignment]

try:
    from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper
except ImportError:
    TetrisMacroActionWrapper = None  # type: ignore[assignment,misc]

# Per-game dependency probes for Orak games that need specific conda envs.
# super_mario runs as a subprocess via ORAK_PYTHON (orak-mario env), so we
# only need the binary to exist — gym-super-mario-bros is NOT required in
# the current process.
# pokemon_red requires pyboy (in game-ai-agent env).
import os as _os
_ORAK_GAME_AVAILABLE: Dict[str, bool] = {}
_orak_python = _os.environ.get("ORAK_PYTHON", "")
if _orak_python and _os.path.isfile(_orak_python) and _os.access(_orak_python, _os.X_OK):
    _ORAK_GAME_AVAILABLE["super_mario"] = True
else:
    try:
        import gym_super_mario_bros  # noqa: F401
        _ORAK_GAME_AVAILABLE["super_mario"] = True
    except ImportError:
        _ORAK_GAME_AVAILABLE["super_mario"] = False

try:
    import pyboy  # noqa: F401
    _ORAK_GAME_AVAILABLE["pokemon_red"] = True
except ImportError:
    _ORAK_GAME_AVAILABLE["pokemon_red"] = False

# ---------------------------------------------------------------------------
# Qwen3-8B model defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen3-8B"

# ---------------------------------------------------------------------------
# Game categories for routing
# ---------------------------------------------------------------------------
LMGAME_BENCH_NAMES = {"twenty_forty_eight", "sokoban", "candy_crush", "tetris"}
EVOLVER_GAME_NAMES = {"avalon", "diplomacy"}
ORAK_EVAL_GAME_NAMES = {"super_mario", "pokemon_red"}

EVOLVER_GAME_INFO: Dict[str, Dict[str, Any]] = {
    "avalon": {
        "task": "Win a game of Avalon through social deduction, strategic voting, and deception.",
        "max_steps": 200,
        "display_name": "Avalon",
    },
    "diplomacy": {
        "task": "Gain the most supply centres in Diplomacy through strategic orders and alliances.",
        "max_steps": 200,
        "display_name": "Diplomacy",
    },
}

DIPLOMACY_POWERS = [
    "AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY",
]

ORAK_EVAL_INFO: Dict[str, Dict[str, Any]] = {
    "super_mario": {
        "task": "Advance Mario as far right as possible. Score = x_pos / 3161 * 100.",
        "max_steps": 100,
        "display_name": "Super Mario (Orak)",
    },
    "pokemon_red": {
        "task": "Progress through Pokemon Red storyline milestones (0-12 flags).",
        "max_steps": 200,
        "display_name": "Pokemon Red (Orak)",
    },
}

def load_skill_bank(
    bank_path: str,
    *,
    use_query_engine: bool = True,
) -> Tuple[Any, Any]:
    """Load a SkillBankMVP from a JSONL file or directory.

    Same logic as scripts/run_inference.py — ensures GPT and Qwen paths
    load and query the skill bank identically.
    """
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
    """Query the skill bank for guidance, using the same select_skill_from_bank
    function as VLMDecisionAgent.

    Returns the guidance dict (with protocol, micro_plan, etc.) or None.
    """
    if skill_bank is None:
        return None
    key = state_text[:500]
    try:
        result = select_skill_from_bank(skill_bank, key, top_k=1)
        if result and result.get("skill_id"):
            # Enrich with skill name / description if missing (SkillQueryEngine
            # returns a minimal dict; ensure both paths produce identical fields)
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
    """Format skill guidance as text to inject into LLM prompts.

    Produces the same information that VLMDecisionAgent._format_active_skill()
    exposes, so both GPT and Qwen see identical skill context.
    """
    if guidance is None or not guidance.get("skill_id"):
        return ""

    parts = [f"\n--- Active Skill: {guidance.get('skill_name', guidance['skill_id'])} ---"]
    if guidance.get("execution_hint"):
        parts.append(f"  Strategy: {guidance['execution_hint'][:120]}")

    protocol = guidance.get("protocol", {})
    steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
    if steps:
        parts.append(f"  Plan ({len(steps)} steps):")
        for i, step in enumerate(steps[:5], 1):
            parts.append(f"    {i}. {step}")

    preconditions = protocol.get("preconditions", []) if isinstance(protocol, dict) else []
    if preconditions:
        parts.append(f"  When: {'; '.join(preconditions[:2])}")

    success = protocol.get("success_criteria", []) if isinstance(protocol, dict) else []
    if success:
        parts.append(f"  Done when: {'; '.join(success[:2])}")

    abort = protocol.get("abort_criteria", []) if isinstance(protocol, dict) else []
    if abort:
        parts.append(f"  Abort if: {'; '.join(abort[:2])}")

    if guidance.get("failure_modes"):
        parts.append(f"  Watch for: {'; '.join(guidance['failure_modes'][:2])}")

    parts.append("--- end skill ---\n")
    return "\n".join(parts)


QWEN_SYSTEM_PROMPT = (
    "You are an expert game-playing agent powered by Qwen3-8B.\n"
    "You receive a textual description of the current game state and must choose exactly one action.\n\n"
    "Before choosing, briefly reason about:\n"
    "1. Key elements of the current state (key=value facts).\n"
    "2. Your immediate [TAG] sub-goal (e.g. [MERGE] Combine tiles, [CLEAR] Remove rows, [SURVIVE] Avoid loss).\n"
    "3. Which valid action best achieves that sub-goal.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences of chain-of-thought>\n"
    "ACTION: <exact action name from the valid actions list>\n"
)

QWEN_USER_TEMPLATE = (
    "Game state:\n\n{state}\n\n"
    "Valid actions: {actions}\n\n"
    "Think step-by-step, then choose one action using the format above."
)

# --- Avalon ---
QWEN_AVALON_SYSTEM = (
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

QWEN_AVALON_USER = (
    "Current game state:\n\n{state}\n\n"
    "Choose your action using the format above."
)

# --- Diplomacy ---
QWEN_DIPLOMACY_SYSTEM = (
    "You are an expert Diplomacy player powered by Qwen3-8B.\n"
    "You control one power and must issue orders for your units this phase.\n\n"
    "Before choosing, briefly reason about:\n"
    "1. Your current territorial position and supply-centre count (key=value facts).\n"
    "2. Your [TAG] sub-goal (e.g. [ATTACK] Capture SER, [DEFEND] Hold BUD, [POSITION] Ally with France).\n"
    "3. Whether to attack, defend, or support, and which borders matter most.\n\n"
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

QWEN_DIPLOMACY_USER = (
    "Current game state:\n\n{state}\n\n"
    "Submit your orders using the format above."
)

# --- Orak (Super Mario, Pokemon Red) ---
QWEN_ORAK_SYSTEM = (
    "You are an expert game-playing agent powered by Qwen3-8B.\n"
    "You receive a textual description of the current game state and must choose exactly one action.\n\n"
    "Before choosing, briefly reason about:\n"
    "1. Key elements of the current state (key=value facts).\n"
    "2. Your immediate [TAG] sub-goal (e.g. [NAVIGATE] Reach flag, [COLLECT] Get coin, [SURVIVE] Avoid enemy).\n"
    "3. Which valid action best achieves that sub-goal.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences of chain-of-thought>\n"
    "ACTION: <exact action name from the valid actions list>"
)

QWEN_ORAK_USER = (
    "Game state:\n\n{state}\n\n"
    "Valid actions: {actions}\n\n"
    "Think step-by-step, then choose one action using the format above."
)


# ---------------------------------------------------------------------------
# Qwen3-8B action function
# ---------------------------------------------------------------------------

def _parse_qwen_response(reply: str, valid_actions: List[str]) -> Tuple[str, Optional[str]]:
    """Parse Qwen response to extract action and reasoning.

    Handles ``<think>...</think>`` blocks emitted by reasoning models.
    """
    if not reply:
        return (valid_actions[0] if valid_actions else "stay"), None

    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply

    reasoning = None

    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nACTION|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    action_m = re.search(r"ACTION\s*:\s*(\S+.*)", cleaned, re.IGNORECASE)
    if action_m:
        raw_action = action_m.group(1).strip().lower()
        lower_map = {a.lower(): a for a in valid_actions}
        canonical = lower_map.get(raw_action)
        if canonical:
            return canonical, reasoning

    extracted = extract_action(cleaned, GAME_GAMINGAGENT, "Valid actions: " + ", ".join(valid_actions))
    if extracted:
        return extracted, reasoning

    return (valid_actions[0] if valid_actions else "stay"), reasoning


def qwen3_agent_action(
    state_nl: str,
    action_names: List[str],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    skill_guidance: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """Query Qwen3-8B via vLLM. Returns (action, reasoning)."""
    if ask_model is None:
        return (action_names[0] if action_names else "stay"), None

    user_content = QWEN_USER_TEMPLATE.format(
        state=state_nl,
        actions=", ".join(action_names),
    )
    skill_text = format_skill_guidance_for_prompt(skill_guidance)
    prompt = QWEN_SYSTEM_PROMPT + skill_text + "\n" + user_content

    try:
        reply = ask_model(prompt, model=model, temperature=temperature, max_tokens=1024)
        if reply and not reply.startswith("Error"):
            return _parse_qwen_response(reply, action_names)
    except Exception as exc:
        print(f"    [WARN] Qwen3-8B call failed ({exc}), using fallback")

    return (action_names[0] if action_names else "stay"), None


# ---------------------------------------------------------------------------
# Avalon: Qwen3 action (one player per call)
# ---------------------------------------------------------------------------

def qwen3_avalon_action(
    state_nl: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    skill_guidance: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """Query Qwen3-8B for one Avalon player. Returns (action_str, reasoning)."""
    if ask_model is None:
        return "wait", None

    skill_text = format_skill_guidance_for_prompt(skill_guidance)
    prompt = QWEN_AVALON_SYSTEM + skill_text + "\n\n" + QWEN_AVALON_USER.format(state=state_nl)
    try:
        reply = ask_model(prompt, model=model, temperature=temperature, max_tokens=1024)
        if not reply or reply.startswith("Error"):
            return "wait", None
    except Exception as exc:
        print(f"    [WARN] Avalon Qwen3 call failed ({exc})")
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
# Diplomacy: Qwen3 action (one power per call)
# ---------------------------------------------------------------------------

def _parse_diplomacy_orders_from_reply(reply: str) -> List[str]:
    """Extract order strings from a Qwen3 reply."""
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


def qwen3_diplomacy_action(
    state_nl: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    skill_guidance: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Optional[str]]:
    """Query Qwen3-8B for one Diplomacy power. Returns (orders_list, reasoning)."""
    if ask_model is None:
        return [], None

    skill_text = format_skill_guidance_for_prompt(skill_guidance)
    prompt = QWEN_DIPLOMACY_SYSTEM + skill_text + "\n\n" + QWEN_DIPLOMACY_USER.format(state=state_nl)
    try:
        reply = ask_model(prompt, model=model, temperature=temperature, max_tokens=1200)
        if not reply or reply.startswith("Error"):
            return [], None
    except Exception as exc:
        print(f"    [WARN] Diplomacy Qwen3 call failed ({exc})")
        return [], None

    cleaned = strip_think_tags(reply) or reply
    reasoning = None
    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nORDERS|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    orders = _parse_diplomacy_orders_from_reply(reply)
    return orders, reasoning


# ---------------------------------------------------------------------------
# Orak (Super Mario, Pokemon Red): Qwen3 action
# ---------------------------------------------------------------------------

def qwen3_orak_action(
    state_nl: str,
    action_names: List[str],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    skill_guidance: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """Query Qwen3-8B for one Orak game step. Returns (action, reasoning)."""
    if ask_model is None:
        return (action_names[0] if action_names else "stay"), None

    user_content = QWEN_ORAK_USER.format(
        state=state_nl,
        actions=", ".join(action_names),
    )
    skill_text = format_skill_guidance_for_prompt(skill_guidance)
    prompt = QWEN_ORAK_SYSTEM + skill_text + "\n" + user_content

    try:
        reply = ask_model(prompt, model=model, temperature=temperature, max_tokens=1024)
        if reply and not reply.startswith("Error"):
            return _parse_qwen_response(reply, action_names)
    except Exception as exc:
        print(f"    [WARN] Orak Qwen3 call failed ({exc}), using fallback")

    return (action_names[0] if action_names else "stay"), None


# ---------------------------------------------------------------------------
# Episode runner: LMGame-Bench (with intention + summary)
# ---------------------------------------------------------------------------

def run_qwen3_episode(
    game: str,
    max_steps: int,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    verbose: bool = False,
    label: bool = True,
    label_model: Optional[str] = None,
    skill_bank: Any = None,
    use_macro: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode with Qwen3-8B, calling get_state_summary and
    infer_intention at every step."""

    game_cfg = GAME_CONFIGS.get(game)
    task = game_cfg.description if game_cfg else f"Play {game}"

    base_env = make_gaming_env(game=game, max_steps=max_steps)
    env = GamingAgentNLWrapper(base_env)

    if use_macro and game == "tetris" and TetrisMacroActionWrapper is not None:
        env = TetrisMacroActionWrapper(env)
        print(f"    [macro] Tetris macro-action wrapper enabled (placement-level actions)")

    obs_nl, info = env.reset()
    action_names = info.get("action_names", [])
    structured_state = info.get("structured_state")

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    last_state_summary = ""
    current_intention = ""

    while step_count < max_steps:
        step_actions = action_names if action_names else ["stay"]

        # --- Call get_state_summary ---
        summary = get_state_summary(
            obs_nl,
            structured_state=structured_state,
            game=GAME_GAMINGAGENT,
            model=model,
        )
        if summary:
            last_state_summary = summary[:HARD_SUMMARY_CHAR_LIMIT]

        # --- Call infer_intention ---
        intention = infer_intention(
            last_state_summary or obs_nl,
            game=GAME_GAMINGAGENT,
            model=model,
            context={
                "last_actions": [e.action for e in experiences[-5:]],
                "task": task,
            },
        )
        if intention:
            current_intention = intention

        # --- Skill guidance (same path as VLMDecisionAgent) ---
        guidance = get_skill_guidance(skill_bank, last_state_summary or obs_nl, game_name=game)

        # --- Get action from LLM ---
        action, reasoning = qwen3_agent_action(
            state_nl=obs_nl,
            action_names=step_actions,
            model=model,
            temperature=temperature,
            skill_guidance=guidance,
        )

        combined_intentions = current_intention

        # --- Step environment ---
        try:
            next_obs_nl, reward, terminated, truncated, next_info = env.step(action)
        except Exception as e:
            print(f"    [ERROR at step {step_count}] env.step failed: {e}")
            break

        done = terminated or truncated
        total_reward += reward
        next_action_names = next_info.get("action_names", action_names)
        next_structured_state = next_info.get("structured_state")

        # --- Build Experience ---
        exp = Experience(
            state=obs_nl,
            action=str(action),
            reward=float(reward),
            next_state=next_obs_nl,
            done=done,
            intentions=combined_intentions if combined_intentions else None,
            tasks=task,
            sub_tasks=None,
        )
        exp.idx = step_count
        exp.action_type = "macro" if (use_macro and game == "tetris") else "primitive"
        exp.summary_state = last_state_summary if last_state_summary else None
        exp.available_actions = list(step_actions) if step_actions else None
        exp.interface = {"env_name": "gamingagent", "game_name": game, "macro_actions": use_macro and game == "tetris"}

        # --- Generate experience summary: key=value facts + strategic note ---
        if ask_model is not None:
            try:
                ss = exp.summary_state or ""
                summary_prompt = (
                    "Compress this game step into a short strategic note (max 10 words). "
                    "Focus on the key threat or opportunity.\n"
                    f"Facts: {ss}\n"
                    f"Action: {action}\n"
                    f"State: {obs_nl[:600]}\n"
                    "Note:"
                )
                raw_note = ask_model(
                    summary_prompt, model=model,
                    temperature=0.2, max_tokens=40,
                )
                if raw_note and not raw_note.startswith("Error"):
                    note = strip_think_tags(raw_note) or raw_note
                    note = note.split("\n")[0].strip().strip('"').strip("'")[:80]
                    exp.summary = f"{ss} | note={note}"[:HARD_SUMMARY_CHAR_LIMIT] if ss else note
                else:
                    exp.summary = ss or None
            except Exception:
                exp.summary = exp.summary_state or None

        experiences.append(exp)

        if verbose:
            intent_short = (current_intention[:60] + "...") if len(current_intention) > 60 else current_intention
            reason_short = (reasoning[:60] + "...") if reasoning and len(reasoning) > 60 else reasoning
            print(
                f"  step {step_count}: action={action}, reward={reward:.2f}, "
                f"cum={total_reward:.2f}\n"
                f"    intention: {intent_short}\n"
                f"    reasoning: {reason_short}\n"
                f"    summary:   {last_state_summary[:80]}"
            )

        obs_nl = next_obs_nl
        action_names = next_action_names
        structured_state = next_structured_state
        step_count += 1

        if done:
            break

    env.close()

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="gamingagent",
        game_name=game,
        metadata={
            "done": terminated or truncated,
            "steps": step_count,
            "total_reward": total_reward,
            "model": model,
            "agent_type": "qwen3_8b",
            "final_intention": current_intention,
            "final_state_summary": last_state_summary,
            "macro_actions": use_macro and game == "tetris",
        },
    )
    episode.set_outcome()

    stats = {
        "game": game,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "qwen3_8b",
        "macro_actions": use_macro and game == "tetris",
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Episode runner: Sokoban (specialized grid prompts, memory, reflection)
# ---------------------------------------------------------------------------

def run_qwen3_sokoban_episode(
    max_steps: int = 200,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    verbose: bool = False,
    label: bool = True,
    reflect_every: int = 3,
    skill_bank: Any = None,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one Sokoban episode using the SokobanNLWrapper with domain-specific
    grid prompts, rolling memory, and periodic reflection — all powered by Qwen3."""

    task = "Push all boxes onto target locations in Sokoban"
    game_cfg = GAME_CONFIGS.get("sokoban")
    if game_cfg:
        task = game_cfg.description

    base_env = make_gaming_env(game="sokoban", max_steps=max_steps)
    env = SokobanNLWrapper(base_env, reflect_every=reflect_every)

    obs_nl, info = env.reset()

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    last_state_summary = ""
    current_intention = ""

    while step_count < max_steps:
        # --- Optional reflection via Qwen3 ---
        if env.should_reflect() and ask_model is not None:
            env.generate_reflection(ask_model, model)

        # --- Skill guidance (same path as VLMDecisionAgent) ---
        guidance = get_skill_guidance(skill_bank, obs_nl, game_name="sokoban")
        skill_text = format_skill_guidance_for_prompt(guidance)

        # --- Build Sokoban-specific prompt and query LLM ---
        user_prompt = env.build_user_prompt()
        system_prompt = env.system_prompt
        full_prompt = system_prompt + skill_text + "\n\n" + user_prompt

        action = "up"
        reasoning = None
        if ask_model is not None:
            try:
                reply = ask_model(
                    full_prompt, model=model,
                    temperature=temperature, max_tokens=1024,
                )
                if reply and not reply.startswith("Error"):
                    cleaned = strip_think_tags(reply) or reply
                    action = env.parse_action(cleaned)
                    reasoning = env.parse_reasoning(cleaned)
            except Exception as exc:
                if verbose:
                    print(f"    [WARN] Sokoban Qwen3 call failed ({exc}), using fallback")

        # --- State summary (deterministic) ---
        summary = get_state_summary(
            obs_nl,
            structured_state=info.get("structured_state"),
            game=GAME_GAMINGAGENT,
            model=model,
        )
        if summary:
            last_state_summary = summary[:HARD_SUMMARY_CHAR_LIMIT]

        # --- Intention (Qwen3) ---
        intention = infer_intention(
            last_state_summary or obs_nl,
            game=GAME_GAMINGAGENT,
            model=model,
            context={
                "last_actions": [e.action for e in experiences[-5:]],
                "task": task,
            },
        )
        if intention:
            current_intention = intention

        # --- Step environment ---
        try:
            next_obs_nl, reward, terminated, truncated, next_info = env.step(action)
        except Exception as e:
            if verbose:
                print(f"    [ERROR at step {step_count}] env.step failed: {e}")
            break

        done = terminated or truncated
        total_reward += reward

        # --- Build Experience ---
        exp = Experience(
            state=obs_nl,
            action=str(action),
            reward=float(reward),
            next_state=next_obs_nl,
            done=done,
            intentions=current_intention if current_intention else reasoning,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.summary_state = last_state_summary if last_state_summary else None
        exp.available_actions = list(env.action_names)
        exp.interface = {"env_name": "gamingagent", "game_name": "sokoban"}

        if ask_model is not None:
            try:
                ss = exp.summary_state or ""
                grid_ctx = info.get("grid_string", obs_nl[:600])
                summary_prompt = (
                    "Compress this Sokoban step into a short strategic note (max 10 words). "
                    "Focus on the key threat or opportunity.\n"
                    f"Facts: {ss}\n"
                    f"Board:\n{grid_ctx}\n"
                    f"Action: {action}\n"
                    "Note:"
                )
                raw_note = ask_model(
                    summary_prompt, model=model,
                    temperature=0.2, max_tokens=40,
                )
                if raw_note and not raw_note.startswith("Error"):
                    note = strip_think_tags(raw_note) or raw_note
                    note = note.split("\n")[0].strip().strip('"').strip("'")[:80]
                    exp.summary = f"{ss} | note={note}"[:HARD_SUMMARY_CHAR_LIMIT] if ss else note
                else:
                    exp.summary = ss or None
            except Exception:
                exp.summary = exp.summary_state or None

        experiences.append(exp)

        if verbose:
            intent_short = (current_intention[:60] + "...") if len(current_intention) > 60 else current_intention
            reason_short = (reasoning[:60] + "...") if reasoning and len(reasoning) > 60 else reasoning
            progress = info.get("element_summary", "")
            prog_line = [l for l in progress.splitlines() if "Progress" in l]
            print(
                f"  step {step_count}: action={action}, reward={reward:.2f}, "
                f"cum={total_reward:.2f}"
                + (f", {prog_line[0].strip()}" if prog_line else "")
                + f"\n    reasoning: {reason_short}"
            )

        obs_nl = next_obs_nl
        info = next_info
        step_count += 1

        if done:
            break

    env.close()

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="gamingagent",
        game_name="sokoban",
        metadata={
            "done": terminated or truncated,
            "steps": step_count,
            "total_reward": total_reward,
            "model": model,
            "agent_type": "qwen3_8b_sokoban",
            "final_intention": current_intention,
            "final_state_summary": last_state_summary,
        },
    )
    episode.set_outcome()

    stats = {
        "game": "sokoban",
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "qwen3_8b_sokoban",
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Episode runner: Avalon (multi-agent, natural end condition)
# ---------------------------------------------------------------------------

DIPLOMACY_MAX_PHASES = 20


def _sanitize_keys(obj: Any) -> Any:
    """Recursively convert all dict keys to plain ``str`` for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_keys(v) for v in obj]
    return obj


def run_qwen3_avalon_episode(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    verbose: bool = False,
    num_players: int = 5,
    seed: int = 42,
    skill_bank: Any = None,
    controlled_player: Optional[int] = None,
    opponent_model: Optional[str] = None,
    **kwargs,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one Avalon episode.

    Three modes:
    1. controlled_player=None — all players use *model* with skill bank (self-play).
    2. controlled_player=N, opponent_model="gpt-5.4" — player N uses *model*
       with skill bank guidance, all other players use *opponent_model* (no bank).
    3. controlled_player=N, opponent_model=None — player N uses *model*, others
       use the same model without bank guidance.
    """
    if AvalonNLWrapper is None:
        raise ImportError("AvalonNLWrapper not available. Install AgentEvolver deps.")

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

    while not env.done:
        active = info.get("active_players", [])
        actions: Dict[int, Any] = {}
        step_reasonings: List[str] = []

        players_to_query = [
            (pid, obs.get(pid, "")) for pid in active if obs.get(pid, "")
        ]

        representative_state = next((s for _, s in players_to_query), "")
        guidance = get_skill_guidance(skill_bank, representative_state, game_name="avalon")

        with ThreadPoolExecutor(max_workers=max(len(players_to_query), 1)) as pool:
            futures = {}
            for pid, state_nl in players_to_query:
                if mixed_model and pid != controlled_player:
                    f = pool.submit(qwen3_avalon_action, state_nl, opponent_model, temperature, None)
                else:
                    f = pool.submit(qwen3_avalon_action, state_nl, model, temperature, guidance)
                futures[f] = pid

            for future in as_completed(futures):
                pid = futures[future]
                try:
                    action, reasoning = future.result()
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

        combined_reasoning = "\n".join(step_reasonings) if step_reasonings else None

        structured = info.get("structured_state")
        state_summary = get_state_summary("", structured_state=structured) if structured else ""

        intention = infer_intention(
            state_summary or next(iter(obs.values()), "")[:1500],
            game="avalon", model=model,
            context={"last_actions": [e.action for e in experiences[-5:]], "task": task},
        )

        exp = Experience(
            state=json.dumps({str(k): v for k, v in obs.items()}, ensure_ascii=False, default=str),
            action=json.dumps({str(k): v for k, v in actions.items()}, ensure_ascii=False, default=str),
            reward=float(reward_val),
            next_state=json.dumps(
                {str(k): v for k, v in next_obs.items()}, ensure_ascii=False, default=str
            ) if isinstance(next_obs, dict) else str(next_obs),
            done=done,
            intentions=intention or combined_reasoning,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.summary_state = state_summary if state_summary else None
        exp.interface = {"env_name": "avalon", "game_name": "avalon", "num_players": num_players}

        if ask_model is not None:
            try:
                ss = exp.summary_state or ""
                obs_preview = next(iter(obs.values()), "")[:600] if isinstance(obs, dict) else str(obs)[:600]
                summary_prompt = (
                    "Compress this Avalon step into a short strategic note (max 10 words). "
                    "Focus on the key information revealed or threat.\n"
                    f"Facts: {ss}\n"
                    f"State: {obs_preview}\n"
                    "Note:"
                )
                raw_note = ask_model(
                    summary_prompt, model=model,
                    temperature=0.2, max_tokens=40,
                )
                if raw_note and not raw_note.startswith("Error"):
                    note = strip_think_tags(raw_note) or raw_note
                    note = note.split("\n")[0].strip().strip('"').strip("'")[:80]
                    exp.summary = f"{ss} | note={note}"[:HARD_SUMMARY_CHAR_LIMIT] if ss else note
                else:
                    exp.summary = ss or None
            except Exception:
                exp.summary = exp.summary_state or None

        experiences.append(exp)

        if verbose:
            phase = next_info.get("phase_name", next_info.get("phase", ""))
            print(f"  step {step_count}: reward={reward_val:.2f}, cum={total_reward:.2f}, phase={phase}")

        obs = next_obs
        info = next_info
        step_count += 1
        if done:
            break

    agent_type = "decision_agent_mixed" if mixed_model else "qwen3_8b"
    meta = {
        "done": True,
        "steps": step_count,
        "total_reward": total_reward,
        "model": model,
        "agent_type": agent_type,
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
# Episode runner: Diplomacy (multi-agent, 20-phase cap)
# ---------------------------------------------------------------------------

def _match_power_name(controlled: str, active_names) -> Optional[str]:
    """Case-insensitive match of *controlled* against env power names."""
    ctrl = controlled.upper()
    for name in active_names:
        if str(name).upper() == ctrl:
            return name
    return None


def run_qwen3_diplomacy_episode(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    verbose: bool = False,
    seed: int = 42,
    skill_bank: Any = None,
    controlled_power: Optional[str] = None,
    opponent_model: Optional[str] = None,
    **kwargs,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one Diplomacy episode.

    Two modes:
    1. controlled_power=None — all powers use *model* with skill bank (self-play).
    2. controlled_power="FRANCE", opponent_model="gpt-5.4" — the controlled
       power uses *model* with skill bank, all others use *opponent_model*.
    """
    if DiplomacyNLWrapper is None:
        raise ImportError("DiplomacyNLWrapper not available. Install AI_Diplomacy deps.")

    mixed_model = controlled_power is not None and opponent_model is not None
    task = EVOLVER_GAME_INFO["diplomacy"]["task"]
    env = DiplomacyNLWrapper(seed=seed, max_phases=DIPLOMACY_MAX_PHASES)
    obs, info = env.reset()

    resolved_cp: Optional[str] = None
    if mixed_model:
        all_power_names = list(obs.keys()) if isinstance(obs, dict) else []
        resolved_cp = _match_power_name(controlled_power, all_power_names)
        if resolved_cp is None and all_power_names:
            resolved_cp = all_power_names[0]
            print(f"    [WARN] Could not match '{controlled_power}' in {all_power_names}, "
                  f"falling back to {resolved_cp}")
        if verbose:
            print(f"    controlled power {resolved_cp} = {model}, opponents = {opponent_model}")

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0

    prev_sc_counts: Dict[str, int] = {}
    if env.game is not None:
        prev_sc_counts = {
            pn: len(po.centers) for pn, po in env.game.powers.items()
            if not po.is_eliminated()
        }

    while not env.done:
        actions: Dict[str, Union[List[str], str]] = {}
        step_reasonings: Dict[str, str] = {}

        active_powers = info.get("active_powers", {})
        powers_to_query = [
            (pname, obs[pname]) for pname in obs
            if obs.get(pname) and pname in active_powers
        ]

        representative_state = next((s for _, s in powers_to_query), "")
        guidance = get_skill_guidance(skill_bank, representative_state, game_name="diplomacy")

        with ThreadPoolExecutor(max_workers=max(len(powers_to_query), 1)) as pool:
            futures = {}
            for pname, state_nl in powers_to_query:
                if mixed_model and pname != resolved_cp:
                    f = pool.submit(qwen3_diplomacy_action, state_nl, opponent_model, temperature, None)
                else:
                    f = pool.submit(qwen3_diplomacy_action, state_nl, model, temperature, guidance)
                futures[f] = pname

            for future in as_completed(futures):
                power_name = futures[future]
                try:
                    orders, reasoning = future.result()
                except Exception as exc:
                    print(f"    [WARN] {power_name} call failed ({exc})")
                    orders, reasoning = [], None
                if parse_orders is not None and env.game is not None:
                    orders = parse_orders(orders, env.game, power_name)
                actions[power_name] = orders
                if reasoning:
                    step_reasonings[power_name] = reasoning
                if verbose:
                    tag = ""
                    if mixed_model:
                        tag = f" [{model}]" if power_name == resolved_cp else f" [{opponent_model}]"
                    preview = orders[:3]
                    print(f"  {power_name}{tag}: {len(orders)} orders, e.g. {preview}")

        phase_before = info.get("phase", "")

        next_obs, rewards, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated
        reward_val = sum(rewards.values()) if isinstance(rewards, dict) else 0.0
        total_reward += reward_val

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

        primary_power = resolved_cp or "AUSTRIA"
        structured = info.get("structured_state")
        if env.game is not None and _diplo_structured_summary is not None:
            structured = _diplo_structured_summary(
                env.game, primary_power, prev_sc_counts=prev_sc_counts
            )
        state_summary = get_state_summary("", structured_state=structured) if structured else ""

        primary_obs_text = obs.get(primary_power, "")[:1500] if isinstance(obs, dict) else str(obs)[:1500]
        intention = infer_intention(
            state_summary or primary_obs_text,
            game="diplomacy", model=model,
            context={
                "last_actions": [e.action for e in experiences[-5:]],
                "task": task,
                "power_name": primary_power,
                "phase": phase_before,
                "sc_delta": sc_delta_str,
            },
        )

        combined_reasoning_parts = [f"{pn}: {r}" for pn, r in step_reasonings.items()]
        combined_reasoning = "\n".join(combined_reasoning_parts) if combined_reasoning_parts else None
        primary_reasoning = step_reasonings.get(primary_power)
        effective_intention = intention or primary_reasoning or combined_reasoning

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
        exp.summary_state = state_summary if state_summary else None
        exp.interface = {"env_name": "diplomacy", "game_name": "diplomacy"}

        if ask_model is not None:
            try:
                ss = exp.summary_state or ""
                my_orders = actions.get(primary_power, [])
                orders_str = "; ".join(my_orders) if isinstance(my_orders, list) else str(my_orders)

                summary_prompt = (
                    f"Compress this Diplomacy turn into a short strategic note (max 10 words) "
                    f"for {primary_power}. Focus on what CHANGED — territory gains/losses or threats.\n"
                    f"Facts: {ss}\n"
                    f"Phase: {phase_before}\n"
                    f"Orders: {orders_str[:300]}\n"
                    f"SC changes: {sc_delta_str or 'none'}\n"
                    "Note:"
                )
                raw_note = ask_model(
                    summary_prompt, model=model,
                    temperature=0.2, max_tokens=40,
                )
                if raw_note and not raw_note.startswith("Error"):
                    note = strip_think_tags(raw_note) or raw_note
                    note = note.split("\n")[0].strip().strip('"').strip("'")[:80]
                    exp.summary = f"{ss} | note={note}"[:HARD_SUMMARY_CHAR_LIMIT] if ss else note
                else:
                    exp.summary = ss or None
            except Exception:
                exp.summary = exp.summary_state or None

        experiences.append(exp)

        if verbose:
            phase = next_info.get("phase", "")
            intent_short = (effective_intention[:80] + "...") if effective_intention and len(effective_intention) > 80 else effective_intention
            print(
                f"  step {step_count}: reward={reward_val:.2f}, cum={total_reward:.2f}, "
                f"phase={phase}, sc_delta=[{sc_delta_str}]\n"
                f"    intention: {intent_short}"
            )

        prev_sc_counts = cur_sc_counts
        obs = next_obs
        info = next_info
        step_count += 1
        if done:
            break

    final_rewards = {}
    if isinstance(rewards, dict):
        final_rewards = {str(k): float(v) for k, v in rewards.items()}

    agent_type = "decision_agent_mixed" if mixed_model else "qwen3_8b"
    meta = {
        "done": True,
        "steps": step_count,
        "total_reward": total_reward,
        "model": model,
        "agent_type": agent_type,
        "final_sc_rewards": final_rewards,
    }
    if resolved_cp:
        meta["controlled_power"] = resolved_cp
    if opponent_model:
        meta["opponent_model"] = opponent_model

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="diplomacy",
        game_name="diplomacy",
        metadata=meta,
    )
    episode.set_outcome()

    stats = {
        "game": "diplomacy",
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": agent_type,
        "max_phases": DIPLOMACY_MAX_PHASES,
        "final_sc_rewards": final_rewards,
    }
    if resolved_cp:
        stats["controlled_power"] = resolved_cp
        if isinstance(rewards, dict) and resolved_cp:
            stats["controlled_power_reward"] = float(rewards.get(resolved_cp, 0.0))
    if opponent_model:
        stats["opponent_model"] = opponent_model
    return episode, stats


# ---------------------------------------------------------------------------
# Episode runner: Orak games (Super Mario, Pokemon Red)
# ---------------------------------------------------------------------------

def run_qwen3_orak_episode(
    game: str,
    max_steps: int = 100,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    verbose: bool = False,
    skill_bank: Any = None,
    **kwargs,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one Orak game episode with Qwen3-8B."""
    if make_orak_env is None:
        raise ImportError("Orak env not available. Install Orak deps and activate the correct conda env.")

    orak_info = ORAK_EVAL_INFO.get(game, {})
    task = orak_info.get("task", f"Play {game}")

    env = make_orak_env(game, max_steps=max_steps)
    action_names = env.action_names
    obs, info = env.reset()

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    last_state_summary = ""

    while step_count < max_steps:
        step_actions = action_names if action_names else ["stay"]

        # State summary (deterministic, no LLM)
        summary = get_state_summary(obs)
        if summary:
            last_state_summary = summary[:HARD_SUMMARY_CHAR_LIMIT]

        # Intention via Qwen3
        intention = infer_intention(
            last_state_summary or obs[:1500],
            game="orak", model=model,
            context={"last_actions": [e.action for e in experiences[-5:]], "task": task},
        )

        # Skill guidance (same path as VLMDecisionAgent)
        guidance = get_skill_guidance(skill_bank, last_state_summary or obs[:500], game_name=game)

        action, reasoning = qwen3_orak_action(
            state_nl=obs,
            action_names=step_actions,
            model=model,
            temperature=temperature,
            skill_guidance=guidance,
        )

        try:
            next_obs, reward, terminated, truncated, next_info = env.step(action)
        except Exception as e:
            print(f"    [ERROR at step {step_count}] env.step failed: {e}")
            break

        done = terminated or truncated
        total_reward += reward

        exp = Experience(
            state=obs,
            action=str(action),
            reward=float(reward),
            next_state=next_obs,
            done=done,
            intentions=intention or reasoning,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.summary_state = last_state_summary if last_state_summary else None
        exp.available_actions = list(step_actions)
        step_info = next_info if isinstance(next_info, dict) else {}
        exp.interface = {
            "env_name": "orak",
            "game_name": game,
            "step": step_count,
            "terminated": terminated,
            "truncated": truncated,
            "score": step_info.get("score"),
            "cumulative_reward": total_reward,
        }

        if ask_model is not None:
            try:
                ss = exp.summary_state or ""
                summary_prompt = (
                    f"Compress this {game} step into a short strategic note (max 10 words). "
                    "Focus on the key threat or opportunity.\n"
                    f"Facts: {ss}\n"
                    f"Action: {action}\n"
                    f"State: {obs[:600]}\n"
                    "Note:"
                )
                raw_note = ask_model(
                    summary_prompt, model=model,
                    temperature=0.2, max_tokens=40,
                )
                if raw_note and not raw_note.startswith("Error"):
                    note = strip_think_tags(raw_note) or raw_note
                    note = note.split("\n")[0].strip().strip('"').strip("'")[:80]
                    exp.summary = f"{ss} | note={note}"[:HARD_SUMMARY_CHAR_LIMIT] if ss else note
                else:
                    exp.summary = ss or None
            except Exception:
                pass

        experiences.append(exp)

        if verbose:
            act_short = str(action)[:60]
            reason_short = (reasoning[:80] + "...") if reasoning and len(reasoning) > 80 else reasoning
            term_label = "TERM" if terminated else ("TRUNC" if truncated else "")
            print(f"  step {step_count}: action={act_short}, reward={reward:.3f}, "
                  f"cum={total_reward:.3f} {term_label}  reason={reason_short}")

        obs = next_obs
        step_count += 1
        if done:
            break

    env.close()

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="orak",
        game_name=game,
        metadata={
            "done": terminated or truncated,
            "steps": step_count,
            "total_reward": total_reward,
            "model": model,
            "agent_type": "qwen3_8b",
        },
    )
    episode.set_outcome()

    final_info = next_info if (step_count > 0 and isinstance(next_info, dict)) else {}
    stats = {
        "game": game,
        "steps": step_count,
        "total_reward": total_reward,
        "final_score": final_info.get("score", 0),
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "qwen3_8b",
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Batch rollout helpers
# ---------------------------------------------------------------------------

def count_existing_episodes(game_dir: Path) -> int:
    if not game_dir.exists():
        return 0
    return sum(1 for f in game_dir.glob("episode_*.json") if f.name != "episode_buffer.json")


def save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict[str, Any]):
    record = _sanitize_keys(episode.to_dict())
    record["rollout_metadata"] = stats
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def save_game_summary(
    game_name: str,
    game_dir: Path,
    all_stats: List[Dict[str, Any]],
    elapsed: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "game": game_name,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "qwen3_8b",
        "total_episodes": len(all_stats),
        "target_episodes": args.episodes,
        "max_steps": args.max_steps,
        "labeled": args.label,
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    valid_stats = [s for s in all_stats if "error" not in s]
    if valid_stats:
        rewards = [s["total_reward"] for s in valid_stats]
        steps = [s["steps"] for s in valid_stats]
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["mean_steps"] = sum(steps) / len(steps)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)

    summary_path = game_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def run_game_rollouts(
    game_name: str,
    args: argparse.Namespace,
    game_run_dir: Path,
    skill_bank: Any = None,
    use_macro: bool = False,
) -> Dict[str, Any]:
    """Run all episodes for one game and save outputs. game_run_dir = output/<model>/<game>/<timestamp>."""
    game_dir = game_run_dir
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    start_idx = 0
    if args.resume:
        start_idx = count_existing_episodes(game_dir)
        if start_idx >= args.episodes:
            print(f"  [SKIP] {game_name}: {start_idx}/{args.episodes} episodes already done")
            return {"game": game_name, "skipped": True, "existing": start_idx}
        if start_idx > 0:
            print(f"  [RESUME] {game_name}: resuming from episode {start_idx}")

    # Determine max_steps from the appropriate config source
    game_cfg = GAME_CONFIGS.get(game_name)
    if args.max_steps:
        effective_max_steps = args.max_steps
    elif game_cfg:
        effective_max_steps = game_cfg.max_steps
    elif game_name in EVOLVER_GAME_INFO:
        effective_max_steps = EVOLVER_GAME_INFO[game_name]["max_steps"]
    elif game_name in ORAK_EVAL_INFO:
        effective_max_steps = ORAK_EVAL_INFO[game_name]["max_steps"]
    else:
        effective_max_steps = 50

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    seed_base = getattr(args, "seed", 42) or 42

    for ep_idx in range(start_idx, args.episodes):
        print(f"\n  [{game_name}] Episode {ep_idx + 1}/{args.episodes}")

        try:
            if game_name == "sokoban" and SokobanNLWrapper is not None:
                episode, stats = run_qwen3_sokoban_episode(
                    max_steps=effective_max_steps,
                    model=args.model,
                    temperature=args.temperature,
                    verbose=args.verbose,
                    label=args.label,
                    reflect_every=getattr(args, "reflect_every", 3),
                    skill_bank=skill_bank,
                )
            elif game_name == "avalon":
                opp_model = getattr(args, "opponent_model", None)
                cp = None
                if getattr(args, "per_role", False):
                    num_p = getattr(args, "num_players", 5)
                    cp = ep_idx % num_p
                episode, stats = run_qwen3_avalon_episode(
                    model=args.model,
                    temperature=args.temperature,
                    verbose=args.verbose,
                    num_players=getattr(args, "num_players", 5),
                    seed=seed_base + ep_idx,
                    skill_bank=skill_bank,
                    controlled_player=cp,
                    opponent_model=opp_model,
                )
            elif game_name == "diplomacy":
                opp_model = getattr(args, "opponent_model", None)
                cp_power = None
                if getattr(args, "per_power", False):
                    cp_power = DIPLOMACY_POWERS[ep_idx % len(DIPLOMACY_POWERS)]
                episode, stats = run_qwen3_diplomacy_episode(
                    model=args.model,
                    temperature=args.temperature,
                    verbose=args.verbose,
                    seed=seed_base + ep_idx,
                    skill_bank=skill_bank,
                    controlled_power=cp_power,
                    opponent_model=opp_model,
                )
            elif game_name in ORAK_EVAL_GAME_NAMES:
                episode, stats = run_qwen3_orak_episode(
                    game=game_name,
                    max_steps=effective_max_steps,
                    model=args.model,
                    temperature=args.temperature,
                    verbose=args.verbose,
                    skill_bank=skill_bank,
                )
            else:
                episode, stats = run_qwen3_episode(
                    game=game_name,
                    max_steps=effective_max_steps,
                    model=args.model,
                    temperature=args.temperature,
                    verbose=args.verbose,
                    label=args.label,
                    label_model=args.label_model,
                    skill_bank=skill_bank,
                    use_macro=use_macro,
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

            save_episode_jsonl(episode, jsonl_path, stats)

        except Exception as e:
            print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
            traceback.print_exc()
            all_stats.append({
                "game": game_name,
                "episode_index": ep_idx,
                "error": str(e),
                "steps": 0,
                "total_reward": 0.0,
            })
            continue

    elapsed = time.time() - t0

    # Don't keep empty runs: no successful episodes and no pre-existing episodes from resume
    if len(episode_buffer) == 0 and start_idx == 0:
        if game_dir.exists():
            shutil.rmtree(game_dir)
        print(f"\n  [INVALID] {game_name}: 0 successful episodes, removed output dir")
        return {
            "game": game_name,
            "total_episodes": 0,
            "invalid": True,
            "episode_stats": all_stats,
        }

    buffer_path = game_dir / "episode_buffer.json"
    episode_buffer.save_to_json(str(buffer_path))
    print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    summary = save_game_summary(game_name, game_dir, all_stats, elapsed, args)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _list_all_eval_games() -> List[str]:
    """Return sorted list of all games the eval script can run."""
    games = set()
    try:
        games.update(list_games())
    except Exception:
        games.update(LMGAME_BENCH_NAMES)
    if AvalonNLWrapper is not None:
        games.add("avalon")
    if DiplomacyNLWrapper is not None:
        games.add("diplomacy")
    if make_orak_env is not None:
        for g in ORAK_EVAL_GAME_NAMES:
            games.add(g)
    return sorted(games)


def _game_description(game_name: str) -> str:
    """Return a short description for any supported game."""
    cfg = GAME_CONFIGS.get(game_name)
    if cfg:
        return cfg.description
    info = EVOLVER_GAME_INFO.get(game_name) or ORAK_EVAL_INFO.get(game_name)
    if info:
        return info.get("task", info.get("display_name", ""))
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-8B decision agent evaluation across LMGame-Bench, AgentEvolver, and Orak games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--games", type=str, nargs="+", default=None,
        help="Games to evaluate (default: all available). Use --list-games to see options.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per game (default: 3)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per episode (default: per-game config)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3)")
    parser.add_argument("--label", action="store_true", help="Generate experience summaries via LLM")
    parser.add_argument("--label_model", type=str, default=None, help="Model for labeling (default: same as --model)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print step-by-step details")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--list-games", action="store_true", help="List available games and exit")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for Avalon/Diplomacy (default: 42)")
    parser.add_argument("--num_players", type=int, default=5, help="Number of Avalon players (default: 5)")
    parser.add_argument("--reflect_every", type=int, default=3, help="Sokoban: run reflection every N steps (default: 3)")
    parser.add_argument(
        "--bank", type=str, default=None,
        help="Path to skill bank JSONL file or directory (same format as run_inference.py --bank).",
    )
    parser.add_argument(
        "--no-bank", action="store_true",
        help="Explicitly run without a skill bank (baseline / ablation).",
    )
    parser.add_argument(
        "--no-query-engine", action="store_true",
        help="Disable SkillQueryEngine (use plain SkillBankMVP for skill queries).",
    )
    parser.add_argument(
        "--macro-actions", action="store_true",
        help="Use macro-action wrapper for Tetris (placement-level actions instead of primitives).",
    )
    parser.add_argument(
        "--opponent_model", type=str, default=None,
        help="Model for opponent players/powers in Avalon/Diplomacy (e.g. gpt-5.4). "
             "When set with --per_role or --per_power, the controlled player/power uses "
             "--model with the skill bank while all others use this model.",
    )
    parser.add_argument(
        "--per_role", action="store_true",
        help="Avalon: cycle controlled_player through 0..num_players-1 across episodes.",
    )
    parser.add_argument(
        "--per_power", action="store_true",
        help="Diplomacy: cycle controlled_power through 7 powers across episodes.",
    )

    args = parser.parse_args()

    if args.list_games:
        all_games = _list_all_eval_games()
        print("Available games:")
        for g in all_games:
            desc = _game_description(g)
            cat = ("evolver" if g in EVOLVER_GAME_NAMES
                   else "orak" if g in ORAK_EVAL_GAME_NAMES
                   else "lmgame-bench")
            suffix = f" - {desc}" if desc else ""
            print(f"  {g} [{cat}]{suffix}")
        return

    # Storage: output/<model>/<game>/<timestamp> (timestamp = run start)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        model_name = Path(args.model).name or args.model.split("/")[-1] or "model"
        model_slug = re.sub(r"[^\w\-.]", "_", model_name)
        base_dir = CODEBASE_ROOT / "output" / model_slug
    base_dir.mkdir(parents=True, exist_ok=True)

    all_known_games = _list_all_eval_games()

    if args.games:
        requested = args.games
    else:
        requested = all_known_games

    available_games: List[str] = []
    skipped_games: List[str] = []
    for g in requested:
        # Evolver games
        if g in EVOLVER_GAME_NAMES:
            if g == "avalon" and AvalonNLWrapper is None:
                print(f"[WARNING] Game '{g}' wrapper not importable (install AgentEvolver), skipping.")
                skipped_games.append(g)
            elif g == "diplomacy" and DiplomacyNLWrapper is None:
                print(f"[WARNING] Game '{g}' wrapper not importable (install AI_Diplomacy), skipping.")
                skipped_games.append(g)
            else:
                available_games.append(g)
            continue
        # Orak games
        if g in ORAK_EVAL_GAME_NAMES:
            if make_orak_env is None:
                print(f"[WARNING] Game '{g}' requires Orak env wrappers, skipping.")
                skipped_games.append(g)
            elif not _ORAK_GAME_AVAILABLE.get(g, True):
                hint = {
                    "super_mario": "gym-super-mario-bros (activate orak-mario conda env)",
                    "pokemon_red": "pyboy (activate game-ai-agent conda env)",
                }.get(g, "game-specific deps")
                print(f"[WARNING] Game '{g}' requires {hint}, skipping.")
                skipped_games.append(g)
            else:
                available_games.append(g)
            continue
        # LMGame-Bench and other games
        cfg = GAME_CONFIGS.get(g)
        if cfg is None:
            if g in all_known_games:
                available_games.append(g)
            else:
                print(f"[WARNING] Game '{g}' not available, skipping.")
                skipped_games.append(g)
            continue
        if not cfg.available:
            print(f"[WARNING] Game '{g}' marked as unavailable (ROM/purchase required), skipping.")
            skipped_games.append(g)
            continue
        available_games.append(g)

    if not available_games:
        print("[ERROR] No games available. Ensure GamingAgent / AgentEvolver / Orak is installed.")
        sys.exit(1)

    # -- Load skill bank (same function as run_inference.py) --
    skill_bank_obj = None
    if not args.no_bank and args.bank:
        bank, engine = load_skill_bank(
            args.bank,
            use_query_engine=not args.no_query_engine,
        )
        skill_bank_obj = engine if engine is not None else bank
    elif args.no_bank:
        print("[eval] Running without skill bank (--no-bank).")

    print("=" * 78)
    print("  Decision Agent Evaluation")
    print("=" * 78)
    print(f"  Games:       {', '.join(available_games)}")
    if skipped_games:
        print(f"  Skipped:     {', '.join(skipped_games)}")
    print(f"  Episodes:    {args.episodes} per game")
    print(f"  Max steps:   {'per-game config' if args.max_steps is None else args.max_steps}")
    print(f"  Model:       {args.model}")
    if getattr(args, "opponent_model", None):
        print(f"  Opponents:   {args.opponent_model}")
        if "avalon" in available_games:
            print(f"  Avalon:      {getattr(args, 'num_players', 5)} players, per_role={getattr(args, 'per_role', False)}")
        if "diplomacy" in available_games:
            print(f"  Diplomacy:   per_power={getattr(args, 'per_power', False)}")
    print(f"  Temperature: {args.temperature}")
    bank_desc = "none"
    if skill_bank_obj is not None:
        bank_size = len(skill_bank_obj) if hasattr(skill_bank_obj, "__len__") else "?"
        bank_desc = f"{args.bank} ({bank_size} skills)"
    print(f"  Skill Bank:  {bank_desc}")
    print(f"  Labeling:    {args.label}")
    print(f"  Resume:      {args.resume}")
    print(f"  Output:      {base_dir} (model/game/{run_timestamp})")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []

    for game_name in available_games:
        print(f"\n{'━' * 78}")
        print(f"  GAME: {game_name} ({args.episodes} episodes)")
        print(f"{'━' * 78}")

        game_run_dir = base_dir / game_name / run_timestamp
        use_macro = getattr(args, "macro_actions", False) and game_name == "tetris"
        summary = run_game_rollouts(game_name, args, game_run_dir, skill_bank=skill_bank_obj, use_macro=use_macro)
        game_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    evolver_games = [g for g in available_games if g in EVOLVER_GAME_NAMES]
    orak_games = [g for g in available_games if g in ORAK_EVAL_GAME_NAMES]
    lmgame_games = [g for g in available_games if g not in EVOLVER_GAME_NAMES and g not in ORAK_EVAL_GAME_NAMES]
    benchmarks = []
    if lmgame_games:
        benchmarks.append("LMGame-Bench")
    if evolver_games:
        benchmarks.append("AgentEvolver")
    if orak_games:
        benchmarks.append("Orak")

    master_summary = {
        "benchmark": "+".join(benchmarks) if benchmarks else "multi",
        "run_started": run_timestamp,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "qwen3_8b",
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "labeled": args.label,
        "total_elapsed_seconds": overall_elapsed,
        "total_games_in_benchmark": len(available_games),
        "games_run": len([s for s in game_summaries if not s.get("skipped")]),
        "games_completed": list(available_games),
        "games_skipped": skipped_games,
        "results": game_summaries,
    }
    batch_summary_dir = base_dir / run_timestamp
    batch_summary_dir.mkdir(parents=True, exist_ok=True)
    master_path = batch_summary_dir / "eval_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  QWEN3-14B EVALUATION COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Games processed: {len(available_games)}")
    total_eps = sum(
        s.get("total_episodes", 0)
        for s in game_summaries
        if not s.get("skipped") and not s.get("invalid")
    )
    print(f"  Total episodes:  {total_eps}")
    print(f"  Elapsed:         {overall_elapsed:.1f}s")
    print(f"  Output:          {base_dir} (model/game/{run_timestamp})")
    print(f"  Summary:         {master_path}")

    successful = [
        s for s in game_summaries
        if not s.get("skipped") and not s.get("invalid") and "mean_reward" in s
    ]
    if successful:
        avg_reward = sum(s["mean_reward"] for s in successful) / len(successful)
        avg_steps = sum(s["mean_steps"] for s in successful) / len(successful)
        print(f"  Avg reward:      {avg_reward:.2f}")
        print(f"  Avg steps:       {avg_steps:.1f}")

    print(f"{'=' * 78}")
    print()
    print("  Load rollouts:")
    print("    from data_structure.experience import Episode_Buffer")
    print(f"    buf = Episode_Buffer.load_from_json('{base_dir}/<game>/{run_timestamp}/episode_buffer.json')")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
