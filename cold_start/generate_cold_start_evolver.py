#!/usr/bin/env python
"""
Cold-start rollout generation for Avalon and Diplomacy using GPT-5.4.

Generates decision-making trajectories for the multi-agent social deduction
(Avalon) and strategic negotiation (Diplomacy) games using GPT-5.4 as the
backbone.  Each active player/power is queried independently with structured
chain-of-thought reasoning; the per-agent reasoning traces are stored in
the ``intentions`` field of each Experience.

The env wrappers live in ``env_wrappers/`` (AvalonNLWrapper, DiplomacyNLWrapper)
and expect the AgentEvolver game engines on ``PYTHONPATH``.

Output structure (cold_start/output/gpt54_evolver/<game_name>/):
  - episode_NNN.json        Individual episode (Episode.to_dict())
  - episode_buffer.json     All episodes in Episode_Buffer format
  - rollouts.jsonl          Append-friendly JSONL (one Episode per line)
  - rollout_summary.json    Per-game run stats

Usage (from Game-AI-Agent root):

    export OPENROUTER_API_KEY="sk-or-..."
    export PYTHONPATH="$(pwd):$(pwd)/../AgentEvolver:$PYTHONPATH"

    # Both games, 20 episodes each
    python cold_start/generate_cold_start_evolver.py

    # Avalon only
    python cold_start/generate_cold_start_evolver.py --games avalon

    # Diplomacy only, fewer episodes
    python cold_start/generate_cold_start_evolver.py --games diplomacy --episodes 5

    # Resume an interrupted run
    python cold_start/generate_cold_start_evolver.py --resume
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
_workspace = CODEBASE_ROOT.parent

for p in [str(CODEBASE_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

for _candidate in [_workspace / "AgentEvolver", CODEBASE_ROOT / "AgentEvolver"]:
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

for _candidate in [_workspace / "AI_Diplomacy", CODEBASE_ROOT / "AI_Diplomacy"]:
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from data_structure.experience import Experience, Episode, Episode_Buffer

try:
    from env_wrappers.avalon_nl_wrapper import AvalonNLWrapper
except ImportError:
    AvalonNLWrapper = None  # type: ignore

try:
    from env_wrappers.diplomacy_nl_wrapper import DiplomacyNLWrapper
except ImportError:
    DiplomacyNLWrapper = None  # type: ignore

try:
    import openai
    from api_keys import openai_api_key, open_router_api_key
except (ImportError, AttributeError):
    openai = None
    openai_api_key = None
    open_router_api_key = None

try:
    from API_func import OPENROUTER_BASE
except ImportError:
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"

from cold_start.generate_cold_start import label_trajectory

# ---------------------------------------------------------------------------
# Model constant
# ---------------------------------------------------------------------------
MODEL_GPT54 = "gpt-5.4"

DIPLOMACY_POWERS = [
    "AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY",
]

# ---------------------------------------------------------------------------
# Game registry for evolver games
# ---------------------------------------------------------------------------
EVOLVER_GAMES: Dict[str, Dict[str, Any]] = {
    "avalon": {
        "env_class": AvalonNLWrapper,
        "task": "Win a game of Avalon through social deduction, strategic voting, and deception.",
        "env_kwargs": {"num_players": 5},
    },
    "diplomacy": {
        "env_class": DiplomacyNLWrapper,
        "task": "Gain the most supply centres in Diplomacy through strategic orders and alliances.",
        "env_kwargs": {},
    },
}

# ---------------------------------------------------------------------------
# Chain-of-thought system prompts (enhanced for cold-start reasoning traces)
# ---------------------------------------------------------------------------
AVALON_COT_SYSTEM = (
    "You are an expert Avalon player powered by GPT-5.4.\n"
    "You receive the current game state for a specific player and must choose an action.\n\n"
    "Before choosing, briefly reason about:\n"
    "1. What you know about other players' roles based on observations so far.\n"
    "2. The current phase and what the optimal play is for your role/alignment.\n"
    "3. What information your action reveals and whether that helps or hurts your team.\n\n"
    "Phase actions:\n"
    "- Team Selection (you are leader): comma-separated player IDs, e.g. '0, 2, 3'\n"
    "- Team Voting: exactly 'approve' or 'reject'\n"
    "- Quest Voting (on team): exactly 'pass' or 'fail'\n"
    "- Assassination (Assassin only): a player ID, e.g. '2'\n"
    "- Not your turn: 'wait'\n\n"
    "Call the `choose_action` function with your chosen action."
)

DIPLOMACY_COT_SYSTEM = (
    "You are an expert Diplomacy player powered by GPT-5.4.\n"
    "You control one power and must issue orders for your units this phase.\n\n"
    "Before choosing, briefly reason about:\n"
    "1. Your current territorial position and supply-centre count.\n"
    "2. Which neighbours are threats vs potential allies.\n"
    "3. Whether to attack, defend, or support, and which borders matter most.\n\n"
    "Order formats:\n"
    "  Hold:         A PAR H\n"
    "  Move:         A PAR - BUR\n"
    "  Support hold: A MAR S A PAR\n"
    "  Support move: A MAR S A PAR - BUR\n"
    "  Convoy:       F ENG C A LON - BRE\n"
    "  Retreat:      A PAR R MAR\n"
    "  Build:        A PAR B\n"
    "  Disband:      A PAR D\n\n"
    "Call the `submit_orders` function with your orders list."
)

# ---------------------------------------------------------------------------
# Function-calling tool builders (with reasoning parameter)
# ---------------------------------------------------------------------------

def _build_avalon_cot_tools() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": (
                    "Choose your Avalon action for this phase.  "
                    "Team selection: comma-separated player IDs. "
                    "Voting: 'approve'/'reject'. Quest: 'pass'/'fail'. "
                    "Assassination: player ID. Not your turn: 'wait'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Brief chain-of-thought reasoning for this action.",
                        },
                        "action": {
                            "type": "string",
                            "description": "Your action string.",
                        },
                    },
                    "required": ["action"],
                },
            },
        }
    ]


def _build_diplomacy_cot_tools() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_orders",
                "description": (
                    "Submit Diplomacy orders for this phase. "
                    "Each order is e.g. 'A PAR - BUR'. "
                    "Empty list if no orders."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Brief chain-of-thought reasoning for these orders.",
                        },
                        "orders": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of order strings.",
                        },
                    },
                    "required": ["orders"],
                },
            },
        }
    ]


# ---------------------------------------------------------------------------
# GPT-5.4 agent helpers
# ---------------------------------------------------------------------------

_SHARED_CLIENT: Optional[Any] = None
_SHARED_USE_ROUTER: Optional[bool] = None


def _get_client() -> Tuple[Any, bool]:
    """Return a shared (openai.OpenAI client, use_router_flag).  Thread-safe for reads."""
    global _SHARED_CLIENT, _SHARED_USE_ROUTER
    if _SHARED_CLIENT is None:
        use_router = bool(open_router_api_key and open_router_api_key.strip())
        if use_router:
            _SHARED_CLIENT = openai.OpenAI(base_url=OPENROUTER_BASE, api_key=open_router_api_key.strip())
        else:
            _SHARED_CLIENT = openai.OpenAI(api_key=openai_api_key)
        _SHARED_USE_ROUTER = use_router
    return _SHARED_CLIENT, _SHARED_USE_ROUTER


def _effective_model(model: str, use_router: bool) -> str:
    if use_router and "/" not in model:
        return f"openai/{model}"
    return model


def _call_gpt54(
    system_prompt: str,
    user_content: str,
    tools: list,
    func_name: str,
    model: str = MODEL_GPT54,
    temperature: float = 0.4,
) -> Dict[str, Any]:
    """Call GPT-5.4 with function calling. Returns parsed tool-call args dict."""
    if openai is None:
        return {}
    client, use_router = _get_client()
    eff_model = _effective_model(model, use_router)

    try:
        response = client.chat.completions.create(
            model=eff_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": func_name}},
            temperature=temperature,
            max_tokens=600,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            raw = getattr(msg.tool_calls[0], "arguments", None) or getattr(msg.tool_calls[0].function, "arguments", None) or "{}"
            return json.loads(raw) if isinstance(raw, str) else (raw or {})
        return {}
    except Exception as exc:
        print(f"    [WARN] GPT-5.4 call failed ({exc})")
        return {}


# ---------------------------------------------------------------------------
# Per-agent action helpers
# ---------------------------------------------------------------------------

def avalon_agent_action(
    state_nl: str,
    model: str = MODEL_GPT54,
    temperature: float = 0.4,
) -> Tuple[str, Optional[str]]:
    """Query GPT-5.4 for one Avalon player. Returns (action, reasoning)."""
    args = _call_gpt54(
        system_prompt=AVALON_COT_SYSTEM,
        user_content=f"Current game state:\n\n{state_nl}\n\nChoose your action.",
        tools=_build_avalon_cot_tools(),
        func_name="choose_action",
        model=model,
        temperature=temperature,
    )
    action = args.get("action", "wait")
    reasoning = args.get("reasoning")
    return action, reasoning


def _parse_diplomacy_orders_from_text(reply: str) -> List[str]:
    """Extract Diplomacy order strings from a plain-text LLM reply."""
    orders: List[str] = []
    for line in reply.splitlines():
        line = line.strip().strip("'\",-•*`)>0123456789.")
        line = line.strip()
        if re.match(r"^[A-Z]\s+[A-Z]{3}", line):
            orders.append(line)
    if not orders:
        json_m = re.search(r"\[([^\]]+)\]", reply)
        if json_m:
            try:
                parsed = json.loads("[" + json_m.group(1) + "]")
                orders = [str(o).strip() for o in parsed if o and re.match(r"^[A-Z]\s+[A-Z]{3}", str(o).strip())]
            except (json.JSONDecodeError, TypeError):
                pass
    return orders


def _call_text_completion(
    system_prompt: str,
    user_content: str,
    model: str = MODEL_GPT54,
    temperature: float = 0.4,
) -> str:
    """Call a model via OpenRouter using plain chat completion (no tools)."""
    if openai is None:
        return ""
    client, use_router = _get_client()
    eff_model = _effective_model(model, use_router)
    try:
        response = client.chat.completions.create(
            model=eff_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=800,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        print(f"    [WARN] text-completion call failed ({exc})")
        return ""


DIPLOMACY_TEXT_SYSTEM = (
    "You are an expert Diplomacy player.\n"
    "You control one power and must issue orders for your units this phase.\n\n"
    "Before choosing, briefly reason about:\n"
    "1. Your current territorial position and supply-centre count.\n"
    "2. Which neighbours are threats vs potential allies.\n"
    "3. Whether to attack, defend, or support, and which borders matter most.\n\n"
    "Order formats:\n"
    "  Hold:         A PAR H\n"
    "  Move:         A PAR - BUR\n"
    "  Support hold: A MAR S A PAR\n"
    "  Support move: A MAR S A PAR - BUR\n"
    "  Convoy:       F ENG C A LON - BRE\n"
    "  Retreat:      A PAR R MAR\n"
    "  Build:        A PAR B\n"
    "  Disband:      A PAR D\n\n"
    "Reply with:\n"
    "REASONING: <your brief reasoning>\n"
    "ORDERS:\n<one order per line>\n"
)


def diplomacy_agent_action(
    state_nl: str,
    model: str = MODEL_GPT54,
    temperature: float = 0.4,
) -> Tuple[List[str], Optional[str]]:
    """Query an LLM for one Diplomacy power. Returns (orders_list, reasoning).

    Tries function calling first (works for GPT-5.4); falls back to
    plain text completion + regex parsing for models that don't support
    tool_choice (e.g. Gemini, GPT-OSS-120B via OpenRouter).
    """
    args = _call_gpt54(
        system_prompt=DIPLOMACY_COT_SYSTEM,
        user_content=f"Current game state:\n\n{state_nl}\n\nSubmit your orders.",
        tools=_build_diplomacy_cot_tools(),
        func_name="submit_orders",
        model=model,
        temperature=temperature,
    )
    orders = args.get("orders", [])
    if not isinstance(orders, list):
        orders = [str(orders)] if orders else []
    orders = [str(o) for o in orders if o]
    reasoning = args.get("reasoning")

    if orders:
        return orders, reasoning

    # Fallback: plain text completion (no function calling)
    reply = _call_text_completion(
        system_prompt=DIPLOMACY_TEXT_SYSTEM,
        user_content=f"Current game state:\n\n{state_nl}\n\nSubmit your orders.",
        model=model,
        temperature=temperature,
    )
    if not reply:
        return [], None

    reasoning = None
    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nORDERS|\Z)", reply, re.DOTALL | re.IGNORECASE)
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    orders = _parse_diplomacy_orders_from_text(reply)
    return orders, reasoning


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_avalon_episode(
    num_players: int = 5,
    seed: int = 42,
    model: str = MODEL_GPT54,
    temperature: float = 0.4,
    verbose: bool = False,
    controlled_player: Optional[int] = None,
    opponent_model: Optional[str] = None,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one Avalon episode.

    Three modes depending on *controlled_player* and *opponent_model*:

    1. ``controlled_player=None`` — all players use *model* (self-play).
    2. ``controlled_player=N, opponent_model=None`` — player *N* uses
       *model*, others use the env's random partner policy (legacy).
    3. ``controlled_player=N, opponent_model="gpt-5.4"`` — player *N* uses
       *model*, all other players use *opponent_model* via LLM calls.
       The env runs in multi-agent mode (no random policy).
    """
    if AvalonNLWrapper is None:
        raise ImportError("AvalonNLWrapper not available. Check AgentEvolver install.")

    mixed_model = controlled_player is not None and opponent_model is not None
    single_agent = controlled_player is not None and opponent_model is None
    task = EVOLVER_GAMES["avalon"]["task"]

    if mixed_model:
        env = AvalonNLWrapper(num_players=num_players, seed=seed)
    else:
        env = AvalonNLWrapper(num_players=num_players, seed=seed,
                              controlled_player=controlled_player)
    obs, info = env.reset()

    cp_role_name = cp_side = None
    if controlled_player is not None:
        roles = env.roles
        cp_role_id, cp_role_name, cp_is_good = roles[controlled_player]
        cp_side = "good" if cp_is_good else "evil"
        if verbose:
            mode_tag = "mixed-model" if mixed_model else "single-agent"
            opp_tag = f", opponents={opponent_model}" if mixed_model else ""
            print(f"    {mode_tag}: player {controlled_player} = {cp_role_name} ({cp_side}){opp_tag}")

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0

    while not env.done:
        active = info.get("active_players", [])
        step_reasonings: List[str] = []

        if single_agent:
            action_str, reasoning = avalon_agent_action(obs, model, temperature)
            if reasoning:
                step_reasonings.append(f"Player {controlled_player}: {reasoning}")
            if verbose:
                short = (reasoning[:80] + "...") if reasoning and len(reasoning) > 80 else reasoning
                print(f"  Player {controlled_player} action={action_str!r}  reason={short}")

            next_obs, rewards, terminated, truncated, next_info = env.step(action_str)
            done = terminated or truncated
            reward_val = float(rewards) if not isinstance(rewards, dict) else 0.0

        else:
            actions: Dict[int, Any] = {}
            players_to_query = [
                (pid, obs.get(pid, "")) for pid in active if obs.get(pid, "")
            ]

            with ThreadPoolExecutor(max_workers=len(players_to_query) or 1) as pool:
                futures = {}
                for pid, state_nl in players_to_query:
                    if mixed_model:
                        pid_model = model if pid == controlled_player else opponent_model
                    else:
                        pid_model = model
                    futures[pool.submit(avalon_agent_action, state_nl, pid_model, temperature)] = pid

                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        action, reasoning = future.result()
                    except Exception as exc:
                        print(f"    [WARN] Player {pid} agent call failed ({exc})")
                        action, reasoning = "wait", None
                    actions[pid] = action
                    if reasoning:
                        step_reasonings.append(f"Player {pid}: {reasoning}")
                    if verbose:
                        tag = f"[{model}]" if mixed_model and pid == controlled_player else (f"[{opponent_model}]" if mixed_model else "")
                        short = (reasoning[:80] + "...") if reasoning and len(reasoning) > 80 else reasoning
                        print(f"  Player {pid} {tag} action={action!r}  reason={short}")

            next_obs, rewards, terminated, truncated, next_info = env.step(actions)
            done = terminated or truncated
            reward_val = sum(rewards.values()) if isinstance(rewards, dict) and rewards else 0.0

        total_reward += reward_val
        combined_reasoning = "\n".join(step_reasonings) if step_reasonings else None

        phase_id = info.get("phase", 0)
        if phase_id == 0:
            avail_actions = {
                "type": "team_selection",
                "active_players": list(active),
                "team_size": info.get("team_size", 0),
                "player_ids": list(range(num_players)),
            }
        elif phase_id == 1:
            avail_actions = {
                "type": "team_vote",
                "active_players": list(active),
                "choices": ["approve", "reject"],
            }
        elif phase_id == 2:
            avail_actions = {
                "type": "quest_vote",
                "active_players": list(active),
                "choices": ["pass", "fail"],
            }
        elif phase_id == 3:
            avail_actions = {
                "type": "assassination",
                "active_players": list(active),
                "targets": list(range(num_players)),
            }
        else:
            avail_actions = {"type": "unknown", "active_players": list(active)}

        if single_agent:
            state_json = json.dumps(obs, ensure_ascii=False, default=str) if not isinstance(obs, str) else obs
            action_json = json.dumps(action_str, ensure_ascii=False, default=str) if not isinstance(action_str, str) else action_str
            next_state_json = json.dumps(next_obs, ensure_ascii=False, default=str) if not isinstance(next_obs, str) else next_obs
        else:
            state_json = json.dumps({str(k): v for k, v in obs.items()}, ensure_ascii=False, default=str)
            action_json = json.dumps({str(k): v for k, v in actions.items()}, ensure_ascii=False, default=str)
            next_state_json = json.dumps(
                {str(k): v for k, v in next_obs.items()}, ensure_ascii=False, default=str
            ) if isinstance(next_obs, dict) else str(next_obs)

        exp = Experience(
            state=state_json,
            action=action_json,
            reward=float(reward_val),
            next_state=next_state_json,
            done=done,
            intentions=combined_reasoning,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.raw_state = info
        exp.raw_next_state = next_info
        exp.available_actions = avail_actions
        exp.interface = {"env_name": "avalon", "game_name": "avalon", "num_players": num_players}
        experiences.append(exp)

        if verbose:
            phase = next_info.get("phase_name", next_info.get("phase", ""))
            print(f"  step {step_count}: reward={reward_val:.2f}, cum={total_reward:.2f}, phase={phase}")

        obs = next_obs
        info = next_info
        step_count += 1

        if done:
            break

    env_done = env.done
    good_victory = info.get("good_victory")

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="avalon",
        game_name="avalon",
    )
    episode.set_outcome()

    if mixed_model:
        agent_type = "mixed_model"
    elif single_agent:
        agent_type = "gpt54_single"
    else:
        agent_type = "gpt54_base"

    stats = {
        "game": "avalon",
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": env_done,
        "truncated": False,
        "model": model,
        "agent_type": agent_type,
        "num_players": num_players,
        "seed": seed,
        "good_victory": good_victory,
    }
    if controlled_player is not None:
        stats["controlled_player"] = controlled_player
        stats["role_name"] = cp_role_name
        stats["role_side"] = cp_side
    if opponent_model:
        stats["opponent_model"] = opponent_model
    return episode, stats


DIPLOMACY_MAX_PHASES = 20  # matches DiplomacyConfig.max_phases


def _match_power_name(controlled: str, active_names) -> Optional[str]:
    """Case-insensitive match of *controlled* against env power names."""
    ctrl = controlled.upper()
    for name in active_names:
        if str(name).upper() == ctrl:
            return name
    return None


def run_diplomacy_episode(
    seed: int = 42,
    model: str = MODEL_GPT54,
    temperature: float = 0.4,
    verbose: bool = False,
    controlled_power: Optional[str] = None,
    opponent_model: Optional[str] = None,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one Diplomacy episode.

    Two modes depending on *controlled_power* and *opponent_model*:

    1. ``controlled_power=None`` — all powers use *model* (self-play).
    2. ``controlled_power="FRANCE", opponent_model="gpt-5.4"`` — the
       controlled power uses *model*, all other powers use *opponent_model*.
    """
    if DiplomacyNLWrapper is None:
        raise ImportError("DiplomacyNLWrapper not available. Check AI_Diplomacy install.")

    mixed_model = controlled_power is not None and opponent_model is not None
    task = EVOLVER_GAMES["diplomacy"]["task"]
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
            print(f"    Mixed-model: {resolved_cp}={model}, opponents={opponent_model}")

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0

    while not env.done:
        actions: Dict[str, Union[List[str], str]] = {}
        step_reasonings: List[str] = []

        active_powers = info.get("active_powers", {})

        powers_to_query = [
            (pname, obs[pname]) for pname in obs
            if obs.get(pname) and pname in active_powers
        ]

        with ThreadPoolExecutor(max_workers=len(powers_to_query) or 1) as pool:
            futures = {}
            for pname, state_nl in powers_to_query:
                if mixed_model:
                    p_model = model if pname == resolved_cp else opponent_model
                else:
                    p_model = model
                futures[pool.submit(diplomacy_agent_action, state_nl, p_model, temperature)] = pname

            for future in as_completed(futures):
                power_name = futures[future]
                try:
                    orders, reasoning = future.result()
                except Exception as exc:
                    print(f"    [WARN] {power_name} agent call failed ({exc})")
                    orders, reasoning = [], None
                actions[power_name] = orders
                if reasoning:
                    step_reasonings.append(f"{power_name}: {reasoning}")
                if verbose:
                    tag = f"[{model}]" if mixed_model and power_name == resolved_cp else (f"[{opponent_model}]" if mixed_model else "")
                    preview = orders[:3]
                    print(f"  {power_name} {tag}: {len(orders)} orders, e.g. {preview}")

        next_obs, rewards, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated

        reward_val = sum(rewards.values()) if isinstance(rewards, dict) and rewards else 0.0
        total_reward += reward_val

        combined_reasoning = "\n".join(step_reasonings) if step_reasonings else None

        exp = Experience(
            state=json.dumps(dict(obs), ensure_ascii=False, default=str),
            action=json.dumps(dict(actions), ensure_ascii=False, default=str),
            reward=float(reward_val),
            next_state=json.dumps(
                dict(next_obs), ensure_ascii=False, default=str
            ) if isinstance(next_obs, dict) else str(next_obs),
            done=done,
            intentions=combined_reasoning,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.raw_state = _sanitize_keys(info)
        exp.raw_next_state = _sanitize_keys(next_info)
        exp.available_actions = _sanitize_keys(info.get("possible_orders", {}))
        exp.interface = {"env_name": "diplomacy", "game_name": "diplomacy"}
        experiences.append(exp)

        if verbose:
            phase = next_info.get("phase", "")
            print(f"  step {step_count}: reward={reward_val:.2f}, cum={total_reward:.2f}, phase={phase}")

        obs = next_obs
        info = next_info
        step_count += 1

        if done:
            break

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="diplomacy",
        game_name="diplomacy",
    )
    episode.set_outcome()

    final_rewards = {}
    if isinstance(rewards, dict):
        final_rewards = {str(k): float(v) for k, v in rewards.items()}

    stats = {
        "game": "diplomacy",
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "mixed_model" if mixed_model else "gpt54_base",
        "seed": seed,
        "max_phases": DIPLOMACY_MAX_PHASES,
        "final_sc_rewards": final_rewards,
    }
    if mixed_model:
        stats["controlled_power"] = resolved_cp
        stats["opponent_model"] = opponent_model
        if isinstance(rewards, dict) and resolved_cp:
            stats["controlled_power_reward"] = float(rewards.get(resolved_cp, 0.0))
    return episode, stats


# ---------------------------------------------------------------------------
# Batch rollout helpers (same structure as generate_cold_start_gpt54)
# ---------------------------------------------------------------------------

def _sanitize_keys(obj: Any) -> Any:
    """Recursively convert all dict keys to plain ``str`` so ``json.dump`` never
    chokes on engine-internal key types (e.g. diplomacy's ``StringComparator``)."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_keys(v) for v in obj]
    return obj


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
    opp = getattr(args, "opponent_model", None)
    summary: Dict[str, Any] = {
        "game": game_name,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "opponent_model": opp,
        "agent_type": "mixed_model" if opp else "gpt54_base",
        "total_episodes": len(all_stats),
        "target_episodes": args.episodes,
        "end_condition": "natural",
        "labeled": args.label and not args.no_label,
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    if all_stats:
        rewards = [s["total_reward"] for s in all_stats]
        steps = [s["steps"] for s in all_stats]
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
    output_dir: Path,
) -> Dict[str, Any]:
    """Run all episodes for one game and save outputs."""
    game_dir = output_dir / game_name
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

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx in range(start_idx, args.episodes):
        print(f"\n  [{game_name}] Episode {ep_idx + 1}/{args.episodes}")
        seed = args.seed + ep_idx

        try:
            opp_model = getattr(args, "opponent_model", None)

            if game_name == "avalon":
                cp = getattr(args, "controlled_player", None)
                if getattr(args, "per_role", False):
                    cp = ep_idx % args.num_players
                episode, stats = run_avalon_episode(
                    num_players=args.num_players,
                    seed=seed,
                    model=args.model,
                    temperature=args.temperature,
                    verbose=args.verbose,
                    controlled_player=cp,
                    opponent_model=opp_model,
                )
            elif game_name == "diplomacy":
                cp_power = getattr(args, "controlled_power", None)
                if getattr(args, "per_power", False):
                    cp_power = DIPLOMACY_POWERS[ep_idx % len(DIPLOMACY_POWERS)]
                episode, stats = run_diplomacy_episode(
                    seed=seed,
                    model=args.model,
                    temperature=args.temperature,
                    verbose=args.verbose,
                    controlled_power=cp_power,
                    opponent_model=opp_model,
                )
            else:
                print(f"    [ERROR] Unknown game: {game_name}")
                continue

            stats["episode_index"] = ep_idx
            print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

            if args.label and not args.no_label:
                episode = label_trajectory(episode, args.label_model)

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

    buffer_path = game_dir / "episode_buffer.json"
    episode_buffer.save_to_json(str(buffer_path))
    print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    summary = save_game_summary(game_name, game_dir, all_stats, elapsed, args)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPT-5.4 cold-start rollouts for Avalon and Diplomacy (evolver wrappers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        choices=list(EVOLVER_GAMES.keys()),
                        help="Games to generate rollouts for (default: all)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes per game (default: 20)")
    parser.add_argument("--model", type=str, default=MODEL_GPT54,
                        help=f"LLM model for the agent (default: {MODEL_GPT54})")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature (default: 0.4)")
    parser.add_argument("--label", action="store_true",
                        help="Label trajectories with LLM (default: off; use labeling/ for that)")
    parser.add_argument("--no_label", action="store_true",
                        help="Skip trajectory labeling")
    parser.add_argument("--label_model", type=str, default="gpt-5-mini",
                        help="Model used for trajectory labeling (default: gpt-5-mini)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run (skip completed episodes)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print step-by-step details")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: cold_start/output/gpt54_evolver)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (incremented per episode)")
    parser.add_argument("--num_players", type=int, default=5,
                        help="Number of Avalon players (default: 5)")
    parser.add_argument("--controlled_player", type=int, default=None,
                        help="Avalon: single-agent mode controlling this player (0-4). "
                             "Others use random partner policy. None = self-play (default).")
    parser.add_argument("--per_role", action="store_true",
                        help="Avalon: cycle controlled_player through 0..num_players-1 "
                             "across episodes (single-agent, one role per episode).")
    parser.add_argument("--opponent_model", type=str, default=None,
                        help="Model for opponent players/powers (e.g. gpt-5.4). "
                             "When set with --per_role or --per_power, the controlled "
                             "player/power uses --model while all others use this model.")
    parser.add_argument("--controlled_power", type=str, default=None,
                        help="Diplomacy: control a specific power (e.g. FRANCE). "
                             "Others use --opponent_model. None = all-same-model (default).")
    parser.add_argument("--per_power", action="store_true",
                        help="Diplomacy: cycle controlled_power through the 7 powers "
                             "across episodes (one power per episode). "
                             "Requires --opponent_model.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output" / "gpt54_evolver"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from api_keys import open_router_api_key as _or_key
        has_key = bool(
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or (_or_key and _or_key.strip())
        )
    except Exception:
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if not has_key:
        print("[WARNING] No API key set. LLM calls will fail.")
        print("  Set open_router_api_key in api_keys.py or export OPENROUTER_API_KEY='sk-or-...'")

    requested = args.games if args.games else list(EVOLVER_GAMES.keys())
    available_games: List[str] = []
    skipped_games: List[str] = []

    for g in requested:
        if g not in EVOLVER_GAMES:
            print(f"[WARNING] Game '{g}' not recognised, skipping.")
            skipped_games.append(g)
            continue
        if EVOLVER_GAMES[g]["env_class"] is None:
            print(f"[WARNING] Game '{g}' env wrapper not importable, skipping.")
            skipped_games.append(g)
            continue
        available_games.append(g)

    if not available_games:
        print("[ERROR] No games available. Ensure AgentEvolver / AI_Diplomacy are installed.")
        sys.exit(1)

    print("=" * 78)
    print("  Avalon & Diplomacy Baseline Rollout Generation")
    print("=" * 78)
    print(f"  Games:       {', '.join(available_games)}")
    if skipped_games:
        print(f"  Skipped:     {', '.join(skipped_games)}")
    print(f"  Episodes:    {args.episodes} per game")
    print(f"  End cond:    natural (avalon=engine done, diplomacy=max_phases {DIPLOMACY_MAX_PHASES})")
    print(f"  Model:       {args.model}")
    if args.opponent_model:
        print(f"  Opponents:   {args.opponent_model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Labeling:    {args.label and not args.no_label} (label model: {args.label_model})")
    print(f"  Resume:      {args.resume}")
    print(f"  Seed:        {args.seed}")
    if "avalon" in available_games:
        print(f"  Avalon:      {args.num_players} players, per_role={args.per_role}")
    if "diplomacy" in available_games:
        print(f"  Diplomacy:   per_power={args.per_power}")
    print(f"  Output:      {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []

    for game_name in available_games:
        print(f"\n{'━' * 78}")
        print(f"  GAME: {game_name.upper()} ({args.episodes} episodes)")
        print(f"{'━' * 78}")

        summary = run_game_rollouts(game_name, args, output_dir)
        game_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "opponent_model": args.opponent_model,
        "agent_type": "mixed_model" if args.opponent_model else "gpt54_base",
        "episodes_per_game": args.episodes,
        "end_condition": "natural",
        "temperature": args.temperature,
        "labeled": args.label and not args.no_label,
        "label_model": args.label_model,
        "total_elapsed_seconds": overall_elapsed,
        "games_completed": list(available_games),
        "games_skipped": skipped_games,
        "per_game_summaries": game_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  GPT-5.4 AVALON & DIPLOMACY — BATCH ROLLOUT COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Games processed: {len(available_games)}")
    total_eps = sum(
        s.get("total_episodes", 0) for s in game_summaries if not s.get("skipped")
    )
    print(f"  Total episodes:  {total_eps}")
    print(f"  Elapsed:         {overall_elapsed:.1f}s")
    print(f"  Output:          {output_dir}")
    print(f"  Master summary:  {master_path}")

    successful = [s for s in game_summaries if not s.get("skipped") and "mean_reward" in s]
    if successful:
        avg_reward = sum(s["mean_reward"] for s in successful) / len(successful)
        avg_steps = sum(s["mean_steps"] for s in successful) / len(successful)
        print(f"  Avg reward:      {avg_reward:.2f}")
        print(f"  Avg steps:       {avg_steps:.1f}")

    print(f"{'=' * 78}")
    print()
    print("  Load into skill pipeline:")
    print("    from cold_start.load_rollouts import load_all_game_rollouts")
    print(f"    rollouts = load_all_game_rollouts('{output_dir}')")
    print()
    print("  Load into trainer:")
    print("    from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records")
    print(f"    eps = load_episodes_from_jsonl('{output_dir}/<game>/rollouts.jsonl')")
    print("    records = episodes_to_rollout_records(eps)")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
