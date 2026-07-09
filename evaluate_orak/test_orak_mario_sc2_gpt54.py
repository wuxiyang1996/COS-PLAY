#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-5.4 evaluation for Super Mario and StarCraft II via the Orak env wrappers.

Runs one (or both) Orak game environments using GPT-5.4 as the backbone model,
with structured chain-of-thought reasoning before each action.  Each action
call is routed through an OpenAI function-calling tool (``choose_action``),
matching the pattern established by ``generate_cold_start_gpt54.py`` for
LM-Game Bench games.

Usage (from Game-AI-Agent root, with Orak src on PYTHONPATH):

    # --- Super Mario ---
    source evaluate_orak/setup_orak_mario.sh
    python evaluate_orak/test_orak_mario_sc2_gpt54.py --game super_mario

    # --- StarCraft II ---
    source evaluate_orak/setup_orak_sc2.sh
    python evaluate_orak/test_orak_mario_sc2_gpt54.py --game star_craft

    # Both games (needs both envs set up; star_craft first, then mario)
    python evaluate_orak/test_orak_mario_sc2_gpt54.py --game both

    # With experience collection
    python evaluate_orak/test_orak_mario_sc2_gpt54.py --game super_mario \\
        --use_experience_collection \\
        --save_episode_buffer output/gpt54_orak_mario.json

    # Override model / episodes / max_steps
    python evaluate_orak/test_orak_mario_sc2_gpt54.py --game star_craft \\
        --model gpt-5.4 --episodes 5 --max_steps 500

Environment setup:
    # Super Mario  -> source evaluate_orak/setup_orak_mario.sh
    # StarCraft II -> source evaluate_orak/setup_orak_sc2.sh
    # Or use the convenience wrappers:
    #   bash evaluate_orak/run_gpt54_mario.sh
    #   bash evaluate_orak/run_gpt54_sc2.sh
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_CODEBASE_ROOT = _SCRIPT_DIR.parent
_ORAK_SRC = _CODEBASE_ROOT.parent / "Orak" / "src"

for _p in [str(_CODEBASE_ROOT), str(_ORAK_SRC)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from evaluate_orak.orak_nl_wrapper import make_orak_env, ORAK_GAMES
from data_structure.experience import Experience, Episode, Episode_Buffer

from decision_agents.dummy_agent import (
    language_agent_action,
    detect_game,
    GAME_ORAK,
    AgentBufferManager,
)

try:
    import openai
    from api_keys import openai_api_key, open_router_api_key
except (ImportError, AttributeError):
    openai = None
    openai_api_key = None  # type: ignore[assignment]
    open_router_api_key = None  # type: ignore[assignment]

try:
    from API_func import OPENROUTER_BASE
except ImportError:
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-5.4"

SUPPORTED_GAMES = {
    "super_mario": {
        "max_steps": 100,
        "display_name": "Super Mario Bros",
    },
    "star_craft": {
        "max_steps": 1000,
        "display_name": "StarCraft II (1v1 Protoss)",
    },
}

# ---------------------------------------------------------------------------
# GPT-5.4 system prompt & templates
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an expert game-playing agent powered by GPT-5.4, competing in the "
    "Orak benchmark.\n"
    "You receive a textual description of the current game state and must "
    "choose exactly one action.\n\n"
    "Before choosing, briefly reason about:\n"
    "1. Key elements of the current state (positions, scores, threats, "
    "opportunities).\n"
    "2. What your immediate sub-goal should be.\n"
    "3. How each candidate action moves you toward that goal.\n\n"
    "Then call the `choose_action` function with your chosen action.\n"
    "Use the EXACT action name from the valid actions list."
)

_SYSTEM_PROMPT_SC2 = (
    "You are an expert StarCraft II agent powered by GPT-5.4, playing Protoss "
    "vs Zerg in the Orak benchmark.\n"
    "Each turn you must provide EXACTLY 5 sequential macro actions.\n\n"
    "Before choosing, briefly reason about:\n"
    "1. Current economy (minerals, gas, supply), army composition, and "
    "opponent threat level.\n"
    "2. Which build order phase you are in and what the next priority is.\n"
    "3. Whether to focus on economy, tech, or aggression this turn.\n\n"
    "Then call the `choose_actions` function with a list of 5 actions.\n"
    "Use EXACT action names from the valid actions list."
)

_USER_TEMPLATE = (
    "Game state:\n\n{state}\n\n"
    "Valid actions: {actions}\n\n"
    "Think step-by-step, then choose one action."
)

_USER_TEMPLATE_SC2 = (
    "Game state:\n\n{state}\n\n"
    "Valid actions: {actions}\n\n"
    "Think step-by-step, then choose exactly 5 sequential macro actions."
)


# ═══════════════════════════════════════════════════════════════════════════
# GPT-5.4 function-calling agent
# ═══════════════════════════════════════════════════════════════════════════

def _build_tools_single(action_names: List[str]) -> list:
    """Tool definition for single-action games (Mario)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": "Choose the single action for your turn.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Brief chain-of-thought reasoning.",
                        },
                        "action": {
                            "type": "string",
                            "description": (
                                f"One of: {', '.join(action_names)}"
                            ),
                        },
                    },
                    "required": ["action"],
                },
            },
        }
    ]


def _build_tools_sc2(action_names: List[str]) -> list:
    """Tool definition for StarCraft II (5 sequential actions per step)."""
    action_desc = f"One of: {', '.join(action_names)}"
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_actions",
                "description": "Choose exactly 5 sequential macro actions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Brief chain-of-thought reasoning.",
                        },
                        "action_1": {"type": "string", "description": action_desc},
                        "action_2": {"type": "string", "description": action_desc},
                        "action_3": {"type": "string", "description": action_desc},
                        "action_4": {"type": "string", "description": action_desc},
                        "action_5": {"type": "string", "description": action_desc},
                    },
                    "required": [
                        "action_1", "action_2", "action_3",
                        "action_4", "action_5",
                    ],
                },
            },
        }
    ]


def _extract_action(text: str, valid_actions: List[str]) -> Optional[str]:
    """Best-effort extraction of a valid action from free-text."""
    if not text:
        return None
    reply = text.strip().lower()
    for v in valid_actions:
        if v.lower() in reply:
            return v
    words = re.findall(r"[\w_]+", reply)
    for w in words:
        for v in valid_actions:
            if w == v.lower():
                return v
    return valid_actions[0] if valid_actions else None


def _canonicalize(raw: str, action_names: List[str]) -> str:
    lower_map = {a.lower().strip(): a for a in action_names}
    canonical = lower_map.get(raw.lower().strip())
    if canonical:
        return canonical
    extracted = _extract_action(raw, action_names)
    return extracted or (action_names[0] if action_names else raw)


def _make_openai_client() -> Optional["openai.OpenAI"]:
    if openai is None:
        return None
    use_router = open_router_api_key and str(open_router_api_key).strip()
    if use_router:
        return openai.OpenAI(
            base_url=OPENROUTER_BASE,
            api_key=str(open_router_api_key).strip(),
        )
    if openai_api_key:
        return openai.OpenAI(api_key=openai_api_key)
    env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        base_url = OPENROUTER_BASE if os.environ.get("OPENROUTER_API_KEY") else None
        kw: Dict[str, Any] = {"api_key": env_key}
        if base_url:
            kw["base_url"] = base_url
        return openai.OpenAI(**kw)
    return None


def _effective_model(model: str) -> str:
    if open_router_api_key and str(open_router_api_key).strip():
        return model if "/" in model else f"openai/{model}"
    return model


def gpt54_mario_action(
    state_nl: str,
    action_names: List[str],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> Tuple[str, Optional[str]]:
    """Query GPT-5.4 for a single Mario action. Returns (action, reasoning)."""
    client = _make_openai_client()
    if client is None:
        return action_names[0] if action_names else "Jump Level : 0", None

    tools = _build_tools_single(action_names)
    user_content = _USER_TEMPLATE.format(
        state=state_nl,
        actions=", ".join(action_names),
    )

    try:
        resp = client.chat.completions.create(
            model=_effective_model(model),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "choose_action"}},
            temperature=temperature,
            max_tokens=400,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            raw_args = (
                getattr(msg.tool_calls[0], "arguments", None)
                or getattr(msg.tool_calls[0].function, "arguments", None)
                or "{}"
            )
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            action = _canonicalize(args.get("action", ""), action_names)
            return action, args.get("reasoning")

        content = msg.content or ""
        return _canonicalize(content, action_names), None

    except Exception as exc:
        print(f"    [WARN] GPT-5.4 Mario call failed ({exc}), using fallback")
        return action_names[0] if action_names else "Jump Level : 0", None


def gpt54_sc2_action(
    state_nl: str,
    action_names: List[str],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> Tuple[str, Optional[str]]:
    """
    Query GPT-5.4 for 5 sequential StarCraft II macro actions.

    Returns a single string "1: A\\n2: B\\n...\\n5: E" (the format Orak SC2 expects)
    plus the chain-of-thought reasoning.
    """
    client = _make_openai_client()
    default_action = "EMPTY ACTION"
    if client is None:
        five = "\n".join(f"{i}: {default_action}" for i in range(1, 6))
        return five, None

    tools = _build_tools_sc2(action_names)
    user_content = _USER_TEMPLATE_SC2.format(
        state=state_nl,
        actions=", ".join(action_names),
    )

    try:
        resp = client.chat.completions.create(
            model=_effective_model(model),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_SC2},
                {"role": "user", "content": user_content},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "choose_actions"}},
            temperature=temperature,
            max_tokens=600,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            raw_args = (
                getattr(msg.tool_calls[0], "arguments", None)
                or getattr(msg.tool_calls[0].function, "arguments", None)
                or "{}"
            )
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            reasoning = args.get("reasoning")
            actions_list = []
            for i in range(1, 6):
                raw = args.get(f"action_{i}", default_action)
                actions_list.append(_canonicalize(raw, action_names))
            five = "\n".join(f"{i}: {a}" for i, a in enumerate(actions_list, 1))
            return five, reasoning

        content = msg.content or ""
        return content.strip() or "\n".join(
            f"{i}: {default_action}" for i in range(1, 6)
        ), None

    except Exception as exc:
        print(f"    [WARN] GPT-5.4 SC2 call failed ({exc}), using fallback")
        five = "\n".join(f"{i}: {default_action}" for i in range(1, 6))
        return five, None


# ═══════════════════════════════════════════════════════════════════════════
# Unified agent dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def gpt54_orak_action(
    state_nl: str,
    game_name: str,
    action_names: List[str],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    use_function_call: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    Choose an action for the given Orak game using GPT-5.4.

    When *use_function_call* is True (default), uses the structured
    function-calling path with CoT reasoning.  Otherwise falls back to
    ``decision_agents.dummy_agent.language_agent_action``.

    Returns (action_string, reasoning_or_None).
    """
    if not use_function_call:
        action = language_agent_action(
            state_nl=state_nl,
            game=GAME_ORAK,
            model=model,
            use_function_call=True,
            temperature=temperature,
        )
        return str(action), None

    if game_name in ("star_craft", "star_craft_multi"):
        return gpt54_sc2_action(state_nl, action_names, model, temperature)
    else:
        return gpt54_mario_action(state_nl, action_names, model, temperature)


# ═══════════════════════════════════════════════════════════════════════════
# Episode runners
# ═══════════════════════════════════════════════════════════════════════════

def run_mario_episode(
    model: str = DEFAULT_MODEL,
    max_steps: int = 100,
    temperature: float = 0.3,
    verbose: bool = True,
    use_function_call: bool = True,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one Super Mario episode with GPT-5.4."""
    game_name = "super_mario"
    env = make_orak_env(game_name, max_steps=max_steps)
    task = ORAK_GAMES[game_name]["task"]
    action_names = ORAK_GAMES[game_name]["action_names"]

    obs, info = env.reset()
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while step_count < max_steps:
        action, reasoning = gpt54_orak_action(
            state_nl=obs,
            game_name=game_name,
            action_names=action_names,
            model=model,
            temperature=temperature,
            use_function_call=use_function_call,
        )

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        exp = Experience(
            state=obs,
            action=str(action),
            reward=float(reward),
            next_state=next_obs,
            done=done,
            intentions=reasoning,
            tasks=task,
        )
        exp.idx = step_count
        experiences.append(exp)

        if verbose:
            reason_short = (
                (reasoning[:80] + "...") if reasoning and len(reasoning) > 80
                else reasoning
            )
            print(
                f"  step {step_count}: action={action}, reward={reward:.3f}, "
                f"cum={total_reward:.3f}, reason={reason_short}"
            )

        obs = next_obs
        info = next_info
        step_count += 1
        if done:
            break

    env.close()

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="orak",
        game_name=game_name,
    )
    episode.set_outcome()

    stats: Dict[str, Any] = {
        "game": game_name,
        "display_name": "Super Mario Bros",
        "steps": step_count,
        "total_reward": total_reward,
        "final_score": info.get("score", 0),
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "gpt54_orak",
    }
    return episode, stats


def run_sc2_episode(
    model: str = DEFAULT_MODEL,
    max_steps: int = 1000,
    temperature: float = 0.3,
    verbose: bool = True,
    use_function_call: bool = True,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one StarCraft II episode with GPT-5.4."""
    game_name = "star_craft"
    env = make_orak_env(game_name, max_steps=max_steps)
    task = ORAK_GAMES[game_name]["task"]
    action_names = ORAK_GAMES[game_name]["action_names"]

    obs, info = env.reset()
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while step_count < max_steps:
        action, reasoning = gpt54_orak_action(
            state_nl=obs,
            game_name=game_name,
            action_names=action_names,
            model=model,
            temperature=temperature,
            use_function_call=use_function_call,
        )

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        exp = Experience(
            state=obs,
            action=str(action),
            reward=float(reward),
            next_state=next_obs,
            done=done,
            intentions=reasoning,
            tasks=task,
        )
        exp.idx = step_count
        experiences.append(exp)

        if verbose:
            reason_short = (
                (reasoning[:80] + "...") if reasoning and len(reasoning) > 80
                else reasoning
            )
            print(
                f"  step {step_count}: action={action[:60]}, reward={reward:.3f}, "
                f"cum={total_reward:.3f}, reason={reason_short}"
            )

        obs = next_obs
        info = next_info
        step_count += 1
        if done:
            break

    env.close()

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="orak",
        game_name=game_name,
    )
    episode.set_outcome()

    stats: Dict[str, Any] = {
        "game": game_name,
        "display_name": "StarCraft II (1v1 Protoss)",
        "steps": step_count,
        "total_reward": total_reward,
        "final_score": info.get("score", 0),
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "gpt54_orak",
    }
    return episode, stats


# ═══════════════════════════════════════════════════════════════════════════
# Experience-collection variants
# ═══════════════════════════════════════════════════════════════════════════

def _run_with_experience_collection(
    game_name: str,
    model: str,
    max_steps: int,
    temperature: float,
    verbose: bool,
    experience_buffer_size: int,
    episode_buffer_size: int,
    save_episode_buffer: Optional[str],
) -> Dict[str, Any]:
    """Run a single episode using AgentBufferManager for experience collection."""
    env = make_orak_env(game_name, max_steps=max_steps)
    display = SUPPORTED_GAMES[game_name]["display_name"]
    task = ORAK_GAMES[game_name]["task"]

    buffer_manager = AgentBufferManager(
        experience_buffer_size=experience_buffer_size,
        episode_buffer_size=episode_buffer_size,
    )
    episode = buffer_manager.run_episode(
        env=env,
        task=f"{display} (GPT-5.4)",
        game=GAME_ORAK,
        model=model,
        use_function_call=True,
        temperature=temperature,
        max_steps=max_steps,
        verbose=verbose,
    )
    env.close()

    if save_episode_buffer:
        buffer_manager.save_episode_buffer(save_episode_buffer)
        if verbose:
            print(f"  Saved {game_name} episode buffer to {save_episode_buffer}")

    history = [
        {
            "state": exp.state,
            "action": exp.action,
            "reward": exp.reward,
            "next_state": exp.next_state,
            "done": exp.done,
        }
        for exp in episode.experiences
    ]
    return {
        "game": game_name,
        "display_name": display,
        "model": model,
        "obs": episode.experiences[-1].next_state if episode.experiences else "",
        "rewards": episode.get_reward(),
        "info": {"episode_length": len(episode.experiences), "task": task},
        "history": history,
        "steps": len(episode.experiences),
        "episode": episode,
        "buffer_stats": buffer_manager.get_buffer_stats(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Multi-episode game runner
# ═══════════════════════════════════════════════════════════════════════════

def run_game(
    game_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run all episodes for one game and persist results."""
    game_dir = output_dir / game_name
    game_dir.mkdir(parents=True, exist_ok=True)

    max_steps = args.max_steps or SUPPORTED_GAMES[game_name]["max_steps"]
    display = SUPPORTED_GAMES[game_name]["display_name"]
    run_fn = run_mario_episode if game_name == "super_mario" else run_sc2_episode

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx in range(args.episodes):
        print(f"\n  [{display}] Episode {ep_idx + 1}/{args.episodes}")

        if args.use_experience_collection:
            suffix = f"_{game_name}.json"
            save_path = (
                args.save_episode_buffer.replace(".json", suffix)
                if args.save_episode_buffer
                else None
            )
            try:
                result = _run_with_experience_collection(
                    game_name=game_name,
                    model=args.model,
                    max_steps=max_steps,
                    temperature=args.temperature,
                    verbose=not args.quiet,
                    experience_buffer_size=args.experience_buffer_size,
                    episode_buffer_size=args.episode_buffer_size,
                    save_episode_buffer=save_path,
                )
                ep = result.get("episode")
                if ep:
                    episode_buffer.add_episode(ep)
                all_stats.append({
                    "game": game_name,
                    "episode_index": ep_idx,
                    "steps": result["steps"],
                    "total_reward": result.get("rewards", 0),
                    "model": args.model,
                    "agent_type": "gpt54_orak",
                })
                print(f"    Steps: {result['steps']}, Reward: {result.get('rewards', 0):.3f}")
            except Exception as e:
                print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
                traceback.print_exc()
                continue
        else:
            try:
                episode, stats = run_fn(
                    model=args.model,
                    max_steps=max_steps,
                    temperature=args.temperature,
                    verbose=not args.quiet,
                    use_function_call=not args.no_function_call,
                )
                stats["episode_index"] = ep_idx
                print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.3f}")

                episode_buffer.add_episode(episode)
                all_stats.append(stats)

                ep_data = episode.to_dict()
                ep_data["metadata"] = stats
                ep_path = game_dir / f"episode_{ep_idx:03d}.json"
                with open(ep_path, "w", encoding="utf-8") as f:
                    json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

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

    summary: Dict[str, Any] = {
        "game": game_name,
        "display_name": display,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "gpt54_orak",
        "total_episodes": len([s for s in all_stats if "error" not in s]),
        "target_episodes": args.episodes,
        "max_steps": max_steps,
        "elapsed_seconds": round(elapsed, 2),
        "episode_stats": all_stats,
    }
    if all_stats:
        rewards = [s.get("total_reward", 0) for s in all_stats if "error" not in s]
        steps = [s.get("steps", 0) for s in all_stats if "error" not in s]
        if rewards:
            summary["mean_reward"] = sum(rewards) / len(rewards)
            summary["mean_steps"] = sum(steps) / len(steps)

    summary_path = game_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Combined runner
# ═══════════════════════════════════════════════════════════════════════════

def run_all(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Run selected games sequentially and return combined results."""
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else _CODEBASE_ROOT / "orak_gpt54_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.game == "both":
        games = ["super_mario", "star_craft"]
    else:
        games = [args.game]

    results: Dict[str, Dict[str, Any]] = {}
    for game_name in games:
        display = SUPPORTED_GAMES[game_name]["display_name"]
        print(f"\n{'='*78}")
        print(f"  GPT-5.4 Evaluation — {display}")
        print(f"  Model: {args.model}")
        print(f"{'='*78}\n")

        summary = run_game(game_name, args, output_dir)
        results[game_name] = summary
        _print_game_summary(game_name, summary)

    return results


def _print_game_summary(game_name: str, summary: Dict[str, Any]) -> None:
    display = summary.get("display_name", game_name)
    print(f"\n{'-'*78}")
    print(f"  {display} — Summary (GPT-5.4)")
    print(f"{'-'*78}")
    print(f"  Episodes:     {summary.get('total_episodes', 0)}/{summary.get('target_episodes', 0)}")
    print(f"  Elapsed:      {summary.get('elapsed_seconds', 0):.2f}s")
    if "mean_reward" in summary:
        print(f"  Mean Reward:  {summary['mean_reward']:.3f}")
        print(f"  Mean Steps:   {summary['mean_steps']:.1f}")
    print(f"{'-'*78}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPT-5.4 evaluation for Super Mario and StarCraft II (Orak)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Super Mario with GPT-5.4 (default 3 episodes)
  python evaluate_orak/test_orak_mario_sc2_gpt54.py --game super_mario

  # StarCraft II, 5 episodes, 500 max steps
  python evaluate_orak/test_orak_mario_sc2_gpt54.py --game star_craft \\
      --episodes 5 --max_steps 500

  # Both games
  python evaluate_orak/test_orak_mario_sc2_gpt54.py --game both

  # With experience collection
  python evaluate_orak/test_orak_mario_sc2_gpt54.py --game super_mario \\
      --use_experience_collection \\
      --save_episode_buffer output/gpt54_orak.json

  # Fall back to dummy_agent.language_agent_action (no structured CoT)
  python evaluate_orak/test_orak_mario_sc2_gpt54.py --game star_craft \\
      --no_function_call
""",
    )

    parser.add_argument(
        "--game", type=str, default="super_mario",
        choices=["super_mario", "star_craft", "both"],
        help="Which game(s) to run (default: super_mario)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes per game (default: 3)",
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Max steps per episode (None = game default: Mario=100, SC2=1000)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--no_function_call", action="store_true",
        help="Skip structured function-calling; use dummy_agent.language_agent_action",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: orak_gpt54_output/)")

    exp_group = parser.add_argument_group("Experience collection")
    exp_group.add_argument(
        "--use_experience_collection", action="store_true",
        help="Collect experiences/episodes into buffers via AgentBufferManager",
    )
    exp_group.add_argument(
        "--experience_buffer_size", type=int, default=10_000,
        help="Experience replay buffer capacity (default: 10000)",
    )
    exp_group.add_argument(
        "--episode_buffer_size", type=int, default=1_000,
        help="Episode buffer capacity (default: 1000)",
    )
    exp_group.add_argument(
        "--save_episode_buffer", type=str, default=None,
        help="Path to save episode buffer as JSON (suffixed per game)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        try:
            from api_keys import open_router_api_key as _k
            api_key = _k
        except Exception:
            pass
    if not api_key:
        print(
            "[WARNING] No API key found (OPENROUTER_API_KEY / OPENAI_API_KEY / "
            "api_keys.py). LLM calls may fall back to default actions."
        )

    games_label = args.game if args.game != "both" else "super_mario + star_craft"
    print("=" * 78)
    print("  GPT-5.4 Orak Evaluation — Super Mario & StarCraft II")
    print("=" * 78)
    print(f"  Game(s):     {games_label}")
    print(f"  Model:       {args.model}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Temperature: {args.temperature}")
    if args.max_steps:
        print(f"  Max steps:   {args.max_steps}")
    else:
        print(f"  Max steps:   per-game default (Mario=100, SC2=1000)")
    if args.use_experience_collection:
        print(f"  Experience:  buffer={args.experience_buffer_size}, "
              f"episodes={args.episode_buffer_size}")
    print(f"  Func-call:   {not args.no_function_call}")
    print("=" * 78)

    t_global = time.time()
    results = run_all(args)
    total_elapsed = time.time() - t_global

    print("\n" + "=" * 78)
    print("  OVERALL RESULTS")
    print("=" * 78)
    for game, res in results.items():
        display = res.get("display_name", game)
        eps = res.get("total_episodes", 0)
        mr = res.get("mean_reward")
        elapsed = res.get("elapsed_seconds", 0)
        reward_str = f", mean_reward={mr:.3f}" if mr is not None else ""
        print(f"  {display:30s}: {eps} episodes in {elapsed:.1f}s{reward_str}")
    print(f"  Total time:   {total_elapsed:.2f}s")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
