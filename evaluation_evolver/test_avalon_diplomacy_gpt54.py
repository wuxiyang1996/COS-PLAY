#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-5.4 evaluation for Avalon and Diplomacy via the evolver env wrappers.

Runs one (or both) game environments using GPT-5.4 as the backbone model,
with all players/powers controlled by the same LLM through
decision_agents.dummy_agent.language_agent_action.

Usage (from Game-AI-Agent root, with AgentEvolver on PYTHONPATH):

    export PYTHONPATH="$(pwd):$(pwd)/../AgentEvolver:$PYTHONPATH"
    export OPENROUTER_API_KEY="sk-or-..."   # or OPENAI_API_KEY

    # Run both games (default)
    python evaluation_evolver/test_avalon_diplomacy_gpt54.py

    # Avalon only
    python evaluation_evolver/test_avalon_diplomacy_gpt54.py --game avalon

    # Diplomacy only, 10 phases
    python evaluation_evolver/test_avalon_diplomacy_gpt54.py --game diplomacy --max_phases 10

    # Both games with experience collection
    python evaluation_evolver/test_avalon_diplomacy_gpt54.py --use_experience_collection \\
        --save_episode_buffer output/gpt54_evolver_episodes.json

Visualizer (watch rollout in browser):
    python AgentEvolver/games/web/server.py
    # Avalon:    http://localhost:8000/avalon/observe
    # Diplomacy: http://localhost:8000/diplomacy/observe
    python evaluation_evolver/test_avalon_diplomacy_gpt54.py --visualizer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Path setup — ensure codebase root, AgentEvolver and AI_Diplomacy are importable
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent          # Game-AI-Agent/
_workspace = _root.parent                                # parent dir (e.g. game_agent/)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# AgentEvolver can live as a sibling of Game-AI-Agent or inside it
for _candidate in [_workspace / "AgentEvolver", _root / "AgentEvolver"]:
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

# AI_Diplomacy: same sibling-or-child search
for _candidate in [_workspace / "AI_Diplomacy", _root / "AI_Diplomacy"]:
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from env_wrappers.avalon_nl_wrapper import AvalonNLWrapper
from env_wrappers.diplomacy_nl_wrapper import DiplomacyNLWrapper

from decision_agents.dummy_agent import (
    GAME_AVALON,
    GAME_DIPLOMACY,
    language_agent_action,
    AgentBufferManager,
)

DEFAULT_MODEL = "gpt-5.4"


# ═══════════════════════════════════════════════════════════════════════════
# Visualizer helpers
# ═══════════════════════════════════════════════════════════════════════════

def _push_visualizer_state(visualizer_url: str, state: Dict[str, Any]) -> None:
    """POST state to the games web server push-state API."""
    url = f"{visualizer_url.rstrip('/')}/api/push-state"
    data = json.dumps(state).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print(f"[visualizer] push-state returned {resp.status}")
    except Exception as e:
        print(f"[visualizer] failed to push state: {e}")


def _avalon_visualizer_state(
    env: AvalonNLWrapper, info: dict, status: str = "running"
) -> Dict[str, Any]:
    roles_data = [
        {"role_id": int(rid), "role_name": str(rname), "is_good": bool(ig)}
        for rid, rname, ig in env.roles
    ]
    state = {
        "game": "avalon",
        "status": status,
        "phase": info.get("phase", 0),
        "mission_id": info.get("turn", 0),
        "round_id": info.get("round", 0),
        "leader": info.get("leader", 0),
        "roles": roles_data,
        "logs": getattr(env, "_discussion_log", []) or [],
    }
    if env.done and hasattr(env, "env") and env.env is not None:
        state["good_wins"] = getattr(env.env, "good_victory", None)
    return state


def _diplomacy_visualizer_state(
    env: DiplomacyNLWrapper, info: dict, status: str = "running"
) -> Dict[str, Any]:
    state = {
        "game": "diplomacy",
        "status": status,
        "phase": info.get("phase", ""),
        "round": info.get("round", 0),
    }
    map_svg = env.get_map_svg()
    if map_svg:
        state["map_svg"] = map_svg
    if env.game and hasattr(env.game, "powers"):
        state["sc_counts"] = {
            p: len(power.centers) for p, power in env.game.powers.items()
        }
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Avalon runner
# ═══════════════════════════════════════════════════════════════════════════

def run_avalon(
    num_players: int = 5,
    max_phases: int = 50,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    verbose: bool = True,
    visualizer_url: Optional[str] = None,
    use_experience_collection: bool = False,
    experience_buffer_size: int = 10_000,
    episode_buffer_size: int = 1_000,
    save_episode_buffer: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one full Avalon game with all players controlled by GPT-5.4."""

    if use_experience_collection:
        return _run_avalon_experience(
            num_players=num_players,
            max_phases=max_phases,
            model=model,
            seed=seed,
            verbose=verbose,
            experience_buffer_size=experience_buffer_size,
            episode_buffer_size=episode_buffer_size,
            save_episode_buffer=save_episode_buffer,
        )

    env = AvalonNLWrapper(num_players=num_players, seed=seed)
    obs, info = env.reset()
    history: List[dict] = []
    rewards: Dict[int, float] = {}
    step_count = 0

    if visualizer_url:
        _push_visualizer_state(
            visualizer_url,
            _avalon_visualizer_state(env, info, status="running"),
        )

    while not env.done and step_count < max_phases:
        active = info.get("active_players", [])
        actions: Dict[int, Any] = {}
        for pid in active:
            state_nl = obs.get(pid, "")
            if not state_nl:
                continue
            action = language_agent_action(
                state_nl,
                game=GAME_AVALON,
                model=model,
                use_function_call=True,
                temperature=0.3,
            )
            actions[pid] = action
            if verbose:
                print(f"  Player {pid} action: {action!r}")

        obs, rewards, terminated, truncated, info = env.step(actions)
        history.append({"obs": obs, "rewards": rewards, "info": dict(info)})
        step_count += 1

        if visualizer_url:
            status = "finished" if env.done else "running"
            _push_visualizer_state(
                visualizer_url,
                _avalon_visualizer_state(env, info, status=status),
            )

        if verbose:
            phase_name = info.get("phase_name", info.get("phase", ""))
            print(f"Step {step_count} | Phase: {phase_name} | Done: {terminated}")

    if verbose and env.done:
        good_victory = info.get("good_victory")
        print(f"Game over. Good wins: {good_victory}. Rewards: {rewards}")

    return {
        "game": "avalon",
        "model": model,
        "obs": obs,
        "rewards": rewards if env.done else {},
        "info": info,
        "history": history,
        "steps": step_count,
    }


def _run_avalon_experience(
    num_players: int = 5,
    max_phases: int = 50,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    verbose: bool = True,
    experience_buffer_size: int = 10_000,
    episode_buffer_size: int = 1_000,
    save_episode_buffer: Optional[str] = None,
) -> Dict[str, Any]:
    env = AvalonNLWrapper(num_players=num_players, seed=seed)
    buffer_manager = AgentBufferManager(
        experience_buffer_size=experience_buffer_size,
        episode_buffer_size=episode_buffer_size,
    )
    episode = buffer_manager.run_episode(
        env=env,
        task=f"Avalon game with {num_players} players (GPT-5.4)",
        game=GAME_AVALON,
        model=model,
        use_function_call=True,
        temperature=0.3,
        max_steps=max_phases,
        verbose=verbose,
    )
    if save_episode_buffer:
        buffer_manager.save_episode_buffer(save_episode_buffer)
        if verbose:
            print(f"Saved Avalon episode buffer to {save_episode_buffer}")

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
        "game": "avalon",
        "model": model,
        "obs": episode.experiences[-1].next_state if episode.experiences else "",
        "rewards": episode.get_reward(),
        "info": {"episode_length": len(episode.experiences), "task": episode.task},
        "history": history,
        "steps": len(episode.experiences),
        "episode": episode,
        "buffer_stats": buffer_manager.get_buffer_stats(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Diplomacy runner
# ═══════════════════════════════════════════════════════════════════════════

def run_diplomacy(
    max_phases: int = 5,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    verbose: bool = True,
    visualizer_url: Optional[str] = None,
    use_experience_collection: bool = False,
    experience_buffer_size: int = 10_000,
    episode_buffer_size: int = 1_000,
    save_episode_buffer: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Diplomacy for several phases with all powers controlled by GPT-5.4."""

    if use_experience_collection:
        return _run_diplomacy_experience(
            max_phases=max_phases,
            model=model,
            seed=seed,
            verbose=verbose,
            experience_buffer_size=experience_buffer_size,
            episode_buffer_size=episode_buffer_size,
            save_episode_buffer=save_episode_buffer,
        )

    env = DiplomacyNLWrapper(seed=seed)
    obs, info = env.reset()
    history: List[dict] = []
    rewards: Dict[str, float] = {}
    step_count = 0

    if visualizer_url:
        _push_visualizer_state(
            visualizer_url,
            _diplomacy_visualizer_state(env, info, status="running"),
        )

    while not env.done and step_count < max_phases:
        actions: Dict[str, Union[List[str], str]] = {}
        for power_name, state_nl in obs.items():
            if not state_nl:
                continue
            action = language_agent_action(
                state_nl,
                game=GAME_DIPLOMACY,
                model=model,
                use_function_call=True,
                temperature=0.3,
            )
            if isinstance(action, list):
                actions[power_name] = action
            else:
                actions[power_name] = [action] if action else []
            if verbose:
                orders = actions[power_name]
                preview = orders[:3] if isinstance(orders, list) else []
                print(
                    f"  {power_name}: {len(orders) if isinstance(orders, list) else 0} orders, e.g. {preview}"
                )

        obs, rewards, terminated, truncated, info = env.step(actions)
        history.append({"obs": obs, "rewards": dict(rewards), "info": dict(info)})
        step_count += 1

        if visualizer_url:
            status = "finished" if env.done else "running"
            _push_visualizer_state(
                visualizer_url,
                _diplomacy_visualizer_state(env, info, status=status),
            )

        if verbose:
            phase = info.get("phase", "")
            print(f"Step {step_count} | Phase: {phase} | Done: {terminated}")

    if verbose and env.done:
        print(f"Game over. Rewards (SC count / 18): {rewards}")

    return {
        "game": "diplomacy",
        "model": model,
        "obs": obs,
        "rewards": rewards,
        "info": info,
        "history": history,
        "steps": step_count,
    }


def _run_diplomacy_experience(
    max_phases: int = 5,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    verbose: bool = True,
    experience_buffer_size: int = 10_000,
    episode_buffer_size: int = 1_000,
    save_episode_buffer: Optional[str] = None,
) -> Dict[str, Any]:
    env = DiplomacyNLWrapper(seed=seed)
    buffer_manager = AgentBufferManager(
        experience_buffer_size=experience_buffer_size,
        episode_buffer_size=episode_buffer_size,
    )
    episode = buffer_manager.run_episode(
        env=env,
        task="Diplomacy game (GPT-5.4)",
        game=GAME_DIPLOMACY,
        model=model,
        use_function_call=True,
        temperature=0.3,
        max_steps=max_phases,
        verbose=verbose,
    )
    if save_episode_buffer:
        buffer_manager.save_episode_buffer(save_episode_buffer)
        if verbose:
            print(f"Saved Diplomacy episode buffer to {save_episode_buffer}")

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
        "game": "diplomacy",
        "model": model,
        "obs": episode.experiences[-1].next_state if episode.experiences else "",
        "rewards": episode.get_reward(),
        "info": {"episode_length": len(episode.experiences), "task": episode.task},
        "history": history,
        "steps": len(episode.experiences),
        "episode": episode,
        "buffer_stats": buffer_manager.get_buffer_stats(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Combined runner
# ═══════════════════════════════════════════════════════════════════════════

def run_both(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Run both Avalon and Diplomacy sequentially and return combined results."""
    results: Dict[str, Dict[str, Any]] = {}

    games = args.game if args.game != "both" else ["avalon", "diplomacy"]
    if isinstance(games, str):
        games = [games]

    for game in games:
        print(f"\n{'='*78}")
        print(f"  GPT-5.4 Evaluation — {game.upper()}")
        print(f"  Model: {args.model}")
        print(f"{'='*78}\n")

        t0 = time.time()

        if game == "avalon":
            result = run_avalon(
                num_players=args.num_players,
                max_phases=args.max_phases_avalon,
                model=args.model,
                seed=args.seed,
                verbose=not args.quiet,
                visualizer_url=args.visualizer_url if args.visualizer else None,
                use_experience_collection=args.use_experience_collection,
                experience_buffer_size=args.experience_buffer_size,
                episode_buffer_size=args.episode_buffer_size,
                save_episode_buffer=(
                    args.save_episode_buffer.replace(".json", "_avalon.json")
                    if args.save_episode_buffer
                    else None
                ),
            )
        elif game == "diplomacy":
            result = run_diplomacy(
                max_phases=args.max_phases_diplomacy,
                model=args.model,
                seed=args.seed,
                verbose=not args.quiet,
                visualizer_url=args.visualizer_url if args.visualizer else None,
                use_experience_collection=args.use_experience_collection,
                experience_buffer_size=args.experience_buffer_size,
                episode_buffer_size=args.episode_buffer_size,
                save_episode_buffer=(
                    args.save_episode_buffer.replace(".json", "_diplomacy.json")
                    if args.save_episode_buffer
                    else None
                ),
            )
        else:
            print(f"[WARNING] Unknown game: {game}, skipping.")
            continue

        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 2)
        results[game] = result

        _print_game_summary(game, result, elapsed)

    return results


def _print_game_summary(game: str, result: Dict[str, Any], elapsed: float) -> None:
    print(f"\n{'-'*78}")
    print(f"  {game.upper()} — Summary (GPT-5.4)")
    print(f"{'-'*78}")
    print(f"  Steps/Phases: {result['steps']}")
    print(f"  Elapsed:      {elapsed:.2f}s")

    if game == "avalon":
        good_victory = result["info"].get("good_victory")
        print(f"  Good Victory: {good_victory}")
        print(f"  Rewards:      {result['rewards']}")
    elif game == "diplomacy":
        print(f"  Game Done:    {result['info'].get('is_game_done', 'N/A')}")
        if isinstance(result.get("rewards"), dict):
            for power, sc in sorted(result["rewards"].items()):
                print(f"    {power:10s}: {sc:.3f}")

    if "buffer_stats" in result:
        print(f"  Buffer Stats: {result['buffer_stats']}")
    print(f"{'-'*78}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPT-5.4 evaluation for Avalon and Diplomacy (evolver wrappers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both games with defaults
  python evaluation_evolver/test_avalon_diplomacy_gpt54.py

  # Avalon only, 30 phases, verbose
  python evaluation_evolver/test_avalon_diplomacy_gpt54.py --game avalon --max_phases_avalon 30

  # Diplomacy only, 10 phases, with experience collection
  python evaluation_evolver/test_avalon_diplomacy_gpt54.py --game diplomacy --max_phases_diplomacy 10 \\
      --use_experience_collection --save_episode_buffer output/gpt54_diplomacy.json

  # Both games with visualizer
  python evaluation_evolver/test_avalon_diplomacy_gpt54.py --visualizer
""",
    )

    parser.add_argument(
        "--game",
        type=str,
        default="both",
        choices=["avalon", "diplomacy", "both"],
        help="Which game(s) to run (default: both)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output")

    avalon_group = parser.add_argument_group("Avalon options")
    avalon_group.add_argument(
        "--num_players", type=int, default=5, help="Number of Avalon players (default: 5)"
    )
    avalon_group.add_argument(
        "--max_phases_avalon",
        type=int,
        default=50,
        help="Max phases for Avalon (default: 50)",
    )

    diplomacy_group = parser.add_argument_group("Diplomacy options")
    diplomacy_group.add_argument(
        "--max_phases_diplomacy",
        type=int,
        default=5,
        help="Max phases for Diplomacy (default: 5)",
    )

    viz_group = parser.add_argument_group("Visualizer")
    viz_group.add_argument(
        "--visualizer",
        action="store_true",
        help="Push state to games web server for browser visualization",
    )
    viz_group.add_argument(
        "--visualizer_url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of games web server (default: http://localhost:8000)",
    )

    exp_group = parser.add_argument_group("Experience collection")
    exp_group.add_argument(
        "--use_experience_collection",
        action="store_true",
        help="Collect experiences/episodes into buffers",
    )
    exp_group.add_argument(
        "--experience_buffer_size",
        type=int,
        default=10_000,
        help="Experience replay buffer capacity (default: 10000)",
    )
    exp_group.add_argument(
        "--episode_buffer_size",
        type=int,
        default=1_000,
        help="Episode buffer capacity (default: 1000)",
    )
    exp_group.add_argument(
        "--save_episode_buffer",
        type=str,
        default=None,
        help="Path to save episode buffer as JSON (suffixed per game)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "[WARNING] Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set. "
            "LLM calls may fall back to default actions."
        )

    print("=" * 78)
    print("  GPT-5.4 Avalon + Diplomacy Evaluation (evolver wrappers)")
    print("=" * 78)
    print(f"  Game(s):     {args.game}")
    print(f"  Model:       {args.model}")
    print(f"  Seed:        {args.seed}")
    if args.game in ("avalon", "both"):
        print(f"  Avalon:      {args.num_players} players, max {args.max_phases_avalon} phases")
    if args.game in ("diplomacy", "both"):
        print(f"  Diplomacy:   max {args.max_phases_diplomacy} phases")
    if args.use_experience_collection:
        print(f"  Experience:  buffer={args.experience_buffer_size}, episodes={args.episode_buffer_size}")
    if args.visualizer:
        print(f"  Visualizer:  {args.visualizer_url}")
    print("=" * 78)

    t_global = time.time()
    results = run_both(args)
    total_elapsed = time.time() - t_global

    print("\n" + "=" * 78)
    print("  OVERALL RESULTS")
    print("=" * 78)
    for game, res in results.items():
        print(f"  {game:12s}: {res['steps']} steps in {res.get('elapsed_s', '?')}s")
    print(f"  Total time:   {total_elapsed:.2f}s")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
