#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script: run dummy_agent in GamingAgent (LMGame-Bench) via GamingAgentNLWrapper.

Uses env_wrappers.GamingAgentNLWrapper and agents.dummy_agent.language_agent_action
to run episodes where the agent chooses actions from natural language state.

Usage (from the Game-AI-Agent codebase root):

    # Ensure GamingAgent is on PYTHONPATH
    export PYTHONPATH=$(pwd):$(pwd)/../GamingAgent
    python evaluate_gamingagent/test_gamingagent_dummy.py --game twenty_forty_eight

Options:
    --game        Game name (default: twenty_forty_eight)
    --max_steps   Episode length (default: 50)
    --episodes    Number of episodes (default: 1)
    --mode        Agent mode: "llm" | "random_nl" | "fallback" (default: fallback)
    --model       LLM model for "llm" mode (default: gpt-4o-mini)
    --verbose     Print full NL observations each step
    --list-games  List available games and exit
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
try:
    from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper, state_to_natural_language
except ImportError as e:
    print(f"[ERROR] Cannot import env_wrappers: {e}")
    print("  Run from Game-AI-Agent root. Ensure env_wrappers is on PYTHONPATH.")
    sys.exit(1)

try:
    from decision_agents.dummy_agent import (
        language_agent_action,
        GAME_GAMINGAGENT,
        _default_action,
    )
except ImportError:
    GAME_GAMINGAGENT = "gamingagent"
    language_agent_action = None
    _default_action = lambda g: "stay" if g != "gamingagent" else "no-op"

try:
    from evaluate_gamingagent.gym_like import make_gaming_env, list_games
except ImportError:
    try:
        from gym_like import make_gaming_env, list_games
    except ImportError:
        make_gaming_env = None
        list_games = lambda: []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_nl_action(action_names: List[str]) -> str:
    """Pick a random valid action from the game's action names."""
    if not action_names:
        return "stay"
    return random.choice(action_names)


def choose_action(
    obs_nl: str,
    action_names: List[str],
    mode: str,
    model: str,
) -> str:
    """Choose an NL action for one step."""
    if mode == "llm" and language_agent_action:
        try:
            action = language_agent_action(
                state_nl=obs_nl,
                game=GAME_GAMINGAGENT,
                model=model,
                use_function_call=True,
                temperature=0.3,
            )
            if isinstance(action, str) and action:
                return action
        except Exception as e:
            print(f"  [WARNING] LLM action failed: {e}")
    elif mode == "random_nl":
        return random_nl_action(action_names)
    # fallback
    return action_names[0] if action_names else "stay"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    game: str,
    max_steps: int,
    mode: str,
    model: str,
    verbose: bool,
    episode_id: int,
) -> Dict[str, Any]:
    """Run one GamingAgent episode with the dummy agent."""

    if make_gaming_env is None:
        print("[ERROR] GamingAgent gym_like not found. Add GamingAgent to PYTHONPATH.")
        return {"error": "GamingAgent not found"}

    base_env = make_gaming_env(game=game, max_steps=max_steps)
    env = GamingAgentNLWrapper(base_env)

    obs_nl, info = env.reset()
    action_names = info.get("action_names", [])

    total_reward = 0.0
    step_count = 0
    actions_log: List[str] = []

    print(f"\n{'='*78}")
    print(f"  Episode {episode_id + 1}  |  Game: {game}  |  Max steps: {max_steps}  |  Mode: {mode}")
    print(f"{'='*78}")

    if verbose:
        print(f"\nInitial state (first 500 chars):\n{obs_nl[:500]}...")
    else:
        print(f"  Action names: {action_names}")

    terminated = False
    truncated = False

    while not (terminated or truncated) and step_count < max_steps:
        step_count += 1
        action = choose_action(obs_nl, action_names, mode, model)
        actions_log.append(action)

        try:
            obs_nl, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"\n  [ERROR at step {step_count}] {e}")
            break

        total_reward += reward
        action_names = info.get("action_names", action_names)

        if verbose:
            print(f"\n  Step {step_count}: action={action}, reward={reward:.2f}, cum={total_reward:.2f}")
            print(f"    State (first 300 chars): {obs_nl[:300]}...")
        else:
            print(f"  Step {step_count}: {action} -> reward {reward:.2f}")

    env.close()

    action_counts: Dict[str, int] = {}
    for a in actions_log:
        action_counts[a] = action_counts.get(a, 0) + 1

    result = {
        "episode_id": episode_id,
        "game": game,
        "max_steps": max_steps,
        "mode": mode,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "action_distribution": action_counts,
        "perf_score": info.get("perf_score", 0.0),
    }

    print(f"\n{'-'*78}")
    print(f"  Episode {episode_id + 1} Summary")
    print(f"{'-'*78}")
    print(f"  Steps:         {step_count}")
    print(f"  Total Reward:  {total_reward:.2f}")
    print(f"  Perf Score:    {info.get('perf_score', 'N/A')}")
    print(f"  Terminated:    {terminated}  |  Truncated: {truncated}")
    print(f"  Actions:       {action_counts}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test dummy agent in GamingAgent (LMGame-Bench)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--game", type=str, default="twenty_forty_eight", help="Game name")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--mode", type=str, default="fallback",
                        choices=["llm", "random_nl", "fallback"], help="Agent mode")
    parser.add_argument("--model", type=str, default="gpt-5.4", help="LLM model for llm mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full NL observations")
    parser.add_argument("--list-games", action="store_true", help="List available games and exit")

    args = parser.parse_args()

    if args.list_games:
        games = list_games() if callable(list_games) else []
        if games:
            print("Available games:")
            for g in games:
                print(f"  {g}")
        else:
            print("Could not list games. Ensure GamingAgent is on PYTHONPATH.")
        return

    if args.mode == "llm":
        if not os.environ.get("OPENAI_API_KEY"):
            print("[WARNING] OPENAI_API_KEY not set. LLM mode may fall back to default actions.")

    print("GamingAgent Dummy Agent Test")
    print(f"  Game:       {args.game}")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Mode:       {args.mode}")
    if args.mode == "llm":
        print(f"  Model:      {args.model}")

    all_results = []
    t0 = time.time()

    for ep in range(args.episodes):
        r = run_episode(
            game=args.game,
            max_steps=args.max_steps,
            mode=args.mode,
            model=args.model,
            verbose=args.verbose,
            episode_id=ep,
        )
        if "error" not in r:
            all_results.append(r)

    elapsed = time.time() - t0

    if all_results:
        print(f"\n{'='*78}")
        print("  OVERALL RESULTS")
        print(f"{'='*78}")
        total_rewards = [r["total_reward"] for r in all_results]
        total_steps = [r["steps"] for r in all_results]
        print(f"  Mean Reward:  {sum(total_rewards) / len(total_rewards):.2f}")
        print(f"  Mean Steps:   {sum(total_steps) / len(total_steps):.1f}")
        print(f"  Total Time:   {elapsed:.2f}s")
        print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
