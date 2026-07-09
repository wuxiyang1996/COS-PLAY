#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LMGame-Bench Evaluation Suite
==============================

Run the full benchmark (or a subset) across all supported GamingAgent games
with a single command.  Results are printed as a summary table and optionally
saved to a JSON file.

Usage examples:

    # Run ALL available games with default settings
    python evaluate_gamingagent/run_benchmark.py --model gpt-5.4

    # Run specific games
    python evaluate_gamingagent/run_benchmark.py --games tictactoe candy_crush tetris

    # Run only games in a category
    python evaluate_gamingagent/run_benchmark.py --category custom

    # Dry-run: just show what would be run
    python evaluate_gamingagent/run_benchmark.py --dry-run

    # Override episodes / steps for all games
    python evaluate_gamingagent/run_benchmark.py --episodes 5 --max-steps 100

    # Save results to JSON
    python evaluate_gamingagent/run_benchmark.py --output results.json

    # List all games in the benchmark
    python evaluate_gamingagent/run_benchmark.py --list
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"

for p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

from evaluate_gamingagent.game_configs import (
    GAME_CONFIGS,
    ALL_GAME_NAMES,
    AVAILABLE_GAME_NAMES,
    UNAVAILABLE_GAME_NAMES,
    TOTAL_GAMES,
    AVAILABLE_GAMES,
    GameConfig,
)

try:
    from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
except ImportError:
    GamingAgentNLWrapper = None

try:
    from evaluate_gamingagent.gym_like import make_gaming_env
except ImportError:
    make_gaming_env = None

try:
    from decision_agents.dummy_agent import (
        language_agent_action,
        GAME_GAMINGAGENT,
    )
except ImportError:
    GAME_GAMINGAGENT = "gamingagent"
    language_agent_action = None

import random


# ── Helpers ─────────────────────────────────────────────────────────────────

def _choose_action(
    obs_nl: str,
    action_names: List[str],
    mode: str,
    model: str,
) -> str:
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
        except Exception:
            pass
    elif mode == "random_nl":
        return random.choice(action_names) if action_names else "stay"
    return action_names[0] if action_names else "stay"


def _run_single_episode(
    game: str,
    max_steps: int,
    mode: str,
    model: str,
    episode_idx: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one episode, return a results dict."""
    if make_gaming_env is None or GamingAgentNLWrapper is None:
        return {"error": "Missing imports (gym_like or GamingAgentNLWrapper)"}

    base_env = make_gaming_env(game=game, max_steps=max_steps)
    env = GamingAgentNLWrapper(base_env)

    obs_nl, info = env.reset()
    action_names = info.get("action_names", [])

    total_reward = 0.0
    step_count = 0
    terminated = truncated = False

    while not (terminated or truncated) and step_count < max_steps:
        step_count += 1
        action = _choose_action(obs_nl, action_names, mode, model)
        try:
            obs_nl, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            if verbose:
                print(f"    [ERROR step {step_count}] {e}")
            break
        total_reward += reward
        action_names = info.get("action_names", action_names)

    env.close()
    return {
        "episode": episode_idx,
        "steps": step_count,
        "reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "perf_score": info.get("perf_score", 0.0),
    }


def _run_game(
    cfg: GameConfig,
    mode: str,
    model: str,
    episodes_override: Optional[int],
    max_steps_override: Optional[int],
    verbose: bool,
) -> Dict[str, Any]:
    """Run all episodes for one game, return aggregated results."""
    max_steps = max_steps_override if max_steps_override else cfg.max_steps
    episodes = episodes_override if episodes_override else cfg.episodes

    saved_env = os.environ.copy()
    for k, v in cfg.env_vars.items():
        os.environ[k] = v

    print(f"\n  {'─'*60}")
    print(f"  {cfg.display_name}  ({cfg.name})")
    print(f"    category={cfg.category}  episodes={episodes}  max_steps={max_steps}")
    print(f"  {'─'*60}")

    episode_results: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep in range(episodes):
        ep_result = _run_single_episode(
            game=cfg.name,
            max_steps=max_steps,
            mode=mode,
            model=model,
            episode_idx=ep,
            verbose=verbose,
        )
        if "error" in ep_result:
            print(f"    Episode {ep+1}: ERROR - {ep_result['error']}")
            break
        episode_results.append(ep_result)
        r = ep_result["reward"]
        s = ep_result["steps"]
        print(f"    Episode {ep+1}/{episodes}:  reward={r:.2f}  steps={s}")

    elapsed = time.time() - t0

    for k, v in cfg.env_vars.items():
        if k in saved_env:
            os.environ[k] = saved_env[k]
        else:
            os.environ.pop(k, None)

    if not episode_results:
        return {
            "game": cfg.name,
            "display_name": cfg.display_name,
            "category": cfg.category,
            "status": "FAILED",
            "error": "No episodes completed",
            "elapsed_s": elapsed,
        }

    rewards = [r["reward"] for r in episode_results]
    steps = [r["steps"] for r in episode_results]
    return {
        "game": cfg.name,
        "display_name": cfg.display_name,
        "category": cfg.category,
        "status": "OK",
        "episodes_run": len(episode_results),
        "max_steps": max_steps,
        "mean_reward": sum(rewards) / len(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_steps": sum(steps) / len(steps),
        "elapsed_s": round(elapsed, 2),
        "episode_details": episode_results,
    }


# ── Table printing ──────────────────────────────────────────────────────────

def _print_results_table(results: List[Dict[str, Any]], model: str, mode: str):
    """Print a nicely formatted summary table."""
    W = 90
    print(f"\n{'━'*W}")
    print(f"  LMGame-Bench Results   model={model}  mode={mode}")
    print(f"{'━'*W}")

    hdr = f"  {'Game':<22} {'Cat':<8} {'Eps':>4} {'Steps':>7} {'Mean R':>9} {'Min R':>8} {'Max R':>8} {'Time':>7}  {'Status':<6}"
    print(hdr)
    print(f"  {'─'*86}")

    for r in results:
        name = r["display_name"]
        cat = r["category"]
        status = r["status"]
        if status == "OK":
            eps = r["episodes_run"]
            steps = f"{r['mean_steps']:.0f}"
            mean_r = f"{r['mean_reward']:.2f}"
            min_r = f"{r['min_reward']:.2f}"
            max_r = f"{r['max_reward']:.2f}"
            elapsed = f"{r['elapsed_s']:.1f}s"
        else:
            eps = 0
            steps = mean_r = min_r = max_r = "—"
            elapsed = f"{r.get('elapsed_s', 0):.1f}s"
        print(f"  {name:<22} {cat:<8} {eps:>4} {steps:>7} {mean_r:>9} {min_r:>8} {max_r:>8} {elapsed:>7}  {status:<6}")

    print(f"{'━'*W}")

    total_elapsed = sum(r.get("elapsed_s", 0) for r in results)
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = len(results) - ok
    print(f"  Games: {ok} passed, {fail} failed  |  Total time: {total_elapsed:.1f}s")
    print(f"{'━'*W}\n")


def _print_game_inventory():
    """Print the full game inventory table."""
    W = 90
    print(f"\n{'━'*W}")
    print(f"  LMGame-Bench Game Inventory  ({TOTAL_GAMES} games, {AVAILABLE_GAMES} available)")
    print(f"{'━'*W}")

    hdr = f"  {'#':>2}  {'Game':<22} {'Display':<20} {'Cat':<8} {'Steps':>6} {'Eps':>4}  {'Status':<12}"
    print(hdr)
    print(f"  {'─'*86}")

    for i, name in enumerate(ALL_GAME_NAMES, 1):
        cfg = GAME_CONFIGS[name]
        status = "AVAILABLE" if cfg.available else "ROM NEEDED"
        print(f"  {i:>2}  {name:<22} {cfg.display_name:<20} {cfg.category:<8} {cfg.max_steps:>6} {cfg.episodes:>4}  {status:<12}")

    print(f"{'━'*W}")
    print(f"  Available: {', '.join(AVAILABLE_GAME_NAMES)}")
    if UNAVAILABLE_GAME_NAMES:
        print(f"  ROM needed: {', '.join(UNAVAILABLE_GAME_NAMES)}")
    print(f"{'━'*W}\n")


# ── Main ────────────────────────────────────────────────────────────────────

def run_benchmark(
    games: Optional[List[str]] = None,
    category: Optional[str] = None,
    model: str = "gpt-5.4",
    mode: str = "llm",
    episodes_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run the benchmark and return results list.

    This is the main API entry point for programmatic use:

        from evaluate_gamingagent.run_benchmark import run_benchmark
        results = run_benchmark(games=["tictactoe", "tetris"], model="gpt-5.4")
    """
    if games:
        game_list = games
    elif category:
        game_list = [n for n in AVAILABLE_GAME_NAMES
                     if GAME_CONFIGS[n].category == category]
    else:
        game_list = list(AVAILABLE_GAME_NAMES)

    unavailable = [g for g in game_list if g not in AVAILABLE_GAME_NAMES]
    if unavailable:
        print(f"[WARNING] Skipping unavailable games: {unavailable}")
        game_list = [g for g in game_list if g in AVAILABLE_GAME_NAMES]

    unknown = [g for g in game_list if g not in GAME_CONFIGS]
    if unknown:
        print(f"[ERROR] Unknown games: {unknown}")
        print(f"  Available: {AVAILABLE_GAME_NAMES}")
        return []

    if not game_list:
        print("[ERROR] No games to run.")
        return []

    W = 90
    print(f"{'━'*W}")
    print(f"  LMGame-Bench Evaluation Suite")
    print(f"{'━'*W}")
    print(f"  Model:      {model}")
    print(f"  Mode:       {mode}")
    print(f"  Games:      {len(game_list)} — {', '.join(game_list)}")
    if episodes_override:
        print(f"  Episodes:   {episodes_override} (override)")
    if max_steps_override:
        print(f"  Max steps:  {max_steps_override} (override)")
    print(f"{'━'*W}")

    all_results: List[Dict[str, Any]] = []
    total_t0 = time.time()

    for game_name in game_list:
        cfg = GAME_CONFIGS[game_name]
        result = _run_game(
            cfg=cfg,
            mode=mode,
            model=model,
            episodes_override=episodes_override,
            max_steps_override=max_steps_override,
            verbose=verbose,
        )
        all_results.append(result)

    total_elapsed = time.time() - total_t0
    _print_results_table(all_results, model, mode)

    if output_path:
        output_data = {
            "benchmark": "LMGame-Bench",
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "mode": mode,
            "total_elapsed_s": round(total_elapsed, 2),
            "total_games_in_benchmark": TOTAL_GAMES,
            "games_run": len(all_results),
            "results": all_results,
        }
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"  Results saved to: {output_path}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="LMGame-Bench Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--games", nargs="+", default=None,
                        help="Specific games to run (space-separated)")
    parser.add_argument("--category", type=str, default=None,
                        choices=["custom", "retro", "zoo"],
                        help="Run all available games in a category")
    parser.add_argument("--model", type=str, default="gpt-5.4",
                        help="LLM model name (default: gpt-5.4)")
    parser.add_argument("--mode", type=str, default="llm",
                        choices=["llm", "random_nl", "fallback"],
                        help="Agent mode (default: llm)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episodes per game")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max steps per episode")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output per step")
    parser.add_argument("--list", action="store_true",
                        help="List all games and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be run, don't execute")

    args = parser.parse_args()

    if args.list:
        _print_game_inventory()
        return

    games = args.games
    if not games and not args.category:
        games = None

    if args.dry_run:
        if games:
            game_list = [g for g in games if g in AVAILABLE_GAME_NAMES]
        elif args.category:
            game_list = [n for n in AVAILABLE_GAME_NAMES
                         if GAME_CONFIGS[n].category == args.category]
        else:
            game_list = list(AVAILABLE_GAME_NAMES)

        print(f"\n[DRY RUN] Would run {len(game_list)} games:\n")
        for g in game_list:
            cfg = GAME_CONFIGS[g]
            eps = args.episodes or cfg.episodes
            steps = args.max_steps or cfg.max_steps
            print(f"  {cfg.display_name:<22} episodes={eps}  max_steps={steps}")
        print()
        return

    run_benchmark(
        games=games,
        category=args.category,
        model=args.model,
        mode=args.mode,
        episodes_override=args.episodes,
        max_steps_override=args.max_steps,
        output_path=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
