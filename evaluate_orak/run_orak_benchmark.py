#!/usr/bin/env python
"""
Run Orak benchmark (krafton-ai/Orak) games through the Game-AI-Agent pipeline.

Supports 12 games across 6 genres, with both the VLM decision agent and
the dummy language agent.

Games: twenty_fourty_eight, baba_is_you, super_mario, street_fighter,
       star_craft, star_craft_multi, slay_the_spire, pokemon_red,
       darkest_dungeon, pwaat, her_story, minecraft, stardew_valley

Usage:

    export PYTHONPATH="$(pwd):$(pwd)/../Orak/src:$PYTHONPATH"

    # Run a single Orak game with the dummy agent
    python evaluate_orak/run_orak_benchmark.py \\
        --game star_craft --episodes 1 --max_steps 100 --model gpt-5-mini

    # Run all Orak games
    python evaluate_orak/run_orak_benchmark.py --all_games

    # Run with VLM decision agent
    python evaluate_orak/run_orak_benchmark.py \\
        --game twenty_fourty_eight --agent_type vlm --episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent

for p in [str(CODEBASE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate_orak.orak_nl_wrapper import make_orak_env, ORAK_GAMES
from data_structure.experience import Experience, Episode, Episode_Buffer

try:
    from decision_agents.agent import VLMDecisionAgent, run_episode_vlm_agent
except ImportError:
    VLMDecisionAgent = None
    run_episode_vlm_agent = None

from decision_agents.dummy_agent import (
    language_agent_action,
    detect_game,
    GAME_ORAK,
)


ORAK_GAME_NAMES = list(ORAK_GAMES.keys())


def run_dummy_episode(
    game_name: str,
    model: str,
    max_steps: int,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode with the dummy language agent on an Orak game."""
    env = make_orak_env(game_name, max_steps=max_steps)
    task = ORAK_GAMES[game_name]["task"]
    action_names = ORAK_GAMES[game_name]["action_names"]

    obs, info = env.reset()
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    game_type = detect_game(obs)

    while step_count < max_steps:
        prompt = obs
        if action_names:
            prompt += f"\n\nValid actions: {', '.join(action_names[:20])}. Choose one."

        action = language_agent_action(
            state_nl=prompt,
            game=game_type,
            model=model,
            use_function_call=True,
            temperature=0.3,
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
            tasks=task,
        )
        exp.idx = step_count
        experiences.append(exp)

        if verbose:
            print(f"  step {step_count}: action={str(action)[:60]}, reward={reward:.3f}, cum={total_reward:.3f}")

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

    stats = {
        "game": game_name,
        "steps": step_count,
        "total_reward": total_reward,
        "final_score": info.get("score", 0),
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "dummy",
    }
    return episode, stats


def run_vlm_episode(
    game_name: str,
    model: str,
    max_steps: int,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode with VLMDecisionAgent on an Orak game."""
    if VLMDecisionAgent is None or run_episode_vlm_agent is None:
        raise ImportError("VLMDecisionAgent not available. Check decision_agents.agent imports.")

    env = make_orak_env(game_name, max_steps=max_steps)
    task = ORAK_GAMES[game_name]["task"]

    agent = VLMDecisionAgent(model=model, game=GAME_ORAK)
    episode = run_episode_vlm_agent(env, agent=agent, task=task, max_steps=max_steps, verbose=verbose)

    env.close()

    meta = episode.metadata or {}
    stats = {
        "game": game_name,
        "steps": meta.get("steps", len(episode.experiences)),
        "total_reward": episode.get_reward(),
        "total_shaped_reward": episode.get_total_reward(),
        "model": model,
        "agent_type": "vlm",
    }
    return episode, stats


def main():
    parser = argparse.ArgumentParser(
        description="Run Orak benchmark games through Game-AI-Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--game", type=str, default="twenty_fourty_eight",
                        choices=ORAK_GAME_NAMES,
                        help="Orak game to evaluate")
    parser.add_argument("--all_games", action="store_true",
                        help="Run all Orak games")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per game")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (None = game default)")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        help="LLM model")
    parser.add_argument("--agent_type", type=str, default="dummy",
                        choices=["dummy", "vlm"],
                        help="Agent type")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    default_steps = {
        "twenty_fourty_eight": 1000,
        "baba_is_you": 200,
        "super_mario": 100,
        "street_fighter": 500,
        "star_craft": 1000,
        "star_craft_multi": 1000,
        "slay_the_spire": 500,
        "pokemon_red": 200,
        "darkest_dungeon": 300,
        "pwaat": 300,
        "her_story": 400,
        "minecraft": 200,
        "stardew_valley": 300,
    }

    output_dir = Path(args.output_dir) if args.output_dir else CODEBASE_ROOT / "orak_benchmark_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    games = ORAK_GAME_NAMES if args.all_games else [args.game]
    run_fn = run_vlm_episode if args.agent_type == "vlm" else run_dummy_episode

    print("=" * 78)
    print("  Orak Benchmark")
    print("=" * 78)
    print(f"  Games:      {', '.join(games)}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Model:      {args.model}")
    print(f"  Agent:      {args.agent_type}")
    print(f"  Output:     {output_dir}")
    print("=" * 78)

    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for game_name in games:
        max_steps = args.max_steps or default_steps.get(game_name, 1000)
        print(f"\n{'─' * 78}")
        print(f"  Game: {game_name} (max_steps={max_steps})")
        print(f"{'─' * 78}")

        game_dir = output_dir / game_name
        game_dir.mkdir(parents=True, exist_ok=True)
        episode_buffer = Episode_Buffer(buffer_size=1000)

        for ep_idx in range(args.episodes):
            print(f"\n  Episode {ep_idx + 1}/{args.episodes}")
            try:
                episode, stats = run_fn(
                    game_name=game_name,
                    model=args.model,
                    max_steps=max_steps,
                    verbose=args.verbose,
                )
                print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.3f}")

                episode_buffer.add_episode(episode)
                all_stats.append(stats)

                ep_path = game_dir / f"episode_{ep_idx:03d}.json"
                ep_data = episode.to_dict()
                ep_data["metadata"] = stats
                with open(ep_path, "w", encoding="utf-8") as f:
                    json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

            except Exception as e:
                print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        buffer_path = game_dir / "episode_buffer.json"
        episode_buffer.save_to_json(str(buffer_path))
        print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    elapsed = time.time() - t0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": args.agent_type,
        "games": games,
        "episodes_per_game": args.episodes,
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    summary_path = output_dir / "orak_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  ORAK BENCHMARK COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Total episodes: {len(all_stats)}")
    if all_stats:
        rewards = [s["total_reward"] for s in all_stats]
        steps = [s["steps"] for s in all_stats]
        print(f"  Mean reward:  {sum(rewards) / len(rewards):.3f}")
        print(f"  Mean steps:   {sum(steps) / len(steps):.1f}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print(f"  Output:       {output_dir}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
