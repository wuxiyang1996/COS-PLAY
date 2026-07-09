#!/usr/bin/env python
"""
Batch cold-start rollout generation: 100 episodes per GamingAgent game.

Generates decision-making agent trajectories using GPT-5-mini, stores them in
the Episode/Experience format used by the co-evolution framework so they can be
directly ingested by the skill pipeline and trainer.

Output structure (cold_start/output/<game_name>/):
  - episode_NNN.json        Individual episode (Episode.to_dict())
  - episode_buffer.json     All episodes in Episode_Buffer format
  - rollouts.jsonl          Append-friendly JSONL (one Episode per line)
  - rollout_summary.json    Run-level stats for this game

The JSONL and episode_buffer formats are compatible with:
  - trainer/skillbank/ingest_rollouts.py (TrajectoryForEM conversion)
  - skill_agents/pipeline.py (SkillBankAgent.ingest_episodes)
  - data_structure/experience.py (Episode_Buffer.load_from_json)

Usage (from Game-AI-Agent root):

    # Prefer OpenRouter (used by default): set open_router_api_key in api_keys.py or:
    export OPENROUTER_API_KEY="sk-or-..."
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

    # All available games, 100 episodes each
    python cold_start/run_100_rollouts.py

    # Specific game(s)
    python cold_start/run_100_rollouts.py --games twenty_forty_eight sokoban tetris

    # Fewer episodes (for testing)
    python cold_start/run_100_rollouts.py --episodes 5 --max_steps 30

    # Resume an interrupted run (skips already-completed episodes)
    python cold_start/run_100_rollouts.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
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
# Imports from Game-AI-Agent
# ---------------------------------------------------------------------------
from API_func import ask_model
from data_structure.experience import Experience, Episode, Episode_Buffer
from decision_agents.dummy_agent import (
    language_agent_action,
    _default_action,
    GAME_GAMINGAGENT,
)

# Reuse the existing cold_start module
from cold_start.generate_cold_start import (
    GAME_REGISTRY,
    ColdStartEnvWrapper,
    get_cold_start_max_steps,
    run_dummy_agent_episode,
    run_vlm_agent_episode,
    label_trajectory,
)

_DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def count_existing_episodes(game_dir: Path) -> int:
    """Count episode_NNN.json files already present (for --resume)."""
    if not game_dir.exists():
        return 0
    return sum(1 for f in game_dir.glob("episode_*.json") if f.name != "episode_buffer.json")


def save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict[str, Any]):
    """Append one episode to a JSONL file (trainer-compatible format)."""
    record = episode.to_dict()
    record["rollout_metadata"] = stats
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def save_game_summary(
    game_name: str,
    game_dir: Path,
    all_stats: List[Dict[str, Any]],
    elapsed: float,
    args: argparse.Namespace,
    max_steps_used: Optional[int] = None,
):
    """Save per-game rollout summary."""
    summary = {
        "game": game_name,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": args.agent_type,
        "total_episodes": len(all_stats),
        "target_episodes": args.episodes,
        "max_steps": max_steps_used if max_steps_used is not None else args.max_steps,
        "labeled": not args.no_label,
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
    run_fn,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run all episodes for a single game and save outputs."""
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

    effective_max_steps = args.max_steps if args.max_steps is not None else get_cold_start_max_steps(game_name)

    for ep_idx in range(start_idx, args.episodes):
        print(f"\n  [{game_name}] Episode {ep_idx + 1}/{args.episodes}")

        try:
            env = ColdStartEnvWrapper(game_name, max_steps=effective_max_steps)
            episode, stats = run_fn(
                env=env,
                game_name=game_name,
                model=args.model,
                max_steps=effective_max_steps,
                verbose=args.verbose,
            )
            env.close()

            stats["episode_index"] = ep_idx
            print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

            if not args.no_label:
                episode = label_trajectory(episode, args.model)

            episode_buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = episode.to_dict()
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

    summary = save_game_summary(game_name, game_dir, all_stats, elapsed, args, max_steps_used=effective_max_steps)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate 100 cold-start rollouts per game for co-evolution framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        help="Games to generate rollouts for (default: all available)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes per game (default: 100)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (default: per-game natural end)")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        help="LLM model for agent and labeling")
    parser.add_argument("--agent_type", type=str, default="dummy",
                        choices=["dummy", "vlm"],
                        help="Agent type: 'dummy' (language_agent_action) or 'vlm' (VLMDecisionAgent)")
    parser.add_argument("--no_label", action="store_true",
                        help="Skip trajectory labeling (faster, unlabeled only)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run (skip completed episodes)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print step-by-step details")
    parser.add_argument("--output_dir", type=str, default=None,
                        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from api_keys import open_router_api_key
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or (open_router_api_key and open_router_api_key.strip()))
    except Exception:
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if not has_key:
        print("[WARNING] No API key set. LLM calls will fail.")
        print("  Prefer: open_router_api_key in api_keys.py or export OPENROUTER_API_KEY='sk-or-...'")
        print("  Or: export OPENAI_API_KEY='sk-...'")

    if args.games:
        requested = args.games
    else:
        requested = list(GAME_REGISTRY.keys())

    available_games = []
    skipped_games = []
    for g in requested:
        if g not in GAME_REGISTRY:
            print(f"[WARNING] Game '{g}' not in registry, skipping.")
            skipped_games.append(g)
            continue
        if GAME_REGISTRY[g]["env_class"] is None:
            print(f"[WARNING] Game '{g}' env class not importable, skipping.")
            skipped_games.append(g)
            continue
        available_games.append(g)

    if not available_games:
        print("[ERROR] No games available. Ensure GamingAgent is installed.")
        sys.exit(1)

    run_fn = run_vlm_agent_episode if args.agent_type == "vlm" else run_dummy_agent_episode

    print("=" * 78)
    print("  Cold-Start Batch Rollout Generation")
    print("=" * 78)
    print(f"  Games:      {', '.join(available_games)}")
    if skipped_games:
        print(f"  Skipped:    {', '.join(skipped_games)}")
    print(f"  Episodes:   {args.episodes} per game")
    print(f"  Max steps:  {'per-game (natural end)' if args.max_steps is None else args.max_steps}")
    print(f"  Model:      {args.model}")
    print(f"  Agent:      {args.agent_type}")
    print(f"  Labeling:   {not args.no_label}")
    print(f"  Resume:     {args.resume}")
    print(f"  Output:     {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []

    for game_name in available_games:
        print(f"\n{'━' * 78}")
        print(f"  GAME: {game_name} ({args.episodes} episodes)")
        print(f"{'━' * 78}")

        summary = run_game_rollouts(game_name, args, run_fn, output_dir)
        game_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": args.agent_type,
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,  # None means per-game natural end
        "labeled": not args.no_label,
        "total_elapsed_seconds": overall_elapsed,
        "games_completed": [g for g in available_games],
        "games_skipped": skipped_games,
        "per_game_summaries": game_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  BATCH ROLLOUT GENERATION COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Games processed: {len(available_games)}")
    total_eps = sum(
        s.get("total_episodes", 0) for s in game_summaries if not s.get("skipped")
    )
    print(f"  Total episodes:  {total_eps}")
    print(f"  Elapsed:         {overall_elapsed:.1f}s")
    print(f"  Output:          {output_dir}")
    print(f"  Master summary:  {master_path}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
