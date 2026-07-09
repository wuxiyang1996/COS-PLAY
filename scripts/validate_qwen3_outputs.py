#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate Qwen3-8B (or other model) rollout outputs: report and optionally remove invalid/empty runs.

A run is considered VALID if the game dir has at least one real episode:
  - at least one episode_NNN.json (excluding episode_buffer.json), OR
  - rollouts.jsonl exists and has at least one line.

Otherwise the run is INVALID (empty or failed).

Usage (from Game-AI-Agent root):

  # Report only (default: output/Qwen3-8B)
  python -m scripts.validate_qwen3_outputs

  # Custom base dir
  python -m scripts.validate_qwen3_outputs --base_dir output/MyModel

  # Remove invalid run dirs
  python -m scripts.validate_qwen3_outputs --remove
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_valid_game_dir(game_dir: Path) -> tuple[bool, str]:
    """
    Return (valid, reason). Valid iff at least one episode exists.
    """
    if not game_dir.is_dir():
        return False, "not a directory"

    episode_files = [
        f for f in game_dir.glob("episode_*.json")
        if f.name != "episode_buffer.json"
    ]
    if episode_files:
        return True, f"{len(episode_files)} episode file(s)"

    rollouts_jsonl = game_dir / "rollouts.jsonl"
    if rollouts_jsonl.exists():
        try:
            lines = sum(1 for _ in open(rollouts_jsonl, encoding="utf-8"))
            if lines > 0:
                return True, f"rollouts.jsonl has {lines} line(s)"
        except OSError:
            pass
        return False, "rollouts.jsonl empty or unreadable"

    # Optional: check rollout_summary for 0 successful
    summary_path = game_dir / "rollout_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                data = json.load(f)
            total = data.get("total_episodes", 0)
            stats = data.get("episode_stats", [])
            valid_stats = [s for s in stats if "error" not in s]
            if total == 0 or len(valid_stats) == 0:
                return False, "rollout_summary: 0 successful episodes"
        except (json.JSONDecodeError, OSError):
            pass

    return False, "no episode files or rollouts.jsonl"


def scan_base_dir(base_dir: Path):
    """Scan layout model/game/timestamp. Yield (game_parent_dir, game_name, run_dir, valid, reason)."""
    if not base_dir.is_dir():
        return
    for game_name_dir in sorted(base_dir.iterdir()):
        if not game_name_dir.is_dir():
            continue
        game_name = game_name_dir.name
        for run_dir in sorted(game_name_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            valid, reason = is_valid_game_dir(run_dir)
            yield game_name_dir, game_name, run_dir, valid, reason


def main():
    parser = argparse.ArgumentParser(
        description="Validate rollout outputs and optionally remove invalid/empty runs",
    )
    parser.add_argument(
        "--base_dir", type=str, default=None,
        help="Base dir to scan (default: output/Qwen3-8B)",
    )
    parser.add_argument(
        "--remove", action="store_true",
        help="Remove invalid game dirs (and timestamp dirs that become empty)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    codebase_root = script_dir.parent
    base_dir = Path(args.base_dir) if args.base_dir else codebase_root / "output" / "Qwen3-8B"
    if not base_dir.is_dir():
        print(f"Base dir does not exist: {base_dir}")
        return

    results = list(scan_base_dir(base_dir))
    valid_list = [(ts, g, path, reason) for ts, g, path, v, reason in results if v]
    invalid_list = [(ts, g, path, reason) for ts, g, path, v, reason in results if not v]

    print(f"Base dir: {base_dir}")
    print(f"Valid runs:   {len(valid_list)}")
    print(f"Invalid runs: {len(invalid_list)}")
    print()

    if valid_list:
        print("VALID (kept):")
        for game_parent, game_name, run_dir, reason in valid_list:
            print(f"  {game_name}/{run_dir.name}  ({reason})")
        print()

    if invalid_list:
        print("INVALID:")
        for game_parent, game_name, run_dir, reason in invalid_list:
            print(f"  {game_name}/{run_dir.name}  — {reason}")
        print()

    if args.remove and invalid_list:
        import shutil
        removed_dirs = []
        for game_parent, game_name, run_dir, _ in invalid_list:
            if run_dir.exists():
                shutil.rmtree(run_dir)
                removed_dirs.append(f"{game_name}/{run_dir.name}")
        for d in removed_dirs:
            print(f"  Removed: {d}")
        # Remove game dirs that are now empty (no timestamp runs left)
        for game_parent in sorted({gp for gp, _, _, _ in invalid_list}):
            if game_parent.exists() and not any(game_parent.iterdir()):
                shutil.rmtree(game_parent)
                print(f"  Removed empty game dir: {game_parent.name}")
    elif args.remove and not invalid_list:
        print("Nothing to remove (no invalid runs).")


if __name__ == "__main__":
    main()
