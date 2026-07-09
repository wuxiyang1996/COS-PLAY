"""
Utilities for loading cold-start rollout outputs into the co-evolution framework.

Provides converters from Episode → RolloutRecord (for trainer) and Episode → list
(for skill pipeline), plus convenience loaders for the JSONL and episode_buffer formats.

Usage::

    from cold_start.load_rollouts import (
        load_episodes_from_jsonl,
        load_episode_buffer,
        episodes_to_rollout_records,
    )

    # Load for skill pipeline
    episodes = load_episodes_from_jsonl("cold_start/output/tetris/rollouts.jsonl")
    skill_agent.ingest_episodes(episodes)

    # Load for trainer
    records = episodes_to_rollout_records(episodes)
    trajectories = ingest_rollouts(records)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_structure.experience import Episode, Episode_Buffer, Experience
from trainer.common.metrics import RolloutRecord, RolloutStep


def load_episodes_from_jsonl(jsonl_path: str) -> List[Episode]:
    """Load Episode objects from a JSONL file (one Episode.to_dict() per line)."""
    episodes: List[Episode] = []
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ep = Episode.from_dict(d)
                episodes.append(ep)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [WARNING] Skipping malformed line {line_num}: {e}")
                continue

    return episodes


def load_episode_buffer(buffer_path: str) -> Episode_Buffer:
    """Load an Episode_Buffer from the episode_buffer.json file."""
    return Episode_Buffer.load_from_json(buffer_path)


def episode_to_rollout_record(episode: Episode) -> RolloutRecord:
    """Convert an Episode to a RolloutRecord for the trainer pipeline.

    Maps Experience fields to RolloutStep fields:
      - state → obs_id (observation identifier/text)
      - action → action
      - action_type → action_type
      - reward → r_env
      - reward_details → r_follow, r_cost, r_total
      - sub_tasks → active_skill_id
      - done → done
    """
    steps: List[RolloutStep] = []
    for i, exp in enumerate(episode.experiences):
        rd = exp.reward_details or {}

        step = RolloutStep(
            step=exp.idx if exp.idx is not None else i,
            obs_id=exp.summary_state or exp.state or "",
            action=str(exp.action) if exp.action is not None else "",
            action_type=exp.action_type or "primitive",
            r_env=float(exp.reward) if exp.reward is not None else 0.0,
            r_follow=float(rd.get("r_follow", 0.0)),
            r_cost=float(rd.get("r_cost", 0.0)),
            r_total=float(rd.get("r_total", exp.reward or 0.0)),
            done=bool(exp.done),
            episode_id=episode.episode_id or "",
            active_skill_id=exp.sub_tasks if isinstance(exp.sub_tasks, str) else None,
        )
        steps.append(step)

    record = RolloutRecord(
        episode_id=episode.episode_id or "",
        env_name=episode.env_name or "",
        game_name=episode.game_name or "",
        steps=steps,
    )
    record.finalize()
    return record


def episodes_to_rollout_records(episodes: List[Episode]) -> List[RolloutRecord]:
    """Batch-convert Episodes to RolloutRecords for trainer ingestion."""
    return [episode_to_rollout_record(ep) for ep in episodes]


def load_all_game_rollouts(output_dir: str) -> Dict[str, List[Episode]]:
    """Load all rollout episodes from the output directory, organized by game.

    Returns a dict mapping game_name → list of Episode objects.
    """
    out = Path(output_dir)
    result: Dict[str, List[Episode]] = {}

    if not out.exists():
        return result

    for game_dir in sorted(out.iterdir()):
        if not game_dir.is_dir():
            continue
        game_name = game_dir.name
        jsonl = game_dir / "rollouts.jsonl"
        if jsonl.exists():
            result[game_name] = load_episodes_from_jsonl(str(jsonl))
        else:
            buffer_path = game_dir / "episode_buffer.json"
            if buffer_path.exists():
                buf = load_episode_buffer(str(buffer_path))
                result[game_name] = list(buf.buffer)

    return result
