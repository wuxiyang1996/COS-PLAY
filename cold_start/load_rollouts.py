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


# ---------------------------------------------------------------------------
# Honest pass@1 aggregator
# ---------------------------------------------------------------------------
#
# The per-task ``rollout_summary.json`` already has ``solved`` /
# ``unscored`` / ``pass_rate`` since the May-2026 fix, but historical
# runs do not. This aggregator walks an entire run directory and
# computes the breakdown directly from each
# ``<run_dir>/<domain>/<task_id>/rollout_summary.json``, treating any
# episode with ``eval_score is None`` as 0 by default. Use this for
# runs collected before the eval-score-on-FAIL/truncate fix landed.

def aggregate_run_pass_at_1(
    run_dir: str,
    *,
    treat_null_as_zero: bool = True,
) -> Dict[str, Any]:
    """Compute honest pass@1 over every per-task ``rollout_summary.json``
    under ``run_dir``.

    Parameters
    ----------
    run_dir : str
        Directory like ``Cold-start-out-osworld/<timestamp>`` that
        contains one subdirectory per domain, each containing one
        subdirectory per task, each containing ``rollout_summary.json``.
    treat_null_as_zero : bool
        When True (default), episodes with ``eval_score == None`` are
        counted as failures (numerator unchanged, denominator includes
        them). When False, those episodes are excluded — matching the
        old buggy ``mean_eval_score`` behaviour. Use ``False`` only when
        comparing to the legacy summary.

    Returns
    -------
    dict
        Schema::
          {
            "run_dir": str,
            "treat_null_as_zero": bool,
            "total_tasks": int,
            "solved": int,
            "errored": int,
            "unscored": int,
            "pass_rate": float,
            "per_domain": {
              "<domain>": {
                "tasks": int,
                "solved": int,
                "errored": int,
                "unscored": int,
                "pass_rate": float,
              },
              ...
            }
          }
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")

    per_domain: Dict[str, Dict[str, Any]] = {}
    total_tasks = 0
    total_solved = 0
    total_errored = 0
    total_unscored = 0

    for domain_dir in sorted(root.iterdir()):
        if not domain_dir.is_dir():
            continue
        if domain_dir.name.startswith("_"):
            continue
        dom = domain_dir.name
        d_tasks = 0
        d_solved = 0
        d_errored = 0
        d_unscored = 0
        for task_dir in sorted(domain_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            rs = task_dir / "rollout_summary.json"
            if not rs.is_file():
                continue
            try:
                summary = json.loads(rs.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            d_tasks += 1
            episode_stats = summary.get("episode_stats") or []
            # Pass@1 = best of episodes for this task. (Most cold-start
            # runs use episodes_per_task=1 so this collapses to the
            # single episode's score.)
            best_score: Optional[float] = None
            had_error = False
            for ep in episode_stats:
                if ep.get("error"):
                    had_error = True
                    continue
                sc = ep.get("eval_score")
                if isinstance(sc, (int, float)):
                    best_score = max(best_score or 0.0, float(sc))
            if had_error and not episode_stats:
                d_errored += 1
                continue
            if best_score is None:
                if treat_null_as_zero:
                    d_unscored += 1
                else:
                    d_tasks -= 1  # exclude entirely
                continue
            if best_score > 0:
                d_solved += 1
        if d_tasks > 0:
            per_domain[dom] = {
                "tasks": d_tasks,
                "solved": d_solved,
                "errored": d_errored,
                "unscored": d_unscored,
                "pass_rate": d_solved / d_tasks if d_tasks else 0.0,
            }
            total_tasks += d_tasks
            total_solved += d_solved
            total_errored += d_errored
            total_unscored += d_unscored

    return {
        "run_dir": str(root),
        "treat_null_as_zero": treat_null_as_zero,
        "total_tasks": total_tasks,
        "solved": total_solved,
        "errored": total_errored,
        "unscored": total_unscored,
        "pass_rate": total_solved / total_tasks if total_tasks else 0.0,
        "per_domain": per_domain,
    }


def _format_aggregate_table(agg: Dict[str, Any]) -> str:
    """Pretty-print ``aggregate_run_pass_at_1`` output."""
    lines = []
    lines.append(f"Run:                   {agg['run_dir']}")
    lines.append(f"Treat null as zero:    {agg['treat_null_as_zero']}")
    lines.append("")
    header = (
        f"  {'Domain':<22} {'Tasks':>6} {'Solved':>7} {'Pass@1':>8} "
        f"{'Errored':>8} {'Unscored':>9}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for dom, st in sorted(agg["per_domain"].items()):
        lines.append(
            f"  {dom:<22} {st['tasks']:>6} {st['solved']:>7} "
            f"{st['pass_rate']*100:>7.1f}% {st['errored']:>8} "
            f"{st['unscored']:>9}"
        )
    lines.append("  " + "-" * (len(header) - 2))
    lines.append(
        f"  {'TOTAL':<22} {agg['total_tasks']:>6} {agg['solved']:>7} "
        f"{agg['pass_rate']*100:>7.1f}% {agg['errored']:>8} "
        f"{agg['unscored']:>9}"
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entry point: ``python -m cold_start.load_rollouts --root <dir>``."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate honest pass@1 from a cold-start run directory. "
            "Walks <root>/<domain>/<task_id>/rollout_summary.json and "
            "treats unscored episodes (eval_score=None) as failures by "
            "default — recovering the metric from runs collected "
            "before the eval-on-FAIL/truncate fix landed."
        )
    )
    parser.add_argument(
        "--root", required=True, type=str,
        help="Run directory (e.g. Cold-start-out-osworld/<timestamp>).",
    )
    parser.add_argument(
        "--treat-null-as-zero", dest="treat_null_as_zero",
        action="store_true", default=True,
        help="Count unscored episodes as 0 (default).",
    )
    parser.add_argument(
        "--legacy-mean", dest="treat_null_as_zero",
        action="store_false",
        help="Match the old (buggy) mean_eval_score behaviour: "
             "exclude unscored episodes from the denominator.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output the raw aggregate JSON instead of the table.",
    )
    args = parser.parse_args()

    agg = aggregate_run_pass_at_1(
        args.root, treat_null_as_zero=args.treat_null_as_zero,
    )
    if args.json:
        print(json.dumps(agg, indent=2))
    else:
        print(_format_aggregate_table(agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
