#!/usr/bin/env python
"""
Convert ``Cold-start-out{,-gymv}/<run>/...`` rollouts into the
``triples.jsonl`` shape that
``trainer/SFT/schema_gen/data_loader.py`` consumes.

Why this script
---------------
Phase-0 of PLAN-VISUAL-GROUNDING-MILESTONES would normally re-run
``labeling/grounding/collect_gymv.py`` to gather
``(frame, heuristic_schema, vision_schema)`` triples from scratch.
That requires re-stepping every env and re-paying the gpt-5.5 schema
bill (~32k frames × $0.01 ≈ $300+).  We already paid it once: the
canonical SFT cold-start rollouts under
``Cold-start-out-gymv/sft_gpt5p4_e20_s100_stream_*`` and
``Cold-start-out/sft_envw_e20_gpt5p4_*`` carry both the saved frame
(``metadata.frame_path``) and the gpt-5.4 vision schema
(``metadata.schema``) for every step.

This converter just walks those rollouts and emits JSONL rows in the
exact format the schema_gen loader's ``_select_target`` expects::

    {
      "frame_path":      "<absolute path on disk>",
      "vision_schema":   "<state>...</state>",   # gpt-5.4 / gpt-5.5 output
      "heuristic_schema": null,                  # not produced by cold-start
      "env_id":          "Temporal_Airstriker-v0",
      "episode":         0,
      "step":            7,
      "description":     "<episode goal / task>",
      "obs_text":        "<short raw state line>",
      "valid_actions":   [...],
      "domain":          "gymv" | "env_wrappers",
      "schema_source":   "<value of metadata.schema_source>",
      "schema_error":    "<value or null>"
    }

Output layout
-------------
``labeling/output/grounding/{gymv,env_wrappers}/<env_or_game>/triples.jsonl``

The schema_gen loader's ``rglob('triples.jsonl')`` picks them all up.
Per-game and per-corpus stats land in
``labeling/output/grounding/_run_summary.json``.

CLI::

    # Default — pick the most recent gymv + env_wrappers SFT run.
    python labeling/build_schema_gen_triples.py

    # Pin specific runs.
    python labeling/build_schema_gen_triples.py \\
        --gymv-run Cold-start-out-gymv/sft_gpt5p4_e20_s100_stream_<ts> \\
        --envw-run Cold-start-out/sft_envw_e20_gpt5p4_<ts>

    # Smoke (1 episode per game, drops error rows).
    python labeling/build_schema_gen_triples.py --limit-episodes 1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent

DEFAULT_GYMV_PARENT = CODEBASE_ROOT / "Cold-start-out-gymv"
DEFAULT_ENVW_PARENT = CODEBASE_ROOT / "Cold-start-out"
DEFAULT_OUTPUT_ROOT = CODEBASE_ROOT / "labeling" / "output" / "grounding"


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def _latest_run(parent: Path, prefix: str) -> Optional[Path]:
    """Return the most-recently-modified ``<parent>/<prefix>*`` dir."""
    if not parent.is_dir():
        return None
    runs = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def _resolve_gymv_run(arg: Optional[Path]) -> Optional[Path]:
    if arg is not None:
        return arg if arg.is_dir() else None
    return _latest_run(DEFAULT_GYMV_PARENT, "sft_gpt5p4_")


def _resolve_envw_run(arg: Optional[Path]) -> Optional[Path]:
    if arg is not None:
        return arg if arg.is_dir() else None
    return _latest_run(DEFAULT_ENVW_PARENT, "sft_envw_")


# ---------------------------------------------------------------------------
# Per-step row builder
# ---------------------------------------------------------------------------

def _make_triple_row(
    step: Dict[str, Any],
    *,
    domain: str,
    env_id: str,
    episode_idx: int,
    step_idx: int,
    description: str,
    drop_errors: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return a triples.jsonl row for one step, or ``None`` when invalid."""
    meta = step.get("metadata") or {}
    schema = meta.get("schema")
    frame_path = meta.get("frame_path") or step.get("frame_path")

    if not isinstance(schema, str) or not schema.strip():
        return None
    if not frame_path:
        return None
    if drop_errors and meta.get("schema_error"):
        return None
    if not Path(frame_path).exists():
        return None

    obs_text = step.get("state") or step.get("raw_state") or ""
    valid_actions = (
        step.get("available_actions")
        or meta.get("valid_actions")
        or []
    )

    return {
        "frame_path": str(frame_path),
        "vision_schema": schema.strip(),
        "heuristic_schema": None,
        "env_id": env_id,
        "episode": episode_idx,
        "step": step_idx,
        "description": description,
        "obs_text": str(obs_text),
        "valid_actions": list(valid_actions) if isinstance(valid_actions, list) else [],
        "domain": domain,
        "schema_source": meta.get("schema_source"),
        "schema_error": meta.get("schema_error"),
    }


# ---------------------------------------------------------------------------
# Per-episode iterator
# ---------------------------------------------------------------------------

def _iter_episode_rows(
    ep_path: Path,
    *,
    domain: str,
    env_id: str,
    drop_errors: bool,
) -> Iterable[Dict[str, Any]]:
    with ep_path.open("r") as f:
        ep = json.load(f)
    exps = ep.get("experiences") or ep.get("steps") or []
    description = (
        ep.get("task") or ep.get("query") or ep.get("summary") or ""
    )
    if isinstance(description, dict):
        description = description.get("text") or ""
    description = (description or "").strip()
    # ``episode_id`` is a string in newer cold-start dumps; fall back to
    # the episode_NNN.json index if it parses to an int.
    ep_idx_raw = ep.get("episode_id") or ep_path.stem.replace("episode_", "")
    try:
        ep_idx = int(str(ep_idx_raw))
    except (ValueError, TypeError):
        ep_idx = -1

    for s_idx, step in enumerate(exps):
        row = _make_triple_row(
            step,
            domain=domain, env_id=env_id,
            episode_idx=ep_idx, step_idx=s_idx,
            description=description, drop_errors=drop_errors,
        )
        if row is not None:
            yield row


# ---------------------------------------------------------------------------
# Per-corpus drivers
# ---------------------------------------------------------------------------

def _process_gymv(
    run_dir: Path,
    out_root: Path,
    *,
    limit_episodes: Optional[int],
    drop_errors: bool,
) -> Dict[str, Any]:
    """Convert every ``Temporal_*-v0/episode_*.json`` under *run_dir*."""
    domain = "gymv"
    out_corpus = out_root / "gymv"
    out_corpus.mkdir(parents=True, exist_ok=True)
    per_env: List[Dict[str, Any]] = []
    total_rows = 0

    env_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("Temporal_"))
    for env_dir in env_dirs:
        env_id = env_dir.name
        ep_files = sorted(env_dir.glob("episode_*.json"))
        if limit_episodes is not None:
            ep_files = ep_files[:limit_episodes]
        out_env = out_corpus / env_id
        out_env.mkdir(parents=True, exist_ok=True)
        out_path = out_env / "triples.jsonl"
        n_rows = 0
        with out_path.open("w") as fout:
            for ep_path in ep_files:
                for row in _iter_episode_rows(
                    ep_path, domain=domain, env_id=env_id,
                    drop_errors=drop_errors,
                ):
                    fout.write(json.dumps(row) + "\n")
                    n_rows += 1
        per_env.append({"env_id": env_id, "n_episodes": len(ep_files), "n_rows": n_rows})
        total_rows += n_rows
        print(f"[build_schema_gen] gymv/{env_id}: eps={len(ep_files)} rows={n_rows}")

    return {"domain": domain, "n_envs": len(env_dirs), "n_rows": total_rows, "per_env": per_env}


def _process_envw(
    run_dir: Path,
    out_root: Path,
    *,
    limit_episodes: Optional[int],
    drop_errors: bool,
) -> Dict[str, Any]:
    """Convert env_wrappers rollouts (game-ai-agent, orak-mario buckets)."""
    domain = "env_wrappers"
    out_corpus = out_root / "env_wrappers"
    out_corpus.mkdir(parents=True, exist_ok=True)
    per_game: List[Dict[str, Any]] = []
    total_rows = 0

    # Buckets are arbitrary subdirs that contain ``<game>/episode_*.json``.
    bucket_dirs = [p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    for bucket_dir in sorted(bucket_dirs):
        for game_dir in sorted(p for p in bucket_dir.iterdir() if p.is_dir()):
            ep_files = sorted(game_dir.glob("episode_*.json"))
            if not ep_files:
                continue
            if limit_episodes is not None:
                ep_files = ep_files[:limit_episodes]
            game = game_dir.name
            out_game = out_corpus / game
            out_game.mkdir(parents=True, exist_ok=True)
            out_path = out_game / "triples.jsonl"
            n_rows = 0
            with out_path.open("w") as fout:
                for ep_path in ep_files:
                    for row in _iter_episode_rows(
                        ep_path, domain=domain, env_id=game,
                        drop_errors=drop_errors,
                    ):
                        fout.write(json.dumps(row) + "\n")
                        n_rows += 1
            per_game.append({
                "bucket": bucket_dir.name, "game": game,
                "n_episodes": len(ep_files), "n_rows": n_rows,
            })
            total_rows += n_rows
            print(f"[build_schema_gen] env_wrappers/{game}: eps={len(ep_files)} rows={n_rows}")

    return {"domain": domain, "n_games": len(per_game), "n_rows": total_rows, "per_game": per_game}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--gymv-run", type=Path, default=None,
        help=f"Gym-V SFT rollout dir (default: latest under {DEFAULT_GYMV_PARENT}).",
    )
    ap.add_argument(
        "--envw-run", type=Path, default=None,
        help=f"env_wrappers SFT rollout dir (default: latest under {DEFAULT_ENVW_PARENT}).",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    ap.add_argument(
        "--limit-episodes", type=int, default=None,
        help="Process only the first N episodes per env/game (smoke).",
    )
    ap.add_argument(
        "--keep-errors", action="store_true",
        help="Keep rows whose metadata.schema_error is set (default: drop).",
    )
    ap.add_argument(
        "--skip-gymv", action="store_true",
    )
    ap.add_argument(
        "--skip-envw", action="store_true",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    drop_errors = not args.keep_errors
    summary: Dict[str, Any] = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "output_root": str(out_root),
        "drop_errors": drop_errors,
        "limit_episodes": args.limit_episodes,
    }

    if not args.skip_gymv:
        gymv_run = _resolve_gymv_run(args.gymv_run)
        if gymv_run is None:
            print("[build_schema_gen] no gymv run found; skipping (use --gymv-run to pin).")
            summary["gymv"] = {"skipped": True}
        else:
            print(f"[build_schema_gen] gymv run: {gymv_run}")
            summary["gymv_run"] = str(gymv_run)
            summary["gymv"] = _process_gymv(
                gymv_run, out_root,
                limit_episodes=args.limit_episodes,
                drop_errors=drop_errors,
            )

    if not args.skip_envw:
        envw_run = _resolve_envw_run(args.envw_run)
        if envw_run is None:
            print("[build_schema_gen] no env_wrappers run found; skipping (use --envw-run to pin).")
            summary["env_wrappers"] = {"skipped": True}
        else:
            print(f"[build_schema_gen] env_wrappers run: {envw_run}")
            summary["envw_run"] = str(envw_run)
            summary["env_wrappers"] = _process_envw(
                envw_run, out_root,
                limit_episodes=args.limit_episodes,
                drop_errors=drop_errors,
            )

    n_total = (
        summary.get("gymv", {}).get("n_rows", 0)
        + summary.get("env_wrappers", {}).get("n_rows", 0)
    )
    summary["n_total_rows"] = n_total
    summary["completed_at"] = datetime.utcnow().isoformat() + "Z"
    out_path = out_root / "_run_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[build_schema_gen] TOTAL rows: {n_total:,}")
    print(f"[build_schema_gen] summary -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
