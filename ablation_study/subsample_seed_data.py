#!/usr/bin/env python3
"""Subsample GPT-5.4 SFT cold-start data by number of unique episodes.

Used for R6 seed-scaling curve: vary N = number of expert episodes
used to train the decision agent's SFT cold-start.

The source layout under ``--src`` is expected to be:

    <src>/<game>/action_taking.jsonl     (rows have field 'episode')
    <src>/<game>/skill_selection.jsonl   (rows have field 'episode')

Output mirrors that layout under ``--out``, keeping only rows whose
``episode`` field belongs to the first ``--n`` unique episode IDs
encountered in the source file order (deterministic).

Note: action_taking and skill_selection are subsampled by the SAME
N episodes so the two decision adapters see exactly the same demos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Set


def _first_n_episodes(path: Path, n: int) -> List[str]:
    seen: List[str] = []
    seen_set: Set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line).get("episode")
            except json.JSONDecodeError:
                continue
            if ep is None or ep in seen_set:
                continue
            seen.append(ep)
            seen_set.add(ep)
            if len(seen) >= n:
                break
    return seen


def _filter_rows(src: Path, dst: Path, keep_episodes: Set[str]) -> int:
    n_kept = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("episode") in keep_episodes:
                fout.write(json.dumps(row) + "\n")
                n_kept += 1
    return n_kept


def subsample_game(
    src_root: Path,
    out_root: Path,
    game: str,
    n: int,
) -> dict:
    src_game = src_root / game
    out_game = out_root / game
    files = ["action_taking.jsonl", "skill_selection.jsonl"]
    for f in files:
        if not (src_game / f).exists():
            raise FileNotFoundError(f"missing source: {src_game / f}")

    # Use action_taking ordering as the canonical episode order; both
    # files share the same set/order in our pipeline.
    keep_list = _first_n_episodes(src_game / "action_taking.jsonl", n)
    keep_set = set(keep_list)
    if len(keep_list) < n:
        print(f"[subsample] WARNING: requested n={n} but only "
              f"{len(keep_list)} unique episodes available")

    stats = {"game": game, "requested_n": n, "actual_n": len(keep_list),
             "kept_episodes": keep_list, "files": {}}
    for f in files:
        kept = _filter_rows(src_game / f, out_game / f, keep_set)
        stats["files"][f] = kept

    # Write a manifest for traceability.
    (out_game / "subsample_manifest.json").write_text(
        json.dumps(stats, indent=2)
    )
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Source root, e.g. labeling/output/gpt54_skill_labeled/grpo_coldstart")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output root, e.g. runs/seed_scaling/data/N20")
    ap.add_argument("--n", type=int, required=True,
                    help="Number of unique episodes to keep")
    ap.add_argument("--games", type=str, nargs="+", default=["candy_crush"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    overall = {"n": args.n, "src": str(args.src), "out": str(args.out),
               "games": []}
    for game in args.games:
        s = subsample_game(args.src, args.out, game, args.n)
        overall["games"].append(s)
        print(f"[subsample] {game:>14s}  n={args.n:3d}  "
              f"kept_episodes={s['actual_n']:3d}  "
              f"rows: {s['files']}")
    (args.out / "subsample_summary.json").write_text(
        json.dumps(overall, indent=2)
    )
    print(f"[subsample] manifest → {args.out / 'subsample_summary.json'}")


if __name__ == "__main__":
    main()
