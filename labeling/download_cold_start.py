#!/usr/bin/env python3
"""Download pre-generated cold-start data from HuggingFace.

Downloads the dataset and restructures it to match the layout that the
training pipeline expects:

    labeling/output/gpt54_skill_labeled/
    ├── tetris/
    │   ├── episode_000.json
    │   ├── episode_001.json
    │   └── labeling_summary.json
    ├── candy_crush/
    │   └── ...
    ├── grpo_coldstart/
    │   ├── tetris/
    │   │   ├── action_taking.jsonl
    │   │   └── skill_selection.jsonl
    │   └── ...
    └── labeling_batch_summary.json

Usage:
    python labeling/download_cold_start.py
    python labeling/download_cold_start.py --games tetris candy_crush
    python labeling/download_cold_start.py --output-dir labeling/output/gpt54_skill_labeled
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "IntelligenceLab/Cos-Play-Cold-Start"
ALL_GAMES = [
    "avalon", "candy_crush", "diplomacy", "pokemon_red",
    "sokoban", "super_mario", "tetris", "twenty_forty_eight",
]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "gpt54_skill_labeled"


def restructure(raw_dir: Path, output_dir: Path, games: list[str]):
    """Convert HuggingFace layout into the format the training pipeline expects."""
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_dir = raw_dir / "data" / "episodes"
    for game in games:
        jsonl = episodes_dir / f"{game}.jsonl"
        if not jsonl.exists():
            print(f"  {game}: no data found, skipping")
            continue
        game_dir = output_dir / game
        game_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(jsonl) as f:
            for line in f:
                ep = json.loads(line)
                out_file = game_dir / f"episode_{count:03d}.json"
                with open(out_file, "w") as fout:
                    json.dump(ep, fout, indent=2, ensure_ascii=False)
                count += 1
        print(f"  {game}: {count} episodes -> {game_dir}")

        summary_file = episodes_dir / "summaries" / f"{game}_summary.json"
        if summary_file.exists():
            shutil.copy2(summary_file, game_dir / "labeling_summary.json")

    grpo_src = raw_dir / "data" / "grpo_coldstart"
    if grpo_src.is_dir():
        grpo_dst = output_dir / "grpo_coldstart"
        if grpo_dst.exists():
            shutil.rmtree(grpo_dst)
        shutil.copytree(grpo_src, grpo_dst)
        print(f"  grpo_coldstart/ copied")

    batch_summary = raw_dir / "data" / "labeling_batch_summary.json"
    if batch_summary.exists():
        shutil.copy2(batch_summary, output_dir / "labeling_batch_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Download COS-PLAY cold-start data")
    parser.add_argument(
        "--games", nargs="+", default=ALL_GAMES, choices=ALL_GAMES,
        help="Games to download (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp) / "raw"
        print(f"Downloading from {REPO_ID} ...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
        print(f"\nRestructuring into pipeline-ready format...")
        restructure(raw_dir, args.output_dir, args.games)

    print(f"\nDone! Data ready at {args.output_dir}")


if __name__ == "__main__":
    main()
