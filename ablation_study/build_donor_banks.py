#!/usr/bin/env python3
"""R9 — Cross-game skill transfer: build donor banks (leave-one-out).

For each target game T in {twenty_forty_eight, candy_crush, tetris, super_mario,
avalon, diplomacy}, builds a donor bank = union of the OTHER 5 games' best
skill banks.  Every skill is tagged with `origin_game` so we can analyse
which games' skills get used at inference time.

To avoid skill_id collisions across games, IDs are rewritten as
``<origin_game>::<original_skill_id>``.  This is purely cosmetic — the skill's
strategic_description and protocol remain unchanged.

Sources (best-step banks):
  - twenty_forty_eight : runs/Qwen3-8B_2048_20260322_071227/best/banks/twenty_forty_eight/skill_bank.jsonl  (13 skills)
  - candy_crush        : runs/Qwen3-8B_20260321_213813_(Candy_crush)/best/banks/candy_crush/skill_bank.jsonl (6)
  - tetris             : runs/Qwen3-8B_tetris_20260322_170438/best/banks/tetris/skill_bank.jsonl              (6)
  - super_mario        : runs/Qwen3-8B_super_mario_20260323_030839/best/banks/super_mario/skill_bank.jsonl    (20)
  - avalon             : runs/Qwen3-8B_avalon_20260322_200424/best/banks/avalon/combined_skill_bank.jsonl    (16, good+evil)
  - diplomacy          : runs/Qwen3-8B_diplomacy_20260322_234548/best/banks/diplomacy/combined_skill_bank.jsonl (64, 7 countries)

Output:
  <out_root>/donor_<TARGET>.jsonl                  donor bank for target=<TARGET>
  <out_root>/donor_manifest.json                   provenance + counts
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (game name as known by COSPLAY, path-to-source-bank relative to PROJECT_ROOT)
SOURCE_BANKS: Dict[str, str] = {
    "twenty_forty_eight":
        "runs/Qwen3-8B_2048_20260322_071227/best/banks/twenty_forty_eight/skill_bank.jsonl",
    "candy_crush":
        "runs/Qwen3-8B_20260321_213813_(Candy_crush)/best/banks/candy_crush/skill_bank.jsonl",
    "tetris":
        "runs/Qwen3-8B_tetris_20260322_170438/best/banks/tetris/skill_bank.jsonl",
    "super_mario":
        "runs/Qwen3-8B_super_mario_20260323_030839/best/banks/super_mario/skill_bank.jsonl",
    "avalon":
        "runs/Qwen3-8B_avalon_20260322_200424/best/banks/avalon/combined_skill_bank.jsonl",
    "diplomacy":
        "runs/Qwen3-8B_diplomacy_20260322_234548/best/banks/diplomacy/combined_skill_bank.jsonl",
}

ALL_GAMES = list(SOURCE_BANKS.keys())


def load_bank(path: Path) -> List[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠ skipping bad line in {path}: {e}")
    return rows


def retag(rec: dict, origin: str) -> dict:
    """Annotate a skill record with its origin game; rewrite skill_id to
    avoid collisions across games."""
    skill = rec.get("skill", rec)
    sid_orig = skill.get("skill_id", "UNKNOWN")
    skill["skill_id"] = f"{origin}::{sid_orig}"
    skill["origin_game"] = origin
    skill["origin_skill_id"] = sid_orig
    if "skill" in rec:
        rec["skill"] = skill
    else:
        rec = skill
    return rec


def build_donor_for(target: str, src_skills: Dict[str, List[dict]]) -> List[dict]:
    """Donor = union of all OTHER games' skills (leave-one-out)."""
    out: List[dict] = []
    for g in ALL_GAMES:
        if g == target:
            continue
        for rec in src_skills[g]:
            out.append(retag(dict(rec), g))  # shallow copy + retag
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True,
                    help="Output dir for donor_<TARGET>.jsonl files.")
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    # 1. Load every source bank (without retagging).
    src_raw: Dict[str, List[dict]] = {}
    for g, rel in SOURCE_BANKS.items():
        p = (args.project_root / rel).resolve()
        if not p.exists():
            raise FileNotFoundError(f"missing source bank: {p}")
        rows = load_bank(p)
        # Pre-tag (we will copy + retag again inside build_donor_for, but the
        # original retag here ensures we don't have to deepcopy)
        src_raw[g] = rows
        print(f"  loaded {g:>18s}: {len(rows):3d} skills  ({p.relative_to(args.project_root)})")

    # 2. Build a donor bank for every target.
    manifest = {
        "experiment": "R9_cross_game_transfer",
        "design": "leave_one_out_union",
        "source_banks": {g: str((args.project_root / rel).resolve()) for g, rel in SOURCE_BANKS.items()},
        "source_counts": {g: len(rows) for g, rows in src_raw.items()},
        "donor_banks": {},
    }
    for target in ALL_GAMES:
        donor = build_donor_for(target, src_raw)
        out_path = args.out_root / f"donor_{target}.jsonl"
        with out_path.open("w") as f:
            for rec in donor:
                f.write(json.dumps(rec) + "\n")
        # tally per-origin
        per_origin: Dict[str, int] = {}
        for rec in donor:
            ori = (rec.get("skill", rec) if "skill" in rec else rec).get("origin_game", "?")
            per_origin[ori] = per_origin.get(ori, 0) + 1
        manifest["donor_banks"][target] = {
            "path": str(out_path),
            "total_skills": len(donor),
            "per_origin": per_origin,
        }
        print(f"  ✓ donor for target={target:>18s}: {len(donor):3d} skills "
              f"from {len([g for g in ALL_GAMES if g != target])} other games "
              f"→ {out_path.name}")

    manifest_path = args.out_root / "donor_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  manifest → {manifest_path}")


if __name__ == "__main__":
    main()
