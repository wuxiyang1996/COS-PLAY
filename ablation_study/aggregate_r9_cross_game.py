#!/usr/bin/env python3
"""Aggregate R9 cross-game transfer results into a single table.

For each target game cell, this reads:
  - cell_meta.json (target_game, donor_n_skills, eval_module, episodes, etc.)
  - eval_out/{eval_summary*.json | summary.json}  — produced by the per-game evaluator

Outputs:
  summary.json    machine-readable
  summary.md      markdown table for the rebuttal
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Paper Table 1 self-bank reference numbers (best-of-training mean reward).
# Used purely as a reference column in the markdown table; we annotate the
# source so reviewers can verify.
SELF_BANK_REFERENCE: Dict[str, Dict[str, Any]] = {
    "twenty_forty_eight": {"mean_reward": None, "src": "paper Table 1 (self-bank)"},
    "candy_crush":        {"mean_reward": None, "src": "paper Table 1 (self-bank)"},
    "tetris":             {"mean_reward": None, "src": "paper Table 1 (self-bank)"},
    "super_mario":        {"mean_reward": None, "src": "paper Table 1 (self-bank)"},
    "avalon":             {"mean_reward": None, "src": "paper Table 1 (vs gpt-5.4)"},
    "diplomacy":          {"mean_reward": None, "src": "paper Table 1 (vs gpt-5.4)"},
}


def _safe_jl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _safe_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _find_eval_summary(eval_dir: Path) -> Optional[Dict[str, Any]]:
    """Probe common naming conventions used by the 3 evaluators.

    Returns the FIRST candidate that actually contains usable reward data
    (i.e. one of: 'results' with episode_stats, 'per_episode', 'episode_stats').
    Falls back to the first non-empty file otherwise.
    """
    if not eval_dir.is_dir():
        return None
    # Order: most-likely-to-have-data first.  rollout_summary (diplomacy)
    # is in the nested game/<ts>/ subdir.
    candidates = (
        list(eval_dir.glob("**/rollout_summary.json"))
        + list(eval_dir.glob("**/eval_summary*.json"))
        + list(eval_dir.glob("**/summary*.json"))
    )
    seen = set()
    primary = None  # first non-empty file (fallback)
    for p in candidates:
        if str(p) in seen:
            continue
        seen.add(str(p))
        d = _safe_json(p)
        if not d:
            continue
        if primary is None:
            primary = (p, d)
        # Heuristic: does this file have actual per-episode rewards?
        has_rewards = (
            (isinstance(d.get("results"), list)
             and any(isinstance(r, dict) and r.get("episode_stats") for r in d["results"]))
            or isinstance(d.get("per_episode"), list)
            or isinstance(d.get("episode_stats"), list)
            or isinstance(d.get("avg_reward"), (int, float))
        )
        if has_rewards:
            d["_source_file"] = str(p)
            return d
    if primary is not None:
        p, d = primary
        d["_source_file"] = str(p)
        return d
    return None


def _extract_rewards(es: Dict[str, Any], target: str) -> Optional[List[float]]:
    """Extract the per-episode reward list (used to compute mean/std/min/max).

    Tries (in order):
      1. es['results'][i]['episode_stats'][j]['total_reward']  (run_qwen3_8b_eval shape)
      2. es['games'][target]['episode_stats'][...]            (alt shape)
      3. es['per_role'] / es['per_power'] values' 'rewards'   (multi-player shape)
      4. es['rewards']                                          (flat)
    Returns None if nothing found.
    """
    if not es:
        return None
    # Shape 1: results[].episode_stats[].total_reward
    # NOTE: skip episodes with a non-empty 'error' field — those represent
    # eval-infrastructure failures (e.g. missing gym module) rather than
    # real game-play outcomes.
    results = es.get("results")
    if isinstance(results, list):
        rewards = []
        for r in results:
            if not isinstance(r, dict):
                continue
            if r.get("game") and r["game"] != target:
                continue
            stats = r.get("episode_stats") or []
            for s in stats:
                if not isinstance(s, dict):
                    continue
                if s.get("error"):
                    continue
                if isinstance(s.get("total_reward"), (int, float)):
                    rewards.append(float(s["total_reward"]))
        if rewards:
            return rewards
    # Shape 1b: top-level episode_stats[] (diplomacy rollout_summary.json shape)
    if isinstance(es.get("episode_stats"), list):
        rewards = [s.get("total_reward") for s in es["episode_stats"]
                   if isinstance(s, dict) and isinstance(s.get("total_reward"), (int, float))]
        if rewards:
            return rewards
    # Shape 1c: top-level per_episode[] (avalon eval_summary.json shape)
    if isinstance(es.get("per_episode"), list):
        rewards = [s.get("total_reward") for s in es["per_episode"]
                   if isinstance(s, dict) and isinstance(s.get("total_reward"), (int, float))]
        if rewards:
            return rewards
    # Shape 2: games[target]
    games = es.get("games") or {}
    if isinstance(games, dict) and target in games:
        gs = games[target]
        if isinstance(gs, dict):
            stats = gs.get("episode_stats") or []
            rewards = [s["total_reward"] for s in stats
                       if isinstance(s, dict) and isinstance(s.get("total_reward"), (int, float))]
            if rewards:
                return rewards
    # Shape 3: per_role / per_power
    for key in ("per_role", "per_power"):
        per = es.get(key)
        if isinstance(per, dict):
            rewards = []
            for v in per.values():
                if isinstance(v, dict):
                    rs = v.get("rewards") or v.get("episode_rewards") or []
                    rewards.extend([x for x in rs if isinstance(x, (int, float))])
            if rewards:
                return [float(x) for x in rewards]
    # Shape 4: flat
    if isinstance(es.get("rewards"), list):
        return [float(x) for x in es["rewards"] if isinstance(x, (int, float))]
    if isinstance(es.get("episode_rewards"), list):
        return [float(x) for x in es["episode_rewards"] if isinstance(x, (int, float))]
    return None


def _extract_reward(es: Dict[str, Any], target: str) -> Optional[float]:
    rs = _extract_rewards(es, target)
    if not rs:
        return None
    return sum(rs) / len(rs)


def collect_cell(cell_dir: Path) -> Optional[Dict[str, Any]]:
    import statistics
    meta = _safe_json(cell_dir / "cell_meta.json")
    if not meta:
        return None
    eval_dir = cell_dir / "eval_out"
    es = _find_eval_summary(eval_dir) or {}
    target = meta.get("target_game", "")
    rewards = _extract_rewards(es, target) or []
    mean_r = (sum(rewards) / len(rewards)) if rewards else None
    std_r = statistics.stdev(rewards) if len(rewards) >= 2 else None
    min_r = min(rewards) if rewards else None
    max_r = max(rewards) if rewards else None
    return dict(
        target=target,
        donor_n_skills=meta.get("donor_n_skills"),
        is_multiplayer=meta.get("is_multiplayer"),
        opponent_model=meta.get("opponent_model"),
        episodes=meta.get("episodes"),
        n_episodes_collected=len(rewards),
        eval_module=meta.get("eval_module"),
        eval_wall_s=meta.get("eval_wall_s"),
        exit_code=meta.get("exit_code", -1),
        eval_summary_file=es.get("_source_file"),
        mean_reward=mean_r,
        std_reward=std_r,
        min_reward=min_r,
        max_reward=max_r,
        rewards=rewards,
        raw_summary_keys=sorted(list(es.keys())) if es else [],
    )


def build_markdown(rows: List[Dict[str, Any]], manifest: Dict[str, Any]) -> str:
    out = []
    out.append("# R9 — Cross-Game Skill Transfer")
    out.append("")
    out.append("Each target game is evaluated using a **donor skill bank** = union")
    out.append("of the OTHER 5 games' best skill banks (leave-one-out).  Decision")
    out.append("LoRAs and base model are unchanged — only the bank is swapped.")
    out.append("")
    out.append("- Single-player targets (twenty_forty_eight / candy_crush / tetris /")
    out.append("  super_mario): 16 episodes per game, COSPLAY agent plays solo.")
    out.append("- Multi-player targets (avalon / diplomacy): 10 episodes,")
    out.append("  COSPLAY agent vs GPT-5.4 opponent for all other seats.")
    out.append("")
    if not rows:
        out.append("_(no cells aggregated yet)_")
        return "\n".join(out)

    out.append("| Target | Donor skills | Mean ± Std (min · max) | n_ep | Wall(s) | Exit |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        tgt = r.get("target")
        donor_n = r.get("donor_n_skills")
        mr = r.get("mean_reward")
        sd = r.get("std_reward")
        mn = r.get("min_reward")
        mx = r.get("max_reward")
        if isinstance(mr, (int, float)):
            mr_s = f"{mr:.2f}"
            if isinstance(sd, (int, float)):
                mr_s += f" ± {sd:.2f}"
            if isinstance(mn, (int, float)) and isinstance(mx, (int, float)):
                mr_s += f" ({mn:.1f} · {mx:.1f})"
        else:
            mr_s = "—"
        n_ep = r.get("n_episodes_collected", "?")
        wall = r.get("eval_wall_s")
        wall_s = f"{wall:.0f}" if isinstance(wall, (int, float)) else "—"
        rc = r.get("exit_code")
        rc_s = "✓" if rc == 0 else f"✗(rc={rc})"
        out.append(f"| {tgt} | {donor_n} | {mr_s} | {n_ep} | {wall_s} | {rc_s} |")

    out.append("")
    out.append("## Donor bank composition (per-target skill counts by origin)")
    out.append("")
    db = manifest.get("donor_banks", {})
    if db:
        origins = sorted({o for d in db.values() for o in (d.get("per_origin") or {})})
        out.append("| Target ↓ \\ Origin → | " + " | ".join(origins) + " | total |")
        out.append("|---|" + "|".join(["---"] * (len(origins) + 1)) + "|")
        for tgt in sorted(db.keys()):
            row = db[tgt].get("per_origin", {})
            cells = [str(row.get(o, 0)) for o in origins]
            out.append(f"| {tgt} | " + " | ".join(cells) + f" | {db[tgt].get('total_skills','?')} |")
        out.append("")

    out.append("## Interpretation guide")
    out.append("- Compare each row's donor-bank reward against the paper Table 1's")
    out.append("  self-bank reward (same game, same eval protocol) and the R1 no-bank")
    out.append("  baseline.  Three regimes are possible:")
    out.append("  - **donor ≈ self**: skills transfer cleanly across games")
    out.append("  - **donor between self and no-bank**: partial transfer; skills are")
    out.append("    related but not interchangeable")
    out.append("  - **donor ≈ no-bank**: effectively no transfer — skills are")
    out.append("    game-specific (this is *evidence the method learns grounded")
    out.append("    semantics*, not generic templates)")
    out.append("- We expect *asymmetric* transfer: single-player↔single-player likely")
    out.append("  shows measurable transfer (grid mechanics generic), but multi-")
    out.append("  player↔single-player should be near-zero because the social /")
    out.append("  territorial semantics are absent.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--donor-manifest", type=Path, default=None)
    ap.add_argument("--summary", type=Path, default=None)
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for sub in sorted(args.output_root.glob("*_donor")):
        if not sub.is_dir():
            continue
        r = collect_cell(sub)
        if r:
            rows.append(r)

    manifest = {}
    if args.donor_manifest and args.donor_manifest.exists():
        manifest = json.loads(args.donor_manifest.read_text())

    summary = dict(
        experiment="R9_cross_game_transfer",
        rows=rows,
        donor_manifest=manifest,
    )
    summary_path = args.summary or (args.output_root / "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[r9-aggregate] wrote {summary_path}")

    md_path = args.output_root / "summary.md"
    md_path.write_text(build_markdown(rows, manifest))
    print(f"[r9-aggregate] wrote {md_path}")


if __name__ == "__main__":
    main()
