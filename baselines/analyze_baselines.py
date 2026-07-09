#!/usr/bin/env python3
"""Compute per-character Avalon win rates and per-power Diplomacy supply centers with 95% CIs."""

import json
import math
import os
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent / "output"

# ── Run registry: (model_label, game, run_dir, source) ──────────────────────
# source = "summary" means rollout_summary.json has episode_stats
# source = "jsonl" means parse rollouts.jsonl -> rollout_metadata
AVALON_RUNS = [
    ("GPT-5.4",        "gpt54_avalon_single_20260323_185147", "summary"),
    ("GPT-OSS-120B",   "gptoss120b_avalon_20260326_113706",   "summary"),
    ("Gemini-3.1-Pro", "gemini31pro_avalon_20260326_093347",   "summary"),
    ("Claude-4.6",     "claude46_avalon_20260326_074558",      "summary"),
]

DIPLOMACY_RUNS = [
    ("GPT-5.4",        "gpt54_diplomacy_20260323_182844",        "summary", True),   # self-play
    ("Gemini-3.1-Pro", "gemini31pro_diplomacy_20260326_114604",   "jsonl",   False),
    ("Claude-4.6",     "claude46_diplomacy_20260326_074611",      "summary", False),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def wilson_ci(wins, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = wins / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0, centre - margin), min(1, centre + margin)


def t_ci(values, confidence=0.95):
    """Mean and 95% CI using t-distribution (scipy-free approx for small n)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(var / n)
    # t critical values for 95% CI (two-tailed) by df; fallback to z=1.96 for large n
    t_crit_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021,
        50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984,
    }
    df = n - 1
    t_crit = 1.96
    for k in sorted(t_crit_table.keys()):
        if df <= k:
            t_crit = t_crit_table[k]
            break
    lo = mean - t_crit * se
    hi = mean + t_crit * se
    return mean, lo, hi


def load_episode_stats(run_dir, game, source):
    """Return list of episode_stat dicts from the best available source."""
    game_dir = BASE / run_dir / game
    if source == "summary":
        summary_path = game_dir / "rollout_summary.json"
        with open(summary_path) as f:
            return json.load(f)["episode_stats"]
    else:
        jsonl_path = game_dir / "rollouts.jsonl"
        stats = []
        with open(jsonl_path) as f:
            for line in f:
                ep = json.loads(line.strip())
                rm = ep.get("rollout_metadata", {})
                if rm:
                    stats.append(rm)
        return stats


# ── Avalon: win rate per character ───────────────────────────────────────────

def analyze_avalon():
    print("=" * 90)
    print("AVALON — Win Rate per Character (controlled player's side wins)")
    print("  Win = good_victory for Good roles, NOT good_victory for Evil roles")
    print("=" * 90)

    ROLES_ORDER = ["Merlin", "Servant", "Assassin", "Minion"]
    ROLE_SIDE = {"Merlin": "good", "Servant": "good", "Assassin": "evil", "Minion": "evil"}

    for model_label, run_dir, source in AVALON_RUNS:
        stats = load_episode_stats(run_dir, "avalon", source)
        role_wins = defaultdict(int)
        role_total = defaultdict(int)

        for ep in stats:
            role = ep["role_name"]
            side = ep.get("role_side", ROLE_SIDE.get(role, "unknown"))
            good_victory = ep["good_victory"]
            won = good_victory if side == "good" else not good_victory
            role_total[role] += 1
            if won:
                role_wins[role] += 1

        print(f"\n{'─' * 90}")
        print(f"  Model: {model_label}   ({sum(role_total.values())} episodes)")
        print(f"{'─' * 90}")
        print(f"  {'Role':<12} {'Side':<6} {'N':>4}  {'Wins':>5}  {'Win Rate':>9}  {'95% CI':>20}")
        print(f"  {'─'*12} {'─'*6} {'─'*4}  {'─'*5}  {'─'*9}  {'─'*20}")

        all_wins = 0
        all_n = 0
        for role in ROLES_ORDER:
            if role not in role_total:
                continue
            n = role_total[role]
            w = role_wins[role]
            wr, lo, hi = wilson_ci(w, n)
            side = ROLE_SIDE[role]
            print(f"  {role:<12} {side:<6} {n:>4}  {w:>5}  {wr:>8.1%}  [{lo:>7.1%}, {hi:>7.1%}]")
            all_wins += w
            all_n += n

        wr, lo, hi = wilson_ci(all_wins, all_n)
        print(f"  {'OVERALL':<12} {'—':<6} {all_n:>4}  {all_wins:>5}  {wr:>8.1%}  [{lo:>7.1%}, {hi:>7.1%}]")


# ── Diplomacy: supply centers per power ──────────────────────────────────────

def analyze_diplomacy():
    print("\n\n" + "=" * 90)
    print("DIPLOMACY — Supply Centers per Power (final_sc_rewards × 18)")
    print("=" * 90)

    POWERS_ORDER = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

    for model_label, run_dir, source, self_play in DIPLOMACY_RUNS:
        stats = load_episode_stats(run_dir, "diplomacy", source)

        # For mixed_model runs: each episode has one controlled_power
        # For self-play: each episode contributes data for ALL 7 powers
        power_sc = defaultdict(list)

        skipped = 0
        if self_play:
            for ep in stats:
                fsr = ep["final_sc_rewards"]
                if any(v < 0 for v in fsr.values()):
                    skipped += 1
                    continue
                for power in POWERS_ORDER:
                    sc = fsr[power] * 18
                    power_sc[power].append(sc)
        else:
            for ep in stats:
                if ep.get("controlled_power_reward", 0) < 0:
                    skipped += 1
                    continue
                cp = ep["controlled_power"]
                sc = ep["controlled_power_reward"] * 18
                power_sc[cp].append(sc)

        total_episodes = len(stats)
        used = total_episodes - skipped
        print(f"\n{'─' * 90}")
        skip_note = f", {skipped} error episodes removed" if skipped else ""
        print(f"  Model: {model_label}   ({used} episodes"
              f"{', self-play' if self_play else ', vs GPT-5.4'}{skip_note})")
        print(f"{'─' * 90}")
        print(f"  {'Power':<10} {'N':>4}  {'Mean SC':>8}  {'95% CI':>20}")
        print(f"  {'─'*10} {'─'*4}  {'─'*8}  {'─'*20}")

        all_sc = []
        for power in POWERS_ORDER:
            vals = power_sc[power]
            if not vals:
                print(f"  {power:<10} {0:>4}  {'N/A':>8}  {'N/A':>20}")
                continue
            n = len(vals)
            mean, lo, hi = t_ci(vals)
            print(f"  {power:<10} {n:>4}  {mean:>8.2f}  [{lo:>7.2f}, {hi:>7.2f}]")
            all_sc.extend(vals)

        if all_sc:
            mean, lo, hi = t_ci(all_sc)
            n = len(all_sc)
            print(f"  {'OVERALL':<10} {n:>4}  {mean:>8.2f}  [{lo:>7.2f}, {hi:>7.2f}]")


if __name__ == "__main__":
    analyze_avalon()
    analyze_diplomacy()
