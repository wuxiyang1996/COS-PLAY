"""Aggregate WebShop 50-task runs into a paper-ready 4-way comparison report.

Walks every ``Cold-start-out-browsergym/webshop_50task_<tag>/`` directory
matching the canonical naming, reads each ``rollout_summary.json`` /
``episode_000.json`` pair, classifies every step into the inner-MDP
hop family (``G/E/C/R``), and emits a Markdown report at
``Cold-start-out-browsergym/REPORT_4way_comparison.md`` with:

  * mean reward + 95% CI (t-distribution and bootstrap)
  * Wilson-score 95% CI for three success-rate definitions
    (strict r=1.0, pass r>=0.5, any r>0)
  * pairwise CI-overlap significance test
  * per-task head-to-head table with win share
  * protocol-family hop breakdown
  * vs published WebShop baselines

This is the script that produced the canonical
``REPORT_4way_comparison.md`` cited in ``webshop_wrapper/README.md``.
Re-run after a new model run lands to refresh the report with the
new column.

Usage::

    python -m webshop_wrapper._make_report
    # or, with non-default cases:
    python -m webshop_wrapper._make_report --case gpt-5.4-low webshop_50task_low \
        --case qwen3-vl-235b webshop_50task_qwen
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from typing import Iterable

# ----------------------------------------------------------------------
# CI / stats helpers
# ----------------------------------------------------------------------

# t critical value for 95% two-sided CI, df=49.
_T_CRIT_DF49 = 2.0096

def t_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float, float, float, float]:
    n = len(values)
    mean = sum(values) / n
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    half = _T_CRIT_DF49 * se
    return mean, mean - half, mean + half, sd, se


def bootstrap_ci(values: list[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2))]


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.96  # 95%
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


# ----------------------------------------------------------------------
# Hop classification (mirrors the live aggregator that produced
# ``REPORT_4way_comparison.md``).
# ----------------------------------------------------------------------

def _url_phase(url: str) -> str:
    if "search_results" in url:
        return "results"
    if "/item_page/" in url:
        return "item"
    if "/item_sub_page/" in url:
        return "item_subpage"
    if "/done/" in url:
        return "done"
    return "landing"


def _classify_hop(exp: dict) -> str:
    state = exp.get("state", "") or ""
    nxt = exp.get("next_state", "") or ""
    action = exp.get("action", "") or ""
    cur = _url_phase(re.search(r"url=(\S+)", state).group(1)) if "url=" in state else "?"
    new = _url_phase(re.search(r"url=(\S+)", nxt).group(1)) if "url=" in nxt else "?"
    if action.startswith("fill"):
        return "G"
    if action.startswith(("noop", "scroll", "go_back", "go_forward")):
        return "C"
    if action.startswith("click"):
        if cur == "landing" and new == "results":
            return "G"
        if cur == "results" and new == "item":
            return "G"
        if cur == "item" and new == "item_subpage":
            return "R"
        if cur == "item_subpage" and new == "item":
            return "C"
        if cur == "item" and new == "done":
            return "E"
        if cur == "item":
            return "E"
        if cur == "results" and new == "results":
            return "C"
        return "E"
    return "C"


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

# (label, full model slug to print, sub-directory under Cold-start-out-browsergym/)
_DEFAULT_CASES: list[tuple[str, str, str]] = [
    ("gpt-5.4-low",       "openai/gpt-5.4 (reasoning_effort=low)",  "webshop_50task_low"),
    ("qwen3-vl-235b",     "qwen/qwen3-vl-235b-a22b-instruct",       "webshop_50task_qwen"),
    ("gemini-3.1-pro",    "google/gemini-3.1-pro-preview",          "webshop_50task_gemini"),
    ("claude-sonnet-4.5", "anthropic/claude-sonnet-4.5",            "webshop_50task_claude"),
]

# Vs published baselines (Yao et al. 2022 + ReAct paper).
_BASELINES: list[tuple[str, float]] = [
    ("Human expert (Mech-Turk, no time-limit)", 0.604),
    ("ReAct + GPT-4 (with prompting)",          0.455),
    ("ReAct + GPT-3.5",                         0.430),
    ("ReAct + GPT-3",                           0.402),
    ("IL+RL (BERT-base)",                       0.300),
    ("IL    (BERT-base)",                       0.299),
    ("WebGPT (GPT-3 ReAct)",                    0.252),
    ("Rule-based bot",                          0.097),
]


def _load_case(root_dir: str, sub: str) -> dict:
    """Return aggregate stats + per-task rows + hop counter for one run."""
    root = os.path.join(root_dir, sub)
    if not os.path.isdir(root):
        return {"missing": True, "rows": [], "hopc": {}, "tr": 0, "ts": 0, "te": 0}
    dirs = sorted(
        (d for d in os.listdir(root) if d.startswith("webshop.")),
        key=lambda s: int(s.split(".")[-1]),
    )
    rows = []
    hopc = {"G": 0, "E": 0, "C": 0, "R": 0}
    tr = ts = te = 0.0
    for d in dirs:
        sp = os.path.join(root, d, "rollout_summary.json")
        ep = os.path.join(root, d, "episode_000.json")
        if not (os.path.exists(sp) and os.path.exists(ep)):
            continue
        s = json.load(open(sp))
        e = s["episode_stats"][0]
        epi = json.load(open(ep))
        for h in (_classify_hop(x) for x in epi.get("experiences", [])):
            hopc[h] = hopc.get(h, 0) + 1
        tr += e["total_reward"]
        ts += e["steps"]
        te += e["elapsed_seconds"]
        rows.append((d, e["steps"], e["total_reward"], e["terminated"], e["elapsed_seconds"]))
    return {"missing": False, "rows": rows, "hopc": hopc, "tr": tr, "ts": ts, "te": te}


def _emit_report(cases: list[tuple[str, str, str]], root_dir: str, out_path: str) -> None:
    results = {
        tag: {"model": model, **_load_case(root_dir, sub)}
        for tag, model, sub in cases
    }

    out: list[str] = []
    p = out.append
    p("# WebShop frontier-model comparison\n")
    p(f"**Bench:** WebShop 50 tasks (`browsergym/webshop.0`–`webshop.49`), real Flask server, lite mode  |  **Stack:** BrowserGym + WebShop bridge  |  **Max steps:** 20\n")

    # ------------- TL;DR ---------------------------------------------
    p("## TL;DR — mean reward (95% CI) and success rate\n")
    p("Mean reward CI: standard t-distribution interval (df = 49, t₀.₉₇₅ = 2.0096) with a bootstrap cross-check (n_boot = 10 000). Success-rate CIs: Wilson-score (better than Wald near 0/1 with n = 50).\n")
    p("| Model | Mean reward (95% t-CI) | 95% bootstrap CI | σ / SE | SR strict (r=1.0) | SR pass (r≥0.5) | SR any (r>0) |")
    p("|---|---|---|---|---|---|---|")
    for tag, _, _ in cases:
        R = results[tag]
        if R["missing"] or not R["rows"]:
            continue
        rs = [r for _, _, r, _, _ in R["rows"]]
        n = len(rs)
        mean, lo_t, hi_t, sd, se = t_ci(rs)
        lo_b, hi_b = bootstrap_ci(rs)
        n_perfect = sum(1 for r in rs if r == 1.0)
        n_pass    = sum(1 for r in rs if r >= 0.5)
        n_any     = sum(1 for r in rs if r > 0)
        p_p, p_p_lo, p_p_hi = wilson_ci(n_perfect, n)
        p_h, p_h_lo, p_h_hi = wilson_ci(n_pass, n)
        p_a, p_a_lo, p_a_hi = wilson_ci(n_any, n)
        p(f"| `{R['model']}` | **{mean:.3f}** [{lo_t:.3f}, {hi_t:.3f}] | [{lo_b:.3f}, {hi_b:.3f}] | "
          f"{sd:.3f} / {se:.3f} | "
          f"{n_perfect}/{n} = **{p_p*100:.0f}%** [{p_p_lo*100:.0f}, {p_p_hi*100:.0f}] | "
          f"{n_pass}/{n} = **{p_h*100:.0f}%** [{p_h_lo*100:.0f}, {p_h_hi*100:.0f}] | "
          f"{n_any}/{n} = **{p_a*100:.0f}%** [{p_a_lo*100:.0f}, {p_a_hi*100:.0f}] |")
    p("")

    # ------------- pairwise significance -----------------------------
    p("## Pairwise significance — 95% CI overlap on mean reward\n")
    intervals = {}
    for tag, _, _ in cases:
        R = results[tag]
        if R["missing"] or not R["rows"]:
            continue
        rs = [r for _, _, r, _, _ in R["rows"]]
        intervals[tag] = (t_ci(rs)[1], t_ci(rs)[2], sum(rs) / len(rs))
    p("| pair | Δ mean | CIs overlap? | verdict |")
    p("|---|---:|:-:|---|")
    keys = list(intervals.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            la, ha, ma = intervals[a]
            lb, hb, mb = intervals[b]
            overlap = not (ha < lb or hb < la)
            diff = ma - mb
            verdict = "tied at 95%" if overlap else (
                f"**{a if ma > mb else b} significantly higher**"
            )
            p(f"| {a} vs {b} | {diff:+.3f} | {'YES' if overlap else 'NO'} | {verdict} |")
    p("")

    # ------------- protocol-family hops ------------------------------
    p("## Protocol-family hop classification\n")
    p("Hop family: **G** (GROUND = search/click product), **E** (EXECUTE = filter/buy), **C** (CHECK = noop/scroll/anti-thrash), **R** (RETRIEVE = open Description/Features/Reviews tab).\n")
    p("| model | total hops | GROUND | EXECUTE | CHECK | RETRIEVE |")
    p("|---|---:|---:|---:|---:|---:|")
    for tag, _, _ in cases:
        R = results[tag]
        if R["missing"]:
            continue
        total = sum(R["hopc"].values()) or 1
        g, e, c, r = R["hopc"].get("G", 0), R["hopc"].get("E", 0), R["hopc"].get("C", 0), R["hopc"].get("R", 0)
        p(f"| **{tag}** | {total} | {g} ({100*g/total:.1f}%) | {e} ({100*e/total:.1f}%) | {c} ({100*c/total:.1f}%) | {r} ({100*r/total:.1f}%) |")
    p("")

    # ------------- per-task table ------------------------------------
    p("## Head-to-head per task (winner = highest reward)\n")
    rows_per_tag = {tag: {d: r for d, _, r, _, _ in results[tag]["rows"]} for tag, _, _ in cases if not results[tag]["missing"]}
    if rows_per_tag:
        all_tasks = sorted(set().union(*[set(d.keys()) for d in rows_per_tag.values()]),
                           key=lambda s: int(s.split(".")[-1]))
        wins: dict[str, float] = {tag: 0.0 for tag in rows_per_tag}
        tie_zero = tie_perfect = 0
        header = "| task | " + " | ".join(rows_per_tag.keys()) + " | winner |"
        p(header)
        p("|---" * (len(rows_per_tag) + 2) + "|")
        for d in all_tasks:
            vals = [(tag, rows_per_tag[tag].get(d, 0.0)) for tag in rows_per_tag]
            mx = max(v for _, v in vals)
            winners = [tag for tag, v in vals if v == mx]
            if len(winners) == len(rows_per_tag) and mx == 0:
                w = "**all = 0**"; tie_zero += 1
            elif len(winners) == len(rows_per_tag) and mx >= 1.0:
                w = "**all = 1.0**"; tie_perfect += 1
            else:
                for ww in winners:
                    wins[ww] += 1.0 / len(winners)
                w = "+".join(winners)
            p("| " + d + " | " + " | ".join(f"{v:+.2f}" for _, v in vals) + f" | {w} |")
        p(f"\n**Win share (excl. {tie_zero + tie_perfect} fully-tied tasks):** "
          + ", ".join(f"{tag} = {ws:.1f}" for tag, ws in wins.items())
          + f"  |  All-failed (r=0) ties: {tie_zero}  |  All-perfect (r=1) ties: {tie_perfect}\n")

    # ------------- vs published baselines ----------------------------
    p("## Vs published WebShop baselines\n")
    ours = [(tag, sum(r for _, _, r, _, _ in results[tag]["rows"]) / max(1, len(results[tag]["rows"])))
            for tag, _, _ in cases if not results[tag]["missing"] and results[tag]["rows"]]
    combined = [(name, v, False) for name, v in _BASELINES] + [(tag, v, True) for tag, v in ours]
    combined.sort(key=lambda x: -x[1])
    p("| baseline | mean reward |")
    p("|---|---:|")
    for name, v, ours_flag in combined:
        marker = "  ◀ this run" if ours_flag else ""
        p(f"| {name}{marker} | {v:.3f} |")
    p("")

    with open(out_path, "w") as fh:
        fh.write("\n".join(out))
    print(f"[ok] wrote {out_path}  ({sum(len(l) for l in out)} chars, {len(out)} lines)")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--root",
        default=os.environ.get(
            "WEBSHOP_REPORT_ROOT",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "Cold-start-out-browsergym"),
        ),
        help="Directory containing webshop_50task_<tag>/ subdirs.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output Markdown path (default: <root>/REPORT_4way_comparison.md).",
    )
    ap.add_argument(
        "--case",
        nargs=3,
        metavar=("TAG", "MODEL_SLUG", "SUBDIR"),
        action="append",
        help="Override the default case list.  Pass once per case.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    cases = [tuple(c) for c in args.case] if args.case else _DEFAULT_CASES
    out_path = args.out or os.path.join(args.root, "REPORT_4way_comparison.md")
    _emit_report(cases, args.root, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
