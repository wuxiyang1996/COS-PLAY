"""Evaluator: pairwise IoU between canonical / text_llm / image_llm.

Run with::

    python visual_grounding_tests/_eval_agreement.py [--run <run_id>]

Produces a per-step table and per-game averages for:

  - canonical  vs  text_llm     (C-T)
  - canonical  vs  image_llm    (C-I)
  - text_llm   vs  image_llm    (T-I)

The first one tells us how well the LLM follows the canonical spec.
The third one tells us whether two LLM-derived schemas can be mixed
during inference when env state is unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output" / "envwrappers"
GAMES = ["twenty_forty_eight", "candy_crush", "tetris", "super_mario"]

_ENT = re.compile(r"^(e\S*?)\[(.+)\]\s*$", re.M)
_ACT = re.compile(r"^a\d+=(.+)$", re.M)


def _attrs(blob: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in re.split(r",\s*(?=[a-z_]+=)", blob):
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse(text: str | None) -> dict | None:
    if not text:
        return None
    ents = [{"id": m.group(1), **_attrs(m.group(2))} for m in _ENT.finditer(text)]
    actions = sorted([m.group(1).strip() for m in _ACT.finditer(text)])
    return {"entities": ents, "actions": actions}


def keys(s: dict | None) -> Counter:
    if not s:
        return Counter()
    return Counter(
        ((e.get("label", "").lower(), e.get("pos", "null")) for e in s["entities"])
    )


def jaccard(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 1.0
    K = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in K)
    union = sum(max(a[k], b[k]) for k in K)
    return inter / union if union else 0.0


def latest_run() -> str:
    runs: set[str] = set()
    for g in GAMES:
        gd = OUT / g
        if not gd.exists():
            continue
        for ep in gd.glob("*_ep*"):
            runs.add(ep.name)
    if not runs:
        sys.exit("no runs found")
    return sorted(runs)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = args.run or latest_run()
    print(f"\n=== Evaluating run: {run} ===\n")

    print(f"{'GAME':22}{'STEP':<5}{'C|n':<6}{'T|n':<6}{'I|n':<6}"
          f"{'C-T':<7}{'C-I':<7}{'T-I':<7}{'ACT C-T':<9}{'ACT T-I':<9}")
    print("-" * 90)
    agg: dict[str, list] = {}

    for g in GAMES:
        p = OUT / g / run / "steps.jsonl"
        if not p.exists():
            print(f"  {g:22}(missing) {p}")
            continue
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        bag: list = []
        for r in rows:
            c = parse(r.get("schema_canonical"))
            t = parse((r.get("schema_text_llm") or {}).get("schema"))
            i = parse((r.get("schema_image_llm") or {}).get("schema"))
            ck, tk, ik = keys(c), keys(t), keys(i)
            ct, ci, ti = jaccard(ck, tk), jaccard(ck, ik), jaccard(tk, ik)
            ca = set(c["actions"]) if c else set()
            ta = set(t["actions"]) if t else set()
            ia = set(i["actions"]) if i else set()
            sj = lambda a, b: (len(a & b) / len(a | b)) if (a or b) else 1.0
            act_ct, act_ti = sj(ca, ta), sj(ta, ia)
            print(f"{g:22}{r['step']:<5}{len(ck):<6}{len(tk):<6}{len(ik):<6}"
                  f"{ct:<7.2f}{ci:<7.2f}{ti:<7.2f}{act_ct:<9.2f}{act_ti:<9.2f}")
            bag.append((ct, ci, ti, act_ct, act_ti))
        agg[g] = bag

    print("\n" + "=" * 90)
    print(f"{'GAME':22}{'avg C-T':<10}{'avg C-I':<10}{'avg T-I':<10}"
          f"{'avg actC-T':<12}{'avg actT-I':<12}")
    print("-" * 90)
    for g, b in agg.items():
        if not b:
            continue
        n = len(b)
        avg = lambda i: sum(s[i] for s in b) / n
        print(f"{g:22}{avg(0):<10.2f}{avg(1):<10.2f}{avg(2):<10.2f}"
              f"{avg(3):<12.2f}{avg(4):<12.2f}")


if __name__ == "__main__":
    main()
