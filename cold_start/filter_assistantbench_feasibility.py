"""Pre-screen AssistantBench tasks for "easy-to-fail" patterns.

Why
---
AssistantBench tasks include real-world web research questions. Some
categories are systematically out of reach for a generic public-web agent
(no login, no purchases, no time machine):

  REQUIRES_LOGIN     — needs a personal account / authenticated session
  REAL_TIME          — answer changes daily/weekly (today's price, weather…)
  TRANSACTIONAL      — needs to buy / book / submit a form
  OPEN_ENDED         — judgemental opinion / very subjective

Rather than waste 5+ minutes of agent + LLM time on each predetermined-fail
task, this script asks a cheap classifier (gpt-4o-mini) to bucket the
tasks first, then writes a filtered task-id list for the main eval driver.

Output
------
  cold_start/task_samples/assistantbench_feasibility.json   — per-task labels
  cold_start/task_samples/browsergym_assistantbench_feasible.txt — filtered list

Usage
-----
  python cold_start/filter_assistantbench_feasibility.py \
      --split test \
      --classifier_model gpt-4o-mini \
      --out cold_start/task_samples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
if str(CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEBASE_ROOT))

from generate_cold_start_actor_browsergym import _bootstrap_api_keys_from_file  # type: ignore  # noqa: E402

_bootstrap_api_keys_from_file()

import openai  # noqa: E402

try:
    from API_func import make_openai_client, effective_openai_model
except Exception:  # pragma: no cover
    make_openai_client = None
    effective_openai_model = None


_PROMPT = """\
You are screening web-research questions for a generic AI agent that can:
  - browse public websites (search + click + read)
  - call a server-side search API (search_web)
  - submit short text answers

The agent CANNOT:
  - log in to personal accounts (Gmail, Spotify, Netflix, banking, etc.)
  - perform purchases / bookings / form submissions
  - access content behind paywalls or local-only services
  - resolve answers that change daily (today's stock price, this week's box office)

Classify the task into ONE of:
  FEASIBLE        — answerable from public web with effort
  REQUIRES_LOGIN  — needs a personal account / authenticated session
  REAL_TIME       — answer is current-state / time-sensitive (today, now, this week)
  TRANSACTIONAL   — requires buying, booking, signing up, or completing a transaction
  OPEN_ENDED      — purely subjective / opinion / no single defensible answer

Output STRICT JSON only, no prose:
  {{"category": "...", "reason": "<one short sentence>"}}

Task: {task}
"""


def classify_one(client, model: str, task_text: str) -> dict:
    prompt = _PROMPT.format(task=task_text.strip())
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=100,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        cat = (data.get("category") or "").strip().upper()
        reason = (data.get("reason") or "").strip()
        if cat not in {"FEASIBLE", "REQUIRES_LOGIN", "REAL_TIME", "TRANSACTIONAL", "OPEN_ENDED"}:
            return {"category": "UNKNOWN", "reason": f"unparsed: {raw[:120]}"}
        return {"category": cat, "reason": reason}
    except Exception as exc:  # noqa: BLE001
        return {"category": "UNKNOWN", "reason": f"error: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument(
        "--classifier_model",
        default="gpt-4o-mini",
        help="Cheap LLM for the screen (default: gpt-4o-mini).",
    )
    ap.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "task_samples"),
        help="Directory for output JSON + filtered task list.",
    )
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument(
        "--keep",
        nargs="+",
        default=["FEASIBLE"],
        help="Categories to KEEP in the filtered list (default: FEASIBLE).",
    )
    args = ap.parse_args()

    try:
        import datasets
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        return 2

    print(f"Loading AssistantBench {args.split} split…")
    ds = datasets.load_dataset("AssistantBench/AssistantBench", split=args.split)
    print(f"  loaded {len(ds)} tasks")

    if make_openai_client is None:
        print("ERROR: API_func.make_openai_client unavailable", file=sys.stderr)
        return 2
    client = make_openai_client(prefer="auto")
    if client is None:
        print("ERROR: no API credentials found (set OPENAI_API_KEY or OPENROUTER_API_KEY)", file=sys.stderr)
        return 2
    model = effective_openai_model(args.classifier_model, prefer="auto") if effective_openai_model else args.classifier_model
    print(f"  classifier model: {model}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"assistantbench_feasibility_{args.split}.json"
    list_path = out_dir / f"browsergym_assistantbench_{args.split}_feasible.txt"

    results: dict[int, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {
            pool.submit(classify_one, client, model, ds[i]["task"] or ""): i
            for i in range(len(ds))
        }
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            res = fut.result()
            row = ds[i]
            results[i] = {
                "index": i,
                "task_id": f"browsergym/assistantbench.{args.split}.{i}",
                "set": row.get("set"),
                "task": (row.get("task") or "")[:300],
                "category": res["category"],
                "reason": res["reason"],
            }
            done += 1
            if done % 20 == 0 or done == len(ds):
                elapsed = time.time() - t0
                print(f"  classified {done}/{len(ds)} ({elapsed:.0f}s)")

    ordered = [results[i] for i in range(len(ds))]
    json_path.write_text(json.dumps(ordered, indent=2))
    print(f"\nWrote per-task JSON: {json_path}")

    from collections import Counter
    counts = Counter(r["category"] for r in ordered)
    set_counts: dict = {}
    for r in ordered:
        set_counts.setdefault(r["set"] or "?", Counter())[r["category"]] += 1

    print("\n=== Category breakdown ===")
    for cat, n in counts.most_common():
        pct = 100.0 * n / len(ordered)
        print(f"  {cat:18s}  {n:4d}  ({pct:.1f}%)")
    print("\n=== Per-set breakdown ===")
    for set_name, c in set_counts.items():
        total = sum(c.values())
        kept_n = sum(c[k] for k in args.keep if k in c)
        print(f"  {set_name:14s}  total={total:4d}  feasible_kept={kept_n}")

    keep_set = {k.upper() for k in args.keep}
    kept = [r for r in ordered if r["category"] in keep_set]
    list_path.write_text(
        "# AssistantBench feasibility-filtered task list\n"
        f"# split={args.split}  classifier={model}  kept={','.join(sorted(keep_set))}\n"
        f"# total={len(ordered)}  kept={len(kept)}  dropped={len(ordered) - len(kept)}\n"
        + "\n".join(r["task_id"] for r in kept)
        + "\n"
    )
    print(f"\nWrote filtered task list: {list_path} ({len(kept)} tasks)")

    print("\n=== Sample DROPPED tasks (each non-FEASIBLE bucket, up to 3) ===")
    dropped_by_cat: dict = {}
    for r in ordered:
        if r["category"] in keep_set:
            continue
        dropped_by_cat.setdefault(r["category"], []).append(r)
    for cat in ("REQUIRES_LOGIN", "REAL_TIME", "TRANSACTIONAL", "OPEN_ENDED", "UNKNOWN"):
        bucket = dropped_by_cat.get(cat, [])
        if not bucket:
            continue
        print(f"\n  [{cat}]")
        for r in bucket[:3]:
            print(f"    test.{r['index']:3d}: {r['task'][:140]}")
            print(f"           reason: {r['reason'][:160]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
