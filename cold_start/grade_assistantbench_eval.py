"""Grade an AssistantBench cold-start eval run + write the AB-server submission.

What this does
--------------
After ``run_coldstart_actor_browsergym_shard.sh`` finishes (or even mid-run),
this script walks ``<output_dir>/<safe_task_id>/episode_000.json`` files and:

  1. **Validation split** — locally aggregates the DROP F1 rewards that the
     BrowserGym AssistantBench env already populated (``experience.reward``
     on the terminal step). Produces a headline ``mean_reward`` directly
     comparable to the AB-paper validation table.

  2. **Test split** — extracts the agent's final ``send_msg_to_user(...)``
     answer (the official AB submission format expects a free-text string)
     and writes a JSONL keyed by the canonical AB ``id`` field, ready to
     upload to https://huggingface.co/spaces/AssistantBench/leaderboard.

Outputs (all under ``<output_dir>``)
------------------------------------
  ``grading_summary.json``                       per-task table (val + test)
  ``grading_summary.csv``                        same, easy-to-eyeball
  ``assistantbench_validation_score.json``       headline val numbers
  ``assistantbench_test_predictions.jsonl``      AB-server upload format
  ``assistantbench_test_predictions_human.json`` same, with task text for review

Safe to run mid-run — it just skips tasks that don't yet have an
``episode_000.json``.

Usage
-----
  python cold_start/grade_assistantbench_eval.py \
      --run_dir Cold-start-out-browsergym/ab_full_eval_v1
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
if str(CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEBASE_ROOT))


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
#
# The actor records the literal action string in ``experience.action`` —
# typically one of:
#
#   send_msg_to_user("Adrenalinpark Köln")
#   send_msg_to_user('multi-word answer with spaces')
#   report_infeasible("no public listing available")
#
# We accept BOTH terminal forms but only ``send_msg_to_user`` produces an
# answer for the AB submission. The regex tolerates either single or
# double quotes; ``ast.literal_eval`` then handles escaping correctly.

_SEND_MSG_RE = re.compile(
    r'send_msg_to_user\(\s*([\'"].*?[\'"]|.*?)\s*\)',
    re.DOTALL,
)
_REPORT_INFEASIBLE_RE = re.compile(
    r'report_infeasible\(\s*([\'"].*?[\'"]|.*?)\s*\)',
    re.DOTALL,
)


def _safe_unquote(payload: str) -> str:
    """Best-effort string-literal unquote for ``send_msg_to_user`` payloads."""
    payload = payload.strip()
    if not payload:
        return ""
    try:
        out = ast.literal_eval(payload)
        if isinstance(out, str):
            return out
        return str(out)
    except (ValueError, SyntaxError):
        if (payload.startswith('"') and payload.endswith('"')) or (
            payload.startswith("'") and payload.endswith("'")
        ):
            return payload[1:-1]
        return payload


def extract_answer_from_episode(ep: dict) -> tuple[str, str]:
    """Return ``(answer, terminal_kind)`` from an episode JSON.

    ``terminal_kind`` is one of:
        ``send_msg``       — agent submitted an answer (used for AB)
        ``infeasible``     — agent gave up
        ``truncated``      — hit max_steps without terminating
        ``no_answer``      — terminated but no usable terminal action
    """
    experiences = ep.get("experiences") or []
    if not experiences:
        return "", "no_answer"

    last_send_msg: str | None = None
    last_infeasible: bool = False
    for exp in reversed(experiences):
        act = exp.get("action") or ""
        if not isinstance(act, str):
            continue
        m = _SEND_MSG_RE.search(act)
        if m and last_send_msg is None:
            last_send_msg = _safe_unquote(m.group(1))
            break
        m2 = _REPORT_INFEASIBLE_RE.search(act)
        if m2 and not last_infeasible:
            last_infeasible = True

    if last_send_msg is not None:
        return last_send_msg, "send_msg"
    if last_infeasible:
        return "", "infeasible"
    last = experiences[-1]
    if last.get("done"):
        return "", "no_answer"
    return "", "truncated"


def _terminal_reward(ep: dict) -> float:
    experiences = ep.get("experiences") or []
    if not experiences:
        return 0.0
    last = experiences[-1]
    r = last.get("reward")
    try:
        return float(r) if r is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _final_step_count(ep: dict) -> int:
    return len(ep.get("experiences") or [])


def _episode_extras_search_meta(ep: dict) -> dict[str, int]:
    """Aggregate search_web telemetry across all step extras."""
    counts = Counter()
    for exp in ep.get("experiences") or []:
        meta = (exp.get("metadata") or {}).get("search_web_meta") if isinstance(exp.get("metadata"), dict) else None
        if isinstance(meta, dict):
            counts["search_web_calls"] += 1
            counts[f"search_backend_{meta.get('backend', 'unknown')}"] += 1
            if meta.get("fallback_to_goto"):
                counts["search_fallback_to_goto"] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Dataset metadata loader
# ---------------------------------------------------------------------------

def _load_ab_metadata() -> dict[tuple[str, int], dict[str, Any]]:
    """Map ``(split, index) -> {id, set, task, gold_answer}``.

    AB ``test`` answers are ``None`` (locally hidden); ``validation`` ships
    its gold answers. We use ``id`` as the canonical submission key.
    """
    try:
        import datasets  # type: ignore
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        return {}

    out: dict[tuple[str, int], dict[str, Any]] = {}
    for split in ("validation", "test"):
        try:
            ds = datasets.load_dataset("AssistantBench/AssistantBench", split=split)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: failed to load AB {split}: {exc}", file=sys.stderr)
            continue
        for i, row in enumerate(ds):
            out[(split, i)] = {
                "id": row.get("id"),
                "set": row.get("set"),
                "task": row.get("task"),
                "gold_answer": row.get("answer"),
            }
    return out


# ---------------------------------------------------------------------------
# Run-dir walker
# ---------------------------------------------------------------------------

_TASK_DIR_RE = re.compile(r"^assistantbench\.(validation|test)\.(\d+)$")


def collect_task_rows(run_dir: Path, ab_meta: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = _TASK_DIR_RE.match(sub.name)
        if not m:
            continue
        split, idx_s = m.group(1), int(m.group(2))
        ep_path = sub / "episode_000.json"
        if not ep_path.exists():
            continue
        try:
            ep = json.loads(ep_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: failed to parse {ep_path}: {exc}", file=sys.stderr)
            continue

        answer, terminal_kind = extract_answer_from_episode(ep)
        reward = _terminal_reward(ep)
        meta = ab_meta.get((split, idx_s), {})
        rows.append({
            "split": split,
            "index": idx_s,
            "task_dir": sub.name,
            "ab_id": meta.get("id"),
            "ab_set": meta.get("set"),
            "task": meta.get("task") or ep.get("task", ""),
            "gold_answer": meta.get("gold_answer"),
            "predicted_answer": answer,
            "terminal_kind": terminal_kind,
            "reward": reward,
            "outcome": ep.get("outcome"),
            "n_steps": _final_step_count(ep),
            "search_meta": _episode_extras_search_meta(ep),
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregation + writers
# ---------------------------------------------------------------------------

def _split_rows(rows, split: str):
    return [r for r in rows if r["split"] == split]


def aggregate_validation(rows) -> dict[str, Any]:
    val_rows = _split_rows(rows, "validation")
    n = len(val_rows)
    if n == 0:
        return {"n": 0, "mean_reward": 0.0}
    total_reward = sum(r["reward"] for r in val_rows)
    answered = sum(1 for r in val_rows if r["terminal_kind"] == "send_msg")
    truncated = sum(1 for r in val_rows if r["terminal_kind"] == "truncated")
    infeasible = sum(1 for r in val_rows if r["terminal_kind"] == "infeasible")
    perfect = sum(1 for r in val_rows if r["reward"] >= 0.999)
    nonzero = sum(1 for r in val_rows if r["reward"] > 0.001)
    mean_steps = sum(r["n_steps"] for r in val_rows) / n
    search_total = sum(r["search_meta"].get("search_web_calls", 0) for r in val_rows)
    return {
        "n": n,
        "mean_reward": total_reward / n,
        "perfect_count": perfect,
        "perfect_rate": perfect / n,
        "nonzero_count": nonzero,
        "nonzero_rate": nonzero / n,
        "answered_count": answered,
        "answered_rate": answered / n,
        "truncated_count": truncated,
        "infeasible_count": infeasible,
        "mean_steps": mean_steps,
        "search_web_calls_total": search_total,
        "search_web_calls_per_task": search_total / n,
    }


def aggregate_test(rows) -> dict[str, Any]:
    test_rows = _split_rows(rows, "test")
    n = len(test_rows)
    if n == 0:
        return {"n": 0}
    answered = sum(1 for r in test_rows if r["terminal_kind"] == "send_msg")
    truncated = sum(1 for r in test_rows if r["terminal_kind"] == "truncated")
    infeasible = sum(1 for r in test_rows if r["terminal_kind"] == "infeasible")
    by_set = Counter(r["ab_set"] or "?" for r in test_rows)
    return {
        "n": n,
        "answered_count": answered,
        "answered_rate": answered / n,
        "truncated_count": truncated,
        "infeasible_count": infeasible,
        "by_set": dict(by_set),
    }


def write_grading_artifacts(run_dir: Path, rows, val_summary, test_summary) -> None:
    json_path = run_dir / "grading_summary.json"
    csv_path = run_dir / "grading_summary.csv"
    val_score_path = run_dir / "assistantbench_validation_score.json"
    pred_jsonl_path = run_dir / "assistantbench_test_predictions.jsonl"
    pred_human_path = run_dir / "assistantbench_test_predictions_human.json"

    json_path.write_text(json.dumps({
        "schema": "ab_grading_summary_v1",
        "n_total": len(rows),
        "validation_summary": val_summary,
        "test_summary": test_summary,
        "per_task": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "split", "index", "ab_id", "ab_set",
        "reward", "terminal_kind", "n_steps",
        "predicted_answer", "gold_answer", "task",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    val_score_path.write_text(json.dumps({
        "schema": "ab_validation_score_v1",
        **val_summary,
    }, indent=2), encoding="utf-8")

    test_rows = [r for r in rows if r["split"] == "test"]
    with pred_jsonl_path.open("w", encoding="utf-8") as fh:
        for r in test_rows:
            ab_id = r.get("ab_id")
            if not ab_id:
                continue
            fh.write(json.dumps({
                "id": ab_id,
                "answer": r["predicted_answer"] or "",
            }, ensure_ascii=False) + "\n")

    pred_human_path.write_text(json.dumps([
        {
            "id": r.get("ab_id"),
            "test_index": r["index"],
            "ab_set": r["ab_set"],
            "task": r["task"],
            "answer": r["predicted_answer"] or "",
            "terminal_kind": r["terminal_kind"],
            "n_steps": r["n_steps"],
        }
        for r in test_rows if r.get("ab_id")
    ], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  wrote: {json_path.relative_to(run_dir.parent)}")
    print(f"  wrote: {csv_path.relative_to(run_dir.parent)}")
    print(f"  wrote: {val_score_path.relative_to(run_dir.parent)}")
    print(f"  wrote: {pred_jsonl_path.relative_to(run_dir.parent)}  ({sum(1 for _ in open(pred_jsonl_path))} predictions)")


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def print_report(rows, val_summary, test_summary, run_dir: Path) -> None:
    print()
    print("=" * 72)
    print(f"  AssistantBench grading — {run_dir}")
    print("=" * 72)
    print(f"  total tasks graded: {len(rows)}")
    print()
    print(f"  --- Validation ({val_summary['n']}/33 tasks) ---")
    if val_summary["n"]:
        print(f"    mean_reward (DROP F1):  {val_summary['mean_reward']:.3f}")
        print(f"    perfect (=1.0):         {val_summary['perfect_count']}/{val_summary['n']} ({_fmt_pct(val_summary['perfect_rate'])})")
        print(f"    nonzero (>0):           {val_summary['nonzero_count']}/{val_summary['n']} ({_fmt_pct(val_summary['nonzero_rate'])})")
        print(f"    answered (send_msg):    {val_summary['answered_count']}/{val_summary['n']} ({_fmt_pct(val_summary['answered_rate'])})")
        print(f"    truncated (max_steps):  {val_summary['truncated_count']}")
        print(f"    infeasible:             {val_summary['infeasible_count']}")
        print(f"    mean steps:             {val_summary['mean_steps']:.1f}")
        print(f"    search_web/task:        {val_summary['search_web_calls_per_task']:.1f}")
    else:
        print("    (no validation tasks completed yet)")
    print()
    print(f"  --- Test ({test_summary['n']}/161 feasible-filtered tasks) ---")
    if test_summary["n"]:
        print(f"    answered (send_msg):    {test_summary['answered_count']}/{test_summary['n']} ({_fmt_pct(test_summary['answered_rate'])})")
        print(f"    truncated (max_steps):  {test_summary['truncated_count']}")
        print(f"    infeasible:             {test_summary['infeasible_count']}")
        for k, v in test_summary.get("by_set", {}).items():
            print(f"    by set: {k}={v}")
    else:
        print("    (no test tasks completed yet)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--no_dataset", action="store_true",
                    help="Skip HF dataset load (predictions JSONL will lack ab_id).")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: run_dir does not exist: {run_dir}", file=sys.stderr)
        return 2

    ab_meta = {} if args.no_dataset else _load_ab_metadata()
    rows = collect_task_rows(run_dir, ab_meta)
    if not rows:
        print(f"WARN: no completed AB episodes found under {run_dir}", file=sys.stderr)
        return 1

    val_summary = aggregate_validation(rows)
    test_summary = aggregate_test(rows)
    write_grading_artifacts(run_dir, rows, val_summary, test_summary)
    print_report(rows, val_summary, test_summary, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
