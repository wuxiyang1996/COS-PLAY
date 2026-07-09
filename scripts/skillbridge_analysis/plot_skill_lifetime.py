"""Skill lifetime distribution (block E3).

For every ``skill_id`` we observe in
``<run-dir>/lifecycle_log/transitions.jsonl``, compute:

* ``birth_step``   = step at the first DRAFT/PROVISIONAL transition (or
  the first transition we observe involving the skill).
* ``death_step``   = step at the first DEPRECATED/RETIRED transition,
  or ``None`` if the skill is still alive at the end of training.
* ``lifetime``     = ``death_step - birth_step`` (or ``last_seen_step -
  birth_step`` for living skills, with a ``censored`` flag).

Outputs:

* ``skill_lifetime_summary.json`` — per-skill records + percentiles.
* ``skill_lifetime_hist.png``     — histogram of skill lifetimes
  (separate stack for censored vs uncensored).
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


_DEAD = {"DEPRECATED", "RETIRED"}
_BORN = {"DRAFT", "PROVISIONAL"}


def _lifetimes(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_skill: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sid = r.get("skill_id")
        if not sid:
            continue
        by_skill[str(sid)].append(r)

    last_step = max(
        (int(r.get("step") or 0) for r in rows), default=0
    )

    records: List[Dict[str, Any]] = []
    for sid, transitions in by_skill.items():
        transitions = sorted(
            transitions,
            key=lambda r: (r.get("step") or 0, r.get("ts") or 0.0),
        )
        birth_step = None
        last_seen = None
        death_step = None
        for r in transitions:
            step = int(r.get("step") or 0)
            last_seen = step
            to_status = (r.get("to_status") or "").upper()
            from_status = (r.get("from_status") or "").upper()
            if birth_step is None and (
                to_status in _BORN or from_status in _BORN or to_status
            ):
                birth_step = step
            if to_status in _DEAD and death_step is None:
                death_step = step
        if birth_step is None:
            continue
        if death_step is not None:
            lifetime = max(0, death_step - birth_step)
            censored = False
        else:
            lifetime = max(0, (last_seen or last_step) - birth_step)
            censored = True
        records.append({
            "skill_id": sid,
            "birth_step": birth_step,
            "death_step": death_step,
            "lifetime": lifetime,
            "censored": censored,
            "n_transitions": len(transitions),
        })

    if records:
        sorted_lifetimes = sorted(r["lifetime"] for r in records)
        n = len(sorted_lifetimes)
        def pct(p: float) -> float:
            idx = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
            return float(sorted_lifetimes[idx])
        percentiles = {
            "p10": pct(10), "p25": pct(25), "p50": pct(50),
            "p75": pct(75), "p90": pct(90), "p99": pct(99),
        }
    else:
        percentiles = {}

    return {
        "schema_version": 1,
        "n_skills": len(records),
        "n_died": sum(1 for r in records if not r["censored"]),
        "n_alive": sum(1 for r in records if r["censored"]),
        "percentiles": percentiles,
        "records": records,
    }


def _maybe_plot(summary: Dict[str, Any], out_path: Path) -> None:
    records: List[Dict[str, Any]] = summary.get("records", [])
    if not records:
        logger.info("no skill lifetimes — skipping plot")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable (%s)", exc)
        return
    died = [r["lifetime"] for r in records if not r["censored"]]
    alive = [r["lifetime"] for r in records if r["censored"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = max(20, min(60, len(records) // 8))
    ax.hist(
        [died, alive], bins=bins, stacked=True,
        label=["died", "alive at run-end"],
        color=["#fb5607", "#3a86ff"],
        edgecolor="white", linewidth=0.4,
    )
    ax.set_xlabel("lifetime (trainer steps)")
    ax.set_ylabel("# skills")
    ax.set_title("Skill lifetime distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    out_dir = args.out_dir or (args.run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(args.run_dir / "lifecycle_log" / "transitions.jsonl")
    summary = _lifetimes(rows)

    summary_path = out_dir / "skill_lifetime_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s (n=%d)", summary_path, summary["n_skills"])

    if not args.no_plot:
        _maybe_plot(summary, out_dir / "skill_lifetime_hist.png")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
