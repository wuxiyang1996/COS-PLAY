"""Harness failure-mode pie chart (block E4).

Counts veto codes from ``<run-dir>/harness_log/rejections.jsonl`` and
breaks down the per-event failure mode of skill-eligibility filtering.
The 8 canonical codes are defined in :mod:`harness.eligibility`:

* ``status_not_runnable``
* ``shadow_disallowed``
* ``domain_mismatch``
* ``task_mismatch``
* ``skill_type_mismatch``
* ``no_adapter``
* ``adapter_raised``
* ``adapter_cannot_handle``

Optionally also breaks down ``validate_invocation`` failure axes
(``binding_ok / precondition_ok / evidence_ok / adapter_ok``) from
``harness_log/validate.jsonl`` for the §4.3 mutation→repair table.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _aggregate_rejections(path: Path) -> Dict[str, Any]:
    veto_counts: Counter = Counter()
    by_domain: Dict[str, Counter] = {}
    n_rows = 0
    for r in _iter_jsonl(path):
        veto = r.get("veto") or "UNKNOWN"
        domain = r.get("domain") or "unknown"
        veto_counts[veto] += 1
        by_domain.setdefault(domain, Counter())[veto] += 1
        n_rows += 1
    return {
        "n_rows": n_rows,
        "veto_counts": dict(veto_counts),
        "by_domain": {d: dict(c) for d, c in by_domain.items()},
    }


def _aggregate_validate(path: Path) -> Dict[str, Any]:
    axes_fail: Counter = Counter()
    veto_reasons: Counter = Counter()
    n_total = 0
    n_ok = 0
    for r in _iter_jsonl(path):
        n_total += 1
        if r.get("ok"):
            n_ok += 1
            continue
        if not r.get("binding_ok", True):
            axes_fail["binding"] += 1
        if not r.get("precondition_ok", True):
            axes_fail["precondition"] += 1
        if not r.get("evidence_ok", True):
            axes_fail["evidence"] += 1
        if not r.get("adapter_ok", True):
            axes_fail["adapter"] += 1
        for reason in r.get("veto_reasons") or []:
            veto_reasons[str(reason)] += 1
    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_failed": n_total - n_ok,
        "axes_fail": dict(axes_fail),
        "top_veto_reasons": dict(veto_reasons.most_common(20)),
    }


def _maybe_plot_pie(counts: Dict[str, int], out_path: Path, title: str) -> None:
    if not counts:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable (%s)", exc)
        return
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    labels, values = zip(*items)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title(title)
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

    rejections = _aggregate_rejections(
        args.run_dir / "harness_log" / "rejections.jsonl"
    )
    validates = _aggregate_validate(
        args.run_dir / "harness_log" / "validate.jsonl"
    )

    summary = {
        "schema_version": 1,
        "run_dir": str(args.run_dir),
        "rejections": rejections,
        "validate_invocation": validates,
    }
    summary_path = out_dir / "failure_modes_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s", summary_path)

    if not args.no_plot:
        _maybe_plot_pie(
            rejections["veto_counts"],
            out_dir / "failure_modes_eligibility_pie.png",
            "Eligibility veto codes (per-event)",
        )
        _maybe_plot_pie(
            validates["axes_fail"],
            out_dir / "failure_modes_validate_axes.png",
            "validate_invocation failure axes",
        )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
