"""Cross-domain SkillBridge eval aggregator (block C7).

Reads the per-domain ``*_result.json`` files emitted by the C2-C6
drivers (``eval_browsergym``, ``eval_osworld``, ``eval_visual_reasoning``,
``eval_video``, ``eval_gymv``) and produces a single CSV / JSON / Markdown
table that mirrors what the NeurIPS paper's main-results table needs.

Usage (typical)::

    python -m scripts.skillbridge_eval.eval_aggregator \\
        --run-dir runs/skillbridge_v12 \\
        --output runs/skillbridge_v12/eval/aggregate.json \\
        --markdown runs/skillbridge_v12/eval/aggregate.md

Per-domain primary metric is normalised so the aggregate table is
directly comparable:

* ``browsergym``         -> ``success_rate_macro``
* ``osworld``            -> ``success_rate_macro``  (also keeps eval_score)
* ``visual_reasoning``   -> ``accuracy_micro``
* ``video``              -> ``accuracy_micro``
* ``gymv``               -> ``mean_reward_macro``
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_DOMAIN_PRIMARY: Dict[str, Tuple[str, str]] = {
    "browsergym":       ("success_rate_macro",   "overall"),
    "osworld":          ("success_rate_macro",   "overall"),
    "visual_reasoning": ("accuracy_micro",       "overall"),
    "video":            ("accuracy_micro",       "overall"),
    "gymv":             ("mean_reward_macro",    "overall"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--eval-dir", type=Path, default=None,
        help="Override location of per-domain *_result.json files. "
             "Defaults to <run-dir>/eval/.",
    )
    p.add_argument(
        "--results", nargs="*", default=None,
        help="Explicit list of *_result.json paths to aggregate. "
             "If unset, the eval-dir is scanned.",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--markdown", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _discover_results(eval_dir: Path) -> List[Path]:
    if not eval_dir.exists():
        return []
    suffixes = {
        "browsergym_result_",
        "osworld_result_",
        "visual_reasoning_result_",
        "video_result_",
        "gymv_result_",
    }
    candidates = sorted(eval_dir.glob("*_result_*.json"))
    out: Dict[str, Path] = {}
    for path in candidates:
        for prefix in suffixes:
            if path.name.startswith(prefix):
                domain = prefix.removesuffix("_result_")
                # Most recent timestamp wins.
                if domain not in out or path.stat().st_mtime > out[domain].stat().st_mtime:
                    out[domain] = path
                break
    return [out[d] for d in sorted(out)]


def _extract_primary(
    domain: str, payload: Dict[str, Any]
) -> Optional[float]:
    spec = _DOMAIN_PRIMARY.get(domain)
    if not spec:
        return None
    metric, scope = spec
    bucket = payload.get(scope, {}) or {}
    val = bucket.get(metric)
    return float(val) if isinstance(val, (int, float)) else None


def _row(domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    primary = _extract_primary(domain, payload)
    overall = payload.get("overall", {}) or {}
    return {
        "domain": domain,
        "label": payload.get("label", "skillbridge"),
        "primary_metric": _DOMAIN_PRIMARY.get(domain, ("?",))[0],
        "primary_value": primary,
        "n_tasks": overall.get(
            "n_tasks", overall.get("n_benchmarks", overall.get("n_games"))
        ),
        "support": overall.get("samples_completed_total")
            or overall.get("n_episodes_total")
            or overall.get("n_tasks_completed"),
        "model": payload.get("model"),
        "out_dir": payload.get("out_dir"),
    }


def _to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_(no domain results found)_\n"
    headers = [
        "domain", "label", "primary_metric", "primary_value",
        "n_tasks", "support", "model",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h)
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    eval_dir = args.eval_dir or (args.run_dir / "eval")
    if args.results:
        result_paths = [Path(p) for p in args.results]
    else:
        result_paths = _discover_results(eval_dir)

    if not result_paths:
        logger.warning("no per-domain *_result*.json under %s", eval_dir)

    rows: List[Dict[str, Any]] = []
    raw: Dict[str, Any] = {}
    for path in result_paths:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip %s: %s", path, exc)
            continue
        domain = payload.get("domain") or path.stem.split("_result_")[0]
        rows.append(_row(domain, payload))
        raw[domain] = payload

    out_payload = {
        "schema_version": 1,
        "run_dir": str(args.run_dir),
        "rows": rows,
        "n_domains": len(rows),
    }

    out_path = args.output or (eval_dir / "aggregate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, default=str)
    logger.info("wrote aggregate JSON %s", out_path)

    if args.csv or rows:
        csv_path = args.csv or (eval_dir / "aggregate.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "domain", "label", "primary_metric",
                    "primary_value", "n_tasks", "support", "model",
                ],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in writer.fieldnames})
        logger.info("wrote aggregate CSV  %s", csv_path)

    if args.markdown or rows:
        md_path = args.markdown or (eval_dir / "aggregate.md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_to_markdown(rows))
        logger.info("wrote aggregate MD   %s", md_path)

    print("\n=== skillbridge cross-domain aggregate ===")
    print(_to_markdown(rows))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
