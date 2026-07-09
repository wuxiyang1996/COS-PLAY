"""Paired-bootstrap significance test (block E7).

Given two per-domain ``*_result.json`` files (one for ``model A`` and
one for ``model B`` over the *same* tasks), computes the paired
bootstrap distribution of the difference in the primary metric and
reports a one- and two-sided p-value with a 95% CI.

Supports both task-level (BrowserGym / OSWorld) and benchmark-level
(visual_reasoning / video) granularities by reading whichever
``per_task`` / ``per_benchmark`` block is present.

Usage::

    python -m scripts.skillbridge_analysis.compute_significance \\
        --domain browsergym \\
        --baseline runs/skillbridge_v12/eval/browsergym_result_baseline.json \\
        --treatment runs/skillbridge_v12/eval/browsergym_result_skillbridge.json \\
        --n-bootstrap 5000

The baseline + treatment files must share the same task / benchmark
keys; tasks present in only one file are dropped with a warning.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


_PRIMARY_KEY = {
    "browsergym":       ("per_task", "success_rate"),
    "osworld":          ("per_task", "success_rate"),
    "visual_reasoning": ("per_benchmark", "accuracy"),
    "video":            ("per_benchmark", "accuracy"),
    "gymv":             ("per_game", "success_rate"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--domain", required=True, choices=list(_PRIMARY_KEY))
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--treatment", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _load_paired(
    args: argparse.Namespace,
) -> List[Tuple[str, float, float]]:
    bucket_key, metric = _PRIMARY_KEY[args.domain]

    def _read(path: Path) -> Dict[str, float]:
        try:
            data = json.loads(Path(path).read_text())
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"could not read {path}: {exc}") from None
        bucket = data.get(bucket_key) or {}
        out: Dict[str, float] = {}
        for k, v in bucket.items():
            if v is None:
                continue
            val = v.get(metric) if isinstance(v, dict) else None
            if isinstance(val, (int, float)) and not math.isnan(val):
                out[str(k)] = float(val)
        return out

    base = _read(args.baseline)
    treat = _read(args.treatment)

    common = sorted(base.keys() & treat.keys())
    only_base = base.keys() - common
    only_treat = treat.keys() - common
    if only_base:
        logger.warning("ignoring %d task(s) only in baseline", len(only_base))
    if only_treat:
        logger.warning("ignoring %d task(s) only in treatment", len(only_treat))
    return [(k, base[k], treat[k]) for k in common]


def _paired_bootstrap(
    paired: List[Tuple[str, float, float]],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    diffs = [t - b for _, b, t in paired]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean_diff": 0.0, "p_two_sided": None}

    observed = sum(diffs) / n
    boots = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()

    def _quantile(p: float) -> float:
        idx = max(0, min(len(boots) - 1, int(round(p * (len(boots) - 1)))))
        return boots[idx]

    p_one_sided = sum(1 for b in boots if b <= 0) / n_bootstrap
    p_two_sided = 2 * min(p_one_sided, 1.0 - p_one_sided)

    return {
        "n_paired": n,
        "n_bootstrap": n_bootstrap,
        "mean_baseline": sum(b for _, b, _ in paired) / n,
        "mean_treatment": sum(t for _, _, t in paired) / n,
        "mean_diff": observed,
        "ci95_low": _quantile(0.025),
        "ci95_high": _quantile(0.975),
        "p_one_sided_treatment_le_baseline": p_one_sided,
        "p_two_sided": p_two_sided,
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    paired = _load_paired(args)
    result = _paired_bootstrap(paired, args.n_bootstrap, args.seed)
    payload = {
        "schema_version": 1,
        "domain": args.domain,
        "baseline": str(args.baseline),
        "treatment": str(args.treatment),
        **result,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("wrote %s", args.output)

    print("=== paired bootstrap ===")
    for k, v in payload.items():
        if isinstance(v, float):
            print(f"  {k:32s} : {v:.6f}")
        else:
            print(f"  {k:32s} : {v}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
