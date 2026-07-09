"""CLI driver for cross-corpus skill bank extraction.

Examples:

    # LLM-free smoke run on all 6 corpora, 50 samples / episodes each:
    python -m skill_transfer_test.extract.runner \
        --output-root skill_transfer_test/skill_bank_local \
        --corpora all --max-samples 50

    # One corpus only:
    python -m skill_transfer_test.extract.runner \
        --corpora siv_bench --max-samples 100

The runner picks the right driver per :class:`CorpusSpec.lift_kind`
(``single_shot`` -> :func:`single_shot_lift.lift_corpus`;
``sequence`` -> :func:`sequence_lift.lift_corpus_per_episode` for now).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from . import single_shot_lift, sequence_lift
from ._corpus_specs import CorpusSpec, all_names, get_spec


def _run_one(
    spec: CorpusSpec,
    *,
    output_dir: Path,
    max_samples: int | None,
    include_incorrect: bool,
    verbose: bool,
):
    if spec.lift_kind == "single_shot":
        return single_shot_lift.lift_corpus(
            spec,
            output_dir=output_dir,
            max_samples=max_samples,
            include_incorrect=include_incorrect,
            verbose=verbose,
        )
    if spec.lift_kind == "sequence":
        return sequence_lift.lift_corpus_per_episode(
            spec,
            output_dir=output_dir,
            max_episodes=max_samples,
            verbose=verbose,
        )
    raise ValueError(f"unknown lift_kind {spec.lift_kind!r} for {spec.name!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpora", nargs="+", default=["all"],
                   help=f"one of {sorted(all_names())} or 'all'")
    p.add_argument("--output-root", type=Path,
                   default=Path("skill_transfer_test/skill_bank_local"))
    p.add_argument("--run-id", default=None,
                   help="run dir name; default: smoke_<utc-ts>")
    p.add_argument("--max-samples", type=int, default=None,
                   help="per-corpus cap on samples / episodes")
    p.add_argument("--include-incorrect", action="store_true",
                   help="include single-shot samples with correct=False")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    corpora = list(all_names()) if args.corpora == ["all"] else args.corpora
    run_id = args.run_id or datetime.utcnow().strftime("smoke_%Y%m%d_%H%M%S")
    out_root = args.output_root / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[runner] output_root={out_root}")
    print(f"[runner] corpora={corpora}")
    print(f"[runner] max_samples={args.max_samples}")
    summaries = []
    for name in corpora:
        spec = get_spec(name)
        print(f"  - {name} (lift_kind={spec.lift_kind}, modality={spec.modality})")
        try:
            summary = _run_one(
                spec,
                output_dir=out_root,
                max_samples=args.max_samples,
                include_incorrect=args.include_incorrect,
                verbose=args.verbose,
            )
            summaries.append(summary)
            n_lifted = summary.get("n_samples_lifted") or summary.get("n_episodes_lifted")
            n_seen = summary.get("n_samples_seen") or summary.get("n_episodes_seen")
            fr = summary.get("fallback_rate", 0.0)
            print(f"    lifted={n_lifted}/{n_seen}  fallback={fr:.1%}")
        except Exception as exc:                                       # noqa: BLE001
            print(f"    ERROR: {exc}")
            summaries.append({"corpus": name, "error": str(exc)})

    rollup = out_root / "rollup.json"
    rollup.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"[runner] rollup -> {rollup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
