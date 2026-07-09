"""Combined Stage-0 audit runner.

Invokes the three Stage-0 audits (vocab_jaccard, predicate_firing_static,
slot_binding_feasibility) in sequence with a shared ``run_id`` so all three
drop into the same ``<output_root>/<run_id>/`` directory, then joins their
per-cell outputs into a unified ``upper_bounds.csv``.

The join is on ``(source_corpus, target_domain)``. ``upper_bound_admit_rate``
for each row is ``min(predicate_can_fire_rate, slot_some_bind_rate)``.

See ``implementation_notes/legacy/phase5-cross-domain-measurement.md`` Section 3.4.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from skill_transfer_test.extract.audits import (
    predicate_firing_static,
    slot_binding_feasibility,
    vocab_jaccard,
)


_DEFAULT_BANKS_ROOT: list[Path] = [
    Path("labeling/skill_bank_out"),
    Path("skill_transfer_test/skill_bank_local/full_v5"),
]
_AUDIT_NAMES: tuple[str, ...] = (
    "vocab_jaccard",
    "predicate_firing_static",
    "slot_binding_feasibility",
)
_AUDIT_MODULES = {
    "vocab_jaccard": vocab_jaccard,
    "predicate_firing_static": predicate_firing_static,
    "slot_binding_feasibility": slot_binding_feasibility,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all Stage-0 audits and emit upper_bounds.csv.",
    )
    p.add_argument(
        "--banks-root",
        nargs="+",
        type=Path,
        default=list(_DEFAULT_BANKS_ROOT),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("cross_domain_results/_phase0"),
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=f"phase0_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--skip",
        nargs="+",
        choices=list(_AUDIT_NAMES),
        default=[],
        help="Audit names to skip (useful for re-running just the join step).",
    )
    return p


def _build_sub_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--output-root",
        str(args.output_root),
        "--run-id",
        args.run_id,
        "--banks-root",
        *(str(p) for p in args.banks_root),
    ]
    if args.verbose:
        argv.append("--verbose")
    return argv


def _first_present(cell: dict, keys: tuple[str, ...], default: float) -> float:
    """Return the first value present in ``cell`` for any of ``keys``.

    Lets the runner survive small naming differences in B's predicate-firing
    JSON schema without needing a hard contract on exact field names.
    """
    for k in keys:
        if k in cell and cell[k] is not None:
            return cell[k]
    return default


def _join_upper_bounds(out_dir: Path) -> tuple[Path, list[dict]]:
    """Read the three audit outputs and write ``upper_bounds.csv``."""
    pred_path = out_dir / "predicate_firing_static.json"
    slot_path = out_dir / "slot_binding_feasibility.json"
    vocab_path = out_dir / "vocab_jaccard.json"
    for required in (pred_path, slot_path, vocab_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"_runner: cannot build upper_bounds.csv; missing {required}"
            )

    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    slot = json.loads(slot_path.read_text(encoding="utf-8"))

    pred_by_key: dict[tuple[str, str], dict] = {
        (c["source_corpus"], c["target_domain"]): c for c in pred.get("cells", [])
    }
    slot_by_key: dict[tuple[str, str], dict] = {
        (c["source_corpus"], c["target_domain"]): c for c in slot.get("cells", [])
    }

    keys = sorted(set(pred_by_key) & set(slot_by_key))
    rows: list[dict] = []
    for key in keys:
        p_cell = pred_by_key[key]
        s_cell = slot_by_key[key]
        pred_can_fire = float(_first_present(p_cell, (
            "cell_max_admit_rate",
            "cell_can_fire_rate",
            "predicate_can_fire_rate",
            "can_fire_rate",
        ), default=0.0))
        pred_mean_cov = float(_first_present(p_cell, (
            "mean_predicate_coverage",
            "predicate_mean_coverage",
            "mean_coverage",
        ), default=0.0))
        slot_full = float(s_cell["cell_full_bind_rate"])
        slot_some = float(s_cell["cell_some_bind_rate"])
        slot_mean = float(s_cell["mean_bindable_fraction"])
        rows.append({
            "source_corpus": s_cell["source_corpus"],
            "source_cluster": s_cell["source_cluster"],
            "target_domain": s_cell["target_domain"],
            "n_skills": s_cell["n_skills"],
            "predicate_can_fire_rate": pred_can_fire,
            "predicate_mean_coverage": pred_mean_cov,
            "slot_full_bind_rate": slot_full,
            "slot_some_bind_rate": slot_some,
            "slot_mean_bindable_fraction": slot_mean,
            "upper_bound_admit_rate": min(pred_can_fire, slot_some),
        })

    rows.sort(key=lambda r: (r["source_cluster"], r["source_corpus"], r["target_domain"]))

    csv_path = out_dir / "upper_bounds.csv"
    fieldnames = [
        "source_corpus", "source_cluster", "target_domain",
        "n_skills",
        "predicate_can_fire_rate", "predicate_mean_coverage",
        "slot_full_bind_rate", "slot_some_bind_rate", "slot_mean_bindable_fraction",
        "upper_bound_admit_rate",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, rows


def _print_summary(out_dir: Path, rows: list[dict]) -> None:
    by_target: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["source_cluster"] == "game":
            by_target[r["target_domain"]].append(r["upper_bound_admit_rate"])

    def _mean(domain: str) -> float:
        vals = by_target.get(domain, [])
        return statistics.fmean(vals) if vals else 0.0

    print("_phase0 runner complete:")
    print(f"  out_dir: {out_dir}/")
    print(f"  n_cells: {len(rows)}")
    print("  headline upper-bound admit rates (game-source -> cross-cluster targets):")
    print(f"    gym_v -> visual_reasoning : {_mean('visual_reasoning'):.2f}  (Stage 1 image-VR oracle)")
    print(f"    gym_v -> video            : {_mean('video'):.2f}  (Stage 2 video-VR oracle)")
    print(f"    gym_v -> osworld          : {_mean('osworld'):.2f}  (Stage 3 osworld oracle)")
    print(f"    gym_v -> browser          : {_mean('browser'):.2f}  (Stage 4 browsergym oracle)")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = args.output_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_argv = _build_sub_argv(args)
    for name in _AUDIT_NAMES:
        if name in args.skip:
            print(f"_runner: skipping {name} (per --skip)", file=sys.stderr)
            continue
        module = _AUDIT_MODULES[name]
        print(f"_runner: invoking {name}.main(...)", file=sys.stderr)
        rc = module.main(sub_argv)
        if rc != 0:
            print(f"_runner: {name} exited with rc={rc}; aborting", file=sys.stderr)
            return rc

    _, rows = _join_upper_bounds(out_dir)
    _print_summary(out_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
