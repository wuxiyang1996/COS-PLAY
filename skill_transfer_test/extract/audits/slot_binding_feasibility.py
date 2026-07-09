"""Stage-0 audit: per-skill slot-binding feasibility against target domains.

For each ``(source_bank, target_domain)`` cell, count how many source skills
have slot types the target's adapter + schema producer can bind values for.
``target_domain`` is one of ``TARGET_DOMAINS`` and the binding oracle is
:data:`TARGET_SLOT_TYPE_VOCAB`.

A skill with zero slot types is treated as trivially "fully bindable" (1.0):
there is nothing to fail to bind.

See ``implementation_notes/legacy/phase5-cross-domain-measurement.md`` Section 3.3.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from datetime import datetime
from pathlib import Path

from skill_transfer_test.extract.audits._loaders import (
    CorpusVocab,
    SkillVocab,
    collect_corpus_vocab,
    discover_banks,
)
from skill_transfer_test.extract.audits._target_vocabularies import (
    TARGET_DOMAINS,
    TARGET_SLOT_TYPE_VOCAB,
)


_DEFAULT_BANKS_ROOT: list[Path] = [
    Path("labeling/skill_bank_out"),
    Path("skill_transfer_test/skill_bank_local/full_v5"),
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage-0 slot-binding feasibility audit.",
    )
    p.add_argument(
        "--banks-root",
        nargs="+",
        type=Path,
        default=list(_DEFAULT_BANKS_ROOT),
        help="One or more roots to walk for skill_bank.jsonl files.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("cross_domain_results/_phase0"),
        help="Output dir; per-run subdir uses --run-id.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=f"phase0_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        help="Run id (becomes leaf dir under --output-root).",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def _per_skill_row(
    skill: SkillVocab,
    target_vocab: frozenset[str],
) -> dict:
    skill_slots = skill.slot_types
    bindable = skill_slots & target_vocab
    n_bindable = len(bindable)
    n_total = len(skill_slots)
    bindable_fraction = n_bindable / n_total if n_total > 0 else 1.0
    fully_bindable = bindable_fraction == 1.0
    some_bindable = (n_bindable >= 1) or (n_total == 0)
    return {
        "skill_id": skill.skill_id,
        "n_slot_types": n_total,
        "n_bindable": n_bindable,
        "bindable_fraction": bindable_fraction,
        "fully_bindable": fully_bindable,
        "some_bindable": some_bindable,
    }


def _build_cell(
    corpus_vocab: CorpusVocab,
    target_domain: str,
) -> tuple[dict, list[dict]]:
    target_vocab = TARGET_SLOT_TYPE_VOCAB[target_domain]
    per_skill: list[dict] = []
    for skill in corpus_vocab.skills:
        per_skill.append(_per_skill_row(skill, target_vocab))

    n_skills = corpus_vocab.n_skills
    n_full = sum(1 for r in per_skill if r["fully_bindable"])
    n_some = sum(1 for r in per_skill if r["some_bindable"])
    if per_skill:
        mean_frac = statistics.fmean(r["bindable_fraction"] for r in per_skill)
    else:
        mean_frac = 0.0

    bi = corpus_vocab.bank_info
    cell = {
        "source_corpus": bi.label,
        "source_cluster": bi.cluster,
        "target_domain": target_domain,
        "n_skills": n_skills,
        "n_fully_bindable": n_full,
        "n_some_bindable": n_some,
        "cell_full_bind_rate": (n_full / n_skills) if n_skills else 0.0,
        "cell_some_bind_rate": (n_some / n_skills) if n_skills else 0.0,
        "mean_bindable_fraction": mean_frac,
        "source_slot_type_vocab": sorted(corpus_vocab.slot_types),
        "bindable_intersection": sorted(corpus_vocab.slot_types & target_vocab),
        "non_bindable_slot_types": sorted(corpus_vocab.slot_types - target_vocab),
    }

    per_skill_lines: list[dict] = []
    for r in per_skill:
        per_skill_lines.append({
            "skill_id": r["skill_id"],
            "source_corpus": bi.label,
            "source_cluster": bi.cluster,
            "target_domain": target_domain,
            "n_slot_types": r["n_slot_types"],
            "n_bindable": r["n_bindable"],
            "bindable_fraction": r["bindable_fraction"],
            "fully_bindable": r["fully_bindable"],
            "some_bindable": r["some_bindable"],
        })
    return cell, per_skill_lines


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = args.output_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        banks = discover_banks(args.banks_root)
    print(
        f"slot_binding_feasibility: discovered {len(banks)} banks "
        f"(suppressed UserWarnings during walk)",
        file=sys.stderr,
    )

    cells: list[dict] = []
    per_skill_path = out_dir / "slot_binding_per_skill.jsonl"
    with per_skill_path.open("w", encoding="utf-8") as per_skill_fh:
        for bi in banks:
            corpus_vocab = collect_corpus_vocab(bi)
            if args.verbose:
                print(
                    f"  {bi.cluster}/{bi.label}: n_skills={corpus_vocab.n_skills} "
                    f"slot_types={sorted(corpus_vocab.slot_types)}",
                    file=sys.stderr,
                )
            for target_domain in TARGET_DOMAINS:
                cell, per_skill_lines = _build_cell(corpus_vocab, target_domain)
                cells.append(cell)
                for line in per_skill_lines:
                    per_skill_fh.write(json.dumps(line) + "\n")

    summary = {
        "run_id": args.run_id,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_target_domains": len(TARGET_DOMAINS),
        "target_slot_type_vocab_sizes": {
            d: len(TARGET_SLOT_TYPE_VOCAB[d]) for d in TARGET_DOMAINS
        },
        "cells": cells,
    }
    summary_path = out_dir / "slot_binding_feasibility.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    headline_targets = {"visual_reasoning", "video", "browser", "osworld"}
    headline_rates = [
        c["cell_some_bind_rate"]
        for c in cells
        if c["source_cluster"] == "game" and c["target_domain"] in headline_targets
    ]
    headline_mean = statistics.fmean(headline_rates) if headline_rates else 0.0
    n_cells = f"{len(banks)}x{len(TARGET_DOMAINS)}"
    print(
        f"slot_binding_feasibility: {summary_path} | n_cells={n_cells} | "
        f"mean cell_some_bind_rate across cross-cluster game->{{vr,video,browser,osworld}} "
        f"= {headline_mean:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
