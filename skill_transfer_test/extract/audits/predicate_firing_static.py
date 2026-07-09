"""Predicate-firing static audit (Stage-0 admission upper bound).

For each ``(source_bank, target_domain)`` cell, counts how many source
skills carry **at least one** predicate type that the target domain's
``success_fn`` knows how to evaluate (per
``_target_vocabularies.TARGET_PREDICATE_VOCAB``). This is the static
upper bound on the cross-cluster admission rate before any runtime
schema-producer wiring lands; live admission rates can only be at or
below this.

The "predicate set" of a skill is computed as
``hop_predicates ∪ contract_predicates`` so the same metric is well-defined
across game banks (which expose predicate types via per-hop
``effects_add/del``) and cross-domain banks (which expose them via the
skill-level ``contract.effects_*``). See the ``_loaders.extract_skill_vocab``
docstring for shape details.

Outputs ``predicate_firing_static.json`` (per-cell aggregate) and
``predicate_firing_per_skill.jsonl`` (per ``(skill, target_domain)`` row)
under ``<output-root>/<run-id>/``.

Invocation::

    python -m skill_transfer_test.extract.audits.predicate_firing_static --verbose
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def _utcnow() -> datetime:
    """Timezone-aware UTC ``datetime.now`` (avoids ``utcnow`` deprecation)."""
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    """ISO-8601 UTC timestamp ending in ``Z`` for stable JSON output."""
    return _utcnow().replace(tzinfo=None).isoformat(timespec="seconds") + "Z"

from ._loaders import (
    BankInfo,
    CorpusVocab,
    SkillVocab,
    collect_corpus_vocab,
    discover_banks,
)
from ._target_vocabularies import TARGET_DOMAINS, TARGET_PREDICATE_VOCAB


_DEFAULT_BANKS_ROOT: tuple[Path, ...] = (
    Path("labeling/skill_bank_out"),
    Path("skill_transfer_test/skill_bank_local/full_v5"),
)
_DEFAULT_OUTPUT_ROOT = Path("cross_domain_results/_phase0")

_HEADLINE_TARGETS_FOR_GAME: tuple[str, ...] = (
    "visual_reasoning",
    "video",
    "browser",
    "osworld",
)


def _skill_predicates(sv: SkillVocab) -> frozenset[str]:
    """Full predicate set of a skill across hop and contract surfaces."""
    return sv.hop_predicates | sv.contract_predicates


def _per_skill_row(
    sv: SkillVocab,
    cv: CorpusVocab,
    target_domain: str,
    target_vocab: frozenset[str],
) -> dict:
    skill_preds = _skill_predicates(sv)
    supported = skill_preds & target_vocab
    n_total = len(skill_preds)
    n_supp = len(supported)
    return {
        "skill_id": sv.skill_id,
        "source_corpus": cv.bank_info.label,
        "source_corpus_subdir": cv.bank_info.corpus_subdir,
        "source_cluster": cv.bank_info.cluster,
        "target_domain": target_domain,
        "n_predicates": n_total,
        "n_supported": n_supp,
        "coverage": round(n_supp / n_total, 4) if n_total > 0 else 0.0,
        "can_fire": n_supp >= 1,
    }


def _cell_row(
    cv: CorpusVocab,
    target_domain: str,
    target_vocab: frozenset[str],
    skill_rows: list[dict],
) -> dict:
    """Aggregate per-skill rows for one ``(source_corpus, target_domain)`` cell."""
    n_skills = cv.n_skills
    n_can_fire = sum(1 for r in skill_rows if r["can_fire"])
    coverages_with_preds = [r["coverage"] for r in skill_rows if r["n_predicates"] > 0]
    n_with_preds = len(coverages_with_preds)
    source_predicate_vocab = cv.hop_predicates | cv.contract_predicates
    return {
        "source_corpus": cv.bank_info.label,
        "source_corpus_subdir": cv.bank_info.corpus_subdir,
        "source_cluster": cv.bank_info.cluster,
        "target_domain": target_domain,
        "n_skills": n_skills,
        "n_skills_with_predicates": n_with_preds,
        "n_skills_can_fire": n_can_fire,
        "cell_max_admit_rate": round(n_can_fire / n_skills, 4) if n_skills > 0 else 0.0,
        "mean_coverage": round(mean(coverages_with_preds), 4) if coverages_with_preds else 0.0,
        "source_predicate_vocab": sorted(source_predicate_vocab),
        "supported_intersection": sorted(source_predicate_vocab & target_vocab),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Static predicate-firing audit (per-cell admission upper bound).",
    )
    p.add_argument("--banks-root", nargs="+", type=Path, default=list(_DEFAULT_BANKS_ROOT))
    p.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    p.add_argument(
        "--run-id",
        type=str,
        default=f"phase0_{_utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    out_dir = args.output_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        banks: list[BankInfo] = discover_banks(args.banks_root)
    n_warnings_suppressed = len(caught)

    if not banks:
        print(
            f"predicate_firing_static: ERROR no banks discovered under "
            f"{args.banks_root}; nothing to audit",
        )
        return 1

    if args.verbose:
        print(f"predicate_firing_static: discovered {len(banks)} banks "
              f"({n_warnings_suppressed} discovery warnings suppressed)")

    corpus_vocabs: list[CorpusVocab] = [collect_corpus_vocab(bi) for bi in banks]

    cells: list[dict] = []
    per_skill_rows: list[dict] = []
    for cv in corpus_vocabs:
        for target in TARGET_DOMAINS:
            target_vocab = TARGET_PREDICATE_VOCAB[target]
            cell_skill_rows = [
                _per_skill_row(sv, cv, target, target_vocab) for sv in cv.skills
            ]
            per_skill_rows.extend(cell_skill_rows)
            cells.append(_cell_row(cv, target, target_vocab, cell_skill_rows))
            if args.verbose:
                cell = cells[-1]
                print(
                    f"  {cv.bank_info.label:32s} -> {target:18s} "
                    f"can_fire={cell['n_skills_can_fire']:4d}/{cell['n_skills']:4d} "
                    f"max_admit={cell['cell_max_admit_rate']:.2f} "
                    f"mean_cov={cell['mean_coverage']:.2f}"
                )

    aggregate = {
        "run_id": args.run_id,
        "generated_at": _utc_iso(),
        "banks_root": [str(p) for p in args.banks_root],
        "n_banks_total": len(banks),
        "n_banks_game": sum(1 for b in banks if b.cluster == "game"),
        "n_banks_cross_domain": sum(1 for b in banks if b.cluster == "cross_domain"),
        "warnings_suppressed": n_warnings_suppressed,
        "n_target_domains": len(TARGET_DOMAINS),
        "target_domains": list(TARGET_DOMAINS),
        "target_predicate_vocab_sizes": {
            t: len(TARGET_PREDICATE_VOCAB[t]) for t in TARGET_DOMAINS
        },
        "n_source_corpora": len(corpus_vocabs),
        "n_skills_total": sum(cv.n_skills for cv in corpus_vocabs),
        "n_per_skill_rows": len(per_skill_rows),
        "cells": cells,
    }

    json_path = out_dir / "predicate_firing_static.json"
    jsonl_path = out_dir / "predicate_firing_per_skill.jsonl"
    json_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n")
    _write_jsonl(jsonl_path, per_skill_rows)

    n_corpora = len(corpus_vocabs)
    n_targets = len(TARGET_DOMAINS)
    headline_rates: list[float] = []
    for cell in cells:
        if cell["source_cluster"] == "game" and cell["target_domain"] in _HEADLINE_TARGETS_FOR_GAME:
            headline_rates.append(cell["cell_max_admit_rate"])
    headline_str = (
        f"{mean(headline_rates):.2f}" if headline_rates else "n/a"
    )

    print(
        f"predicate_firing_static: {json_path},{jsonl_path} "
        f"| n_cells={n_corpora}x{n_targets}={len(cells)} "
        f"| n_per_skill_rows={len(per_skill_rows)} "
        f"| mean cell_max_admit_rate across cross-cluster "
        f"game->{{vr,video,browser,osworld}} = {headline_str}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
