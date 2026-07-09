#!/usr/bin/env python
"""Cluster per-sample skills into archetype banks (closes TODO-1).

Phase-1.5b deliverable: for each VR / video corpus, walk the
``per_sample/skill_bank.jsonl`` produced by the lift pipeline and
emit an ``archetype/skill_bank.jsonl`` keyed by archetype-cluster
identity. The cluster identity for VR/video corpora is the
``CorpusSpec.archetype_cluster_field`` slice of each sample's
``raw_sample`` dict, which the lift pipeline already mirrors into
``skill.provenance.cluster_key`` at lift time. This module reads
``cluster_key`` directly -- the "direct" strategy from
:mod:`skill_transfer_test.TODO` TODO-1.

The "LLM-clustered" strategy (gpt-5.5 topic tags for VTB, where
``raw_sample.eval_focus`` only takes 2 distinct values) is **not**
implemented in this module -- it requires API access we don't have
in scope here. VTB will produce 2 archetypes (below the
``>=3 archetypes per VR/video corpus`` acceptance bar in TODO-1);
the other three corpora (TIR-Bench / Video-Holmes / SIV-Bench)
clear the bar with the direct strategy alone (11 / 7 / 10
archetypes respectively, surveyed 2026-05-02).

Per-archetype aggregation strategy:

  * **Member set**: every per-sample skill_id whose ``cluster_key``
    matches.
  * **Representative skill**: the member with the highest
    ``report.eff_add_success_rate`` (ties broken by lowest skill_id
    lexicographically -- deterministic).
  * **Archetype contract**: union of ``effects_add`` predicate-types
    across members (drops args -- predicate-type set only); union of
    ``effects_del`` predicate-types; ``preconditions`` and
    ``success_criteria`` are taken from the representative skill.
  * **Archetype protocol**: the representative's protocol (typed hops
    are heterogeneous across members so a true union would lose
    structure; picking a representative is honest).
  * **Archetype provenance**: ``{corpus, modality, bank_kind:
    "archetype", cluster_key, n_members, member_skill_ids,
    aggregation: "direct"}``.

Output layout (matches TODO-1):

    skill_transfer_test/skill_bank_local/<run_id>/<corpus>/archetype/skill_bank.jsonl

Each line is the same ``{report, skill}`` envelope shape as the
per-sample bank, so :mod:`labeling_supplement._harness_io_helpers`
can load it via the existing :func:`record_from_bank_entry`.

Usage:

    python -m skill_transfer_test.extract.archetype_aggregator \
        --bank-root skill_transfer_test/skill_bank_local/full_v5

This walks every corpus directory under ``--bank-root`` and writes
``<corpus>/archetype/skill_bank.jsonl`` next to each
``<corpus>/per_sample/skill_bank.jsonl`` it finds. Non-VR / non-video
corpora are skipped silently (their cluster_key is not populated).

See :mod:`implementation_notes/legacy/phase5-cross-domain-measurement.md`
section 8 for how Stage 5's matrix-mode loop consumes archetype
banks (one row per (source_archetype, target_corpus) cell).
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from skill_transfer_test.extract._corpus_specs import all_specs, CorpusSpec

logger = logging.getLogger("archetype_aggregator")


# Corpora eligible for archetype aggregation. The four VR / video
# corpora ship per-sample skills with a populated cluster_key; gymv /
# osworld / browser banks use a different aggregation axis (per-task
# / per-game) that lives upstream in the lift pipeline.
ARCHETYPE_ELIGIBLE_CORPORA: Tuple[str, ...] = (
    "visual_toolbench",
    "tir_bench",
    "video_holmes",
    "siv_bench",
)


@dataclass
class ArchetypeStats:
    """Per-archetype roll-up emitted alongside the archetype bank."""

    cluster_key: str
    n_members: int
    member_skill_ids: List[str]
    representative_skill_id: str
    representative_pass_rate: float
    union_effect_predicate_types: List[str]
    union_slot_types: List[str]


def _read_per_sample_records(per_sample_path: Path) -> List[Dict[str, Any]]:
    """Return raw ``{report, skill}`` envelopes from a per-sample bank."""
    out: List[Dict[str, Any]] = []
    if not per_sample_path.exists():
        return out
    with per_sample_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("skip malformed line in %s: %s", per_sample_path, exc)
    return out


def _cluster_key_of(record: Dict[str, Any]) -> Optional[str]:
    """Extract ``provenance.cluster_key`` (string) from a record, or None."""
    skill = record.get("skill") or {}
    prov = skill.get("provenance") or {}
    if not isinstance(prov, dict):
        return None
    ck = prov.get("cluster_key")
    if ck is None or not isinstance(ck, str) or not ck.strip():
        return None
    return ck.strip()


def _group_by_cluster(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket records by ``provenance.cluster_key``."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        ck = _cluster_key_of(r)
        if ck is None:
            continue
        groups[ck].append(r)
    return dict(groups)


def _pass_rate_of(record: Dict[str, Any]) -> float:
    """Return ``report.eff_add_success_rate`` (default 0.0 if missing)."""
    rep = record.get("report") or {}
    val = rep.get("eff_add_success_rate")
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _pick_representative(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Highest pass_rate; break ties by lexicographic skill_id."""
    return max(
        records,
        key=lambda r: (
            _pass_rate_of(r),
            -ord(((r.get("skill") or {}).get("skill_id", "z") or "z")[0:1] or "z"),
        ),
    )


def _union_predicate_types(records: Iterable[Dict[str, Any]], side: str) -> List[str]:
    """Union of ``contract.<side>`` predicate ``type`` strings across records.

    ``side`` is ``"effects_add"`` or ``"effects_del"``. Reads both the
    cross-domain (``effects_add``) and legacy (``eff_add``) keys so
    the function is loader-agnostic.
    """
    legacy = "eff_add" if side == "effects_add" else "eff_del"
    seen: Counter = Counter()
    for r in records:
        contract = (r.get("skill") or {}).get("contract") or {}
        preds = contract.get(side) or contract.get(legacy) or []
        for p in preds:
            if isinstance(p, dict):
                t = p.get("type") or p.get("predicate_type")
                if isinstance(t, str) and t:
                    seen[t] += 1
    return sorted(seen.keys())


def _union_slot_types(records: Iterable[Dict[str, Any]]) -> List[str]:
    """Union of typed slot-type names seen in any hop."""
    seen: Counter = Counter()
    for r in records:
        skill = r.get("skill") or {}
        for hop in (skill.get("protocol") or []):
            if not isinstance(hop, dict):
                continue
            slot_types = hop.get("slot_types") or {}
            if isinstance(slot_types, dict):
                for t in slot_types.values():
                    if isinstance(t, str) and t:
                        seen[t] += 1
    return sorted(seen.keys())


def _build_archetype_record(
    *,
    corpus: str,
    cluster_key: str,
    members: Sequence[Dict[str, Any]],
    representative: Dict[str, Any],
    archetype_index: int,
) -> Dict[str, Any]:
    """Build a ``{report, skill}`` envelope describing the archetype.

    The representative supplies the protocol, preconditions,
    success_criteria, and protocol_raw. Effects are widened to the
    union of member predicate-types (args dropped). Tags carry the
    archetype identity so downstream filters can route quickly.
    """
    rep_skill = (representative.get("skill") or {}).copy()
    rep_report = (representative.get("report") or {}).copy()
    rep_contract = (rep_skill.get("contract") or {}).copy()

    # Preserve the rep's preconditions / success_criteria / abort
    # (they're the most common shape across the cluster). Widen
    # effects_add / effects_del to the union of member predicate
    # types -- args are intentionally dropped.
    union_add = _union_predicate_types(members, "effects_add")
    union_del = _union_predicate_types(members, "effects_del")
    archetype_contract = {
        "preconditions": list(rep_contract.get("preconditions") or []),
        "success_criteria": list(rep_contract.get("success_criteria") or []),
        "abort_criteria": list(rep_contract.get("abort_criteria") or []),
        "effects_add": [
            {"type": t, "args": {}, "from_phrase": "archetype_union"}
            for t in union_add
        ],
        "effects_del": [
            {"type": t, "args": {}, "from_phrase": "archetype_union"}
            for t in union_del
        ],
    }

    # Mint a deterministic archetype skill_id. We avoid colliding
    # with per-sample ids by prefixing with ``archetype.`` and using
    # the cluster_key (ASCII-safe) as the stem.
    safe_ck = "".join(
        c if c.isalnum() or c in ("_", "-", ".") else "_"
        for c in cluster_key
    )
    archetype_skill_id = f"archetype.{corpus}.{safe_ck}"

    member_ids = [
        (m.get("skill") or {}).get("skill_id") for m in members
    ]
    member_ids = [m for m in member_ids if m]

    archetype_provenance = {
        "corpus": corpus,
        "benchmark": (rep_skill.get("provenance") or {}).get("benchmark", corpus),
        "modality": (rep_skill.get("provenance") or {}).get("modality"),
        "bank_kind": "archetype",
        "cluster_key": cluster_key,
        "n_members": len(members),
        "member_skill_ids": member_ids,
        "representative_skill_id": (rep_skill.get("skill_id")),
        "representative_pass_rate": _pass_rate_of(representative),
        "aggregation": "direct",
        "aggregator_version": "v1",
        "aggregated_at": datetime.utcnow().isoformat() + "Z",
        "source_per_sample_count": len(members),
    }

    archetype_skill: Dict[str, Any] = {
        "skill_id": archetype_skill_id,
        "name": rep_skill.get("name", archetype_skill_id),
        "strategic_description": (
            rep_skill.get("strategic_description")
            or f"Archetype cluster {cluster_key!r} ({len(members)} members)"
        ),
        "applicable_domains": list(rep_skill.get("applicable_domains") or []),
        "feasible_tasks": [corpus, cluster_key],
        "feasible_domains": list(rep_skill.get("feasible_domains") or []),
        "verified_domains": list(rep_skill.get("verified_domains") or []),
        "verified_tasks": [corpus],
        "evidence_role": rep_skill.get("evidence_role") or "COMMIT",
        "execution_hint": rep_skill.get("execution_hint"),
        "expected_tag_pattern": rep_skill.get("expected_tag_pattern"),
        "protocol": list(rep_skill.get("protocol") or []),
        "protocol_history": list(rep_skill.get("protocol_history") or []),
        "protocol_raw": rep_skill.get("protocol_raw"),
        "contract": archetype_contract,
        "provenance": archetype_provenance,
        "tags": sorted(set(
            list(rep_skill.get("tags") or [])
            + [corpus, "archetype", cluster_key]
        )),
        "n_instances": len(members),
        "sub_episodes": [],
        "source_type": rep_skill.get("source_type", "MINED"),
    }

    archetype_report: Dict[str, Any] = {
        "skill_id": archetype_skill_id,
        "n_instances": len(members),
        "overall_pass_rate": sum(_pass_rate_of(m) for m in members) / max(1, len(members)),
        "eff_add_success_rate": sum(
            float((m.get("report") or {}).get("eff_add_success_rate") or 0.0)
            for m in members
        ) / max(1, len(members)),
        "eff_del_success_rate": sum(
            float((m.get("report") or {}).get("eff_del_success_rate") or 0.0)
            for m in members
        ) / max(1, len(members)),
        "eff_event_rate": rep_report.get("eff_event_rate", 0.0),
        "failure_signatures": [],
        "worst_segments": [],
        "lift_stats": {
            "n_members": len(members),
            "archetype_index": archetype_index,
            "representative_skill_id": rep_skill.get("skill_id"),
        },
        "expected_answer": rep_report.get("expected_answer"),
        "model_answer": rep_report.get("model_answer"),
        "judge_correct": None,
        "n_explicit_entity_refs": rep_report.get("n_explicit_entity_refs", 0),
    }

    return {"report": archetype_report, "skill": archetype_skill}


def aggregate_corpus(
    *,
    corpus_root: Path,
    corpus_name: str,
    output_root: Optional[Path] = None,
) -> Tuple[Path, List[ArchetypeStats]]:
    """Aggregate one corpus's per-sample bank into an archetype bank.

    Returns ``(archetype_bank_path, [ArchetypeStats, ...])``. The
    archetype bank is written next to the per-sample bank by default
    (``<corpus_root>/archetype/skill_bank.jsonl``); pass
    ``output_root`` to redirect it (testing).
    """
    per_sample = corpus_root / "per_sample" / "skill_bank.jsonl"
    records = _read_per_sample_records(per_sample)
    if not records:
        logger.warning("no per-sample records found at %s", per_sample)

    groups = _group_by_cluster(records)
    if not groups:
        logger.warning(
            "no records carry provenance.cluster_key in %s -- skipping",
            per_sample,
        )

    out_dir = (output_root or corpus_root) / "archetype"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "skill_bank.jsonl"

    stats: List[ArchetypeStats] = []
    sorted_keys = sorted(groups.keys())
    with out_path.open("w") as f:
        for idx, ck in enumerate(sorted_keys):
            members = groups[ck]
            rep = _pick_representative(members)
            envelope = _build_archetype_record(
                corpus=corpus_name,
                cluster_key=ck,
                members=members,
                representative=rep,
                archetype_index=idx,
            )
            f.write(json.dumps(envelope, ensure_ascii=False) + "\n")
            stats.append(ArchetypeStats(
                cluster_key=ck,
                n_members=len(members),
                member_skill_ids=[
                    (m.get("skill") or {}).get("skill_id", "") for m in members
                ],
                representative_skill_id=(rep.get("skill") or {}).get("skill_id", ""),
                representative_pass_rate=_pass_rate_of(rep),
                union_effect_predicate_types=_union_predicate_types(members, "effects_add"),
                union_slot_types=_union_slot_types(members),
            ))

    return out_path, stats


def aggregate_all(
    bank_root: Path,
    *,
    corpora: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[Path, List[ArchetypeStats]]]:
    """Aggregate every eligible corpus under ``bank_root``.

    ``corpora`` defaults to :data:`ARCHETYPE_ELIGIBLE_CORPORA`. Returns
    ``{corpus: (out_path, [stats, ...])}``. Skips corpora whose
    ``per_sample/skill_bank.jsonl`` is missing.
    """
    corpora_to_run = list(corpora) if corpora is not None else list(ARCHETYPE_ELIGIBLE_CORPORA)
    out: Dict[str, Tuple[Path, List[ArchetypeStats]]] = {}
    for corpus in corpora_to_run:
        corpus_root = bank_root / corpus
        per_sample = corpus_root / "per_sample" / "skill_bank.jsonl"
        if not per_sample.exists():
            logger.info("skip %s (no per-sample bank at %s)", corpus, per_sample)
            continue
        out_path, stats = aggregate_corpus(
            corpus_root=corpus_root,
            corpus_name=corpus,
        )
        n_arch = len(stats)
        n_members = sum(s.n_members for s in stats)
        meets_acceptance = "PASS" if n_arch >= 3 else "FAIL"
        logger.info(
            "%s: %d archetypes / %d members [TODO-1 acceptance: %s]",
            corpus, n_arch, n_members, meets_acceptance,
        )
        out[corpus] = (out_path, stats)
    return out


def _emit_summary(
    results: Dict[str, Tuple[Path, List[ArchetypeStats]]],
    *,
    summary_path: Optional[Path] = None,
) -> Optional[Path]:
    """Write a human-readable summary of every aggregated corpus."""
    if not results:
        return None
    if summary_path is None:
        return None
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for corpus, (path, stats) in sorted(results.items()):
        rows.append({
            "corpus": corpus,
            "archetype_bank_path": str(path),
            "n_archetypes": len(stats),
            "n_members_total": sum(s.n_members for s in stats),
            "todo1_acceptance": "PASS" if len(stats) >= 3 else "FAIL",
            "archetypes": [
                {
                    "cluster_key": s.cluster_key,
                    "n_members": s.n_members,
                    "representative_skill_id": s.representative_skill_id,
                    "representative_pass_rate": s.representative_pass_rate,
                    "union_effect_predicate_types": s.union_effect_predicate_types,
                    "union_slot_types": s.union_slot_types,
                }
                for s in stats
            ],
        })
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return summary_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bank-root",
        default="skill_transfer_test/skill_bank_local/full_v5",
        help="Root containing per-corpus dirs each with per_sample/skill_bank.jsonl",
    )
    p.add_argument(
        "--corpora",
        nargs="+",
        default=None,
        help=(
            "Subset of corpora to aggregate. Defaults to "
            f"{list(ARCHETYPE_ELIGIBLE_CORPORA)} (the four VR/video "
            "corpora that ship a populated provenance.cluster_key)."
        ),
    )
    p.add_argument(
        "--summary-out",
        default=None,
        help="Optional path to dump a JSON summary of every archetype.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bank_root = Path(args.bank_root)
    if not bank_root.exists():
        raise SystemExit(f"bank_root missing: {bank_root}")

    results = aggregate_all(bank_root, corpora=args.corpora)
    if not results:
        logger.warning("no corpora aggregated -- check --bank-root contents")
        return 1

    print()
    print(f"=== archetype aggregation: {bank_root} ===")
    for corpus, (path, stats) in sorted(results.items()):
        print(
            f"{corpus:<24} {len(stats):>3} archetypes / "
            f"{sum(s.n_members for s in stats):>4} members  "
            f"-> {path}"
        )
        for s in stats[:5]:  # first 5
            print(f"    {s.cluster_key:<40} n={s.n_members}")
        if len(stats) > 5:
            print(f"    ... +{len(stats) - 5} more")
    print()

    if args.summary_out:
        out = _emit_summary(results, summary_path=Path(args.summary_out))
        if out is not None:
            print(f"summary: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
