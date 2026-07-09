"""Unified skill index for cross-corpus banks (closes TODO-2).

Walks ``<output_root>/<corpus>/<bank_kind>/skill_bank.jsonl`` for the six
cross-domain corpora (browsergym, osworld, visual_toolbench, tir_bench,
video_holmes, siv_bench) and emits a corpus-tagged unified index under
``<output_root>/_unified/``::

    <output_root>/_unified/
    +-- skill_index.jsonl       # one row per skill (flat, RAG-ingest-ready)
    +-- skill_catalog_all.json  # grouped by corpus -> bank_kind -> skills
    +-- skill_rag_index.json    # flat list with id / type / text fields

This is the cross-domain analogue of
:mod:`labeling.unify_skill_index` (which targets the legacy
``skill_bank_out/<run>/{env_wrappers,gym_v}/<source>/skill_bank.jsonl``
layout). The cross-domain layout is fundamentally different -- per-corpus
folders containing per-bank-kind sub-dirs, with each row using
``contract.effects_add`` / ``contract.effects_del`` (typed predicate
dicts) rather than the legacy ``contract.eff_add`` / ``contract.eff_del``
(flat string lists). We can't reuse :func:`labeling.unify_skill_index.unify_roots`
directly because its row->entry projection assumes the legacy keys.

Run as::

    python -m skill_transfer_test.extract._unify \\
        --output-root skill_transfer_test/skill_bank_local/<run_id>

Idempotent: re-running on the same root overwrites the unified outputs.
Never mutates the per-corpus banks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger("skill_transfer_test.extract._unify")

__all__ = [
    "CROSS_DOMAIN_CORPORA",
    "discover_banks",
    "load_bank",
    "unify_root",
    "main",
]

# The six cross-domain corpora the skill_transfer_test/extract/ pipeline
# emits (mirrors `_corpus_specs._SPECS`). Listed here verbatim so a
# misconfigured `_corpus_specs` doesn't silently exclude a corpus from
# the unified index.
CROSS_DOMAIN_CORPORA: tuple[str, ...] = (
    "browsergym",
    "osworld",
    "visual_toolbench",
    "tir_bench",
    "video_holmes",
    "siv_bench",
)

# Bank-kind folder names produced by `runner.py` per corpus. Keep this
# small and explicit so we never pick up a stray ``_normalized_episodes/``
# or other transient dir as a "bank kind".
_BANK_KINDS: tuple[str, ...] = ("per_sample", "per_episode", "archetype")


def discover_banks(
    output_root: Path,
    *,
    corpora: Iterable[str] = CROSS_DOMAIN_CORPORA,
) -> List[tuple[str, str, Path]]:
    """Walk ``<output_root>/<corpus>/<bank_kind>/skill_bank.jsonl``.

    Returns a sorted list of ``(corpus, bank_kind, bank_path)`` triples.
    Missing corpora / bank-kinds are skipped silently -- not every corpus
    ships every bank-kind (e.g. ``browsergym`` ships ``per_episode``,
    ``visual_toolbench`` ships ``per_sample`` and ``archetype``).
    """
    output_root = Path(output_root)
    if not output_root.exists():
        return []
    found: List[tuple[str, str, Path]] = []
    for corpus in corpora:
        corpus_dir = output_root / corpus
        if not corpus_dir.is_dir():
            continue
        for kind in _BANK_KINDS:
            bank = corpus_dir / kind / "skill_bank.jsonl"
            if bank.exists():
                found.append((corpus, kind, bank))
    return sorted(found, key=lambda t: (t[0], t[1]))


def _stable_skill_id(name: str, corpus: str, bank_kind: str, idx: int) -> str:
    """Derive a stable, content-addressed skill_id for a cross-domain row.

    The cross-domain banks don't carry a ``skill.skill_id`` field
    (legacy game banks do). We mint one from a SHA1 of the row's
    coordinates so the unified index is round-trippable across re-runs
    of `_unify` on the same bank.
    """
    payload = f"{corpus}|{bank_kind}|{idx}|{name}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{corpus}.{bank_kind}.{digest}"


def _project_predicates(typed_predicates: Any) -> List[str]:
    """Extract the ``.type`` from each typed predicate dict, sorted+dedup."""
    if not isinstance(typed_predicates, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for pred in typed_predicates:
        if not isinstance(pred, dict):
            continue
        t = pred.get("type")
        if isinstance(t, str) and t and t not in seen:
            seen.add(t)
            out.append(t)
    return sorted(out)


def _row_to_entry(
    row: Dict[str, Any],
    *,
    corpus: str,
    bank_kind: str,
    idx: int,
) -> Dict[str, Any] | None:
    """Project one ``{report, skill}`` envelope to a canonical catalog entry."""
    skill = row.get("skill") or {}
    if not skill:
        return None
    name = skill.get("name") or ""
    if not name:
        return None
    provenance = skill.get("provenance") or {}
    contract = skill.get("contract") or {}
    eff_add = _project_predicates(contract.get("effects_add"))
    eff_del = _project_predicates(contract.get("effects_del"))
    skill_id = _stable_skill_id(name, corpus, bank_kind, idx)
    summary = skill.get("execution_hint") or skill.get("name") or ""
    return {
        "skill_id": skill_id,
        "name": name,
        "summary": summary,
        "description": skill.get("execution_hint") or summary,
        "tag": (provenance.get("modality") or ""),
        "eff_add": eff_add,
        "eff_del": eff_del,
        "eff_event": [],
        "n_instances": int(skill.get("n_instances") or 0),
        # `aggregator_version` is a string label like "v1" / "v4" in the
        # cross-domain banks. Strip the leading 'v' if present, otherwise
        # fall back to 1. This keeps the unified schema's int field
        # contract intact across legacy + cross-domain rows.
        "version": _coerce_version(provenance.get("aggregator_version")),
    }


def _coerce_version(raw: Any) -> int:
    if raw is None:
        return 1
    if isinstance(raw, int):
        return raw
    s = str(raw).strip().lstrip("vV")
    try:
        return int(s)
    except ValueError:
        return 1


def load_bank(
    bank_path: Path,
    *,
    corpus: str,
    bank_kind: str,
) -> List[Dict[str, Any]]:
    """Read a cross-domain ``skill_bank.jsonl`` and return canonical entries."""
    entries: List[Dict[str, Any]] = []
    if not bank_path.exists():
        return entries
    with open(bank_path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("bad row %d in %s: %s", idx, bank_path, exc)
                continue
            entry = _row_to_entry(row, corpus=corpus, bank_kind=bank_kind, idx=idx)
            if entry is not None:
                entries.append(entry)
    return entries


def unify_root(
    output_root: Path,
    *,
    corpora: Iterable[str] = CROSS_DOMAIN_CORPORA,
    pipeline: str = "skill_transfer_test.extract",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Walk ``output_root`` and emit ``_unified/`` outputs. Returns a summary."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    banks = discover_banks(output_root, corpora=corpora)
    if verbose:
        for c, k, p in banks:
            print(f"  bank: corpus={c} kind={k} path={p}")

    unified_dir = output_root / "_unified"
    unified_dir.mkdir(parents=True, exist_ok=True)

    flat_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    rag_entries: List[Dict[str, Any]] = []

    for corpus, bank_kind, bank_path in banks:
        entries = load_bank(bank_path, corpus=corpus, bank_kind=bank_kind)
        source_name = f"{corpus}__{bank_kind}"
        grouped.setdefault(corpus, {}).setdefault(source_name, []).extend(entries)
        for entry in entries:
            row = dict(entry)
            row["corpus"] = corpus
            row["source_name"] = source_name
            row["source_path"] = str(bank_path)
            row["bank_kind"] = bank_kind
            flat_rows.append(row)

            rag_text_parts = [
                f"corpus={corpus}",
                f"bank_kind={bank_kind}",
                f"name={entry.get('name')}",
            ]
            if entry.get("eff_add"):
                rag_text_parts.append(f"eff_add={','.join(entry['eff_add'])}")
            rag_entries.append({
                "id": f"{corpus}/{source_name}/{entry['skill_id']}",
                "type": "skill",
                "corpus": corpus,
                "source_name": source_name,
                "skill_id": entry["skill_id"],
                "name": entry.get("name") or entry["skill_id"],
                "tag": entry.get("tag", ""),
                "text": entry.get("summary") or " | ".join(rag_text_parts),
                "description": entry.get("description") or "",
            })

    timestamp = datetime.now().isoformat()
    flat_path = unified_dir / "skill_index.jsonl"
    with open(flat_path, "w", encoding="utf-8") as f:
        for row in flat_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    grouped_payload = {
        "timestamp": timestamp,
        "pipeline": pipeline,
        "n_banks": len(banks),
        "n_skills": len(flat_rows),
        "n_corpora": len(grouped),
        "output_root": str(output_root),
        "corpora": grouped,
    }
    catalog_all_path = unified_dir / "skill_catalog_all.json"
    with open(catalog_all_path, "w", encoding="utf-8") as f:
        json.dump(grouped_payload, f, indent=2, ensure_ascii=False, default=str)
    rag_payload = {
        "timestamp": timestamp,
        "n_entries": len(rag_entries),
        "entries": rag_entries,
    }
    rag_path = unified_dir / "skill_rag_index.json"
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(rag_payload, f, indent=2, ensure_ascii=False, default=str)

    return {
        "n_banks": len(banks),
        "n_skills": len(flat_rows),
        "n_corpora": len(grouped),
        "skill_index_path": str(flat_path),
        "skill_catalog_all_path": str(catalog_all_path),
        "skill_rag_index_path": str(rag_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="run-id root, e.g. skill_transfer_test/skill_bank_local/<run_id>",
    )
    parser.add_argument(
        "--corpora", nargs="+", default=list(CROSS_DOMAIN_CORPORA),
        help=f"subset of {list(CROSS_DOMAIN_CORPORA)} (default: all 6)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    summary = unify_root(
        args.output_root, corpora=args.corpora, verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
