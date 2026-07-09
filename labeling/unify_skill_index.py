#!/usr/bin/env python
"""Unified skill-index aggregator.

Walks one or more skill-extraction output roots — each root may host any
mix of env_wrappers (``labeling/extract_skillbank_gpt54.py``) and gym-v
(``labeling/extract_skillbank_gymv_gpt54.py``) per-source folders — and
emits a canonical, cross-corpus index under
``<output_dir>/_unified/``.

Per-source folder convention recognised by this aggregator
----------------------------------------------------------

A "source" is any directory with a ``skill_bank.jsonl`` and (optionally)
a sibling ``skill_catalog.json``. The folder *name* is treated as
``source_name``; if no sibling catalog is found the aggregator builds
one from the bank rows themselves so older runs without the unified
catalog still aggregate cleanly.

Layout produced
---------------

::

    <output_dir>/_unified/
    ├── skill_index.jsonl       # one row per skill (flat, RAG-ingest-ready)
    ├── skill_catalog_all.json  # grouped by corpus → source_name → skills
    └── skill_rag_index.json    # flat list with `id` / `type` / `text`
                                # fields, mirrors the env_wrappers schema

The aggregator is idempotent: re-running on the same roots overwrites
the unified outputs. It never mutates the source folders.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("labeling.unify_skill_index")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_GYMV = "gym_v"
CORPUS_ENV_WRAPPERS = "env_wrappers"
CORPUS_UNKNOWN = "unknown"

# Folders we deliberately skip when walking a root for source dirs.
_RESERVED_FOLDER_NAMES = {
    "_unified",
    "_logs",
    "_dispatch_logs",
    "_normalized_episodes",
    "reports",
    "episode_snapshots",
    "per_episode_bank_management",
}

# Canonical keys carried on every per-skill row in the unified index.
_CANONICAL_SKILL_KEYS = (
    "skill_id",
    "name",
    "summary",
    "description",
    "tag",
    "eff_add",
    "eff_del",
    "eff_event",
    "n_instances",
    "version",
)


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------


def _classify_corpus(source_dir: Path, catalog: Dict[str, Any]) -> str:
    """Return ``gym_v`` / ``env_wrappers`` / ``unknown`` for a source.

    Priority order:

    1. Explicit ``corpus`` field on the catalog (set by gym-v driver and
       by any future-canonical env_wrappers writer).
    2. Folder-name heuristic: gym-v envs are ``Temporal_<Title>-v0``.
    3. Fallback to ``unknown`` (still aggregated, just untagged).
    """
    explicit = (catalog or {}).get("corpus")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if source_dir.name.startswith("Temporal_"):
        return CORPUS_GYMV
    # env_wrappers writes ``"game"`` in its catalog header — treat that
    # as a strong signal even when ``corpus`` is missing.
    if (catalog or {}).get("game"):
        return CORPUS_ENV_WRAPPERS
    return CORPUS_UNKNOWN


def _source_name_from_catalog(source_dir: Path, catalog: Dict[str, Any]) -> str:
    cat = catalog or {}
    for key in ("source_name", "game", "env"):
        v = cat.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return source_dir.name


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------


def _load_bank_rows(bank_path: Path) -> List[Dict[str, Any]]:
    """Read a ``skill_bank.jsonl`` produced by ``SkillBankMVP.save()``."""
    rows: List[Dict[str, Any]] = []
    if not bank_path.exists():
        return rows
    with open(bank_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                logger.warning("bad row in %s: %s", bank_path, exc)
    return rows


def _bank_row_to_catalog_entry(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Project a ``SkillBankMVP`` row down to the canonical catalog entry."""
    skill = row.get("skill") or {}
    if not skill:
        return None
    sid = skill.get("skill_id")
    if not sid:
        return None
    if skill.get("retired"):
        return None
    contract = skill.get("contract") or {}
    eff_add = sorted(contract.get("eff_add") or [])
    eff_del = sorted(contract.get("eff_del") or [])
    eff_event = sorted(contract.get("eff_event") or [])
    tags = skill.get("tags") or []
    summary = (
        skill.get("strategic_description")
        or skill.get("summary")
        or skill.get("description")
        or ""
    )
    return {
        "skill_id": sid,
        "name": skill.get("name") or sid,
        "summary": summary,
        "description": skill.get("strategic_description") or summary,
        "tag": tags[0] if tags else "",
        "eff_add": eff_add,
        "eff_del": eff_del,
        "eff_event": eff_event,
        "n_instances": int(contract.get("n_instances") or skill.get("n_instances") or 0),
        "version": int(contract.get("version") or skill.get("version") or 1),
    }


def _normalize_catalog_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a catalog entry to exactly the canonical key set."""
    out: Dict[str, Any] = {k: raw.get(k, "" if k != "n_instances" and k != "version" else 0) for k in _CANONICAL_SKILL_KEYS}
    # Guard list fields against None.
    for k in ("eff_add", "eff_del", "eff_event"):
        v = raw.get(k)
        out[k] = list(v) if isinstance(v, (list, tuple)) else []
    # Guard numeric fields.
    out["n_instances"] = int(raw.get("n_instances") or 0)
    out["version"] = int(raw.get("version") or 1)
    return out


def load_source(source_dir: Path) -> Optional[Tuple[str, str, List[Dict[str, Any]]]]:
    """Load a single per-source folder.

    Returns ``(corpus, source_name, [canonical_entries])`` or ``None`` if
    the folder has no usable bank.
    """
    bank_path = source_dir / "skill_bank.jsonl"
    if not bank_path.exists():
        return None

    catalog_path = source_dir / "skill_catalog.json"
    catalog: Dict[str, Any] = {}
    if catalog_path.exists():
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f) or {}
        except Exception as exc:
            logger.warning("bad catalog at %s: %s", catalog_path, exc)
            catalog = {}

    corpus = _classify_corpus(source_dir, catalog)
    source_name = _source_name_from_catalog(source_dir, catalog)

    entries: List[Dict[str, Any]] = []
    raw_entries = catalog.get("skills") if isinstance(catalog, dict) else None
    if isinstance(raw_entries, list) and raw_entries:
        entries = [_normalize_catalog_entry(e) for e in raw_entries if isinstance(e, dict)]
    else:
        # Catalog missing or empty → reconstruct from the bank rows.
        for row in _load_bank_rows(bank_path):
            entry = _bank_row_to_catalog_entry(row)
            if entry is not None:
                entries.append(entry)

    return corpus, source_name, entries


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_sources(roots: Iterable[Path]) -> List[Path]:
    """Find every per-source folder under any of ``roots``.

    A "source folder" is any directory containing ``skill_bank.jsonl``
    that is not in the reserved-name list (so we don't pick up
    per-episode snapshot dirs that happen to carry a snapshot bank).
    """
    found: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            logger.warning("root does not exist: %s", root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune reserved dir names so we never descend into snapshot trees.
            dirnames[:] = [d for d in dirnames if d not in _RESERVED_FOLDER_NAMES]
            if "skill_bank.jsonl" not in filenames:
                continue
            p = Path(dirpath)
            if p.name in _RESERVED_FOLDER_NAMES:
                continue
            # Skip per-episode snapshot banks even if pruning missed them
            # (e.g. the user passed the snapshot folder itself as a root).
            if any(part in _RESERVED_FOLDER_NAMES for part in p.parts):
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(p)
    return sorted(found)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def unify_roots(
    roots: List[Path],
    output_dir: Path,
    *,
    pipeline: str = "skill_agents",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Walk ``roots`` and emit unified outputs under ``output_dir/_unified``.

    Returns a small summary dict (counts + paths).
    """
    sources = discover_sources(roots)
    if verbose:
        for p in sources:
            print(f"  source: {p}")

    unified_dir = output_dir / "_unified"
    unified_dir.mkdir(parents=True, exist_ok=True)

    flat_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    rag_entries: List[Dict[str, Any]] = []
    n_total_skills = 0

    for src in sources:
        loaded = load_source(src)
        if loaded is None:
            continue
        corpus, source_name, entries = loaded
        n_total_skills += len(entries)

        grouped.setdefault(corpus, {}).setdefault(source_name, []).extend(entries)

        for entry in entries:
            row = dict(entry)
            row["corpus"] = corpus
            row["source_name"] = source_name
            row["source_path"] = str(src)
            flat_rows.append(row)

            rag_text_parts = [
                f"corpus={corpus}",
                f"source={source_name}",
                f"skill={entry.get('name') or entry.get('skill_id')}",
            ]
            if entry.get("tag"):
                rag_text_parts.append(f"tag={entry['tag']}")
            if entry.get("eff_add"):
                rag_text_parts.append(f"eff_add={','.join(entry['eff_add'])}")
            if entry.get("eff_event"):
                rag_text_parts.append(f"eff_event={','.join(entry['eff_event'])}")
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
        "n_sources": len(sources),
        "n_skills": n_total_skills,
        "n_corpora": len(grouped),
        "roots": [str(r) for r in roots],
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
        "n_sources": len(sources),
        "n_skills": n_total_skills,
        "n_corpora": len(grouped),
        "skill_index_path": str(flat_path),
        "skill_catalog_all_path": str(catalog_all_path),
        "skill_rag_index_path": str(rag_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-source skill banks (env_wrappers + gym-v) into "
            "a unified, corpus-tagged index suitable for cross-domain RAG."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--root", action="append", required=True, type=Path,
        help="One or more skill-extraction output roots. Repeat the flag "
             "to merge multiple roots (e.g. labeling/output/gpt54_skillbank "
             "and skill_bank_sft).",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Where to write the _unified/ tree. Defaults to the FIRST "
             "--root.",
    )
    parser.add_argument(
        "--pipeline", type=str, default="skill_agents",
        help="Tag stamped on the grouped catalog (default: skill_agents).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s %(name)s: %(message)s")

    roots: List[Path] = [r.resolve() for r in args.root]
    output_dir = (args.output_dir or roots[0]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  labeling/unify_skill_index — cross-corpus aggregator")
    print("=" * 62)
    for r in roots:
        print(f"  Root          : {r}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Pipeline tag  : {args.pipeline}")
    print()

    result = unify_roots(
        roots=roots, output_dir=output_dir,
        pipeline=args.pipeline, verbose=args.verbose,
    )

    print()
    print("-" * 62)
    print(f"  Sources scanned : {result['n_sources']}")
    print(f"  Skills indexed  : {result['n_skills']}")
    print(f"  Corpora seen    : {result['n_corpora']}")
    print(f"  skill_index     : {result['skill_index_path']}")
    print(f"  skill_catalog   : {result['skill_catalog_all_path']}")
    print(f"  skill_rag_index : {result['skill_rag_index_path']}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
