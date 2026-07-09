"""Shared bank-discovery and vocabulary-extraction surface for Stage-0 audits.

Walks ``skill_bank.jsonl`` files under the canonical roots, classifies each
discovered bank by cluster (``game`` vs ``cross_domain``), and emits per-skill
and per-corpus vocabulary slices (protocol ops, slot types, predicate types)
that the three Stage-0 audit scripts consume.

Dataclass field names and function signatures are a hard contract for
sub-agents B and C; do not rename or restructure.

See ``implementation_notes/legacy/phase5-cross-domain-measurement.md`` Section 3 and
the ``audits/__init__.py`` docstring for the broader plan.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_BANK_ROOTS: tuple[Path, Path] = (
    Path("labeling/skill_bank_out"),
    Path("skill_transfer_test/skill_bank_local"),
)


RESERVED_DIR_NAMES: frozenset[str] = frozenset({
    "_unified",
    "_logs",
    "_dispatch_logs",
    "_normalized_episodes",
    "_audits",
    "reports",
    "episode_snapshots",
    "per_episode_bank_management",
})


_VALID_BANK_KINDS: frozenset[str] = frozenset({"per_sample", "per_episode"})


@dataclass(frozen=True)
class BankInfo:
    """One discovered ``skill_bank.jsonl`` file plus its provenance."""

    label: str
    path: Path
    bank_kind: str
    cluster: str
    parent_root: Path
    corpus_subdir: str


@dataclass(frozen=True)
class SkillVocab:
    """Per-skill vocabulary slices extracted from one ``{report, skill}`` envelope."""

    skill_id: str
    protocol_ops: frozenset[str]
    slot_types: frozenset[str]
    hop_predicates: frozenset[str]
    contract_predicates: frozenset[str]
    n_hops: int
    n_predicate_instances: int
    n_slot_instances: int


@dataclass
class CorpusVocab:
    """Aggregated per-corpus vocabulary (set unions + per-skill list)."""

    bank_info: BankInfo
    n_skills: int
    protocol_ops: frozenset[str]
    slot_types: frozenset[str]
    hop_predicates: frozenset[str]
    contract_predicates: frozenset[str]
    skills: list[SkillVocab] = field(default_factory=list)


def _classify_cluster(path: Path) -> str:
    parts = set(path.resolve().parts)
    if "skill_bank_out" in parts:
        return "game"
    if "skill_bank_local" in parts:
        return "cross_domain"
    warnings.warn(
        f"bank at {path} matches neither skill_bank_out nor skill_bank_local; "
        "classifying cluster as 'unknown'",
        stacklevel=2,
    )
    return "unknown"


def _walk_for_banks(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    if root.name in RESERVED_DIR_NAMES:
        return
    candidate = root / "skill_bank.jsonl"
    if candidate.is_file():
        yield candidate
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield from _walk_for_banks(child)


def discover_banks(
    roots: Iterable[Path] = DEFAULT_BANK_ROOTS,
) -> list[BankInfo]:
    """Walk ``roots`` recursively looking for ``skill_bank.jsonl`` files.

    Missing roots emit a warning and are skipped (the game-bank root is
    gitignored and frequently absent). Directories whose name appears in
    :data:`RESERVED_DIR_NAMES` are skipped.

    Returns a list sorted by ``(cluster, label)``.
    """
    found: list[BankInfo] = []
    for root in roots:
        if not root.exists():
            warnings.warn(
                f"bank root {root} does not exist; skipping",
                stacklevel=2,
            )
            continue
        for bank_path in _walk_for_banks(root):
            parent = bank_path.parent
            grandparent = parent.parent
            great_grandparent = grandparent.parent

            if parent.name in _VALID_BANK_KINDS:
                bank_kind = parent.name
                parent_root = great_grandparent
                try:
                    corpus_subdir = grandparent.relative_to(parent_root).as_posix()
                except ValueError:
                    corpus_subdir = grandparent.name
                label = grandparent.name
            else:
                warnings.warn(
                    f"bank at {bank_path} has non-standard bank_kind dir "
                    f"{parent.name!r}; expected 'per_sample' or 'per_episode'",
                    stacklevel=2,
                )
                bank_kind = parent.name
                parent_root = great_grandparent
                try:
                    corpus_subdir = parent.relative_to(parent_root).as_posix()
                except ValueError:
                    corpus_subdir = parent.name
                label = parent.name

            cluster = _classify_cluster(bank_path)
            found.append(
                BankInfo(
                    label=label,
                    path=bank_path,
                    bank_kind=bank_kind,
                    cluster=cluster,
                    parent_root=parent_root,
                    corpus_subdir=corpus_subdir,
                )
            )

    found.sort(key=lambda b: (b.cluster, b.label))
    return found


def iter_bank_records(bank_path: Path) -> Iterator[dict]:
    """Yield ``{"report": ..., "skill": ...}`` envelopes from a JSONL bank.

    Skips empty / whitespace-only lines. Re-raises ``json.JSONDecodeError``
    with ``path:line`` context on malformed lines.
    """
    with bank_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"{bank_path}:{lineno}: {exc.msg}", exc.doc, exc.pos
                ) from exc


def _collect_predicate_types(items: object) -> tuple[list[str], int]:
    """Pull predicate ``type`` strings out of an effects_add/del list.

    Bare strings are accepted (legacy shape) and treated as the type itself.
    Dicts missing ``"type"`` are skipped silently. Anything else is ignored.
    Returns ``(types, n_instances)`` where ``n_instances`` counts every
    contributing element (with duplicates).
    """
    out: list[str] = []
    n = 0
    if not isinstance(items, list):
        return out, n
    for elem in items:
        if isinstance(elem, str):
            out.append(elem)
            n += 1
        elif isinstance(elem, dict):
            t = elem.get("type")
            if isinstance(t, str):
                out.append(t)
            n += 1
    return out, n


def extract_skill_vocab(envelope: dict) -> SkillVocab:
    """Extract per-skill vocabulary from one envelope."""
    skill = envelope.get("skill") or {}
    skill_id = str(skill.get("skill_id", ""))

    protocol = skill.get("protocol") or []
    if not isinstance(protocol, list):
        protocol = []

    protocol_ops: set[str] = set()
    slot_types: set[str] = set()
    hop_predicates: set[str] = set()
    n_hops = 0
    n_pred_instances = 0
    n_slot_instances = 0

    for hop in protocol:
        if not isinstance(hop, dict):
            continue
        n_hops += 1
        op = hop.get("op")
        if isinstance(op, str) and op:
            protocol_ops.add(op)
        hop_slot_types = hop.get("slot_types") or {}
        if isinstance(hop_slot_types, dict):
            for v in hop_slot_types.values():
                if isinstance(v, str) and v:
                    slot_types.add(v)
                    n_slot_instances += 1
        for key in ("effects_add", "effects_del"):
            types, n = _collect_predicate_types(hop.get(key))
            hop_predicates.update(types)
            n_pred_instances += n

    contract = skill.get("contract") or {}
    if not isinstance(contract, dict):
        contract = {}
    contract_predicates: set[str] = set()
    for key in ("effects_add", "effects_del"):
        types, n = _collect_predicate_types(contract.get(key))
        contract_predicates.update(types)
        n_pred_instances += n

    return SkillVocab(
        skill_id=skill_id,
        protocol_ops=frozenset(protocol_ops),
        slot_types=frozenset(slot_types),
        hop_predicates=frozenset(hop_predicates),
        contract_predicates=frozenset(contract_predicates),
        n_hops=n_hops,
        n_predicate_instances=n_pred_instances,
        n_slot_instances=n_slot_instances,
    )


def collect_corpus_vocab(bank_info: BankInfo) -> CorpusVocab:
    """Aggregate per-skill vocabs into per-corpus union by reading the bank once."""
    skills: list[SkillVocab] = []
    ops: set[str] = set()
    slots: set[str] = set()
    hop_preds: set[str] = set()
    contract_preds: set[str] = set()
    for envelope in iter_bank_records(bank_info.path):
        sv = extract_skill_vocab(envelope)
        skills.append(sv)
        ops.update(sv.protocol_ops)
        slots.update(sv.slot_types)
        hop_preds.update(sv.hop_predicates)
        contract_preds.update(sv.contract_predicates)
    return CorpusVocab(
        bank_info=bank_info,
        n_skills=len(skills),
        protocol_ops=frozenset(ops),
        slot_types=frozenset(slots),
        hop_predicates=frozenset(hop_preds),
        contract_predicates=frozenset(contract_preds),
        skills=skills,
    )
