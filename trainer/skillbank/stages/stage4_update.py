"""
Stage 4: Update — refine contracts, materialize NEW clusters, merge/split skills.

Handles the bank mutation phase of the EM loop:
  - Refine: tighten contract effects based on new evidence
  - Materialize NEW: cluster NEW segments and promote qualifying clusters
  - Merge: combine skills with nearly identical contracts
  - Split: break apart skills with low pass rates into sub-skills
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from trainer.skillbank.stages.stage2_decode import DecodeResult, DecodedSegment, NEW_LABEL
from trainer.skillbank.stages.stage3_contracts import LearnedContract

logger = logging.getLogger(__name__)


@dataclass
class UpdateConfig:
    """Configuration for the update stage."""

    min_new_cluster_size: int = 5
    split_pass_rate_threshold: float = 0.7
    child_pass_rate_threshold: float = 0.8
    merge_jaccard_threshold: float = 0.85
    merge_embedding_threshold: float = 0.90
    min_child_size: int = 3
    refine_delta_threshold: float = 0.05


@dataclass
class UpdateEvent:
    """Record of a single bank update event."""

    event_type: str  # "refine" | "materialize" | "merge" | "split"
    skill_ids: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateResult:
    """Result of the update stage."""

    events: List[UpdateEvent] = field(default_factory=list)
    n_refine: int = 0
    n_materialize: int = 0
    n_merge: int = 0
    n_split: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_refine": self.n_refine,
            "n_materialize": self.n_materialize,
            "n_merge": self.n_merge,
            "n_split": self.n_split,
            "events": [{"type": e.event_type, "skills": e.skill_ids} for e in self.events],
        }


def run_update(
    decode_results: List[DecodeResult],
    contracts: Dict[str, LearnedContract],
    bank: Any,
    config: Optional[UpdateConfig] = None,
) -> UpdateResult:
    """Execute the full update stage.

    Runs refine, materialize, merge, and split in sequence.

    Args:
        decode_results: decoded segments from Stage 2
        contracts: learned contracts from Stage 3
        bank: current SkillBankMVP (mutated in place)
        config: update hyperparameters

    Returns:
        UpdateResult with event log and counts.
    """
    cfg = config or UpdateConfig()
    result = UpdateResult()

    _refine(contracts, bank, cfg, result)
    _materialize_new(decode_results, bank, cfg, result)
    _merge_similar(bank, cfg, result)
    _split_weak(contracts, bank, cfg, result)

    return result


def _refine(
    contracts: Dict[str, LearnedContract],
    bank: Any,
    cfg: UpdateConfig,
    result: UpdateResult,
) -> None:
    """Tighten existing contracts based on new evidence."""
    for skill_id, lc in contracts.items():
        if not lc.verified:
            continue

        existing = bank.get_contract(skill_id) if hasattr(bank, "get_contract") else None
        if existing is None:
            continue

        old_add = getattr(existing, "eff_add", set()) or set()
        new_add = lc.eff_add
        delta = len(old_add.symmetric_difference(new_add))
        total = max(len(old_add | new_add), 1)
        relative_delta = delta / total

        if relative_delta >= cfg.refine_delta_threshold:
            existing.eff_add = new_add
            existing.eff_del = lc.eff_del
            existing.support = lc.support
            existing.n_instances = lc.n_instances
            existing.bump_version()
            bank.add_or_update(existing)

            result.n_refine += 1
            result.events.append(UpdateEvent(
                event_type="refine",
                skill_ids=[skill_id],
                details={"delta": relative_delta, "n_instances": lc.n_instances},
            ))


def _materialize_new(
    decode_results: List[DecodeResult],
    bank: Any,
    cfg: UpdateConfig,
    result: UpdateResult,
) -> None:
    """Cluster NEW segments and promote qualifying clusters to real skills."""
    new_segments: List[DecodedSegment] = []
    for dr in decode_results:
        for seg in dr.segments:
            if seg.skill_label == NEW_LABEL:
                new_segments.append(seg)

    if len(new_segments) < cfg.min_new_cluster_size:
        return

    by_sig: Dict[str, List[DecodedSegment]] = defaultdict(list)
    for seg in new_segments:
        sig = _effect_signature(seg.eff_add, seg.eff_del)
        by_sig[sig].append(seg)

    ts = int(time.time())
    created = 0

    for sig, cluster in by_sig.items():
        if len(cluster) < cfg.min_new_cluster_size:
            continue

        new_id = f"S_new_{ts}_{created}"

        add_counts: Dict[str, int] = defaultdict(int)
        del_counts: Dict[str, int] = defaultdict(int)
        for seg in cluster:
            for p in seg.eff_add:
                add_counts[p] += 1
            for p in seg.eff_del:
                del_counts[p] += 1

        n = len(cluster)
        freq = 0.8
        consensus_add = {p for p, c in add_counts.items() if c >= freq * n}
        consensus_del = {p for p, c in del_counts.items() if c >= freq * n}

        if not consensus_add and not consensus_del:
            continue

        try:
            from skill_agents.stage3_mvp.schemas import SkillEffectsContract
            contract = SkillEffectsContract(
                skill_id=new_id,
                eff_add=consensus_add,
                eff_del=consensus_del,
                n_instances=n,
            )
            bank.add_or_update(contract)
            created += 1
            result.n_materialize += 1
            result.events.append(UpdateEvent(
                event_type="materialize",
                skill_ids=[new_id],
                details={"n_instances": n, "sig": sig},
            ))
        except Exception as exc:
            logger.warning("Failed to materialize %s: %s", new_id, exc)


def _merge_similar(bank: Any, cfg: UpdateConfig, result: UpdateResult) -> None:
    """Merge skills with nearly identical contracts."""
    skill_ids = list(getattr(bank, "skill_ids", []))
    merged_away: set = set()

    for i in range(len(skill_ids)):
        if skill_ids[i] in merged_away:
            continue
        ci = bank.get_contract(skill_ids[i])
        if ci is None:
            continue

        for j in range(i + 1, len(skill_ids)):
            if skill_ids[j] in merged_away:
                continue
            cj = bank.get_contract(skill_ids[j])
            if cj is None:
                continue

            jaccard = _contract_jaccard(ci, cj)
            if jaccard >= cfg.merge_jaccard_threshold:
                keep = skill_ids[i] if (ci.n_instances or 0) >= (cj.n_instances or 0) else skill_ids[j]
                remove = skill_ids[j] if keep == skill_ids[i] else skill_ids[i]

                bank.remove(remove)
                merged_away.add(remove)
                result.n_merge += 1
                result.events.append(UpdateEvent(
                    event_type="merge",
                    skill_ids=[keep, remove],
                    details={"jaccard": jaccard},
                ))


def _split_weak(
    contracts: Dict[str, LearnedContract],
    bank: Any,
    cfg: UpdateConfig,
    result: UpdateResult,
) -> None:
    """Split skills with pass rates below threshold (placeholder).

    Full split logic requires re-clustering segments within the skill,
    which is deferred to the full EM iteration. Here we flag candidates.
    """
    for skill_id, lc in contracts.items():
        if lc.pass_rate < cfg.split_pass_rate_threshold and lc.n_instances >= cfg.min_child_size * 2:
            result.n_split += 1
            result.events.append(UpdateEvent(
                event_type="split",
                skill_ids=[skill_id],
                details={"pass_rate": lc.pass_rate, "n_instances": lc.n_instances},
            ))
            logger.info("Split candidate: %s (pass_rate=%.2f, n=%d)",
                         skill_id, lc.pass_rate, lc.n_instances)


def _effect_signature(eff_add: Set[str], eff_del: Set[str]) -> str:
    a = ",".join(sorted(eff_add)) if eff_add else ""
    d = ",".join(sorted(eff_del)) if eff_del else ""
    return f"A:{a}|D:{d}"


def _contract_jaccard(c1: Any, c2: Any) -> float:
    s1 = (getattr(c1, "eff_add", set()) or set()) | (getattr(c1, "eff_del", set()) or set())
    s2 = (getattr(c2, "eff_add", set()) or set()) | (getattr(c2, "eff_del", set()) or set())
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)
