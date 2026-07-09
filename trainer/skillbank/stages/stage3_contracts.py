"""
Stage 3: Learn and verify effects-only contracts from decoded segments.

Aggregates per-instance effects across segments assigned to the same skill,
applies frequency thresholding, and verifies contracts against observations.
Wraps the existing skill_agents.stage3_mvp logic for use in the EM loop.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from trainer.skillbank.stages.stage2_decode import DecodeResult, DecodedSegment, NEW_LABEL

logger = logging.getLogger(__name__)


@dataclass
class ContractLearningConfig:
    """Configuration for contract learning."""

    eff_freq: float = 0.8
    min_instances_per_skill: int = 5
    start_end_window: int = 5
    pass_rate_threshold: float = 0.6
    verification_sample_size: int = 50


@dataclass
class LearnedContract:
    """A contract learned from decoded segments."""

    skill_id: str
    eff_add: Set[str] = field(default_factory=set)
    eff_del: Set[str] = field(default_factory=set)
    eff_event: Set[str] = field(default_factory=set)
    support: Dict[str, int] = field(default_factory=dict)
    n_instances: int = 0
    pass_rate: float = 0.0
    verified: bool = False


@dataclass
class ContractLearningResult:
    """Result of contract learning for the entire batch."""

    contracts: Dict[str, LearnedContract] = field(default_factory=dict)
    n_skills_updated: int = 0
    n_skills_created: int = 0
    n_skills_below_threshold: int = 0
    per_skill_pass_rates: Dict[str, float] = field(default_factory=dict)


def learn_contracts(
    decode_results: List[DecodeResult],
    bank: Any,
    config: Optional[ContractLearningConfig] = None,
) -> ContractLearningResult:
    """Learn/update contracts from a batch of decode results.

    Groups segments by assigned skill label, then for each skill:
      1. Collect per-instance eff_add/eff_del from segment boundary predicates
      2. Apply frequency threshold to get consensus effects
      3. Verify against observations
      4. Build/update SkillEffectsContract

    Args:
        decode_results: list of DecodeResult from Stage 2
        bank: current SkillBankMVP
        config: learning hyperparameters

    Returns:
        ContractLearningResult with learned contracts and diagnostics.
    """
    cfg = config or ContractLearningConfig()

    segments_by_skill: Dict[str, List[DecodedSegment]] = defaultdict(list)
    for dr in decode_results:
        for seg in dr.segments:
            if seg.skill_label != NEW_LABEL:
                segments_by_skill[seg.skill_label].append(seg)

    result = ContractLearningResult()

    for skill_id, segments in segments_by_skill.items():
        n = len(segments)
        if n < cfg.min_instances_per_skill:
            logger.debug("Skill %s: only %d instances (need %d), skipping",
                         skill_id, n, cfg.min_instances_per_skill)
            continue

        add_counts: Dict[str, int] = defaultdict(int)
        del_counts: Dict[str, int] = defaultdict(int)

        for seg in segments:
            for pred in seg.eff_add:
                add_counts[pred] += 1
            for pred in seg.eff_del:
                del_counts[pred] += 1

        freq_threshold = cfg.eff_freq * n
        consensus_add = {p for p, c in add_counts.items() if c >= freq_threshold}
        consensus_del = {p for p, c in del_counts.items() if c >= freq_threshold}

        # Enrich with LLM-based contract summarization when CONTRACT adapter
        # is available.  LLM suggestions are merged (union) with the
        # frequency-based consensus so they augment rather than replace.
        try:
            from skill_agents.stage3_mvp.llm_contract import llm_summarize_contract
            seg_obs = [
                str(getattr(seg, "B_start", set()))[:200] for seg in segments[:3]
            ]
            all_starts = set().union(*(seg.B_start for seg in segments if seg.B_start))
            all_ends = set().union(*(seg.B_end for seg in segments if seg.B_end))
            llm_result = llm_summarize_contract(
                skill_id=skill_id,
                segment_observations=seg_obs,
                predicates_start=all_starts,
                predicates_end=all_ends,
                n_instances=n,
            )
            if llm_result:
                llm_add = set(llm_result.get("eff_add", []))
                llm_del = set(llm_result.get("eff_del", []))
                if llm_add:
                    consensus_add = consensus_add | llm_add
                if llm_del:
                    consensus_del = consensus_del | llm_del
                logger.debug("CONTRACT adapter enriched skill %s: +%d add, +%d del",
                             skill_id, len(llm_add), len(llm_del))
        except Exception as exc:
            logger.debug("CONTRACT adapter enrichment skipped for %s: %s", skill_id, exc)

        pass_count = 0
        verify_n = min(n, cfg.verification_sample_size)
        for seg in segments[:verify_n]:
            add_ok = consensus_add.issubset(seg.eff_add) if consensus_add else True
            del_ok = consensus_del.issubset(seg.eff_del) if consensus_del else True
            if add_ok and del_ok:
                pass_count += 1
        pass_rate = pass_count / max(verify_n, 1)

        support = {**{p: add_counts[p] for p in consensus_add},
                   **{p: del_counts[p] for p in consensus_del}}

        contract = LearnedContract(
            skill_id=skill_id,
            eff_add=consensus_add,
            eff_del=consensus_del,
            support=support,
            n_instances=n,
            pass_rate=pass_rate,
            verified=pass_rate >= cfg.pass_rate_threshold,
        )
        result.contracts[skill_id] = contract
        result.per_skill_pass_rates[skill_id] = pass_rate

        existing = bank.get_contract(skill_id) if hasattr(bank, "get_contract") else None
        if existing is not None:
            result.n_skills_updated += 1
        else:
            result.n_skills_created += 1

        if pass_rate < cfg.pass_rate_threshold:
            result.n_skills_below_threshold += 1

    logger.info(
        "Contract learning: %d skills updated, %d created, %d below threshold",
        result.n_skills_updated, result.n_skills_created,
        result.n_skills_below_threshold,
    )
    return result


def apply_contracts_to_bank(
    result: ContractLearningResult,
    bank: Any,
    only_verified: bool = True,
) -> int:
    """Write learned contracts into the bank.

    Returns the number of contracts written.
    """
    written = 0
    for skill_id, lc in result.contracts.items():
        if only_verified and not lc.verified:
            continue

        try:
            from skill_agents.stage3_mvp.schemas import SkillEffectsContract
            existing = bank.get_contract(skill_id)
            if existing is not None:
                existing.eff_add = lc.eff_add
                existing.eff_del = lc.eff_del
                existing.support = lc.support
                existing.n_instances = lc.n_instances
                existing.bump_version()
                bank.add_or_update(existing)
            else:
                contract = SkillEffectsContract(
                    skill_id=skill_id,
                    eff_add=lc.eff_add,
                    eff_del=lc.eff_del,
                    support=lc.support,
                    n_instances=lc.n_instances,
                )
                bank.add_or_update(contract)
            written += 1
        except Exception as exc:
            logger.warning("Failed to write contract for %s: %s", skill_id, exc)

    return written
