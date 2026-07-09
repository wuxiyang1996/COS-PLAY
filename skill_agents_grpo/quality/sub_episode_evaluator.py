"""
Sub-episode quality evaluation, drift detection, and protocol synthesis.

Stage 4.5 in the SkillBankAgent pipeline: after contract learning (Stage 3)
and before bank maintenance (Stage 4), this module inspects sub-episode
**pointers** (``SubEpisodeRef``) attached to each skill and makes
aggregate / update / drop decisions.

SubEpisodeRef carries only lightweight metadata (summary, intention_tags,
cumulative_reward, outcome) — the actual Experience data stays in rollout
storage.  All quality scoring here operates on those cached fields without
needing to load full trajectories.

Segment quality dimensions (per sub-episode):
  - episode_credit:  min-max normalized cumulative reward from the game
  - local_progress:  success/partial/failure outcome + follow-through
  - seg_validity:    tag consistency + compactness (segmentation quality)
  - contract_valid:  does this segment's skill have a passing contract?
  - novelty_bonus:   extra credit for new skills that pass validity gates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from skill_agents_grpo.stage3_mvp.schemas import (
    Protocol,
    Skill,
    SubEpisodeRef,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SegmentQualityWeights:
    """Configurable weights for segment (sub-episode) quality scoring."""

    episode_credit: float = 0.30
    local_progress: float = 0.25
    seg_validity: float = 0.20
    contract_validity: float = 0.15
    novelty_bonus: float = 0.10

    contract_pass_threshold: float = 0.5
    novelty_seg_validity_gate: float = 0.4


@dataclass
class SkillQualityWeights:
    """Configurable weights for skill-level quality scoring."""

    mean_segment_quality: float = 0.30
    reuse_success: float = 0.25
    contract_pass: float = 0.20
    cross_episode_consistency: float = 0.15
    exploration_value: float = 0.10

    min_reuse_for_full_credit: int = 5
    min_episodes_for_consistency: int = 2


DEFAULT_SEGMENT_WEIGHTS = SegmentQualityWeights()
DEFAULT_SKILL_WEIGHTS = SkillQualityWeights()


# ─────────────────────────────────────────────────────────────────────
# Quality scoring
# ─────────────────────────────────────────────────────────────────────

def _tag_consistency(
    intention_tags: List[str],
    expected_pattern: List[str],
) -> float:
    """Measure how well the observed intention tags match the expected pattern.

    Returns 1.0 for perfect alignment, 0.0 for no overlap.  When no
    expected_pattern is set, returns a neutral 0.5.
    """
    if not expected_pattern:
        return 0.5
    if not intention_tags:
        return 0.0
    expected_set = set(expected_pattern)
    observed_set = set(intention_tags)
    if not expected_set:
        return 0.5
    overlap = len(expected_set & observed_set)
    return overlap / len(expected_set)


def score_sub_episode(
    sub_ep: SubEpisodeRef,
    skill: Skill,
    *,
    reward_range: Tuple[float, float] = (0.0, 1.0),
    median_length: Optional[int] = None,
    contract_pass_rate: Optional[float] = None,
    is_novel: bool = False,
    weights: Optional[SegmentQualityWeights] = None,
) -> float:
    """Compute a composite quality score in [0, 1] for a single sub-episode.

    Dimensions:
      - episode_credit:    min-max normalized cumulative game reward
      - local_progress:    success/partial/failure outcome
      - seg_validity:      tag consistency + compactness (segmentation quality)
      - contract_validity: does this skill's contract pass verification?
      - novelty_bonus:     extra credit for novel skills that pass seg_validity

    Bootstrap mode: when no protocol exists yet or all rewards are zero,
    the scoring relaxes to avoid penalizing sub-episodes that simply lack
    reward signals (common in sparse-reward games).
    """
    w = weights or DEFAULT_SEGMENT_WEIGHTS

    # -- episode_credit: normalized cumulative reward from the game
    r_min, r_max = reward_range
    r_range = r_max - r_min if r_max > r_min else 0.0
    if r_range > 0:
        episode_credit = max(0.0, min(1.0, (sub_ep.cumulative_reward - r_min) / r_range))
    else:
        episode_credit = 0.5

    # -- local_progress: outcome-based
    follow_map = {"success": 1.0, "partial": 0.5, "failure": 0.0}
    local_progress = follow_map.get(sub_ep.outcome, 0.5)

    # -- seg_validity: tag consistency + compactness
    has_protocol = skill.protocol.steps and skill.protocol.expected_duration > 1
    if has_protocol:
        expected = skill.protocol.expected_duration
    elif median_length is not None and median_length > 1:
        expected = median_length
    else:
        expected = max(sub_ep.length, 1)
    compactness = max(0.0, 1.0 - abs(sub_ep.length - expected) / max(expected, 1))
    consistency = _tag_consistency(sub_ep.intention_tags, skill.expected_tag_pattern)
    seg_validity = 0.5 * compactness + 0.5 * consistency

    # -- contract_validity: does the skill's contract pass?
    if contract_pass_rate is not None:
        contract_valid = min(1.0, contract_pass_rate / max(w.contract_pass_threshold, 0.01))
    elif skill.contract and skill.contract.total_literals > 0:
        contract_valid = 0.5
    else:
        contract_valid = 0.0

    # -- novelty_bonus: only awarded if novel AND passes segmentation validity gate
    novelty = 0.0
    if is_novel and seg_validity >= w.novelty_seg_validity_gate and contract_valid > 0:
        novelty = 1.0

    score = (
        w.episode_credit * episode_credit
        + w.local_progress * local_progress
        + w.seg_validity * seg_validity
        + w.contract_validity * contract_valid
        + w.novelty_bonus * novelty
    )
    return max(0.0, min(1.0, score))


def score_all_sub_episodes(
    skill: Skill,
    reward_range: Optional[Tuple[float, float]] = None,
    contract_pass_rate: Optional[float] = None,
    is_novel: bool = False,
    weights: Optional[SegmentQualityWeights] = None,
) -> None:
    """Score all sub-episodes of a skill in place."""
    if not skill.sub_episodes:
        return

    if reward_range is None:
        rewards = [se.cumulative_reward for se in skill.sub_episodes]
        r_min = min(rewards) if rewards else 0.0
        r_max = max(rewards) if rewards else 1.0
        reward_range = (r_min, r_max)

    lengths = sorted(se.length for se in skill.sub_episodes)
    median_length = lengths[len(lengths) // 2] if lengths else None

    for se in skill.sub_episodes:
        se.quality_score = score_sub_episode(
            se, skill,
            reward_range=reward_range,
            median_length=median_length,
            contract_pass_rate=contract_pass_rate,
            is_novel=is_novel,
            weights=weights,
        )


# ─────────────────────────────────────────────────────────────────────
# Drift detection
# ─────────────────────────────────────────────────────────────────────

def protocol_drift_detected(
    new_sub_eps: List[SubEpisodeRef],
    skill: Skill,
    drift_threshold: float = 0.2,
) -> bool:
    """Detect whether new sub-episodes show significantly different patterns
    from the existing protocol.

    Heuristic: if the success rate of new sub-episodes differs from
    the overall success rate by more than *drift_threshold*, drift is detected.
    """
    if not new_sub_eps or not skill.sub_episodes:
        return False

    overall_sr = skill.success_rate
    new_successes = sum(1 for se in new_sub_eps if se.outcome == "success")
    new_sr = new_successes / len(new_sub_eps)

    return abs(new_sr - overall_sr) > drift_threshold


# ─────────────────────────────────────────────────────────────────────
# Aggregate / Update / Drop decisions
# ─────────────────────────────────────────────────────────────────────

def run_quality_check(
    skill: Skill,
    *,
    drop_threshold: float = 0.2,
    min_aggregate_count: int = 3,
    min_viable_count: int = 2,
    protocol_quality_threshold: float = 0.6,
    bootstrap_quality_threshold: float = 0.35,
    contract_pass_rate: Optional[float] = None,
    is_novel: bool = False,
    segment_weights: Optional[SegmentQualityWeights] = None,
) -> Dict:
    """Run the full quality-check pipeline for one skill.

    Steps:
      1. Score all sub-episodes (using episode reward + contract validity)
      2. DROP bottom-quality sub-episodes
      3. AGGREGATE: mark skill for protocol update if enough high-quality exist
      4. RETIRE if too few sub-episodes remain

    Bootstrap mode: when no protocol exists yet, the quality threshold
    for protocol synthesis is lowered to ``bootstrap_quality_threshold``
    so the first protocol can be bootstrapped from available data.

    Returns a dict summarizing the actions taken.
    """
    score_all_sub_episodes(
        skill,
        contract_pass_rate=contract_pass_rate,
        is_novel=is_novel,
        weights=segment_weights,
    )

    result = {
        "skill_id": skill.skill_id,
        "before_count": len(skill.sub_episodes),
        "dropped": 0,
        "needs_protocol_update": False,
        "retired": False,
    }

    # DROP: remove bottom-quality sub-episodes
    if skill.sub_episodes:
        scores = sorted(se.quality_score for se in skill.sub_episodes)
        adaptive_threshold = max(
            drop_threshold,
            scores[len(scores) // 5] if len(scores) >= 5 else 0.0,
        )
        before = len(skill.sub_episodes)
        skill.sub_episodes = [
            se for se in skill.sub_episodes
            if se.quality_score >= adaptive_threshold
        ]
        result["dropped"] = before - len(skill.sub_episodes)

    # AGGREGATE: check if enough high-quality sub-episodes exist.
    # Use a lower threshold when no protocol exists yet (bootstrap).
    has_protocol = bool(skill.protocol and skill.protocol.steps)
    q_threshold = protocol_quality_threshold if has_protocol else bootstrap_quality_threshold
    high_quality = [se for se in skill.sub_episodes if se.quality_score >= q_threshold]
    if len(high_quality) >= min_aggregate_count:
        result["needs_protocol_update"] = True

    # RETIRE: either too few sub-episodes remain, or skill_score is
    # persistently low (< 0.2) with enough evidence (>= 5 sub-episodes).
    if result["before_count"] > 0 and len(skill.sub_episodes) < min_viable_count:
        skill.retired = True
        result["retired"] = True
    elif len(skill.sub_episodes) >= 5:
        ss = skill.compute_skill_score(contract_pass_rate=contract_pass_rate)
        if ss < 0.2:
            skill.retired = True
            result["retired"] = True
            result["retire_reason"] = f"low skill_score ({ss:.2f})"

    # Update instance count (keep the higher of linked vs historical)
    if skill.sub_episodes:
        skill.n_instances = max(skill.n_instances, len(skill.sub_episodes))
    result["after_count"] = len(skill.sub_episodes)

    return result


def run_quality_check_batch(
    skills: List[Skill],
    bank: Any = None,
    **kwargs,
) -> List[Dict]:
    """Run quality check on a batch of skills.

    When *bank* is provided, contract pass rates are looked up from
    verification reports so that ``score_sub_episode`` can incorporate
    contract validity into the quality score.
    """
    results = []
    for skill in skills:
        if skill.retired:
            continue
        cpr = None
        if bank is not None and hasattr(bank, "get_report"):
            report = bank.get_report(skill.skill_id)
            if report is not None:
                cpr = report.overall_pass_rate
        r = run_quality_check(skill, contract_pass_rate=cpr, **kwargs)
        results.append(r)
        if r["dropped"] > 0:
            logger.info(
                "Skill %s: dropped %d low-quality sub-episodes (%d remaining)",
                skill.skill_id, r["dropped"], r["after_count"],
            )
        if r["retired"]:
            logger.info("Skill %s: retired (too few sub-episodes)", skill.skill_id)
    return results
