"""
Reward functions for each GRPO-wrapped stage.

Each reward function has the signature expected by ``GRPOCallWrapper``:
    ``reward_fn(sample_output, *original_args, **original_kwargs) -> float``

Design philosophy — three stages reinforce each other:
  Segmentation  → find the most valuable skills (high episode reward)
  Contract      → produce solid start/end condition descriptions
  Curator       → base decisions on skill quality; encourage new skill exploration

All reward computations are CPU-only — no additional LLM inference.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_RAW_BLEND = 0.08  # weight for raw-text fingerprint in contract / curator rewards


def _raw_completion_fingerprint(llm_output: Any) -> float:
    """Deterministic fingerprint in [0, 1] derived from raw LLM text.

    When ``llm_output`` is a ``SkillBankLLMOutput`` carrying
    ``_grpo_raw_completion``, the hash of that text is used so that
    structurally identical (same parsed JSON) but textually different
    completions get different rewards — critical for GRPO learning.
    Falls back to 0.5 (neutral) when raw text is unavailable.
    """
    raw = getattr(llm_output, "_grpo_raw_completion", None) or ""
    if not raw:
        return 0.5
    h = 0
    for ch in raw:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return (h % 10000) / 10000.0


# ── Stage 3: CONTRACT reward ─────────────────────────────────────────
#
# A good contract is *solid*: it accurately describes what predicates
# hold at segment start and what changes by segment end.  We reward
# contracts that cover both eff_add (gained conditions) and eff_del
# (lost conditions) and that hold across holdout instances — especially
# high-reward ones.

def contract_reward(
    llm_output: Optional[Dict[str, Any]],
    skill_id: str,
    segment_observations: List[str],
    predicates_start: Set[str],
    predicates_end: Set[str],
    n_instances: int = 0,
    *,
    consensus_add: Optional[Set[str]] = None,
    consensus_del: Optional[Set[str]] = None,
    holdout_instances: Optional[list] = None,
    verify_config: Optional[Any] = None,
    instance_rewards: Optional[List[float]] = None,
    **kwargs: Any,
) -> float:
    """Reward a contract for being solid on start/end conditions.

    Evaluates the LLM's proposed effects **standalone** (no union-merge
    with consensus) to ensure each completion is scored on its own merit.
    Specifically rewards contracts that cover BOTH eff_add and eff_del
    (start→end transition) and verify on high-reward instances.
    """
    if llm_output is None:
        return 0.0

    eff_add = llm_output.get("eff_add", [])
    eff_del = llm_output.get("eff_del", [])

    r_raw = _raw_completion_fingerprint(llm_output)

    if not eff_add and not eff_del:
        return 0.05 * (1.0 - _RAW_BLEND) + _RAW_BLEND * r_raw

    if holdout_instances is not None and verify_config is not None:
        base = _contract_reward_with_verification(
            llm_output, skill_id, consensus_add, consensus_del,
            holdout_instances, verify_config, instance_rewards,
            predicates_start, predicates_end,
        )
    else:
        base = _contract_reward_start_end_coverage(
            eff_add, eff_del, predicates_start, predicates_end,
        )

    return max(0.0, min(1.0, base * (1.0 - _RAW_BLEND) + _RAW_BLEND * r_raw))


def _contract_reward_start_end_coverage(
    eff_add: list,
    eff_del: list,
    predicates_start: Optional[Set[str]],
    predicates_end: Optional[Set[str]],
) -> float:
    """Score by how well the contract describes the start→end transition.

    Uses F1 + content fingerprint so different effect IDENTITIES produce
    different rewards, even when structural metrics (precision, recall,
    literal count) happen to be equal.

    Components:
      0.22 * r_add_f1      -- F1 of eff_add vs predicates_end
      0.22 * r_del_f1      -- F1 of eff_del vs predicates_start
      0.18 * r_both_sides  -- bonus for covering both add and del
      0.13 * r_sparsity    -- fewer, higher-quality effects
      0.13 * r_specificity -- ratio of effects that are actual changes
      0.12 * r_fingerprint -- content-sensitive tiebreaker
    """
    llm_add = set(eff_add)
    llm_del = set(eff_del)
    p_start = predicates_start or set()
    p_end = predicates_end or set()

    r_add_f1 = _set_f1(llm_add, p_end)
    r_del_f1 = _set_f1(llm_del, p_start)

    r_both = 1.0 if (llm_add and llm_del) else 0.3

    n_literals = len(llm_add) + len(llm_del)
    r_sparsity = max(0.0, 1.0 - n_literals / 20.0)

    if p_start and p_end:
        actual_gains = p_end - p_start
        actual_losses = p_start - p_end
        spec_hits = len(llm_add & actual_gains) + len(llm_del & actual_losses)
        r_specificity = spec_hits / max(n_literals, 1)
    else:
        r_specificity = 0.5

    # r_fingerprint: deterministic hash of effect identities → [0, 1]
    # Different effect sets produce different tiebreaker values.
    r_fingerprint = _effect_fingerprint(llm_add, llm_del)

    return max(0.0, min(1.0,
        0.22 * r_add_f1
        + 0.22 * r_del_f1
        + 0.18 * r_both
        + 0.13 * r_sparsity
        + 0.13 * r_specificity
        + 0.12 * r_fingerprint,
    ))


def _effect_fingerprint(eff_add: Set[str], eff_del: Set[str]) -> float:
    """Deterministic fingerprint in [0, 1] that varies with effect identity.

    Uses a simple sum-of-hashes approach so that different effect sets
    always produce different values, breaking ties in structural metrics.
    """
    h = 0
    for p in sorted(eff_add):
        h = (h * 31 + hash(p)) & 0xFFFFFFFF
    for p in sorted(eff_del):
        h = (h * 37 + hash(p)) & 0xFFFFFFFF
    return (h % 10000) / 10000.0


def _set_f1(predicted: Set[str], reference: Set[str]) -> float:
    """F1 score between predicted and reference sets."""
    if not predicted and not reference:
        return 0.5
    if not predicted or not reference:
        return 0.0
    overlap = len(predicted & reference)
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _contract_reward_with_verification(
    llm_output: Dict[str, Any],
    skill_id: str,
    consensus_add: Optional[Set[str]],
    consensus_del: Optional[Set[str]],
    holdout_instances: list,
    verify_config: Any,
    instance_rewards: Optional[List[float]] = None,
    predicates_start: Optional[Set[str]] = None,
    predicates_end: Optional[Set[str]] = None,
) -> float:
    """Evaluate the LLM's effects standalone, weighted by instance reward.

    Components (weights sum to 1.0):
      0.30 * r_standalone     -- reward-weighted pass rate on holdout
      0.25 * r_start_end      -- coverage of start/end conditions
      0.15 * r_precision      -- LLM effects that appear in consensus
      0.15 * r_recall         -- consensus effects the LLM recovers
      0.15 * r_reward_align   -- effects hold better for high-reward instances
    """
    from skill_agents_grpo.stage3_mvp.schemas import SkillEffectsContract

    llm_add = set(llm_output.get("eff_add", []))
    llm_del = set(llm_output.get("eff_del", []))

    standalone = SkillEffectsContract(
        skill_id=skill_id,
        eff_add=llm_add,
        eff_del=llm_del,
    )

    # -- r_standalone: reward-weighted pass rate
    rewards = instance_rewards or [0.0] * len(holdout_instances)
    abs_total = sum(abs(r) for r in rewards) or 1.0
    weighted_pass = 0.0
    total_weight = 0.0
    pass_hi = 0
    pass_lo = 0
    n_hi = 0
    n_lo = 0
    reward_median = sorted(rewards)[len(rewards) // 2] if rewards else 0.0

    for inst, r in zip(holdout_instances, rewards):
        passes = _instance_passes(standalone, inst, verify_config)
        w = 1.0 + abs(r) / abs_total
        weighted_pass += w if passes else 0.0
        total_weight += w
        if r >= reward_median:
            n_hi += 1
            if passes:
                pass_hi += 1
        else:
            n_lo += 1
            if passes:
                pass_lo += 1

    r_standalone = weighted_pass / max(total_weight, 1e-8)

    # -- r_start_end: do eff_add/eff_del cover the start→end transition?
    r_start_end = _contract_reward_start_end_coverage(
        list(llm_add), list(llm_del), predicates_start, predicates_end,
    )

    # -- r_precision / r_recall vs consensus
    all_consensus = (consensus_add or set()) | (consensus_del or set())
    all_llm = llm_add | llm_del
    if all_llm and all_consensus:
        overlap = len(all_llm & all_consensus)
        r_precision = overlap / len(all_llm)
        r_recall = overlap / len(all_consensus)
    elif all_llm:
        r_precision = 0.5
        r_recall = 0.0
    else:
        r_precision = 0.0
        r_recall = 0.0

    # -- r_reward_align: pass rate on high-reward vs low-reward instances
    hi_rate = pass_hi / max(n_hi, 1)
    lo_rate = pass_lo / max(n_lo, 1)
    r_reward_align = min(1.0, max(0.0, 0.5 + (hi_rate - lo_rate)))

    return max(0.0, min(1.0,
        0.30 * r_standalone
        + 0.25 * r_start_end
        + 0.15 * r_precision
        + 0.15 * r_recall
        + 0.15 * r_reward_align,
    ))


def _instance_passes(contract: Any, instance: Any, config: Any) -> bool:
    """Check if a single instance passes the contract."""
    fails = 0
    for p in contract.eff_add:
        if p not in instance.eff_add and p not in instance.B_end:
            fails += 1
    for p in contract.eff_del:
        if p not in instance.eff_del and p in instance.B_end:
            fails += 1
    total = contract.total_literals
    if total == 0:
        return True
    return (total - fails) / total >= config.instance_pass_literal_frac


# ── Stage 4: CURATOR reward ──────────────────────────────────────────
#
# The curator bases decisions on *skill quality* (compute_skill_score)
# while actively encouraging exploration of new skills.  Good curators:
#  - Approve high-quality skills and veto genuinely poor ones
#  - Approve promising new skills (materialize/promote) even with limited data
#  - Use skill_score as a continuous signal, not a binary threshold

def curator_reward(
    decisions: Optional[Dict[str, Any]],
    candidates: list,
    bank: Any,
    *args: Any,
    action_outcomes: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> float:
    """Reward for curator decisions grounded in skill quality + exploration.

    Parameters
    ----------
    decisions : dict
        ``{"decisions": [{"idx": 0, "verdict": "approve", "reason": "..."}, ...]}``
    candidates : list
        Candidate bank mutations (from the algorithmic proposer).
    bank : SkillBankMVP
        Current skill bank state.
    action_outcomes : list[dict], optional
        Per-candidate ground truth: ``{"succeeded": bool, "quality_delta": float}``.
    """
    if decisions is None:
        return 0.0

    decision_list = decisions.get("decisions", [])
    r_raw = _raw_completion_fingerprint(decisions)

    if not decision_list:
        return 0.05 * (1.0 - _RAW_BLEND) + _RAW_BLEND * r_raw

    if action_outcomes is not None and len(action_outcomes) == len(candidates):
        base = _curator_reward_with_outcomes(
            decision_list, candidates, action_outcomes,
        )
    else:
        base = _curator_reward_evidence(decision_list, candidates)

    return max(0.0, min(1.0, base * (1.0 - _RAW_BLEND) + _RAW_BLEND * r_raw))


def _curator_reward_evidence(
    decision_list: List[Dict[str, Any]],
    candidates: list,
) -> float:
    """Score curator decisions using continuous skill quality + exploration.

    Components (weights sum to 1.0):
      0.40 * r_quality_align   -- verdict consistency with continuous skill_score
      0.25 * r_exploration     -- credit for approving promising new skills
      0.35 * r_reason_quality  -- reasons cite concrete evidence from candidate data
    """
    n_scored = 0
    quality_align_score = 0.0
    exploration_score = 0.0
    reason_score = 0.0

    for d in decision_list:
        idx = d.get("idx")
        verdict = (d.get("verdict") or "").lower()
        reason = d.get("reason") or ""
        if idx is None or not isinstance(idx, int) or idx >= len(candidates):
            continue

        cand = candidates[idx]
        n_scored += 1

        ss = cand.get("skill_score", 0.5)
        pr = cand.get("pass_rate")
        ni = cand.get("n_instances", 0)
        atype = cand.get("type", "")

        # -- r_quality_align: continuous alignment with skill_score
        # Approve is good when skill_score is high; veto when low
        if verdict == "approve":
            quality_align_score += ss
        elif verdict == "veto":
            quality_align_score += (1.0 - ss)
        elif verdict == "defer":
            mid_distance = 1.0 - 2.0 * abs(ss - 0.5)
            quality_align_score += 0.5 * mid_distance + 0.3
        else:
            quality_align_score += 0.3

        # -- r_exploration: encourage approving new/young skills
        is_new_skill = atype in ("materialize", "promote")
        is_young = ni <= 5
        has_some_evidence = ni >= 1

        if is_new_skill and verdict == "approve" and has_some_evidence:
            if pr is not None and pr >= 0.4:
                exploration_score += 1.0
            elif ss >= 0.3:
                exploration_score += 0.8
            else:
                exploration_score += 0.5
        elif is_new_skill and verdict == "defer" and has_some_evidence:
            exploration_score += 0.3
        elif is_new_skill and verdict == "veto" and is_young:
            exploration_score += 0.1
        elif not is_new_skill:
            exploration_score += 0.5

        # -- r_reason_quality: does the reason cite concrete evidence?
        spec = 0.0
        if pr is not None and (f"{pr:.2f}" in reason or f"{pr:.1f}" in reason):
            spec += 0.35
        if ss is not None and (f"{ss:.2f}" in reason or f"{ss:.1f}" in reason):
            spec += 0.25
        if ni and str(ni) in reason:
            spec += 0.2
        sid = cand.get("skill_id", "")
        if sid and sid in reason:
            spec += 0.2
        if not reason:
            spec = 0.0
        elif len(reason) > 10 and spec == 0.0:
            spec = 0.1
        reason_score += min(1.0, spec)

    if n_scored == 0:
        return 0.05

    r_quality = quality_align_score / n_scored
    r_explore = exploration_score / n_scored
    r_reason = reason_score / n_scored

    return max(0.0, min(1.0,
        0.40 * r_quality + 0.25 * r_explore + 0.35 * r_reason,
    ))


def _curator_reward_with_outcomes(
    decision_list: List[Dict[str, Any]],
    candidates: list,
    action_outcomes: List[Dict[str, Any]],
) -> float:
    """Reward based on outcomes, weighted by quality_delta.

    Also includes exploration awareness: approving new skills that succeed
    gets extra credit, vetoing new skills that would have succeeded is penalized.
    """
    weighted_correct = 0.0
    total_weight = 0.0
    exploration_bonus = 0.0
    n_new = 0

    for d in decision_list:
        idx = d.get("idx")
        verdict = (d.get("verdict") or "").lower()
        if idx is None or not isinstance(idx, int) or idx >= len(action_outcomes):
            continue
        outcome = action_outcomes[idx]
        cand = candidates[idx] if idx < len(candidates) else {}
        succeeded = outcome.get("succeeded", True)
        qd = outcome.get("quality_delta", 0.0)
        atype = cand.get("type", "")

        w = 1.0 + abs(qd)
        total_weight += w

        if verdict == "approve" and succeeded:
            weighted_correct += w
        elif verdict == "veto" and not succeeded:
            weighted_correct += w
        elif verdict == "defer":
            weighted_correct += w * (0.3 if abs(qd) < 0.05 else 0.1)

        is_new_skill = atype in ("materialize", "promote")
        if is_new_skill:
            n_new += 1
            if verdict == "approve" and succeeded:
                exploration_bonus += 1.0
            elif verdict == "approve" and not succeeded:
                exploration_bonus += 0.2
            elif verdict == "veto" and succeeded:
                exploration_bonus += 0.0

    if total_weight < 1e-8:
        return 0.05

    accuracy = weighted_correct / total_weight
    r_explore = exploration_bonus / max(n_new, 1) if n_new > 0 else 0.5
    r_evidence = _curator_reward_evidence(decision_list, candidates)

    return max(0.0, min(1.0,
        0.40 * (0.3 + 0.7 * accuracy)
        + 0.25 * r_evidence
        + 0.15 * accuracy
        + 0.20 * r_explore,
    ))


# ── Stage 2: SEGMENT reward ──────────────────────────────────────────
#
# A good segmentation finds the most *valuable* skills: it assigns
# high-reward segments to high-quality existing skills, and only
# proposes __NEW__ when existing skills genuinely don't fit.

# When decode-based terms (reuse, margins, reward-capture) all saturate at
# their maximum, every GRPO sample in a group can get reward 1.0 even if the
# LLM produced different rankings.  Blend in a preference fingerprint so
# group-normalized GRPO still sees spread (mirrors r_winner_fp on fallback).
_SEGMENT_DECODE_PREF_BLEND = 0.14


def _hash_raw_rollout_texts(texts: List[str]) -> int:
    """Stable hash over multiset of raw LLM strings (order-independent)."""
    h = 0
    for t in sorted(texts or []):
        body = t if isinstance(t, str) else str(t)
        limit = min(len(body), 12000)
        for i in range(limit):
            h = (h * 137 + ord(body[i])) & 0xFFFFFFFF
    return h


def _preference_list_fingerprint(preference_list: list) -> float:
    """Map preference_list to [0, 1] for GRPO tie-breaking.

    Uses sorted rows of (segment, winner, loser, evidence_norm).  Including
    **evidence** differentiates completions that fuzzy-map to the same
    (win, lose) pairs but carry different LLM reasoning.

    When ``preference_list`` is a :class:`PreferenceListWithRollouts`, also
    hashes **raw JSON / text** from each segment LLM call.  Different
    stochastic rollouts often collapse to identical parsed prefs; raw text
    keeps GRPO rewards aligned with actual rollout diversity.
    """
    if not preference_list:
        return 0.5
    parts: List[tuple] = []
    for p in preference_list:
        try:
            ev = getattr(p, "evidence", None) or ""
            ev_norm = " ".join(str(ev).split())[:512]
            parts.append(
                (
                    int(p.segment_start),
                    int(p.segment_end),
                    str(p.skill_win),
                    str(p.skill_lose),
                    ev_norm,
                )
            )
        except Exception:
            parts.append((str(type(p)), str(p), "", "", ""))
    sig = tuple(sorted(parts))
    h = 0
    for tup in sig:
        for x in tup:
            h = (h * 31 + hash(x)) & 0xFFFFFFFF
    raw_rollouts = getattr(preference_list, "raw_rollouts", None) or []
    if raw_rollouts:
        h = (h * 41 + _hash_raw_rollout_texts(list(raw_rollouts))) & 0xFFFFFFFF
    return (h % 10000) / 10000.0


def segmentation_reward(
    preference_list: list,
    segments: list,
    observations: Any,
    actions: Any,
    skill_names: List[str],
    *args: Any,
    scorer_factory: Optional[Any] = None,
    decode_fn: Optional[Any] = None,
    predicates: Optional[list] = None,
    per_step_rewards: Optional[List[float]] = None,
    episode_total_reward: Optional[float] = None,
    bank_skill_scores: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> float:
    """Reward for finding the most valuable skill assignments.

    Parameters
    ----------
    bank_skill_scores : dict[str, float], optional
        Per-skill quality scores from Skill.compute_skill_score().
        Used to reward assignments to high-quality bank skills.
    """
    if not preference_list:
        return 0.0

    if scorer_factory is not None and decode_fn is not None:
        return _segmentation_reward_with_decode(
            preference_list, segments, observations, actions,
            skill_names, scorer_factory, decode_fn, predicates,
            per_step_rewards=per_step_rewards,
            episode_total_reward=episode_total_reward,
            bank_skill_scores=bank_skill_scores,
        )

    # Fallback (no scorer/decoder) — must produce distinct rewards for
    # different LLM preference outputs (different winners/rankings).
    existing_skills = {s for s in skill_names if s != "__NEW__"}

    # r_reuse_strength: weighted reuse signal — fraction of preferences
    # where an existing skill wins, scaled by how many distinct existing
    # skills win (diversity of reuse).
    n_existing_wins = sum(1 for p in preference_list if p.skill_win in existing_skills)
    r_reuse_frac = n_existing_wins / max(len(preference_list), 1)
    winning_existing = {p.skill_win for p in preference_list if p.skill_win in existing_skills}
    skill_diversity = len(winning_existing) / max(len(existing_skills), 1) if existing_skills else 0.0
    r_reuse = 0.7 * r_reuse_frac + 0.3 * skill_diversity

    # r_dominance: how consistently each segment has a clear winner
    # (high dominance = one skill wins most comparisons per segment)
    seg_winners: Dict[tuple, Dict[str, int]] = {}
    for p in preference_list:
        key = (p.segment_start, p.segment_end)
        seg_winners.setdefault(key, {})
        seg_winners[key][p.skill_win] = seg_winners[key].get(p.skill_win, 0) + 1
    if seg_winners:
        dominances = []
        for counts in seg_winners.values():
            total = sum(counts.values())
            top = max(counts.values())
            dominances.append(top / max(total, 1))
        r_dominance = sum(dominances) / len(dominances)
    else:
        r_dominance = 0.0

    # r_coverage: fraction of segments that have at least one preference
    segment_coverage = len(seg_winners)
    r_coverage = segment_coverage / max(len(segments), 1)

    # r_quality_hint: bank skill quality of winning skills
    r_quality_hint = 0.0
    if bank_skill_scores:
        quality_wins = [
            bank_skill_scores.get(p.skill_win, 0.0)
            for p in preference_list
            if p.skill_win in existing_skills
        ]
        if quality_wins:
            r_quality_hint = sum(quality_wins) / len(quality_wins)

    # r_winner_fingerprint: align with decode-path fingerprint (incl. evidence)
    r_winner_fp = _preference_list_fingerprint(preference_list)

    if bank_skill_scores:
        return max(0.0, min(1.0,
            0.25 * r_reuse
            + 0.15 * r_dominance
            + 0.15 * r_coverage
            + 0.30 * r_quality_hint
            + 0.15 * r_winner_fp,
        ))
    return max(0.0, min(1.0,
        0.35 * r_reuse
        + 0.25 * r_dominance
        + 0.25 * r_coverage
        + 0.15 * r_winner_fp,
    ))


def _segmentation_reward_with_decode(
    preference_list: list,
    segments: list,
    observations: Any,
    actions: Any,
    skill_names: List[str],
    scorer_factory: Any,
    decode_fn: Any,
    predicates: Optional[list],
    per_step_rewards: Optional[List[float]] = None,
    episode_total_reward: Optional[float] = None,
    bank_skill_scores: Optional[Dict[str, float]] = None,
) -> float:
    """Reward grounded in episode reward + skill bank quality.

    Components (weights sum to 1.0):
      0.15 * r_decode_score   -- normalized Viterbi total_score
      0.15 * r_reuse          -- fraction of segments assigned to existing skills
      0.30 * r_reward_reuse   -- fraction of positive reward captured by existing skills
      0.25 * r_value_match    -- high-quality skills assigned to high-reward segments
      0.15 * r_margin         -- mean decode margin quality
    """
    r_decode_score = 0.3
    r_reuse = 0.0
    r_reward_reuse = 0.0
    r_value_match = 0.0
    r_margin = 0.0

    try:
        scorer = scorer_factory(preference_list)
        result = decode_fn(scorer, segments, observations, actions, skill_names, predicates)
        if not (hasattr(result, "segments") and result.segments):
            return 0.1

        n_total = len(result.segments)

        if hasattr(result, "total_score") and math.isfinite(result.total_score):
            raw = result.total_score / max(n_total, 1)
            r_decode_score = min(1.0, max(0.0, (raw + 5.0) / 10.0))

        n_existing = sum(
            1 for seg in result.segments
            if seg.assigned_skill != "__NEW__"
        )
        r_reuse = n_existing / max(n_total, 1)

        if per_step_rewards:
            pos_existing = 0.0
            pos_total = 0.0
            for seg in result.segments:
                seg_r = sum(per_step_rewards[seg.start: seg.end + 1])
                pos_r = max(0.0, seg_r)
                pos_total += pos_r
                if seg.assigned_skill != "__NEW__":
                    pos_existing += pos_r
                else:
                    pos_existing += 0.3 * pos_r
            r_reward_reuse = pos_existing / (pos_total + 1e-8) if pos_total > 0 else 0.5

        # r_value_match: high-quality skills get high-reward segments
        if bank_skill_scores and per_step_rewards:
            value_scores = []
            for seg in result.segments:
                skill_q = bank_skill_scores.get(seg.assigned_skill, 0.0)
                seg_r = sum(per_step_rewards[seg.start: seg.end + 1])
                seg_r_norm = max(0.0, seg_r) / (abs(episode_total_reward) + 1e-8) if episode_total_reward else 0.0
                if seg.assigned_skill != "__NEW__":
                    value_scores.append(skill_q * (0.5 + 0.5 * seg_r_norm))
                else:
                    value_scores.append(0.2 * max(0.0, seg_r_norm))
            r_value_match = sum(value_scores) / max(len(value_scores), 1) if value_scores else 0.0
            r_value_match = min(1.0, r_value_match)
        elif bank_skill_scores:
            quals = [
                bank_skill_scores.get(seg.assigned_skill, 0.0)
                for seg in result.segments
                if seg.assigned_skill != "__NEW__"
            ]
            r_value_match = sum(quals) / max(len(quals), 1) if quals else 0.0

        margins = []
        for seg in result.segments:
            if hasattr(seg, "margin") and math.isfinite(seg.margin):
                margins.append(seg.margin)
        if margins:
            avg_margin = sum(margins) / len(margins)
            r_margin = min(1.0, max(0.0, avg_margin / 10.0))

    except Exception:
        logger.debug("Segmentation reward decode failed", exc_info=True)
        return 0.1

    r_pref_fp = _preference_list_fingerprint(preference_list)
    blend = _SEGMENT_DECODE_PREF_BLEND
    if bank_skill_scores:
        base = max(0.0, min(1.0,
            0.15 * r_decode_score
            + 0.15 * r_reuse
            + 0.30 * r_reward_reuse
            + 0.25 * r_value_match
            + 0.15 * r_margin,
        ))
    else:
        base = max(0.0, min(1.0,
            0.20 * r_decode_score
            + 0.25 * r_reuse
            + 0.35 * r_reward_reuse
            + 0.20 * r_margin,
        ))
    # Prefer structural signal, but never return identical rewards for every
    # group member when the LLM outputs differ.
    return max(0.0, min(1.0, base * (1.0 - blend) + blend * r_pref_fp))


# ── Skill Selection reward (delayed, for decision-agent GRPO) ────────

def skill_selection_reward(
    reward_on_skill: float,
    steps_on_skill: int,
    max_skill_duration: int = 10,
    success_met: bool = False,
    abort_triggered: bool = False,
    confidence: float = 0.5,
) -> float:
    """Reward for a skill selection decision, assigned at skill-switch time.

    This is a *delayed* reward: it is computed when the skill tracker
    triggers a reselection (duration exceeded, success met, abort, or
    zero-reward stall) and retroactively assigned to the skill-selection
    sample in the GRPO buffer.

    Parameters
    ----------
    reward_on_skill : float
        Cumulative environment reward earned while the skill was active.
    steps_on_skill : int
        Number of steps the skill was active.
    max_skill_duration : int
        Maximum allowed duration for the skill.
    success_met : bool
        Whether the skill's success criteria were satisfied.
    abort_triggered : bool
        Whether the skill was terminated due to abort criteria.
    confidence : float
        RAG confidence score of the selected skill (prior).

    Returns
    -------
    float
        Reward in [0, 1].

    Components (weights sum to 1.0):
        0.40 * env_reward — normalized cumulative reward during skill
        0.20 * efficiency — ratio of useful steps vs. max duration
        0.20 * success    — 1.0 if success criteria met, 0.0 otherwise
        0.10 * no_abort   — 1.0 if no abort triggered, 0.0 otherwise
        0.10 * confidence — RAG confidence as a soft prior
    """
    r_env = min(1.0, max(0.0, reward_on_skill / max(steps_on_skill, 1)))

    if steps_on_skill <= 0:
        r_efficiency = 0.0
    else:
        r_efficiency = min(1.0, max_skill_duration / max(steps_on_skill, 1))

    r_success = 1.0 if success_met else 0.0
    r_no_abort = 0.0 if abort_triggered else 1.0
    r_confidence = max(0.0, min(1.0, confidence))

    return (
        0.40 * r_env
        + 0.20 * r_efficiency
        + 0.20 * r_success
        + 0.10 * r_no_abort
        + 0.10 * r_confidence
    )
