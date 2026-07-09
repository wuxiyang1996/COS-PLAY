"""
Viterbi DP decoder (HSMM-style) for InferSegmentation.

Finds the globally optimal segmentation + skill labeling given the scorer.

dp[j, k] = best total score up to boundary j if the last segment ends at j
            and is labeled k.

Recurrence:
    dp[j, k] = max_{i in C, i < j, k'}  dp[i, k'] + Score(i+1, j, k | k')

Backtracking recovers the best path.
Only candidate boundaries C ∪ {1, T} are considered.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from skill_agents.infer_segmentation.config import SegmentationConfig
from skill_agents.infer_segmentation.scorer import SegmentScorer
from skill_agents.infer_segmentation.diagnostics import (
    SegmentationResult,
    SegmentDiagnostic,
    SkillCandidate,
)

_NEG_INF = float("-inf")


def _get_obs_actions(
    observations: Sequence,
    actions: Sequence,
    start: int,
    end: int,
) -> Tuple[Sequence, Sequence]:
    """Slice observations and actions for segment [start, end] (inclusive)."""
    return observations[start : end + 1], actions[start : end + 1]


def _get_predicates(
    predicates: Optional[List[Optional[dict]]],
    idx: int,
) -> Optional[dict]:
    if predicates is None or idx < 0 or idx >= len(predicates):
        return None
    return predicates[idx]


def viterbi_decode(
    candidates: List[int],
    T: int,
    scorer: SegmentScorer,
    observations: Sequence,
    actions: Sequence,
    predicates: Optional[List[Optional[dict]]] = None,
    config: Optional[SegmentationConfig] = None,
) -> SegmentationResult:
    """
    Run Viterbi DP over candidate boundaries.

    Parameters
    ----------
    candidates : list[int]
        Candidate boundary positions from Stage 1 (0-indexed timesteps).
    T : int
        Total trajectory length.
    scorer : SegmentScorer
        Composite scorer for evaluating segments.
    observations : Sequence
        Observation/state embeddings, len T.
    actions : Sequence
        Actions taken, len T.
    predicates : list[dict], optional
        Per-timestep predicate dicts for contract compatibility.
    config : SegmentationConfig, optional
        Overrides scorer.config if provided.

    Returns
    -------
    SegmentationResult
        Best segmentation with full diagnostics.
    """
    cfg = config or scorer.config
    skills = scorer.skill_names
    num_skills = len(skills)
    top_k = cfg.decoder.top_k_diagnostics

    # Boundary set: {0} ∪ C ∪ {T-1}  (0-indexed, inclusive)
    boundary_set = sorted(set([0] + candidates + [T - 1]))
    if boundary_set[0] != 0:
        boundary_set = [0] + boundary_set

    num_bounds = len(boundary_set)
    bnd_to_idx = {b: idx for idx, b in enumerate(boundary_set)}

    # dp[b_idx][k_idx] = best score ending at boundary b with skill k
    dp = [[_NEG_INF] * num_skills for _ in range(num_bounds)]
    # backpointer: (prev_b_idx, prev_k_idx)
    bp: List[List[Optional[Tuple[int, int]]]] = [
        [None] * num_skills for _ in range(num_bounds)
    ]
    # top-K candidates per (b_idx, k_idx) for diagnostics
    all_candidates: Dict[Tuple[int, int], List[SkillCandidate]] = {}

    # Restrict skills per segment if top_m_skills is set
    top_m = cfg.decoder.top_m_skills

    # ── Base case: first segment starts at boundary_set[0] ──────────
    first_b = boundary_set[0]  # should be 0
    use_batch = hasattr(scorer, "score_breakdown_batch") and callable(getattr(scorer, "score_breakdown_batch", None))
    for bi in range(1, num_bounds):
        j = boundary_set[bi]
        seg_obs, seg_act = _get_obs_actions(observations, actions, first_b, j)
        p_start = _get_predicates(predicates, first_b)
        p_end = _get_predicates(predicates, j)

        if use_batch:
            batch_reqs = [
                (first_b, j, sk, None, seg_obs, seg_act, p_start, p_end)
                for sk in skills
            ]
            breakdowns = scorer.score_breakdown_batch(batch_reqs)
            scored = [(bd["total"], ki, bd) for ki, bd in enumerate(breakdowns)]
        else:
            scored = []
            for ki, sk in enumerate(skills):
                bd = scorer.score_breakdown(
                    first_b, j, sk, None, seg_obs, seg_act, p_start, p_end
                )
                scored.append((bd["total"], ki, bd))
        scored.sort(key=lambda x: -x[0])

        cands = [
            SkillCandidate(skill=skills[s[1]], total_score=s[0], breakdown=s[2])
            for s in scored[:top_k]
        ]

        for rank, (sc, ki, bd) in enumerate(scored):
            if top_m is not None and rank >= top_m:
                break
            if sc > dp[bi][ki]:
                dp[bi][ki] = sc
                bp[bi][ki] = None  # no predecessor (first segment)
                all_candidates[(bi, ki)] = cands

    # ── Fill DP table ───────────────────────────────────────────────
    for bi in range(2, num_bounds):
        j = boundary_set[bi]
        for prev_bi in range(1, bi):
            i = boundary_set[prev_bi] + 1  # segment starts right after prev boundary
            if i > j:
                continue

            seg_obs, seg_act = _get_obs_actions(observations, actions, i, j)
            p_start = _get_predicates(predicates, i)
            p_end = _get_predicates(predicates, j)

            for prev_ki, prev_sk in enumerate(skills):
                if dp[prev_bi][prev_ki] == _NEG_INF:
                    continue
                base = dp[prev_bi][prev_ki]

                if use_batch:
                    batch_reqs = [
                        (i, j, sk, prev_sk, seg_obs, seg_act, p_start, p_end)
                        for sk in skills
                    ]
                    breakdowns = scorer.score_breakdown_batch(batch_reqs)
                    scored_inner = [(base + bd["total"], ki, bd) for ki, bd in enumerate(breakdowns)]
                else:
                    scored_inner = []
                    for ki, sk in enumerate(skills):
                        bd = scorer.score_breakdown(
                            i, j, sk, prev_sk, seg_obs, seg_act, p_start, p_end
                        )
                        scored_inner.append((base + bd["total"], ki, bd))
                scored_inner.sort(key=lambda x: -x[0])

                cands = [
                    SkillCandidate(
                        skill=skills[s[1]], total_score=s[0], breakdown=s[2]
                    )
                    for s in scored_inner[:top_k]
                ]

                for rank, (total, ki, bd) in enumerate(scored_inner):
                    if top_m is not None and rank >= top_m:
                        break
                    if total > dp[bi][ki]:
                        dp[bi][ki] = total
                        bp[bi][ki] = (prev_bi, prev_ki)
                        all_candidates[(bi, ki)] = cands

    # ── Backtrack ───────────────────────────────────────────────────
    last_bi = num_bounds - 1
    best_ki = max(range(num_skills), key=lambda ki: dp[last_bi][ki])
    best_score = dp[last_bi][best_ki]

    if best_score == _NEG_INF:
        return SegmentationResult(total_score=_NEG_INF)

    path: List[Tuple[int, int]] = []  # (boundary_idx, skill_idx)
    cur_bi, cur_ki = last_bi, best_ki
    while cur_bi is not None:
        path.append((cur_bi, cur_ki))
        prev = bp[cur_bi][cur_ki]
        if prev is None:
            break
        cur_bi, cur_ki = prev
    path.reverse()

    # ── Build SegmentationResult ────────────────────────────────────
    segments: List[SegmentDiagnostic] = []
    for idx in range(len(path)):
        bi, ki = path[idx]
        if idx == 0:
            seg_start = boundary_set[0]
        else:
            prev_bi = path[idx - 1][0]
            seg_start = boundary_set[prev_bi] + 1

        seg_end = boundary_set[bi]
        cands = all_candidates.get((bi, ki), [])

        segments.append(
            SegmentDiagnostic(
                start=seg_start,
                end=seg_end,
                assigned_skill=skills[ki],
                candidates=cands,
            )
        )

    return SegmentationResult(
        segments=segments,
        total_score=best_score,
    )
