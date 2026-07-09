"""
Beam search decoder for InferSegmentation.

Agent-friendly alternative to Viterbi DP:
  - Keeps top-B partial segmentations
  - Can stop early, inspect ambiguity, let the agent refine contracts
  - Returns full diagnostics per segment

State = (last_cut, last_skill, total_score, path)
Expand by choosing next cut and next skill, prune to top-B.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from skill_agents.infer_segmentation.config import SegmentationConfig
from skill_agents.infer_segmentation.scorer import SegmentScorer
from skill_agents.infer_segmentation.diagnostics import (
    SegmentationResult,
    SegmentDiagnostic,
    SkillCandidate,
)


@dataclass(order=True)
class _BeamEntry:
    """Priority-queue entry (negated score for max-heap via min-heap)."""

    neg_score: float
    last_cut: int = field(compare=False)
    last_skill: Optional[str] = field(compare=False)
    path: List[Tuple[int, int, str, List[SkillCandidate]]] = field(
        compare=False, default_factory=list
    )


def beam_decode(
    candidates: List[int],
    T: int,
    scorer: SegmentScorer,
    observations: Sequence,
    actions: Sequence,
    predicates: Optional[List[Optional[dict]]] = None,
    config: Optional[SegmentationConfig] = None,
) -> SegmentationResult:
    """
    Run beam search over candidate boundaries.

    Parameters
    ----------
    candidates : list[int]
        Candidate boundary positions from Stage 1.
    T : int
        Total trajectory length.
    scorer : SegmentScorer
    observations, actions : Sequence
    predicates : list[dict], optional
    config : SegmentationConfig, optional

    Returns
    -------
    SegmentationResult
    """
    cfg = config or scorer.config
    skills = scorer.skill_names
    B = cfg.decoder.beam_width
    top_k = cfg.decoder.top_k_diagnostics
    top_m = cfg.decoder.top_m_skills
    max_segs = cfg.decoder.beam_max_segments

    boundary_set = sorted(set([0] + candidates + [T - 1]))
    if boundary_set[0] != 0:
        boundary_set = [0] + boundary_set

    use_batch = hasattr(scorer, "score_breakdown_batch") and callable(getattr(scorer, "score_breakdown_batch", None))

    # Seed beam: one entry starting at time 0, no skill yet
    beam: List[_BeamEntry] = [_BeamEntry(neg_score=0.0, last_cut=0, last_skill=None, path=[])]

    completed: List[_BeamEntry] = []

    for _ in range(len(boundary_set)):
        next_beam: List[_BeamEntry] = []

        for entry in beam:
            # Find next boundaries strictly after entry.last_cut
            for next_b in boundary_set:
                if next_b <= entry.last_cut:
                    continue

                seg_start = entry.last_cut if not entry.path else entry.last_cut + 1
                seg_end = next_b
                if seg_start > seg_end or seg_start >= T:
                    continue

                obs_slice = observations[seg_start : seg_end + 1]
                act_slice = actions[seg_start : seg_end + 1]
                p_start = predicates[seg_start] if predicates and seg_start < len(predicates) else None
                p_end = predicates[seg_end] if predicates and seg_end < len(predicates) else None

                if use_batch:
                    batch_reqs = [
                        (seg_start, seg_end, sk, entry.last_skill, obs_slice, act_slice, p_start, p_end)
                        for sk in skills
                    ]
                    breakdowns = scorer.score_breakdown_batch(batch_reqs)
                    scored = [(bd["total"], sk, bd) for sk, bd in zip(skills, breakdowns)]
                else:
                    scored = []
                    for sk in skills:
                        bd = scorer.score_breakdown(
                            seg_start, seg_end, sk, entry.last_skill,
                            obs_slice, act_slice, p_start, p_end,
                        )
                        scored.append((bd["total"], sk, bd))
                scored.sort(key=lambda x: -x[0])
                cands = [
                    SkillCandidate(skill=s[1], total_score=s[0], breakdown=s[2])
                    for s in scored[:top_k]
                ]

                for rank, (sc, sk, bd) in enumerate(scored):
                    if top_m is not None and rank >= top_m:
                        break

                    new_score = -entry.neg_score + sc
                    new_path = entry.path + [(seg_start, seg_end, sk, cands)]
                    new_entry = _BeamEntry(
                        neg_score=-new_score,
                        last_cut=next_b,
                        last_skill=sk,
                        path=new_path,
                    )

                    if next_b == boundary_set[-1]:
                        completed.append(new_entry)
                    else:
                        next_beam.append(new_entry)

        if not next_beam:
            break

        # Prune to top-B
        next_beam.sort()
        beam = next_beam[:B]

        if max_segs is not None and beam and len(beam[0].path) >= max_segs:
            completed.extend(beam)
            break

    # Pick best completed path
    if not completed:
        return SegmentationResult(total_score=float("-inf"))

    best = min(completed)  # min neg_score = max score

    segments: List[SegmentDiagnostic] = []
    for seg_start, seg_end, sk, cands in best.path:
        segments.append(
            SegmentDiagnostic(
                start=seg_start,
                end=seg_end,
                assigned_skill=sk,
                candidates=cands,
            )
        )

    return SegmentationResult(
        segments=segments,
        total_score=-best.neg_score,
    )
