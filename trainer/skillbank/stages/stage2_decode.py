"""
Stage 2: Decode — assign skill labels to segments via Viterbi/DP.

Given proposed boundary cuts and a current skill bank, find the best
assignment of skill labels (including NEW) to each segment. Uses dynamic
programming over candidate skills with effect-matching scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from trainer.skillbank.ingest_rollouts import TrajectoryForEM
from trainer.skillbank.stages.stage0_predicates import booleanize
from trainer.skillbank.stages.stage1_propose_cuts import CutCandidate, cuts_to_segments

logger = logging.getLogger(__name__)

NEW_LABEL = "__NEW__"


@dataclass
class DecodeConfig:
    """Configuration for the decode stage."""

    top_m_candidates: int = 10
    segment_min_len: int = 3
    segment_max_len: int = 100
    new_skill_penalty: float = 5.0
    eff_freq: float = 0.8
    p_thresh: float = 0.7


@dataclass
class DecodedSegment:
    """One decoded segment with assigned skill label and diagnostics."""

    t_start: int
    t_end: int
    skill_label: str = NEW_LABEL
    score: float = 0.0
    margin: float = 0.0
    runner_up: str = ""
    runner_up_score: float = 0.0
    B_start: Set[str] = field(default_factory=set)
    B_end: Set[str] = field(default_factory=set)
    eff_add: Set[str] = field(default_factory=set)
    eff_del: Set[str] = field(default_factory=set)
    confusers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "skill_label": self.skill_label,
            "score": self.score,
            "margin": self.margin,
            "runner_up": self.runner_up,
            "B_start": sorted(self.B_start),
            "B_end": sorted(self.B_end),
            "eff_add": sorted(self.eff_add),
            "eff_del": sorted(self.eff_del),
            "confusers": self.confusers,
        }


@dataclass
class DecodeResult:
    """Result of decoding a single trajectory."""

    traj_id: str = ""
    segments: List[DecodedSegment] = field(default_factory=list)
    total_score: float = 0.0
    n_new: int = 0
    n_known: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traj_id": self.traj_id,
            "n_segments": len(self.segments),
            "total_score": self.total_score,
            "n_new": self.n_new,
            "n_known": self.n_known,
            "segments": [s.to_dict() for s in self.segments],
        }


def decode_trajectory(
    trajectory: TrajectoryForEM,
    cuts: List[CutCandidate],
    bank: Any,
    config: Optional[DecodeConfig] = None,
) -> DecodeResult:
    """Assign skill labels to segments using effect-matching DP.

    For each segment, scores every skill in the bank by overlap between
    the segment's observed effects and the skill's contract effects.
    Assigns NEW if no skill scores well enough.

    Args:
        trajectory: enriched trajectory with predicates
        cuts: boundary candidates from Stage 1
        bank: SkillBankMVP with current contracts
        config: decode hyperparameters

    Returns:
        DecodeResult with labelled segments and diagnostics.
    """
    cfg = config or DecodeConfig()
    frames = trajectory.frames
    n = len(frames)

    seg_ranges = cuts_to_segments(cuts, n)
    skill_ids = list(getattr(bank, "skill_ids", []))
    decoded_segments: List[DecodedSegment] = []

    for (t_start, t_end) in seg_ranges:
        seg_len = t_end - t_start
        if seg_len < cfg.segment_min_len:
            continue

        B_start = booleanize(frames[t_start].predicates, cfg.p_thresh)
        B_end = booleanize(frames[min(t_end - 1, n - 1)].predicates, cfg.p_thresh)
        obs_eff_add = B_end - B_start
        obs_eff_del = B_start - B_end

        scores: List[Tuple[float, str]] = []
        for sid in skill_ids:
            contract = bank.get_contract(sid)
            if contract is None:
                continue
            s = _score_skill(obs_eff_add, obs_eff_del, contract, seg_len)
            scores.append((s, sid))

        new_score = -cfg.new_skill_penalty
        scores.append((new_score, NEW_LABEL))

        scores.sort(key=lambda x: -x[0])

        # Optional LLM-based re-ranking via RETRIEVAL adapter.
        # Boosts scores of skills that the LLM ranks highly.
        if skill_ids:
            try:
                from skill_agents.skill_bank.llm_retrieval import llm_retrieve_skills
                descs = {}
                for sid in skill_ids[:20]:
                    c = bank.get_contract(sid)
                    if c:
                        add = sorted(getattr(c, "eff_add", set()) or set())
                        descs[sid] = f"adds {add}" if add else sid
                state_dict = {"B_start": sorted(B_start), "eff_needed": sorted(obs_eff_add)}
                llm_ranking = llm_retrieve_skills(
                    query=f"segment t={t_start}-{t_end}",
                    current_state=state_dict,
                    skill_descriptions=descs,
                )
                if llm_ranking:
                    boost = 0.3
                    score_dict = {sid: sc for sc, sid in scores}
                    for rank, sid in enumerate(llm_ranking):
                        if sid in score_dict:
                            score_dict[sid] += boost / (rank + 1)
                    scores = [(sc, sid) for sid, sc in score_dict.items()]
                    scores.sort(key=lambda x: -x[0])
            except Exception:
                pass
        best_score, best_label = scores[0]
        runner_score, runner_label = scores[1] if len(scores) > 1 else (0.0, "")
        margin = best_score - runner_score

        confusers = [
            sid for s, sid in scores[1:4]
            if s > 0 and (best_score - s) < 1.0 and sid != NEW_LABEL
        ]

        seg = DecodedSegment(
            t_start=t_start,
            t_end=t_end,
            skill_label=best_label,
            score=best_score,
            margin=margin,
            runner_up=runner_label,
            runner_up_score=runner_score,
            B_start=B_start,
            B_end=B_end,
            eff_add=obs_eff_add,
            eff_del=obs_eff_del,
            confusers=confusers,
        )
        decoded_segments.append(seg)

    n_new = sum(1 for s in decoded_segments if s.skill_label == NEW_LABEL)
    total_score = sum(s.score for s in decoded_segments)

    return DecodeResult(
        traj_id=trajectory.traj_id,
        segments=decoded_segments,
        total_score=total_score,
        n_new=n_new,
        n_known=len(decoded_segments) - n_new,
    )


def _score_skill(
    obs_add: Set[str],
    obs_del: Set[str],
    contract: Any,
    seg_len: int,
) -> float:
    """Score a candidate skill against observed segment effects.

    Uses Jaccard-like overlap between observed and contracted effects.
    """
    c_add = getattr(contract, "eff_add", set()) or set()
    c_del = getattr(contract, "eff_del", set()) or set()

    if not c_add and not c_del:
        return 0.0

    add_overlap = len(obs_add & c_add) / max(len(obs_add | c_add), 1)
    del_overlap = len(obs_del & c_del) / max(len(obs_del | c_del), 1)

    score = 0.6 * add_overlap + 0.4 * del_overlap

    n_instances = getattr(contract, "n_instances", 0) or 0
    support_bonus = min(n_instances / 20.0, 0.5)
    score += 0.1 * support_bonus

    return score


def decode_batch(
    trajectories: List[TrajectoryForEM],
    cuts_per_traj: Dict[str, List[CutCandidate]],
    bank: Any,
    config: Optional[DecodeConfig] = None,
) -> List[DecodeResult]:
    """Decode a batch of trajectories."""
    results = []
    for traj in trajectories:
        traj_cuts = cuts_per_traj.get(traj.traj_id, [])
        result = decode_trajectory(traj, traj_cuts, bank, config)
        results.append(result)
    return results
