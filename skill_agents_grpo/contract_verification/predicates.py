"""
Step 1 — Build segment predicate summaries.

Given per-timestep predicate dicts (from Stage 1/2 extractors or a custom
``extract_predicates`` callable), compute robust P_start, P_end, P_all for
each segment and populate ``SegmentRecord`` objects.

Temporal smoothing uses majority-vote / average-probability over a configurable
window at segment boundaries to resist noisy single-frame readings.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from skill_agents_grpo.contract_verification.config import PredicateConfig
from skill_agents_grpo.contract_verification.schemas import SegmentRecord
from skill_agents_grpo.default_predicates import default_extract_predicates


# ── Core: build predicate summaries for a segment ───────────────────

def _average_predicate_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Average predicate probabilities across multiple timestep dicts."""
    if not dicts:
        return {}
    accum: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            accum[k] = accum.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: accum[k] / counts[k] for k in accum}


def _collect_timestep_predicates(
    observations: Sequence,
    t_start: int,
    t_end: int,
    extract_fn: Callable,
    cached_predicates: Optional[List[Optional[Dict[str, float]]]] = None,
) -> List[Dict[str, float]]:
    """Collect predicate dicts for timesteps [t_start..t_end] inclusive."""
    result: List[Dict[str, float]] = []
    for t in range(t_start, t_end + 1):
        if cached_predicates is not None and t < len(cached_predicates):
            preds = cached_predicates[t]
            if preds is not None:
                result.append({k: (float(v) if not isinstance(v, float) else v)
                               for k, v in preds.items()})
                continue
        if t < len(observations):
            preds = extract_fn(observations[t])
            result.append(preds if preds else {})
        else:
            result.append({})
    return result


def build_segment_predicates(
    seg_id: str,
    traj_id: str,
    t_start: int,
    t_end: int,
    skill_label: str,
    observations: Sequence,
    config: PredicateConfig,
    extract_fn: Callable = default_extract_predicates,
    cached_predicates: Optional[List[Optional[Dict[str, float]]]] = None,
    embedding: Optional[np.ndarray] = None,
) -> SegmentRecord:
    """Build a ``SegmentRecord`` with smoothed P_start, P_end, P_all.

    Parameters
    ----------
    seg_id, traj_id : str
        Identifiers for the segment and its parent trajectory.
    t_start, t_end : int
        Inclusive timestep range.
    skill_label : str
        Assigned skill id from Stage 2 (or ``"__NEW__"``).
    observations : Sequence
        Full trajectory observations (indexed by timestep).
    config : PredicateConfig
        Smoothing window and downsampling settings.
    extract_fn : callable
        ``extract_predicates(obs) -> {predicate: prob}``.
    cached_predicates : list, optional
        Pre-computed per-timestep predicate dicts (avoids re-extraction).
    embedding : np.ndarray, optional
        Segment embedding vector from Stage 2.
    """
    all_preds = _collect_timestep_predicates(
        observations, t_start, t_end, extract_fn, cached_predicates,
    )

    w = min(config.start_end_window, max(1, (t_end - t_start + 1) // 2))
    P_start = _average_predicate_dicts(all_preds[:w])
    P_end = _average_predicate_dicts(all_preds[-w:] if len(all_preds) >= w else all_preds)

    step = max(1, config.downsample_interval)
    P_all = all_preds[::step] if step > 1 else all_preds

    return SegmentRecord(
        seg_id=seg_id,
        traj_id=traj_id,
        t_start=t_start,
        t_end=t_end,
        skill_label=skill_label,
        P_start=P_start,
        P_end=P_end,
        P_all=P_all,
        embedding=embedding,
    )


# ── Batch: build records from Stage 2 SegmentationResult ────────────

def build_records_from_result(
    result,
    traj_id: str,
    observations: Sequence,
    config: PredicateConfig,
    extract_fn: Callable = default_extract_predicates,
    cached_predicates: Optional[List[Optional[Dict[str, float]]]] = None,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> List[SegmentRecord]:
    """Convert a Stage 2 ``SegmentationResult`` into ``SegmentRecord`` list.

    Parameters
    ----------
    result : SegmentationResult
        Output of Stage 2 decoder (has ``.segments`` list of ``SegmentDiagnostic``).
    traj_id : str
        Trajectory identifier.
    observations : Sequence
        Full trajectory observations.
    config : PredicateConfig
    extract_fn : callable
    cached_predicates : list, optional
    embeddings : dict, optional
        Mapping seg_id -> embedding vector.
    """
    records: List[SegmentRecord] = []
    for idx, seg in enumerate(result.segments):
        seg_id = f"{traj_id}_seg{idx:04d}"
        emb = embeddings.get(seg_id) if embeddings else None
        rec = build_segment_predicates(
            seg_id=seg_id,
            traj_id=traj_id,
            t_start=seg.start,
            t_end=seg.end,
            skill_label=seg.assigned_skill,
            observations=observations,
            config=config,
            extract_fn=extract_fn,
            cached_predicates=cached_predicates,
            embedding=emb,
        )
        records.append(rec)
    return records
