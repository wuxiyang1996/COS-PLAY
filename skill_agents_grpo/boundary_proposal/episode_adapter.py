"""
Episode adapter: bridge between the framework's Episode/Experience data
structures and the boundary proposal pipeline.

This is the main integration point.  Given an Episode, it:
  1. Extracts signals (predicates, events) via a SignalExtractor.
  2. Optionally computes embeddings and change-point scores via the RAG TextEmbedder.
  3. Runs the boundary proposal to get candidate cut points C.
  4. Segments the Episode into SubTask_Experience objects at those boundaries.

Signal extraction strategies (via ``env_name``):
  - ``"overcooked"`` etc.       — rule-based (fast, brittle keyword matching)
  - ``"llm"``                   — LLM-based (general, adaptive to any env)
  - ``"llm+overcooked"`` etc.   — hybrid: LLM predicates + per-env hard events (recommended)

Usage:
    from skill_agents_grpo.boundary_proposal.episode_adapter import segment_episode
    from skill_agents_grpo.boundary_proposal import ProposalConfig
    from data_structure.experience import Episode

    # Recommended: hybrid LLM + rule-based
    sub_episodes = segment_episode(
        episode,
        env_name="llm+overcooked",
        config=ProposalConfig(merge_radius=5),
        extractor_kwargs={"model": "gpt-4o-mini"},
    )

    # Or fully general (no per-env rules needed)
    sub_episodes = segment_episode(
        episode,
        env_name="llm",
        config=ProposalConfig(merge_radius=5),
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from skill_agents_grpo.boundary_proposal.proposal import (
    propose_boundary_candidates,
    BoundaryCandidate,
    ProposalConfig,
    candidate_centers_only,
)
from skill_agents_grpo.boundary_proposal.signal_extractors import (
    SignalExtractorBase,
    get_signal_extractor,
)
from skill_agents_grpo.boundary_proposal.changepoint import compute_changepoint_scores

# Allow importing data_structure from the repo root
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _get_summaries(experiences: list) -> List[Optional[str]]:
    """Get existing summaries from experiences (do NOT call LLM)."""
    return [
        getattr(exp, "summary", None) or getattr(exp, "summary_state", None)
        for exp in experiences
    ]


def _embed_summaries(
    summaries: List[str],
    embedder,
) -> np.ndarray:
    """
    Embed experience summaries using a TextEmbedderBase.

    Returns shape (n, dim) or empty (0, 0) if no summaries available.
    """
    valid = [s for s in summaries if s is not None and len(s.strip()) > 0]
    if not valid or embedder is None:
        return np.zeros((0, 0), dtype=np.float32)
    return embedder.encode(valid, prompt_name="passage")


def extract_signals(
    experiences: list,
    env_name: str = "generic",
    embedder=None,
    changepoint_method: str = "cusum",
    changepoint_drift: float = 0.05,
    changepoint_window: int = 10,
    extractor_kwargs: Optional[dict] = None,
) -> dict:
    """
    Extract all four signal types from an Experience list.

    Returns a dict ready to be unpacked into ``propose_boundary_candidates``:
      - predicates, event_times, changepoint_scores
      - plus the raw summaries/embeddings for downstream use.
    """
    ext = get_signal_extractor(env_name, **(extractor_kwargs or {}))
    predicates, event_times = ext.extract(experiences)

    # Reward spike events (merged into event_times)
    reward_events = ext.detect_reward_spike_events(experiences)
    event_times = sorted(set(event_times) | set(reward_events))

    # Embedding-based change-point scores
    changepoint_scores = None
    summaries = _get_summaries(experiences)
    if embedder is not None:
        valid_summaries = [s if s else "" for s in summaries]
        if any(s.strip() for s in valid_summaries):
            embeddings = embedder.encode(valid_summaries, prompt_name="passage")
            changepoint_scores = compute_changepoint_scores(
                embeddings,
                method=changepoint_method,
                drift=changepoint_drift,
                window_size=changepoint_window,
            )

    # Intention tags (e.g. "[DEFEND] Do something") — extract the [TAG] part
    intention_tags: Optional[list] = None
    done_flags: Optional[list] = None
    _raw_tags = []
    _raw_dones = []
    for exp in experiences:
        intent = getattr(exp, "intentions", None)
        if isinstance(intent, str) and intent:
            import re as _re
            m = _re.match(r"\[([A-Z_]+)\]", intent)
            _raw_tags.append(m.group(1) if m else intent.split("]")[0].lstrip("[") if "]" in intent else "")
        elif isinstance(intent, dict):
            _raw_tags.append(intent.get("tag", ""))
        else:
            _raw_tags.append("")
        _raw_dones.append(bool(getattr(exp, "done", False)))
    if any(t for t in _raw_tags):
        intention_tags = _raw_tags
        done_flags = _raw_dones

    return {
        "predicates": predicates,
        "event_times": event_times,
        "changepoint_scores": changepoint_scores,
        "intention_tags": intention_tags,
        "done_flags": done_flags,
        # Metadata (not passed to propose_boundary_candidates directly)
        "_summaries": summaries,
    }


def propose_from_episode(
    episode,
    env_name: str = "generic",
    config: Optional[ProposalConfig] = None,
    embedder=None,
    surprisal: Optional[np.ndarray] = None,
    changepoint_method: str = "cusum",
    changepoint_drift: float = 0.05,
    changepoint_window: int = 10,
    event_window: int = 1,
    extractor_kwargs: Optional[dict] = None,
) -> List[BoundaryCandidate]:
    """
    Run the full Stage 1 pipeline on an Episode.

    Parameters
    ----------
    episode : Episode
        From data_structure.experience.
    env_name : str
        "overcooked", "avalon", "diplomacy", or "generic".
    config : ProposalConfig, optional
    embedder : TextEmbedderBase, optional
        If provided, computes embedding change-point scores from summaries.
    surprisal : np.ndarray, optional
        Pre-computed action surprisal; shape (T,).  If None, skipped.
    changepoint_method / changepoint_drift / changepoint_window
        Passed to compute_changepoint_scores.
    event_window : int
        Passed to propose_boundary_candidates.
    extractor_kwargs : dict, optional
        Extra kwargs for the signal extractor (e.g. controlled_power).

    Returns
    -------
    list[BoundaryCandidate]
    """
    experiences = episode.experiences
    T = len(experiences)
    if T == 0:
        return []

    signals = extract_signals(
        experiences,
        env_name=env_name,
        embedder=embedder,
        changepoint_method=changepoint_method,
        changepoint_drift=changepoint_drift,
        changepoint_window=changepoint_window,
        extractor_kwargs=extractor_kwargs,
    )

    # Phase-transition boundaries: detect game phases and add transitions
    # as high-priority event triggers so the decoder sees finer segments.
    from skill_agents_grpo.infer_segmentation.phase_detector import detect_phases
    phase_events = []
    try:
        phases = detect_phases(experiences, game_name=env_name)
        for t in range(1, len(phases)):
            if phases[t] != phases[t - 1]:
                phase_events.append(t)
    except Exception:
        pass

    all_events = sorted(set((signals["event_times"] or []) + phase_events))

    candidates = propose_boundary_candidates(
        T,
        predicates=signals["predicates"],
        surprisal=surprisal,
        changepoint_scores=signals["changepoint_scores"],
        event_times=all_events if all_events else signals["event_times"],
        intention_tags=signals.get("intention_tags"),
        done_flags=signals.get("done_flags"),
        config=config,
        event_window=event_window,
    )
    return candidates


def segment_episode(
    episode,
    env_name: str = "generic",
    config: Optional[ProposalConfig] = None,
    embedder=None,
    surprisal: Optional[np.ndarray] = None,
    outcome_length: int = 5,
    changepoint_method: str = "cusum",
    changepoint_drift: float = 0.05,
    changepoint_window: int = 10,
    event_window: int = 1,
    extractor_kwargs: Optional[dict] = None,
) -> list:
    """
    Full pipeline: propose boundaries, then segment Episode into SubTask_Experience objects.

    This replaces or augments ``Episode.separate_into_sub_episodes()`` for the
    unsupervised case where sub_task labels are NOT pre-set.

    Parameters
    ----------
    episode : Episode
        From data_structure.experience.
    env_name / config / embedder / surprisal / event_window / extractor_kwargs
        Passed to propose_from_episode.
    outcome_length : int
        Number of look-ahead steps for outcome window (same semantics as
        Episode.separate_into_sub_episodes).

    Returns
    -------
    list[SubTask_Experience]
        Segments of the episode, one per inter-boundary interval.
        Each segment's sub_task is set to "segment_{i}" (a placeholder for
        downstream LLM-based labeling).
    """
    # Lazy import to avoid circular dependency
    from data_structure.experience import SubTask_Experience

    candidates = propose_from_episode(
        episode,
        env_name=env_name,
        config=config,
        embedder=embedder,
        surprisal=surprisal,
        changepoint_method=changepoint_method,
        changepoint_drift=changepoint_drift,
        changepoint_window=changepoint_window,
        event_window=event_window,
        extractor_kwargs=extractor_kwargs,
    )

    T = len(episode.experiences)
    centers = candidate_centers_only(candidates)

    # Build cut indices: [0, c1, c2, ..., T]
    cut_indices = sorted(set([0] + centers + [T]))

    sub_episodes = []
    for i in range(len(cut_indices) - 1):
        start = cut_indices[i]
        end = cut_indices[i + 1]
        if start >= end:
            continue

        segment_exps = episode.experiences[start:end]
        # Outcome: look-ahead window after the segment
        outcome_start = end
        outcome_end = min(end + outcome_length, T)
        outcome_exps = (
            episode.experiences[outcome_start:outcome_end]
            if outcome_start < outcome_end
            else None
        )

        sub_ep = SubTask_Experience(
            sub_task=f"segment_{i}",
            final_goal=episode.task,
            experiences=segment_exps,
            outcome=outcome_exps,
        )
        sub_episodes.append(sub_ep)

    return sub_episodes


def annotate_episode_boundaries(
    episode,
    env_name: str = "generic",
    config: Optional[ProposalConfig] = None,
    embedder=None,
    surprisal: Optional[np.ndarray] = None,
    event_window: int = 1,
    extractor_kwargs: Optional[dict] = None,
) -> list:
    """
    Annotate each Experience in the Episode with sub_task_done = True at
    proposed boundaries.  This lets the existing
    ``Episode.separate_into_sub_episodes()`` work with the proposed cuts.

    Returns the list of BoundaryCandidates (for inspection / logging).
    """
    candidates = propose_from_episode(
        episode,
        env_name=env_name,
        config=config,
        embedder=embedder,
        surprisal=surprisal,
        event_window=event_window,
        extractor_kwargs=extractor_kwargs,
    )
    centers = set(candidate_centers_only(candidates))

    segment_idx = 0
    for t, exp in enumerate(episode.experiences):
        if t in centers:
            # Mark the previous experience as sub_task_done
            if t > 0:
                episode.experiences[t - 1].sub_task_done = True
            # Label segment start
            if exp.sub_tasks is None:
                exp.sub_tasks = f"segment_{segment_idx}"
                segment_idx += 1
        elif exp.sub_tasks is None and segment_idx > 0:
            exp.sub_tasks = f"segment_{segment_idx - 1}"

    return candidates
