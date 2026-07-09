"""
Episode adapter: bridge between Stage 1 (boundary_proposal) and Stage 2 (InferSegmentation).

Pipeline:
  1. Stage 1 proposes candidate boundaries C.
  2. LLM teacher ranks skills for each candidate segment → pairwise preferences.
  3. Train a PreferenceScorer from those preferences (Bradley-Terry).
  4. Decode with the trained scorer (DP or beam) → skill sequence + diagnostics.
  5. On uncertain segments, query the LLM for more preferences → retrain → re-decode.
  6. Return SubTask_Experience segments with learned skill labels.

The LLM never produces numeric scores.  It only provides rankings/preferences.
All numeric scoring comes from the trained PreferenceScorer.

Usage:
    from skill_agents_grpo.infer_segmentation import infer_and_segment

    result, sub_episodes, store = infer_and_segment(
        episode,
        skill_names=["move", "attack", "gather", "craft"],
        env_name="overcooked",
    )
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

from skill_agents_grpo.infer_segmentation.config import SegmentationConfig
from skill_agents_grpo.infer_segmentation.scorer import SegmentScorer
from skill_agents_grpo.infer_segmentation.dp_decoder import viterbi_decode
from skill_agents_grpo.infer_segmentation.beam_decoder import beam_decode
from skill_agents_grpo.infer_segmentation.diagnostics import SegmentationResult
from skill_agents_grpo.infer_segmentation.preference import (
    PreferenceStore,
    PreferenceScorer,
    generate_preference_queries,
)

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


# ── GRPO episode context ─────────────────────────────────────────────
# Thread-local storage so concurrent segmentations (one per game in the
# co-evolution loop) don't overwrite each other's context.

_grpo_episode_ctx = threading.local()


def _set_grpo_episode_context(
    skill_names: List[str],
    config: "SegmentationConfig",
    intention_fit_fn: Optional[Callable] = None,
    compat_fn: Optional[Callable] = None,
) -> None:
    """Update the per-thread episode context used by :func:`grpo_scorer_factory`."""
    _grpo_episode_ctx.data = {
        "skill_names": skill_names,
        "config": config,
        "intention_fit_fn": intention_fit_fn,
        "compat_fn": compat_fn,
    }


def _get_grpo_episode_context() -> dict:
    return getattr(_grpo_episode_ctx, "data", {})


def grpo_scorer_factory(preference_list: list) -> "SegmentScorer":
    """Build a SegmentScorer from *preference_list* + current episode context.

    Designed to be passed to ``enable_segment_grpo`` / ``segmentation_reward``
    so that the GRPO reward evaluation reconstructs the same scorer (including
    ``intention_fit_fn``) that the main pipeline uses.
    """
    ctx = _get_grpo_episode_context()
    if not ctx:
        raise RuntimeError(
            "grpo_scorer_factory called but no episode context has been set. "
            "Ensure _set_grpo_episode_context() is called before the LLM "
            "teacher runs."
        )
    store = PreferenceStore()
    store.add_batch(preference_list)
    return _build_scorer_from_preferences(
        skill_names=ctx["skill_names"],
        store=store,
        config=ctx["config"],
        compat_fn=ctx.get("compat_fn"),
        intention_fit_fn=ctx.get("intention_fit_fn"),
    )


def grpo_decode_fn(
    scorer: "SegmentScorer",
    segments: list,
    observations: Sequence,
    actions: Sequence,
    skill_names: List[str],
    predicates: Optional[list],
) -> "SegmentationResult":
    """Thin decode wrapper for GRPO reward evaluation."""
    ctx = _get_grpo_episode_context()
    config = ctx.get("config") or SegmentationConfig()
    T = len(observations)
    candidates = sorted({pt for seg in segments for pt in [seg[0], seg[1]]})
    return _decode(candidates, T, scorer, observations, actions, predicates, config)


def _extract_obs_actions(experiences: list) -> Tuple[list, list]:
    """Pull observations and actions from Experience objects."""
    observations = []
    actions = []
    for exp in experiences:
        obs = getattr(exp, "summary_state", None) or getattr(exp, "summary", None) or getattr(exp, "state", None)
        observations.append(obs)
        actions.append(getattr(exp, "action", None))
    return observations, actions


def _extract_predicates(experiences: list) -> List[Optional[dict]]:
    """Build per-timestep predicate dicts from experiences.

    When an experience carries an ``intentions`` string with a ``[TAG]``
    prefix, we decompose it into ``tag_<tag>`` one-hot keys and a
    ``<tag>_completed`` flag so that ``_effects_compat_score`` can match
    intention-extracted contracts (Strategy B).
    """
    from skill_agents_grpo.boundary_proposal.signal_extractors import parse_intention_tag

    predicates = []
    for exp in experiences:
        preds: dict = {}

        if getattr(exp, "sub_tasks", None) is not None:
            preds["sub_task"] = exp.sub_tasks

        intent = getattr(exp, "intentions", None)
        if intent is not None:
            preds["intention"] = intent
            tag = parse_intention_tag(intent)
            if tag != "UNKNOWN":
                preds[f"tag_{tag.lower()}"] = 1.0
                preds[f"{tag.lower()}_completed"] = float(
                    getattr(exp, "done", False)
                )

        if getattr(exp, "done", None) is not None:
            preds["done"] = exp.done

        predicates.append(preds if preds else None)
    return predicates


def _build_intention_fit_fn(
    experiences: list,
    game_name: str = "generic",
    skill_names: Optional[List[str]] = None,
    skill_tags: Optional[Dict[str, List[str]]] = None,
) -> Optional["Callable[[str, int, int], float]"]:
    """Build a closure that scores intention-tag agreement for a segment.

    Uses the phase detector to produce per-step **compound labels**
    (``"phase:tag"``) so that the same raw tag in different game phases
    results in different skill assignments.

    Parameters
    ----------
    skill_tags : dict, optional
        Maps skill IDs to their associated tags (e.g.
        ``{"skill_tetris_clear_0": ["CLEAR"]}``).  When a bank-sourced
        skill (not a compound label) is scored, its tags are used to
        match against the per-step compound labels so that seeded skills
        integrate with the intention-fit signal.

    Returns ``None`` when no intention tags are available so the scorer
    degrades gracefully to LLM-only mode.
    """
    from skill_agents_grpo.boundary_proposal.signal_extractors import parse_intention_tag
    from skill_agents_grpo.infer_segmentation.phase_detector import (
        detect_phases,
        make_compound_label,
    )

    raw_tags: List[str] = []
    for exp in experiences:
        intent = getattr(exp, "intentions", None)
        tag = parse_intention_tag(intent) if intent else "UNKNOWN"
        raw_tags.append(tag)

    if all(t == "UNKNOWN" for t in raw_tags):
        return None

    phases = detect_phases(experiences, game_name=game_name)
    compound_labels = [
        make_compound_label(p, t) for p, t in zip(phases, raw_tags)
    ]

    # Guard: if BOTH raw tags AND compound labels are monotone, the
    # intention signal carries no information.  When raw tags are
    # monotone but phases add diversity (e.g. "early:SETUP" vs
    # "midgame:SETUP"), keep the signal — it still differentiates.
    _monotone_dampen = 1.0
    known_tags = [t for t in raw_tags if t != "UNKNOWN"]
    if known_tags:
        from collections import Counter
        tag_counts = Counter(known_tags)
        dominant_frac = tag_counts.most_common(1)[0][1] / len(known_tags)
        if dominant_frac > 0.9:
            known_compounds = [c for c in compound_labels if not c.endswith("UNKNOWN")]
            if known_compounds:
                compound_counts = Counter(known_compounds)
                compound_dominant = compound_counts.most_common(1)[0][1] / len(known_compounds)
            else:
                compound_dominant = 1.0
            if compound_dominant > 0.9:
                logger.debug(
                    "Intention tags are monotone (%.0f%% %s) and phases "
                    "add no diversity — disabling intention_fit",
                    dominant_frac * 100, tag_counts.most_common(1)[0][0],
                )
                return None
            _monotone_dampen = 0.5
            logger.debug(
                "Raw tags monotone (%.0f%% %s) but phases add diversity "
                "(%d unique compounds) — keeping intention_fit at %.1fx",
                dominant_frac * 100, tag_counts.most_common(1)[0][0],
                len(compound_counts), _monotone_dampen,
            )

    # Pre-compute which simple tags have compound variants among candidates.
    # When both "CLEAR" and "early:CLEAR" exist, the simple "CLEAR" must NOT
    # subsume compound-labeled steps — otherwise compound labels can never
    # win and skill diversity collapses to the simple labels only.
    _has_compound_variant: set = set()
    if skill_names:
        compound_skills = {s for s in skill_names if ":" in s}
        for cs in compound_skills:
            _tag = cs.split(":", 1)[1]
            if _tag in skill_names:
                _has_compound_variant.add(_tag)

    # Build a mapping from bank skill IDs to uppercase tag sets so that
    # seeded skills (e.g. "skill_tetris_clear_0" → tags ["CLEAR"]) match
    # compound labels containing "CLEAR".
    _skill_tag_map: Dict[str, set] = {}
    if skill_tags:
        for sid, tags in skill_tags.items():
            if tags:
                _skill_tag_map[sid] = {t.upper() for t in tags}

    def _intention_fit(skill: str, i: int, j: int) -> float:
        seg_labels = compound_labels[i : j + 1]
        length = len(seg_labels)
        if length == 0:
            return 0.0

        # Bank-sourced skills that aren't compound labels: match via tags
        if skill in _skill_tag_map:
            stags = _skill_tag_map[skill]
            matches = sum(
                1 for lb in seg_labels
                if any(
                    lb.upper() == t or lb.upper().endswith(f":{t}")
                    for t in stags
                )
            )
            match_frac = matches / length
            score = match_frac * length if match_frac > 0 else -0.3 * length
            return score * _monotone_dampen

        if ":" in skill:
            matches = sum(1 for lb in seg_labels if lb == skill)
        elif skill in _has_compound_variant:
            matches = sum(1 for lb in seg_labels if lb == skill)
        else:
            matches = sum(
                1 for lb in seg_labels
                if lb == skill or lb.endswith(f":{skill}")
            )
        match_frac = matches / length
        if match_frac > 0:
            return match_frac * length * _monotone_dampen
        # Stronger mismatch penalty when the correct phase-specific label
        # exists as a candidate — prevents behavior_fit from overriding
        # the intention signal and collapsing diversity.
        if _has_compound_variant and (":" in skill or skill in _has_compound_variant):
            return -2.0 * length * _monotone_dampen
        return -0.5 * length * _monotone_dampen
    return _intention_fit


def _build_scorer_from_preferences(
    skill_names: List[str],
    store: PreferenceStore,
    config: SegmentationConfig,
    compat_fn=None,
    intention_fit_fn=None,
) -> SegmentScorer:
    """Train a PreferenceScorer and wrap it in a SegmentScorer."""
    pref_scorer = PreferenceScorer(
        skill_names=skill_names,
        lr=config.preference.learning_rate,
    )
    if len(store) > 0:
        pref_scorer.train(store, epochs=config.preference.training_epochs)

    return SegmentScorer(
        skill_names=skill_names,
        config=config,
        behavior_fit_fn=pref_scorer.behavior_fit,
        transition_fn=pref_scorer.transition_prior,
        compat_fn=compat_fn,
        intention_fit_fn=intention_fit_fn,
    )


def _decode(
    candidates: List[int],
    T: int,
    scorer: SegmentScorer,
    observations: Sequence,
    actions: Sequence,
    predicates: Optional[List[Optional[dict]]],
    config: SegmentationConfig,
) -> SegmentationResult:
    """Run DP or beam decoder."""
    if config.method == "beam":
        return beam_decode(candidates, T, scorer, observations, actions, predicates, config)
    return viterbi_decode(candidates, T, scorer, observations, actions, predicates, config)


def _segments_to_sub_episodes(result, experiences: list, task, outcome_length: int) -> list:
    """Build SubTask_Experience list from segmentation result and episode experiences."""
    from data_structure.experience import SubTask_Experience
    T = len(experiences)
    sub_episodes = []
    for seg in result.segments:
        start, end = seg.start, min(seg.end + 1, T)
        segment_exps = experiences[start:end]
        outcome_start = end
        outcome_end = min(end + outcome_length, T)
        outcome_exps = (
            experiences[outcome_start:outcome_end]
            if outcome_start < outcome_end
            else None
        )
        sub_ep = SubTask_Experience(
            sub_task=seg.assigned_skill,
            final_goal=task,
            experiences=segment_exps,
            outcome=outcome_exps,
            seg_id=getattr(seg, "seg_id", None),
        )
        sub_episodes.append(sub_ep)
    return sub_episodes


# ── Low-level API ────────────────────────────────────────────────────

def infer_segmentation(
    candidates: List[int],
    T: int,
    skill_names: List[str],
    observations: Sequence,
    actions: Sequence,
    predicates: Optional[List[Optional[dict]]] = None,
    config: Optional[SegmentationConfig] = None,
    scorer: Optional[SegmentScorer] = None,
    behavior_fit_fn=None,
    transition_fn=None,
    duration_stats=None,
    compat_fn=None,
    intention_fit_fn=None,
) -> SegmentationResult:
    """
    Core InferSegmentation: run DP or beam search over candidates.

    If ``scorer`` is provided, uses it directly.
    Otherwise builds a SegmentScorer from the given functions.
    """
    cfg = config or SegmentationConfig()

    if scorer is not None:
        active_scorer = scorer
    else:
        active_scorer = SegmentScorer(
            skill_names=skill_names,
            config=cfg,
            behavior_fit_fn=behavior_fit_fn,
            transition_fn=transition_fn,
            duration_stats=duration_stats,
            compat_fn=compat_fn,
            intention_fit_fn=intention_fit_fn,
        )

    return _decode(candidates, T, active_scorer, observations, actions, predicates, cfg)


# ── High-level API ───────────────────────────────────────────────────

def infer_and_segment(
    episode,
    skill_names: List[str],
    env_name: str = "generic",
    config: Optional[SegmentationConfig] = None,
    proposal_config=None,
    embedder=None,
    surprisal=None,
    outcome_length: int = 5,
    preference_store: Optional[PreferenceStore] = None,
    extractor_kwargs=None,
    compat_fn=None,
    game_name: Optional[str] = None,
    skill_descriptions: Optional[Dict[str, str]] = None,
    skill_tags: Optional[Dict[str, List[str]]] = None,
    bank_skill_scores: Optional[Dict[str, float]] = None,
) -> Tuple[SegmentationResult, list, PreferenceStore]:
    """
    Inference pipeline with LLM (e.g. GPT-5):
      1. Stage 1 → candidate boundaries C.
      2. LLM teacher ranks skills for candidate segments → preferences.
      3. Train PreferenceScorer from preferences.
      4. Decode with trained scorer.
      5. Query LLM on uncertain segments → more preferences → retrain.
      6. Return segments + diagnostics + preference store.

    Use this when you want to interface with an LLM at inference time.

    Parameters
    ----------
    episode : Episode
        From data_structure.experience.
    skill_names : list[str]
        Known skill labels.
    env_name : str
        Passed to Stage 1 boundary proposal.
    config : SegmentationConfig, optional
    proposal_config : ProposalConfig, optional
    embedder : TextEmbedderBase, optional
    surprisal : np.ndarray, optional
    outcome_length : int
        Look-ahead steps for outcome window.
    preference_store : PreferenceStore, optional
        Existing preferences to bootstrap from.
    extractor_kwargs : dict, optional

    Returns
    -------
    (SegmentationResult, list[SubTask_Experience], PreferenceStore)
    """
    from data_structure.experience import SubTask_Experience
    from skill_agents_grpo.boundary_proposal import (
        propose_from_episode,
        candidate_centers_only,
    )
    from skill_agents_grpo.infer_segmentation.llm_teacher import (
        collect_segment_preferences,
        collect_transition_preferences,
        collect_uncertain_preferences,
    )

    cfg = config or SegmentationConfig()
    experiences = episode.experiences
    T = len(experiences)
    if T == 0:
        return SegmentationResult(), [], PreferenceStore()

    # Stage 1: boundary proposal
    boundary_candidates = propose_from_episode(
        episode,
        env_name=env_name,
        config=proposal_config,
        embedder=embedder,
        surprisal=surprisal,
        extractor_kwargs=extractor_kwargs,
    )
    centers = candidate_centers_only(boundary_candidates)

    _sources = {}
    for bc in boundary_candidates:
        for s in bc.source.split("+"):
            _sources[s] = _sources.get(s, 0) + 1
    logger.info(
        "Segmentation T=%d: %d boundary candidates (sources: %s), centers=%s",
        T, len(boundary_candidates), _sources, centers,
    )

    observations, actions = _extract_obs_actions(experiences)
    predicates = _extract_predicates(experiences)
    per_step_rewards = [getattr(exp, "reward", 0.0) or 0.0 for exp in experiences]
    episode_total_reward = sum(per_step_rewards)

    # Build segment list from boundaries
    cut_indices = sorted(set([0] + centers + [T - 1]))
    segments = []
    for idx in range(len(cut_indices) - 1):
        segments.append((cut_indices[idx], cut_indices[idx + 1]))
    if not segments:
        segments = [(0, T - 1)]

    logger.info(
        "Segmentation T=%d: %d segments, lengths=%s",
        T, len(segments), [e - s for s, e in segments],
    )

    store = preference_store or PreferenceStore()

    # Safety net: the pipeline should always provide ≥ 2 skill names
    # (via game-stage default seeds), but guard in case called directly.
    if len(skill_names) < 2:
        if "__NEW__" not in skill_names:
            skill_names = list(skill_names) + ["__NEW__"]
        if len(skill_names) < 2:
            skill_names = list(skill_names) + ["__EXPLORE__"]
        logger.warning(
            "skill_names had < 2 entries — padded to %d: %s.  "
            "This likely means game-stage seeds are missing from the pipeline.",
            len(skill_names), skill_names,
        )

    # ── Build intention-fit signal from per-step compound labels ────
    _game = game_name or env_name
    intention_fit_fn = _build_intention_fit_fn(
        experiences, game_name=_game, skill_names=skill_names,
        skill_tags=skill_tags,
    )

    # Update GRPO episode context so the scorer_factory used by the GRPO
    # reward function rebuilds an equivalent scorer (with intention_fit_fn).
    _set_grpo_episode_context(
        skill_names=skill_names,
        config=cfg,
        intention_fit_fn=intention_fit_fn,
        compat_fn=compat_fn,
    )

    # ── Collect preferences from LLM teacher ─────────────────────────
    # Re-collect when the store is empty (cold-start) OR when the skill
    # vocabulary has expanded beyond what the store covers.  Without this,
    # newly seeded skills have no preference data and the scorer assigns
    # them zero/default scores, so they can never win.
    unseen_skills = set(skill_names) - store.known_skills()
    if len(store) == 0 or unseen_skills:
        segment_prefs = collect_segment_preferences(
            segments, observations, actions, skill_names,
            predicates=predicates, config=cfg.llm_teacher,
            skill_descriptions=skill_descriptions,
            per_step_rewards=per_step_rewards,
            episode_total_reward=episode_total_reward,
            bank_skill_scores=bank_skill_scores,
        ) or []
        store.add_batch(segment_prefs)

        if cfg.preference.collect_transitions:
            transition_prefs = collect_transition_preferences(
                skill_names, config=cfg.llm_teacher,
                skill_descriptions=skill_descriptions,
            )
            store.add_batch(transition_prefs)

    # ── Train scorer and decode ─────────────────────────────────────
    scorer = _build_scorer_from_preferences(
        skill_names, store, cfg, compat_fn=compat_fn,
        intention_fit_fn=intention_fit_fn,
    )
    result = _decode(centers, T, scorer, observations, actions, predicates, cfg)

    # ── Active learning iterations ──────────────────────────────────
    for iteration in range(cfg.preference.num_iterations):
        new_prefs = collect_uncertain_preferences(
            result, observations, actions,
            margin_threshold=cfg.preference.margin_threshold,
            max_queries=cfg.preference.max_queries_per_iter,
            config=cfg.llm_teacher,
        )
        if not new_prefs:
            break

        store.add_batch(new_prefs)
        scorer = _build_scorer_from_preferences(
            skill_names, store, cfg, compat_fn=compat_fn,
            intention_fit_fn=intention_fit_fn,
        )
        result = _decode(centers, T, scorer, observations, actions, predicates, cfg)

    # ── Post-decode: relabel segments using intention-tag majority ──
    # The decoder picks labels to maximise a score that mixes LLM
    # preferences with intention tags.  When compound phase labels exist,
    # preferences can still dominate and collapse diversity.  Relabelling
    # each segment with the majority per-step compound label preserves
    # the decoder's *boundaries* (which are behaviourally meaningful)
    # while ensuring the *labels* reflect the ground-truth intention tags.
    if intention_fit_fn is not None:
        _relabel_segments_by_intention(
            result, experiences, skill_names, game_name=_game,
        )

    sub_episodes = _segments_to_sub_episodes(result, experiences, episode.task, outcome_length)
    return result, sub_episodes, store


def _relabel_segments_by_intention(
    result,
    experiences: list,
    skill_names: List[str],
    game_name: str = "generic",
) -> None:
    """Relabel decoded segments using majority compound intention label.

    Only relabels when the majority label exists in ``skill_names``
    and differs from the decoder's assignment.  Leaves the decoder's
    boundaries untouched.
    """
    from collections import Counter
    from skill_agents_grpo.boundary_proposal.signal_extractors import parse_intention_tag
    from skill_agents_grpo.infer_segmentation.phase_detector import (
        detect_phases,
        make_compound_label,
    )

    raw_tags = []
    for exp in experiences:
        intent = getattr(exp, "intentions", None)
        tag = parse_intention_tag(intent) if intent else "UNKNOWN"
        raw_tags.append(tag)

    if all(t == "UNKNOWN" for t in raw_tags):
        return

    phases = detect_phases(experiences, game_name=game_name)
    compound_labels = [
        make_compound_label(p, t) for p, t in zip(phases, raw_tags)
    ]

    skill_set = set(skill_names)
    T = len(experiences)
    for seg in result.segments:
        start = seg.start
        end = min(seg.end + 1, T)
        seg_labels = compound_labels[start:end]
        counts = Counter(lb for lb in seg_labels if lb != "UNKNOWN")
        if not counts:
            continue
        majority_label, majority_count = counts.most_common(1)[0]
        if majority_label in skill_set and majority_label != seg.assigned_skill:
            seg.assigned_skill = majority_label


def infer_and_segment_offline(
    episode,
    skill_names: List[str],
    env_name: str = "generic",
    config: Optional[SegmentationConfig] = None,
    proposal_config=None,
    embedder=None,
    surprisal=None,
    outcome_length: int = 5,
    behavior_fit_fn=None,
    transition_fn=None,
    duration_stats=None,
    extractor_kwargs=None,
    compat_fn=None,
    game_name: Optional[str] = None,
    skill_tags: Optional[Dict[str, List[str]]] = None,
) -> Tuple[SegmentationResult, list]:
    """
    Offline pipeline (no LLM): decode using provided scoring functions only.

    No LLM (e.g. GPT-5) is called. Use when (1) you have a pre-trained
    PreferenceScorer and want inference without API, (2) training/decoding
    from pre-collected preferences, or (3) testing without API keys.
    """
    from data_structure.experience import SubTask_Experience
    from skill_agents_grpo.boundary_proposal import (
        propose_from_episode,
        candidate_centers_only,
    )

    cfg = config or SegmentationConfig()
    experiences = episode.experiences
    T = len(experiences)
    if T == 0:
        return SegmentationResult(), []

    boundary_candidates = propose_from_episode(
        episode,
        env_name=env_name,
        config=proposal_config,
        embedder=embedder,
        surprisal=surprisal,
        extractor_kwargs=extractor_kwargs,
    )
    centers = candidate_centers_only(boundary_candidates)

    observations, actions = _extract_obs_actions(experiences)
    predicates = _extract_predicates(experiences)
    _game = game_name or env_name
    intention_fit_fn = _build_intention_fit_fn(
        experiences, game_name=_game, skill_names=skill_names,
        skill_tags=skill_tags,
    )

    result = infer_segmentation(
        candidates=centers,
        T=T,
        skill_names=skill_names,
        observations=observations,
        actions=actions,
        predicates=predicates,
        config=cfg,
        behavior_fit_fn=behavior_fit_fn,
        transition_fn=transition_fn,
        duration_stats=duration_stats,
        compat_fn=compat_fn,
        intention_fit_fn=intention_fit_fn,
    )
    if intention_fit_fn is not None:
        _game = game_name or env_name
        _relabel_segments_by_intention(
            result, experiences, skill_names, game_name=_game,
        )
    sub_episodes = _segments_to_sub_episodes(result, experiences, episode.task, outcome_length)
    return result, sub_episodes
