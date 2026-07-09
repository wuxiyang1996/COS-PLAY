"""
LLM preference teacher: the LLM provides skill rankings (not numeric scores).

The LLM is asked to **rank** candidate skills for a segment, and to **rank**
likely next-skills given a previous skill.  These rankings are converted to
pairwise preferences and stored for training.

The LLM never produces numeric scores.  Numeric scores come from the
``PreferenceScorer`` in ``preference.py``, which is *trained* on these
collected preferences.

Two collection modes:
  1. **Proactive** — ``collect_segment_preferences``: for every segment in a
     trajectory, ask the LLM to rank skills.  Used for initial cold-start.
  2. **Active** — ``collect_uncertain_preferences``: only query segments where
     the current scorer is uncertain (low margin).  Used in the iterative loop.

GRPO wrapping: ``enable_segment_grpo()`` wraps ``collect_segment_preferences``
at the batch level (per episode). G complete preference sets are generated,
each evaluated via scorer-rebuild + decode, and the best is returned.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import logging
import time as _time
from dataclasses import dataclass, field, asdict

from skill_agents_grpo.infer_segmentation.config import LLMTeacherConfig

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

logger = logging.getLogger(__name__)

# ── Global concurrency limiter for LLM teacher calls ─────────────────
# When multiple episodes are segmented in parallel (each spawning its own
# ThreadPoolExecutor), the per-episode semaphore from _wrap_ask_with_semaphore
# does NOT limit cross-episode concurrency.  Without a global cap, N episodes
# × M workers can flood the vLLM servers with hundreds of simultaneous
# requests, causing queue buildup and effective deadlock.
#
# This module-level semaphore caps the TOTAL in-flight LLM teacher calls
# across all episodes/threads in this process.
_GLOBAL_MAX_CONCURRENT_LLM_CALLS = int(
    os.environ.get("SKILLBANK_GLOBAL_MAX_LLM_CALLS", "32")
)
_global_llm_semaphore = threading.Semaphore(_GLOBAL_MAX_CONCURRENT_LLM_CALLS)


# ── Cold-start I/O recording ─────────────────────────────────────────
# Every LLM teacher call is recorded so that (prompt, response) pairs
# can serve as supervised fine-tuning data for Qwen3-8B cold-start,
# and as reference outputs for GRPO reward comparison.

@dataclass
class TeacherIORecord:
    """One LLM teacher call with full prompt/response for cold-start replay."""
    function: str                             # segment_ranking | transition_ranking | pairwise_choice | skill_naming
    prompt: str = ""
    response: str = ""
    parsed: Optional[dict] = None             # structured output (ranking list, choice, name)
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    elapsed_s: float = 0.0
    # Segment-level context (when applicable)
    segment_start: Optional[int] = None
    segment_end: Optional[int] = None
    skill_names: List[str] = field(default_factory=list)
    prev_skill: Optional[str] = None          # for transition rankings
    skill_a: Optional[str] = None             # for pairwise
    skill_b: Optional[str] = None             # for pairwise
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None and v != "" and v != [] and v != 0 and v != 0.0}
        if "function" not in d:
            d["function"] = self.function
        return d


_teacher_io_records: List[TeacherIORecord] = []
_teacher_io_lock = threading.Lock()


def _record_teacher_io(rec: TeacherIORecord) -> None:
    """Thread-safe append to the module-level recording buffer."""
    with _teacher_io_lock:
        _teacher_io_records.append(rec)


def get_teacher_io_records() -> List[dict]:
    """Return all accumulated records as dicts (non-destructive)."""
    with _teacher_io_lock:
        return [r.to_dict() for r in _teacher_io_records]


def flush_teacher_io_records() -> List[dict]:
    """Return and clear all accumulated records."""
    with _teacher_io_lock:
        out = [r.to_dict() for r in _teacher_io_records]
        _teacher_io_records.clear()
        return out


def reset_teacher_io_records() -> None:
    """Clear all accumulated records without returning them."""
    with _teacher_io_lock:
        _teacher_io_records.clear()


def _get_ask_model():
    """Lazy import to avoid pulling in API dependencies at module load.

    Prefers the LoRA segment adapter when available, otherwise falls back
    to the API-based ``ask_model``.  The returned callable is wrapped for
    reasoning-model compatibility (Qwen3 ``/no_think``, think-tag stripping).

    The returned callable is always wrapped with the global concurrency
    semaphore so that cross-episode parallelism cannot flood the vLLM
    servers.
    """
    from skill_agents_grpo._llm_compat import wrap_ask_for_reasoning_models

    _hint = "Qwen/Qwen3-8B"
    raw_ask = None
    try:
        from skill_agents_grpo.lora import MultiLoraSkillBankLLM, SkillFunction
        llm = MultiLoraSkillBankLLM.get_shared_instance()
        if llm is not None:
            raw_ask = wrap_ask_for_reasoning_models(
                llm.as_ask_fn(SkillFunction.SEGMENT), model_hint=_hint,
            )
    except Exception:
        pass
    if raw_ask is None:
        from API_func import ask_model
        raw_ask = wrap_ask_for_reasoning_models(ask_model, model_hint=_hint)

    def _globally_limited_ask(*args, **kwargs):
        with _global_llm_semaphore:
            return raw_ask(*args, **kwargs)

    return _globally_limited_ask


def _wrap_ask_with_semaphore(ask: Callable, max_concurrent: int):
    """Wrap ask so at most max_concurrent calls run at once (e.g. for local GPU)."""
    sem = threading.Semaphore(max_concurrent)

    def limited_ask(*args, **kwargs):
        with sem:
            return ask(*args, **kwargs)

    return limited_ask


def _parse_json_from_response(response: str) -> Optional[dict]:
    """Extract JSON from an LLM response that may contain extra text."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _fuzzy_match_skill(name: str, skill_names: List[str]) -> Optional[str]:
    """Match an LLM-produced skill name to canonical skill names.

    Handles common LLM quirks: extra whitespace, casing differences,
    surrounding quotes, underscores vs spaces.
    """
    cleaned = name.strip().strip('"').strip("'").strip()
    if cleaned in skill_names:
        return cleaned
    lower_map = {s.lower(): s for s in skill_names}
    if cleaned.lower() in lower_map:
        return lower_map[cleaned.lower()]
    normed = cleaned.lower().replace(" ", "_")
    normed_map = {s.lower().replace(" ", "_"): s for s in skill_names}
    if normed in normed_map:
        return normed_map[normed]
    return None


# ── Ranking prompts ──────────────────────────────────────────────────

def _format_skill_candidates(
    skill_names: List[str],
    skill_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Format candidate skills, enriching with descriptions when available."""
    if not skill_descriptions:
        return "[" + ", ".join(f'"{s}"' for s in skill_names) + "]"
    lines: List[str] = []
    for sid in skill_names:
        desc = skill_descriptions.get(sid, "")
        if desc:
            lines.append(f'  - "{sid}": {desc[:150]}')
        else:
            lines.append(f'  - "{sid}"')
    return "\n".join(lines)


def _build_segment_ranking_prompt(
    observations: Sequence,
    actions: Sequence,
    skill_names: List[str],
    segment_start: int,
    segment_end: int,
    predicates_start: Optional[dict] = None,
    predicates_end: Optional[dict] = None,
    skill_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """
    Ask the LLM to rank skills for one segment (best fit first).

    When *skill_descriptions* is provided (mapping ``skill_id`` to a
    short summary), each candidate is shown with its description so the
    model can make an informed ranking.
    """
    obs_str = str(list(observations))
    act_str = str(list(actions))
    length = len(observations)
    candidates_block = _format_skill_candidates(skill_names, skill_descriptions)
    preds_s = str(predicates_start) if predicates_start else "N/A"
    preds_e = str(predicates_end) if predicates_end else "N/A"

    return (
        f"You are an expert at recognizing skills in agent trajectories.\n"
        f"\n"
        f"A trajectory segment spans timesteps {segment_start} to {segment_end} "
        f"(length {length}).\n"
        f"\n"
        f"Observations:\n{obs_str}\n"
        f"\n"
        f"Actions:\n{act_str}\n"
        f"\n"
        f"State at segment start: {preds_s}\n"
        f"State at segment end:   {preds_e}\n"
        f"\n"
        f"Candidate skills:\n{candidates_block}\n"
        f"\n"
        f"Rank ALL candidate skills from best fit to worst fit for this "
        f"segment.  Consider:\n"
        f"  - Do the actions match what this skill would produce?\n"
        f"  - Is the segment length reasonable for this skill?\n"
        f"  - Is the state change consistent with this skill's purpose?\n"
        f"\n"
        f"Return ONLY a JSON object (no extra text):\n"
        f'{{"ranking": ["best_skill", "second_best", ...], '
        f'"reasoning": "brief explanation"}}\n'
    )


def _build_transition_ranking_prompt(
    prev_skill: str,
    skill_names: List[str],
    skill_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """
    Ask the LLM to rank which skills most naturally follow prev_skill.
    """
    candidates_block = _format_skill_candidates(skill_names, skill_descriptions)
    prev_desc = ""
    if skill_descriptions and prev_skill in skill_descriptions:
        prev_desc = f" ({skill_descriptions[prev_skill][:100]})"

    return (
        f"You are an expert at modeling skill sequences in agent trajectories.\n"
        f"\n"
        f"The agent just finished executing skill: \"{prev_skill}\"{prev_desc}\n"
        f"\n"
        f"Candidate next skills:\n{candidates_block}\n"
        f"\n"
        f"Rank ALL candidate skills from most likely to follow to least "
        f"likely.  Consider natural task ordering, common behavior patterns, "
        f"and logical dependencies between skills.\n"
        f"\n"
        f"Return ONLY a JSON object (no extra text):\n"
        f'{{"ranking": ["most_likely", "second", ...], '
        f'"reasoning": "brief explanation"}}\n'
    )


def _build_pairwise_prompt(
    observations: Sequence,
    actions: Sequence,
    skill_a: str,
    skill_b: str,
    segment_start: int,
    segment_end: int,
    skill_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """
    Ask the LLM which of two skills better fits this segment.
    Used for active learning on uncertain segments.
    """
    obs_str = str(list(observations))
    act_str = str(list(actions))

    desc_a = f" ({skill_descriptions[skill_a][:100]})" if skill_descriptions and skill_a in skill_descriptions else ""
    desc_b = f" ({skill_descriptions[skill_b][:100]})" if skill_descriptions and skill_b in skill_descriptions else ""

    return (
        f"You are an expert at recognizing skills in agent trajectories.\n"
        f"\n"
        f"Segment: timesteps {segment_start} to {segment_end}\n"
        f"Observations:\n{obs_str}\n"
        f"Actions:\n{act_str}\n"
        f"\n"
        f"Which skill better explains this segment?\n"
        f"  A: \"{skill_a}\"{desc_a}\n"
        f"  B: \"{skill_b}\"{desc_b}\n"
        f"\n"
        f"Return ONLY a JSON object:\n"
        f'{{"choice": "A" or "B", "evidence": "brief explanation"}}\n'
    )


# ── Ranking → pairwise conversion ────────────────────────────────────

def ranking_to_pairwise(
    ranking: List[str],
    segment_start: int,
    segment_end: int,
    source: str = "llm",
    evidence: str = "",
) -> list:
    """
    Convert a ranked list [best, ..., worst] into pairwise preferences.

    For a ranking [A, B, C] produces: A≻B, A≻C, B≻C.
    """
    from skill_agents_grpo.infer_segmentation.preference import PreferenceExample

    pairs = []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            pairs.append(PreferenceExample(
                segment_start=segment_start,
                segment_end=segment_end,
                skill_win=ranking[i],
                skill_lose=ranking[j],
                evidence=evidence,
                source=source,
            ))
    return pairs


# ── Public API ────────────────────────────────────────────────────────

def _collect_one_segment_prefs(
    start: int,
    end: int,
    observations: Sequence,
    actions: Sequence,
    skill_names: List[str],
    predicates: Optional[List[Optional[dict]]],
    cfg: LLMTeacherConfig,
    ask,
    *,
    skill_descriptions: Optional[Dict[str, str]] = None,
    raw_sink: Optional[Dict[Tuple[int, int], str]] = None,
) -> list:
    """Get pairwise preferences for a single segment (used by batch workers)."""
    seg_obs = observations[start: end + 1]
    seg_act = actions[start: end + 1]
    p_start = predicates[start] if predicates and start < len(predicates) else None
    p_end = predicates[end] if predicates and end < len(predicates) else None
    prompt = _build_segment_ranking_prompt(
        seg_obs, seg_act, skill_names, start, end, p_start, p_end,
        skill_descriptions=skill_descriptions,
    )
    t0 = _time.time()
    response = ask(
        prompt, model=cfg.model,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )
    elapsed = _time.time() - t0
    if raw_sink is not None:
        raw_sink[(start, end)] = response if isinstance(response, str) else str(response)
    parsed = _parse_json_from_response(response)

    _record_teacher_io(TeacherIORecord(
        function="segment_ranking",
        prompt=prompt,
        response=response or "",
        parsed=parsed,
        model=cfg.model or "",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        elapsed_s=round(elapsed, 3),
        segment_start=start,
        segment_end=end,
        skill_names=list(skill_names),
        error=None if parsed and "ranking" in parsed else "parse_failed",
    ))

    if not parsed or "ranking" not in parsed:
        return []
    raw_ranking = parsed["ranking"] if isinstance(parsed["ranking"], list) else []
    ranking = []
    seen = set()
    for s in raw_ranking:
        matched = _fuzzy_match_skill(str(s), skill_names)
        if matched and matched not in seen:
            ranking.append(matched)
            seen.add(matched)
    evidence = parsed.get("reasoning", "")
    if len(ranking) < 2:
        return []
    return ranking_to_pairwise(ranking, start, end, "llm", evidence)


def collect_segment_preferences(
    segments: List[Tuple[int, int]],
    observations: Sequence,
    actions: Sequence,
    skill_names: List[str],
    predicates: Optional[List[Optional[dict]]] = None,
    config: Optional[LLMTeacherConfig] = None,
    skill_descriptions: Optional[Dict[str, str]] = None,
    per_step_rewards: Optional[List[float]] = None,
    episode_total_reward: Optional[float] = None,
    **_kw: Any,
) -> list:
    """
    Proactive preference collection: ask the LLM to rank skills for
    each segment.  Used for cold-start before any scorer is trained.

    When *skill_descriptions* is provided, each candidate skill is shown
    with its name and strategic description so the model can make an
    informed ranking.

    Parameters
    ----------
    segments : list[(start, end)]
        Segment boundaries (inclusive).
    observations, actions : Sequence
        Full trajectory data.
    skill_names : list[str]
        Candidate skills to rank.
    predicates : list[dict], optional
        Per-timestep predicates.
    config : LLMTeacherConfig, optional
    skill_descriptions : dict, optional
        Mapping ``skill_id`` → short summary string.
    per_step_rewards : list[float], optional
        Per-timestep game rewards from the episode.
    episode_total_reward : float, optional
        Sum of all per-step rewards in the episode.

    Returns
    -------
    PreferenceListWithRollouts | list
        Pairwise preferences derived from LLM rankings, normally wrapped as
        ``PreferenceListWithRollouts`` (a ``list`` subclass) carrying
        ``raw_rollouts`` — one raw LLM string per segment — for GRPO rewards.
    """
    from skill_agents_grpo.infer_segmentation.preference import PreferenceListWithRollouts

    if len(skill_names) < 2:
        # With < 2 skills there is nothing to rank.  Return the typed container
        # (not bare []) so GRPO fingerprinting never crashes on missing attribute.
        return PreferenceListWithRollouts([], raw_rollouts=[])
    cfg = config or LLMTeacherConfig()
    if "temperature" in _kw:
        from copy import copy
        cfg = copy(cfg)
        cfg.temperature = _kw["temperature"]
    ask = _get_ask_model()
    raw_by_seg: Dict[Tuple[int, int], str] = {}

    def _ordered_rollouts() -> List[str]:
        return [raw_by_seg[k] for k in sorted(raw_by_seg.keys())]

    max_workers = getattr(cfg, "max_workers", None)
    if max_workers is None or max_workers <= 1:
        all_prefs = []
        for start, end in segments:
            prefs = _collect_one_segment_prefs(
                start, end, observations, actions, skill_names, predicates, cfg, ask,
                skill_descriptions=skill_descriptions,
                raw_sink=raw_by_seg,
            )
            all_prefs.extend(prefs)
        return PreferenceListWithRollouts(all_prefs, raw_rollouts=_ordered_rollouts())

    max_concurrent = getattr(cfg, "max_concurrent_llm_calls", None)
    if max_concurrent is not None and max_concurrent > 0:
        ask = _wrap_ask_with_semaphore(ask, max_concurrent)
    all_prefs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _collect_one_segment_prefs,
                start, end, observations, actions, skill_names, predicates, cfg, ask,
                skill_descriptions=skill_descriptions,
                raw_sink=raw_by_seg,
            ): (start, end)
            for start, end in segments
        }
        for future in as_completed(futures):
            try:
                prefs = future.result()
                all_prefs.extend(prefs)
            except Exception:
                pass  # skip failed segments
    return PreferenceListWithRollouts(all_prefs, raw_rollouts=_ordered_rollouts())


def _collect_one_transition_prefs(
    prev_skill: str,
    skill_names: List[str],
    cfg: LLMTeacherConfig,
    ask,
    skill_descriptions: Optional[Dict[str, str]] = None,
) -> list:
    """Get transition preferences for one prev_skill (used by batch workers)."""
    from skill_agents_grpo.infer_segmentation.preference import PreferenceExample

    prompt = _build_transition_ranking_prompt(
        prev_skill, skill_names, skill_descriptions=skill_descriptions,
    )
    t0 = _time.time()
    response = ask(
        prompt, model=cfg.model,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )
    elapsed = _time.time() - t0
    parsed = _parse_json_from_response(response)

    _record_teacher_io(TeacherIORecord(
        function="transition_ranking",
        prompt=prompt,
        response=response or "",
        parsed=parsed,
        model=cfg.model or "",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        elapsed_s=round(elapsed, 3),
        prev_skill=prev_skill,
        skill_names=list(skill_names),
        error=None if parsed and "ranking" in parsed else "parse_failed",
    ))

    if not parsed or "ranking" not in parsed:
        return []
    raw_ranking = parsed["ranking"] if isinstance(parsed["ranking"], list) else []
    ranking = []
    seen: set = set()
    for s in raw_ranking:
        matched = _fuzzy_match_skill(str(s), skill_names)
        if matched and matched not in seen:
            ranking.append(matched)
            seen.add(matched)
    evidence = parsed.get("reasoning", "")
    prefs = []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            prefs.append(PreferenceExample(
                segment_start=-1,
                segment_end=-1,
                skill_win=f"{prev_skill}->{ranking[i]}",
                skill_lose=f"{prev_skill}->{ranking[j]}",
                evidence=evidence,
                source="llm",
            ))
    return prefs


def collect_transition_preferences(
    skill_names: List[str],
    config: Optional[LLMTeacherConfig] = None,
    skill_descriptions: Optional[Dict[str, str]] = None,
) -> list:
    """
    Ask the LLM to rank transition likelihoods for each skill.

    For each skill as prev_skill, collects pairwise preferences on
    which next-skill is more likely. When config.max_workers > 1,
    LLM calls run in parallel.
    """
    from skill_agents_grpo.infer_segmentation.preference import PreferenceExample

    cfg = config or LLMTeacherConfig()
    ask = _get_ask_model()
    max_workers = getattr(cfg, "max_workers", None)
    if max_workers is None or max_workers <= 1:
        all_prefs = []
        for prev_skill in skill_names:
            all_prefs.extend(
                _collect_one_transition_prefs(
                    prev_skill, skill_names, cfg, ask, skill_descriptions,
                ),
            )
        return all_prefs

    max_concurrent = getattr(cfg, "max_concurrent_llm_calls", None)
    if max_concurrent is not None and max_concurrent > 0:
        ask = _wrap_ask_with_semaphore(ask, max_concurrent)
    all_prefs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _collect_one_transition_prefs,
                ps, skill_names, cfg, ask, skill_descriptions,
            ): ps
            for ps in skill_names
        }
        for future in as_completed(futures):
            try:
                all_prefs.extend(future.result())
            except Exception:
                pass
    return all_prefs


def _collect_one_uncertain_pref(
    seg,
    observations: Sequence,
    actions: Sequence,
    cfg: LLMTeacherConfig,
    ask,
) -> Optional[object]:
    """Get one A/B preference for an uncertain segment (used by batch workers)."""
    from skill_agents_grpo.infer_segmentation.preference import PreferenceExample

    if len(seg.candidates) < 2:
        return None
    skill_a = seg.candidates[0].skill
    skill_b = seg.candidates[1].skill
    seg_obs = observations[seg.start: seg.end + 1]
    seg_act = actions[seg.start: seg.end + 1]
    prompt = _build_pairwise_prompt(
        seg_obs, seg_act, skill_a, skill_b, seg.start, seg.end,
    )
    t0 = _time.time()
    response = ask(
        prompt, model=cfg.model,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )
    elapsed = _time.time() - t0
    parsed = _parse_json_from_response(response)

    _record_teacher_io(TeacherIORecord(
        function="pairwise_choice",
        prompt=prompt,
        response=response or "",
        parsed=parsed,
        model=cfg.model or "",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        elapsed_s=round(elapsed, 3),
        segment_start=seg.start,
        segment_end=seg.end,
        skill_a=skill_a,
        skill_b=skill_b,
        error=None if parsed else "parse_failed",
    ))

    if not parsed:
        return None
    choice = parsed.get("choice", "").upper().strip()
    evidence = parsed.get("evidence", "")
    if choice == "A":
        win, lose = skill_a, skill_b
    elif choice == "B":
        win, lose = skill_b, skill_a
    else:
        return None
    return PreferenceExample(
        segment_start=seg.start,
        segment_end=seg.end,
        skill_win=win,
        skill_lose=lose,
        evidence=evidence,
        source="llm",
    )


# ── Naming new skills ─────────────────────────────────────────────────

def _build_skill_naming_prompt(
    observation_slices: List[Sequence],
    eff_add: Optional[Sequence[str]] = None,
    eff_del: Optional[Sequence[str]] = None,
    eff_event: Optional[Sequence[str]] = None,
) -> str:
    """Build a prompt for the LLM to suggest a human-readable name for a new skill."""
    parts = [
        "You are a game-AI skill naming expert.",
        "",
        "Below are trajectory segments grouped as the same skill.",
        "Generate a concrete, actionable skill name and description.",
        "",
        "Segment observations (state summaries):",
    ]
    for i, obs_slice in enumerate(observation_slices[:3]):
        parts.append(f"  Segment {i + 1}: {str(list(obs_slice))[:800]}")
    if eff_add or eff_del or eff_event:
        parts.append("")
        parts.append("Learned effects of this skill:")
        if eff_add:
            parts.append(f"  adds: {list(eff_add)}")
        if eff_del:
            parts.append(f"  deletes: {list(eff_del)}")
        if eff_event:
            parts.append(f"  events: {list(eff_event)}")
    parts.extend([
        "",
        "RULES:",
        "- The NAME must be a concrete imperative verb phrase (2-6 words) describing",
        "  the actual game action. Reference game objects, mechanics, or goals.",
        "- The DESCRIPTION must be specific: describe WHAT the skill does and WHEN",
        "  to invoke it. Avoid generic phrases like 'applies effects' or 'does skill'.",
        "- Use snake_case or short phrases (e.g. \"pick_up_onion\", \"clear_bottom_row\").",
        "",
        "Return ONLY a JSON object (no extra text):",
        '{"name": "short skill name", "description": "one line description"}',
    ])
    return "\n".join(parts)


def suggest_skill_name(
    observation_slices: List[Sequence],
    eff_add: Optional[Sequence[str]] = None,
    eff_del: Optional[Sequence[str]] = None,
    eff_event: Optional[Sequence[str]] = None,
    config: Optional[LLMTeacherConfig] = None,
) -> Optional[Dict[str, str]]:
    """
    Ask the LLM to suggest a human-readable name and description for a new skill.

    Used when materializing __NEW__ segments into a real skill so the bank
    has a readable label (e.g. \"pick_up_onion\") instead of only an ID.

    Parameters
    ----------
    observation_slices : list of observation sequences
        One or more segment observation slices (e.g. from cluster segments).
    eff_add, eff_del, eff_event : optional
        Effect literals learned for this skill (from contract).
    config : LLMTeacherConfig, optional
        Model and sampling settings.

    Returns
    -------
    dict with "name" and "description" keys, or None if the LLM response could not be parsed.
    """
    cfg = config or LLMTeacherConfig()
    ask = _get_ask_model()
    prompt = _build_skill_naming_prompt(
        observation_slices,
        eff_add=eff_add,
        eff_del=eff_del,
        eff_event=eff_event,
    )
    t0 = _time.time()
    response = ask(
        prompt,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    elapsed = _time.time() - t0
    parsed = _parse_json_from_response(response)

    _record_teacher_io(TeacherIORecord(
        function="skill_naming",
        prompt=prompt,
        response=response or "",
        parsed=parsed,
        model=cfg.model or "",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        elapsed_s=round(elapsed, 3),
        error=None if parsed and "name" in (parsed or {}) else "parse_failed",
    ))

    if not parsed or "name" not in parsed:
        return None
    name = (parsed.get("name") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "description": (parsed.get("description") or "").strip() or None,
    }


def collect_uncertain_preferences(
    result,
    observations: Sequence,
    actions: Sequence,
    margin_threshold: float = 1.0,
    max_queries: int = 10,
    config: Optional[LLMTeacherConfig] = None,
) -> list:
    """
    Active learning: query the LLM only on uncertain segments (low margin).

    For each uncertain segment, asks the LLM to choose between the top-2
    skill candidates.     When config.max_workers > 1, LLM calls run in parallel.
    """
    cfg = config or LLMTeacherConfig()
    ask = _get_ask_model()
    uncertain = result.uncertain_segments(margin_threshold)
    uncertain.sort(key=lambda s: s.margin)
    to_query = [seg for seg in uncertain[:max_queries] if len(seg.candidates) >= 2]
    if not to_query:
        return []

    max_workers = getattr(cfg, "max_workers", None)
    if max_workers is None or max_workers <= 1:
        all_prefs = []
        for seg in to_query:
            pref = _collect_one_uncertain_pref(seg, observations, actions, cfg, ask)
            if pref is not None:
                all_prefs.append(pref)
        return all_prefs

    max_concurrent = getattr(cfg, "max_concurrent_llm_calls", None)
    if max_concurrent is not None and max_concurrent > 0:
        ask = _wrap_ask_with_semaphore(ask, max_concurrent)
    all_prefs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_collect_one_uncertain_pref, seg, observations, actions, cfg, ask): seg
            for seg in to_query
        }
        for future in as_completed(futures):
            try:
                pref = future.result()
                if pref is not None:
                    all_prefs.append(pref)
            except Exception:
                pass
    return all_prefs


# ── GRPO integration ──────────────────────────────────────────────────

_grpo_original_fn: Optional[Callable] = None


def enable_segment_grpo(
    buffer: "Any",
    group_size: int = 4,
    temperature: float = 0.7,
    scorer_factory: Optional[Callable] = None,
    decode_fn: Optional[Callable] = None,
) -> None:
    """Activate GRPO wrapping on ``collect_segment_preferences``.

    Wraps at the **batch level** (per episode): each GRPO sample is a
    complete set of preferences for all segments. The reward requires
    rebuilding the scorer and running the decoder per sample.

    Parameters
    ----------
    buffer : GRPOBuffer
    group_size : int
    temperature : float
    scorer_factory : callable
        ``scorer_factory(preference_list) -> SegmentScorer``
    decode_fn : callable
        ``decode_fn(scorer, segments, obs, actions, skill_names, preds) -> result``
    """
    import skill_agents_grpo.infer_segmentation.llm_teacher as _mod
    from skill_agents_grpo.grpo.rewards import segmentation_reward
    from skill_agents_grpo.grpo.wrapper import GRPOCallWrapper
    from skill_agents_grpo.lora.skill_function import SkillFunction
    from functools import partial

    global _grpo_original_fn

    if _grpo_original_fn is not None:
        logger.warning("Segment GRPO already enabled — skipping")
        return

    _grpo_original_fn = _mod.collect_segment_preferences

    bound_reward = partial(
        segmentation_reward,
        scorer_factory=scorer_factory,
        decode_fn=decode_fn,
    )

    def _prompt_extractor(
        segments: List[Tuple[int, int]],
        observations: Sequence,
        actions: Sequence,
        skill_names: List[str],
        predicates: Optional[List[Optional[dict]]] = None,
        config: Optional[LLMTeacherConfig] = None,
        **kw: Any,
    ) -> str:
        parts = [f"Segment preferences for {len(segments)} segments"]
        parts.append(f"Skills: {skill_names}")
        return "\n".join(parts)

    def _metadata_extractor(
        segments: List[Tuple[int, int]],
        *a: Any,
        **kw: Any,
    ) -> Dict[str, Any]:
        return {"n_segments": len(segments)}

    wrapper = GRPOCallWrapper(
        adapter=SkillFunction.SEGMENT,
        reward_fn=bound_reward,
        buffer=buffer,
        group_size=group_size,
        temperature=temperature,
        prompt_extractor=_prompt_extractor,
        metadata_extractor=_metadata_extractor,
    )

    _mod.collect_segment_preferences = wrapper.wrap(_grpo_original_fn)
    logger.info("Segment GRPO enabled: G=%d, temp=%.2f", group_size, temperature)


def disable_segment_grpo() -> None:
    """Deactivate GRPO wrapping, restore original function."""
    import skill_agents_grpo.infer_segmentation.llm_teacher as _mod

    global _grpo_original_fn
    if _grpo_original_fn is not None:
        _mod.collect_segment_preferences = _grpo_original_fn
        _grpo_original_fn = None
        logger.info("Segment GRPO disabled")
