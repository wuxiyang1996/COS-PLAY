"""
Per-environment signal extractors.

Each extractor takes a list of Experience objects (from an Episode) and
returns the signals expected by the boundary proposal pipeline:

  - predicates:       list[dict]   -- discrete state facts per timestep
  - event_times:      list[int]    -- timesteps of hard events
  - reward_array:     np.ndarray   -- per-step reward (for reward-spike events)

Two extraction strategies:

  Rule-based (per-env):  Fast, zero-cost, but brittle keyword matching.
                         Best for structured state dicts with known keys.

  LLM-based (general):   Uses ask_model() to extract predicates from NL
                         state descriptions.  Environment-agnostic.  See
                         ``llm_extractor.py`` for details.

  Hybrid (recommended):  Per-env hard-event detection (rule-based) +
                         LLM-based predicate extraction.  Get this via
                         ``get_signal_extractor("llm+overcooked")`` etc.

Usage:
    # Pure rule-based (legacy, per-env)
    extractor = get_signal_extractor("overcooked")

    # Pure LLM (fully general, no per-env rules)
    extractor = get_signal_extractor("llm")

    # Hybrid: LLM predicates + per-env hard events (recommended)
    extractor = get_signal_extractor("llm+overcooked")
"""

from __future__ import annotations

import re
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

# Avoid hard import of data_structure to keep this module self-contained.
# We rely on duck typing: experience objects must have .state, .action,
# .reward, .done, .next_state, and optionally .idx.


class SignalExtractorBase(ABC):
    """Base class for environment-specific signal extraction."""

    @abstractmethod
    def extract_predicates(self, experiences: list) -> List[Optional[dict]]:
        """Return a list of predicate dicts, one per timestep."""
        ...

    @abstractmethod
    def extract_event_times(self, experiences: list) -> List[int]:
        """Return timesteps of hard events (reward spikes, resets, phase changes, etc.)."""
        ...

    def extract_rewards(self, experiences: list) -> np.ndarray:
        """Return reward array (T,). Default: pull from experience.reward."""
        rewards = []
        for exp in experiences:
            r = exp.reward if exp.reward is not None else 0.0
            if isinstance(r, dict):
                r = sum(r.values())
            rewards.append(float(r))
        return np.array(rewards, dtype=np.float64)

    def detect_reward_spike_events(
        self,
        experiences: list,
        std_factor: float = 2.0,
    ) -> List[int]:
        """Detect reward spikes as hard events."""
        rewards = self.extract_rewards(experiences)
        if len(rewards) == 0:
            return []
        mean_r = float(np.nanmean(rewards))
        std_r = float(np.nanstd(rewards))
        if std_r < 1e-9:
            return []
        threshold = mean_r + std_factor * std_r
        return [int(t) for t in range(len(rewards)) if rewards[t] >= threshold]

    def extract(
        self, experiences: list
    ) -> Tuple[List[Optional[dict]], List[int]]:
        """
        Convenience: extract both predicates and event_times.

        Returns (predicates, event_times).
        """
        predicates = self.extract_predicates(experiences)
        event_times = self.extract_event_times(experiences)
        return predicates, event_times


# ---------------------------------------------------------------------------
# Overcooked
# ---------------------------------------------------------------------------


class OvercookedSignalExtractor(SignalExtractorBase):
    """
    Extract signals from Overcooked experiences.

    Predicates (from state NL or state dict):
      - held_object: what the agent is holding
      - position: agent grid position
      - pot_status: soup cooking/ready/empty
      - order_pending: whether there's a pending order

    Events:
      - reward spikes (soup delivered)
      - done flags
    """

    def extract_predicates(self, experiences: list) -> List[Optional[dict]]:
        predicates = []
        for exp in experiences:
            preds: dict = {}
            state = exp.state
            if isinstance(state, dict):
                # Structured state dict from env info
                if "overcooked_state" in state:
                    preds["has_overcooked_state"] = True
                preds["done"] = bool(exp.done)
            elif isinstance(state, str):
                # NL state string — extract keywords
                sl = state.lower()
                preds["holding_something"] = "holding" in sl and "nothing" not in sl
                preds["soup_ready"] = "ready" in sl
                preds["soup_cooking"] = "cooking" in sl
                preds["at_counter"] = "counter" in sl
                preds["at_pot"] = "pot" in sl
                preds["at_serving"] = "serving" in sl or "deliver" in sl
            else:
                preds["unknown_state"] = True
            predicates.append(preds)
        return predicates

    def extract_event_times(self, experiences: list) -> List[int]:
        events = []
        for t, exp in enumerate(experiences):
            if exp.done:
                events.append(t)
            # Reward spike: soup delivered gives +20 in Overcooked
            r = exp.reward if exp.reward is not None else 0.0
            if isinstance(r, (tuple, list)):
                r = sum(r)
            if isinstance(r, dict):
                r = sum(r.values())
            if float(r) > 0:
                events.append(t)
        return sorted(set(events))


# ---------------------------------------------------------------------------
# Avalon
# ---------------------------------------------------------------------------


class AvalonSignalExtractor(SignalExtractorBase):
    """
    Extract signals from Avalon experiences.

    Predicates:
      - phase / phase_name (Team Selection, Team Voting, Quest Voting, Assassination)
      - turn, round
      - quest_results (list)
      - leader

    Events:
      - phase transitions
      - quest completion (new result added)
      - game end
    """

    def extract_predicates(self, experiences: list) -> List[Optional[dict]]:
        predicates = []
        for exp in experiences:
            preds: dict = {}
            state = exp.state
            if isinstance(state, dict):
                preds["phase"] = state.get("phase")
                preds["phase_name"] = state.get("phase_name")
                preds["turn"] = state.get("turn")
                preds["round"] = state.get("round")
                preds["leader"] = state.get("leader")
                preds["quest_results"] = str(state.get("quest_results", []))
                preds["done"] = state.get("done", False)
            elif isinstance(state, str):
                sl = state.lower()
                preds["team_selection"] = "team selection" in sl or "propose" in sl
                preds["voting"] = "vote" in sl
                preds["quest"] = "quest" in sl
                preds["assassination"] = "assassin" in sl
            predicates.append(preds)
        return predicates

    def extract_event_times(self, experiences: list) -> List[int]:
        events = []
        prev_phase = None
        prev_quest_results = None
        for t, exp in enumerate(experiences):
            state = exp.state if isinstance(exp.state, dict) else {}
            phase = state.get("phase")
            quest_results = state.get("quest_results", [])

            # Phase transition
            if phase is not None and phase != prev_phase and prev_phase is not None:
                events.append(t)
            prev_phase = phase

            # Quest completion
            if isinstance(quest_results, list) and prev_quest_results is not None:
                if len(quest_results) > len(prev_quest_results):
                    events.append(t)
            prev_quest_results = quest_results if isinstance(quest_results, list) else prev_quest_results

            # Game end
            if exp.done:
                events.append(t)
        return sorted(set(events))


# ---------------------------------------------------------------------------
# Diplomacy
# ---------------------------------------------------------------------------


class DiplomacySignalExtractor(SignalExtractorBase):
    """
    Extract signals from Diplomacy experiences.

    Predicates:
      - phase (e.g. "S1901M", "F1901M")
      - phase_type (M/R/A)
      - num_centers per power
      - eliminated powers

    Events:
      - phase transitions
      - supply center changes (gains/losses)
      - power elimination
      - game end
    """

    def __init__(self, controlled_power: Optional[str] = None):
        self.controlled_power = controlled_power

    def extract_predicates(self, experiences: list) -> List[Optional[dict]]:
        predicates = []
        for exp in experiences:
            preds: dict = {}
            state = exp.state
            if isinstance(state, dict):
                preds["phase"] = state.get("phase")
                preds["phase_type"] = state.get("phase_type")
                powers = state.get("powers", {})
                if self.controlled_power and self.controlled_power in powers:
                    p = powers[self.controlled_power]
                    preds["num_centers"] = p.get("num_centers", 0)
                    preds["num_units"] = len(p.get("units", []))
                    preds["eliminated"] = p.get("eliminated", False)
                else:
                    # Track all powers center counts
                    for pname, pdata in powers.items():
                        preds[f"{pname}_centers"] = pdata.get("num_centers", 0)
                preds["is_game_done"] = state.get("is_game_done", False)
            elif isinstance(state, str):
                sl = state.lower()
                preds["movement"] = "movement" in sl
                preds["retreat"] = "retreat" in sl
                preds["adjustment"] = "build" in sl or "disband" in sl
            predicates.append(preds)
        return predicates

    def extract_event_times(self, experiences: list) -> List[int]:
        events = []
        prev_phase = None
        prev_centers: Optional[dict] = None
        for t, exp in enumerate(experiences):
            state = exp.state if isinstance(exp.state, dict) else {}
            phase = state.get("phase")
            powers = state.get("powers", {})

            # Phase transition
            if phase is not None and phase != prev_phase and prev_phase is not None:
                events.append(t)
            prev_phase = phase

            # Supply center change
            centers = {pn: pd.get("num_centers", 0) for pn, pd in powers.items()}
            if prev_centers is not None and centers != prev_centers:
                events.append(t)
            prev_centers = centers

            # Game end
            if exp.done:
                events.append(t)
        return sorted(set(events))


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------


class GenericSignalExtractor(SignalExtractorBase):
    """
    Generic extractor: treats state as opaque, uses reward spikes and done flags.
    Works with any environment but has lower signal quality.
    """

    def extract_predicates(self, experiences: list) -> List[Optional[dict]]:
        predicates = []
        for exp in experiences:
            preds: dict = {}
            if isinstance(exp.state, dict):
                for k, v in exp.state.items():
                    if isinstance(v, (bool, int, float, str)):
                        preds[k] = v
            preds["done"] = bool(exp.done)
            predicates.append(preds)
        return predicates

    def extract_event_times(self, experiences: list) -> List[int]:
        events = []
        for t, exp in enumerate(experiences):
            if exp.done:
                events.append(t)
        events.extend(self.detect_reward_spike_events(experiences))
        return sorted(set(events))


# ---------------------------------------------------------------------------
# Intention-based: extract predicates from [TAG] intention strings
# ---------------------------------------------------------------------------

_SUBGOAL_TAGS_DEFAULT = (
    "SETUP", "CLEAR", "MERGE", "ATTACK", "DEFEND",
    "NAVIGATE", "POSITION", "COLLECT", "BUILD", "SURVIVE",
    "OPTIMIZE", "EXPLORE", "EXECUTE",
)

_TAG_ALIASES: Dict[str, str] = {
    "PLACE": "SETUP", "DROP": "EXECUTE", "MOVE": "NAVIGATE",
    "SWAP": "EXECUTE", "PUSH": "NAVIGATE", "JUMP": "NAVIGATE",
    "MATCH": "CLEAR", "PLAN": "SETUP", "ARRANGE": "SETUP",
    "ROTATE": "SETUP", "ORGANIZE": "OPTIMIZE", "SCORE": "EXECUTE",
    "PROTECT": "DEFEND", "GRAB": "COLLECT", "FLEE": "SURVIVE",
    "RUN": "NAVIGATE", "CREATE": "BUILD", "FIND": "EXPLORE",
    "FIX": "OPTIMIZE", "ALIGN": "POSITION", "TARGET": "ATTACK",
    "SECURE": "DEFEND", "EXPAND": "ATTACK", "RETREAT": "DEFEND",
}

_TAG_RE = re.compile(r"\[(\w+)\]")


def parse_intention_tag(intention: str, tags: tuple = _SUBGOAL_TAGS_DEFAULT) -> str:
    """Extract and normalise the [TAG] from an intention string.

    Returns the canonical tag (e.g. ``"CLEAR"``) or ``"UNKNOWN"`` if no
    recognised tag is found.
    """
    m = _TAG_RE.match((intention or "").strip())
    if not m:
        return "UNKNOWN"
    raw = m.group(1).upper()
    if raw in tags:
        return raw
    return _TAG_ALIASES.get(raw, "UNKNOWN")


class IntentionSignalExtractor(SignalExtractorBase):
    """Extract predicates and events from ``[TAG]`` intention annotations.

    Produces per-timestep predicate dicts with one-hot ``tag_<tag>`` keys
    that match the vocabulary used by intention-extracted
    ``SkillEffectsContract`` objects (``eff_add={..._completed}``,
    ``eff_event={tag_...}``).

    Works on any ``Experience`` that carries an ``intentions`` attribute
    (i.e. LLM-labeled episodes).  For unlabeled data the predicates will
    be empty dicts and events will only contain ``done`` flags, which is a
    safe no-op.

    Parameters
    ----------
    tags : tuple[str, ...]
        Canonical tag vocabulary.  Defaults to ``SUBGOAL_TAGS``.
    consistency_penalty : float
        Weight for penalizing rapid tag switching (ping-pong).
    min_segment_length : int
        Minimum steps between boundaries; switches within this window
        are penalized.
    min_skill_length : int
        Post-processing: segments shorter than this are merged into
        the longer neighbor.
    boundary_score_threshold : float
        Minimum score for a boundary to be included in the event list.
    """

    def __init__(
        self,
        tags: tuple = _SUBGOAL_TAGS_DEFAULT,
        consistency_penalty: float = 0.3,
        min_segment_length: int = 3,
        min_skill_length: int = 2,
        boundary_score_threshold: float = 0.3,
    ):
        self._tags = tags
        self._tag_set = frozenset(tags)
        self.consistency_penalty = consistency_penalty
        self.min_segment_length = min_segment_length
        self.min_skill_length = min_skill_length
        self.boundary_score_threshold = boundary_score_threshold

    def extract_predicates(self, experiences: list) -> List[Optional[dict]]:
        predicates: List[Optional[dict]] = []
        for exp in experiences:
            intent = getattr(exp, "intentions", None) or ""
            tag = parse_intention_tag(intent, self._tags)

            preds: dict = {f"tag_{t.lower()}": float(t == tag) for t in self._tags}

            done = bool(getattr(exp, "done", False))
            if tag != "UNKNOWN":
                preds[f"{tag.lower()}_completed"] = float(done)

            preds["done"] = done
            predicates.append(preds)
        return predicates

    def _extract_tag_sequence(self, experiences: list) -> List[str]:
        """Extract the tag at each timestep."""
        tags = []
        for exp in experiences:
            intent = getattr(exp, "intentions", None) or ""
            tags.append(parse_intention_tag(intent, self._tags))
        return tags

    def score_boundary_candidates(self, experiences: list) -> List:
        """Score each tag-change boundary with the penalty model.

        Returns ``List[ScoredBoundary]`` (imported lazily to avoid circular deps).

        Scoring logic:
          - base_score = 1.0 for every tag change
          - ping-pong penalty: if tag[t-1] == tag[t+1] (A->B->A), penalty
          - rapid-switch penalty: if time since last change < min_segment_length
          - reward-signal bonus: if reward at boundary is a spike
          - done bonus: if the experience is a terminal state
        """
        from skill_agents_grpo.stage3_mvp.schemas import ScoredBoundary

        tag_seq = self._extract_tag_sequence(experiences)
        T = len(tag_seq)
        if T < 2:
            return []

        reward_spikes = set(self.detect_reward_spike_events(experiences))

        scored: List[ScoredBoundary] = []
        last_change_t = -999

        for t in range(1, T):
            if tag_seq[t] == tag_seq[t - 1] or tag_seq[t] == "UNKNOWN":
                continue

            base_score = 1.0
            time_since_last = t - last_change_t

            is_ping_pong = False
            if t + 1 < T and tag_seq[t - 1] == tag_seq[t + 1]:
                is_ping_pong = True

            rapid_switch = 0.0
            if is_ping_pong:
                rapid_switch = 1.0
            elif time_since_last < self.min_segment_length:
                rapid_switch = 0.5

            penalty = self.consistency_penalty * rapid_switch
            reward_bonus = 0.3 if t in reward_spikes else 0.0
            done_bonus = 0.2 if getattr(experiences[t], "done", False) else 0.0

            score = base_score - penalty + reward_bonus + done_bonus

            scored.append(ScoredBoundary(
                time=t,
                score=max(0.0, score),
                tag_before=tag_seq[t - 1],
                tag_after=tag_seq[t],
                is_ping_pong=is_ping_pong,
                time_since_last=time_since_last,
            ))

            last_change_t = t

        return scored

    def extract_event_times(self, experiences: list) -> List[int]:
        """Return boundary timesteps, filtered by score threshold.

        Uses ``score_boundary_candidates`` internally and returns only
        boundaries with score >= ``boundary_score_threshold``.
        """
        scored = self.score_boundary_candidates(experiences)
        events = [
            sb.time for sb in scored
            if sb.score >= self.boundary_score_threshold
        ]

        for t, exp in enumerate(experiences):
            if getattr(exp, "done", False):
                events.append(t)

        events.extend(self.detect_reward_spike_events(experiences))
        return sorted(set(events))

    def extract_event_times_scored(self, experiences: list) -> List:
        """Return the full ``List[ScoredBoundary]`` for Stage 2 consumption."""
        return self.score_boundary_candidates(experiences)

    @staticmethod
    def merge_short_segments(
        segments: list,
        min_length: int = 2,
    ) -> list:
        """Post-processing: merge segments shorter than *min_length* into
        the longer adjacent neighbor.

        *segments* is a list of objects with ``.start``, ``.end``, and
        ``.assigned_skill`` attributes (e.g. ``SegmentDiagnostic``).
        Returns a new list (does not mutate input).
        """
        if not segments or min_length < 2:
            return list(segments)

        merged = list(segments)
        changed = True
        while changed:
            changed = False
            new_merged = []
            i = 0
            while i < len(merged):
                seg = merged[i]
                seg_len = seg.end - seg.start + 1
                if seg_len < min_length and len(merged) > 1:
                    left_len = (merged[i - 1].end - merged[i - 1].start + 1) if i > 0 else 0
                    right_len = (merged[i + 1].end - merged[i + 1].start + 1) if i + 1 < len(merged) else 0

                    if left_len >= right_len and i > 0:
                        new_merged[-1].end = seg.end
                        changed = True
                    elif right_len > 0 and i + 1 < len(merged):
                        merged[i + 1].start = seg.start
                        changed = True
                    else:
                        new_merged.append(seg)
                else:
                    new_merged.append(seg)
                i += 1
            merged = new_merged
        return merged


# ---------------------------------------------------------------------------
# Hybrid: LLM predicates + rule-based hard events
# ---------------------------------------------------------------------------


class HybridSignalExtractor(SignalExtractorBase):
    """
    Combines LLM-based predicate extraction with rule-based hard event
    detection from a per-environment extractor.

    This is the **recommended** extractor for production use:
    - Predicates are extracted by the LLM (general, adaptive)
    - Hard events come from cheap per-env rules (reliable, free)

    Parameters
    ----------
    llm_extractor : LLMSignalExtractor
        Handles predicate extraction.
    rule_extractor : SignalExtractorBase
        Handles hard event detection (done, phase transitions, reward spikes).
    """

    def __init__(
        self,
        llm_extractor,
        rule_extractor: SignalExtractorBase,
    ):
        self._llm = llm_extractor
        self._rule = rule_extractor

    def extract_predicates(self, experiences: list) -> List[Optional[dict]]:
        """Predicates from LLM (general, adaptive)."""
        return self._llm.extract_predicates(experiences)

    def extract_event_times(self, experiences: list) -> List[int]:
        """Hard events from rule-based extractor (cheap, reliable)."""
        return self._rule.extract_event_times(experiences)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_RULE_EXTRACTORS = {
    "overcooked": OvercookedSignalExtractor,
    "avalon": AvalonSignalExtractor,
    "diplomacy": DiplomacySignalExtractor,
    "generic": GenericSignalExtractor,
}


def get_signal_extractor(
    env_name: str,
    **kwargs,
) -> SignalExtractorBase:
    """
    Factory: return the signal extractor for the given environment.

    Supported patterns:

    - ``"overcooked"`` / ``"avalon"`` / ``"diplomacy"`` / ``"generic"``
        Pure rule-based extractor (legacy, per-env).

    - ``"llm"``
        Pure LLM-based extractor (fully general).
        kwargs are passed to LLMSignalExtractor (model, chunk_size, etc.).

    - ``"llm+overcooked"`` / ``"llm+avalon"`` / ``"llm+diplomacy"``
        Hybrid: LLM predicates + per-env hard events (recommended).
        LLM kwargs: model, chunk_size, temperature, filter_significance.
        Env kwargs: e.g. controlled_power for Diplomacy.

    Args:
        env_name: Extractor identifier (see patterns above).
        **kwargs: Passed to the extractor constructor.

    Returns:
        SignalExtractorBase instance.
    """
    key = env_name.lower().strip()

    # Pure rule-based
    if key in _RULE_EXTRACTORS:
        # Filter kwargs to only those accepted by the rule extractor
        return _RULE_EXTRACTORS[key](**_filter_kwargs(_RULE_EXTRACTORS[key], kwargs))

    # Pure LLM
    if key == "llm":
        from skill_agents_grpo.boundary_proposal.llm_extractor import LLMSignalExtractor
        llm_kwargs = _extract_llm_kwargs(kwargs)
        return LLMSignalExtractor(**llm_kwargs)

    # Hybrid: "llm+envname"
    if key.startswith("llm+"):
        env_part = key[4:]
        if env_part not in _RULE_EXTRACTORS:
            raise ValueError(
                f"Unknown env '{env_part}' in hybrid '{env_name}'. "
                f"Available: {list(_RULE_EXTRACTORS.keys())}"
            )
        from skill_agents_grpo.boundary_proposal.llm_extractor import LLMSignalExtractor
        llm_kwargs = _extract_llm_kwargs(kwargs)
        env_kwargs = {k: v for k, v in kwargs.items() if k not in llm_kwargs}
        llm_ext = LLMSignalExtractor(**llm_kwargs)
        rule_ext = _RULE_EXTRACTORS[env_part](**_filter_kwargs(_RULE_EXTRACTORS[env_part], env_kwargs))
        return HybridSignalExtractor(llm_ext, rule_ext)

    # Pure intention-based
    if key == "intention":
        return IntentionSignalExtractor(**_filter_kwargs(IntentionSignalExtractor, kwargs))

    # Hybrid: "intention+envname" — intention predicates + per-env hard events
    if key.startswith("intention+"):
        env_part = key[len("intention+"):]
        if env_part not in _RULE_EXTRACTORS:
            raise ValueError(
                f"Unknown env '{env_part}' in hybrid '{env_name}'. "
                f"Available: {list(_RULE_EXTRACTORS.keys())}"
            )
        intent_ext = IntentionSignalExtractor(**_filter_kwargs(IntentionSignalExtractor, kwargs))
        env_kwargs = {k: v for k, v in kwargs.items() if k != "tags"}
        rule_ext = _RULE_EXTRACTORS[env_part](**_filter_kwargs(_RULE_EXTRACTORS[env_part], env_kwargs))
        return HybridSignalExtractor(intent_ext, rule_ext)

    raise ValueError(
        f"Unknown extractor '{env_name}'. "
        f"Available: {list(_RULE_EXTRACTORS.keys())} | 'llm' | 'llm+<env>' | "
        f"'intention' | 'intention+<env>'"
    )


def _extract_llm_kwargs(kwargs: dict) -> dict:
    """Pull LLMSignalExtractor-relevant kwargs from the combined dict."""
    llm_keys = {
        "ask_model_fn", "model", "chunk_size", "temperature",
        "filter_significance", "max_state_chars", "reward_spike_std",
    }
    return {k: v for k, v in kwargs.items() if k in llm_keys}


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Filter kwargs to only those accepted by cls.__init__."""
    import inspect
    try:
        sig = inspect.signature(cls.__init__)
        valid = set(sig.parameters.keys()) - {"self"}
        return {k: v for k, v in kwargs.items() if k in valid}
    except (ValueError, TypeError):
        return kwargs
