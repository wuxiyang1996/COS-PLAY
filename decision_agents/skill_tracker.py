"""Skill lifecycle tracker for the Actor Agent.

Implements the control structure described in PLAN-ACTION-AGENT §3
("Protocol-aware lifecycle") and §10 ("Slot coverage check before skill
execution").  The tracker is intentionally small, deterministic, and
algorithmic — it is NOT a learned policy.  Its job is to off-load
control structure so the actor's LLM can focus on action selection
rather than "when do I switch skills?".

What the tracker owns:

* **Lifecycle state** — which skill is active, which protocol step is
  current, how long it has been active, reward-on-skill so far.
* **Re-selection triggers** — no active skill, duration cap exceeded,
  reward stall (≥N steps with reward ≤0), success/abort criteria
  keyword-matched.
* **Slot coverage check** — consults the parsed :class:`StateSchema`
  before activation and flags any required slots that are not populated.
  The actor uses that flag to decide whether to insert a GROUND hop.

What the tracker does NOT own:

* Skill selection itself — the actor calls
  :meth:`SkillProvider.select` and then hands the result to the tracker
  via :meth:`activate`.
* Action choice — the tracker only reports "the current protocol step
  is X"; the actor decides whether to follow it.
* Reward computation — that lives in ``reward_func.py``.

See ``plans/02-action-agent/PLAN-ACTION-AGENT.md`` §1 step 3 and §10 for the full spec
these rules implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .schema_parser import StateSchema
from .skill_interface import SkillGuidance


DEFAULT_STALL_WINDOW: int = 4
"""Number of consecutive zero-or-negative-reward steps that count as a stall."""

DEFAULT_DURATION_CAP: int = 16
"""Upper bound on steps-in-skill when the guidance provides no expected duration."""


@dataclass
class ActivationCheck:
    """Result of :meth:`SkillTracker.activate`.

    Tells the actor whether the skill is ready to execute or whether the
    inner MDP should insert a GROUND hop to fill in missing slots.

    Attributes
    ----------
    activated
        True when the skill has been stored as the new active skill.
        (Activation ALWAYS succeeds — missing slots are resolved via a
        GROUND hop rather than a hard rejection.)
    ready
        True when every required slot is populated.  When False, the
        actor should schedule a GROUND hop before executing the first
        protocol step.
    missing_slots
        List of slot names that are not populated in the schema.  Empty
        when ``ready`` is True.
    """

    activated: bool = False
    ready: bool = False
    missing_slots: List[str] = field(default_factory=list)


@dataclass
class TrackerState:
    """Persistent state for the tracker, exposed for logging."""

    active_skill_id: Optional[str] = None
    active_guidance: Optional[SkillGuidance] = None
    protocol_step: int = 0                  # index into protocol_steps
    steps_in_skill: int = 0                 # total steps since activation
    steps_without_progress: int = 0         # reward<=0 stall counter
    reward_on_skill: float = 0.0            # accumulated env reward on skill
    switch_count: int = 0                   # skills activated this episode
    needs_ground: bool = False              # True until required slots are filled
    pending_ground_slots: List[str] = field(default_factory=list)
    last_outcome: Optional[str] = None      # "success" | "abort" | "stall" | ...
    history: List[Dict[str, Any]] = field(default_factory=list)


class SkillTracker:
    """Protocol-aware skill lifecycle manager.

    Usage pattern (per step, called by the actor):

    >>> tracker = SkillTracker()
    >>> if tracker.should_reselect(state):
    ...     guidance = provider.select(query, ...)[0]
    ...     check = tracker.activate(guidance, state)
    ...     if not check.ready:
    ...         # insert a GROUND hop for check.missing_slots before EXECUTE
    ...         ...
    >>> step_hint = tracker.current_step()        # for the prompt
    >>> tracker.record_step(reward, new_state)    # after env.step
    """

    def __init__(
        self,
        *,
        stall_window: int = DEFAULT_STALL_WINDOW,
        default_duration_cap: int = DEFAULT_DURATION_CAP,
    ) -> None:
        self.stall_window = max(1, int(stall_window))
        self.default_duration_cap = max(1, int(default_duration_cap))
        self.state = TrackerState()

    # ── Episode lifecycle ───────────────────────────────────────────

    def reset(self) -> None:
        """Forget all state — call at the start of every episode."""
        self.state = TrackerState()

    # ── Re-selection decision ───────────────────────────────────────

    def should_reselect(self, schema: Optional[StateSchema]) -> Tuple[bool, str]:
        """Return ``(reselect, reason)``.  PLAN-ACTION-AGENT §1 step 3.

        Triggers (in priority order):

        1. No active skill → ``"no_active_skill"``.
        2. Abort criteria match in current schema → ``"abort_matched"``.
        3. Success criteria match in current schema → ``"success_matched"``.
        4. Duration cap exceeded → ``"duration_exceeded"``.
        5. Reward stall (≥ stall_window steps without progress) → ``"stall"``.

        The string reasons are surfaced in the episode metadata so
        offline analysis can tell *why* a skill was swapped.
        """
        s = self.state
        if s.active_skill_id is None or s.active_guidance is None:
            return True, "no_active_skill"

        g = s.active_guidance

        if schema is not None:
            if _criteria_match(g.abort_criteria, schema):
                return True, "abort_matched"
            if _criteria_match(g.success_criteria, schema):
                return True, "success_matched"

        duration_cap = g.expected_duration or self.default_duration_cap
        if s.steps_in_skill >= duration_cap:
            return True, "duration_exceeded"

        if s.steps_without_progress >= self.stall_window:
            return True, "stall"

        return False, ""

    # ── Activation + slot coverage ──────────────────────────────────

    def activate(
        self,
        guidance: SkillGuidance,
        schema: Optional[StateSchema],
    ) -> ActivationCheck:
        """Install *guidance* as the active skill and check slot coverage.

        PLAN-ACTION-AGENT §10 — if any of ``guidance.required_slots`` is
        missing from the current schema, the tracker flags
        :attr:`ActivationCheck.ready` as False and the actor should
        insert a GROUND hop 0.  When ``schema`` is None (pre-Phase-2
        callers without structured state), all required slots are
        treated as unknown ⇒ ready=False.
        """
        s = self.state

        # Record outcome of the previous skill, if any, before overwriting.
        if s.active_skill_id is not None and s.active_skill_id != guidance.skill_id:
            s.history.append(
                {
                    "skill_id": s.active_skill_id,
                    "steps": s.steps_in_skill,
                    "reward": s.reward_on_skill,
                    "outcome": s.last_outcome or "switch",
                }
            )

        s.active_skill_id = guidance.skill_id
        s.active_guidance = guidance
        s.protocol_step = 0
        s.steps_in_skill = 0
        s.steps_without_progress = 0
        s.reward_on_skill = 0.0
        s.last_outcome = None
        s.switch_count += 1

        if not guidance.required_slots:
            s.needs_ground = False
            s.pending_ground_slots = []
            return ActivationCheck(activated=True, ready=True, missing_slots=[])

        if schema is None:
            s.needs_ground = True
            s.pending_ground_slots = list(guidance.required_slots)
            return ActivationCheck(
                activated=True,
                ready=False,
                missing_slots=list(guidance.required_slots),
            )

        missing = schema.missing_slots(guidance.required_slots)
        s.pending_ground_slots = missing
        s.needs_ground = bool(missing)
        return ActivationCheck(
            activated=True,
            ready=not s.needs_ground,
            missing_slots=missing,
        )

    def clear_ground_flag(self, schema: Optional[StateSchema] = None) -> List[str]:
        """Re-check the required slots after a GROUND hop.

        Called by the actor after an inner-MDP GROUND step produces a
        fresh schema.  Returns the remaining missing slots (empty when
        the skill is now ready).
        """
        s = self.state
        g = s.active_guidance
        if g is None or not g.required_slots:
            s.needs_ground = False
            s.pending_ground_slots = []
            return []
        if schema is None:
            return list(s.pending_ground_slots)
        missing = schema.missing_slots(g.required_slots)
        s.pending_ground_slots = missing
        s.needs_ground = bool(missing)
        return missing

    # ── Per-step update ─────────────────────────────────────────────

    def record_step(
        self,
        *,
        reward: float,
        schema_after: Optional[StateSchema] = None,
        advance_protocol: bool = True,
    ) -> None:
        """Update counters after one ``env.step``.

        Parameters
        ----------
        reward
            Raw environment reward from the step.  Used only for stall
            detection (``reward <= 0`` ⇒ ``steps_without_progress += 1``).
        schema_after
            The schema produced after the step — used to re-evaluate
            success/abort criteria on the next :meth:`should_reselect`
            call.  Pass ``None`` when the actor isn't in Phase 2.
        advance_protocol
            When True (default), move the protocol cursor forward by one.
            The actor may pass False for no-op / thinking / GROUND hops
            that should not count as protocol progress.
        """
        s = self.state
        if s.active_skill_id is None:
            return
        s.steps_in_skill += 1
        s.reward_on_skill += float(reward or 0.0)
        if reward is None or float(reward) <= 0:
            s.steps_without_progress += 1
        else:
            s.steps_without_progress = 0

        if advance_protocol and s.active_guidance is not None:
            max_step = len(s.active_guidance.protocol_steps)
            if max_step > 0:
                s.protocol_step = min(s.protocol_step + 1, max_step)

    # ── Outcome reporting ───────────────────────────────────────────

    def finalize_active_skill(
        self,
        outcome: str,
    ) -> Optional[Dict[str, Any]]:
        """Mark the active skill as finished.

        Returns a dict suitable to forward to
        :meth:`SkillProvider.record_outcome`, or ``None`` when no skill
        is active.
        """
        s = self.state
        if s.active_skill_id is None:
            return None
        record = {
            "skill_id": s.active_skill_id,
            "outcome": outcome,
            "reward": s.reward_on_skill,
            "steps_taken": s.steps_in_skill,
        }
        s.last_outcome = outcome
        s.history.append(
            {
                "skill_id": s.active_skill_id,
                "steps": s.steps_in_skill,
                "reward": s.reward_on_skill,
                "outcome": outcome,
            }
        )
        return record

    # ── Accessors ───────────────────────────────────────────────────

    @property
    def active_skill_id(self) -> Optional[str]:
        return self.state.active_skill_id

    @property
    def active_guidance(self) -> Optional[SkillGuidance]:
        return self.state.active_guidance

    @property
    def needs_ground(self) -> bool:
        return self.state.needs_ground

    @property
    def pending_ground_slots(self) -> List[str]:
        return list(self.state.pending_ground_slots)

    def current_step(self) -> Optional[str]:
        """Return the description of the current protocol step, or None."""
        g = self.state.active_guidance
        if g is None or not g.protocol_steps:
            return None
        idx = min(self.state.protocol_step, len(g.protocol_steps) - 1)
        return g.protocol_steps[idx]

    def progress_marker(self) -> str:
        """Return ``"k/n"`` — steps completed / total protocol steps."""
        g = self.state.active_guidance
        if g is None or not g.protocol_steps:
            return "0/0"
        return f"{self.state.protocol_step}/{len(g.protocol_steps)}"


# ──────────────────────────────────────────────────────────────────────
# Internal: keyword-level criteria matcher
# ──────────────────────────────────────────────────────────────────────


def _criteria_match(
    criteria: List[str],
    schema: StateSchema,
) -> bool:
    """Return True when any entry in *criteria* is keyword-present in *schema*.

    We use the schema's compact summary plus the labels of declared
    entities as the haystack.  This is intentionally lightweight — the
    existing legacy ``_SkillTracker`` in ``scripts/qwen3_decision_agent``
    uses substring matching against a state string, and we preserve that
    semantic here.

    Returns False on empty/None *criteria*.
    """
    if not criteria:
        return False

    haystack_parts = [schema.compact_summary(max_chars=600)]
    for ent in schema.entities.values():
        if ent.label:
            haystack_parts.append(ent.label)
    sf = schema.state_flags
    if sf.error:
        haystack_parts.append(sf.error)
    haystack = " | ".join(haystack_parts).lower()

    for crit in criteria:
        if not crit:
            continue
        words = [w for w in crit.lower().split() if len(w) >= 3]
        if not words:
            # Very short criterion — require exact substring.
            if crit.lower() in haystack:
                return True
            continue
        if all(w in haystack for w in words):
            return True
    return False


__all__ = [
    "ActivationCheck",
    "SkillTracker",
    "TrackerState",
    "DEFAULT_STALL_WINDOW",
    "DEFAULT_DURATION_CAP",
]
