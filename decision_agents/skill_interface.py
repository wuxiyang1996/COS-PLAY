"""Pluggable skill interface for the Actor Agent.

The Actor Agent should NOT know how skills are stored, ranked, or
retrieved.  It only needs three things per step:

1. A :class:`SkillGuidance` package telling it *what* to do, *when* it is
   valid, and *when* it is done.
2. A way to mark a skill attempt as succeeded, aborted, or stalled so the
   skill-use agent / skill bank can update its statistics.
3. A "required slots" list so the actor can decide — via its inner MDP —
   whether to insert a GROUND hop before executing (PLAN-ACTION-AGENT
   §10).

This module defines the contract between the Actor (Agent 1) and the
Skill-Use Agent (Agent 2) from ``plans/02-action-agent/PLAN-ACTION-AGENT.md`` §2.3.  Two
concrete providers ship with this package:

* :class:`NullSkillProvider` — a no-op used for skill-less runs and
  unit tests.  The actor falls back to raw LLM action selection.
* :class:`SkillBankProvider` — a thin adapter around the existing
  ``SkillQueryEngine`` / ``SkillBankMVP`` stack.  The adapter is
  intentionally thin; it does NOT reimplement RAG, scoring, or
  lifecycle logic, only converts between the bank's return types and
  the actor-facing :class:`SkillGuidance` dataclass.

When the Skill-Use Agent (Agent 2) is implemented as a learned policy,
it will simply provide another class that implements
:class:`SkillProvider` — no actor changes needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ──────────────────────────────────────────────────────────────────────
# Skill guidance dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SkillGuidance:
    """Structured skill-guidance package returned by :meth:`SkillProvider.select`.

    Mirrors the fields of ``skill_agents.query.SkillSelectionResult`` but
    is *this* package's canonical type so the actor never imports from
    ``skill_agents`` directly.  That keeps the dependency direction
    one-way: ``skill_agents`` may implement ``SkillProvider``, but the
    Actor Agent is unaware of skill_agents internals.

    See PLAN-ACTION-AGENT §3 for the scoring axes that populate these
    fields when the underlying store is a SkillQueryEngine.
    """

    skill_id: str
    skill_name: str = ""
    why_selected: str = ""

    # Scoring — actor uses these for logging only, not for re-ranking.
    confidence: float = 0.0
    relevance: float = 0.0
    applicability_score: float = 0.0
    pass_rate: Optional[float] = None

    # Execution description — rendered into the action-selection prompt.
    execution_hint: str = ""
    strategic_description: str = ""

    # Protocol (what the skill actually does).
    protocol_steps: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    abort_criteria: List[str] = field(default_factory=list)
    expected_duration: int = 0
    termination_hint: str = ""
    failure_modes: List[str] = field(default_factory=list)

    # Slot contract — which schema slots the skill NEEDS populated before
    # it can execute.  Drives the inner-MDP GROUND-insertion logic in
    # :mod:`decision_agents.skill_tracker` (PLAN §10).
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)

    # Effects contract (for r_follow reward shaping).  Kept as sets
    # (serialised to lists in ``to_dict``) so set operations against the
    # current state are O(n).
    eff_add: List[str] = field(default_factory=list)
    eff_del: List[str] = field(default_factory=list)

    # Fallback action sequence when no LLM is available.  Each entry is a
    # dict; the recognised keys are ``action`` (env action string) and,
    # optionally, ``effect`` (predicate that should become true).
    micro_plan: List[Dict[str, Any]] = field(default_factory=list)

    # Opaque passthrough for provider-specific diagnostics (timings,
    # scores breakdown, selection trace).  Ignored by the actor.
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "why_selected": self.why_selected,
            "confidence": round(self.confidence, 4),
            "relevance": round(self.relevance, 4),
            "applicability_score": round(self.applicability_score, 4),
            "pass_rate": (
                round(self.pass_rate, 3) if self.pass_rate is not None else None
            ),
            "execution_hint": self.execution_hint,
            "strategic_description": self.strategic_description,
            "protocol_steps": list(self.protocol_steps),
            "preconditions": list(self.preconditions),
            "success_criteria": list(self.success_criteria),
            "abort_criteria": list(self.abort_criteria),
            "expected_duration": self.expected_duration,
            "termination_hint": self.termination_hint,
            "failure_modes": list(self.failure_modes),
            "required_slots": list(self.required_slots),
            "optional_slots": list(self.optional_slots),
            "eff_add": list(self.eff_add),
            "eff_del": list(self.eff_del),
            "micro_plan": list(self.micro_plan),
        }


# ──────────────────────────────────────────────────────────────────────
# Provider protocol
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class SkillProvider(Protocol):
    """Minimal contract every skill backend must implement.

    Instances are injected into :class:`decision_agents.actor_agent.ActorAgent`.
    The actor calls :meth:`select` when its lifecycle tracker flags a
    re-selection event (new episode, stall, abort, success) and
    :meth:`record_outcome` after every episode terminates.

    The protocol is deliberately thin.  Scoring, RAG, applicability
    filtering, and lifecycle heuristics live inside the provider.  The
    actor is concerned only with *using* guidance, not producing it.
    """

    def select(
        self,
        query: str,
        *,
        state_summary: str = "",
        structured_state: Optional[Any] = None,
        current_predicates: Optional[Dict[str, float]] = None,
        top_k: int = 1,
    ) -> List[SkillGuidance]:
        """Return up to *top_k* candidate skills for the current state.

        Parameters
        ----------
        query
            Free-text query, typically ``"<game_name> | <intention>"`` or
            a compact state summary.
        state_summary
            Compact ``key=value`` summary of the current state, for
            providers that want extra retrieval signal.
        structured_state
            The parsed :class:`StateSchema` when the actor is in
            Phase 2/3 (PLAN §7).  Providers that can score slot
            compatibility should use this.
        current_predicates
            Optional ``{predicate: float}`` dict used by applicability
            scoring (PLAN §3).  Set if the provider supports this API;
            otherwise pass ``None``.
        top_k
            Desired number of results.  Providers may return fewer.
        """
        ...

    def record_outcome(
        self,
        skill_id: str,
        *,
        outcome: str,
        reward: float = 0.0,
        steps_taken: int = 0,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record how the last skill attempt terminated.

        ``outcome`` is one of:
        ``"success"`` | ``"abort"`` | ``"stall"`` | ``"switch"`` |
        ``"timeout"``.  Providers use this to update pass-rate
        statistics (PLAN §3) and, for learned providers, to feed the
        Skill-Use Agent's GRPO reward.
        """
        ...

    def available_skills(self) -> List[str]:
        """Return the skill IDs this provider can currently select from.

        Used by the actor's prompt builder to tell the LLM which skills
        exist.  A ``NullSkillProvider`` returns ``[]``.
        """
        ...


# ──────────────────────────────────────────────────────────────────────
# Null provider
# ──────────────────────────────────────────────────────────────────────


class NullSkillProvider:
    """Provider that returns no guidance at all — the actor runs in
    skill-free mode.

    Useful for baselines, unit tests, and the first few rollouts of a
    fresh domain where the skill bank is still empty.  Every method is a
    cheap no-op.
    """

    def select(
        self,
        query: str,
        *,
        state_summary: str = "",
        structured_state: Optional[Any] = None,
        current_predicates: Optional[Dict[str, float]] = None,
        top_k: int = 1,
    ) -> List[SkillGuidance]:
        return []

    def record_outcome(
        self,
        skill_id: str,
        *,
        outcome: str,
        reward: float = 0.0,
        steps_taken: int = 0,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        return None

    def available_skills(self) -> List[str]:
        return []


# ──────────────────────────────────────────────────────────────────────
# Skill-bank adapter
# ──────────────────────────────────────────────────────────────────────


class SkillBankProvider:
    """Adapter wrapping an existing ``SkillQueryEngine`` / ``SkillBankMVP``.

    The adapter performs exactly one responsibility: translating between
    the existing ``SkillSelectionResult`` / ``Skill`` / ``Protocol``
    types and the actor-facing :class:`SkillGuidance`.  It does not add
    new scoring, caching, or lifecycle logic.

    Parameters
    ----------
    skill_bank
        A skill-bank instance.  Accepted shapes (in priority order):

        * ``SkillQueryEngine`` — preferred.  ``.select()`` is called
          directly and produces the richest guidance packages.
        * ``SkillBankMVP`` wrapped in something exposing ``.skill_ids`` /
          ``.get_skill`` / ``.get_contract`` — falls back to a simple
          keyword match (delegated to ``select_skill_from_bank``).
    """

    def __init__(self, skill_bank: Any) -> None:
        self._bank = skill_bank

    # ── SkillProvider interface ──────────────────────────────────────

    def select(
        self,
        query: str,
        *,
        state_summary: str = "",
        structured_state: Optional[Any] = None,
        current_predicates: Optional[Dict[str, float]] = None,
        top_k: int = 1,
    ) -> List[SkillGuidance]:
        if self._bank is None:
            return []

        # Rich path: SkillQueryEngine.select
        if hasattr(self._bank, "select"):
            try:
                results = self._bank.select(
                    query,
                    current_state=current_predicates,
                    current_predicates=current_predicates,
                    top_k=top_k,
                )
                if results:
                    return [
                        self._selection_result_to_guidance(r)
                        for r in results[:top_k]
                    ]
            except Exception:
                pass

        # Convenience wrapper
        if hasattr(self._bank, "query_for_decision_agent"):
            try:
                result = self._bank.query_for_decision_agent(
                    query,
                    current_state=current_predicates,
                    current_predicates=current_predicates,
                    top_k=top_k,
                )
                if result and result.get("skill_id"):
                    return [self._dict_to_guidance(result)]
            except Exception:
                pass

        # Fallback: agent_helper.select_skill_from_bank (keyword match)
        try:
            from .agent_helper import select_skill_from_bank
            result = select_skill_from_bank(
                self._bank,
                query,
                current_state=current_predicates,
                top_k=top_k,
            )
            if result and result.get("skill_id"):
                return [self._dict_to_guidance(result)]
        except Exception:
            pass

        return []

    def record_outcome(
        self,
        skill_id: str,
        *,
        outcome: str,
        reward: float = 0.0,
        steps_taken: int = 0,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._bank is None or not skill_id:
            return
        record = getattr(self._bank, "record_outcome", None)
        if record is not None:
            try:
                record(
                    skill_id,
                    outcome=outcome,
                    reward=reward,
                    steps_taken=steps_taken,
                    info=info or {},
                )
                return
            except Exception:
                pass

        # Best-effort fallback: bump pass-rate report if we can find one.
        bank = getattr(self._bank, "bank", self._bank)
        report_fn = getattr(bank, "record_execution", None)
        if report_fn is not None:
            try:
                report_fn(skill_id, success=(outcome == "success"))
            except Exception:
                pass

    def available_skills(self) -> List[str]:
        if self._bank is None:
            return []
        bank = getattr(self._bank, "bank", self._bank)
        ids = getattr(bank, "skill_ids", None)
        if ids is None:
            return []
        try:
            return list(ids)
        except TypeError:
            return []

    # ── Conversion helpers ───────────────────────────────────────────

    def _selection_result_to_guidance(self, r: Any) -> SkillGuidance:
        """Convert ``SkillSelectionResult`` → :class:`SkillGuidance`.

        Tolerant of missing fields (dataclass defaults are used for
        anything the result doesn't carry).  Also falls back to reading
        the full Skill object from the bank when the selection result
        only carries scoring — that way ``protocol_steps`` and contract
        are always populated when the bank has them.
        """
        def _get(attr: str, default: Any = None) -> Any:
            return getattr(r, attr, default)

        skill_id = _get("skill_id", "")
        guidance = SkillGuidance(
            skill_id=skill_id,
            skill_name=_get("skill_name", "") or "",
            why_selected=_get("why_selected", "") or "",
            confidence=float(_get("confidence", 0.0) or 0.0),
            relevance=float(_get("relevance", 0.0) or 0.0),
            applicability_score=float(_get("applicability_score", 0.0) or 0.0),
            pass_rate=_get("pass_rate"),
            execution_hint=_get("execution_hint", "") or "",
            preconditions=list(_get("preconditions", []) or []),
            termination_hint=_get("termination_hint", "") or "",
            failure_modes=list(_get("failure_modes", []) or []),
            eff_add=list(_get("expected_effects", []) or []),
            micro_plan=list(_get("micro_plan", []) or []),
        )

        # If the result already carries a contract dict, pick up eff_del.
        contract = _get("contract") or {}
        if isinstance(contract, dict):
            guidance.eff_add = list(contract.get("eff_add") or guidance.eff_add)
            guidance.eff_del = list(contract.get("eff_del") or [])

        # Enrich with protocol / required slots from the backing skill.
        self._enrich_from_skill(guidance, skill_id)
        return guidance

    def _dict_to_guidance(self, d: Dict[str, Any]) -> SkillGuidance:
        """Convert the dict returned by the fallback paths → :class:`SkillGuidance`."""
        protocol = d.get("protocol") or {}
        if not isinstance(protocol, dict):
            protocol = {}

        guidance = SkillGuidance(
            skill_id=d.get("skill_id", "") or "",
            skill_name=d.get("skill_name", "") or "",
            why_selected=d.get("why_selected", "") or "",
            confidence=float(d.get("confidence", 0.0) or 0.0),
            relevance=float(d.get("relevance", 0.0) or 0.0),
            applicability_score=float(d.get("applicability_score", 0.0) or 0.0),
            pass_rate=d.get("pass_rate"),
            execution_hint=d.get("execution_hint", "") or "",
            protocol_steps=list(protocol.get("steps", []) or []),
            preconditions=list(protocol.get("preconditions", []) or []),
            success_criteria=list(protocol.get("success_criteria", []) or []),
            abort_criteria=list(protocol.get("abort_criteria", []) or []),
            expected_duration=int(protocol.get("expected_duration", 0) or 0),
            termination_hint=d.get("termination_hint", "") or "",
            failure_modes=list(d.get("failure_modes", []) or []),
            eff_add=list(d.get("expected_effects", []) or []),
            micro_plan=list(d.get("micro_plan", []) or []),
        )
        self._enrich_from_skill(guidance, guidance.skill_id)
        return guidance

    def _enrich_from_skill(
        self,
        guidance: SkillGuidance,
        skill_id: str,
    ) -> None:
        """Populate protocol / contract / required_slots by reading the Skill object."""
        if not skill_id or self._bank is None:
            return
        bank = getattr(self._bank, "bank", self._bank)
        skill = None
        if hasattr(bank, "get_skill"):
            try:
                skill = bank.get_skill(skill_id)
            except Exception:
                skill = None

        if skill is not None:
            proto = getattr(skill, "protocol", None)
            if proto is not None:
                if not guidance.protocol_steps:
                    guidance.protocol_steps = list(getattr(proto, "steps", []) or [])
                if not guidance.preconditions:
                    guidance.preconditions = list(
                        getattr(proto, "preconditions", []) or []
                    )
                if not guidance.success_criteria:
                    guidance.success_criteria = list(
                        getattr(proto, "success_criteria", []) or []
                    )
                if not guidance.abort_criteria:
                    guidance.abort_criteria = list(
                        getattr(proto, "abort_criteria", []) or []
                    )
                if not guidance.expected_duration:
                    guidance.expected_duration = int(
                        getattr(proto, "expected_duration", 0) or 0
                    )

            if not guidance.skill_name:
                guidance.skill_name = getattr(skill, "name", "") or skill_id
            if not guidance.strategic_description:
                guidance.strategic_description = (
                    getattr(skill, "strategic_description", "") or ""
                )
            eh = getattr(skill, "execution_hint", None)
            if eh is not None and not guidance.execution_hint:
                guidance.execution_hint = (
                    getattr(eh, "execution_description", "") or ""
                )

        # Pull slot bindings when the skill exposes them (PLAN §10).
        slot_bindings = None
        if skill is not None:
            slot_bindings = getattr(skill, "slot_bindings", None)
        if slot_bindings is None:
            slot_bindings = getattr(bank, "get_slot_bindings", None)
            if callable(slot_bindings):
                try:
                    slot_bindings = slot_bindings(skill_id)
                except Exception:
                    slot_bindings = None
        if slot_bindings:
            required = getattr(slot_bindings, "required_slots", None)
            optional = getattr(slot_bindings, "optional_slots", None)
            if isinstance(slot_bindings, dict):
                required = slot_bindings.get("required_slots") or required
                optional = slot_bindings.get("optional_slots") or optional
            if required:
                guidance.required_slots = list(required)
            if optional:
                guidance.optional_slots = list(optional)

        # Fill eff_add / eff_del from the contract if not already set.
        contract = None
        if hasattr(bank, "get_contract"):
            try:
                contract = bank.get_contract(skill_id)
            except Exception:
                contract = None
        if contract is not None:
            if not guidance.eff_add:
                guidance.eff_add = list(
                    getattr(contract, "eff_add", set()) or set()
                )
            if not guidance.eff_del:
                guidance.eff_del = list(
                    getattr(contract, "eff_del", set()) or set()
                )


__all__ = [
    "NullSkillProvider",
    "SkillBankProvider",
    "SkillGuidance",
    "SkillProvider",
]
