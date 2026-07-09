"""Actor Agent — Tier-2 decision agent for the COS-PLAY pipeline.

This module implements **Agent 1** from ``plans/02-action-agent/PLAN-ACTION-AGENT.md``
§2.3: the trained actor that consumes a structured ``<state>…</state>``
schema from the Visual Grounding pipeline (``vlm_wrapper``), selects a
skill via an injected :class:`~decision_agents.skill_interface.SkillProvider`,
and emits one environment action per outer step.

Design goals (per the plan):

* **Schema-native** — the state representation is a parsed
  :class:`StateSchema`, not raw observation text (PLAN §7 Phase 2).
  Entity references like ``click(e5)`` resolve through the schema to
  real actionable handles (``click(bid)``) (PLAN §7 Phase 3).
* **Skill-agnostic** — skills are injected via the
  :class:`~decision_agents.skill_interface.SkillProvider` interface.
  The actor never imports from ``skill_agents`` directly.  That keeps
  Agent 1 and Agent 2 (Skill-Use) loosely coupled, per PLAN §2.3.
* **Single-MDP / Harness-driven** — the action vocabulary and the
  semantics of ``step(action)`` are owned by an injected
  :class:`~decision_agents.core.harness.Harness`.  Five reference
  harnesses ship under ``decision_agents/core/`` (game / web / OS /
  visual-reasoning / video-understanding); the actor stays
  task-agnostic.  This replaces the old two-level outer/inner MDP
  framing — see ``decision_agents/README.md`` "How the actor agent
  works" for the migration rationale.
* **Reward-aware** — forwards through
  :class:`~decision_agents.reward_func.RewardComputer` so the
  ``r_env + r_follow + r_cost`` shaping keeps working unchanged.
  When a harness is bound, ``observe_result`` consults
  ``harness.action_kind(action)`` so per-action-kind costs (VR
  ``LOOK`` / ``RETRIEVE``, video ``JUMP`` / ``FOCUS``) flow into
  ``r_cost`` correctly.

Backward compatibility
----------------------
Callers that pre-date the harness contract keep working unchanged:
when no ``harness=`` is supplied the actor auto-binds a
:class:`~decision_agents.core.harness_gym.GymHarness` over the env /
``info`` it receives.  The legacy ``hop_policy=`` and
``max_hops_per_step=`` kwargs are still accepted (and silently
ignored, with a one-shot :class:`DeprecationWarning`) so existing
``decision_agents.SFT.GPT4oCollectorActor`` and
``decision_agents.grpo.QwenVLActor`` constructors compile without
edits — see PHASE-7 entry in the patch-set log.

Relationship to :class:`~decision_agents.agent.VLMDecisionAgent`: the
older ``VLMDecisionAgent`` is a *text-observation* agent kept for
backward compatibility with ``scripts/qwen3_decision_agent.py`` and the
Pipeline A / B flows in :mod:`decision_agents.README`.  ``ActorAgent``
is its schema-native successor and the designated GRPO training target
(PLAN §6 Phase 1).
"""

from __future__ import annotations

import difflib
import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from API_func import ask_model
except ImportError:  # pragma: no cover — optional dep for offline tests
    ask_model = None

_LOGGER = logging.getLogger(__name__)

from .agent_helper import (
    EpisodicMemoryStore,
    HARD_SUMMARY_CHAR_LIMIT,
    infer_intention,
)
from .core.harness import Harness, HarnessState
from .core.harness_gym import GymHarness
from .reward_func import RewardComputer, RewardConfig, RewardResult
from .schema_parser import (
    Entity,
    ResolvedAction,
    StateSchema,
    parse_state_schema,
    resolve_entity_action,
)
from .skill_interface import (
    NullSkillProvider,
    SkillGuidance,
    SkillProvider,
)
from .skill_tracker import SkillTracker


# ──────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL: str = "Qwen/Qwen3.5-9B"  # project-wide actor backbone; see common/models.py BACKBONE_MODEL
MAX_LAST_ACTIONS: int = 5
MAX_PROGRESS_NOTES: int = 3
MAX_VALID_ACTIONS_IN_PROMPT: int = 16
MAX_ENTITIES_IN_PROMPT: int = 12
ANTI_REPETITION_WINDOW: int = 3
"""If the last N actions are identical with reward ≤ 0, pick an alternative."""


# ──────────────────────────────────────────────────────────────────────
# Per-step decision record
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ActorDecision:
    """Structured output of :meth:`ActorAgent.step`.

    The runner uses this to drive ``env.step`` and to build the
    :class:`Experience` payload.  Everything the actor did on this outer
    step is captured here so episodes can be replayed or scored offline.
    """

    action: str                                            # env-level action string
    resolved: Optional[ResolvedAction] = None               # entity-reference resolution
    intention: str = ""                                     # [TAG] subgoal phrase
    summary: str = ""                                       # compact key=value summary
    active_skill_id: Optional[str] = None
    reselected: bool = False
    reselect_reason: str = ""
    anti_repetition_triggered: bool = False
    valid_actions: List[str] = field(default_factory=list)
    reasoning: str = ""                                     # raw LLM reasoning (best-effort)
    queried_skill: bool = False                             # this step ran skill_provider.select
    queried_mem: bool = False                               # last harness.step was a RETRIEVE-class action
    parse_path: str = ""                                    # which strategy produced the action
    action_kind: str = ""                                   # harness.action_kind(action) — drives r_cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "resolved_action": (
                self.resolved.resolved if self.resolved else self.action
            ),
            "entity_ref": self.resolved.eid if self.resolved else None,
            "intention": self.intention,
            "summary": self.summary,
            "active_skill_id": self.active_skill_id,
            "reselected": self.reselected,
            "reselect_reason": self.reselect_reason,
            "anti_repetition_triggered": self.anti_repetition_triggered,
            "valid_actions": list(self.valid_actions),
            "reasoning": self.reasoning,
            "queried_skill": self.queried_skill,
            "queried_mem": self.queried_mem,
            "parse_path": self.parse_path,
            "action_kind": self.action_kind,
        }


# ──────────────────────────────────────────────────────────────────────
# Actor internal state (owned by ActorAgent, not the runner)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class InnerScratchpad:
    """Per-step / per-skill scratchpad shared across the harness loop.

    Accumulated across outer steps (memory hits + notes persist across a
    skill period; ``grounded_slots`` is reset on skill re-select).  In
    the unified single-MDP design the harness owns the side-effect
    semantics: VR / Video harnesses write into this scratchpad from
    their ``step()`` (``LOOK`` → ``grounded_slots``, ``RETRIEVE`` →
    ``memory_hits``, ``NOTE`` → ``notes``), and the actor's prompt
    builder renders the cumulative content on the next step.  Game /
    web / OS harnesses leave it untouched — their world-mutating
    actions don't need the scratchpad.

    Fields
    ------
    pending_ground_slots
        Slots the :class:`SkillTracker` flagged as missing at skill
        activation time.  Surfaced into the action prompt so the LLM
        knows what to ``LOOK`` at first when running against
        :class:`~decision_agents.core.harness_vr.VRHarness`.
    grounded_slots
        Slots that have been observed during this skill period, mapped
        to a best-effort resolved value (``"observed"`` from the VR
        harness; arbitrary tool output once Option B's visual-grounding
        handlers are wired).
    memory_hits
        Top-k hits from the most recent ``RETRIEVE`` actions.
        Rendered into the action prompt and logged on each
        :class:`Experience` so Skill-Bank mining can see which memories
        mattered.
    notes
        ``NOTE(text)`` arguments committed during this skill period —
        free-form strings the LLM can use as intermediate lemmas.
    """

    pending_ground_slots: List[str] = field(default_factory=list)
    grounded_slots: Dict[str, str] = field(default_factory=dict)
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.grounded_slots or self.memory_hits or self.notes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pending_ground_slots": list(self.pending_ground_slots),
            "grounded_slots": dict(self.grounded_slots),
            "memory_hits": list(self.memory_hits),
            "notes": list(self.notes),
        }


@dataclass
class ActorState:
    """Per-episode state the actor maintains across outer steps."""

    current_intention: str = ""
    last_actions: List[str] = field(default_factory=list)
    progress_notes: List[str] = field(default_factory=list)
    last_rewards: List[float] = field(default_factory=list)
    last_summary: str = ""
    last_schema: Optional[StateSchema] = None
    scratchpad: InnerScratchpad = field(default_factory=InnerScratchpad)


# ──────────────────────────────────────────────────────────────────────
# Actor agent
# ──────────────────────────────────────────────────────────────────────


class ActorAgent:
    """Schema-native decision agent.

    Parameters
    ----------
    model
        LLM used for intention inference and action selection.  Any
        model reachable through ``API_func.ask_model`` works
        (``Qwen/Qwen3.5-9B`` via vLLM by default — see
        ``common/models.py`` ``BACKBONE_MODEL``; ``gpt-5.5``, Claude
        Sonnet, and the deferred Qwen3-8B / Qwen3-VL tracks are
        reachable too).
    skill_provider
        Implementation of :class:`SkillProvider`.  Defaults to
        :class:`NullSkillProvider` (skill-free baseline).
    harness
        Optional :class:`~decision_agents.core.harness.Harness`
        instance.  When supplied it owns the per-step action vocabulary
        (``valid_actions``) and the cost bucket for reward shaping
        (``action_kind``).  When ``None`` the actor stays compatible
        with legacy callers: :func:`run_actor_episode` will auto-bind
        a :class:`~decision_agents.core.harness_gym.GymHarness` over
        the supplied env.
    reward_config
        Optional :class:`RewardConfig` — uses defaults if omitted.
    memory
        Optional :class:`EpisodicMemoryStore` for retrieval-class
        actions (e.g. ``VRHarness.step(RETRIEVE(q))``).  Defaults to
        an in-memory store with auto-detected embedder.
    intention_model
        Optional separate model for ``infer_intention`` calls, e.g. if
        you want a cheaper model for the tag prediction.  Falls back to
        *model* when None.
    hop_policy / max_hops_per_step
        **Deprecated** — accepted for backward compatibility with the
        pre-harness API but silently ignored.  The unified single-MDP
        loop replaces the inner-MDP scaffold; emit a one-shot
        :class:`DeprecationWarning` when supplied so callers know to
        drop them.  See ``decision_agents/README.md`` migration notes.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        skill_provider: Optional[SkillProvider] = None,
        harness: Optional[Harness] = None,
        reward_config: Optional[RewardConfig] = None,
        memory: Optional[EpisodicMemoryStore] = None,
        intention_model: Optional[str] = None,
        stall_window: int = 4,
        anti_repetition_window: int = ANTI_REPETITION_WINDOW,
        # ── deprecated kwargs (kept for back-compat) ─────────────────
        hop_policy: Any = None,
        max_hops_per_step: Optional[int] = None,
    ) -> None:
        if hop_policy is not None or max_hops_per_step is not None:
            warnings.warn(
                "ActorAgent: 'hop_policy' and 'max_hops_per_step' are "
                "deprecated and ignored — the inner-MDP scaffold has been "
                "replaced by the per-task Harness contract. See "
                "decision_agents/README.md (migration status).",
                DeprecationWarning,
                stacklevel=2,
            )
        self.model = model or DEFAULT_MODEL
        self.intention_model = intention_model or self.model
        self.skill_provider: SkillProvider = skill_provider or NullSkillProvider()
        self.reward_config = reward_config or RewardConfig()
        self.reward_computer = RewardComputer(self.reward_config)
        self.tracker = SkillTracker(stall_window=stall_window)
        self.memory = memory if memory is not None else _build_default_memory()
        self.anti_repetition_window = max(1, int(anti_repetition_window))
        self.state = ActorState()

        # Harness binding — None means "auto-bind GymHarness in
        # run_actor_episode".  When supplied we wire the side-effect
        # channels (scratchpad / memory / tracker) into the harness
        # immediately so VR / Video harnesses can mutate the actor's
        # scratchpad from their step().
        self.harness: Optional[Harness] = None
        if harness is not None:
            self.bind_harness(harness)

    # ── Episode lifecycle ────────────────────────────────────────────

    def bind_harness(self, harness: Harness) -> None:
        """Attach (or replace) the harness driving this actor.

        Wires the harness's optional side-effect channels into the
        actor's scratchpad / memory / tracker.  Safe to call repeatedly
        — :func:`run_actor_episode` calls it on every reset to refresh
        the scratchpad reference (it's rebuilt on skill re-select).
        """
        self.harness = harness
        bind_actor = getattr(harness, "bind_actor", None)
        if callable(bind_actor):
            try:
                bind_actor(
                    scratchpad=self.state.scratchpad,
                    memory=self.memory,
                    tracker=self.tracker,
                )
            except Exception as exc:  # pragma: no cover — defensive
                _LOGGER.warning("harness.bind_actor failed: %s", exc)

    def reset(self) -> None:
        """Forget episode-local state.  Call before each episode."""
        self.state = ActorState()
        self.tracker.reset()
        self.reward_computer.reset()
        # Re-seat the harness binding so it points at the fresh scratchpad.
        if self.harness is not None:
            self.bind_harness(self.harness)

    # ── Main per-step entry point ────────────────────────────────────

    def step(
        self,
        *,
        observation: str,
        schema_text: Optional[str] = None,
        schema: Optional[StateSchema] = None,
        task: str = "",
        valid_actions: Optional[List[str]] = None,
        info: Optional[Dict[str, Any]] = None,
        harness: Optional[Harness] = None,
    ) -> ActorDecision:
        """Run one MDP step.

        The caller supplies the structured schema (either pre-parsed via
        *schema* or as text via *schema_text*); when neither is given the
        actor falls back to a raw-observation path so it still works for
        callers that have not yet integrated the VLM grounding head
        (PLAN §7 Phase 1).

        ``harness`` (per-step override) lets a runner inject a different
        :class:`Harness` mid-episode (rare — typically you bind once via
        the constructor or :meth:`bind_harness`).  When the actor has
        no harness bound and none is supplied here, ``valid_actions`` /
        ``schema`` / ``info`` drive the legacy fallback in
        :func:`_resolve_valid_actions`.

        Returns an :class:`ActorDecision` the runner feeds to
        ``harness.step``.  The actor does NOT call the env — keeping it
        side-effect-free makes it trivially testable and
        GRPO-compatible.
        """
        info = info or {}
        active_harness = harness or self.harness

        # 1. Parse schema (Phase 2 input).
        parsed = schema or (parse_state_schema(schema_text) if schema_text else None)

        # 1b. Apply any in-step schema delta produced by the previous
        #     ``harness.step`` call (Phase 8.0).  Perception ops like
        #     ``LOOK("close button")`` write a new entity into
        #     ``info["schema_delta"]`` instead of having to re-emit the
        #     full ``<state>…</state>`` text.  The merge runs *before*
        #     ``compact_summary`` / ``_pick_action`` so the next action
        #     prompt sees the freshly-grounded entity.
        delta = info.get("schema_delta")
        if delta:
            parsed = self._merge_schema_delta(parsed, delta)
        self.state.last_schema = parsed

        # 2. Compact summary — either from the schema or from raw text.
        summary = (
            parsed.compact_summary()
            if parsed is not None
            else _compact_from_text(observation)
        )
        self.state.last_summary = summary

        # 3. Intention inference — runs BEFORE reselect (PLAN-ACTION-AGENT §1
        #    step 2) so it feeds the RAG skill-query and the action prompt
        #    with a fresh subgoal instead of last step's stale tag.
        intention = self._infer_intention(summary, task)
        self.state.current_intention = intention

        # 4. Determine the valid-action list.  Prefer the harness when
        #    bound — that's the unified MDP's source of truth.  Otherwise
        #    fall back to the legacy info / schema lookup so callers that
        #    haven't migrated yet keep working.
        valid = self._resolve_valid_actions(
            valid_actions=valid_actions,
            schema=parsed,
            observation=observation,
            info=info,
            harness=active_harness,
            intention=intention,
        )

        # 5. Re-selection check (PLAN §1 step 3).
        reselected = False
        reselect_reason = ""
        queried_skill = False
        should_reselect, reason = self.tracker.should_reselect(parsed)
        if should_reselect:
            # Report outcome of the skill we are about to replace.
            if self.tracker.active_skill_id is not None:
                record = self.tracker.finalize_active_skill(
                    outcome=reason or "switch"
                )
                if record:
                    self.skill_provider.record_outcome(
                        record["skill_id"],
                        outcome=record["outcome"],
                        reward=record["reward"],
                        steps_taken=record["steps_taken"],
                    )
            # Calling the provider at all counts as a QUERY_SKILL event
            # (PLAN §4 — reselect cost fires regardless of whether any
            # guidance comes back).
            queried_skill = True
            new_guidance = self._select_skill(parsed, summary, task)
            if new_guidance is not None:
                check = self.tracker.activate(new_guidance, parsed)
                reselected = True
                reselect_reason = reason or "reselect"
                # PLAN §10 — the tracker owns slot-coverage.  Seed the
                # scratchpad with its missing_slots so the harness can
                # treat them as observed-slot priors.
                self.state.scratchpad = InnerScratchpad(
                    pending_ground_slots=list(check.missing_slots),
                )
                # Re-seat the harness's scratchpad reference too.
                if active_harness is not None and active_harness is self.harness:
                    self.bind_harness(active_harness)
        # On non-reselect steps, keep last step's scratchpad but forget
        # transient bookkeeping — memory hits are the one thing worth
        # carrying across steps so the LLM can keep citing them.
        else:
            self.state.scratchpad = InnerScratchpad(
                memory_hits=list(self.state.scratchpad.memory_hits[-3:]),
                notes=list(self.state.scratchpad.notes[-3:]),
            )
            if active_harness is not None and active_harness is self.harness:
                self.bind_harness(active_harness)

        # 6. Action selection — single LLM call against the unified
        #    valid-action vocabulary the harness (or legacy fallback)
        #    just produced.
        action_text, reasoning, parse_path = self._pick_action(
            schema=parsed,
            summary=summary,
            task=task,
            valid_actions=valid,
            observation=observation,
        )

        # 7. Resolve entity references in the action (PLAN §7 Phase 3).
        resolved = resolve_entity_action(action_text, parsed)
        final_action = resolved.resolved or action_text

        # 8. Anti-repetition guard.
        final_action, anti_rep = self._anti_repetition(final_action, valid)

        # 9. Look up the cost bucket from the harness when available so
        #    observe_result can fold it into r_cost.  The legacy game /
        #    web / OS path returns "primitive" — same as today's r_cost
        #    semantics, so r_total stays unchanged for those tasks.
        kind = ""
        if active_harness is not None:
            try:
                kind = active_harness.action_kind(final_action) or ""
            except Exception:  # pragma: no cover — defensive
                kind = ""

        decision = ActorDecision(
            action=final_action,
            resolved=resolved,
            intention=intention,
            summary=summary,
            active_skill_id=self.tracker.active_skill_id,
            reselected=reselected,
            reselect_reason=reselect_reason,
            anti_repetition_triggered=anti_rep,
            valid_actions=valid,
            reasoning=reasoning,
            queried_skill=queried_skill,
            queried_mem=("retrieve" in (kind or "").lower()),
            parse_path=parse_path,
            action_kind=kind,
        )
        return decision

    # ── Schema-delta merge (Phase 8.0) ───────────────────────────────

    @staticmethod
    def _merge_schema_delta(
        schema: Optional[StateSchema],
        delta: Any,
    ) -> StateSchema:
        """Merge a list of in-step entity additions/updates into ``schema``.

        Called from :meth:`step` when the previous ``harness.step``
        attached ``info["schema_delta"]``.  Two acceptable input shapes:

        * ``list[Entity]`` — fully-typed entities (preferred; what
          ``VRHarness`` produces).
        * ``list[dict]``   — raw dicts with at least an ``eid`` key;
          unknown keys are stashed in ``Entity.extra`` so the parser
          contract isn't broken.  Used by lightweight harnesses that
          don't want a hard dep on ``schema_parser``.

        Merge semantics:

        * **New eid** → appended to ``entities`` and ``entity_order``.
        * **Existing eid** → fields with non-empty new values overwrite
          the old ones; ``attributes`` and ``extra`` dicts are merged
          (delta wins on key collision).  Lists (``affords``) are
          de-duplicated, preserving original order then appending new
          items.

        When ``schema`` is ``None`` (no ``schema_text`` was supplied),
        a fresh empty :class:`StateSchema` is created so callers always
        get a usable object back.  This matters for VR / Video where
        the actor may rely entirely on perception ops to populate the
        entity table.
        """
        if not delta:
            return schema if schema is not None else StateSchema()

        out = schema if schema is not None else StateSchema()
        for raw in delta:
            entity = ActorAgent._coerce_entity(raw)
            if entity is None:
                continue
            existing = out.entities.get(entity.eid)
            if existing is None:
                out.entities[entity.eid] = entity
                if entity.eid not in out.entity_order:
                    out.entity_order.append(entity.eid)
                continue

            # In-place field-wise merge (delta wins on non-empty).
            for fname in ("type", "label", "ontology"):
                new_val = getattr(entity, fname, "")
                if new_val:
                    setattr(existing, fname, new_val)
            if entity.bid is not None:
                existing.bid = entity.bid
            if entity.pos is not None:
                existing.pos = entity.pos
            if entity.state is not None:
                existing.state = entity.state
            if entity.value is not None:
                existing.value = entity.value
            if entity.attributes:
                existing.attributes.update(entity.attributes)
            if entity.uncertainty:
                existing.uncertainty.update(entity.uncertainty)
            if entity.extra:
                existing.extra.update(entity.extra)
            if entity.affords:
                seen = set(existing.affords)
                for aff in entity.affords:
                    if aff not in seen:
                        existing.affords.append(aff)
                        seen.add(aff)
        return out

    @staticmethod
    def _coerce_entity(raw: Any) -> Optional[Entity]:
        """Return a typed :class:`Entity` from either an Entity or a dict.

        Returns ``None`` for malformed input (no ``eid``) — callers
        skip such rows rather than raise so a sloppy harness never
        crashes the actor mid-rollout.
        """
        if isinstance(raw, Entity):
            return raw
        if not isinstance(raw, dict):
            return None
        eid = raw.get("eid")
        if not isinstance(eid, str) or not eid:
            return None

        # Pull the fields the parser knows about; rest go to ``extra``.
        known = {
            "eid", "type", "label", "bid", "pos", "ontology",
            "state", "value", "attributes", "affords", "uncertainty",
            "extra",
        }
        extra = dict(raw.get("extra") or {})
        for k, v in raw.items():
            if k not in known:
                extra[k] = v

        return Entity(
            eid=eid,
            type=str(raw.get("type", "") or ""),
            label=str(raw.get("label", "") or ""),
            bid=raw.get("bid"),
            pos=tuple(raw["pos"]) if raw.get("pos") is not None else None,  # type: ignore[arg-type]
            ontology=str(raw.get("ontology", "") or ""),
            extra=extra,
            state=raw.get("state"),
            value=raw.get("value"),
            attributes=dict(raw.get("attributes") or {}),
            affords=list(raw.get("affords") or []),
            uncertainty=dict(raw.get("uncertainty") or {}),
        )

    # ── Update after env.step ────────────────────────────────────────

    def observe_result(
        self,
        decision: ActorDecision,
        *,
        reward: float,
        next_observation: str = "",
        next_schema_text: Optional[str] = None,
        next_schema: Optional[StateSchema] = None,
        done: bool = False,
    ) -> RewardResult:
        """Fold env feedback back into the actor state.

        Returns the :class:`RewardResult` for the step — the runner
        should stash it on the :class:`Experience` it builds.
        """
        schema_after = next_schema or (
            parse_state_schema(next_schema_text) if next_schema_text else None
        )

        # Update short history buffers.
        st = self.state
        st.last_actions.append(decision.action)
        if len(st.last_actions) > MAX_LAST_ACTIONS:
            st.last_actions = st.last_actions[-MAX_LAST_ACTIONS:]
        st.last_rewards.append(float(reward or 0.0))
        if len(st.last_rewards) > MAX_LAST_ACTIONS:
            st.last_rewards = st.last_rewards[-MAX_LAST_ACTIONS:]
        if reward and float(reward) > 0:
            st.progress_notes.append(
                f"+{float(reward):g} after {decision.action}"[:80]
            )
            if len(st.progress_notes) > MAX_PROGRESS_NOTES:
                st.progress_notes = st.progress_notes[-MAX_PROGRESS_NOTES:]

        # Tracker update + reward computation.
        self.tracker.record_step(
            reward=reward,
            schema_after=schema_after,
            advance_protocol=not _is_noop_action(decision.action),
        )

        # Derive the action_type for reward-cost bookkeeping.
        action_type = (
            "CALL_SKILL"
            if self.tracker.active_skill_id is not None
            else "primitive"
        )
        contract = self._active_skill_contract_view()
        rr = self.reward_computer.compute_reward(
            r_env=float(reward or 0.0),
            action_type=action_type,
            observation=next_observation or "",
            active_skill_id=self.tracker.active_skill_id,
            skill_contract=contract,
            queried_skill=bool(decision.queried_skill),
            queried_mem=bool(decision.queried_mem),
            action_kind=decision.action_kind or "",
        )

        # Push to memory so RETRIEVE hops get a grounded store.
        if self.memory is not None:
            try:
                next_summary = (
                    schema_after.compact_summary()
                    if schema_after is not None
                    else _compact_from_text(next_observation)
                )
                self.memory.add_experience(
                    state_summary=decision.summary,
                    action=decision.action,
                    next_state_summary=next_summary,
                    done=done,
                )
            except Exception:
                pass

        # Episode termination: finalize the active skill with the right
        # outcome so the provider can update its statistics.
        if done and self.tracker.active_skill_id is not None:
            outcome = "success" if float(reward or 0.0) > 0 else "timeout"
            record = self.tracker.finalize_active_skill(outcome=outcome)
            if record:
                self.skill_provider.record_outcome(
                    record["skill_id"],
                    outcome=record["outcome"],
                    reward=record["reward"],
                    steps_taken=record["steps_taken"],
                )

        return rr

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _select_skill(
        self,
        schema: Optional[StateSchema],
        summary: str,
        task: str,
    ) -> Optional[SkillGuidance]:
        """Consult the injected :class:`SkillProvider` for guidance."""
        intention = self.state.current_intention or ""
        query_parts = [p for p in (task, intention, summary) if p]
        query = " | ".join(query_parts)[:1500] or summary or task

        predicates = _predicates_from_schema(schema)
        results = self.skill_provider.select(
            query,
            state_summary=summary,
            structured_state=schema,
            current_predicates=predicates,
            top_k=1,
        )
        return results[0] if results else None

    def _resolve_valid_actions(
        self,
        *,
        valid_actions: Optional[List[str]],
        schema: Optional[StateSchema],
        observation: str,
        info: Dict[str, Any],
        harness: Optional[Harness],
        intention: str,
    ) -> List[str]:
        """Pick the actor's action vocabulary for this step.

        Priority:

        1. Bound :class:`Harness` (the unified MDP source of truth).
        2. Explicit *valid_actions* argument.
        3. ``info["valid_actions"]`` / ``info["available_actions"]``.
        4. ``schema.actions`` from the parsed schema.
        5. Regex over *observation* (legacy fallback for the
           text-observation gymv runner).
        """
        if harness is not None:
            try:
                state = HarnessState(
                    observation=observation,
                    schema=schema,
                    info=dict(info or {}),
                    intention=intention or "",
                    t=len(self.state.last_actions),
                )
                actions = list(harness.valid_actions(state))
                if actions:
                    return actions[:MAX_VALID_ACTIONS_IN_PROMPT]
            except Exception as exc:  # pragma: no cover — defensive
                _LOGGER.warning("harness.valid_actions failed: %s", exc)
        return _resolve_valid_actions_legacy(
            valid_actions, schema, observation, info
        )

    def _pick_action(
        self,
        *,
        schema: Optional[StateSchema],
        summary: str,
        task: str,
        valid_actions: List[str],
        observation: str,
    ) -> Tuple[str, str, str]:
        """Return ``(action_text, reasoning, parse_path)``.

        Priority:

        1. Active skill's current protocol step (when it is a literal
           action and present in *valid_actions*).
        2. LLM over the compact prompt (when ``ask_model`` is available).
           The reply is decoded via the multi-strategy pipeline in
           :func:`_extract_action_from_reply` (exact → numbered →
           entity-ref → edit distance → token overlap).
        3. Deterministic fallback — first valid action or ``"no-op"``.

        ``parse_path`` is a short tag identifying which branch produced
        the action; it's logged on ``ActorDecision`` to let the GRPO
        pipeline quantify how often the LoRA actually chose a valid
        action without the fallback stack rescuing it.
        """
        # 1. Follow the active skill's protocol step when it matches.
        current_step = self.tracker.current_step()
        if current_step:
            exact = _match_valid_action(current_step, valid_actions)
            if exact is not None:
                return exact, f"protocol step: {current_step}", "protocol"

        # 2. LLM prompt path — routed through the ``_call_llm`` seam so
        #    subclasses can swap backend (e.g. Qwen/Qwen3.5-9B via vLLM)
        #    and attach images without touching this pipeline.
        prompt = self._build_action_prompt(
            schema=schema,
            summary=summary,
            task=task,
            valid_actions=valid_actions,
            observation=observation,
        )
        reply = self._call_llm(prompt, temperature=0.3, max_tokens=200) or ""
        if reply:
            action_text, parse_path = _extract_action_from_reply(reply, valid_actions)
            if action_text:
                return action_text, reply[:400], f"llm:{parse_path}"

        # 3. Deterministic fallback.
        if valid_actions:
            return valid_actions[0], "fallback: first valid action", "fallback_first"
        return "no-op", "fallback: no valid actions", "fallback_noop"

    def _build_action_prompt(
        self,
        *,
        schema: Optional[StateSchema],
        summary: str,
        task: str,
        valid_actions: List[str],
        observation: str,
    ) -> str:
        """Compose the action-selection prompt (compact, schema-aware).

        Renders — in order — task, intention, state summary, entity
        block, active-skill block, the inner-MDP scratchpad (memory hits
        + grounded slots + committed subgoals), recent actions / rewards,
        recent progress notes, and a numbered valid-action list.

        Falls back to ``schema.goal`` / ``schema.task`` when the caller
        passes an empty ``task`` so the prompt never renders
        ``"Task: (unspecified)"`` just because the external runner
        forgot to thread the task text through.
        """
        g = self.tracker.active_guidance
        skill_block = _format_skill_block(g, self.tracker.progress_marker())
        entity_block = _format_entity_block(schema)
        scratchpad_block = _format_scratchpad(self.state.scratchpad)
        recent = " | ".join(self.state.last_actions[-3:]) or "(none)"
        recent_rewards = (
            ", ".join(f"{r:+.2f}" for r in self.state.last_rewards[-3:]) or "-"
        )
        recent_notes = " | ".join(self.state.progress_notes[-3:]) or ""

        numbered_actions = "\n".join(
            f"  {i+1}. {a}" for i, a in enumerate(valid_actions)
        )

        effective_task = task or (
            (getattr(schema, "goal", "") or getattr(schema, "task", ""))
            if schema is not None
            else ""
        )

        parts: List[str] = [
            "You are an Actor Agent choosing ONE environment action.",
            "",
            f"Task: {effective_task or '(unspecified)'}",
            f"Intention: {self.state.current_intention or '(tbd)'}",
            f"State summary: {summary}",
        ]
        if schema is None and observation:
            parts.extend([
                "",
                "Observation (text fallback):",
                observation[:1500],
            ])
        if entity_block:
            parts.extend(["", "Entities:", entity_block])
        if skill_block:
            parts.extend(["", "Active skill:", skill_block])
        if scratchpad_block:
            parts.extend(["", "Inner reasoning so far:", scratchpad_block])
        parts.extend([
            "",
            f"Recent actions: {recent}",
            f"Recent rewards: {recent_rewards}",
        ])
        if recent_notes:
            parts.append(f"Recent progress: {recent_notes}")
        parts.extend([
            "",
            "Valid actions (pick EXACTLY one, copy the string verbatim; "
            "entity references like click(e5) are also accepted when "
            "the action list permits):",
            numbered_actions,
            "",
            "Output format (strict):",
            "THOUGHT: <at most 2 sentences>",
            "ACTION: <one of the valid actions, or its 1-based number>",
        ])
        return "\n".join(parts)

    # ── LLM seam ─────────────────────────────────────────────────────
    #
    # ``_call_llm`` is the single point of contact between the actor's
    # per-step pipeline and a language model.  The default uses the
    # text-only ``API_func.ask_model`` so existing callers (and the
    # offline test suite that monkeypatches ``ask_model = None``) keep
    # working unchanged.
    #
    # Two specialised subclasses live alongside this module:
    #
    # * :class:`decision_agents.SFT.GPT4oCollectorActor` — uses the SFT
    #   teacher (``gpt-5.5``; class name retained for back-compat),
    #   sends the screenshot as a vision content part, and writes
    #   per-step SFT records that ``trainer/SFT/data_loader.py`` can
    #   consume directly.
    # * :class:`decision_agents.grpo.QwenVLActor` — routes the prompt
    #   through :class:`trainer.coevolution.vllm_client.AsyncVLLMClient`
    #   against ``Qwen/Qwen3.5-9B`` with hot-swappable LoRA adapters,
    #   and emits :class:`trainer.common.metrics.RolloutStep` records
    #   for the GRPO trainer.
    #
    # Subclasses MUST keep this contract:
    #   - return a *string* (empty string on failure, never None);
    #   - never raise — the ``_pick_action`` pipeline expects to fall
    #     through to the deterministic fallback when the LLM is silent.
    #
    def _call_llm(
        self,
        prompt: str,
        *,
        images: Optional[List[Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 200,
    ) -> str:
        """Default text-only LLM call via :func:`API_func.ask_model`.

        ``images`` is accepted for forward compatibility — the default
        impl ignores it because the GPT-4o text path predates the
        multimodal split.  The vision-aware subclasses
        (``GPT4oCollectorActor``, ``QwenVLActor``) override this method
        and consume the list of :class:`~decision_agents.core.VisualInput`
        objects.
        """
        if ask_model is None:
            return ""
        try:
            reply = ask_model(
                prompt,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            return ""
        return reply or ""

    def _infer_intention(self, summary: str, task: str) -> str:
        """Wrapper around :func:`agent_helper.infer_intention`."""
        if not summary or ask_model is None:
            return self.state.current_intention
        try:
            return infer_intention(
                summary,
                game=None,
                model=self.intention_model,
                context={
                    "last_actions": self.state.last_actions,
                    "progress_notes": self.state.progress_notes,
                    "task": task,
                },
            )
        except Exception:
            return self.state.current_intention

    def _anti_repetition(
        self,
        action: str,
        valid_actions: List[str],
    ) -> Tuple[str, bool]:
        """Return ``(action, triggered)``.

        PLAN §1 step 7.  If the last *anti_repetition_window* actions
        were identical AND all rewards in that window were ≤ 0, we swap
        to a different valid action (deterministically the next one in
        the list so tests are reproducible).
        """
        st = self.state
        window = self.anti_repetition_window
        if len(st.last_actions) < window or not valid_actions:
            return action, False
        recent = st.last_actions[-window:]
        recent_r = st.last_rewards[-window:] if st.last_rewards else []
        if len(recent_r) < window:
            return action, False
        if all(a == action for a in recent) and all(r <= 0 for r in recent_r):
            alternatives = [a for a in valid_actions if a != action]
            if alternatives:
                return alternatives[0], True
        return action, False

    def _active_skill_contract_view(self) -> Any:
        """Return a shim object exposing ``eff_add`` / ``eff_del`` for
        :class:`RewardComputer`.

        We don't want to pull in ``skill_agents.stage3_mvp.schemas``
        here — RewardComputer only calls ``getattr(contract,
        "eff_add", set())`` — so we build a tiny anonymous object.
        """
        g = self.tracker.active_guidance
        if g is None:
            return None

        class _ContractView:
            eff_add = set(g.eff_add)
            eff_del = set(g.eff_del)

        return _ContractView


# ──────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────


def run_actor_episode(
    env: Any,
    *,
    agent: Optional[ActorAgent] = None,
    model: Optional[str] = None,
    skill_provider: Optional[SkillProvider] = None,
    harness: Optional[Harness] = None,
    reward_config: Optional[RewardConfig] = None,
    task: str = "",
    max_steps: int = 200,
    schema_from_info: Callable[[Dict[str, Any]], Optional[str]] = None,
    verbose: bool = False,
    # Deprecated kwargs preserved for back-compat.
    hop_policy: Any = None,
) -> "Episode":
    """End-to-end runner that drives :class:`ActorAgent` against an env.

    The runner auto-binds a
    :class:`~decision_agents.core.harness_gym.GymHarness` over *env*
    when no explicit ``harness=`` is supplied (and the agent has none
    bound yet).  This keeps every legacy caller —
    ``scripts/qwen3_decision_agent.py``, the unit tests, the SFT /
    GRPO runners — working byte-identically against the unified
    single-MDP loop.

    Both 4-tuple and 5-tuple ``env.step`` returns are accepted; the
    GymHarness folds ``terminated or truncated`` into a single
    ``done`` bool.

    The structured schema is expected under ``info["schema"]``
    (override via *schema_from_info*).  When the schema is absent the
    actor falls back to the text-observation path so this runner
    works with both Phase 1 (text) and Phase 2 (schema) environments
    (PLAN §7).
    """
    from data_structure.experience import Experience, Episode

    if hop_policy is not None:
        warnings.warn(
            "run_actor_episode: 'hop_policy' is deprecated and ignored.",
            DeprecationWarning,
            stacklevel=2,
        )

    if agent is None:
        agent = ActorAgent(
            model=model,
            skill_provider=skill_provider,
            harness=harness,
            reward_config=reward_config,
        )

    # Bind a harness if neither the actor nor the caller supplied one
    # (the legacy contract).  GymHarness's ``valid_actions`` mirrors
    # the priority used by the pre-harness ``_resolve_valid_actions``
    # so the action vocabulary the actor sees is unchanged.
    bound_harness = harness or agent.harness
    if bound_harness is None:
        bound_harness = GymHarness(env)
        agent.bind_harness(bound_harness)
    elif agent.harness is not bound_harness:
        agent.bind_harness(bound_harness)

    agent.reset()

    if schema_from_info is None:
        schema_from_info = lambda info: (info or {}).get("schema") or \
            (info or {}).get("schema_text")

    obs, info = bound_harness.reset()
    observation = str(obs) if obs is not None else ""
    info = dict(info or {})
    episode_task = task or info.get("task", "")

    experiences: List[Experience] = []
    done = False
    step_count = 0

    while step_count < max_steps:
        schema_text = schema_from_info(info)
        valid_actions = info.get("valid_actions") or info.get("available_actions")

        decision = agent.step(
            observation=observation,
            schema_text=schema_text,
            task=episode_task,
            valid_actions=valid_actions,
            info=info,
        )

        env_action = (
            decision.resolved.resolved
            if decision.resolved is not None
            else decision.action
        )
        next_obs, reward, done, next_info = bound_harness.step(env_action)

        next_observation = str(next_obs) if next_obs is not None else ""
        next_info = dict(next_info or {})
        next_schema_text = schema_from_info(next_info)

        rr = agent.observe_result(
            decision,
            reward=float(reward or 0.0),
            next_observation=next_observation,
            next_schema_text=next_schema_text,
            done=done,
        )

        exp = Experience(
            state=observation,
            action=decision.action,
            reward=float(reward or 0.0),
            next_state=next_observation,
            done=done,
            intentions=decision.intention or None,
            tasks=episode_task or None,
            sub_tasks=decision.active_skill_id,
        )
        exp.idx = step_count
        exp.summary_state = decision.summary or None
        exp.reward_details = rr.to_dict()
        exp.action_type = (
            "CALL_SKILL" if decision.active_skill_id else "primitive"
        )
        exp.available_actions = list(decision.valid_actions) or None
        extras = getattr(exp, "extras", None)
        if extras is None:
            extras = {}
        extras["reselected"] = decision.reselected
        extras["reselect_reason"] = decision.reselect_reason
        extras["anti_repetition_triggered"] = decision.anti_repetition_triggered
        extras["queried_skill"] = decision.queried_skill
        extras["queried_mem"] = decision.queried_mem
        extras["parse_path"] = decision.parse_path
        extras["action_kind"] = decision.action_kind
        extras["scratchpad"] = agent.state.scratchpad.to_dict()
        if decision.reasoning:
            extras["reasoning"] = decision.reasoning[:400]
        exp.extras = extras
        experiences.append(exp)

        if verbose:
            print(
                f"  step {step_count}: action={decision.action!r} "
                f"kind={decision.action_kind} "
                f"skill={decision.active_skill_id} "
                f"{rr}"
            )

        observation = next_observation
        info = next_info
        step_count += 1
        if done:
            break

    cumulative = agent.reward_computer.cumulative
    episode = Episode(
        experiences=experiences,
        task=episode_task or "Unspecified task",
        env_name=info.get("env_name") or info.get("game"),
        game_name=info.get("game_name") or info.get("game"),
        metadata={
            "done": done,
            "steps": step_count,
            "cumulative_reward": cumulative.to_dict(),
            "skill_history": agent.tracker.state.history,
            "agent_state": {
                "current_intention": agent.state.current_intention,
                "last_actions": agent.state.last_actions,
                "progress_notes": agent.state.progress_notes,
                "active_skill_id": agent.tracker.active_skill_id,
            },
        },
    )
    episode.set_outcome()
    return episode


# ──────────────────────────────────────────────────────────────────────
# Module-private helpers
# ──────────────────────────────────────────────────────────────────────


_ACTION_LINE_RE = re.compile(r"ACTION\s*:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)


def _compact_from_text(observation: str) -> str:
    """Lightweight fallback when the actor has no parsed schema."""
    from .agent_helper import compact_text_observation
    return compact_text_observation(observation, max_chars=HARD_SUMMARY_CHAR_LIMIT)


def _resolve_valid_actions_legacy(
    valid_actions: Optional[List[str]],
    schema: Optional[StateSchema],
    observation: str,
    info: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Pre-harness valid-action resolution (used as the safety net).

    Priority: explicit *valid_actions* argument →
    ``info["valid_actions"]`` / ``info["available_actions"]`` →
    schema ``<actions>`` → regex over ``observation``.

    Used when the actor has no harness bound — the unified-MDP path
    prefers ``Harness.valid_actions(state)`` instead.
    """
    if valid_actions:
        return [str(a) for a in valid_actions][:MAX_VALID_ACTIONS_IN_PROMPT]
    info = info or {}
    candidate = info.get("valid_actions") or info.get("available_actions")
    if candidate:
        return [str(a) for a in candidate][:MAX_VALID_ACTIONS_IN_PROMPT]
    if schema and schema.actions:
        return schema.actions[:MAX_VALID_ACTIONS_IN_PROMPT]

    m = re.search(
        r"[Vv]alid\s+actions?\s*[:\-]\s*(.+?)(?:\n|\.|$)", observation or ""
    )
    if m:
        raw = m.group(1).strip()
        return [a.strip() for a in re.split(r"[,;]", raw) if a.strip()]
    return []


def _match_valid_action(
    candidate: str,
    valid_actions: List[str],
) -> Optional[str]:
    """Return the valid-action string that matches *candidate*, or None."""
    if not candidate or not valid_actions:
        return None
    c = candidate.strip().strip(".").strip()
    for a in valid_actions:
        if a == c:
            return a
    lc = c.lower()
    for a in valid_actions:
        if a.lower() == lc:
            return a
    # prefix / substring tolerance
    for a in valid_actions:
        al = a.lower()
        if al in lc or lc in al:
            return a
    return None


_NUMBERED_RE = re.compile(r"^\s*(\d+)\s*[\.\)\-:]?\s*$")
_ENTITY_REF_RE = re.compile(r"\w+\s*\(\s*e\d+\s*\)")
_TOKEN_RE = re.compile(r"\w+")


def _extract_action_from_reply(
    reply: str,
    valid_actions: List[str],
) -> Tuple[Optional[str], str]:
    """Pull an action string from the LLM reply.

    Returns ``(action, parse_path)`` where ``parse_path`` names which
    strategy succeeded (``"exact"``, ``"numbered"``, ``"entity_ref"``,
    ``"edit_distance"``, ``"token_overlap"``, ``"loose"``) or ``""``
    when nothing matched.  This lets the pipeline log how often the
    LLM needed the fallback stack, which is directly actionable for
    GRPO reward shaping (PLAN-ACTION-AGENT §1 step 6).
    """
    if not reply:
        return None, ""

    # Strategy 0 — pull the ``ACTION:`` line if present.
    m = _ACTION_LINE_RE.search(reply)
    cand = m.group(1).strip().strip("`").strip('"').strip() if m else ""

    if cand:
        # 1. Exact / caseless / substring match against the valid list.
        matched = _match_valid_action(cand, valid_actions)
        if matched is not None:
            return matched, "exact"

        # 2. Numbered selection ("1", "1.", "1)", "1-", "1:").
        num = _NUMBERED_RE.match(cand)
        if num:
            idx = int(num.group(1)) - 1
            if 0 <= idx < len(valid_actions):
                return valid_actions[idx], "numbered"

        # 3. Entity-reference pass-through (env runner resolves them).
        if _ENTITY_REF_RE.fullmatch(cand):
            return cand, "entity_ref"

        # 4. Edit-distance fallback — catches typos like "Righ" vs "[Right]".
        if valid_actions:
            close = difflib.get_close_matches(cand, valid_actions, n=1, cutoff=0.6)
            if close:
                return close[0], "edit_distance"

            # Same, but caseless, since difflib is case-sensitive.
            lowered = [a.lower() for a in valid_actions]
            close = difflib.get_close_matches(cand.lower(), lowered, n=1, cutoff=0.6)
            if close:
                return valid_actions[lowered.index(close[0])], "edit_distance"

        # 5. Token-overlap fallback — rank valid actions by shared tokens.
        c_tokens = {t for t in _TOKEN_RE.findall(cand.lower()) if len(t) >= 2}
        if c_tokens and valid_actions:
            scored: List[Tuple[int, int, str]] = []
            for a in valid_actions:
                a_tokens = {t for t in _TOKEN_RE.findall(a.lower()) if len(t) >= 2}
                overlap = len(a_tokens & c_tokens)
                if overlap > 0:
                    scored.append((-overlap, len(a), a))
            if scored:
                scored.sort()
                return scored[0][2], "token_overlap"

    # Strategy 6 — loose search over the whole reply.
    for a in valid_actions:
        if a and a in reply:
            return a, "loose"

    # Strategy 7 — a trailing digit in the reply is a numbered selection.
    trailing = re.search(r"(\d+)\s*$", reply.strip())
    if trailing:
        idx = int(trailing.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx], "numbered"

    return None, ""


def _predicates_from_schema(
    schema: Optional[StateSchema],
) -> Optional[Dict[str, float]]:
    """Convert schema slots into a ``{predicate: float}`` dict.

    The skill-selection engine scores skill effects against these; we
    emit conservative 1.0 indicators for flags that are clearly TRUE
    and leave anything uncertain / null absent.  See PLAN-SKILL-BANK §3
    for the contract.
    """
    if schema is None:
        return None
    predicates: Dict[str, float] = {}
    sf = schema.state_flags
    if sf.scene_type:
        predicates[f"scene_type={sf.scene_type}"] = 1.0
    if sf.phase:
        predicates[f"phase={sf.phase}"] = 1.0
    if sf.progress is not None:
        predicates["has_progress"] = 1.0
    if sf.error:
        predicates["error_present"] = 1.0
    if sf.dialog_open:
        predicates["dialog_open"] = 1.0
    if sf.input_pending:
        predicates["input_pending"] = 1.0
    for ent in schema.entities.values():
        if ent.ontology:
            predicates[f"has:{ent.ontology}"] = 1.0
    if schema.targets.target:
        predicates["has_target"] = 1.0
    if schema.targets.blocker:
        predicates["has_blocker"] = 1.0
    if schema.targets.candidate_set:
        predicates["has_candidate_set"] = 1.0
    return predicates or None


def _format_skill_block(
    g: Optional[SkillGuidance],
    progress: str,
) -> str:
    """Render a skill guidance package for the prompt."""
    if g is None:
        return ""
    parts: List[str] = [f"[{g.skill_id}] {g.skill_name or g.skill_id}"]
    if g.execution_hint:
        parts.append(f"  how: {g.execution_hint[:140]}")
    if g.strategic_description and not g.execution_hint:
        parts.append(f"  strategy: {g.strategic_description[:140]}")
    if g.protocol_steps:
        parts.append(f"  plan ({progress}):")
        for i, step in enumerate(g.protocol_steps[:6]):
            parts.append(f"    {i+1}. {step}")
    if g.preconditions:
        parts.append(f"  preconditions: {'; '.join(g.preconditions[:3])}")
    if g.termination_hint:
        parts.append(f"  done when: {g.termination_hint[:100]}")
    if g.failure_modes:
        parts.append(f"  watch for: {'; '.join(g.failure_modes[:2])}")
    return "\n".join(parts)


def _format_entity_block(schema: Optional[StateSchema]) -> str:
    """Render the most-relevant entities from the schema."""
    if schema is None or not schema.entities:
        return ""

    rows: List[str] = []
    ordered = [
        schema.entities[eid]
        for eid in schema.entity_order[:MAX_ENTITIES_IN_PROMPT]
        if eid in schema.entities
    ]
    for ent in ordered:
        pieces = [f"{ent.eid}"]
        if ent.label:
            pieces.append(ent.label)
        if ent.ontology:
            pieces.append(f"<{ent.ontology}>")
        if ent.affords:
            pieces.append("affords=" + ",".join(ent.affords[:4]))
        if ent.bid:
            pieces.append(f"bid={ent.bid}")
        rows.append("  " + " · ".join(pieces))
    if schema.targets.target:
        rows.append(f"  ** target={schema.targets.target} **")
    return "\n".join(rows)


_NOOP_PATTERNS = ("no-op", "noop", "stay", "wait")


def _is_noop_action(action: str) -> bool:
    a = (action or "").lower().strip()
    return any(p == a for p in _NOOP_PATTERNS)


def _format_scratchpad(sp: Optional[InnerScratchpad]) -> str:
    """Render :class:`InnerScratchpad` for the action prompt.

    Returns an empty string when there's nothing to show so the prompt
    doesn't get polluted with ``Inner reasoning so far: (empty)``.
    """
    if sp is None or sp.is_empty():
        return ""
    rows: List[str] = []
    if sp.grounded_slots:
        slots = ", ".join(f"{k}={v}" for k, v in list(sp.grounded_slots.items())[:5])
        rows.append(f"  grounded: {slots}")
    for hit in sp.memory_hits[-3:]:
        rows.append(f"  memory: {hit.get('hit', '')}")
    for note in sp.notes[-3:]:
        rows.append(f"  subgoal: {note}")
    return "\n".join(rows)


def _build_default_memory() -> Optional[EpisodicMemoryStore]:
    try:
        from rag import get_text_embedder
        return EpisodicMemoryStore(embedder=get_text_embedder())
    except Exception as exc:
        _LOGGER.debug("default memory with embedder unavailable: %s", exc)
        try:
            return EpisodicMemoryStore(embedder=None)
        except Exception as exc2:
            _LOGGER.warning("EpisodicMemoryStore unavailable: %s", exc2)
            return None


__all__ = [
    "ActorAgent",
    "ActorDecision",
    "ActorState",
    "run_actor_episode",
]
