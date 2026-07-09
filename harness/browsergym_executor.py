"""BrowserGym hop executor — deterministic-stub stage-1 cut.

Reference: ``harness/gymv_executor.py`` (the canonical shape this
module mirrors); rollout memo §6.1 / §11.5.5. Stage 4's first cut
does **not** drive a real browser via Playwright — each hop returns
a deterministic ``observation`` + ``EvidenceRef`` so the few-shot
adapter and per-hop predicate evaluator can run end-to-end against
cold-start demonstrations without a live browser.

Real BrowserGym binding (Playwright `_step_dom`, action parsing via
`browsergym.core.action.highlevel.HighLevelActionParser`) plugs in
later by replacing the closure `make_browsergym_executor` returns.
The closure shape matches `make_gymv_executor`:

    executor, holder = make_browsergym_executor(
        domain="browser", task="assistantbench",
        on_unresolved="skip", schema_producer=...,
    )
    adapter = BrowserAdapter()
    adapter.set_executor(executor)

Action verb taxonomy (matches BrowserGym's
``highlevel_action_parser`` action set): DOM-mutating (``CLICK`` /
``FILL`` / ``SELECT_OPTION`` / ``CHECK`` / ``UNCHECK`` / ``HOVER`` /
``SCROLL`` / ``KEY_PRESS``), navigation / no-op (``GO_BACK`` /
``GO_FORWARD`` / ``GOTO`` / ``NEW_TAB`` / ``TAB_FOCUS`` /
``TAB_CLOSE`` / ``NOOP`` / ``DONE``), observational (``OBSERVE`` /
``INSPECT`` / ``LOCATE`` / ``RECALL_*``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from common.state_schema import EvidenceRef, StateSchema
from harness.adapters.browser_adapter import HopExecutor
from harness.gym_schema_producer import SchemaProducer
from harness.skill_adapter import AdapterRunContext

logger = logging.getLogger("harness.browsergym_executor")


# DOM-mutating verbs — would call into the live browser in a real
# executor; here we only echo the payload + emit an EvidenceRef.
DOM_MUTATING_OPS: frozenset[str] = frozenset({
    "CLICK", "FILL", "SELECT_OPTION", "CHECK", "UNCHECK",
    "HOVER", "SCROLL", "KEY_PRESS", "PRESS",
})

# Navigation / no-op verbs — top-of-page chrome, no DOM change.
NAV_OPS: frozenset[str] = frozenset({
    "GO_BACK", "GO_FORWARD", "GOTO", "NEW_TAB",
    "TAB_FOCUS", "TAB_CLOSE", "NOOP", "DONE",
})

# Observational verbs — never step the browser; emit a GATHER
# EvidenceRef so the harness's G0 invariant (non-empty evidence on
# success for non-ACTION skills) holds.
OBSERVATIONAL_OPS: frozenset[str] = frozenset({
    "OBSERVE", "INSPECT", "LOCATE", "READ", "TRACK", "GATHER",
    "RECALL", "RECALL_TASK", "RECALL_PLAN", "RECALL_OBSERVATION",
    "VERIFY", "EVALUATE", "COMPARE", "SIMULATE",
    "REASON", "INFER", "DEDUCE", "PLAN", "DECIDE",
})


def _classify_op(op: str) -> str:
    """Return ``"dom"`` / ``"nav"`` / ``"observational"`` / ``"unknown"``
    for ``op`` (case-insensitive). Exact membership wins; the
    startswith fallback catches lifted prose verbs (``READ_DOM``,
    ``CLICK_BUTTON``)."""
    upper = op.upper()
    if upper in DOM_MUTATING_OPS:
        return "dom"
    if upper in NAV_OPS:
        return "nav"
    if upper in OBSERVATIONAL_OPS:
        return "observational"
    for verb in DOM_MUTATING_OPS:
        if upper.startswith(verb):
            return "dom"
    for verb in NAV_OPS:
        if upper.startswith(verb):
            return "nav"
    for verb in OBSERVATIONAL_OPS:
        if upper.startswith(verb):
            return "observational"
    return "unknown"


@dataclass
class BrowserExecutorState:
    """Mutable state the executor closure carries across hops.

    Mirrors `GymvExecutorState` but tracks browser-shaped fields
    (URL, focused element bid) instead of cumulative reward.
    """

    last_url: str = ""
    last_focused_bid: Optional[str] = None
    last_observation: Any = None
    last_info: Dict[str, Any] = field(default_factory=dict)
    outer_step: int = 0
    last_post_state: Optional[StateSchema] = None


def make_browsergym_executor(
    *,
    domain: str = "browser",
    task: str = "",
    on_unresolved: str = "skip",
    schema_producer: Optional[SchemaProducer] = None,
    state_holder: Optional[BrowserExecutorState] = None,
) -> Tuple[HopExecutor, BrowserExecutorState]:
    """Build a deterministic-stub `HopExecutor` for BrowserGym.

    Args:
      domain: Domain tag projected onto each post-step `StateSchema`.
      task: Task-id prefix (``"assistantbench"`` / ``"miniwob"`` / …).
      on_unresolved: ``"skip"`` (default) returns a soft-skip
        observational hop with reason ``no_browser_action`` for
        unrecognised verbs; ``"abort"`` returns ``ok=False``.
      schema_producer: Optional Day-4 schema producer. The stub
        executor doesn't invoke it (no live obs); a future
        real-browser version will pass the post-step DOM through it.
      state_holder: Optional shared `BrowserExecutorState` so the
        caller can read out `last_url` / `last_focused_bid` between
        runs.

    Returns: ``(executor, state_holder)``.
    """
    if on_unresolved not in {"skip", "abort"}:
        raise ValueError(
            f"on_unresolved={on_unresolved!r} must be 'skip' or 'abort'"
        )

    holder = state_holder or BrowserExecutorState()
    _ = schema_producer  # accepted for shape symmetry; reserved.

    def _ev(op: str, role: str, *, locator: str, conf: float,
            payload: Optional[Dict[str, Any]] = None) -> EvidenceRef:
        return EvidenceRef(
            source=f"browser:{op.lower()}",
            locator=locator,
            role=role,
            confidence=conf,
            payload=payload or {},
        )

    def executor(
        action_type: str, payload: Dict[str, Any], ctx: AdapterRunContext
    ) -> Dict[str, Any]:
        op = (action_type or "NOOP").upper()
        kind = _classify_op(op)
        t0 = time.time()

        if kind in {"observational", "nav"}:
            holder.last_post_state = ctx.state
            ev_out = _ev(op, role="GATHER",
                         locator=f"step={ctx.state.inner_step}",
                         conf=0.85)
            ev_in: List[EvidenceRef] = list(ctx.state.evidence or [])
            return {
                "ok": True,
                "observation": {
                    "echo_action": op,
                    "echo_payload": dict(payload),
                    "no_env_step": True,
                },
                "evidence": [ev_out],
                "evidence_in": ev_in,
                "evidence_out": [ev_out],
                "post_state": ctx.state.to_json(),
            }

        if kind == "dom":
            holder.outer_step += 1
            elapsed_ms = (time.time() - t0) * 1000.0
            ev_out = _ev(
                op, role="GATHER",
                locator=f"step={holder.outer_step},action={op.lower()}",
                conf=0.85,
                payload={"task": task, "elapsed_ms": elapsed_ms},
            )
            ev_in = list(ctx.state.evidence or [])
            holder.last_post_state = ctx.state
            return {
                "ok": True,
                "observation": {
                    "echo_action": op,
                    "echo_payload": dict(payload),
                    "elapsed_ms": elapsed_ms,
                },
                "evidence": [ev_out],
                "evidence_in": ev_in,
                "evidence_out": [ev_out],
                "post_state": ctx.state.to_json(),
            }

        # Unknown verb — soft-skip or abort.
        reason = f"no_browser_action_for_op={op}_payload={dict(payload)}"
        if on_unresolved == "abort":
            return {"ok": False, "reason": reason, "evidence": []}
        ev_out = _ev(op, role="GATHER",
                     locator=f"step={ctx.state.inner_step},skip",
                     conf=0.5,
                     payload={"reason": "no_browser_action"})
        ev_in = list(ctx.state.evidence or [])
        holder.last_post_state = ctx.state
        return {
            "ok": True,
            "observation": {
                "echo_action": op,
                "echo_payload": dict(payload),
                "no_env_step": True,
                "skip_reason": reason,
            },
            "evidence": [ev_out],
            "evidence_in": ev_in,
            "evidence_out": [ev_out],
            "post_state": ctx.state.to_json(),
        }

    return executor, holder


__all__ = [
    "BrowserExecutorState",
    "DOM_MUTATING_OPS",
    "NAV_OPS",
    "OBSERVATIONAL_OPS",
    "make_browsergym_executor",
]
