"""OSWorld hop executor — deterministic-stub binding for the
``OsworldAdapter``.

Stage 3 of the Phase-5 cross-domain transfer rollout (memo §6.1,
§11.5.5). Mirror of :mod:`harness.gymv_executor`, but the first cut is
**deterministic**: we do not actually invoke ``pyautogui`` or any real
desktop tool. The job here is only to keep the dispatch chain running
so the per-step success_fn fires on producer-emitted facts.

Recognises the OSWorld action-verb taxonomy: primitive desktop ops
(``CLICK`` / ``DOUBLE_CLICK`` / ``TYPE`` / ``HOTKEY`` / …) plus the
protocol-lift observational verbs (``INSPECT`` / ``OBSERVE`` /
``RECALL_*`` / …). Each call returns ``ok: True`` plus an
``EvidenceRef`` so the G0 invariant (non-empty evidence on success
for non-ACTION skills) holds. Real OSWorld binding lands in a later
cut via ``OsworldAdapter.set_executor``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from common.state_schema import EvidenceRef, StateSchema
from harness.adapters._stub_base import HopExecutor
from harness.gym_schema_producer import SchemaProducer
from harness.skill_adapter import AdapterRunContext

logger = logging.getLogger("harness.osworld_executor")


# ---------------------------------------------------------------------------
# Verb taxonomies
# ---------------------------------------------------------------------------

# Primitive desktop actions emitted by OSWorld's actor / lifted
# protocols. The deterministic stub doesn't invoke ``pyautogui`` for
# any of these; it just echoes the action_type + payload back through
# an EvidenceRef so the harness has something to record.
PRIMITIVE_DESKTOP_OPS: frozenset[str] = frozenset({
    "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK",
    "TYPE", "KEY_PRESS", "HOTKEY",
    "SCROLL", "DRAG", "MOVE_MOUSE",
    "WAIT", "DONE", "FINISH",
})

# Observational hops — never touch the desktop, only emit evidence.
# Mirrors :data:`harness.gymv_executor.OBSERVATIONAL_OPS` in spirit;
# OSWorld adds a few RECALL_* variants that the cross-domain protocol
# lift uses on AT-SPI snapshots.
OBSERVATIONAL_OPS: frozenset[str] = frozenset({
    "INSPECT", "OBSERVE", "LOCATE",
    "RECALL", "RECALL_TASK", "RECALL_GOAL", "RECALL_ENTITY",
    "READ", "TRACK", "COMPARE", "EVALUATE", "SIMULATE",
    "VERIFY", "KEEP", "STOP", "CONTINUE",
})


# ---------------------------------------------------------------------------
# Mutable state holder (parity with GymvExecutorState)
# ---------------------------------------------------------------------------


@dataclass
class OsworldExecutorState:
    """Mutable counters / last-seen objects that the executor closure
    carries between hops.

    Mirrors :class:`harness.gymv_executor.GymvExecutorState`. The
    deterministic stub doesn't actually step a desktop, but we still
    advance ``outer_step`` so producer calls receive a monotonically
    increasing step index and ``last_post_state`` can be inspected
    between hops for tests.
    """

    outer_step: int = 0
    last_payload: Dict[str, Any] = field(default_factory=dict)
    last_post_state: Optional[StateSchema] = None
    n_calls: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapt_schema_producer(
    producer: SchemaProducer,
    *,
    domain: str,
    task: str,
) -> Callable[[AdapterRunContext, str, Mapping[str, Any]], Optional[str]]:
    """Adapt the Day-4 ``SchemaProducer(info, obs, *, step, task,
    goal)`` signature into something the OSWorld executor can call
    once per hop. Returns a callable ``(ctx, action_type, info) ->
    Optional[str]``. The closure carries a mutable step counter so
    consecutive calls advance ``step``.
    """

    counter = {"step": 0}

    def _builder(
        ctx: AdapterRunContext,
        action_type: str,
        info: Mapping[str, Any],
    ) -> Optional[str]:
        try:
            block = producer(
                info, None, step=counter["step"], task=task, domain=domain,
            )
        except Exception as exc:                                    # noqa: BLE001
            logger.debug(
                "osworld schema_producer raised (%s); skipping refresh", exc,
            )
            return None
        counter["step"] += 1
        return block or None

    return _builder


def _refresh_state_from_producer(
    builder: Callable[[AdapterRunContext, str, Mapping[str, Any]], Optional[str]],
    ctx: AdapterRunContext,
    action_type: str,
    info: Mapping[str, Any],
    *,
    domain: str,
) -> None:
    """Re-run the producer over a synthetic info dict and overlay any
    new ``state.facts`` keys onto ``ctx.state``. The deterministic
    stub's info dict carries whatever the cold-start step had under
    ``metadata.candidate_actions`` / ``metadata.schema_canonical``;
    when neither is present this is a no-op.
    """

    block = builder(ctx, action_type, info)
    if not block or "<state>" not in block:
        return
    try:
        from labeling_supplement._harness_io_helpers import parse_schema_canonical
    except Exception as exc:                                        # noqa: BLE001
        logger.debug("parse_schema_canonical import failed: %s", exc)
        return
    try:
        refreshed = parse_schema_canonical(block, default_domain=domain)
    except Exception as exc:                                        # noqa: BLE001
        logger.debug("parse_schema_canonical raised: %s", exc)
        return
    # Merge refreshed facts into the current state. We don't replace
    # ``ctx.state.elements`` since the harness expects the state object
    # identity to remain stable across hops; instead we overlay the
    # facts so per-hop predicate evaluation has up-to-date numerics.
    ctx.state.facts.update(refreshed.facts or {})


# ---------------------------------------------------------------------------
# Executor factory
# ---------------------------------------------------------------------------


def make_osworld_executor(
    *,
    domain: str = "osworld",
    task: str = "",
    on_unresolved: str = "skip",
    schema_producer: Optional[SchemaProducer] = None,
    state_holder: Optional[OsworldExecutorState] = None,
) -> Tuple[HopExecutor, Dict[str, Any]]:
    """Return ``(executor, holder)`` for ``OsworldAdapter.set_executor``.

    ``domain`` tags emitted EvidenceRefs; ``task`` is the friendly
    OSWorld domain name (``"vlc"``, ``"gimp"``) and is forwarded to
    the schema producer. ``on_unresolved`` is reserved for parity
    with :func:`harness.gymv_executor.make_gymv_executor` (the
    deterministic stub never fails to resolve an action; both
    ``"skip"`` and ``"abort"`` behave identically until a real
    desktop driver lands). ``schema_producer`` and ``state_holder``
    are optional; the holder dict lets the dispatcher treat the
    return value as opaque ``Dict[str, Any]``.
    """

    if on_unresolved not in {"skip", "abort"}:
        raise ValueError(
            f"on_unresolved={on_unresolved!r} must be 'skip' or 'abort'"
        )

    holder_state = state_holder or OsworldExecutorState()
    holder: Dict[str, Any] = {
        "state": holder_state,
        "domain": domain,
        "task": task,
        "on_unresolved": on_unresolved,
    }

    builder = (
        _adapt_schema_producer(schema_producer, domain=domain, task=task)
        if schema_producer is not None else None
    )

    def executor(
        action_type: str,
        payload: Dict[str, Any],
        ctx: AdapterRunContext,
    ) -> Dict[str, Any]:
        op = (action_type or "STEP").upper()
        holder_state.n_calls += 1
        holder_state.last_payload = dict(payload or {})

        # ---- Observational verb: no desktop step, just an EvidenceRef.
        if op in OBSERVATIONAL_OPS:
            holder_state.last_post_state = ctx.state
            if builder is not None:
                _refresh_state_from_producer(
                    builder, ctx, op, payload or {}, domain=domain,
                )
            return {
                "ok": True,
                "observation": {
                    "echo_action": op,
                    "echo_payload": dict(payload or {}),
                    "no_env_step": True,
                },
                "evidence": [
                    EvidenceRef(
                        source=f"osworld:{op.lower()}",
                        locator=f"step={ctx.state.inner_step}",
                        role="GATHER",
                        confidence=0.85,
                    )
                ],
                "post_state": ctx.state.to_json(),
            }

        # ---- Primitive desktop verb: deterministic stub.
        if op in PRIMITIVE_DESKTOP_OPS:
            holder_state.outer_step += 1
            if builder is not None:
                _refresh_state_from_producer(
                    builder, ctx, op, payload or {}, domain=domain,
                )
            holder_state.last_post_state = ctx.state
            return {
                "ok": True,
                "observation": {
                    "echo_action": op,
                    "echo_payload": dict(payload or {}),
                    "stub": True,
                },
                "evidence": [
                    EvidenceRef(
                        source=f"osworld:{op.lower()}",
                        locator=f"step={ctx.state.inner_step}",
                        role="COMMIT" if op in {"DONE", "FINISH"} else "GATHER",
                        confidence=0.85,
                    )
                ],
                "post_state": ctx.state.to_json(),
            }

        # ---- Catch-all: unknown action_type. Don't fail — emit
        # GATHER evidence and a structured note so the success_fn can
        # still score the hops that DID resolve. This matches the gymv
        # ``on_unresolved="skip"`` behaviour.
        holder_state.last_post_state = ctx.state
        return {
            "ok": True,
            "observation": {
                "echo_action": op,
                "echo_payload": dict(payload or {}),
                "no_env_step": True,
                "note": f"unknown_action_type={op!r}",
            },
            "evidence": [
                EvidenceRef(
                    source=f"osworld:{op.lower()}",
                    locator=f"step={ctx.state.inner_step},unknown",
                    role="GATHER",
                    confidence=0.5,
                    payload={"reason": "unknown_action_type"},
                )
            ],
            "post_state": ctx.state.to_json(),
        }

    return executor, holder


__all__ = [
    "OBSERVATIONAL_OPS",
    "PRIMITIVE_DESKTOP_OPS",
    "OsworldExecutorState",
    "make_osworld_executor",
]
