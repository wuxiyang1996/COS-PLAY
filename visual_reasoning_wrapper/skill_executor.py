"""Skill executor that binds the visual_reasoning tool registries to the
harness ``VisualReasoningAdapter``.

The harness adapter
(:mod:`harness.adapters.visual_reasoning_adapter`) walks
``SkillRecord.protocol`` hop by hop and, for each hop, calls a pluggable
``HopExecutor`` callable with signature
``(action_type, payload, ctx) -> dict``.  The default stub is
deterministic; this module provides a *real* executor backed by
:mod:`visual_reasoning_wrapper.tools_visual` (perception) and
:mod:`visual_reasoning_wrapper.tools_reasoning` (typed symbolic
reasoning).

Inner-MDP action → tool mapping
-------------------------------
The protocol's hop ``action`` field is a value from
``common.enums.InnerAction``.  We map it to one of our concrete tools and
emit an :class:`EvidenceRef` whose ``role`` matches the canonical
``EVIDENCE_ROLES = (GATHER, VERIFY, REASON, COMMIT)`` taxonomy::

    GROUND   →  grounded_detect / detect_objects     →  GATHER
    RETRIEVE →  describe_region / read_text_region   →  GATHER
    CHECK    →  count_value / compute_ratio /
                compare_values / spatial_query        →  REASON
    VERIFY   →  verify_claim                          →  VERIFY
    COMMIT   →  verify_claim (final)                  →  COMMIT
    EXECUTE  →  no-op (image QA has no env effects)   →  COMMIT

The ``CHECK`` mapping is keyed off ``payload["kind"]``: protocols
authored against ``DERIVATION_KINDS`` (``COUNT`` / ``RATIO`` /
``COMPARE``) automatically pick the matching reasoning tool.  Protocols
that do not specify a ``kind`` fall through to ``spatial_query`` (a
read-only perception probe) and are still emitted with ``role=REASON``
because the *intent* of a ``CHECK`` hop is to derive, not to gather.

Slot resolution
---------------
The adapter has already substituted ``${slot}`` placeholders before the
payload reaches us (see :func:`harness.adapters._common.HopBindings`),
so the executor sees concrete strings/numbers.  Anything left as
``${...}`` is treated as an unbound slot and surfaces as an error in
the result so the harness aborts the hop rather than silently passing
garbage to a tool.

Usage
-----
::

    from PIL import Image
    from harness.adapters.visual_reasoning_adapter import VisualReasoningAdapter
    from visual_reasoning_wrapper.skill_executor import bind_executor

    adapter = VisualReasoningAdapter()
    img = Image.open("frame.png")
    executor = bind_executor(adapter, image=img)
    # adapter.run(skill, ctx) now uses the real visual / reasoning tools.

The returned ``executor`` exposes ``derivation_log`` (the
:class:`tools_reasoning._DerivationLog`) so callers can render the
``<derivations>`` block from the resulting trace, mirroring what
:func:`vlm_wrapper.ground.cascaded_ground` does for free-form runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

from common.state_schema import EvidenceRef
from harness.skill_adapter import AdapterRunContext

from .tools_reasoning import _DerivationLog
from .tools_visual import build_visual_registry
from vlm_wrapper.tools import ToolRegistry

logger = logging.getLogger(__name__)


# ── Action → tool mapping ──────────────────────────────────────────────

#: Default observation tool to call for ``GROUND`` hops without a
#: ``query`` payload.  ``grounded_detect`` is used when the payload has
#: ``query`` (and optionally ``confidence_threshold``).
_DEFAULT_GROUND_TOOL = "detect_objects"

#: Default observation tool to call for ``RETRIEVE`` hops without a
#: ``bbox``.  ``read_text_region`` is used when the payload has bbox
#: coordinates (``x``/``y``/``w``/``h``).
_DEFAULT_RETRIEVE_TOOL = "describe_region"

#: ``CHECK`` payload key that disambiguates which reasoning tool to use.
_CHECK_KIND_TO_TOOL: Dict[str, str] = {
    "COUNT": "count_value",
    "RATIO": "compute_ratio",
    "COMPARE": "compare_values",
}

#: Inner-MDP action → evidence role.  Mirrors PLAN-SKILL-BANK §0.3 Clause B.
_ACTION_TO_ROLE: Dict[str, str] = {
    "GROUND": "GATHER",
    "RETRIEVE": "GATHER",
    "CHECK": "REASON",
    "VERIFY": "VERIFY",
    "COMMIT": "COMMIT",
    "EXECUTE": "COMMIT",
}


# ── Helpers ────────────────────────────────────────────────────────────

def _coerce_image(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, (str, bytes)):
        return Image.open(img)  # type: ignore[arg-type]
    raise TypeError(f"unsupported image type: {type(img).__name__}")


def _scrub_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return only string / numeric / bool entries (drops scratch values)."""
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
    return out


def _has_unbound_slot(payload: Dict[str, Any]) -> Optional[str]:
    for k, v in payload.items():
        if isinstance(v, str) and "${" in v and "}" in v:
            return f"unbound slot in payload[{k!r}]: {v}"
    return None


# ── Executor ───────────────────────────────────────────────────────────

@dataclass
class VisualReasoningExecutor:
    """Concrete ``HopExecutor`` for the ``visual_reasoning`` adapter.

    Holds the per-image tool registry plus the typed derivation log and
    dispatches each hop to the right tool.
    """

    image: Image.Image
    registry: ToolRegistry
    derivation_log: _DerivationLog
    confidence: float = 0.8
    _last_grounded_entities: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_image(
        cls,
        image: Any,
        *,
        prefer_gdino: bool = True,
        confidence: float = 0.8,
    ) -> "VisualReasoningExecutor":
        pil_image = _coerce_image(image)
        registry = build_visual_registry(
            pil_image,
            prefer_gdino=prefer_gdino,
            include_reasoning=True,
        )
        log = getattr(registry, "derivation_log", None)
        if log is None:
            raise RuntimeError(
                "build_visual_registry must attach a derivation_log "
                "(include_reasoning=True). The reasoning sub-registry "
                "was not merged."
            )
        return cls(
            image=pil_image,
            registry=registry,
            derivation_log=log,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # HopExecutor protocol
    # ------------------------------------------------------------------

    def __call__(
        self,
        action_type: str,
        payload: Dict[str, Any],
        ctx: AdapterRunContext,
    ) -> Dict[str, Any]:
        unbound = _has_unbound_slot(payload)
        if unbound:
            return {
                "ok": False,
                "reason": unbound,
                "evidence": [],
            }

        action = action_type.upper()
        try:
            tool_name, tool_args = self._select_tool(action, payload, ctx)
        except _NoToolForAction as exc:
            return {
                "ok": False,
                "reason": str(exc),
                "evidence": [],
            }

        if tool_name is None:
            # EXECUTE / commit-only hop with no tool call.
            return {
                "ok": True,
                "observation": {
                    "echo_action": action,
                    "echo_payload": _scrub_payload(payload),
                    "note": "no-op for visual_reasoning (no env side effects)",
                },
                "evidence": [self._make_evidence(action, action.lower(), step=ctx.state.inner_step)],
            }

        result = self.registry.dispatch(tool_name, tool_args)
        if result.error is not None:
            return {
                "ok": False,
                "reason": f"tool {tool_name!r} failed: {result.error}",
                "evidence": [],
            }

        if tool_name in ("detect_objects", "grounded_detect"):
            elements = (result.result or {}).get("elements", []) if isinstance(result.result, dict) else []
            self._last_grounded_entities = list(elements)

        evidence = [
            self._make_evidence(
                action,
                tool_name,
                step=ctx.state.inner_step,
                payload_snippet=_scrub_payload(tool_args),
            )
        ]

        # COMMIT hops *also* emit a VERIFY ref alongside the COMMIT one
        # so the answer block can satisfy both
        # ``expected_evidence_roles=['VERIFY','COMMIT']`` contracts.
        if action == "COMMIT" and tool_name == "verify_claim":
            evidence.insert(
                0,
                self._make_evidence(
                    "VERIFY",
                    tool_name,
                    step=ctx.state.inner_step,
                    payload_snippet=_scrub_payload(tool_args),
                ),
            )

        return {
            "ok": True,
            "observation": {
                "tool": tool_name,
                "result": result.result,
            },
            "evidence": evidence,
        }

    # ------------------------------------------------------------------
    # Action → tool selection
    # ------------------------------------------------------------------

    def _select_tool(
        self,
        action: str,
        payload: Dict[str, Any],
        ctx: AdapterRunContext,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        if action == "GROUND":
            if "query" in payload:
                args = {
                    "query": str(payload["query"]),
                    "confidence_threshold": float(
                        payload.get("confidence_threshold", 0.20)
                    ),
                    "max_results": int(payload.get("max_results", 10)),
                }
                return "grounded_detect", args
            args = {
                "max_elements": int(payload.get("max_elements", 25)),
                "confidence_threshold": float(
                    payload.get("confidence_threshold", 0.20)
                ),
            }
            return _DEFAULT_GROUND_TOOL, args

        if action == "RETRIEVE":
            if all(k in payload for k in ("x", "y", "w", "h")):
                args = {
                    "x": int(payload["x"]),
                    "y": int(payload["y"]),
                    "w": int(payload["w"]),
                    "h": int(payload["h"]),
                }
                # ``use_ocr`` is a *protocol* flag we honour by switching
                # tools — the underlying tools do not take it as a kwarg.
                tool = (
                    "read_text_region"
                    if payload.get("use_ocr")
                    else _DEFAULT_RETRIEVE_TOOL
                )
                return tool, args
            if "entity_index" in payload:
                idx = int(payload["entity_index"])
                if 0 <= idx < len(self._last_grounded_entities):
                    bbox = self._last_grounded_entities[idx].get("bbox", {})
                    args = {
                        "x": int(bbox.get("x", 0)),
                        "y": int(bbox.get("y", 0)),
                        "w": int(bbox.get("w", 0)),
                        "h": int(bbox.get("h", 0)),
                    }
                    return _DEFAULT_RETRIEVE_TOOL, args
            raise _NoToolForAction(
                "RETRIEVE hop needs either bbox (x,y,w,h) or entity_index "
                "(into the last GROUND result)."
            )

        if action == "CHECK":
            kind = str(payload.get("kind", "")).upper()
            tool = _CHECK_KIND_TO_TOOL.get(kind)
            if tool is None:
                if "element_a" in payload and "element_b" in payload:
                    return "spatial_query", {
                        "element_a": str(payload["element_a"]),
                        "element_b": str(payload["element_b"]),
                    }
                raise _NoToolForAction(
                    "CHECK hop requires payload['kind'] in "
                    f"{sorted(_CHECK_KIND_TO_TOOL)} or "
                    "spatial_query args (element_a, element_b)."
                )
            args = {k: v for k, v in payload.items() if k != "kind"}
            return tool, args

        if action in ("VERIFY", "COMMIT"):
            claim = payload.get("claim") or payload.get("answer")
            if claim is None:
                raise _NoToolForAction(
                    f"{action} hop requires payload['claim'] or "
                    "payload['answer']."
                )
            evidence_refs = (
                payload.get("evidence_refs")
                or payload.get("evidence")
                or payload.get("refs")
            )
            if evidence_refs is None:
                raise _NoToolForAction(
                    f"{action} hop requires payload['evidence_refs'] "
                    "naming hops/entities/derivations to bind to."
                )
            if isinstance(evidence_refs, (list, tuple)):
                evidence_refs = ",".join(str(r) for r in evidence_refs)
            args = {
                "claim": str(claim),
                "evidence_refs": str(evidence_refs),
                "confidence": str(payload.get("confidence", "medium")),
            }
            return "verify_claim", args

        if action == "EXECUTE":
            # No env side effects in image QA — but still emit evidence.
            return None, {}

        raise _NoToolForAction(
            f"action {action!r} has no visual_reasoning tool mapping. "
            "Supported: GROUND, CHECK, RETRIEVE, VERIFY, COMMIT, EXECUTE."
        )

    # ------------------------------------------------------------------
    # Evidence construction
    # ------------------------------------------------------------------

    def _make_evidence(
        self,
        action: str,
        tool_name: str,
        *,
        step: int,
        payload_snippet: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRef:
        role = _ACTION_TO_ROLE.get(action, "GATHER")
        return EvidenceRef(
            source=f"visual_reasoning:{tool_name}",
            locator=f"step={step}",
            role=role,
            confidence=self.confidence,
            payload=payload_snippet,
        )


class _NoToolForAction(ValueError):
    """Raised when an action_type cannot be mapped to a tool."""


# ── Wiring helpers ─────────────────────────────────────────────────────

def bind_executor(
    adapter: Any,
    *,
    image: Any,
    prefer_gdino: bool = True,
    confidence: float = 0.8,
) -> VisualReasoningExecutor:
    """Build a :class:`VisualReasoningExecutor` and attach it to ``adapter``.

    ``adapter`` is expected to be the
    :class:`~harness.adapters.visual_reasoning_adapter.VisualReasoningAdapter`
    instance the harness will dispatch to.  The function returns the
    executor so callers can inspect ``executor.derivation_log`` after
    the run to render the ``<derivations>`` block.
    """
    if not hasattr(adapter, "set_executor"):
        raise TypeError(
            f"adapter {type(adapter).__name__} has no set_executor — "
            "is it a StubTransferTargetAdapter?"
        )
    executor = VisualReasoningExecutor.from_image(
        image,
        prefer_gdino=prefer_gdino,
        confidence=confidence,
    )
    adapter.set_executor(executor)
    return executor


def make_visual_reasoning_executor(
    image: Any,
    *,
    prefer_gdino: bool = True,
    confidence: float = 0.8,
) -> VisualReasoningExecutor:
    """Functional alias for :meth:`VisualReasoningExecutor.from_image`."""
    return VisualReasoningExecutor.from_image(
        image,
        prefer_gdino=prefer_gdino,
        confidence=confidence,
    )


__all__ = [
    "VisualReasoningExecutor",
    "bind_executor",
    "make_visual_reasoning_executor",
]
