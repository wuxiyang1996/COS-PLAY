"""Reasoning / derivation tools for tool-loop grounding.

These tools are *symbolic* — they do not look at pixels.  They take
values that the model has already grounded via observation tools
(`detect_objects`, `read_text_region`, `describe_frame`, …) and turn
them into typed derivation steps that can be cited inside the
``<derivations>`` block of the structured ``<state>`` schema.

Why a separate module?  The image / video tool registries are full of
*observation* tools (locate, describe, OCR, compare frames).  When we
asked the VLM to answer a "what proportion of foo is bar" question on
TIR-Bench it called `detect_objects`, got back perfect bboxes, then
hallucinated the ratio — the perception was fine, the missing piece
was an explicit *computation* step.  These tools make the computation
step a first-class tool call that lands in the ``tool_trace``,
producing a stable derivation id (``d1``, ``d2`` …) that downstream
skills can cite.

Tools provided
--------------
* ``count_value`` — record a count produced from observation tool calls.
* ``compute_ratio`` — numerator / denominator with units.
* ``compare_values`` — two-argument comparison with a typed operator.
* ``verify_claim`` — commit to a claim and bind it to evidence ids.

Each call appends a row to the per-registry ``_DerivationLog`` and the
returned dict carries the row's ``derivation_id`` so the model can
reference it inside ``<derivations>`` and ``<answer>``.

The log itself is exposed via ``get_derivations()`` on the registry's
state, so the orchestrator can attach the rendered ``<derivations>``
block to the final schema if the model omits it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from vlm_wrapper.tools import ToolDef, ToolRegistry

logger = logging.getLogger(__name__)


# ── Derivation kinds (closed enum, mirrors INNER_MDP_OPS) ────────────

DERIVATION_KINDS: tuple[str, ...] = (
    "COUNT", "RATIO", "COMPARE", "VERIFY",
)


@dataclass
class _DerivationRow:
    """One typed derivation step with stable id."""

    derivation_id: str
    kind: str
    inputs: dict[str, Any]
    output: Any
    label: str | None = None
    refs: list[str] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "derivation_id": self.derivation_id,
            "kind": self.kind,
            "inputs": self.inputs,
            "output": self.output,
            "label": self.label,
            "refs": list(self.refs),
            "note": self.note,
        }


class _DerivationLog:
    """Per-registry log of typed derivation steps.

    The orchestrator can fetch the rendered ``<derivations>`` block via
    :meth:`render_section` and attach it to the schema if the model
    forgot the block (the validator will warn but the answer can still
    be salvaged).
    """

    def __init__(self) -> None:
        self._rows: list[_DerivationRow] = []

    def append(self, kind: str, **fields: Any) -> _DerivationRow:
        if kind not in DERIVATION_KINDS:
            logger.warning(
                "derivation kind %r is not in canonical enum %s",
                kind, DERIVATION_KINDS,
            )
        derivation_id = f"d{len(self._rows) + 1}"
        row = _DerivationRow(derivation_id=derivation_id, kind=kind, **fields)
        self._rows.append(row)
        return row

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def to_list(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._rows]

    def render_section(self) -> str:
        """Render the log as a ``<derivations>`` schema block.

        Format (one row per derivation, no surrounding tag — caller
        wraps as ``<derivations>…</derivations>``)::

            d1.kind=COUNT    d1.label=apples         d1.inputs={refs:[e1,e2,e3]} d1.output=3
            d2.kind=RATIO    d2.label=apples_total   d2.inputs={num:3,den:10}    d2.output=0.30
        """
        lines: list[str] = []
        for r in self._rows:
            inputs_str = _short_repr(r.inputs)
            output_str = _short_repr(r.output)
            parts = [
                f"{r.derivation_id}.kind={r.kind}",
                f"{r.derivation_id}.inputs={inputs_str}",
                f"{r.derivation_id}.output={output_str}",
            ]
            if r.label:
                parts.append(f"{r.derivation_id}.label={r.label}")
            if r.refs:
                parts.append(
                    f"{r.derivation_id}.refs=[{','.join(r.refs)}]"
                )
            if r.note:
                parts.append(f"{r.derivation_id}.note={r.note}")
            lines.append("  ".join(parts))
        return "\n".join(lines)


def _short_repr(value: Any, max_len: int = 80) -> str:
    """Compact one-line repr for derivation rendering."""
    if isinstance(value, str):
        s = value.strip()
    elif isinstance(value, dict):
        items = []
        for k, v in value.items():
            items.append(f"{k}:{_short_repr(v, max_len=40)}")
        s = "{" + ",".join(items) + "}"
    elif isinstance(value, (list, tuple)):
        s = "[" + ",".join(_short_repr(v, max_len=20) for v in value) + "]"
    elif isinstance(value, float):
        s = f"{value:.4g}"
    else:
        s = str(value)
    s = s.replace("\n", " ")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def _coerce_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items: Iterable[str] = (
            p.strip() for p in value.replace(";", ",").split(",")
        )
    elif isinstance(value, (list, tuple)):
        items = (str(v).strip() for v in value)
    else:
        items = (str(value).strip(),)
    return [s for s in items if s]


# ── Tool definitions ──────────────────────────────────────────────────

TOOL_COUNT_VALUE = ToolDef(
    name="count_value",
    description=(
        "Record a count you have already grounded with an observation "
        "tool (e.g. detect_objects / grounded_detect / "
        "detect_objects_at_frame).  Use this when the question is "
        '"how many", "count of …" — pass the integer count and the '
        "entity / hop ids that justify it.  Returns a stable "
        "derivation_id (d1, d2 …) you can cite inside <derivations> "
        "and <answer>."
    ),
    parameters={
        "type": "object",
        "properties": {
            "value": {
                "type": "integer",
                "description": "The count itself.",
            },
            "label": {
                "type": "string",
                "description": (
                    "What is being counted, in 1-3 words "
                    '(e.g. "rebars", "people in frame", '
                    '"red blocks").'
                ),
            },
            "refs": {
                "type": "string",
                "description": (
                    "Comma-separated entity / hop ids that ground "
                    "this count, e.g. 'e1,e2,e3' or 'hop2,hop3'."
                ),
            },
            "note": {
                "type": "string",
                "description": "Optional short justification.",
            },
        },
        "required": ["value", "label"],
    },
    domain="reasoning",
)

TOOL_COMPUTE_RATIO = ToolDef(
    name="compute_ratio",
    description=(
        "Compute and record numerator / denominator (a ratio, "
        "proportion, percentage, fraction-of-area, etc.).  Use this "
        'whenever the question asks for "what proportion", "what '
        'percentage", "fraction of …".  Pass concrete numeric values '
        "you have already obtained from observation tools.  Returns a "
        "stable derivation_id and the rounded ratio in [0,1] plus a "
        "percentage string."
    ),
    parameters={
        "type": "object",
        "properties": {
            "numerator": {
                "type": "number",
                "description": "Numerator (e.g. count of matching items, area of subset).",
            },
            "denominator": {
                "type": "number",
                "description": "Denominator (e.g. total count, total area).",
            },
            "label": {
                "type": "string",
                "description": (
                    "What this ratio represents in 1-5 words, "
                    'e.g. "rebars / total" or "shaded_area / image".'
                ),
            },
            "refs": {
                "type": "string",
                "description": (
                    "Comma-separated entity / hop / derivation ids that "
                    "ground numerator or denominator, e.g. "
                    "'d1,d2' or 'e1,e2,e3'."
                ),
            },
            "unit": {
                "type": "string",
                "enum": ["fraction", "percent", "ratio"],
                "description": (
                    "Output formatting.  'fraction' returns 0-1, "
                    "'percent' returns 0-100, 'ratio' returns 'a:b'."
                ),
            },
        },
        "required": ["numerator", "denominator", "label"],
    },
    domain="reasoning",
)

TOOL_COMPARE_VALUES = ToolDef(
    name="compare_values",
    description=(
        "Compare two grounded values with a typed operator and record "
        "the result.  Use this for questions like 'is the red car "
        'larger than the blue car", "which is closer to the building", '
        "or any comparative reasoning.  Returns a derivation_id and "
        "the boolean / categorical outcome."
    ),
    parameters={
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "First value (numeric).",
            },
            "b": {
                "type": "number",
                "description": "Second value (numeric).",
            },
            "op": {
                "type": "string",
                "enum": ["<", "<=", "==", "!=", ">=", ">"],
                "description": "Comparison operator.",
            },
            "label_a": {
                "type": "string",
                "description": "Short name for value a (e.g. 'red_car_area').",
            },
            "label_b": {
                "type": "string",
                "description": "Short name for value b (e.g. 'blue_car_area').",
            },
            "refs": {
                "type": "string",
                "description": "Comma-separated entity / hop / derivation ids.",
            },
        },
        "required": ["a", "b", "op", "label_a", "label_b"],
    },
    domain="reasoning",
)

TOOL_VERIFY_CLAIM = ToolDef(
    name="verify_claim",
    description=(
        "Commit to a candidate answer (or sub-claim) and bind it to "
        "evidence ids.  Call this as the FINAL reasoning step before "
        "writing <answer>: it records a CONCLUDE-type derivation that "
        "names the answer string, the supporting entities / hops / "
        "derivations, and a confidence label.  Returns a stable "
        "derivation_id you should cite inside <answer>."
    ),
    parameters={
        "type": "object",
        "properties": {
            "claim": {
                "type": "string",
                "description": (
                    "The candidate answer or claim, in the exact form "
                    "you intend to put in <answer> (e.g. 'C', '6', "
                    "'2945.24 mm²', 'yes')."
                ),
            },
            "evidence_refs": {
                "type": "string",
                "description": (
                    "Comma-separated entity / hop / derivation ids "
                    "supporting the claim, e.g. 'e1,e2,d1,d2'.  At "
                    "least ONE must be cited."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Self-reported confidence in the claim.",
            },
        },
        "required": ["claim", "evidence_refs"],
    },
    domain="reasoning",
)


# ── Handler implementations ──────────────────────────────────────────

def _h_count_value(
    log: _DerivationLog,
    *,
    value: int,
    label: str,
    refs: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    refs_list = _coerce_refs(refs)
    row = log.append(
        "COUNT",
        inputs={"refs": refs_list},
        output=int(value),
        label=label.strip()[:40],
        refs=refs_list,
        note=(note.strip()[:120] if note else None),
    )
    return {
        "derivation_id": row.derivation_id,
        "kind": row.kind,
        "label": row.label,
        "value": row.output,
        "refs": row.refs,
    }


def _h_compute_ratio(
    log: _DerivationLog,
    *,
    numerator: float,
    denominator: float,
    label: str,
    refs: str | None = None,
    unit: str = "fraction",
) -> dict[str, Any]:
    if denominator == 0:
        return {
            "error": (
                "denominator=0 — cannot compute ratio.  Re-ground the "
                "denominator with an observation tool before retrying."
            ),
        }
    refs_list = _coerce_refs(refs)
    fraction = float(numerator) / float(denominator)
    if unit == "percent":
        output: Any = round(fraction * 100.0, 2)
        formatted = f"{output:.2f}%"
    elif unit == "ratio":
        output = f"{numerator}:{denominator}"
        formatted = output
    else:
        output = round(fraction, 4)
        formatted = f"{output:.4f}"
    row = log.append(
        "RATIO",
        inputs={
            "num": float(numerator),
            "den": float(denominator),
            "unit": unit,
        },
        output=output,
        label=label.strip()[:60],
        refs=refs_list,
        note=None,
    )
    return {
        "derivation_id": row.derivation_id,
        "kind": row.kind,
        "label": row.label,
        "fraction": round(fraction, 6),
        "formatted": formatted,
        "unit": unit,
        "refs": row.refs,
    }


def _h_compare_values(
    log: _DerivationLog,
    *,
    a: float,
    b: float,
    op: str,
    label_a: str,
    label_b: str,
    refs: str | None = None,
) -> dict[str, Any]:
    a_f = float(a)
    b_f = float(b)
    ops = {
        "<": a_f < b_f,
        "<=": a_f <= b_f,
        "==": a_f == b_f,
        "!=": a_f != b_f,
        ">=": a_f >= b_f,
        ">": a_f > b_f,
    }
    if op not in ops:
        return {
            "error": (
                f"unsupported op={op!r}; choose one of "
                f"{sorted(ops.keys())}"
            ),
        }
    result = bool(ops[op])
    refs_list = _coerce_refs(refs)
    if op in ("<", "<=") and a_f < b_f:
        winner = label_b
    elif op in (">", ">=") and a_f > b_f:
        winner = label_a
    elif a_f > b_f:
        winner = label_a
    elif b_f > a_f:
        winner = label_b
    else:
        winner = None
    row = log.append(
        "COMPARE",
        inputs={
            "a": a_f, "b": b_f, "op": op,
            "label_a": label_a.strip()[:30],
            "label_b": label_b.strip()[:30],
        },
        output=result,
        label=f"{label_a.strip()[:20]}_{op}_{label_b.strip()[:20]}",
        refs=refs_list,
        note=None,
    )
    return {
        "derivation_id": row.derivation_id,
        "kind": row.kind,
        "result": result,
        "larger": winner,
        "delta": round(a_f - b_f, 6),
        "refs": row.refs,
    }


def _h_verify_claim(
    log: _DerivationLog,
    *,
    claim: str,
    evidence_refs: str,
    confidence: str = "medium",
) -> dict[str, Any]:
    refs_list = _coerce_refs(evidence_refs)
    if not refs_list:
        return {
            "error": (
                "verify_claim requires evidence_refs to cite at least "
                "one entity / hop / derivation id."
            ),
        }
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"
    row = log.append(
        "VERIFY",
        inputs={
            "claim": claim.strip()[:120],
            "confidence": confidence,
        },
        output=claim.strip()[:120],
        label="answer_candidate",
        refs=refs_list,
        note=None,
    )
    return {
        "derivation_id": row.derivation_id,
        "kind": row.kind,
        "claim": row.output,
        "confidence": confidence,
        "refs": row.refs,
        "instruction": (
            "Cite this derivation_id inside <answer> "
            "evidence_chain=[…] and <derivations>."
        ),
    }


# ── Public: build registry ───────────────────────────────────────────

def build_reasoning_registry(
    log: _DerivationLog | None = None,
) -> tuple[ToolRegistry, _DerivationLog]:
    """Create a ``ToolRegistry`` exposing the four reasoning tools.

    Parameters
    ----------
    log
        Optional pre-existing log to append to (so two registries can
        share derivation ids).  If omitted, a fresh log is created.

    Returns
    -------
    (registry, log)
        ``registry`` is the ``ToolRegistry`` to merge into the
        per-frame visual or video registry; ``log`` is the
        ``_DerivationLog`` the caller can render onto the schema.
    """
    log = log if log is not None else _DerivationLog()
    reg = ToolRegistry(domain="reasoning")
    reg.register(TOOL_COUNT_VALUE, lambda **kw: _h_count_value(log, **kw))
    reg.register(TOOL_COMPUTE_RATIO, lambda **kw: _h_compute_ratio(log, **kw))
    reg.register(TOOL_COMPARE_VALUES, lambda **kw: _h_compare_values(log, **kw))
    reg.register(TOOL_VERIFY_CLAIM, lambda **kw: _h_verify_claim(log, **kw))
    return reg, log


__all__ = [
    "DERIVATION_KINDS",
    "TOOL_COUNT_VALUE",
    "TOOL_COMPUTE_RATIO",
    "TOOL_COMPARE_VALUES",
    "TOOL_VERIFY_CLAIM",
    "_DerivationRow",
    "_DerivationLog",
    "build_reasoning_registry",
]
