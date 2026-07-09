"""Pure-function metrics for the grounding-pipeline eval harness.

Every function in this module takes plain Python data (gold + predicted
schema strings, optional bbox lists, the raw ``GroundingResult`` dict)
and returns a numeric score plus a small explanation dict — there is no
I/O, no state, no model calls.  The driver in ``harness.py`` composes
these into per-sample reports and an aggregated ``EvalMetrics``.

All metrics are 0–1 floats unless noted; "n_*" fields are absolute
counts so the harness can compute weighted averages over heterogeneous
benchmarks (VisualToolBench + TIR-Bench + Video-Holmes + …).
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..schema import (
    count_entities,
    parse_answer_from_schema,
    parse_evidence_refs,
    parse_schema_output,
    semantic_validate,
    validate_schema,
)

# Section presence is the cheapest signal — a schema that lacks the
# section can't possibly score above zero on the per-section field
# checks, so we short-circuit.
_REQUIRED_SECTIONS_FOR_FIELD_ACC: tuple[str, ...] = (
    "entities", "attributes", "relations", "state_flags", "targets",
)

_EID_RE = re.compile(r"\be(\d+)\b")
_ENTITY_LINE_RE = re.compile(r"^(e\d+)\s*\[(.*?)\]\s*$", re.MULTILINE)
_POS_RE = re.compile(r"pos\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
_LABEL_RE = re.compile(r"label\s*=\s*([^,\]]+)")
_TARGET_RE = re.compile(r"^target\s*=\s*(\S+)\s*$", re.MULTILINE)
_BLOCKER_RE = re.compile(r"^blocker\s*=\s*(\S+)\s*$", re.MULTILINE)
_HOP_TOOL_RE = re.compile(r"^hop\d+\.tool\s*=\s*([\w./-]+)\s*$", re.MULTILINE)


# ======================================================================
# Per-sample dataclasses
# ======================================================================

@dataclass
class PerSampleResult:
    """One row in the per-sample report jsonl produced by ``run_eval``.

    Each metric is a float 0–1 (or ``None`` when not applicable for
    this sample — e.g. ``answer_correct`` is ``None`` for env tasks).
    Lists / dicts carry the underlying detail the aggregate metrics are
    built from.
    """

    sample_id: str
    domain: str
    head_used: str | None = None
    n_escalations: int = 0
    format_ok: bool = False
    format_warnings: list[str] = field(default_factory=list)
    semantic_valid: bool = False
    semantic_errors: list[str] = field(default_factory=list)
    field_accuracy: dict[str, float] = field(default_factory=dict)
    answer_correct: bool | None = None
    target_correct: bool | None = None
    blocker_correct: bool | None = None
    entity_iou: float | None = None
    entity_iou_breakdown: dict[str, float] = field(default_factory=dict)
    tool_precision: float | None = None
    tool_recall: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "domain": self.domain,
            "head_used": self.head_used,
            "n_escalations": self.n_escalations,
            "format_ok": self.format_ok,
            "format_warnings": list(self.format_warnings),
            "semantic_valid": self.semantic_valid,
            "semantic_errors": list(self.semantic_errors),
            "field_accuracy": dict(self.field_accuracy),
            "answer_correct": self.answer_correct,
            "target_correct": self.target_correct,
            "blocker_correct": self.blocker_correct,
            "entity_iou": self.entity_iou,
            "entity_iou_breakdown": dict(self.entity_iou_breakdown),
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
            "error": self.error,
        }


@dataclass
class EvalMetrics:
    """Aggregated metrics across a stream of samples (PLAN-V-G-MILESTONES §5)."""

    n_samples: int = 0
    n_with_schema: int = 0
    format_compliance: float = 0.0
    semantic_valid_rate: float = 0.0
    field_accuracy: dict[str, float] = field(default_factory=dict)
    answer_accuracy: float | None = None
    target_accuracy: float | None = None
    blocker_accuracy: float | None = None
    entity_iou_mean: float | None = None
    tool_precision: float | None = None
    tool_recall: float | None = None
    path_a_rate: float = 0.0
    path_b_rate: float = 0.0
    path_c_rate: float = 0.0
    head_usage: dict[str, int] = field(default_factory=dict)
    escalation_rate: float = 0.0
    per_domain: dict[str, "EvalMetrics"] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_with_schema": self.n_with_schema,
            "format_compliance": round(self.format_compliance, 4),
            "semantic_valid_rate": round(self.semantic_valid_rate, 4),
            "field_accuracy": {
                k: round(v, 4) for k, v in self.field_accuracy.items()
            },
            "answer_accuracy": (
                round(self.answer_accuracy, 4)
                if self.answer_accuracy is not None else None
            ),
            "target_accuracy": (
                round(self.target_accuracy, 4)
                if self.target_accuracy is not None else None
            ),
            "blocker_accuracy": (
                round(self.blocker_accuracy, 4)
                if self.blocker_accuracy is not None else None
            ),
            "entity_iou_mean": (
                round(self.entity_iou_mean, 4)
                if self.entity_iou_mean is not None else None
            ),
            "tool_precision": (
                round(self.tool_precision, 4)
                if self.tool_precision is not None else None
            ),
            "tool_recall": (
                round(self.tool_recall, 4)
                if self.tool_recall is not None else None
            ),
            "path_a_rate": round(self.path_a_rate, 4),
            "path_b_rate": round(self.path_b_rate, 4),
            "path_c_rate": round(self.path_c_rate, 4),
            "head_usage": dict(self.head_usage),
            "escalation_rate": round(self.escalation_rate, 4),
            "per_domain": {
                k: v.to_dict() for k, v in self.per_domain.items()
            },
        }


# ======================================================================
# Format compliance + semantic validity
# ======================================================================

def compute_format_compliance(
    schema_text: str | None,
    *,
    required_sections: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Return ``(ok, warnings)`` from the structural ``validate_schema``.

    ``ok`` is True iff parsing succeeded *and* there are no missing-tag
    warnings.  Use the warnings list for diagnostics.
    """
    if not schema_text:
        return False, ["empty schema"]
    parsed = parse_schema_output(schema_text)
    if parsed is None:
        return False, ["could not parse <state>...</state> block"]
    warnings = validate_schema(parsed, required_sections=required_sections)
    return (len(warnings) == 0), warnings


# ======================================================================
# Field accuracy (per-section comparison vs gold schema)
# ======================================================================

def _split_sections_lite(schema_text: str) -> dict[str, str]:
    """Lightweight section splitter — duplicates ``schema._split_sections``
    locally so this module stays self-contained for tests."""
    out: dict[str, str] = {}
    pattern = re.compile(
        r"<(\w+)>(.*?)(?=<\w+>|</state>)", re.DOTALL,
    )
    for m in pattern.finditer(schema_text):
        out[m.group(1)] = m.group(2).strip()
    return out


def _section_lines(body: str) -> set[str]:
    """Strip blank/comment lines + normalise whitespace before comparing."""
    lines: set[str] = set()
    for ln in body.splitlines():
        s = ln.strip()
        if not s or (s.startswith("(") and s.endswith(")")):
            continue
        # Collapse internal whitespace so ``e1.state= visible`` matches
        # ``e1.state=visible``.
        lines.add(re.sub(r"\s+", "", s))
    return lines


def compute_field_accuracy(
    pred_schema: str | None,
    gold_schema: str | None,
    *,
    sections: Iterable[str] = _REQUIRED_SECTIONS_FOR_FIELD_ACC,
) -> dict[str, float]:
    """Per-section line-level F1 between predicted and gold schemas.

    A "field" here is one normalised line of a section body.  We use
    micro-F1 because it tolerates the VLM listing entities in a
    different order while still penalising missed / fabricated rows.
    The result is a dict keyed by section name; missing sections score
    0 unless the gold also lacks them (then 1.0 — vacuously correct).
    """
    if not pred_schema or not gold_schema:
        return {s: 0.0 for s in sections}

    pred_secs = _split_sections_lite(pred_schema)
    gold_secs = _split_sections_lite(gold_schema)

    out: dict[str, float] = {}
    for sec in sections:
        gold_lines = _section_lines(gold_secs.get(sec, ""))
        pred_lines = _section_lines(pred_secs.get(sec, ""))
        if not gold_lines and not pred_lines:
            out[sec] = 1.0
            continue
        if not gold_lines or not pred_lines:
            out[sec] = 0.0
            continue
        tp = len(pred_lines & gold_lines)
        precision = tp / len(pred_lines)
        recall = tp / len(gold_lines)
        if precision + recall == 0:
            out[sec] = 0.0
        else:
            out[sec] = 2 * precision * recall / (precision + recall)
    return out


# ======================================================================
# Answer / target / blocker accuracy
# ======================================================================

def _norm_answer(s: str | None) -> str | None:
    if s is None:
        return None
    return s.strip().strip(".").lower()


def compute_answer_accuracy(
    pred: str | None, gold: str | None,
) -> bool | None:
    """Case-insensitive exact-match for benchmark QA answers.

    Returns ``None`` when either side is missing — the harness treats
    ``None`` as "not applicable" and excludes it from the aggregate.
    """
    np_, ng = _norm_answer(pred), _norm_answer(gold)
    if np_ is None or ng is None:
        return None
    return np_ == ng


def _extract_target(schema_text: str | None) -> str | None:
    if not schema_text:
        return None
    m = _TARGET_RE.search(schema_text)
    if not m:
        return None
    val = m.group(1).strip()
    return val if val.lower() != "null" else None


def _extract_blocker(schema_text: str | None) -> str | None:
    if not schema_text:
        return None
    m = _BLOCKER_RE.search(schema_text)
    if not m:
        return None
    val = m.group(1).strip()
    return val if val.lower() != "null" else None


def compute_target_accuracy(
    pred_schema: str | None, gold_schema: str | None,
) -> tuple[bool | None, bool | None]:
    """Return ``(target_correct, blocker_correct)``.

    Both are ``None`` when the gold schema doesn't fix that slot — for
    QA benchmarks the gold ``target=`` is often null.
    """
    g_target = _extract_target(gold_schema)
    g_blocker = _extract_blocker(gold_schema)

    p_target = _extract_target(pred_schema)
    p_blocker = _extract_blocker(pred_schema)

    target_ok = (
        (p_target == g_target)
        if g_target is not None and p_target is not None
        else None
    )
    blocker_ok = (
        (p_blocker == g_blocker)
        if g_blocker is not None and p_blocker is not None
        else None
    )
    return target_ok, blocker_ok


# ======================================================================
# Entity bbox IoU
# ======================================================================

def _parse_entities_with_bbox(
    schema_text: str,
) -> list[tuple[str, str, tuple[int, int, int, int] | None]]:
    """Return list of ``(eid, label, (x, y, w, h) | None)`` for every entity."""
    out: list[tuple[str, str, tuple[int, int, int, int] | None]] = []
    if not schema_text:
        return out
    for m in _ENTITY_LINE_RE.finditer(schema_text):
        eid = m.group(1)
        body = m.group(2)
        label_m = _LABEL_RE.search(body)
        label = label_m.group(1).strip() if label_m else ""
        pos_m = _POS_RE.search(body)
        pos = (
            (int(pos_m.group(1)), int(pos_m.group(2)),
             int(pos_m.group(3)), int(pos_m.group(4)))
            if pos_m else None
        )
        out.append((eid, label, pos))
    return out


def _bbox_iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int],
) -> float:
    """Standard Pascal-VOC IoU on ``(x, y, w, h)`` axis-aligned boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_entity_iou(
    pred_schema: str | None,
    gold_entities: list[dict[str, Any]] | None,
    *,
    iou_threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Greedy matched mean-IoU of predicted entity boxes vs gold boxes.

    ``gold_entities`` is a list of dicts with keys ``label`` and
    ``bbox`` (``[x, y, w, h]``) — same shape produced by
    ``_extract_entity_oracle`` in ``vlm_wrapper.labeling`` (built from
    OmniParser / GroundingDINO / scene graphs).

    Returns ``(mean_iou_over_matched, breakdown)`` where breakdown has
    keys ``precision`` / ``recall`` / ``mean_iou`` / ``n_matched``.
    """
    pred_entities = [
        (eid, label, pos)
        for eid, label, pos in _parse_entities_with_bbox(pred_schema or "")
        if pos is not None
    ]
    gold = [
        (g.get("label", ""), tuple(g["bbox"]))
        for g in (gold_entities or [])
        if g.get("bbox") and len(g["bbox"]) == 4
    ]
    if not pred_entities or not gold:
        return 0.0, {
            "precision": 0.0, "recall": 0.0,
            "mean_iou": 0.0, "n_matched": 0,
        }

    used_gold: set[int] = set()
    matched_ious: list[float] = []
    for _, _, pred_box in pred_entities:
        best_iou = 0.0
        best_j = -1
        for j, (_, gold_box) in enumerate(gold):
            if j in used_gold:
                continue
            iou = _bbox_iou(pred_box, gold_box)  # type: ignore[arg-type]
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            used_gold.add(best_j)
            matched_ious.append(best_iou)

    n_pred = len(pred_entities)
    n_gold = len(gold)
    n_matched = len(matched_ious)
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_gold if n_gold else 0.0
    mean_iou = (sum(matched_ious) / n_matched) if n_matched else 0.0
    return mean_iou, {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "n_matched": float(n_matched),
    }


# ======================================================================
# Tool-call precision / recall vs an oracle of "needed" tools
# ======================================================================

def compute_tool_precision(
    tool_trace: list[dict[str, Any]] | None,
    oracle_needed_tools: Iterable[str] | None,
) -> tuple[float | None, float | None]:
    """Return ``(precision, recall)`` of called tools vs the oracle.

    The oracle is the set of tools the heuristic / scene-graph adapter
    decided would have been needed to derive the gold schema (e.g. for
    an image-QA item with 3 spheres the oracle includes ``count_objects``).
    Returns ``(None, None)`` when no oracle is provided.
    """
    if oracle_needed_tools is None:
        return None, None
    oracle_set = {t.strip() for t in oracle_needed_tools if t}
    if not oracle_set:
        return None, None

    called: set[str] = set()
    for entry in tool_trace or []:
        name = (entry.get("tool") or entry.get("name") or "").strip()
        if name:
            called.add(name)

    if not called:
        return 0.0, 0.0

    tp = len(called & oracle_set)
    precision = tp / len(called)
    recall = tp / len(oracle_set)
    return precision, recall


# ======================================================================
# Cascade telemetry — Path A / B / C breakdown
# ======================================================================

# A run's ``head_used`` strings come from ``ground.cascaded_ground``.
# Anything in PATH_A_HEADS is "fast path"; PATH_B_HEADS is tool repair;
# PATH_C_HEADS is offline / teacher fallback.
PATH_A_HEADS: tuple[str, ...] = ("heuristic", "vlm", "omniparser")
PATH_B_HEADS: tuple[str, ...] = ("tool_loop", "tool_repair")
PATH_C_HEADS: tuple[str, ...] = ("teacher", "escalation_failure")


def _classify_path(
    head_used: str | None, escalation_trace: list[dict[str, Any]] | None,
) -> str:
    """Bucket a run as Path A / B / C (PLAN-V-G-MILESTONES §3)."""
    if head_used in PATH_C_HEADS:
        return "C"
    if head_used in PATH_B_HEADS:
        return "B"
    if head_used in PATH_A_HEADS:
        # If we got to a Path-A head only after escalating from another
        # Path-A head, count it as "B-lite" — the cascade did fire.
        if escalation_trace and len(escalation_trace) > 1:
            return "B"
        return "A"
    return "A"  # unknown head → assume direct parse


def compute_path_breakdown(
    head_used: str | None,
    escalation_trace: list[dict[str, Any]] | None,
) -> str:
    """Public wrapper around ``_classify_path`` for the harness."""
    return _classify_path(head_used, escalation_trace)


# ======================================================================
# Aggregation
# ======================================================================

def _mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarise_metrics(
    rows: list[PerSampleResult],
    *,
    by_domain: bool = True,
) -> EvalMetrics:
    """Aggregate a list of per-sample rows into ``EvalMetrics``.

    Computes overall metrics and (optionally) per-domain breakdowns.
    Skips ``None`` values so a benchmark that doesn't define `target`
    doesn't drag the mean target accuracy to 0.
    """
    n = len(rows)
    if n == 0:
        return EvalMetrics()

    n_with_schema = sum(1 for r in rows if r.format_ok or r.field_accuracy)
    fmt_ok_n = sum(1 for r in rows if r.format_ok)
    sem_ok_n = sum(1 for r in rows if r.semantic_valid)

    # Per-section field accuracy: average over samples that emitted that
    # section in either gold or pred (i.e. recorded a non-trivial score).
    section_scores: dict[str, list[float]] = {}
    for r in rows:
        for sec, score in r.field_accuracy.items():
            section_scores.setdefault(sec, []).append(score)
    field_acc = {
        sec: statistics.mean(scores)
        for sec, scores in section_scores.items()
    }

    answer_vals = [
        float(r.answer_correct) for r in rows if r.answer_correct is not None
    ]
    target_vals = [
        float(r.target_correct) for r in rows if r.target_correct is not None
    ]
    blocker_vals = [
        float(r.blocker_correct) for r in rows if r.blocker_correct is not None
    ]
    iou_vals = [r.entity_iou for r in rows if r.entity_iou is not None]
    tp_vals = [r.tool_precision for r in rows if r.tool_precision is not None]
    tr_vals = [r.tool_recall for r in rows if r.tool_recall is not None]

    paths = [
        _classify_path(r.head_used, [{"i": i} for i in range(r.n_escalations)])
        for r in rows
    ]
    head_counts: dict[str, int] = {}
    for r in rows:
        head = r.head_used or "unknown"
        head_counts[head] = head_counts.get(head, 0) + 1

    metrics = EvalMetrics(
        n_samples=n,
        n_with_schema=n_with_schema,
        format_compliance=fmt_ok_n / n,
        semantic_valid_rate=sem_ok_n / n,
        field_accuracy=field_acc,
        answer_accuracy=_mean_or_none(answer_vals),
        target_accuracy=_mean_or_none(target_vals),
        blocker_accuracy=_mean_or_none(blocker_vals),
        entity_iou_mean=_mean_or_none(iou_vals),
        tool_precision=_mean_or_none(tp_vals),
        tool_recall=_mean_or_none(tr_vals),
        path_a_rate=paths.count("A") / n,
        path_b_rate=paths.count("B") / n,
        path_c_rate=paths.count("C") / n,
        head_usage=head_counts,
        escalation_rate=(
            sum(r.n_escalations for r in rows) / n
        ),
    )

    if by_domain:
        by: dict[str, list[PerSampleResult]] = {}
        for r in rows:
            by.setdefault(r.domain or "unknown", []).append(r)
        metrics.per_domain = {
            d: summarise_metrics(samples, by_domain=False)
            for d, samples in by.items()
        }

    return metrics


__all__ = [
    "PerSampleResult",
    "EvalMetrics",
    "compute_format_compliance",
    "compute_field_accuracy",
    "compute_answer_accuracy",
    "compute_target_accuracy",
    "compute_entity_iou",
    "compute_tool_precision",
    "compute_path_breakdown",
    "summarise_metrics",
]
