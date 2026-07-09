"""evaluation/answer_evaluator.py — deterministic answer + evidence judge.

Spec: ``PLAN-EVAL-FIRST-TARGET.md`` §5 (Joint Success) and §6 (failure
taxonomy F1–F7).

Joint Success ≜ ``answer_correct AND evidence_supported``
(``PLAN-SYSTEM-NORTHSTAR.md`` §3 headline metric).

This module is *pure logic*: it takes the per-instance evaluation
inputs that the driver produces (or that a label file supplies) and
returns

  - ``compute_joint_success(instance) -> bool``,
  - the failure-class label (``F1..F7``) when joint success is false,
  - aggregated per-setting / overall numbers used by the scoreboard.

Two consumers:

  * :class:`evaluation.driver.EvalDriver` — calls this on every
    instance during an eval run.
  * :class:`evaluation.scoreboard.ScoreboardAssembler` — calls this
    when reading a previously-saved per-instance JSONL.

The class :class:`AnswerEvaluator` is the spec-named surface (per
``IMPLEMENTATION-STATUS.md``); :func:`compute_joint_success` is the
free-function shortcut tests + drivers use most.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional


__all__ = [
    "AnswerEvaluator",
    "EvalInstance",
    "FailureClass",
    "compute_joint_success",
]


# ---------------------------------------------------------------------------
# Failure taxonomy (PLAN-EVAL-FIRST-TARGET.md §6)
# ---------------------------------------------------------------------------


class FailureClass(str, Enum):
    """Canonical F1–F7 failure classes."""

    F1 = "F1"  # answer_wrong + evidence_wrong
    F2 = "F2"  # answer_wrong + evidence_insufficient
    F3 = "F3"  # answer_correct + evidence_missing
    F4 = "F4"  # answer_correct + evidence_mismatched
    F5 = "F5"  # grounding_incomplete
    F6 = "F6"  # over_grounding / unnecessary_tool_use
    F7 = "F7"  # budget_exhaustion / runaway_reasoning


# Deterministic order so the scoreboard table is stable.
FAILURE_CLASSES_ORDERED: tuple[FailureClass, ...] = tuple(FailureClass)


# ---------------------------------------------------------------------------
# Per-instance record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalInstance:
    """Single evaluation instance — the unit the scoreboard rolls up.

    Fields trace 1:1 to the canonical 10-column table:

      * ``answer_correct`` → Answer Acc
      * ``evidence_supported`` → Evidence Support; combined →
        ``Joint Success`` (§3)
      * ``path_a`` → Path A
      * ``binding_success`` → Binding Success
      * ``transfer_pass`` (Optional) → Transfer Pass (only counted on
        instances marked ``transfer``)
      * ``rolled_back`` → Rollback Rate (release-window)
      * ``tool_calls`` → Avg Tool Calls
      * ``cost_usd`` → Cost ($/inst)
      * ``latency_ms`` → Latency (s/inst, p50/p95)

    ``setting`` is one of the 10 canonical row labels
    (``overall``-style aggregations are computed by the driver, not
    stored on the instance).

    ``failure_class`` is filled by :class:`AnswerEvaluator` whenever
    joint success is false; it may be left ``None`` for joint-success
    instances.
    """

    instance_id: str
    setting: str
    domain: str
    answer_correct: bool
    evidence_supported: bool
    path_a: bool = False
    binding_success: bool = True
    transfer: bool = False
    transfer_pass: Optional[bool] = None
    rolled_back: bool = False
    tool_calls: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    failure_class: Optional[str] = None
    grounding_complete: bool = True
    over_grounded: bool = False
    budget_exhausted: bool = False
    target_domain: Optional[str] = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "setting": self.setting,
            "domain": self.domain,
            "answer_correct": self.answer_correct,
            "evidence_supported": self.evidence_supported,
            "path_a": self.path_a,
            "binding_success": self.binding_success,
            "transfer": self.transfer,
            "transfer_pass": self.transfer_pass,
            "rolled_back": self.rolled_back,
            "tool_calls": self.tool_calls,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "failure_class": self.failure_class,
            "grounding_complete": self.grounding_complete,
            "over_grounded": self.over_grounded,
            "budget_exhausted": self.budget_exhausted,
            "target_domain": self.target_domain,
        }
        if self.extras:
            d["extras"] = dict(self.extras)
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "EvalInstance":
        return EvalInstance(
            instance_id=str(d["instance_id"]),
            setting=str(d["setting"]),
            domain=str(d["domain"]),
            answer_correct=bool(d["answer_correct"]),
            evidence_supported=bool(d["evidence_supported"]),
            path_a=bool(d.get("path_a", False)),
            binding_success=bool(d.get("binding_success", True)),
            transfer=bool(d.get("transfer", False)),
            transfer_pass=(
                None
                if d.get("transfer_pass") is None
                else bool(d["transfer_pass"])
            ),
            rolled_back=bool(d.get("rolled_back", False)),
            tool_calls=int(d.get("tool_calls", 0)),
            cost_usd=float(d.get("cost_usd", 0.0)),
            latency_ms=int(d.get("latency_ms", 0)),
            failure_class=(
                None if d.get("failure_class") is None else str(d["failure_class"])
            ),
            grounding_complete=bool(d.get("grounding_complete", True)),
            over_grounded=bool(d.get("over_grounded", False)),
            budget_exhausted=bool(d.get("budget_exhausted", False)),
            target_domain=(
                None if d.get("target_domain") is None else str(d["target_domain"])
            ),
            extras=dict(d.get("extras", {}) or {}),
        )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def compute_joint_success(instance: EvalInstance) -> bool:
    """``True`` iff the instance counts toward Joint Success Rate."""
    return bool(instance.answer_correct) and bool(instance.evidence_supported)


class AnswerEvaluator:
    """Deterministic per-instance judge.

    The evaluator is intentionally *pure logic*: callers (drivers,
    label loaders, tests) construct an :class:`EvalInstance` from
    whatever upstream signal they have (executor outcome,
    grounding-completeness flag, evidence pointer, etc.) and call
    :meth:`evaluate` to attach the failure class.

    Replacing this class with an LLM-based judge later does not
    require any change to the driver or scoreboard.
    """

    def evaluate(self, instance: EvalInstance) -> EvalInstance:
        """Return a new :class:`EvalInstance` with ``failure_class``
        filled in when joint success is false.
        """
        if compute_joint_success(instance):
            if instance.failure_class is not None:
                # Joint success ⇒ no failure class.
                return _replace(instance, failure_class=None)
            return instance

        cls = self._classify(instance)
        if instance.failure_class == cls:
            return instance
        return _replace(instance, failure_class=cls)

    def _classify(self, ins: EvalInstance) -> str:
        """Apply F1..F7 in priority order matching PLAN-EVAL-FIRST-TARGET §6."""

        # F7 — budget exhaustion overrides all others (the loop didn't
        # produce a usable answer at all).
        if ins.budget_exhausted:
            return FailureClass.F7.value

        # F5 — grounding never completed; downstream answer/evidence
        # signals are untrustworthy. Counted before answer-vs-evidence
        # because their truth value is undefined here.
        if not ins.grounding_complete:
            return FailureClass.F5.value

        if ins.answer_correct:
            # Answer right — taxonomy is about the evidence side.
            # F4 takes priority over F3: if the system *cited*
            # evidence but it doesn't actually support the answer
            # (mismatched), that is the more severe right-for-the-
            # wrong-reasons mode.
            if ins.extras.get("evidence_present") and not ins.evidence_supported:
                return FailureClass.F4.value
            return FailureClass.F3.value

        # Answer wrong.
        if ins.evidence_supported is False and ins.extras.get(
            "evidence_present", False
        ):
            # Wrong answer + cited evidence is wrong / mismatched.
            return FailureClass.F1.value

        # F6 — wrong answer reached only after over-grounding.
        if ins.over_grounded:
            return FailureClass.F6.value

        # Default: wrong answer with insufficient evidence.
        return FailureClass.F2.value

    # -- bulk helpers ------------------------------------------------------

    def evaluate_all(self, instances: Iterable[EvalInstance]) -> List[EvalInstance]:
        return [self.evaluate(ins) for ins in instances]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _replace(ins: EvalInstance, **changes: Any) -> EvalInstance:
    """Frozen-dataclass safe replace (dataclasses.replace does not
    play well with default factories on Mapping types in some
    Pythons; this wrapper keeps behavior explicit)."""
    payload = ins.to_dict()
    payload.update(changes)
    return EvalInstance.from_dict(payload)
