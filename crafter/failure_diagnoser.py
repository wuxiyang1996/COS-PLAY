"""`FailureDiagnoser` — first-pass diagnosis of a single FailureTrace.

Spec: PLAN-SKILL-CRAFTER §6.4.

The diagnoser maps a `FailureTrace` to a `FailureDiagnosis` containing a
recommended `RecoveryStrategy`. The MVP uses a deterministic rule table;
a frozen-teacher LLM hook is provided for future replacement
(`set_llm_diagnoser`) but defaults to the rule path so the pipeline is
runnable without any external model.
"""

from __future__ import annotations

from typing import Callable, Optional

from common.enums import RecoveryStrategy
from data_structure.extensions.failure_trace import FailureDiagnosis, FailureTrace


_LLMDiagnoser = Callable[[FailureTrace], FailureDiagnosis]


class FailureDiagnoser:
    def __init__(self, llm: Optional[_LLMDiagnoser] = None) -> None:
        self._llm = llm

    def set_llm_diagnoser(self, llm: _LLMDiagnoser) -> None:
        self._llm = llm

    def diagnose(self, trace: FailureTrace) -> FailureDiagnosis:
        if self._llm is not None:
            try:
                return self._llm(trace)
            except Exception:                                  # noqa: BLE001
                # Fall through to the rule path on LLM failure.
                pass
        return self._rule_diagnose(trace)

    # -- rule path --------------------------------------------------------

    def _rule_diagnose(self, trace: FailureTrace) -> FailureDiagnosis:
        cls = trace.failure_class.upper()

        # T1.3c — lane-(a) retrieval-centric classes (canonical names
        # from ``configs/failure_routing.yaml::lane_a_taxonomy``). These
        # come from the synthesis layer (``_crafter_hook.
        # _synthesize_failures`` and ``reflect_per_episode_gpt54``), not
        # from ``harness/skill_harness.py::_classify_abort``.
        if cls == "BANK_GAP":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="retrieval",
                root_cause="bank does not contain a skill matching the situation",
                recommended_strategy=RecoveryStrategy.BANK_GAP,
                confidence=0.7,
                notes="route to Hypothesizer; consider Composer for adjacent merges",
            )
        if cls == "RETRIEVAL_MISLEAD":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="retrieval",
                root_cause="retrieved skill matched contract but ran inappropriately for context",
                recommended_strategy=RecoveryStrategy.RETRIEVAL_MISLEAD,
                confidence=0.6,
                notes="route to Composer (primary) / Hypothesizer (fallback)",
            )
        if cls == "STALE_DESCRIPTION":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="retrieval",
                root_cause="skill description / NLI fields drifted from contract",
                recommended_strategy=RecoveryStrategy.STALE_DESCRIPTION,
                confidence=0.6,
                notes="route to Rewriter (primary) / Hypothesizer (fallback)",
            )

        # Harness-derived classes (existing).
        if cls == "PRECONDITION_VIOLATION":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="precondition",
                root_cause="precondition not satisfied at invocation",
                recommended_strategy=RecoveryStrategy.PRECONDITION_STRENGTHENING,
                confidence=0.7,
            )
        if cls == "INVARIANT_VIOLATION":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="effect_check",
                root_cause="claimed effect lacks evidence (G0)",
                recommended_strategy=RecoveryStrategy.HOP_INSERTION,
                confidence=0.6,
                notes="insert a VERIFY hop before COMMIT",
            )
        if cls == "BUDGET_EXCEEDED":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="protocol_step",
                root_cause="protocol exceeded its hop / token budget",
                recommended_strategy=RecoveryStrategy.SKILL_DECOMPOSITION,
                confidence=0.55,
            )
        if cls == "MISSING_ADAPTER":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="protocol_step",
                root_cause="no adapter registered for (domain, type)",
                recommended_strategy=RecoveryStrategy.SKILL_RETIREMENT,
                confidence=0.4,
            )
        if cls == "ADAPTER_EXCEPTION":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="protocol_step",
                root_cause="adapter raised an exception during a hop",
                recommended_strategy=RecoveryStrategy.FALLBACK_INJECTION,
                confidence=0.5,
            )

        # T1.3c — synthesis-signal aliases from
        # ``configs/failure_routing.yaml::synthesis_signals``. Consulted
        # only when the harness class wasn't matched above (so the
        # closed-loop "INVARIANT_VIOLATION → HOP_INSERTION" mapping
        # used by ``_synthesize_failures`` keeps working). Lane-(a) mode
        # opts in by *writing* ``failure_class`` directly as one of
        # BANK_GAP / RETRIEVAL_MISLEAD / STALE_DESCRIPTION (matched at
        # the top of this method).
        signal = ""
        try:
            signal = str(trace.extra.get("synthesis_signal", "")).upper()
        except Exception:  # noqa: BLE001
            signal = ""
        if signal in {"OUTCOME_FAILURE", "NO_SKILL_BOUND", "LOW_APPLICABILITY"}:
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="retrieval",
                root_cause=f"synthesis signal {signal!r} -> bank gap",
                recommended_strategy=RecoveryStrategy.BANK_GAP,
                confidence=0.65,
                notes="lane-(a) alias from configs/failure_routing.yaml",
            )
        if signal == "MISSING_EFFECTS":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="retrieval",
                root_cause="claimed effects not realized -> stale description",
                recommended_strategy=RecoveryStrategy.STALE_DESCRIPTION,
                confidence=0.6,
                notes="lane-(a) alias from configs/failure_routing.yaml",
            )

        return FailureDiagnosis(
            failure_id=trace.failure_id,
            locus="unknown",
            root_cause=f"unclassified failure: {trace.failure_class}",
            recommended_strategy=RecoveryStrategy.PROTOCOL_PATCH,
            confidence=0.3,
        )


__all__ = ["FailureDiagnoser"]
