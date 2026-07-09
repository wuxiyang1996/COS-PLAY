"""Lightweight LLM-judge stage for the live Promotion driver.

Bridges the documented Phase-1 gap where ``--gate-mode offline-synthetic``
runs only rule-based Stage 0 plus LIMITED_PASS placeholders for Stages
1-4, never touching the 35B-A3B teacher backbone.  The judge here is a
single 35B call per proposal that rates whether the proposal is sound
enough to enter the bank, and is intentionally *additive* — it sits
alongside the synthetic stages, can independently FAIL a proposal, and
otherwise leaves the Phase-1 verdict structure untouched.

Routing: the model identifier (default ``BACKBONE_JUDGE_MODEL``) is
resolved through ``API_func.ask_model``, which honours
``VLLM_BASE_URL_MAP`` so calls land on the dedicated 35B endpoint
without any extra plumbing in this module.

Failure-mode contract: if the LLM call raises, times out, or returns
unparseable text, this module returns ``GateVerdict.LIMITED_PASS`` with
a diagnostic ``notes`` field so the caller never crashes the Promotion
driver on a flaky judge — the synthetic stages still gate the
proposal as before.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from common.enums import GateStage, GateVerdict
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
    RewriteProposal,
)
from data_structure.extensions.gate_verdict import StageVerdict
from data_structure.extensions.skill_record import SkillRecord

logger = logging.getLogger("labeling_supplement.llm_skill_judge")


# Maximum chars from any free-text field we copy into the prompt — keeps
# token budgets bounded for very chatty rationales / protocols.
_MAX_FIELD_CHARS = 800
_MAX_PROTOCOL_STEPS = 12

# Verdict map — how the LLM's coarse grade translates back to the
# unified gate vocabulary.  ``poor`` is the only escape hatch that can
# override the synthetic LIMITED_PASS into a hard FAIL.
_GRADE_TO_VERDICT: Dict[str, GateVerdict] = {
    "fail":     GateVerdict.FAIL,
    "poor":     GateVerdict.FAIL,
    "limited":  GateVerdict.LIMITED_PASS,
    "limited_pass": GateVerdict.LIMITED_PASS,
    "fair":     GateVerdict.LIMITED_PASS,
    "pass":     GateVerdict.PASS,
    "good":     GateVerdict.PASS,
    "excellent": GateVerdict.PASS,
}


@dataclass
class JudgeOutcome:
    verdict: GateVerdict
    rationale: str
    raw_response: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _summarize_proposal(proposal: BankMutationProposal) -> Dict[str, Any]:
    """Produce a short JSON-able dict describing the proposal kind and any
    proposal-specific fields the judge should weigh."""
    base: Dict[str, Any] = {
        "kind":              type(proposal).__name__,
        "proposal_id":       proposal.proposal_id,
        "rationale":         (proposal.rationale or "")[:_MAX_FIELD_CHARS],
        "parent_skill_ids":  list(proposal.parent_skill_ids),
        "target_domains":    list(proposal.target_domains),
        "teacher_model":     proposal.teacher_model,
    }
    if isinstance(proposal, ComposeProposal):
        base["component_skill_ids"] = list(proposal.component_skill_ids)
        base["composed_protocol"] = proposal.composed_protocol[:_MAX_PROTOCOL_STEPS]
    elif isinstance(proposal, GeneralizeProposal):
        base["base_skill_id"] = proposal.base_skill_id
        base["abstracted_protocol"] = proposal.abstracted_protocol[:_MAX_PROTOCOL_STEPS]
        base["source_domain"] = proposal.source_domain
        base["target_domain"] = proposal.target_domain
    elif isinstance(proposal, HypothesisProposal):
        base["novel_protocol"] = proposal.novel_protocol[:_MAX_PROTOCOL_STEPS]
        base["seed_failure_pattern_ids"] = list(proposal.source_failure_pattern_ids)
    elif isinstance(proposal, PatchProposal):
        base["base_skill_id"] = proposal.base_skill_id
        base["patched_protocol"] = proposal.patched_protocol[:_MAX_PROTOCOL_STEPS]
        base["recovery_strategy"] = proposal.recovery_strategy
    elif isinstance(proposal, RewriteProposal):
        base["base_skill_id"] = getattr(proposal, "base_skill_id", "")
    elif isinstance(proposal, RetireProposal):
        base["target_skill_id"] = proposal.target_skill_id
        base["reason"] = (proposal.reason or "")[:_MAX_FIELD_CHARS]
    return base


def _summarize_skill(skill: SkillRecord) -> Dict[str, Any]:
    contract = skill.contract
    return {
        "skill_id":       skill.skill_id,
        "name":           getattr(skill, "name", ""),
        "skill_type":     skill.skill_type.value,
        "source_type":    skill.source_type.value,
        "status":         skill.status.value,
        "source_domains": list(skill.source_domains),
        "feasible_domains": list(skill.feasible_domains),
        "transfer_target_domains": list(getattr(skill, "transfer_target_domains", []) or []),
        "protocol":       (skill.protocol or [])[:_MAX_PROTOCOL_STEPS],
        "contract": {
            "expected_evidence_roles": list(getattr(contract, "expected_evidence_roles", []) or []),
            "expected_outcome_signals": list(getattr(contract, "expected_outcome_signals", []) or []),
            "preconditions": (getattr(contract, "preconditions", "") or "")[:_MAX_FIELD_CHARS],
            "postconditions": (getattr(contract, "postconditions", "") or "")[:_MAX_FIELD_CHARS],
        },
    }


_VERDICT_BLOCK = (
    "Respond with EXACTLY one JSON object on a single line, with these\n"
    "fields and nothing else:\n"
    "  {\"verdict\": \"pass\" | \"limited_pass\" | \"fail\",\n"
    "   \"reason\":  \"<one short sentence>\"}\n"
    "\n"
    "Verdict semantics:\n"
    "  - \"pass\"          → the proposal is well-formed and likely useful;\n"
    "  - \"limited_pass\"  → plausible but unproven (default for cold-start);\n"
    "  - \"fail\"          → nonsensical, incoherent, contradictory, or\n"
    "                      clearly redundant.\n"
)


# Kind-aware criteria — keyed by ``type(proposal).__name__``.  The default
# (skill-mutation kinds: AddProposal/PatchProposal/ComposeProposal/
# GeneralizeProposal/RewriteProposal) keeps the original "needs a sound
# protocol" contract.  HypothesisProposal / RetireProposal use specialised
# blocks because they are *not* skill-mutations:
#   - HypothesisProposal is a probe asking the harness to collect evidence
#     for a claimed effect; an empty ``novel_protocol`` is by-design and
#     must NOT be treated as a fail reason (otherwise every cold-start
#     hypothesis is auto-vetoed and Phase B′ never grows the bank).
#   - RetireProposal targets an existing skill_id — the evaluation lives
#     in "is the retirement justified?", not in protocol coherence.
_CRITERIA_HYPOTHESIZE = (
    "This proposal is a HYPOTHESIS — a probe that asks the harness to\n"
    "collect evidence for a claimed effect on a future step.  It is *not*\n"
    "expected to ship an executable protocol; an empty ``novel_protocol``\n"
    "is normal for fresh hypotheses and MUST NOT be a fail reason.  Once\n"
    "promoted, the hypothesis enters the bank as a low-confidence stub\n"
    "(role=VERIFY) that the actor / harness can probe next step.\n"
    "\n"
    "Use these criteria:\n"
    "  - Worth investigating: rationale points to a real, non-trivial\n"
    "    open question (not a tautology, not an effect already settled).\n"
    "  - Grounded in seed_failure_pattern_ids: the hypothesis ties to an\n"
    "    actual failure pattern observed in this game / domain.\n"
    "  - Target domains plausible: the hypothesis is meaningfully scoped,\n"
    "    not 'applies to everything' boilerplate.\n"
    "  - Non-redundant with subject_skill: the hypothesis doesn't simply\n"
    "    restate what the parent skill's contract already promises.\n"
    "  - Coherence (when populated): if novel_protocol / preconditions /\n"
    "    effects_add ARE supplied, they are well-formed; missing or empty\n"
    "    is fine and should grade as ``limited_pass`` not ``fail``.\n"
)

_CRITERIA_RETIRE = (
    "This proposal RETIRES an existing skill from the bank.  Use these\n"
    "criteria — *do not* require a protocol (retirements remove a skill,\n"
    "they don't ship one):\n"
    "  - Target validity: target_skill_id is non-empty and references a\n"
    "    real skill in this bank.\n"
    "  - Justification: the ``reason`` cites concrete evidence of\n"
    "    obsolescence, harm, or persistent low admit-rate — not vague\n"
    "    dislike or duplicate-of-a-better-skill without naming the better\n"
    "    skill.\n"
    "  - Non-circular: the retirement does not orphan dependent skills\n"
    "    relied on elsewhere in the bank.\n"
)

_CRITERIA_DEFAULT = (
    "Use these criteria:\n"
    "  - Coherence: protocol steps are well-formed and consistent with\n"
    "    the contract and the skill_type.\n"
    "  - Generality: the proposal is plausibly useful in the listed\n"
    "    target/feasible domains, not over-specialised to one trace.\n"
    "  - Soundness: action vocabulary is sane, no contradictions, no\n"
    "    obvious empty/garbage fields.\n"
    "  - Novelty / non-redundancy: the proposal adds a meaningful\n"
    "    capability rather than restating the parent skill.\n"
)


def _criteria_for(proposal: BankMutationProposal) -> str:
    """Pick the criteria block matching the proposal's structural kind.

    Skill-mutation proposals (Add/Patch/Compose/Generalize/Rewrite) reuse
    the default "protocol must be sound" contract.  Hypothesize / Retire
    bypass the protocol requirement because their semantics don't ship a
    protocol — see the rationale on ``_CRITERIA_HYPOTHESIZE`` /
    ``_CRITERIA_RETIRE``.
    """
    if isinstance(proposal, HypothesisProposal):
        return _CRITERIA_HYPOTHESIZE
    if isinstance(proposal, RetireProposal):
        return _CRITERIA_RETIRE
    return _CRITERIA_DEFAULT


def _build_prompt(
    *, proposal: BankMutationProposal, skill: SkillRecord, game_hint: Optional[str],
) -> str:
    summary = {
        "game": game_hint or "(unspecified)",
        "proposal": _summarize_proposal(proposal),
        "subject_skill": _summarize_skill(skill),
    }
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    criteria = _criteria_for(proposal)
    return (
        "You are an offline judge for a multi-game RL agent's skill bank.\n"
        "Decide whether the following PROPOSAL is sound enough to be\n"
        "PROMOTED into the agent's live shared skill bank, given the subject\n"
        "skill it would attach to.\n"
        "\n"
        + criteria
        + "\n"
        + _VERDICT_BLOCK
        + "\n"
        "INPUT (JSON):\n"
        + summary_json
        + "\n"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_grade(response: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(grade_str, reason_str)`` from a noisy LLM response.

    Tolerates the model wrapping JSON in fences / preamble / trailing
    whitespace.  Returns ``(None, None)`` only if no plausible grade
    token can be recovered.
    """
    if not response:
        return (None, None)
    txt = response.strip()
    # Pass 1 — try strict JSON parse.
    try:
        obj = json.loads(txt)
        v = obj.get("verdict") if isinstance(obj, dict) else None
        r = obj.get("reason", "")  if isinstance(obj, dict) else ""
        if isinstance(v, str):
            return (v.strip().lower(), str(r).strip())
    except Exception:
        pass
    # Pass 2 — find the first {...} blob in the text.
    m = _JSON_OBJ_RE.search(txt)
    if m is not None:
        try:
            obj = json.loads(m.group(0))
            v = obj.get("verdict") if isinstance(obj, dict) else None
            r = obj.get("reason", "")  if isinstance(obj, dict) else ""
            if isinstance(v, str):
                return (v.strip().lower(), str(r).strip())
        except Exception:
            pass
    # Pass 3 — keyword fallback.  Pick the most-severe token if multiple
    # appear (so a model that hedges with "limited but possibly fails"
    # still maps to FAIL).
    low = txt.lower()
    severity_order = ["fail", "poor", "limited_pass", "limited", "fair",
                      "pass", "good", "excellent"]
    for tok in severity_order:
        if tok in low:
            return (tok, "")
    return (None, None)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def judge_proposal(
    *,
    proposal: BankMutationProposal,
    skill: SkillRecord,
    model: str,
    game_hint: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 256,
    enable_thinking: bool = False,
) -> JudgeOutcome:
    """One LLM call → :class:`JudgeOutcome`.

    Always returns a value; the caller never has to wrap this in
    try/except.  On any failure path the verdict degrades gracefully to
    ``LIMITED_PASS`` so the offline-synthetic floor is preserved.

    ``enable_thinking`` (default ``False``) toggles Qwen3 ``<think>``
    chain-of-thought via ``API_func.ask_vllm``.  Stage 1 in-domain
    promotion gating keeps it off (fast verdict, ≤256 tokens / call);
    Stage 2 cross-domain runs opt in for higher-fidelity judgements,
    in which case the caller must also bump ``max_tokens`` to ≥ 2048
    so the ``<think>`` block has room to complete before the JSON
    verdict.
    """
    prompt = _build_prompt(proposal=proposal, skill=skill, game_hint=game_hint)
    raw = ""
    try:
        import time as _t
        from API_func import ask_model
        _t0 = _t.monotonic()
        raw = ask_model(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        ) or ""
        # Block A5: attribute promotion-judge wall-time.  Best-effort
        # import — the labeling_supplement module is also called outside
        # the trainer (offline judge runs), so the import may fail.
        try:
            from trainer.coevolution._run_loggers import (  # noqa: WPS433
                record_component_call,
            )
            record_component_call(
                "promotion.judge",
                latency_ms=(_t.monotonic() - _t0) * 1000.0,
            )
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "llm_skill_judge: ask_model raised; defaulting to LIMITED_PASS "
            "(model=%s proposal=%s err=%s)",
            model, getattr(proposal, "proposal_id", "?"), exc,
        )
        return JudgeOutcome(
            verdict=GateVerdict.LIMITED_PASS,
            rationale="llm-judge: call_failed",
            raw_response="",
            error=str(exc),
        )

    grade, reason = _parse_grade(raw)
    if grade is None:
        logger.warning(
            "llm_skill_judge: response did not parse to a verdict; "
            "defaulting to LIMITED_PASS (proposal=%s raw=%r)",
            getattr(proposal, "proposal_id", "?"), (raw or "")[:200],
        )
        return JudgeOutcome(
            verdict=GateVerdict.LIMITED_PASS,
            rationale="llm-judge: parse_failed",
            raw_response=raw,
            error="parse_failed",
        )

    verdict = _GRADE_TO_VERDICT.get(grade, GateVerdict.LIMITED_PASS)
    return JudgeOutcome(
        verdict=verdict,
        rationale=(reason or grade or "")[:_MAX_FIELD_CHARS],
        raw_response=raw,
    )


def build_stage_verdict(outcome: JudgeOutcome) -> StageVerdict:
    """Wrap a :class:`JudgeOutcome` into a :class:`StageVerdict` we can
    append onto an existing offline-synthetic verdict's stage list.

    Re-uses ``GateStage.STATIC`` (no enum extension needed for backward
    compat); the ``notes`` field carries the ``llm-judge:`` provenance
    prefix so dashboards can split it back out.
    """
    notes_prefix = "llm-judge: "
    notes = notes_prefix + (outcome.rationale or "(no reason)")
    if outcome.error:
        notes = notes_prefix + f"error={outcome.error}; floor=LIMITED_PASS"
    failures = [outcome.rationale] if outcome.verdict == GateVerdict.FAIL else []
    return StageVerdict(
        stage=GateStage.STATIC,
        verdict=outcome.verdict,
        metrics={"llm_judge.invoked": 1.0},
        failures=failures,
        notes=notes,
    )


__all__ = [
    "JudgeOutcome",
    "build_stage_verdict",
    "judge_proposal",
]
