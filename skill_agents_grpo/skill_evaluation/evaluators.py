"""
LLM-as-a-Judge evaluators for skill quality assessment.

Each evaluator builds a structured prompt with the skill's contract,
representative instances, and bank-wide context, then calls the LLM
to produce a quality judgement.  The LLM returns a JSON response with
a score (1-10), grade, evidence bullets, and actionable recommendations.

No hardcoded heuristic thresholds — all quality reasoning is delegated
to the LLM judge.

Follows the same ``ask_model`` pattern as ``llm_teacher.py`` and
``llm_extractor.py``.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from skill_agents_grpo.skill_evaluation.config import LLMJudgeConfig
from skill_agents_grpo.skill_evaluation.schemas import (
    DimensionScore,
    QualityDimension,
)
from skill_agents_grpo.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)
from skill_agents_grpo.bank_maintenance.schemas import SkillProfile

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# LLM call helpers
# ═════════════════════════════════════════════════════════════════════


def _get_ask_model(config: LLMJudgeConfig) -> Callable:
    """Return the LLM call function, preferring user-supplied, then API_func.

    Wrapped for reasoning-model compatibility (Qwen3 /no_think, think-tag stripping).
    """
    from skill_agents_grpo._llm_compat import wrap_ask_for_reasoning_models

    _hint = "Qwen/Qwen3-8B"
    if config.ask_model_fn is not None:
        return wrap_ask_for_reasoning_models(config.ask_model_fn, model_hint=_hint)
    from API_func import ask_model
    return wrap_ask_for_reasoning_models(ask_model, model_hint=_hint)


def _call_llm(prompt: str, config: LLMJudgeConfig) -> str:
    """Call the LLM with the given prompt and return raw response text."""
    ask = _get_ask_model(config)
    kwargs: dict = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.model is not None:
        kwargs["model"] = config.model
    return ask(prompt, **kwargs)


def _parse_json_from_response(response: str) -> Optional[dict]:
    """Extract JSON object from LLM response (handles markdown fences, extra text)."""
    text = re.sub(r"```(?:json)?\s*", "", response)
    text = text.strip().rstrip("`")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        candidate = match.group(0)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


def _parse_dimension_response(
    response: str,
    dimension: QualityDimension,
) -> DimensionScore:
    """Parse LLM JSON response into a DimensionScore."""
    parsed = _parse_json_from_response(response)
    if parsed is None:
        logger.warning("Failed to parse LLM response for %s", dimension.value)
        return DimensionScore(
            dimension=dimension,
            score=0.0,
            evidence=["LLM response could not be parsed"],
            details={"raw_response": response[:500]},
        )

    raw_score = parsed.get("score", 5)
    if isinstance(raw_score, (int, float)):
        score = max(0.0, min(float(raw_score) / 10.0, 1.0))
    else:
        score = 0.5

    evidence = parsed.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]

    return DimensionScore(
        dimension=dimension,
        score=score,
        evidence=evidence,
        details={
            k: v for k, v in parsed.items()
            if k not in ("score", "evidence")
        },
    )


# ═════════════════════════════════════════════════════════════════════
# Context formatting helpers
# ═════════════════════════════════════════════════════════════════════


def _format_contract(contract: SkillEffectsContract) -> str:
    """Format a skill contract for prompt injection."""
    lines = [
        f"Skill ID: {contract.skill_id} (version {contract.version})",
        f"Effects-Add: {sorted(contract.eff_add) if contract.eff_add else '(none)'}",
        f"Effects-Del: {sorted(contract.eff_del) if contract.eff_del else '(none)'}",
        f"Effects-Event: {sorted(contract.eff_event) if contract.eff_event else '(none)'}",
        f"Total literals: {contract.total_literals}",
        f"N instances: {contract.n_instances}",
    ]
    if contract.support:
        top_support = sorted(contract.support.items(), key=lambda x: -x[1])[:10]
        lines.append(f"Top support: {dict(top_support)}")
    return "\n".join(lines)


def _format_report(report: Optional[VerificationReport]) -> str:
    """Format a verification report for prompt injection."""
    if report is None:
        return "No verification report available."
    lines = [
        f"Overall pass rate: {report.overall_pass_rate:.3f}",
        f"N instances verified: {report.n_instances}",
    ]
    if report.eff_add_success_rate:
        lines.append(f"Eff-add success rates: {report.eff_add_success_rate}")
    if report.eff_del_success_rate:
        lines.append(f"Eff-del success rates: {report.eff_del_success_rate}")
    if report.eff_event_rate:
        lines.append(f"Eff-event rates: {report.eff_event_rate}")
    if report.worst_segments:
        lines.append(f"Worst segments: {report.worst_segments[:5]}")
    if report.failure_signatures:
        top_sigs = sorted(report.failure_signatures.items(), key=lambda x: -x[1])[:5]
        lines.append(f"Top failure signatures: {dict(top_sigs)}")
    return "\n".join(lines)


def _format_instances(
    instances: List[SegmentRecord],
    max_instances: int,
) -> str:
    """Format a representative sample of instances for prompt injection."""
    sample = instances[:max_instances]
    lines = []
    for i, inst in enumerate(sample):
        lines.append(
            f"  Instance {i+1}: seg_id={inst.seg_id}, traj={inst.traj_id}, "
            f"t=[{inst.t_start},{inst.t_end}], duration={inst.t_end - inst.t_start}, "
            f"eff_add={sorted(inst.eff_add)}, eff_del={sorted(inst.eff_del)}, "
            f"eff_event={sorted(inst.eff_event)}, "
            f"signature={inst.effect_signature()}"
        )
    if len(instances) > max_instances:
        lines.append(f"  ... ({len(instances) - max_instances} more instances omitted)")
    return "\n".join(lines)


def _format_peer_skills(
    skill_id: str,
    all_profiles: Dict[str, SkillProfile],
    max_peers: int = 8,
) -> str:
    """Format other skills in the bank for discriminability context."""
    peers = {k: v for k, v in all_profiles.items() if k != skill_id}
    if not peers:
        return "(no other skills in the bank)"
    lines = []
    for pid, prof in list(peers.items())[:max_peers]:
        lines.append(
            f"  {pid}: eff_add={sorted(prof.eff_add)}, "
            f"eff_del={sorted(prof.eff_del)}, "
            f"eff_event={sorted(prof.eff_event)}, "
            f"n_instances={prof.n_instances}, "
            f"pass_rate={prof.overall_pass_rate:.2f}"
        )
    if len(peers) > max_peers:
        lines.append(f"  ... ({len(peers) - max_peers} more skills omitted)")
    return "\n".join(lines)


def _format_transitions(
    skill_id: str,
    transition_bigrams: Dict[str, Counter],
) -> str:
    """Format transition statistics for composability context."""
    prev_counts = transition_bigrams.get(f"_to_{skill_id}", Counter())
    next_counts = transition_bigrams.get(f"{skill_id}_to_", Counter())
    lines = []
    if prev_counts:
        lines.append(f"Predecessors: {dict(prev_counts.most_common(8))}")
    else:
        lines.append("Predecessors: (none observed)")
    if next_counts:
        lines.append(f"Successors: {dict(next_counts.most_common(8))}")
    else:
        lines.append("Successors: (none observed)")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# JSON response schema (shared across all dimension prompts)
# ═════════════════════════════════════════════════════════════════════

_RESPONSE_SCHEMA = """\
Return ONLY a JSON object (no extra text) with this structure:
{
  "score": <integer 1-10, where 10 is best>,
  "evidence": ["reason 1", "reason 2", ...],
  "issues": ["issue 1", ...],
  "recommendation": "KEEP" | "REFINE" | "SPLIT" | "MERGE" | "DISCARD"
}"""


# ═════════════════════════════════════════════════════════════════════
# 1. COHERENCE
# ═════════════════════════════════════════════════════════════════════

_COHERENCE_PROMPT = """\
You are an expert evaluator for learned skills in an agentic game-playing pipeline.

Your task: evaluate the **COHERENCE** of the following skill — do all instances of this skill perform the same logical sub-task, or are they a grab-bag of unrelated behaviors lumped under one label?

=== Skill Contract ===
{contract}

=== Verification Report ===
{report}

=== Instances ===
{instances}

Evaluate coherence by reasoning about:
1. Do the effect signatures (eff_add, eff_del, eff_event) across instances tell a consistent story, or are they scattered / contradictory?
2. Are the durations (t_end - t_start) roughly consistent, or wildly varying (suggesting mixed sub-tasks)?
3. Do the support counts suggest that certain effects only fire for a subset of instances (indicating the skill conflates multiple behaviors)?
4. Would a human looking at these instances agree they all represent "the same kind of action"?

A score of 10 means every instance is doing essentially the same thing.
A score of 1 means instances are completely unrelated behaviors.

{response_schema}"""


def evaluate_coherence(
    skill_id: str,
    contract: SkillEffectsContract,
    instances: List[SegmentRecord],
    report: Optional[VerificationReport],
    config: LLMJudgeConfig,
) -> DimensionScore:
    """LLM-judged coherence: are all instances semantically consistent?"""
    prompt = _COHERENCE_PROMPT.format(
        contract=_format_contract(contract),
        report=_format_report(report),
        instances=_format_instances(instances, config.max_instances_in_prompt),
        response_schema=_RESPONSE_SCHEMA,
    )
    response = _call_llm(prompt, config)
    return _parse_dimension_response(response, QualityDimension.COHERENCE)


# ═════════════════════════════════════════════════════════════════════
# 2. DISCRIMINABILITY
# ═════════════════════════════════════════════════════════════════════

_DISCRIMINABILITY_PROMPT = """\
You are an expert evaluator for learned skills in an agentic game-playing pipeline.

Your task: evaluate the **DISCRIMINABILITY** of the following skill — is it clearly distinct from every other skill in the bank, or is it a near-duplicate / heavily overlapping with another skill?

=== Target Skill ===
{contract}

=== Other Skills in the Bank ===
{peer_skills}

Evaluate discriminability by reasoning about:
1. Does this skill have unique effects that no other skill produces? Or does it share most/all effects with another skill?
2. If two skills have similar effect sets, do they differ meaningfully in context (e.g., different preconditions, different game phases)?
3. Could a planner distinguish between this skill and its closest neighbour, or would they be interchangeable?
4. Would merging this skill with its closest match lose meaningful behavioral information?

A score of 10 means this skill is clearly unique in the bank.
A score of 1 means it is essentially a duplicate of another skill.

If you recommend MERGE, include the skill ID it should merge with in the "issues" field.

{response_schema}"""


def evaluate_discriminability(
    skill_id: str,
    contract: SkillEffectsContract,
    all_profiles: Dict[str, SkillProfile],
    config: LLMJudgeConfig,
) -> DimensionScore:
    """LLM-judged discriminability: is this skill distinct from peers?"""
    prompt = _DISCRIMINABILITY_PROMPT.format(
        contract=_format_contract(contract),
        peer_skills=_format_peer_skills(skill_id, all_profiles),
        response_schema=_RESPONSE_SCHEMA,
    )
    response = _call_llm(prompt, config)
    return _parse_dimension_response(response, QualityDimension.DISCRIMINABILITY)


# ═════════════════════════════════════════════════════════════════════
# 3. COMPOSABILITY
# ═════════════════════════════════════════════════════════════════════

_COMPOSABILITY_PROMPT = """\
You are an expert evaluator for learned skills in an agentic game-playing pipeline.

Your task: evaluate the **COMPOSABILITY** of the following skill — does it fit naturally into multi-step plans, chaining with other skills as predecessors and successors?

=== Skill Contract ===
{contract}

=== Transition Statistics ===
{transitions}

=== All Skills in the Bank ===
{all_skill_ids}

Evaluate composability by reasoning about:
1. Does this skill have clear, diverse predecessors (skills that naturally lead into it)?
2. Does this skill have clear, diverse successors (skills that naturally follow it)?
3. Is there an excessive self-loop (the skill is followed by itself too often, suggesting poor boundary detection)?
4. Does the skill act as a useful building block in a plan, or is it an isolated "dead-end" that doesn't connect to the rest of the skill vocabulary?
5. Do the transition patterns make logical sense (e.g., "pick_up" followed by "place" is natural)?

A score of 10 means the skill chains naturally with diverse neighbours.
A score of 1 means the skill is isolated with no meaningful transitions.

{response_schema}"""


def evaluate_composability(
    skill_id: str,
    contract: SkillEffectsContract,
    transition_bigrams: Dict[str, Counter],
    all_skill_ids: List[str],
    config: LLMJudgeConfig,
) -> DimensionScore:
    """LLM-judged composability: does this skill chain well with others?"""
    prompt = _COMPOSABILITY_PROMPT.format(
        contract=_format_contract(contract),
        transitions=_format_transitions(skill_id, transition_bigrams),
        all_skill_ids=all_skill_ids,
        response_schema=_RESPONSE_SCHEMA,
    )
    response = _call_llm(prompt, config)
    return _parse_dimension_response(response, QualityDimension.COMPOSABILITY)


# ═════════════════════════════════════════════════════════════════════
# 4. GENERALIZATION
# ═════════════════════════════════════════════════════════════════════

_GENERALIZATION_PROMPT = """\
You are an expert evaluator for learned skills in an agentic game-playing pipeline.

Your task: evaluate the **GENERALIZATION** of the following skill — does it work consistently across different trajectories / episodes / contexts, or is it overfitted to a specific scenario?

=== Skill Contract ===
{contract}

=== Verification Report ===
{report}

=== Cross-Trajectory Distribution ===
{traj_distribution}

=== Instances (sampled across trajectories) ===
{instances}

Evaluate generalization by reasoning about:
1. How many distinct trajectories does this skill appear in? A skill seen in only one trajectory may be a spurious pattern.
2. Is the contract pass rate stable across trajectories, or does it work well in some and fail in others?
3. Do the instances from different trajectories show similar effect patterns, or does the skill behave differently depending on context?
4. Would this skill be useful if applied to a new, unseen trajectory?

A score of 10 means the skill generalizes robustly across all observed contexts.
A score of 1 means it is specific to one trajectory and unlikely to transfer.

{response_schema}"""


def evaluate_generalization(
    skill_id: str,
    contract: SkillEffectsContract,
    instances: List[SegmentRecord],
    report: Optional[VerificationReport],
    per_traj_pass_rates: Optional[Dict[str, float]],
    config: LLMJudgeConfig,
) -> DimensionScore:
    """LLM-judged generalization: does this skill transfer across contexts?"""
    traj_ids = set(inst.traj_id for inst in instances)
    traj_counts: Counter = Counter(inst.traj_id for inst in instances)

    traj_lines = [f"Appears in {len(traj_ids)} distinct trajectories."]
    traj_lines.append(f"Per-trajectory instance counts: {dict(traj_counts.most_common())}")
    if per_traj_pass_rates:
        traj_lines.append(f"Per-trajectory pass rates: {per_traj_pass_rates}")
    traj_distribution = "\n".join(traj_lines)

    prompt = _GENERALIZATION_PROMPT.format(
        contract=_format_contract(contract),
        report=_format_report(report),
        traj_distribution=traj_distribution,
        instances=_format_instances(instances, config.max_instances_in_prompt),
        response_schema=_RESPONSE_SCHEMA,
    )
    response = _call_llm(prompt, config)
    return _parse_dimension_response(response, QualityDimension.GENERALIZATION)


# ═════════════════════════════════════════════════════════════════════
# 5. UTILITY
# ═════════════════════════════════════════════════════════════════════

_UTILITY_PROMPT = """\
You are an expert evaluator for learned skills in an agentic game-playing pipeline.

Your task: evaluate the **UTILITY** of the following skill — does it meaningfully contribute to the agent's downstream task completion, or is it a vacuous / trivial action that changes nothing important?

=== Skill Contract ===
{contract}

=== Task Outcome Correlation ===
{outcome_info}

=== Instances ===
{instances}

Evaluate utility by reasoning about:
1. Does this skill produce meaningful state changes (eff_add, eff_del, eff_event)? A skill with zero or trivial effects is not useful.
2. Is the skill correlated with task success? Does it appear more in successful episodes than failed ones?
3. Would a planner benefit from having this skill available, or is it redundant / trivially replaceable?
4. Does the skill capture an important sub-task that advances the agent toward its goal?

A score of 10 means the skill is critical for task completion.
A score of 1 means the skill is vacuous or actively harmful.

{response_schema}"""


def evaluate_utility(
    skill_id: str,
    contract: SkillEffectsContract,
    instances: List[SegmentRecord],
    episode_outcomes: Optional[Dict[str, bool]],
    config: LLMJudgeConfig,
) -> DimensionScore:
    """LLM-judged utility: does this skill contribute to task success?"""
    outcome_lines = []
    if episode_outcomes:
        skill_trajs = set(inst.traj_id for inst in instances)
        success_trajs = {tid for tid, ok in episode_outcomes.items() if ok}
        failure_trajs = {tid for tid, ok in episode_outcomes.items() if not ok}
        in_success = len(skill_trajs & success_trajs)
        in_failure = len(skill_trajs & failure_trajs)
        outcome_lines.append(
            f"Present in {in_success}/{len(success_trajs)} successful episodes "
            f"and {in_failure}/{len(failure_trajs)} failed episodes."
        )
    else:
        outcome_lines.append("No episode outcome data available.")

    n_effects = len(contract.eff_add) + len(contract.eff_del) + len(contract.eff_event)
    outcome_lines.append(f"Total effect magnitude: {n_effects}")

    prompt = _UTILITY_PROMPT.format(
        contract=_format_contract(contract),
        outcome_info="\n".join(outcome_lines),
        instances=_format_instances(instances, config.max_instances_in_prompt),
        response_schema=_RESPONSE_SCHEMA,
    )
    response = _call_llm(prompt, config)
    return _parse_dimension_response(response, QualityDimension.UTILITY)


# ═════════════════════════════════════════════════════════════════════
# 6. GRANULARITY
# ═════════════════════════════════════════════════════════════════════

_GRANULARITY_PROMPT = """\
You are an expert evaluator for learned skills in an agentic game-playing pipeline.

Your task: evaluate the **GRANULARITY** of the following skill — is it at the right level of abstraction? Not too atomic (single-frame actions) and not too coarse (entire episodes lumped into one skill).

=== Skill Contract ===
{contract}

=== Duration Statistics ===
{duration_info}

=== Instances ===
{instances}

Evaluate granularity by reasoning about:
1. Is the mean duration reasonable for a reusable sub-task? Very short (1-2 timesteps) suggests it's just a single action, not a meaningful skill. Very long (hundreds of timesteps) suggests it's an entire episode, not a decomposable unit.
2. Is the duration consistent across instances? High variance may indicate the skill conflates short and long behaviors.
3. Is the contract appropriately sized? Too few literals (0-1) means the contract is uninformative. Too many (50+) means it's over-specified / brittle.
4. Could this skill be further decomposed into meaningful sub-skills, or is it already at a natural granularity?

A score of 10 means the skill is at the ideal level of abstraction.
A score of 1 means it is either trivially atomic or uselessly coarse.

{response_schema}"""


def evaluate_granularity(
    skill_id: str,
    contract: SkillEffectsContract,
    instances: List[SegmentRecord],
    profile: Optional[SkillProfile],
    config: LLMJudgeConfig,
) -> DimensionScore:
    """LLM-judged granularity: is the skill at the right abstraction level?"""
    lengths = [inst.t_end - inst.t_start for inst in instances]
    if lengths:
        dur_mean = sum(lengths) / len(lengths)
        dur_min = min(lengths)
        dur_max = max(lengths)
    elif profile:
        dur_mean = profile.duration_mean
        dur_min = dur_max = dur_mean
    else:
        dur_mean = dur_min = dur_max = 0.0

    duration_info = (
        f"Mean duration: {dur_mean:.1f} timesteps\n"
        f"Min duration: {dur_min}, Max duration: {dur_max}\n"
        f"Contract size: {contract.total_literals} literals"
    )

    prompt = _GRANULARITY_PROMPT.format(
        contract=_format_contract(contract),
        duration_info=duration_info,
        instances=_format_instances(instances, config.max_instances_in_prompt),
        response_schema=_RESPONSE_SCHEMA,
    )
    response = _call_llm(prompt, config)
    return _parse_dimension_response(response, QualityDimension.GRANULARITY)


# ═════════════════════════════════════════════════════════════════════
# HOLISTIC — final synthesis pass
# ═════════════════════════════════════════════════════════════════════

_HOLISTIC_PROMPT = """\
You are an expert evaluator for learned skills in an agentic game-playing pipeline.

You have already evaluated the skill "{skill_id}" across six quality dimensions. Now synthesise these individual assessments into a **holistic quality judgement** and a final **recommended action**.

=== Skill Contract ===
{contract}

=== Dimension Scores ===
{dimension_scores}

Based on the above dimension-level evaluations, provide your overall assessment:

1. What is the overall quality of this skill (1-10)?
2. What is the single most important action to take?
   - KEEP: the skill is good as-is
   - REFINE: the skill has potential but its contract needs adjustment
   - SPLIT: the skill conflates multiple behaviors and should be decomposed
   - MERGE: the skill overlaps heavily with another and should be merged (specify which)
   - DISCARD: the skill is too low quality to be useful

Return ONLY a JSON object:
{{
  "score": <integer 1-10>,
  "evidence": ["holistic reason 1", "holistic reason 2", ...],
  "recommendation": "KEEP" | "REFINE" | "SPLIT" | "MERGE" | "DISCARD",
  "merge_with": "<skill_id or null>",
  "reasoning": "one-paragraph synthesis of the overall quality"
}}"""


def evaluate_holistic(
    skill_id: str,
    contract: SkillEffectsContract,
    dimension_scores: Dict[str, DimensionScore],
    config: LLMJudgeConfig,
) -> dict:
    """Final LLM pass that synthesises dimension scores into overall judgement.

    Returns a dict with keys: score, evidence, recommendation, merge_with, reasoning.
    """
    dim_lines = []
    for dim_name, ds in dimension_scores.items():
        dim_lines.append(
            f"  {dim_name}: {ds.score:.2f}/1.0 ({ds.grade.value})"
        )
        for ev in ds.evidence[:3]:
            dim_lines.append(f"    - {ev}")

    prompt = _HOLISTIC_PROMPT.format(
        skill_id=skill_id,
        contract=_format_contract(contract),
        dimension_scores="\n".join(dim_lines),
    )
    response = _call_llm(prompt, config)
    parsed = _parse_json_from_response(response)

    if parsed is None:
        logger.warning("Failed to parse holistic LLM response for %s", skill_id)
        return {
            "score": 5,
            "evidence": ["Holistic LLM response could not be parsed"],
            "recommendation": "KEEP",
            "merge_with": None,
            "reasoning": "",
        }

    return {
        "score": parsed.get("score", 5),
        "evidence": parsed.get("evidence", []),
        "recommendation": parsed.get("recommendation", "KEEP"),
        "merge_with": parsed.get("merge_with"),
        "reasoning": parsed.get("reasoning", ""),
    }
