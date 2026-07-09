"""
LLM curator filter for bank maintenance candidate actions.

The algorithmic pipeline in ``run_bank_maintenance()`` proposes candidate
mutations (refine, merge, split, materialize, promote). This module adds
a single-turn LLM filter that reviews candidates and returns approve /
veto / defer decisions.

GRPO wrapping: ``enable_curator_grpo()`` activates the GRPO wrapper on
``filter_candidates()``. G samples are generated, evaluated via
``curator_reward()``, and the best is returned. The maintenance pipeline
is unaware of the wrapping.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Dynamic reward context (thread-safe) ─────────────────────────────

_reward_ctx = threading.local()


def set_curator_reward_context(
    *,
    action_outcomes: Optional[list] = None,
) -> None:
    """Update the per-thread curator reward context.

    Called from the maintenance pipeline before ``filter_candidates``
    so that the GRPO reward uses the outcome-based scoring path.

    Parameters
    ----------
    action_outcomes : list[dict]
        One entry per candidate: ``{"succeeded": bool, "quality_delta": float}``.
    """
    _reward_ctx.data = {
        "action_outcomes": action_outcomes,
    }


def _get_curator_reward_context() -> dict:
    return getattr(_reward_ctx, "data", {})

_CURATOR_PROMPT_TEMPLATE = """\
You are a skill bank maintenance curator. Review the proposed actions and decide \
whether to approve, veto, or defer each one. Base your decisions on skill quality \
(skill_score) and encourage new skill exploration when evidence supports it.

## Bank Summary
Total skills: {n_skills}
Mean pass rate: {mean_pass_rate:.2f}
Mean skill score: {mean_skill_score:.2f}
Skills with low pass rate (<0.60): {n_low_pass}

## Proposed Actions
{actions_text}

For each action, respond with a JSON object:
{{"decisions": [{{"idx": 0, "verdict": "approve|veto|defer", "reason": "brief reason citing skill_score and metrics"}}, ...]}}

Action types: SPLIT, MERGE, REFINE, MATERIALIZE, PROMOTE.

Guidelines:
- Base decisions primarily on **skill_score** (0-1) which reflects episode reward, \
reuse success, contract quality, and cross-episode consistency.
- APPROVE actions on skills with skill_score > 0.5 that have clear evidence.
- VETO actions where skill_score is low and evidence is contradictory.
- For MATERIALIZE/PROMOTE: **encourage new skill exploration** — approve if the \
skill has a valid contract and reasonable pass rate, even with limited instances. \
New skills expand the bank's coverage of game behaviors.
- DEFER only when evidence is truly insufficient (no contract, zero instances).
- Cite specific metric values (skill_score, pass_rate, n_instances) in your reasoning.
"""


def _format_action(idx: int, action: Dict[str, Any]) -> str:
    """Format one candidate action for the prompt."""
    action_type = action.get("type", "unknown")
    skill_id = action.get("skill_id", "?")
    parts = [f"  Action {idx}: {action_type.upper()} on {skill_id}"]

    if "skill_score" in action:
        parts.append(f"    Skill score: {action['skill_score']:.2f}")
    if "trigger" in action:
        parts.append(f"    Trigger: {action['trigger']}")
    if "pass_rate" in action:
        parts.append(f"    Pass rate: {action['pass_rate']:.2f}")
    if "n_instances" in action:
        parts.append(f"    Instances: {action['n_instances']}")
    if "details" in action:
        details = action["details"]
        if isinstance(details, dict):
            for k, v in list(details.items())[:5]:
                parts.append(f"    {k}: {v}")
        else:
            parts.append(f"    Details: {str(details)[:200]}")

    return "\n".join(parts)


def _get_curator_ask_fn() -> Optional[Callable[..., str]]:
    """Return a CURATOR-routed ask function.

    Resolution order:
      1. CURATOR LoRA adapter via ``MultiLoraSkillBankLLM``
      2. Local vLLM (``ask_vllm``) — avoids OpenRouter rate limits
      3. ``ask_model`` (routes through OpenRouter)
    """
    from skill_agents_grpo._llm_compat import wrap_ask_for_reasoning_models

    _hint = "Qwen/Qwen3-8B"
    try:
        from skill_agents_grpo.lora import MultiLoraSkillBankLLM, SkillFunction
        llm = MultiLoraSkillBankLLM.get_shared_instance()
        if llm is not None:
            return wrap_ask_for_reasoning_models(
                llm.as_ask_fn(SkillFunction.CURATOR), model_hint=_hint,
            )
    except Exception:
        pass
    try:
        from API_func import ask_vllm, _probe_vllm
        if _probe_vllm():
            logger.debug("CURATOR fallback: using local vLLM")
            return wrap_ask_for_reasoning_models(ask_vllm, model_hint=_hint)
    except Exception:
        pass
    from API_func import ask_model
    return wrap_ask_for_reasoning_models(ask_model, model_hint=_hint)


def _build_curator_prompt(
    candidates: List[Dict[str, Any]],
    bank_summary: Dict[str, Any],
) -> str:
    """Build the CURATOR prompt from candidates and bank summary."""
    actions_text = "\n\n".join(
        _format_action(i, c) for i, c in enumerate(candidates)
    )
    return _CURATOR_PROMPT_TEMPLATE.format(
        n_skills=bank_summary.get("n_skills", 0),
        mean_pass_rate=bank_summary.get("mean_pass_rate", 0.0),
        mean_skill_score=bank_summary.get("mean_skill_score", 0.5),
        n_low_pass=bank_summary.get("n_low_pass", 0),
        actions_text=actions_text or "(no actions proposed)",
    )


def make_bank_summary(bank: Any) -> Dict[str, Any]:
    """Extract a summary dict from a SkillBankMVP for the curator prompt."""
    skills = []
    if hasattr(bank, "skills"):
        skills = list(bank.skills.values()) if isinstance(bank.skills, dict) else bank.skills
    elif hasattr(bank, "list_skills"):
        skills = bank.list_skills()

    n_skills = len(skills)
    pass_rates = []
    skill_scores = []
    for s in skills:
        if hasattr(s, "contract") and s.contract:
            pr = getattr(s, "pass_rate", None)
            if pr is not None:
                pass_rates.append(pr)
        if hasattr(s, "compute_skill_score"):
            try:
                skill_scores.append(s.compute_skill_score())
            except Exception:
                pass

    mean_pr = sum(pass_rates) / max(len(pass_rates), 1) if pass_rates else 0.0
    mean_ss = sum(skill_scores) / max(len(skill_scores), 1) if skill_scores else 0.5
    n_low = sum(1 for pr in pass_rates if pr < 0.60)

    return {
        "n_skills": n_skills,
        "mean_pass_rate": mean_pr,
        "mean_skill_score": mean_ss,
        "n_low_pass": n_low,
    }


def filter_candidates(
    candidates: List[Dict[str, Any]],
    bank: Any,
    *,
    bank_summary: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Use the CURATOR adapter to filter bank maintenance candidates.

    Returns ``{"decisions": [{"idx": 0, "verdict": "approve", "reason": "..."}, ...]}``
    or None if the adapter is unavailable.

    When GRPO wrapping is active (via ``enable_curator_grpo``), the
    wrapper intercepts this call automatically.
    """
    import time as _time
    from skill_agents_grpo.coldstart_io import record_io, ColdStartRecord

    if not candidates:
        return {"decisions": []}

    ask_fn = _get_curator_ask_fn()
    if ask_fn is None:
        return None

    summary = bank_summary or make_bank_summary(bank)
    prompt = _build_curator_prompt(candidates, summary)

    try:
        from skill_agents_grpo._llm_retry import sync_ask_with_retry

        t0 = _time.time()
        raw = sync_ask_with_retry(
            ask_fn,
            prompt,
            log_label="CURATOR:filter_candidates",
            temperature=temperature,
        )
        elapsed = _time.time() - t0
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = None
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])

        record_io(ColdStartRecord(
            module="bank_curator",
            function="filter_candidates",
            prompt=prompt,
            response=raw or "",
            parsed=parsed,
            model="",
            temperature=temperature,
            elapsed_s=round(elapsed, 3),
            extra={"n_candidates": len(candidates)},
            error=None if parsed and "decisions" in parsed else "parse_failed",
        ))

        if parsed and "decisions" in parsed:
            from skill_agents_grpo.grpo.grpo_outputs import SkillBankLLMOutput

            return SkillBankLLMOutput(dict(parsed), raw_completion=raw or "")
    except Exception as exc:
        logger.warning("CURATOR adapter call failed: %s", exc)

    return None


def apply_curator_decisions(
    candidates: List[Dict[str, Any]],
    decisions: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter candidates by curator decisions, returning only approved ones.

    If decisions is None (curator unavailable), returns all candidates
    (algorithmic default).
    """
    if decisions is None:
        return candidates

    decision_list = decisions.get("decisions", [])
    approved_indices = {
        d["idx"] for d in decision_list
        if d.get("verdict") == "approve" and "idx" in d
    }

    if not approved_indices and not decision_list:
        return candidates

    return [c for i, c in enumerate(candidates) if i in approved_indices]


# ── GRPO integration ──────────────────────────────────────────────────

_grpo_original_fn: Optional[Callable] = None


def enable_curator_grpo(
    buffer: Any,
    group_size: int = 4,
    temperature: float = 0.7,
) -> None:
    """Activate GRPO wrapping on ``filter_candidates``.

    Reward context (compute_quality_fn, execute_fn) is read dynamically
    from the thread-local ``_reward_ctx`` at scoring time.  Call
    :func:`set_curator_reward_context` before maintenance runs.
    """
    import skill_agents_grpo.bank_maintenance.llm_curator as _mod
    from skill_agents_grpo.grpo.rewards import curator_reward
    from skill_agents_grpo.grpo.wrapper import GRPOCallWrapper
    from skill_agents_grpo.lora.skill_function import SkillFunction

    global _grpo_original_fn

    if _grpo_original_fn is not None:
        logger.warning("Curator GRPO already enabled — skipping")
        return

    _grpo_original_fn = _mod.filter_candidates

    def _dynamic_curator_reward(decisions, *args, **kwargs):
        ctx = _get_curator_reward_context()
        passthrough = {
            k: v for k, v in kwargs.items()
            if k not in ("action_outcomes",)
        }
        return curator_reward(
            decisions, *args,
            action_outcomes=ctx.get("action_outcomes"),
            **passthrough,
        )

    def _prompt_extractor(
        candidates: List[Dict[str, Any]],
        bank: Any,
        *,
        bank_summary: Optional[Dict[str, Any]] = None,
        **kw: Any,
    ) -> str:
        summary = bank_summary or make_bank_summary(bank)
        return _build_curator_prompt(candidates, summary)

    def _metadata_extractor(
        candidates: List[Dict[str, Any]],
        *a: Any,
        **kw: Any,
    ) -> Dict[str, Any]:
        return {"n_candidates": len(candidates)}

    wrapper = GRPOCallWrapper(
        adapter=SkillFunction.CURATOR,
        reward_fn=_dynamic_curator_reward,
        buffer=buffer,
        group_size=group_size,
        temperature=temperature,
        prompt_extractor=_prompt_extractor,
        metadata_extractor=_metadata_extractor,
    )

    _mod.filter_candidates = wrapper.wrap(_grpo_original_fn)
    logger.info("Curator GRPO enabled: G=%d, temp=%.2f", group_size, temperature)


def disable_curator_grpo() -> None:
    """Deactivate GRPO wrapping, restore original function."""
    import skill_agents_grpo.bank_maintenance.llm_curator as _mod

    global _grpo_original_fn
    if _grpo_original_fn is not None:
        _mod.filter_candidates = _grpo_original_fn
        _grpo_original_fn = None
        logger.info("Curator GRPO disabled")
