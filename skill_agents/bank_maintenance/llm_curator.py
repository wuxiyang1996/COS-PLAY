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
    from skill_agents._llm_compat import wrap_ask_for_reasoning_models

    _hint = "Qwen/Qwen3.5-9B"
    try:
        from skill_agents.lora import MultiLoraSkillBankLLM, SkillFunction
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
    from skill_agents.coldstart_io import record_io, ColdStartRecord

    if not candidates:
        return {"decisions": []}

    ask_fn = _get_curator_ask_fn()
    if ask_fn is None:
        return None

    summary = bank_summary or make_bank_summary(bank)
    prompt = _build_curator_prompt(candidates, summary)

    try:
        from skill_agents._llm_retry import sync_ask_with_retry

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
            from skill_agents.grpo.grpo_outputs import SkillBankLLMOutput

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


# T2.7 — curator overfit mitigation. The CURATOR LoRA is small + trained
# on a tight objective (approve/veto/defer); under noisy early-GRPO
# rewards it overfits before the actor has produced enough variety in
# candidate proposals to give a useful signal. Mitigation: scale the
# scalar reward seen by the GRPO wrapper by a linear ramp from 0 to
# ``weight`` over ``warmup_steps`` outer-loop steps. Module-level state
# is the simplest plumbing — the orchestrator calls
# ``set_curator_warmup(...)`` once per outer step before
# ``filter_candidates`` runs (the wrapper reads the state on every
# scoring call). Defaults preserve the legacy "no scaling" behaviour.
_curator_warmup_state: Dict[str, Any] = {
    "weight": 1.0,
    "warmup_steps": 0,
    "current_step": 0,
}


def set_curator_warmup(
    *,
    weight: Optional[float] = None,
    warmup_steps: Optional[int] = None,
    current_step: Optional[int] = None,
) -> None:
    """T2.7: configure the per-step ramp on ``curator_reward``.

    Any argument left as ``None`` keeps its previously-set value.
    Call this from the trainer once per outer step before the
    skill-bank pipeline fires (e.g. inside
    ``PerGameSkillBankManager.reset_for_step``).
    """

    if weight is not None:
        _curator_warmup_state["weight"] = float(weight)
    if warmup_steps is not None:
        _curator_warmup_state["warmup_steps"] = max(0, int(warmup_steps))
    if current_step is not None:
        _curator_warmup_state["current_step"] = max(0, int(current_step))


def get_curator_warmup_state() -> Dict[str, Any]:
    """Read-only snapshot of the current warmup state (for tests / logs)."""

    return dict(_curator_warmup_state)


def _curator_reward_weight() -> float:
    """Resolve the current ramp multiplier in ``[0, weight]``."""

    weight = float(_curator_warmup_state["weight"])
    warmup = int(_curator_warmup_state["warmup_steps"])
    if warmup <= 0:
        return weight
    step = int(_curator_warmup_state["current_step"])
    ramp = min(1.0, max(0.0, step) / float(max(1, warmup)))
    return weight * ramp


def enable_curator_grpo(
    buffer: Any,
    group_size: int = 4,
    temperature: float = 0.7,
) -> None:
    """Activate GRPO wrapping on ``filter_candidates``.

    Reward context (compute_quality_fn, execute_fn) is read dynamically
    from the thread-local ``_reward_ctx`` at scoring time.  Call
    :func:`set_curator_reward_context` before maintenance runs.

    T2.7: the scalar reward returned to the GRPO buffer is multiplied
    by :func:`_curator_reward_weight` (a linear ramp configured via
    :func:`set_curator_warmup`). The default state (``weight=1.0,
    warmup_steps=0``) is a no-op identity scaling; configure the ramp
    once at trainer start to enable overfit mitigation.
    """
    import skill_agents.bank_maintenance.llm_curator as _mod
    from skill_agents.grpo.rewards import curator_reward
    from skill_agents.grpo.wrapper import GRPOCallWrapper
    from skill_agents.lora.skill_function import SkillFunction

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
        base = curator_reward(
            decisions, *args,
            action_outcomes=ctx.get("action_outcomes"),
            **passthrough,
        )
        # T2.7 — multiply by the live ramp weight; defaults to 1.0.
        return float(base) * _curator_reward_weight()

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
    import skill_agents.bank_maintenance.llm_curator as _mod

    global _grpo_original_fn
    if _grpo_original_fn is not None:
        _mod.filter_candidates = _grpo_original_fn
        _grpo_original_fn = None
        logger.info("Curator GRPO disabled")
