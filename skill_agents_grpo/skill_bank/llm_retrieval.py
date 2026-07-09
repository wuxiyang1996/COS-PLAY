"""
LLM-assisted skill retrieval via the RETRIEVAL LoRA adapter.

Provides ``llm_retrieve_skills`` which uses the trained RETRIEVAL adapter
to rewrite queries and rank skills, augmenting the algorithmic
effect-matching scoring in Stage 2 decode.

When the multi-LoRA model is not configured, falls back gracefully.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_RETRIEVAL_PROMPT_TEMPLATE = """\
You are a skill retrieval assistant for a game-playing agent.

The agent's current goal: {query}
Current state: {current_state}

Available skills:
{skill_list}

Rewrite the query to better match relevant skills, then rank the top skills.

Return ONLY a JSON object:
{{"rewritten_query": "improved query", "ranking": ["best_skill", ...], "reasoning": "brief explanation"}}"""


def _get_retrieval_ask_fn() -> Optional[Callable[..., str]]:
    """Return a RETRIEVAL-routed ask function, or None if unavailable.

    The returned callable is wrapped for reasoning-model compatibility
    (Qwen3 ``/no_think``, think-tag stripping).
    """
    from skill_agents_grpo._llm_compat import wrap_ask_for_reasoning_models

    _hint = "Qwen/Qwen3-8B"
    try:
        from skill_agents_grpo.lora import MultiLoraSkillBankLLM, SkillFunction
        llm = MultiLoraSkillBankLLM.get_shared_instance()
        if llm is not None:
            return wrap_ask_for_reasoning_models(
                llm.as_ask_fn(SkillFunction.RETRIEVAL), model_hint=_hint,
            )
    except Exception:
        pass
    return None


def llm_retrieve_skills(
    query: str,
    current_state: Dict[str, Any],
    skill_descriptions: Dict[str, str],
    top_k: int = 5,
) -> Optional[List[str]]:
    """Use the RETRIEVAL adapter to rank skills for a given query/state.

    Returns a ranked list of skill IDs, or None if the adapter is
    unavailable or the call fails.
    """
    import time as _time
    from skill_agents_grpo.coldstart_io import record_io, ColdStartRecord

    ask_fn = _get_retrieval_ask_fn()
    if ask_fn is None:
        return None

    skill_list = "\n".join(
        f"  - {sid}: {desc}" for sid, desc in list(skill_descriptions.items())[:30]
    )

    prompt = _RETRIEVAL_PROMPT_TEMPLATE.format(
        query=query,
        current_state=json.dumps(current_state, default=str),
        skill_list=skill_list,
    )

    try:
        t0 = _time.time()
        raw = ask_fn(prompt, temperature=0.1)
        elapsed = _time.time() - t0
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = None
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])

        ranking = parsed.get("ranking", []) if parsed else []
        valid = [s for s in ranking if s in skill_descriptions]

        record_io(ColdStartRecord(
            module="skill_retrieval",
            function="retrieve_skills",
            prompt=prompt,
            response=raw or "",
            parsed=parsed,
            model="",
            temperature=0.1,
            elapsed_s=round(elapsed, 3),
            skill_names=list(skill_descriptions.keys())[:30],
            extra={"query": query, "top_k": top_k},
            error=None if valid else "no_valid_ranking",
        ))

        return valid[:top_k] if valid else None
    except Exception as exc:
        logger.debug("RETRIEVAL adapter call failed: %s", exc)

    return None
