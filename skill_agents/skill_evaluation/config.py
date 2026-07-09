"""
Configuration for the Skill Evaluation module (LLM-agentic evaluation).

All quality judgements are produced by LLM-as-a-judge calls — no
hardcoded heuristic thresholds.  This config controls LLM call
parameters, prompt behaviour, and output routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


def _default_judge_model() -> Optional[str]:
    """Resolve the canonical judge model from ``common.models``.

    Falls back to ``None`` (which the caller interprets as "use whatever
    ``API_func.ask_model`` defaults to") only when the constant import
    itself fails — extremely unusual but keeps this dataclass importable
    in stripped-down test environments without the full repo on path.
    """
    try:
        from common.models import BACKBONE_JUDGE_MODEL
        return BACKBONE_JUDGE_MODEL
    except Exception:  # noqa: BLE001
        return None


@dataclass
class LLMJudgeConfig:
    """Parameters for the LLM judge used across all evaluation dimensions.

    The judge is intentionally distinct from the actor backbone (the
    actor is the LoRA-trained 9B; the judge is the frozen 35B-A3B
    teacher). By default it uses
    ``common.models.BACKBONE_JUDGE_MODEL`` (``Qwen/Qwen3.5-35B-A3B``),
    which means judge calls hit the local 35B vLLM server with no
    API spend. The 35B is shared with the crafter / harness / orchestrator
    teacher (same weights, different role) so both can be served by a
    single ``inference/serve_qwen35_35b_a3b.sh`` instance.

    For paper / formal eval where within-Qwen-family self-preference
    bias must be controlled, override to an off-distribution oracle
    (e.g. ``gpt-5.5``) by exporting
    ``VLM_AGENT_BACKBONE_JUDGE_MODEL=gpt-5.5`` — see
    ``implementation_notes/coevolution-cross-domain-integration.md``
    §"Judge family bias" for the spot-check protocol.
    """

    model: Optional[str] = field(default_factory=_default_judge_model)
    temperature: float = 0.3
    max_tokens: int = 2048

    # Maximum number of instances to include in the prompt context
    # (to control prompt length; a representative sample is selected)
    max_instances_in_prompt: int = 10

    # Maximum characters per state/observation string in the prompt
    max_state_chars: int = 400

    # Whether to include a chain-of-thought request in prompts
    chain_of_thought: bool = True

    # Optional custom ask_model function; if None, imports from API_func
    ask_model_fn: Optional[Callable] = None


@dataclass
class SkillEvaluationConfig:
    """Top-level configuration for the full skill evaluation pipeline."""

    llm: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)

    # Per-dimension weights for overall score aggregation (all default to 1.0)
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "coherence": 1.0,
        "discriminability": 1.0,
        "composability": 0.8,
        "generalization": 1.0,
        "utility": 1.2,
        "granularity": 0.8,
    })

    # Minimum instances required before evaluating a skill
    min_instances_for_eval: int = 3

    # Whether to run all six dimensions or a subset
    enabled_dimensions: List[str] = field(default_factory=lambda: [
        "coherence",
        "discriminability",
        "composability",
        "generalization",
        "utility",
        "granularity",
    ])

    # Whether to run a final LLM pass that synthesises dimension scores
    # into an overall judgement with holistic reasoning
    run_holistic_pass: bool = True

    # Path for saving evaluation reports
    report_path: Optional[str] = None
