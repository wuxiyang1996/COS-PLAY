"""
Configuration for the Skill Evaluation module (LLM-agentic evaluation).

All quality judgements are produced by LLM-as-a-judge calls — no
hardcoded heuristic thresholds.  This config controls LLM call
parameters, prompt behaviour, and output routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class LLMJudgeConfig:
    """Parameters for the LLM judge used across all evaluation dimensions."""

    model: Optional[str] = None
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
