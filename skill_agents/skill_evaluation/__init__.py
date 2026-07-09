"""
Skill Evaluation — LLM-agentic quality assessment of extracted skills.

All quality judgements are produced by LLM-as-a-judge calls across six
dimensions (no hardcoded heuristic thresholds):

  1. Coherence — intra-skill semantic consistency.
  2. Discriminability — inter-skill separability.
  3. Composability — transition-graph connectivity.
  4. Generalization — cross-trajectory consistency.
  5. Utility — downstream task contribution.
  6. Granularity — appropriate abstraction level.

Plus an optional holistic synthesis pass that produces a final overall
judgement with reasoning.

Quick start::

    from skill_agents.skill_evaluation import run_skill_evaluation
    summary = run_skill_evaluation(bank, all_segments)
    print(summary.format_for_llm())
"""

from skill_agents.skill_evaluation.run_evaluation import run_skill_evaluation
from skill_agents.skill_evaluation.schemas import (
    DimensionScore,
    EvaluationSummary,
    QualityDimension,
    QualityGrade,
    SkillQualityReport,
)
from skill_agents.skill_evaluation.config import (
    LLMJudgeConfig,
    SkillEvaluationConfig,
)

__all__ = [
    "run_skill_evaluation",
    "DimensionScore",
    "EvaluationSummary",
    "QualityDimension",
    "QualityGrade",
    "SkillQualityReport",
    "LLMJudgeConfig",
    "SkillEvaluationConfig",
]
