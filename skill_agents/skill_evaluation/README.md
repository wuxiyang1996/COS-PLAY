# Skill Evaluation — LLM-Agentic Quality Assessment

**Location:** `skill_agents/skill_evaluation/`

## Overview

Skill Evaluation runs **LLM-as-a-judge** on every skill in the bank. All quality judgements come from LLM calls — no hardcoded heuristic thresholds. Six dimensions plus an optional holistic synthesis:

| Dimension | Focus |
|-----------|--------|
| **Coherence** | Intra-skill semantic consistency |
| **Discriminability** | Inter-skill separability |
| **Composability** | Transition-graph connectivity |
| **Generalization** | Cross-trajectory consistency |
| **Utility** | Downstream task contribution |
| **Granularity** | Appropriate abstraction level |
| **Holistic** (optional) | Overall judgement with reasoning |

## Integration

- **Input:** [SkillBankMVP](../skill_bank/README.md), segment records (same types as [Stage 3](../stage3_mvp/README.md) and [bank_maintenance](../bank_maintenance/README.md)).
- **Output:** `EvaluationSummary` usable by:
  - Bank maintenance (split/merge/refine priorities)
  - Stage 2 LLM teacher (re-label low-quality skills)
  - Downstream agents (filter or weight skills)

## High-level API

```python
from skill_agents.skill_evaluation import run_skill_evaluation

summary = run_skill_evaluation(bank, all_segments)
print(summary.format_for_llm())
```

## Key modules

| Module | Purpose |
|--------|--------|
| `run_evaluation` | Orchestrator; builds lightweight profiles, runs evaluators |
| `evaluators` | Per-dimension LLM judge: coherence, discriminability, composability, generalization, utility, granularity, holistic |
| `schemas` | `EvaluationSummary`, `SkillQualityReport`, `QualityDimension`, `DimensionScore` |
| `config` | `SkillEvaluationConfig`, `LLMJudgeConfig` (model, temperature, prompt limits) |
