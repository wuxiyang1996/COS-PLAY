# Stage 3 MVP: Effects-Only Contract Learning

**Location:** `skill_agents/stage3_mvp/`

## Overview

Stage 3 MVP learns **effects-only** contracts from segmented trajectories (Stage 2). Each skill gets:

- **eff_add** — predicates that reliably become true by segment end
- **eff_del** — predicates that reliably become false by segment end
- **eff_event** — event-like predicates that occur during the segment

No preconditions, invariants, SPLIT, or MATERIALIZE_NEW. Used by the main pipeline and by [bank_maintenance](../bank_maintenance/README.md) and [skill_bank](../skill_bank/README.md).

## Pipeline

1. Extract per-timestep predicates; summarize each segment.
2. Compute per-instance add/delete/event effects.
3. Aggregate into initial effects contracts per skill.
4. Verify contracts against instances.
5. Refine by dropping unreliable literals.
6. Persist into a Skill Bank (JSONL) with versioning.

## Integration

- **Input:** Segment list (e.g. from [infer_segmentation](../infer_segmentation/README.md)); use `specs_from_segmentation_result()` to convert.
- **Output:** [SkillBankMVP](../skill_bank/README.md) with `SkillEffectsContract` and `VerificationReport` per skill.
- **Closed loop → Stage 2:** Learned contracts feed back into Stage 2 via `SkillBankMVP.compat_fn`. When `contract_feedback_mode` is `"weak"` or `"strong"`, the Stage 2 `SegmentScorer` uses contract compatibility as a soft bias to guide skill assignment toward skills whose effects match observed predicate changes. See [infer_segmentation/README.md](../infer_segmentation/README.md) §4 for details.

## High-level API

```python
from skill_agents.stage3_mvp import run_stage3_mvp, specs_from_segmentation_result

# From Stage 2 SegmentationResult
specs = specs_from_segmentation_result(result, traj_id, observations_by_traj)
summary = run_stage3_mvp(specs, observations_by_traj, bank, config=...)
```

## Key modules

| Module | Purpose |
|--------|--------|
| `run_stage3_mvp` | Orchestrator: summarize → effects → learn → verify → refine |
| `extract_predicates` | Per-timestep predicate extraction |
| `segment_summarize` | Segment-level predicate summaries |
| `effects_compute` | Add/delete/event effects per instance |
| `contract_learn` | Aggregate into effects contract |
| `contract_verify` | Verify contract vs instances |
| `contract_refine` | Drop unreliable literals |
