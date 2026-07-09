# Stage 3: Contract Verification — Full Contract Learn / Verify / Refine

**Location:** `skill_agents/contract_verification/`

## Overview

Contract Verification is the full Stage 3 pipeline for building and maintaining a **Skill Bank** from segmented trajectories (Stage 2 output). Each skill gets a verifiable contract with:

- **Pre** — preconditions
- **Eff+** / **Eff-** — add/delete effects
- **Inv** — invariants

Verification produces update actions:

- **KEEP** — contract holds, no changes
- **REFINE** — drop or soften unstable literals
- **SPLIT** — instances cluster into distinct sub-skills
- **MATERIALIZE_NEW** — repeated `__NEW__` segments become a new skill

## Pipeline

1. Extract per-timestep predicates and build smoothed segment summaries.
2. Compute per-instance add/delete effects and invariants.
3. Aggregate into initial contracts per skill.
4. Verify contracts and collect counterexamples.
5. Decide KEEP / REFINE / SPLIT per skill.
6. Materialize `__NEW__` segments into new skills.
7. Persist updates with versioning; produce agent-facing summary.

## Integration

- **Input:** `SegmentationResult` from [Stage 2 (infer_segmentation)](../infer_segmentation/README.md).
- **Output:** `SkillBank` with `compat_fn` for Stage 2’s `SegmentScorer`; `Stage3Summary` for LLM teacher and [bank_maintenance](../bank_maintenance/README.md).

## High-level API

```python
from skill_agents.contract_verification import run_stage3, run_stage3_batch, SkillBank

# Single trajectory
summary = run_stage3(result, traj_id, observations, config=...)

# Batch
summaries = run_stage3_batch(results_and_obs, config=...)

# Bank provides compat_fn for Stage 2
bank = SkillBank(path="data/skill_bank.jsonl")
scorer = SegmentScorer(skill_names=bank.get_skill_names(), compat_fn=bank.compat_fn, ...)
```

## Key modules

| Module | Purpose |
|--------|--------|
| `run_stage3` | Orchestrator; agent-facing summary |
| `predicates` | Build segment predicates from observations |
| `contract_init` | Compute effects, build initial contracts |
| `contract_verify` | Verify contracts, counterexamples |
| `updates` | REFINE / SPLIT / materialize new skills |
| `skill_bank` | Persistent bank + versioning |
| `action_language` | Export to PDDL / STRIPS / compact text |
