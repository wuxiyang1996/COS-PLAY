# Skill Bank — Persistent Storage for Skill Contracts

**Location:** `skill_agents/skill_bank/`

## Overview

The Skill Bank provides **persistent storage** for learned skill contracts and manages **NEW skill discovery**. Two main classes:

### SkillBankMVP

Stores effects-only contracts from [Stage 3 MVP](../stage3_mvp/README.md):

- `SkillEffectsContract` per skill (eff_add, eff_del, eff_event)
- Optional `VerificationReport` per skill
- Versioning and JSONL persistence
- Mutation history for reproducibility
- **`compat_fn`**: contract compatibility scorer for Stage 3 → Stage 2 closed-loop feedback

### NewPoolManager

Rich tracking and promotion of `__NEW__` segments into real skills:

- Per-candidate metadata: effect signature, duration, predecessor/successor context
- Jaccard-based agglomerative clustering (not exact string match)
- Promotion criteria: support + consistency + separability
- Persistence (JSON save/load)

Used by the full [skill_agents](../README.md) pipeline, [contract_verification](../contract_verification/README.md), [bank_maintenance](../bank_maintenance/README.md), and [skill_evaluation](../skill_evaluation/README.md).

## High-level API

```python
from skill_agents.skill_bank import SkillBankMVP

bank = SkillBankMVP(path="data/skill_bank.jsonl")
bank.load()

# Queries
bank.skill_ids
bank.get_contract("nav_to_pot")
bank.get_report("nav_to_pot")
bank.has_skill("nav_to_pot")

# Mutations (usually via Stage 3 or bank_maintenance)
bank.add_or_update(contract, report=...)
bank.remove("old_skill")

# Stage 2 integration: pass compat_fn to SegmentScorer
# scorer = SegmentScorer(skill_names, compat_fn=bank.compat_fn)
bank.compat_fn("nav_to_pot", predicates_start, predicates_end)
# → float in [-1, +1]: effects-based compatibility score

bank.save()
```

### NewPoolManager

```python
from skill_agents.skill_bank.new_pool import NewPoolManager, NewPoolConfig

pool = NewPoolManager(config=NewPoolConfig(min_cluster_size=5))
pool.add(record, predecessor_skill="move", successor_skill="attack")

# Inspect mature clusters
candidates = pool.get_candidates()
for c in candidates:
    print(c.cluster_id, c.size, c.consistency, c.representative_sig)

# Promote qualifying clusters to real skills
created_ids = pool.promote(bank, observations_by_traj)
```

## Contract compatibility scoring (`compat_fn`)

The `compat_fn` provides the Stage 3 → Stage 2 closed-loop feedback signal. For each segment-skill pair it scores how well the observed predicate changes match the skill's learned effects contract:

| Condition | Score |
|-----------|-------|
| `eff_add` literal observed true at end | +1.0 |
| `eff_add` literal observed false at end | -1.0 (contradiction) |
| `eff_add` / `eff_del` literal not in predicates | -0.5 (missing) |
| `eff_del` literal observed false at end | +1.0 (correctly deleted) |
| `eff_del` literal still true at end | -1.0 (contradiction) |

Normalised by contract size → range approx. [-1, +1]. Returns 0.0 when the skill has no contract.

## Key modules

| Module | Purpose |
|--------|--------|
| `bank` | `SkillBankMVP`: in-memory contracts + reports, JSONL load/save, version log, `compat_fn` for Stage 2 |
| `new_pool` | `NewPoolManager`: rich NEW tracking, Jaccard clustering, promotion with support + consistency + separability criteria |
