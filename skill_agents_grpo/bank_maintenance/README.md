# Stage 4: Bank Maintenance — Split / Merge / Refine

**Location:** `skill_agents/bank_maintenance/`

## Overview

Bank Maintenance keeps the [Skill Bank](../skill_bank/README.md) high-quality by applying three update operations:

| Operation | When | Effect |
|-----------|------|--------|
| **SPLIT** | One skill has multiple modes | Split into child skills |
| **MERGE** | Two skills are near-duplicates | Merge into one canonical skill |
| **REFINE** | Contracts too strong or weak | Drop fragile or add discriminative literals |

It also updates:

- Duration model \( p(\ell \mid k) \)
- Indices: effect inverted index, MinHash/LSH, optional ANN
- Local re-decode for affected trajectories after splits/merges

## Pipeline (run order)

1. Build/update **SkillProfiles** (only for changed skills).
2. **SplitQueue** → execute splits → local re-decode.
3. **MergeCandidates** → verify → execute merges → re-decode if needed.
4. **RefineQueue** → refine contracts + duration/start–end updates.
5. Update indices incrementally.
6. Emit bank diff report.

## Integration

- **Input:** [SkillBankMVP](../skill_bank/README.md), segment records, optional [Stage 3](../stage3_mvp/README.md) / [contract_verification](../contract_verification/README.md) outputs.
- **Output:** Updated bank, `BankDiffReport`, `RedecodeRequest` list for downstream re-segmentation.

## High-level API

```python
from skill_agents.bank_maintenance import run_bank_maintenance, BankMaintenanceConfig

result = run_bank_maintenance(
    bank=bank,
    segments_by_traj=...,
    config=BankMaintenanceConfig(...),
)
# result.diff_report, result.split_results, result.redecode_requests, ...
```

## Key modules

| Module | Purpose |
|--------|--------|
| `run_bank_maintenance` | Orchestrator |
| `split` | Split triggers, execute split, re-decode requests |
| `merge` | Merge candidates, verify pair, execute merge |
| `refine` | Refine triggers, refine skill, duration model update |
| `duration_model` | Store and query \( p(\ell \mid k) \) |
| `indices` | Effect index, MinHash/LSH, optional ANN |
| `local_redecode` | Re-decode windows after splits/merges |
