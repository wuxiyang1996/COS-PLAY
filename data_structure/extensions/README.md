# `data_structure/extensions/` — P0 extension records

Higher-level artefacts that close the loop across `harness/`, `orchestrator/`, `crafter/`, and `skill_bank/`. These records **extend** the legacy per-tick / per-rollout / per-segment ground truth in [`data_structure/`](../) (`Experience`, `Episode`, `SubTask_Experience`); they do **not** replace it.

Spec: [`plans/00-system/PLAN-EXTENSION.md`](../../plans/00-system/PLAN-EXTENSION.md) and [`plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md`](../../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) §8 (P0 step).

```python
from data_structure.extensions import (
    SkillEpisode, SkillEpisodeStep, SkillEpisodeOutcome,
    SkillRecord, SkillContract,
    SkillEvaluationRecord,
    GateVerdictPayload, StageVerdict,
    BankMutationProposal,
        ComposeProposal, GeneralizeProposal,
        HypothesisProposal, PatchProposal, RetireProposal,
    FailureTrace, FailureDiagnosis,
    RunRelease,
)
```

---

## Record map

| Record | What it represents | Owner / writer | Consumers |
|---|---|---|---|
| `SkillEpisode` | One full skill invocation by the Harness — the typed step trace (`SkillEpisodeStep[]`), the `SkillEpisodeOutcome` (success / fail / aborted), and the evidence interface (`evidence_in / evidence_out / evidence_warrant`). `finalize()` raises if any non-`ACTION` step has empty evidence — this is where invariant **G0 (evidence-driven)** lives. `SkillEpisodeStep.__post_init__` rejects any action type starting with `QUERY_MEM` / `WRITE_MEM` (invariant 2 — no memory). | `harness.SkillHarness.run_skill` | `orchestrator.GateService` (replay + evidence checks); `crafter.FailureMemory` (failure ingestion) |
| `SkillRecord` | The bank entry being evaluated / promoted. Carries `skill_id`, `status`, `source_type`, `contract` (typed input/output schema), `expected_evidence_roles`, `feasible_domains`, `source_domains`, `verified_domains`, `adapter_history`, `content_hash`. `verified_domains` is **gate-owned** — only `SkillLifecycleManager.record_transfer_verification(...)` may mutate it (invariant 8). | `skill_bank.SkillLifecycleManager` (only writer) | Everyone reads via `SkillRepository` |
| `SkillContract` | Typed slot spec attached to every `SkillRecord`: input slots, output slots, pre/post conditions. The harness's `EligibilityFilter` matches against this. | `crafter` (proposes), `lifecycle` (validates) | `harness.EligibilityFilter` |
| `SkillEvaluationRecord` | The gate's signed verdict for one promotion attempt: `(skill_id, content_hash, gate_verdict_payload, eligible_domains, evaluator, timestamp, signature)`. Carried as the proof object on every `PromotionPlan`. | `orchestrator.GateService.evaluate` | `orchestrator.PromotionOrchestrator` (rejects FAIL, refuses ACTIVE on LIMITED_PASS) |
| `GateVerdictPayload` | Per-stage breakdown attached to the eval: ordered `StageVerdict[]` for stages 0 (static) → 1 (replay) → 2 (shadow) → 3a (transfer) → 3b (non-regression) → 4 (provisional) → 5 (active). Each `StageVerdict` is `(stage, verdict, metrics, notes, eligible_domains)`. | `orchestrator.GateService.{Stage0…Stage5}` | `PromotionOrchestrator`, telemetry dashboards |
| `BankMutationProposal` (sum type) | Typed crafter proposal — see Crafter README. The five concrete subclasses: `ComposeProposal` (chain N existing skills), `GeneralizeProposal` (lift slot bindings), `HypothesisProposal` (novel skill from failure pattern), `PatchProposal` (repair a known failure), `RetireProposal` (deprecate). | `crafter.SkillCrafterService.propose_*` | `orchestrator.ArtifactStore.put_proposal` → `lifecycle.ingest_draft` |
| `FailureTrace` | One observed failure of a skill / step: `(skill_id, episode_id, step_idx, failure_mode, evidence_snapshot, recovery_attempt)`. Failure modes come from the closed enum in `SubTask_Experience.failure_mode` (slot binding, adapter exec mismatch, evidence insufficient, temporal mismatch, UI grounding mismatch, desktop object mismatch, overconfident commit, contract mismatch). | `harness.SkillHarness.run_skill` (on failure) | `crafter.FailureMemory`, `crafter.FailureDiagnoser` |
| `FailureDiagnosis` | Output of `crafter.FailureDiagnoser`: which `BankMutationProposal` family is appropriate (compose / generalize / hypothesise / patch / retire) and why. | `crafter.FailureDiagnoser` | `crafter.SkillCrafterService` |
| `RunRelease` | Frozen snapshot of `(bank ⊕ adapter weights ⊕ config)` at promotion time. The atomic unit for rollback in `orchestrator.SnapshotManager`. | `orchestrator.SnapshotManager.create_release` | `orchestrator.PromotionOrchestrator.rollback`, eval drivers |

---

## Why these records, not extensions to `Experience` / `Episode`?

The legacy per-tick records (`data_structure/experience.py`) are **episode-local** — they capture the rollout itself. The records in this package are **promotion-local** and **lifecycle-local** — they capture what the system decides about a skill across many rollouts. Mixing the two would either bloat every `Experience` with bank-bookkeeping fields or scatter promotion state through the rollout traces. Keeping them separate also lets the no-memory invariant (#2) bite at exactly one place: `SkillEpisodeStep.__post_init__`.

---

## Cross-references

- [`../README.md`](../README.md) — the legacy `Experience` / `Episode` / `SubTask_Experience` records these extend.
- [`../../plans/05-harness/PLAN-HARNESS.md`](../../plans/05-harness/PLAN-HARNESS.md) §5 — `SkillEpisode` extension and the G0 evidence-driven contract.
- [`../../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) §6 — `SkillRecord` lifecycle and `SkillEvaluationRecord` signing rules.
- [`../../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md`](../../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md) §4 — `BankMutationProposal` taxonomy.
