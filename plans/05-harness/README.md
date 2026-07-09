# 05-harness — Component: Skill Harness (per-invocation runtime)

The **micro-runtime** for one skill call: retrieve → bind slots → attach adapter → check evidence → execute → log `SkillEpisode`. Frozen 72B verifier / candidate filter / veto layer — **not** the online policy (that stays with the [Actor](../02-action-agent/PLAN-ACTION-AGENT.md)).

## Status (repo snapshot — 2026-05-02)

**Shipped:** Harness MVP + `SkillHarnessHook`; **`RewardLogger.log_grpo_record`** wired into the co-evolution rollout path (audit ~~T2.4~~); eligibility + `validate_invocation` + `RejectedSkillSink` plumbing.
**Open:** Full **G0–G5** gate stack vs current `GateRunner` surface; Q4 harness ablations on intra-`gymv` probe — [`implementation_notes/pre-training-readiness-audit.md`](../../implementation_notes/pre-training-readiness-audit.md) §5.

| Document | Purpose |
|----------|---------|
| [`PLAN-HARNESS.md`](PLAN-HARNESS.md) | `SkillEpisode`, `SkillHarness`, `AdapterRegistry`, `TransferManager`, `ReplayValidator`, `RewardLogger`. Six promotion gates **G0 — Evidence-driven contract**, G1 binding, G2 adapter, G3 replay, G4 shadow, G5 non-regression. Two-phase shadow → active transfer protocol. Transfer-failure diagnostics (§10a). Phase 0 + Phase 1 as the immediate implementation target. |

Cross-cutting consumers: [Pipeline Orchestrator](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) (consumes `GateVerdict` for promotion / rollback), [Unified Skill Gate](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) (Harness owns gate **execution**), [First Eval Target](../00-system/PLAN-EVAL-FIRST-TARGET.md) (Harness filter / veto secondary axes).

Back to [plans/README.md](../README.md).
