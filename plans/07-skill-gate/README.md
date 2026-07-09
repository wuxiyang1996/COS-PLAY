# 07-skill-gate — Component: Unified Skill Gate

Canonical lifecycle and promotion specification shared by [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md), [Harness](../05-harness/PLAN-HARNESS.md), and [Pipeline Orchestrator](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md). Pins ownership so "no promotion without gate" is **mechanically enforceable**.

## Status (repo snapshot — 2026-05-02)

**Shipped:** Versioned policy in [`configs/skill_gate.yaml`](../../configs/skill_gate.yaml) (~~audit T2.5~~); eval-suite loader integration described in PLAN-UNIFIED-SKILL-GATE §4; lifecycle enums + proposal types wired for lane-(a) (`BankMutationProposal`, shadow semantics).
**Open:** Mechanical drift between YAML and any remaining hard-coded `GateThresholds` fallbacks — grep `orchestrator/config.py` when changing thresholds.

| Document | Purpose |
|----------|---------|
| [`PLAN-UNIFIED-SKILL-GATE.md`](PLAN-UNIFIED-SKILL-GATE.md) | `SkillStatus` state machine (`draft → candidate → shadow → provisional → active`, plus `deprecated / rejected / rolled_back`); `SkillSourceType`; canonical record types (`SkillRecord`, `SkillEvaluationRecord`, `GateVerdict`); ownership split (`SkillLifecycleManager` / `GateRunner` / `PromotionOrchestrator`); storage split (`draft_store / candidate_store / active_store / archive_store`); shared gate stack `static → replay → shadow → transfer → non-regression`. All skill sources — mined, crafted, repaired, transferred, teacher-proposed, human-seeded — pass the same stack. |

Back to [plans/README.md](../README.md).
