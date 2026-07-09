# 06-orchestrator — Component: Pipeline Orchestrator (control plane)

The **macro DAG** that runs grounding → reasoning → skill retrieval → action → verification → promotion / rollback across many episodes and many runs. Owns artifact / log schema, episode-local trajectory bookkeeping, training cadence, evaluation matrix, budget controller, escalation ladder. Composes with the [Harness](../05-harness/PLAN-HARNESS.md) (which is its per-invocation micro-runtime).

## Status (repo snapshot — 2026-05-02)

**Shipped:** Orchestrator MVP (`runner`, `promotion_orchestrator`, `gate_service`, snapshots, budget); scoreboard / eval-suite wiring aligned with North-Star §7.3 where implemented — see [`IMPLEMENTATION-STATUS.md`](../../IMPLEMENTATION-STATUS.md).
**Outstanding training gate:** launch **fast-loop GRPO on `gymv` only** ([`PLAN-PIPELINE-ORCHESTRATOR.md`](PLAN-PIPELINE-ORCHESTRATOR.md) §5.0); thresholds also live in [`configs/skill_gate.yaml`](../../configs/skill_gate.yaml).

| Document | Purpose |
|----------|---------|
| [`PLAN-PIPELINE-ORCHESTRATOR.md`](PLAN-PIPELINE-ORCHESTRATOR.md) | Hot-path / warm-path DAGs, **four-way Actor / Harness / Bank / Orchestrator boundary** (§0a), required record types (§2.2), promotion / rollback transactions (§3a), **episode-local evidence & trace bookkeeping** (§4), evaluation matrix (§6), budget controller, observability, escalation ladder L0–L3. |

Cross-cutting consumers: every component plan; the [System North-Star](../00-system/PLAN-SYSTEM-NORTHSTAR.md) (the orchestrator emits the canonical scoreboard); [Failure Routing](../08-cross-cutting/PLAN-FAILURE-ROUTING.md) (orchestrator applies routing rules).

Back to [plans/README.md](../README.md).
