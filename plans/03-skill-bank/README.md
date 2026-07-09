# 03-skill-bank — Stage 3: Skill Bank

Cross-task bank of **transferable** reasoning, grounding, and control skills written as **general protocols feasible across all five target domains** (game, webagent, os-agent, video-understanding, visual reasoning). Stores skill objects, retrieves top-k candidates, manages lifecycle states.

## Status (repo snapshot — 2026-05-02)

**Shipped:** Split storage + `SkillLifecycleManager`; lane **(a)** — skills as retrieval context ([`implementation_notes/legacy/skill-lane-decision.md`](../../implementation_notes/legacy/skill-lane-decision.md)); offline promotion driver executed (**375 / 489** rows writeback-eligible per [`IMPLEMENTATION-STATUS.md`](../../IMPLEMENTATION-STATUS.md) §S1).
**Open:** Fast-loop GRPO + continued catalogue hygiene; cross-domain executors remain mostly stubs outside `gymv` — see audit §4.

| Document | Purpose |
|----------|---------|
| [`PLAN-SKILL-BANK.md`](PLAN-SKILL-BANK.md) | Skill as structured-state program (§0.5), the **general-protocol invariant** (§0.1), the **evidence-driven invariant** (§0.3), shared inner primitives + adapter-based binding (§1.5), unified structured state interface with entity ontology (§3), typed slots + domain adapters in data model (§4), 5-stage discovery pipeline, effect families + 3-layer hierarchy (§8), 6 transferable skill families (§9), asymmetric GRPO co-evolution with acceptance gates (§7), query/select API (§6). |

Cross-cutting consumers: [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) (typed proposals), [Harness](../05-harness/PLAN-HARNESS.md) (retrieval / binding / G0 evidence check), [Unified Skill Gate](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) (lifecycle owner).

Back to [plans/README.md](../README.md).
