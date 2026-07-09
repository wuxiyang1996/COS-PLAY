# implementation_notes/legacy — finished design memos

This folder holds the implementation-notes memos that have **already been
delivered**: each one captures a design decision plus a snapshot of the
code that landed against it. They are kept here as design provenance so
the rationale can still be cited from the active plans, but they are not
the place to look for current state.

## What lives here

| Doc | Status | What shipped |
|---|---|---|
| [`single-vs-two-mdp-tradeoff.md`](single-vs-two-mdp-tradeoff.md) | ✅ DONE | Single-MDP / harness-driven design. `decision_agents/inner_mdp.py` deleted; `_run_inner_mdp` removed from `actor_agent.py`; only two GRPO LoRAs (`skill_selection` + `action_taking`). T3.6 closed. |
| [`skill-lane-decision.md`](skill-lane-decision.md) | ✅ DONE | Lane (a) — skills as retrieval payloads. Default-on via `SkillCrafterService(enable_protocol_patching=False)`; lane-(a) banners on the four PLAN docs + `harness/README.md` §22 + `crafter-harness-orchestrator-roles.md` §7.3 / §7.5. T1.3a / T1.3e / T1.3f closed. |
| [`vllm-topology.md`](vllm-topology.md) | ✅ DONE | S2 governance for T2.8. Default option (ii) — `schema_gen` is offline-only on `Qwen3.5-35B-A3B`; trainer `VLLMServerManager` runs a single `Qwen/Qwen3.5-9B` base across `vllm_gpu_ids`. Wired in `trainer/coevolution/config.py`. |
| [`protocol-lift-design.md`](protocol-lift-design.md) | ✅ DONE | Implementation lives at [`labeling/_protocol_lift.py`](../../labeling/_protocol_lift.py) (`lift_protocol_to_typed_hops`, `classify_prose_step`, `mine_effects`, `LiftStats`, `build_schema_index_for_game`) wired into [`labeling/_decorate_skill_records.py`](../../labeling/_decorate_skill_records.py). Tests at [`tests/test_protocol_lift.py`](../../tests/test_protocol_lift.py). |
| [`crafter-harness-orchestrator-roles.md`](crafter-harness-orchestrator-roles.md) | ✅ DONE | Three-role split shipped — live `crafter/service.py`, `harness/gate_runner.py`, `orchestrator/promotion_orchestrator.py`. Three offline `labeling_supplement/*_gpt54.py` mirrors landed (`decide_skill_crafting_gpt54.py`, `dump_harness_io_gpt54.py`, `decide_promotion_gpt54.py`). |
| [`harness-usability-and-intra-gymv-transfer.md`](harness-usability-and-intra-gymv-transfer.md) | ✅ DONE | `skill_transfer_test/` scaffolding + `_phase4_transfer_cycle.py` + gymv real executor + protocol lift all shipped. T1.2 ran end-to-end 2026-05-02 — 375 / 489 cold-start records flipped to `ACTIVE / PROVISIONAL / SHADOW` across 17 pair banks. The §17 keystone has fired. |
| [`phase5-cross-domain-measurement.md`](phase5-cross-domain-measurement.md) | ✅ DONE | All six measurement stages (S0 audits → S6 NxN matrix + report) shipped 2026-05-02. Per-target executors run as deterministic stubs; real-env binding tracked in [`../cross-domain-transfer-suite-rollout.md`](../cross-domain-transfer-suite-rollout.md). |

## What is *not* here (lives one level up in `implementation_notes/`)

* [`pre-training-readiness-audit.md`](../pre-training-readiness-audit.md)
  — rolling sprint ledger (S0–S4 open / closed items).
* [`cross-domain-transfer-suite-rollout.md`](../cross-domain-transfer-suite-rollout.md)
  — measurement infra DONE, but per-target executors are deterministic
  stubs and real-env binding is still planned.

## Reading rules

> **Where this folder and any active plan or audit ledger disagree, the
> active plan / ledger wins.** These memos record the design *as it was
> when the decision shipped*; subsequent code changes are tracked in
> [`../pre-training-readiness-audit.md`](../pre-training-readiness-audit.md)
> and the per-folder `plans/0X-…/README.md` status sections.
