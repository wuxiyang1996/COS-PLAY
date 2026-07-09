# 09-implementation — Cursor-ready build sheet

Implementation-ordering plan. Does not duplicate canonical specs — links into the [Harness](../05-harness/PLAN-HARNESS.md), [Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md), [Pipeline Orchestrator](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md), and [Unified Skill Gate](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) plans for the *why* and *what*; this folder owns the *how* and *when*.

## Status (repo snapshot — 2026-05-02)

**Ground truth:** [`IMPLEMENTATION-STATUS.md`](../../IMPLEMENTATION-STATUS.md) — Phases A–C MVP + Day 7–10 trainer hooks + threshold YAMLs + SFT corpus + offline promotion **shipped**; **six-way SFT re-run** after [`trainer/SFT/lora_targets.py`](../../trainer/SFT/lora_targets.py) (T2.11) then **T1.1′** probe still gates trusting `schema_gen`. PLAN prose: lane-(a) banners on major PLANs; Phase F GRPO target list should stay aligned with **two** actor LoRAs only (`skill_selection`, `action_taking`).

| Document | Purpose |
|----------|---------|
| [`PLAN-COMPONENTS-IMPLEMENTATION.md`](PLAN-COMPONENTS-IMPLEMENTATION.md) | Pinned target repo layout (`src/harness/ src/crafter/ src/orchestrator/ src/skill_bank/ src/common/`); strict phase order **A (Harness MVP) → B (Orchestrator MVP) → C (Crafter MVP) → D (Transfer + Replay) → E (Eval + dashboards) → F (optional trainable extensions)**; per-phase acceptance criteria; architectural boundaries that must hold across phases (Harness ≠ Crafter ≠ Orchestrator); paste-ready Cursor prompt encoding all required invariants. |

Back to [plans/README.md](../README.md).
