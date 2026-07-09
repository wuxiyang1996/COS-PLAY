# 08-cross-cutting — Cross-cutting contracts

Contracts that touch every stage and every component but are not themselves modules. They define how failure signals, uncertainty signals, and runtime extension records flow across the system.

## Status (repo snapshot — 2026-05-02)

**Shipped:** Extension-record scaffolding under `data_structure/extensions/`; versioned [`configs/failure_routing.yaml`](../../configs/failure_routing.yaml) (lane-(a) failure taxonomy + `policy_version`).
**Open:** Full **R0–R5** failure-router implementation + typed `FailureRoutingRecord` parity — audit §4 (`T3.2`); uncertainty calibration mostly spec-level.

| Document | Purpose |
|----------|---------|
| [`PLAN-FAILURE-ROUTING.md`](PLAN-FAILURE-ROUTING.md) | Single canonical policy that converts every observed failure (Harness diagnostics, grounding verdicts, judge `F1`–`F7`, budget events, human-audit triggers) into a typed [`FailureRoutingRecord`](PLAN-EXPERIENCE-EXTENSION.md#d-failureroutingrecord--making-failures-governable) with one downstream owner. |
| [`PLAN-UNCERTAINTY-CALIBRATION.md`](PLAN-UNCERTAINTY-CALIBRATION.md) | Cross-cutting uncertainty contract: scopes (field / entity / state / evidence / answer), sources (parser / validator / tool / cross-view / missing-field), routing thresholds (Path A / B / C), calibration (reliability curves and ECE per slice). |
| [`PLAN-EXPERIENCE-EXTENSION.md`](PLAN-EXPERIENCE-EXTENSION.md) | Thin **extension layer** above `data_structure/` that adds system-contract records (`SkillEpisode`, `GateVerdict`, `SkillRecord`, `FailureRoutingRecord`, run / release metadata) as append-only side-tables. **Hard non-goal:** no memory subsystem, no cross-episode mutable state. |

Back to [plans/README.md](../README.md).
