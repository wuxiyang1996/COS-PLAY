# PLAN: Unified Skill Gate — Lifecycle, Promotion, and Rollback

**Scope:** Define the **single shared skill lifecycle** so that every new skill proposal — whether **mined** from traces, **crafted** by composition, **repaired** from failures, **transferred** across domains, **teacher-proposed** (frozen 32B/72B), or **human-seeded** — must follow the same path before any of it can affect the active actor:

```
draft → candidate → shadow → provisional → active
draft → candidate → rejected
active → deprecated / rolled_back
```

**Problem statement.** Today the lifecycle is described in three places — [PLAN-SKILL-BANK.md §7 acceptance gate](../03-skill-bank/PLAN-SKILL-BANK.md#7-grpo-co-evolution), [PLAN-HARNESS.md §10 promotion gates](../05-harness/PLAN-HARNESS.md#10-promotion-gates), and [PLAN-PIPELINE-ORCHESTRATOR.md §3 promotion / rollback rules](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3-promotion--rollback-rules) — but no document concretely defines (i) the canonical state machine, (ii) the canonical record types every component reads/writes, (iii) the ownership boundary between Skill Bank Agent / Harness / Pipeline Orchestrator, (iv) the storage split that makes "no promotion without gate" mechanically impossible to bypass, or (v) the phased build order. This plan is the canonical specification of all five.

**Upstream:** [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) (skill data model, `evidence_role`, `evidence_interface`, lineage / provenance, negative knowledge), [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) (`SkillEpisode`, `SkillHarness`, `AdapterRegistry`, `TransferManager`, `ReplayValidator`, six promotion gates G0–G5, transfer-failure diagnostics), [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) (artifact / log schema, episode-local trajectory bookkeeping, evaluation matrix, budget controller, escalation ladder).

**Downstream:** Concrete `gate/`, `skill_bank/`, `harness/`, and `orchestrator/` module trees; configs (`configs/skill_gate.yaml`); JSONL artifact schemas backing `BankMutationProposal` / `GateVerdict` / `SkillEpisode` / `AuditRecord`.

**Architectural sentence to paste into the repo.**

> The Skill Bank Agent owns the skill lifecycle and promotion policy, but the **skill gate is implemented as a shared protocol** across the Skill Bank, Harness, and Pipeline Orchestrator. All skill proposals — mined, crafted, repaired, transferred, or teacher-proposed — must pass the same multi-stage validation path: **static checks → replay validation → shadow execution → transfer validation → non-regression checks** before promotion into the active bank. **No module can write to the active bank without passing this gate.**

---

## 1. Ownership model

The gate is one shared protocol with three owners. Each row below pins which module *executes* a stage and which module *records the verdict*. No other module may write the indicated artifact directly.

| Owner | Owns (writes) | Does NOT own |
|-------|---------------|--------------|
| **Skill Bank Agent** ([PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md)) | skill state machine, versioning, provenance, candidate registration, promotion *recommendation*, final bank writes for `candidate / shadow / provisional / active / deprecated / rejected / rolled_back` | runtime validation execution; promotion *transactions* across multiple stores |
| **Harness** ([PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md)) | static validation execution, replay validation, shadow execution, transfer validation, non-regression evaluation, gate-metrics emission (`SkillEpisode`, `GateVerdict` payloads) | bank pointer moves; snapshot creation; rollback transactions |
| **Pipeline Orchestrator** ([PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)) | batch scheduling, snapshot/checkpointing, **promotion + rollback transactions**, frozen eval-suite execution, audit trail, decision logging | per-skill semantic decisions; per-episode validation logic |

This split is consistent with what each plan already says is its scope. The unified gate just makes the boundary executable rather than a convention.

---

## 2. Canonical lifecycle

### 2.1 Skill states (`SkillStatus`)

```python
class SkillStatus(str, Enum):
    DRAFT       = "draft"        # raw proposal (mined / crafted / repaired / transferred / seeded)
    CANDIDATE   = "candidate"    # passed Stage 0 (static / contract / schema)
    SHADOW      = "shadow"       # passed Stage 1 (replay); allowed only via SkillHarness.run_shadow
    PROVISIONAL = "provisional"  # passed Stages 2–4 (shadow + transfer + non-regression); limited active use
    ACTIVE      = "active"       # promoted to the live bank; default-retrievable in run_active
    DEPRECATED  = "deprecated"   # superseded by a newer version, retained for rollback
    REJECTED    = "rejected"     # failed any gate stage; stays addressable for crafter learning
    ROLLED_BACK = "rolled_back"  # was active, was reverted; restoration target preserved in rollback_links
```

### 2.2 Skill source types (`SkillSourceType`)

```python
class SkillSourceType(str, Enum):
    MINED            = "mined_from_trace"
    CRAFTED          = "crafted_by_composition"
    REPAIRED         = "repaired_from_failure"
    TRANSFERRED      = "transferred_from_other_domain"        # legacy generic transfer
    FEW_SHOT_ADAPTED = "few_shot_adapted_from_source"         # PLAN-SKILL-BANK §0.4
    TEACHER          = "teacher_proposed"                     # frozen 32B/72B output
    SEEDED           = "human_seeded"
```

`FEW_SHOT_ADAPTED` is the source type emitted by the Crafter's
`GeneralizeProposal` whenever the proposal carries an explicit
`(source_domain, target_domain, k_shot_budget, slot_remap)` recipe
(see [PLAN-SKILL-CRAFTER.md §5.2](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) and `data_structure/extensions/bank_mutation_proposal.py::GeneralizeProposal`). The legacy `TRANSFERRED` value is retained so historical proposals still round-trip; new proposals that bind a game-foundry skill to a transfer-target adapter must use `FEW_SHOT_ADAPTED`.

**Hard rule.** All source types share *the same* gate; there is no fast path based on model size, lineage, or human authorship. This makes the "frozen 32B/72B proposals stay candidates until they pass the same gate stack" rule from [PLAN-PIPELINE-ORCHESTRATOR.md §3.4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#34-asymmetric-teacher-outputs) mechanical rather than aspirational.

### 2.3 Allowed transitions

```
                         ┌── reject ─────────────────────────────────────────┐
                         │                                                   ▼
DRAFT ── static ──► CANDIDATE ── replay ──► SHADOW ── transfer + ─► PROVISIONAL ── promote ─► ACTIVE
                         │                     │      non-regression                │            │
                         │                     └── borderline (more shadow trials) ─┘            │
                         │                                                                       │
                         └── (skipped: teacher / mined still must enter via DRAFT) ──────────────┘
                                                                                                 │
                       ┌─ rollback_to(prev_active_version) ─ ROLLED_BACK ◄────── ACTIVE ─────────┘
                       └─ deprecate(reason)               ─ DEPRECATED ◄────── ACTIVE
```

**Invariants (mechanically enforced).**

1. No skill is retrievable in `SkillHarness.run_active` unless its current `SkillStatus ∈ {ACTIVE, PROVISIONAL}`.
2. No skill is retrievable in `SkillHarness.run_shadow` unless its current `SkillStatus ∈ {ACTIVE, PROVISIONAL, SHADOW, CANDIDATE}`.
3. `DRAFT` skills are **not** in any retrieval index (they live in `draft_store`, §6).
4. `REJECTED` and `ROLLED_BACK` skills retain their full `SkillEvaluationRecord` history for the [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) repair / reflection loop.
5. Every state transition records both a `SkillEvaluationRecord` (the per-stage verdict, §3.2) and an `AuditRecord` ([PLAN-PIPELINE-ORCHESTRATOR.md §8.3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#83-audit-artifact)).

---

## 3. Canonical data structures

These three records are the **only** objects that cross the Skill-Bank ↔ Harness ↔ Orchestrator boundary.

### 3.1 `SkillRecord` (owned by Skill Bank)

The bank-side canonical object. Concretizes the typed-slot / domain-adapter / evidence-driven items already required by [PLAN-SKILL-BANK.md §4](../03-skill-bank/PLAN-SKILL-BANK.md#4-skill-data-model) and [§0.3](../03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills).

```python
@dataclass
class SkillRecord:
    skill_id: str
    version: int
    name: str
    summary: str
    status: SkillStatus
    source_type: SkillSourceType

    # Cross-domain ontology (PLAN-SKILL-BANK §3.2 / §4.3)
    applicable_domains: list[str]            # subset of {gymv, browser, osworld, video, visual_reasoning}
    verified_domains: list[str]              # populated by gate stage 3 (transfer)

    # Typed program (PLAN-SKILL-BANK §0.5 / §4.1)
    input_slots: dict[str, str]              # slot_name -> type
    preconditions: list[dict]
    procedure: list[dict]                    # inner-hop chain over shared primitives
    success_criteria: list[dict]
    abort_criteria: list[dict]

    # Evidence-driven invariant (PLAN-SKILL-BANK §0.3)
    evidence_role: str                       # GATHER | VERIFY | REASON | COMMIT
    evidence_interface: dict                 # required evidence kinds in / out
    evidence_requirements: list[dict]

    # Outcome contract (PLAN-SKILL-BANK §4.2)
    contract: dict                           # eff_add / eff_del / belief effects
    failure_modes: list[str]                 # negative knowledge, populated post-rejection

    # Lineage (PLAN-SKILL-BANK §4.3a)
    provenance: dict                         # parent_skill_ids, source_episode_ids, teacher_model, crafter_op

    metrics: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
```

### 3.2 `SkillEvaluationRecord` (owned by Harness, consumed by Orchestrator + Skill Bank)

The shared per-evaluation artifact, written once per gate-stage execution. Sits under `artifacts/gates/{gate_run_id}/` in [PLAN-PIPELINE-ORCHESTRATOR.md §2.3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#23-storage-layout-logical).

```python
@dataclass
class SkillEvaluationRecord:
    skill_id: str
    version: int
    source_type: str
    status_before: str                       # SkillStatus
    status_after: str                        # SkillStatus

    # Per-stage payloads (any may be empty if the stage was skipped)
    static_check:        dict
    replay_metrics:      dict
    shadow_metrics:      dict
    transfer_metrics:    dict
    non_regression_metrics: dict

    final_decision:   str                    # PASS | FAIL | LIMITED_PASS
    decision_reason:  str
    diagnostic_labels: list[str]             # from PLAN-HARNESS §10a
    approved_domains: list[str]
    rejected_domains: list[str]

    rollback_target:  str | None             # prior version to restore, if applicable
    bank_snapshot_id: str                    # snapshot the eval ran against
    eval_suite_id:    str                    # frozen eval slice identifier
    adapter_versions: dict                   # per-domain adapter version pins
    ontology_version: str

    artifacts: list[str]                     # paths to mismatch traces, slot-binding errors, regression deltas
    evaluated_at: str
```

### 3.3 `GateVerdict` (already named in [PLAN-PIPELINE-ORCHESTRATOR.md §2.2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#22-required-record-types) and [PLAN-HARNESS.md §10a](../05-harness/PLAN-HARNESS.md#10a-transfer-failure-diagnostics-domain-specific))

Concretized as:

```python
class GateVerdict(str, Enum):
    PASS          = "pass"
    FAIL          = "fail"
    LIMITED_PASS  = "limited_pass"           # e.g., approved for source domain only

@dataclass
class GateVerdictPayload:
    stage:   str                             # static | replay | shadow | transfer | non_regression
    verdict: GateVerdict
    metrics: dict
    diagnostic_labels: list[str]             # PLAN-HARNESS §10a labels
    failing_checks: list[str]
    artifacts: list[str]
```

`SkillEvaluationRecord` is the **roll-up** across the per-stage `GateVerdictPayload`s for one full evaluation pass.

---

## 4. Module / file layout

The unified gate adds three module trees and lightly extends the existing two.

```
gate/                              # NEW — owned conceptually by the Harness, called by the Orchestrator
  __init__.py
  gate_types.py                    # SkillStatus, SkillSourceType, GateVerdict
  gate_record.py                   # SkillEvaluationRecord, GateVerdictPayload
  static_checker.py                # Stage 0
  replay_gate.py                   # Stage 1 (wraps harness/replay_validator.py)
  shadow_gate.py                   # Stage 2 (wraps harness/skill_harness.run_shadow)
  transfer_gate.py                 # Stage 3 (wraps harness/transfer_manager.py)
  non_regression_gate.py           # Stage 4
  gate_policy.py                   # thresholds; loads configs/skill_gate.yaml
  promotion_manager.py             # Stage 5 (provisional → active two-step)

skill_bank/                        # NEW concrete sub-tree under skill_agents/
  skill_record.py                  # SkillRecord
  skill_registry.py                # store-aware index over draft / candidate / active stores
  skill_lifecycle_manager.py       # SkillLifecycleManager (§5.1)
  skill_versioning.py              # version_history, parent/child links

harness/                           # extends PLAN-HARNESS §12
  skill_episode.py                 # already planned
  skill_harness.py                 # already planned
  replay_validator.py              # already planned
  transfer_manager.py              # already planned
  gate_runner.py                   # NEW — single entry point the Orchestrator calls (§5.2)

orchestrator/                      # NEW concrete sub-tree
  promotion_orchestrator.py        # PromotionOrchestrator (§5.3)
  rollback_manager.py              # Stage 6 (rollback transactions)
  eval_suite.py                    # frozen eval slice loader
  snapshot_manager.py              # bank snapshot create / pointer move

configs/
  skill_gate.yaml                  # all thresholds in one file (§9)
```

This mirrors the file responsibilities already declared in [PLAN-HARNESS.md §12.1](../05-harness/PLAN-HARNESS.md#121-file-responsibilities) and the implementation checklist in [PLAN-PIPELINE-ORCHESTRATOR.md §9](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#9-implementation-checklist-cursor-ready).

---

## 5. APIs

### 5.1 Skill Bank Agent — `SkillLifecycleManager`

The **only** API allowed to mutate skill status or write into any skill store.

```python
class SkillLifecycleManager:
    def register_draft(self, skill: SkillRecord) -> str: ...
    def mark_candidate(self,    skill_id: str, eval_record: SkillEvaluationRecord) -> None: ...
    def mark_shadow(self,       skill_id: str, eval_record: SkillEvaluationRecord) -> None: ...
    def mark_provisional(self,  skill_id: str, eval_record: SkillEvaluationRecord) -> None: ...
    def promote_active(self,    skill_id: str, eval_record: SkillEvaluationRecord) -> None: ...
    def reject(self,            skill_id: str, eval_record: SkillEvaluationRecord) -> None: ...
    def deprecate(self,         skill_id: str, reason: str) -> None: ...
    def rollback(self,          skill_id: str, target_version: int, reason: str) -> None: ...
```

**Contract.** Every mutation requires a backing `SkillEvaluationRecord` (except `register_draft`, which records a `SourceProposalRecord`, and `deprecate`, which records an `AuditRecord`). `SkillCrafter`, `SkillHarness`, and `PromotionOrchestrator` all call this — no module bypasses it.

### 5.2 Harness — `GateRunner`

The single entry point that runs gate stages and emits `SkillEvaluationRecord`. Wraps the existing `SkillHarness`, `ReplayValidator`, `TransferManager`, `AdapterRegistry`.

```python
class GateRunner:
    def run_static_check(self, skill: SkillRecord) -> GateVerdictPayload: ...
    def run_replay(self,       skill: SkillRecord, datasets: list[str]) -> GateVerdictPayload: ...
    def run_shadow(self,       skill: SkillRecord, rollout_batch: list[dict]) -> GateVerdictPayload: ...
    def run_transfer(self,     skill: SkillRecord, target_domains: list[str]) -> GateVerdictPayload: ...
    def run_non_regression(self, skill: SkillRecord, eval_suite: dict) -> GateVerdictPayload: ...

    def assemble_evaluation(
        self, skill: SkillRecord, payloads: list[GateVerdictPayload]
    ) -> SkillEvaluationRecord: ...
```

`run_shadow` / `run_transfer` / `run_non_regression` MUST set `SkillEpisode.shadow = True` for every episode they generate (consistent with [PLAN-HARNESS.md §6.1](../05-harness/PLAN-HARNESS.md#61-phase-a--shadow-mode)) so the shadow constraints are enforced by the Harness, not by the gate code.

### 5.3 Orchestrator — `PromotionOrchestrator`

Owns the *transactions*: schedules evaluation batches, calls `GateRunner`, asks `SkillLifecycleManager` to apply state transitions, creates / moves bank snapshots, and writes the `AuditRecord`.

```python
class PromotionOrchestrator:
    def evaluate_candidate(self, skill_id: str)             -> SkillEvaluationRecord: ...
    def promote_if_passed(self, skill_id: str)              -> bool: ...
    def rollback_if_needed(self, skill_id: str)             -> bool: ...
    def batch_evaluate_candidates(
        self, candidate_ids: list[str]
    ) -> list[SkillEvaluationRecord]: ...
```

`promote_if_passed` is a **transaction**: snapshot create → `SkillLifecycleManager.promote_active` → `current_production` pointer move → `AuditRecord` write. Failure at any step rolls the whole transaction back; the bank pointer never moves on partial success.

---

## 6. Storage split

This is the structural reason the "no promotion without gate" invariant cannot be bypassed: a `DRAFT` skill is *physically not in the same store* as anything `run_active` looks at.

| Store | Holds | Visible to |
|-------|-------|------------|
| `draft_store`     | `DRAFT`                                 | `SkillCrafter`, gate Stage 0 |
| `candidate_store` | `CANDIDATE`, `SHADOW`, `PROVISIONAL`    | gate Stages 1–4, `SkillHarness.run_shadow`, `run_active` (PROVISIONAL only, downweighted) |
| `active_store`    | `ACTIVE`                                | `SkillHarness.run_active` (default rank), all online retrieval |
| `archive_store`   | `DEPRECATED`, `REJECTED`, `ROLLED_BACK` | rollback target lookup, crafter repair input |

Plus three index tables co-located with the bank snapshots from [PLAN-PIPELINE-ORCHESTRATOR.md §2.3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#23-storage-layout-logical):

- `version_history(skill_id, version) → SkillRecord`
- `gate_history(skill_id, version) → list[SkillEvaluationRecord]`
- `rollback_links(skill_id, from_version, to_version, reason, audit_id)`

**`SkillHarness.run_active` retrieval policy.** Sees `active_store` at full weight + `candidate_store(PROVISIONAL)` with the [PLAN-HARNESS.md §7.2](../05-harness/PLAN-HARNESS.md#72-shadow-origin-penalty) shadow-origin penalty applied. Never sees `CANDIDATE`, `SHADOW`, `DRAFT`, `REJECTED`, `DEPRECATED`, or `ROLLED_BACK`.

**`SkillHarness.run_shadow` retrieval policy.** Sees `active_store` + `candidate_store` (CANDIDATE / SHADOW / PROVISIONAL) at full rank.

---

## 7. Gate stages (the canonical pipeline)

This stack is the unified specification of what [PLAN-SKILL-BANK.md §7](../03-skill-bank/PLAN-SKILL-BANK.md#7-grpo-co-evolution), [PLAN-HARNESS.md §10](../05-harness/PLAN-HARNESS.md#10-promotion-gates), and [PLAN-PIPELINE-ORCHESTRATOR.md §3.1](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#31-gate-stages-ordered) already each describe in part. Existing **G0 (Evidence-driven contract)** from [PLAN-HARNESS.md §10](../05-harness/PLAN-HARNESS.md#10-promotion-gates) is **orthogonal** to this stack — it is checked at every `SkillHarness.finalize_episode` (active and shadow) and can demote at any time. The stages below run *as a batch* on candidates and produce one `SkillEvaluationRecord` per pass.

### Stage 0 — Static sanity check (`gate/static_checker.py`)

| Item | Value |
|------|-------|
| **Inputs** | `SkillRecord`, schema spec, slot type spec, domain registry |
| **Checks** | required fields present; no unresolved slots; preconditions / success_criteria / abort_criteria / procedure non-empty; `evidence_requirements` declared; `contract` schema valid; `applicable_domains` valid; `evidence_role` consistent with §8 effect family ([PLAN-SKILL-BANK.md §0.3 Clause B](../03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills)); no environment-specific hardcoding in semantic body |
| **Maps to existing gate** | Subsumes [PLAN-PIPELINE-ORCHESTRATOR.md §3.1.1 + §3.1.2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#31-gate-stages-ordered) (static contract + symbolic consistency) and [PLAN-HARNESS.md G1 binding feasibility](../05-harness/PLAN-HARNESS.md#10-promotion-gates) at the schema layer |
| **Transition** | `pass → CANDIDATE`; `fail → REJECTED` |

### Stage 1 — Offline replay validation (`gate/replay_gate.py` → `harness/replay_validator.py`)

| Item | Value |
|------|-------|
| **Inputs** | `SkillRecord`, held-out trajectories, candidate slot bindings, expected contract effects |
| **Metrics** | precondition precision / recall; slot binding success; completion rate; success-criteria match; abort correctness; contract consistency; unsupported-step ratio; avg hop count; avg cost |
| **Maps to existing gate** | [PLAN-HARNESS.md G3 Replay](../05-harness/PLAN-HARNESS.md#10-promotion-gates) + [PLAN-PIPELINE-ORCHESTRATOR.md §3.1.3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#31-gate-stages-ordered) |
| **Transition** | `strong pass → SHADOW (all applicable_domains)`; `weak pass → SHADOW (subset)`; `fail → REJECTED` |

### Stage 2 — Shadow execution (`gate/shadow_gate.py` → `SkillHarness.run_shadow`)

| Item | Value |
|------|-------|
| **Behavior** | Runs candidate in shadow only: retrieved, slot-bound, adapter-attached, scored against observed transitions. **MUST NOT** affect environment, training reward, or actor action — these are the constraints already pinned in [PLAN-HARNESS.md §6.1](../05-harness/PLAN-HARNESS.md#61-phase-a--shadow-mode) |
| **Metrics** | shadow invocation appropriateness; slot binding stability; severe stall rate; contradiction rate; evidence grounding quality; contract progress reliability; shadow pass rate |
| **Maps to existing gate** | [PLAN-HARNESS.md G4 Shadow](../05-harness/PLAN-HARNESS.md#10-promotion-gates) |
| **Transition** | `pass → Stage 3`; `fail → REJECTED`; `borderline → remain SHADOW with more trials` |

### Stage 3a — Few-shot transfer validation (`gate/transfer_gate.py` → `harness/few_shot_adapter.py`)

This is the asymmetric realisation of the old "Stage 3 — Transfer validation". The skill being evaluated has, by construction, a game-foundry lineage (`source_domains ⊆ SOURCE_DOMAINS`); Stage 3a verifies that the *same protocol* binds to the declared transfer-target adapters using only a handful of target-domain demonstrations. This is what earns each entry in `SkillRecord.verified_domains`.

| Item | Value |
|------|-------|
| **Goal** | Verify that the skill is not merely a game-foundry trick — for each declared `target_domain`, prove the abstract protocol binds to the target adapter under a *small* demo budget (PLAN-SKILL-BANK §0.4). |
| **Inputs** | source-domain `SkillRecord` (with `source_domains`, `transfer_target_domains`); target-domain adapters from the `AdapterRegistry`; an optional `few_shot_demos: Dict[target_domain, Sequence[FewShotDemo]]` mapping; the [`FewShotConfig`](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) thresholds. |
| **K-shot adaptation protocol** | For each `target_domain ∈ skill.transfer_target_domains`: (i) take up to `k_shot_default` demonstrations (capped at `k_shot_max`); (ii) re-tag each demo's `<state>.domain` to the target; (iii) apply the proposal's `slot_remap`; (iv) invoke `harness.SkillHarness.run_skill(...)` through the target adapter; (v) score each shot via `success_fn` (default: `outcome.success ∧ outcome.contract_satisfied`). The adapter aborts the budget if cumulative `cost_tokens > adaptation_cost_max_tokens`. |
| **Per-target output** | `AdaptResult { target_domain, k_used, pass_rate, n_success, n_total, aborted, cost_tokens, cost_ms, diagnostic_label, episode_ids }`. Diagnostics: `target_domain_demo_unavailable`, `few_shot_budget_exceeded`, `adaptation_overfitting`. |
| **Per-target verdict** | `PASS` iff `n_total > 0 ∧ pass_rate ≥ target_domain_pass_rate_min`; otherwise the target *does not earn* a `verified_domains` entry on this run. |
| **Stage verdict** | `PASS` if `# verified targets ≥ transfer_min_target_domains_verified`; `LIMITED_PASS` if ≥1 verified but below threshold; `FAIL` otherwise. |
| **Side-effect on the bank** | `GateService` *produces* the verified-target list (carried in the `SkillEvaluationRecord` and exposed via `GateVerdictPayload.eligible_domains ∩ TRANSFER_TARGET_DOMAINS`). The actual mutation of `SkillRecord.verified_domains` and `SkillRecord.adapter_history` is performed by `SkillLifecycleManager.record_transfer_verification(...)`, called from `PromotionOrchestrator.promote(...)` *before* the status transition so the ACTIVE invariant (PLAN-SKILL-BANK §0.4) sees the updated list. The lifecycle manager is the only sanctioned writer of either field; no other component (Crafter, Harness, Bank query path) may mutate them. On `FAIL`, no verification is recorded; diagnostic labels propagate through the gate verdict only. |
| **Diagnostic labels** | populates [PLAN-HARNESS.md §10a](../05-harness/PLAN-HARNESS.md#10a-transfer-failure-diagnostics-domain-specific) labels, plus the three new few-shot-specific labels above. |
| **Transition** | `strong pass → Stage 4 (all verified target domains)`; `partial pass → PROVISIONAL for verified target domains only (LIMITED_PASS)`; `fail → REJECTED or back to DRAFT for repair`. |
| **Backward compatibility** | If a candidate carries no source/target metadata (legacy proposals from before the asymmetry), `GateService._run_transfer` falls back to the older "≥ `transfer_min_domains` feasible domains" check; the new path takes precedence as soon as `source_domains` is populated. |

### Stage 4 — Non-regression (`gate/non_regression_gate.py`)

| Item | Value |
|------|-------|
| **Goal** | Ensure newly admitted skill does not damage existing system |
| **Frozen eval suite** | source-domain core tasks, target-domain transfer tasks, retrieval hit@k, slot-binding success, shadow transfer pass rate, promotion rate, overall task reward, unsupported reasoning rate, avg hop cost — composed by `orchestrator/eval_suite.py`, drawn from [PLAN-PIPELINE-ORCHESTRATOR.md §6](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#6-evaluation-matrix) |
| **Maps to existing gate** | [PLAN-HARNESS.md G5 Non-regression](../05-harness/PLAN-HARNESS.md#10-promotion-gates) + [PLAN-PIPELINE-ORCHESTRATOR.md §3.1.4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#31-gate-stages-ordered) |
| **Transition** | `pass → PROVISIONAL`; `fail → REJECTED, plus rollback any existing PROVISIONAL of this skill_id` |

### Stage 5 — Promotion (`gate/promotion_manager.py`)

Two-step activation:

1. **PROVISIONAL** — limited applicable_domains, limited invocation budget, increased logging, shadow-origin penalty applied in `run_active` ranking.
2. **ACTIVE** — full promotion after stable online stats over a configurable window (default: ≥ N successful active episodes with no Gate G0 violations and no non-regression alerts).

The promotion *transaction* is owned by `PromotionOrchestrator.promote_if_passed` (§5.3). Snapshot pointer moves only on transaction success.

### Stage 6 — Rollback / deprecation (`orchestrator/rollback_manager.py`)

| Trigger | Action |
|---------|--------|
| Non-regression later fails | `ACTIVE → ROLLED_BACK`; restore `prev_active_version`; quarantine offending skills |
| Severe instability in production | same as above; raise to L2 in [PLAN-PIPELINE-ORCHESTRATOR.md §8.1](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#81-escalation-ladder) |
| Conflict with better version | `ACTIVE → DEPRECATED` (newer version wins) |
| Transfer harms source-domain performance | `ACTIVE → ROLLED_BACK` for the offending domain only (`LIMITED_PASS` → drop one domain); skill stays active where verified |
| Repeated G0 violations in production | `ACTIVE → DEPRECATED`; route episodes to crafter `evidence_starved` cluster ([PLAN-SKILL-CRAFTER.md §6.2](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)) |

All rollbacks go through `PromotionOrchestrator.rollback_if_needed` and emit an `AuditRecord` ([PLAN-PIPELINE-ORCHESTRATOR.md §8.3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#83-audit-artifact)) tagged with the trigger reason.

---

## 8. Integration with existing modules

### 8.1 Skill Crafter

`SkillCrafter` **never** writes to the active bank. Its only path is:

```python
skill_lifecycle_manager.register_draft(crafted_skill)   # status = DRAFT
```

This applies uniformly to mined / crafted / repaired / transferred / **teacher** / seeded — see §2.2 hard rule. Frozen 32B/72B output is admitted via the same `register_draft` call as a mined skill.

### 8.2 Action Agent (and `SkillHarness.run_active`)

Per §6 storage split:

| Status | Visible in `run_active` | Visible in `run_shadow` |
|--------|-------------------------|-------------------------|
| ACTIVE        | ✅ default rank         | ✅ |
| PROVISIONAL   | ✅ shadow-origin penalty | ✅ |
| SHADOW        | ❌                      | ✅ |
| CANDIDATE     | ❌                      | ✅ |
| DRAFT         | ❌                      | ❌ |
| REJECTED / DEPRECATED / ROLLED_BACK | ❌ | ❌ |

The `skill_select` head in [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) sees only what `SkillHarness.run_active` returns; it has no other channel to the bank.

### 8.3 Skill Bank Agent

`SkillBankAgent` ([PLAN-SKILL-BANK.md §12](../03-skill-bank/PLAN-SKILL-BANK.md#12-core-components)) gains one new subcomponent (`SkillLifecycleManager`) and no longer holds runtime validation logic — that belongs to the Harness. The bank's existing `TransferManager` (§12) is **renamed** to `LegacyTransferProposer` and proposes `TransferProposal` records that flow through `register_draft`; the runtime transfer validation lives in [PLAN-HARNESS.md §5.4](../05-harness/PLAN-HARNESS.md#54-transfermanager).

### 8.4 Harness

`SkillHarness` keeps its existing surface. Adds `gate_runner.py` as a thin orchestration layer the Orchestrator calls. The existing six-gate set (G0–G5) in [PLAN-HARNESS.md §10](../05-harness/PLAN-HARNESS.md#10-promotion-gates) remains the per-episode contract; the unified gate composes those gates into the *batch lifecycle* described in §7.

### 8.5 Pipeline Orchestrator

The Orchestrator's centralized acceptance-gate role from [PLAN-PIPELINE-ORCHESTRATOR.md §3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3-promotion--rollback-rules) is now executed via `PromotionOrchestrator`. The Orchestrator keeps end-to-end DAG control, snapshot management, frozen eval-suite execution, and audit logging.

---

## 9. Threshold policy

All thresholds live in **one** file: `configs/skill_gate.yaml`. Initial conservative defaults:

```yaml
static:
  required_fields_present: true        # binary — no soft pass

replay:
  contract_consistency_min: 0.80
  slot_binding_success_min: 0.80
  unsupported_step_ratio_max: 0.15

shadow:
  pass_rate_min: 0.75
  severe_instability_max: 0.05
  contradiction_rate_max: 0.10

transfer:
  # Asymmetric, few-shot-driven (PLAN-SKILL-BANK §0.4).
  min_target_domains_verified: 1       # minimum |verified_domains ∩ TRANSFER_TARGET_DOMAINS|
  per_domain_pass_rate_min: 0.65       # lower than source-domain initially
  degradation_ratio_max: 0.20
  ontology_remap_consistency_min: 0.75

few_shot:                              # per-(skill, target_domain) adaptation budget
  k_shot_default: 5                    # default shots per target
  k_shot_max: 16                       # cap; enforced by FewShotAdapter
  target_domain_pass_rate_min: 0.50    # minimum pass_rate to verify a target
  adaptation_cost_max_tokens: 8000     # abort the few-shot run if exceeded

non_regression:
  source_drop_max: 0.02                # ε from PLAN-PIPELINE-ORCHESTRATOR §3.1.4
  retrieval_hit_at_k_drop_max: 0.05

promotion:
  provisional_active_window_episodes: 200
  provisional_min_pass_rate: 0.70
  shadow_origin_penalty: 0.30
```

Do not overfit thresholds at this stage — exposing them centrally is the goal so the [PLAN-PIPELINE-ORCHESTRATOR.md §8.2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#82-human-audit-points) audit point ("change to acceptance thresholds") fires consistently when they move.

---

## 10. Logging and audit

Every gate run emits the fields below to `artifacts/gates/{gate_run_id}/verdict.json` ([PLAN-PIPELINE-ORCHESTRATOR.md §2.3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#23-storage-layout-logical)).

**Required fields.** `skill_id`, `source_type`, `version`, `stage`, `verdict`, `metrics`, `artifacts`, `dataset/task slice`, `adapter_version` (per domain), `ontology_version`, `bank_snapshot_id`, `eval_suite_id`, `timestamp`.

**Required artifacts.** Replay mismatch traces; shadow failure traces; slot-binding errors; transfer remap reports; regression deltas vs prior `snapshot_id`; full `SkillEpisode` set used for the evaluation.

This subsumes [PLAN-HARNESS.md §10a.1](../05-harness/PLAN-HARNESS.md#10a1-consumers) (diagnostic-label tallying) and [PLAN-PIPELINE-ORCHESTRATOR.md §8.3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#83-audit-artifact) (signed `AuditRecord` for human decisions).

---

## 11. Phased implementation plan

| Phase | Goal | New / changed code | Exit criterion |
|-------|------|--------------------|----------------|
| **1 — Infrastructure skeleton** | The state machine, records, and dummy gate exist | `gate/gate_types.py`, `gate/gate_record.py`, `skill_bank/skill_record.py`, `skill_bank/skill_lifecycle_manager.py`, `harness/gate_runner.py` (stubs), `orchestrator/promotion_orchestrator.py` (stubs) | A `DRAFT` skill can be registered and walked through `DRAFT → CANDIDATE → SHADOW → PROVISIONAL → ACTIVE` with dummy verdicts; storage split (§6) enforced |
| **2 — Static + replay** | Mined skill can pass / fail Stages 0–1 | `gate/static_checker.py`, `gate/replay_gate.py`, held-out trajectory loader, contract-consistency scoring | Mined skill from a real trace produces a real `SkillEvaluationRecord` and is correctly routed to `CANDIDATE` or `REJECTED` |
| **3 — Shadow execution** | Crafted / transferred skills can run in shadow without affecting active behavior | `gate/shadow_gate.py`, candidate retrieval in `run_shadow`, shadow metrics aggregation, no-environment-effect guarantee | Transferred skill runs N shadow episodes; produces `SkillEpisode.shadow=True` records; cannot influence reward or actor action |
| **4 — Transfer gate** | Adapter validation + per-domain approval | `gate/transfer_gate.py`, `harness/adapter_registry.py` validation hook, ontology mapping validation, per-domain approval results | A skill is approved for one target domain (e.g., `video`) but rejected for another (e.g., `osworld`) and recorded as `LIMITED_PASS` |
| **5 — Non-regression + promotion** | Provisional → active only after frozen eval-suite passes | `gate/non_regression_gate.py`, `orchestrator/eval_suite.py`, `orchestrator/snapshot_manager.py`, `orchestrator/rollback_manager.py`, `gate/promotion_manager.py` | A `PROVISIONAL` skill becomes `ACTIVE` after passing source-domain non-regression; a triggered rollback restores `prev_active_version` atomically |
| **6 — Crafter / failure loop** | Rejection reasons re-enter the crafter | `SkillCrafter` consumes `SkillEvaluationRecord` + diagnostic labels; failure-cluster export; repair proposal path; candidate re-submission | A `REJECTED` skill is repaired by the crafter and re-enters `register_draft`; the new draft carries a `provenance.repaired_from` link |

Phase 1 is the immediate implementation target. Phase 2 directly closes the **P0 acceptance-gate item** in [PLAN-SKILL-BANK.md §14 TODO](../03-skill-bank/PLAN-SKILL-BANK.md#14-todo).

---

## 12. What not to do

- **Do not** let `SkillCrafter` (or any other module) write to `active_store` directly. Only `PromotionOrchestrator` may move pointers; only `SkillLifecycleManager` may write to any skill store.
- **Do not** let transferred skills enter `active_store` without passing Stages 2–4. The two-phase shadow → active protocol from [PLAN-HARNESS.md §6](../05-harness/PLAN-HARNESS.md#6-two-phase-transfer-protocol) is non-negotiable.
- **Do not** push runtime replay code into `SkillBankAgent`. Replay execution belongs in the Harness (`harness/replay_validator.py`).
- **Do not** let `ActionAgent` retrieve raw `CANDIDATE` or `DRAFT` skills under `run_active`. The retrieval policy in §6 is the single source of truth.
- **Do not** keep cross-episode memory interfaces in this repo. The orchestrator's only state-keeping surface is the [§4 episode-local trajectory](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping); no episodic / semantic / state memory subsystem exists or should be added.
- **Do not** use model size as a promotion shortcut. Frozen 32B/72B teacher proposals enter as `DRAFT` and must clear the same gate stack — already pinned in [PLAN-PIPELINE-ORCHESTRATOR.md §3.4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#34-asymmetric-teacher-outputs).

---

## 13. Cross-plan edits (companion to this plan)

The following companion edits land in the existing plan files so they all reference this canonical specification:

1. **[PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md)** — new subsection "Unified skill lifecycle and promotion ownership" (in §7); `§14 TODO` replaces the vague "Acceptance gate pipeline" row with concrete sub-items (`SkillStatus` enum, `SkillRecord`, `SkillLifecycleManager`, `GatePolicy`, store split).
2. **[PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md)** — new section "Gate Execution Runtime" (§10b) that names `GateRunner` and lists per-stage entry points.
3. **[PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)** — new section "Promotion transaction and rollback protocol" (§3a) that names `PromotionOrchestrator`, the snapshot / pointer-move sequence, and the audit-logging contract.
4. **[README.md](../README.md)** — adds this plan to the plan-documents table.

These edits are **link-only**: every cross-plan section points back to this file rather than restating the lifecycle, records, or APIs.

---

## 14. Related documents

| Document | Relationship |
|----------|--------------|
| [README](../README.md) | Pipeline overview; lists this plan |
| [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) | Skill data model + lifecycle owner |
| [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) | Per-episode gate G0–G5; gate-runtime executor |
| [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) | Promotion / rollback transactions; snapshot management; audit |
| [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) | Producer of `DRAFT` skills (mined / crafted / repaired / transferred); consumer of `REJECTED` records |
| [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) | Retrieves only what `SkillHarness.run_active` returns |
| [PLAN-EDITS-HARNESS-CONTROL-PLANE.md](../legacy/10-edits/PLAN-EDITS-HARNESS-CONTROL-PLANE.md) | Predecessor edit pass; this plan extends its "control plane" framing with the explicit lifecycle |
