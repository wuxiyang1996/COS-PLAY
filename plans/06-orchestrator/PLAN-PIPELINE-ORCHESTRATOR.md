# PLAN: Pipeline Orchestrator (End-to-End Harness)

**Scope:** Define the **single top-level runner** that closes the loop across [Visual Grounding](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md), [Action Agent](../02-action-agent/PLAN-ACTION-AGENT.md), [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md), and [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md). The orchestrator is the **domain-general control plane** for grounding → reasoning → skill retrieval → action → verification → promotion → rollback, with **episode-local evidence & trace bookkeeping** plus budgets, artifacts, gates, and full-system evaluation.

**Scope boundaries (deliberate).** The orchestrator is domain-general across game / webagent / os-agent / video-understanding / visual reasoning. Its **first evaluation track** is **short-video evidence-grounded reasoning** (Video-Holmes-style). The orchestrator's only state-keeping surface is the episode-local trajectory described in §4 — current `<state>`, the short typed hop trace, intermediate belief state, and within-episode evidence references. Long-horizon video and any cross-episode storage layer are out of scope and have no APIs in this orchestrator.

**Problem statement:** Sub-plans already specify module orchestrators (e.g., bank maintenance, grounding evaluation harness). What is missing is one **executable DAG** that repeatedly: collects rollouts → grounds → runs inner-hop reasoning → acts → logs traces → updates the bank → runs the crafter → **verifies** → promotes or rolls back → schedules training → re-evaluates — with explicit **acceptance gates**, **budget control**, and **observability**.

**Upstream:** All component plans; shared `<state>` schema ([README § Canonical `<state>`](../README.md)); two-level MDP framing ([`LONG_HORIZON_REASONING.md`](../../LONG_HORIZON_REASONING.md)).

**Downstream:** Implementations of runners, job queues, artifact stores, CI-style verification, and monitoring dashboards.

**Non-goals:** Replacing optional [Visual Skills](../01-visual-grounding/PLAN-VISUAL-SKILLS.md); duplicating milestone-level grounding SFT detail ([PLAN-VISUAL-GROUNDING-MILESTONES](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md)).

---

## 0a. Actor–Harness–Skill Bank–Orchestrator Boundary

The system follows a **four-way separation of responsibilities**. The orchestrator's job is to make this separation enforceable end-to-end; the boundaries themselves are stated here so other modules can reference a single canonical version.

### 0a.1 Four-way separation

| Module | Responsibilities |
|--------|------------------|
| **Skill Bank** ([PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md)) | stores skill objects (with `source_domains` / `transfer_target_domains` / `verified_domains`, see [§0.4](../03-skill-bank/PLAN-SKILL-BANK.md#04-source-domain--transfer-target-asymmetry)); retrieves top-k candidates; manages lifecycle states; receives promotion / rollback results. **Never** writes `verified_domains` itself — that field is owned by the gate. |
| **Harness** ([PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md)) | filters retrieved candidates; validates binding and evidence; runs the [`FewShotAdapter`](../05-harness/PLAN-HARNESS.md#542-fewshotadapter-stage-3a-runtime) for Stage 3a target-binding probes; provides advisory scores; performs veto when necessary. Frozen 72B; **never** the online policy. |
| **Actor** ([PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md)) | remains the online policy; decides continue / switch / no-skill / reasoning / action; consumes only Harness-filtered eligible candidates; produces online trajectories and experience (in the source domain — game — by default; target-domain inference happens through the slow loop, see [§5](#5-training-cadence-by-timescale)). |
| **Orchestrator** (this document) | manages batch evaluation; runs promotion / rollback; schedules validation; **schedules the source / target asymmetric cadence ([§5](#5-training-cadence-by-timescale))**; maintains snapshots and experiments; owns the audit trail; is the only writer of `SkillRecord.verified_domains` (via `GateService`'s Stage 3a, see [PLAN-UNIFIED-SKILL-GATE Stage 3a](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)). |

This separation prevents any single module from becoming overloaded or semantically ambiguous. In particular, the orchestrator does **not** reach inside the Actor's policy choices, the Harness does **not** mutate the bank, and the Bank does **not** decide which candidate the Actor should run.

### 0a.2 Online control path

The online path is:

1. structured `<state>` is produced by [Visual Grounding](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md);
2. **Skill Bank** retrieves top-k skill candidates (see [PLAN-SKILL-BANK.md §6](../03-skill-bank/PLAN-SKILL-BANK.md));
3. **Harness** filters them into `eligible_skills` (see [PLAN-HARNESS.md §1a.3](../05-harness/PLAN-HARNESS.md#1a3-harness-output-contract));
4. **Actor** makes the final policy decision:
   - continue current skill,
   - switch to an eligible skill,
   - no skill,
   - reasoning step,
   - primitive action;
5. **Harness** performs invocation-time validation if a skill is proposed; on veto the Actor falls back ([PLAN-HARNESS.md §1a.4](../05-harness/PLAN-HARNESS.md#1a4-actor-proposal-harness-veto));
6. execution proceeds; the chosen action is emitted to the environment;
7. traces are logged for replay, shadow evaluation, mining, and future promotion decisions (§2 artifact schema).

This preserves the central role of the Actor while fully using the frozen Harness as a high-capacity verifier.

### 0a.3 Architectural principle

> **A strong frozen Harness does not replace the trainable Actor. Instead, it constrains and improves the Actor's action space by filtering and validating candidate skills at runtime.**

This principle should guide both implementation and evaluation. Any orchestrator change that would route final policy choice through the Harness violates this principle and must be rejected.

### 0a.4 Promotion / runtime separation

Runtime skill filtering must be separated from promotion logic:

- **Runtime filtering** belongs to the [Harness](../05-harness/PLAN-HARNESS.md).
- **Final online choice** belongs to the [Actor](../02-action-agent/PLAN-ACTION-AGENT.md).
- **Promotion, rollback, and bank mutation** belong to the orchestrator (§3, §3a) and to Skill Bank lifecycle logic ([PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)).

This separation is important to avoid mixing:

- online policy,
- runtime validation,
- offline governance.

### 0a.5 Evaluation implication

The evaluation protocol (§6) must measure not only whether the Harness improves runtime safety, but also whether **the Actor itself becomes stronger**.

Reported metrics should therefore separately analyze:

- Actor decision quality,
- Harness filtering / veto quality,
- overall system performance,
- skill-use efficiency,
- reasoning-step usefulness,
- transfer robustness.

This ensures the project does not collapse into "the big frozen model did the hard part."

### 0a.6 Online skill invocation interface

The §0a.2 path is realized by the following reference interface. Each line maps to one of the four modules; nothing else is allowed to make the call on the Actor's behalf.

```python
retrieved_skills = skill_bank.retrieve(
    schema_state, intention, top_k=K
)

eligible_skills = harness.filter_and_score(
    schema_state=schema_state,
    intention=intention,
    retrieved_skills=retrieved_skills,
    active_skill=active_skill,
    local_reasoning_trace=local_reasoning_trace,
)

actor_decision = actor.step(
    schema_state=schema_state,
    intention=intention,
    eligible_skills=eligible_skills,
    active_skill=active_skill,
    valid_actions=valid_actions,
    local_reasoning_trace=local_reasoning_trace,
)

if actor_decision.proposed_skill is not None:
    veto = harness.validate_invocation(
        actor_decision.proposed_skill, schema_state, ...
    )
    if veto:
        actor_decision = actor.fallback(...)
```

Design summary:

- **Skill Bank** retrieves,
- **Harness** filters and vetoes,
- **Actor** decides,
- **Orchestrator** governs offline lifecycle.

This is the canonical control pattern that the orchestrator must preserve at every stage of the rollout DAG (§1).

---

## 1. Rollout DAG

The orchestrator is a **directed acyclic graph of stages** with explicit inputs/outputs. Stages may run **inline** (same process), **queued** (async workers), or **scheduled** (cron / Airflow-style), but the **data dependencies** are fixed.

### 1.1 Online episode subgraph (hot path)

Every environment step (or batched chunk for video):

```
observe (pixels / DOM / API)
  → ground          # Visual Grounding: pixels → <state>
  → inner_mdp       # Action Agent: GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE
  → act             # environment action from valid set
  → log_step        # append to EpisodeTrace (see §2)
```

**Contract:** `inner_mdp` may recurse (re-ground on uncertainty) only through the **budget controller** (§7); the orchestrator records each sub-call as a child span.

### 1.2 End-of-episode subgraph

```
finalize_episode
  → segment_traces     # Skill Bank: boundaries on outer + inner traces
  → ingest_bank        # provisional skill updates / experience buffer
  → emit_signals       # success, failure class, stall, contract violations
```

### 1.3 Offline evolution subgraph (warm path)

Runs between episodes or on a timer; **never blocks** the hot path unless configured for shadow mode only.

```
select_batch
  → crafter_propose    # Skill Crafter: compose / hypothesize / patch
  → acceptance_gate    # §3 — contract checks, replay, non-regression
  → promote_or_rollback
  → schedule_train     # GRPO / LoRA jobs by timescale (§5)
  → re_evaluate        # §6 — full-system matrix
```

### 1.4 DAG invariants

| Invariant | Meaning |
|-----------|---------|
| **No bank promotion without gate** | Crafter outputs are **candidates** until acceptance completes. |
| **Replay before promote** | Any skill or protocol change that affects `skill_select` / `hop_select` must pass **held-out replay** on a frozen trace slice. |
| **Idempotent artifacts** | Every stage writes versioned artifacts (§2); re-runs do not corrupt prior versions. |
| **Explicit failure edges** | Gate failure → rollback or quarantine; never silent partial promotion. |

### 1.5 Optional: Visual Skills

If enabled, insert `visual_skill_retrieve` **after** `ground` and **before** `inner_mdp`, subject to the same budgets and logging. Default: **off** for pipeline-closure work.

---

## 2. Artifact / log schema

The harness is only as debuggable as its **unified telemetry**. Treat logs as **append-only event streams** with stable IDs.

### 2.1 Core identifiers

| Field | Description |
|-------|-------------|
| `run_id` | Orchestrator run (training or eval job). |
| `episode_id` | One task instance from reset to terminal. |
| `step_id` | Monotonic within episode. |
| `span_id` | Inner-hop or tool sub-call (tree under `step_id`). |
| `skill_id` / `skill_version` | Bank pointer at time of use. |
| `schema_hash` | Hash of `<state>` schema version + adapter. |

### 2.2 Required record types

1. **`EpisodeMeta`** — domain, task, goal, seed, model adapters, budget snapshot at episode start.
2. **`GroundingRecord`** — raw routing (Path A/B/C if applicable), latency, escalation reason, optional tool traces; **canonical `evidence_out`** for `GATHER`-role skills (see [PLAN-VISUAL-GROUNDING.md §3a](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md#3a-shared-output-schema-both-domains-emit-this)): carries `evidence_id`, `source`, `kind`, `anchor`, `confidence`, `verified_by`.
3. **`InnerHopRecord`** — sequence of inner actions; slot coverage flags; uncertainty scores; re-ground triggers.
4. **`ActionRecord`** — chosen env action, parse path, valid-set constraints, **`evidence_warrant: List[EvidenceRef]`** (non-empty for any committed env action or final answer, per [PLAN-ACTION-AGENT.md §5.3-bis](../02-action-agent/PLAN-ACTION-AGENT.md)).
5. **`RewardRecord`** — `r_env`, `r_follow`, `r_cost`, and components.
6. **`SkillEpisode`** — per-skill-invocation record from the Harness (see [PLAN-HARNESS.md §5.1](../05-harness/PLAN-HARNESS.md#51-skillepisode)); carries `evidence_role`, `evidence_in`, `evidence_out`, `evidence_warrant`, `verify_verdict`, `reason_warrant`, `contract_progress`, `outcome`, and transfer-diagnostic labels. This is the record Gate G0 operates on.
7. **`BankMutationProposal`** — segmented spans, contract deltas, merge/split ops — **staged**, not live until gate. Typed subclasses: `PatchProposal | ComposeProposal | TransferProposal | RetireProposal` (see [PLAN-SKILL-CRAFTER.md §2.5](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)); every subclass declares `evidence_role` and `evidence_interface`.
8. **`GateVerdict`** — pass/fail, failing checks list, replay diffs, non-regression metrics; carries zero or more typed diagnostic labels including `opaque_skill_violation`, `evidence_interface_mismatch`, `skill_role_mismatch` (see [PLAN-HARNESS.md §10a](../05-harness/PLAN-HARNESS.md#10a-transfer-failure-diagnostics-domain-specific)).
9. **`TrainJobSpec`** — which LoRA/timescale, data snapshot IDs, seed, cluster target.

### 2.3 Storage layout (logical)

```
artifacts/
  runs/{run_id}/
    episodes/{episode_id}/
      trace.jsonl          # stream of step + span records
      summary.json         # aggregates for eval
  bank/
    snapshots/{snapshot_id}/
    proposals/{proposal_id}/   # pre-promotion
  gates/
    {gate_run_id}/verdict.json
  train/
    {job_id}/spec.json
```

Physical backend (object store, DB, lakehouse) is an implementation choice; the **schema** is normative.

---

## 3. Promotion / rollback rules

The **acceptance gate** is the safety spine of self-evolution. Both Skill Bank and Skill Crafter depend on it; the orchestrator **centralizes** policy here.

### 3.1 Gate stages (ordered)

1. **Static contract check** — schemas, typed slots, effect families, adapter bindings; no execution.
2. **Symbolic consistency** — preconditions/effects compose without contradiction (bounded solver or LLM-judge + checks).
3. **Replay verification** — replay against **frozen traces** with deterministic settings; compare action distributions / outcomes within tolerance.
4. **Non-regression filter** — compare against prior `snapshot_id` on a **fixed eval slice** (§6); reject if regression beyond ε.
5. **Canary** (optional) — small live traffic or shadow scoring before full promotion.

### 3.2 Promotion

Promotion **creates a new bank snapshot**; pointers (`current_production`) move only after gate success. Previous snapshot remains addressable for rollback.

### 3.3 Rollback

| Trigger | Action |
|---------|--------|
| Gate failure | Discard proposal; optionally file **failure artifact** for crafter learning. |
| Post-promotion regression on the **source-domain** (game) frozen slice | Revert pointer to last good `snapshot_id`; quarantine offending skills. The source-domain regression is treated as a system-level rollback because it indicates the foundry's hardening signal has degraded. |
| Post-promotion regression on **one transfer-target** domain only | **Partial deprecation**: drop the offending entry from `SkillRecord.verified_domains`, append the diagnostic to `adapter_history` and (if the failure pattern recurs) to `false_binding_patterns` (PLAN-SKILL-BANK §4.3a/b). The skill *stays ACTIVE* if at least one other verified target remains; otherwise the skill falls back to PROVISIONAL pending a fresh Stage 3a run. The source-domain (game) lineage is never revoked by a target-domain regression. |
| Data corruption / schema drift | Halt scheduled training; require manual audit (§8). |

### 3.4 Asymmetric teacher outputs

Frozen 32B/72B proposals remain **candidates** until they pass the same gate stack as mined skills — no fast path based on model size.

---

## 3a. Promotion transaction and rollback protocol

The gate stages in §3.1 are the *what*. The **promotion transaction** is the *how* the orchestrator turns a passing `SkillEvaluationRecord` into an actual change in `current_production`, and the *rollback transaction* is the symmetric undo. Both run through one component — `PromotionOrchestrator` (`orchestrator/promotion_orchestrator.py`) — so that bank pointers never move on partial success.

This section is the canonical specification of those two transactions. The full lifecycle, ownership boundary, record types, and storage split are in the [Unified Skill Gate](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) plan; this section pins the orchestrator-side responsibilities and the audit-logging contract.

### 3a.1 `PromotionOrchestrator` (`orchestrator/promotion_orchestrator.py`)

```python
class PromotionOrchestrator:
    def evaluate_candidate(self, skill_id: str)         -> SkillEvaluationRecord: ...
    def promote_if_passed(self, skill_id: str)          -> bool: ...
    def rollback_if_needed(self, skill_id: str)         -> bool: ...
    def batch_evaluate_candidates(
        self, candidate_ids: list[str]
    ) -> list[SkillEvaluationRecord]: ...
```

`evaluate_candidate` is the only path that calls into `harness/gate_runner.py` ([PLAN-HARNESS.md §10b](../05-harness/PLAN-HARNESS.md#10b-gate-execution-runtime)) for a candidate. `promote_if_passed` and `rollback_if_needed` wrap the pointer-move + status-mutation as a single transaction (§3a.3 / §3a.4).

### 3a.2 Batch evaluation schedule

Owned by the Orchestrator, fed by the offline evolution subgraph (§1.3):

- **Trigger.** Every N episodes *or* when `select_batch` produces ≥ K candidates *or* on the slow timescale (§5.3).
- **Input.** A list of `(skill_id, version)` pairs in `candidate_store` whose latest `SkillEvaluationRecord` is older than the current `bank_snapshot_id`.
- **Eval suite.** `orchestrator/eval_suite.py` resolves the frozen evaluation slice from the §6 evaluation matrix, pinned by `eval_suite_id`.
- **Snapshot.** `orchestrator/snapshot_manager.py` records `bank_snapshot_id` before evaluation begins so every per-stage `GateVerdictPayload` can be replayed against a stable bank state.
- **Adapter / ontology pin.** Each evaluation pins `adapter_versions` and `ontology_version` so verdicts are reproducible when adapters or the cross-domain ontology drift.
- **Fairness.** Honors the §7.2 fairness-across-domains budget so rare target domains are not starved of evaluation slots.

### 3a.3 Promotion transaction (the *commit* path)

`promote_if_passed(skill_id)` runs the following ordered steps. **Failure at any step rolls the entire transaction back**; `current_production` does not move on partial success.

1. **Read** the latest `SkillEvaluationRecord` for `skill_id`. Refuse to proceed unless `final_decision ∈ {PASS, LIMITED_PASS}` and `status_after ∈ {PROVISIONAL, ACTIVE}`.
2. **Snapshot create.** `snapshot_manager.create(parent=current_production)` produces a new `bank_snapshot_id` containing the proposed change.
3. **Apply state transition.** Call `SkillLifecycleManager.mark_provisional` or `promote_active` ([PLAN-UNIFIED-SKILL-GATE.md §5.1](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md#51-skill-bank-agent--skilllifecyclemanager)). The bank stores receive the write under the new snapshot only.
4. **Pointer move.** Atomically advance `current_production` to the new `bank_snapshot_id`. The previous snapshot remains addressable.
5. **Audit.** Emit a signed `AuditRecord` (§8.3) that links `skill_id`, `version`, `from_snapshot`, `to_snapshot`, `eval_suite_id`, `final_decision`, `decision_reason`, and the operator (human or automated trigger).
6. **Notify.** Push the verdict to the gate dashboard (§9 item 6) and to the [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) failure-cluster export if `final_decision == LIMITED_PASS`.

If steps 2–4 fail, `snapshot_manager.discard(new_snapshot_id)` runs and `SkillLifecycleManager` reverts the in-memory status to `status_before`. No write to `active_store` happens outside this transaction.

### 3a.4 Rollback transaction (the *revert* path)

`rollback_if_needed(skill_id)` is triggered by:

| Trigger | Source |
|---------|--------|
| Post-promotion non-regression failure | `re_evaluate` (§1.3) detects regression beyond ε against the prior `snapshot_id` |
| Sustained Gate G0 violations in production | `SkillHarness.finalize_episode` aggregator ([PLAN-HARNESS.md §10b.2](../05-harness/PLAN-HARNESS.md#10b2-per-stage-delegation-table)) |
| Repeated transfer-domain failures | `re_evaluate` per-domain slice |
| Human escalation L2+ | §8.1 escalation ladder |

Steps:

1. **Read** the current `SkillRecord` and the `rollback_target` from the most recent passing `SkillEvaluationRecord` (or, if none, the prior `ACTIVE` version from `version_history`).
2. **Snapshot create.** `snapshot_manager.create(parent=current_production)` for the revert state.
3. **Apply state transition.** `SkillLifecycleManager.rollback(skill_id, target_version, reason)` flips the current version to `ROLLED_BACK` (or `DEPRECATED` for non-emergency supersession) and restores the target version to `ACTIVE`. The `rollback_links` index is updated.
4. **Pointer move.** Atomically advance `current_production`.
5. **Quarantine.** Add the rolled-back `(skill_id, version)` to a quarantine list so the [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) can pick it up for repair without it being re-proposed unchanged.
6. **Audit.** Emit `AuditRecord` with the full trigger chain.

Rollback is **always** atomic: either `current_production` references the post-rollback state and the audit record is written, or nothing changes.

### 3a.5 Audit-logging contract

Every promotion / rollback / deprecation produces *both* a `SkillEvaluationRecord` (for evaluation evidence) and an `AuditRecord` (for the human-readable decision trail). The contract:

| Field | Source |
|-------|--------|
| `who` | `human_operator_id` for L2+ decisions; `"automated:promotion_orchestrator"` otherwise |
| `when` | UTC timestamp |
| `rationale` | For automated decisions: the verdict + threshold breach summary. For human decisions: free text mandatory above L1 |
| `linked_snapshot_ids` | `from_snapshot`, `to_snapshot` |
| `linked_eval_records` | `SkillEvaluationRecord` IDs that justified the action |
| `triggered_by` | enum: `gate_pass`, `gate_fail`, `non_regression_fail`, `g0_violation_aggregate`, `human_escalation`, `crafter_supersession` |

Stored alongside `gates/{gate_run_id}/verdict.json` per the §2.3 storage layout.

### 3a.6 What stays out of the orchestrator

- **Per-skill semantic decisions** (status mutation rules, evidence-role checks, contract validity): owned by `SkillLifecycleManager` and `SkillHarness.finalize_episode` respectively.
- **Stage execution** (static / replay / shadow / transfer / non-regression): owned by `harness/gate_runner.py`.
- **Threshold *interpretation***: owned by `gate/gate_policy.py` reading `configs/skill_gate.yaml`.

The orchestrator's lane is purely *transaction control + scheduling + audit*.

---

## 4. Episode-local evidence & trace bookkeeping

The orchestrator maintains an **episode-local trajectory** as its only state-keeping surface: current structured `<state>`, a short typed hop trace, an intermediate belief state, and within-episode evidence references. Everything resets at the episode boundary; the only durable artifacts are skill-bank snapshots (§3) and the append-only logs in §2. There is no cross-episode storage layer in this orchestrator.

The bookkeeping is richer than a plain "log all tool calls" contract because of the **evidence-driven invariant** ([PLAN-SKILL-BANK.md §0.3](../03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills)): every skill is required to either consume or produce addressable evidence. If `<state>.evidence_refs` and the claim–evidence links below are not preserved episode-wide and rolled up into `SkillEpisode` records, Gate G0 cannot be evaluated, and skills would silently drift away from actually assisting reasoning. The bookkeeping below is the substrate that lets the Harness enforce "ALL skills are evidence-driven" mechanically rather than by convention.

### 4.1 What the orchestrator maintains per episode

| Slice | Contents | Lifetime |
|-------|----------|----------|
| **Current structured state** | Latest `<state>` from grounding (entities, attributes, relations, targets, uncertainty) | Replaced every outer step |
| **Short typed hop trace** | Ordered `InnerHopRecord` + `ActionRecord` from this episode | Cleared at episode boundary |
| **Intermediate belief state** | Per-entity confidence, accumulated `CHECK` verdicts, pending `REASON` warrants | Episode-scoped |
| **Within-episode evidence references** | Domain-appropriate pointers: clip/frame IDs for video, DOM-node + screenshot region IDs for web, desktop object + window IDs for os, tool-call IDs for visual reasoning | Episode-scoped |
| **Claim–evidence links** | `(claim_id → evidence_ref[])` produced by `CHECK` / `COMMIT` inner actions | Episode-scoped |
| **Transfer diagnostics** | For any reused skill: `(skill_id, source_domain, target_domain, binding_verdict, replay_pass)` | Episode-scoped, rolled up into §6a metrics |

### 4.2 Where each inner action reads and writes

| Inner action | Reads from | Writes to |
|--------------|------------|-----------|
| `GROUND` | Current `<state>`, raw observation, grounding tools | Current `<state>` (entities/attributes/relations), evidence references |
| `CHECK` | Current `<state>`, evidence references, intermediate belief state | Intermediate belief state (verdict), claim–evidence links |
| `RETRIEVE` | Active `bank_snapshot_id` (skill bank, read-only) | Hop trace (selected skill_id), intermediate belief state (active skill) |
| `COMMIT` | Intermediate belief state, evidence references | Claim–evidence links, hop trace |
| `EXECUTE` | Current `<state>`, intermediate belief state, evidence references | `ActionRecord` (with `evidence_warrant`), environment |

`RETRIEVE` only fetches skills from the bank — it carries no episode-spanning lookup channel. Anything the agent wants to cite in a later hop must already exist as a current-context evidence reference produced by grounding or by a tool call earlier in the same episode.

### 4.3 Grounding alignment

When grounding revises entities mid-episode (schema drift), the orchestrator:

1. Records a `schema_revision_notice` on the next `GroundingRecord` (see [PLAN-VISUAL-GROUNDING.md §`GroundingRecord`]).
2. Invalidates claim–evidence links whose entity anchors no longer resolve.
3. Re-issues `GROUND` for any affected slot before the next `COMMIT` / `EXECUTE`.

Re-issuing `GROUND` rebuilds the affected slice of current context directly; nothing else needs to be re-anchored because the trajectory above is the only state-keeping surface.

---

## 5. Training cadence by timescale

Map jobs to the three-agent separation ([README § Three-agent role split](../README.md)) with explicit **triggers** and **data dependencies**. The cadence is asymmetric in the same way the bank is (PLAN-SKILL-BANK §0.4): the **fast loop runs in the source domain (game)**, where rollouts are cheap and verification is dense; the **slow loop runs the few-shot transfer machinery against target domains**, where each rollout is expensive and the only thing being asked is "does this game-learned protocol bind here?"

### 5.0 Source / target asymmetry of the cadence

- **Fast loop = game rollouts only.** The Actor's GRPO updates and skill mining feed off `gymv` rollouts. This is where new candidate skills are *born* and where the bulk of training compute goes.
- **Medium loop = mining + single-domain replay validation.** Bank-ops jobs operate over the source-domain trace pool only.
- **Slow loop = few-shot transfer.** Target-domain rollouts (`browser`, `osworld`, `video`, `visual_reasoning`) are scheduled explicitly through the few-shot adapter (PLAN-UNIFIED-SKILL-GATE Stage 3a, [PLAN-HARNESS §5.4.2](../05-harness/PLAN-HARNESS.md)) and are budgeted in *demos per skill per target* rather than continuous rollouts.
- **Non-regression is measured on the source-domain frozen slice** (gate G5, see [PLAN-HARNESS §10](../05-harness/PLAN-HARNESS.md#10-promotion-gates)). Target-domain regressions trigger only the partial-deprecation path in §3.3.

### 5.1 Fast (Actor — every iteration / continuous, **source domain only**)

- **Targets:** `hop_select`, `skill_select`, action execution adapters tied to GRPO.
- **Inputs:** Fresh `gymv` trajectories from `EpisodeTrace`, reward streams.
- **Trigger:** Buffer full, KL stable, or wall-clock micro-batch.
- **Blocking:** Must not wait on slow teacher; uses last **promoted** bank snapshot.

### 5.2 Medium (Skill Bank ops — every few iterations, **source domain only**)

- **Targets:** segmentation, contract heads, curator policy.
- **Inputs:** Segmented `gymv` traces + gate-verified labels where available.
- **Trigger:** N new episodes or drift in segmentation loss proxy.

### 5.3 Slow (Synthesis / reflection + **few-shot transfer to target domains** — batched)

- **Targets:** composition, hypothesis, counterfactuals, protocol patches (frozen teacher), and **per-(skill, target_domain) `FewShotAdapter.adapt()` runs** that earn `verified_domains` entries.
- **Inputs:** Failure clusters, gate failures, curated slices, target-domain demonstration sets indexed by `(target_domain, slot_signature)`.
- **Trigger:** every N episodes **or** backlog threshold **or** a new `GeneralizeProposal` carrying a few-shot recipe. Always passes through the **unified gate** (Stage 3a is the transfer-specific stage) before affecting production pointers.

### 5.4 Grounding (its own schedule)

- Visual grounding training runs on **grounding milestones** ([Milestones plan](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md)); orchestrator **coordinates** snapshots so the actor always declares which `schema_gen` checkpoint it assumes.

---

## 6. Evaluation matrix

Visual grounding has strong module metrics; the orchestrator adds **full-pipeline** metrics so regressions are attributable.

### 6.1 Actor + inner MDP

| Metric | Definition | Notes |
|--------|------------|-------|
| **Hop efficiency** | Inner actions to successful `EXECUTE` | Stratify by task difficulty. |
| **Action success @ budget** | Env success under fixed hop/token budgets | Primary outer outcome. |
| **Skill reuse quality** | Frequency + outcome of retrieved skills vs. baseline | Ablate retrieval off. |
| **Reasoning overhead** | Latency + tokens per outer step | Compare to budget policy. |
| **Re-ground rate** | `GROUND` insertions per episode | Should correlate with uncertainty, not explode. |

### 6.2 Bank + crafter

| Metric | Definition |
|--------|------------|
| **Contract validity rate** | Fraction of invoked skills whose contracts hold on replay |
| **Transfer lift** | Cross-domain slices: same skill_id usage → outcome delta |
| **Gate pass rate** | Proposals accepted / total proposals |
| **Bank churn** | Promotions + rollbacks per 1k episodes |

### 6.2a Few-shot transfer (target-domain only)

These metrics measure the project's **central thesis** — that game-mined skills generalize to other domains under K-shot adaptation — and **must be reported per `target_domain ∈ TRANSFER_TARGET_DOMAINS`**. They are computed from the artifacts written by `GateService._run_transfer` (Stage 3a) and the per-target `verified_domains` log.

| Metric | Definition | Target |
|--------|------------|--------|
| **K-shot pass rate** | Fraction of `(skill_id, target_domain)` pairs that earn a `verified_domains` entry on first Stage 3a run, given `K = few_shot.k_shot_default` demonstrations | Per target ≥ `transfer_min_target_domains_verified` policy; report curve over K ∈ {1, 5, k_shot_max} |
| **Transfer skill coverage** | Fraction of `ACTIVE` skills that have at least one target-domain entry in `verified_domains` | Should rise monotonically as the bank matures |
| **Multi-target generalization** | For skills with ≥1 verified target, the mean number of target domains they verify on | Skills generalizing across many targets are the high-value subset |
| **Adaptation cost** | Mean tokens consumed per successful Stage 3a `adapt()` call | Bounded above by `few_shot.adaptation_cost_max_tokens` |
| **Target-domain regression rate** | Frequency at which a previously-verified `(skill, target)` is dropped from `verified_domains` (partial deprecation, §3.3) | Tracked separately from full bank rollback |
| **Source-vs-target gap** | Source-domain (game) success rate of a skill minus its mean target-domain success rate after adaptation | Large positive gap = the few-shot path is the bottleneck, not the underlying skill |

### 6.3 Evidence & trace quality

| Metric | Definition |
|--------|------------|
| **Evidence sufficiency** | Fraction of `CHECK` / `COMMIT` hops whose claim is backed by at least one valid `evidence_ref` |
| **Anchor consistency** | Fraction of claim–evidence links that still resolve after the next `GroundingRecord` (zero schema drift = 1.0) |
| **Short-video chain validity** | On Video-Holmes-style tasks: fraction of final answers whose full evidence chain replays successfully on frozen frames |
| **Cross-domain evidence coverage** | For reused skills: fraction of `evidence_refs` populated by the target-domain adapter (vs. empty / stubbed) |

### 6.4 Slices

Report all metrics by **domain** (game / web / video / embodied), **task length**, and **schema version**. Store in `summary.json` per `run_id`.

---

## 7. Budget controller

Centralize **limits** that are currently implicit across plans: one **policy object** per episode (and nested per inner MDP) that all stages consult.

### 7.1 Budget dimensions

| Dimension | Example knobs |
|-----------|----------------|
| **Token budget** | Max tokens per step, per hop, per episode |
| **Hop budget** | Max inner hops before forced `EXECUTE`. Default **0–2 hops**; **≤3 hops** under uncertainty. See [Action Agent §5.4](../02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control). |
| **Grounding escalation budget** | Max tool/GPT calls; Path C limits |
| **Replay budget** | Max CPU time per gate stage |
| **Teacher budget** | Max 32B/72B calls per day; queue discipline |

### 7.2 Policy styles

- **Hard caps** — non-negotiable ceilings (safety).
- **Adaptive caps** — raise grounding budget when uncertainty spikes; **log trigger**.
- **Fairness across domains** — minimum replay and eval slices so rare domains are not starved.

### 7.3 Interface

```text
BudgetController.state(episode_id) → remaining budgets
BudgetController.consume(span_id, cost_vector) → ok | deny | degrade
```

`degrade` must map to **explicit** behaviors (e.g., skip optional `CHECK`, drop an `evidence_ref` enrichment, shorten the typed trace) — never silent omission without logging.

---

## 8. Failure escalation / human audit points

Continuous self-evolution requires **circuit breakers** and **human-readable audit trails**.

### 8.1 Escalation ladder

| Level | Condition | System response |
|-------|-----------|-----------------|
| **L0** | Single step failure | Log; continue with recovery inner actions if budget allows. |
| **L1** | Repeated gate failures on same proposal class | Quarantine proposal generator; switch to shadow-only. |
| **L2** | Domain-wide regression on eval slice | Block promotion; freeze bank pointer; alert. |
| **L3** | Safety / policy violation in traces | Halt rollouts; require human sign-off. |

### 8.2 Human audit points

- **First promotion** of a new skill family or effect chain.
- **Any rollback** of a previously stable snapshot.
- **Change to acceptance thresholds** (ε in non-regression).
- **Escalation of teacher budget** or disabling gate stages for speed (should be rare and logged).

### 8.3 Audit artifact

Each human decision produces a signed `AuditRecord` (who, when, rationale, linked `snapshot_id`s) stored beside gate verdicts.

---

## 9. Implementation checklist (Cursor-ready)

1. **Runner binary / service** — one entrypoint that loads config, wires stages, and writes `run_id` artifacts.
2. **Artifact schemas** — JSON Schema or pydantic models for §2 record types, including `SkillRecord`, `SkillEvaluationRecord`, `GateVerdictPayload` from [PLAN-UNIFIED-SKILL-GATE.md §3](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md#3-canonical-data-structures).
3. **Gate service** — `PromotionOrchestrator` (§3a), `orchestrator/snapshot_manager.py`, `orchestrator/rollback_manager.py`, `orchestrator/eval_suite.py`; calls into `harness/gate_runner.py` for stage execution.
4. **Budget module** — shared library used by action agent and grounding routers.
5. **Eval driver** — runs §6 matrix on frozen checkpoints + bank snapshots; supplies the frozen eval suite to non-regression checks.
6. **Dashboards** — gate pass rate, rollback count, budget denials, per-domain metrics, per-domain transfer-failure-label distributions ([PLAN-HARNESS.md §10a](../05-harness/PLAN-HARNESS.md#10a-transfer-failure-diagnostics-domain-specific)).

---

## 10. Related documents

| Document | Relationship |
|----------|----------------|
| [README](../README.md) | Pipeline overview and shared schema |
| [PLAN-VISUAL-GROUNDING.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md) | Grounding module; feeds `GroundingRecord` |
| [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) | Inner MDP; feeds `InnerHopRecord` |
| [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) | Segmentation & bank; `BankMutationProposal` |
| [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) | Crafter proposals; teacher policies |
| [PLAN-VISUAL-GROUNDING-MILESTONES.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) | Module-level training schedule (coordinate with §5) |
| [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) | Canonical lifecycle + record types + storage split + APIs that §3a transactions operate over |
| [PLAN-HARNESS.md §10b](../05-harness/PLAN-HARNESS.md#10b-gate-execution-runtime) | `GateRunner` — the per-stage entry points called by `PromotionOrchestrator.evaluate_candidate` |

---

*This document does not duplicate module internals; it specifies what must exist **between** modules for a closed, safe, measurable pipeline.*
