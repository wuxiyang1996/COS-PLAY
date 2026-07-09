# PLAN: Failure Routing Policy

**Status:** Single canonical policy that converts observed failures into governed downstream actions.
**Owner:** Pipeline Orchestrator (rule application). Detection is delegated to producers (Harness, Visual Grounding, Judge, Budget Controller, Human Audit).
**Substrate record:** [`FailureRoutingRecord`](PLAN-EXPERIENCE-EXTENSION.md#d-failureroutingrecord--making-failures-governable) (extension layer, P3).
**Companions:** [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) §3, §3a, §8; [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) §10; [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md); [PLAN-EVAL-FIRST-TARGET.md](../00-system/PLAN-EVAL-FIRST-TARGET.md) §6.

---

## 1. Why failure routing is needed

The system already produces a lot of failure signal:

- Harness emits per-invocation diagnostics (`opaque_skill_violation`, `evidence_interface_mismatch`, `slot_binding_failed`, `adapter_execution_mismatch`, `evidence_starved`, `temporal_mismatch`, …).
- Visual Grounding emits schema-validity verdicts on every `GroundingRecord`.
- The Judge emits per-instance verdicts (`F1`–`F7` from [PLAN-EVAL-FIRST-TARGET.md §6](../00-system/PLAN-EVAL-FIRST-TARGET.md#6-failure-taxonomy)).
- The Budget Controller emits `budget_exceeded`, `degrade`, `deny` events.
- The orchestrator's escalation ladder (L0–L3) emits human-audit triggers.

Today these signals fan out into ad-hoc consumers: some land in dashboards, some are dropped, some get re-mined by the Crafter only when the operator remembers to do it. **For a self-evolving system this is not acceptable** — every failure either teaches the system or becomes accumulated drift.

This plan defines:

1. **A small set of failure layers** (where the failure happened).
2. **A small set of routing targets** (what we do about it).
3. **A concrete rules table** that maps the cross-product to a target.
4. **Severity tags** that gate whether a route fires.
5. **A hard-case buffer policy** that bounds storage growth.
6. **An interface contract** that says exactly what the Crafter / Gate / Replay subsystems are allowed to receive.

The policy must be small enough to implement as a single rule table and audit by reading. Anything more elaborate belongs in a follow-up plan.

---

## 2. Failure layers

Every observed failure is tagged with **exactly one** layer. The layer answers *where the failure occurred*, not *what to do about it*. Routing (§3, §4) handles "what to do".

### L1 — Grounding failure

Producer: Visual Grounding (`schema_gen`, Path A/B/C tool loop), `GroundingRecord` validator.

Examples:

- Missing required schema fields (entities, attributes, targets).
- Malformed schema (parse error, type violation, illegal slot).
- High grounding uncertainty above the configured threshold.
- Wrong object or relationship extraction (entity-class mismatch, missing relation that the rest of the trace then assumes).
- Tool-call failure or tool-call timeout that left the schema incomplete.

L1 failures are upstream of every reasoning or skill decision. A missed `L1` corrupts every downstream signal.

### L2 — Invocation failure

Producer: Harness (per-invocation gates), `SkillInvocationRecord`.

Examples:

- Slot binding failed (required slot absent or wrong type after adapter binding).
- Precondition failed (skill's declared precondition does not hold on current `<state>`).
- Evidence insufficient (skill consumed evidence below the declared `evidence_in` requirement, or produced none of the declared `evidence_out`).
- Adapter invalid (`adapter_execution_mismatch`, missing target-domain adapter).
- Harness veto at invocation time (`validate_invocation` returned non-empty veto reason).
- Opaque skill violation (Gate G0 — empty evidence on both sides).

L2 is where the Harness asserts that a skill *can be run safely on this state*; failure here is a runtime contract violation, not a content error.

### L3 — Reasoning failure

Producer: Action Agent (inner MDP), Harness `verify_verdict`, Judge (when run as part of inner reasoning verification).

Examples:

- Wrong hop (selected hop type does not advance the claim graph; `hop_select` regression).
- Weak support chain (`reason_warrant` exists but does not connect to a grounded evidence ref).
- Claim contradicted by a later `CHECK` verdict that the agent ignored.
- Verification failed (verifier returns `not_supported` and the agent committed anyway — overconfident commit).
- Runaway reasoning / inner-MDP non-convergence within the hop budget.

L3 is about the *quality of thought* given that grounding and invocation were structurally OK.

### L4 — Outcome failure

Producer: Pipeline Orchestrator (end-of-episode), Judge (`AnswerSupportRecord.judge_verdict`), non-regression eval.

Examples:

- Final answer wrong (Judge verdict `incorrect`, or MCQ exact-match failure).
- Task failed (environment terminal failure on game / web / os tasks).
- Regression after transfer (a transferred skill caused a measurable Joint Success Rate drop on a frozen eval slice — see [PLAN-EVAL-FIRST-TARGET.md §5](../00-system/PLAN-EVAL-FIRST-TARGET.md#5-joint-success-definition-headline)).
- Unsafe or unstable execution (policy violation, environment-side error, repeated crash).

L4 is the only layer the *user* sees. L1–L3 exist so we can attribute L4 back to a fixable root.

### Layer assignment rule

A single root failure may emit signals at multiple layers (e.g. malformed schema → invocation fails → answer wrong). The detector tags **its own layer**. The orchestrator may link related routing records via `parent_failure_id` ([PLAN-EXPERIENCE-EXTENSION.md §3.D](PLAN-EXPERIENCE-EXTENSION.md#d-failureroutingrecord--making-failures-governable)) but never collapses them.

---

## 3. Routing targets

Closed set. Adding a target requires bumping the policy version and updating the table in §4.

| Target | Owner | What it does | Storage |
|--------|-------|--------------|---------|
| `audit_only` | Human-audit dashboard | Surfaces the failure for review. No automatic action. | `failures/audit/` |
| `replay_buffer` | Replay validator (gate stage 3) | Adds the episode/step to the held-out replay slice for future non-regression checks. | `failures/replay/` |
| `relabel_queue` | Annotation pipeline | Sends the offending record back for relabeling (gold answer suspect, schema mis-labeled). | `failures/relabel/` |
| `crafter_refine_queue` | [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) | Becomes a candidate input for skill repair / composition / hypothesis. | `failures/crafter/` |
| `rollback_candidate` | `PromotionOrchestrator.rollback_if_needed` ([§3a.4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a4-rollback-transaction-the-revert-path)) | Adds the implicated `(skill_id, version)` to the rollback candidate list. | `failures/rollback/` |
| `adapter_repair_queue` | `AdapterRegistry` owner ([PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md)) | Marks an adapter as needing repair (bad binding, broken slot mapping). | `failures/adapter/` |
| `ignore_noise` | — | The route was deliberately a no-op. Recorded so we can audit how often we ignore. | `failures/noise/` |

`ignore_noise` is a real route, not a missing route. A failure with no routing record is a bug; a failure routed to `ignore_noise` is a deliberate decision.

---

## 4. Routing rules table

The orchestrator applies the **first matching rule** top-to-bottom. Rules are version-pinned in `configs/failure_routing.yaml`. The table is read every release.

| # | Layer | Failure type (label) | Severity gate | Recoverability gate | Repeat gate | Route(s) |
|---|-------|-----------------------|---------------|---------------------|-------------|----------|
| R1 | L1 | `schema_malformed` | ≥ `error` | any | any | `audit_only` + `relabel_queue` |
| R2 | L1 | `schema_incomplete` (missing required field) | ≥ `error` | `recoverable` | any | `replay_buffer` |
| R3 | L1 | `grounding_uncertainty_high` | `warn` | `recoverable` | ≥ 3 in 100 ep | `replay_buffer` |
| R4 | L1 | `wrong_entity_extraction` | ≥ `error` | any | any | `relabel_queue` + `audit_only` |
| R5 | L1 | `tool_call_failure` | `warn` | `rerun_required` | any | `replay_buffer` |
| R6 | L2 | `slot_binding_failed` | ≥ `error` | any | any | `adapter_repair_queue` |
| R7 | L2 | `precondition_failed` | `warn` | `recoverable` | ≥ 5 in 1k ep | `crafter_refine_queue` |
| R8 | L2 | `evidence_insufficient` / `evidence_starved` | ≥ `error` | any | any | `crafter_refine_queue` |
| R9 | L2 | `adapter_execution_mismatch` | ≥ `error` | any | any | `adapter_repair_queue` |
| R10 | L2 | `harness_veto` (correctly fired) | `info` | `recoverable` | any | `ignore_noise` |
| R11 | L2 | `harness_veto` (false positive — Actor fallback succeeded > N times) | `warn` | `recoverable` | ≥ 5 in 1k ep | `crafter_refine_queue` |
| R12 | L2 | `opaque_skill_violation` | `critical` | `rollback_required` | any | `rollback_candidate` + `audit_only` |
| R13 | L3 | `wrong_hop` | `warn` | `recoverable` | ≥ 5 in 1k ep | `crafter_refine_queue` |
| R14 | L3 | `weak_support_chain` | ≥ `error` | any | any | `crafter_refine_queue` |
| R15 | L3 | `claim_contradicted` | ≥ `error` | any | any | `crafter_refine_queue` + `replay_buffer` |
| R16 | L3 | `verification_failed` (overconfident commit) | ≥ `error` | any | any | `crafter_refine_queue` + `audit_only` |
| R17 | L4 | `final_answer_wrong` (F1/F2 from eval) | ≥ `error` | any | any | `crafter_refine_queue` + `replay_buffer` |
| R18 | L4 | `final_answer_wrong` (F3/F4 — answer correct, evidence broken) | ≥ `error` | any | any | `crafter_refine_queue` + `audit_only` |
| R19 | L4 | `transfer_regression` (post-promotion) | `critical` | `rollback_required` | ≥ 1 confirmed | `rollback_candidate` |
| R20 | L4 | `unsafe_execution` | `critical` | `rollback_required` | any | `rollback_candidate` + `audit_only` |
| R21 | * | `annotation_noise_suspected` (judge–human disagreement, gold flagged) | any | `fatal_for_label` | any | `audit_only` |
| R22 | * | rare, low-severity, non-repeating | `info` | `recoverable` | < 2 in 10k ep | `ignore_noise` |

Notes:

- "Severity gate `≥ error`" means rules apply only when severity is `error` or `critical` (see §5).
- Repeat gates use a sliding window over recent episodes (`recent_window_eps` in policy config).
- A rule may emit multiple routes; the orchestrator writes one `FailureRoutingRecord` per route, all linked via `related_routing_ids`.
- `R10` is the explicit "the Harness did its job" rule — vetoes that the Actor recovered from cheaply are noise, not work for the Crafter.

---

## 5. Severity and recoverability

Every routable failure is tagged with four dimensions before the rules table is consulted. These reuse the field set already defined for `FailureRoutingRecord` plus two derived fields the orchestrator computes at routing time.

| Dimension | Values | Source |
|-----------|--------|--------|
| `severity` | `info` / `warn` / `error` / `critical` | Detector (Harness, Grounding, Judge, Budget). Defaults documented per `failure_type`. |
| `recoverability` | `recoverable` / `rerun_required` / `rollback_required` / `fatal` | Detector. `rollback_required` is the only value that can short-circuit to `rollback_candidate`. |
| `repeat_count` | int | Orchestrator. Computed over the sliding window keyed on `(failure_type, skill_id?, adapter_id?)`. |
| `blast_radius` | `instance` / `episode` / `skill` / `adapter` / `domain` / `bank_snapshot` | Orchestrator. How far the failure can propagate if left unrouted. |

### Default severity defaults

- L1 schema malformed / wrong entity extraction → `error`.
- L1 high uncertainty → `warn`.
- L2 contract violations → `error` (G0 violation is `critical`).
- L3 weak support chain / claim contradicted → `error`.
- L4 answer wrong → `error`. Transfer regression / unsafe execution → `critical`.

### Blast radius rules

- `bank_snapshot` blast radius forces inclusion of `rollback_candidate` regardless of repeat count.
- `domain` blast radius forces `replay_buffer` to make the regression visible at the next non-regression check.
- `instance` blast radius is eligible for `ignore_noise` only when severity is `info` and repeat count is below the §6 buffer threshold.

These four dimensions are the only inputs the rules table is allowed to read. Any new dimension requires a policy version bump.

---

## 6. Hard-case buffer policy

The system would otherwise drown in routing records. The buffer policy decides which failures are *kept* in the routable corpus (and therefore visible to the Crafter, Replay validator, and Adapter repair queue) versus collapsed into counters.

A failure enters the **hard-case buffer** if at least one of the following holds:

1. **Frequent.** `repeat_count ≥ frequent_threshold` (default 5) within `recent_window_eps`.
2. **High loss.** Severity is `critical`, or it directly caused an L4 outcome failure (linked via `parent_failure_id`).
3. **Cross-domain repeated.** Same `failure_type` + same `skill_id` observed across ≥ 2 domains within the window.
4. **High-uncertainty near-miss.** Judge reports `EvidenceValid = False` *but* `AnswerCorrect = True` (F3/F4 in [PLAN-EVAL-FIRST-TARGET.md §6](../00-system/PLAN-EVAL-FIRST-TARGET.md#6-failure-taxonomy)) — these are exactly the cases where the system is "right for the wrong reasons" and we want to learn from them aggressively.

Failures that **do not** match any of these criteria are routed to `ignore_noise`, counted, and dropped from per-failure storage. The counter is preserved so we can prove how often `ignore_noise` was applied.

Storage rule: hard-case buffer is per-target. Each routing target has its own bounded queue with a documented eviction policy:

| Target | Default capacity | Eviction |
|--------|------------------|----------|
| `crafter_refine_queue` | 5,000 | LRU within `severity` band |
| `replay_buffer` | 10,000 | Random reservoir within slice |
| `adapter_repair_queue` | 1,000 | LIFO; oldest unresolved escalates to `audit_only` |
| `rollback_candidate` | unbounded | Drained per promotion cycle |
| `relabel_queue` | 2,000 | LRU |
| `audit_only` | unbounded | Drained on human review |

Capacities live in `configs/failure_routing.yaml`. They are policy, not code.

---

## 7. Interface to Crafter / Gate / Replay

The routing layer is the *only* path by which detected failures reach downstream subsystems. No subsystem reads detector output directly.

### 7.1 Skill Crafter ([PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md))

**Valid Crafter inputs** (from `crafter_refine_queue` only):

- L2: `evidence_insufficient`, `evidence_starved`, `precondition_failed`, `harness_veto` (false-positive variant).
- L3: every reasoning failure (`wrong_hop`, `weak_support_chain`, `claim_contradicted`, `verification_failed`).
- L4: `final_answer_wrong` (all F-classes), specifically including F3/F4 from the hard-case buffer.

**Invalid Crafter inputs** (the orchestrator MUST NOT route these to Crafter, even when severity is high):

- L1 raw parser glitches (`schema_malformed`, `tool_call_failure`) — these are grounding bugs, not skill bugs.
- L1 `wrong_entity_extraction` when severity stems from labeling drift — goes to `relabel_queue`, not Crafter.
- L2 `slot_binding_failed` / `adapter_execution_mismatch` — these are adapter bugs; Crafter cannot fix bindings, `adapter_repair_queue` can.
- Anything tagged `annotation_noise_suspected` — Crafter would learn the wrong lesson from labeling noise.
- Anything routed to `ignore_noise`.

The Crafter consumes records from `crafter_refine_queue` via `failures/crafter/*.jsonl` and turns them into `BankMutationProposal` candidates ([PLAN-PIPELINE-ORCHESTRATOR.md §2.2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#22-required-record-types)). Proposals still pass the full gate stack — failure-driven proposals are not privileged.

### 7.2 Gate (Harness `GateRunner` + `PromotionOrchestrator`)

**Inputs the gate is allowed to consume from routing:**

- `rollback_candidate` records — fed into `PromotionOrchestrator.rollback_if_needed` ([§3a.4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a4-rollback-transaction-the-revert-path)).
- `replay_buffer` records — added to the frozen replay slice consumed by gate stage 3 ([§3.1](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#31-gate-stages-ordered)).

**The gate never reads routing records to decide a verdict mid-stage.** Routing records influence *which traces enter the slice*, not how a candidate is scored on that slice.

### 7.3 Replay validator

- Consumes from `replay_buffer` only.
- Replay slices are versioned by `eval_suite_id` so adding a routed failure to the slice produces a new pinned suite (no silent slice drift).
- Failures with `recoverability == fatal` are excluded from the replay slice — they cannot teach a stable check.

### 7.4 Visual Grounding re-train queue

- Consumes from `relabel_queue` and from L1 `replay_buffer` records tagged `schema_incomplete`.
- Used by [PLAN-VISUAL-GROUNDING-MILESTONES.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) training rounds.

### 7.5 Adapter repair queue

- Consumes from `adapter_repair_queue`.
- Each entry pins `(adapter_id, adapter_version, failure_type, exemplar_routing_ids)`.
- Adapter owners drain this queue; a routed entry is closed only when a new adapter version passes the binding gate ([PLAN-HARNESS.md §10](../05-harness/PLAN-HARNESS.md#10-promotion-gates)).

### 7.6 Human audit

- Consumes `audit_only`.
- Drives `AuditRecord` writes ([PLAN-PIPELINE-ORCHESTRATOR.md §3a.5](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a5-audit-logging-contract)).
- Human verdicts can re-route a record (e.g. confirm an `annotation_noise_suspected` case → flip to `relabel_queue`); re-routes append a new `FailureRoutingRecord` with `parent_failure_id` set.

---

## 8. File / object alignment

This plan does not introduce a new record type. It uses [`FailureRoutingRecord`](PLAN-EXPERIENCE-EXTENSION.md#d-failureroutingrecord--making-failures-governable) (extension layer P3, file `data_structure/extensions/failure_routing_record.py`).

### 8.1 Two-step write

The record is written in **two steps** — this is the same two-step contract the Experience Extension plan already pins; this plan operationalizes it.

1. **Detection write** (producer-owned). The detector (Harness gate, schema validator, Judge, Budget Controller, human auditor) creates a `FailureRoutingRecord` with everything *except* `route_to`. `route_to` stays `None`. The record is appended to `failures/incoming/<date>.jsonl`.
2. **Routing write** (orchestrator-owned). The `FailureRouter` (this plan's component, lives in `orchestrator/failure_router.py`) reads `failures/incoming/`, applies the §4 rules table using §5 dimensions, sets `route_to`, and writes the record to the per-target file under `failures/<target>/<date>.jsonl`. Hard-case buffer policy (§6) is applied at this step.

A record with `route_to is None` after step 2 is a routing bug, not a valid state.

### 8.2 What is created

```
failures/
├── incoming/
│   └── YYYY-MM-DD.jsonl        # step 1: detector writes
├── audit/
├── replay/
├── relabel/
├── crafter/
├── rollback/
├── adapter/
└── noise/                      # step 2: router writes (one of these)
```

`failures/incoming/` is drained continuously; the orchestrator must not let it grow without bound.

### 8.3 What is consumed

| Reader | Reads from | Cadence |
|--------|------------|---------|
| Skill Crafter | `failures/crafter/` | Every Crafter batch ([PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) cadence). |
| Replay validator | `failures/replay/` | Every gate stage 3 invocation. |
| `PromotionOrchestrator.rollback_if_needed` | `failures/rollback/` | Every promotion cycle. |
| Adapter owners | `failures/adapter/` | Per adapter-repair sprint. |
| Annotation pipeline | `failures/relabel/` | Per relabel sprint. |
| Human audit dashboard | `failures/audit/` | Continuously. |
| Counters dashboard | `failures/noise/` (counts only) | Continuously. |

Readers consume `FailureRoutingRecord.from_dict(...)` only. They never reach into the originating substrate object — they follow `source_record_type` + `source_record_id` if they need the underlying artifact.

### 8.4 Append-only and re-routing

Re-routing (e.g. human audit downgrades a `crafter_refine_queue` record to `ignore_noise`) appends a **new** `FailureRoutingRecord` with `parent_failure_id` pointing at the original. The original is never mutated. This matches [PLAN-EXPERIENCE-EXTENSION.md §3.D](PLAN-EXPERIENCE-EXTENSION.md#d-failureroutingrecord--making-failures-governable) "append-only".

---

## 9. Minimal implementation order

Each phase is independently shippable. None of them changes substrate code; all code lives under `orchestrator/`.

| Phase | Deliverable | Why this order |
|-------|-------------|----------------|
| **R0** | `configs/failure_routing.yaml` with the §4 rules table, §5 default severities, §6 thresholds, §7 capacities. | The rules are policy. Pinning the YAML first lets the rest be tested against fixtures. |
| **R1** | `orchestrator/failure_router.py`: `FailureRouter.route(record) -> List[FailureRoutingRecord]`. Pure function over a record + policy. | Pure function, fully unit-testable, no I/O. Detector writes still go to `failures/incoming/` but no router runs yet. |
| **R2** | `orchestrator/failure_router_runner.py`: drains `failures/incoming/`, calls `FailureRouter.route`, writes per-target files. Implements §6 hard-case buffer. | Wires R1 to disk. Makes the routing observable end-to-end. |
| **R3** | Detector adapters (no new code in detectors — thin shims that build the step-1 `FailureRoutingRecord` from existing diagnostics). One shim per producer: Harness, Visual Grounding, Judge, Budget Controller. | Until shims exist, detectors keep emitting only their existing diagnostics; the router has nothing to route. |
| **R4** | Reader integrations: Crafter reads `failures/crafter/`; `PromotionOrchestrator` reads `failures/rollback/`; Replay validator reads `failures/replay/`. | Closes the loop; failures now drive downstream change. |
| **R5** | Dashboards: per-target queue depth, eviction count, `ignore_noise` count, judge–human re-route rate. | Necessary to detect routing-policy drift. |
| **R6** | Policy review cadence: every release, the §4 table is re-read; rules with zero hits in 4 weeks are flagged for removal; `ignore_noise` rate above threshold flags a missing rule. | The policy is small *because* it is reviewed regularly. |

The minimum to be useful in the first release is R0 + R1 + R2 + the Harness shim from R3 + the Crafter and rollback readers from R4. That is enough to make L2 and L4 failures governable end-to-end.

---

## 10. Anti-goals

Stated to keep the policy small and operational.

1. **No memory subsystem.** Routing records are episode-anchored (`episode_id`, `source_record_id`). They are not queryable by content from outside their episode. Routing exists to govern action, not to reintroduce recall.
2. **No ML-driven router in v1.** The router is a deterministic rule application over the §4 table. No learned dispatcher, no embedding similarity over failures. ML routing is a separate plan if it is ever needed.
3. **No automatic Crafter-driven mutation of Visual Grounding labels.** Even though F3/F4 cases are juicy, routing them to `crafter_refine_queue` MUST NOT cause silent label rewrites. Label changes go through `relabel_queue` and human review.
4. **No silent dropping.** Every detected failure produces a `FailureRoutingRecord`. The only way to "drop" a failure is to route it to `ignore_noise`, which is itself recorded.
5. **No bypass of Harness gates.** Routing influences which candidates get proposed and which slices get replayed. It never overrides a gate verdict.
6. **No raw-noise → Crafter shortcut.** §7.1 is enforced by the rules table. A future rule that routes L1 parser glitches to `crafter_refine_queue` is rejected by review.
7. **No new severity levels.** The four-value severity scale is closed. New `failure_type`s pick from existing severities.
8. **No long-lived `failures/incoming/`.** If the runner falls behind, raise an L2 escalation ([PLAN-PIPELINE-ORCHESTRATOR.md §8.1](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#81-escalation-ladder)). Backlog is itself a failure.

---

## 11. Related documents

| Document | Relationship |
|----------|--------------|
| [PLAN-EXPERIENCE-EXTENSION.md](PLAN-EXPERIENCE-EXTENSION.md) | Defines `FailureRoutingRecord` and the two-step write contract. |
| [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) | §3 gate stages, §3a promotion / rollback transactions, §8 escalation ladder. |
| [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) | Source of L2 invocation diagnostics; consumer of `rollback_candidate`. |
| [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) | Sole consumer of `crafter_refine_queue`; produces `BankMutationProposal` from routed failures. |
| [PLAN-EVAL-FIRST-TARGET.md](../00-system/PLAN-EVAL-FIRST-TARGET.md) | §6 failure taxonomy; F3/F4 are the canonical hard-case buffer entries. |
| [PLAN-VISUAL-GROUNDING.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md) | Source of L1 grounding failures; consumer of `relabel_queue`. |
| [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) | Lifecycle states that `rollback_candidate` records flip. |
