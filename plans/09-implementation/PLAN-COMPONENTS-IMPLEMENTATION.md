# PLAN: Components Implementation — Skill Harness, Skill Crafter, Pipeline Orchestrator (Cursor-Ready)

> **Lane decision (2026-05-01) — lane (a), Context-only skills.** This
> implementation plan was authored under the lane-(b) assumption
> (skills as runnable programs); the lane was closed in favour of
> retrieval-payload semantics. Authoritative record:
> [`implementation_notes/legacy/skill-lane-decision.md`](../../implementation_notes/legacy/skill-lane-decision.md).
> Practical implications for the build sheet:
>
> * **The mental-model row "Skill Harness — *Can this skill run here
>   right now?*"** is now interpreted as *"is this retrieval payload
>   eligible for the actor right now?"* (G0/G2 invariants + F2′ task
>   axis). The Harness in the live trainer never invokes
>   `run_skill(...)`.
> * **Phase ordering for §16 (executor) and `inner_mdp` integration
>   stays in tree** but ships as **offline / lane-(b) diagnostic
>   tooling**, not as a launch-blocker for the live trainer. The
>   live trainer's Crafter / Harness wires landed at Day-10 without it
>   (see `IMPLEMENTATION-STATUS.md` "Delivered" rows for Day-7→10).
> * **Single-MDP companion decision (T3.6):** every reference to
>   `inner_mdp` / `hop_select` / a separate hop-selection LoRA in the
>   ordered build sheet below is **obsolete**. The actor is one MDP
>   with two GRPO LoRAs (`skill_selection` + `action_taking`); the
>   warm-start adapters listed in
>   `runs/sft_coldstart/sft_summary_all.json` correspond to that
>   architecture. Companion record:
>   [`implementation_notes/legacy/single-vs-two-mdp-tradeoff.md`](../../implementation_notes/legacy/single-vs-two-mdp-tradeoff.md).
> * **Crafter `enable_protocol_patching` flag (T1.3a)** is the single
>   source of truth for whether a build target is live or
>   offline-diagnostic. `False` (default) → live trainer; `True` →
>   `labeling_supplement/` drivers and lane-(b) regression suites.
> * **`PatchProposal` / `RecoveryStrategy.{HOP_INSERTION,
>   PROTOCOL_PATCH, FALLBACK_INJECTION, REGROUNDING_TRIGGER,
>   SKILL_DECOMPOSITION}` proposal types** stay in the typed proposal
>   union for binary compatibility but are minted only behind the
>   protocol-patching flag.
>
> Sections below remain useful as the *full lane-(b) build sheet* — the
> escalation target if the rollback condition in
> [`skill-lane-decision.md` §4](../../implementation_notes/legacy/skill-lane-decision.md) trips. Treat them as the build
> sheet for the offline gate / diagnostic stack unless a row is
> explicitly tagged "live."

**Scope.** This is a **Cursor-ready implementation plan** that turns the three component-level design plans into a concrete, ordered coding sequence. It does **not** replace the design plans; it is the build sheet that tells Cursor *what files to create, in what order, and where to stop each phase*.

**Inputs (canonical specs — do not duplicate, link instead).**

- [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) — design of `SkillEpisode`, `SkillHarness`, `AdapterRegistry`, `TransferManager`, `ReplayValidator`, `RewardLogger`, six-gate promotion (G0 evidence-driven / G1 binding / G2 adapter / G3 replay / G4 shadow / G5 non-regression), and the Phase 0 + Phase 1 immediate target.
- [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) — composer / generalizer / hypothesizer creation modes, typed proposals (`PatchProposal | ComposeProposal | TransferProposal | RetireProposal`), failure trace + private failure memory, frozen 32B/72B teacher policy.
- [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) — hot-path / warm-path DAG, artifact schemas (`SkillRecord`, `SkillEvaluationRecord`, `GateVerdict`, `AuditRecord`), promotion / rollback transactions (§3a), four-way Actor / Harness / Bank / Orchestrator boundary (§0a), budget controller, evaluation matrix.
- [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) — canonical lifecycle states, ownership split (`SkillLifecycleManager`, `GateRunner`, `PromotionOrchestrator`), storage split (`draft_store / candidate_store / active_store / archive_store`).
- [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) — skill object, retrieval API, lineage / negative-knowledge fields.
- [PLAN-ACTION-AGENT.md §1a](../02-action-agent/PLAN-ACTION-AGENT.md#1a-actor-role-and-boundary) — Actor consumes Harness-filtered `eligible_skills`, Actor remains the online policy.

**Mental model for coding.**

| Module | Asks |
|--------|------|
| **Skill Crafter** | *What new skill should exist next?* |
| **Skill Harness** | *Can this skill run here right now?* |
| **Pipeline Orchestrator** | *Should this proposal become part of the next system version?* |

Do **not** collapse these three modules. The architectural reasons are pinned in [PLAN-PIPELINE-ORCHESTRATOR.md §0a](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#0a-actor-harness-skill-bank-orchestrator-boundary).

---

## 0. Architectural boundaries (must hold across phases)

| Module | Owns | Must NOT do |
|--------|------|-------------|
| **Skill Harness** (per-invocation runtime) | retrieval, slot binding, adapter attachment, execution, tracing, `SkillEpisode`, shadow / replay validation hooks | invent new skills; rewrite contracts; promote skills; mutate production bank snapshots; replace the Actor as the online policy |
| **Skill Crafter** (slow-timescale proposal layer) | failure analysis, composition, generalization, hypothesis generation, typed proposal outputs | execute skills online; perform slot binding on live states; decide promotion; mutate production bank snapshots |
| **Pipeline Orchestrator** (system control plane) | episode runs, artifact IDs / storage, acceptance gates, promotion / rollback transactions, train scheduling, re-evaluation, audit trail | per-invocation execution; author skill contents; reach inside Actor policy choices |

These boundaries are normative for every phase below. Any code that violates them must be rejected at review.

---

## 1. Repo structure (target layout)

```
src/
  skill_bank/
    models.py           # SkillRecord, SkillSpec, lifecycle enums
    query.py            # retrieve(top_k) — read-only client used by Harness
    store.py            # snapshot reader (active_store / archive_store)

  harness/
    skill_episode.py    # PLAN-HARNESS §5.1
    skill_harness.py    # PLAN-HARNESS §5.2
    adapter_registry.py # PLAN-HARNESS §5.3
    transfer_manager.py # PLAN-HARNESS §5.4.1 (Phase D)  — lifecycle bookkeeping
    few_shot_adapter.py # PLAN-HARNESS §5.4.2 (Phase D)  — Stage 3a K-shot machinery
    replay_validator.py # PLAN-HARNESS §5.5 (Phase D)
    reward_logger.py    # PLAN-HARNESS §5.6
    gate_runner.py      # PLAN-HARNESS §10b (Phase D)
    policies.py         # ranking weights, shadow-origin penalty
    adapters/
      _stub_base.py     # shared scaffolding for transfer-target stub adapters
      gymv.py           # SOURCE_DOMAINS adapter (game, real)
      browser.py        # TRANSFER_TARGET_DOMAINS adapter (stub in Phase A)
      osworld.py        # TRANSFER_TARGET_DOMAINS adapter (stub)
      video.py          # TRANSFER_TARGET_DOMAINS adapter (stub) — first transfer arena
      visual_reasoning.py # TRANSFER_TARGET_DOMAINS adapter (stub)

  crafter/
    proposal_types.py   # ComposeProposal, GeneralizeProposal, HypothesisProposal, RetireProposal
    composer.py
    generalizer.py
    hypothesizer.py
    failure_trace.py
    failure_diagnoser.py
    failure_memory.py   # Crafter-private; never read by online actor
    recovery_selector.py
    counterfactual.py
    service.py          # SkillCrafterService — single entrypoint

  orchestrator/
    runner.py                # PipelineOrchestrator: run_episode + run_offline_cycle
    stage_graph.py           # hot-path / warm-path DAG wiring
    schemas.py               # EpisodeTrace, StepRecord, BankMutationProposal, GateVerdict, AuditRecord, TrainJobSpec
    artifact_store.py        # local file-backed store; matches PLAN-PIPELINE-ORCHESTRATOR §2.3 layout
    gate_service.py          # static / replay / shadow / transfer / non-regression entrypoints
    promotion_orchestrator.py  # PLAN-PIPELINE-ORCHESTRATOR §3a transactions
    snapshot_manager.py      # bank-snapshot create / discard / pointer move
    rollback_manager.py
    eval_suite.py            # frozen evaluation slice for non-regression
    eval_driver.py           # PLAN-PIPELINE-ORCHESTRATOR §6 metrics
    budget.py                # BudgetController (PLAN-PIPELINE-ORCHESTRATOR §7)
    config.py

  common/
    state_schema.py     # canonical <state> typing
    ids.py              # run_id / episode_id / step_id / span_id helpers
    enums.py
    typing.py
```

Where this layout differs from existing on-disk code, **align to this layout** as part of Phase A — do not branch the directory structure.

---

## 2. Skill Harness (build first)

The Harness plan explicitly names Phase 0 + Phase 1 as the immediate implementation target ([PLAN-HARNESS.md §19](../05-harness/PLAN-HARNESS.md#19-immediate-next-target)). Build `SkillEpisode`, `SkillHarness`, and `AdapterRegistry`, then route all current skill usage through them **before** transfer promotion.

### 2.1 `harness/skill_episode.py`

Implement the dataclass exactly per [PLAN-HARNESS.md §5.1](../05-harness/PLAN-HARNESS.md#51-skillepisode); the additional ID fields below are required by the orchestrator's artifact contract ([PLAN-PIPELINE-ORCHESTRATOR.md §2.1](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#21-core-identifiers)).

```python
class SkillEpisode(BaseModel):
    # Identifiers (orchestrator artifact contract)
    run_id: str
    episode_id: str
    step_id_start: int
    step_id_end: int | None
    span_id: str

    # Skill provenance
    skill_id: str
    skill_version: str
    skill_type: Literal["reasoning", "action", "grounding", "mixed"]
    proposal_id: str | None = None

    # Domain context
    source_domain: str
    target_domain: str
    schema_hash: str
    goal: str
    adapter_id: str | None = None
    adapter_version: str | None = None

    # Evidence-driven invariant (PLAN-SKILL-BANK §0.3)
    evidence_role: Literal["GATHER", "VERIFY", "REASON", "COMMIT"]
    evidence_in:        list = []
    evidence_out:       list = []
    evidence_warrant:   list | None = None  # required iff role == COMMIT
    verify_verdict:     Literal["PASS", "FAIL", "INSUFFICIENT"] | None = None  # iff VERIFY
    reason_warrant:     list | None = None  # iff REASON; subset of evidence_in

    # Execution
    slot_bindings: dict
    protocol_trace: list  # inner-hop sequence
    evidence_trace: list  # superset of evidence_in/out

    # Outcome
    contract_progress: dict
    outcome: Literal["success", "fail", "abort", "stall", "shadow_only"]
    abort_reason: str | None = None
    failure_type: str | None = None

    # Reward + diagnostics
    reward_components: dict = {}
    reward_delta: float = 0.0
    diagnostics: list[str] = []
    shadow: bool = False
    metadata: dict = {}
```

`finalize_episode` must enforce the role-specific field requirements from [PLAN-HARNESS.md §5.1 evidence-role table](../05-harness/PLAN-HARNESS.md#51-skillepisode); violations stamp `outcome = "fail"` with `abort_reason ∈ {"opaque-skill-violation", "skill-role-mismatch"}` and disqualify the episode from promotion evidence.

### 2.2 `harness/adapter_registry.py`

Method signatures per [PLAN-HARNESS.md §5.3](../05-harness/PLAN-HARNESS.md#53-adapterregistry):

```python
class AdapterRegistry:
    def register(self, skill_type: str, domain: str, adapter: SkillAdapter) -> None: ...
    def get(self, skill_type: str, domain: str) -> SkillAdapter | None: ...
    def validate(self, adapter: SkillAdapter, skill: SkillSpec, state: State) -> AdapterVerdict: ...
    def request_synthesis(self, skill: SkillSpec, domain: str) -> AdapterProposal: ...  # calls 72B
```

**Phase 1 wiring (asymmetric).** Register the *real* `gymv` adapter (`SOURCE_DOMAINS`) and **stub** adapters for every domain in `TRANSFER_TARGET_DOMAINS` (`browser`, `osworld`, `video`, `visual_reasoning`). Stub adapters share `harness/adapters/_stub_base.py` and produce deterministic short hop-loops; they are sufficient for Stage 3a few-shot adaptation runs to exercise the gate end-to-end before any of those domains has a real backend. Real adapters replace the stubs domain-by-domain in later phases without changing the `AdapterRegistry` interface or the gate.

### 2.3 `harness/skill_harness.py`

Public entry points per [PLAN-HARNESS.md §5.2](../05-harness/PLAN-HARNESS.md#52-skillharness). The runtime sequence is **fixed** and matches §4 of the harness plan:

```
normalize_state
  → retrieve_candidates (skill_bank.query)
  → rank_candidates (policies.py)
  → filter to eligible_skills        # consumed by Actor as ActorInput.eligible_skills
  → bind_skill                       # G1
  → attach_adapter                   # G2
  → execute_step (protocol-aware)
  → update_episode (trace / evidence / contract)
  → continue | switch | abort        # budget-aware (orchestrator/budget.py)
  → finalize_episode                 # G0 enforced here
  → return action(s) + SkillEpisode
```

**Critical:** the Harness produces `eligible_skills` and may **veto** an Actor-proposed invocation, but it does **not** select the final skill. The Actor decides ([PLAN-ACTION-AGENT.md §1a.6](../02-action-agent/PLAN-ACTION-AGENT.md#1a6-actorharness-interaction), [PLAN-HARNESS.md §1a.4](../05-harness/PLAN-HARNESS.md#1a4-actor-proposal-harness-veto)).

### 2.4 `harness/reward_logger.py`

Single sink for `r_env` / `r_follow` / `r_cost` / `r_transfer` / `r_adapter` per [PLAN-HARNESS.md §5.6](../05-harness/PLAN-HARNESS.md#56-rewardlogger). After this lands, no other module is allowed to write reward components directly.

### 2.5 Phase A acceptance criteria (Harness MVP)

Stop Phase A only when **all** of the following are true:

1. Every skill invocation in the current `ActionAgent` goes through `SkillHarness`.
2. Every invocation writes one `SkillEpisode` (JSONL) under `artifacts/runs/{run_id}/episodes/{episode_id}/skill_episodes.jsonl`.
3. Semantic skill and domain adapter are cleanly separated, with **at least two adapters** (`gymv`, `browser`) attached through `AdapterRegistry`.
4. `finalize_episode` enforces evidence-role field requirements; opaque-skill / role-mismatch violations are stamped on the episode.
5. The Actor receives `eligible_skills` from the Harness rather than querying the bank directly.

These match [PLAN-HARNESS.md Phases 0–1](../05-harness/PLAN-HARNESS.md#14-phased-implementation-plan) success criteria.

### 2.6 What stays out of the Harness

- No skill invention.
- No contract rewriting.
- No bank promotion or snapshot mutation.
- No final policy choice over which skill to run (that is the Actor; see [PLAN-ACTION-AGENT.md §1a.2](../02-action-agent/PLAN-ACTION-AGENT.md#1a2-actor-decision-scope)).

---

## 3. Pipeline Orchestrator (build second)

Build a **small control plane**, not a distributed system. The orchestrator's job in MVP is to (a) own `run_id` / artifact IDs, (b) route candidates through gates, and (c) move the bank pointer atomically.

### 3.1 `orchestrator/schemas.py`

Pydantic models for every artifact named in [PLAN-PIPELINE-ORCHESTRATOR.md §2.2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#22-required-record-types):

- `EpisodeMeta`
- `GroundingRecord`
- `InnerHopRecord`
- `ActionRecord` (carries `evidence_warrant`)
- `RewardRecord`
- `SkillEpisode` *(re-export from `harness/skill_episode.py`)*
- `BankMutationProposal` (subclasses: `PatchProposal | ComposeProposal | TransferProposal | RetireProposal`)
- `GateVerdict` + `GateVerdictPayload`
- `SkillRecord`, `SkillEvaluationRecord` *(per [PLAN-UNIFIED-SKILL-GATE.md §3](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md))*
- `TrainJobSpec`
- `AuditRecord`

### 3.2 `orchestrator/artifact_store.py`

Local file-backed, append-only. Layout pinned by [PLAN-PIPELINE-ORCHESTRATOR.md §2.3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#23-storage-layout-logical):

```
artifacts/
  runs/{run_id}/
    episodes/{episode_id}/
      trace.jsonl
      summary.json
      skill_episodes.jsonl
  bank/
    snapshots/{snapshot_id}/
    proposals/{proposal_id}/
  gates/
    {gate_run_id}/verdict.json
  train/
    {job_id}/spec.json
```

Every write must be **idempotent** (DAG invariant in [PLAN-PIPELINE-ORCHESTRATOR.md §1.4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#14-dag-invariants)).

### 3.3 `orchestrator/runner.py`

Two public APIs that map directly to the documented DAG subgraphs:

```python
class PipelineOrchestrator:
    def run_episode(self, env, policy, cfg) -> EpisodeResult:
        """Hot path: observe → ground → inner_mdp → act → log_step (PLAN-PIPELINE-ORCHESTRATOR §1.1).
        Calls into SkillHarness on every inner_mdp step where a skill is invoked."""

    def run_offline_cycle(self, cfg) -> OfflineCycleResult:
        """Warm path: select_batch → crafter_propose → acceptance_gate
                      → promote_or_rollback → schedule_train → re_evaluate
        (PLAN-PIPELINE-ORCHESTRATOR §1.3)."""
```

The runner must always assign `run_id`, `episode_id`, `step_id`, `span_id` and persist them on every record.

### 3.4 `orchestrator/gate_service.py`

Phase B implements the **four MVP checks** before all six gates come online:

| MVP check | Maps to | Owner inside the harness |
|-----------|---------|--------------------------|
| Proposal schema validity | static contract check ([§3.1.1](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#31-gate-stages-ordered)) | `gate/static_checker.py` |
| Evidence-interface validity | G0 ([PLAN-HARNESS.md §10](../05-harness/PLAN-HARNESS.md#10-promotion-gates)) | `SkillHarness.finalize_episode` |
| Replay pass / fail | G3 | `harness/replay_validator.py` *(stub in MVP, full in Phase D)* |
| Non-regression threshold | G5 | `harness/eval_harness.py` over frozen eval suite |

The full six-gate `GateRunner` ([PLAN-HARNESS.md §10b](../05-harness/PLAN-HARNESS.md#10b-gate-execution-runtime)) lands in Phase D; the MVP gate service forwards the same `GateVerdictPayload` shape so callers do not change.

### 3.5 `orchestrator/promotion_orchestrator.py` and `rollback_manager.py`

Implement the [PLAN-PIPELINE-ORCHESTRATOR.md §3a](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a-promotion-transaction-and-rollback-protocol) transactions verbatim:

```python
class PromotionOrchestrator:
    def evaluate_candidate(self, skill_id: str)         -> SkillEvaluationRecord: ...
    def promote_if_passed(self, skill_id: str)          -> bool: ...
    def rollback_if_needed(self, skill_id: str)         -> bool: ...
    def batch_evaluate_candidates(
        self, candidate_ids: list[str]
    ) -> list[SkillEvaluationRecord]: ...
```

Promotion rule (atomic, all-or-nothing):

1. Read latest `SkillEvaluationRecord`; require `final_decision ∈ {PASS, LIMITED_PASS}`.
2. `snapshot_manager.create(parent=current_production)`.
3. `SkillLifecycleManager.mark_provisional` / `promote_active`.
4. Atomically advance `current_production` pointer.
5. Emit signed `AuditRecord`.
6. Notify gate dashboard + (on `LIMITED_PASS`) Crafter failure-cluster export.

Rollback symmetric per §3a.4. **No partial pointer move ever**.

### 3.6 `orchestrator/budget.py`

Per [PLAN-PIPELINE-ORCHESTRATOR.md §7](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#7-budget-controller). One `BudgetController` per episode + nested per inner MDP. The `degrade` verdict must map to **explicit** logged behaviors, never silent omission.

### 3.7 Phase B acceptance criteria (Orchestrator MVP)

Stop Phase B only when **all** of the following are true:

1. Every episode is assigned a `run_id` and an `episode_id`; every record carries them.
2. `EpisodeTrace`, `SkillEpisode`, `RewardRecord` are persisted under the §2.3 storage layout.
3. Crafter proposals go through `gate_service` before any snapshot change. *(MVP gates: schema / evidence / replay-stub / non-regression.)*
4. `promote_if_passed` creates a new snapshot atomically and moves `current_production` only on success.
5. `rollback_if_needed` can restore the previous snapshot atomically; emits an `AuditRecord` with the trigger chain.
6. No Crafter or Harness code path can mutate `current_production` directly.

---

## 4. Skill Crafter (build third)

Start as a **slow proposal service**, not a large autonomous subsystem. The 32B/72B teacher is **frozen first**; outputs are typed proposals subject to the orchestrator gate ([PLAN-SKILL-CRAFTER.md §2](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)).

### 4.1 Scope for v1

Implement only three creation modes ([PLAN-SKILL-CRAFTER.md three creation modes](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)):

| Mode | What it does | First implementation budget |
|------|--------------|-----------------------------|
| **Composer** | combine 2+ existing skills into a compound skill | full implementation |
| **Generalizer** | adapt one skill to another domain via shared schema slots | full implementation |
| **Hypothesizer-lite** | propose a new skill **only after** repeated failure patterns recur | minimal — single-pass proposal, no Best-of-N yet |

`PatchProposal` and `RetireProposal` are typed but not actively generated in v1; they exist so the orchestrator's gate plumbing is type-complete.

### 4.2 `crafter/proposal_types.py`

Carry the evidence-driven invariant on every proposal (per the Revision Note in [PLAN-EDITS-HARNESS-CONTROL-PLANE.md](../legacy/10-edits/PLAN-EDITS-HARNESS-CONTROL-PLANE.md) and [PLAN-SKILL-CRAFTER.md §2.5](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)):

```python
class BaseProposal(BaseModel):
    proposal_id: str
    source_episode_ids: list[str]
    source_skill_ids: list[str]
    target_domains: list[str]                                # must be subset of the five general domains
    evidence_role: Literal["GATHER", "VERIFY", "REASON", "COMMIT"]
    evidence_interface: dict                                  # required + produced evidence kinds
    rationale: str
    diagnostics: list[str] = []

class ComposeProposal(BaseProposal):
    component_skill_ids: list[str]
    composition_protocol: list[dict]
    expected_effects: list[dict]

class GeneralizeProposal(BaseProposal):
    source_skill_id: str
    source_domain: str | None = None     # must be in SOURCE_DOMAINS when set (currently {"gymv"})
    target_domain: str                    # must be in TRANSFER_TARGET_DOMAINS when source_domain set
    slot_mapping: dict                    # legacy in-bank rewrite
    slot_remap: dict | None = None        # few-shot recipe: source-slot → target-slot
    demo_episode_ids: list[str] = []      # K demos consumed by FewShotAdapter
    demo_selection: Literal["random", "diverse", "hardest", "manual"] = "diverse"
    k_shot_budget: int | None = None      # ≤ few_shot.k_shot_max; defaults to few_shot.k_shot_default
    adapter_proposal: dict | None = None
    # When source_domain + target_domain + slot_remap + demo_episode_ids are all set,
    # the proposal becomes a *few-shot transfer recipe* and source_type resolves to
    # SkillSourceType.FEW_SHOT_ADAPTED. Otherwise it falls back to legacy
    # SkillSourceType.TRANSFERRED. See PLAN-SKILL-CRAFTER §4 and
    # PLAN-UNIFIED-SKILL-GATE §7 (Stage 3a).

class HypothesisProposal(BaseProposal):
    failure_cluster_id: str
    proposed_protocol: list[dict]
    expected_effects: list[dict]

class RetireProposal(BaseProposal):
    skill_id: str
    reason: Literal["superseded", "evidence_starved", "non_regression_fail", "g0_chronic", "other"]
```

### 4.3 `crafter/failure_trace.py`

```python
class FailureTrace(BaseModel):
    run_id: str
    episode_id: str
    skill_episode_id: str | None = None
    root_cause_step: int | None = None
    failure_type: str
    state_before: dict
    state_after: dict
    trace: list[dict]
    diagnostics: list[str]
```

Constructed from `SkillEpisode` records with `outcome ∈ {"fail", "abort", "stall"}` plus the orchestrator's surrounding trace. The Harness already emits the diagnostic labels listed in [PLAN-HARNESS.md §10a](../05-harness/PLAN-HARNESS.md#10a-transfer-failure-diagnostics-domain-specific) — pass them through verbatim.

### 4.4 `crafter/failure_memory.py`

**Crafter-private.** This is *not* a system memory and is *never* read by the online actor — the repo is no-memory for the current short-video execution target.

```python
class FailureMemory(BaseModel):
    entries: list[FailureDiagnosis]
    recurrence_counts: dict          # failure_type → count
    cross_skill_patterns: dict
    recovery_effectiveness: dict
    high_regret_patterns: dict = {}
```

Persisted under `artifacts/crafter/failure_memory/` (separate from `bank/` and `runs/`) so its read scope is enforceable by directory permissions.

### 4.5 `crafter/service.py`

Single entrypoint:

```python
class SkillCrafterService:
    def propose_from_failures(self, batch: list[FailureTrace]) -> list[BaseProposal]: ...
    def propose_from_bank(self, skills: list[SkillSpec]) -> list[BaseProposal]: ...
```

### 4.6 Crafter execution policy (when `service.py` is invoked)

Triggers, in priority order, mirror [PLAN-SKILL-CRAFTER.md §2 cadence](../04-skill-crafter/PLAN-SKILL-CRAFTER.md):

1. After repeated failure clusters (recurrence threshold in `failure_memory.recurrence_counts`).
2. Periodically every N episodes for composition / generalization sweeps.
3. When a new domain adapter appears in `AdapterRegistry`.
4. Before cold-start trajectory generation (one-shot, gated).

The teacher is **frozen** — no fine-tuning in v1. Improvement happens through better input distribution, evidence organization, replay validation, and verification ([PLAN-ACTION-AGENT.md §6 frozen-teacher channels](../02-action-agent/PLAN-ACTION-AGENT.md#frozen-teacher-improvement-channels)).

### 4.7 Phase C acceptance criteria (Crafter MVP)

Stop Phase C only when **all** of the following are true:

1. `SkillCrafterService.propose_from_failures` and `propose_from_bank` produce well-typed `ComposeProposal` / `GeneralizeProposal` / `HypothesisProposal` instances.
2. Every proposal carries `evidence_role`, `evidence_interface`, and `target_domains` ⊆ the five general domains.
3. Proposals are written to `artifacts/bank/proposals/{proposal_id}/` and routed to `gate_service` — never directly into `bank/snapshots/`.
4. `FailureMemory` persists under `artifacts/crafter/` and is **not** importable from the online actor or the harness modules.
5. The frozen teacher is invoked only from `crafter/`; no other module imports the 72B client.

### 4.8 What stays out of the Crafter

- No online execution.
- No live slot binding.
- No promotion decision.
- No direct snapshot mutation.

---

## 5. Phase D — Transfer & replay (after MVP)

Only after Phases A–C are green:

1. **`harness/transfer_manager.py`** — two-phase shadow → active transfer protocol per [PLAN-HARNESS.md §6](../05-harness/PLAN-HARNESS.md#6-two-phase-transfer-protocol).
2. **`harness/replay_validator.py`** — held-out replay checks for G3.
3. **`harness/gate_runner.py`** — full six-gate runner per [PLAN-HARNESS.md §10b](../05-harness/PLAN-HARNESS.md#10b-gate-execution-runtime), replacing the Phase B MVP gate service.
4. **Shadow-only transfer first** — `run_shadow` produces `SkillEpisode(shadow=True)` records that flow through gates G0–G5 but never affect `r_env` or `current_production`.
5. **Active promotion** — only after the shadow + non-regression metrics are stable for K cycles.

This sequence matches [PLAN-HARNESS.md §14 Phases 2–4](../05-harness/PLAN-HARNESS.md#14-phased-implementation-plan).

---

## 6. Concrete implementation order (the Cursor sequence)

| Phase | Goal | Files added | Stop condition |
|-------|------|-------------|----------------|
| **A — Harness MVP** | one path for every skill invocation | `harness/skill_episode.py`, `harness/adapter_registry.py`, `harness/skill_harness.py`, `harness/reward_logger.py`, `harness/policies.py`, `harness/adapters/{gymv,browser}.py` | §2.5 acceptance criteria |
| **B — Orchestrator MVP** | one control plane, atomic snapshots | `orchestrator/schemas.py`, `orchestrator/artifact_store.py`, `orchestrator/runner.py`, `orchestrator/gate_service.py`, `orchestrator/promotion_orchestrator.py`, `orchestrator/snapshot_manager.py`, `orchestrator/rollback_manager.py`, `orchestrator/budget.py`, `orchestrator/config.py` | §3.7 acceptance criteria |
| **C — Crafter MVP** | typed, gated, frozen-teacher proposals | `crafter/proposal_types.py`, `crafter/failure_trace.py`, `crafter/failure_diagnoser.py`, `crafter/failure_memory.py`, `crafter/composer.py`, `crafter/generalizer.py`, `crafter/hypothesizer.py`, `crafter/service.py` | §4.7 acceptance criteria |
| **D — Transfer + Replay** | shadow-first cross-domain transfer | `harness/transfer_manager.py`, `harness/replay_validator.py`, `harness/gate_runner.py`, additional adapters (`osworld`, `video`, `visual_reasoning`) | shadow pass rate ≥ threshold for K cycles, then enable active promotion |
| **E — Eval + dashboards** | measurable reuse / transfer | `orchestrator/eval_suite.py`, `orchestrator/eval_driver.py`, dashboards for §6.4 slices and §10a label distributions | metrics from [PLAN-HARNESS.md §15](../05-harness/PLAN-HARNESS.md#15-metrics) and [PLAN-PIPELINE-ORCHESTRATOR.md §6](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#6-evaluation-matrix) reproducible per `run_id` |
| **F — Trainable extensions (optional)** | learned skill-use decisions | `skill_select`, `continue_vs_switch`, `accept_transfer`, `adapter_refine` LoRAs | measurable gain over rule-based baselines |

Phases A → B → C → D → E → F is strict; do not start a phase before its predecessor passes.

---

## 7. Required invariants (apply to every phase)

1. **No proposal reaches production without a gate pass.** Crafter outputs are candidates only ([PLAN-PIPELINE-ORCHESTRATOR.md §1.4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#14-dag-invariants)).
2. **Replay before promote** for any proposal affecting `skill_select` / `hop_select` behavior.
3. **Evidence-driven invariant.** `SkillHarness.finalize_episode` enforces G0; opaque-skill / role-mismatch episodes are not promotable regardless of reward.
4. **Atomic promotion / rollback.** Either `current_production` references the new state and the audit record is written, or nothing changes.
5. **Crafter is candidate-only.** No direct bank writes.
6. **Harness is invocation-only.** No skill invention, no snapshot mutation, no final policy choice.
7. **Orchestrator is system-only.** No per-invocation execution, no skill authoring.
8. **Frozen teacher.** Only `crafter/` imports the 72B client. Only proposals exit; no proposal bypasses gates.
9. **Episode-local state surface.** No cross-episode storage layer ([PLAN-PIPELINE-ORCHESTRATOR.md §4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping)).
10. **Actor is the online policy.** Harness produces `eligible_skills` and may veto; Actor decides ([PLAN-ACTION-AGENT.md §1a](../02-action-agent/PLAN-ACTION-AGENT.md#1a-actor-role-and-boundary)).

---

## 8. Cursor prompt (paste-ready)

The prompt below is engineered to be dropped into Cursor verbatim. It encodes the boundaries from §0, the order from §6, and the invariants from §7.

````
Implement three separate modules: `harness/`, `crafter/`, and `orchestrator/`.
Follow the canonical specs:
  - plans/05-harness/PLAN-HARNESS.md
  - plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md
  - plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md
  - plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md
  - plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md  (this file — order + acceptance criteria)

Architectural boundaries (must hold):
- Skill Harness = per-invocation runtime only. It owns retrieval, slot binding,
  adapter attachment, execution, tracing, `SkillEpisode`, and shadow/replay
  validation hooks. It must NOT invent new skills or mutate production bank
  snapshots, and it must NOT make the final skill choice — that belongs to the
  Actor (PLAN-ACTION-AGENT.md §1a).
- Skill Crafter = slow-timescale proposal layer only. It owns failure analysis,
  composition, generalization, hypothesis generation, and typed proposal outputs.
  It must NOT execute skills online or decide promotion.
- Pipeline Orchestrator = system control plane only. It owns episode runs,
  artifact IDs / storage, acceptance gates, promotion/rollback transactions,
  train scheduling, and re-evaluation. It must NOT do invocation-local execution
  or author skill contents.

Implement strictly in this order. Do not start a phase before the previous
phase's acceptance criteria (see PLAN-COMPONENTS-IMPLEMENTATION.md) are green.

Phase A — Harness MVP
  - Add `harness/skill_episode.py`     (PLAN-HARNESS §5.1, with evidence-role enforcement in finalize_episode)
  - Add `harness/adapter_registry.py`  (PLAN-HARNESS §5.3)
  - Add `harness/skill_harness.py`     (PLAN-HARNESS §5.2)
  - Add `harness/reward_logger.py`     (PLAN-HARNESS §5.6)
  - Add `harness/policies.py`          (ranking + shadow-origin penalty)
  - Add `harness/adapters/gymv.py` and `harness/adapters/browser.py`
  - Route ALL current `ActionAgent` skill invocation through `SkillHarness`
  - Log one `SkillEpisode` per invocation
  - The Actor receives `eligible_skills` from the Harness; it does NOT call the
    Skill Bank directly.

Phase B — Orchestrator MVP
  - Add `orchestrator/schemas.py`      (every record type in PLAN-PIPELINE-ORCHESTRATOR §2.2)
  - Add `orchestrator/artifact_store.py`   (layout from §2.3, idempotent writes)
  - Add `orchestrator/runner.py`           (run_episode + run_offline_cycle)
  - Add `orchestrator/gate_service.py`     (MVP: schema / evidence / replay-stub / non-regression)
  - Add `orchestrator/promotion_orchestrator.py` and `rollback_manager.py`
       (atomic transactions per PLAN-PIPELINE-ORCHESTRATOR §3a)
  - Add `orchestrator/snapshot_manager.py`
  - Add `orchestrator/budget.py`           (PLAN-PIPELINE-ORCHESTRATOR §7)
  - Persist artifacts under `artifacts/runs/{run_id}/...` and `artifacts/bank/...`

Phase C — Crafter MVP
  - Add `crafter/proposal_types.py`    (ComposeProposal, GeneralizeProposal,
                                         HypothesisProposal, RetireProposal —
                                         each with evidence_role + evidence_interface)
  - Add `crafter/failure_trace.py`
  - Add `crafter/failure_diagnoser.py`
  - Add `crafter/failure_memory.py`    (Crafter-private; persisted under
                                         `artifacts/crafter/failure_memory/`;
                                         NOT importable from harness/ or actor)
  - Add `crafter/composer.py`, `crafter/generalizer.py`, `crafter/hypothesizer.py`
  - Add `crafter/service.py`
  - Wire Crafter outputs into the Orchestrator gate.
  - Keep the 32B/72B teacher frozen. Only `crafter/` imports the 72B client.

Required invariants (verify before marking any phase done):
  1. No proposal reaches production without a gate pass.
  2. Replay before promote for any proposal affecting skill behavior.
  3. SkillHarness.finalize_episode enforces the evidence-role contract (G0).
  4. Promotion creates a new snapshot atomically; rollback restores previous
     snapshot atomically. No partial pointer move.
  5. Crafter outputs are candidates only; never written directly into the bank.
  6. Harness never invents skills, never mutates snapshots, never selects the
     final skill.
  7. Orchestrator never executes per-invocation logic and never authors skills.
  8. Episode-local state only — no cross-episode storage layer.

Use pydantic models, clean interfaces, and unit-testable pure functions where
possible. Every public method should have a docstring linking to the canonical
spec section in the relevant PLAN-*.md.
````

---

## 9. Honest simplification advice

The best simplification is **not** to merge Crafter and Harness. The better simplification is:

- keep **Harness** strong and narrow — it is the unit of runtime verification and attribution;
- keep **Crafter** small and slow at first — it earns scope by proving its proposals survive the gate;
- keep **Orchestrator** minimal but **authoritative** — it is the only place that mutates `current_production`.

That fits the existing plans: the Harness is the micro runtime, the Crafter is the proposal layer, and the Orchestrator is the macro DAG and gate owner. Any temptation to fold one into another should be resisted; the four-way boundary in [PLAN-PIPELINE-ORCHESTRATOR.md §0a](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#0a-actor-harness-skill-bank-orchestrator-boundary) exists precisely to prevent that drift.

---

## 10. Related documents

| Document | Relationship |
|----------|--------------|
| [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) | Canonical Harness design — record types, six gates, two-phase transfer |
| [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) | Canonical Crafter design — creation modes, typed proposals, failure memory |
| [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) | Canonical Orchestrator design — DAG, artifact schema, promotion/rollback transactions, four-way boundary |
| [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) | Canonical lifecycle + ownership split (`SkillLifecycleManager`, `GateRunner`, `PromotionOrchestrator`) and storage split |
| [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) | Skill object, retrieval API, lineage / negative-knowledge fields read by the Harness |
| [PLAN-ACTION-AGENT.md §1a](../02-action-agent/PLAN-ACTION-AGENT.md#1a-actor-role-and-boundary) | Actor consumes `eligible_skills`, remains the online policy |
| [PLAN-EDITS-HARNESS-CONTROL-PLANE.md](../legacy/10-edits/PLAN-EDITS-HARNESS-CONTROL-PLANE.md) | Edit-plan that aligned existing plan files for the evidence-driven invariant |

---

*This document specifies **how** to build the three components, in what order, with which acceptance criteria. The **why** and **what** live in the canonical PLAN-* documents linked above; this plan does not duplicate them.*
