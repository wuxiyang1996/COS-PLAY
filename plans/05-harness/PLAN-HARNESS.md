# PLAN: Skill Harness — Unified Runtime for Skill Use, Validation, and Transfer

> **Lane decision (2026-05-01) — lane (a), Context-only skills.** In
> the live trainer the Harness is an **eligibility filter and
> validator**, *not* a skill executor. The Day-10 trainer integration
> calls only `harness.select_eligible_skills(...)` and
> `harness.validate_invocation(...)` — never `harness.run_skill(...)`.
> Authoritative record: [`implementation_notes/legacy/skill-lane-decision.md`](../../implementation_notes/legacy/skill-lane-decision.md);
> see also [`harness/README.md` §22](../../harness/README.md).
> Practical implications for this plan:
>
> * **§16 (Skill executor / `run_skill` / inner-MDP dispatch) and the
>   typed-hop dispatch surface stay in tree as offline diagnostic
>   infrastructure**, not as the live runtime substrate. They are
>   exercised only by `labeling_supplement/_phase4_transfer_cycle.py`
>   and `labeling_supplement/_phase2_real_env_skill_smoke.py`, plus
>   the offline `GateRunner` (Stage-1 replay, Stage-2 shadow).
> * The §1 framing "skills are *executable units*" describes the
>   **offline gate's view** of a skill, not the live actor's view. The
>   live actor consumes a skill as **prompt context** (one env action
>   per LLM call) and emits one `SkillEpisode` per skill consultation.
> * The §11 relationship "the orchestrator calls the Harness at every
>   `inner_mdp` step where a skill is invoked" is **superseded**: there
>   is no inner-MDP step in the live trainer (single-MDP companion
>   decision, T3.6 — see
>   [`implementation_notes/legacy/single-vs-two-mdp-tradeoff.md`](../../implementation_notes/legacy/single-vs-two-mdp-tradeoff.md)).
>   The trainer calls the Harness once per actor decision, as filter +
>   validator only.
> * **`SkillEpisode` is still emitted** — once per skill consultation,
>   with `evidence_in / evidence_out / role` slots populated. The G0
>   evidence-driven invariant is preserved (set-containment of
>   `expected_evidence_roles ⊆ state.evidence`).
> * **Replay validation, few-shot adaptation (G3a), and shadow gating**
>   remain the *gate*'s evidence base for promotion decisions — they
>   produce `verified_tasks` entries that the live eligibility filter
>   then consults via `feasible_tasks` / F2′. The promotion machinery
>   (`PromotionOrchestrator`) is the only path that flips a record's
>   lifecycle status; the Harness never promotes.
>
> Sections below were authored under the lane-(b) assumption that the
> Harness is also the live executor. Treat invocation / dispatch
> language as the *offline* surface unless a section is explicitly
> tagged "live."

**Scope:** Define a **Skill Harness** that sits on top of the existing framework as a unified runtime orchestration layer for skill retrieval, binding, execution, validation, and cross-domain transfer. The Harness is **not a new agent** — it is a thin orchestration layer that wraps the [Action Agent](../02-action-agent/PLAN-ACTION-AGENT.md), the [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md), and the [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md), and provides a single entry point for every skill invocation in the system.

**Problem statement:** Today, skills live in the Skill Bank and are called by the Action Agent, but there is no shared execution surface that (a) normalizes state into slots, (b) attaches domain-specific adapters, (c) records standardized execution traces, and (d) gates transferred skills behind replay + shadow validation. Without this layer, skill transfer risks destabilizing rollouts, cross-domain metrics are inconsistent, and there is no uniform reward signal for later GRPO on skill-use decisions.

**Upstream:** Canonical `<state>` schema ([README § Canonical `<state>`](../README.md)); shared slot names (`target`, `blocker`, `constraint`, `candidate_set`, `history_anchor`); three-agent role split ([Action Agent §2](../02-action-agent/PLAN-ACTION-AGENT.md)); Skill Bank query/select API ([Skill Bank §6](../03-skill-bank/PLAN-SKILL-BANK.md)); Skill Crafter transfer proposals ([Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)).

**Downstream:** [Pipeline Orchestrator](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) acceptance gates (Harness emits `SkillEpisode` records consumed by the orchestrator's gate logic); unified reward signals for GRPO; evaluation harness for reuse and transfer benchmarks.

**Relation to Pipeline Orchestrator.** The Pipeline Orchestrator is the **system-level DAG** that runs grounding → action → bank → crafter → gates across many episodes and runs. The Skill Harness is the **per-invocation runtime** for a single skill call. Pipeline Orchestrator = macro scheduler; Skill Harness = micro runtime for skill use. They compose: the orchestrator calls the Harness at every `inner_mdp` step where a skill is invoked.

**Non-goals:** Replacing the Action Agent, Skill Bank, or Skill Crafter. Introducing a fourth agent. Making the 32B/72B teacher the default online controller. Adding new trainable models before the execution + validation loop works. **Narrowing the Harness to a single domain, or admitting domain-specific skills.** Every skill the Harness binds, runs, and validates is a **general protocol feasible across all five target domains** (game / webagent / os-agent / video-understanding / visual reasoning) — see [Skill Bank §0.1](../03-skill-bank/PLAN-SKILL-BANK.md#01-general-protocol-invariant-no-domain-specific-skill-families). The Harness is the *domain-general transfer runtime*; short-video evidence-grounded reasoning is the first proving ground where that broad transfer is *validated*, not the definition of the Harness's scope.

**Episode-local state surface.** The Harness reads skills from the bank and reads its episode-local trajectory — current `<state>`, short typed hop trace, intermediate belief state, and within-episode evidence references — directly from the orchestrator (see [Pipeline Orchestrator §4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping)). It maintains no other lookup channel.

---

## 1. Goal

Make skills **executable units** — not static objects in the bank — that can be:

- retrieved,
- bound to the current task / domain state,
- executed with trace logging,
- validated before promotion,
- reused across tasks,
- transferred across domains safely.

The Harness should become the **default path** for all skill invocation and should produce standardized execution records (`SkillEpisode`) that feed the acceptance gates ([Pipeline Orchestrator §3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)) and the reward logger.

---

## 1a. Harness Role as Frozen 72B Runtime Layer

In this project, the Harness is a **frozen 72B model** wrapped by the orchestration layer described above. It should **not** replace the [Action Agent](../02-action-agent/PLAN-ACTION-AGENT.md) as the online policy. Instead, it should serve as a high-capacity **runtime verifier, candidate filter, veto layer, and teacher-like advisor**.

Its role is to make skill usage safer, more executable, and more transferable at runtime, **without taking over the final policy decision**.

This distinction is critical. If the Harness directly makes final online skill choices, then the system effectively becomes a 72B-driven policy with an 8B execution shell. That would weaken the role of the Actor and blur the architectural story of the project.

### 1a.1 What the Harness should do

The Harness should be responsible for:

- **candidate filtering**
  - evaluate retrieved skills,
  - discard invalid or unsafe candidates,
  - keep only runtime-eligible skills.
- **binding validation**
  - check slot binding feasibility,
  - validate schema-to-skill alignment,
  - verify required arguments are grounded.
- **precondition checking**
  - determine whether the candidate skill is actually applicable now.
- **evidence and contract checking**
  - verify that the required evidence interface is available,
  - ensure invocation satisfies runtime evidence requirements ([§5.1 Evidence-role field requirements](#51-skillepisode)),
  - check protocol-level constraints.
- **adapter / transfer feasibility**
  - verify domain adapter compatibility,
  - reject cross-domain invocation when mapping is invalid.
- **runtime veto**
  - reject an Actor-proposed skill if binding, evidence, precondition, or runtime safety fails.
- **advisory scoring**
  - provide fit score,
  - provide risk score,
  - provide evidence sufficiency score,
  - optionally rank eligible candidates.

### 1a.2 What the Harness should not do

The Harness should **not** become the final online policy. It should not directly decide:

- whether the system should definitely continue the current skill,
- whether the Actor must switch skill,
- whether no-skill mode is preferable,
- whether a reasoning step should be emitted,
- which primitive action should be taken.

Those remain Actor decisions ([PLAN-ACTION-AGENT.md §1a.2](../02-action-agent/PLAN-ACTION-AGENT.md#1a2-actor-decision-scope)).

The Harness may **advise, filter, rank, or veto**, but it should not fully replace policy-level choice.

### 1a.3 Harness output contract

The Harness should transform raw retrieved candidates into a constrained **eligible set**:

```python
eligible_skills = [
    {
        "skill_id": str,
        "binding_ok": bool,
        "precondition_ok": bool,
        "evidence_ok": bool,
        "adapter_ok": bool,
        "fit_score": float,
        "risk_score": float,
        "veto": bool,
        "veto_reason": str | None,
    },
    ...
]
```

This output is then consumed by the Actor.

The key design principle is:

> **The Harness narrows the choice space; the Actor makes the final choice.**

### 1a.4 Actor proposal, Harness veto

At invocation time, the interaction should follow:

1. Actor proposes:
   - continue current skill,
   - switch to skill X,
   - no skill,
   - reasoning step,
   - primitive action.
2. If the Actor proposes a skill invocation, the Harness validates:
   - binding,
   - preconditions,
   - evidence,
   - adapter compatibility,
   - runtime constraints.
3. If validation passes, execution continues.
4. If validation fails, the Harness returns a veto reason, and the Actor must fallback to:
   - another eligible skill,
   - no-skill mode,
   - a reasoning step,
   - or a primitive action.

This gives the system a clear control pattern:

> **Actor proposes; Harness constrains or vetoes.**

### 1a.5 Why the frozen 72B Harness should not replace the Actor

Because the Harness is frozen and high-capacity, it is attractive to let it choose skills directly. However, doing so would create several problems:

- the trainable Actor would stop learning core skill-use policy,
- the Harness would become a hidden policy model,
- reasoning-step choice and skill choice would become fragmented,
- the architecture would drift away from the intended COS-PLAY-style Decision Agent ([PLAN-ACTION-AGENT.md §1a](../02-action-agent/PLAN-ACTION-AGENT.md#1a-actor-role-and-boundary)),
- the final system would rely too heavily on the frozen large model.

Therefore, the Harness should remain a **runtime support and verification layer** rather than the main policy.

### 1a.6 Harness as teacher-like advisor

The Harness may still produce strong **advisory signals**, such as:

- ranked eligible skills,
- top-1 recommended skill,
- invocation risk,
- evidence sufficiency warnings,
- transfer confidence,
- binding confidence.

These signals can be given to the Actor as extra inputs.
However, they should remain **advisory rather than fully controlling** — the trainable Actor decides whether to follow them.

---

## 2. Design principle: semantic skill vs. domain adapter

The Harness enforces a strict separation between the **semantic skill** (domain-agnostic, shared across tasks) and the **domain adapter** (task-specific execution layer).

### 2.1 Semantic skill (shared, reusable)

The portable part of a skill:

- `summary`
- `preconditions`
- `plan` / `protocol` (inner-hop chain over shared primitives)
- `success_criteria`
- `abort_criteria`
- `contract` / expected effects
- `evidence_requirements`

This is what lives in the [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) and is subject to composition / generalization by the [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md).

### 2.2 Domain adapter (task-specific)

The execution layer that maps abstract skill steps into concrete operations in:

- games (Gym-V)
- web (BrowserGym)
- desktop (OSWorld)
- video reasoning
- visual grounding / reasoning

Adapters are owned by the `AdapterRegistry` (§5.3) and looked up per `(skill_type, target_domain)`.

### 2.3 Transfer rule

Transfer is **not** "copy a skill end-to-end." Transfer is:

```
reuse semantic skill   +   rebind or synthesize target-domain adapter
```

This is the single most important invariant the Harness enforces.

---

## 3. Responsibilities

The Harness is responsible for exactly five things. Anything beyond this list belongs in the existing modules.

| # | Responsibility | Input | Output |
|---|----------------|-------|--------|
| 1 | **State normalization** | raw env / task state | shared structured state (typed slots) |
| 2 | **Skill retrieval + ranking** | normalized state, goal | ranked candidate skill list |
| 3 | **Skill binding + execution** | semantic skill + adapter | per-step actions + updates |
| 4 | **Transfer validation** | transferred skill candidate | replay verdict + shadow verdict |
| 5 | **Logging + reward hooks** | execution trace | `SkillEpisode` + reward signals |

---

## 4. Minimal runtime flow

The Harness wraps the actor loop:

```
raw state
  → normalize_state               (§5.2 / §5.3)
  → retrieve_candidates           (Skill Bank §6)
  → rank_candidates               (§7)
  → bind_skill                    (slot binding)
  → attach_adapter                (AdapterRegistry §5.3)
  → execute_step                  (protocol-aware)
  → update_episode                (trace / evidence / contract)
  → continue | switch | abort     (budget-aware)
  → finalize_episode              (reward + metrics)
  → return action(s) + reward signals
```

This becomes the **default path** for all skill usage in the Action Agent and the `skill_select` inner action path.

---

## 5. New core abstractions

### 5.1 `SkillEpisode`

Standardized execution record — one per skill invocation. This is the unit of evidence the acceptance gates consume.

```python
class SkillEpisode:
    episode_id: str
    step_id_start: int
    step_id_end: int | None
    skill_id: str
    skill_version: str
    skill_type: str              # reasoning | action | grounding | mixed
    evidence_role: str           # GATHER | VERIFY | REASON | COMMIT
                                 # must match the declared evidence_role of the skill in
                                 # the bank; see PLAN-SKILL-BANK §0.3 Clause B
    source_domain: str
    target_domain: str
    adapter_id: str | None
    slot_bindings: dict          # typed slots → concrete entity IDs
    protocol_trace: list         # inner-hop sequence actually executed

    # ── Evidence-driven interface (PLAN-SKILL-BANK §0.3 Clause A) ──
    # Opaque episodes (evidence_in ∪ evidence_out == ∅ and no evidence_warrant)
    # are rejected at Gate G0 regardless of reward.
    evidence_in:       list        # EvidenceRef list read from <state>.evidence_refs or prior inner hops
    evidence_out:      list        # EvidenceRef list written by this skill (grounding, verdicts, hypotheses)
    evidence_warrant:  list | None # required and non-empty iff evidence_role == COMMIT
    verify_verdict:    str | None  # PASS | FAIL | INSUFFICIENT; required iff evidence_role == VERIFY
    reason_warrant:    list | None # subset of evidence_in cited as warrant; required iff evidence_role == REASON
    # ─────────────────────────────────────────────────────────────

    evidence_trace: list         # chronological grounding / tool-call evidence (superset of evidence_in/out)
    contract_progress: dict      # per-effect progress in [0,1]
    outcome: str                 # success | fail | abort | stall
    abort_reason: str | None
    reward_components: dict      # r_env, r_follow, r_cost, transfer bonus
    reward_delta: float
    shadow: bool                 # True if running in shadow mode
    metadata: dict               # seeds, adapter_version, model_tier used
```

Purpose:

- unify execution logging,
- support replay-based validation,
- measure transfer success,
- enforce the evidence-driven invariant (Gate G0) at promotion time,
- generate reward signals for later GRPO on `skill_select` / `continue_vs_switch` / `accept_transfer`.

**Evidence-role field requirements (checked at `finalize_episode` time).**
Any episode that violates these requirements is marked `outcome = fail` with `abort_reason = "opaque-skill-violation"` or `"skill-role-mismatch"` and is not eligible for promotion:

| `evidence_role` | must hold |
|-----|-----|
| `GATHER` | `evidence_out ≠ ∅` |
| `VERIFY` | `evidence_in ≠ ∅` and `verify_verdict ∈ {PASS, FAIL, INSUFFICIENT}` |
| `REASON` | `evidence_in ≠ ∅` and `reason_warrant ⊆ evidence_in`, `reason_warrant ≠ ∅` |
| `COMMIT` | `evidence_warrant ≠ ∅` |
| *any* | `evidence_in ∪ evidence_out ≠ ∅` (opacity precondition) |

### 5.2 `SkillHarness`

Main orchestration entry point. Should expose the following methods; anything else is internal.

```python
class SkillHarness:
    def normalize_state(self, raw_state) -> NormalizedState: ...
    def retrieve_candidates(self, state, goal, k: int) -> list[SkillCandidate]: ...
    def rank_candidates(self, state, candidates) -> list[RankedSkill]: ...
    def bind_skill(self, skill, state) -> BoundSkill | BindFailure: ...
    def attach_adapter(self, skill, target_domain) -> AdapterHandle | AdapterMiss: ...
    def execute_step(self, bound_skill, state, trace) -> StepResult: ...
    def update_episode(self, episode, step_result) -> None: ...
    def decide_continue_switch_abort(self, episode, budget) -> Decision: ...
    def finalize_episode(self, episode, outcome) -> SkillEpisode: ...

    # entry points
    def run_active(self, state, goal, budget) -> (Action, SkillEpisode): ...
    def run_shadow(self, state, goal, candidate) -> SkillEpisode: ...
```

Runtime responsibilities:

- normalize input state,
- call retrieval engine,
- rank candidates (§7),
- bind slots,
- attach adapter,
- execute current skill,
- monitor progress,
- decide continue / switch / abort,
- emit execution logs,
- trigger shadow validation for transferred skills.

### 5.3 `AdapterRegistry`

Registry of domain-specific adapters.

```python
class AdapterRegistry:
    def register(self, skill_type: str, domain: str, adapter) -> None: ...
    def get(self, skill_type: str, domain: str) -> Adapter | None: ...
    def validate(self, adapter, skill, state) -> AdapterVerdict: ...
    def request_synthesis(self, skill, domain) -> AdapterProposal: ...  # calls 72B
```

Initial domains: `gymv`, `browser`, `osworld`, `video`, `visual_reasoning`.

Responsibilities:

- register adapters per domain,
- retrieve adapter by `(skill_type, target_domain)`,
- validate adapter availability / syntactic sanity,
- optionally call the **slow teacher** (32B/72B, frozen) to refine or propose a new adapter.

### 5.4 Transfer subsystem (`TransferProposer` + `FewShotAdapter`)

Transfer **does not** happen inside the Skill Bank. It happens inside the Harness, because it requires execution-level evidence that only the Harness can produce. Under the source/target asymmetry ([PLAN-SKILL-BANK §0.4](../03-skill-bank/PLAN-SKILL-BANK.md#04-source-domain--transfer-target-asymmetry)) the responsibilities split into two cleanly separable units:

#### 5.4.1 `TransferProposer` (lifecycle)

Proposes transfer attempts and manages their lifecycle. Owned by the Crafter / Orchestrator side of the Harness.

```python
class TransferProposer:
    def propose_transfer(self, skill, target_domain) -> TransferProposal: ...
    def select_or_synthesize_adapter(self, skill, target_domain) -> Adapter: ...
    def dry_run_transfer(self, proposal, replay_slice) -> ReplayVerdict: ...
    def shadow_run_transfer(self, proposal, live_states) -> ShadowVerdict: ...
    def reject(self, proposal, reason) -> None: ...
```

Responsibilities: propose transfer, pick the right target adapter, run replay-based dry checks, run shadow-mode online checks. The proposer **never** writes `verified_domains` and never promotes — that path goes through the gate's Stage 3a (below) and the orchestrator's `PromotionOrchestrator`.

#### 5.4.2 `FewShotAdapter` (Stage 3a runtime)

The K-shot adaptation engine — this is the actual realisation of [PLAN-UNIFIED-SKILL-GATE Stage 3a](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md). Lives at `harness/few_shot_adapter.py`.

```python
class FewShotAdapter:
    def adapt(self, *, skill, target_domain, demos=(), k=None) -> AdaptResult: ...
    def adapt_many(self, *, skill, target_domains, demos_by_domain=None, k=None) -> List[AdaptResult]: ...
```

Responsibilities:

- Validate `(skill.source_domains ⊆ SOURCE_DOMAINS, target_domain ∈ TRANSFER_TARGET_DOMAINS)`.
- For each `(skill, target_domain)` pair, take up to `k_shot_max` `FewShotDemo`s, re-tag each demo's `<state>.domain` to the target, apply the proposal's `slot_remap`, and execute the skill through `SkillHarness.run_skill()` against the registered target adapter.
- Score each shot via a pluggable `success_fn` (default: `outcome.success ∧ outcome.contract_satisfied`).
- Honour the `adaptation_cost_max_tokens` budget; abort the run with diagnostic `few_shot_budget_exceeded` if exceeded.
- Return per-target `AdaptResult { k_used, pass_rate, n_success, n_total, aborted, cost_*, diagnostic_label, episode_ids }`.

The adapter is **stateless** across calls: it never mutates `SkillRecord`, never writes `verified_domains` itself, and never logs to the long-term artifact store. The `GateService._run_transfer` consumes the `AdaptResult`s and is the sole writer of `verified_domains`.

### 5.5 `ReplayValidator`

Offline validation over logged transitions / held-out state slices ([Pipeline Orchestrator §3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)).

Checks:

- slot bindability on historical states,
- preconditions satisfied,
- expected effects match observed deltas,
- protocol does not contradict observed transitions,
- evidence-trace coherence.

### 5.6 `RewardLogger`

Central place for all skill-use reward signals. Existing components should not write reward directly anymore; they write through this.

Emits:

- `r_env` — environment reward
- `r_follow` — protocol following reward
- `r_cost` — latency / token / adapter cost penalty
- `r_transfer` — transfer success metrics
- `r_adapter` — adapter validation signals

---

## 6. Two-phase transfer protocol

This is the most load-bearing part of the plan. Transferred skills must never be allowed into the active policy before passing both phases.

### 6.1 Phase A — Shadow mode

Transferred skills **can** be:

- retrieved,
- slot-bound,
- adapter-attached,
- evaluated against current states and observed transitions.

Transferred skills **cannot**:

- control the active actor policy,
- change the environment,
- affect `r_env` or any reward that feeds training.

Shadow-mode checks per step:

- are slots bindable?
- are preconditions satisfied?
- does the protocol make sense in the target domain (type-check over primitives)?
- do expected effects align with observed transitions?
- is the evidence trace coherent?

### 6.2 Phase B — Active mode

Only after a transferred skill passes the promotion gates (§10) may it enter active policy execution.

Rationale: protect rollout quality while the bank and adapters are still evolving, and avoid feedback loops where a bad transferred skill pollutes the experience buffer that trains the actor.

---

## 7. Retrieval and ranking update

Current skill selection likely ranks by relevance and pass rate. Transfer-aware ranking must be added so transferred skills are not prematurely chosen on semantic similarity alone.

### 7.1 Suggested scoring

```
score =
    0.35 * retrieval_relevance
  + 0.25 * applicability            # preconditions satisfiable on current state
  + 0.20 * historical_pass_rate
  + 0.10 * adapter_validity
  + 0.10 * slot_binding_confidence
```

Weights are initial defaults; they should be tunable and logged per `run_id`.

### 7.2 Shadow-origin penalty

Skills that have not yet cleared §10 promotion gates receive a hard cap in `run_active` ranking; they are still retrievable in `run_shadow` at full weight.

---

## 8. Unified skill interface

Action skills, reasoning skills, and grounding skills must share a single interface so the Harness can treat them uniformly.

```python
class Skill:
    def preconditions(self, state) -> bool: ...
    def bind(self, state) -> SlotBindings | BindFailure: ...
    def step(self, state, trace) -> StepAction: ...
    def expected_effects(self) -> list[Effect]: ...
    def evidence_requirements(self) -> list[EvidenceSpec]: ...
    def get_adapter(self, domain) -> Adapter | None: ...
```

This lets the framework treat reasoning skills, grounding skills, and action skills as **different types of the same object**. It is a prerequisite for cross-domain transfer and for unified GRPO reward shaping.

---

## 9. Where 7B/8B and 72B fit

The Harness is **not** a 72B-only inference layer. Model assignment follows the [three-agent role split](../README.md#three-agent-role-split--model-convention).

### 9.1 Fast loop — 7B/8B (Qwen3-8B)

Default model for:

- online state interpretation,
- skill selection,
- step-level execution,
- short reasoning traces,
- rollout-time continue/switch/abort decisions.

### 9.2 Slow loop — 32B/72B (frozen teacher)

Escalation only, used for:

- transfer proposal review,
- adapter refinement / synthesis,
- protocol rewriting,
- failure reflection,
- merge / split critique,
- hard-case judgment.

The Harness must default to 7B/8B and escalate to the teacher only when an explicit rule fires (e.g., repeated stall, adapter miss, shadow failure on a promising candidate).

---

## 10. Promotion gates

A transferred (or newly promoted) skill is only admitted to active use when it passes **all six** gate categories, in order. Gate **G0 precedes all others** and is evaluated on every `SkillEpisode`, not only at transfer time — a skill that stops touching evidence in production is demoted. Verdicts are recorded in `GateVerdict` ([Pipeline Orchestrator §2.2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)).

| Gate | Check | Source |
|------|-------|--------|
| **G0 — Evidence-driven contract** | For every episode used as evidence for promotion: `evidence_in ∪ evidence_out ≠ ∅`; `evidence_role` matches the skill's declared role ([PLAN-SKILL-BANK.md §0.3](../03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills)); role-specific fields are populated (GATHER⇒`evidence_out`; VERIFY⇒`verify_verdict`; REASON⇒`reason_warrant ⊆ evidence_in`; COMMIT⇒`evidence_warrant ≠ ∅`). Failure rejects the skill as `opaque-skill-violation` or `skill-role-mismatch`, independent of reward or success rate. | `SkillHarness.finalize_episode` |
| **G1 — Binding** | target slots ground; abstract predicates map to target ontology | `SkillHarness.bind_skill` |
| **G2 — Adapter** | adapter exists (or synthesized adapter is valid); passes domain syntax / execution sanity | `AdapterRegistry.validate` |
| **G3 — Replay** | expected effects match held-out transitions; protocol does not contradict observed data | `ReplayValidator` |
| **G3a — Few-shot adaptation** | for each declared `target_domain`, the skill binds to the target adapter and reaches `pass_rate ≥ target_domain_pass_rate_min` within `k_shot_max` shots; ≥1 verified target required for ACTIVE | `FewShotAdapter` ([§5.4.2](#542-fewshotadapter-stage-3a-runtime)) → consumed by `GateService._run_transfer` ([PLAN-UNIFIED-SKILL-GATE Stage 3a](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)) |
| **G4 — Shadow** | shadow pass rate ≥ threshold; no severe instability / repeated stalls | `TransferProposer.shadow_run_transfer` |
| **G5 — Non-regression** | enabling transfer does not degrade prior source-domain competence beyond tolerance, **measured on the source-domain (game) frozen slice** | cross-run eval on frozen source slice |

Any failing gate → rejection with reason; candidate returns to the crafter for revision or is quarantined. G0 failures are routed to the crafter's `evidence-starved skill` failure cluster (see [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)).

---

## 10a. Transfer-failure diagnostics (domain-specific)

Because the Harness is the place where safe broad transfer is actually validated, every gate rejection and every shadow abort is labeled with a **typed diagnostic** that names *why* transfer failed. This is what makes cross-domain transfer auditable when the target domains span very different observation/action spaces.

Each `GateVerdict` carries zero or more of the following labels; each label is populated by a specific Harness component.

| Label | Meaning | Producing component |
|-------|---------|---------------------|
| `evidence_interface_mismatch` | The target-domain adapter cannot produce or consume the evidence kinds declared in the skill's `evidence_interface` (e.g., protocol expects `frame_span` but target domain only emits `dom_node`), so the episode would violate Gate G0 even if slots bound | `AdapterRegistry.validate` + `ReplayValidator` |
| `opaque_skill_violation` | A shadow or replay episode produced `evidence_in ∪ evidence_out == ∅` or missed the `evidence_role`-specific fields ([§5.1](#51-skillepisode)); the skill is not actually assisting reasoning in the target domain | `SkillHarness.finalize_episode` (Gate G0) |
| `slot_binding_failed` | Required typed slots do not ground from the target `<state>` (missing `target`, empty `candidate_set`, no `blocker` to anchor, ...) | `SkillHarness.bind_skill` |
| `adapter_execution_mismatch` | Adapter exists but its execution surface disagrees with the skill's abstract predicates (e.g., `select($target)` is not realizable with the adapter's action set) | `AdapterRegistry.validate` |
| `evidence_insufficient` | Skill's `evidence_required` cannot be filled from the target domain's within-episode `evidence_refs` (no clip/frame/DOM/desktop-object pointer of the required kind) | `ReplayValidator` |
| `temporal_mismatch` | Video-understanding transfer: temporal `candidate_set` members do not align with the claim's time anchor, or evidence frames are out of order vs. protocol | `ReplayValidator` (video path) |
| `ui_grounding_mismatch` | Webagent transfer: UI elements expected by the protocol (e.g., a "submit" control) are not grounded or are ambiguous in the DOM / screenshot state | `SkillHarness.bind_skill` (browser adapter) |
| `desktop_object_mismatch` | OS-agent transfer: required desktop objects (windows, files, tray icons) are not grounded or belong to a different application | `SkillHarness.bind_skill` (desktop adapter) |
| `overconfident_commit` | Shadow mode: the skill's `COMMIT` fires despite anti-preconditions / `do_not_transfer_if` predicates holding in the target state | `TransferProposer.shadow_run_transfer` |
| `contract_mismatch` | Replay: the realized effects diverge from `eff_add` / `eff_del` beyond tolerance, or belief-effects do not hold after execution | `ReplayValidator` |
| `few_shot_budget_exceeded` | Stage 3a: cumulative `cost_tokens > adaptation_cost_max_tokens` before the K-shot run completes for `(skill, target_domain)` — adapter aborts with this label | `FewShotAdapter` |
| `target_domain_demo_unavailable` | Stage 3a: no target-domain `FewShotDemo`s available for the candidate (or no adapter registered for the target). Adapter returns an empty `AdaptResult` so the gate can flag the target binding as untested rather than failed | `FewShotAdapter` |
| `adaptation_overfitting` | Stage 3a: per-shot `pass_rate < target_domain_pass_rate_min` despite reaching `k_shot_max`. The skill binds syntactically but does not generalize over the target-domain demos | `FewShotAdapter` |

### 10a.1 Consumers

- The [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) appends each label to the target skill's `known_failure_modes` (§4.3b) and, if a pattern recurs, to `do_not_transfer_if` / `false_binding_patterns`.
- The [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) uses the labels to route patch / compose / transfer-adaptation proposals to the right failure cluster.
- The [Pipeline Orchestrator](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) tallies labels into per-domain dashboards so that "transfer is safe in domain X" is a claim backed by diagnostic distributions, not just aggregate success rates.

### 10a.2 First validation arena (for the diagnostics, not for the skills)

The diagnostic labels above are defined for **all five target domains from day one**, and every skill passing through the Harness is a general protocol expected to be feasible in all of them. The **first arena** in which these diagnostics are *exercised and tuned* is short-video evidence-grounded reasoning: `evidence_insufficient`, `temporal_mismatch`, and `opaque_skill_violation` are the highest-signal labels there, followed by `overconfident_commit` (claims made without adequate frame-level backing) and `evidence_interface_mismatch` (adapter cannot supply the declared evidence kinds). As the webagent / os-agent / game adapters come online, `ui_grounding_mismatch`, `desktop_object_mismatch`, and `adapter_execution_mismatch` start firing against the same general protocols — the protocols do not change, only the diagnostics that happen to trigger on them. Gate G0 (`opaque_skill_violation`) fires uniformly across all domains and is the primary defense against a skill *silently ceasing to assist reasoning* after a transfer.

---

## 10b. Gate Execution Runtime

The six per-episode gates G0–G5 in §10 are the *what*. The **Gate Execution Runtime** is the *how*: a single `GateRunner` entry point — owned by the Harness — that the [Pipeline Orchestrator](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) calls to execute the unified skill gate ([PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)) over a candidate `SkillRecord`.

This subsection defines `GateRunner`, the per-stage entry points, and where each stage delegates inside the existing Harness. It does **not** redefine the gate semantics or thresholds — those are pinned in [PLAN-UNIFIED-SKILL-GATE.md §7](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md#7-gate-stages-the-canonical-pipeline) and `configs/skill_gate.yaml` ([PLAN-UNIFIED-SKILL-GATE.md §9](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md#9-threshold-policy)).

### 10b.1 `GateRunner` (`harness/gate_runner.py`)

```python
class GateRunner:
    def run_static_check(self,    skill: SkillRecord) -> GateVerdictPayload: ...
    def run_replay(self,          skill: SkillRecord, datasets: list[str]) -> GateVerdictPayload: ...
    def run_shadow(self,          skill: SkillRecord, rollout_batch: list[dict]) -> GateVerdictPayload: ...
    def run_transfer(self,        skill: SkillRecord, target_domains: list[str]) -> GateVerdictPayload: ...
    def run_non_regression(self,  skill: SkillRecord, eval_suite: dict) -> GateVerdictPayload: ...

    def assemble_evaluation(
        self, skill: SkillRecord, payloads: list[GateVerdictPayload]
    ) -> SkillEvaluationRecord: ...
```

`GateRunner` is the **only** Harness-side entry point the Orchestrator may call to evaluate a candidate. It does not move bank pointers or mutate skill status — it produces `GateVerdictPayload` and `SkillEvaluationRecord` artifacts that the Orchestrator hands to `SkillLifecycleManager` under transaction control ([PLAN-PIPELINE-ORCHESTRATOR.md §3a](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a-promotion-transaction-and-rollback-protocol)).

### 10b.2 Per-stage delegation table

| Unified-gate stage | `GateRunner` method | Delegates to | Per-episode gate (§10) |
|--------------------|--------------------|--------------|------------------------|
| Stage 0 — Static sanity        | `run_static_check`     | `gate/static_checker.py` (schema, slot types, contract); calls into `SkillBank.skill_record` validators | G1 (binding feasibility) at the schema level |
| Stage 1 — Offline replay       | `run_replay`           | `harness/replay_validator.py` ([§5.5](#55-replayvalidator)) | G3 (Replay) |
| Stage 2 — Shadow execution     | `run_shadow`           | `SkillHarness.run_shadow` ([§5.2](#52-skillharness)); MUST set `SkillEpisode.shadow = True`; respects [§6.1](#61-phase-a--shadow-mode) constraints | G4 (Shadow) |
| Stage 3 — Transfer validation  | `run_transfer`         | `harness/transfer_manager.py` ([§5.4](#54-transfermanager)); per-target-domain `SkillHarness.run_shadow` + `AdapterRegistry.validate` | G2 (Adapter) + G3 (Replay) on target domain |
| Stage 4 — Non-regression       | `run_non_regression`   | `harness/eval_harness.py` (existing) over the orchestrator-supplied frozen eval suite | G5 (Non-regression) |
| Continuous (every episode)     | n/a — runs in `SkillHarness.finalize_episode` regardless of source | `SkillHarness.finalize_episode` per [§5.1](#51-skillepisode) | **G0 (Evidence-driven contract)** |

**G0 is orthogonal to the batch lifecycle.** It is checked on every `SkillEpisode` produced by `run_active` *and* `run_shadow`, and a sustained pattern of G0 failures in production triggers `ACTIVE → DEPRECATED` via [PLAN-UNIFIED-SKILL-GATE.md §7 Stage 6](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md#stage-6--rollback--deprecation-orchestratorrollback_managerpy).

### 10b.3 Diagnostic-label routing

Each `GateVerdictPayload` carries the [§10a](#10a-transfer-failure-diagnostics-domain-specific) diagnostic labels. `assemble_evaluation` rolls them up into `SkillEvaluationRecord.diagnostic_labels`, which the Orchestrator forwards to:

- the bank for `known_failure_modes` / `do_not_transfer_if` updates ([PLAN-SKILL-BANK.md §4.3b](../03-skill-bank/PLAN-SKILL-BANK.md#43b-negative-knowledge)),
- the [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) for failure-cluster routing,
- the orchestrator dashboards for per-domain transfer-safety distributions ([PLAN-PIPELINE-ORCHESTRATOR.md §6.4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#64-slices)).

### 10b.4 What stays out of `GateRunner`

- **Status mutation.** `GateRunner` does not call `SkillLifecycleManager`; the Orchestrator does, under transaction.
- **Bank-pointer moves and snapshot creation.** Owned by `orchestrator/snapshot_manager.py`.
- **Threshold *interpretation*.** `GatePolicy` (`gate/gate_policy.py`) loads thresholds and turns numeric metrics into `GateVerdict` enum values; `GateRunner` only emits raw metrics + the policy-rendered verdict.

---

## 11. No new trainable model required at first

Phase 0 implementation is **inference-only**. It uses:

- the existing Qwen3-8B runtime model,
- the optional 32B/72B frozen slow teacher,
- the current retrieval and Skill Bank machinery,
- rule-based and replay-based validation.

Only after the Harness is working end-to-end should we consider adding trainable LoRAs for:

- `skill_select` (Harness-aware)
- `continue_vs_switch`
- `accept_transfer`
- `adapter_refine`

This mirrors the [Action Agent co-evolution schedule](../02-action-agent/PLAN-ACTION-AGENT.md) — the harness first, the learning later.

---

## 12. Suggested code structure

A new top-level module inside the repo:

```
harness/
  __init__.py
  skill_harness.py          # main orchestration
  skill_episode.py          # SkillEpisode dataclass + serialization
  transfer_manager.py       # two-phase transfer protocol
  adapter_registry.py       # per-domain adapter lookup + synthesis hooks
  replay_validator.py       # offline validation on held-out transitions
  reward_logger.py          # unified r_env / r_follow / r_cost / r_transfer
  eval_harness.py           # reuse + transfer benchmarks
  gate_runner.py            # §10b — single entry point for the unified skill gate
  adapters/
    gymv.py
    browser.py
    osworld.py
    video.py
    visual_reasoning.py
```

### 12.1 File responsibilities

| File | Implements |
|------|------------|
| `skill_harness.py` | `normalize_state`, `retrieve_candidates`, `rank_candidates`, `bind_skill`, `attach_adapter`, `execute_step`, `update_episode`, `finalize_episode`, `run_active`, `run_shadow` |
| `skill_episode.py` | `SkillEpisode` record + JSONL serialization compatible with [Pipeline Orchestrator §2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) |
| `transfer_manager.py` | `propose_transfer`, `dry_run_transfer`, `shadow_run_transfer`, `promote`, `reject` |
| `adapter_registry.py` | adapter registration, lookup, validation, synthesis escalation to 72B |
| `replay_validator.py` | held-out replay checks (effects, protocol, evidence) |
| `reward_logger.py` | central reward emission + metric collation |
| `eval_harness.py` | reuse + transfer benchmark runner (metrics from §15) |
| `gate_runner.py` | `GateRunner` (§10b) — `run_static_check`, `run_replay`, `run_shadow`, `run_transfer`, `run_non_regression`, `assemble_evaluation`; single entry point the [Pipeline Orchestrator](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) calls to execute the [Unified Skill Gate](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) |

---

## 13. Integration points with the existing framework

The Harness **does not replace** any existing module. It sits on top.

### 13.1 What stays where

| Module | Keeps responsibility for |
|--------|--------------------------|
| Skill Bank | storage, retrieval backend, contract learning, curation |
| Skill Crafter | composition, abstraction, novel-skill hypothesis |
| Action Agent | actor policy inference, hop selection, action emission |
| Pipeline Orchestrator | full-system DAG, acceptance gates, training schedule |

### 13.2 What moves into the Harness

- runtime orchestration of skill invocation,
- transfer gating (shadow → active),
- replay + shadow validation driver,
- unified logging of `SkillEpisode`,
- evaluation hooks for reuse / transfer metrics.

### 13.3 Wiring pattern

```
ActionAgent
  └─ SkillHarness
       ├─ SkillBank.query (retrieval backend)
       ├─ AdapterRegistry
       ├─ TransferManager
       │    ├─ ReplayValidator
       │    └─ (shadow loop via SkillHarness.run_shadow)
       └─ RewardLogger
```

The Pipeline Orchestrator drives `ActionAgent` and, on end-of-episode / offline evolution, consumes `SkillEpisode` streams from the Harness.

---

## 14. Phased implementation plan

| Phase | Goal | Deliverables | Success criteria |
|-------|------|--------------|------------------|
| **0 — Harness MVP** | unify skill invocation before touching transfer | `SkillEpisode`, `SkillHarness`, basic retrieval + binding + execution wrapper, centralized logging | every skill invocation goes through the Harness; traces and outcomes recorded consistently |
| **1 — Adapterized execution** | separate semantic skill from domain adapter | `AdapterRegistry`, adapter attachment API, adapter validity checking, adapters for `gymv`, `browser` | same semantic skill attaches different adapters across domains |
| **2 — Shadow transfer validation** | support transfer safely before active use | `TransferManager`, `ReplayValidator`, shadow-mode runner | transferred skills can be tested without affecting active rollout |
| **3 — Promotion gates** | decide when a transferred skill becomes active | gate implementations (§10), non-regression checks, contract consistency checks | only validated transferred skills enter active policy |
| **4 — Evaluation harness** | measure reuse and transfer explicitly | `eval_harness.py`, in-domain reuse + cross-domain transfer benchmarks, metrics dashboard | metrics from §15 are logged centrally and comparable across runs |
| **5 — Trainable extensions (optional)** | add learning on skill-use decisions | `skill_select` / `continue_vs_switch` / `accept_transfer` / `adapter_refine` LoRAs | measurable gain over rule-based baselines |

**Immediate next implementation target:** **Phase 0 + Phase 1 only.** Build `SkillEpisode`, `SkillHarness`, and `AdapterRegistry`, and route all current skill usage through them **before** implementing transfer promotion.

---

## 15. Metrics

The Harness (not scattered modules) owns these metrics.

| Metric | Definition |
|--------|------------|
| `retrieval_hit_at_k` | fraction of invocations where the ground-truth useful skill appears in top-k retrieval |
| `slot_binding_success_rate` | fraction of candidate skills whose slots bind on current state |
| `adapter_validation_pass_rate` | fraction of `(skill, domain)` pairs with a valid adapter (existing or synthesized) |
| `shadow_transfer_pass_rate` | fraction of shadow episodes passing all shadow-mode checks |
| `promotion_rate` | fraction of transfer proposals reaching active mode |
| `active_transfer_success_rate` | success rate of transferred skills once in active mode |
| `avg_steps_to_success` | mean inner-hop count per successful skill invocation |
| `source_domain_non_regression` | Δ on frozen source-domain eval after enabling transfer |
| `target_domain_reward_delta` | Δ on target-domain reward vs. baseline without transfer |

These feed the [Pipeline Orchestrator §6](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) evaluation matrix.

---

## 16. Implementation notes for Cursor

Priority edits, in order:

1. Create the `harness/` module and define `SkillEpisode` (`harness/skill_episode.py`).
2. Wrap the current `ActionAgent` skill-usage path with `SkillHarness` (`harness/skill_harness.py`) — Phase 0.
3. Refactor current skill invocation into: retrieval → slot binding → adapter attachment → execution → logging.
4. Add `AdapterRegistry` with two initial adapters (`gymv`, `browser`) — Phase 1.
5. Add `TransferManager` but keep it **shadow-only** initially — Phase 2.
6. Add `ReplayValidator` and the promotion gate (§10) — Phase 3.
7. Add `eval_harness.py` with the metrics from §15 — Phase 4.

All new code should write `SkillEpisode` records in a format compatible with the Pipeline Orchestrator's artifact schema ([§2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)).

---

## 17. What not to do

Do **not**:

- introduce a fourth agent just for the harness,
- make the 32B/72B teacher the default online controller,
- couple transfer directly to bank insertion (bank stays storage + retrieval; transfer lives in the Harness),
- allow transferred skills into active policy without shadow + replay validation,
- train new models before the execution / validation loop is working,
- scatter reward emission — all reward goes through `RewardLogger`.

---

## 18. Short design summary (for the repo)

We introduce a **Skill Harness** as a unified runtime and evaluation orchestration layer that mediates between the Action Agent, Skill Bank, and Skill Crafter. The Harness normalizes state into shared slots, retrieves candidate skills, binds semantic skills to target-domain adapters, executes them with protocol-aware tracing, and records evidence traces, contract progress, and outcomes in standardized `SkillEpisode` objects. For cross-domain transfer, the Harness uses a **two-stage protocol**: transferred skills are first evaluated in replay and shadow mode, and only promoted to active execution after passing **binding, adapter, replay, shadow, and non-regression** gates. This design lets action, reasoning, and grounding skills share a unified execution interface and makes skill reuse and transfer **measurable, safe, and extensible** to later GRPO-based decision learning.

---

## 19. Immediate next target

**Phase 0 + Phase 1.** Build `SkillEpisode`, `SkillHarness`, and `AdapterRegistry`, then route all current skill usage through them **before** implementing transfer promotion. That is the cleanest starting point; everything else (TransferManager, ReplayValidator, promotion gates, eval harness, trainable LoRAs) stacks cleanly on top once the Harness owns the execution surface.

---

## 20. (Optional) Harness Ablations

**Status:** Optional appendix. Run once Phase 0–4 (§14) are functional and the joint task evaluation contract ([PLAN-EVAL-FIRST-TARGET.md](../00-system/PLAN-EVAL-FIRST-TARGET.md)) reports stable numbers. Skip this section if the project is still pre-MVP.

This appendix defines a **minimal ablation suite** for the Harness itself: what the Harness contributes *beyond the Actor* and *beyond the skill bank*. It is intentionally small. The default development loop should not wait on it; it exists so that, when the Harness is shipped, we can defend the claim that the Harness — not just the Actor or the skill bank — is doing measurable work.

### 20.1 Why Harness needs explicit ablations

The Harness sits between the Actor (trainable, 7B/8B) and the Skill Bank (storage + retrieval). A naive read of the architecture invites two failure modes:

- **"The Actor did it."** A skeptic can argue that any improvement attributed to the Harness is really the Actor learning to call good skills, and the Harness is just plumbing.
- **"The bank did it."** A skeptic can argue that retrieval alone (with no binding/precondition/evidence/adapter checks) already returns the right skill most of the time, so the Harness's filtering / veto / scoring add nothing.

Without explicit ablations these claims cannot be refuted. Module-level metrics (§15) measure *Harness internals* (slot binding rate, adapter pass rate) but they do not isolate the Harness's *contribution to system outcome*. The orchestrator-level eval ([PLAN-PIPELINE-ORCHESTRATOR.md §6](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md), [PLAN-EVAL-FIRST-TARGET.md](../00-system/PLAN-EVAL-FIRST-TARGET.md)) explicitly requires separate analysis of actor quality, harness filtering/veto quality, system performance, skill-use efficiency, reasoning-step usefulness, and transfer robustness — this appendix supplies the experimental design that makes those axes separable.

The suite is also the place where the **frozen-72B-as-runtime-validator** claim ([§1a](#1a-harness-role-as-frozen-72b-runtime-layer)) is checked: ablation **A0** answers the converse — does the Actor still improve when the validation layer is removed?

### 20.2 Core evaluation questions

The suite must answer at least these four questions. Every ablation cell exists to move one of them. Anything that doesn't is dropped.

| Q | Question | Primary signal |
|---|----------|----------------|
| **Q1** | Does the Harness improve **skill invocation validity**? | invalid invocation rate ↓; slot binding success rate ↑; precondition pass rate ↑; evidence pass rate ↑ |
| **Q2** | Does the Harness improve **transfer safety**? | shadow→active promotion precision ↑; regression rate after transfer ↓; `opaque_skill_violation` / `evidence_insufficient` rates ↓ on cross-domain slices |
| **Q3** | Does the Harness **reduce harmful or low-value skill execution**? | veto precision / veto recall (where ground truth available); avg skill-use cost / latency for unsuccessful invocations ↓; abort rate on bad candidates ↑ before side effects |
| **Q4** | Does the **Actor itself** still improve, or is the frozen validation layer doing all the work? | actor decision quality (top-1 / top-k accuracy on the Harness-eligible set) over training time; gap between A0 and A4 attributable to the *trained Actor* vs. *frozen validation* |

Q4 is the load-bearing question. If A0 (no Harness) shows the Actor failing to improve while A4 (full system) succeeds, **and** the Actor's standalone decision quality on the eligible set is rising, then both the Harness *and* the Actor are doing real work. If A0 and A4 differ but Actor decision quality on eligible sets is flat, the system is a 72B-driven policy in disguise ([§1a.5](#1a5-why-the-frozen-72b-harness-should-not-replace-the-actor)) and the architecture story has failed.

### 20.3 Ablation matrix

A staged matrix. Each row removes or adds exactly one Harness capability vs. its neighbor, so deltas are attributable.

| Cell | What is enabled | What is disabled | Purpose |
|------|-----------------|------------------|---------|
| **A0 — No Harness** | Actor + Skill Bank retrieval only. Actor calls retrieved skill directly; no binding/precondition/evidence/adapter checks; no veto. `SkillEpisode` still logged for measurement. | All Harness validation, scoring, veto, transfer gating. | Baseline. Isolates "Actor + bank alone." Answers Q4. |
| **A1 — Harness-lite** | A0 + slot binding check + precondition check (G1 + applicability part of §7.1). Veto on bind/precondition failure only. | Evidence-role checks (G0), adapter validation (G2), replay (G3), shadow (G4), non-regression (G5), runtime veto on evidence/adapter, transfer gating. | Cheapest validation. Measures the contribution of *structural* checks alone. |
| **A2 — Harness-core** | A1 + evidence-role / Gate G0 checks ([§5.1](#51-skillepisode)) + adapter validation (G2) + runtime veto on any failed check + advisory scoring ([§1a.6](#1a6-harness-as-teacher-like-advisor)). | Replay validation (G3), shadow execution (G4), non-regression (G5), promotion/rollback. | The core "filter + veto + advise" runtime. Measures Q1 and Q3. |
| **A3 — Harness-transfer** | A2 + replay validation (G3) + shadow execution (G4) + transfer-aware ranking (§7.2). Cross-domain skills are admitted only after replay+shadow. | Promotion/rollback transactions, non-regression gate (G5). | Adds the transfer safety layer. Measures Q2 and the `opaque_skill_violation` / `evidence_insufficient` reductions on cross-domain slices. |
| **A4 — Full system** | A3 + non-regression gate (G5) + promotion/rollback hooks ([PLAN-PIPELINE-ORCHESTRATOR.md §3a](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a-promotion-transaction-and-rollback-protocol)). Sustained G0 failures trigger demotion. | nothing. | Reference configuration. The shipped Harness. |

**Constants across cells.** Same Actor checkpoint, same Skill Bank snapshot, same retrieval index, same evaluation slices, same seeds, same teacher escalation policy. Only the Harness configuration changes between cells.

**Cell pairs that matter.**

- **A0 → A1**: contribution of structural validation alone.
- **A1 → A2**: contribution of evidence-driven invariant + veto (the part that defends [§1a.5](#1a5-why-the-frozen-72b-harness-should-not-replace-the-actor)).
- **A2 → A3**: contribution of replay+shadow to transfer safety.
- **A3 → A4**: contribution of promotion/rollback (defense against silent regressions).
- **A0 vs. A4**: total Harness contribution.

### 20.4 Metrics

Recorded for every (cell × slice) pair. Numbers come from `SkillEpisode` ([§5.1](#51-skillepisode)) plus the orchestrator's task-level eval ([PLAN-EVAL-FIRST-TARGET.md §6–7](../00-system/PLAN-EVAL-FIRST-TARGET.md)).

| Group | Metric | Source |
|-------|--------|--------|
| **System outcome** | task success / answer accuracy | task eval (joint success rate where defined) |
| **System outcome** | evidence support rate | `support_package.evidence_warrant` non-empty + judge pass |
| **Validity (Q1)** | invalid invocation rate | fraction of invocations with any of `binding_ok=False`, `precondition_ok=False`, `evidence_ok=False`, `adapter_ok=False` reaching execution |
| **Validity (Q1)** | slot binding success rate | `SkillHarness.bind_skill` pass / attempts |
| **Validity (Q1)** | precondition pass rate | `Skill.preconditions` pass / attempts |
| **Validity (Q1)** | evidence pass rate | Gate G0 pass / `finalize_episode` calls |
| **Validity (Q1)** | adapter pass rate | `AdapterRegistry.validate` pass / attempts |
| **Veto quality (Q3)** | veto precision | vetoed-and-truly-bad / vetoed (ground truth from logged outcomes when veto is overridden in a sweep cell, plus replay) |
| **Veto quality (Q3)** | veto recall | vetoed-and-truly-bad / all-truly-bad attempts |
| **Veto quality (Q3)** | avg skill-use cost / latency | per `SkillEpisode`, broken down by outcome |
| **Transfer (Q2)** | transfer pass rate | shadow → active promotion fraction |
| **Transfer (Q2)** | regression rate after transfer | `source_domain_non_regression` (§15) flipped to a rate |
| **Actor (Q4)** | actor top-1 / top-k accuracy on eligible set | Actor's choice vs. ground-truth-best skill *within* the Harness-eligible set, per cell |

Veto precision/recall is only computable when truth is available (replay slices with logged outcomes, or sweep runs in cells where the veto is logged but not enforced). Where it is not, mark as `n/a` rather than fabricate.

### 20.5 Dataset slices

Same instance pool as the joint eval, partitioned along the four axes below. Every cell × slice combination is measured; do not collapse slices into a single grand mean.

| Slice axis | Values | Purpose |
|------------|--------|---------|
| **Domain reuse** | `in_domain_reuse` (skill ran ≥1× in the source domain), `cross_domain_transfer` (skill is being applied to a domain it has not run in) | Separates Q1 (validity) from Q2 (transfer safety). |
| **Promotion stage** | `before_promotion` (skill in shadow / pre-G4), `after_promotion` (skill ACTIVE) | Verifies that the gates actually filter — `after_promotion` numbers must dominate `before_promotion` numbers in A3/A4. |
| **Difficulty** | `easy`, `hard` (per [PLAN-EVAL-FIRST-TARGET.md](../00-system/PLAN-EVAL-FIRST-TARGET.md) `difficulty` metadata) | Detects ceilings: if A0 already saturates `easy`, the Harness's value will only show on `hard`. |
| **Domain** | the five target domains ([§13.1](#131-what-stays-where), bank §0.1) | Required to back the "transfer is safe in domain X" claim with diagnostic distributions, not just aggregates. |

Minimum slice budget per cell × slice: large enough that a 5-point absolute change in task success is distinguishable from noise at the slice level. Below that, mark the slice as under-powered and report the cell × slice number with an explicit confidence note rather than dropping the slice.

### 20.6 Analysis templates

Three reports per ablation run. Keep them short — one page each. They must keep the three signals **separated**; do not merge "actor decision quality" and "harness filtering quality" into a single chart.

**(a) Actor decision quality — per cell.**
For each cell, plot Actor top-1 accuracy on the Harness-eligible set over training steps. Same Actor, different upstream filters. The question is whether the Actor's *choice quality on a fixed eligible set* improves; if it does, the Actor is learning real skill-use policy and is not just a passenger.

**(b) Harness filtering quality — per cell × slice.**
Per cell, report the Validity and Veto metrics from §20.4. Compute the A0→A1, A1→A2, A2→A3 deltas with confidence intervals. This is the report that defends "the Harness reduces harmful or low-value skill execution."

**(c) Overall system outcome — per cell × slice.**
Task success / answer accuracy / evidence support rate per cell × slice, plus the Joint Success Rate from [PLAN-EVAL-FIRST-TARGET.md §7](../00-system/PLAN-EVAL-FIRST-TARGET.md). Cross-domain rows here are how Q2 is settled.

A run is reported as **"Harness contributes"** only if all three reports tell consistent stories: Actor decision quality is non-flat across cells, Harness filtering metrics improve monotonically A0→A4, and overall system outcome rises with the same shape — especially on `cross_domain_transfer` and `hard` slices.

### 20.7 Minimal rollout order

Run cells in the order that buys the most diagnostic value per unit of compute. Do not start at A4.

1. **A0 and A4 first**, on a single small slice (one domain, mixed difficulty). Establishes that there is *any* gap to explain. If `A4 − A0 ≈ 0`, the rest of the suite is not worth running yet — go fix the Harness or the Actor first.
2. **A2** on the same slice. Tells whether the gap is mostly the core filter+veto layer (A2≈A4) or mostly the transfer machinery (A2≪A4).
3. **A1 and A3** to fill in the staircase, on the same slice.
4. **All five cells × all four slice axes**, only after the staircase looks coherent on the small slice.
5. **Cross-domain re-run** for A2/A3/A4 on the `cross_domain_transfer` slice with diagnostic-label breakdowns ([§10a](#10a-transfer-failure-diagnostics-domain-specific)).

Stop early at any step whose result invalidates the next step's premise.

### 20.8 Anti-goals

- **Do not build a massive benchmark matrix in v1.** Five cells × four slice axes × five domains is the ceiling for the first version. Anything more belongs in a successor plan.
- **Do not rely on one giant teacher model to "solve everything" during ablations.** The 72B is the frozen runtime validator ([§1a](#1a-harness-role-as-frozen-72b-runtime-layer)) and may be escalated to per the existing rules ([§9.2](#92-slow-loop--3272b-frozen-teacher)); it must not be promoted to *online policy* inside any ablation cell, including A0. The point of the suite is to attribute outcome to Harness components, not to the teacher.
- **Do not redefine gate semantics, thresholds, or `SkillEpisode` schema inside this appendix.** Those live in [§5.1](#51-skillepisode), [§10](#10-promotion-gates), and [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md). The ablation suite *consumes* them; if it needs to change them, that is a signal to update the upstream plans first.
- **Do not introduce new trainable models** to make a cell run. Cells differ only in which Harness capabilities are enabled, not in model identity or weights.
- **Do not collapse Q1, Q2, Q3, Q4 into one number.** The point of the suite is that they are separable; merging them re-creates the ambiguity the suite was built to remove.
