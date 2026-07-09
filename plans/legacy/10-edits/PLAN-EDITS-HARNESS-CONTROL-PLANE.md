# PLAN: Edit Plan — Harness as Control Plane

> **Revision note 2 (evidence-driven invariant, April 2026).** The evidence-driven invariant ("ALL skills are evidence-driven; every skill's purpose is to assist reasoning / decision-making") is now enforced at the Harness gate. The following changes have been applied directly to the live plan files on this pass and supersede any looser wording anywhere in this edit plan (including P2 / P4 / P5 drafts):
>
> - `PLAN-SKILL-BANK.md §0.3` adds the **evidence-driven invariant** with Clause A (`evidence_in ∪ evidence_out ≠ ∅`) and Clause B (`evidence_role ∈ {GATHER, VERIFY, REASON, COMMIT}`). `§4.1`/`§4.2` carry `evidence_role` and an `evidence_interface` block on every skill.
> - `PLAN-HARNESS.md §5.1` extends `SkillEpisode` with `evidence_role`, `evidence_in`, `evidence_out`, `evidence_warrant`, `verify_verdict`, `reason_warrant` and the `evidence_role`-specific field requirements. `§10` promotes the gate set from five to **six** gates, with **Gate G0 — Evidence-driven contract** first; `§10a` adds `opaque_skill_violation` and `evidence_interface_mismatch` diagnostics.
> - `PLAN-SKILL-CRAFTER.md §2.5` introduces typed proposals (`PatchProposal | ComposeProposal | TransferProposal | RetireProposal`), each carrying `evidence_role` + `evidence_interface`; `§6.2` adds the `evidence_starved` failure category.
> - `PLAN-ACTION-AGENT.md §5.3-bis` pins the inner-hop ↔ `evidence_role` contract; skill-role mismatches are raised by the Harness.
> - `PLAN-VISUAL-SKILLS.md §2` maps visual/grounding skills to `GATHER` / `VERIFY`; `PLAN-VISUAL-GROUNDING.md §3a` declares `GroundingRecord` the canonical `evidence_out` for `GATHER` skills.
> - `PLAN-PIPELINE-ORCHESTRATOR.md §2.2` promotes `SkillEpisode` to a first-class artifact and carries the new fields; `§4` references the invariant as the *reason* for evidence bookkeeping.
> - `../README.md` lists the two invariants (general-protocol + evidence-driven) up front.
>
> The P2 Harness Contract, P4 Typed Proposal Outputs, and P5 `GroundingRecord` drafts below should be read with these applied edits in mind — any proposal contract, action-agent output record, or grounding record they describe now also carries `evidence_role` / `evidence_in` / `evidence_out` / `evidence_warrant` as applicable. Where the drafts below and the applied text disagree, the applied text wins.

> **Revision note 1 (episode-local trajectory + broad ontology, short-video-first).** The repo's only state-keeping surface is the orchestrator's **episode-local trajectory** — current `<state>`, short typed hop trace, intermediate belief state, within-episode evidence references — over a **broad, cross-domain skill ontology** (game / webagent / os-agent / video-understanding / visual reasoning). The corresponding edits have been applied directly to the live plan files on this same pass:
>
> - `PLAN-PIPELINE-ORCHESTRATOR.md §4` is the canonical **Episode-local evidence & trace bookkeeping** section (current `<state>`, short typed hop trace, intermediate belief state, within-episode evidence references, claim–evidence links, transfer diagnostics).
> - `../02-action-agent/PLAN-ACTION-AGENT.md`, `../03-skill-bank/PLAN-SKILL-BANK.md`, `../04-skill-crafter/PLAN-SKILL-CRAFTER.md`, `../05-harness/PLAN-HARNESS.md`, and `../README.md` consistently describe state-keeping as episode-local trajectory bookkeeping plus a "broad ontology, short-video first" framing.
> - `PLAN-HARNESS.md §10a` adds **domain-specific transfer-failure diagnostics** (`slot_binding_failed`, `adapter_execution_mismatch`, `evidence_insufficient`, `temporal_mismatch`, `ui_grounding_mismatch`, `desktop_object_mismatch`, `overconfident_commit`, `contract_mismatch`).
> - `PLAN-SKILL-BANK.md §4.3a / §4.3b` add **lineage/provenance** and **negative-knowledge** fields on every skill.
>
> Where any draft below talks about cross-episode storage / retrieval interfaces / alignment integrations, the applied text in the orchestrator §4 and the other plan files wins. The applied text uses `RETRIEVE` for the skill bank only; everything the agent needs in subsequent hops must already exist in the episode-local trajectory.

**Scope.** This is a *plan of edits* to the existing plan files, not a new module plan. It turns the earlier diagnosis ("the harness is a core contribution, not glue") into a concrete, ordered, Cursor-executable refactor of:

- `../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`
- `../02-action-agent/PLAN-ACTION-AGENT.md`
- `../03-skill-bank/PLAN-SKILL-BANK.md`
- `../04-skill-crafter/PLAN-SKILL-CRAFTER.md`
- `../05-harness/PLAN-HARNESS.md` (terminology reconciliation only)
- `../01-visual-grounding/PLAN-VISUAL-GROUNDING.md` (integration section only)
- `../README.md` (pipeline overview + three-agent framing)

**Thesis to enforce across all edits.**

> The Harness is the **control plane** that turns heterogeneous domains into one shared typed decision process, governs skill evolution and transfer, and enforces verification before any update reaches production. It is the unit of generalization, verification, and attribution.

**Non-goals of this edit pass.**

- No new modules, no new agents, no new trainable heads.
- No changes to grounding algorithms, MDP action set, GRPO design, or bank pipeline semantics.
- No renaming of `<state>`, slot names, or inner primitives (`GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE`).
- No content duplication — when a concept lands in the orchestrator plan, other plans **link** to it instead of restating it.

---

## 0. Terminology reconciliation (do this first)

The repo currently has **two files both using "harness"** for different things:

| File | Current role | Under the new framing |
|------|--------------|-----------------------|
| `../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md` | Macro DAG runner ("glue") | **Harness = control plane** (this is what the directive upgrades) |
| `../05-harness/PLAN-HARNESS.md` | Per-invocation **skill** runtime (`SkillHarness`, `SkillEpisode`) | A component *inside* the control plane: the **Skill-Invocation Runtime** |

**Resolution (chosen to minimize churn).** Keep both files. Add a one-paragraph disambiguation block at the top of each, and consistently call the control plane **"the Harness"** (capital H) in prose, while keeping the class name `SkillHarness` for the per-invocation runtime.

**Action items (terminology only, before any content edits):**

1. In `../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`, immediately below the H1 title, add:

   > **Terminology.** This document defines **the Harness** — the system-level control plane. The class `SkillHarness` ([PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md)) is a component *inside* the Harness, specifically the per-invocation runtime for skill use. Macro vs. micro: Harness (this plan) = control plane; `SkillHarness` = skill-invocation runtime.

2. In `../05-harness/PLAN-HARNESS.md`, update the existing "Relation to Pipeline Orchestrator" paragraph (already in §intro) to:

   > **Relation to the Harness (control plane).** The system-level **Harness** ([PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)) is the control plane. The `SkillHarness` defined here is its **skill-invocation runtime** — the micro-layer the control plane calls at every `inner_mdp` step where a skill is invoked. Terms: Harness = macro, control plane; `SkillHarness` = micro, per-invocation skill runtime.

3. In `../README.md` row 6 of the plan table, change the "Scope" cell for `../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md` from the current "End-to-end harness — ..." to:

   > **The Harness (control plane)** — single top-level DAG, typed trajectory interface, acceptance gate, promotion economy, cross-domain transfer protocol, episode-local evidence & trace bookkeeping, budget contracts, full-system evaluation, harness-level ablations.

4. In `../README.md` row 7 for `../05-harness/PLAN-HARNESS.md`, rename the display title from "Skill Harness" to **"Skill-Invocation Runtime (micro-harness)"** in the "Plan" cell; leave the file path unchanged.

**Verification.** After steps 1–4, grep for the bare word `harness` across `plans/*.md` and confirm each occurrence is either (a) the control-plane sense, (b) the class name `SkillHarness`, or (c) the phrase "skill-invocation runtime". Fix any ambiguous usage.

---

## 1. Execution order (phases)

The edits are ordered so that each later phase can link to stable anchors established earlier.

| Phase | Target file(s) | Why this order |
|-------|----------------|----------------|
| **P0** | Terminology reconciliation (§0) | Removes ambiguity before any cross-file linking. |
| **P1** | `../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md` | Establishes the canonical definitions (Why-it-matters, Three Layers, Promotion Economy, Transfer Protocol, Ablations). All later files link here. |
| **P2** | `../02-action-agent/PLAN-ACTION-AGENT.md` | Adds **Harness Contract** (consumer side), **Failure Taxonomy**, tightens 8B/72B split, elevates inner MDP as stable reasoning API. |
| **P3** | `../03-skill-bank/PLAN-SKILL-BANK.md` | Reframes as harness-governed; adds Transfer Readiness, Dual Storage, Lifecycle Under Harness, Shared Primitives as transfer substrate. |
| **P4** | `../04-skill-crafter/PLAN-SKILL-CRAFTER.md` | Narrows scope to three jobs (patch/compose/transfer), standardizes typed proposal outputs. |
| **P5** | `../01-visual-grounding/PLAN-VISUAL-GROUNDING.md` | Enriches `GroundingRecord` output contract so the evidence harness can log unresolved entities / ambiguity. |
| **P6** | `../README.md` | Updates pipeline-overview framing, scope cells, and adds a one-paragraph "Harness as control plane" blurb. |
| **P7** | Verification pass | Cross-link check, terminology grep, ablation-table sanity. |

Each phase below specifies exact section headers to add, insertion points, and verbatim text to paste.

---

## 2. P1 — Edits to `../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`

**File state check.** Current section map (verified): `# title`, `## 1. Rollout DAG`, `## 2. Artifact / log schema`, `## 3. Promotion / rollback rules`, `## 4. Episode-local evidence & trace bookkeeping`, `## 5. Training cadence by timescale`, `## 6. Evaluation matrix`, `## 7. Budget controller`, `## 8. Failure escalation / human audit points`, `## 9. Implementation checklist`, `## 10. Related documents`.

### 2.1 Rewrite the opening (Scope + Problem statement)

**Replace** the current `**Scope:**` line (which calls the file "glue" and "not a new research module") with:

> **Scope.** Define **the Harness** — the system-level control plane for cross-domain reasoning and control. The Harness maps heterogeneous observations and traces into a shared typed trajectory interface, coordinates grounding, episode-local evidence & trace bookkeeping, skill retrieval, action execution, verification, promotion, rollback, and training, and makes skill transfer and self-evolution **measurable and safe**. The orchestrator's only state-keeping surface is the episode-local trajectory in [`PLAN-PIPELINE-ORCHESTRATOR.md §4`](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping).

**Replace** the `**Problem statement:**` paragraph with:

> **Problem statement.** Sub-plans specify *what* each module does, but not the shared runtime under which heterogeneous domains become one decision process, skill changes are verified before reaching production, and gains or regressions can be attributed to a specific subsystem. The Harness fills exactly this role: one executable DAG with typed artifacts, acceptance gates, a promotion economy, a native cross-domain transfer workflow, budget control, and observability.

**Keep** the `**Upstream:**`, `**Downstream:**`, and `**Non-goals:**` lines unchanged.

### 2.2 Insert new section — "Why the Harness Matters"

Insert **immediately after** the opening metadata block and **before** `## 1. Rollout DAG`, as a new `## 0.` section:

```markdown
## 0. Why the Harness Matters

The Harness is the central mechanism that gives the project scientific structure.

1. **Generalization.** It maps games, browser tasks, desktop / OS tasks, short-video reasoning, visual reasoning, and embodied control into one typed decision process (shared `<state>` schema, shared slot names, shared inner primitives). Domains become *instantiations*, not special cases.
2. **Verification.** It blocks unverified skill or protocol changes from entering production. Self-evolution is only allowed through the acceptance gate (§3) and the promotion economy (§3a).
3. **Attribution.** Typed artifacts (`GroundingRecord`, `InnerHopRecord`, `ActionRecord`, `BankMutationProposal`, `GateVerdict`) let us localize gains or regressions to a specific subsystem — grounding, reasoning, retrieval, bank update, crafter proposal, or training change.
4. **Efficiency.** The tier split is a harness decision: the 8B actor stays in the fast online loop; 32B/72B models operate offline for synthesis, diagnosis, and transfer; the Harness decides when offline proposals are admitted.

This framing is the scientific claim of the system; the rest of this document is its mechanism.
```

### 2.3 Restructure §1 as "Three Layers of the Harness"

**Rename** `## 1. Rollout DAG` → `## 1. Three Layers of the Harness`.

**Prepend** a short lead paragraph before the existing §1.1:

> The Harness is structured as three layers. The existing DAG subgraphs (§1.1–1.3) are how these layers execute; the layering below is *what they are responsible for*.

**Insert three labeled subsections** (do not delete existing §1.1–1.5; slot them under Layer 1 / Layer 2 / Layer 3 as noted):

```markdown
### 1.0 Layer map

| Layer | Role | Existing subgraph |
|-------|------|-------------------|
| **L1 Runtime Harness** | Online execution loop: observe → ground → inner_mdp → act → log_step → finalize_episode | §1.1, §1.2 |
| **L2 Evidence Harness** | Makes every important decision traceable: grounding evidence, retrieval evidence, contract clauses checked, replay slices supporting promotion/rollback | §2 artifact schema + new hooks in §4 |
| **L3 Transfer Harness** | Moves skills across domains safely: discover → canonicalize → bind adapters → replay on target traces → promote or quarantine | New §3b (added in 2.5 below) |
```

After this subsection, **keep** `### 1.1 Online episode subgraph`, `### 1.2 End-of-episode subgraph`, `### 1.3 Offline evolution subgraph`, `### 1.4 DAG invariants`, `### 1.5 Optional: Visual Skills` unchanged.

### 2.4 Insert "Promotion Economy" as §3a

**Insert** a new subsection **immediately after** the existing `## 3. Promotion / rollback rules` (keep §3.1–§3.4 intact; append §3a below them, then continue to §4):

```markdown
### 3a. Promotion economy

The acceptance gate in §3.1 is pass/fail; the **promotion economy** scores *which* passing proposals actually update the production snapshot. Every candidate is evaluated along these dimensions:

| Dimension | Source signal |
|-----------|----------------|
| Contract validity | §3.1 stage 1–2 |
| Symbolic consistency | §3.1 stage 2 |
| Replay pass rate | §3.1 stage 3 |
| Non-regression delta | §3.1 stage 4 on fixed eval slice |
| Transfer lift | §6.2 cross-domain slice |
| Cost overhead | §7 budget consumption per use |
| Novelty | Coverage delta vs. current bank |
| Uncertainty reduction | `GroundingRecord` / `InnerHopRecord` deltas on replay |

A proposal is **promoted** only if it passes §3.1 **and** improves a declared aggregate of the above (configurable per proposal class). Otherwise it is **quarantined** — kept as a candidate with its score vector attached so the Crafter can iterate.
```

### 2.5 Insert "Harness-Native Transfer Protocol" as §3b

**Insert** immediately after §3a:

```markdown
### 3b. Harness-native transfer protocol

Cross-domain transfer is a first-class harness workflow, not an implicit Crafter behavior. The protocol has five ordered stages:

1. **Source discovery.** Identify reusable skills in a source domain using bank telemetry (support, contract validity, transfer-readiness score).
2. **Canonicalization.** Rewrite the skill using shared typed slots and shared inner primitives (`GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE`). Strip domain-specific surface.
3. **Adapter binding.** Resolve target-domain entities, actions, and observations via the `AdapterRegistry` ([PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md)).
4. **Target replay.** Run the transferred skill against frozen target-domain traces under deterministic settings.
5. **Promotion or quarantine.** Apply the promotion economy (§3a) scoped to the *target* domain. Only verified transfers enter the target bank snapshot.

**Invariant.** Transfer never promotes by analogy alone; it always requires replay evidence in the *target* domain.
```

**Also update** `### 1.0 Layer map` (from 2.3) so the "Transfer Harness" row points to `§3b`.

### 2.6 Insert "Harness Ablations" as §6a

**Insert** immediately after `## 6. Evaluation matrix` (after §6.4):

```markdown
### 6a. Harness ablations

The Harness is itself a manipulated variable. The evaluation matrix must include ablations that isolate its contribution:

| Ablation | What is removed | What it measures |
|----------|-----------------|------------------|
| No Harness | Direct actor, no bank evolution, no gate | Upper bound of "vanilla" agent quality |
| No acceptance gate | Gate stages §3.1 disabled | Value of verified promotion |
| No replay verification | §3.1 stage 3 disabled | Value of frozen-trace replay |
| No slow-teacher proposals | Crafter `ComposeProposal` / `TransferProposal` disabled | Marginal value of 32B/72B synthesis |
| Frozen bank | All promotions blocked after initialization | Value of continual bank updates |
| No transfer workflow | §3b disabled; per-domain banks only | Value of cross-domain transfer |
| No evidence logging | L2 records omitted | Cost of debuggability, downstream attribution loss |

Each ablation must report the §6.1–§6.3 metric suite and the §6.4 slice breakdown. Results live under `artifacts/runs/{run_id}/ablation/{variant}/summary.json`.
```

### 2.7 Update §10 (Related documents) cross-link to `../05-harness/PLAN-HARNESS.md`

Add a row to the §10 table:

| `../05-harness/PLAN-HARNESS.md` | Skill-invocation runtime (micro-harness); used by §1.1 `inner_mdp` for every skill invocation |

### 2.8 Verification for P1

- [ ] Opening no longer contains the strings "glue" or "not a new research module".
- [ ] New section headers present: `## 0. Why the Harness Matters`, `### 1.0 Layer map`, `### 3a. Promotion economy`, `### 3b. Harness-native transfer protocol`, `### 6a. Harness ablations`.
- [ ] All seven ablation rows reference existing §§ correctly.
- [ ] Terminology: every use of "Harness" (capital H) refers to the control plane; `SkillHarness` is only used for the class.

---

## 3. P2 — Edits to `../02-action-agent/PLAN-ACTION-AGENT.md`

**File state check.** Current section map (verified): `## 1. Architecture overview`, `## 2. Tiered model architecture`, `## 3. Skill-guided decision making`, `## 4. Reward computation`, `## 5. Two-level MDP`, `## 6. Co-evolution & GRPO decomposition`, `## 7. Integration with Visual Grounding`, `## 8. Two pipeline variants`, `## 9. Supported environments`, `## 10. Uncertainty-driven GROUND triggering`, `## 11. TODO`, `## 12. Implementation`.

### 3.1 Insert "Harness Contract" as §1a

**Insert** immediately after `## 1. Architecture overview` and **before** `## 2. Tiered model architecture`:

```markdown
## 1a. Harness contract

The Action Agent is a component *inside* the Harness ([PLAN-PIPELINE-ORCHESTRATOR.md §0](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)). It consumes a typed input bundle and emits typed records. This section defines that contract.

### 1a.1 Inputs from the Harness

| Field | Source | Purpose |
|-------|--------|---------|
| `structured_state` | Visual Grounding (`<state>` schema) | Typed observation |
| `schema_hash` | Grounding | Detect schema drift; pin retrieval |
| `bank_snapshot_id` | Harness promotion pointer | Freeze which skills are available this step |
| `budget_state` | [Orchestrator §7](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#7-budget-controller) | Remaining tokens / hops / tool calls |
| `evidence_refs` (within-episode) | [Orchestrator §4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping) | Clip/frame / DOM / desktop-object / tool-call IDs carried on `<state>` |
| `allowed_skill_scope` | Harness policy | Domain filter, quarantine filter |

### 1a.2 Outputs to the Harness

| Record | When emitted | Consumer |
|--------|--------------|----------|
| `InnerHopRecord` | Every inner action | Evidence Harness, GRPO |
| `ActionRecord` | Every outer step | Evidence Harness, Bank segmentation |
| `RewardRecord` | Every outer step | GRPO, acceptance gate non-regression |
| `FailureSignal` | On stall / contract violation | Crafter input, failure taxonomy |
| `SkillUseTrace` | On skill invocation | `SkillHarness`, bank telemetry |

### 1a.3 Invariant

The Action Agent does **not** read from or write to the bank directly. All bank access goes through `bank_snapshot_id` + the retrieval interface, and all bank updates go through Harness-gated promotion.
```

### 3.2 Tighten the 8B / 32B-72B story inside §2

**Append** the following subsection at the end of `## 2. Tiered model architecture` (after the last existing subsection; do not rewrite existing §2 content):

```markdown
### 2.x Tier split as a Harness decision

The tier split is intentional and enforced by the Harness:

- **8B (Tier 2)** — bounded online control under structured state, valid-action constraints, and active-skill tracking. Trainable via GRPO on the fast timescale.
- **32B / 72B (Tier 1)** — offline-only synthesis, diagnosis, transfer, and protocol proposal. Never in the online action loop.
- **Harness** — the layer that decides *when* offline proposals are created, *how* they are verified, and *whether* they are allowed to affect the online actor.

Breaking this split (e.g., calling 72B in the hot path) is a harness-level violation and is blocked by the budget controller.
```

### 3.3 Insert "Failure Taxonomy" as §10a

**Insert** immediately after `## 10. Uncertainty-driven GROUND triggering` and **before** `## 11. TODO`:

```markdown
## 10a. Failure taxonomy

Every step-level or episode-level failure emits one or more labels on the `FailureSignal` record so the Harness can route it to the right downstream consumer.

| Label | Meaning | Primary consumer |
|-------|---------|-------------------|
| `grounding_failure` | Schema incomplete or wrong entity/relation set | Grounding re-ground / Path C escalation |
| `retrieval_failure` | No bank hit or wrong skill retrieved | Bank `skill_select` training data |
| `skill_mismatch` | Retrieved skill's preconditions unmet at execution | Crafter `PatchProposal` |
| `protocol_following_failure` | Inner-hop chain deviated from retrieved skill's protocol | Actor GRPO negative reward |
| `execution_failure` | Env rejected the action despite protocol compliance | Adapter registry, domain-level debug |
| `reward_shaping_mismatch` | `r_env` / `r_follow` / `r_cost` produced misleading signal | Reward-shaping audit |
| `schema_drift` | `schema_hash` changed mid-episode; previously grounded claim–evidence links no longer resolve | Re-issue `GROUND` for affected slots; orchestrator alert (the episode-local trajectory in §4 is the only state-keeping surface, so re-grounding rebuilds the affected slice directly) |

Labels are attached to the replay slice used by the acceptance gate so that Crafter proposals targeting a label are evaluated against exactly the failures they claim to fix.
```

### 3.4 Re-frame §5 ("Two-level MDP") as the stable reasoning API

**Insert** a lead paragraph at the top of `## 5. Two-level MDP (long-horizon reasoning)`, immediately after the section header and before the existing content:

> **Framing.** The inner MDP is the **stable reasoning API** the Action Agent exposes to the Harness. Its action alphabet (`GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE`) is what makes reasoning trajectories *typed, replayable, and transferable* across domains. Everything below is the mechanism; the API itself is a Harness-level commitment and must not be domain-specialized without a Harness-level change.

Do not modify the rest of §5.

### 3.5 Verification for P2

- [ ] New sections present: `## 1a. Harness contract`, `### 2.x Tier split as a Harness decision`, `## 10a. Failure taxonomy`.
- [ ] §5 opens with the "stable reasoning API" framing paragraph.
- [ ] Every output record name in §1a.2 matches an existing record in `PLAN-PIPELINE-ORCHESTRATOR.md §2.2`. If a name is new (`FailureSignal`, `SkillUseTrace`), add it to the orchestrator §2.2 list as part of this phase.

---

## 4. P3 — Edits to `../03-skill-bank/PLAN-SKILL-BANK.md`

**File state check.** Current section map (verified): `## 0`, `## 0.5`, `## 1`, `## 1.5`, `## 2`–`## 14`.

### 4.1 Reframe the opening of §0

**Prepend** a new paragraph to `## 0. Goal` (before the existing text):

> **Framing.** The Skill Bank is **not an autonomous store**. It is a **Harness-governed asset base** whose contents, versions, transfers, promotions, and retirements are controlled by verified orchestration policies ([PLAN-PIPELINE-ORCHESTRATOR.md §3, §3a, §3b](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)). Everything below describes the internal structure of that asset base; the *governance* is centralized in the Harness.

### 4.2 Insert "Transfer Readiness" as §4a

**Insert** immediately after `## 4. Skill data model` and **before** `## 5. Phase and context detection`:

```markdown
## 4a. Transfer readiness

Every `SkillProfile` carries a `transfer_readiness` block consumed by the Harness transfer protocol ([Orchestrator §3b](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3b-harness-native-transfer-protocol)).

| Field | Meaning |
|-------|---------|
| `source_domains` | Domains where the skill was originally mined or learned |
| `validated_target_domains` | Domains where transfer has passed replay + promotion |
| `adapter_requirements` | List of `(target_domain, adapter_id)` needed for binding |
| `transfer_risk` | Scalar in [0, 1] from past transfer verdicts |
| `replay_support_count` | Number of frozen-trace replays supporting current contract |
| `evidence_count` | Distinct episodes contributing to the profile |
| `retirement_risk` | Scalar based on usage decline + contract-validity drop |

These fields are *read-only from the Action Agent's perspective*; they are maintained by the Harness during promotion and maintenance.
```

### 4.3 Insert "Dual Storage Model" as §4b

**Insert** immediately after §4a:

```markdown
## 4b. Dual storage model

The bank separates **protocol storage** (what the skill *is*) from **execution evidence** (how it has *behaved*). This separation is what makes the promotion economy (§3a) and transfer protocol (§3b) auditable.

### 4b.1 Protocol store

Human-readable skill definitions:

- `trigger`
- `slots` (typed, over shared schema)
- `protocol` (inner-hop chain over shared primitives)
- `effects` (typed effect families)
- `adapters` (domain bindings)
- `success_criteria` / `abort_criteria`

### 4b.2 Execution evidence store

Operational history:

- invocation counts per domain
- success / failure rates stratified by failure label (§10a of Action Agent)
- replay support set (which frozen traces the current contract passed on)
- transfer outcomes (target domain × verdict)
- typical failure modes (top-k clusters)

Promotion/rollback decisions combine both stores; Crafter proposals must cite both a protocol change *and* the evidence slice that motivated it.
```

### 4.4 Insert "Lifecycle Under the Harness" as §4c

**Insert** immediately after §4b:

```markdown
## 4c. Lifecycle under the Harness

```
candidate → staged → canary → promoted → adapted → merged/split → retired/quarantined
```

Each transition is a Harness event:

| Transition | Trigger | Gate |
|------------|---------|------|
| `candidate → staged` | Crafter proposal passes static contract check | §3.1 stages 1–2 |
| `staged → canary` | Replay pass on source domain | §3.1 stage 3 |
| `canary → promoted` | Non-regression + promotion economy score | §3.1 stage 4 + §3a |
| `promoted → adapted` | Transfer protocol succeeds in a new domain | §3b |
| `adapted → merged/split` | Maintenance job | §4b.2 evidence-driven |
| `any → retired/quarantined` | Regression, gate failure, or retirement risk above threshold | §3.3 rollback rules |

This is a *refinement* of the existing staging/maintenance flow (§7, §8); it is not a parallel system.
```

### 4.5 Strengthen §1.5 with "shared primitives as transfer substrate"

**Prepend** to `## 1.5. Cross-task transfer objective` (before existing content):

> **Central claim.** The shared primitive vocabulary (`GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE`) plus shared typed slots are the **transfer substrate** of the system. Skills transfer not because domains are visually similar, but because they can be expressed through the same typed reasoning-control operators. Any extension to the primitive vocabulary is a Harness-level change requiring coordinated updates across `PLAN-ACTION-AGENT.md §5`, this file's §1.5, and `PLAN-SKILL-CRAFTER.md §4`.

### 4.6 Verification for P3

- [ ] New sections present: `## 4a. Transfer readiness`, `## 4b. Dual storage model`, `## 4c. Lifecycle under the Harness`.
- [ ] §0 opens with the "Harness-governed asset base" framing.
- [ ] §1.5 opens with the "transfer substrate" claim.
- [ ] Every lifecycle transition in §4c points to an existing gate in orchestrator §3.

---

## 5. P4 — Edits to `../04-skill-crafter/PLAN-SKILL-CRAFTER.md`

**File state check.** Current section map (verified): `## 1. Motivation`, `## 2. Architecture`, `## 3. Skill Composer`, `## 4. Skill Generalizer`, `## 5. Skill Hypothesizer`, `## 6. Failure Reflection`, `## 7–12`.

### 5.1 Narrow scope at the top of §1

**Append** a new paragraph at the end of `## 1. Motivation`:

> **Scope invariant.** The Crafter does exactly three offline jobs: (a) **failure-to-patch**, (b) **skill composition**, (c) **transfer adaptation**. It is *not* a general idea generator; every output is a typed proposal (§2a) and every proposal must cite evidence, a replay target, and an expected delta. Sections §3–§6 are the mechanisms for these three jobs; anything outside them belongs in a future plan, not this one.

### 5.2 Insert "Typed Proposal Outputs" as §2a

**Insert** immediately after `## 2. Architecture` and **before** `## 3. Skill Composer`:

```markdown
## 2a. Typed proposal outputs

Every Crafter run emits **exactly one** of four typed proposals, consumed by the Harness acceptance gate ([Orchestrator §3, §3a](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)).

| Proposal | Job | Produced by |
|----------|-----|-------------|
| `PatchProposal` | Failure-to-patch | §6 Failure Reflection |
| `ComposeProposal` | Skill composition | §3 Skill Composer |
| `TransferProposal` | Transfer adaptation (semantic skill + new adapter) | §4 Skill Generalizer + §7 Transferable families |
| `RetireProposal` | Retire or quarantine underperforming skills | Maintenance (§8 here, §4c in Skill Bank) |

### 2a.1 Required fields (all proposal types)

| Field | Meaning |
|-------|---------|
| `source_traces` | Episode + step IDs motivating the proposal |
| `failure_cluster_or_trigger` | Failure label(s) (§10a Action Agent) or positive trigger |
| `expected_effect_delta` | Predicted change on `RewardRecord` + contract-validity rate |
| `required_adapters` | For `TransferProposal`: target-domain adapter IDs |
| `replay_target_slice` | Frozen trace IDs the gate should replay against |
| `confidence` | Teacher self-reported calibration |
| `uncertainty_reduction_claim` | Predicted drop in re-ground rate / CHECK rate, if any |

### 2a.2 Gate compatibility

The acceptance gate in orchestrator §3.1 is parameterized by proposal type: `PatchProposal` runs stages 1–4 scoped to the failure cluster; `TransferProposal` additionally runs §3b target replay; `RetireProposal` only requires that §3.1 stage 4 does not *worsen* on the target slice.
```

### 5.3 Verification for P4

- [ ] §1 ends with the "three jobs" scope invariant.
- [ ] New section present: `## 2a. Typed proposal outputs`.
- [ ] Each proposal type in §2a maps to an existing §3/§4/§6 mechanism in this file.
- [ ] `RetireProposal` is linked to Skill Bank §4c lifecycle.

---

## 6. P5 — Edits to `../01-visual-grounding/PLAN-VISUAL-GROUNDING.md`

**Goal.** Enrich `GroundingRecord` so the Evidence Harness (L2) can log ambiguity and escalation signals that §10a of the Action Agent depends on.

**Single edit.** Find the subsection that defines `GroundingRecord` output fields (likely in the "Tool traces / Path A/B/C routing" area — do a local search for `GroundingRecord`). Add the following fields to the record schema (without removing existing fields):

| Field | Meaning |
|-------|---------|
| `unresolved_entities` | Entities referenced by the task/goal but not grounded this step |
| `uncertain_relations` | Relations whose confidence falls below a threshold |
| `ambiguity_class` | One of `{entity_identity, entity_count, relation_direction, state_value, temporal_anchor}` |
| `escalation_trigger` | Reason Path B/C was invoked, if any |
| `schema_revision_notice` | Flag + delta when the grounding head revised the `<state>` schema mid-episode |

**Additionally**, at the end of the section, add:

> These fields are consumed by (a) Action Agent §10a failure taxonomy (to label grounding failures precisely), (b) Orchestrator §4 episode-local evidence & trace bookkeeping (to re-issue `GROUND` for affected slots on schema revisions — re-grounding rebuilds the affected slice of current context directly, since the episode-local trajectory is the only state-keeping surface), and (c) the promotion economy §3a (as inputs to the uncertainty-reduction score).

**Verification for P5.** Grep `GroundingRecord` across the plans; every reference should be consistent with the enriched schema.

---

## 7. P6 — Edits to `../README.md`

### 7.1 Pipeline overview diagram

Under `## Pipeline overview`, append (do not replace the existing ASCII diagram):

```markdown
**The Harness wraps this pipeline.** All four stages (Grounding, Action Agent, Skill Bank, Skill Crafter) run under a single control plane that owns the typed trajectory interface, the acceptance gate, the promotion economy, and the transfer protocol. See [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md).
```

### 7.2 New one-paragraph blurb "Harness as the control plane"

Insert **immediately before** the `## Plan documents` section:

```markdown
### Harness as the control plane

The harness is the **unit of generalization** (one typed decision process across domains), the **unit of verification** (no skill or protocol change reaches production without passing the gate), and the **unit of attribution** (typed artifacts localize gains and regressions). The tier split (8B online, 32B/72B offline) is a harness decision, not a model choice. The full specification is [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md); the per-invocation skill runtime ([PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md)) is one component inside it.
```

### 7.3 Updated scope cells

Apply the §0 terminology fixes to rows 6 and 7 of the `## Plan documents` table.

### 7.4 Updated one-sentence framing

Replace the existing `## One-sentence framing` paragraph with:

> A **harness-governed cross-domain skill system** in which grounding, action, skill bank, and skill crafter operate under a shared typed trajectory interface, a verified promotion policy, and a native cross-domain transfer workflow — with an 8B online actor trained by GRPO and a frozen 32B/72B teacher whose proposals are admitted only through the harness acceptance gate.

**Verification for P6.** `grep -n "glue\|not a new research module" README.md` returns no matches.

---

## 8. P7 — Final verification pass

Run these checks after P1–P6. Each is a blocker for "edit plan complete".

### 8.1 Terminology grep

```bash
rg -n "\bharness\b" plans/ | less
```

Every hit must be one of:
- "the Harness" (capital H) = control plane
- `SkillHarness` = class name
- "skill-invocation runtime" = the micro-layer
- Inside a verbatim quote of old text being replaced

### 8.2 Cross-link integrity

For each new section anchor, verify that every cross-file link resolves. Manually check this list:

- `../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#0-why-the-harness-matters`
- `../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a-promotion-economy`
- `../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3b-harness-native-transfer-protocol`
- `../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#6a-harness-ablations`
- `../02-action-agent/PLAN-ACTION-AGENT.md#1a-harness-contract`
- `../02-action-agent/PLAN-ACTION-AGENT.md#10a-failure-taxonomy`
- `../03-skill-bank/PLAN-SKILL-BANK.md#4a-transfer-readiness`
- `../03-skill-bank/PLAN-SKILL-BANK.md#4b-dual-storage-model`
- `../03-skill-bank/PLAN-SKILL-BANK.md#4c-lifecycle-under-the-harness`
- `../04-skill-crafter/PLAN-SKILL-CRAFTER.md#2a-typed-proposal-outputs`

### 8.3 Record-type consistency

The union of record types referenced must match across files:

- `PLAN-PIPELINE-ORCHESTRATOR.md §2.2` is the **authoritative list**.
- `PLAN-ACTION-AGENT.md §1a.2` outputs (`InnerHopRecord`, `ActionRecord`, `RewardRecord`, `FailureSignal`, `SkillUseTrace`) must all appear in §2.2.
- If `FailureSignal` or `SkillUseTrace` is new, add a row to orchestrator §2.2 during P2 (already called out in §3.5 verification).

### 8.4 Ablation table sanity

Every ablation in orchestrator §6a must be *toggleable* via an existing config knob or a clearly identifiable component to remove. If any ablation requires code that does not exist yet, mark it `FUTURE` in the table so it is not mistaken for an executable experiment.

### 8.5 No scope creep

Confirm that none of the edits added:
- New trainable heads
- New agents beyond the three in README §Three-agent role split
- Any cross-episode storage layer (orchestrator §4 is the only state-keeping surface and is strictly episode-local)
- New inner primitives beyond `GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE`

If any did, revert to the authoritative definition and link.

---

## 9. What this edit plan deliberately does *not* do

To keep the refactor tight and reviewable:

- **No new plan files.** Every change is an in-place edit of an existing file, except this edit plan itself.
- **No algorithm changes.** Grounding routing, MDP action set, GRPO losses, bank pipeline stages, and Crafter mechanisms are untouched.
- **No renaming of files.** Terminology is reconciled by disambiguation blocks (§0), not by moving files.
- **Episode-local trajectory is the only state-keeping surface.** Orchestrator §4 carries the current `<state>`, short typed hop trace, intermediate belief state, within-episode evidence references, and claim–evidence links — and nothing else. All enrichment happens via the enriched `GroundingRecord` (P5) and the claim–evidence links carried on `<state>`.
- **No promotion-economy numerics.** Weights and thresholds are deferred to implementation; this plan only fixes the *dimensions* scored.

---

## 10. One-sentence strategic summary

> The best version of this project is not "grounding + action + bank + crafter"; it is **a harness-governed cross-domain skill system where all four components operate under a shared typed trajectory interface, a verified promotion policy, and a native transfer workflow** — and the purpose of this edit pass is to make the plan files say that plainly.
