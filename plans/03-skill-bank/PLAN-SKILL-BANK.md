# PLAN: Cross-Task Skill Bank for Reasoning and Control

> **Lane decision (2026-05-01) — lane (a), Context-only skills.** A
> skill in this bank is a *retrieval payload* — name, summary,
> strategic description, contract (preconditions / effects /
> evidence-roles), tags, optional NL or typed protocol consumed *as
> guidance text* — that the actor LLM consults during decision-making.
> Skills are **not** runnable programs at training time. Authoritative
> record: [`implementation_notes/legacy/skill-lane-decision.md`](../../implementation_notes/legacy/skill-lane-decision.md).
> Practical implications for this plan:
>
> * **Multi-domain `ACTIVE` invariant (`feasible_domains ≥ 2`) is
>   replaced by `min_retrievals_per_skill`** under lane (a) (T1.3d, S2). The
>   §0.1 "general protocol" framing stays — it remains the *source-of-
>   truth* for cross-domain transfer evidence — but cold-start single-
>   domain skills are now eligible for `ACTIVE` once they meet a
>   retrieval-utility floor. The lane-(b) invariant is preserved in
>   tree as the lane-(b) rollback path.
> * **Typed protocols and effect predicates** are gate evidence, not
>   runtime substrate. The harness's offline `gate_runner.py`,
>   `replay_validator.py`, `gymv_executor.py`, and `few_shot_adapter.py`
>   continue to consume them; the live actor only reads NL surfaces
>   from the retrieval payload.
> * **`SkillRepository.runnable()`** still gates which records the live
>   trainer sees (`ACTIVE` ∪ `SHADOW`). The §17 keystone (T1.2) is to
>   run the offline promotion loop once so the runnable set is
>   non-empty at trainer launch.
> * **Single-MDP companion decision (T3.6):** the actor consumes
>   skills from this bank with one MDP and two GRPO LoRAs
>   (`skill_selection` + `action_taking`). `hop_select` is a
>   non-target. Companion record:
>   [`implementation_notes/legacy/single-vs-two-mdp-tradeoff.md`](../../implementation_notes/legacy/single-vs-two-mdp-tradeoff.md).
>
> Sections below were authored under the lane-(b) assumption. Where
> they describe protocol dispatch / executable invocation, treat that
> as the *offline gate / diagnostic* surface unless the section is
> tagged otherwise.

**Scope:** Build and maintain a cross-domain Skill Bank from structured trajectories across **games, web agents, desktop / OS agents, short-video reasoning, visual reasoning, and embodied tasks**. The bank stores **transferable reasoning, grounding, and control skills** defined over shared state abstractions and verified outcome contracts, and exposes them to the [Action Agent](../02-action-agent/PLAN-ACTION-AGENT.md) through retrieval and selection APIs.

**Scope boundaries (deliberate).** Every skill in this bank is a **general protocol feasible across all five target domains** (game / webagent / os-agent / video-understanding / visual reasoning); see [§0.1](#01-general-protocol-invariant-no-domain-specific-skill-families). The **current execution/evaluation priority** is **short-video evidence-grounded reasoning** (Video-Holmes-style) — that is a deployment/measurement choice for adapters and eval slices, **not a narrowing of the skill ontology**. The bank carries no long-video assumptions; everything it consumes and emits is grounded in the orchestrator's episode-local trajectory (see [PLAN-PIPELINE-ORCHESTRATOR.md §4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping)).

**Upstream:** Structured episode trajectories from the Action Agent; structured schemas from [Visual Grounding](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md); intermediate reasoning traces; visual grounding outputs; within-episode evidence references (clip/frame / DOM / desktop / tool-call IDs); execution outcomes / verification signals. These inputs may come from multiple domains, but are converted into a shared typed structured representation before skill discovery and maintenance.
**Downstream:** Skill guidance consumed by the Action Agent and the Reasoning Agent; skill contracts used for reward shaping; bank curation consumed by GRPO-based training loops.
**Co-evolves with:** [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) (which composes and creates new skills); [Visual Skills](../01-visual-grounding/PLAN-VISUAL-SKILLS.md) (optional grounding strategy layer).

---

## 0. Goal

Build and maintain a **Skill Bank** from structured trajectories across **games, web agents, desktop / OS agents, short-video reasoning, visual reasoning, and embodied tasks**. The Skill Bank should discover reusable skills, learn their protocols and symbolic contracts, and expose them to the Action Agent through retrieval and selection APIs.

The key design goal is to store **transferable reasoning, grounding, and control skills**, not only environment-specific action motifs. A skill should be reusable across tasks through shared state abstractions, shared inner primitives, and verifiable outcome contracts.

### 0.1 General-protocol invariant (games as the foundry, other domains as few-shot transfer targets)

**Every skill in the bank is a general protocol** written over the shared `<state>` schema (§3) and the shared inner primitives (§1.5). A skill is only admitted if its protocol is **feasible across all five target domains** — game, webagent, os-agent, video-understanding, visual reasoning — through adapter binding (§4.3). There is no "short-video skill family," no "browser-only skill family," no per-domain sub-bank.

The five domains are *not symmetric*. Games (the `gymv` adapter, see [§0.4](#04-source-domain--transfer-target-asymmetry)) are the **source domain** in which skills are first mined, abstracted, and stress-tested under dense verifiable reward. Webagent, os-agent, video, and visual reasoning are **transfer targets**: their adapter bindings are *claimed* up front (so we never approve a skill that is structurally inadmissible elsewhere) but only become *verified* after the skill passes the few-shot adaptation stage of the gate ([Stage 3a](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md), see also [PLAN-HARNESS.md `FewShotAdapter`](../05-harness/PLAN-HARNESS.md)). A "general protocol" therefore means: *one typed protocol over evidence; adapter bindings to all five domains; lineage in the source domain; verified entries in target domains earned shot-by-shot*.

Examples of general protocols (the bank's actual content):

| General protocol | What it *means* (domain-independent) |
|------------------|--------------------------------------|
| `locate_filter_select` | Find-and-pick among typed candidates using a role / filter criterion |
| `blocker_prerequisite_replan` | Detect a blocker, resolve its prerequisite, then resume the original target |
| `disambiguate_target` | Resolve reference ambiguity from surrounding evidence and relations |
| `collect_evidence_chain` | Assemble a claim-backing evidence set until a sufficiency criterion is met |
| `verify_constraint` | Check a typed predicate against grounded evidence before committing |
| `actor_action_binding` | Bind an actor entity to an action entity via relations in `<state>` |

Each protocol instantiates across every target domain via the same `candidate_set` / `target` / `blocker` / `constraint` / `history_anchor` slots (see [Visual Skills §0.2](../01-visual-grounding/PLAN-VISUAL-SKILLS.md#02-cross-domain-candidate_set--one-abstraction-five-bindings) for the five-domain binding table).

### 0.2 What "execution priority" does and does not mean

Narrowing the **execution/evaluation target** to short-video (Video-Holmes-style) is a **deployment and measurement choice**, not a skill-ontology choice. Concretely:

- **What narrows:** which adapters are implemented first, which replay slices are built first, which transfer-failure diagnostics are exercised first (see [PLAN-HARNESS.md §10a.2](../05-harness/PLAN-HARNESS.md)), which `DomainEvalMatrix` slice is populated first.
- **What does not narrow:** the skill format, the protocol vocabulary, the effect families, the typed slots, the promotion gates, or which protocols the Crafter is allowed to synthesize. A skill that only works on short video is, by construction, *not a skill* for this bank — it is a failed transfer candidate and belongs in `known_failure_modes` / `do_not_transfer_if` (see §4.3b).

When short-video evaluation exercises `collect_evidence_chain` first, it is exercising a **general protocol** whose adapter bindings for game / webagent / os-agent / visual reasoning exist from day one; the short-video arena is simply where the first `verified_domains` entry is written.

### 0.3 Evidence-driven invariant (no opaque skills)

**Every skill in the bank is evidence-driven and exists only to assist reasoning / decision-making.** Together with [§0.1](#01-general-protocol-invariant-no-domain-specific-skill-families), this is a hard admission rule enforced by the Harness gate (see [PLAN-HARNESS.md §10 Gate G0](../05-harness/PLAN-HARNESS.md)); it is not a stylistic guideline. A skill is admitted only if it satisfies **both** clauses below.

**Clause A — Evidence interface (mechanically enforced).**
Every successful `SkillEpisode e(s)` must record at least one of:

- `evidence_in: List[EvidenceRef]` — references read by `s` from `<state>.evidence_refs` (clip/frame IDs, DOM node IDs, desktop element IDs, tool-call IDs, or prior inner-hop outputs in the same outer step);
- `evidence_out: List[EvidenceRef]` — the evidence delta written by `s` (new grounding records, new claim–evidence links, new verification outcomes, new hypothesis-with-warrant entries).

An episode whose `evidence_in ∪ evidence_out = ∅` is an **opaque-skill violation** and rejects the skill at promotion. `evidence_in ∪ evidence_out ≠ ∅` is a hard precondition for entering the bank.

**Clause B — Evidence role (declared, typed, checked).**
Every skill declares exactly one `evidence_role` drawn from the closed set below. This is **orthogonal to** the task-effect taxonomy in [§8](#8-effect-families-and-skill-hierarchy) (Acquisition / Verification / Tracking / …): §8 is about *what state change* a skill produces in the world or belief; `evidence_role` is about *what role* the skill plays in the evidence-and-decision contract. Both fields are carried on every skill; §8's label must be consistent with `evidence_role` (e.g., a §8 `Verification` skill has `evidence_role = VERIFY`).

`evidence_role` determines which evidence fields are required at episode time and which inner-MDP action may invoke the skill (see [PLAN-ACTION-AGENT.md §5](../02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control)).

| `evidence_role` | Purpose | Required episode fields | Invokable under inner action |
|-----|-----|-----|-----|
| `GATHER` | produce new evidence (grounding, retrieval, inspection, segmentation) | `evidence_out ≠ ∅` | `GROUND`, `RETRIEVE` |
| `VERIFY` | check evidence against a typed predicate (anchor, constraint, consistency) | `evidence_in ≠ ∅`; `verdict ∈ {PASS, FAIL, INSUFFICIENT}` | `CHECK` |
| `REASON` | derive a new hypothesis, ranking, or sufficiency judgment warranted by a recorded evidence subset | `evidence_in ≠ ∅`; `warrant: List[EvidenceRef]` (subset of `evidence_in`) | `CONCLUDE` |
| `COMMIT` | emit a decision / action / answer with explicit evidence backing | `evidence_warrant: List[EvidenceRef]` (non-empty, required) | `COMMIT`, `EXECUTE` |

No other `evidence_role` is admissible. In particular:

- **Pure motor macros** (action sequences with no evidence warrant) are **not** skills.
- **Pure templates** (e.g., prompt templates that don't touch `<state>`) are **not** skills.
- **Pure planners / decomposers** that don't themselves cite what they conditioned on are **not** skills as a standalone entry; if they condition on `<state>` they become `REASON` skills with the conditioning evidence as `evidence_in`; otherwise they belong in the Crafter's `ComposeProposal` path as a composition of evidence-driven sub-skills (see [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)).

The right-most column is part of the Action Agent contract: the Action Agent MAY NOT invoke a skill whose `evidence_role` does not match the inner-MDP action it is instantiating. Mismatches are raised as `contract-violation: skill-role-mismatch` events by the Harness.

### 0.4 Source-domain / transfer-target asymmetry

The five canonical domains split into two roles:

| Role | Members (`common.enums.SOURCE_DOMAINS` / `TRANSFER_TARGET_DOMAINS`) | What the bank does here |
|------|--------------------------------------------------------------------|-------------------------|
| **Source domain (foundry)** | `gymv` (game) | Mine candidate skills, run the bulk of training rollouts, harden under dense verifiable reward, stress-test contracts, populate `false_binding_patterns`. *Every active skill must have a source-domain lineage.* |
| **Transfer target** | `browser`, `osworld`, `video`, `visual_reasoning` | Receive skills *via few-shot adaptation only*. Each skill must declare an adapter binding here, but the binding is **provisional** until it passes [Stage 3a](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) on a small budget of target-domain demonstrations. |

The asymmetry shows up on every `SkillRecord` (see [§4.3a](#43a-lineage--provenance)) as three required fields:

| Field | Constraint | Set by |
|-------|------------|--------|
| `source_domains` | ⊆ `SOURCE_DOMAINS`; non-empty for any record that ever reaches `ACTIVE` | Mining + Crafter |
| `transfer_target_domains` | ⊆ `TRANSFER_TARGET_DOMAINS`; the bindings the skill *claims* | Crafter (`GeneralizeProposal`) |
| `verified_domains` | ⊆ `DOMAINS`; populated when Stage 3a passes for a target. **Written only by `SkillLifecycleManager.record_transfer_verification(...)`**, called from `PromotionOrchestrator.promote(...)` based on the `GateVerdictPayload`. | `SkillLifecycleManager` (sole writer), driven by `GateService` outputs. Never the Crafter, Harness, Actor, or any direct bank caller. |

**Hard rules enforced by `SkillLifecycleManager` ([§7a](#7a-unified-skill-lifecycle-and-promotion-ownership)):**

1. A skill cannot be promoted to `ACTIVE` if `source_domains` does not intersect `SOURCE_DOMAINS` — i.e. without a game-foundry lineage. (Legacy records produced before this invariant was introduced fall back to the older "≥2 feasible_domains" check; see `skill_bank/lifecycle.py::_validate_invariants`.)
2. A skill cannot be promoted to `ACTIVE` if `verified_domains` does not contain at least one element of `TRANSFER_TARGET_DOMAINS` — i.e. without a few-shot transfer success in at least one non-game arena.
3. `SkillLifecycleManager.record_transfer_verification(...)` is the **only** writer of `verified_domains` and `adapter_history`; it is invoked by `PromotionOrchestrator.promote(...)` *before* the status transition so the ACTIVE invariant sees the just-written list. The Crafter, Harness, and Actor are read-only on these fields.

Why this asymmetry: games are the only domain where we cheaply get all four properties needed to *learn* multi-hop reasoning skills — dense rewards, deterministic resets, hard-to-game verification, and rich evidence chains over a controllable visual state. Other domains are excellent *evaluators* of the resulting skills (and excellent sources of failure modes that drive Crafter repairs) but are not where the skills are first discovered. The gate's [Stage 3a few-shot adaptation](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) is the bridge.

---

## 0.5. Core design principle: skill as a structured-state program

A skill is a **reusable reasoning-and-control program over a typed structured state**.

Instead of defining skills as raw environment-dependent action macros, we define them as:

- a typed applicability condition over structured state
- an executable protocol composed of shared reasoning/control primitives
- a symbolic outcome contract describing expected state changes
- a transfer interface that binds the abstract skill to different domains via adapters

This lets the same high-level skill transfer across:
- games
- web / UI environments
- desktop / OS agent environments
- short-video reasoning settings (first eval target)
- visual reasoning tasks
- embodied / robotics tasks

In this plan, the Skill Bank stores **reasoning templates + contracts**, while environment-specific parsing, grounding, and action realization are delegated to adapters.

---

## 1. Architecture overview

### High-level loop

```
Structured episode trajectories (from any domain)
    ↓
Stage 1: Boundary Proposal       — candidate cut points from signals + state deltas
    ↓
Stage 2: Segmentation            — decode segments + skill labels via preference learning
    ↓
Stage 3: Contract Learning       — learn/verify/refine symbolic contracts per skill
    ↓
Stage 4: Bank Maintenance        — propose → filter (CURATOR) → execute (split/merge/refine/promote)
    ↓
Stage 4.5: Quality Evaluation   — score on effectiveness, transferability, contract validity
    ↓
Query / Select API               — action agent retrieves skills with structured guidance
```

The bank co-evolves with the policy: improved skills improve policy rollouts, and improved rollouts improve the bank.

### Upstream inputs

The Skill Bank consumes:
- structured episode trajectories
- intermediate reasoning traces (inner MDP hop chains)
- visual grounding outputs (entities, relations, evidence)
- within-episode evidence references (clip / frame / DOM / desktop-object / tool-call IDs)
- execution outcomes / verification signals

These inputs may come from multiple domains, but are converted into a shared typed structured representation before skill discovery and maintenance.

### Downstream consumers

The Skill Bank serves:
- the Action Agent (skill selection, protocol following, active skill tracking)
- the Reasoning Agent (hop chain templates, evidence strategies)
- bank curation and GRPO-based training loops

The bank's only persistent surface is its own snapshots; everything else — entity references, evidence pointers, intermediate belief state — is read from and written to the orchestrator's episode-local trajectory (see [Pipeline Orchestrator §4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping)).

---

## 1.5. Cross-task transfer objective

The bank should support **cross-task transfer** through three mechanisms:

### 1. Shared state abstraction

Different environments are mapped to a common typed state interface (§3). The canonical `<state>` schema (defined in [Visual Grounding §3](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md#3-canonical-schema)) provides the shared representation: entities, attributes, relations, state_flags, targets, uncertainty. All skill preconditions and effects are written over this shared schema.

### 2. Shared inner primitives

Skills are written using reusable reasoning/control primitives from the inner MDP action vocabulary (see [Action Agent §5](../02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control)):

| Primitive | Purpose |
|-----------|---------|
| `GROUND` | Locate / bind entities from visual or structured input |
| `CHECK` | Verify a relation, attribute, or constraint |
| `RETRIEVE` | Query the skill bank (by `bank_snapshot_id`) for a reusable skill/protocol; within-episode evidence is already in `<state>.evidence_refs` |
| `COMMIT` | Commit an intermediate result or subgoal |
| `ACT` / `EXECUTE` | Emit an environment action (exits inner loop) |
| `VERIFY` | Confirm that expected effects hold after execution |

### 3. Adapter-based binding

Each environment provides an adapter that:
- parses observations into structured state
- grounds entities / relations / evidence
- binds abstract actions to concrete actions
- reports verification signals back to the bank

This means skills transfer through **state + protocol + contract**, not through raw action strings. See [Visual Skills §6](../01-visual-grounding/PLAN-VISUAL-SKILLS.md#6-separating-semantics-from-execution) for the full semantic/execution separation and the abstract-operator-to-domain mapping table.

---

## 2. Five-stage bank pipeline

All stages operate on **typed structured trajectories** rather than only task-local action traces.

### Stage 1: Boundary Proposal

Given a long trajectory, propose candidate boundaries for reusable skill segments.

Signals may include:
- abrupt changes in local reward / progress
- predicate flips or state deltas
- phase transitions (see §5)
- action-mode switches (grounding → reasoning → action)
- intention / subgoal changes
- evidence handoffs (claim → supporting `evidence_refs`)
- grounding failures or uncertainty spikes reported by adapters
- tool usage pattern changes

Not GRPO-wrapped (algorithmic from predicates + signals).

- **Implementation:** `skill_agents/boundary_proposal/`

### Stage 2: Segmentation / skill decoding

Given candidate segments, decode them into known skills or propose `__NEW__` skills.

Segmentation operates over:
- typed state transitions
- reasoning traces (inner MDP hops)
- action traces (outer MDP actions)
- grounding / evidence traces

The decoder may produce:
- action-level skills
- reasoning-level skills (hop chain templates)
- mixed reasoning-and-control skills
- grounding-level skills (multi-step perception strategies, see [Visual Skills §7](../01-visual-grounding/PLAN-VISUAL-SKILLS.md#7-grounding-skill-bank))

If no existing skill fits well, assign `__NEW__` and forward to contract learning / bank maintenance.

**Scoring formula (6 terms):**

```
Score(i, j, k | k_prev) =
    1.0 * behavior_fit       [LLM preferences — Bradley-Terry]
  + 2.0 * intention_fit      [per-step phase:tag compound labels]
  + 0.3 * duration_prior     [Gaussian]
  + 1.0 * transition_prior   [LLM preferences — Bradley-Terry]
  + 0.0 * contract_compat    [Stage 3 feedback, off by default]
  + 0.5 * boundary_preference
```

**Compound skill labels:** Phase detector + intention tags → `"endgame:MERGE"`, `"opening:POSITION"`, etc. Same tactic in different phases becomes a distinct skill.

**GRPO:** SEGMENT LoRA wraps `collect_segment_preferences()`; reward = `SegmentationDiagnostics`.

- **Implementation:** `skill_agents/infer_segmentation/`

### Stage 3: Contract Learning

For each candidate skill, infer:
- what preconditions tend to hold (over the typed structured state)
- what effects are consistently produced (world-effects and belief/grounding-effects)
- what evidence or verification signals support those effects
- what parts are cross-domain invariants vs domain-specific bindings

The goal is to learn a compact symbolic contract that can be used for:
- retrieval (effect-based search)
- reward shaping (r_follow from eff_add predicates)
- verification (contract pass rate)
- transfer (shared predicates across domains)

Contracts feed back into Stage 2 via `compat_fn`.

**GRPO:** CONTRACT LoRA wraps `llm_summarize_contract()`; reward = `verify_effects_contract().overall_pass_rate`.

- **Implementation:** `skill_agents/stage3_mvp/`

### Stage 4: Bank Maintenance

**Propose → Filter → Execute** flow:

1. **Propose:** Build `SkillProfile` per skill, propose candidates (refine, merge, split, materialize, promote, retire).
2. **Filter:** CURATOR LoRA — approve/veto/defer per candidate.
3. **Execute:** Approved actions run (refine = weaken + strengthen; merge/split with alias map).

Maintenance decisions should consider:
- protocol similarity
- contract similarity (shared effect predicates)
- signature compatibility (typed slot overlap)
- transferability across domains (domain coverage)
- verification consistency (pass rate stability)
- usage frequency / utility (selection rate by action agent)

New skills enter through **proto-skill staging:** `__NEW__` clusters → materialize → proto-skill → verify → promote.

**GRPO:** CURATOR LoRA wraps `filter_candidates()`; reward = `bank_quality_delta`.

- **Implementation:** `skill_agents/bank_maintenance/`

### Stage 4.5: Quality Evaluation

Evaluate candidate or updated skills on:
- **In-domain effectiveness:** outcome_reward, follow_through, consistency, compactness
- **Cross-domain transferability:** domain coverage, slot coverage, protocol quality
- **Contract validity:** pass rate on holdout instances
- **Verification consistency:** stability across episodes
- **Grounding cost:** tool calls and latency per skill invocation
- **Evidence sufficiency:** whether evidence_required fields are satisfiable
- **Downstream utility:** selection rate and reward improvement for the action agent

Low-quality segments can be dropped before bank maintenance.

- **Implementation:** `skill_agents/quality/`

---

## 3. Unified structured state interface

All tasks should be mapped into a typed structured state representation. The canonical `<state>` schema (defined in [Visual Grounding §3](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md#3-canonical-schema)) provides the shared format. Skills are defined over this schema, not over raw observations.

### 3.1. State schema

```yaml
state:
  domain: str                    # gymv | browser | desktop | image_qa | video_qa | embodied
  task: str
  goal: str
  step: int

  entities: list[Entity]         # typed objects with attributes
  attributes: dict[eid, dict]    # per-entity key-value attributes
  relations: list[Relation]      # spatial, functional, temporal, grouping
  events: list[Event]            # state changes, temporal markers
  goals: list[Goal]              # active subgoals and completion status

  state_flags:
    progress: float              # 0–1
    phase: str                   # early | mid | late | domain-specific
    error: str | null
    dialog_open: bool
    input_pending: bool

  targets:
    target: eid | null
    blocker: eid | null
    constraint: str | null
    candidate_set: list[eid]
    history_anchor: eid | null

  uncertainty: dict[eid, str]    # per-entity confidence (high | medium | low)
  evidence_refs: list[str]       # within-episode evidence pointers (clip/frame, DOM node, desktop object, tool-call id)
  action_candidates: list[str]   # valid actions in the current state
```

### 3.2. Entity types (cross-domain ontology)

For skills to transfer, domain-specific objects must map into shared types. See [Visual Skills §5](../01-visual-grounding/PLAN-VISUAL-SKILLS.md#5-cross-domain-entity-ontology) for the full ontology.

| Ontology type | Purpose | Example domains |
|---------------|---------|-----------------|
| `selectable_entity` | Objects that can be targeted / selected | Buttons, icons, tiles, items, people in video |
| `interactive_entity` | Objects whose state can be changed | Form fields, levers, doors, switches |
| `container_entity` | Objects that contain other objects | Dropdowns, folders, chests, rooms, scenes |
| `textual_anchor` | Text that identifies or locates other entities | Labels, titles, score displays, subtitles |
| `navigable_region` | Areas that can be moved to / focused on | Page sections, desktop areas, map zones, timeline segments |
| `tracked_entity` | Objects that persist across time | Session elements, persistent windows, moving characters |
| `goal_indicator` | Objects that signal goal progress | Success messages, score thresholds, answer evidence |
| `blocking_entity` | Objects that prevent actions | Modals, disabled states, walls, occlusions |

### 3.3. Two kinds of effect predicates

Skills produce two kinds of state changes, both expressed as predicates over the schema:

| Effect type | Changes | Example predicates |
|-------------|---------|-------------------|
| **World-effects** | External environment state | `selected(target)=true`, `opened(container)=true`, `distance(agent,target) decreases` |
| **Belief/grounding-effects** | Internal binding / evidence / confidence state | `binding(target)=resolved`, `confidence(target)≥τ`, `candidate_count=1`, `evidence_collected=true` |

World-effects are the existing `eff_add` / `eff_del` / `eff_event` predicates. Belief/grounding-effects are new — they describe changes to the agent's understanding rather than changes to the environment. Both types use the same contract format.

---

## 4. Skill data model

Each skill has three logical parts:

### 4.1. Protocol store (what the action agent sees)

- `skill_id`, `name`, `strategic_description`, `tags`
- `skill_type`: `"reasoning"` | `"action"` | `"grounding"` | `"mixed"`
- `evidence_role` **(required, §0.3 Clause B)**: one of `GATHER | VERIFY | REASON | COMMIT`. Determines which inner-MDP action may invoke the skill and which evidence fields are required at episode time.
- `category`: task-effect family label used for retrieval/tagging (see §8); must be consistent with `evidence_role` (e.g., §8 `Verification` ⇒ `evidence_role = VERIFY`).
- `protocol`: steps (using shared inner primitives), preconditions, success_criteria, abort_criteria, expected_duration
- `typed_slots`: slot variables with ontology types (§3.2)
- `confidence`
- Used by `skill_bank_to_text()`, `query_skill_bank()`, and to set `active_skill_plan`

### 4.2. Contract (what the reward system and the Harness gate use)

- `eff_add`: predicates that should become true (world-effects + belief-effects)
- `eff_del`: predicates that should become false
- `eff_event`: event predicates
- `evidence_interface` **(required, §0.3 Clause A)**:
  - `evidence_inputs_spec`: typed description of `EvidenceRef` kinds the skill expects to read (may be empty only if `evidence_role == GATHER`)
  - `evidence_outputs_or_warrant_spec`: typed description of what the skill is expected to write, per its `evidence_role`:
    - `GATHER` → declared `evidence_out` kinds (grounding records, retrieved items, segmentation spans)
    - `VERIFY` → declared `verdict` domain and referenced evidence kinds
    - `REASON` → declared hypothesis schema plus `warrant` shape (which `evidence_in` subset is cited)
    - `COMMIT` → declared decision/action schema plus `evidence_warrant` shape (non-empty, required)
  - `opacity_check`: the Harness rejects any episode whose recorded `evidence_in ∪ evidence_out = ∅` (opaque-skill violation, §0.3 Clause A).
- `evidence_required` (legacy field, kept for reward shaping): which `evidence_out` entries must be present for a reward signal to fire; a subset of `evidence_outputs_or_warrant_spec`.
- Used for segmentation verification, reward shaping (r_follow), stage 2↔3 feedback, and **the Gate G0 evidence-driven contract check** in [PLAN-HARNESS.md §10](../05-harness/PLAN-HARNESS.md).

**The agent plans from protocols, is rewarded for making progress on the contract's eff_add predicates, and is gated on the evidence interface: a skill whose episodes do not touch evidence is not promoted regardless of its reward.**

### 4.3. Transfer interface (what enables cross-domain reuse)

- `slot_bindings`: maps typed slots to domain-specific schema fields (see [Visual Skills §3b](../01-visual-grounding/PLAN-VISUAL-SKILLS.md#3b-typed-slot-variables))
- `abstract_predicates`: parameterised eff_add/eff_del using `$slot` placeholders, with per-domain instantiations
- `domain_adapters`: per-domain execution realizations (how abstract operators map to concrete tools/actions)
- `transfer_hints`: domains where this skill has been validated
- `reasoning_protocol`: hop chain template using inner MDP actions

### 4.3a. Lineage / provenance

Every skill carries its own audit trail so the acceptance gate and the transfer protocol can reason about *why* it exists and *where* it has been proven to work. The starred fields are **required** under the source/target asymmetry (§0.4) and are validated by the `SkillLifecycleManager`'s ACTIVE-promotion check.

| Field | Required for ACTIVE? | Meaning |
|-------|---------------------|---------|
| `origin_trace_ids` | yes | Episode + step IDs of the trajectories from which the skill was mined or composed |
| `source_domains` ★ | **yes** | Foundry domains the skill was originally extracted from. Must intersect `SOURCE_DOMAINS = ("gymv",)`. |
| `transfer_target_domains` ★ | **yes** | The non-game adapter bindings the skill *claims*. Must be ⊆ `TRANSFER_TARGET_DOMAINS = ("browser", "osworld", "video", "visual_reasoning")`. |
| `verified_domains` ★ | **yes** (≥1 target) | Domains where the skill has passed replay **and** Stage 3a few-shot adaptation. Mutated *only* via `SkillLifecycleManager.record_transfer_verification(...)`, which `PromotionOrchestrator.promote(...)` calls based on the `GateVerdictPayload.eligible_domains` produced by `GateService`. |
| `failure_clusters` | no | IDs of failure clusters (see [Skill Crafter §6.7](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)) this skill was intended to patch |
| `promotion_reason` | yes | Short text + pointer to the `GateVerdict` that last promoted the current version |
| `rollback_reason` | conditional | Present only for retired/quarantined skills; short text + pointer to the triggering regression |
| `adapter_history` ★ | yes | List of `{target_domain, evaluation_id, verified_at, rationale, metrics: {pass_rate, k_used}}` binding attempts in append order; written exclusively by `SkillLifecycleManager.record_transfer_verification(...)` (one entry per verified target per Stage 3a run, accumulating across re-evaluations so the full transfer lineage is reconstructible). |

### 4.3b. Negative knowledge

The bank broadens transfer risk as it broadens domain coverage; negative knowledge bounds that risk.

| Field | Meaning |
|-------|---------|
| `anti_preconditions` | Typed predicates that, if true in `<state>`, make this skill *unsafe* to invoke (short-circuit negatives, the dual of `preconditions`) |
| `known_failure_modes` | Named failure patterns observed in replay / rollouts, each with a short signature (state predicates + trace shape) |
| `do_not_transfer_if` | Target-domain predicates that block the transfer protocol from attempting to bind this skill (e.g., `domain == video_qa AND requires_actuation`) |
| `false_binding_patterns` | Slot-binding patterns that historically looked plausible but produced contract failures (e.g., `candidate_set` from visual similarity without role filter) |

These fields are populated by: (a) failure reflection in the Crafter (see [PLAN-SKILL-CRAFTER.md §6](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)), (b) gate rollbacks in the orchestrator (§3.3), and (c) transfer-protocol quarantines in the Harness (see [PLAN-HARNESS.md §6](../05-harness/PLAN-HARNESS.md)).

### 4.4. Reasoning skills (inner MDP hop chain templates)

Under the two-level MDP (see [Action Agent §5](../02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control)), skills capture *how to think*, not just *what to do*. Each reasoning skill is a multi-step policy over inner MDP actions (GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE).

**Example reasoning skill:**

```
skill: constraint_satisfaction
skill_type: reasoning
evidence_role: REASON        # derives a blocker-resolution hypothesis from cited evidence
category: verification
evidence_interface:
  evidence_inputs_spec:  [blocker_grounding_record, constraint_check_result, retrieved_past_resolution]
  evidence_outputs_or_warrant_spec:
    hypothesis: {subgoal: str, blocker_id: EvidenceRef}
    warrant:    [blocker_grounding_record, constraint_check_result]   # non-empty subset of evidence_in
trigger: state_flags.error != null OR target.blocker != null
slots:
  blocker: blocking_entity
  constraint: str
protocol:
  hop1: GROUND(blocker entity)
  hop2: CHECK(what constraint is violated)
  hop3: RETRIEVE(similar past resolution)
  hop4: COMMIT(subgoal = resolve blocker first)
  hop5: EXECUTE(action addressing blocker)
effects:
  eff_add: [blocker_resolved, constraint_satisfied]
  eff_del: [blocker_active]
domain_adapters:
  game: EXECUTE → game-specific action
  browser: EXECUTE → click/type/navigate
  video: EXECUTE → temporal navigation + evidence collection
```

**Example cross-domain skill:**

```
skill: locate_filter_select
skill_type: mixed
evidence_role: COMMIT          # emits a target selection with non-empty evidence_warrant
category: acquisition
evidence_interface:
  evidence_inputs_spec:  [candidate_grounding_records, filter_check_results]
  evidence_outputs_or_warrant_spec:
    decision: {selected_target: EvidenceRef}
    evidence_warrant: [filter_check_results, candidate_grounding_records]   # non-empty
trigger: candidate_set is non-empty AND target is unresolved
slots:
  target: selectable_entity
  candidate_set: list[selectable_entity]
  filter_criterion: str
protocol:
  hop1: GROUND(candidate_set)
  hop2: CHECK(filter_criterion against each candidate)
  hop3: COMMIT(best candidate)
  hop4: EXECUTE(select best)
effects:
  eff_add: [target_selected, candidate_resolved]
  eff_del: [target_unresolved]
domain_adapters:
  game: candidates = legal moves → best legal move
  browser: candidates = UI elements → relevant control
  image_qa: candidates = detected objects → answer target
  video_qa: candidates = temporal moments → key frame
  embodied: candidates = reachable objects → grasp target
```

**Mapping COS-PLAY components to the unified system:**

| COS-PLAY Component | Role in Cross-Task Skills |
|---------------------|--------------------------|
| Skill Bank | Stores reusable **hop chain templates** with typed slots and domain adapters |
| Skill protocols (trigger, steps, abort/success) | Become **reasoning protocols** — when to ground, when to check, when to conclude |
| GRPO training | Optimizes the full reasoning chain end-to-end |
| Co-evolution loop | Discovers new reasoning patterns from trajectories across all domains |
| RAG + embeddings | Retrieves relevant reasoning templates for current visual/structured state |

**Trajectory format:** Under the two-level MDP, episode trajectories include both inner hops and outer actions:

```
(schema_0, GROUND, schema_0') → (schema_0', CHECK, schema_0'') → (schema_0'', EXECUTE(click), schema_1) → ...
```

These trajectories are segmentable by Stage 1–2 — reasoning hop chains become discoverable skills alongside action-level skills.

---

## 5. Phase and context detection

Per-step intention tags capture tactical intent; the phase detector adds strategic context. Phase detection generalizes across domains — not just games.

```
raw tag:      MERGE          MERGE          MERGE
phase:        opening        midgame        endgame
compound:     opening:MERGE  midgame:MERGE  endgame:MERGE   ← 3 distinct skills
```

### Domain-specific extractors

| Domain | Environment | State Feature | Phases |
|--------|------------|--------------|--------|
| Game | 2048 | Board occupancy + highest tile | opening, midgame, endgame |
| Game | Tetris | Board fill ratio | opening, midgame, endgame |
| Game | Super Mario | Mario x-position | early_level, mid_level, late_level |
| Game | Avalon | Round signals | team_building, quest, endgame |
| Game | Diplomacy | Turn/season signals | opening, orders, retreat, adjustment |
| Game | Candy Crush | Temporal position | early, mid, late |
| Browser | WebArena | Task completion % + page depth | exploration, form_filling, verification |
| Browser | MiniWoB++ | Element count + interaction history | identification, interaction, confirmation |
| Desktop | OSWorld | Window/app state | navigation, configuration, verification |
| Image QA | VisualToolBench/TIR-Bench | Evidence chain length | grounding, reasoning, answering |
| Video QA | SIV-Bench | Timeline position + evidence count | scanning, focusing, concluding |
| Video QA | Video-Holmes | Clue chain length + scene coverage | exploration, investigation, synthesis |

**Impact:** Phase detection makes the same abstract reasoning pattern (e.g., "filter candidates") produce distinct skills when applied in different strategic contexts. 2048 goes from 1 skill (MERGE) to 9 compound skills. Browser tasks gain phase-aware exploration vs. verification skills.

---

## 6. Query / Select API

### Simple retrieval

```python
results = agent.query_skill("propose team and vote on quest", top_k=3)
```

### Rich skill selection (preferred for action agent)

```python
results = agent.select_skill(
    query="propose team for quest",
    current_state={"is_leader": True, "quest_round": 2},
    domain="game",
    top_k=3,
)
# Each result has: skill_id, skill_type, relevance, applicability, confidence,
# matched_effects, missing_effects, protocol, micro_plan, failure_modes,
# domain_adapter (if available for current domain)
```

### Effect-based retrieval

```python
results = agent.query_by_effects(
    desired_add={"team_proposed", "quest_active"},
    desired_del={"waiting_for_leader"},
    top_k=3,
)
```

### Cross-domain transfer retrieval

```python
results = agent.query_transferable(
    source_domain="game",
    target_domain="browser",
    effect_family="acquisition",
    top_k=3,
)
# Returns skills with validated adapters for the target domain,
# or skills with high structural similarity that need adapter creation
```

### Scoring

Each candidate skill is scored on three axes:

| Component | Weight | Source |
|-----------|--------|--------|
| Retrieval relevance | 40% | RAG embedding cosine similarity + keyword Jaccard |
| Execution applicability | 35% | Effect compatibility against current state predicates + slot coverage check |
| Historical pass rate | 25% | Success rate from past executions (domain-aware when available) |

---

## 7. GRPO co-evolution

### Asymmetric co-evolution framework

The Skill Bank sits at the **medium timescale** within the three-agent co-evolution framework (see [Action Agent §6](../02-action-agent/PLAN-ACTION-AGENT.md#6-co-evolution--grpo-decomposition)). Its operational components (retrieval, scoring, tracking) update more frequently than the synthesis-reflection agent but less frequently than the actor.

**Co-evolution loop for the Skill Bank:**

```
Actor rolls out with current bank (fast timescale)
        ↓
Collect experience trajectories + skill-use statistics (from all domains)
        ↓
Skill Bank operational update (medium timescale):
  • Segmentation of new trajectories (typed structured)
  • Contract learning/refinement (world + belief effects)
  • Bank maintenance (propose → filter → execute)
  • Quality evaluation (including cross-domain transferability)
        ↓
Synthesis-reflection agent proposes (slow timescale):
  • New skills from failure reflection
  • Compositions from effect chaining
  • Cross-domain transfers (new adapters for existing skills)
  • "New skill vs. new adapter" decisions
        ↓
Acceptance gate (before anything enters the bank):
  1. Contract completeness checks against schema
  2. Retrieval compatibility with current bank entries
  3. Replay or held-out verification on stored trajectories
  4. Non-regression filtering (must not lower pass rate on prior successes)
  5. Cross-domain consistency (shared predicates must hold across validated domains)
        ↓
Accepted artifacts enter bank → actor uses updated bank
```

**Update cadence:** 5–10 actor GRPO update cycles, then 1 offline skill-bank update cycle. Skill-bank refinement runs after the actor's update converges enough that traces are meaningful. Bank updates should not run every iteration — that creates a moving target the actor chases too aggressively.

### LoRA adapters (8B, trained)

Three LoRA adapters on Qwen3-8B, trained during co-evolution:

| Adapter | Stage | Wrapped function | Reward |
|---------|-------|------------------|--------|
| **CONTRACT** (P0) | 3 | `llm_summarize_contract()` | `verify_effects_contract().overall_pass_rate` |
| **CURATOR** (P1) | 4 | `filter_candidates()` | `bank_quality_delta` |
| **SEGMENT** (P1) | 2 | `collect_segment_preferences()` | `SegmentationDiagnostics` |

These adapters belong to the **skill-use / operational agent** (Agent 2 in the [three-agent split](../02-action-agent/PLAN-ACTION-AGENT.md#three-agent-role-split)). They handle the sequential bank-management decisions that benefit from GRPO: segmentation, contract quality, and curation. Simple retrieval, applicability scoring, pass-rate lookup, and `_SkillTracker` lifecycle logic remain algorithmic — GRPO is not applied to these.

**Additional skill-use GRPO targets** (selective, for sequential decisions only):

| Decision | Policy output | Reward |
|---|---|---|
| Continue / switch active skill | binary | Downstream reward improvement, reduced stall |
| Accept / reject candidate segment as skill instance | binary | Contract satisfaction, skill reuse rate |
| Merge / split / retire / keep skill | categorical | Bank compactness regularization, downstream actor improvement |
| Protocol revision choice from candidate set | categorical | Follow-through rate, skill pass rate delta |
| Accept / reject cross-domain transfer proposal | binary | Transfer success rate, adapter validation pass rate |

### Two-phase architecture

1. **Phase 1 (Rollout):** Pipeline calls LLM as normal → GRPO wrapper generates G samples → reward per sample → best returned to pipeline + all stored in buffer.
2. **Phase 2 (Training):** Read buffer → recompute log_probs → group-normalized rewards → GRPO policy gradient → update LoRA adapter → clear buffer.

### Synthesis-reflection outputs (from frozen 32B/72B)

The Skill Bank receives candidate artifacts from the synthesis-reflection agent ([Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)):
- New skill proposals (from Composer, Hypothesizer, Generalizer)
- Revised protocols (from Failure Reflector recovery actions)
- Contract patches (precondition strengthening, effect updates)
- Cross-domain transfer mappings (new adapters for existing abstract skills)
- "New skill vs. new adapter" decisions (see [Visual Skills §11](../01-visual-grounding/PLAN-VISUAL-SKILLS.md#11-how-the-synthesis-reflection-agent-helps-with-transfer))

All of these are treated as **candidate proposals**, not ground truth. They enter the bank only after passing the acceptance gate. The Skill Bank does not blindly trust the 32B/72B — it verifies, replays, and gates every output. See [Skill Crafter §2](../04-skill-crafter/PLAN-SKILL-CRAFTER.md#2-architecture) for the frozen teacher design rationale.

---

## 7a. Unified skill lifecycle and promotion ownership

The acceptance-gate sketch in §7 above is concretized by the **Unified Skill Gate** plan ([PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)) — that file is the canonical specification of the lifecycle, the canonical record types, and the cross-module ownership boundary. This subsection is a pointer plus the ownership commitments the Skill Bank carries.

### 7a.1 Lifecycle (the only path into the bank)

```
draft → candidate → shadow → provisional → active
draft → candidate → rejected
active → deprecated / rolled_back
```

Every skill — regardless of source (`mined / crafted / repaired / transferred / teacher_proposed / human_seeded`) — enters at `draft` and follows the same path. There is no fast path for frozen 32B/72B teacher outputs (see [PLAN-PIPELINE-ORCHESTRATOR.md §3.4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#34-asymmetric-teacher-outputs)).

### 7a.2 What the Skill Bank Agent owns

| Responsibility | Where it lives |
|----------------|----------------|
| `SkillStatus` state machine | `gate/gate_types.py` (definition); enforced by `skill_bank/skill_lifecycle_manager.py` |
| `SkillRecord` (canonical bank object) | `skill_bank/skill_record.py` |
| Versioning + lineage / provenance | `skill_bank/skill_versioning.py` (consumes [§4.3a](#43a-lineage--provenance)) |
| Candidate registration (the only entry point) | `SkillLifecycleManager.register_draft` |
| Promotion *recommendation* (per-stage `mark_*` calls) | `SkillLifecycleManager.mark_candidate / mark_shadow / mark_provisional / promote_active / reject / deprecate / rollback` |
| Final writes for each `SkillStatus` | `SkillLifecycleManager` (no other module writes any skill store) |

### 7a.3 What the Skill Bank Agent does NOT own

- **Runtime gate execution** — the Harness owns it ([PLAN-HARNESS.md §10b](../05-harness/PLAN-HARNESS.md#10b-gate-execution-runtime)). The bank calls `GateRunner` only via the Orchestrator; it does not embed replay / shadow / transfer / non-regression code itself.
- **Promotion *transactions*** — the Orchestrator owns them ([PLAN-PIPELINE-ORCHESTRATOR.md §3a](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a-promotion-transaction-and-rollback-protocol)). The bank applies the state transition under transaction control; it does not move snapshot pointers itself.

### 7a.4 Storage split (mechanical enforcement)

The bank is physically split into four stores so that the "no write to active without gate" invariant cannot be bypassed by a buggy module call:

| Store | Holds | Visible to |
|-------|-------|------------|
| `draft_store`     | `DRAFT`                                 | `SkillCrafter`, gate Stage 0 |
| `candidate_store` | `CANDIDATE`, `SHADOW`, `PROVISIONAL`    | gate Stages 1–4, `SkillHarness.run_shadow`, `run_active` (PROVISIONAL only, downweighted) |
| `active_store`    | `ACTIVE`                                | `SkillHarness.run_active` |
| `archive_store`   | `DEPRECATED`, `REJECTED`, `ROLLED_BACK` | rollback target lookup; crafter repair input |

See [PLAN-UNIFIED-SKILL-GATE.md §6](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md#6-storage-split) for the full retrieval-policy table and [§8.2](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md#82-action-agent-and-skillharnessrun_active) for what the Action Agent sees.

### 7a.5 Component reorganization (delta against §12)

- `SkillBankAgent` keeps its pipeline orchestrator role but delegates lifecycle writes to a new `SkillLifecycleManager` subcomponent.
- `TransferManager` (the existing §12 entry, which is a *proposer* of cross-domain mappings) is renamed to `LegacyTransferProposer` to avoid name collision with the runtime [PLAN-HARNESS.md §5.4 `TransferManager`](../05-harness/PLAN-HARNESS.md#54-transfermanager). The proposer emits `TransferProposal` records that flow through `register_draft`; the runtime transfer validation lives in the Harness.

---

## 8. Effect families and skill hierarchy

### Effect families

Skills organized by the kind of state change they create — the primary axis for cross-domain transfer. See [Visual Skills §4](../01-visual-grounding/PLAN-VISUAL-SKILLS.md#4-effect-families) for the full taxonomy.

| Family | State change | Game example | Browser example | Video example | Embodied example |
|--------|-------------|--------------|-----------------|---------------|------------------|
| **Acquisition** | Bring target into focus / possession | Select best move | Click target element | Lock onto tracked person | Grasp object |
| **Navigation** | Move attention or agent to region | Move to area | Scroll / navigate to section | Seek to temporal region | Move to location |
| **Inspection** | Reveal hidden information | Observe board state | Read tooltip / expand details | Sample frames for evidence | Look at / examine object |
| **Manipulation** | Change target state | Merge tiles | Fill form / toggle switch | — | Push / rotate / assemble |
| **Verification** | Test whether desired state holds | Check goal condition | Verify form submission | Confirm answer evidence | Check grasp stability |
| **Disambiguation** | Resolve multiple candidates | Choose between moves | Select from similar controls | Identify correct person | Choose correct object |
| **Tracking** | Maintain entity identity over time | Track moving objects | Maintain session state | Follow person across cuts | Track moving target |
| **Recovery** | Restore after failure / loss | Undo / reposition | Go back / retry | Re-scan for lost entity | Re-grasp / reposition |

### Three-layer skill hierarchy

The bank organizes into three layers to support both transfer and domain-specific robustness. See [Visual Skills §8](../01-visual-grounding/PLAN-VISUAL-SKILLS.md#8-three-layer-skill-bank-hierarchy) for the full design.

**Layer 1: Abstract transferable skills** — shared across all domains, defined by semantic contracts only.

```
acquire_target, inspect_region, navigate_to_goal, verify_condition,
disambiguate_candidate, track_entity, open_reveal_interact,
recover_after_loss, collect_evidence, localize_temporal_event,
constraint_satisfaction, blocker_prerequisite_replan,
history_hidden_state_act, compare_under_constraint
```

**Layer 2: Domain adapters** — per-domain execution realizations.

```
game.acquire_target, browser.acquire_target, video.acquire_target, embodied.acquire_target
```

**Layer 3: Environment-specific tactics** — concrete low-level wrappers.

```
game.acquire_target.walk_then_interact
browser.acquire_target.button_click
video.acquire_target.find_then_track
embodied.acquire_target.reach_and_grasp
```

### How the actor uses the hierarchy

The actor does not directly choose from a flat bank of environment-specific skills. The decision process is three-step:

1. **What effect is needed?** → select abstract skill (Layer 1) based on current state + desired state change
2. **Which entities fill the slots?** → bind typed slots from current grounded state
3. **Which adapter executes it here?** → select domain adapter (Layer 2), fall back to specific tactic (Layer 3)

---

## 9. Transferable skill extraction

Skills discovered in one domain can transfer to others. The extraction pipeline analyses per-domain banks and produces domain-agnostic templates.

### Skill template format (`TransferableSkill`)

Each transferable skill wraps the concrete `Skill` schema with cross-domain abstractions:

| Component | Purpose |
|-----------|---------|
| **SlotBinding** | Maps domain predicates to shared ontology slots (`target`, `blocker`, `constraint`, `candidate_set`, `history_anchor`) |
| **AbstractPredicate** | Parameterised eff_add/eff_del using `$slot` placeholders, with per-domain instantiations |
| **ReasoningProtocol** | Hop chain template using inner MDP actions (GROUND → CHECK → RETRIEVE → COMMIT → EXECUTE) |
| **DomainAdapters** | Per-domain execution realizations (tool calls, action bindings) |

### Transferable skill families

| Family | Hop chain | Game | Browser | Visual QA | Video | Embodied |
|--------|-----------|------|---------|-----------|-------|----------|
| **locate_filter_select** | GROUND → CHECK → COMMIT → EXECUTE | Candidates → best legal move | UI elements → relevant control | Objects → attributes → answer | Frames → key moment | Objects → grasp target |
| **blocker_prerequisite_replan** | GROUND → CHECK → RETRIEVE → COMMIT → EXECUTE | Deadlock → resolve prerequisite | Disabled control → fill missing field | Weak evidence → gather anchor | Missing context → scan earlier | Obstacle → clear path |
| **history_hidden_state_act** | RETRIEVE → CHECK → GROUND → COMMIT → EXECUTE | Dialogue → infer alliance → act | Prior pages → session state → next step | Prior frames → disambiguate | Earlier scenes → identify person | Prior interactions → predict affordance |
| **compare_under_constraint** | GROUND → CHECK → CHECK → COMMIT → EXECUTE | Move preserving structure | Path minimising risk | Candidate consistent with constraints | Moment consistent with timeline | Action within force/reach limits |
| **disambiguate_target** | GROUND → RETRIEVE → CHECK → COMMIT | Multiple game objects → correct one | Similar UI elements → correct control | Ambiguous objects → correct match | Multiple people → correct track | Cluttered objects → correct grasp |
| **collect_evidence_chain** | GROUND → CHECK → GROUND → CHECK → COMMIT | Multi-step board analysis | Multi-page form verification | Multi-hop visual reasoning | Multi-scene clue chaining | Multi-step task verification |

### Extraction pipeline

```
Per-domain skill banks
    ↓
Stage A: Predicate normalisation     — map domain predicates to ontology slots via regex + LLM
    ↓
Stage B: Structural clustering       — agglomerative clustering by role signatures (cross-domain)
    ↓
Stage C: Template abstraction        — produce TransferableSkill with protocol + slot bindings + adapters
    ↓
Stage D: Transferability scoring     — domain coverage × slot coverage × protocol quality × evidence
    ↓
Stage E: Export                      — transferable_skills.jsonl + transfer_index.json + families
```

**Extension to grounding skills:** The extraction pipeline also applies to visual grounding strategies — multi-step grounding patterns (disambiguation, target recovery, evidence collection) can be extracted as transferable grounding skills with belief/binding-effect contracts. See [Visual Skills](../01-visual-grounding/PLAN-VISUAL-SKILLS.md) for the full grounding skill format and how grounding segments integrate into Stages A–D.

### Usage

```python
# From the SkillBankAgent
templates = agent.extract_transferable_skills(
    other_banks={"tetris": bank_tetris, "webarena": bank_webarena, "image_qa": bank_image_qa},
    output_dir="output/transferable",
)

# Import a template into a new domain
agent.import_transferable_skill(templates[0], slot_map={
    "target": "focused_element",
    "blocker": "validation_error",
}, domain="browser")

# Or standalone
from skill_agents.extract_transferable import extract_transferable_skills
templates = extract_transferable_skills(
    banks={"2048": bank_2048, "tetris": bank_tetris, "webarena": bank_webarena},
    output_dir="output/transferable",
)
```

---

## 10. Tool-call reward (agentic RL)

For QUERY_SKILL, QUERY_MEM, CALL_SKILL the reward includes:

**r_total = w_relevance × r_relevance + w_utility × r_utility**

- **r_relevance**: RAG retrieval score × relevance_scale
- **r_utility**: fraction of eff_add predicates satisfied in outcome × utility_per_predicate, plus completion bonus

---

## 11. Cold-start I/O recording

Every LLM call that will be replaced by GRPO records its prompt/response:
- `teacher_io_coldstart.jsonl` — Stage 2 teacher calls
- `coldstart_io_all.jsonl` — all other stages

Used for: SFT cold-start for Qwen3-8B, reference outputs for GRPO reward comparison.

---

## 12. Core components

| Component | Purpose |
|-----------|---------|
| **SkillBankAgent** | Full pipeline orchestrator: ingest → segment → learn → maintain → query |
| **SkillQueryEngine** | Rich retrieval + selection: keyword, effect-based, state-aware, and cross-domain search (RAG + applicability + pass rate) |
| **SkillBankMVP** | Persistent JSONL storage for skill contracts, transfer interfaces, and verification reports |
| **NewPoolManager** | Tracking of `__NEW__` segments; clusters graduate via proto-skill staging |
| **ProtoSkillManager** | Staging area: materialize → verify → promote to real bank |
| **TransferManager** | Cross-domain transfer: extract → score → import → validate adapters |

---

## 13. Implementation

| Directory | Purpose |
|-----------|---------|
| `skill_agents/pipeline.py` | SkillBankAgent orchestrator |
| `skill_agents/query.py` | SkillQueryEngine + SkillSelectionResult |
| `skill_agents/skill_template.py` | TransferableSkill, SlotBinding, ReasoningProtocol, AbstractPredicate, DomainAdapter |
| `skill_agents/extract_transferable.py` | Cross-domain extraction pipeline (normalise → cluster → abstract → score → export) |
| `skill_agents/boundary_proposal/` | Stage 1 |
| `skill_agents/infer_segmentation/` | Stage 2 |
| `skill_agents/stage3_mvp/` | Stage 3 |
| `skill_agents/bank_maintenance/` | Stage 4 |
| `skill_agents/quality/` | Stage 4.5 |
| `skill_agents/skill_bank/` | Persistent storage + NEW pool |
| `skill_agents/grpo/` | GRPO infrastructure |
| `skill_agents/lora/` | Multi-LoRA model |
| `skill_agents/extract_skillbank/` | Extraction scripts |
| `skill_agents/tool_call_reward.py` | Agentic RL reward |
| `skill_agents/coldstart_io.py` | Cold-start I/O recording |

---

## 14. TODO

| Task | Priority | Status |
|------|----------|--------|
| Transferable skill template + extraction pipeline | P0 | **Done** |
| Unified skill gate (canonical lifecycle + ownership) — see [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) | P0 | Not started — broken into the sub-items below |
|   ↳ `SkillStatus` + `SkillSourceType` enums (`gate/gate_types.py`) | P0 | Not started |
|   ↳ `SkillRecord` (`skill_bank/skill_record.py`) | P0 | Not started |
|   ↳ `SkillEvaluationRecord` + `GateVerdictPayload` (`gate/gate_record.py`) | P0 | Not started |
|   ↳ `SkillLifecycleManager` (`skill_bank/skill_lifecycle_manager.py`) — only writer to any skill store | P0 | Not started |
|   ↳ Storage split (`draft_store / candidate_store / active_store / archive_store`) + `version_history` / `gate_history` / `rollback_links` indices | P0 | Not started |
|   ↳ `GatePolicy` + `configs/skill_gate.yaml` (centralized thresholds) | P0 | Not started |
|   ↳ Static contract check (Stage 0, `gate/static_checker.py`) | P0 | Not started |
|   ↳ Replay validation gate (Stage 1, `gate/replay_gate.py`, wraps `harness/replay_validator.py`) | P0 | Not started |
|   ↳ Non-regression gate (Stage 4, `gate/non_regression_gate.py`) | P0 | Not started |
| Implement timescale separation for bank update cadence (medium timescale) | P0 | Not started |
| Extend segmentation to inner MDP hop traces | P1 | Not started |
| Reasoning skill discovery (hop chain templates) | P1 | Not started |
| Inner hop reward signal for GRPO (hop quality + outer reward) | P1 | Not started |
| Skill-use GRPO: continue/switch, accept/reject, merge/split decisions | P1 | Not started |
| Integration with synthesis-reflection agent outputs (gated candidate ingestion) | P1 | Not started |
| Unified structured state interface (§3) across all domains | P1 | Not started |
| Cross-domain entity ontology mapping (§3.2, heuristic + LLM) | P1 | Not started |
| Belief/grounding-effect contracts (§3.3) | P1 | Not started |
| Typed slot variables in skill data model (§4.1, §4.3) | P1 | Not started |
| Domain adapter registry (abstract operator → concrete tool call) | P1 | Not started |
| Effect family taxonomy and skill hierarchy (§8) | P1 | Not started |
| Extend skill families from 4 to 6 (§9, +disambiguate_target, +collect_evidence_chain) | P1 | Not started |
| Phase detection for non-game domains (browser, video, embodied) | P1 | Not started |
| Cross-domain transfer retrieval API (§6) | P2 | Not started |
| Cross-domain transfer acceptance GRPO target (§7) | P2 | Not started |
| Reasoning protocol contracts (trigger → hops → EXECUTE) | P2 | Not started |
| RAG retrieval over reasoning templates (not just action skills) | P2 | Not started |
| LLM-based slot binding (replace regex heuristics with LLM inference) | P2 | Not started |
| Cross-domain transfer evaluation harness (transfer success rate metric) | P2 | Not started |
| TransferManager component for cross-domain skill import/export | P2 | Not started |
| Embodied task adapter (observation → schema, abstract action → motor command) | P3 | Not started |
