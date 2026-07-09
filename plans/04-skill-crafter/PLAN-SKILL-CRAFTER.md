# PLAN: Skill Crafter Agent

> **Lane decision (2026-05-01) — lane (a), Context-only skills.** A
> skill is a *retrieval payload + procedural guidance the actor LLM
> consults*, **not** a runnable program executed by the harness. See
> the canonical record:
> [`implementation_notes/legacy/skill-lane-decision.md`](../../implementation_notes/legacy/skill-lane-decision.md).
> Practical implications for this plan:
>
> * The Crafter ships with **Repairer parked behind
>   `SkillCrafterService(enable_protocol_patching=False)`** in the
>   live trainer (T1.3a). `RecoveryStrategy.{HOP_INSERTION,
>   PROTOCOL_PATCH, FALLBACK_INJECTION, REGROUNDING_TRIGGER,
>   SKILL_DECOMPOSITION}` and the `PatchProposal` mint path stay in
>   tree as **offline / lane-(b) diagnostic infrastructure**;
>   `labeling_supplement/` drivers opt them back on with
>   `enable_protocol_patching=True`.
> * The live failure taxonomy is the lane-(a) one (`BANK_GAP`,
>   `RETRIEVAL_MISLEAD`, `STALE_DESCRIPTION`); see
>   [`configs/failure_routing.yaml`](../../configs/failure_routing.yaml).
> * The Crafter's primary mode under lane (a) is the **Hypothesizer**
>   (mint a sibling retrieval payload). When a "known skill, recurring
>   failure" arrives with the Repairer parked, the dispatcher's
>   existing `_STATUS_NO_OP` fall-through routes the signal to the
>   Hypothesizer — no Crafter rewrite required.
> * **Single-MDP architecture (T3.6):** the actor is one MDP with two
>   GRPO LoRAs (`skill_selection` + `action_taking`). `hop_select` is
>   a non-target — references to a separate hop-selection LoRA below
>   are obsolete. Companion record:
>   [`implementation_notes/legacy/single-vs-two-mdp-tradeoff.md`](../../implementation_notes/legacy/single-vs-two-mdp-tradeoff.md).
>
> Sections below were authored under the lane-(b) assumption; treat
> protocol-edit machinery as lane-(b) / offline unless the section is
> tagged otherwise.

**Scope:** Compose, create, and refine new skills from existing Skill Bank primitives. The Skill Crafter is the creative layer that discovers higher-order strategies by combining existing skills, generalizing across games/domains, and proposing novel skill hypotheses that the [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) can test and adopt.

**Scope boundaries (deliberate).** Every proposal the Crafter emits must be a **general protocol feasible across all five target domains** — game / webagent / os-agent / video-understanding / visual reasoning — written over the shared schema and shared inner primitives (see [Skill Bank §0.1](../03-skill-bank/PLAN-SKILL-BANK.md#01-general-protocol-invariant-no-domain-specific-skill-families)). A proposal that only works on one domain is rejected by the acceptance gate; it does not become a "short-video-only skill" or a "browser-only skill". The Crafter's **first evaluation arena** is short-video (Video-Holmes-style) — that is where `verified_domains` entries are filled in first and where transfer-failure diagnostics ([PLAN-HARNESS.md §10a](../05-harness/PLAN-HARNESS.md)) are exercised first, **not** where a separate class of skills is synthesized. The Crafter-private `FailurePatternStore` (§6.7) is an offline pattern-aggregation index over `FailureDiagnosis` records; it is never read by the online actor and never extends the agent's episode-local trajectory.

**Upstream:** Existing skill bank (contracts, protocols, execution traces); structured schemas from [Visual Grounding](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md); episode trajectories from [Action Agent](../02-action-agent/PLAN-ACTION-AGENT.md).
**Downstream:** New/refined skills injected into the Skill Bank; cross-domain skill transfer proposals.

---

## 1. Motivation

The Skill Bank (Stage 1–4) discovers skills *bottom-up* from observed trajectories — it segments what the agent actually did. But some valuable skills are never observed because:

1. **Composition gap:** The agent never chains two skills in the right order (e.g., "build corner" in 2048 = `POSITION` → `MERGE` → `POSITION`).
2. **Transfer gap:** A skill discovered in one game (e.g., "clear bottom row" in Tetris) has a structural analogue in another game (e.g., "clear bottom boxes" in Sokoban) but the surface-level observations differ.
3. **Hypothesis gap:** Some strategies are known from game theory or human play but never appear in the agent's rollouts because the agent hasn't explored that part of the strategy space.

The Skill Crafter addresses these gaps by operating *top-down*: it proposes new skills, the Skill Bank tests them, and the Action Agent tries them.

---

## 2. Architecture

### Model tier assignment

The Skill Crafter runs entirely on **Tier 1 (Qwen3-32B/72B, inference-only, frozen)** — see [Action Agent §2](../02-action-agent/PLAN-ACTION-AGENT.md#2-tiered-model-architecture) for the full tiered architecture rationale and the [three-agent role split](../02-action-agent/PLAN-ACTION-AGENT.md#three-agent-role-split). The Skill Crafter IS the **synthesis-reflection agent** (Agent 3). All three creation modes (Composer, Generalizer, Hypothesizer) and the Failure Reflector require multi-step counterfactual reasoning, cross-domain analogy, and structured diagnosis that exceed the 8B reasoning ceiling. Because these components run offline (between episodes, not per-step), the larger model adds no latency to the Action Agent's decision loop.

**Frozen-first design:** The 32B/72B backbone is kept frozen initially. Its outputs (candidate skills, revised protocols, recovery patches, diagnoses) are treated as **proposals**, not ground truth. Every output must pass multi-pass verification and held-out replay checks before entering the skill bank or training buffer. This avoids feedback-loop drift where the teacher becomes overconfident in the system's own biases. See [Action Agent §6](../02-action-agent/PLAN-ACTION-AGENT.md#6-co-evolution--grpo-decomposition) for the full acceptance gate specification.

**Multi-run reasoning requirement:** Even at 32B/72B scale, single-pass inference is insufficient for the Skill Crafter's tasks. The main bottleneck is not "lack of optimization" but "reasoning is hard and noisy." Each creation or reflection task requires multiple reasoning passes:

- **Composer:** Pass 1 — propose candidate compositions; Pass 2 — verify effect chain validity per pair; Pass 3 — generate protocol + test expectations.
- **Generalizer:** Pass 1 — identify shared structural slots; Pass 2 — propose mapping candidates; Pass 3 — instantiate and sanity-check with target-domain examples.
- **Hypothesizer:** Best-of-N sampling (N=3–5) — generate N proposals, score by contract completeness + novelty, keep top-K.
- **Failure Reflector:** Pass 1 — identify symptom step; Pass 2 — re-evaluate each prior hop with targeted prompts; Pass 3 — confirm root cause via counterfactual; Pass 4 — evaluate alternative outcomes at the root cause step, compute regret, store `CounterfactualTrace` (§6.9).

This multi-run design costs ~3–6× the tokens of a single 32B/72B call per task (the Failure Reflector's 4-pass chain is the most expensive at ~4–6×), but remains negligible compared to GRPO rollout costs since these tasks run once per episode batch, not per step. The additional Pass 4 adds ~500–1000 tokens per failure reflection — a modest increase that produces high-value counterfactual evidence for skill synthesis.

### Frozen teacher improvement channels

The synthesis-reflection agent (32B/72B) improves over time WITHOUT weight updates through five channels:

1. **Better input distribution** — as the 8B actor and skill bank improve through GRPO, the Skill Crafter receives cleaner trajectories, more reusable segments, better failure logs, and richer skill statistics. The same frozen model produces much better outputs when reasoning over better evidence.
2. **Better evidence serialization & transfer diagnostics** — Crafter context improves as the artifact stores that feed it get better at their jobs: (a) *evidence serialization* (how within-episode `evidence_refs`, tool traces, and claim–evidence links are laid out for the teacher), (b) *transfer diagnostics* (typed labels from the Harness, see [PLAN-HARNESS.md §10a](../05-harness/PLAN-HARNESS.md)), (c) *replay slices* (frozen trace selection pipelines), (d) *skill clustering* (how failure clusters and transfer-candidate groups are indexed), and (e) *verification prompts* (templates for Stage-4 acceptance). The Crafter-private `FailurePatternStore` (§6.7) is one of these artifact stores; it is offline-only and never read by the online actor.
3. **Better inference procedure** — improve the multi-pass reasoning without touching weights: better decomposition, proposal-then-verify chains, best-of-N with smarter selection, counterfactual replay, stricter acceptance filters. This is where most early "improvement" should come from.
4. **Better verification and selection** — as the actor and bank mature, downstream usefulness can be measured more reliably. A frozen model that emits many candidate skills becomes much more useful when the system can better select which candidates to keep.
5. **Distillation into smaller specialized modules** — train smaller adapters or submodules on outputs that pass verification: a small failure-localizer, a contract writer, a protocol patch ranker, a transfer-mapping scorer. This builds a synthesis pipeline that improves over time without changing the main teacher.

**Principle:** The thing that improves is the synthesis *system*, not necessarily the frozen model weights. Think of improvement as: frozen LLM + improving (data, evidence serialization, transfer diagnostics, replay slices, skill clustering, verification prompts, acceptance rules).

### Phased teacher adaptation policy

The 32B/72B teacher should NOT be fine-tuned from day one. Follow this phased approach:

**Phase 1: Frozen teacher (default starting point)**
- Keep 32B/72B frozen.
- Improve prompts, decompositions, and multi-pass workflows.
- Add strict verification before any teacher output enters the skill bank or GRPO data.
- This phase may be sufficient for the entire project if verification quality is high enough.

**Phase 2: Light adaptation only if needed**
- Fine-tune only if repeated, narrow failures are observed:
  - Poor formatting of contracts/protocols despite good evidence
  - Weak domain vocabulary mapping
  - Systematic failure in the schema language
  - Low agreement with replay-based verification
- If fine-tuning: start with small supervised or preference-style adaptation on narrow tasks:
  - Failure localization
  - Contract writing
  - Protocol synthesis
  - Schema-slot mapping
  - Candidate ranking / judging
- Do NOT do broad end-to-end RL on the teacher first.

**Phase 3: GRPO on the teacher (last resort)**
- Only after Phase 2 still shows a clear bottleneck.
- Risk: teaching the teacher to optimize toward shallow proxies — writing plausible but wrong diagnoses, inventing too many "new skills", overfitting to the current bank schema, producing flashy but wrong recovery patches. Those errors then contaminate the bank.
- If done: train only on structured subproblems with narrow rewards, not the full synthesis pipeline end-to-end.

**Rule of thumb:** Train the actor, not the judge, until there is evidence the judge is the limiting factor. The bigger gains will likely come from better verifier design and replay checks, not from teacher fine-tuning.

### Pipeline overview

```
                     Skill Bank (existing skills)
                            ↓
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
    Skill Composer    Skill Generalizer   Skill Hypothesizer
              ↓             ↓             ↓
              └─────────────┼─────────────┘
                            ↓
                    Proto-Skill Proposals
                            ↓
                    Skill Bank (Stage 4: verify → promote)
                            ↓
                    Action Agent (trial execution)
                            ↓ (on failure)
                    ┌───────────────┐
                    │  Failure       │
                    │  Reflector     │
                    │  (§6)         │
                    └───────┬───────┘
                            ↓
              ┌─────────────┼──────────────┐
              ↓             ↓              ↓
         Localize       Diagnose       Recover
         (where?)       (why?)         (how to fix?)
              ↓             ↓              ↓
              └─────────────┼──────────────┘
                            ↓
              ┌─────────────┼─────────────┐
              ↓                           ↓
        Skill Bank                  Skill Crafter
        (patch existing)            (compose/hypothesize new)
              ↓                           ↓
              └─────────┬─────────────────┘
                        ↓
                  Retry with improved skills
```

### Three creation modes

| Mode | Input | Output | Method |
|------|-------|--------|--------|
| **Composer** | 2+ existing skills | Compound skill (chain/branch/loop) | Sequence planning + effect chaining |
| **Generalizer** | 1 skill + different domain | Transferred skill with adapted contract | Structural analogy via shared schema slots |
| **Hypothesizer** | Game description + failure patterns | Novel skill proposal | LLM reasoning over rules + failure analysis |

---

## 2.5 Typed proposal outputs (evidence-driven, domain-general)

Every Crafter output is one of the four typed proposals below. All four carry the **evidence-driven fields** required by [Skill Bank §0.3](../03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills); proposals missing them are gate-rejected before they reach replay. Evidence fields are **declarations of intent** on the proposal — the Harness confirms them empirically at Gate G0 after shadow / replay runs.

```python
class EvidenceInterfaceDecl:
    evidence_inputs_spec:   list     # typed EvidenceRef kinds the proposed skill expects to read
    evidence_outputs_or_warrant_spec: dict
                                     # role-dependent spec (see Skill Bank §4.2):
                                     #   GATHER  -> evidence_out kinds
                                     #   VERIFY  -> verdict domain + referenced evidence kinds
                                     #   REASON  -> hypothesis schema + warrant shape
                                     #   COMMIT  -> decision schema + evidence_warrant shape

class _BaseProposal:
    proposal_id: str
    proposer:    str                 # "composer" | "generalizer" | "hypothesizer" | "reflector"
    evidence_role: str               # GATHER | VERIFY | REASON | COMMIT  (Skill Bank §0.3 Clause B)
    evidence_interface: EvidenceInterfaceDecl
    target_domains: list             # MUST include all 5: game, webagent, os-agent, video, visual_reasoning
    adapter_plan:   dict             # per-domain adapter strategy (even if stub)
    replay_slice_ids: list           # replay traces the harness should use for G0+G3
    rationale: str                   # short English justification (teacher output)

class PatchProposal(_BaseProposal):
    target_skill_id: str
    patch_kind: str                  # "precondition" | "protocol" | "contract" | "warrant-strengthen"
    patch_body:  dict

class ComposeProposal(_BaseProposal):
    components: list                 # ordered list of sub-skill_ids (evidence-driven, already in the bank)
    compose_op: str                  # "sequence" | "branch" | "loop" | "while-insufficient"
    # If the composed macro would have no evidence_in/out of its own, this MUST be realized
    # as a ComposeProposal over evidence-driven sub-skills, NEVER as a standalone PLAN-family
    # skill (which does not exist under §0.3 Clause B).

class TransferProposal(_BaseProposal):
    source_skill_id: str
    new_adapter:     dict            # target-domain adapter spec
    evidence_interface_remap: dict   # how source-domain EvidenceRef kinds map to target-domain kinds
                                     # (if this is empty and source/target evidence kinds differ,
                                     #  the harness will reject with `evidence_interface_mismatch`)

class RetireProposal(_BaseProposal):
    target_skill_id: str
    retire_reason: str               # "opaque" | "evidence-starved" | "subsumed" | "unsafe"
                                     # | "regressing" | "superseded"
    evidence_stats: dict             # recent episodes' evidence_in/out coverage, to justify retirement
```

**Gate-time rejection rules (applied before the proposal reaches Stage-4 staging):**

- `evidence_role` missing or outside `{GATHER, VERIFY, REASON, COMMIT}` → reject.
- `evidence_interface.evidence_inputs_spec` and `evidence_outputs_or_warrant_spec` both empty → reject (opaque-skill violation, §0.3 Clause A).
- `target_domains` not covering all five → reject (general-protocol invariant, §0.1).
- `TransferProposal` with differing source/target evidence kinds and empty `evidence_interface_remap` → reject (anticipated `evidence_interface_mismatch`).
- `ComposeProposal` whose components are not all themselves evidence-driven (any `component.evidence_role` missing) → reject.

These rules make "ALL skills are evidence-driven" a property the Crafter cannot violate by construction: any proposal that would create an opaque skill is blocked before it ever gets replayed.

---

## 3. Skill Composer

### What it does

Takes two or more existing skills and proposes a compound skill that chains them together with explicit transition conditions.

### Composition operators

| Operator | Semantics | Example |
|----------|-----------|---------|
| `sequence(A, B)` | Execute A until success, then B | `sequence(POSITION_corner, MERGE_large)` |
| `if_then_else(cond, A, B)` | Check condition, branch | `if_then_else(stack_high, CLEAR_row, BUILD_tetris)` |
| `repeat_until(A, cond)` | Loop skill A | `repeat_until(MERGE_small, highest_tile >= 512)` |
| `fallback(A, B)` | Try A, on abort try B | `fallback(NAVIGATE_direct, NAVIGATE_detour)` |

### Effect chaining

A composition is valid when the postconditions of skill A satisfy the preconditions of skill B:

```
Skill A: eff_add = {corner_built, high_tile_positioned}
Skill B: preconditions = {corner_built}
         eff_add = {large_merge_completed}

Composed: sequence(A, B)
  preconditions = A.preconditions
  eff_add = A.eff_add ∪ B.eff_add
  eff_del = A.eff_del ∪ B.eff_del
```

### Discovery algorithm

1. For each pair of skills (A, B) in the bank:
   - Check: do A's eff_add predicates overlap with B's preconditions?
   - If yes: propose `sequence(A, B)` as a candidate compound skill.
2. For skills with high abort rates:
   - Find skills whose eff_add addresses the abort condition.
   - Propose `fallback(original, recovery)`.
3. For skills that are often repeated:
   - Check if repetition correlates with progress.
   - Propose `repeat_until(skill, progress_threshold)`.
4. **Comparative evaluation** (counterfactual ranking)**:**
   - Before submitting to Stage 4, compare alternative compositions against each other.
   - For each skill A where multiple valid continuations exist (e.g., `sequence(A, B)`, `sequence(A, C)`, `sequence(A, D)` all pass effect chaining), run a composition-level counterfactual:
     - Draw representative state snapshots from the Failure Pattern Store (§6.7) where skill A was recently active.
     - For each candidate composition, predict cumulative effect coverage: how many of the composed skill's `eff_add` predicates would be satisfied starting from that state?
     - Rank compositions by predicted coverage × historical pass rate of the component skills.
   - Submit only the top-K compositions (default K=3 per skill A) rather than all valid ones. This reduces Stage 4 verification load while prioritizing the most promising combinations.
   - Compositions that address high-regret patterns from `CounterfactualTrace` records (§6.9) get a ranking bonus, since they target known decision-point weaknesses.
5. Submit top-K proposals to Skill Bank Stage 4 (proto-skill staging → verify → promote).

---

## 4. Skill Generalizer

### What it does

The Generalizer is the **few-shot transfer engine** of the Crafter ([PLAN-UNIFIED-SKILL-GATE.md Stage 3a](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md#7-stage-3a--few-shot-transfer-validation), [PLAN-HARNESS.md §5.4.2](../05-harness/PLAN-HARNESS.md#542-fewshotadapter-stage-3a-runtime)). It takes a skill that was **mined from the source domain (game)** and proposes a *recipe* for adapting it to a specific **transfer-target domain** (`browser` / `osworld` / `video` / `visual_reasoning`) using a small number (`K = few_shot.k_shot_default`) of target-domain demonstrations.

The recipe is what eventually becomes a `GeneralizeProposal` carrying:

- `source_domain` (must be in `SOURCE_DOMAINS`, currently `{"gymv"}`),
- `target_domain` (must be in `TRANSFER_TARGET_DOMAINS`),
- `slot_remap` (slot-level rewrite of the source-domain protocol into target-domain slot names),
- `demo_episode_ids` + `demo_selection` (which K demos the FewShotAdapter should consume),
- `k_shot_budget` (≤ `few_shot.k_shot_max`).

The Generalizer is **not** the verifier — it does not decide whether the skill works in the target domain. That decision is made offline by `GateService._run_transfer` invoking the `FewShotAdapter`. The Generalizer's job ends when a well-formed recipe is emitted; the recipe earns a `verified_domains` entry only after Stage 3a passes.

When the Generalizer is invoked **without** explicit source/target metadata, it falls back to a legacy in-bank generalization (slot-level rewrite without a target-domain probe), produces a `TRANSFERRED` proposal, and that proposal does **not** earn a `verified_domains` entry until Stage 3a runs separately.

### Transfer via shared schema slots

> **See also:** [Visual Skills](../01-visual-grounding/PLAN-VISUAL-SKILLS.md) extends this transfer mechanism with a cross-domain entity ontology (§5) and a three-layer skill bank hierarchy (abstract skill → domain adapter → environment-specific tactic) that applies to both grounding and reasoning skills.

The canonical schema (from Visual Grounding) uses shared slot names across domains:

| Slot | Game example | Browser example | Video example |
|------|-------------|-----------------|---------------|
| `target` | tile to merge | button to click | person to track |
| `blocker` | wall preventing push | modal blocking click | occlusion blocking view |
| `constraint` | cannot pull boxes | form validation | temporal ordering |
| `candidate_set` | movable tiles | clickable elements | visible people |
| `progress` | score / boxes solved | form fields completed | clues found |

**Key insight:** Because the schema format is the same across domains, a skill defined in terms of schema slots can transfer to any domain that populates those slots.

### Transfer algorithm

1. For a source skill with contract `{eff_add, eff_del, preconditions}`:
   - Express predicates in terms of schema slots (e.g., `target.value >= 512` → `target.value >= threshold`).
   - Parameterize domain-specific constants.
2. In the target domain:
   - Find schema instances where the parameterized predicates can be instantiated.
   - Propose the transferred skill with adapted contract.
3. Validate:
   - Run the transferred skill in the target domain.
   - Verify eff_add predicates are achievable.
   - If pass rate > threshold → promote to bank.

### Cross-domain examples

| Source (game) | Target (browser) | Shared structure |
|--------------|-------------------|-----------------|
| 2048: `MERGE` (combine adjacent tiles) | Form: `FILL_ADJACENT` (fill adjacent fields) | `adjacent(target, e_next)` + `target.value → combined` |
| Sokoban: `PUSH_TO_GOAL` (push box to target) | Shopping: `ADD_TO_CART` (navigate item to cart) | `navigate(target) → apply_action(target)` |
| Tetris: `CLEAR_ROW` (complete and clear) | Todo: `COMPLETE_SECTION` (fill all fields in section) | `fill(candidate_set) → clear(candidate_set)` |

### Cross-domain examples (video benchmarks)

| Source (SIV-Bench) | Target (Video-Holmes) | Shared structure |
|--------------------|----------------------|-----------------|
| `TRACK_INTERACTION` (follow social exchange) | `TRACK_CLUE` (follow clue across scenes) | `track_object(query) → temporal chain of evidence` |
| `DETECT_EMOTION_CHANGE` (find expression shift) | `DETECT_PLOT_TWIST` (find narrative shift) | `detect_scene_changes() → compare_elements(before, after)` |

---

## 5. Skill Hypothesizer

### What it does

Proposes entirely new skills that don't exist in the bank, based on:
- Game rules / task descriptions
- Common failure patterns (high-abort skills, low-reward trajectories)
- Human strategy knowledge (via LLM)

### Hypothesis generation

1. **Failure analysis** (informed by Failure Reflector §6)**:**
   - Query the Failure Pattern Store (§6.7) for skills with high abort rates, recurring failure patterns, or retired skills.
   - Use `FailureDiagnosis` records — specifically the `violated_assumption` and `suggested_fix` fields — as structured input rather than raw failure counts.
   - Ask LLM: "Given these diagnosed failures and their root causes, what strategy would avoid them?"
   - Propose the LLM's suggestion as a proto-skill, linking back to the failure patterns it addresses.

2. **Rule-based reasoning:**
   - Parse game description (`env.description`) for strategic implications.
   - Ask LLM: "What skills would be useful for a game with these rules?"
   - Cross-reference against existing bank skills.
   - Propose skills that cover gaps.

3. **Archetype matching:**
   - Maintain a library of game-theoretic archetypes (minimax, resource management, spatial planning, social deduction, etc.).
   - Match game characteristics to archetypes.
   - Instantiate archetype-specific skill templates.

4. **Counterfactual synthesis** (informed by §6.9)**:**
   - Query the Failure Pattern Store (§6.7) for `CounterfactualTrace` entries with high regret (`regret_estimate > threshold`).
   - Cluster by `best_alternative` patterns: if the same type of alternative repeatedly appears as the "better choice" across different episodes, that recurring pattern is a candidate skill.
   - Ask LLM: "These N episodes all would have benefited from {alternative_pattern}. Generalize this into a reusable skill with a contract and protocol."
   - Produces proto-skills with higher initial confidence (`0.5` vs. `0.3` for other sources) because they are grounded in specific counterfactual evidence rather than abstract failure patterns.
   - **Synthesis trigger:** Automatic when a `(skill, decision_level, best_alternative)` tuple accumulates 3+ occurrences in the Failure Pattern Store's `high_regret_patterns` (see §6.7).
   - **Advantage over failure analysis (source 1):** Failure analysis asks "what strategy would avoid this failure?" — a speculative question. Counterfactual synthesis asks "this specific alternative was predicted to work N times — generalize it" — a grounded question. The resulting proto-skills have tighter contracts because the counterfactual traces provide concrete precondition/postcondition examples.

### Hypothesis format

```python
ProtoSkill(
    name="corner_defense",
    strategic_description="Maintain highest tile in corner while building merge chains",
    proposed_by="hypothesizer:failure_analysis",
    source_evidence=["skill:MERGE has 40% abort rate when corner is lost"],
    proposed_contract=SkillEffectsContract(
        eff_add={"corner_maintained", "merge_chain_intact"},
        eff_del={"corner_lost"},
    ),
    proposed_protocol=Protocol(
        steps=["Position highest tile in corner", "Build descending chain along edge", "Merge from opposite end"],
        preconditions=["highest_tile >= 128"],
        success_criteria=["corner_maintained after 5+ merges"],
        abort_criteria=["highest_tile displaced from corner"],
    ),
    confidence=0.3,  # low — needs verification
)
```

Submitted to Skill Bank Stage 4 for proto-skill staging → verify → promote/reject.

---

## 6. Failure Reflection & Reasoning Recovery

### Motivation

When the agent executes a multi-hop reasoning chain and it fails — wrong answer, timeout, degraded reward, aborted skill — the failure is rarely random. It originates at a specific step in the chain, for a specific reason. Without the ability to **locate**, **diagnose**, and **learn from** these failures, the Skill Crafter proposes blind fixes. This section defines a structured failure reflection loop that closes that gap.

### 6.1 Failure Trace Capture

Every reasoning chain (hop sequence) produces an execution trace. On failure, the trace is captured as a `FailureTrace`:

```python
FailureTrace(
    episode_id="ep_0042",
    skill_name="MERGE_chain",
    hop_sequence=[
        HopRecord(step=0, action="GROUND", input="locate highest tile", output="tile_256 at (0,0)", status="ok"),
        HopRecord(step=1, action="CHECK",  input="adjacent mergeable?", output="tile_128 at (0,1)", status="ok"),
        HopRecord(step=2, action="EXECUTE", input="merge (0,0)+(0,1)", output="blocked by tile_64", status="FAIL"),
        HopRecord(step=3, action="COMMIT", input="—", output="—", status="skipped"),
    ],
    failure_step=2,
    failure_type="precondition_violated",
    context_snapshot={...},  # full state at failure point
    expected_outcome="tile_384 at (0,0)",
    actual_outcome="no merge; board unchanged",
)
```

Key fields:
- **`failure_step`** — the exact index in the hop sequence where things went wrong.
- **`failure_type`** — classified category (see §6.2).
- **`context_snapshot`** — the full `<state>` schema at the moment of failure, so the reflector can reason over what the agent *actually saw*.

### 6.2 Failure Classification Taxonomy

Not all failures are the same. The reflector classifies each failure into a category that determines the recovery strategy:

| Category | Description | Example | Recovery bias |
|----------|-------------|---------|---------------|
| **grounding_error** | Entity was misidentified or not found | VLM returned wrong tile position | Re-ground with refined query |
| **precondition_violated** | Step assumed a condition that wasn't true | Tried to merge but path was blocked | Backtrack and re-check preconditions |
| **stale_context** | Relied on outdated state information | Board changed between GROUND and EXECUTE | Re-observe before acting |
| **logical_error** | Reasoning step drew wrong conclusion from correct inputs | Chose wrong merge direction despite seeing the board correctly | Revise reasoning rule / protocol |
| **missing_information** | Needed data the agent didn't have | Couldn't infer hidden tile from history | Add RETRIEVE hop or request more context |
| **evidence_starved** | Skill executed with empty `evidence_in ∪ evidence_out`, or `evidence_role`-required fields unset, across ≥ N recent episodes — skill is no longer assisting reasoning (Gate G0 violation, [PLAN-HARNESS.md §10](../05-harness/PLAN-HARNESS.md#10-promotion-gates)) | A `REASON` skill that stopped citing any `evidence_in` after a slot-binding drift; a `COMMIT` skill with empty `evidence_warrant` | Emit a `PatchProposal{patch_kind: "warrant-strengthen"}` requiring citation of specific evidence kinds, or a `RetireProposal{retire_reason: "evidence-starved"}` if the pattern persists |
| **cascading_failure** | Earlier soft error amplified into hard failure at later step | Slightly wrong grounding → wrong CHECK → wrong EXECUTE | Trace back to root cause step |
| **resource_exhaustion** | Ran out of hops / time / tokens before completion | Complex chain exceeded hop budget | Simplify protocol or increase budget |

### 6.3 Failure Localization (Where)

The reflector walks backward through the hop sequence to find the **root cause step** — which may differ from the step that visibly failed.

**Algorithm: Backward Trace Analysis**

```
Input: FailureTrace T
Output: root_cause_step, root_cause_type

1. Start at T.failure_step (the step that raised the error).
2. For each prior step i = failure_step-1 ... 0:
   a. Re-evaluate step i's output against the context_snapshot at step i.
   b. If step i's output was already degraded (wrong entity, stale state, 
      incorrect inference):
      - Mark i as candidate root cause.
      - Classify the degradation type.
   c. If step i's output was correct given its inputs → stop.
      The root cause is the most recent candidate (or failure_step itself 
      if no prior degradation found).
3. Return (root_cause_step, root_cause_type).
```

This distinguishes between:
- **Direct failures** — the failing step itself is the problem (e.g., wrong EXECUTE action).
- **Cascading failures** — an earlier step silently produced bad output that propagated forward.

### 6.4 Failure Diagnosis (Why)

Once the root cause step is located, the reflector generates a structured diagnosis by prompting the LLM with the failure context:

```
Prompt template (failure_diagnosis):
───────────────────────────────────
You are analyzing a reasoning failure in a multi-hop chain.

**Skill:** {skill_name}
**Task:** {task_description}
**Full hop trace:** {hop_sequence}
**Root cause step:** Step {root_cause_step} — action: {action}, input: {input}
**Expected output:** {expected}
**Actual output:** {actual}
**State at failure:** {context_snapshot}

1. Why did step {root_cause_step} produce the wrong output?
2. What assumption was violated?
3. Was the input to this step correct? If not, trace the error further.
4. What specific change to the skill's protocol would prevent this failure?
───────────────────────────────────
```

The diagnosis output is a structured `FailureDiagnosis`:

```python
FailureDiagnosis(
    root_cause_step=2,
    root_cause_type="precondition_violated",
    explanation="Step 2 attempted merge without checking that the path between tiles was clear. "
                "The GROUND step correctly found the tiles, but the CHECK step only verified "
                "value compatibility, not spatial accessibility.",
    violated_assumption="Adjacent tiles with matching values can always merge",
    suggested_fix="Add a CHECK_PATH hop between GROUND and EXECUTE that verifies no blocking "
                  "tiles exist on the merge path.",
    confidence=0.75,
)
```

**Pass 4 — Counterfactual evaluation** (feeds §6.9):

After diagnosis, a fourth reasoning pass evaluates what the agent *should* have done instead. This pass takes the root cause step and diagnosis as input and produces a `CounterfactualTrace` (see §6.9 for data structures).

```
Prompt template (counterfactual_evaluation):
───────────────────────────────────
You are evaluating alternative actions for a reasoning failure.

**Skill:** {skill_name}
**Root cause step:** Step {root_cause_step} — chose: {chosen_action}
**Diagnosis:** {explanation}
**Violated assumption:** {violated_assumption}
**State at decision point:** {context_snapshot}
**Available alternatives:** {alternative_actions}

For each alternative action:
1. If step {root_cause_step} had done {alternative} instead of {chosen_action},
   what would the subsequent steps have produced?
2. Would this alternative have avoided the diagnosed failure? Why or why not?
3. What is the estimated reward improvement over the actual outcome? (float, can be negative)
4. Confidence that this alternative would have succeeded? (0.0–1.0)

Rank alternatives by expected improvement. Identify the single best alternative.
───────────────────────────────────
```

The multi-run reasoning budget for the Failure Reflector thus becomes: Pass 1 (symptom) → Pass 2 (re-evaluate hops) → Pass 3 (confirm root cause) → **Pass 4 (counterfactual alternatives)**. See §2 for updated token budget.

### 6.5 Recovery Strategies (How to Improve)

Based on the diagnosis, the reflector proposes one or more recovery actions. These aren't just retries — they are structured improvements that feed back into the Skill Crafter's three creation modes:

| Recovery strategy | When to apply | What it produces | Feeds into |
|-------------------|---------------|------------------|------------|
| **Protocol patch** | Logical error or missing check | Revised protocol with added/modified hop | Skill Bank (update existing skill) |
| **Precondition strengthening** | Precondition violated | Tighter precondition predicates on the skill contract | Skill Bank (contract update) |
| **Fallback injection** | Grounding error or stale context | `fallback(original_step, recovery_step)` composition | Composer (§3) |
| **Hop insertion** | Missing information | New RETRIEVE or CHECK hop inserted into protocol | Skill Bank (protocol update) |
| **Skill decomposition** | Cascading failure across many steps | Break monolithic skill into smaller, independently verifiable sub-skills | Composer (§3) |
| **Re-grounding trigger** | Stale context | Add re-observe checkpoints at key protocol boundaries | Skill Bank (protocol update) |
| **Skill retirement** | Persistent failure despite multiple fixes | Demote skill; let Hypothesizer (§5) propose replacement | Hypothesizer (§5) |

### 6.6 Reflection Loop Integration

The failure reflection loop runs as a post-episode process and connects to the rest of the Skill Crafter pipeline:

```
Episode execution (Action Agent)
        ↓ (on failure)
  FailureTrace capture
        ↓
  Failure Localization (§6.3)  →  root_cause_step
        ↓
  Failure Diagnosis (§6.4)     →  FailureDiagnosis
        ↓
  Recovery Proposal (§6.5)     →  RecoveryAction[]
        ↓
  ┌─────┴──────────────────┐
  ↓                        ↓
Skill Bank               Skill Crafter
(patch existing)         (compose/hypothesize new)
  ↓                        ↓
  └────────┬───────────────┘
           ↓
  Action Agent (retry with improved skills)
           ↓
  Evaluate: did the fix work?
           ↓
  ┌────────┴────────┐
  ↓                 ↓
 Yes: promote      No: escalate
 fix + update       (deeper reflection
 confidence)        or retire skill)
```

### 6.7 Failure Pattern Store & Pattern Aggregation

Individual failure diagnoses are stored in a Crafter-private, offline-only **Failure Pattern Store** that enables pattern-level learning. It is an aggregation index over `FailureDiagnosis` records, never an episode-spanning lookup channel for the online actor:

- **Recurrence detection:** If the same `(skill, failure_type, root_cause_step)` tuple appears N+ times, escalate from "patch" to "redesign."
- **Cross-skill patterns:** If multiple skills fail with the same `failure_type` (e.g., many skills have `stale_context` errors), propose a systemic fix (e.g., add re-grounding checkpoints to all long-chain protocols).
- **Failure clustering:** Group failures by shared violated assumptions. Each cluster may point to a missing primitive skill or a flawed schema mapping.
- **Improvement tracking:** For each recovery action applied, track whether it reduced the failure rate. Recovery actions with low success rates are themselves subject to reflection.

```python
FailurePatternStore(
    entries=[FailureDiagnosis, ...],
    
    # Aggregated patterns
    recurrence_counts={("MERGE_chain", "precondition_violated", 2): 7},
    cross_skill_patterns={"stale_context": ["MERGE_chain", "POSITION_corner", "NAVIGATE_path"]},
    recovery_effectiveness={
        "add_CHECK_PATH_hop": {"applied": 5, "resolved": 4, "rate": 0.80},
        "strengthen_precondition": {"applied": 3, "resolved": 1, "rate": 0.33},
    },

    # Counterfactual aggregation (§6.9)
    counterfactual_traces=[CounterfactualTrace, ...],
    high_regret_patterns={
        # key: (skill, decision_level, best_alternative_type)
        # value: occurrence count, mean regret, whether synthesis was triggered
        ("MERGE_chain", "action", "CHECK_PATH"): {
            "count": 5, "mean_regret": 0.18, "skill_synthesized": False,
        },
        ("POSITION_corner", "skill", "DEFEND_corner"): {
            "count": 3, "mean_regret": 0.22, "skill_synthesized": True,
        },
    },
    regret_driven_skills=["corner_defense", ...],  # skills born from counterfactual synthesis
)
```

**Counterfactual-specific aggregation logic:**

- **Regret accumulation:** Each `CounterfactualTrace` is indexed by `(skill_name, decision_level, best_alternative)`. When the same tuple recurs, increment its count and update the running mean regret.
- **Synthesis trigger:** When a high-regret pattern reaches 3+ occurrences (configurable), auto-dispatch to the Hypothesizer's counterfactual synthesis mode (§5, source 4). This is analogous to the escalation policy (§6.8) but driven by regret rather than failure recurrence.
- **Effectiveness tracking:** After a regret-driven skill is synthesized and deployed, continue collecting `CounterfactualTrace` records at similar decision points. If regret at those points decreases, the synthesized skill is validated. If regret persists or increases, escalate to redesign.
- **Cross-referencing with recovery effectiveness:** When a recovery action (§6.5) was informed by a `CounterfactualTrace.best_alternative`, track that linkage. This lets the system measure whether counterfactual-informed recoveries outperform non-counterfactual ones.

### 6.8 Escalation Policy

Not every failure warrants the same level of response. The escalation policy determines how much effort to invest:

| Occurrence count | Response level | Action |
|------------------|---------------|--------|
| 1st occurrence | **Log** | Store diagnosis, no immediate action |
| 2nd occurrence (same pattern) | **Patch** | Apply lightest recovery strategy |
| 3rd–5th occurrence | **Redesign** | Decompose or rewrite the skill protocol |
| 6+ occurrences | **Retire & replace** | Demote skill, ask Hypothesizer for alternative |
| Cross-skill pattern (3+ skills) | **Systemic fix** | Propose architectural change (new primitive, schema update) |

### 6.9 Counterfactual Reasoning & Alternative Outcome Synthesis

The Failure Reflector (§6.3–6.4) diagnoses *what went wrong on the path the agent took*. But it never asks the complementary question: **"What would have happened if the agent had chosen differently?"** Counterfactual reasoning closes this gap. Instead of only producing "this failed because X", the system produces "this failed because X, **and choosing Y instead would have succeeded because Z**" — a much richer signal for skill synthesis, decision boundary sharpening, and regret-driven learning.

#### Three levels of counterfactual analysis

Counterfactual reasoning applies at every level of the two-level MDP:

| Level | Scope | Question | Cost |
|-------|-------|----------|------|
| **Action-level** (inner MDP) | Single hop within a skill | "What if hop 2 had done CHECK_PATH instead of CHECK_VALUE?" | Low — one extra LLM pass using existing `context_snapshot` |
| **Skill-level** (outer MDP) | Skill selection decision point | "What if we'd selected POSITION_corner instead of MERGE_chain at this state?" | Medium — requires state schema at selection moment + candidate skill set |
| **Composition-level** (Composer) | Proposed skill compositions | "What if we'd composed sequence(A, B) instead of sequence(A, C)?" | High — simulates alternative effect chains against representative states |

**Action-level counterfactuals** operate on the `HopRecord` sequence within a single skill execution. The Failure Reflector already captures the `context_snapshot` at each step — the counterfactual pass feeds this snapshot to the LLM along with each alternative inner action (from the inner action vocabulary: GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE) and asks for the predicted downstream outcome. This is the cheapest level and runs as Pass 4 of failure diagnosis (see §6.4).

**Skill-level counterfactuals** operate on the skill selection decision points logged by `_SkillTracker`. They require the state schema at the selection moment plus the set of candidate skills that were scored but not chosen — both already available from `select_skill_from_bank()` scoring (see [Action Agent §3](../02-action-agent/PLAN-ACTION-AGENT.md#3-skill-guided-decision-making)). For each runner-up skill, the LLM predicts whether its protocol would have avoided the failure, given the state at that moment.

**Composition-level counterfactuals** operate during the Composer's batch jobs (§3). When the Composer proposes `sequence(A, B)`, it also evaluates `sequence(A, C)`, `sequence(A, D)`, etc. against representative state snapshots drawn from the Failure Pattern Store. This ranks alternative compositions by predicted cumulative effect coverage before submitting them to Stage 4, reducing verification load.

#### Data structures

```python
@dataclass
class AlternativeOutcome:
    option: str                  # the unchosen action/skill/composition
    predicted_outcome: str       # LLM-predicted result of taking this alternative
    predicted_reward_delta: float # estimated reward difference vs. what actually happened
    confidence: float            # LLM self-assessed confidence (0–1)
    reasoning: str               # chain-of-thought justification

@dataclass
class CounterfactualTrace:
    episode_id: str
    decision_point_step: int
    decision_level: str          # "action" | "skill" | "composition"
    chosen_option: str
    chosen_outcome: str
    alternative_options: list[AlternativeOutcome]
    best_alternative: str        # the alternative with highest predicted_reward_delta
    regret_estimate: float       # max(predicted_reward_delta) across alternatives
```

**Example:**

```python
CounterfactualTrace(
    episode_id="ep_0042",
    decision_point_step=2,
    decision_level="action",
    chosen_option="CHECK_VALUE(tile_128)",
    chosen_outcome="FAIL: precondition_violated",
    alternative_options=[
        AlternativeOutcome(
            option="CHECK_PATH(tile_256, tile_128)",
            predicted_outcome="path_blocked detected → fallback triggered",
            predicted_reward_delta=+0.15,
            confidence=0.7,
            reasoning="Path check would have detected tile_64 blocker before merge attempt",
        ),
        AlternativeOutcome(
            option="GROUND(adjacent_tiles)",
            predicted_outcome="re-grounding reveals updated board state",
            predicted_reward_delta=+0.05,
            confidence=0.4,
            reasoning="Stale context possible but not confirmed",
        ),
    ],
    best_alternative="CHECK_PATH(tile_256, tile_128)",
    regret_estimate=0.15,
)
```

**Key field: `regret_estimate`** — the predicted reward delta between what happened and what the best alternative would have produced. High-regret decision points are the richest source material for new skill synthesis (see §5, counterfactual synthesis).

#### Counterfactual generation algorithm

```
Input: FailureDiagnosis D, FailureTrace T, available_actions (or candidate_skills)
Output: CounterfactualTrace

1. At the root_cause_step identified by D:
   a. Enumerate alternative actions/skills available at that state.
      - Action-level: inner action vocabulary minus the chosen action.
      - Skill-level: top-K scored candidate skills from select_skill_from_bank().
   b. Filter to top 2–3 alternatives by surface plausibility
      (LLM pre-screen: "Given this state, which of these alternatives 
       could plausibly address the diagnosed root cause?").
2. For each surviving alternative:
   a. Prompt LLM with context_snapshot + alternative action + remaining protocol steps.
   b. Ask: "If step {root_cause_step} had done {alternative} instead of {chosen},
      what would steps {root_cause_step+1}...{N} have produced?"
   c. Extract predicted_outcome, predicted_reward_delta, confidence.
3. Select best_alternative = argmax(predicted_reward_delta) among alternatives
   with confidence ≥ 0.4.
4. Compute regret_estimate = best_alternative.predicted_reward_delta.
5. Return CounterfactualTrace.
```

#### Integration with the reflection loop

Counterfactual analysis slots into the existing reflection loop (§6.6) as a post-diagnosis step:

```
Episode execution (Action Agent)
        ↓ (on failure)
  FailureTrace capture
        ↓
  Failure Localization (§6.3)  →  root_cause_step
        ↓
  Failure Diagnosis (§6.4)     →  FailureDiagnosis
        ↓
  Counterfactual Analysis (§6.9) → CounterfactualTrace
        ↓
  Recovery Proposal (§6.5)     →  RecoveryAction[]
        ↓                          (now informed by best_alternative)
  ┌─────┴──────────────────┐
  ↓                        ↓
Skill Bank               Skill Crafter
(patch existing)         (compose/hypothesize new)
```

Recovery proposals (§6.5) now have access to the `CounterfactualTrace`: the `best_alternative` directly informs what recovery action to propose. A protocol patch can insert the counterfactually-validated alternative step rather than guessing at a fix.

#### What NOT to do

- **No environment rollback / replay.** Counterfactuals are LLM-predicted ("what would likely happen"), not simulated. If an environment supports cheap forking, that is a future optimization, not a requirement.
- **No exhaustive enumeration.** Only the top 2–3 alternatives are evaluated per decision point. The pre-screen filter keeps cost bounded.
- **No counterfactuals on success.** Only failed episodes trigger counterfactual analysis. Successful episodes already demonstrate a working path.

---

## 7. Transferable skill families (long-horizon reasoning)

> **Note:** Failure reflection (§6) applies to reasoning chains within these skill families — when a locate→filter→select chain fails at the "filter" step, the reflector localizes that step, diagnoses why the filter criteria were wrong, and proposes a protocol patch or fallback.

Under the two-level MDP (see [Action Agent §5](../02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control)), the Skill Crafter composes and transfers *reasoning policies* — not single-call chain-of-thought templates, but actual multi-step policies that can be trained, composed, and transferred across domains.

### Cross-domain skill families

| Family | Game | Web | Visual Reasoning |
|--------|------|-----|------------------|
| **Locate → filter → select** | Candidate moves → best legal | UI candidates → relevant control | Objects → attributes → answer target |
| **Blocker → prerequisite → replan** | Deadlock → missing setup | Disabled control → missing field | Weak evidence → gather anchor |
| **History → hidden state → act** | Dialogue → alliance/threat | Prior pages → next step | Prior frames → disambiguate |
| **Compare under future constraint** | Move preserving structure | Path lowering risk/steps | Candidate consistent with constraints |

Each family is a reusable multi-step reasoning policy whose protocol maps to inner MDP actions (GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE).

### Composition under the inner MDP

Skill composition (§2) gains a new dimension: composing *reasoning hops* rather than just environment actions.

- **Sequence composition** now chains hop protocols: Skill A's COMMIT feeds Skill B's GROUND trigger.
- **Fallback composition** tries alternative reasoning strategies: if GROUND fails to locate the entity, fall back to RETRIEVE from the skill bank.
- **Nested composition**: an inner RETRIEVE hop can invoke a sub-skill's entire reasoning protocol.

### Transfer via shared reasoning vocabulary

Because all domains share the same inner action vocabulary (GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE) and the same `<state>` schema structure, transfer becomes a matter of **schema-slot mapping** rather than domain-specific engineering:

1. **Source domain** skill: `Locate → filter → select` over entities {piece, obstacle, board_position} in a game.
2. **Target domain** mapping: {piece → form_field, obstacle → validation_error, board_position → form_section} in browser.
3. **Reasoning protocol** is unchanged: GROUND(target) → CHECK(constraints) → COMMIT(best) → EXECUTE(action).

---

## 8. Integration with Visual Grounding & Failure Reflection

The Skill Crafter leverages multi-hop visual reasoning from the vlm_wrapper:

### For Composition

- Use `spatial_query` to verify that composed skills' preconditions/postconditions are visually grounded.
- Use `detect_objects` to confirm entity existence before proposing entity-dependent compositions.

### For Generalization

- Use the shared schema as the transfer medium — both source and target domains produce `<state>` schemas with the same entity/relation/target structure.
- Visual grounding ensures transferred skills map to real visual entities in the target domain.

### For Hypothesis generation

- Use `classify_scene` to characterize the game state and match to archetypes.
- Use tool traces from failed episodes to identify where the agent's visual understanding broke down.

### For Failure Reflection

- Use `detect_objects` and `spatial_query` during backward trace analysis (§6.3) to re-evaluate whether GROUND steps produced correct visual grounding at the failure point.
- Compare the VLM's current perception against the `context_snapshot` stored in the `FailureTrace` to detect grounding drift.
- Feed visual grounding errors into the failure classification taxonomy (§6.2) as `grounding_error` — the most common failure type in visual reasoning domains.

---

## 9. Evaluation

### Metrics for crafted skills

| Metric | How measured | Threshold |
|--------|-------------|-----------|
| Contract pass rate | Stage 3 verification on trial episodes | ≥ 0.7 for promotion |
| Improvement over base | Reward delta when using crafted skill vs. existing | > 0 |
| Transfer success rate | % of cross-domain proposals that pass verification | Track |
| Composition utility | % of composed skills selected by action agent | Track |
| Novelty | Jaccard distance from nearest existing skill | ≥ 0.25 |

### Metrics for failure reflection

| Metric | How measured | Threshold |
|--------|-------------|-----------|
| Localization accuracy | % of root cause steps confirmed correct by manual review | ≥ 0.7 |
| Recovery success rate | % of applied recovery actions that resolved the failure pattern | ≥ 0.5 |
| Mean diagnoses to fix | Average number of reflection cycles before a failure pattern is resolved | ≤ 3 |
| Failure recurrence rate | % of fixed failures that reappear within N episodes | ≤ 0.15 |
| Systemic fix coverage | % of cross-skill patterns addressed by systemic fixes | Track |

### Metrics for counterfactual reasoning

| Metric | How measured | Threshold |
|--------|-------------|-----------|
| Counterfactual prediction accuracy | % of predicted "better" alternatives that actually improve reward when tried | ≥ 0.5 |
| Regret-driven skill utility | % of regret-synthesized skills that pass Stage 4 verification | ≥ 0.6 |
| Regret reduction rate | Mean regret at similar decision points before vs. after synthesized skill deployment | Track (expect downward trend) |
| Counterfactual-informed recovery lift | Recovery success rate for counterfactual-informed fixes vs. non-counterfactual fixes | > 0 delta |

### Evaluation protocol

1. Crafter proposes N proto-skills per iteration.
2. Each proto-skill enters Stage 4 staging → verification.
3. Promoted skills are tried by the action agent for K episodes.
4. Skills that improve reward are kept; others are demoted or removed.

---

## 10. Rollout order

> **Co-evolution alignment:** The Skill Crafter operates on the **slow timescale** within the three-agent co-evolution framework (see [Action Agent §6](../02-action-agent/PLAN-ACTION-AGENT.md#6-co-evolution--grpo-decomposition)). Crafter proposals run after failed episodes, periodically for effect chaining, when the Failure Pattern Store accumulates, when adding new domains, and before GRPO for cold-start trajectory generation. They are gated by the acceptance pipeline before entering the skill bank or training buffer.

**Phase 0 — Frozen teacher bootstrap**
1. Deploy 32B/72B frozen, inference-only.
2. Validate multi-pass prompting and verification pipeline.
3. Generate initial candidate skills from seed trajectories (from Tier 0 or 8B actor early rollouts).
4. All outputs go through acceptance gate before entering bank.

**Phase 1 — Skill Composer (within-game)**
1. Implement effect chaining algorithm.
2. Propose sequence/fallback compositions for existing game skills.
3. Verify and promote via Stage 4.
4. Measure: do composed skills improve episode reward?

**Phase 2 — Skill Hypothesizer (within-game)**
1. Implement failure analysis pipeline.
2. Propose novel skills based on failure patterns + game rules.
3. Verify and promote via Stage 4.
4. Measure: do hypothesized skills cover previously unaddressed situations?

**Phase 3 — Skill Generalizer (cross-domain)**
1. Implement schema-slot-based transfer.
2. Transfer skills from games → browser (or vice versa).
3. Verify transferred skills in target domain.
4. Measure: does transfer reduce cold-start time in new domains?

**Phase 4 — Failure Reflection Loop**
1. Implement `FailureTrace` capture and `FailurePatternStore` storage.
2. Implement backward trace analysis for failure localization.
3. Implement LLM-based failure diagnosis with structured output.
4. Implement recovery strategy selection and application.
5. Implement escalation policy and pattern aggregation.
6. Measure: do reflected fixes reduce failure recurrence?

**Phase 4b — Counterfactual Reasoning & Regret-Driven Synthesis**
1. Implement `CounterfactualTrace` generation (Pass 4 of failure diagnosis, §6.4/§6.9).
2. Implement `AlternativeOutcome` prediction pipeline (LLM prompting + structured output).
3. Integrate counterfactual traces into the `FailurePatternStore` — regret accumulation and `high_regret_patterns` indexing.
4. Implement regret-driven synthesis trigger in Hypothesizer (§5, source 4).
5. Implement comparative composition ranking in Composer (§3, step 4).
6. Measure: counterfactual prediction accuracy, regret-driven skill utility, regret reduction rate.

**Phase 5 — Cross-modality transfer (image ↔ video)**
1. Transfer visual reasoning skills from image benchmarks to video.
2. Transfer temporal reasoning patterns from video benchmarks to interactive environments.

**Phase 6 — Teacher adaptation (only if needed)**
1. Identify if frozen teacher is the bottleneck (check: narrow repeated failures, poor contract formatting, low verification agreement).
2. If yes: light SFT/preference tuning on narrow tasks (failure localization, contract writing, protocol synthesis, ranking).
3. Measure: does teacher adaptation improve acceptance rate and downstream actor performance?
4. See §2 Phased teacher adaptation policy.

---

## 11. TODO

| Task | Priority | Status |
|------|----------|--------|
| Effect chaining algorithm for skill composition | P0 | Not started |
| Composition operators (sequence, fallback, repeat_until) | P0 | Not started |
| Acceptance gate pipeline (contract check, replay verification, non-regression filter) | P0 | Not started |
| Multi-pass verification workflow for 32B/72B outputs | P0 | Not started |
| Failure analysis pipeline for hypothesis generation | P1 | Not started |
| Schema-slot transfer algorithm | P1 | Not started |
| FailureTrace capture & HopRecord serialization | P0 | Not started |
| Backward trace analysis (failure localization) | P0 | Not started |
| Failure classification taxonomy implementation | P1 | Not started |
| LLM-based failure diagnosis with structured output | P1 | Not started |
| Recovery strategy selector & applicator | P1 | Not started |
| FailurePatternStore & pattern aggregation | P1 | Not started |
| Frozen teacher inference procedure optimization (prompt engineering, best-of-N, decomposition) | P1 | Not started |
| Escalation policy engine | P2 | Not started |
| Archetype library for hypothesis matching | P2 | Not started |
| Cross-domain transfer evaluation harness | P2 | Not started |
| Integration with vlm_wrapper tool traces | P2 | Not started |
| Hop-chain composition operators (inner MDP) | P1 | Not started |
| Cross-domain reasoning protocol transfer | P2 | Not started |
| Failure reflection metrics & dashboarding | P2 | Not started |
| CounterfactualTrace generation (Pass 4 of failure diagnosis) | P1 | Not started |
| AlternativeOutcome prediction pipeline & LLM prompting | P1 | Not started |
| Regret accumulation & high_regret_patterns indexing in FailurePatternStore | P1 | Not started |
| Regret-driven synthesis trigger in Hypothesizer | P1 | Not started |
| Comparative composition ranking in Composer | P2 | Not started |
| Counterfactual prediction accuracy evaluation harness | P2 | Not started |
| Distillation of accepted outputs into smaller specialized modules (Phase 2 teacher adaptation) | P2 | Not started |
| Narrow SFT/preference tuning for teacher if frozen approach bottlenecks (Phase 3) | P2 | Not started |

---

## 12. Implementation (planned)

| File (planned) | Purpose |
|----------------|---------|
| `skill_agents/crafter/composer.py` | Skill composition via effect chaining |
| `skill_agents/crafter/generalizer.py` | Cross-domain skill transfer |
| `skill_agents/crafter/hypothesizer.py` | Novel skill proposal from failures + rules |
| `skill_agents/crafter/archetypes.py` | Game-theoretic archetype library |
| `skill_agents/crafter/evaluation.py` | Crafted-skill quality metrics |
| `skill_agents/crafter/failure_trace.py` | FailureTrace / HopRecord data structures and capture |
| `skill_agents/crafter/failure_reflector.py` | Backward trace analysis, failure localization & classification |
| `skill_agents/crafter/failure_diagnosis.py` | LLM-based diagnosis prompt construction & structured output |
| `skill_agents/crafter/recovery.py` | Recovery strategy selection, application, and feedback |
| `skill_agents/crafter/failure_patterns.py` | Persistent FailurePatternStore: aggregation index over FailureDiagnosis records, escalation |
| `skill_agents/crafter/counterfactual.py` | CounterfactualTrace / AlternativeOutcome generation, regret estimation, alternative enumeration |

Note: counterfactual reasoning also integrates into existing planned files — `failure_diagnosis.py` (Pass 4 prompt), `failure_patterns.py` (regret aggregation), `hypothesizer.py` (counterfactual synthesis mode), `composer.py` (comparative composition ranking).

Currently no implementation exists — this is the newest component of the pipeline.
