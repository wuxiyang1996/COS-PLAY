# PLAN: Edit Plan — Transferable Reasoning Skills (IR + Inner-Hop Discovery + Crafter Engine)

> **Document type.** This is a *plan of edits* across the existing plan files, in the same spirit as [PLAN-EDITS-HARNESS-CONTROL-PLANE.md](PLAN-EDITS-HARNESS-CONTROL-PLANE.md). It does **not** introduce a new module, agent, or trainable head. Every concrete addition lands in an already-existing plan file via the dispatch table in §6 and the per-phase "Where this lands" subsections.
>
> **Reading order.** Start with §1 (thesis + main line), then §2 / §3 / §4 (Phases A / B / C), then §5 (the integrated closed loop), then §6 (file-by-file dispatch table), then §7 (priorities). Skip directly to §6 if you are looking for the smallest "what do I touch in which file" map.

---

## 0. Scope and non-goals

**Scope.** Move the system from "we have a Skill Bank that *describes* cross-task transfer" to "we have a pipeline that *actually produces* transferable reasoning skills." This requires three coordinated additions, all of which are already directionally present in the repo but not yet wired into a runnable end-to-end loop:

1. **Phase A — Transferable IR layer.** A typed-slot, ontology-mapped, adapter-bound *intermediate representation* for skills, so the same skill object can be exercised across game / webagent / os-agent / video-understanding / visual reasoning without rewriting prompt text. This sharpens what [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) §3, §4, and [PLAN-VISUAL-SKILLS.md](../01-visual-grounding/PLAN-VISUAL-SKILLS.md) §0.2 already gesture at into a single hard contract.
2. **Phase B — Inner-hop reasoning skill discovery.** A discovery pipeline that mines *reasoning protocols* from the inner reasoning hops of the two-level MDP defined in [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) §5, instead of only segmenting outer-action trajectories. Today the inner-hop trace is logged but not mined; this phase makes it the primary substrate for new reasoning skills.
3. **Phase C — Crafter as a verifiable synthesis / repair engine.** Convert the [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) from a "responsibilities-and-prompts" specification into a concrete engine with explicit composition operators, backward failure localization, a discrete repair taxonomy, typed proposal outputs, and mandatory routing through the unified gate stack ([PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)).

**Non-goals.**

- No new agent. The three phases are absorbed into Agents 1–3 already defined in [README §three-agent role split](../README.md#three-agent-role-split--model-convention).
- No new trainable head. All Phase A / B / C work is either schema / contract work (Phase A), offline mining (Phase B), or frozen-teacher pipeline (Phase C). Existing GRPO-trained heads (`hop_select`, `skill_select`, `segment`, `contract`) are unchanged.
- No new gate. Crafter outputs go through the existing unified gate stack: `static → replay → shadow → transfer → non-regression`. This plan adds inputs to that stack, not new stack stages.
- No new state-keeping surface. Inner-hop traces are part of the orchestrator's episode-local trajectory ([PLAN-PIPELINE-ORCHESTRATOR.md §4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)). Pattern aggregation across episodes is offline only and lives in the Crafter-private `FailurePatternStore` already described in [PLAN-SKILL-CRAFTER.md §6.7](../04-skill-crafter/PLAN-SKILL-CRAFTER.md).

**Compatibility.** All edits in this plan respect the two existing invariants enforced at the Harness gate:

- **General-protocol invariant** — every IR object, every mined protocol, every Crafter proposal must be feasible across all five target domains ([PLAN-SKILL-BANK.md §0.1](../03-skill-bank/PLAN-SKILL-BANK.md#01-general-protocol-invariant-no-domain-specific-skill-families)).
- **Evidence-driven invariant** — every skill, including Phase B mined protocols and Phase C composed / repaired candidates, declares one `evidence_role ∈ {GATHER, VERIFY, REASON, COMMIT}` and a non-empty `evidence_interface` ([PLAN-SKILL-BANK.md §0.3](../03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills)).

---

## 1. Thesis and main line

> **Thesis.** Cross-task reasoning skill transfer is real only if (a) skills live in a typed intermediate representation that is independent of any single domain's surface vocabulary, (b) the discovery pipeline mines reasoning procedures from inner reasoning hops as well as outer actions, and (c) the Skill Crafter is a verifiable synthesis / repair engine rather than a textual reflection module. The three are useless in isolation and complete in combination.

**Main line (one closed loop).**

```
Actor inner-hop traces (Phase B input)
    │
    ▼
Trace segmentation + hop-utility attribution (Phase B)
    │
    ▼
Reasoning protocols lifted into typed-slot IR over canonical ontology (Phase A)
    │
    ▼
Crafter operates on typed IR: compose / repair / cross-domain transfer (Phase C)
    │
    ▼
Unified gate stack: static → replay → shadow → transfer → non-regression
    │
    ▼
Promoted into Skill Bank → consumed by Actor → new traces feed back
```

The three phases are *sequenced* but must be *integrated*. Building Phase B without Phase A produces clusters that cannot be lifted into reusable skills. Building Phase C without Phase A produces patches that only work in one domain. Building Phase A without Phase B starves the Crafter of input. The planned execution order is A → B → C only because each later phase consumes the previous one's outputs; in practice the three are co-developed, with §7 fixing the P0 / P1 / P2 split.

---

## 2. Phase A — Transferable Skill IR (typed slots + canonical ontology + adapters)

### 2.1 Goal

Define a single, typed `SkillIR` object that is the canonical in-memory and on-disk representation of every skill in the bank. A skill's reusability across domains becomes a property of its slot signature and ontology bindings, not of the natural-language wording in its protocol.

### 2.2 Canonical cross-domain ontology

A skill cannot be transferable unless the entities, events, and evidence it refers to live in a domain-independent vocabulary. The Skill Bank already names this requirement ([PLAN-SKILL-BANK.md §3](../03-skill-bank/PLAN-SKILL-BANK.md), [PLAN-VISUAL-SKILLS.md §0.2](../01-visual-grounding/PLAN-VISUAL-SKILLS.md)); Phase A pins it as a hard contract.

**Top-level canonical ontology types.**

| Category | Canonical types |
|----------|-----------------|
| Entities | `Agent`, `Object`, `Region`, `UIElement`, `Person`, `Group`, `Container`, `Document`, `TrackableEntity` |
| Events | `ActionEvent`, `InteractionEvent`, `TemporalEvent`, `StateTransition`, `UIEvent`, `NavigationEvent` |
| Evidence | `EvidenceSpan`, `FrameSpan`, `RegionPatch`, `TrajectorySegment`, `DialogueSnippet`, `UIObservation` |
| Relations / attributes | `Attribute`, `Relation`, `IntentHypothesis`, `IdentityHypothesis`, `Constraint`, `GoalState` |

The set is closed at the canonical layer. Domain-specific subtypes are introduced only inside per-domain adapters and never appear in `SkillIR` slot type signatures.

### 2.3 Domain ontology adapters

Each of the five domains supplies a static mapping table from native object types to the canonical ontology. Adapters are pure data; they live next to the existing domain adapters in the harness's `AdapterRegistry` (see [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md)).

| Domain | Native → canonical mapping (illustrative; full table per domain) |
|--------|------------------------------------------------------------------|
| game | `object → Object`; `npc / player → Agent`; `grid / room / tile → Region`; `pickup / attack / move → ActionEvent` |
| webagent | `button / link / textbox → UIElement`; `modal / panel / form → Container`; `click / submit / type → UIEvent` |
| os-agent | `window / app → UIElement`; `file → Document`; `dialog → Container`; `switch_window / open_file → NavigationEvent / ActionEvent` |
| video | `character / suspect / pedestrian → Person / Agent`; `interaction / handoff / chase → InteractionEvent`; `frame range / clip span → FrameSpan / EvidenceSpan` |
| visual reasoning | `image object → Object`; `image region → RegionPatch`; `relation / comparison target → Relation / Attribute` |

Adapter compatibility (does adapter X cover all canonical types referenced by skill S?) is checked by an `adapter_compatibility_checker` in the Harness (see §6).

### 2.4 Typed Slot Skill IR

Every skill in the bank, every Crafter proposal, and every Phase B mined protocol is represented as the following object. Field names are chosen to be additive over what already exists in [PLAN-SKILL-BANK.md §4.1 / §4.2](../03-skill-bank/PLAN-SKILL-BANK.md); existing fields are preserved.

```text
SkillIR = {
  "skill_id":            str,
  "name":                str,
  "version":             int,
  "skill_family":        str,                    # inspect | compare | verify | resolve_identity | ...
  "evidence_role":       "GATHER" | "VERIFY" | "REASON" | "COMMIT",
  "applicable_domains":  list[str],              # subset of {game, webagent, os-agent, video, visual_reasoning}

  "input_slots": {                               # typed slot signature (canonical types only)
      "<slot_name>": "<CanonicalType>",
      ...
  },
  "output_slots": {
      "<slot_name>": "<CanonicalType>",
      ...
  },

  "preconditions":      list[Predicate],
  "procedure":          list[HopStep],           # see §2.5; typed protocol, not domain prose
  "success_criteria":   list[Predicate],
  "abort_criteria":     list[Predicate],

  "evidence_interface": {                        # required by §0.3 Clause A/B
      "evidence_in":      list[EvidenceContract],
      "evidence_out":     list[EvidenceContract],
      "evidence_warrant": list[EvidenceContract] | None,
      "verdict":          {"PASS","FAIL","INSUFFICIENT"} | None,
  },

  "effects": {                                   # cf. PLAN-VISUAL-SKILLS.md §2
      "world_effects":     list[Effect],
      "belief_effects":    list[Effect],
      "grounding_effects": list[Effect],
  },

  "adapter_requirements": list[str],             # canonical types this skill references; checked against adapters
  "lineage":               LineageRecord,        # cf. PLAN-SKILL-BANK.md §4.3a
  "negative_knowledge":    NegativeKnowledge,    # cf. PLAN-SKILL-BANK.md §4.3b
  "verified_domains":      list[str],            # filled in by the Harness gate, not by the author
}
```

`procedure` is **not** free-form text. It is a list of typed operators (§2.5) over the typed slots; this is what makes the IR transferable.

### 2.5 Skill procedures as typed operators

Skills must be written as sequences of typed operators rather than as domain-specific action descriptions. The first-version operator vocabulary is:

| Operator | Signature | Meaning |
|----------|-----------|---------|
| `inspect` | `inspect(target_entity: Entity) → EvidenceSpan` | Produce evidence about a specific entity |
| `compare` | `compare(a: Entity, b: Entity, attribute: Attribute) → Relation` | Produce a typed comparison result |
| `verify` | `verify(claim_or_event, evidence_span) → {PASS, FAIL, INSUFFICIENT}` | Check a typed predicate against evidence |
| `resolve_identity` | `resolve_identity(entity: Entity, cue_set: list[EvidenceSpan]) → IdentityHypothesis` | Bind an entity reference to a concrete identity |
| `track_state_change` | `track_state_change(entity: Entity, span: EvidenceSpan) → StateTransition` | Detect and record a typed state transition |
| `commit` | `commit(hypothesis, warrant: list[EvidenceRef]) → COMMIT` | Emit a decision / answer with explicit warrant |
| `ground` | `ground(entity_hint, evidence_source) → EvidenceSpan` | Bind a textual / partial reference to a concrete grounded entity |

These operators bind directly to the inner-MDP primitives in [PLAN-ACTION-AGENT.md §5](../02-action-agent/PLAN-ACTION-AGENT.md): `inspect / ground` map under `GROUND` and `RETRIEVE`; `verify` under `CHECK`; `compare / resolve_identity / track_state_change` under `CONCLUDE / REASON`; `commit` under `COMMIT / EXECUTE`. The mapping is enforced by the Harness contract at §0.3 Clause B.

### 2.6 Minimal first-cut reasoning skill families

The first-version Skill Bank should converge on a *small* set of cross-domain reasoning skill families, not 50. README and Skill Bank already mention "6 transferable skill families" / "4 cross-domain families"; Phase A pins the first version to the following six. Each is one skill family in `SkillIR.skill_family`:

1. `inspect(target_entity)`
2. `compare(entity_a, entity_b, attribute)`
3. `verify(claim_or_event, evidence_span)`
4. `resolve_identity(entity, cue_set)`
5. `track_state_change(entity, event_span)`
6. `commit_subgoal_or_belief(hypothesis, warrant)`

These six are sufficient to cover the bulk of reusable reasoning primitives across game / webagent / os-agent / video / visual reasoning. Phase B is allowed to mine additional families only if it can demonstrate (a) non-redundancy with these six and (b) feasibility across at least three domains.

### 2.7 Where this lands

| Plan file | Section to add |
|-----------|----------------|
| [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) | §3a *Canonical Cross-Domain Ontology*; §4.1a *Typed Slot Skill IR*; §4.3c *Domain Adapter Contract*; §9.0 *Transferable Skill Families (minimal set)* |
| [PLAN-VISUAL-SKILLS.md](../01-visual-grounding/PLAN-VISUAL-SKILLS.md) | §2a *World vs Belief vs Grounding Effect — unified definition over canonical ontology* |
| [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) | §10b *Slot binding validator*; §10c *Ontology remap validator*; §10d *Adapter compatibility checker* (all run inside the existing G0 / G2 / G3 gates, no new gate) |

---

## 3. Phase B — Inner-hop reasoning skill discovery

### 3.1 Goal

Move skill discovery from "look at outer action trajectories" to "look at the inner reasoning hops of the two-level MDP, segment them, attribute hop-level utility, and lift recurring fragments into typed reasoning protocols." The Action Agent's two-level MDP ([PLAN-ACTION-AGENT.md §5](../02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control)) already produces the substrate; Phase B adds the mining pipeline.

### 3.2 New artifact — `HopTrace`

Inner reasoning chains must be persisted as first-class artifacts, not folded into outer-action logs. `HopTrace` is logged per outer step by the Action Agent and consumed by the offline mining pipeline.

```text
HopTrace = {
  "episode_id":     str,
  "outer_step_idx": int,
  "hop_seq": [
      {
        "hop_idx":       int,
        "hop_type":      "GROUND" | "CHECK" | "RETRIEVE" | "CONCLUDE" | "COMMIT" | "EXECUTE",
        "input_slots":   dict,                 # typed slots bound at hop entry
        "output_slots":  dict,                 # typed slots produced
        "evidence_in":   list[EvidenceRef],
        "evidence_out":  list[EvidenceRef],
        "confidence":    float,
        "success_flag":  bool,
        "latency_cost":  float,
        "skill_invoked": str | None,           # SkillIR.skill_id, if any
      },
      ...
  ],
  "final_outcome": dict,
  "reward_delta":  float,
  "failure_type":  str | None,                  # see §3.7 taxonomy
}
```

`HopTrace` is part of the orchestrator's episode-local trajectory ([PLAN-PIPELINE-ORCHESTRATOR.md §4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping)); cross-episode aggregation is offline-only and never read by the online actor.

### 3.3 Hop segmentation (do this *before* clustering)

Naively clustering raw hop sequences is the wrong starting move. Phase B first segments each `HopTrace` into candidate fragments using typed segmentation signals:

- hop-type change,
- target-entity change,
- evidence-source change,
- hypothesis change,
- sharp confidence drop or rise,
- success / failure boundary,
- outer-step `COMMIT` boundary.

Each segment is labeled with a coarse fragment family: `grounding_fragment`, `verification_fragment`, `comparison_fragment`, `identity_resolution_fragment`, `commit_fragment`. These labels are *segmentation-time* and are not yet skills.

### 3.4 Hop-level utility attribution

The outer reward is too coarse to drive reasoning-skill discovery. Each hop is scored against five utility components:

1. `evidence_gain` — change in `evidence_out` cardinality / coverage,
2. `ambiguity_reduction` — change in candidate-set size or hypothesis entropy,
3. `binding_success` — whether the hop's typed slot bindings remained consistent downstream,
4. `verification_success` — whether `CHECK` hops returned `PASS / FAIL` (not `INSUFFICIENT`),
5. `commit_reliability` — whether downstream `COMMIT` succeeded conditional on this hop's outputs.

Each fragment carries a 5-dimensional utility vector. This is the input to clustering in §3.5 and to Crafter failure localization in §4.3.

### 3.5 Reasoning trace clustering (cluster the *signature*, not the text)

Clustering operates over typed signatures, not raw token strings. The clustering vector for each fragment combines:

- hop-type sequence,
- typed slot signature (slot names → canonical types),
- ontology type pattern (e.g., `Person → IdentityHypothesis → InteractionEvent`),
- evidence-flow pattern (which evidence types are read / written),
- utility-vector profile,
- success / failure profile.

This is what makes cross-domain protocol discovery possible. Two fragments from very different domains —

- *video:* `GROUND(Person) → CHECK(IdentityHypothesis) → VERIFY(InteractionEvent)`
- *webagent:* `GROUND(UIElement) → CHECK(GoalState) → VERIFY(UIEvent)`

— land in the same cluster because their typed signatures coincide, even though their surface text does not. The cluster is then lifted into a single higher-order reasoning skill family such as `inspect_and_verify(target_entity, condition, evidence_span)`.

### 3.6 Lift clusters into reusable `ReasoningProtocol` objects

Cluster centroids are *not* dumped back into the bank as raw traces. They are abstracted into `ReasoningProtocol` objects that fit directly into the Phase A `SkillIR.procedure` field:

```text
ReasoningProtocol = {
  "protocol_id":            str,
  "slot_signature":         dict,           # canonical-typed slots
  "canonical_hop_sequence": list[HopStep],  # typed operators (Phase A §2.5)
  "allowed_variants":       list[HopStep],  # permitted hop substitutions
  "preconditions":          list[Predicate],
  "success_criteria":       list[Predicate],
  "abort_criteria":         list[Predicate],
  "evidence_contract":      EvidenceInterface,
  "source_domains":         list[str],
  "utility_profile":        dict,           # aggregated 5-dim vector across cluster members
}
```

Every `ReasoningProtocol` produced by Phase B is treated as a *candidate* and routed to the Crafter (Phase C) and the unified gate stack — never written directly to the active bank.

### 3.7 Failed hop pattern mining

Phase B must mine **failed** trace patterns with the same discipline as successful ones, because these are the primary input to Crafter repair (§4.4). Failure type taxonomy:

- `grounding_failure`,
- `entity_binding_failure`,
- `verification_insufficient`,
- `wrong_hop_order`,
- `premature_commit`,
- `missing_abort`,
- `over_reasoning`,
- `under_reasoning`.

Failed-pattern clusters are written to the Crafter-private `FailurePatternStore` ([PLAN-SKILL-CRAFTER.md §6.7](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)); they are never written to the bank and never read by the online actor.

### 3.8 Where this lands

| Plan file | Section to add |
|-----------|----------------|
| [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) | §5.4 *HopTrace logging contract*; §5.5 *Reasoning-step trace export*; §5.6 *Bounded inner-hop trace format* |
| [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) | §5a *Inner-hop skill discovery pipeline* (sub-sections: segmentation, hop-utility attribution, signature clustering, protocol lifting, failed-pattern mining) |
| [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) | §6.7a *Crafter consumes failed-hop pattern clusters* (input contract from §3.7) |
| [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) | §4a *HopTrace as part of the episode-local trajectory*; §2.3 *HopTrace artifact schema* |

---

## 4. Phase C — Skill Crafter as a synthesis / repair engine

### 4.1 Goal

Promote the Crafter from "responsibilities-and-prompts" to a concrete engine with: (a) discrete composition operators, (b) backward failure localization at three layers, (c) a fixed repair taxonomy, (d) typed proposal outputs, and (e) mandatory routing through the unified gate stack. The Crafter remains a frozen 32B/72B teacher per [PLAN-SKILL-CRAFTER.md §2](../04-skill-crafter/PLAN-SKILL-CRAFTER.md); only its *control flow* is being made concrete.

### 4.2 Composition operators (first-version set)

The first-version Crafter must support exactly these four discrete composition operations. Every `ComposeProposal` (cf. [PLAN-SKILL-CRAFTER.md §2.5](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)) is built using one of them.

1. **Sequential chaining.** `A → B`. Example: `inspect(target_entity) ▷ verify(event, evidence_span)` becomes `inspect_then_verify(target_entity, event, evidence_span)`.
2. **Conditional branching.** `if cond then A else B`. Example: if `IdentityHypothesis.confidence < τ` then `resolve_identity` else `verify_event`.
3. **Slot substitution.** Replace one canonical type binding while preserving the protocol. Example: rebind `target_entity: Object → target_entity: UIElement`. Slot substitution is a transferable-IR operation; it is the most common cross-domain composition.
4. **Substep replacement.** Replace one fragile hop with a more reliable fragment. Example: `CHECK` → `GROUND ▷ CHECK ▷ VERIFY`.

### 4.3 Backward failure localization (three layers)

Repair without localization is guesswork. The Crafter localizes every failure to one of three layers before choosing a repair.

| Layer | Symptom | Localized cause |
|-------|---------|-----------------|
| **L1 — Binding** | `entity_binding_failure`, slot type mismatch | Wrong slot fill; wrong ontology mapping; wrong entity grounding |
| **L2 — Protocol** | `wrong_hop_order`, `premature_commit`, `verification_insufficient` | Hop order; missing verification; premature stopping; missing abort |
| **L3 — Scope** | Cross-domain regression; over-broad applicability | Precondition too wide; `applicable_domains` too wide; evidence requirement too weak |

Localization output is a typed `FailureDiagnosis` ([PLAN-SKILL-CRAFTER.md §6](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)) with the `layer` field set, not a free-form prose explanation.

### 4.4 Repair taxonomy (discrete operations, not "reflection")

Repair is a closed set of operations, each parameterized by the diagnosis from §4.3. The first-version repair operators are:

- `tighten_precondition`,
- `narrow_domain_scope`,
- `insert_verification_step`,
- `replace_atomic_hop`,
- `add_abort_condition`,
- `add_evidence_check`,
- `split_skill`,
- `merge_with_support_skill`.

Each repair produces a typed `PatchProposal` (existing type in [PLAN-SKILL-CRAFTER.md §2.5](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)) with the operator name and parameters recorded for audit and rollback.

### 4.5 Crafter outputs are *candidates* — never direct bank writes

This is non-negotiable and lines up with the existing unified gate spec. Crafter never writes to the active store. It only emits typed proposals into the candidate store:

- `candidate_skill` (full new `SkillIR`),
- `candidate_skill_patch` (`PatchProposal` against an existing skill),
- `candidate_composite_protocol` (`ComposeProposal` over existing skills).

These are routed through the unified gate stack in [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md): `static → replay → shadow → transfer → non-regression → promote / reject / repaired-retry`.

### 4.6 Multi-pass verification (the four passes already exist; pin them)

Crafter outputs cycle through four verification passes inside the existing gate stack. None of these passes is new; this section pins the responsibility split:

| Pass | Owner | What it checks |
|------|-------|----------------|
| **1 — Static sanity** | `SkillHarness` static gate (G0 / G1) | Schema completeness, slot legality, evidence-interface closure, adapter compatibility |
| **2 — Replay validation** | `ReplayValidator` ([PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md)) | New / repaired protocol is at least as stable as the previous version on historical traces |
| **3 — Shadow execution** | `TransferManager` shadow phase | Side-by-side execution against the active skill in real rollouts |
| **4 — Transfer validation** | `TransferManager` transfer phase | New skill generalizes across `applicable_domains` rather than overfitting to a source domain |

A skill that fails Pass 4 is not promoted; it is sent back as a `RepairProposal` with `narrow_domain_scope` already pre-staged, closing the loop with §4.4.

### 4.7 Crafter input / output contract

The Crafter's runtime contract is fixed so that the offline pipeline scheduler (the Pipeline Orchestrator) can batch Crafter calls without bespoke wiring per call site.

**Input.**

```text
CrafterInput = {
  "active_bank_skills":          list[SkillIR],
  "candidate_protocol_clusters": list[ReasoningProtocol],     # from Phase B §3.6
  "failed_hop_patterns":         list[FailedPatternCluster],  # from Phase B §3.7
  "failure_cases":               list[FailureDiagnosis],      # from Harness gate
  "transfer_mismatch_reports":   list[TransferDiagnostic],    # from Harness §10a
}
```

**Output.**

```text
CrafterOutput = {
  "candidate_skills":       list[SkillIR],
  "repair_patches":         list[PatchProposal],
  "composition_proposals":  list[ComposeProposal],
  "rationale_records":      list[RationaleRecord],            # for audit, never used as evidence by gate
}
```

`rationale_records` are explicitly *not* admissible as evidence inside the gate stack; they are audit-only.

### 4.8 Where this lands

| Plan file | Section to add |
|-----------|----------------|
| [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) | §3a *Composition Operators*; §6.4a *Backward Failure Localization (3 layers)*; §6.5a *Repair Taxonomy*; §2.6 *Crafter Output Contract*; §6a *Crafter Verification Path (4 passes pinned to existing gates)* |
| [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) | §11a *Replay / shadow / transfer validation hooks for crafted skills* (no new gate; documents which Pass 1–4 routes through which existing gate) |
| [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) | §3a *Crafter batch scheduling*; §3b *Crafted-candidate promotion path*; §3c *Rollback for faulty repaired skills* |
| [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) | §SourceTypes addendum: confirm `crafted` and `repaired` source types route through the same five-stage stack (no special-casing) |

---

## 5. The integrated closed loop (do not implement A / B / C in isolation)

The single most common failure mode for plans like this is to ship Phase A, Phase B, and Phase C as three independent tracks. The point of this document is the opposite: the value only appears when the three are wired into one loop. Concretely:

| Step | Producer | Consumer | Artifact |
|------|----------|----------|----------|
| 1 | Action Agent (online) | Episode-local trajectory | `HopTrace` per outer step |
| 2 | Skill Bank discovery (offline) | Phase B mining | Segmented, utility-attributed fragments |
| 3 | Phase B clustering / lifting | Phase A IR | `ReasoningProtocol` over typed slots |
| 4 | Phase A IR | Crafter (Phase C) input | `SkillIR` candidates over canonical ontology |
| 5 | Crafter (Phase C) | Unified gate stack | `candidate_skill / patch / composition` |
| 6 | Unified gate stack | Skill Bank active store | Promoted skill, or `rejected` / `repaired-retry` |
| 7 | Active Skill Bank | Action Agent (online) | Retrieved skill at next episode → new `HopTrace` |

This is the same loop already implied by [README §pipeline overview](../README.md#pipeline-overview), with Phase B added as an explicit producer between trajectory storage and the Crafter, and Phase A added as the typed substrate everyone shares. The Harness gate stack and the orchestrator's promotion economy are unchanged — they are the points of integration, not new components.

**Two integration invariants the loop must preserve.**

- **No skill bypasses the gate stack.** Phase B-mined protocols, Phase C-composed skills, Phase C-repaired skills, and Phase A-rebound skills (e.g., slot-substitution clones) all enter as candidates and exit through `static → replay → shadow → transfer → non-regression`. There is no "trusted source" exception, including the frozen 32B/72B teacher.
- **No state-keeping surface other than the episode-local trajectory.** `HopTrace` extends what the orchestrator already keeps per episode; pattern aggregation across episodes lives only in the Crafter-private `FailurePatternStore` and is offline-only.

---

## 6. File-by-file dispatch table

A single table of every section to add, by file. Each entry is a section header to *insert* into the named file; the §-numbers are advisory and should be reconciled with the file's existing numbering when the edits are applied.

| File | New / extended sections |
|------|-------------------------|
| [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) | §3a Canonical cross-domain ontology; §4.1a Typed Slot Skill IR; §4.3c Domain adapter contract; §5a Inner-hop skill discovery pipeline (segmentation / utility / clustering / lifting / failed-pattern mining); §9.0 Transferable skill families (minimal six) |
| [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) | §2.6 Crafter output contract; §3a Composition operators; §6.4a Backward failure localization (L1/L2/L3); §6.5a Repair taxonomy (8 operations); §6.7a Crafter consumes failed-hop pattern clusters; §6a Crafter verification path (4 passes) |
| [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) | §5.4 HopTrace logging contract; §5.5 Reasoning-step trace export; §5.6 Bounded inner-hop trace format |
| [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) | §10b Slot binding validator; §10c Ontology remap validator; §10d Adapter compatibility checker; §11a Replay / shadow / transfer hooks for crafted skills |
| [PLAN-VISUAL-SKILLS.md](../01-visual-grounding/PLAN-VISUAL-SKILLS.md) | §2a World vs belief vs grounding effect — unified definition over canonical ontology |
| [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) | §2.3 HopTrace artifact schema; §3a Crafter batch scheduling; §3b Crafted-candidate promotion path; §3c Rollback for faulty repaired skills; §4a HopTrace as part of episode-local trajectory |
| [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) | §SourceTypes addendum: `crafted` and `repaired` route through unchanged five-stage stack |
| [README.md](../README.md) | §key-shared-concepts addendum: pointer to canonical ontology; pointer to typed-operator skill procedures; pointer to inner-hop skill discovery |

---

## 7. Priorities

The three phases must be co-developed, but the order in which the *individual sub-deliverables* land matters. The split below mirrors the priority tags from the original directive.

### 7.1 P0 — must ship first

These items unblock everything else:

- Canonical cross-domain ontology (§2.2).
- Typed Slot `SkillIR` (§2.4).
- `HopTrace` logging contract on the Action Agent (§3.2).
- Inner-hop segmentation (§3.3).
- Basic hop-utility attribution (§3.4) — at least `evidence_gain` and `verification_success`.
- Crafter composition operators (§4.2).
- Crafter repair taxonomy (§4.4).
- Hard rule: every Crafter output goes through the unified gate stack (§4.5).

### 7.2 P1 — second wave

These items make Phase A and Phase C *cross-domain*; without them the system is still mostly single-domain:

- Cross-domain ontology adapters for all five domains (§2.3).
- Reasoning trace clustering over typed signatures (§3.5).
- Reusable hop-chain abstraction into `ReasoningProtocol` (§3.6).
- Backward failure localization across the three layers (§4.3).
- Multi-pass verification with the Pass 1–4 ownership pinned (§4.6).

### 7.3 P2 — later optimization

These items are pure quality / scale improvements once the loop is closed:

- Failed-hop pattern mining at scale (§3.7) — beyond the first taxonomy.
- Automatic merge / split of reasoning skills (an additional Crafter operator beyond §4.2).
- Ontology refinement driven by transfer statistics (closing the loop from `verified_domains` failure back into the canonical ontology).

---

## 8. One-paragraph summary (drop-in for higher-level docs)

> To make cross-task reasoning skill transfer real rather than prompt-template reuse, the system introduces a transferable intermediate layer based on typed slots, a canonical cross-domain ontology, and per-domain adapters. Skills are represented as typed reasoning protocols — `inspect(target_entity)`, `compare(entity_a, entity_b, attribute)`, `verify(event, evidence_span)`, `resolve_identity(entity, cue_set)`, `track_state_change(entity, event_span)`, `commit(hypothesis, warrant)` — rather than domain-specific textual procedures. In parallel, skill discovery moves beyond outer action trajectories and mines reusable reasoning procedures from inner-hop traces of the two-level MDP, with hop segmentation, hop-level utility attribution, signature clustering, protocol lifting, and failed-pattern mining as discrete pipeline stages. Finally, the Skill Crafter evolves from a conceptual module into a verifiable synthesis / repair engine with explicit composition operators, three-layer backward failure localization, a discrete repair taxonomy, and mandatory multi-pass validation through the existing replay / shadow / transfer / non-regression gates before any crafted skill enters the bank.

---

## 9. Related plans

- [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) — IR and ontology land here; minimal six skill families list lives here.
- [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) — composition operators, failure localization, repair taxonomy, output contract land here.
- [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) — `HopTrace` logging contract lands here.
- [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) — slot binding / ontology / adapter validators and crafter-validation hooks land here.
- [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) — `HopTrace` artifact schema and crafted-candidate scheduling / rollback land here.
- [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) — confirms `crafted` / `repaired` source types route through the unchanged stack.
- [PLAN-VISUAL-SKILLS.md](../01-visual-grounding/PLAN-VISUAL-SKILLS.md) — unified world / belief / grounding effect definition over the canonical ontology lands here.
- [PLAN-EDITS-HARNESS-CONTROL-PLANE.md](PLAN-EDITS-HARNESS-CONTROL-PLANE.md) — sibling edit-plan; this document follows its style and respects its terminology reconciliation (Harness = control plane; `SkillHarness` = micro-runtime).
