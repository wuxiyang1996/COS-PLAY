# PLAN: Edit Plan — Visual Grounding as a Lightweight Perception Layer

> **Document type.** This is a *plan of edits* across the existing plan files, in the same spirit as [PLAN-EDITS-HARNESS-CONTROL-PLANE.md](PLAN-EDITS-HARNESS-CONTROL-PLANE.md) and [PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md](PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md). It does **not** introduce a new module, agent, or trainable head. Every concrete addition lands in an already-existing plan file via the dispatch table in §11.
>
> **Reading order.** Start with §0 (thesis + non-goals), then §1–§10 (the twelve directives, grouped), then §11 (file-by-file dispatch table), then §12 (staged roadmap), then §13 (guiding paragraph drop-in).

---

## 0. Thesis and non-goals

**Thesis (one paragraph, drop-in for higher-level docs — see §13 for the canonical wording).**

> We treat visual grounding as a **structured perception layer**, not a main GRPO-trained policy. Its job is to convert multi-domain visual input into a usable schema with confidence, evidence references, and uncertainty flags that downstream modules can consume. In the first version, grounding is improved primarily through teacher inference, distillation, and hard-case relabeling, while GRPO is reserved for policy and skill-related decision modules.

**Non-goals (explicit).**

- Visual grounding is **not** trained with GRPO in the first version.
- There is **no** large standalone grounding evaluation framework in the core training loop.
- Grounding is **not** the center of the project; it is a support layer for the Actor, Skill Bank, Crafter, and failure analysis.
- No new agent, no new gate, no new state-keeping surface beyond the orchestrator's episode-local trajectory ([PLAN-PIPELINE-ORCHESTRATOR.md §4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping)).

**Compatibility.** The two existing invariants enforced at the Harness gate are preserved: the **general-protocol invariant** ([PLAN-SKILL-BANK.md §0.1](../03-skill-bank/PLAN-SKILL-BANK.md#01-general-protocol-invariant-no-domain-specific-skill-families)) and the **evidence-driven invariant** ([PLAN-SKILL-BANK.md §0.3](../03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills)). Phase A typed `SkillIR` and Phase B `HopTrace` from [PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md](PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md) consume the `GroundingRecord` defined here.

---

## 1. Training strategy: SFT/distillation, not GRPO

`schema_gen` is a **perception / structured parsing module**. It is trained with **supervised fine-tuning and distillation**, not GRPO. The training signal comes from a high-capacity frozen teacher (frozen 72B / GPT-4o-style), used for:

- label generation,
- hard-case relabeling,
- verification / judging,
- uncertainty calibration.

GRPO remains focused on the policy and bank-side decision heads:

- `hop_select`,
- `skill_select`,
- `action_execute`,
- `segment`,
- `contract`,
- `curator`.

The distinction is hard:

| Layer | Training method | Owner |
|-------|-----------------|-------|
| Visual grounding (perception / schema generation) | **SFT / distillation only** | `schema_gen` |
| Actor + skill usage + reasoning policy | **GRPO** | `hop_select`, `skill_select`, `action_execute` |
| Bank-side discovery / governance | **GRPO where appropriate** | `segment`, `contract`, `curator` |

There is no "grounding GRPO" head in the first version. Optional future RL is restricted to small grounding-control decisions and is described in §12 Stage 4 — never to the schema parser itself.

---

## 2. Adapter / module layout (no ambiguity)

The LoRA adapter table in [README §LoRA adapter layout](../README.md#lora-adapter-layout) is restated explicitly so that `schema_gen` is never read as "just another GRPO LoRA".

**Non-GRPO (perception / schema generation):**

- `schema_gen` — SFT / distillation only. Inputs: pixels (game frame / screenshot / video). Outputs: structured `<state>` schema + confidence + evidence + uncertainty flags (see §4 `GroundingRecord`).

**GRPO — Actor side:**

- `hop_select` — typed next-hop router over `{GROUND, CHECK, RETRIEVE, CONCLUDE, COMMIT, EXECUTE}`.
- `skill_select` — schema → which reasoning skill to invoke.
- `action_execute` — final environment action selection.

**GRPO — Skill Bank side:**

- `segment` — trajectory → skill boundary detection.
- `contract` — segment → effects contract.
- `curator` — accept / reject / merge / split / retire decisions on candidate skills.

The README LoRA table should be updated to add a "Training" column with values `{SFT/distillation, GRPO}` so the split is mechanically obvious.

---

## 3. Minimal role of visual grounding

Visual grounding is responsible for **producing a structured state representation that downstream modules can consume**. Its main outputs are:

- structured state schema,
- per-field confidence,
- evidence references / evidence spans,
- uncertainty flags,
- optional grounding error tags.

It is a **support layer** for:

- Actor decision-making (primary state input),
- skill slot filling / binding (typed slots in `SkillIR`),
- evidence-aware reasoning (`evidence_in` / `evidence_out` for `evidence_role`),
- failure analysis (the `GroundingFailureRecord` of §7),
- hard-case relabeling (feeds back into `schema_gen` SFT data).

Grounding is **not** a separate planner, autonomous policy, or competing decision agent. It does not decide what to do; it tells the rest of the system what is on the screen and how much to trust each field.

---

## 4. Runtime output object — `GroundingRecord`

Every grounding invocation emits one `GroundingRecord`. This is the canonical perception → downstream contract.

```python
GroundingRecord = {
    "state_schema":      ...,            # canonical <state> (see PLAN-VISUAL-GROUNDING.md §3)
    "field_confidence":  {...},          # per top-level field: {domain, entities, relations, ...}
    "slot_confidence":   {...},          # per shared slot: {target, blocker, constraint, candidate_set, history_anchor}
    "evidence_refs":     [...],          # EvidenceRef list: clip/frame IDs, DOM nodes, desktop elements, tool-call IDs
    "uncertainty_flags": [...],          # {low_confidence_field, ambiguous_target, missing_evidence, ...}
    "grounding_errors":  [...],          # optional: parse failures, OCR failures, region misses
    "schema_gen_ckpt":   "...",          # checkpoint id used to produce this record (for audit / rollback)
}
```

The record is the only thing passed downstream from the perception layer. Everything in the plans that previously said "the Actor reads the schema" should be updated to "the Actor reads the `GroundingRecord`."

`evidence_refs` is the canonical `evidence_out` for `GATHER` skills as already declared in [PLAN-VISUAL-GROUNDING.md §3a](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md) and [PLAN-EDITS-HARNESS-CONTROL-PLANE.md](PLAN-EDITS-HARNESS-CONTROL-PLANE.md) (P5 draft).

---

## 5. How grounding connects to the Actor

The Actor consumes `GroundingRecord` as its primary perceptual input. The connection is intentionally simple and uncertainty-aware.

- **Primary input.** The Actor reads `state_schema` as primary input, and reads `field_confidence` / `slot_confidence` / `uncertainty_flags` alongside it.
- **Uncertainty-triggered hops.** Low-confidence or uncertainty-flagged fields can trigger typed reasoning steps:
  - low confidence on a referenced entity → `GROUND`,
  - low confidence on a relation / state predicate → `CHECK`,
  - claim under review → `VERIFY`.
  Mapping rules live in [PLAN-ACTION-AGENT.md §5](../02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control) and [PLAN-ACTION-AGENT.md §10](../02-action-agent/PLAN-ACTION-AGENT.md) (uncertainty-driven `GROUND` triggering already exists; this plan pins it to `GroundingRecord` fields).
- **`skill_select` precondition.** `skill_select` may only depend on slots that are present and at or above a slot-specific confidence threshold (recorded on each `SkillIR.input_slots` entry). Slots below threshold are treated as unbound.
- **`action_execute` discipline.** `action_execute` must not treat unsupported / low-confidence fields as reliable facts. Acting on an unsupported field is logged as `unsupported_reasoning` (see §8 metric).
- **No assumption of perfect grounding.** The Actor uses uncertainty-aware state; grounding errors are first-class and bounded by `uncertainty_flags`, not silently ignored.

---

## 6. How grounding connects to the Skill Bank / Harness

Grounding affects skill use through a small, typed set of mechanisms — all already named in [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) and [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md), pinned here:

- **Slot availability** — does the `GroundingRecord` carry the slot the skill needs?
- **Slot confidence** — is `slot_confidence` above the skill's per-slot threshold?
- **Evidence availability** — does `evidence_refs` cover what the skill's `evidence_in` requires?
- **Adapter feasibility** — does the active domain adapter support the canonical types the skill references? (Cf. Phase A §2.3 in [PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md](PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md).)
- **Binding success** — do the typed slot bindings remain consistent under the current `GroundingRecord`?

Operational rules:

- Skills requiring missing or unreliable slots are **filtered or vetoed** at the Harness candidate-selection layer (existing `SkillHarness` veto path).
- Harness / runtime validation can **reject skill invocations** when grounding support is insufficient, surfaced as a typed diagnostic `grounding_support_insufficient` (parallel to existing `slot_binding_failed` / `evidence_insufficient` in [PLAN-HARNESS.md §10a](../05-harness/PLAN-HARNESS.md)).
- Evidence-carrying grounding outputs (`evidence_refs`) are the canonical input to **`VERIFY`-role skills**; a `VERIFY` skill is not invokable when its required evidence is absent from `GroundingRecord.evidence_refs`.

This keeps Harness as the control-plane gate for grounding-driven feasibility — no new component is added.

---

## 7. Grounding failure feedback — `GroundingFailureRecord`

When a downstream failure originates in grounding (rather than in skill selection or reasoning), it must be attributable. A lightweight, typed record makes this possible:

```python
GroundingFailureRecord = {
    "episode_id":        ...,
    "outer_step":        ...,
    "error_type":        ...,            # see taxonomy below
    "affected_slots":    [...],          # which shared slots were impacted
    "evidence_missing":  [...],          # which EvidenceRefs were expected but absent
    "downstream_effect": ...,            # {wrong_action, wrong_skill, premature_commit, abort, ...}
}
```

**Error type taxonomy (first version).** `entity_missed`, `entity_misidentified`, `relation_wrong`, `region_off`, `ocr_failure`, `state_flag_wrong`, `evidence_localization_failure`, `low_confidence_unflagged`.

**Consumers.**

- **Failure analysis** — orchestrator-level failure attribution; counted alongside the existing transfer-failure diagnostics in [PLAN-HARNESS.md §10a](../05-harness/PLAN-HARNESS.md).
- **Crafter repair input** — `GroundingFailureRecord` joins `FailureDiagnosis` as Crafter input (cf. Phase C §4.7 of [PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md](PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md)). Crafter does not edit `schema_gen`; it only adjusts skills whose feasibility depends on grounding signals it should not have trusted.
- **Hard-case relabeling** — failed grounding instances are routed back through the frozen teacher for relabeling and added to the next `schema_gen` SFT batch.
- **Future `schema_gen` data improvement** — aggregated failure clusters drive what the teacher should be asked to relabel next.

**Attribution discipline.** Not every downstream failure should be blamed on skill selection or reasoning. The existence of `GroundingFailureRecord` makes the alternative attribution path mechanical: when both a `FailureDiagnosis` and a `GroundingFailureRecord` exist for the same outer step and the grounding error temporally precedes the skill failure, the failure is attributed to grounding first.

---

## 8. Lightweight grounding evaluation

Grounding evaluation is **downstream-oriented and small**. There is no large independent grounding-evaluation framework in the core loop.

**Core grounding-facing metrics (kept in the main loop).**

- `schema_completeness` — fraction of canonical schema fields produced per episode.
- `actor_consumable_schema_rate` — fraction of episodes whose `GroundingRecord` is sufficient for the Actor to make a decision without an additional `GROUND` hop.
- `evidence_sufficiency` — fraction of `VERIFY`-role skill invocations whose required `evidence_in` was present in `GroundingRecord.evidence_refs`.
- `unsupported_reasoning_rate` — rate at which `action_execute` acted on a low-confidence / uncertainty-flagged field.

**Optional diagnostics (appendix only — not core loop metrics).**

- field-level accuracy,
- slot-level extraction accuracy,
- uncertainty calibration (e.g., ECE, reliability diagrams),
- evidence localization (IoU / span-level F1).

These diagnostics are computed offline on held-out slices and are not gated by; they inform the next SFT batch.

---

## 9. Missing implementation pieces (the real list)

The following are the actual gaps in visual grounding. The list is deliberately **not** "GRPO training is missing", because GRPO training is intentionally out of scope.

1. **Benchmark loaders** — unified loaders for the benchmark slices already named in [PLAN-VISUAL-GROUNDING.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md) (VisualToolBench / TIR-Bench / SIV-Bench / Video-Holmes plus Gym-V / BrowserGym / OSWorld paired data).
2. **`schema_gen` training pipeline** — SFT / distillation pipeline driven by frozen-teacher labels, with a hard-case relabeling lane.
3. **Grounding evaluation harness** — small, downstream-oriented eval (the four core metrics in §8) plus the offline diagnostics appendix.
4. **Grounding output contract** — `GroundingRecord` (§4) implemented and adopted by all downstream consumers.
5. **Uncertainty propagation** — wiring of `field_confidence` / `slot_confidence` / `uncertainty_flags` into Actor (§5) and Skill Bank / Harness (§6).
6. **Grounding failure feedback path** — `GroundingFailureRecord` (§7) wired to Crafter inputs and to the hard-case relabeling lane.

---

## 10. Staged roadmap

A simple, four-stage roadmap. Stage 4 is explicitly optional and is the only place where any RL appears.

**Stage 1 — Bootstrap (frozen teacher, no training).**

- Use frozen high-capacity teacher inference (frozen 72B / GPT-4o-style) to generate structured labels.
- Build benchmark loaders.
- Define the canonical `<state>` schema output (cf. [PLAN-VISUAL-GROUNDING.md §3](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md)).
- Define `GroundingRecord` (§4).

**Stage 2 — Distill and integrate.**

- Train `schema_gen` with SFT / distillation against teacher labels.
- Add confidence + evidence outputs to `GroundingRecord`.
- Connect grounding to Actor (§5) and Harness (§6).

**Stage 3 — Hard-case loop.**

- Add lightweight hard-case relabeling driven by `GroundingFailureRecord`.
- Improve uncertainty handling (calibration, threshold tuning per slot).
- Evaluate **downstream usefulness** (the §8 core metrics), not standalone grounding accuracy.

**Stage 4 — Optional future RL (grounding control only).**

- If — and only if — Stages 1–3 leave grounding-control decisions on the table, explore RL for *small grounding-control decisions* such as:
  - re-observation,
  - region refinement,
  - tool-path routing,
  - uncertainty-triggered re-grounding.
- This RL applies **only to a grounding-control policy**, never to the core schema parser. The core schema parser remains SFT / distillation.

---

## 11. File-by-file dispatch table

A single table of every section to add or update, by file. Section numbers are advisory and should be reconciled with each file's existing numbering when the edits are applied.

| File | Updates |
|------|---------|
| [PLAN-VISUAL-GROUNDING.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md) | Reposition entire document as **SFT / distillation-based perception**: explicit "no GRPO for `schema_gen`" statement (§1); restated module split (§2); minimal-role section (§3); add §3a-bis `GroundingRecord` definition (§4); add §6a Grounding evaluation — lightweight only (§8); add §7a `GroundingFailureRecord` and feedback path (§7); add §11 missing-pieces list (§9); add §12 staged roadmap (§10). Remove or qualify any wording that suggests `schema_gen` is GRPO-trained. |
| [PLAN-VISUAL-GROUNDING-MILESTONES.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) | Rewrite milestone backbone as **teacher → distill → integrate → hard-case-improve**. Drop GRPO-training milestones for `schema_gen`. Add Stage 4 as explicitly optional and grounding-control-only. |
| [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) | Add §5.7 *Consuming `GroundingRecord`*: uncertainty-aware state, uncertainty-triggered `GROUND`/`CHECK`/`VERIFY` hops (§5), `skill_select` confidence-threshold rule, `action_execute` `unsupported_reasoning` rule. Cross-link from existing §10 (uncertainty-driven `GROUND` triggering). |
| [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) | Add §10e *Grounding-driven eligibility / binding / veto*: slot availability, slot confidence, evidence availability, adapter feasibility, binding success (§6). Add `grounding_support_insufficient` to the §10a transfer-failure diagnostic taxonomy. |
| [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) | Add §6a *Slot confidence and evidence availability as part of skill usability*: per-slot confidence thresholds on `SkillIR.input_slots`; `VERIFY`-role skills require coverage by `GroundingRecord.evidence_refs`. |
| [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) | Add §2.4 *`GroundingRecord` artifact schema*; add §6a *Grounding as a tracked module with snapshots and fixed lightweight evaluation* — pins the §8 core metrics into the cadence and adds `schema_gen_ckpt` to the snapshot store; treat grounding as a tracked module, not an evaluation framework. |
| [README.md](../README.md) | Update §pipeline overview and §LoRA adapter layout: add a "Training" column to the LoRA table with values `{SFT/distillation, GRPO}`; mark `schema_gen` as `SFT/distillation`. Add the §13 guiding paragraph (below) to the §key-shared-concepts area. |

---

## 12. Integrated execution order

These edits are independent of [PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md](PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md) but compose with it:

- `GroundingRecord.evidence_refs` is the canonical `evidence_out` source for `GATHER`-role skills in the typed `SkillIR` of Phase A.
- `GroundingRecord.field_confidence` / `slot_confidence` / `uncertainty_flags` are the typed signals that the Actor's `hop_select` reads to trigger inner-MDP hops, which are then logged as `HopTrace` in Phase B.
- `GroundingFailureRecord` joins Phase C's Crafter inputs (§7), so grounding-attributed failures drive Crafter repair the same way reasoning-attributed failures do.

Apply the edits in this order:

1. P0 — `GroundingRecord` (§4) and the `schema_gen` training declaration (§1, §2). Without these, all downstream wiring is blocked.
2. P0 — Actor wiring (§5) and Harness wiring (§6).
3. P1 — `GroundingFailureRecord` (§7) and the Crafter / hard-case relabeling feedback path.
4. P1 — Lightweight evaluation (§8) wired into the orchestrator.
5. P2 — Stage 4 grounding-control RL exploration (only if Stages 1–3 are insufficient).

---

## 13. Guiding paragraph (drop-in)

Add the following paragraph to [README.md](../README.md) §key-shared-concepts and to the top of [PLAN-VISUAL-GROUNDING.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md):

> We treat visual grounding as a structured perception layer rather than a main GRPO-trained policy. Its job is to convert multi-domain visual input into a usable schema with confidence, evidence references, and uncertainty flags that downstream modules can consume. In the first version, grounding is improved primarily through teacher inference, distillation, and hard-case relabeling, while GRPO is reserved for policy and skill-related decision modules.

---

## 14. Related plans

- [PLAN-VISUAL-GROUNDING.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md) — primary target of this edit pass; repositioned as SFT/distillation-based perception.
- [PLAN-VISUAL-GROUNDING-MILESTONES.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) — milestone backbone rewritten to teacher → distill → integrate → hard-case-improve.
- [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) — Actor consumes `GroundingRecord`; uncertainty-aware state and uncertainty-triggered hops land here.
- [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) — grounding-driven eligibility / binding / veto land here.
- [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) — slot confidence and evidence availability as skill-usability constraints land here.
- [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) — `GroundingRecord` artifact schema, snapshot tracking, and lightweight evaluation cadence land here.
- [README.md](../README.md) — LoRA training-method column, simplified grounding role, guiding paragraph land here.
- [PLAN-EDITS-HARNESS-CONTROL-PLANE.md](PLAN-EDITS-HARNESS-CONTROL-PLANE.md) — sibling edit-plan; this document follows its style and respects its terminology reconciliation.
- [PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md](PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md) — sibling edit-plan; `GroundingRecord` is the canonical `evidence_out` for `GATHER`-role skills in that plan's typed `SkillIR`.
