# DISCUSSION: Who Does What — Skill Bank Agent vs. Skill Harness vs. Pipeline Orchestrator

**Purpose.** This file is a *role walkthrough*, not a new module spec. It uses concrete short-video reasoning scenarios to make the component split crisp: when the same workflow runs end-to-end, who owns which step, and — equally important — what each component **must not** do. It is a companion to:

- [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) — skill content production and curation (Agent 2 medium-timescale + Crafter/Agent 3 proposals).
- [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) — typed proposals fed into the bank's lifecycle (`PatchProposal | ComposeProposal | TransferProposal | RetireProposal`).
- [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) — per-invocation skill runtime (the `SkillHarness` class).
- [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) — system-level control plane (the macro DAG, gates, snapshots, training cadence).
- [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) — the gate stack that all of the above share.

---

## 0. Terminology note (read first)

The repo has **two senses of "harness"** ([PLAN-EDITS-HARNESS-CONTROL-PLANE.md §0](../legacy/10-edits/PLAN-EDITS-HARNESS-CONTROL-PLANE.md#0-terminology-reconciliation-do-this-first)):

- **Skill-invocation runtime (micro-harness, `SkillHarness`)** — defined in [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md). One skill call. Throughout this file, **"Harness"** means this micro-runtime.
- **System-level control plane (the macro Harness)** — defined in [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md). The DAG that runs many episodes, owns gates, snapshots, and lifecycle. Throughout this file, **"Orchestrator"** means this control plane.

**"Skill Bank Agent"** is shorthand for the bank's content-production responsibilities: Agent 2 (medium-timescale: `SEGMENT`, `CONTRACT`, `CURATOR`) plus the Crafter / Agent 3 (slow-timescale: composition, transfer, hypothesis), as defined in the [README three-agent split](../README.md#three-agent-role-split--model-convention). It is the component that *creates or revises skill contents* in the bank; it is **not** the component that runs a single skill on a single state.

---

## 1. The intuition in one line

| Component | One-line role |
|-----------|---------------|
| **Skill Bank Agent** | Invent / revise skills (segment trajectories, learn contracts, curate the bank). |
| **Harness** (`SkillHarness`) | Execute one skill call safely (retrieve → bind slots → attach adapter → check evidence → run → log `SkillEpisode`). |
| **Orchestrator** | Manage the whole lifecycle (rollout DAG, snapshots, acceptance gates, promotion, rollback, training schedule, re-evaluation). |

Software analogy:

- **Skill Bank Agent** = code generator / maintainer.
- **Harness** = unit-test + runtime wrapper for one call.
- **Orchestrator** = CI/CD + release manager.

R&D analogy:

- **Skill Bank Agent** = R&D team that designs new tools.
- **Harness** = technician who tries one tool on one job safely.
- **Orchestrator** = factory manager who decides which tool version is approved for production.

---

## 2. Four worked scenarios

All scenarios are framed around the project's near-term arena — short-video evidence-grounded reasoning ([README](../README.md#current-execution-focus)) — but the same split holds in any of the five target domains.

### 2.1 Scenario A — Using an existing skill during one short-video QA episode

**Situation.** The actor is answering: *"Who noticed the hidden object first?"* A reusable skill already exists in the bank, e.g. `track_object_visibility_across_shots` (a `GATHER` / `VERIFY` protocol bound through a video adapter).

**Skill Bank Agent — does nothing online at this moment.** It already produced and curated this skill in earlier passes (segmentation, contract learning, bank curation). The runtime use of the skill is not its concern.

**Harness — this is exactly its job.** It receives the current state and one selected skill call, and:

1. Retrieves the candidate skill from the bank.
2. Binds slots: `object = "red folder"`, `character = "woman in blue coat"`.
3. Checks preconditions and evidence sufficiency (Gate G0 — evidence-driven contract).
4. Attaches the video-domain adapter.
5. Runs the skill.
6. Records a `SkillEpisode` with `evidence_role`, `evidence_in`, `evidence_out`, `evidence_warrant` ([PLAN-HARNESS.md §5.1](../05-harness/PLAN-HARNESS.md#51-skillepisode)).
7. Types the outcome as `success / fail / abort / stall`.
8. Logs evidence trace and reward pieces.

This matches the Harness's plan-level definition: *the micro runtime for skill use*, refactoring invocation into **retrieval → slot binding → adapter attachment → evidence check → execution → logging**.

**Orchestrator — does not bind slots or step the skill.** It sits above and says, in effect:

- This episode belongs to run `R12`.
- Use snapshot `bank_v7`.
- Collect the produced `SkillEpisode` into the run's typed trajectory ([PLAN-PIPELINE-ORCHESTRATOR.md §4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping)).
- Keep this trace for later evaluation, mining, and promotion decisions.

**One-sentence summary.** *Harness executes the call; Orchestrator records where that call belongs in the larger run.*

**What must not happen.**

- The Harness must not promote, retire, or rewrite the skill based on this single episode.
- The Skill Bank Agent must not be invoked online inside the inner loop to "patch" the skill mid-episode.
- The Orchestrator must not reach into the skill's slot binding or adapter attachment.

---

### 2.2 Scenario B — Transferring a semantic skill from game / web to short-video reasoning

**Situation.** Reuse a semantic protocol like `identify_primary_blocker_before_next_action`. In a browser task this means *"which popup blocks the button"*; in short-video reasoning it means *"which missing visual evidence blocks answering the question"*.

**Skill Bank Agent — may have created or abstracted this skill earlier as a reusable semantic pattern.** It can later refine its contract if transfer repeatedly fails. It does **not** run the transfer online.

**Harness — this is one of its most important responsibilities.** [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) explicitly defines transfer as: *reuse the semantic skill + rebind or synthesize the target-domain adapter*, and calls this the single most important invariant the Harness enforces.

For this scenario the Harness will:

1. Keep the semantic skill unchanged.
2. Rebind its slots into the video domain (e.g. `blocker → missing_evidence_clip`).
3. Attach or synthesize the target-domain adapter.
4. Run replay checks against held-out frozen traces (Gate G3 — replay).
5. Run **shadow execution first** — retrieved, slot-bound, adapter-attached, evaluated, but **not allowed to control the active actor policy or affect environment reward** (Gate G4 — shadow).
6. Only later allow active use if the gates pass.

**Orchestrator — does not perform the shadow run itself.** It owns the bigger policy:

- Whether this transfer attempt is part of the current experiment.
- Which frozen trace slice to replay on.
- Whether the resulting evidence is sufficient for promotion.
- Whether the bank snapshot pointer should advance or stay frozen.

**One-sentence summary.** *Harness proves "this transferred skill can run locally"; Orchestrator decides "this transferred skill is now allowed to affect the system globally."*

**What must not happen.**

- The Harness must not let a transferred skill influence the active actor policy or environment reward before the shadow → active gate transition is approved.
- The Skill Bank Agent must not edit the semantic skill to make a single transfer attempt succeed; refinement requires accumulated evidence and a typed proposal ([PLAN-SKILL-CRAFTER.md §2.5](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)).
- The Orchestrator must not bypass the shadow → active transition just because one episode looked good.

---

### 2.3 Scenario C — Discovering a new candidate skill after many failures

**Situation.** Across many episodes the system keeps failing on questions like *"Who knew the truth before the confrontation?"* From accumulated traces, a useful pattern recurs: `compare_belief_state_before_and_after_dialogue_turn`.

**Skill Bank Agent — this is where it becomes active.** Per [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) and [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md), it:

1. Looks at accumulated rollouts.
2. Segments recurring useful sub-trajectories (`SEGMENT`).
3. Learns or patches the contract (`CONTRACT`), declaring `evidence_role` and an `evidence_interface` (Clauses A and B of the evidence-driven invariant, [PLAN-SKILL-BANK.md §0.3](../03-skill-bank/PLAN-SKILL-BANK.md)).
4. Decides whether this is a new skill, a merge, a split, or a refinement.
5. Proposes a candidate bank update as a typed proposal (`PatchProposal | ComposeProposal | TransferProposal | RetireProposal`).

This is the COS-PLAY-style role: **skill content production and curation**.

**Harness — once that candidate exists, the Harness tests it as an invocation runtime object.**

- Can it bind on real states?
- Does the evidence check pass (Gate G0)?
- Does replay agree with held-out traces (Gate G3)?
- Does shadow execution behave stably (Gate G4)?

The Harness evaluates **runtime viability** of the proposed skill, not whether the skill should *exist conceptually*.

**Orchestrator — centralizes the acceptance decision.** Per [PLAN-PIPELINE-ORCHESTRATOR.md §3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md):

- No bank promotion without the gate stack.
- Replay before promote.
- Gate failure leads to rollback or quarantine.
- Promotion creates a new bank snapshot.
- Rollback restores the last good snapshot.

After the Skill Bank Agent proposes the candidate and the Harness produces runtime evidence, the Orchestrator asks: did it pass static checks, did replay verification pass, is there non-regression, should `current_production` move to a new snapshot?

**One-sentence summary.** *Skill Bank Agent proposes the new skill; Harness tests whether it runs; Orchestrator decides whether it ships.*

**What must not happen.**

- The Harness must not invent or re-author the skill contract; opaque skills (no `evidence_in` / `evidence_out`) are rejected at G0 rather than silently fixed.
- The Skill Bank Agent must not write directly into the active store; new skills enter `draft_store` / `candidate_store` and only reach `active_store` via the Orchestrator's promotion transaction ([PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)).
- The Orchestrator must not promote based on aggregate counts alone; it consumes the typed gate verdicts (`GateVerdict`) the Harness emits.

---

### 2.4 Scenario D — Production regression after promotion

**Situation.** A newly promoted skill hurts performance on a held-out short-video eval slice.

**Skill Bank Agent — not the rollback actor.** It may later use the failure artifact to refine, split, or retire the offending skill (typed `PatchProposal` or `RetireProposal`), but it does not perform the rollback itself.

**Harness — provides the local evidence.** It supplies the detailed `SkillEpisode` traces showing where the skill broke: which gate diagnostic fired (`slot_binding_failed`, `adapter_execution_mismatch`, `evidence_insufficient`, `temporal_mismatch`, `ui_grounding_mismatch`, `desktop_object_mismatch`, `overconfident_commit`, `contract_mismatch` — see [PLAN-HARNESS.md §10a](../05-harness/PLAN-HARNESS.md)), and what the evidence interface looked like at failure time.

**Orchestrator — squarely its responsibility.** Per [PLAN-PIPELINE-ORCHESTRATOR.md §3](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md), post-promotion regression triggers **pointer reversion to the last good `snapshot_id`**, and offending skills can be **quarantined**. Promotion and rollback are centralized in the Orchestrator's acceptance-gate policy.

**One-sentence summary.** *Harness tells you what broke; Orchestrator decides to roll back.*

**What must not happen.**

- The Harness must not unilaterally disable a skill mid-run; it reports verdicts and lets the Orchestrator act.
- The Skill Bank Agent must not silently overwrite the offending skill in the active store; revisions are typed proposals re-entering the gate stack.
- The Orchestrator must not roll back without a recorded `GateVerdict` and snapshot lineage — rollbacks are auditable transactions, not free-form edits.

---

## 3. The "single rule" decision tree

When unsure where a responsibility belongs, ask exactly one question:

1. **Is this about a single skill invocation?** → **Harness.**
   - Slot binding, precondition check, evidence check, execution trace, failure typing, shadow run for this skill call.
2. **Is this about creating or revising the skill itself?** → **Skill Bank Agent.**
   - Segmentation, contract learning, merge / split / retire, bank curation, typed proposals.
3. **Is this about the whole system version or lifecycle?** → **Orchestrator.**
   - Rollout DAG, acceptance gate stack, promotion / rollback, snapshot pointer, training schedule, experiment versioning.

If a task seems to span two boxes, it is almost always the **interface artifact** (a `SkillEpisode`, a `GateVerdict`, a `SkillRecord`, a snapshot pointer), and the artifact tells you which side owns the read and which side owns the write.

---

## 4. Summary table

| Scenario | Skill Bank Agent | Harness (`SkillHarness`) | Orchestrator | What must not happen |
|----------|------------------|--------------------------|--------------|----------------------|
| **A. Existing skill, one episode** | Idle (skill produced earlier). | Retrieve → bind slots → check evidence → attach adapter → run → log `SkillEpisode`. | Tag the episode to the run, snapshot, and trajectory; collect the `SkillEpisode`. | Harness promotes/rewrites the skill on one episode; Bank Agent edits mid-episode; Orchestrator binds slots. |
| **B. Cross-domain transfer (game/web → short-video)** | Optional later refinement if transfer repeatedly fails. | Keep semantic skill, rebind slots, attach/synthesize target adapter, replay, **shadow run only** until gates pass. | Decide if this transfer is part of the experiment, pick frozen replay slice, decide on shadow → active transition and snapshot advance. | Transferred skill influences active actor or env reward before gate passes; Bank Agent rewrites skill to make one transfer succeed; Orchestrator skips shadow. |
| **C. New candidate from repeated failures** | Segment recurring sub-trajectories, learn/patch contract, declare `evidence_role` + `evidence_interface`, propose typed `Patch / Compose / Transfer / Retire`. | Test runtime viability: binding, G0 evidence, G3 replay, G4 shadow. | Run the gate stack, decide promotion, create new bank snapshot or quarantine. | Harness invents the contract; Bank Agent writes to active store directly; Orchestrator promotes on aggregate counts without typed `GateVerdict`. |
| **D. Post-promotion regression** | Later: refine / split / retire the offender via typed proposal. | Emit `SkillEpisode` traces with typed failure diagnostics. | Revert pointer to last good `snapshot_id`; quarantine offending skill; record audit. | Harness disables a skill unilaterally; Bank Agent overwrites the active store silently; Orchestrator rolls back without recorded verdict / lineage. |

---

## 5. Non-goals of this file

- This file does **not** introduce new modules, new agents, new trainable heads, new gates, or new artifacts.
- It does **not** change the inner-MDP primitives, the `<state>` schema, or the slot ontology.
- It does **not** override any plan it links to. Where the live plan files and this walkthrough disagree, **the live plan files win** — this file exists to make the existing split easier to remember and to onboard new contributors faster.
