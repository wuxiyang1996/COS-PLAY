# Skill lane decision — skills are retrieval payloads, not runnable programs

> **Status:** decided 2026-05-01.
> **Verdict:** **Lane (a) — Context-only skills.** A skill is a
> *semantic retrieval payload* — name, description, preconditions /
> effects / role labels, optionally NL guidance — that the actor LLM
> consults during decision-making. Skills are **not** runnable
> programs invoked by the harness at training time.
> **Cross-refs:**
> [`crafter-harness-orchestrator-roles.md` §7.3 / §7.4](crafter-harness-orchestrator-roles.md)
> (where the lane choice was first surfaced),
> [`pre-training-readiness-audit.md` §0.1 T1.3](../pre-training-readiness-audit.md)
> (the open audit item this doc closes),
> [`single-vs-two-mdp-tradeoff.md`](single-vs-two-mdp-tradeoff.md)
> (companion decision: one MDP, two LoRAs),
> [`harness-usability-and-intra-gymv-transfer.md`](harness-usability-and-intra-gymv-transfer.md)
> (the harness wiring the actor consults),
> [`protocol-lift-design.md`](protocol-lift-design.md)
> (the typed-protocol artefacts produced offline — see §5).

## 1. The decision

A skill in COS-PLAY is:

* **A retrieval payload** — `(skill_id, name, summary,
  strategic_description, tags, preconditions, contract.effects_*,
  expected_evidence_roles, …)` packaged for vector-store retrieval and
  prompt injection.
* **Procedural guidance** for the LLM, not a callable. The actor LLM
  reads the retrieved skill(s) when deciding the next action;
  `skill_selection` LoRA picks which retrieved skill is most
  relevant; `action_taking` LoRA produces the actual action token.
* **Optional `protocol`** — when present, an NL or typed list of hops
  is *guidance text*, not an execution plan. The harness does not
  dispatch protocol hops at training time.

The actor architecture this entails is exactly what already ships:

* one MDP (companion decision in
  [`single-vs-two-mdp-tradeoff.md`](single-vs-two-mdp-tradeoff.md))
* two GRPO LoRAs (`skill_selection` + `action_taking`) — already
  warm-started in
  [`runs/sft_coldstart/decision/`](../../runs/sft_coldstart/decision)
* harness used as an **eligibility filter and validator**, not as a
  skill executor — already wired by
  [`trainer/coevolution/_harness_hook.py`](../../trainer/coevolution/_harness_hook.py)
  (Day-10) which only calls
  `harness_hook.filter_candidates(...)` and
  `harness_hook.validate_choice(...)`, never `harness.run_skill(...)`

## 2. Why lane (a)

Each of the following points was independently surfaced in earlier
audits and notes; the decision crystallises them:

1. **Cold-start data shape matches lane (a) verbatim.** Every output
   of [`labeling/extract_skillbank_*`](../../labeling) ships skills as
   `(name, summary, strategic_description, contract.effects_*,
   expected_evidence_roles, NL protocol)` — the canonical RAG-friendly
   shape. Forcing lane (b) would require a second cold-start pipeline
   that mints adapter-compatible action sequences, which we don't
   have ([`crafter-harness-orchestrator-roles.md` §7.3 row "Cold-start
   fit"](crafter-harness-orchestrator-roles.md)).
2. **Live decision loop is already lane (a).** The actor calls
   `select_skill → update_intention → take_action` and reads the
   retrieved skill *as context*; the harness is a guardrail
   ([`decision_agents/README.md` §"TL;DR"](../../decision_agents/README.md)).
3. **Cross-task transfer rides on retrieval embeddings, not on
   adapter dispatch.** A Tetris `[CLEAR]` retrieval and a webform
   `[CLEAR]` retrieval share an embedding neighbourhood; that's the
   transfer hook that already works
   ([`single-vs-two-mdp-tradeoff.md` §"What does enable transfer (already shipped)"](single-vs-two-mdp-tradeoff.md)).
   Adapter-dispatch transfer requires real env executors for every
   target — which we don't have for `osworld` / `video` (T1.4).
4. **Crafter's lane-(b) machinery has been producing closed-loop
   noise.** All 23 fresh `PatchProposal`s in the post-fix run were
   `recovery_strategy=hop_insertion` — the same canned edit applied N
   times ([`crafter-harness-orchestrator-roles.md` §7.1
   row 4](crafter-harness-orchestrator-roles.md)). The Repairer is
   patching protocols that aren't being executed; the symptoms are
   dead-code symptoms.
5. **Harness already lives behind a single-MDP / lane-(a) wire.** The
   Day-10 trainer integration calls **only** the lane-(a)-compatible
   harness surfaces (`select_eligible_skills` and
   `validate_invocation`); `run_skill` is only ever called offline by
   `labeling_supplement/_phase4_transfer_cycle.py` and
   `labeling_supplement/_phase2_real_env_skill_smoke.py`.
6. **Lane (a) is recoverable.** Lane (b) machinery (typed protocols,
   `effect_predicates`, gymv executor + success_fn, `ReplayValidator`
   action-walk) **stays in tree as offline gate / diagnostic
   infrastructure** (see §5). If the empirical results force a flip
   later, none of that work is lost.

## 3. What ships under lane (a)

### 3.1 The actor decision loop (as today)

```
ActorAgent.step(state):
    candidates  = top_k_skill_candidates(state)            # RAG retrieval
    eligible, rejected = harness_hook.filter_candidates(   # G0/G2 + task axis
        skill_candidates=candidates, state=state)
    chosen      = skill_selection_LoRA(state, eligible)    # LLM picks one
    verdict     = harness_hook.validate_choice(            # second-pass veto
        skill_id=chosen, state=state, bindings=...)
    if verdict.veto:
        chosen = next_eligible(eligible) or None
    action      = action_taking_LoRA(state, chosen)        # LLM emits action
    env.step(action)
```

**No `harness.run_skill(...)` call. No N-step skill invocation. No
inner MDP.** The actor consults the retrieved skill as prompt context
and produces one env action per LLM call.

### 3.2 The retrieval payload contract

A `SkillRecord` in lane (a) is defined by:

| Field | Required? | Used for |
|---|---|---|
| `skill_id`, `name`, `summary` | yes | identity + retrieval text |
| `strategic_description` | yes | "what this skill does and when to use it" — actor prompt |
| `tags` | yes | retrieval routing; `expected_tag_pattern` cross-check |
| `contract.effects_add / del / event` | yes | retrieval scoring (state-effect match) + offline gate diagnostics |
| `contract.preconditions` | yes | actor prompt + harness eligibility filter |
| `contract.expected_evidence_roles` | yes | G0 invariant on `SkillEpisode` (still enforced) |
| `feasible_tasks` / `verified_tasks` | yes | F2′ task-axis veto in `EligibilityFilter` |
| `protocol` (NL or typed) | optional | **prompt guidance only** — actor reads it as text; harness does not dispatch it |
| `execution_hint.{termination_cues, common_failure_modes}` | optional | actor prompt for skill-bind / unbind heuristics |
| `n_instances`, `retired`, `version` | yes | lifecycle bookkeeping |

A skill without a `protocol` is fully valid in lane (a). A skill with
a typed `protocol` is also valid — the harness's offline replay /
shadow gates can use it (see §5), but the live actor only reads its
NL surface.

### 3.3 Crafter modes that retain meaning

| Mode | Role under lane (a) |
|---|---|
| **Hypothesizer** (primary) | Mint a new retrieval payload for a recurring uncovered situation. LLM-backed via `SkillCrafterService.set_teacher_model`. |
| **Composer (re-purposed as Merge)** | Merge two skills with overlapping retrieval profiles into a single richer payload. New proposal type: `MergeProposal{absorbed_ids, merged_description, merged_tags}`. |
| **Retire** | Drop low-utility retrievers. Threshold becomes `min_retrievals_per_skill` (a skill that never gets retrieved is dead, regardless of domain count). |
| **Generalizer (re-scoped)** | "This skill's retrieval profile matches situations in domain Y." Outputs a `feasible_tasks` extension, *not* an adapter binding recipe. |
| **`record_false_binding_pattern` ingest (Day-9c)** | Stays — `RejectedSkill` patterns from the live filter are valid evidence that a retrieval payload is misleading and should be rewritten. |

### 3.4 Crafter modes that go dark (quarantined behind a flag)

| Mode | Why dark | Disposition |
|---|---|---|
| **Repairer** | Patching a non-executable retrieval payload is a no-op | Move `crafter/repairer.py` behind `SkillCrafterService(enable_protocol_patching=False)` default. Code stays in tree for a possible future lane-(b) flip. |
| `RecoveryStrategy.{HOP_INSERTION, PROTOCOL_PATCH, FALLBACK_INJECTION, REGROUNDING_TRIGGER, SKILL_DECOMPOSITION}` | All five are protocol-edit ops on a non-executable payload | Same flag; emit `DeprecationWarning` if instantiated outside offline lane-(b) tooling |
| `PatchProposal` minting from live `FailureTrace` | Closed-loop noise generator (23 / 23 identical patches in the live run) | Disabled by default; offline drivers (`decide_skill_crafting_gpt54.py`) keep the path for diagnostic dumps but do not write back to the live bank |

### 3.5 New `FailureClass` taxonomy (replaces the protocol-edit one)

Lane (a) replaces the six-strategy protocol-edit taxonomy with a
retrieval-centric one. The Crafter's job becomes "tell the bank
curator what to mint, retire, or rewrite":

| Class | Trigger | Crafter response |
|---|---|---|
| `BANK_GAP` | Hypothesizer can't find a matching skill for a recurring `(state, intention)` cluster | Mint via `HypothesizeProposal` |
| `RETRIEVAL_MISLEAD` | The matched skill misled the actor's reasoning (post-action signal: actor took an action contradicting the retrieved skill's contract) | Rewrite the retrieval payload via `RewriteProposal` |
| `STALE_DESCRIPTION` | Skill description is no longer accurate for the situations it's matching (drift from `verified_tasks`) | Rewrite or retire via `RewriteProposal` / `RetireProposal` |

These are additive on top of the existing typed proposal hierarchy in
[`data_structure/extensions/bank_mutation.py`](../../data_structure/extensions/bank_mutation.py);
they do not require deleting the lane-(b) classes (which stay in tree
under the quarantine flag).

### 3.6 G0 invariant — preserved

Lane (a) does **not** weaken the evidence-driven invariant
(`contract.expected_evidence_roles ⊆ state.evidence`). The actor
emits a `SkillEpisode` per skill consultation; the G0 check fires on
that episode regardless of whether the skill had a runnable protocol
or not. This is what the Day-7d typed-evidence work landed: roles are
attached to evidence refs, the G0 check is a set-containment check.

### 3.7 Multi-domain ACTIVE invariant — dropped

The lane-(b) invariant "skill must be `feasible_domains ⊇ ≥2` to reach
`ACTIVE`" is replaced by **`min_retrievals_per_skill`**: a skill that
never gets retrieved (or always gets vetoed) is dead, regardless of
its domain count. This unblocks `ACTIVE` promotion on single-domain
gymv banks (which is what the cold-start corpus produces).

`feasible_tasks` (the Day-2 task axis) and `verified_tasks` (the
Day-7c persistence loop) **stay** — they're orthogonal to the
lane-(b) `feasible_domains ≥ 2` rule and provide the actual eligibility
narrowing the live filter relies on.

## 4. Implications for the trainer launch

| Subsystem | Change | Effort |
|---|---|---|
| `trainer/coevolution/_harness_hook.py` | None — already lane-(a)-compatible. Calls only `filter_candidates` + `validate_choice`. | 0 |
| `trainer/coevolution/_crafter_hook.py` | Pass `enable_protocol_patching=False` when constructing `SkillCrafterService`. Drop the `false_binding_patterns` flush *only if* lane (a) wants to disable it (recommendation: keep — `RejectedSkill` evidence is still actionable). | 30 min |
| `trainer/coevolution/_promotion_hook.py` | None — `PromotionOrchestrator` doesn't depend on the protocol-edit machinery. | 0 |
| `crafter/skill_crafter_service.py` | Add `enable_protocol_patching: bool = False` constructor flag; gate the Repairer dispatch path behind it. | 1 h |
| `decision_agents/skill_interface.py` | Replace `SkillBankProvider.skill_summary` to surface `strategic_description` + `protocol` (NL) as the actor's prompt context (already mostly in place). | 30 min |
| `harness/eligibility.py` | None — F2′ task axis is the right narrowing for lane (a). | 0 |
| `data_structure/extensions/bank_mutation.py` | Add three new proposal subclasses: `RewriteProposal`, `MergeProposal` (rename `ComposeProposal` if cleaner), `RetireProposal` already exists. | 2 h |
| `crafter/{hypothesizer, composer, retirer}.py` | Existing modules — repurpose Composer→Merge per §3.3; no new modules required. | 1-2 sessions |

**Net effort: 2-3 sessions, no architectural rewrites.** The bulk of
the lane-(a) machinery is already shipped; what's left is renaming
`PatchProposal`-family flows to `RewriteProposal` and gating the
Repairer.

## 5. What happens to the lane-(b) work that already shipped

The protocol-lift / typed-effect-predicate / gymv-executor /
action-walk-replay work landed on the assumption of lane (b) but is
**not wasted under lane (a)** — it lives on as **offline gate-side
and diagnostic** infrastructure. Specifically:

| Surface | Live use under lane (a) | Offline / diagnostic use |
|---|---|---|
| `labeling/_protocol_lift.py` (v2.1, 92.5 % verb taxonomy coverage) | None — actor reads NL `protocol` as guidance | Used by `labeling_supplement/_phase4_transfer_cycle.py` to mine typed effects → drive `FewShotAdapter` cross-task verification |
| `harness/gymv_executor.py` + `gymv_success.py` | Not invoked by `EpisodeRunner` | Used by `_phase2_real_env_skill_smoke.py` and the offline `GateRunner` (Stage 1 replay, Stage 2 shadow) for skill-quality diagnostics |
| `harness/replay_validator.py` (action-walk mode) | Not invoked by trainer | Used by `decide_promotion_gpt54.py` Stage 1 to gate promotion |
| `harness/few_shot_adapter.py` (intra-source-domain task transfer) | Not invoked by trainer | Used by Stage 3a (G3a) cross-task probes; produces `verified_tasks` evidence the actor's eligibility filter then consults |
| `data_structure/extensions/skill_episode.py` typed-evidence fields (`evidence_in / evidence_out`, `protocol_index`, citation slots) | Populated by the *offline* gate's stub adapter (Day-7d typed-hop awareness); the live actor populates them when it consults a skill (one-step `SkillEpisode`) | Used by gate scoring + Crafter's `BANK_GAP` / `RETRIEVAL_MISLEAD` analysis |
| `harness/gate_runner.py` (`GateRunner`, `EvalSuite`, `GateRunnerConfig`) | Not invoked by trainer | Used by `decide_promotion_gpt54.py` and `dump_harness_io_gpt54.py --gate-runner` for offline promotion decisions |
| `orchestrator/promotion_orchestrator.py` (atomic promote / rollback, `RunRelease` manifests) | **Used live** — the trainer's `_promotion_hook.py` consumes it on its medium-timescale tick | unchanged |

**Reframing rule of thumb:** typed protocols and effect predicates
are *evidence the gate uses to decide whether a retrieval payload
deserves promotion*. They are **not** the actor's runtime substrate.

## 6. Rollback condition — when to revisit lane (b)

Revisit only if **both** of the following hold after S2 fast-loop
launch:

1. **Retrieval-quality ceiling is hit.** The actor's
   `skill_selection` LoRA + retrieval scoring saturates (i.e. the
   actor picks the right skill ≥ 95 % of the time when one is
   available) **and** Joint Success Rate (NORTHSTAR §7.3) is still
   below the headline target.
2. **Retrieval-quality work has been exhausted.** The escalation
   order in
   [`single-vs-two-mdp-tradeoff.md` §"Escalation order"](single-vs-two-mdp-tradeoff.md)
   has been fully tried: better `strategic_description` quality,
   pattern-tag abstraction layer on `Skill`, and two-call inference
   (predict-pattern → act, both inside one MDP).

If both fail, the next escalation is *not* lane (b) directly — it's
**MCTS with a forward model on games** and **tool-augmented harness
ops on VR / video**. Lane (b) is the last resort, not the first.

## 7. Action items closed by this decision

* **T1.3 (`pre-training-readiness-audit.md` §0.1)** — closed.
* **T3.6 (single-MDP plan-doc cleanup)** — strengthened: the lane decision and the MDP decision both reinforce that `hop_select` is a non-target.
* **`crafter-harness-orchestrator-roles.md` §7.5 "Until a lane is picked"** — superseded; the lane is picked.

## 8. Action items opened by this decision

| ID | Item | Sprint |
|---|---|---|
| T1.3a | Add `enable_protocol_patching: bool = False` flag on `SkillCrafterService`; default False; `_crafter_hook` passes False | S0 |
| T1.3b | Add `RewriteProposal` (and rename / repurpose `ComposeProposal` → `MergeProposal` if cleaner) in `data_structure/extensions/bank_mutation.py` | S2 |
| T1.3c | Implement `BANK_GAP` / `RETRIEVAL_MISLEAD` / `STALE_DESCRIPTION` `FailureClass` taxonomy in the live (not offline-mirror) Crafter path | S2 |
| T1.3d | Replace the multi-domain `ACTIVE` invariant with `min_retrievals_per_skill` in `PromotionOrchestrator` | S2 |
| T1.3e | Update [`harness/README.md`](../../harness/README.md) §22 + [`crafter-harness-orchestrator-roles.md`](crafter-harness-orchestrator-roles.md) §7.3 to mark the lane as decided | S0 (doc-only, alongside T2.6 + T3.6) |
| T1.3f | Update plan documents (PLAN-SKILL-CRAFTER, PLAN-SKILL-BANK, PLAN-HARNESS, PLAN-COMPONENTS-IMPLEMENTATION) to reflect the lane-(a) verdict (the four PLAN docs were written for lane (b)) | S0 (doc-only, alongside T2.6 + T3.6) |

## 9. Headline

**A skill is a retrieval payload.** The actor LLM consults retrieved
skills as prompt context; `skill_selection` LoRA picks among them;
`action_taking` LoRA emits the env action. The harness is an
**eligibility filter and validator**, not an executor. The Crafter
mints, rewrites, merges, and retires retrieval payloads — the
Repairer is parked behind a feature flag.

Typed protocols, effect predicates, replay validators, and
transfer-target executors **stay in tree** as offline gate /
diagnostic infrastructure. They are the gate's evidence base, not
the actor's runtime substrate.

This decision is consistent with one MDP, two GRPO LoRAs, the
shipped cold-start data shape, the Day-10 trainer ↔ harness wire-up,
and the on-disk SFT inventory. It is a description of what the live
system already is — formalised so future Crafter additions stop
re-litigating the lane choice.
