# Skill Crafter / Harness / Orchestrator — roles, I/O contracts, and the offline `labeling_supplement` mirror

> **Status:** design captured. Live Crafter ships the two-tier trigger
> in [`crafter/service.py`](../../crafter/service.py)
> (`reflect_on_episode` per-episode reactive pass + `cycle` per-batch
> reflective pass) plus a frozen [`BankView`](../../crafter/_bank_view.py)
> snapshot for the wider read-scope. The Phase-1 rule-based mirror
> ships offline in
> [`labeling_supplement/decide_skill_crafting_gpt54.py`](../../labeling_supplement/decide_skill_crafting_gpt54.py).
> Harness `GateRunner` mirror and Orchestrator `PromotionOrchestrator`
> mirror are the next two scripts in the same folder; their I/O
> contracts are fixed by this note so they can be implemented
> independently.
> **Last reviewed:** 2026-04-30.
> **Cross-refs:**
> [`plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md`](../../plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md),
> [`plans/05-harness/PLAN-HARNESS.md`](../../plans/05-harness/PLAN-HARNESS.md),
> [`plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`](../../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md),
> [`plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md),
> [`crafter/README.md`](../../crafter/README.md) (live code overview),
> [`labeling/readme.md`](../../labeling/readme.md) (offline labelers it mirrors).

This memo records the design discussion behind splitting Skill Crafter,
Skill Harness, and Pipeline Orchestrator into three separate roles, why
the boundary between them is non-negotiable, and how each role is being
mirrored as a deterministic offline `labeling_supplement/*_gpt54.py`
script for cold-start production.

The goal is that the next person to open
`labeling_supplement/decide_skill_crafting_gpt54.py` and ask "what does
this script *not* do?" gets a precise answer without having to read four
plan documents.

---

## 1. Why one role is not enough

Three plan documents land in this repo (`04-skill-crafter`, `05-harness`,
`06-orchestrator`) plus a shared lifecycle gate spec (`07-skill-gate`).
On a first read the three plans look like they overlap — the Crafter
"validates" skills, the Harness "verifies" skills, the Orchestrator
"promotes" skills. They are not the same operation:

* The **Crafter** is *generative*. It owns *content production*: pick
  good skills out of a frozen teacher, hypothesize new ones, propose
  retiring stale ones. It never moves bank pointers.
* The **Harness** is *runtime*. Online it's the per-decision filter and
  veto layer; offline (the surface this note focuses on) it's the
  `GateRunner` that takes a Crafter proposal + a bank snapshot and
  emits per-stage `GateVerdictPayload`s. It never overwrites a skill's
  `status` or `verified_domains`.
* The **Orchestrator** is *transactional*. It is the only component
  allowed to write to a bank snapshot, mutate `status` /
  `verified_domains`, append `AuditRecord`s, or roll back.

Conflating any two of these breaks one of the four invariants below.

### The four-way ownership boundary (PLAN-PIPELINE-ORCHESTRATOR §2)

| Component       | Owns                                                                  | Does NOT own |
|-----------------|-----------------------------------------------------------------------|--------------|
| **Actor**       | next-token decoding from `(state, intention, retrieved_skills)`       | bank state, gate verdicts, evidence storage, promotion |
| **Harness**     | retrieval scoring, binding, validation, transfer probes, gate stages  | final action selection (Actor), bank state mutation (Orchestrator), per-batch promotion verdicts (Orchestrator) |
| **Skill Bank**  | persistence + retrieval of `SkillRecord`s; snapshot ids                | invocation, scoring, status mutation policy |
| **Orchestrator**| run-graph DAG; promotion / rollback transactions; bank-snapshot moves; `AuditRecord`s | running the gate stages itself (asks Harness); generating proposals (asks Crafter) |

The Crafter sits **outside this four-way table** as a content producer.
It writes to the `draft_store` (its own staging area), and that's all.
Promotion of a draft into the bank is the Orchestrator's job, gated by
the Harness.

This separation is what makes the pipeline *auditable*: every bank
mutation can be traced back to a Crafter proposal, a Harness gate
verdict roll-up, and an Orchestrator audit record — three independent
artefacts produced by three independent components.

---

## 2. Per-role scope

### 2.1 Skill Crafter (PLAN §1, §2.5, §6)

A frozen-teacher (32B/72B) **synthesis-reflection agent** running in
four modes:

| Mode | Trigger | Output |
|---|---|---|
| **Composer** | adjacent skills co-occur often → chain them | `ComposeProposal` |
| **Generalizer** | mature single-domain skill → propose K-shot transfer | `GeneralizeProposal` |
| **Hypothesizer** | gap in skill bank for a phenomenon seen in failures | `HypothesisProposal` (new draft) / `PatchProposal{protocol}` |
| **Failure-Reflector** | one or more failure traces on an existing skill | `RetireProposal` / `PatchProposal{precondition\|warrant-strengthen}` |

#### Two-tier trigger model

The Crafter has *two* invocation cadences. Each Crafter mode is wired
to one of them:

| Pass | Live entry point | Threshold | Modes that fire | Latency expectation |
|---|---|---|---|---|
| **Per-episode reactive** | [`SkillCrafterService.reflect_on_episode(EpisodeReflection)`](../../crafter/service.py) | `min_count=1` (one signal is enough) | Failure-Reflector, per-episode Hypothesizer fall-through, **subsumption-retire** | one teacher call worst case; ms with rule path |
| **Per-batch reflective** | [`SkillCrafterService.cycle(new_failures=...)`](../../crafter/service.py) | `hot_pattern_threshold` (default 3) | Composer, Generalizer, statistical retires/patches | scheduled by the orchestrator every K episodes |

The split exists because some Crafter modes have very different
statistical demands. Failure-Reflector and Hypothesizer can act on a
single trace; Composer needs adjacent-skill co-occurrence; Generalizer
needs multi-episode `n_instances` and `pass_rate`. Mixing them on one
trigger either floods the gate stack with noisy single-episode patches
(if cadence is per-episode) or leaves single-episode regressions
unfixed for the next batch (if cadence is per-batch). The two-tier
split fixes both ends.

Both surfaces share the *same* per-pattern dispatch chain
(`repair > retire > hypothesise`, PLAN-SKILL-CRAFTER §6.5) — only the
threshold and the post-dispatch surfaces (subsumption check, batch
stats) differ. This is mechanically enforced in
[`crafter/service.py::_run_failure_dispatch`](../../crafter/service.py).

#### Read scope — wider than active-only

The per-episode pass needs to see *more* than the active store: it
reads candidate skills the Bank Agent just minted and recent
bank-mgmt actions for dedup. Live code exposes this through the frozen
[`BankView`](../../crafter/_bank_view.py) snapshot built at the start of
every `reflect_on_episode` / `cycle` call:

| Store | Surface in `BankView` | Used by |
|---|---|---|
| Active | `BankView.actives` / `actives_iter()` | repair / retire base lookup, subsumption "did the candidate replace this active?" |
| Candidate | `BankView.candidates` / `candidates_iter()` | subsumption pair detection (`subsumed_pairs(candidate_ids=...)`) |
| Draft | `BankView.drafts` / `drafts_iter()` | dedup against the Crafter's own outstanding proposals |

`BankView` is a `frozen=True` dataclass — proposers receive it as a
parameter and cannot mutate it. The bank-store rule remains intact:
`SkillCrafterService` is the only crafter component that may build a
view (`_take_bank_view`); `Composer`, `Generalizer`, `Hypothesizer`,
`Repairer` accept a view as input and never re-fetch.

#### Subsumption-retire (per-episode only)

When the Skill Bank Agent emits a candidate skill whose
`parent_skill_ids` includes an active skill *and* whose contract
strictly covers the active's (`effects_add`, `effects_del`,
`expected_evidence_roles`, `success_criteria`), the Crafter
automatically emits a `RetireProposal{reason: "subsumed_by=<cand_id> ..."}`.
The check is conservative on purpose — false positives cost one
gate-rejected `RetireProposal`, and the active gets to keep running
until G3/G5 confirm it really is dominated. See
[`crafter/_bank_view.py::_subsumes`](../../crafter/_bank_view.py).

#### Early-training noise filters: coalesce + cooldown

The per-episode reactive pass uses `min_count=1` (any single failure
fires Repair) — that is the *whole point* of the per-episode tier, but
in early training it produces severe write amplification. Empirically,
the offline mirror under [`labeling_supplement/reflect_per_episode_gpt54.py`](../../labeling_supplement/reflect_per_episode_gpt54.py)
showed an **11.3× duplicate-mint factor** across 340 episodes
(260 PatchProposals collapsing to only 23 unique
`(base_skill_id, recovery_strategy)` tuples). To keep the gate stack,
the artifact store, and the audit trail bounded in early training,
[`SkillCrafterService`](../../crafter/service.py) now applies two
deterministic filters to the failure-driven dispatch chain. Both are
on by default; both are observable on every `CrafterCycleResult`
(`n_patches_coalesced`, `n_patches_skipped_cooldown`).

| Filter | When it fires | Effect on the proposal | Cost |
|---|---|---|---|
| **Coalesce** | a still-DRAFT `PatchProposal` already covers `(base_skill_id, recovery_strategy)` | the new `failure_id`s are appended to that proposal's `seed_failure_ids`, the artifact-store JSON is overwritten in place, and an `audit{kind: patch_coalesced}` event is recorded | one in-memory dict lookup + one `put_proposal` overwrite — strictly cheaper than running the Repairer + ingesting a fresh DRAFT |
| **Cooldown** | no coalescable open patch *and* the same base was patched within the last `cooldown_passes` (default **5**) Crafter passes | the mint is skipped; the failure still landed in `FailureMemory.add` (so the per-batch `cycle()` can still pick it up later) and an `audit{kind: patch_skipped_cooldown}` event is recorded | one in-memory dict lookup; no Repairer call |

Ordering is fixed in `_run_failure_dispatch`: **coalesce always wins
over cooldown**. Appending evidence to an in-flight DRAFT is free *and*
is what the gate wants to see (richer `seed_failure_ids` = better
diagnosis quality at the gate). Cooldown gates only the *minting* of
parallel proposals — never the strengthening of an existing one.

A draft skill leaves DRAFT only via the orchestrator (gate decision:
promote → CANDIDATE / archive → ARCHIVE). The Crafter has no
notification channel for that, so `_lookup_open_patch` re-checks
`lifecycle.get(draft_skill_id).status` on every coalesce attempt and
lazily evicts the cache entry the moment the draft moves on. The next
failure on the same `(base, strategy)` then mints a fresh proposal —
this is correct: the gate has formed an opinion on the previous patch
and we should not be poisoning a closed proposal with new evidence.

The pass counter that drives cooldown is advanced **only on
productive Crafter passes** — `reflect_on_episode` short-circuits with
no signal does *not* increment it. Otherwise a few quiet episodes
after a patch would silently re-open the mint window without anyone
making a deliberate decision.

Verified end-to-end on the cold-start corpus
(`labeling_supplement/episode_reflections_out/run_<ts>/_run_summary.json`):

| Metric | Pre-fix | Post-fix |
|---|---|---|
| `n_proposals` (minted) | 260 | 23 |
| `n_patches_coalesced` | n/a | 237 |
| `n_patches_skipped_cooldown` | n/a | 0 (offline mirror has no gate, so DRAFTs never leave DRAFT and coalesce hits 100% — cooldown is exercised by `tests/test_crafter_coalesce_cooldown.py::TestCooldown` instead) |
| effective dedup factor | 11.3× | **1.0×** |
| unique `(base_skill_id, recovery_strategy)` | 23 | 23 |

#### Proposal payload (unchanged)

The Crafter never executes a skill. It writes typed
`BankMutationProposal` records (PLAN-SKILL-CRAFTER §2.5) into a
`draft_store`. Each proposal carries:

* `proposer` ∈ `{composer, generalizer, hypothesizer, reflector}`
* `evidence_role`, `evidence_interface` (mirrored from the source skill)
* `target_domains` — **must cover all five** (gymv, browser, osworld,
  video, visual_reasoning)
* `adapter_plan` — per-target binding strategy stub
* `replay_slice_ids` — segments the gate stages can replay
* free-form `rationale` — for human review and gate dashboard summaries

Phase-1 ships **rule-based + frozen-teacher only** — no trainable
update of the Crafter's parameters. The teacher is called only to draft
better names / rationales; the *decision* of which proposal to emit is
made by deterministic rules so the cold-start corpus is reproducible.

### 2.2 Skill Harness (PLAN-HARNESS §1, §3, §6)

The Harness is structurally one component with two surfaces:

**Online surface (per actor decision)**

```
Inputs  : (schema_state, intention, retrieved_skills, active_skill,
           local_reasoning_trace)
Outputs : (eligible_skills with scores, invocation_veto/permit, SkillEpisode)
```

Used inside `ActorAgent.step` to filter / score `retrieved_skills`
against the current state and (optionally) veto an invocation that
would violate `evidence_role` invariants. Fast, frozen, **never picks
the next action** — the Actor still chooses, the Harness just hands
over candidates with reasons.

**Offline surface (per Crafter proposal)**

```
Inputs  : (proposal: BankMutationProposal, bank_snapshot_id, replay_slice_ids,
           shadow_traces?, batch_metrics?)
Outputs : list[GateVerdictPayload]    # one per gate stage
          + roll-up SkillEvaluationRecord
```

This is the `GateRunner` per PLAN-SKILL-GATE / PLAN-HARNESS §6:

| Stage | Predicate (rough) | Diagnostic labels |
|---|---|---|
| **G0 evidence-driven** | does the proposal declare a non-empty `evidence_interface` consistent with `evidence_role`? | `opaque_skill_violation`, `evidence_starved` |
| **G1 binding** | does the protocol bind cleanly against the current schema (no missing slots)? | `binding_failed_<slot>` |
| **G2 adapter** | per-target adapter present & non-trivial? | `missing_adapter_<domain>` |
| **G3 replay** | running the proposed protocol on `replay_slice_ids` reproduces the observed evidence? | `replay_regression`, `replay_underperforms` |
| **G4 shadow** | shadow execution against current actor matches gate's expectation? | `shadow_disagreement` |
| **G5 non-regression** | promoting this proposal doesn't drop batch metrics on already-active skills? | `non_regression_failed_<metric>` |

The Harness emits *verdicts*. It does not promote, it does not retire,
it does not write `verified_domains`. Those mutations are exclusively
the Orchestrator's.

### 2.3 Pipeline Orchestrator (PLAN-PIPELINE-ORCHESTRATOR §2, §3a, §4)

Top-level loop closer. Two surfaces too:

**Online control plane** — wires
```
skill_bank.retrieve(state) -> harness.filter_and_score(...) ->
actor.step(...) -> harness.validate_invocation(...) -> (action, reward)
```
per outer-step, plus episode-local evidence/trace bookkeeping.

**Offline `PromotionOrchestrator`** — consumes the Harness's
`SkillEvaluationRecord` for each Crafter proposal and decides:

```
PROMOTE      | REJECT      | DEFER (more replay)  | ROLLBACK
```

It is the *only* component that:

* moves a skill between `draft_store → candidate_store → active_store`;
* mutates `status` / `applicable_domains` / `verified_domains`;
* takes a new `bank_snapshot_id`;
* writes `AuditRecord{who, when, rationale, linked_snapshot_ids}`.

If a promotion later regresses on actor batch metrics, the Orchestrator
is also the only component allowed to flip a snapshot pointer back
(`ROLLBACK`).

---

## 3. The I/O contract table

Synthesized from PLAN-04 / 05 / 06 / 07 + the on-disk `skill_actions_out`
and `skill_bank_out` shapes. This is the single source of truth for the
`labeling_supplement/` mirror scripts.

| Component / surface | Inputs (canonical names) | Outputs |
|---|---|---|
| **Crafter — per-episode reactive** ([`reflect_on_episode`](../../crafter/service.py)) | [`EpisodeReflection`](../../data_structure/extensions/episode_reflection.py) carrying `failure_traces`, `skill_episodes`, `new_candidate_skill_ids`, `bank_agent_actions`, `outcome_summary` + a frozen `BankView` snapshot (built by the service) | `PatchProposal` / `RetireProposal` / `HypothesisProposal` written to `draft_store` (plus subsumption `RetireProposal`s) |
| **Crafter — per-batch reflective** ([`cycle`](../../crafter/service.py)) | accumulated `FailureTrace[]` over the last K episodes, current bank state via `BankView` | same proposal taxonomy + Composer / Generalizer outputs |
| **Crafter** (offline mirror) | `bank_snapshot @ snapshot_id`, `skill_actions_run` (usage stats), recent `bank_management_io.json` updates | `PatchProposal \| ComposeProposal \| GeneralizeProposal \| RetireProposal` written to `draft_store` |
| **Harness — online filter** | `(schema_state, intention, retrieved_skills, active_skill, local_trace)` | `eligible_skills (ranked)`, `invocation_veto/permit`, `SkillEpisode` |
| **Harness — `GateRunner` (offline)** | `proposal`, `bank_snapshot_id`, stage-specific args (`replay_slice_ids`, `shadow_traces`, `batch_metrics`) | per-stage `GateVerdictPayload{stage, verdict, metrics, diagnostic_labels}` + roll-up `SkillEvaluationRecord{final_decision, approved_domains, rejected_domains, bank_snapshot_id}` |
| **Orchestrator — online** | actor messages + harness verdicts each step | invocation log, episode artefacts |
| **Orchestrator — `PromotionOrchestrator`** | `SkillEvaluationRecord`, current `bank_snapshot_id`, batch-level actor metrics | `final_decision ∈ {PROMOTE, REJECT, DEFER, ROLLBACK}`, mutated bank state, new `bank_snapshot_id`, `AuditRecord` |

### "Who writes what" cheat sheet

| Artefact | Written by |
|---|---|
| `BankMutationProposal` (`Patch / Compose / Transfer / Retire`) | **Crafter** |
| `SkillEpisode` (per-skill execution record) | **Harness (online)** |
| `GateVerdictPayload` (per stage) | **Harness (`GateRunner`)** |
| `SkillEvaluationRecord` (roll-up) | **Harness (`GateRunner`)**; *consumed* by Orchestrator |
| `bank_snapshot_id` updates / `status` / `verified_domains` | **Orchestrator (`PromotionOrchestrator`)** |
| `AuditRecord` (rationale, who/when, linked snapshots) | **Orchestrator** |
| `SkillRecord` row content | originally Crafter (proposal); finalized + version-bumped by Orchestrator on promotion |

---

## 4. Why mirror these into `labeling_supplement/`?

The repo already has a successful pattern:
[`labeling/`](../../labeling/) — frozen-teacher, deterministic, parallelised
offline labelers (`label_intentions_gpt54.py`,
`extract_skillbank_gpt54.py`, `extract_skillbank_gymv_gpt54.py`,
`label_skill_actions_gpt54.py`) that produce reproducible cold-start
SFT data without ever running the live Actor or live Harness.

For the Crafter / Harness-gate / Orchestrator-promotion loop we want
the **same property**: deterministic, parallel, replay-only, no live
env, so we can build / inspect / unit-test the lifecycle end-to-end on
the existing `skill_bank_out/run_<ts>` and `skill_actions_out/run_<ts>`
artefacts before any of these components gets wired into the live
agent.

That's what `labeling_supplement/` is. Its naming, defaults, and
dispatcher pattern follow `labeling/` 1-to-1. There will eventually be
three sibling drivers under it, mirroring the three roles:

| Sibling driver | Role mirrored | Status |
|---|---|---|
| `decide_skill_crafting_gpt54.py` (+ `run_decide_skill_crafting.sh`) | Crafter Phase-1 | **shipped** |
| `run_harness_gates_gpt54.py` (or `decide_gate_verdicts_gpt54.py`) | Harness `GateRunner` (G0–G5) | **pending** |
| `decide_promotion_gpt54.py` | Orchestrator `PromotionOrchestrator` | **pending** |

The contract between siblings is the same one the live components will
use: each driver consumes a previous run's `out/` directory (bank
snapshot → proposals → gate verdicts → promotion decisions). On-disk
JSONL is the only shared API. No driver imports another driver's code.

---

## 5. Crafter Phase-1 — `decide_skill_crafting_gpt54.py`

The shipped script is a faithful implementation of PLAN-SKILL-CRAFTER's
Phase-1 ("rule-based + frozen teacher"). It takes a `bank_run` snapshot
and (optionally) a `skill_actions_run` snapshot, and applies six
deterministic decision rules. Every decision is recorded with the
numeric inputs that triggered it, so a future Harness `GateRunner` can
re-run the same proposals and a human reviewer can trace any verdict.

### 5.1 Decision rules

| Rule | Predicate | Proposal | Proposer |
|---|---|---|---|
| **R1 — retire (evidence-starved)** | `(eff_add+eff_del+eff_event)==0 ∧ n_inst < min_inst_for_keep`, OR `usage_pct < retire_usage_pct_min` | `RetireProposal{retire_reason: "evidence-starved"}` | `reflector` |
| **R2 — patch (warrant-strengthen)** | `evidence_role==COMMIT ∧ eff_add+eff_event==0 ∧ n_inst ≥ min_inst_for_keep` | `PatchProposal{patch_kind: "warrant-strengthen"}` | `reflector` |
| **R3 — patch (precondition tighten)** | `pass_rate==1.0 ∧ mean_applicability` saturated near `0.5` over `min_usage_for_signal` selections | `PatchProposal{patch_kind: "precondition"}` | `reflector` |
| **R3b — patch (protocol)** | `protocol.steps < protocol_min_steps ∧ n_inst ≥ 2·min_inst_for_keep` | `PatchProposal{patch_kind: "protocol"}` | `hypothesizer` |
| **R4 — compose (sequence)** | `co_occurrence(A→B) ≥ compose_threshold · n_transitions`, neither retired | `ComposeProposal{compose_op: "sequence"}` | `composer` |
| **R5 — transfer** | `len(applicable_domains)==1 ∧ n_inst ≥ transfer_min_instances ∧ pass_rate ≥ transfer_min_pass_rate ∧ verified_domains==∅` | `TransferProposal` | `generalizer` |

### 5.2 Why these thresholds, why these rules

* **Phase-1 thresholds are conservative on purpose.** PLAN §10 sets a
  high-precision target so the downstream Harness gate stack is not
  flooded with junk proposals. False-positives on the Crafter side are
  expensive (they become G0/G3 replays); false-negatives are recoverable
  (a skill that should have been retired this batch will be retired the
  next batch when its `usage_pct` keeps dropping).
* **R1 has two branches** (bank-only `empty-contract+low-support`,
  usage-based `low-usage-pct`) so the rule fires both at cold-start
  (when no usage stats exist yet) and during steady-state usage.
* **R2 specifically targets COMMIT-role skills with no add/event
  effects.** Per PLAN-SKILL-GATE Gate G0, this is exactly the
  `opaque_skill_violation` shape — a skill that emits a decision
  without producing a measurable warrant. The patch is to require an
  `evidence_warrant` citation; if the Crafter can't synthesize one, the
  next batch will retire the skill.
* **R3 catches degenerate discriminators.** A skill whose mean
  applicability is exactly `0.5` across hundreds of selections is
  almost certainly being matched by relevance only (the applicability
  scorer isn't doing work). Patching the precondition is the cheap fix
  before retiring.
* **R4 (compose) only fires when usage stats are available.** Without
  them we have no transition counts and the rule cannot fire — the
  driver records this in the per-skill decision trace.
* **R5 (transfer) is gated on `verified_domains==∅`.** A skill that
  already has verified domains has been promoted by the Orchestrator
  before; re-proposing transfer would be a waste of the gate stack.

### 5.3 What the driver does NOT do

This is the critical list — every line is intentional:

1. **Never mutates the bank.** Output goes to a fresh
   `crafter_proposals_out/run_<ts>/` directory only. The
   `skill_bank_out/run_<ts>/` snapshot is read-only.
2. **Never runs gate stages.** No replay, no shadow, no binding check.
   It only decides *which proposal to emit*. G0–G5 are the Harness's
   job.
3. **Never sets `verified_domains` or `status`.** Even on
   `RetireProposal` we only mark `retire_reason: "evidence-starved"`;
   the actual `status: retired` mutation is a future Orchestrator's job.
4. **Never invokes a live Actor.** All inputs are frozen JSON files.
   This is what makes it cheap to re-run (full corpus completes in
   ~1 s on a single core).
5. **Never collapses `target_domains`.** Per PLAN-SKILL-CRAFTER §2.5,
   every proposal must enumerate all five target domains so the gate
   stack can verify general feasibility. Even a `RetireProposal`
   carries the full list — the Orchestrator decides whether to honor
   it per-domain.

### 5.4 Real-corpus statistics (sanity check)

Running the dispatcher across all 17 `(corpus, source)` pairs of the
real `skill_bank_out/run_20260430_030637` snapshot, with the 5%
`compose_threshold` and other defaults:

```
17/17 pairs ok, 114 skills in -> 224 proposals out
  by_kind     = {patch: 109, compose: 48, retire: 38, transfer: 29}
  by_proposer = {reflector: 147, composer: 48, generalizer: 29}
```

Distribution looks right for Phase-1: most proposals are patches
(low-cost, mostly shadowable in G3/G4); ~17% are retires (the cold-start
bank is noisy, this is expected); ~13% are transfers (most game banks
have at least one mature single-domain skill). If the share of retires
ever climbs above 30% the Crafter is too aggressive and the
Orchestrator will throttle promotion; this is the dial we tune in
Phase-2.

---

## 6. Open questions for the next two siblings

### 6.1 Harness-side mirror

Open design questions, in rough order of priority:

1. **Replay surface.** The Harness `GateRunner` G3 stage needs to
   "re-execute" a protocol on `replay_slice_ids`. For the offline
   mirror we don't run a live env, so re-execution = re-evaluating the
   skill's `success_criteria` against the segment's recorded
   `summary_state` deltas. This is enough for Phase-1; full live replay
   requires `cold_start/generate_cold_start_actor.py`-style env
   re-instantiation, which is Phase-2.
2. **Shadow trace input.** G4 requires a "shadow" run of the actor —
   for the offline mirror we substitute the rollout's recorded actions.
   PLAN-HARNESS §3a calls this `shadow_eq_recorded`. Document and
   commit to this approximation in the docstring.
3. **Per-stage early-exit.** A G0 fail blocks G1–G5 (no point binding
   an opaque skill). The mirror should support `--stop-at <stage>` for
   debugging.
4. **Output naming.** `gate_verdicts_out/run_<ts>/<corpus>/<source>/`
   matching the existing `out/run_<ts>` convention.

### 6.2 Orchestrator-side mirror

1. **Snapshot model.** The mirror does not need a real DB; a
   `bank_snapshots/<id>/` directory with copy-on-write semantics is
   enough for offline cold-start. `bank_snapshot_id` is just the
   directory name.
2. **Batch metrics input.** PLAN-PIPELINE-ORCHESTRATOR §4 `ROLLBACK`
   needs `actor_batch_metrics`. For the offline mirror these come from
   per-game `_run_summary.json` files in `skill_actions_out/`. No live
   eval is needed for the cold-start path.
3. **Audit-record format.** `AuditRecord` should be JSONL, one record
   per promotion / rejection / rollback, with the linked
   `SkillEvaluationRecord` ids. This is the file the gate dashboard
   reads.
4. **`DEFER` semantics.** A `DEFER` decision must produce a follow-on
   instruction for the next Crafter run (e.g. "regenerate proposal
   with more replay slices"). Encode this as a back-edge from the
   Orchestrator output into the Crafter input — `defer_followups.jsonl`
   feeding `--defer-followups` on the next Crafter dispatcher run.

### 6.3 Cross-cutting

* **No cycle in offline mirror.** Online the loop is closed: actor →
  harness → orchestrator → bank → crafter → … Offline we deliberately
  break the cycle: each driver runs exactly once per cold-start
  iteration. The "loop" is implemented by re-running drivers in
  sequence with new run-ts ids.
* **Reproducibility contract.** Every driver must write
  `_run_meta.json` with full argv + thresholds + input run paths. This
  is what makes "re-run with the same inputs and check the same outputs
  fall out" mechanical.
* **Diagnostic-label vocabulary.** PLAN-HARNESS §10a enumerates the
  agreed-upon `diagnostic_labels`. The Crafter's `decision_trace.json`
  already uses a subset (`opaque_skill_violation` for R2,
  `evidence_starved` for R1); the Harness mirror must use the same
  vocabulary verbatim so the dashboard groups them correctly.

---

## 7. Gap analysis: where the Crafter doesn't yet fit our setting

The two-tier trigger + coalesce + cooldown work landed because the
Crafter's *plumbing* was correct but its *operational regime* was noisy.
Stepping back further: the Crafter is built for the steady-state
(actor → harness → SkillEpisode → Crafter → gate → orchestrator)
multi-domain pipeline that the four PLAN documents converge on. We are
not in that regime today. This section captures, on the record, the
five mismatches between Crafter assumptions and the actual data plane,
and offers a binary lane decision the next contributor needs to make
before adding more Crafter logic.

### 7.1 Five mismatches between Crafter assumptions and current state

| # | Crafter assumes | Current reality | Evidence in this repo |
|---|---|---|---|
| 1 | `EpisodeReflection.skill_episodes` is populated by a live Harness emitting one record per skill invocation | No live Harness wired (`Actor rewire` still pending in [`IMPLEMENTATION-STATUS.md`](../../IMPLEMENTATION-STATUS.md) §"Not yet delivered"). Cold-start data has zero `SkillEpisode`s | The offline mirror under [`labeling_supplement/reflect_per_episode_gpt54.py`](../../labeling_supplement/reflect_per_episode_gpt54.py) passes `skill_episodes=[]` and **synthesizes** `FailureTrace`s from per-step heuristics (`OUTCOME_FAILURE`, `EMPTY_QUERY`, `LOW_APPLICABILITY`, `MISSING_EFFECTS`). Failures aren't *observed*, they're *synthesized post-hoc* |
| 2 | The actor invokes a skill — its protocol drives a span of action choices, and a "skill failure" means the protocol's claimed effects didn't materialize | The actor *consults* skills as retrieval context (`decision_agents.skill_interface.SkillBankProvider`); it never delegates control flow to a skill protocol. The actor still picks every action itself | A failed step labelled with a selected skill is at best correlated with that skill, not caused by it. The Repairer's protocol-patch logic is therefore patching a skill that had no agency over the failure |
| 3 | Skills carry typed protocols (ordered list of `{action, payload}` hops) and structured contracts (effects_add/del, expected_evidence_roles) | Cold-start "skills" are intention-axis labels (`COMMIT/ATTACK`, `RECOVER/EVADE`) with NL-string protocols emitted by the teacher | The mirror has [`_wrap_protocol_steps`](../../labeling_supplement/reflect_per_episode_gpt54.py) wrapping NL strings as `{"action": "EXEC", "notes": <string>}` just to keep `Repairer._rule_repair`'s `[dict(h) for h in base.protocol]` from crashing. The wrap is a band-aid; an inserted `VERIFY` hop next to NL transcript steps is semantically incoherent |
| 4 | Skills declare `feasible_domains ⊆ {gymv, browser, osworld, video, visual_reasoning}` with ≥2 for ACTIVE; Generalizer / Composer / subsumption-retire all rely on multi-domain reach | Cold-start bank ships every skill with `feasible_domains=["gymv"]` (single domain) | On the real cold-start corpus (340 episodes, 17 sources): `n_subsumption_retires=0`; `Composer` / `Generalizer` are not wired into the per-episode path and the per-batch path has no cross-domain co-occurrence to chew on. All 23 fresh `PatchProposal`s in the post-fix run were `recovery_strategy=hop_insertion` — the exact same canned edit, applied N times across N skills |
| 5 | The diagnoser → repairer chain extracts new information from a failure (root cause, recommended strategy, concrete edit) | With `_llm=None` defaults (the Phase-1 baseline), the chain is rule-based and deterministic | The closed loop is `synthesizer-label → diagnoser-rule → repairer-rule`. No new analysis enters: my `LOW_APPLICABILITY` heuristic mechanically becomes a `PRECONDITION_STRENGTHENING` patch with `contract.preconditions += ["preconditions_strengthened"]`. The Crafter is currently a relabeling function over the synthesizer's output |

The Crafter is not broken. It is **operating downstream of an empty
data plane**, with synthesized inputs and rule-based proposers, on a
single-domain cold-start bank whose atomic unit doesn't shape-match
its protocol expectations.

### 7.2 What's working — preserve through any redesign

Some of the work landed in this slice is independent of the lane
decision below and pays back in any future state:

* **Two-tier trigger model + coalesce + cooldown** (§"Two-tier trigger
  model"). The 11.3× → 1.0× collapse is real and the noise filters
  scale to the steady state too — they're not cold-start patches,
  they're permanent gates against per-episode write amplification.
* **Lifecycle invariants** — bank-write isolation, ≥2 domains for
  ACTIVE, status-mutation-only-via-lifecycle (PLAN-SKILL-BANK §0.4 +
  `IMPLEMENTATION-STATUS.md` §"Invariants enforced"). These are the
  guards the gate stack rests on; do not relax them under any lane.
* **Architectural invariants** — Crafter never imports
  `skill_bank.stores`, only `SkillCrafterService` may build a
  `BankView`. These prevent a future contributor from accidentally
  growing a back-channel into the active store.
* **Audit trail / artifact-store / proposal schema** — `put_proposal`,
  `append_audit`, the typed `BankMutationProposal` union. The
  serialization story is solid and reusable.

### 7.3 The lane decision: what *is* a skill?

> **Status: closed (2026-05-01) — lane (a), Context-only skills.**
> Authoritative record: [`skill-lane-decision.md`](skill-lane-decision.md);
> readiness audit: [`pre-training-readiness-audit.md`](../pre-training-readiness-audit.md) §0.4 (T1.3 closed → lane (a)).
> The two-lane comparison below stays in tree as the rationale for the
> decision; downstream work (Crafter modes, audit, gate stack)
> follows the lane-(a) implications listed in §3 of the lane-decision
> doc, not §7.4 below. Lane-(b) machinery remains in tree as offline
> diagnostic tooling — gated by `SkillCrafterService(enable_protocol_patching=…)` (default `False`,
> per T1.3a).

The deepest unresolved question is one of definition. The Crafter
makes sense in two coherent regimes; we are currently in neither.
Pick before adding more Crafter logic.

| | **Lane (a) — Context-only skills** | **Lane (b) — Executable skills** |
|---|---|---|
| **What is a skill?** | A semantic retrieval payload: name, description, preconditions/effects/role labels. Used by the actor for *reasoning context* during action choice | A runnable program: ordered, typed protocol hops dispatched via `harness.skill_adapter`. A skill invocation owns N actor steps until success/abort |
| **Who picks the action?** | Actor, every step, with retrieved skills as context | Skill protocol while invoked; actor only picks when no skill is invoked or the active skill aborts |
| **What does failure mean?** | Bank gap (no skill matched) or retrieval-quality failure (the matched skill misled reasoning) | Skill protocol failure (a hop's claimed effect didn't materialize, or the contract's abort criterion fired) |
| **Crafter modes that retain meaning** | Hypothesizer (mint a skill for a recurring uncovered situation), partial Composer (synthesize a better retrieval payload from co-firing labels), Retire (drop low-utility retrievers) | All five modes (Compose / Generalize / Hypothesize / Patch / Retire) work as designed in PLAN-SKILL-CRAFTER §6.5 |
| **Crafter modes that go dark** | **Repairer is largely dead code.** Patching a non-executable retrieval payload is a no-op — there is no protocol to patch. `RecoveryStrategy.{HOP_INSERTION, PROTOCOL_PATCH, FALLBACK_INJECTION, REGROUNDING_TRIGGER, SKILL_DECOMPOSITION}` lose meaning | None — every mode has a target |
| **What `FailureTrace.skill_id` means** | The retrieved skill that was *consulted* during the failed step. Attribution is correlational; needs a `skill_fault_confidence` to be actionable | The skill that *owned* the failed step. Attribution is causal; the existing `failed_step_index` + `contract_violation` fields are sufficient |
| **What `protocol` looks like** | Optional. Could be NL guidance for the actor or empty | Required. List of typed `{action, payload}` hops dispatched through an adapter |
| **Cold-start fit** | High — current bank ships exactly this shape (NL protocol steps consumed as guidance) | Low — would require a second cold-start pipeline that mints adapter-compatible action sequences, not NL transcripts |
| **Multi-domain story** | Skills generalise via *retrieval embeddings*; Generalizer's role becomes "this skill's retrieval profile matches situations in domain Y" | Skills generalise via *adapter binding*; Generalizer emits few-shot adaptation recipes (the current `GeneralizeProposal.{source_domain, target_domain, slot_remap, demo_*}` fields) |
| **Gate stack changes** | G0 needs to redefine "evidence" for a non-executable skill (replay no longer applicable). G3/G5 need new metrics on retrieval quality | Gate stack runs as designed; replay validation = re-execute the patched protocol against the original episode |

#### Honest read

We are de-facto in lane (a) today (the actor reasons over retrieved
skills) but the Crafter is built for lane (b) (it patches protocols).
The misalignment is what produces the symptoms in §7.1. Forcing the
codebase to lane (b) is the bigger lift but matches the four PLAN
documents' intent. Embracing lane (a) shrinks the Crafter sharply but
matches the cold-start data shape.

### 7.4 Phased path forward, by lane

Both lanes share one strict prerequisite: **wire the Harness →
SkillEpisode pipeline first**. Until rollouts emit live `SkillEpisode`s
(and therefore live `FailureTrace`s with causal attribution), every
Crafter pass is operating on synthesized inputs and the §7.1 closed-
loop determinism applies.

#### Common (do this first regardless of lane)

1. **Actor rewire** — replace `decision_agents.skill_interface.SkillBankProvider`
   with a `HarnessSkillProvider` per `IMPLEMENTATION-STATUS.md`
   §"Not yet delivered". Source: `harness/skill_harness.py` already
   exposes `select_eligible_skills`; the wrapper is small.
2. **Live `SkillEpisode` emission** — the actor (or harness) logs one
   record per skill invocation/consultation, with the skill_id,
   step indices, and outcome. Schema already exists in
   `data_structure/extensions/skill_episode.py`.
3. **`FailureTrace` from real signals** — emit `FailureTrace` from the
   Harness when a skill's contract is violated *during execution*, not
   from offline heuristics. Add `skill_fault_confidence ∈ [0,1]` to
   the trace so the Crafter can filter actor-fault failures out.

After this prereq, both lanes have something real to consume.

#### Lane (a) — context-only skills (smaller Crafter scope)

If we accept that skills are retrieval payloads and not runnable
programs:

1. **Quarantine the Repairer.** Move `crafter/repairer.py` and the
   `PatchProposal` mint path behind a feature flag
   (`SkillCrafterService(enable_protocol_patching=False)` default).
   Cold-start tests stay green; live runs default to the Repairer
   being inert.
2. **Reshape `FailureClass` taxonomy.** The current six recovery
   strategies are protocol-edit operations. Replace with a
   retrieval-centric taxonomy: `BANK_GAP`, `RETRIEVAL_MISLEAD`,
   `STALE_DESCRIPTION`. The Crafter's job becomes "tell the bank
   curator what to mint, retire, or rewrite."
3. **Strengthen the Hypothesizer.** Without protocol patching, the
   Hypothesizer is the primary mode. Wire the LLM hook
   (`set_llm_hypothesizer`) and route through the Phase-F frozen
   teacher (`SkillCrafterService.set_teacher_model`).
4. **Repurpose `Composer`.** From "compose protocols" to "compose
   retrieval payloads": merge two skills with overlapping retrieval
   profiles into a single richer payload. Concrete proposal type:
   `MergeProposal{absorbed_ids, merged_description, merged_tags}`.
5. **Drop the multi-domain ACTIVE invariant for retrieval skills.**
   Replace with a `min_retrievals_per_skill` threshold (a skill that
   never gets retrieved is dead, regardless of domain count).

Effort estimate: 2-3 sessions. Most of the existing Crafter code
becomes dead but stays in tree behind the flag for a possible future
lane-(b) flip.

#### Lane (b) — executable skills (full PLAN intent)

If we commit to the four PLAN documents' vision:

1. **Replace the cold-start protocol shape.** Today's NL transcript
   steps need to become typed action sequences compatible with
   `harness/adapters/{gymv_adapter, browser_adapter}.py`. This
   probably means a new cold-start labeling pass that converts
   `Press fire button to initiate enemy volley` → `{"action":
   "PRESS_KEY", "payload": {"key": "FIRE"}}`. Source: an extension to
   `cold_start_labeling/build_skill_bank_gymv.py`.
2. **Two-domain bootstrap for cold-start.** Every cold-start skill
   ships with at least one source domain and one *stub* target
   domain (e.g. `feasible_domains=["gymv", "_pending_transfer"]`)
   that the few-shot adaptation gate eventually replaces with a real
   binding. Unblocks ACTIVE promotion + subsumption-retire.
3. **Wire the LLM hooks on `FailureDiagnoser`, `Hypothesizer`,
   `Repairer`.** Phase-F frozen teacher routing is partially in
   (`SkillCrafterService.with_qwen3_vl_teacher`,
   `phase_f_teacher_from_env`); the missing piece is provider-side
   routing per `IMPLEMENTATION-STATUS.md` line 100-104. Until those
   hooks are live, the rule path produces the §7.1 closed-loop output.
4. **Activate Composer + Generalizer on the per-batch path.** Both
   need cross-episode statistics from real `SkillEpisode`s; once the
   prereq pipe is wired, expose them as failure-driven dispatchers
   alongside the Repairer (the existing `_run_failure_dispatch`
   already accommodates them, the per-batch entry point just needs to
   call them).
5. **Real failure tests.** Replace the synthesized-failure offline
   mirror tests with rollout-fed integration tests once the live
   Harness is wired. Keep `labeling_supplement/` as the deterministic-
   replay sibling.

Effort estimate: 4-6 sessions, plus a fresh cold-start labeling pass.
The Crafter code stays approximately as-is; the surrounding
infrastructure catches up to it.

### 7.5 Until a lane is picked

> **Superseded by §7.3 box (lane closed → lane (a), 2026-05-01).**
> The "until" stance below was the operating rule from 2026-04-23 to
> 2026-04-30; it landed two-tier-trigger / coalesce / cooldown without
> committing a lane. The lane is now closed: future Crafter work is
> either (i) lane-neutral plumbing or (ii) explicitly lane-(a) work,
> per [`skill-lane-decision.md`](skill-lane-decision.md) §3.

Don't add more Crafter modes or proposers. The current code is the
right shape for lane (b) and the wrong shape for lane (a); growing it
in either direction commits a lane choice by accident. The
two-tier-trigger + coalesce + cooldown work was lane-neutral noise
control and that's why it landed cleanly. Future Crafter additions
should be either (i) lane-neutral plumbing (audit-trail extensions,
new metric surfaces), or (ii) explicit lane-(a) / lane-(b) work behind
a flag with the choice documented here.

#### Post-decision rule (2026-05-01 onward)

* **Lane (a) is the live default.** Crafter additions land *as* lane-(a)
  features unless explicitly justified otherwise. New modes that only
  make sense for protocol-edit work (HOP_INSERTION, PROTOCOL_PATCH,
  FALLBACK_INJECTION, REGROUNDING_TRIGGER, SKILL_DECOMPOSITION) belong
  behind the `enable_protocol_patching` feature flag, which is `False`
  by default in `SkillCrafterService.__init__` and threaded through
  `CoEvolutionConfig.crafter_enable_protocol_patching` /
  `scripts/run_coevolution.py --enable-protocol-patching`.
* **Lane (b) machinery is "offline diagnostics," not "deferred."** The
  Repairer, `RecoveryStrategy.PROTOCOL_PATCH`, the typed protocol
  hop registry, and `harness/skill_adapter.run_skill` are kept in tree
  but only fire from `labeling_supplement/` drivers (which opt
  `enable_protocol_patching=True`). Removing them is *not* on the
  roadmap — they are the rollback target if `skill-lane-decision.md`
  §4 trips.
* **The lane-(a) closed-loop signal flows through the Hypothesizer.**
  When the Repairer is parked, the dispatcher's existing
  `_STATUS_NO_OP` fall-through routes any "known skill, recurring
  failure" pattern to the Hypothesizer (mint a sibling skill rather
  than edit the protocol of the consulted one). This is the only
  Crafter behaviour change that ships as part of T1.3 closure.
* **No new lanes.** If a Crafter mode doesn't fit retrieval-payload
  edits *or* protocol edits, it doesn't fit at all — escalate to the
  lane-decision doc before writing code.

---

## 8. TL;DR

* **Three roles, three responsibilities, three artefact families.** The
  Crafter writes proposals, the Harness writes gate verdicts, the
  Orchestrator writes promotion / audit records. Conflating any two
  breaks the audit trail.
* **The Crafter never touches the bank. The Harness never promotes.
  The Orchestrator never invents skills.** This is the rule the four
  plan documents converge on; the I/O contracts in §3 enforce it.
* **The Crafter has two cadences, not one.** `reflect_on_episode` runs
  every episode-end (`min_count=1`, Failure-Reflector + per-episode
  Hypothesizer + subsumption-retire); `cycle` runs every K episodes
  (`hot_pattern_threshold=3`, Composer / Generalizer / statistical
  retires/patches). Both share `_run_failure_dispatch` so the per-
  pattern decision is identical regardless of cadence — only which
  patterns reach it differs.
* **The Crafter reads the full bank, not just `active_store`.** A
  frozen `BankView` snapshot (active ∪ candidate ∪ draft) is built at
  the start of every pass and handed to proposers as a parameter. This
  enables subsumption-retire (a freshly-minted candidate strictly
  covering an existing active) without violating the no-direct-store
  invariant — only `SkillCrafterService` may build the view.
* **`labeling_supplement/` is the offline mirror** of the live three-
  role loop, modeled on `labeling/`. One driver per role, deterministic,
  parallel, replay-only. The Crafter Phase-1 driver is in; the Harness
  and Orchestrator drivers are next.
* **Phase-1 Crafter is rule-based + frozen teacher.** Six rules
  (R1–R5 + R3b) with conservative thresholds. On the real
  `skill_bank_out/run_20260430_030637` snapshot it produces
  ~2 proposals/skill across 17 sources, dominantly `patch`. Tune
  thresholds via CLI; do not weaken the rule structure.
* **Trap to avoid:** do not let any driver under `labeling_supplement/`
  import another driver's code or write into another driver's output
  directory. The on-disk JSONL is the only shared API. This is what
  makes the mirror replaceable — when the live Harness ships, swap
  the driver, keep the JSONL contract.
* **Open question (§7).** The Crafter is built for the steady-state
  pipeline (live Harness → `SkillEpisode` → typed protocol patching →
  multi-domain ACTIVE), but we are operating on synthesized
  per-step heuristics, a single-domain cold-start bank, and an actor
  that *consults* skills rather than *executing* their protocols.
  Read §7 before adding new Crafter modes — the lane decision (skills
  as retrieval payloads vs. runnable programs) needs to land first.
