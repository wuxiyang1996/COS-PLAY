# Harness usability test — intra-gymv ablation, and why the offline cycle is a wire-up prerequisite

> **Status:** design captured. The test driver
> [`labeling_supplement/dump_harness_io_gpt54.py`](../../labeling_supplement/dump_harness_io_gpt54.py)
> (online + offline surfaces) is the substrate. The
> ablation runner under `skill_transfer_test/` is the *next* item; this
> note pins its scope, its canonical anchors, and the question
> **"can we just wire all modules together instead?"**
> The answer is no — not safely — and the reason is structural
> (`harness/README.md` §17). The test is the offline half of the wire-up,
> not parallel work.
> **Last reviewed:** 2026-04-30.
> **Cross-refs:**
> [`plans/05-harness/PLAN-HARNESS.md`](../../plans/05-harness/PLAN-HARNESS.md) (§20 ablations, §10b GateRunner),
> [`harness/README.md`](../../harness/README.md) (audit §9–§22, wire-up §15–§18, suggested work-order),
> [`implementation_notes/legacy/crafter-harness-orchestrator-roles.md`](crafter-harness-orchestrator-roles.md)
> (§3 I/O contract, §6.1 pending Harness mirror),
> [`labeling_supplement/dump_harness_io_gpt54.py`](../../labeling_supplement/dump_harness_io_gpt54.py)
> (the live driver this test wraps),
> [`common/enums.py`](../../common/enums.py)
> (`SOURCE_DOMAINS = ("gymv",)`, `TRANSFER_TARGET_DOMAINS`).

This memo records the design discussion behind testing the **usability
of the Skill Harness** before wiring it into the live online runtime.
It exists because two questions kept coming back during scoping:

1. *"What does it mean to test 'harness usability' in this repo?"*
2. *"Can we skip the test and wire all modules together directly?"*

Both have canonical answers in `PLAN-HARNESS.md` and `harness/README.md`
that are easy to miss because they are spread across §20 (ablations),
§22 callout (intra-gymv first milestone), and §15–§17 (wire-up audit).
This note consolidates them so the next contributor opening
`skill_transfer_test/` knows what is canonical, what is a derived
choice, and what was deliberately deferred.

---

## 1. What "harness usability test" means here

Per [`PLAN-HARNESS.md` §1a](../../plans/05-harness/PLAN-HARNESS.md#1a-harness-role-as-frozen-72b-runtime-layer)
the Harness is a runtime *narrows + may veto* layer — it never picks
the next action and never mutates the bank. The question
"is the Harness usable?" therefore decomposes into the four research
questions named in
[`PLAN-HARNESS.md` §20.2](../../plans/05-harness/PLAN-HARNESS.md#202-core-evaluation-questions):

| Q | Question | Primary signal |
|---|---|---|
| **Q1** | Does the Harness improve **skill invocation validity**? | invalid invocation rate ↓; slot binding success rate ↑; precondition pass rate ↑; evidence (G0) pass rate ↑ |
| **Q2** | Does the Harness improve **transfer safety**? | shadow→active promotion precision ↑; regression rate after transfer ↓; `opaque_skill_violation` / `evidence_insufficient` rates ↓ on cross-domain slices |
| **Q3** | Does the Harness **reduce harmful or low-value skill execution**? | veto precision / veto recall; avg skill-use cost / latency for unsuccessful invocations ↓; abort rate on bad candidates ↑ before side effects |
| **Q4** | Does the **Actor itself** still improve, or is the frozen validation layer doing all the work? | actor decision quality (top-1 / top-k accuracy on the Harness-eligible set) over training time |

Q4 is load-bearing. If A0 (no Harness) shows the Actor failing to
improve while A4 (full system) succeeds, **and** the Actor's standalone
decision quality on the eligible set is rising, then both the Harness
*and* the Actor are doing real work. If A0 and A4 differ but Actor
decision quality on the eligible set is flat, the system is a
72B-driven policy in disguise
([`PLAN-HARNESS.md` §1a.5](../../plans/05-harness/PLAN-HARNESS.md#1a5-why-the-frozen-72b-harness-should-not-replace-the-actor))
and the architecture story has failed.

The **answer-method** for these four questions is also pinned in §20:
the five ablation cells A0–A4 in
[`§20.3`](../../plans/05-harness/PLAN-HARNESS.md#203-ablation-matrix), the
three reports in
[`§20.6`](../../plans/05-harness/PLAN-HARNESS.md#206-analysis-templates),
and the four slice axes in
[`§20.5`](../../plans/05-harness/PLAN-HARNESS.md#205-dataset-slices). We
inherit them verbatim — this note does **not** redefine cell semantics
or thresholds.

---

## 2. Where to run the test — intra-gymv first, by canon

[`harness/README.md` §22](../../harness/README.md#22-feasible_domains-granularity-collapses-gymv-games-into-a-single-bucket)
makes the case that intra-gymv (game ↔ game inside `gymv`, e.g.
`Airstriker → Strider`, `Columns → Tetris`) is the right *first* place
to exercise transfer:

| Cost axis | gymv → browser/osworld/video/vr | gymv → gymv (cross-game) |
|---|---|---|
| Adapter executors to wire ([`§1`](../../harness/README.md#1-transfer-target-adapters-are-deterministic-stubs)/[`§16.1`](../../harness/README.md#161-adapter-executors-are-stubs-so-run_skill-is-a-black-hole)) | 4 transfer + 1 source = 5 | **1** (`GymvAdapter`) |
| New env bindings | full browser DOM, VM control, frame indexer, MCQ resolver | **none** — `cold_start/generate_cold_start_actor*.py` already drives the envs; expose its `step()` |
| Demo corpus ([`§4`](../../harness/README.md#4-fewshotadapter-runs-but-the-scorer-is-a-placeholder)) | doesn't exist | **already on disk** — every `labeling/skill_actions_out/.../<game>/episode_*.json` is a real rollout with state, action, intention, ground-truth `skill_query.selected_skill_id` |
| Domain-aware `success_fn` ([`§4`](../../harness/README.md#4-fewshotadapter-runs-but-the-scorer-is-a-placeholder)) | per target (DOM diff, screen diff, video QA, …) | **single gymv-shape scorer** keyed on consecutive `schema_canonical` blocks + `cumulative_reward` |
| Slot-binding ontology | cross-modal — `tile → DOM_node`, `direction → click`, … | gymv-internal — abstract verbs over `selectable_entity` / `container_entity` / `direction` |
| Protocol lift ([`§21`](../../harness/README.md#21-cold-start-protocol-is-natural-language-prose-not-typed-hops)) | needed | needed (same lift, but only over gymv-shaped skills) |

Five out of six axes are dramatically smaller; only the protocol lift
is identical work either way and is the hard prerequisite. The
[`§22`](../../harness/README.md#22-feasible_domains-granularity-collapses-gymv-games-into-a-single-bucket)
callout therefore says, verbatim:

> So the order is: first prove `harness.run_skill(COMMIT/MERGE,
> twenty_forty_eight_state)` actually executes (lift + gymv executor);
> then add §22's task axis; only then run cross-game transfer probes;
> only then attempt cross-domain. Intra-gymv is the first place to
> discover real binding-failure patterns on actual data.

The §20 ablation cells × the §22 intra-gymv probe is a single
experiment. We do not invent a parallel framework for it.

---

## 3. Cell × probe matrix — concrete

| Cell | Harness config | What is exercised on the intra-gymv probe |
|---|---|---|
| **A0 — No Harness** | Actor + bank retrieval only, no `select_eligible_skills`, no `validate_invocation`, no `run_skill`. `SkillEpisode` still logged for measurement. | Baseline. Read `skill_query.selected_skill_id` from `labeling/skill_actions_out/.../<game>/episode_*.json` directly. |
| **A1 — Harness-lite** | `EligibilityFilter` (binding + precondition checks; G1 only). No G0 evidence check, no adapter validation, no transfer gating. | `dump_harness_io_gpt54.py --surface online` with G0/G2/transfer disabled. Measures whether **structural slot binding** alone narrows cold-start retrieval. |
| **A2 — Harness-core** | A1 + G0 evidence-role check ([`§5.1`](../../plans/05-harness/PLAN-HARNESS.md#51-skillepisode)) + G2 adapter validation + invocation veto + advisory scoring (`fit_score` / `risk_score` per audit [`§9`](../../harness/README.md#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs)). | The "filter + veto + advise" cell. Defends [`PLAN-HARNESS.md` §1a.5](../../plans/05-harness/PLAN-HARNESS.md#1a5-why-the-frozen-72b-harness-should-not-replace-the-actor) (the trained Actor must still do work). |
| **A3 — Harness-transfer** | A2 + Stage 1 replay (G3) + Stage 2 shadow (G4) + Stage 3a few-shot transfer (G3a) **with the §22 task axis active**. | `dump_harness_io_gpt54.py --surface offline` over the Crafter's `TransferProposal`s (which `decide_skill_crafting_gpt54.py` already mints — R5, ~29 of them on the live `skill_bank_out/run_20260430_030637` snapshot). Each `(source_game → target_game)` becomes one Stage-3a probe. |
| **A4 — Full system** | A3 + G5 non-regression + promotion / rollback hooks. | Plug into the existing `decide_promotion_gpt54.py`. Sustained G0 failures trigger demotion. |

Cell-pair deltas that matter (per
[`§20.3`](../../plans/05-harness/PLAN-HARNESS.md#203-ablation-matrix)):
A0→A1 = structural validation alone; A1→A2 = G0 + veto contribution;
A2→A3 = transfer-safety contribution; A3→A4 = promotion / rollback
contribution. The intra-gymv slice populates the §20.5
`cross_domain_transfer` axis at the *task* granularity — exactly the
gap §22 says we need to fill.

---

## 4. Hard prerequisites already named in the audit

[`harness/README.md` §9–§22](../../harness/README.md#spec-contract-gaps-audit-2026-04-30)
enumerates the gaps that would produce silent garbage if we ran cells
today. Three of them gate which cells are actually evaluable:

| Audit item | Effect on cells | Required posture |
|---|---|---|
| **§21 — cold-start `protocol.steps` is NL prose, not typed hops** | `iter_hops(skill)` yields zero hops on every cold-start skill → `GymvAdapter.run` returns a stub `SkillEpisode` regardless of state. **A2/A3/A4 are degenerate without the protocol lift.** | Sequence Suggested-work-order #6 (`labeling/_decorate_skill_records.py`-style transformer) **before** A2. Phase 0 (A0/A1) is unaffected — those cells don't depend on hops executing. |
| **§22 — no `feasible_tasks` axis** | `EligibilityFilter` admits an Airstriker-mined skill into a Columns episode because both have `state.domain == "gymv"`. **No way to tell A2/A3/A4 apart on a transfer probe** — the filter never narrows on `task`. | Sequence Suggested-work-order #7 (additive: `SkillRecord.feasible_tasks` / `verified_tasks`, `target_task` parameter on `FewShotAdapter.adapt`) before reporting transfer-safety numbers. Pure additive contract change. |
| **§16.1 — adapter executors are stubs** | `run_skill` fabricates a plausible episode without touching the env. **Cell A2 is *also* degenerate** because the rollout never advances. | Wire `GymvAdapter.set_executor(real_step)` per Suggested-work-order #8 — `cold_start/generate_cold_start_actor_gymv.py` already drives the real env, so this is exposing its `step()`, not new infrastructure. |

The order in
[`harness/README.md` §"Suggested work-order"](../../harness/README.md#suggested-work-order)
#6 → #7 → #8 is exactly the order required to make all five cells
meaningful. The plan respects it.

---

## 5. Where the test lives — `skill_transfer_test/` is a measurement layer

Per the
[`crafter-harness-orchestrator-roles.md` §3 ownership table](crafter-harness-orchestrator-roles.md#3-the-io-contract-table)
the artefacts that drive promotion are owned by three components:

| Artefact | Written by |
|---|---|
| `BankMutationProposal` (`Patch / Compose / Transfer / Retire`) | **Crafter** |
| `SkillEpisode` (per-skill execution record) | **Harness (online)** |
| `GateVerdictPayload` (per stage) | **Harness (`GateRunner`)** |
| `SkillEvaluationRecord` (roll-up) | **Harness (`GateRunner`)**; *consumed* by Orchestrator |
| `bank_snapshot_id` updates / `status` / `verified_domains` | **Orchestrator (`PromotionOrchestrator`)** |
| `AuditRecord` (rationale, who/when, linked snapshots) | **Orchestrator** |

`skill_transfer_test/` writes **none** of these. It is a *measurement
layer* over the dump driver's outputs and produces only the §20.6
ablation reports. Its inputs are existing on-disk artefacts:

```
labeling/skill_bank_out/run_<ts>/<corpus>/<source>/skill_bank.jsonl
labeling/skill_actions_out/run_<ts>/<corpus>/<source>/episode_*.json
labeling_supplement/{crafter_proposals_out, episode_reflections_out}/run_<ts>/...
labeling_supplement/dump_harness_io_out/run_<ts>/<corpus>/<source>/{online,offline}/...
labeling_supplement/promotion_decisions_out/run_<ts>/...   # for A4
```

Its outputs are limited to:

```
skill_transfer_test/runs/<ts>/<cell>/<corpus>/<source>/...      # per-cell dumps (handed to dump driver)
skill_transfer_test/runs/<ts>/reports/{a,b,c}.md                # the three §20.6 reports
skill_transfer_test/runs/<ts>/_run_meta.json                    # argv + cell configs + input run paths
```

Same trap-to-avoid as
[`crafter-harness-orchestrator-roles.md` §8](crafter-harness-orchestrator-roles.md#8-tldr): no driver under `labeling_supplement/` or
`skill_transfer_test/` may import another driver's code or write into
another driver's output directory. **The on-disk JSONL is the only
shared API.** That's what makes the test layer replaceable — when the
live online runtime ships, the dump driver gets swapped, the JSONL
contract stays.

### Folder layout (target)

```
skill_transfer_test/
├── README.md                          # condensed plan, cross-refs PLAN-HARNESS §20, audit §9–§22
├── conftest.py
│
├── cell_configs/                      # one config per §20.3 cell — no new code paths, just toggles
│   ├── a0_no_harness.yaml             # bypass select_eligible_skills + validate_invocation entirely
│   ├── a1_harness_lite.yaml           # EligibilityFilter only; G0/G2/transfer disabled
│   ├── a2_harness_core.yaml           # + G0, G2, validate_invocation veto, fit/risk scoring
│   ├── a3_harness_transfer.yaml       # + G3 replay, G4 shadow, G3a few-shot transfer (task axis on)
│   └── a4_full_system.yaml            # + G5 non-regression, promotion / rollback
│
├── runner.py                          # CLI: --cells {a0,a1,a2,a3,a4,all} --probe intra_gymv
│                                      # Calls labeling_supplement/dump_harness_io_gpt54.py per cell
│
├── slices.py                          # §20.5 axis builders — in_domain_reuse / cross_domain_transfer
│                                      # (per task!), before_promotion / after_promotion, easy/hard, per-game
│
├── metrics/
│   ├── validity.py                    # Q1: invalid_invocation_rate, slot_binding_pass_rate, ...
│   ├── veto.py                        # Q3: veto precision/recall (where ground truth available)
│   ├── transfer.py                    # Q2: transfer_pass_rate, regression_rate_after_transfer, ...
│   └── actor_quality.py               # Q4: actor top-1 / top-k accuracy on Harness-eligible set
│
├── reports/
│   ├── report_a_actor_decision.py     # §20.6(a)
│   ├── report_b_harness_filtering.py  # §20.6(b)
│   ├── report_c_system_outcome.py     # §20.6(c)
│   └── render_summary.py
│
├── runs/                              # output dir, gitignored
│
└── tests/
    ├── test_cell_configs_load.py
    ├── test_metric_q1_validity.py
    ├── test_actor_quality_q4.py
    └── test_smoke_a0_a4_one_source.py # Airstriker only, --max-episodes 2 --max-steps 5
```

---

## 6. The "directly wire all modules" question

The other recurring scoping question:

> *"Can we skip the test and wire all modules together directly?"*

The canonical answer is in
[`harness/README.md` §15–§17](../../harness/README.md#wire-up-status-audit-2026-04-30):

> **No for the live online runtime, yes for the offline promotion loop.
> The asymmetry is structural — see §17 for why the offline loop is a
> hard prerequisite for the online one.**

### 6.1 The five hard blockers (§16)

| # | Blocker | What goes wrong on a live wire-up today |
|---|---|---|
| §16.1 | All five adapter executors fall back to `_deterministic_executor` because `set_executor(real_step)` is never called in production | `run_skill` becomes a black hole — adapter fabricates a plausible `SkillEpisode` and returns; the env never advances. **Every cold-start rollout regresses silently** (and these are runs that cost ~$1.5 K per full sweep per the README cold-start table). |
| §16.2 | `EpisodeRunner.run` requires `env.step(SkillEpisode) -> (StateSchema, bool)`; production gym envs take primitive actions and return text observations | Crash before any signal. |
| §16.3 | `decision_agents.skill_interface.SkillBankProvider` reads `skill_agents.SkillBankMVP` (legacy 4-stage); `SkillHarness` reads `skill_bank.SkillRepository` (new 4-store). No `skill_bank/legacy_bridge.py` (`IMPLEMENTATION-STATUS.md` §"Not yet delivered") | Even a correctly-wired `HarnessSkillProvider` points at an empty new-bank while the legacy bank does the real work. |
| §16.4 | `EpisodeRunner` expects `ActorLike.choose_action(state, eligible) -> Optional[ActorChoice]`; current `ActorAgent.run` returns text actions and consumes text observations | No live ActorAgent can satisfy the contract — adapter wrapper required first. |
| §16.5 | `SkillProvider.record_outcome(skill_id, *, outcome, reward, …)` is per-attempt scalar; `RewardLogger` and Stage 2 expect a full `SkillEpisode` | Loss of fidelity — Stage 2 shadow can't tell shadow-mode failures from real-mode failures. |

### 6.2 The §17 keystone

Even if every §16 blocker were solved tomorrow, the runtime would still
be skill-starved:

```51:57:Multi-hop-Reasoning-VLM-Agent/skill_bank/repository.py
    def runnable(self, *, include_shadow: bool = True) -> List[SkillRecord]:
        out: List[SkillRecord] = []
        for r in self.active.all():
            if r.status == SkillStatus.SHADOW and not include_shadow:
                continue
            out.append(r)
        return out
```

`bank.runnable()` reads only from the `active` store, which holds
`ACTIVE` and `SHADOW`. Cold-start ingest puts every skill in
`CANDIDATE`. Until the offline promotion loop fires *at least once*
and graduates a skill across `CANDIDATE → SHADOW`, `EpisodeRunner`
sees `[]` and every online step is a no-op. The §17 conclusion:

> **The offline promotion path is a hard prerequisite for the online
> runtime, not the other way around.**

### 6.3 The reframe — the test *is* the offline cycle

Following the dependency chain:

```
audit §21 (protocol lift)
audit §22 (feasible_tasks)
audit §16.1 (gymv real executor)        ← Suggested-work-order #6, #7, #8
        ↓
labeling_supplement/dump_harness_io_gpt54.py --surface offline
        ↓ produces SkillEvaluationRecord per Crafter proposal
labeling_supplement/decide_promotion_gpt54.py
        ↓ promotes CANDIDATE → SHADOW for survivors
SkillRepository.runnable() now non-empty
        ↓
EpisodeRunner.run(...) stops returning no-ops
```

The intra-gymv transfer test sketched in `skill_transfer_test/` is
**the same execution graph as the first offline promotion cycle.** It
is not parallel work. The §20 ablation cells A3/A4 *are* Stage 3a +
Stage 4 of the gate stack; firing them on the cold-start corpus is
exactly what graduates the first batch of skills out of `CANDIDATE`.
So the question collapses:

> "Do we need the transfer test?" ≡ "Do we need to fire the offline
> cycle even once?" ≡ "Do we want a non-empty `bank.runnable()`?"

Per §17, yes.

### 6.4 What the test does **not** solve

Honest boundary: the test does **not** solve §16.2 / §16.3 / §16.4 /
§16.5. Those are independent live-runtime adapter / shape mismatches
that have to be fixed before any online actor wire-up, regardless of
what the test says. The test makes them safe to attempt, not
unnecessary.

| Need | What unblocks it |
|---|---|
| `bank.runnable()` non-empty (§17) | The transfer test, by firing the offline loop. |
| `harness.run_skill` actually advances the env (§16.1) | gymv real executor — same prereq as the test. |
| Cold-start protocols are executable (§21) | Protocol lift — same prereq as the test. |
| `EpisodeRunner.run(env, actor)` doesn't crash (§16.2 / §16.4) | EnvLike + ActorLike shims — **independent** of the test. |
| `HarnessSkillProvider` points at the right bank (§16.3) | `skill_bank/legacy_bridge.py` — **independent** of the test. |
| `RewardLogger` sees full `SkillEpisode`s (§16.5) | Actor emits real `SkillEpisode` — depends on audit §10 schema fix, which the test consumes but doesn't ship. |

### 6.5 Cost / risk asymmetry

| Path | Cost if it works | Cost if it doesn't |
|---|---|---|
| **Wire online directly today** | Doesn't apply — §17 makes it a no-op runtime. Best case = silent no-op; "works" by accident only. | Silent regression on `cold_start/generate_cold_start_actor*.py` runs (~$1.5 K per full sweep) until someone notices the env didn't advance. Each bad sweep is multi-day, multi-hundred-dollar lost cycle. |
| **Phase 0 of the test (A0/A1, existing dump driver, smoke slice)** | ~1 working session, $0 API spend, surfaces all of audit §9–§14 gaps as concrete numbers. Per [`§20.7`](../../plans/05-harness/PLAN-HARNESS.md#207-minimal-rollout-order): if A0 ≈ A1 on the smoke slice, the rest of the suite isn't worth running yet — invaluable signal before investing in #6/#7/#8. | Worst case: discovers a deeper gap, saves us from doing #6/#7/#8 wrong. |
| **Phase 1–4 of the test (#6 protocol lift, #7 task axis, #8 gymv executor)** | ~3–5 working sessions, $0 API spend. Lands additive contract changes that any future wire-up needs anyway. Produces the first non-empty `bank.runnable()`. | Strictly additive — if a phase fails, the previous one still leaves the codebase in a strictly better state. |
| **Phase 5 (full A0–A4 + reports) + §16.2/§16.3/§16.4/§16.5 + online wire-up** | Now safe to attempt. Cold-start runs see real `SkillEpisode`s, `bank.runnable()` is populated, gates have fired at least once. | Same risks as any live integration — but no longer compounded by the §16/§17 silent-failure modes. |

The asymmetry is overwhelming. Phase 0 in particular is dominated —
there is no scenario in which not running it is cheaper.

---

## 7. Recommended phased rollout

Same numbering as
[`harness/README.md` §"Suggested work-order"](../../harness/README.md#suggested-work-order),
restated for the intra-gymv probe:

| Phase | Maps to | Deliverable | Cell coverage |
|---|---|---|---|
| **0 — A0 / A1 only on the existing driver** | Suggested-work-order #5 (already shipped) | `skill_transfer_test/runner.py` invokes `dump_harness_io_gpt54.py` once with `--cells a0,a1` on smoke slice (`--max-episodes 2 --max-steps 5`, one source). Reports §20.6(a) + (c) on `in_domain_reuse` slice. | A0, A1 |
| **1 — protocol lift (gymv only)** | Suggested-work-order #6 + audit §21 | Add a `labeling/_decorate_skill_records.py`-style transformer that lifts `protocol.steps: List[str]` → `protocol: List[Dict]` with abstract verbs + `${slot}` placeholders. Hard prereq for A2/A3/A4. **Outside `skill_transfer_test/`** — reusable by every harness consumer. | unblocks A2 |
| **2 — `feasible_tasks` axis** | Suggested-work-order #7 + audit §22 | Additive to `SkillRecord` + `EligibilityFilter` + `FewShotAdapter.adapt(target_task=…)`. Cold-start ingest seeds `feasible_tasks=[provenance.source_name]`. **Outside `skill_transfer_test/`** but consumed by the cell configs. | unblocks A3 (transfer probe is meaningful) |
| **3 — gymv real executor** | Suggested-work-order #8 first half + audit §16.1 | `GymvAdapter.set_executor(real_step)` from `cold_start/generate_cold_start_actor_gymv.py`'s env loop. Smoke through `tests/test_smoke_a0_a4_one_source.py`. | A0 honest; A2 produces real episodes |
| **4 — intra-gymv Stage 3a probe** | Suggested-work-order #8 second half | Build `FewShotDemo`s from `labeling/skill_actions_out/.../<game>/episode_*.json`. Plug a gymv-shape `success_fn` keyed on consecutive `schema_canonical` blocks + `cumulative_reward`. | A3 transfer cell active |
| **5 — full A0–A4 sweep + reports** | Suggested-work-order #5 (offline half) | Run all five cells on the full 13-game corpus; emit §20.6(a)+(b)+(c). **This is the first offline promotion cycle** (§17). | A4 reference cell |
| **6 — cross-domain follow-up** | Suggested-work-order #16 | Same `runner.py`, swap probe = `cross_domain` (gymv → browser/osworld/video/visual_reasoning) once per-domain executors land. The §20 cells stay the same. | future arena |

Phase 0 is reversible and could start today. Phases 2–4 are exactly
the items the audit already prioritized as "smallest cost / highest
value first".

---

## 8. Decisions to confirm before coding Phase 0

1. **Phase ordering.** Suggested-work-order says #6 → #7 → #8 → run the
   sweep. But Phase 0 (A0/A1 only) is meaningful *before* any of that
   lands and produces actionable signal on whether to invest in
   #6/#7/#8 at all. Default: run Phase 0 first as a pre-investment
   check, then decide.
2. **Audit-fix ownership.** Items §21 (protocol lift) and §22
   (`feasible_tasks` field) are **not** in `skill_transfer_test/` —
   they touch
   [`data_structure/extensions/skill_record.py`](../../data_structure/extensions/skill_record.py),
   [`harness/eligibility.py`](../../harness/eligibility.py),
   [`harness/few_shot_adapter.py`](../../harness/few_shot_adapter.py),
   [`labeling/`](../../labeling/). They land as separate commits owned by
   their canonical homes (additive contract changes);
   `skill_transfer_test/` only consumes them.
3. **Scope of Phase 0 reports.** Limit to §20.6(a) (Actor decision
   quality) + §20.6(c) (System outcome) on the `in_domain_reuse` slice
   only. §20.6(b) (Harness filtering quality) needs G0/G2 active and
   is genuinely degraded without #6+#7. Stating the limit up front
   prevents over-claiming on the first run.

---

## 9. Anti-goals (mirroring `PLAN-HARNESS.md` §20.8)

- **Do not build a parallel transfer framework under
  `skill_transfer_test/`.** It is a measurement layer over
  `dump_harness_io_gpt54.py`. Adapter logic, success scorers, and
  proposal mints belong in `harness/`, `labeling/`, and `crafter/`
  respectively.
- **Do not redefine cell semantics, gate thresholds, or ablation
  metrics inside `skill_transfer_test/`.** Those live in
  [`§5.1`](../../plans/05-harness/PLAN-HARNESS.md#51-skillepisode),
  [`§10`](../../plans/05-harness/PLAN-HARNESS.md#10-promotion-gates),
  [`§20`](../../plans/05-harness/PLAN-HARNESS.md#20-optional-harness-ablations),
  and
  [`PLAN-UNIFIED-SKILL-GATE.md`](../../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md).
  The test consumes them; if it needs to change them, that's a signal
  to update the upstream plan first.
- **Do not skip Phase 0** because A2–A4 sound more interesting. Per
  [`§20.7`](../../plans/05-harness/PLAN-HARNESS.md#207-minimal-rollout-order),
  if `A4 − A0 ≈ 0` on the smoke slice the rest of the suite is not
  worth running — Phase 0 is the cheapest way to find that out.
- **Do not collapse Q1, Q2, Q3, Q4 into one number.** The point of the
  suite is that they are separable.
- **Do not let the test write into `verified_domains` /
  `bank_snapshot_id` / `AuditRecord`.** Those are the Orchestrator's
  exclusive surface
  ([`crafter-harness-orchestrator-roles.md` §3](crafter-harness-orchestrator-roles.md#3-the-io-contract-table)).
  The test reads them.

---

## 10. TL;DR

- **"Test the harness" = run [`PLAN-HARNESS.md` §20](../../plans/05-harness/PLAN-HARNESS.md#20-optional-harness-ablations)
  cells A0–A4 with the §20.6 reports, on the
  [`harness/README.md` §22](../../harness/README.md#intra-gymv-transfer-is-the-right-first-milestone)
  intra-gymv probe.** Five cells, four research questions, three
  reports. Cell semantics and thresholds are pinned upstream.
- **`skill_transfer_test/` is a measurement layer, not a parallel
  framework.** It wraps the existing
  [`labeling_supplement/dump_harness_io_gpt54.py`](../../labeling_supplement/dump_harness_io_gpt54.py)
  and writes only the §20.6 reports. It writes none of the
  audit-trail artefacts (proposals, episodes, gate verdicts, audit
  records).
- **"Wire all modules directly" is not a safe alternative.**
  [`harness/README.md` §16](../../harness/README.md#16-hard-blockers--would-silently-break-the-runtime-if-flipped-today)
  lists five hard blockers; [`§17`](../../harness/README.md#17-the-keystone--bankrunnable-is-empty-until-the-offline-loop-fires-once)
  is the keystone — `bank.runnable()` is empty until the offline
  promotion loop fires once. The test **is** that offline loop.
- **Test does not replace §16.2/§16.3/§16.4/§16.5 fixes.** Those are
  independent live-runtime adapter / shape mismatches that gate the
  online half of the wire-up regardless of test results. The test
  makes them safe to attempt, not unnecessary.
- **Phase 0 (A0 + A1 on the smoke slice) is dominated.** Zero API
  spend, ~1 session, surfaces audit §9–§14 gaps as concrete numbers,
  gives the §20.7 stop/go signal before investing in
  Suggested-work-order #6/#7/#8.
- **Suggested order.** Phase 0 → audit fix #6 (protocol lift) → fix
  #7 (`feasible_tasks`) → fix #8 (gymv real executor) → Phase 5 (full
  A0–A4 sweep, which **is** the first offline promotion cycle) → live
  online wire-up (§16.2 EnvLike, §16.3 legacy bridge, §16.4 ActorLike,
  §16.5 reward shape) → cross-domain follow-up.
- **Trap to avoid.** `skill_transfer_test/` must not import another
  driver's code or write into another driver's output directory. The
  on-disk JSONL is the only shared API — same rule as
  [`labeling_supplement/`](../../labeling_supplement/) per
  [`crafter-harness-orchestrator-roles.md` §8](crafter-harness-orchestrator-roles.md#8-tldr).
