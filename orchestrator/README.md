# `orchestrator/` — System control plane

Spec: [`PLAN-PIPELINE-ORCHESTRATOR`](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md), [`PLAN-COMPONENTS-IMPLEMENTATION`](../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) §4 (Phase B).

Single top-level runner that drives outer-env episodes, persists artefacts, runs the gate, and atomically promotes / rolls back skills. The orchestrator is the **only** module that touches all of `harness/`, `crafter/`, and `skill_bank/` at once — and it's the only place where a skill's lifecycle moves forward.

```python
from orchestrator import (
    EpisodeRunner, EpisodeResult,
    ArtifactStore,
    BudgetController, BudgetExceeded,
    GateService, NonRegressionResult,
    PromotionOrchestrator, PromotionPlan, PromotionResult,
    SnapshotManager,
    OrchestratorConfig, BudgetLimits, GateThresholds,
    TeacherConfig, JudgeConfig, FewShotConfig,
)
```

> **Not the only "orchestrator" in this repo.** This package is the *runtime* control plane (rollout / gate / lifecycle / promotion). Two unrelated *training-side* orchestrators live elsewhere — same word, different jobs, no model training happens in this folder.
>
> | Path | Symbol | Job | Plane |
> |---|---|---|---|
> | `orchestrator/` *(this package)* | `EpisodeRunner` + `GateService` + `PromotionOrchestrator` | Episode rollouts, gate evaluation, atomic skill promote / rollback | runtime / data |
> | [`../trainer/coevolution/orchestrator.py`](../trainer/coevolution/orchestrator.py) | `co_evolution_loop` | Three-phase co-evolution loop: rollouts ↔ skill-bank mining ↔ GRPO LoRA training, with vLLM adapter hot-reload | training |
> | [`../skill_agents/grpo/orchestrator.py`](../skill_agents/grpo/orchestrator.py) | `GRPOOrchestrator` | Wraps the GRPO buffer + trainer for the `segment` / `contract` / `curator` skill-bank stages | training |
>
> Inside this package the only class literally named `*Orchestrator` is `PromotionOrchestrator` (atomic promotion transactions); the rest are sibling components the README composes into "the Pipeline Orchestrator."

---

## Module map

| File | Role |
|---|---|
| `runner.py` | `EpisodeRunner` — drives one outer-env episode end-to-end: resets the env, repeatedly asks the Actor to pick from `SkillHarness.select_eligible_skills`, executes via the harness, logs `SkillEpisode` to `ArtifactStore`. Returns `EpisodeResult(run_id, outer_steps, skill_episodes, final_state, budget_snapshot, aborted, abort_reason)` |
| `artifact_store.py` | `ArtifactStore` — atomic, file-backed storage for every artefact type: `episodes/`, `skill_episodes/`, `proposals/`, `failures/`, `evaluations/`, `snapshots/`, `releases/`, plus an append-only `audit.jsonl`. Atomic = write-to-tmp + `os.replace`; one JSON file per record (the audit log is the only JSONL stream). The orchestrator's only side-effect on disk |
| `budget.py` | `BudgetController` — caps outer steps, inner steps, skill invocations, tokens, wallclock-ms, grounding escalations, and teacher calls per episode. Raises `BudgetExceeded` from inside the runner so partial work is still flushed to the artefact store. Hard-cap-only today; the `degrade` path from `PLAN-PIPELINE-ORCHESTRATOR §7.3` is not yet wired |
| `gate_service.py` | `GateService` — composes the canonical gate stages from [`PLAN-UNIFIED-SKILL-GATE`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) §7: `Stage 0 static` (contract + invariant checks), `Stage 1 replay` (delegates to `harness.ReplayValidator`), `Stage 2 shadow` (reads `harness.RewardLogger`), `Stage 3a transfer` (drives `harness.FewShotAdapter` against `TRANSFER_TARGET_DOMAINS`), `Stage 4 non-regression` (baseline-vs-post score delta). Emits `SkillEvaluationRecord` whose `GateVerdictPayload` carries per-stage `StageVerdict`s and an `eligible_domains` list — the verified-targets source of truth for invariant 8. Stages 5 (promotion) and 6 (rollback) are *actions*, owned by `PromotionOrchestrator` |
| `promotion_orchestrator.py` | `PromotionOrchestrator` — promotion / rollback transactions. `promote(plan: PromotionPlan)`: (1) refuses on FAIL verdict, content-hash drift, or `LIMITED_PASS → ACTIVE`; (2) writes evaluations to the audit trail; (3) calls `lifecycle.record_transfer_verification(...)` *before* the status transition (invariant 8); (4) calls `lifecycle.transition_many(...)` for the atomic store move; (5) takes a snapshot via `SnapshotManager` and mints a `RunRelease`. `rollback(skill_id, reason)`: walks the lifecycle through `DEPRECATED → ROLLED_BACK` and writes an audit record |
| `snapshot_manager.py` | `SnapshotManager` — content-addressed JSON snapshots of the active bank state + adapter signature + config payload. Used by every successful `promote(...)` so the resulting `RunRelease` can pin a frozen system version |
| `config.py` | `OrchestratorConfig` and its sub-configs: `BudgetLimits`, `GateThresholds`, `FewShotConfig` (K, max-tokens, per-target pass-rate floor for `FewShotAdapter`), `TeacherConfig` (Synthesis-Reflection / Crafter teacher — defaults to the project-wide control-plane backbone `Qwen/Qwen3.5-35B-A3B`), `JudgeConfig` (eval-driver judge — defaults to `Qwen/Qwen3.5-35B-A3B`, same weights as the crafter teacher; override to `gpt-5.5` via `VLM_AGENT_BACKBONE_JUDGE_MODEL` for paper / formal eval runs). Single source of truth for tunable knobs; loaded from YAML/JSON or constructed directly in tests |

---

## Pipeline

The orchestrator runs **two loops on different timescales**. They share artefacts through `ArtifactStore` but never block each other.

### Hot path — per-episode rollout (`EpisodeRunner.run`)

```
env.reset()                                       ─── env adapter
  │  state
  ▼
loop until done OR BudgetExceeded:
  budget.add_outer_step()
  candidates = bank.runnable()                    ─── skill_bank/repository.py
  eligible   = harness.select_eligible_skills(    ─── harness  (EligibilityFilter)
                   candidates, state)
  choice     = actor.choose_action(state, eligible)── decision_agents  (Actor: Qwen/Qwen3.5-9B)
  if choice is not None:
      budget.add_skill_invocation()
      ep = harness.run_skill(choice.skill,        ─── harness + per-domain adapter
                             state, parent_run_id, bindings)
      artifact_store.put_skill_episode(ep)        ─── orchestrator/artifact_store.py
  next_state, done = env.step(ep)
                                                   on BudgetExceeded → catch, flush, return
artifact_store.put_episode(meta)                   ─── episode-level meta
artifact_store.append_audit({"kind":"episode_done"})
```

Three rules are baked into this code path, not into convention:
- The Actor never sees the raw bank — it only sees the *harness-narrowed* `EligibleSkill[]`. (`PLAN-PIPELINE-ORCHESTRATOR §0a.2`.)
- The orchestrator never executes a skill itself — it always delegates to `harness.run_skill(...)` and gets back a `SkillEpisode`.
- All artefacts are written through `ArtifactStore`; the orchestrator has no other disk side-effect.

### Warm path — gate + promotion (`GateService` + `PromotionOrchestrator`)

This loop is invoked separately from the rollout — typically every N episodes or when the candidate backlog is full. It turns DRAFTs (produced by `crafter/`) into ACTIVE skills.

```
crafter.SkillCrafterService.cycle(failures)         ─── crafter/service.py
  ├── FailureMemory clusters → FailureDiagnoser
  ├── builds BankMutationProposal
  ├── ArtifactStore.put_proposal(p)
  └── SkillLifecycleManager.ingest_draft(skill)     ─── DRAFT lands in draft_store
              │
              ▼
GateService.evaluate(proposal=, skill=,             ─── orchestrator/gate_service.py
                     replay_seeds=, shadow_log=,
                     baseline_score=, post_score=,
                     few_shot_demos=)
  Stage 0  static          (in-module: feasible_domains, evidence role, lineage, source-type)
  Stage 1  replay          → harness.replay_validate(skill, seeds)
  Stage 2  shadow          → reads harness.RewardLogger.filter(skill_id=...)
  Stage 3a transfer        → for each tgt ∈ skill.transfer_target_domains ∩ TRANSFER_TARGET_DOMAINS:
                                FewShotAdapter.adapt(skill, target_domain=tgt, demos=...)
                             → emits verified_targets + diagnostic labels
  Stage 4  non-regression  (in-module: post_score − baseline_score ≥ −max_delta)
        ▼
SkillEvaluationRecord(verdict ∈ {PASS, LIMITED_PASS, FAIL},
                      eligible_domains = source_domains ∪ verified_targets)

PromotionOrchestrator.promote(plan)                 ─── orchestrator/promotion_orchestrator.py
  │  for each (skill, target_status, evaluation, rationale):
  ├── refuse on FAIL  /  content-hash drift  /  LIMITED_PASS → ACTIVE
  ├── artifact_store.put_evaluation(ev)             (audit trail first)
  ├── lifecycle.record_transfer_verification(...)   ─── ★ load-bearing
  │      writes verified_domains + adapter_history BEFORE the status flip
  ├── lifecycle.transition_many([(id, target_status, rationale), ...])
  │      ─── physical store move via SkillLifecycleManager
  ├── snapshot_manager.take(active_records, adapter_signature, config_payload)
  ├── artifact_store.put_release(RunRelease)
  └── artifact_store.append_audit({"kind":"release", ...})
```

The `record_transfer_verification → transition_many` ordering is **load-bearing**: it's the one place invariant 8 (`verified_domains` is gate-owned) becomes a runtime guarantee. Without it, `SkillLifecycleManager._validate_invariants` would refuse the ACTIVE transition for any source-tagged skill (invariant 7), since `verified_domains` would still be empty.

---

## Sub-agent interaction

The orchestrator is the only place that touches all four sub-agents. Every other module exposes a narrow API and trusts the orchestrator to compose them.

| Sub-agent | What the orchestrator calls | What flows back | Where written |
|---|---|---|---|
| **Visual Grounding** ([`vlm_wrapper/`](../vlm_wrapper/), today via env adapters) | Indirect — `env.reset()` / `env.step()` returns the structured `<state>` (`StateSchema`). The orchestrator never calls grounding directly | `StateSchema` (entities, evidence_refs) | None — episode-local, lives inside `EpisodeRunner.run` |
| **Action Agent / Actor** ([`decision_agents/`](../decision_agents/), `Qwen/Qwen3.5-9B`) | `actor.choose_action(state, eligible) → ActorChoice \| None` (`runner.py:116`). Minimal `ActorLike` protocol — anything implementing it is drivable | `ActorChoice(skill, bindings, rationale)` or `None` (no-skill tick) | Choice is implicit in the resulting `SkillEpisode` (skill_id + bindings) |
| **Skill Harness** ([`harness/`](../harness/)) | Hot-path: `harness.select_eligible_skills(candidates, state)` → `harness.run_skill(skill, state, parent_run_id, bindings)`. Gate: `harness.replay_validate(skill, seeds)` and `FewShotAdapter.adapt(skill, target_domain, demos)` (which calls `harness.run_skill` again under the target adapter) | `EligibleSkill[]`, `SkillEpisode`, `ReplayResult`, `AdaptResult` | `SkillEpisode` → `artifact_store.put_skill_episode`; reward-log entries written by `RewardLogger` inside the harness |
| **Skill Bank** ([`skill_bank/`](../skill_bank/)) | Read: `bank.runnable()` for online retrieval. Write: only via `SkillLifecycleManager` — the orchestrator never touches `SkillStore` directly. Specifically: `lifecycle.transition_many(...)`, `lifecycle.record_transfer_verification(...)`, `lifecycle.transition(skill_id, to=DEPRECATED/ROLLED_BACK, ...)` | `SkillRecord[]` for retrieval; `SkillRecord` after each transition | `skill_bank/{draft,candidate,active,archive}` stores |
| **Skill Crafter** ([`crafter/`](../crafter/)) | The orchestrator does **not** call the crafter. The crafter calls *into* `ArtifactStore.put_proposal(...)` and `SkillLifecycleManager.ingest_draft(...)`. The orchestrator's gate then consumes the resulting DRAFTs | `BankMutationProposal` (read from `ArtifactStore` or `SkillRepository.draft.all()`) | `artifact_store/proposals/`, `skill_bank/draft/` |

Specific contracts worth pinning:

1. **The Actor never sees the bank.** `bank.runnable()` is only called by the orchestrator inside `EpisodeRunner.run`, and the result is *immediately* fed into `harness.select_eligible_skills(...)`. The Actor receives only `EligibleSkill[]`. This is the "harness narrows + may veto, actor decides" boundary made executable.
2. **The Crafter never writes ACTIVE.** Architectural rules at the top of [`crafter/README.md`](../crafter/README.md) are mechanically enforced — the package can't import `skill_bank.stores`. The only path from a crafter proposal to ACTIVE goes `ingest_draft → GateService.evaluate → PromotionOrchestrator.promote → SkillLifecycleManager.transition_many`.
3. **`verified_domains` has exactly one writer.** `GateService._run_transfer` *produces* the verified-target list (carried in `GateVerdictPayload.eligible_domains`). `PromotionOrchestrator._record_transfer_verifications` *mirrors* it via `SkillLifecycleManager.record_transfer_verification(...)`. The lifecycle manager is the only function that physically extends `verified_domains` and `adapter_history`.
4. **The Harness is shared between the two loops.** The same `SkillHarness` instance is used by `EpisodeRunner` (online execution), `GateService._run_replay` (deterministic re-execution), and `FewShotAdapter.adapt` (target-domain probes). That's why a skill's behavior on a target domain at gate time matches what it would do online — the adapter, eligibility, and execution code paths are identical.

---

## Phase boundaries

| Phase | What this package contains | Status |
|---|---|---|
| B (MVP) | `EpisodeRunner`, file-backed `ArtifactStore`, `BudgetController`, `GateService` (Stages 0 / 1 / 2 / 3a / 4), `PromotionOrchestrator`, `SnapshotManager` | **Delivered in isolation** — covered by `tests/test_smoke.py::test_smoke_end_to_end`. Live-runtime integration is partial; see "Live-runtime integration status" below |
| D (transfer + replay) | Action-level deterministic replay (Stage 1); Stage 3a wired to real (non-stub) executors for all four `TRANSFER_TARGET_DOMAINS`; per-skill `evaluate_candidate` / `promote_if_passed` / `rollback_if_needed` API as in `PLAN-PIPELINE-ORCHESTRATOR §3a.1` | Pending |
| E (eval suite + dashboards) | `eval_suite.py` (frozen non-regression slice with `eval_suite_id` pinned across releases); slice / label dashboards | Pending |

---

## Live-runtime integration status

This package is internally consistent and the smoke wiring test passes, but **no live-runtime entry point currently drives it**. The hot-path (`EpisodeRunner`) and warm-path (`GateService` + `PromotionOrchestrator`) are exercised today only by the offline mirrors under [`../labeling_supplement/`](../labeling_supplement/) and the unit / smoke tests under [`../tests/`](../tests/).

### Who imports `orchestrator/*` today

Outside this package and the test suite, only four files import anything from `orchestrator`:

| Importer | Pulls | Plane |
|---|---|---|
| [`../crafter/service.py`](../crafter/service.py) | `ArtifactStore` (audit log sink) | live |
| [`../labeling_supplement/decide_promotion_gpt54.py`](../labeling_supplement/decide_promotion_gpt54.py) | `ArtifactStore`, `GateService`, `OrchestratorConfig`, `PromotionOrchestrator`, `PromotionPlan`, `SnapshotManager` | offline mirror |
| [`../labeling_supplement/dump_harness_io_gpt54.py`](../labeling_supplement/dump_harness_io_gpt54.py) | `GateService` | offline mirror |
| [`../labeling_supplement/reflect_per_episode_gpt54.py`](../labeling_supplement/reflect_per_episode_gpt54.py) | `ArtifactStore` | offline mirror |

Notably absent: `decision_agents/`, `cold_start/`, `inference/`, `trainer/`, `baselines/`, `scripts/qwen3_*.py`. The only `EpisodeRunner` instantiation in the repo is `tests/test_smoke.py`.

### Wiring gaps (in dependency order)

Each row is independent of the next; closing them in order is the cheapest sequence.

| # | Gap | Where it surfaces | Fix shape | Status |
|---|---|---|---|---|
| 1 | **Two disjoint `Harness` types share the name.** [`../decision_agents/core/harness.py`](../decision_agents/core/harness.py) `Harness` (env-step: `reset / step(action) / valid_actions(state)`) is what the live actor consumes; `harness/skill_harness.py` `SkillHarness` (skill-exec: `select_eligible_skills / run_skill / replay_validate`) is what `EpisodeRunner` requires. No bridge module exists | `runner.py:81-84` requires both an `EnvLike` *and* a `SkillHarness` simultaneously | Adapter that lifts `decision_agents.core.Harness.step(action_string)` into something `SkillHarness.run_skill(skill, state)` can consume per hop | Pending |
| 2 | **`ActorLike.choose_action` is not implemented by the live actor.** `EpisodeRunner` calls `actor.choose_action(state, eligible: List[EligibleSkill]) → ActorChoice \| None` (`runner.py:38-46`). The live `ActorAgent.step(observation, schema, valid_actions, …) → ActorDecision` ([`../decision_agents/actor_agent.py:351`](../decision_agents/actor_agent.py)) has a different signature, return type, and abstraction (primitive actions vs skills) | First call inside `EpisodeRunner.run` after `harness.select_eligible_skills(...)` | `RunnerActorAdapter(ActorLike)` that wraps `ActorAgent` and renders `EligibleSkill[]` into a guidance pack before calling `step(...)` | Pending |
| 3 | **Bank duality with no bridge.** This package reads from `skill_bank.SkillRepository` (the new four-store, lifecycle-locked bank). The live actor and every cold-start / inference / trainer entry point reads from `skill_agents.skill_bank.bank.SkillBankMVP` (legacy Stage-3 single-JSONL bank). The orchestrator cannot see legacy mining output; the actor cannot see the new lifecycle bank | `runner.py:113` (`bank.runnable()`) expects `SkillRepository` | `skill_bank/legacy_bridge.py` — one-way migration listed in [`../IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §"Not yet delivered" | Pending |
| 4 | **`AdapterRegistry` has no live registrations.** [`../harness/adapter_registry.py`](../harness/adapter_registry.py) is instantiated only in tests and offline mirrors. The shipped per-domain adapters (`harness/adapters/{gymv,browser,osworld,video,visual_reasoning}_adapter.py`) use `make_deterministic_executor` ([`../harness/adapters/_stub_base.py:36`](../harness/adapters/_stub_base.py)) — emits one `GATHER` evidence per hop, sufficient for gate Stage-3a dry-run only. No live `set_executor(real_executor)` exists | `harness.run_skill(...)` would only generate stub evidence in production | Per-domain real `HopExecutor` registration after `AdapterRegistry()`, wired into the cold-start / inference path. Largest piece of work — domain-by-domain action grammar alignment | Pending |
| 5 | **No live `SkillEpisode` emission.** Gate Stage-1 (Replay) needs `SkillEpisode` seeds; Stage-2 (Shadow) reads `harness.RewardLogger`. The live cold-start emits `Episode` (its own format) with no skill-id attribution. [`../implementation_notes/legacy/crafter-harness-orchestrator-roles.md`](../implementation_notes/legacy/crafter-harness-orchestrator-roles.md) §7.4 calls this out as the strict prerequisite for both lanes | `gate_service.py` Stages 1 + 2 fall back to `LIMITED_PASS` for lack of inputs | Add a `SkillEpisode` emitter to the actor's `observe_result` path (or wrap it) and log to `RewardLogger`. Schema already exists in `data_structure/extensions/skill_episode.py` | Pending |
| 6 | **No subscriber to `RunRelease`.** `PromotionOrchestrator.promote` writes a frozen `RunRelease` (snapshot + adapter signature + config payload) but no live consumer watches for it. The actor's skill provider does not pin to or reload from a release | `promotion_orchestrator.py` after `put_release(...)`; effect is invisible to a running actor | Either (a) actor pins `release_id` at episode start and rebuilds its `SkillProvider` from `bank_snapshot_path`, or (b) a watcher reloads the in-memory `SkillRepository` view on `release.json` change | Pending |

### What is wireable today, no changes required

- **`ArtifactStore`** — pure file I/O, schema-clean. Already used by `crafter/service.py` and the three offline mirrors.
- **`SnapshotManager`** — pure JSON, content-addressed.
- **`PromotionOrchestrator`** — fully transactional and gate-bound. Consumes `(SkillRecord, SkillEvaluationRecord)` pairs; the offline mirror in `decide_promotion_gpt54.py` already proves the path end-to-end.
- **`SkillRepository` + `SkillLifecycleManager`** (sibling package) — the four-store bank with locked writes. Crafter and orchestrator already write through it; only the actor read path is missing.
- **`BudgetController`** — pure accounting, drop-in.
- **`GateService` Stages 0, 4, and 3a (with stub adapters)** — LLM-free, run today against any `SkillRecord`. Stages 1 + 2 wait on Gap 5.

### The offline mirror as the de-facto producer

`labeling_supplement/decide_promotion_gpt54.py` exercises the full warm path against synthesised inputs. It deliberately bypasses every live edge:

- Reads `SkillRecord`s seeded by `cold_start_labeling/build_skill_bank_gymv.py` (Gap 3 sidestepped).
- Reads `BankMutationProposal`s from `decide_skill_crafting_gpt54.py` outputs (no live Crafter call).
- Defaults to `--gate-mode offline-synthetic`: Stage 0 runs rule-based, Stages 1 / 2 / 3a / 4 receive `LIMITED_PASS` (Gaps 4, 5 sidestepped).
- Loads actor-batch metrics from `_skill_actions_summary.json` for the post-promotion regression check (no live `RewardLogger`).
- Writes `RunRelease` to disk where the next offline pass can consume it (Gap 6 sidestepped).

This is the model called out at [`../implementation_notes/legacy/crafter-harness-orchestrator-roles.md`](../implementation_notes/legacy/crafter-harness-orchestrator-roles.md) line 337: deterministic, replay-only, no live components, before any of these gets wired into the live runtime.

### Recommended sequencing if/when the live wire happens

1. **Gap 5 first** — without live `SkillEpisode` emission, every gate decision is synthetic and the §7.1 closed-loop synthesis pathology in the implementation notes applies.
2. **Gaps 1 + 2 together** (`HarnessSkillProvider` + `RunnerActorAdapter`) — one logical unit: making the live actor drivable by `EpisodeRunner` against the new bank.
3. **Gap 3** (`legacy_bridge`). Cheap once 1 + 2 are in. Most legacy skills will fail the G0 invariant on ACTIVE promotion (intentional — the bank starts ~empty under the new lifecycle).
4. **Gap 6** (`RunRelease` subscriber). Cheap.
5. **Gap 4** (real domain executors). Largest effort; do per-domain incrementally — gymv → browser → osworld, etc.

Until at least Gap 5 lands, closing the earlier gaps gives no operational benefit: the orchestrator would run, but on synthesised inputs equivalent to what the offline mirrors already produce.

---

## Known gaps vs. the plans

These are honest deltas between this package and `plans/06-orchestrator/` + `plans/07-skill-gate/`. Nothing here breaks the smoke test, but they should be closed before this package can be trusted in a real promotion loop.

1. **Promotion is not snapshot-restoring on partial failure.** `PromotionOrchestrator.promote` runs `transition_many()` *before* `snapshot_manager.take()` and `put_release()`. If snapshot or release writing fails, the bank has already moved and there is no snapshot to revert to. `transition_many` does best-effort rollback inside its own call, but cross-call atomicity (snapshot + release + lifecycle as one transaction) is not enforced. `PLAN-PIPELINE-ORCHESTRATOR §3a.3` requires snapshot-create *before* the lifecycle apply.
2. **`rollback()` does not restore a prior snapshot.** It only walks the lifecycle through `DEPRECATED → ROLLED_BACK`. The "atomically advance `current_production` to a previous snapshot" semantics from `PLAN §3a.4` step 3-4 is not present.
3. **API surface differs from `PLAN-PIPELINE-ORCHESTRATOR §3a.1`.** The plan specifies `evaluate_candidate(skill_id) → SkillEvaluationRecord`, `promote_if_passed(skill_id) → bool`, `rollback_if_needed(skill_id) → bool`, `batch_evaluate_candidates(...)`. The implementation instead exposes `promote(plan: PromotionPlan)` and `rollback(*, skill_id, reason)`. Invariants are preserved; named entry points differ.
4. **No `eval_suite.py` and no `rollback_manager.py`.** Both are named in `PLAN-UNIFIED-SKILL-GATE §4` and `PLAN-PIPELINE-ORCHESTRATOR §9`. Tracked under Phase E.
5. **Stage 0 only partially covers `PLAN-UNIFIED-SKILL-GATE §7 Stage 0`.** `_run_static` validates feasible-domains, evidence-role presence, protocol non-empty, source-type match, and lineage. It does *not* check `evidence_role` consistency with the §8 effect family or "no environment-specific hardcoding."
6. **Online retrieval policy not enforced here.** `EpisodeRunner.run` calls `bank.runnable()` (default `include_shadow=True`), so `SHADOW`/`PROVISIONAL` skills can reach the eligibility filter. Per `PLAN-UNIFIED-SKILL-GATE §6`, online retrieval should see `ACTIVE` + `PROVISIONAL` only (latter with the shadow-origin penalty). The filter must do the right thing downstream; the orchestrator does not yet make a `run_active` vs `run_shadow` distinction.
7. **`EpisodeRunner` double-bumps `state.outer_step`.** Env adapters return a `next_state` (test env sets `outer_step=tick`), then the runner does `state.outer_step += 1` (`runner.py:131`). Today's smoke env happens to be idempotent under this; a real adapter that sets its own `outer_step` will be off-by-one.
8. **`BudgetController` has no `degrade` path** — it's hard-cap-only. `PLAN §7.3` specifies an explicit graceful-degradation behavior (skip optional `CHECK`, drop optional evidence enrichment, etc.).
9. **`ArtifactStore` is one-JSON-file-per-record, not JSONL streams** as `PLAN §2.3` describes (only `audit.jsonl` is a stream). Functionally equivalent for current consumers; stream consumers will break.

---

## Cross-references

- Root [`readme.md`](../readme.md) §"Architecture" / §"Implementation status" / §"Skill transfer layer".
- [`../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) §0a — the actor / harness / skill-bank / orchestrator boundary; §3a — the canonical promotion / rollback transaction.
- [`../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) §7 — canonical gate-stage spec; §9 — `configs/skill_gate.yaml` thresholds mirrored in `config.py`.
- [`../harness/README.md`](../harness/README.md) — the four call points the orchestrator depends on (`select_eligible_skills`, `run_skill`, `replay_validate`, `FewShotAdapter.adapt`).
- [`../skill_bank/README.md`](../skill_bank/README.md) — invariants this orchestrator must respect on every `promote(...)`, in particular invariants 7 (source / target asymmetry) and 8 (`verified_domains` is gate-owned).
- [`../crafter/README.md`](../crafter/README.md) — proposal taxonomy fed into `GateService.evaluate` via `SkillLifecycleManager.ingest_draft`.
- [`../tests/test_smoke.py`](../tests/test_smoke.py) — runnable end-to-end wiring example for both loops.
- [`../trainer/coevolution/orchestrator.py`](../trainer/coevolution/orchestrator.py) — sibling *training* orchestrator (`co_evolution_loop`); see top-of-file callout. Disjoint from this package — it owns model weights and adapters, not skill-record lifecycle.
- [`../skill_agents/grpo/orchestrator.py`](../skill_agents/grpo/orchestrator.py) — sibling GRPO training helper (`GRPOOrchestrator`); wraps the GRPO buffer + trainer for the segment / contract / curator stages used by the co-evolution loop.
