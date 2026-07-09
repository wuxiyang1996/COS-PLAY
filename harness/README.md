# `harness/` — Per-invocation runtime for skill execution and verification

Spec: [`PLAN-HARNESS`](../plans/05-harness/PLAN-HARNESS.md), [`PLAN-COMPONENTS-IMPLEMENTATION`](../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) §4 (Phase A).

> **Current state (post-Phase-5/6, 2026-05-02):** All 5 target domains are
> registered and dispatchable: `gymv` (canonical), `visual_reasoning`,
> `video`, `osworld`, `browser`. Routing happens via
> [`labeling_supplement/_phase4_target_dispatch.py`](../labeling_supplement/_phase4_target_dispatch.py).
> The 4 cross-domain harness executors (`video_executor`, `osworld_executor`,
> `browsergym_executor`, `visual_reasoning_executor`) ship as deterministic
> stubs at the module level, but the dispatcher binds **real-env per-sample
> wrappers** when cold-start data + runtime infra are available (which they
> are in this workspace, see below).
>
> **All 4 Tier 1 + Tier 2 + Tier 3 closed 2026-05-02.** Image-VR + video Stage
> 1/2 cells exercise real VLM tools via
> [`harness/_vr_per_sample_executor.py`](_vr_per_sample_executor.py) and
> [`harness/_video_per_sample_executor.py`](_video_per_sample_executor.py).
> OSWorld Stage 3 cells exercise real `pyautogui` against the live
> `happysixd/osworld-docker` container fleet via
> [`harness/_osworld_per_sample_executor.py`](_osworld_per_sample_executor.py)
> + [`harness/_executor_helpers/osworld_client.py`](_executor_helpers/osworld_client.py)
> (HTTP client over the container's Flask server). BrowserGym Stage 4 cells
> exercise a real Playwright browser via
> [`harness/_browser_per_sample_executor.py`](_browser_per_sample_executor.py)
> + [`harness/_executor_helpers/browser_helper.py`](_executor_helpers/browser_helper.py)
> (JSON-RPC subprocess hosting `gym.make("browsergym/<task>")` in the
> `browsergym` conda env). The per-domain runtime predicate-translator
> that game->cross-domain transfers depend on for non-zero admit rates
> ships as [`harness/predicate_translator.py`](predicate_translator.py)
> and is wired into all 4 cross-domain target builders. See
> [`../implementation_notes/legacy/phase5-cross-domain-measurement.md`](../implementation_notes/legacy/phase5-cross-domain-measurement.md) §12 for the updated inventory.
>
> **Retraction note:** A prior revision of this README and the §12.1 doc
> classified Tier 1 items 3-4 as "infra-blocked, deferred -- needs an OSWorld
> VM in CI / Playwright in CI". That framing was wrong: the workspace already
> ships dedicated `osworld` and `browsergym` conda envs with all
> dependencies, the upstream OSWorld + BrowserGym sources (editable installs),
> `Xvfb` + `xvfb-run` on PATH, 13 pre-warmed `happysixd/osworld-docker`
> containers, and the WebArena Docker stack. The actual gating constraint was
> code-side wiring, not infra.
>
> Numbers measured against the **dispatcher-bound real-env wrappers** (the
> default path when cold-start data + runtime infra are present, see callout
> above) are mechanism-validating; numbers measured against the bare
> deterministic-stub fallback path (when cold-start data / runtime infra is
> missing) remain infrastructure-validating only. See
> [`implementation_notes/legacy/phase5-cross-domain-measurement.md`](../implementation_notes/legacy/phase5-cross-domain-measurement.md)
> §12 for the per-target rollout status; that memo's §11.5.0 reconciles the
> historical stub-pathology with the §11.5.4 aspirational transferability bands.
>
> **Skill-selection design (2026-05-02):** the live trainer +
> standalone agent both run a four-stage pipeline — **RAG retrieves
> top-K → harness adapts (predicate translator) + filters
> (eligibility) → `skill_selection` LLM picks one → harness validates
> the pick**. The harness *informs* the LLM picker via two refinement
> signals it stamps onto each candidate dict: `_harness_deboost`
> (RAG-time veto-history multiplier from
> [`rejection_deboost.py`](rejection_deboost.py)) and
> `_harness_adaptation_score` (filter-time `[0, 1]` summary of
> task-axis match × adapter native-vs-bridged × predicate translation
> provenance). Both are surfaced to the LLM in the `skill_selection`
> prompt as `Adaptation:` / `Recent veto rate:` lines. See
> [§22.5](#225-skill-selection-design--rag-retrieves-harness-informs-llm-picks-harness-validates)
> for the design memo.

The harness is the *frozen verifier* in the role split (root README §"Three-agent role split" / §"Architecture"):

> The Skill Bank provides candidates, the **Harness narrows + may veto**, the Actor decides, the Orchestrator handles offline promotion.

Concretely, the harness's job for one skill invocation is six steps:

1. **Filter** the bank's candidate set to a domain- and contract-eligible subset (`select_eligible_skills`).
2. **Score** every kept skill with `fit_score` + `risk_score` and per-check booleans (`binding_ok / precondition_ok / evidence_ok / adapter_ok`). _Per-check booleans shipped Day-8a on `EligibleSkill.to_json()`; numeric `fit_score / risk_score` LoRA head still pending — see [§9](#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs)._
3. **Validate the invocation** — once the Actor has bound slots and proposed a skill, the harness re-checks and may veto (`validate_invocation`). _Shipped Day-8a as `SkillHarness.validate_invocation(skill, state, bindings=…) → ValidateInvocationResult`._
4. **Execute** one chosen skill via the right `SkillAdapter` (`run_skill`).
5. **Record** everything as a `SkillEpisode` + reward-log entry — this is where invariant **G0 (evidence-driven)** bites via `SkillEpisode.finalize()`.
6. **Provide replay validation** for the gate (Stage 1, [`PLAN-UNIFIED-SKILL-GATE`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) §7).

The harness has **two surfaces** sharing this module:

| Surface | When it runs | Public API |
|---|---|---|
| **Online runtime** | Per actor decision (steps 1–5 above). Inputs: `(schema_state, intention, retrieved_skills, active_skill, local_reasoning_trace)`. Outputs: `eligible_skills` + `invocation_veto` + `SkillEpisode`. | `select_eligible_skills`, `validate_invocation` (Day-8a), `run_skill` |
| **Offline `GateRunner`** | Per Crafter `BankMutationProposal`. Inputs: `(proposal, candidate_skill, replay_seeds, shadow_log/rollout_batch, target_domains, FewShotDemo[], frozen_eval_suite)`. Outputs: per-stage `GateVerdictPayload` + roll-up `SkillEvaluationRecord`. | `GateRunner` (Day-7a) at [`gate_runner.py`](gate_runner.py); subclasses `orchestrator/gate_service.GateService` so all old callers keep working unchanged |

It does **NOT**:
- Choose which skill to commit to — that's the Actor.
- Mutate `SkillRecord.status` — only `SkillLifecycleManager` may.
- Read/write a "memory" buffer — invariant 2 (no-memory).
- Train policies / call the teacher — that's the offline Orchestrator + Crafter.

```python
from harness import (
    SkillHarness, HarnessConfig,
    AdapterRegistry,
    SkillAdapter, AdapterRunContext, AdapterRunResult,
    EligibilityFilter, EligibleSkill,
    ReplayValidator, ReplayResult,
    RewardLogger,
    FewShotAdapter, FewShotDemo, AdaptResult,
)
```

---

## Module map

Regenerated from `harness/__init__.py`'s `__all__` (post-Phase-5/6, 27 source files). Grouped by role.

### Core runtime

| Symbol / file | Role |
|---|---|
| `skill_harness.py` (`SkillHarness`, `HarnessConfig`) | The public entry point. `select_eligible_skills(candidates, state)`, `validate_invocation(skill, state, bindings=…)` (Day-8a), `run_skill(skill, state, …)`, `replay_validate(skill, seeds=…)`. Owns the per-invocation lifecycle: eligibility filter → invocation veto → adapter run → tracing → `SkillEpisode` → replay validation |
| `eligibility.py` (`EligibilityFilter`, `EligibleSkill`, `task_id_from_state`) | Matches `SkillRecord.contract` (typed input slots, pre-conditions, domain, F2′ task) against the current `StateSchema`; returns `EligibleSkill[]` with per-check booleans. Day-8 added `filter_with_rejections(...) → (admitted, rejected)` so the actor can render a veto log |
| `skill_adapter.py` (`SkillAdapter`, `AdapterRunContext`, `AdapterRunResult`) | Base class + run context every per-domain adapter implements |
| `adapter_registry.py` (`AdapterRegistry`) | Domain → `SkillAdapter` lookup. Adapters self-register on `register(adapter)` |
| `replay_validator.py` (`ReplayValidator`, `ReplayResult`) | Day-7b action-level walk over `seed.steps[i]`; emits per-step `StepDiff` (action_type equality, payload equality, evidence-role non-worsening). Adapter-level mode remains the default |
| `reward_logger.py` (`RewardLogger`) | Append-only JSONL of per-step `r_env / r_follow / r_cost / r_total` |
| `few_shot_adapter.py` (`FewShotAdapter`, `FewShotDemo`, `AdaptResult`, `FewShotAdapterError`, `default_success_fn`) | Stage 3a (transfer) of the gate. Takes K target-domain demos, runs them, emits the `eligible_domains` list mirrored into `SkillRecord.verified_domains` (invariant 8) |

### Adapters (one per domain, registered via `AdapterRegistry`)

| Symbol / file | Role |
|---|---|
| `adapters/gymv_adapter.py` (`GymvAdapter`) | Canonical source domain. Real env via `gymv_executor.set_executor(...)` |
| `adapters/visual_reasoning_adapter.py` (`VisualReasoningAdapter`, `bind_visual_reasoning_executor`) | Image-QA / visual reasoning. Inherits `StubTransferTargetAdapter`; bind helper re-exports `visual_reasoning_wrapper.skill_executor.bind_executor` |
| `adapters/video_adapter.py` (`VideoAdapter`, `bind_video_executor`) | Short-video evidence-grounded reasoning. Inherits `StubTransferTargetAdapter`; bind helper re-exports `harness.video_executor.make_video_executor` |
| `adapters/osworld_adapter.py` (`OsworldAdapter`) | OSWorld desktop. Inherits `StubTransferTargetAdapter` |
| `adapters/browser_adapter.py` (`BrowserAdapter`) | BrowserGym / webagent. Its **own** `SkillAdapter` subclass (not `StubTransferTargetAdapter`) but uses the same hop-loop shape |
| `adapters/_common.py`, `adapters/_stub_base.py` | Shared slot-binding helpers; `StubTransferTargetAdapter` + `make_deterministic_executor` (Day-7d typed verb-→-role table with `evidence_in` / `evidence_out` split) |

### Per-target executors (deterministic stubs at the harness level — see top-of-file "Current state" callout)

| Symbol / file | Role |
|---|---|
| `gymv_executor.py` (`make_gymv_executor`, `initial_state_from_env`, `GymvExecutorState`, `ACTION_ALIAS_MAP`) | **Real** Day-3 env wiring. Plugs into `GymvAdapter.set_executor`; maps typed hop ops to concrete env actions; threads a `schema_producer=…` for decidable post-states |
| `video_executor.py` (`make_video_executor`) | Phase-5 typed deterministic stub at the module level (identity-passes rebound contract predicates against a `video_meta` payload). The dispatcher binds [`_video_per_sample_executor.py:TaskAwareVideoReasoningExecutor`](_video_per_sample_executor.py) over `visual_reasoning_wrapper.video_skill_executor.VideoReasoningExecutor` for real frame decode + VLM tools when cold-start `video_meta` is on disk |
| `osworld_executor.py` (`make_osworld_executor`) | Phase-5 typed deterministic stub at the module level. The dispatcher binds [`_osworld_per_sample_executor.py:TaskAwareOsworldExecutor`](_osworld_per_sample_executor.py) over [`_executor_helpers/osworld_client.py:OsworldClient`](_executor_helpers/osworld_client.py) for real `pyautogui` against the live `happysixd/osworld-docker` container fleet (HTTP) when cold-start tree + container fleet are both present |
| `browsergym_executor.py` (`make_browsergym_executor`, `BrowserExecutorState`) | Phase-5 typed deterministic stub at the module level. The dispatcher binds [`_browser_per_sample_executor.py:TaskAwareBrowserExecutor`](_browser_per_sample_executor.py) over [`_executor_helpers/browser_helper.py`](_executor_helpers/browser_helper.py) (JSON-RPC subprocess hosting real Playwright `gym.Env` in the `browsergym` conda env) when cold-start tree present |
| (`visual_reasoning` has no dedicated executor module — uses `StubTransferTargetAdapter._default_executor()` at module level; dispatcher binds [`_vr_per_sample_executor.py:TaskAwareVisualReasoningExecutor`](_vr_per_sample_executor.py) for real per-sample image loading + VLM tool dispatch when cold-start frames present) | — |

### Schema producers (deterministic `<state>...</state>` renderers — no VLM)

| Symbol / file | Role |
|---|---|
| `gym_schema_producer.py` (`make_gaming_env_producer`, `twenty_forty_eight_producer`, `tetris_producer`, `candy_crush_producer`, `super_mario_producer`, `render_state_block`, `SchemaProducer`) | Day-4B / Day-6 producer for `make_gaming_env(...)` envs. Round-trips through `parse_schema_canonical` so predicates are decidable end-to-end |
| `osworld_schema_producer.py` (`make_osworld_producer`) | Phase-5 producer for the OSWorld stub-executor input shape |
| `browser_schema_producer.py` (`browsergym_canonical_producer`, `make_browsergym_producer`) | Phase-5 producer for the BrowserGym stub-executor input shape |

### Success-fn registrations (`register_success_fn(domain)` from `harness.gymv_success`)

| Domain | Module | Factory |
|---|---|---|
| `gymv` | `gymv_success.py` (`make_per_step_success_fn`, `evaluate_predicate`, `evaluate_hop_effects`, `evaluate_episode_effects`) | `make_per_step_success_fn` — registered at import |
| `visual_reasoning` | `qa_success.py` (`make_qa_success_fn`, `qa_answer_matches`) | QA-style answer-equality scorer |
| `video` | `video_qa_success.py` (`make_video_qa_success_fn`) | Video-QA answer-equality scorer |
| `osworld` | `osworld_success.py` (`make_osworld_per_step_success_fn`) | OSWorld desktop predicate scorer |
| `browser` | `browser_success.py` (`make_browser_per_step_success_fn`) | BrowserGym-shape predicate scorer |

### Few-shot demo loaders (one per target domain)

| File | Builds `FewShotDemo[]` from |
|---|---|
| `few_shot_demos_gymv.py` | `labeling/skill_actions_out/.../<game>/episode_*.json` |
| `few_shot_demos_vr.py` | `Cold-start-out-visual-reasoning/{visual_toolbench,tir_bench}/sample_*.json` (re-tagged `state.domain="visual_reasoning"`) |
| `few_shot_demos_video.py` | Cold-start video samples |
| `few_shot_demos_osworld.py` | Cold-start OSWorld samples |
| `few_shot_demos_browsergym.py` | Cold-start BrowserGym samples |

### Gate / runner

| Symbol / file | Role |
|---|---|
| `gate_runner.py` (`GateRunner`, `GateRunnerConfig`, `EvalSuite`) | Day-7a spec-named offline gate surface (PLAN-UNIFIED-SKILL-GATE §6). Subclasses `orchestrator.gate_service.GateService`; threads reproducibility anchors (`bank_snapshot_id`, `eval_suite_id`, `adapter_versions`, `ontology_version`, `seed`, `judge_model`) into every emitted `SkillEvaluationRecord`. Adds `rollout_batch: Sequence[SkillEpisode]` (Stage 2) and `eval_suite: EvalSuite` (Stage 4) shapes that close §12. Lazy `__getattr__` import to avoid circular orchestrator dep |
| `rejected_skill_sink.py` (`RejectedSkillSink`, `FlushReport`) | Day-9c in-process aggregator between the harness and the Crafter. `observe(rejected, domain=…, task=…)` after every `filter.filter_with_rejections(...)`; `flush_to(lifecycle, min_count=…)` writes `false_binding_patterns` evidence via `lifecycle.record_false_binding_pattern` |
| `rejection_deboost.py` (`compute_deboost`, `apply_deboost_to_candidates`) | Refinement A of [§22.5](#225-skill-selection-design--rag-retrieves-harness-informs-llm-picks-harness-validates) — turns `SkillRecord.false_binding_patterns` into a multiplicative deboost on the RAG candidate list. Pure CPU, count + recency + on-axis weighting, deboost factor stamped onto each candidate as `_harness_deboost`. Wired into [`scripts/qwen3_decision_agent.get_top_k_skill_candidates`](../scripts/qwen3_decision_agent.py) |
| `predicate_translator.py` (`translate_skill_contract`, `PREDICATE_TRANSLATIONS`) | Layer C of the cross-domain integration — rewrites a `SkillRecord.contract`'s `effects_add` / `effects_del` predicates between source and target domain vocabularies. Spliced into `SkillHarnessHook.filter_candidates` so cross-domain skills present a target-grounded contract to the eligibility filter; identity / diagonal cells short-circuit |
| `eligibility.task_id_from_state` | Helper used by F2′ to canonicalise `state.task` (`"make_gaming_env/<game>"` → `"<game>"`) |
| `validate_invocation` (on `SkillHarness`) | Day-8a second-pass invocation veto. Returns `ValidateInvocationResult` with per-check booleans + `veto_reasons / missing_bindings / missing_evidence_in / failed_preconditions` |

---

## Adapter contract

Every concrete adapter under `adapters/` inherits `SkillAdapter` and implements:

```python
class SkillAdapter(ABC):
    domain: str          # one of common.enums.DOMAINS
    name: str

    def can_handle(self, skill: SkillRecord, state: StateSchema) -> bool: ...
    def bind_slots(self, skill, state) -> SlotBinding: ...
    def run(self, ctx: AdapterRunContext) -> AdapterRunResult: ...
```

`AdapterRunResult` carries the executed action sequence, the produced evidence (mapped to the skill's `expected_evidence_roles`), and a `success: bool` flag. The harness wraps the run in a `SkillEpisode`; if any non-`ACTION` step has empty evidence, `SkillEpisode.finalize()` raises and the episode is marked `failed_contract` (invariant G0).

---

## Phase boundaries

| Phase | What this package contains | Status |
|---|---|---|
| A (MVP) | `SkillHarness`, eligibility filter, gymv + browser adapters, reward log, replay-validator stub | **Delivered** — covered by `tests/test_smoke.py` and `tests/test_invariants.py` |
| D (transfer + replay) | Action-level `ReplayValidator` (Day-7b); six-gate `GateRunner` G0–G5 (Day-7a at [`gate_runner.py`](gate_runner.py)); `osworld / video / visual_reasoning / browser` adapters (Phase-5/6, deterministic stubs at the harness level — see top-of-file "Current state" callout); `transfer_manager.py` shadow → active still pending | **Partial** — adapters / executors / schema producers / success_fns / few-shot demo loaders shipped 2026-05-02 as deterministic stubs; full deterministic `ReplayValidator` snapshot path and shadow → active quarantine still pending. See root README §"Pending" |
| F (trainable extensions) | LoRA heads `skill_select`, `continue_vs_switch`, `accept_transfer`, `adapter_refine` consumed by `SkillHarness.select_eligible_skills` | Pending |

---

## What's missing today (pending work in this package)

Grouped by impact. Items 1–4 are blocking the "harness as gate verifier" story; items 5–8 are smaller correctness / completeness gaps inside the existing files.

### 1. Transfer-target adapters are deterministic stubs

Five target domains are registered: `gymv` (canonical, real env via `gymv_executor.py`), `visual_reasoning`, `video`, `osworld`, `browser`. Of these, only `gymv_adapter.py` drives a real environment. `osworld_adapter.py`, `video_adapter.py`, and `visual_reasoning_adapter.py` inherit `StubTransferTargetAdapter` (`adapters/_stub_base.py`); `browser_adapter.py` is its own `SkillAdapter` subclass with the same shape but a separate hop loop. All four transfer-target adapters fall back to `make_deterministic_executor` when no real executor is registered. Day-7d typed the stub: `make_deterministic_executor(...)` now emits a verb-→-role-table-keyed evidence role (`GATHER / VERIFY / REASON / COMMIT` plus common synonyms) with a directional `evidence_in / evidence_out` split rather than a single synthetic `GATHER` per hop. The harness-side cross-domain executors (`harness/video_executor.py`, `harness/osworld_executor.py`, `harness/browsergym_executor.py`) shipped Phase-5/6 as **typed deterministic stubs** that identity-pass the rebound contract's predicates rather than calling a real env or VLM — numbers measured against them are infrastructure-validating, not mechanism-validating (see top-of-file "Current state" callout). Real env binding (BrowserGym / Playwright / OSWorld VM / video frame indexer / VR pixel tools) is still owed by `vlm_wrapper/<domain>_adapter.py` and must be plugged in via `adapter.set_executor(real_executor)`. Helpers: `bind_visual_reasoning_executor` (in `visual_reasoning_adapter`) and `bind_video_executor` (in `video_adapter`) re-export the wire-up from the harness-side stub modules so callers don't have to import the executor module directly.

### 2. `ReplayValidator` is a "dry-run rerun", not a real replay

`replay_validator.py` calls the adapter again from the seed's *initial* state with `dry_run=True`. It never:

- walks `seed.steps` action-by-action,
- compares post-states or evidence locators across the original and replayed runs,
- consults `orchestrator/SnapshotManager` for a frozen environment snapshot.

The PASS threshold is also hardcoded (`0.8` in `as_stage_verdict`) and there is no curated **held-out replay-seed corpus** — `validate(seeds=...)` requires the caller to bring its own.

### 3. The six-gate runner and the transfer manager don't exist yet

Two files named in the plan are not in the package:

| Missing file | Owes |
|---|---|
| `harness/gate_runner.py` | Unified `GateRunner` walking G0–G5 against `gate_service` stages; today only `orchestrator/gate_service.py` composes the stages, and there is no harness-side aggregator that owns the final verdict |
| `harness/transfer_manager.py` | Two-phase **shadow → active** protocol layered on top of Stage 3a (`FewShotAdapter`); without it a transfer pass goes straight to "verified" with no shadow-deploy quarantine |

### 4. `FewShotAdapter` runs, but the scorer is a placeholder

`few_shot_adapter.py` itself is wired correctly. What it doesn't ship:

- **Domain-aware scorer.** `default_success_fn` only checks `episode.outcome.success and contract_satisfied`; it never compares `episode.outcome.answer` to `demo.expected`. The docstring explicitly defers this to "the orchestrator's transfer-eval driver", which has not yet plugged in a real `success_fn`.
- **Real demo corpus.** When `demos` is empty, `adapt(...)` falls back to a synthetic empty-state probe and tags the verdict `target_domain_demo_unavailable`. There is no curated `FewShotDemo` set per target domain.
- **One-sided cost stop.** The K-shot loop only breaks on `cost_tokens > _max_tokens`; `cost_ms` is tracked but never used as a stop condition.

### 5. Trainable selection heads (Phase F) — completely absent

`select_eligible_skills` is purely the deterministic `EligibilityFilter` (contract + slot-binding match). The Phase-F LoRA heads — `skill_select`, `continue_vs_switch`, `accept_transfer`, `adapter_refine` — that are supposed to consume the harness's candidate set are not implemented.

### 6. Budget is collected but not enforced by the harness itself

`SkillHarness._effective_budget` builds `{tokens, hops, ms}` and passes it into `AdapterRunContext`, but the only place that actually checks budget is `BudgetGuard` in `_stub_base.py`, and only for `hops` and `ms`. The harness never:

- compares `result.cost` against the budget after the run,
- aborts mid-run on token overrun,
- reads `r_cost` back from `RewardLogger` to feed the orchestrator's `BudgetController` (today the path is writer-only).

### 7. `_record_failure` always blames the last step

```python
last_step = episode.steps[-1] if episode.steps else None
trace = FailureTrace(..., failed_step_index=last_step.step_index if last_step else None, ...)
```

In a multi-hop adapter run the failing step isn't necessarily the tail. The adapter result has no slot to communicate "step N failed", so the harness has no way to record it accurately.

### 8. `HarnessConfig.fail_on_missing_adapter` is a no-op

Both branches of the `if self._config.fail_on_missing_adapter:` guard inside `run_skill` currently `return episode`. Either the flag should drive distinct behavior (e.g. raise vs. return a failed episode) or be removed.

---

## Spec-contract gaps (audit: 2026-04-30)

Items 1–8 above are runtime/execution holes inside files that already exist. Items 9–14 below are gaps in the **API surface and persisted-artefact shapes** between [`PLAN-HARNESS`](../plans/05-harness/PLAN-HARNESS.md) / [`PLAN-UNIFIED-SKILL-GATE`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) and what the live code actually exposes. They were surfaced while scoping a `labeling_supplement/dump_harness_io_gpt54.py` driver that exercises the live harness against the cold-start corpus.

### 9. Online-surface API gaps (`validate_invocation`, scoring, intention/active_skill inputs)

*Shipped per Day-7/8/9 status block at the top of §22 — see §22.5 Day-8a and the source files [`eligibility.py`](eligibility.py) (`filter_with_rejections`, `RejectedSkill`, per-check booleans on `EligibleSkill.to_json()`) and [`skill_harness.py`](skill_harness.py) (`validate_invocation` returning `ValidateInvocationResult`) for the live API. Remaining structural gaps: §9.2 `intention / active_skill / local_reasoning_trace` plumbed via `state.extra` but not yet typed first-class params (Day-10+); §9.3 numeric `fit_score / risk_score` LoRA head still pending.*

### 10. `SkillEpisode` artefact field gaps

*Shipped per Day-7/8/9 status block at the top of §22 — see §22.5 Day-8b/c and [`../data_structure/extensions/skill_episode.py`](../data_structure/extensions/skill_episode.py) for the live shape. The expansion added `shadow: bool`, `diagnostic_labels: List[str]`, `protocol_trace: List[Optional[int]]`, per-step `evidence_in / evidence_out / protocol_index / evidence_warrant / verify_verdict / reason_warrant`, and `outcome.contract_progress: Dict[str, bool]` + `outcome.reward_components: Dict[str, float]`. All additions are additive (legacy callers see identical JSON for legacy fields plus None / [] for new ones).*

### 11. `SkillEvaluationRecord` reproducibility-anchor gaps

*Shipped per Day-7/8/9 status block at the top of §22 — see §22.5 Day-8b/c and [`../data_structure/extensions/skill_evaluation.py`](../data_structure/extensions/skill_evaluation.py) for the live shape; [`gate_runner.py`](gate_runner.py) (`GateRunnerConfig`) is the writer that pins anchors at construction. Day-9a wires `PromotionOrchestrator.promote(...)` to record `status_before` / `status_after` / `bank_snapshot_id` plus a `reproducibility_anchors` block on every persisted record.*

### 12. `GateService` stage I/O signatures don't match the spec

*Shipped per Day-7/8/9 status block at the top of §22 — see §22.5 Day-7a and [`gate_runner.py`](gate_runner.py) for the live API. `GateRunner` adds `rollout_batch: Sequence[SkillEpisode]` (Stage 2 replacement for `RewardLogger`-only) and `eval_suite: EvalSuite` (Stage 4 replacement for scalar `baseline_score / post_score`). Mixing old + new shapes for the same stage is a `ValueError`.*

### 13. `GateService` lives under `orchestrator/`, not `harness/` — naming mismatch with the spec

*Shipped per Day-7/8/9 status block at the top of §22 — see §22.5 Day-7a. [`gate_runner.py`](gate_runner.py) is the spec-named offline gate surface; subclasses `orchestrator.gate_service.GateService` so all old callers keep working unchanged. Importable from `harness` via lazy `__getattr__` to avoid a circular orchestrator dep.*

### 14. No I/O dump driver — live harness behaviour against the cold-start corpus is unverified

> **Status (2026-04-30, Day-1 of intra-gymv transfer milestone):** Online-surface dump driver landed at [`../labeling_supplement/dump_harness_io_gpt54.py`](../labeling_supplement/dump_harness_io_gpt54.py) with bash dispatcher [`../labeling_supplement/run_dump_harness_io.sh`](../labeling_supplement/run_dump_harness_io.sh). First Phase-0 baseline (twenty_forty_eight + tetris, 6 episodes / 277 steps, online surface only — no `--run-skill`) recorded in [`../labeling_supplement/harness_io_out/run_phase0_20260501_012106/_run_summary.json`](../labeling_supplement/harness_io_out/run_phase0_20260501_012106/) and analysed in [`../labeling_supplement/harness_io_out/_phase0_report.md`](../labeling_supplement/harness_io_out/_phase0_report.md).
>
> One real driver-shape limitation surfaced: the dump driver feeds the actor's per-step `retrieved_skill_ids` (cold-start RAG) as candidates to `select_eligible_skills`. That RAG was built per-game, so cross-game IDs never reach the filter — the dump driver alone cannot exhibit the §22 failure mode. A standalone bypass probe ([`../labeling_supplement/_phase0_cross_eligibility_probe.py`](../labeling_supplement/_phase0_cross_eligibility_probe.py)) feeds all skills in a fused bank as candidates and produces the §22 measurement; it's hardened against `safe_skill_id` collisions across games (e.g. tetris and super_mario both ship `INSPECT/SETUP`). Offline-surface dump and replay-seed round-trip remain unverified.

Nothing today drives the live `SkillHarness` against `labeling/skill_actions_out/` to validate that:

  - the eligibility narrowing agrees with the cold-start actor's bound `skill_query.selected_skill_id`,
  - `run_skill` produces non-empty `SkillEpisode`s on the gymv adapter for the existing `(corpus, source)` pairs,
  - `replay_validate` round-trips synthesized seeds from `labeling/skill_bank_out/.../sub_episodes.json`.

Verifying these is the prerequisite for the actor rewire (`HarnessSkillProvider` per [`../IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §"Not yet delivered"): we need numerical evidence that the harness behaves as designed before it goes in front of the actor.

The natural fit is a sibling driver under [`../labeling_supplement/`](../labeling_supplement/) that mirrors the existing `decide_skill_crafting_gpt54.py` / `reflect_per_episode_gpt54.py` pattern. Data plumbing per surface:

| Driver surface | Bank | Per-step / per-rollout | Proposals |
|---|---|---|---|
| **Online dump** (eligibility, validate_invocation, run_skill, SkillEpisode) | `labeling/skill_bank_out/run_<ts>/<corpus>/<source>/skill_bank.jsonl` | `labeling/skill_actions_out/run_<ts>/<corpus>/<source>/episode_*.json` (state, intention, retrieved_skills, active_skill, raw_intentions, ground-truth `skill_query.selected_skill_id` for agreement metric) | n/a |
| **Offline GateRunner dump** (Stage 0–4, `SkillEvaluationRecord`) | `labeling/skill_bank_out/...` (skill the proposal mutates) | `labeling/skill_actions_out/...` (replay seeds + shadow log + cross-source few-shot demos + `_skill_actions_summary.json` non-regression baseline) | `labeling_supplement/crafter_proposals_out/run_<ts>/...` and `labeling_supplement/episode_reflections_out/run_<ts>/...` |

Note the layering: `labeling_supplement/` only contains *what to verify* (Crafter proposals); the rollouts the harness consumes — replay seeds, shadow logs, transfer demos, non-regression baselines — all still live in `labeling/`.

### 21. Cold-start `protocol` is natural-language prose, not typed hops

> **Navigation note:** §21 and §22 are numbered out of order — they precede §15-§20 below in the file but logically depend on the runtime-topology picture in §15-§20. If you haven't read §15-§20 yet, the [§16.1 stub-executor](#161-adapter-executors-are-stubs-so-run_skill-is-a-black-hole) framing is what makes §21's "no executor call ever happens" claim concrete.

> **Status (2026-04-30):** Day-1 design lock for the lift landed at [`../implementation_notes/legacy/protocol-lift-design.md`](../implementation_notes/legacy/protocol-lift-design.md). Two-loaded-shapes nuance discovered: the audit's "zero hops" framing is correct only for the direct-from-jsonl load path. The dump-driver / probe path already passes prose through [`../labeling_supplement/_harness_io_helpers.py:_wrap_protocol_steps`](../labeling_supplement/_harness_io_helpers.py), which emits `[{"action": "EXEC", "payload": {}, "notes": "<prose>"}, …]` — `iter_hops` yields N hops per skill, but every hop normalises to `"EXEC"` with empty payload. **The shape gap is closed via this workaround; the semantic gap (real verbs, populated `${slot}` placeholders, typed effects) is open.** Empirical sweep across all 80 prose steps in `run_20260430_030637`: a 21-verb gymv-only taxonomy mined from the schema's `<affordances>` block plus subordinator-stripping + downstream-walk classifier covers **74 / 80 = 92.5 %** of prose steps. Implementation lands in [`../labeling/_decorate_skill_records.py`](../labeling/_decorate_skill_records.py) (Day 2). One small pre-existing bug in the schema parser surfaced en route — [`../labeling_supplement/_harness_io_helpers.py:parse_schema_canonical`](../labeling_supplement/_harness_io_helpers.py) does not parse the `<attributes>` block, so `state.facts` only carries `{"goal": ...}`; the success-fn (Day 4–5) needs either a parser extension (Option A) or a direct read from `step.state` (Option B). Tracked in §5.1 of the design doc.

Every adapter — gymv and the four transfer-target stubs — walks `skill.protocol` via `iter_hops`:

```40:44:Multi-hop-Reasoning-VLM-Agent/harness/adapters/_common.py
def iter_hops(skill: SkillRecord) -> Iterator[Tuple[int, Dict[str, Any]]]:
    for i, hop in enumerate(skill.protocol):
        if not isinstance(hop, dict):
            continue
        yield i, hop
```

The contract is `protocol: List[Dict[str, Any]]` (`data_structure/extensions/skill_record.py:89` — *"# ordered hop list"*), where each hop is a dict carrying `action / op / type` plus `${slot}` placeholders the binder can resolve.

The cold-start labeling pipeline produces a different shape. From `labeling/skill_bank_out/run_<ts>/env_wrappers/twenty_forty_eight/skill_bank.jsonl`, skill `COMMIT/MERGE`:

```json
"protocol": {
  "preconditions": [...],
  "steps": [
    "Read the current board and enumerate the four possible slide directions: up, down, left, right.",
    "Determine which directions are legal by checking whether sliding in that direction would move any tile or merge any adjacent equal tiles.",
    "Select the intended direction according to the controlling policy or plan.",
    "Execute a single slide of the full board in the selected direction.",
    ...
  ],
  "success_criteria": [...],
  "abort_criteria": [...],
  "expected_duration": 10
}
```

Two structural problems:

1. **Wrong outer shape.** `protocol` is a *dict-of-string-lists*, not a *list-of-dicts*. When passed through `iter_hops`, the for-loop iterates the dict's keys (strings), each of which fails the `isinstance(hop, dict)` check. **Zero hops are yielded.** No executor call ever happens.
2. **Wrong inner content even if the shape is fixed.** Each step is a natural-language sentence (*"Slides the board in a direction so matching adjacent tiles combine"*), not `{"action": "SLIDE", "args": {"direction": "${dir}"}}`. There is no abstract verb, no typed slot, no `${binding}` placeholder. The contract's `eff_add` / `eff_del` arrays are also empty — there is no formal effect specification to test "did the skill actually do what it promised."

Why this is masked today: every gymv-and-transfer adapter falls back to `_deterministic_executor`, which echoes any input action and emits one synthetic `GATHER` evidence ref ([§1](#1-transfer-target-adapters-are-deterministic-stubs)). The episode that comes back has the right *shape* but no real grounding. The moment a real executor is wired ([§16.1](#161-adapter-executors-are-stubs-so-run_skill-is-a-black-hole)), the prose-protocol input has nowhere to go and `harness.run_skill(...)` will return a no-op episode — even on a same-game state.

This is the upstream blocker for several of the audit items already on the list:

  - [§1](#1-transfer-target-adapters-are-deterministic-stubs) — real adapters can only consume typed hops.
  - [§4](#4-fewshotadapter-runs-but-the-scorer-is-a-placeholder) — `default_success_fn` cannot test "skill did the right thing in target domain" if the skill has no executable semantics in any domain.
  - [§10](#10-skillepisode-artefact-field-gaps) `protocol_trace` row — there is no `skill.protocol[k]` to index `episode.steps[i]` against.

The fix is **not** in `harness/`. It belongs in the cold-start labeling pipeline (`labeling/_decorate_skill_records.py` or a sibling Crafter-style transformer) that lifts `protocol.steps: List[str]` into `protocol: List[Dict]` with abstract verbs and types `eff_add` / `eff_del` from the schema-canonical evidence. The harness inherits the contract; it cannot repair it.

### 22. `feasible_domains` granularity collapses gymv games into a single bucket

> **Status (2026-05-01, Day-7/8 of intra-gymv transfer milestone):** Spec-contract surfaces in items §9–§12 closed at the structural level. Five surfaces landed:
>
> * **Day-7a — `harness/gate_runner.py` lands the spec-named offline gate surface.** `GateRunner` subclasses the existing `orchestrator.gate_service.GateService` (no behaviour change for old callers) and adds two additive shapes that close §12: `rollout_batch: Sequence[SkillEpisode]` replaces the `RewardLogger`-only Stage-2 input (the runner auto-filters to `skill_id`); `eval_suite: EvalSuite` replaces scalar `(baseline_score, post_score)` Stage-4 inputs and threads the `suite_id` into the persisted record. Also exposes `GateRunnerConfig` (`bank_snapshot_id`, `eval_suite_id`, `adapter_versions`, `ontology_version`, `seed`, `judge_model`) — pin once, every emitted `SkillEvaluationRecord` inherits the anchors. Mixing old + new shapes for the same stage is a `ValueError`. Importable from `harness` via lazy `__getattr__` to avoid a circular orchestrator dep.
>
> * **Day-7b — Action-level `ReplayValidator(mode="action_level")`.** The validator now walks `seed.steps[i]` against the proposal's adapter step output and emits a per-step `StepDiff` (action_type equality, payload equality, evidence-role non-worsening — proposed roles ⊇ seed roles). Pass criterion is monotonic-non-worse: extra proposed steps are tolerated, truncation / evidence-role regression is not. Adapter-level mode remains the default and is unchanged. New stage metrics: `step_action_match_rate`, `step_evidence_non_worse_rate`, `n_steps_compared`. Closes Day-6c deferred + the §10 ``protocol_trace`` row's read side (per-step indexing now exists for the gate; the lift-side write still depends on §21).
>
> * **Day-7c — `SkillLifecycleManager.record_task_verification`.** Analog to `record_transfer_verification` for the **task axis** (PLAN-HARNESS §22). When the Stage-3a transfer cycle PASSes, callers now have a sanctioned writer to append to `SkillRecord.verified_tasks` and emit `adapter_history` entries tagged `kind="task_verification"`. Round-trips to disk; idempotent on re-registration; rejects empty rationale / empty `verified_tasks`. Closes the Day-5b in-memory-only persistence gap.
>
> * **Day-8a — `SkillHarness.validate_invocation` + `EligibilityFilter.filter_with_rejections`.** The harness gained the second-pass invocation veto called out in §9. `validate_invocation(skill, state, bindings=…, eligible=…) → ValidateInvocationResult` reports `ok / adapter_ok / binding_ok / precondition_ok / evidence_ok / shadow_only` plus `veto_reasons / missing_bindings / missing_evidence_in / failed_preconditions`. ACTION skills are exempt from the evidence-in check (G0 doesn't apply to ACTION). The eligibility filter gained `filter_with_rejections(...) → (admitted, rejected)`; the previously-silent rejection channel surfaces as `RejectedSkill(skill, veto, veto_reason, …)` so the actor can render a veto log. `EligibleSkill.to_json()` now carries the per-check booleans.
>
> * **Day-8b/c — `SkillEpisode` field expansion + `SkillEvaluationRecord` reproducibility anchors.** The `SkillEpisode` data structure gained `shadow: bool`, `diagnostic_labels: List[str]` (transfer_label auto-mirrors), `protocol_trace: List[Optional[int]]` (step→protocol[k] index, populated lazily on `add_step`); `SkillEpisodeStep` gained `evidence_in / evidence_out` directional split (legacy `evidence` mirrors into `evidence_out`), `protocol_index`, and three citation slots (`evidence_warrant / verify_verdict / reason_warrant`). `SkillEpisodeOutcome` gained `contract_progress: Dict[str, bool]` (per-key contract satisfaction) + `reward_components: Dict[str, float]` (multi-component reward). `SkillEvaluationRecord` gained the §11 anchors: `bank_snapshot_id`, `eval_suite_id`, `adapter_versions`, `ontology_version`, `version`, `status_before` / `status_after`, `rejected_domains`, `rollback_target`, `diagnostic_labels`. `GateRunner.evaluate(...)` populates them directly. All additions are additive; legacy callers see identical JSON for the legacy fields plus None / [] for new ones.
>
> Test deltas: new `test_gate_runner.py` (+5), new `test_replay_validator_action_walk.py` (+6), new `test_lifecycle_task_verification.py` (+6), new `test_validate_invocation.py` (+9), new `test_skill_episode_field_expansion.py` (+7) → +33 unit tests, full suite **375 passed** (1 known pre-existing whitespace-tolerance failure in `test_schema_predicates.py`). Lint clean.
>
> Day-9 follow-ups: (a) wire `GateRunner` into the orchestrator's `PromotionOrchestrator.promote()` so anchors flow into release manifests; (b) `_phase4_transfer_cycle.py --persist` flag that calls `record_task_verification`; (c) Crafter consumes the new `RejectedSkill` channel as `false_binding_patterns` evidence; (d) `osworld_adapter` / `video_adapter` / `visual_reasoning_adapter` real surfaces (work-order item 1 cross-domain stub); (e) `dump_harness_io_gpt54.py` extension to the offline `GateRunner` surface.

> **Status (2026-05-01, Day-9 wiring milestone — Phase-6):** All five Day-9 follow-ups closed at the wire-up level; full report at [`labeling_supplement/harness_io_out/_phase6_report.md`](../labeling_supplement/harness_io_out/_phase6_report.md):
>
> * **Day-9a — `GateRunner` → `PromotionOrchestrator` anchor flow.** `PromotionOrchestrator.promote(...)` now records `status_before` (captured pre-transition), `status_after` (from `PromotionPlan.target_status`), and `bank_snapshot_id` (caller-supplied or just-minted snapshot) on every persisted `SkillEvaluationRecord`. The audit row gains a `reproducibility_anchors` block carrying the union of `eval_suite_ids / ontology_versions / adapter_versions` across all transitions in the batch. New `bank_snapshot_id=…` kwarg lets a multi-promotion batch pin a frozen pre-mutation snapshot.
>
> * **Day-9b — `_phase4_transfer_cycle.py --persist` flag.** With `--persist --persist-bank-root <path>` the driver seeds source skills as DRAFT → CANDIDATE → PROVISIONAL on `<path>`'s SkillRepository (idempotent across re-runs) and calls `lifecycle.record_task_verification(...)` for every promoted skill, with the `(pass_rate, k_used)` metrics that drove the decision. Persistence failures log per-skill but don't abort. New JSON fields: `n_verified_tasks_persisted`, `persist_enabled`, `persist_bank_root`, `persist_errors`.
>
> * **Day-9c — Crafter consumes `RejectedSkill` → `false_binding_patterns`.** Two surfaces close PLAN-SKILL-BANK §4.3b: `SkillLifecycleManager.record_false_binding_pattern(...)` is the sanctioned writer (dedupes on `(veto, domain, task)`, FIFO-caps at `max_patterns=64`, round-trips to disk); `harness.RejectedSkillSink` is the in-process aggregator that the harness / dump driver / orchestrator can call after every `filter.filter_with_rejections(...)` and flush via `sink.flush_to(lifecycle, min_count=…)`. Thread-safe; skips skill_ids the lifecycle doesn't know about (without raising) so transient repos in the dump driver can't blow up the flush.
>
> * **Day-7d — Adapter typed-hop awareness.** `make_deterministic_executor(...)` lifts from "always emit `GATHER`" to a verb-→-role table covering `GATHER / VERIFY / REASON / COMMIT` plus common synonyms (`OBSERVE`, `INFER`, `ANSWER`, …); unknown verbs degrade to `GATHER` (G0 still satisfied); prefix matches work (`VERIFY_TILE → VERIFY`). The stub now emits a directional `evidence_in / evidence_out` split which the `_stub_base.run(...)` shim propagates into per-step records and `SkillHarness.run_skill(...)` copies into `SkillEpisodeStep.evidence_in / evidence_out / protocol_index`. Real env executors (when wired in via `set_executor(...)`) inherit the typed-hop machinery automatically.
>
> * **Day-7e — Dump driver consumes the new offline-gate surface.** The `_validate_invocation_stub` is replaced with a real call into `harness.validate_invocation(...)`; serializes the full `ValidateInvocationResult.to_json()` while preserving legacy `{veto, veto_reason, source}` keys. New `--gate-runner` opt-in switches the offline surface from `orchestrator.GateService` to `harness.GateRunner` with anchors derived from the run context (`bank_snapshot_id ← dump:<bank-stem>`, `eval_suite_id ← cold_start:<actions-stem>`, `adapter_versions={…: "v1"}`, `ontology_version="cold_start_v1"`). The driver's `harness_known_gaps` summary now flags §9.1, §9.3-booleans, §10, §11, §12, §13 as **closed at the structural level**; remaining items (§9.2 planner-context, §9.3 fit_score/risk_score, real cross-domain executors) flagged as Day-10+.
>
> +27 unit tests this round across `test_lifecycle_false_binding_patterns`, `test_rejected_skill_sink`, `test_promotion_orchestrator_anchors`, `test_phase4_persist`, `test_stub_executor_typed_hops`. Full pytest suite at 402 / 403 (one pre-existing whitespace failure unrelated to this work).
>
> **Status (2026-05-01, Day-5/6 of intra-gymv transfer milestone):** Cross-task transfer cycle is end-to-end wired and empirically validated. Three surfaces landed:
>
> * **Day-5a — Lift v2.1: per-game schema-index whitelist + word-set matcher.** [`labeling/_protocol_lift.py`](../labeling/_protocol_lift.py) gained `_SCHEMA_INDEX_LABEL_WHITELIST` (per-(corpus, game)) so tetris's `holes` / `stack_height` / `filled_cells` / `level` / `lines_cleared` bind even when cold-start `schema_canonical` doesn't enumerate them — the producer's canonical labels are now mineable directly. `_first_entity_label` rewritten with a word-set matcher so phrases like ``"no lines are cleared by the placement"`` bind to label `lines_cleared` (substring matching alone wouldn't, because the words are non-adjacent). Singular ↔ plural fold (``"hole count"`` → `holes`); longest-match-wins on ties. Phase-2 smoke pre/post: tetris `Commit/Position` `entity_count_changed` flips from undecidable (Day-4 rate=1.00) to decidable & failing (Day-5a rate=0.00, `count[holes] 1 → 1`) — the rate drop is the rigor signal.
>
> * **Day-5b — Stage 3a `FewShotAdapter` cross-task transfer cycle.** New driver [`labeling_supplement/_phase4_transfer_cycle.py`](../labeling_supplement/_phase4_transfer_cycle.py) wires `GymvAdapter.set_executor(make_gymv_executor(target_env, schema_producer=…))` and runs every source-game skill on target-game demos via `FewShotAdapter.adapt(..., target_task=<game>)`. Demo loader [`harness/few_shot_demos_gymv.py`](few_shot_demos_gymv.py) builds `FewShotDemo`s from `labeling/skill_actions_out/.../<game>/episode_*.json` (StateSchema parsed via `parse_schema_canonical`, bindings carry the recorded action, expected carries reward). `FewShotAdapter._validate` relaxed for **intra-source-domain task transfer** — `gymv → gymv` with `target_task="tetris"` now permitted without listing `gymv` in `TRANSFER_TARGET_DOMAINS`. Empirical headline: 2048 → tetris widens eligibility from 0/3 → 2/3 admit on `(gymv, tetris)`; tetris → 2048 widens 0/6 → 4/6. `verified_tasks` updated in-memory on PASS (persistence is Day-7 lifecycle work). Action-typed skills with task-specific predicates (2048's `Commit/Merge`, tetris's `Commit/Evade`/`Commit/Optimize`) correctly fail → `adaptation_overfitting`; observational/reasoning skills pass on the structural well-formedness check, which is the right Stage-3a semantics.
>
> * **Day-6a — Producer fan-out: candy_crush + super_mario.** [`harness/gym_schema_producer.py`](gym_schema_producer.py) registry grew from `{2048, tetris}` to `{2048, tetris, candy_crush, super_mario}`. `candy_crush_producer` parses the textual obs (`"Board:\n0| R C G C …\nScore: <N>\nMoves Left: <N>"`) for the 8×8 letter-coded grid and emits one aggregate `candy_<color>` text entity per color plus `score` / `moves_remaining` `goal_indicator`s; `phase=gameover` when moves exhaust. `super_mario_producer` parses `"Position of Mario: (X, Y)"` + the `Positions of all objects` table for visible enemies/items, emits one entity per detected object plus `score` / `lives` / `scroll_x` (= `mario.x`) `goal_indicator`s; `progress` normalized over a 3168-px world-1-1 baseline.
>
> * **Day-6b — Domain-keyed `SuccessFn` registry.** `harness.gymv_success` exposes `register_success_fn(domain, factory)` / `success_fn_for_domain(domain, …)` / `registered_success_fn_domains()`. Bootstrap registers `gymv ⇒ make_per_step_success_fn` at import. `FewShotAdapter.adapt` consults the registry: when constructed with the default scorer, swaps in the registered factory for `target_domain`; explicit `success_fn=…` overrides still win. Lets cross-domain transfer (browser/osworld/video/…) plug a domain-aware scorer once at the lifecycle wiring rather than every gate caller.
>
> Test deltas: `test_protocol_lift.py` 33 → 37 (+4), `test_gym_schema_producer.py` 13 → 18 (+5), new `test_few_shot_demos_gymv.py` (+6), new `test_success_fn_registry.py` (+6). Full empirical write-up: [`../labeling_supplement/harness_io_out/_phase4_report.md`](../labeling_supplement/harness_io_out/_phase4_report.md).
>
> Day-7 follow-ups: (a) `AdaptResult` lifecycle persistence — append `target_task` to `verified_tasks` on disk via a `SkillLifecycleManager` transition; (b) action-level `ReplayValidator` walk over `seed.steps`; (c) `harness/gate_runner.py` (work-order item 12); (d) `osworld_adapter` / `video_adapter` / `visual_reasoning_adapter` surfaces for the cross-domain transfer set.
>
> **Status (2026-05-01, Day-4 of intra-gymv transfer milestone):** Predicate-evaluation rigor closed. Two parallel surfaces landed:
>
> * **Lift v2 — predicate-mining vocabulary expansion.** [`labeling/_protocol_lift.py`](../labeling/_protocol_lift.py)::`_PREDICATE_TRIGGERS` widened to cover indirect phrasings real cold-start prose uses (`"valid merges were applied"` → `cumulative_reward_increased`; `"top-out"` → `phase_transitioned`; `"Hole count increases from"` → `entity_count_changed`; `"moves remaining count has decreased"` → `entity_value_decreased`; etc.). [`labeling/_decorate_skill_records.py`](../labeling/_decorate_skill_records.py) gained `--force_relift` to sweep the bank in place. Coverage delta on the existing skill bank: **2 / 12 → 9 / 12 skills** with mined effects. Headline: 2048 `Commit/Merge` was `attribute_changed=3` only → now also `cumulative_reward_increased=3`; tetris `Commit/Optimize` was `entity_count_changed=4` → now also `cumulative_reward_increased=4` and `phase_transitioned=4`.
>
> * **Day-4B — deterministic ``<state>``-block producer for live gym envs.** [`harness/gym_schema_producer.py`](gym_schema_producer.py) renders the same canonical `<state>...</state>` block the cold-start labeler emits, but **deterministically from `env.info`** (no VLM). Two producers ship: `twenty_forty_eight_producer` (board grid + score + max_tile_power) and `tetris_producer` (score + lines + level + holes + active_piece detection). Both round-trip through `parse_schema_canonical`. Plumbed into `make_gymv_executor(env, …, schema_producer=…)` and `initial_state_from_env(env, …, schema_producer=…)` — opt-in, with the Day-3 plain-text obs path as the fallback for unsupported games.
>
> Phase-2 A/B (2048 `Commit/Merge` × 8 deterministic seeds): without producer best_pass_rate=0.67 (undecidable predicates inflate); with producer best_pass_rate=0.33 (only seeds where `up` produces a legal merge actually pass). The drop is the empirical signature of newly-decidable predicates — `attribute_changed` now reports `"score attrs changed"` instead of `"entity_attrs missing on both sides"`. Full empirical write-up: [`../labeling_supplement/harness_io_out/_phase3_report.md`](../labeling_supplement/harness_io_out/_phase3_report.md).
>
> Day-5 follow-ups identified empirically: (a) extend the lift's per-game `schema_index` so tetris binds `holes` / `stack_height` / `filled_cells` as entity labels (the producer surfaces these — the lift just doesn't bind them yet); (b) Stage 3a `FewShotAdapter` cross-task transfer cycle (2048 → tetris and tetris → 2048), now wireable end-to-end through `make_per_step_success_fn` + the producer; (c) producers for the rest of the gymv envs (candy_crush, super_mario, the 13 SEGA gym_v games).
>
> **Status (2026-05-01, Day-3 of intra-gymv transfer milestone):** Real-env wiring landed. `harness.run_skill(skill, state)` now drives a real GamingAgent `make_gaming_env(...)` env via `GymvAdapter.set_executor(...)` ([`gymv_executor.py`](gymv_executor.py)), captures per-hop pre/post `StateSchema` snapshots, and surfaces a structured per-hop `effects_add` verdict on `outcome.extra["per_hop_effects"]` via [`gymv_success.py`](gymv_success.py). Phase-2 smoke driver at [`../labeling_supplement/_phase2_real_env_skill_smoke.py`](../labeling_supplement/_phase2_real_env_skill_smoke.py); summary: COMMIT/MERGE on 2048 → 3 evaluable hops, 3/3 predicates pass; COMMIT/OPTIMIZE on tetris → 4/4 pass. GamingAgent log lines confirm real env steps (`AgentAct='up', R=4.00, …`). Full results in [`../labeling_supplement/harness_io_out/_phase2_report.md`](../labeling_supplement/harness_io_out/_phase2_report.md).
>
> **Status (2026-04-30, Day-1):** Empirically confirmed at 100 % cross-contamination on `twenty_forty_eight ↔ tetris`. Probe at [`../labeling_supplement/_phase0_cross_eligibility_probe.py`](../labeling_supplement/_phase0_cross_eligibility_probe.py); fused 9-skill bank, 277 steps total. Result: every cold-start `COMMIT` / `GATHER`-typed skill from the *other* game is admitted on every step (`6/6` tetris skills on every 2048 step; `2/3` 2048 skills — both `ACTION`-typed — on every tetris step). Only `COMPARE/MERGE` (2048) is filtered, and only because it's `evidence_role=REASON → SkillType.REASONING` and no `(gymv, REASONING)` adapter is registered — orthogonal to §22. The probe measures **admission only**; the agreement-impact follow-up (does cross-game admission steal the actor's pick?) is a Day-2 extension noted in [`../labeling_supplement/harness_io_out/_phase0_report.md`](../labeling_supplement/harness_io_out/_phase0_report.md) §4. Fix lands as the additive contract change called out below.

`SOURCE_DOMAINS = ("gymv",)` has no per-game cell:

```30:36:Multi-hop-Reasoning-VLM-Agent/common/enums.py
SOURCE_DOMAINS: Tuple[str, ...] = ("gymv",)
TRANSFER_TARGET_DOMAINS: Tuple[str, ...] = (
    "browser",
    "osworld",
    "video",
    "visual_reasoning",
)
```

Every cold-start skill is tagged `applicable_domains=["gymv"]`; per-game info lives only in two free-form metadata fields:

  - `state.task = "make_gaming_env/<game>"` (parsed from `metadata.schema_canonical`),
  - `skill.provenance.source_name = "<game>"` (set by the cold-start ingester).

Neither is read by `EligibilityFilter` or `FewShotAdapter`. Three consequences:

  - **A.** `EligibilityFilter` admits a 2048-mined skill into a tetris episode — both have `state.domain == "gymv"`, the filter never reads `state.task`.
  - **B.** `FewShotAdapter.adapt(skill, target_domain="gymv", demos=[tetris_demos])` is semantically undefined — the skill *already* claims `feasible_domains=["gymv"]`, so the gate has nothing to verify. The Stage 3a transfer machinery only models cross-domain transitions.
  - **C.** `verified_domains` has domain granularity. There is no field to record "verified on tetris specifically." A skill that passes a real intra-gymv probe on tetris cannot have that fact persisted.

This blocks intra-gymv cross-game transfer experiments (2048 → tetris, candy_crush → 2048, …), which are otherwise the cheapest first transfer milestone — see the callout below.

The fix is an additive contract change: add `SkillRecord.feasible_tasks: List[str]` and `verified_tasks: List[str]`, plumb a `target_task: Optional[str] = None` parameter through `FewShotAdapter.adapt(...)` and `GateService._run_transfer(...)`, and have `EligibilityFilter` honour both `domain` and (when set) `task`. Cold-start ingestion seeds `feasible_tasks=[provenance.source_name]`.

### Intra-gymv transfer is the right first milestone

[§21](#21-cold-start-protocol-is-natural-language-prose-not-typed-hops) and [§22](#22-feasible_domains-granularity-collapses-gymv-games-into-a-single-bucket) together suggest a smaller first transfer experiment than gymv → browser/osworld. Pinning the first transfer cycle inside gymv (2048 → tetris → candy_crush → …) shrinks four of the six cost axes:

| Cost axis | gymv → browser/osworld/video/vr | gymv → gymv (cross-game) |
|---|---|---|
| Adapter executors to wire ([§1](#1-transfer-target-adapters-are-deterministic-stubs) / [§16.1](#161-adapter-executors-are-stubs-so-run_skill-is-a-black-hole)) | 4 transfer + 1 source = 5 | **1** (`GymvAdapter`) |
| New env bindings | full browser DOM, VM control, frame indexer, MCQ resolver | **none** — `cold_start/generate_cold_start_actor*.py` already drives the envs; expose its `step()` |
| Demo corpus ([§4](#4-fewshotadapter-runs-but-the-scorer-is-a-placeholder)) | doesn't exist | **already on disk** — every `labeling/skill_actions_out/.../<game>/episode_*.json` is a real rollout with state, action, intention, ground-truth `skill_query.selected_skill_id` |
| Domain-aware `success_fn` ([§4](#4-fewshotadapter-runs-but-the-scorer-is-a-placeholder)) | per target (DOM diff, screen diff, video QA, …) | **single gymv-shape scorer** keyed on consecutive `schema_canonical` blocks + `cumulative_reward` |
| Slot-binding ontology | cross-modal — `tile → DOM_node`, `direction → click`, … | gymv-internal — abstract verbs over `selectable_entity` / `container_entity` / `direction` |
| Protocol lift ([§21](#21-cold-start-protocol-is-natural-language-prose-not-typed-hops)) | needed | needed (same lift, but only over gymv-shaped skills) |

Five out of six axes are dramatically smaller. The protocol lift ([§21](#21-cold-start-protocol-is-natural-language-prose-not-typed-hops)) is the same cost either way and is the hard prerequisite — without it, `harness.run_skill(skill, state)` produces a no-op episode even on a *same-game* state. So the order is: first prove `harness.run_skill(COMMIT/MERGE, twenty_forty_eight_state)` actually executes (lift + gymv executor); then add [§22](#22-feasible_domains-granularity-collapses-gymv-games-into-a-single-bucket)'s task axis; only then run cross-game transfer probes; only then attempt cross-domain. Intra-gymv is the first place to discover real binding-failure patterns ([`PLAN-SKILL-BANK §4.3b`](../plans/04-skill-bank/PLAN-SKILL-BANK.md)) on actual data.

### Outside this package, but blocking its value

- **Actor rewire.** `decision_agents.skill_interface.SkillBankProvider` still queries the bank directly. Until it is replaced by a `HarnessSkillProvider` that wraps `SkillHarness.select_eligible_skills`, the "harness narrows + may veto" rule is not in force at runtime. Tracked in [`../IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §"Not yet delivered".
- **Real domain executors** under `vlm_wrapper/<domain>_adapter.py` (item 1's other half).

### Suggested work-order

Smallest cost / highest value first. The reordering below puts the audit's [§9–§22](#spec-contract-gaps-audit-2026-04-30) items ahead of items 1–8 because they are pure additive contract fixes (no behavioural break) and they unblock the I/O-dump driver that validates everything else.

  1. Add `SkillHarness.validate_invocation(skill, state, bindings, *, intention, reasoning_trace) -> {veto, veto_reason, diagnostic_labels}` and propagate `EligibleSkill.shadow_only` into `SkillEpisode` / `RewardLogEntry` ([§9](#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs), [§10](#10-skillepisode-artefact-field-gaps) shadow row).
  2. Extend `EligibleSkill` with the per-skill check booleans (`binding_ok / precondition_ok / evidence_ok / adapter_ok`), `fit_score`, `risk_score`, and `veto / veto_reason` for rejected candidates ([§9](#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs)). Pure additive on `eligibility.py`.
  3. Extend `SkillEpisode` with `evidence_in / evidence_out` split, `evidence_warrant / verify_verdict / reason_warrant`, `protocol_trace` (index from `episode.steps[i]` to `skill.protocol[k]`), per-key `contract_progress`, structured `reward_components`, and a list-typed `diagnostic_labels` ([§10](#10-skillepisode-artefact-field-gaps)). Pure additive on `data_structure/extensions/skill_episode.py`.
  4. Extend `SkillEvaluationRecord` with the reproducibility anchors `bank_snapshot_id`, `eval_suite_id`, `adapter_versions`, `ontology_version`, plus `status_before / status_after / rejected_domains / rollback_target` ([§11](#11-skillevaluationrecord-reproducibility-anchor-gaps)). Wire `bank_snapshot_id` through `GateService.evaluate(...)`.
  5. **[done — Day 1, 2026-04-30]** Stand up `labeling_supplement/dump_harness_io_gpt54.py` (online surface only) per [§14](#14-no-io-dump-driver--live-harness-behaviour-against-the-cold-start-corpus-is-unverified). Doubles as the integration test for items 1–4. Companion bypass probe `_phase0_cross_eligibility_probe.py` shipped alongside, since the dump driver inherits the actor's per-game RAG and cannot exhibit §22 on its own. Phase-0 baseline numbers in `labeling_supplement/harness_io_out/_phase0_report.md`.
  6. **[design locked — Day 1; impl Day 2]** Lift cold-start `protocol.steps` from prose to typed hops in the labeling pipeline — abstract verbs + `${slot}` placeholders + populated `eff_add` / `eff_del` derived from `metadata.schema_canonical` ([§21](#21-cold-start-protocol-is-natural-language-prose-not-typed-hops)). Gymv-only ontology is sufficient for the first pass. Hard prerequisite for items 7, 13, and any meaningful Stage 3a transfer. Design lock + 21-verb taxonomy + 92.5 % coverage measurement: [`../implementation_notes/legacy/protocol-lift-design.md`](../implementation_notes/legacy/protocol-lift-design.md). Implementation site: [`../labeling/_decorate_skill_records.py`](../labeling/_decorate_skill_records.py).
  7. **[Day 2]** Add the task axis — `SkillRecord.feasible_tasks` / `verified_tasks`, `FewShotAdapter.adapt(..., target_task)`, `EligibilityFilter` honours both `domain` and `task` ([§22](#22-feasible_domains-granularity-collapses-gymv-games-into-a-single-bucket)). Cold-start ingestion seeds `feasible_tasks=[provenance.source_name]`. Pure additive contract change. Empirical motivation (100 % cross-contamination on 2048 ↔ tetris) recorded in `labeling_supplement/harness_io_out/_phase0_report.md`.
  8. **[done — Day 3, 2026-05-01]** First intra-gymv real-env execution cycle — wired `GymvAdapter.set_executor(real_step)` via [`gymv_executor.py`](gymv_executor.py), and plugged a gymv-shape `success_fn` keyed on `schema_canonical`-derived `<attributes>` / `<state_flags>` facts via [`gymv_success.py`](gymv_success.py). Phase-2 smoke ([`../labeling_supplement/_phase2_real_env_skill_smoke.py`](../labeling_supplement/_phase2_real_env_skill_smoke.py)) confirms `harness.run_skill(skill, state)` actually steps `make_gaming_env("twenty_forty_eight")` / `make_gaming_env("tetris")` and surfaces a per-hop predicate verdict on `outcome.extra["per_hop_effects"]`. Full report: [`../labeling_supplement/harness_io_out/_phase2_report.md`](../labeling_supplement/harness_io_out/_phase2_report.md). Stage 3a transfer cycle (build `FewShotDemo`s from `labeling/skill_actions_out/.../<game>/episode_*.json`, run `(2048 ↔ tetris)` transfer probes) and lift v2 / VLM schema wrapper are Day-4/5 follow-ups.
  8a. **[done — Day 4, 2026-05-01]** Lift v2 — broadened `_PREDICATE_TRIGGERS` in [`labeling/_protocol_lift.py`](../labeling/_protocol_lift.py) to cover indirect phrasings real cold-start prose uses (`"valid merges were applied"` → `cumulative_reward_increased`, `"top-out"` → `phase_transitioned`, etc.). [`labeling/_decorate_skill_records.py`](../labeling/_decorate_skill_records.py) gained `--force_relift` to sweep the bank in place. Coverage delta: 2 / 12 → 9 / 12 skills with mined effects.
  8b. **[done — Day 4, 2026-05-01]** Deterministic `<state>...</state>` producer for live gym envs — [`gym_schema_producer.py`](gym_schema_producer.py) renders a structured block from `env.info` (no VLM); 2048 + tetris ship; opt-in via `make_gymv_executor(env, …, schema_producer=…)` and `initial_state_from_env(env, …, schema_producer=…)`. A/B on Phase-2 smoke (2048 `Commit/Merge`, 8 deterministic seeds): without producer best_pass_rate=0.67 (undecidable predicates inflate); with producer best_pass_rate=0.33 (only seeds where `up` produces a legal merge actually pass). The drop is the empirical signature of newly-decidable predicates. Full write-up: [`../labeling_supplement/harness_io_out/_phase3_report.md`](../labeling_supplement/harness_io_out/_phase3_report.md).
  9. ~~Plug a real domain-aware `success_fn` into `FewShotAdapter` for cross-domain (existing item 4) — generalises item 8's gymv scorer.~~ — **shipped 2026-05-02** as the 4 cross-domain success-fn modules registered against `harness.gymv_success.register_success_fn(domain)`: [`qa_success.py`](qa_success.py) (`visual_reasoning`), [`video_qa_success.py`](video_qa_success.py) (`video`), [`osworld_success.py`](osworld_success.py) (`osworld`), [`browser_success.py`](browser_success.py) (`browser`). Companion few-shot demo loaders shipped at [`few_shot_demos_vr.py`](few_shot_demos_vr.py), [`few_shot_demos_video.py`](few_shot_demos_video.py), [`few_shot_demos_osworld.py`](few_shot_demos_osworld.py), [`few_shot_demos_browsergym.py`](few_shot_demos_browsergym.py).
  10. Wire the legacy actor to `HarnessSkillProvider`. *(Note: under the T1.3 lane-(a) decision the live trainer instead consumes the eligibility + `validate_invocation` surface directly via `harness_hook` — see §22.1.)*
  11. ~~Implement action-level `ReplayValidator` (walk `seed.steps`, compare actions + evidence) — existing item 2.~~ — **shipped 2026-05-01** Day-7b at [`replay_validator.py`](replay_validator.py) (`mode="action_level"` emits per-step `StepDiff`).
  12. ~~Stand up `harness/gate_runner.py` over `gate_service` stages — existing item 3 + [§13](#13-gateservice-lives-under-orchestrator-not-harness--naming-mismatch-with-the-spec) relocate.~~ — **shipped 2026-05-01** Day-7a at [`gate_runner.py`](gate_runner.py) (`GateRunner`, `GateRunnerConfig`, `EvalSuite`).
  13. ~~Extend the dump driver to the offline GateRunner surface (Stage 0–4 + `SkillEvaluationRecord` per Crafter proposal) — second half of [§14](#14-no-io-dump-driver--live-harness-behaviour-against-the-cold-start-corpus-is-unverified).~~ — **shipped 2026-05-01** Day-7e via the `--gate-runner` flag on [`../labeling_supplement/dump_harness_io_gpt54.py`](../labeling_supplement/dump_harness_io_gpt54.py).
  14. ~~Add `rollout_batch[]` overload on `_run_shadow` and `eval_suite[]` overload on `_run_non_regression` ([§12](#12-gateservice-stage-io-signatures-dont-match-the-spec)).~~ — **shipped 2026-05-01** Day-7a as part of [`gate_runner.py`](gate_runner.py)'s `evaluate(...)` shape.
  15. Add `harness/transfer_manager.py` for shadow → active — existing item 3 second half. *(Still pending — no shadow → active two-phase quarantine yet; under lane (a) this is offline-diagnostic only.)*
  16. Wire `vlm_wrapper/<domain>_adapter.py` executors via `set_executor()` — `browser` → `osworld` → `video` → `visual_reasoning` (existing item 1). Cross-domain transfer follows the path validated by item 8. *(Partial: the harness-side **deterministic stubs** at [`video_executor.py`](video_executor.py), [`osworld_executor.py`](osworld_executor.py), [`browsergym_executor.py`](browsergym_executor.py) — plus matching schema producers [`osworld_schema_producer.py`](osworld_schema_producer.py) and [`browser_schema_producer.py`](browser_schema_producer.py) — **shipped 2026-05-02** as Phase-5/6. They identity-pass rebound contract predicates rather than calling a real env / VLM, so numbers measured against them are infrastructure-validating, not mechanism-validating — see top-of-file "Current state" callout. Real `vlm_wrapper/<domain>_adapter.py` env binding is still pending **for `browser` and `osworld` swap-in**; **for `video` and `visual_reasoning`, the `vlm_wrapper/` real-env adapters do not exist on disk yet and must be authored first** (~600-800 LOC each, mirroring `vlm_wrapper/osworld_adapter.py`).)*
  17. Phase-F LoRA heads in `select_eligible_skills` (existing item 5). *(Still pending.)*

### Additional Day-7/8/9/10 surfaces shipped (not on the original work-order)

  - `harness/gate_runner.py` (`GateRunner` + `GateRunnerConfig` + `EvalSuite`) — Day-7a; threads reproducibility anchors into every `SkillEvaluationRecord`.
  - `harness/rejected_skill_sink.py` (`RejectedSkillSink`, `FlushReport`) — Day-9c; in-process aggregator that flushes `false_binding_patterns` evidence into `SkillLifecycleManager.record_false_binding_pattern` (PLAN-SKILL-BANK §4.3b).
  - `SkillHarness.validate_invocation` + `EligibilityFilter.filter_with_rejections` — Day-8a; second-pass invocation veto + structured `RejectedSkill` channel.
  - `bind_video_executor` (in `harness/adapters/video_adapter.py`) and `bind_visual_reasoning_executor` (in `harness/adapters/visual_reasoning_adapter.py`) — Phase-5 helper re-exports so callers wire executors without importing the executor module directly.

---

## Wire-up status (audit: 2026-04-30)

[§9–§22](#spec-contract-gaps-audit-2026-04-30) catalogue what's missing on the harness's *API surface*. This section answers the orthogonal question: **given the live process today, can the harness actually be wired into the framework?** It was scoped while standing up `labeling_supplement/dump_harness_io_gpt54.py` and grepping every caller of `EpisodeRunner(`, `SkillHarness(`, `ActorAgent(`, and `PromotionOrchestrator.promote(`.

Short answer: **no for the live online runtime, yes for the offline promotion loop.** The asymmetry is structural — see [§17](#17-the-keystone-bankrunnable-is-empty-until-the-offline-loop-fires-once) for why the offline loop is a hard prerequisite for the online one.

### 15. Topology — what's already connected vs not

> **Navigation note:** §15-§20 logically precede §21-§22 (which were originally drafted in the "Spec-contract gaps" section above and grew the Day-7/8/9 status blocks during the post-Phase-5/6 status pass). Read the runtime-topology framing here *first* if you came in through the table of contents; §21 (cold-start protocol lift) and §22 (task-axis + Day-7/8/9 status) layer on top of the §16 / §17 / §18 wire-up picture below. The numerical ordering has been left as-is to avoid breaking deep links from the audit memos.

The wiring code I expected to be missing is already there. `EpisodeRunner.run` is an end-to-end harness driver:

```112:129:Multi-hop-Reasoning-VLM-Agent/orchestrator/runner.py
                eligible = self._harness.select_eligible_skills(
                    self._bank.runnable(),
                    state,
                )
                choice = self._actor.choose_action(state, eligible)
                if choice is None:
                    next_state, done = self._env.step(None)
                else:
                    budget.add_skill_invocation()
                    last_episode = self._harness.run_skill(
                        choice.skill,
                        state,
                        parent_run_id=run_id,
                        bindings=choice.bindings,
                    )
                    skill_eps.append(last_episode)
                    self._store.put_skill_episode(last_episode)
                    next_state, done = self._env.step(last_episode)
```

`GateService.evaluate(...)` likewise drives `harness.replay_validate` (Stage 1) and `FewShotAdapter.adapt → harness.run_skill` (Stage 3a). The connectors exist; the question is whether the *runtime* uses them.

| Component | Production caller? | Wired to `SkillHarness`? |
|---|---|---|
| `orchestrator.EpisodeRunner` | None — only [`../tests/test_smoke.py`](../tests/test_smoke.py) | yes (in code) |
| `decision_agents.ActorAgent` | `cold_start/generate_cold_start_actor*.py`, `decision_agents.run_actor_episode` | **no** — uses `SkillBankProvider` over legacy `skill_agents.SkillBankMVP` |
| `orchestrator.PromotionOrchestrator.promote` | None — only [`../tests/test_smoke.py`](../tests/test_smoke.py) | n/a |
| `crafter.SkillCrafterService.cycle` | None — only the dump driver and unit tests | n/a |

There is also a **name collision** in the codebase: `decision_agents/core/harness.py::Harness` is an *env wrapper* (gym-style `reset`/`step`), totally unrelated to `harness.SkillHarness`. The live `ActorAgent` already takes a `harness=` kwarg, and that kwarg means the env. Any wiring work has to disambiguate, and the spec's "harness" (this package) should probably end up renamed at the call site to keep readers from conflating the two.

### 16. Hard blockers — would silently break the runtime if flipped today

#### 16.1 Adapter executors are stubs, so `run_skill` is a black hole

All five registered adapters fall back to `_deterministic_executor` from `adapters/_stub_base.py` when no real env-step callback is registered. `gymv_wrapper` exposes `set_executor(...)` but **nothing in production calls it**. If the live actor is wired to `harness.run_skill` today, the env doesn't advance — the adapter fabricates a plausible `SkillEpisode` and returns. This would catastrophically regress every cold-start rollout. Same root cause as [§1](#1-transfer-target-adapters-are-deterministic-stubs); restated here because it's the #1 risk for *online* wiring specifically.

#### 16.2 `EnvLike` protocol mismatch with production gym envs

`EpisodeRunner.run` requires `env.step(episode: Optional[SkillEpisode]) -> (StateSchema, bool)` — i.e. the env consumes a typed `SkillEpisode` and returns a typed `StateSchema`. None of the production envs under `env_wrappers/` match that shape; they take primitive actions and return text observations. The smoke test passes only because the smoke env is bespoke. A real wire-up needs an `EnvLike` shim per env (gymv first).

#### 16.3 Bank-pointer mismatch

The live actor's `decision_agents.skill_interface.SkillBankProvider` reads `skill_agents.SkillBankMVP` (legacy four-stage bank). `SkillHarness` reads `skill_bank.SkillRepository` (new typed four-store). The planned `skill_bank/legacy_bridge.py` (status: not yet delivered, see [`../IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) §"Not yet delivered") is what closes that gap. Without it, even a correctly-wired `HarnessSkillProvider` would point at an empty new-bank while the legacy bank does the real work.

#### 16.4 `ActorLike` protocol mismatch with `decision_agents.ActorAgent`

`EpisodeRunner` expects `ActorLike.choose_action(state: StateSchema, eligible: List[EligibleSkill]) -> Optional[ActorChoice]` (`orchestrator/runner.py:38–46`), where `ActorChoice` carries a typed `SkillRecord` + `bindings`. The current `decision_agents.ActorAgent.run` returns text actions and consumes text observations. They'd need an adapter wrapper before the runner can drive the live actor.

#### 16.5 `record_outcome` shape mismatch with `RewardLogger`

The actor-side `SkillProvider.record_outcome(skill_id, *, outcome, reward, steps_taken, info)` (`decision_agents/skill_interface.py:179–196`) is per-attempt scalar feedback. `RewardLogger` and the gate's Stage 2 expect a full `SkillEpisode`. Loss of fidelity unless the actor produces a real `SkillEpisode` itself — which depends on [§10](#10-skillepisode-artefact-field-gaps) landing first.

### 17. The keystone — `bank.runnable()` is empty until the offline loop fires once

`SkillRepository.runnable()` reads from the `active` store only:

```51:57:Multi-hop-Reasoning-VLM-Agent/skill_bank/repository.py
    def runnable(self, *, include_shadow: bool = True) -> List[SkillRecord]:
        out: List[SkillRecord] = []
        for r in self.active.all():
            if r.status == SkillStatus.SHADOW and not include_shadow:
                continue
            out.append(r)
        return out
```

The active store holds statuses `ACTIVE` and `SHADOW`. Cold-start banks ingest skills as `CANDIDATE` (in the `candidate` store). Until the offline promotion loop fires *at least once* and graduates a skill across `CANDIDATE → SHADOW`, `EpisodeRunner` sees `[]`, the eligibility filter has nothing to filter, and every online step is a no-op.

This is the most consequential finding from the audit: **the offline promotion path is a hard prerequisite for the online runtime**, not the other way around. Even if [§16](#16-hard-blockers--would-silently-break-the-runtime-if-flipped-today) were solved tomorrow, the runner would still be skill-starved.

The lifecycle's `feasible_domains ≥ 2` check (`skill_bank/lifecycle.py:253`) only fires on transitions to `ACTIVE`. Single-domain cold-start skills can therefore still reach `SHADOW` — sufficient to populate `bank.runnable(include_shadow=True)` — even before Phase D transfer arenas exist. So one offline cycle on the dump driver's outputs really is enough to unblock the online path *for shadow-mode rollouts*.

### 18. The one wire-up that's safe today — offline promotion

```
labeling_supplement/dump_harness_io_out/<run>/<corpus>/<source>/proposal_*/evaluation.json
        ↓ (new driver, ~150 LOC)
PromotionPlan: [(skill, target_status_from_verdict, evaluation, rationale), ...]
        ↓
PromotionOrchestrator.promote(plan, repository=test_repo)
        ↓
test_repo now has SHADOW / PROVISIONAL skills
        ↓
bank.runnable() returns non-empty
        ↓
EpisodeRunner has something to filter
```

Properties:

  - Reads files only; doesn't touch the live actor or any env.
  - Operates on a test `SkillRepository`, not production state.
  - Surfaces real orchestrator-side breakage (`content_hash` drift, atomic-write conflicts, snapshot-reference issues) on actual data, not stubbed records.
  - Failure mode is loud: a malformed promotion raises `LifecycleError`; nothing in production changes.
  - Is the **only** sequence that makes the online loop meaningful afterward, per [§17](#17-the-keystone-bankrunnable-is-empty-until-the-offline-loop-fires-once).

The natural location is a sibling to the dump driver under [`../labeling_supplement/`](../labeling_supplement/) — e.g. `promote_evaluations_gpt54.py` — mirroring the same dispatcher pattern.

### 19. Recommended sequencing to a fully-wired online runtime

| Stage | What | Risk | Rough effort | Unblocks |
|---|---|---|---|---|
| **0** | Offline promotion driver — consume `evaluation.json` → `PromotionOrchestrator.promote` against a test repo ([§18](#18-the-one-wire-up-thats-safe-today--offline-promotion)) | Low | 1–2 days | Skills reach SHADOW; runner has data ([§17](#17-the-keystone-bankrunnable-is-empty-until-the-offline-loop-fires-once)) |
| **1** | Additive contract fixes — `SkillEpisode` ([§10](#10-skillepisode-artefact-field-gaps)), `SkillEvaluationRecord` reproducibility anchors ([§11](#11-skillevaluationrecord-reproducibility-anchor-gaps)), `EligibleSkill` scoring ([§9](#9-online-surface-api-gaps-validate_invocation-scoring-intentionactive_skill-inputs)) | Low | 5–7 days | Online wiring is *meaningful* when it lands |
| **2** | Additive harness API — `validate_invocation` (§9.1), threaded `intention / active_skill / local_reasoning_trace` (§9.2) | Low | 4–5 days | Online actor can pre-flight check |
| **3** | Real adapter executor for `gymv` via `gymv_wrapper.set_executor(...)`; replace deterministic-stub fallback with explicit `ABORT` so silent stubs can't escape ([§16.1](#161-adapter-executors-are-stubs-so-run_skill-is-a-black-hole)). Pre-requires the protocol lift ([§21](#21-cold-start-protocol-is-natural-language-prose-not-typed-hops)) — without it `harness.run_skill` produces no-op episodes even on same-game state. Pairs with the task axis ([§22](#22-feasible_domains-granularity-collapses-gymv-games-into-a-single-bucket)) to enable the intra-gymv transfer milestone before cross-domain ([Intra-gymv transfer is the right first milestone](#intra-gymv-transfer-is-the-right-first-milestone)). | Medium | 7–10 days | `run_skill` actually moves the env |
| **4** | `EnvLike` shim for gymv ([§16.2](#162-envlike-protocol-mismatch-with-production-gym-envs)); `skill_bank/legacy_bridge.py` for the bank-pointer flip ([§16.3](#163-bank-pointer-mismatch)) | Medium | 4–5 days | Live actor reads the new bank; runner can step real envs |
| **5** | `HarnessSkillProvider` + actor wired in **shadow mode** (legacy provider still authoritative; harness disagreements logged, not acted on) | Low | 4–5 days | Diagnostic signal pre-flip |
| **6** | Actor flip — `HarnessSkillProvider` becomes authoritative; cold-start scripts call `EpisodeRunner` directly; actor produces real `SkillEpisode` ([§16.4](#164-actorlike-protocol-mismatch-with-decision_agentsactoragent), [§16.5](#165-record_outcome-shape-mismatch-with-rewardlogger)) | Higher | 7–10 days | Live online wiring |

Total: ~6–8 weeks. Stage 0 alone gives end-to-end signal in a day or two and is the prerequisite for everything that depends on `bank.runnable()` returning non-empty.

The "Suggested work-order" in [§Suggested work-order](#suggested-work-order) above is harness-internal; this table layers the runtime-topology dimension on top of it. Stage 0 here corresponds to no item there (it's an orchestrator-side driver). Stages 1–2 here ≈ items 1–4 there. Stages 3–6 here are mostly items 7 + 13 there but explicitly sequenced.

### 20. Bottom line

> **Reconciliation note (post-Day-10, T1.3 lane decision):** This section was written for the lane-(b) world where the harness's `run_skill` / typed-executor surface is on the live trainer's critical path. Day-10's [`../implementation_notes/legacy/skill-lane-decision.md`](../implementation_notes/legacy/skill-lane-decision.md) closed lane (a) — skill = retrieval payload — so the live trainer wires the **eligibility + `validate_invocation`** surface only (see §22.1 / §22.5). The "offline loop is the keystone" framing below is still correct *for the lane-(b) regression suite and the offline diagnostic stack* (`labeling_supplement/`, `tests/`), and §22.5 explicitly preserves §16 work as offline diagnostic to be flipped back to "next milestone" only if the rollback condition in §4 of the lane decision fires. Read the bullets below as: "for any consumer that *does* invoke `run_skill` (the gate, the dump driver, the lane-(b) regression suite), the offline promotion loop is the keystone." For the live trainer hot path, §22.5's lane-based framing supersedes — `run_skill` is off the critical path.

  - **The hard part of harness wiring is already in [`../orchestrator/runner.py`](../orchestrator/runner.py) and [`../orchestrator/gate_service.py`](../orchestrator/gate_service.py).** Those code paths are correct in shape, exercised by [`../tests/test_smoke.py`](../tests/test_smoke.py) and the dump driver, and were the riskiest thing to get right.
  - **What's missing is not the wiring but the things on either end** — a real adapter executor (so `run_skill` advances the env), a populated `active` store (so `bank.runnable()` returns anything), and a typed `EnvLike` shim per environment.
  - **For lane-(b) consumers, the offline promotion loop is the keystone.** Until it fires once on real data, those consumers are *structurally* skill-starved. Flipping the lane-(b) harness in today would just make every actor step a no-op (best case) or a stub-driven black hole (worst case, if `run_skill` is invoked). Lane-(a) consumers (the live co-evolution trainer) are unaffected — they read the bank directly via the eligibility filter without invoking `run_skill`.

---

## §22 Trainer integration (Day-10) — what's wired into the co-evolution loop

> **Lane decision: closed (T1.3, lane (a) — skill = retrieval payload).**
> Spec: [`../implementation_notes/legacy/skill-lane-decision.md`](../implementation_notes/legacy/skill-lane-decision.md).
> The harness's role in the live trainer is exactly the **eligibility +
> `validate_invocation`** surface described in this section — i.e. it
> filters / vetoes which retrieval payload the actor LLM consults; it
> does **not** call `run_skill` and it does **not** execute typed
> protocols. Lane-(b) machinery in this README (action-level
> `ReplayValidator`, `GateRunner` Stages 1-4, the few-shot adapter)
> remains live in the **offline diagnostic** stack
> (`labeling_supplement/`, `tests/`) but does not fire on the
> live-trainer hot path. To re-enable the lane-(b) protocol-edit path
> in the live Crafter, pass `--enable-protocol-patching` to
> `scripts/run_coevolution.py` (default off — see T1.3a).

The runtime in [`../orchestrator/runner.py`](../orchestrator/runner.py) is one of two consumers of this harness; the other is the **co-evolution training loop** in [`../trainer/coevolution/orchestrator.py`](../trainer/coevolution/orchestrator.py). Day-10 plugs the harness's two **LLM-free** surfaces into the live training rollouts and feeds the resulting rejection signal back into the existing Crafter hook so the offline-mirror loop now has a *live-trainer* counterpart.

### 22.1 Topology — what's connected today

```
  ┌── Phase A (rollout) ───────────────────────────────────────────┐
  │   episode_runner.run_episode_async(... harness_hook=hook)      │
  │     │                                                          │
  │     ├─ get_top_k_skill_candidates(...)            (RAG)        │
  │     │           │                                              │
  │     ├─ hook.filter_candidates(records, state)                  │
  │     │     └── EligibilityFilter.filter_with_rejections         │
  │     │             ├── admitted ─▶ skill_selection LLM picks    │
  │     │             └── rejected ─▶ RejectedSkillSink.observe()  │
  │     │                                                          │
  │     ├─ hook.validate_choice(skill_id, state)                   │
  │     │     └── SkillHarness.validate_invocation                 │
  │     │             ├── ok=True   ─▶ proceed                     │
  │     │             └── ok=False  ─▶ fall back to next eligible  │
  │     │                                                          │
  │     └─ env.step(...) (unchanged)                               │
  └────────────────────────────────────────────────────────────────┘
                  │
                  ▼
  ┌── Phase B′ (Crafter) ──────────────────────────────────────────┐
  │   _crafter_hook.run_crafter_step(... harness_hooks=hooks)      │
  │     │                                                          │
  │     ├─ _seed_repo_from_legacy_jsonl  →  ephemeral lifecycle    │
  │     │                                                          │
  │     ├─ harness_hook.flush_to_lifecycle(lifecycle)              │
  │     │     └── RejectedSkillSink.flush_to                       │
  │     │             └── lifecycle.record_false_binding_pattern   │
  │     │                     ↳ writes SkillRecord.false_binding_  │
  │     │                       patterns                           │
  │     │                                                          │
  │     ├─ SkillCrafterService.reflect_on_episode (existing)       │
  │     │     └── Repairer reads false_binding_patterns ──▶ emits  │
  │     │         PatchProposal records that previously had no     │
  │     │         live signal to fire on                           │
  │     │                                                          │
  │     └─ writes proposals.jsonl ─▶ Phase B′ Promotion subprocess │
  └────────────────────────────────────────────────────────────────┘
```

### 22.2 What this *does not* do

This wire-up deliberately stops at the eligibility + validate surfaces. Specifically, it does **not**:

1. Call `harness.run_skill(...)` from the trainer. The episode runner still drives the env directly via primitive actions through `action_taking` LoRA (see [§16.1 / §16.2](#161-adapter-executors-are-stubs-so-run_skill-is-a-black-hole)). Plumbing `run_skill` requires a real `gymv` `set_executor(env_step_fn)` plus an `EnvLike` shim per env wrapper — the same multi-day env-binding work tracked in §16.
2. Persist any status mutation. Skills hydrated from the live `skill_bank.jsonl` are mounted as a *runtime view* with `status=PROVISIONAL` (so the F1 status check admits them); the `SkillLifecycleManager` remains the only authority that may write status to disk (PLAN-SKILL-BANK §0.5).
3. Add LLM calls. Both `EligibilityFilter` and `validate_invocation` are deterministic CPU paths (microseconds per step). The trainer's existing 2-LLM-calls-per-env-step budget (intention + skill-selection + action) is unchanged.

### 22.3 CLI surface

```
python scripts/run_coevolution.py \
    --crafter-promotion-enabled \
    --harness-enabled \
    [--no-harness-allow-shadow]
```

Both flags default off / permissive, so existing runs are byte-identical.

### 22.4 Spec gaps this closes for the *trainer* (subset of §9 / §22)

| Gap | Trainer status |
| --- | --- |
| §9.1 second-pass `validate_invocation` | **closed** in trainer's live rollout (per-step) |
| §9.3 per-check booleans on `EligibleSkill` | **closed** (logged into `experiences[].harness.filter[].rejected[]`) |
| §22 task-axis F2′ | **closed** for trainer Phase A (state.task = game name) |
| PLAN-SKILL-BANK §4.3b `false_binding_patterns` from live signal | **closed** via `RejectedSkillSink → record_false_binding_pattern` in Phase B′ |

### 22.5 Skill-selection design — RAG retrieves, **harness** *informs*, LLM picks, harness validates

> **Status (2026-05-02): both refinements landed.** The four-stage
> selection pipeline below is what runs in the live trainer + the
> standalone `qwen3_decision_agent`; the two harness-derived signals
> (`_harness_deboost`, `_harness_adaptation_score`) are decorated onto
> the candidate dicts the `skill_selection` LLM sees.

A recurring design question: *should we let the harness (with its
deterministic predicate checks) replace the RAG retriever and pick
skills directly, or should we keep RAG as the picker and let the
harness only veto?* Neither extreme is right.

* **RAG-only** is what the legacy stack did. It retrieves on dense
  state-text similarity, which is excellent for "which existing skill
  resembles this state". But RAG can't see the skill's runtime
  contract (predicates, `feasible_tasks`, adapter availability), so
  it routinely top-Ks skills the harness will veto microseconds
  later — wasting the LLM's `skill_selection` budget on dead options.
* **Harness-as-picker** is appealing — the harness has perfect
  ground-truth on contract validity — but it has no notion of
  *strategic relevance*. A skill that's runnable on every state is
  not the right pick on every state; the LLM is the only component
  in the loop that can read a state's narrative ("the player is
  trapped in a corner") and match it to a skill's strategic
  description. The harness is also far less expressive than RAG's
  embedding search at scale (≥10⁴ skills).

The right division of labour is the **four-stage pipeline**:

```
   state ──▶ RAG retriever (top-K)        # scalable similarity
              │
              ▼
         Harness eligibility filter       # deterministic vetoes
              │     ├─ predicate translator (Layer C)
              │     ├─ task / domain / adapter / can_handle checks
              │     └─ veto sink ─▶ false_binding_patterns (durable)
              ▼
         skill_selection LLM picks ONE    # interpretive, with priors
              │
              ▼
         Harness.validate_invocation       # post-pick second pass
              │     ├─ ok=True  ─▶ proceed
              │     └─ ok=False ─▶ next eligible candidate
              ▼
            env.step
```

The harness is the **referee**, not the picker. The two refinements
below are how the referee teaches the picker without replacing it.

#### Refinement A — `false_binding_patterns` deboost the RAG ranker

Module: [`harness/rejection_deboost.py`](rejection_deboost.py).
Wired into [`scripts/qwen3_decision_agent.get_top_k_skill_candidates`](../scripts/qwen3_decision_agent.py).

When the harness vetoes a skill, the rejection sink already aggregates
the `(veto, domain, task)` triple onto
`SkillRecord.false_binding_patterns` (PLAN-SKILL-BANK §4.3b). Refinement
A reads that durable history at retrieval time and multiplies the
candidate's `confidence` / `relevance` by a deboost factor in
`[0.10, 1.0]` derived from:

* On-axis count vs. half-life (default 3 vetoes ⇒ 0.5×).
* Recency weighting (default 1-day half-life — ancient vetoes weigh
  less so a rehabilitated skill can rebound).
* An off-axis discount (vetoes from a *different* `(domain, task)`
  contribute at 0.25× by default — they're weak evidence on the
  current axis).

The factor is also stamped on the candidate dict as
`_harness_deboost`, so downstream consumers (the prompt formatter,
audit logs, GRPO records) can see how aggressively a skill was
deboosted. Configurable via the `apply_rejection_deboost=False`
opt-out on `get_top_k_skill_candidates`.

Test surface: [`tests/test_rejection_deboost.py`](../tests/test_rejection_deboost.py)
(27 unit tests covering pure scoring, end-to-end through
`get_top_k_skill_candidates`, fetcher errors, opt-out, and floor
clamping).

#### Refinement B — Adaptation score injected into the prompt

Module: [`trainer/coevolution/_harness_hook.py`](../trainer/coevolution/_harness_hook.py)
(`_compute_adaptation_score`). Surfaced in both copies of
`_format_candidates_for_selection` (the standalone agent and the
trainer's mirror).

Each admitted candidate now carries a numeric
`_harness_adaptation_score ∈ [0, 1]` that summarises *how well the
harness expects this skill to adapt to the current `(domain, task)`*.
The score is the arithmetic mean of three components:

| Component | 1.0 case | Weakened case |
| --- | --- | --- |
| **Task-axis match** | `task_match == "verified"` | `same_task = 0.85`, `agnostic = 0.60` |
| **Adapter-target fit** | `adapter_name == state.domain` (native) | bridged adapter ⇒ 0.70 |
| **Predicate translation** | diagonal cell — no rewrite needed | rewritten ⇒ 0.85, identity-fallback (translator crash) ⇒ 0.55 |

The score appears in the `skill_selection` prompt as `Adaptation: 0.83`
between the candidate's `Confidence:` line and (when meaningful) a
`Recent veto rate: 0.45` line derived from `1 - _harness_deboost`. The
LLM sees both signals as a structured prior — it's free to override
them, but it now picks with more information than dense embeddings
alone provide.

Vetoed candidates do not appear in the filtered list at all (the
harness already removed them); unknown-to-cache candidates pass
through unchanged with no score (the harness has no opinion). The
candidate-level breakdown (per-component scores + translation status)
is written to `_harness_adaptation_breakdown` for offline inspection
but is *not* surfaced in the prompt — it would be redundant noise.

Per-step summary stats land in the `experiences[].harness` payload:
`adaptation_score_min / max / mean` so wandb / TB can trend the
moments.

Test surface: 11 new tests in [`tests/test_trainer_harness_hook.py`](../tests/test_trainer_harness_hook.py)
covering the score range, monotonicity (`verified > same_task >
agnostic`), diagonal-vs-translated-vs-failed translation status,
absent-on-unknown-skills, prompt-formatter rendering, and the
trainer/standalone formatter parity.

#### What this design preserves

* **Single picker.** The LLM is still the only component that picks.
  The harness emits *priors*, not decisions.
* **Architectural composability.** Refinements A and B work
  independently — the deboost is a RAG-time signal, the adaptation
  score is a filter-time signal. Either can be opted out without
  affecting the other (though they're complementary in practice).
* **Backwards-compatible prompts.** Both signals are best-effort:
  candidates assembled outside the harness path simply omit the
  fields and the prompt formatter degrades silently. No regression
  for callers that don't run the harness hook.
* **No new LLM calls.** Both refinements are pure CPU. The trainer's
  per-step LLM budget is unchanged (intention + skill-selection +
  action).

The Crafter loop's [`_crafter_hook → flush_to_lifecycle`](../trainer/coevolution/_crafter_hook.py)
remains the canonical writer for `false_binding_patterns`, so the
deboost in Refinement A automatically picks up on-the-fly veto
history within the same trainer step.

### 22.6 What the trainer integration leaves for §9 / §16 / §22

- §9.2 planner-context (`intention / active_skill / local_reasoning_trace`) is *plumbed* into `state.extra` but not yet a typed first-class param of `select_eligible_skills` — the harness API still takes `state` only.
- §9.3 numeric `fit_score / risk_score` head — still pending (LoRA scoring, PLAN-SKILL-BANK §0.3 Clause D).
- §16.1–§16.5 — unchanged under lane (a). The `run_skill` / typed
  executor work is no longer on the critical path because skills
  are retrieval payloads (no inner-MDP execution by the runtime).
  The §16 work remains in tree as **offline diagnostic** for the
  Stage-3a transfer cycle and the lane-(b) regression suite; flip
  it back to "next milestone" only if the rollback condition in
  [`../implementation_notes/legacy/skill-lane-decision.md`](../implementation_notes/legacy/skill-lane-decision.md) §4 fires (retrieval ceiling hit *and* MCTS / tool-augmented escalations exhausted).

---

## Cross-references

- Root [`readme.md`](../readme.md) §"Architecture" — where the harness sits in the four-stage pipeline.
- [`../plans/05-harness/PLAN-HARNESS.md`](../plans/05-harness/PLAN-HARNESS.md) — full spec, gate stack G0–G5.
- [`../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) — `GateRunner` stages and aggregation contract that drives [§11–§13](#11-skillevaluationrecord-reproducibility-anchor-gaps).
- [`../skill_bank/README.md`](../skill_bank/README.md) — invariant 8 ties `FewShotAdapter` to `SkillLifecycleManager.record_transfer_verification`.
- [`../implementation_notes/legacy/crafter-harness-orchestrator-roles.md`](../implementation_notes/legacy/crafter-harness-orchestrator-roles.md) — three-role I/O contract; §3 cheat sheet enumerates the artefact families [§10–§11](#10-skillepisode-artefact-field-gaps) extend; §7.1 mismatch #2 motivates [§10](#10-skillepisode-artefact-field-gaps)'s `protocol_trace` row.
- [`../labeling_supplement/`](../labeling_supplement/) — sibling location for the `dump_harness_io_gpt54.py` driver in [§14](#14-no-io-dump-driver--live-harness-behaviour-against-the-cold-start-corpus-is-unverified).
- [`../tests/test_smoke.py`](../tests/test_smoke.py) — runnable end-to-end wiring example.

### Phase-5/6 cross-domain dispatch and measurement

- [`../labeling_supplement/_phase4_target_dispatch.py`](../labeling_supplement/_phase4_target_dispatch.py) — central per-target dispatcher (Stage 4 / Phase-5/6); routes `gymv → {visual_reasoning, video, osworld, browser}` through the matching adapter + executor + schema producer + success_fn quad.
- [`../labeling_supplement/_phase5_matrix.py`](../labeling_supplement/_phase5_matrix.py) — Stage 5 within-VR / within-video 4×4 driver.
- [`../labeling_supplement/_phase4_transfer_matrix.py`](../labeling_supplement/_phase4_transfer_matrix.py) — Stage 6 NxN cross-domain transfer driver.
- [`../labeling_supplement/_phase4_transfer_report.py`](../labeling_supplement/_phase4_transfer_report.py) — Stage 6 report generator (G1-G6 verdicts).
- [`../implementation_notes/legacy/phase5-cross-domain-measurement.md`](../implementation_notes/legacy/phase5-cross-domain-measurement.md) — Phase-5/6 plan memo; §11.5.0 reconciles the deterministic-stub pathology with the §11.5.4 aspirational transferability bands. **Required reading** for any consumer interpreting the Stage 5 / Stage 6 numerical outputs.
- [`../implementation_notes/cross-domain-transfer-suite-rollout.md`](../implementation_notes/cross-domain-transfer-suite-rollout.md) §11.5 — transferability assessment across the 5-domain matrix.
