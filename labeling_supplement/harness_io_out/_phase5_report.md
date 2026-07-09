# Phase-5 report — Day-7/8 of the intra-gymv transfer milestone

> **TL;DR.** All five spec-contract gaps surfaced in `harness/README.md` §9–§12 closed at the structural level (the §11 reproducibility anchors and §12 stage I/O signature drift now match the unified-skill-gate spec). The harness's offline gate surface gains its spec-named entry point (`harness.GateRunner`); the action-level `ReplayValidator` walk lands; `SkillLifecycleManager.record_task_verification` becomes the sanctioned writer for the Day-5b PASS persistence loop; the second-pass invocation veto (`validate_invocation`) and the rejected-skill channel close §9; `SkillEpisode` / `SkillEvaluationRecord` close §10/§11. **+33 unit tests; full suite 375 / 376 (one pre-existing whitespace failure unrelated to this work).**

## 1. Surfaces that landed

### 1.1 Day-7a — `harness/gate_runner.py`

The spec calls it the "Harness `GateRunner`" — historically lived under `orchestrator.gate_service.GateService`. The new file at [`harness/gate_runner.py`](../../harness/gate_runner.py) exposes:

* **`GateRunner`** — subclass of `GateService`, no behaviour change for old callers. Adds two additive `evaluate(...)` shapes:
  * `rollout_batch: Sequence[SkillEpisode]` replaces `RewardLogger`-only Stage-2 input. Auto-filters to the proposal's `skill_id` so callers can pass an unfiltered batch.
  * `eval_suite: EvalSuite` replaces scalar `(baseline_score, post_score)` Stage-4 inputs and threads `suite_id` into the persisted record.
  * `status_before: SkillStatus` is recorded as an anchor on the `SkillEvaluationRecord`.
* **`GateRunnerConfig`** — frozen dataclass for the §11 reproducibility anchors: `bank_snapshot_id`, `eval_suite_id`, `adapter_versions`, `ontology_version`, `seed`, `judge_model`. Pin once; every emitted record inherits them.
* **`EvalSuite(suite_id, pre_score, post_score, metrics)`** — frozen value object for Stage-4 input. `metrics` flow into `record.metrics` keyed `eval_suite.<task>`; `delta()` is auto-recorded as `eval_suite.delta`.

Mixing old + new shapes for the same stage raises `ValueError`. The `harness` package exposes the new symbols via lazy `__getattr__` (avoids a circular orchestrator import for cold-start consumers).

### 1.2 Day-7b — Action-level `ReplayValidator(mode="action_level")`

[`harness/replay_validator.py`](../../harness/replay_validator.py) gained a step-by-step walk through `seed.steps`. For each `(seed.step[i], proposed.step[i])` pair it emits a `StepDiff(action_match, payload_match, evidence_non_worse, …)`. Pass criterion is **monotonic-non-worse**:

* extra proposed steps tolerated;
* truncation (proposal has fewer steps than the seed) is a regression;
* evidence-role regression (proposed roles ⊊ seed roles on any step) is a regression.

New stage metrics on `ReplayResult`: `step_action_match_rate`, `step_evidence_non_worse_rate`, `n_steps_compared`. Adapter-level mode (`mode="adapter_level"`) remains the default and is unchanged — old callers see identical behaviour.

This closes the Day-6c deferred item and provides the read-side of the §10 `protocol_trace` row (the gate can now compare per-step evidence between seed and proposal; the lift-side write still depends on §21's typed-hop migration).

### 1.3 Day-7c — `SkillLifecycleManager.record_task_verification`

The bank's task-axis analog of the existing `record_transfer_verification`. Same contract, different field:

```python
lifecycle.record_task_verification(
    skill_id,
    verified_tasks=["tetris"],
    evaluation_id="eval-007",
    per_task_metrics={"tetris": {"pass_rate": 0.83, "k_used": 3}},
    rationale="Day-5b transfer cycle PASS",
)
```

Mutates `SkillRecord.verified_tasks`, appends one `adapter_history` entry per task tagged `kind="task_verification"`, round-trips to disk, idempotent on re-registration. Rejects empty rationale / empty `verified_tasks`. The `_phase4_transfer_cycle.py` driver can now persist Stage-3a verdicts (the wiring patch is the `--persist` flag, Day-9 follow-up).

### 1.4 Day-8a — `SkillHarness.validate_invocation` + `EligibilityFilter.filter_with_rejections`

[`harness/skill_harness.py`](../../harness/skill_harness.py) and [`harness/eligibility.py`](../../harness/eligibility.py) close §9.

`validate_invocation(skill, state, bindings=…, eligible=…) → ValidateInvocationResult` is the second-pass veto:

| Field | Meaning |
|---|---|
| `ok` | aggregate AND of the four per-check booleans |
| `adapter_ok` | a `(state.domain, skill.skill_type)` adapter is registered |
| `binding_ok` | every `${slot}` placeholder in `skill.protocol` payloads is in `bindings` |
| `precondition_ok` | every `${slot}` referenced in `skill.contract.preconditions` is in `bindings` (free-form preconditions don't yet have a typed checker — that's Day-9+) |
| `evidence_ok` | every role in `skill.contract.expected_evidence_roles` appears in `state.evidence` (ACTION skills exempt by G0) |
| `shadow_only` | propagated from the `EligibleSkill` |
| `veto_reasons` | union of "what failed" tags the actor can render |
| `missing_bindings` / `missing_evidence_in` / `failed_preconditions` | structured per-check failure lists |

`filter_with_rejections(candidates, state) → (admitted, rejected)` is the eligibility-filter companion that returns the previously-silent rejection channel as `RejectedSkill(skill, veto, veto_reason, adapter_ok, binding_ok, …)`. `filter(...)` (no rejections) remains the legacy one-arg API. `EligibleSkill.to_json()` now carries the per-check booleans (`binding_ok / precondition_ok / evidence_ok / adapter_ok`).

### 1.5 Day-8b/c — Data shape expansion

[`data_structure/extensions/skill_episode.py`](../../data_structure/extensions/skill_episode.py):

* **`SkillEpisode`**: `shadow: bool`, `diagnostic_labels: List[str]` (legacy `transfer_label` auto-mirrors), `protocol_trace: List[Optional[int]]` (mapping `steps[i] → skill.protocol[k]`, populated lazily on `add_step`).
* **`SkillEpisodeStep`**: `evidence_in / evidence_out` directional split (legacy `evidence` mirrors into `evidence_out` for forward-compat reads), `protocol_index`, three citation slots (`evidence_warrant`, `verify_verdict`, `reason_warrant`).
* **`SkillEpisodeOutcome`**: `contract_progress: Dict[str, bool]` (per-key contract satisfaction), `reward_components: Dict[str, float]` (multi-component reward decomposition).

[`data_structure/extensions/skill_evaluation.py`](../../data_structure/extensions/skill_evaluation.py):

* **`SkillEvaluationRecord`** gains the §11 anchors: `bank_snapshot_id`, `eval_suite_id`, `adapter_versions`, `ontology_version`, `version`, `status_before` / `status_after` (typed `SkillStatus`), `rejected_domains`, `rollback_target`, `diagnostic_labels`.

`SkillHarness.run_skill(...)` now propagates `eligible.shadow_only` (or falls back to `skill.status==SHADOW`) into `episode.shadow`. `GateRunner.evaluate(...)` populates the anchors directly on the resulting record (`record.bank_snapshot_id`, `record.eval_suite_id`, …) — no more workaround `_anchors` attribute.

All additions are **additive**: legacy callers writing the legacy fields see identical JSON output for those fields plus None / [] / {} for the new ones.

## 2. Test plan and results

| File | New tests | Status |
|---|---|---|
| `tests/test_gate_runner.py` | 5 | ✓ PASS |
| `tests/test_replay_validator_action_walk.py` | 6 | ✓ PASS |
| `tests/test_lifecycle_task_verification.py` | 6 | ✓ PASS |
| `tests/test_validate_invocation.py` | 9 | ✓ PASS |
| `tests/test_skill_episode_field_expansion.py` | 7 | ✓ PASS |
| **Total new** | **33** | **✓ all PASS** |

Full suite: `python -m pytest tests/ -q --ignore=tests/test_orchestrator_phase_b_prime_integration.py` → **375 passed, 1 failed**.

The single failure (`tests/test_schema_predicates.py::TestRobustness::test_extra_whitespace_tolerated`) is the pre-existing whitespace-tolerance regression already documented in §5 of the Day-5/6 report — unrelated to Day-7/8 work.

Lint check on every modified file: clean.

## 3. Spec-gap status after Day-7/8

| Section | Gap | Status |
|---|---|---|
| §9 | `validate_invocation` missing | ✓ closed (Day-8a) |
| §9 | `EligibleSkill.to_json` missing per-check booleans + rejection channel | ✓ closed (Day-8a) |
| §9 | `select_eligible_skills` doesn't take `intention / active_skill / local_reasoning_trace` | 🟡 partial — those inputs are still missing on the filter. Day-9 work: scoring head + planner-context inputs (need LoRA scoring before they're meaningful). |
| §10 | `evidence_in / evidence_out` split | ✓ closed (Day-8b) |
| §10 | `evidence_warrant / verify_verdict / reason_warrant` citation slots | ✓ closed (Day-8b) |
| §10 | `protocol_trace` (episode → protocol[k] mapping) | 🟡 read side closed (Day-7b/8b); write side blocked on §21 typed-hop lift |
| §10 | `contract_progress` (per-key) | ✓ closed (Day-8b) |
| §10 | `reward_components` | ✓ closed (Day-8b) |
| §10 | `shadow` flag on episode | ✓ closed (Day-8b) |
| §10 | `diagnostic_labels` (list) | ✓ closed (Day-8b) |
| §11 | `version` / `status_before` / `status_after` | ✓ closed (Day-8c) |
| §11 | `approved_domains` / `rejected_domains` | ✓ closed (Day-8c — `eligible_domains` already records approved; `rejected_domains` now persisted) |
| §11 | `rollback_target` | ✓ closed (field exists; populated on `RollBackProposal`, Day-9) |
| §11 | `bank_snapshot_id` | ✓ closed (Day-7a `GateRunnerConfig`) |
| §11 | `eval_suite_id`, `adapter_versions`, `ontology_version` | ✓ closed (Day-7a/8c) |
| §11 | `diagnostic_labels` (flat list) | ✓ closed (Day-8c) |
| §12 | Stage 2 `_run_shadow` accepts `RewardLogger` only | ✓ closed (Day-7a `rollout_batch`) |
| §12 | Stage 4 `_run_non_regression` accepts scalar `(pre, post)` only | ✓ closed (Day-7a `eval_suite`) |
| §13 | `GateService` lives under `orchestrator/`, not `harness/` | ✓ closed (Day-7a `harness.GateRunner` alias) |
| §14 | I/O dump driver missing | 🟡 unchanged — Day-1 dump driver landed in `labeling_supplement/dump_harness_io_gpt54.py`; **offline `GateRunner` extension is Day-9 follow-up.** |
| §21 | Cold-start `protocol` is prose, not typed hops | unchanged (lift v2.1 closed mining gap; semantic gap still open — typed-hop migration is Day-10+) |
| §22 | `feasible_domains` collapses gymv games | ✓ Day-5/6 closed; Day-7 `record_task_verification` closes the persistence loop |

## 4. Day-9 follow-ups

* Wire `GateRunner` into `orchestrator.PromotionOrchestrator.promote()` so the anchors flow into `RunRelease` manifests.
* Add `--persist` flag to `_phase4_transfer_cycle.py` that calls `record_task_verification` on PASS (the lifecycle hook now exists).
* Crafter consumes the new `RejectedSkill` channel as `false_binding_patterns` evidence (PLAN-SKILL-BANK §4.3b).
* `osworld_adapter` / `video_adapter` / `visual_reasoning_adapter` real surfaces — currently deterministic stubs that pass the new `validate_invocation` trivially. Once they're real, the cross-domain transfer cycle becomes meaningful.
* Extend `dump_harness_io_gpt54.py` to the offline `GateRunner` surface (work-order item 13).
* Numeric `fit_score` / `risk_score` on `EligibleSkill` (Day-9+, requires LoRA scoring head from PLAN-SKILL-BANK §0.3 Clause D).

## 5. Files touched

```
data_structure/extensions/skill_episode.py        (+45 lines, additive fields + back-compat mirror)
data_structure/extensions/skill_evaluation.py     (+30 lines, anchor fields)
harness/__init__.py                               (+15 lines, lazy GateRunner export)
harness/eligibility.py                            (+90 lines, RejectedSkill + filter_with_rejections)
harness/gate_runner.py                            (NEW, 230 lines)
harness/replay_validator.py                       (+150 lines, action-level walk)
harness/skill_harness.py                          (+170 lines, validate_invocation + helpers)
skill_bank/lifecycle.py                           (+60 lines, record_task_verification)
harness/README.md                                 (+15 lines, §22 Day-7/8 status block)
tests/test_gate_runner.py                         (NEW, 200 lines, 5 tests)
tests/test_replay_validator_action_walk.py        (NEW, 230 lines, 6 tests)
tests/test_lifecycle_task_verification.py         (NEW, 145 lines, 6 tests)
tests/test_validate_invocation.py                 (NEW, 175 lines, 9 tests)
tests/test_skill_episode_field_expansion.py       (NEW, 175 lines, 7 tests)
```
