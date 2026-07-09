# Phase-6 report — Day-9 wiring milestone

> **TL;DR.** Closes the four "wire-up" gaps the Phase-5 report parked
> as Day-9 follow-ups. The structural surfaces from Day-7/8
> (`GateRunner`, `RejectedSkill`, `record_task_verification`,
> `validate_invocation`, the `SkillEpisode` / `SkillEvaluationRecord`
> field expansions) now flow end-to-end through the
> `PromotionOrchestrator`, the cross-task transfer driver, the
> Crafter's failure-pattern channel, and the harness-I/O dump
> driver. The deterministic transfer-target adapters lift to typed
> evidence-role emission so the action-level `ReplayValidator` walk
> has a meaningful signal even before real env executors are wired
> in. **+27 unit tests this round; full suite 402 / 403 (one
> pre-existing whitespace failure unrelated to this work).**

## 1. Surfaces wired this round

### 1.1 Day-9a — `GateRunner` → `PromotionOrchestrator` anchor flow

[`orchestrator/promotion_orchestrator.py`](../../orchestrator/promotion_orchestrator.py) `PromotionOrchestrator.promote(...)` now records the GateRunner reproducibility anchors directly on the `SkillEvaluationRecord` it persists, and surfaces a per-release union of those anchors in the audit log:

* **`status_before`** — captured *before* the lifecycle transition fires, so the audit row reflects what the skill was leaving (e.g. `CANDIDATE`).
* **`status_after`** — set to the `target_status` from the `PromotionPlan` (e.g. `ACTIVE`).
* **`bank_snapshot_id`** — pinned to the just-minted snapshot's id, *or* an explicit caller-supplied id (the new `bank_snapshot_id=...` kwarg on `promote()`). Caller's id wins so a multi-promotion batch can pin a frozen pre-mutation snapshot.
* **Audit row** — every `kind: "release"` entry now carries `reproducibility_anchors: { eval_suite_ids, ontology_versions, adapter_versions }` — the union across all transitions in the batch. Searching for "what did suite-A drive on this branch?" becomes a `grep audit.jsonl` query.

The `eval_suite_id`, `adapter_versions`, and `ontology_version` set on the `SkillEvaluationRecord` by the `GateRunner` survive the round-trip unchanged (the orchestrator only fills in *missing* anchors; it never overwrites).

### 1.2 Day-9b — `--persist` flag on `_phase4_transfer_cycle.py`

[`labeling_supplement/_phase4_transfer_cycle.py`](../../labeling_supplement/_phase4_transfer_cycle.py) gains an opt-in `--persist` flag. With `--persist --persist-bank-root <path>` the driver now:

1. Builds a `SkillRepository` rooted at `<path>` (idempotent — re-running over the same root re-uses the seeded records);
2. Seeds each source `SkillRecord` as `DRAFT` and walks it through `CANDIDATE → PROVISIONAL` so `record_task_verification` runs on a runnable status;
3. For every skill that the in-memory transfer loop promoted (`verified_task_promoted=True`), invokes `lifecycle.record_task_verification(...)` with the `(pass_rate, k_used)` metrics that drove the decision.

Persistence failures are logged per-skill but **do not abort** the run — the empirical verdict JSON is the headline output; persistence is a side-effect. The output JSON gains four new fields: `n_verified_tasks_persisted`, `persist_enabled`, `persist_bank_root`, `persist_errors`.

### 1.3 Day-9c — Crafter consumes `RejectedSkill` → `false_binding_patterns`

Two new surfaces close the eligibility-filter ↔ Crafter loop (PLAN-SKILL-BANK §4.3b):

**[`skill_bank/lifecycle.py`](../../skill_bank/lifecycle.py)** gains `record_false_binding_pattern(skill_id, *, veto, veto_reason, domain, task)` — the *only* sanctioned writer of `SkillRecord.false_binding_patterns`. Behaviour:

* Dedupes on `(veto, domain, task)`; the existing entry's `count` is incremented and `last_observed_at` is bumped.
* Different vetoes / domains / tasks become separate entries.
* `max_patterns` (default 64) caps the list; FIFO eviction prevents a misconfigured filter loop from unbounded-growing the record.
* Round-trips to disk via the same store layer the rest of the lifecycle uses.

**[`harness/rejected_skill_sink.py`](../../harness/rejected_skill_sink.py)** is the in-process aggregator that bridges the eligibility filter and the lifecycle writer:

```python
sink = RejectedSkillSink()
admitted, rejected = filter.filter_with_rejections(candidates, state)
sink.observe(rejected, domain=state.domain, task=task_id_from_state(state))
# … later, on the orchestrator's tick or the dump driver's teardown:
report = sink.flush_to(lifecycle, min_count=3)   # only hot patterns land
```

The sink is thread-safe (one shared sink across a multi-threaded harness is fine) and dedupes on `(skill_id, veto, domain, task)`. `flush_to(...)` skips skill_ids the lifecycle doesn't know about (without raising) so a transient repository in the dump driver can't blow up the flush; those skipped ids are returned in the `FlushReport`.

### 1.4 Day-7e — Dump driver consumes the new offline-gate surface

[`labeling_supplement/dump_harness_io_gpt54.py`](../../labeling_supplement/dump_harness_io_gpt54.py) lands two changes:

* **Real `validate_invocation`** — the §9.1 stub `_validate_invocation_stub` is replaced with `_validate_invocation_real`, which calls `harness.validate_invocation(skill, state, bindings, eligible=…)` and serializes the full `ValidateInvocationResult.to_json()` payload while preserving the legacy `{veto, veto_reason, source}` keys for back-compat.
* **`--gate-runner` opt-in** — switches the offline surface from `orchestrator.GateService` to `harness.GateRunner`. When set, the dump driver constructs a `GateRunnerConfig` with anchor values derived from the run context (`bank_snapshot_id` ← `dump:<bank-stem>`, `eval_suite_id` ← `cold_start:<actions-stem>`, `adapter_versions={a.name: "v1" for a in registry}`, `ontology_version="cold_start_v1"`, `seed=0`, `judge_model="dump_driver"`). Persisted `SkillEvaluationRecord`s now carry the §11 anchors.

The driver's `harness_known_gaps` summary updates to reflect §9.1, §9.3 (per-check booleans), §10, §11, §12, and §13 as **closed at the structural level**. The remaining gaps — §9.2 (planner-context plumb on `select_eligible_skills`) and §9.3 numeric `fit_score` / `risk_score` (LoRA scoring head) — are flagged as Day-10+ work.

### 1.5 Day-7d — Adapter typed-hop awareness

[`harness/adapters/_stub_base.py`](../../harness/adapters/_stub_base.py) `make_deterministic_executor(...)` lifts from "always emit `GATHER`" to "map `action_type` → role". The new `_ACTION_VERB_TO_ROLE` table covers the canonical four roles plus common synonyms (e.g. `OBSERVE → GATHER`, `INFER → REASON`, `ANSWER → COMMIT`); `_role_for_action(...)` falls back to `GATHER` for unknown verbs (G0 still satisfied) and accepts prefix matches (`VERIFY_TILE → VERIFY`).

The stub also emits a directional `evidence_in / evidence_out` split:
* `evidence_in` carries the union of roles already on the state at the time of the hop;
* `evidence_out` is what the stub just emitted (a single role-typed `EvidenceRef` for this hop).

`_stub_base.StubTransferTargetAdapter.run(...)` propagates the directional split (and `protocol_index`) into each per-step record so [`harness/skill_harness.py`](../../harness/skill_harness.py) `run_skill(...)` can copy them straight into `SkillEpisodeStep.evidence_in / evidence_out / protocol_index`. The `SkillEpisode.protocol_trace` list (Day-8b) populates correctly.

This is the smallest change that gives the action-level `ReplayValidator` walk a meaningful signal even before real env executors are plugged in. When a real executor is wired in via `set_executor(...)`, it can return the same `{ok, observation, evidence, evidence_in, evidence_out}` shape and inherit the typed-hop machinery for free.

## 2. Spec-contract status (delta vs. Phase-5)

| Section | Phase-5 status | Phase-6 status |
|---|---|---|
| §9.1  `validate_invocation` | structural CLOSED, dump driver still on stub | **CLOSED** — dump driver wired Day-7e |
| §9.2  planner-context on `select_eligible_skills` | open | open (Day-10+, planner refactor) |
| §9.3  `EligibleSkill` per-check booleans | structural CLOSED | **CLOSED** — surfaced through dump driver |
| §9.3  `EligibleSkill.fit_score / risk_score` | open | open (Day-10+, LoRA scoring head) |
| §10  `SkillEpisode` field expansion | structural CLOSED | **fully wired** — typed-hop adapters populate the directional split end-to-end |
| §11  `SkillEvaluationRecord` anchors | structural CLOSED | **fully wired** — populated by `GateRunner` *and* `PromotionOrchestrator` |
| §12  `GateRunner.evaluate(...)` overloads | structural CLOSED | **fully wired** — dump driver opts in via `--gate-runner` |
| §13  `harness.GateRunner` package alias | structural CLOSED | **fully wired** — dump driver consumes via `--gate-runner` |
| §22  `verified_tasks` persistence | partial (in-memory only on phase-4 driver) | **CLOSED** — `--persist` flag on phase-4 driver |
| §4.3b `false_binding_patterns` ingest | open | **CLOSED** — `RejectedSkillSink` + `record_false_binding_pattern` |

## 3. Test additions

| File | Purpose | Tests |
|---|---|---|
| [`tests/test_lifecycle_false_binding_patterns.py`](../../tests/test_lifecycle_false_binding_patterns.py) | Pins `record_false_binding_pattern` writer contract | 6 |
| [`tests/test_rejected_skill_sink.py`](../../tests/test_rejected_skill_sink.py) | Pins `RejectedSkillSink` aggregation + flush | 6 |
| [`tests/test_promotion_orchestrator_anchors.py`](../../tests/test_promotion_orchestrator_anchors.py) | Pins `status_before/after`, `bank_snapshot_id`, anchor union on audit | 5 |
| [`tests/test_phase4_persist.py`](../../tests/test_phase4_persist.py) | Pins `--persist` round-trip seed → record_task_verification → reopen | 3 |
| [`tests/test_stub_executor_typed_hops.py`](../../tests/test_stub_executor_typed_hops.py) | Pins typed-hop role mapping + directional split + `run_skill` propagation | 7 |
| **Total** | | **27** |

Full pytest run: 402 passed, 1 failed. The single failure
(`test_extra_whitespace_tolerated` in `test_schema_predicates.py`) is
pre-existing and unrelated to this work; it has been present in every
run since at least Day-3.

## 4. Out-of-scope items (Day-10+)

These items remain on the roadmap but require external infrastructure
not present in this repo today:

* **§9.2 planner-context on `select_eligible_skills`** — needs the
  Actor-side planner to plumb `intention / active_skill /
  local_reasoning_trace` through the harness call. The harness side
  is ready (the existing kwargs are forward-compat); the gating
  factor is the planner refactor.
* **§9.3 `fit_score` / `risk_score`** — requires the LoRA scoring
  head described in PLAN-SKILL-BANK §0.3 Clause D. The
  `EligibleSkill` dataclass is already shaped to receive them; the
  gating factor is the training pipeline.
* **Real cross-domain adapter executors** — `osworld_adapter`,
  `video_adapter`, `visual_reasoning_adapter` still use the
  deterministic stub by default. With Day-7d the stub no longer
  flattens evidence roles, so when a real executor is wired in via
  `set_executor(...)` the directional-split + role-typed signal
  flows for free. The gating factor is the per-domain env binding
  (OSWorld VM, Video-Holmes data corpus, visual-reasoning tool
  registry).
* **`SkillProvider` actor wiring** — not started this round; the
  harness surface is ready (eligibility + validate_invocation +
  rejected channel all in place), but the legacy actor still
  consumes the bank directly.

The next high-value milestone in this thread is wiring up at least
one real cross-domain executor (likely
`visual_reasoning_adapter`, since the `visual_reasoning_wrapper`
package exposes a ready `bind_executor(...)` symbol) so the
cross-domain transfer cycle becomes meaningful.

— Day-9 / Phase-6 wiring milestone close.
