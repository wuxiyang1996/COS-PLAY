# Phase-2 real-env skill execution — Day-3 wiring landed

> **Status:** Day-3 real-env wiring deployed and empirically validated.
> `harness.run_skill(skill, state)` now drives a real GamingAgent
> `make_gaming_env(...)` env via `GymvAdapter.set_executor(...)`, captures
> per-hop pre/post `StateSchema` snapshots, and surfaces a structured
> per-hop `effects_add` verdict on `outcome.extra["per_hop_effects"]`.
> **Last reviewed:** 2026-05-01.
> **Cross-refs:** [`harness/README.md` §22](../../harness/README.md),
> [`harness/gymv_executor.py`](../../harness/gymv_executor.py),
> [`harness/gymv_success.py`](../../harness/gymv_success.py),
> [`labeling_supplement/_phase2_real_env_skill_smoke.py`](../_phase2_real_env_skill_smoke.py),
> [`labeling_supplement/harness_io_out/_phase1_report.md`](_phase1_report.md).

## 1. What landed

| Surface | Change | Backwards-compat |
|---|---|---|
| `labeling_supplement._harness_io_helpers.parse_schema_canonical` | Now parses the `<attributes>` and `<state_flags>` blocks. New `state.facts` keys: `score`, `highest_tile`, `lines_cleared`, `phase`, `progress`, `entity_attrs` (label → field → value), `entity_label_count` (label → count). The `<entities>` body is still parsed; `entities`-section header just narrows the scan. | All pre-Day-3 callers keep their `goal`, `domain`, `task`, `elements` keys. Schemas without `<attributes>` produce no new fact keys (no false positives on legacy dumps). |
| `harness.gymv_executor` (NEW) | `make_gymv_executor(env, …)` returns a `(HopExecutor, GymvExecutorState)` pair to plug into `GymvAdapter.set_executor`. The executor maps each typed hop op to an env action via `ACTION_ALIAS_MAP` (SLIDE/MOVE → up/down/left/right, ROTATE → rotate_cw/ccw, …) plus a payload-value rescue clause and op-level fallbacks. Observational ops (INSPECT/READ/EVALUATE/COMPARE/VERIFY/…) generate evidence without stepping the env. `on_unresolved="skip"` (default) treats unresolvable env-mutating hops as evidence-only no-ops; `"abort"` is the strict gate-hardening mode. `initial_state_from_env(env, …)` builds the hop-0 pre-state from the env's actual reset observation so `cumulative_reward_increased` etc. have a numeric baseline. | New module — no back-compat surface. |
| `harness.gymv_success` (NEW) | `evaluate_predicate(predicate, pre, post)` and `evaluate_hop_effects(hop, pre, post)` evaluate every predicate type from the protocol-lift taxonomy (`entity_value_increased`, `cumulative_reward_increased`, `phase_transitioned`, `entity_count_changed`, `entity_appeared`, `entity_disappeared`, `entity_value_decreased`, `attribute_changed`). Undecidable (label not surfaced, value not parseable) is non-blocking. `make_per_step_success_fn(...)` returns a `SuccessFn` ready to plug into `FewShotAdapter`. `evaluate_episode_effects(skill, episode)` is the per-episode roll-up. | New module. The runtime predicate taxonomy is asserted equal to the labeling-side `EFFECT_PREDICATE_TYPES` by `tests/test_gymv_success.py::test_taxonomy_completeness` so future lift edits propagate as a single failure here. |
| `harness.adapters.gymv_adapter.GymvAdapter.run` | Records per-hop `pre_state`/`post_state` snapshots (no longer just hop-0 pre). When the executor surfaces `hop_result["post_state"]`, it lands verbatim on the `SkillEpisodeStep`; otherwise we synthesise a facts-only echo so the success_fn at least sees `last_observation`. After the hop loop, `_evaluate_effects` rolls up `effects_add` against the recorded snapshots and writes the verdict to `AdapterRunResult.extra["per_hop_effects"]` — the harness then propagates that to `SkillEpisodeOutcome.extra` as part of the existing `extra=dict(result.extra)` plumb-through. | Skills with no typed `effects_add` (pre-lift cold-start banks) skip the roll-up entirely; `extra` stays unset. The previous "first-hop pre_state only" behaviour is gone but the existing `pre_state`/`post_state` fields were already nullable, so no consumer broke. |
| `harness.__init__` exports | `ACTION_ALIAS_MAP`, `EFFECT_PREDICATE_TYPES`, `GymvExecutorState`, `HopEffectResult`, `PredicateResult`, `evaluate_episode_effects`, `evaluate_hop_effects`, `evaluate_predicate`, `initial_state_from_env`, `make_gymv_executor`, `make_per_step_success_fn`. | Additive; no symbols removed. |

Total: 3 new modules + 1 modified adapter + 1 extended parser + 1 widened
`__init__`. Three new test files (`test_gymv_executor.py`,
`test_gymv_success.py`, `test_schema_canonical_attributes.py`) — 35
passing tests, plus the 67 Day-1/Day-2 tests still green.

## 2. Empirical result

### Setup

* Real GamingAgent env via `env_wrappers.gym_like.make_gaming_env(<game>)`.
* Decorated v2 bank from
  `labeling/skill_bank_out/run_20260430_030637/env_wrappers/<game>/skill_bank.jsonl`.
* `_phase2_real_env_skill_smoke.py` loads each skill via
  `record_from_bank_entry`, force-promotes to `PROVISIONAL` (so we
  bypass the gate's lifecycle path — Phase-2 is about execution, not
  promotion), and runs each skill end-to-end. Slot bindings
  (`--bindings direction=up --bindings target=up`) stand in for the
  actor's RAG/binding pass.

### 2048 — `python labeling_supplement/_phase2_real_env_skill_smoke.py --game twenty_forty_eight --max-skills 3 --bindings direction=up --bindings target=up --include-reasoning`

```
=== Phase-2 smoke summary [twenty_forty_eight, env=gaming_agent] ===
skill_id        type       hops eval pass  rate  ok
COMMIT__MERGE   action        7    3    3  1.00   Y
mid:OPTIMIZE    action        7    0    0  0.00   Y
COMPARE__MERGE  reasoning     7    0    0  0.00   Y
```

GamingAgent log evidence (env actually stepped):

```
[GymEnvAdapter] E3 S1: AgentAct='up', R=4.00, Perf=4.00, Term=False, Trunc=False, T=0.00s
[GymEnvAdapter] E5 S1: AgentAct='up', R=0.00, Perf=0.00, Term=False, Trunc=False
[GymEnvAdapter] E5 S2: AgentAct='up', R=0.00, Perf=0.00, Term=False, Trunc=False
```

Three real env steps for `mid:OPTIMIZE` (MOVE + SELECT after the
payload-value rescue clause kicks in) and one for `COMMIT__MERGE` (the
single SLIDE hop; the redundant EXECUTE() is soft-skipped under
`on_unresolved="skip"`). `COMPARE__MERGE` is reasoning-typed so all
seven hops are observational — no env steps, no aborts.

### Tetris — `--game tetris --bindings direction=left --bindings dir=cw --bindings target=active_piece --include-reasoning`

```
=== Phase-2 smoke summary [tetris, env=gaming_agent] ===
skill_id          type       hops eval pass  rate  ok
COMMIT__EVADE     action        7    0    0  0.00   Y
COMMIT__OPTIMIZE  action        7    4    4  1.00   Y
COMMIT__POSITION  action        7    0    0  0.00   Y
```

`COMMIT__OPTIMIZE` evaluates 4 hops with all predicates passing; both
shifted-left actions land in `[GymEnvAdapter] E7 S{1,2}: AgentAct='left'`.
The `EVADE`/`POSITION` skills carry no typed `effects_add` from the
lift (their prose triggered the predicate-mining gate but didn't bind
to any taxonomy entry), so they run env-clean but score zero
predicates. That's a **lift-coverage signal**, not an executor issue —
it's the next thing to harden in Day-4.

## 3. What we just proved

| Phase | Question | Verdict |
|---|---|---|
| 0 | Does the cold-start `harness_io_out` driver expose cross-task admits as the actor sees them? | **No** (driver-shape limitation) — fixed via `_phase0_cross_eligibility_probe.py`. |
| 0 | Without a task axis, do cross-task admits actually appear? | **Yes — 100 %** (every step admits every cross-game skill). |
| 1 | Does the F2′ task-axis veto remove cross-task admits? | **Yes — 100 % → 0 %** with same-task admits unchanged. |
| **2** | **Does `harness.run_skill(skill, state)` actually step a real env via `GymvAdapter.set_executor` when the bank is decorator-v2 lifted?** | **Yes.** GamingAgent log lines show real env actions; `outcome.extra["per_hop_effects"]` carries structured per-hop predicate verdicts. |
| **2** | **Does the gymv `success_fn` evaluate the lift's typed `effects_add` against post-step `<attributes>`-derived facts?** | **Yes.** Every predicate from the lift's 8-type taxonomy has a runtime evaluator; undecidable cases are correctly classified as non-blocking. |
| **2** | **Does the harness round-trip (pre → step → post → predicate roll-up) preserve `cumulative_reward` and `phase` so `cumulative_reward_increased` / `phase_transitioned` *can* fire when the lift surfaces them?** | **Yes.** `initial_state_from_env` + per-hop closure state both carry `cumulative_reward` numerically; `phase` is set from `terminated`/`truncated`. |

## 4. What this does NOT validate

* **Lift coverage of `cumulative_reward_increased`.** The COMMIT/MERGE
  protocol's `success_criteria` say "the resulting board differs from
  the previous board" → `attribute_changed`, not "score increases" →
  `cumulative_reward_increased`. As a result the executed hops score on
  `attribute_changed` (which is *undecidable* on text-obs envs because
  `entity_attrs` is empty on both sides) rather than on the env's
  numeric reward. The non-blocking rule keeps the verdict correct
  (passed=True with n_undecidable=1), but the diagnostic strength is
  weak. **Day-4** widens the lift's predicate-mining vocabulary so 2048
  SLIDE/MERGE protocols also surface `cumulative_reward_increased`.
* **Schema-canonical observation parity.** GamingAgent's text-obs envs
  return plain text, not `<state>...</state>`. A future Day-4/5 plug
  is `gymv_wrapper.adapter.GymVSchemaWrapper` — wrap the env in a VLM
  schema producer, so each post-step state is the same canonical
  format that fed the lift. Then `entity_attrs`-keyed predicates
  (`entity_value_increased`, `entity_count_changed`) become decidable
  end-to-end.
* **Stage 3a `FewShotAdapter` cross-task transfer probes.** The
  `make_per_step_success_fn` is now wireable into `FewShotAdapter`,
  but Phase-2 measures the **execution path**, not the **transfer
  path**. A 2048-feasible skill running in transfer to tetris would
  surface `target_domain_demo_unavailable` until the orchestrator's
  cross-task probe driver lands — that's Day-5+ work tied to the gate
  service.
* **Slot binding policy.** The smoke driver hard-codes
  `bindings={direction:up, target:up}`. The actor's real RAG-driven
  binding pass is what should populate these. Phase-2 proves the
  *execution surface* is real; the actor wiring is orthogonal.

## 5. Day-3 wrap-up & next step

Real-env wiring + per-hop predicate verdict: **done.** Two follow-ups
remaining for the intra-gymv transfer milestone:

* **Lift v2 (Day 4)** — broaden `_PREDICATE_TRIGGERS` so
  `cumulative_reward_increased` fires whenever the prose mentions a
  scoring outcome, including indirect phrasings ("merge values
  doubled", "lines cleared award points"). Cross-validate predicate
  recall against the live env's reward signal: the gate should be the
  reward-bearing post-state.
* **VLM-in-the-loop schema_canonical wrapper (Day 4–5)** — wire
  `gymv_wrapper.adapter.GymVSchemaWrapper` (or a cheaper
  textual-grounding head) so the env's text obs becomes a real
  `<state>...</state>` block per step. Once that lands,
  `entity_value_increased` / `entity_count_changed` become decidable
  end-to-end and the gymv success_fn returns dense, not sparse,
  verdicts.
* **Stage 3a transfer cycle (Day 5+)** — plug
  `make_per_step_success_fn` into `FewShotAdapter`, run the cross-task
  probe (2048 → tetris and tetris → 2048), append `verified_tasks` on
  PASS / LIMITED_PASS, and re-run the cross-eligibility probe to show
  the F2′ admit set widens for the verified skill.
