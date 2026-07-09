# Phase-1 cross-eligibility probe — task-axis veto landed (Day 2)

> **Status:** Day-2 task-axis contract change deployed and empirically
> validated. Cross-game contamination on `twenty_forty_eight ↔ tetris`
> dropped from **100 % → 0 %** with the same fused 9-skill bank,
> 277 actor-driven steps, and a single additive `EligibilityFilter` rule.
> **Last reviewed:** 2026-04-30.
> **Cross-refs:** [`harness/README.md` §22](../../harness/README.md),
> [`labeling_supplement/_phase0_cross_eligibility_probe.py`](../_phase0_cross_eligibility_probe.py),
> [`labeling_supplement/harness_io_out/_phase0_report.md`](_phase0_report.md),
> [`tests/test_eligibility_task_axis.py`](../../tests/test_eligibility_task_axis.py).

## 1. What landed

| Surface | Change | Backwards-compat |
|---|---|---|
| `data_structure.extensions.skill_record.SkillRecord` | Two new list fields: `feasible_tasks: List[str]` and `verified_tasks: List[str]`. Plumbed through `to_json()` / `new()` / `__post_init__()`. Excluded from `content_hash()` deliberately — they are eligibility metadata, not skill body, so adding a verified task does not invalidate a prior gate evaluation. | Default `[]`; pre-v2 banks load without changes. |
| `harness.eligibility.EligibilityFilter.filter()` | New filter step F2′ between domain (F2) and adapter dispatch (F3): if `skill.feasible_tasks` is non-empty, the trailing path-segment of `state.task` must be in it. | Empty `feasible_tasks` is *task-agnostic* and admits everywhere the domain admits. State with no task tag does not blind-veto; it admits as `agnostic` (single-step adapter / synth state path). |
| `harness.eligibility.EligibleSkill` | New `task_match: str` field (`"agnostic"` / `"same_task"` / `"verified"`). Surfaced in `to_json()` + the `reasons` string. | Unchanged for callers that don't read it. |
| `harness.eligibility.task_id_from_state()` | Public helper that extracts the bare task token from `state.task` (`"make_gaming_env/twenty_forty_eight"` → `"twenty_forty_eight"`). Re-exported from `harness/__init__.py`. | New public function. |
| `harness.few_shot_adapter.FewShotAdapter.adapt(...)` | New optional `target_task: Optional[str] = None` keyword. Synthesised demo state and `_coerce_state_to_target` propagate it into `state.task` so F2′ sees the right task during transfer; the value lands on `AdaptResult.target_task` for downstream `verified_tasks` bookkeeping. `adapt_many` gets a parallel `target_task_by_domain` keyword. | `target_task=None` keeps pre-task-axis behaviour byte-for-byte. |
| `skill_bank.stores._record_from_dict` (loader) | Reads `feasible_tasks` / `verified_tasks` from on-disk JSON; defaults to `[]`. | Existing on-disk records without these fields load unchanged. |
| `labeling_supplement._harness_io_helpers.record_from_bank_entry` | Seeds `feasible_tasks` from explicit `skill["feasible_tasks"]` first, falling back to `[provenance.source_name]`. | Both shapes coexist; the helper bridges them. |
| `labeling._decorate_skill_records.py` | Decorator bumped to `skillrecord_shape_v2`. New rows seed `feasible_tasks=[<source_name>]` and `verified_tasks=[]`; v1-decorated rows are back-filled idempotently. `_lifecycle_meta.json` records the new defaults. | Re-running over a v1 bank lifts forward; re-running over a v2 bank is a no-op. |

Total touched: 7 source files + 1 new test file (12 passing tests). No
behavioural change for any cold-start skill that hasn't yet been
re-decorated to v2 — `feasible_tasks=[]` is admitted unconditionally.

## 2. Empirical result

### Setup (identical to Phase-0)

* Fused 9-skill bank: 3 skills from `twenty_forty_eight` + 6 from `tetris`.
* Actor-driven episodes: `labeling/skill_actions_out/run_20260430_064325/env_wrappers/{twenty_forty_eight,tetris}/episode_*.json`, 3 episodes per game, capped at 50 steps each.
* Bank decorated with `skillrecord_shape_v2` via `python labeling/_decorate_skill_records.py --root labeling/skill_bank_out/run_20260430_030637`. All 12 env_wrappers + 102 gym_v skills back-filled with `feasible_tasks=[<source_name>]`.
* Probe re-run: `python labeling_supplement/_phase0_cross_eligibility_probe.py`.

### Result

```
[twenty_forty_eight] 150 steps
  same-game eligible-count histogram : {2: 150}     ← every step admits 2/3 same-game skills
  CROSS-game eligible-count histogram: {0: 150}     ← 0 cross-game admits on every step
  per-skill eligibility tallies:
    [SAME ] COMMIT__MERGE       eligible 150 steps
    [SAME ] mid:OPTIMIZE        eligible 150 steps
    (COMPARE/MERGE filtered: REASONING type, no (gymv, REASONING) adapter — orthogonal to §22)

[tetris] 127 steps
  same-game eligible-count histogram : {6: 127}     ← every step admits 6/6 same-game skills
  CROSS-game eligible-count histogram: {0: 127}     ← 0 cross-game admits on every step
  per-skill eligibility tallies:
    [SAME ] COMMIT__EVADE       eligible 127 steps
    [SAME ] COMMIT__OPTIMIZE    eligible 127 steps
    [SAME ] COMMIT__POSITION    eligible 127 steps
    [SAME ] COMMIT__SETUP       eligible 127 steps
    [SAME ] COMMIT__SURVIVE     eligible 127 steps
    [SAME ] INSPECT__SETUP      eligible 127 steps
```

| Metric | Phase-0 (no task axis) | Phase-1 (F2′ veto) |
|---|---|---|
| 2048 step → cross-game admits | **6 / step** (every step admits all 6 tetris skills) | **0 / step** |
| Tetris step → cross-game admits | **2 / step** (every step admits 2 of 3 2048 skills) | **0 / step** |
| 2048 step → same-game admits | 2 / step (only `COMPARE/MERGE` filtered for the orthogonal §16 reason) | 2 / step (unchanged) |
| Tetris step → same-game admits | 6 / step | 6 / step (unchanged) |

The same-game admit counts are byte-identical between Phase-0 and
Phase-1 — the F2′ filter is purely additive: it removes cross-task
admits without touching the same-task path. That confirms the back-
compat clause works in vivo, not just in the unit tests.

## 3. Unit-test guard

`tests/test_eligibility_task_axis.py` (12 tests, all green) covers:

* Task-id extraction across both `<game>` and `<prefix>/<game>` shapes,
  including degenerate cases (empty / whitespace / trailing slash).
* Same-task admission with `task_match="same_task"` reason.
* Verified-task admission with `task_match="verified"` reason.
* Cross-task veto in both directions (the §22 regression case).
* Empty-`feasible_tasks` admits as `task_match="agnostic"` regardless
  of state task — pre-v2 back-compat.
* State with no task tag admits as `agnostic` (single-step / synth path).
* Multi-task feasibility (a skill that's been transferred and
  partially verified) admits on both feasible tasks but with the
  correct `task_match` per task.

Together with the Phase-1 probe these constitute the regression net:
the unit tests catch a broken filter at edit-time; the probe catches a
mis-decorated bank or a regression in the helper / loader path.

## 4. What this does NOT validate

The probe measures **admission**: does the filter still pass cross-game
skills through to the actor's pick-list. It does **not** measure:

1. **Does cross-task veto change which skill the actor picks?** The
   actor's RAG was per-game in the original cold-start corpus, so on
   *its own* skill_query.candidates the cross-task admits never
   appeared in Phase-0 either (that was the §14 driver-shape
   limitation). Whether the actor would have picked differently if
   given the cross-task list is a separate question. Resolving it
   requires re-running the actor with the fused bank and a task-aware
   RAG, which is Day-3+ work.
2. **Does the F2′ filter behave correctly under transfer?** Stage 3a
   cross-task probes (`FewShotAdapter.adapt(skill, target_domain,
   target_task=...)`) wire the new `target_task` parameter into
   `_coerce_state_to_target`. The synth state will carry the target
   task, so a tetris-feasible skill in transfer to 2048 will see
   `state.task="few_shot_probe/twenty_forty_eight"` (after the
   `task_id_from_state` projection: `"twenty_forty_eight"`) and the
   filter will veto. **This is intentional** — the gate's Stage 3a is
   exactly the path that should append to `verified_tasks` *before*
   the skill becomes admissible there. Until that wiring lands
   (Day 3–4: real GymV executor + transfer cycle), the FewShotAdapter
   path will return `target_domain_demo_unavailable` for cross-task
   probes against gymv. That is the correct answer; it is not a bug.
3. **Cold-start skills with `applicable_domains=["gymv"]` and
   `feasible_tasks=[<game>]` cannot transfer to a different game
   without going through Stage 3a.** This is the expected and desired
   blocking property — it forces every cross-task admit to be
   verified before the skill can hit ACTIVE in the second task.

## 5. Day-2 wrap-up & next step

Task-axis contract change: **done.** Two follow-ups remaining for the
intra-gymv transfer milestone:

* **Protocol lift** (the second half of Day 2 per
  [`implementation_notes/legacy/protocol-lift-design.md`](../../implementation_notes/legacy/protocol-lift-design.md))
  — replace prose `EXEC` hops with the 21-verb gymv taxonomy,
  populate `${slot}` placeholders, and mine `effects_add` /
  `effects_del` from `success_criteria` / `abort_criteria`. Lands in
  `labeling/_decorate_skill_records.py` next to the v2 task-axis
  back-fill.
* **Real GymV executor wiring + success_fn** (Day 3–4) — the lift's
  payoff comes when `harness.run_skill(...)` actually steps the env
  via `GymvAdapter.set_executor(real_step)` and the success_fn reads
  consecutive `schema_canonical` blocks. That's where Stage 3a
  transfer probes start producing real verdicts and `verified_tasks`
  gets populated.
