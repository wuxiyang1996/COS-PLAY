# Phase-4 / Day-5/6 — Cross-task transfer cycle + producer fan-out

**Status (2026-05-01, Day-5/6):** the intra-gymv task-axis transfer
milestone is end-to-end wired and empirically validated. This report
documents what landed, what the empirical numbers say, and what's
explicitly deferred.

Cross-refs:
* [`harness/README.md` §22](../../harness/README.md) — design context and
  status timeline.
* [`_phase3_report.md`](_phase3_report.md) — Day-4 producer + lift v2.
* [`_phase2_report.md`](_phase2_report.md) — Day-3 first real-env smoke.

---

## 1. Day-5a — Lift v2.1: per-game schema-index whitelist + word-set matcher

### Problem the day was scoped against

Day-4's producer surfaces `holes`, `stack_height`, `filled_cells`,
`level`, `lines_cleared` as entity labels in tetris's
`<state>...</state>` block, but the cold-start
`metadata.schema_canonical` blocks the protocol lift mines
(`labeling/_protocol_lift.py::build_schema_index_for_game`) only
enumerate `board`, `next_pieces`, `current_piece`, etc. The result was
that prose phrases like ``"Hole count increases from 3 to 4"`` mined an
`entity_count_changed` predicate with `args={}` — undecidable at runtime
because the runtime evaluator (`harness/gymv_success.py`) has nothing
to look up.

### What landed

* **`_SCHEMA_INDEX_LABEL_WHITELIST`** (per-(corpus, game)) overlays the
  auto-mined cold-start vocabulary with the producer's canonical
  emission set. Cold-start labels still win on collision; the
  whitelist is the *fallback* vocabulary, not an override. Tetris
  registers `holes`, `stack_height`, `filled_cells`, `level`,
  `lines_cleared`, `score`. 2048 / candy_crush / super_mario also
  registered defensively for Day-6's producer set.
* **`_first_entity_label`** rewritten with a word-set matcher. Phrases
  like ``"no lines are cleared by the placement"`` now bind to label
  `lines_cleared` (the producer-canonical name) — substring matching
  alone wouldn't, because the words are non-adjacent. Singular ↔
  plural fold (``"hole count"`` → `holes`), longest-match-wins on
  ties. Sorted iteration over `entity_labels` for stable picks.
* `--force_relift` re-applied: `Commit/Position` →
  `entity_count_changed entity_label='holes'`; `Commit/Survive` →
  `entity_count_changed entity_label='lines_cleared'` (matches the
  producer's emission).

### Empirical evidence — Phase-2 smoke pre/post

**Tetris (`--n-trials=4 --seed=0`):**

| Skill              | Day-4 (`v1` lift) | Day-5a (`v2.1`) | Day-5a hop-level detail |
| ------------------ | ----------------- | --------------- | --- |
| `COMMIT/POSITION`  | rate=1.00 (5/5)   | rate=0.00 (0/5) | `count[holes] 1 → 1` decidable & failing |
| `COMMIT/SURVIVE`   | rate=0.50 (2/4)   | rate=0.00 (0/4) | `count[lines_cleared] 1 → 1` decidable & failing |

The rate **drop** is the correct rigor signal: previously-undecidable
predicates were silently passing as non-blockers. Now they're
decidable, and the runtime correctly reports that *left shifts in
tetris don't change the hole count or clear lines*. The
`attribute_changed` predicate continues to pass on these hops
(``"holes attrs changed"``) — that's still real evidence of a side
effect.

### Tests

`tests/test_protocol_lift.py` gained 4 new cases (37 passing total):
the whitelist binds, plural fold works, multi-word labels match
non-adjacent phrasing, end-to-end lift-and-bind for tetris hole-count
prose.

---

## 2. Day-5b — Stage 3a `FewShotAdapter` cross-task transfer cycle

### What landed

The transfer cycle is the headline milestone of the harness/README §22
roadmap: take a skill mined on game A, run it against
demonstrations of game B, and record the verdict. PASS / LIMITED_PASS
appends `B` to `verified_tasks`, which the F2′ task-axis filter then
consults to widen admission.

Three new surfaces:

| Surface | Module | Contract |
| --- | --- | --- |
| Demo loader | `harness/few_shot_demos_gymv.py` | `(actions_root, game, …) -> List[FewShotDemo]` from cold-start `episode_*.json` files; `state` parsed via `parse_schema_canonical`, `bindings` carry the recorded action token, `expected` carries the per-step reward. |
| Transfer driver | `labeling_supplement/_phase4_transfer_cycle.py` | CLI: `--source <game> --target <game> --k <int> [--bindings k=v]`. Wires `GymvAdapter.set_executor(make_gymv_executor(target_env, schema_producer=make_gaming_env_producer(target_game)))`, calls `FewShotAdapter.adapt(skill, target_domain="gymv", target_task=<target>, demos, success_fn=make_per_step_success_fn(...))` for every source skill, surfaces per-skill verdicts and the eligibility-set diff. |
| Intra-domain task-transfer validation | `harness/few_shot_adapter.py::FewShotAdapter._validate` | Day-5 relaxation: when `target_task` is set AND `target_domain` is in `SOURCE_DOMAINS`, intra-source-domain task transfer (e.g. `gymv` → `gymv` with `target_task="tetris"`) is allowed without listing `gymv` in `TRANSFER_TARGET_DOMAINS`. Cross-domain semantics are unchanged. |

### Empirical evidence — three cardinal probes

#### A. Same-task baseline (`twenty_forty_eight → twenty_forty_eight`, k=4)

```
skill_id                 type       demos pass  rate  ok  promoted
COMMIT__MERGE            action         4    3  0.75   Y       YES
mid:OPTIMIZE             action         4    4  1.00   Y       YES
COMPARE__MERGE           reasoning      4    4  1.00   Y       YES

eligibility on (gymv, twenty_forty_eight): BEFORE=3/3  AFTER=3/3  (Δ = +0)
```

Sanity check: 2048 skills run on 2048 demos. All three pass; eligibility
unchanged because they were already feasible. `COMMIT/MERGE`'s 0.75
reflects 1/4 demos where `left` doesn't produce a merge in the seeded
state — same rigor signal as Day-4's A/B.

#### B. 2048 → tetris (k=4)

```
skill_id                 type       demos pass  rate  ok  promoted diag
COMMIT__MERGE            action         4    0  0.00   n         - adaptation_overfitting
mid:OPTIMIZE             action         4    4  1.00   Y       YES
COMPARE__MERGE           reasoning      4    4  1.00   Y       YES

eligibility on (gymv, tetris): BEFORE=0/3  AFTER=2/3  (Δ = +2)
```

The headline result: **eligibility on `(gymv, tetris)` widens from
0/3 to 2/3**, driven by two passing skills appending `tetris` to their
`verified_tasks`. `COMMIT/MERGE` correctly fails — its merge predicates
(``"valid merges"`` → `cumulative_reward_increased`) don't fire on
tetris's reward signal — and is *not* promoted. `mid:OPTIMIZE` and
`COMPARE/MERGE` pass because their lifted protocols are
mostly-observational (no env-mutating predicates the producer can
disprove); the success_fn defers to `outcome.success` per its
documented semantics, which is the right call for Stage 3a where we
care about "does the skill complete on this domain at all?"

#### C. tetris → 2048 (k=4)

```
skill_id                 type       demos pass  rate  ok  promoted diag
COMMIT__EVADE            action         4    0  0.00   n         - adaptation_overfitting
COMMIT__OPTIMIZE         action         4    0  0.00   n         - adaptation_overfitting
COMMIT__POSITION         action         4    4  1.00   Y       YES
COMMIT__SETUP            action         4    4  1.00   Y       YES
COMMIT__SURVIVE          action         4    4  1.00   Y       YES
INSPECT__SETUP           grounding      4    4  1.00   Y       YES

eligibility on (gymv, twenty_forty_eight): BEFORE=0/6  AFTER=4/6  (Δ = +4)
```

Symmetric headline: tetris skills probed on 2048 widen the admit set
from 0/6 to 4/6. `COMMIT/EVADE` and `COMMIT/OPTIMIZE` correctly fail —
their predicates (`entity_count_changed entity_label='holes'`,
`phase_transitioned`) require tetris-specific surface that 2048
doesn't expose. The other four are mostly-observational and pass on
the structural well-formedness check.

### Asymmetry note

Tetris cold-start `action` strings are high-level placement
descriptions (`"S-flat col4 (+1hole, h=6)"`); 2048's are clean tokens
(`"left"`). The driver's `--bindings direction=left --bindings
target=left` flag rescues this so the executor's payload-value
resolver finds *something* in the target env's action vocabulary;
unresolvable hops fall through `on_unresolved="skip"` to soft-skip.
This is a property of the cold-start prompt design, not the transfer
cycle.

### Tests

* `tests/test_few_shot_demos_gymv.py` (6 cases) — demo loader
  no-op-skipping, max-cap, malformed-schema-tolerance, multi-episode
  walking, missing-root-graceful, expected-payload preservation.
* The Day-5b transfer driver is exercised end-to-end by the three
  empirical probes above (artifacts in
  `labeling_supplement/harness_io_out/_phase4_transfer_*.json`).

---

## 3. Day-6a — Producer fan-out: candy_crush + super_mario

### What landed

Two new producers in `harness/gym_schema_producer.py`:

* **`candy_crush_producer`**. Parses the env's textual obs (`"Board:
  …\nScore: <N>\nMoves Left: <N>"`) for the 8×8 letter-coded grid and
  emits one aggregate `candy_<color>` text entity per color (counts as
  attribute), plus `score` and `moves_remaining` `goal_indicator`s.
  `phase=gameover` when `moves_remaining ≤ 0`.
* **`super_mario_producer`**. Parses `"Position of Mario: (X, Y)"`
  + the `Positions of all objects` table for visible enemies / items,
  emits one entity per visible object plus aggregate `score`,
  `lives`, `scroll_x` (= `mario.x`) `goal_indicator`s. `progress`
  is the normalized scroll position over a 3168-px world-1-1
  baseline.

Both are wired into `_PRODUCERS` registry so
`make_gaming_env_producer("candy_crush" | "super_mario")` returns
them. Both round-trip through `parse_schema_canonical` and surface
`entity_attrs` / `entity_label_count` / `phase` / `score` cleanly.

### Tests

`tests/test_gym_schema_producer.py` grew from 13 → 18 cases:

* Round-trip parse for both new producers.
* `info` overrides text-parsed `score` / `moves_remaining` (candy).
* `phase=gameover` when no moves left (candy).
* `cumulative_reward_increased` decidable on candy score deltas.
* `entity_value_increased entity_label=scroll_x` decidable on mario
  forward progress.
* No-objects edge case (mario empty world).

---

## 4. Day-6b — Domain-keyed `SuccessFn` registry

### What landed

`harness/gymv_success.py` now exposes:

* `register_success_fn(domain, factory)` — register a per-domain
  `SuccessFn` factory. Bootstrap registers `gymv ⇒
  make_per_step_success_fn` at module import.
* `success_fn_for_domain(domain, *, pass_rate_threshold=0.5,
  require_episode_success=True, fallback=None)` — look up the right
  scorer for a target domain. Falls back to
  `harness.few_shot_adapter.default_success_fn` when no scorer is
  registered (browser, osworld, video, visual_reasoning until those
  surfaces ship).
* `registered_success_fn_domains()` — diagnostic view.

`FewShotAdapter.adapt` now consults the registry: when constructed
with the default `default_success_fn`, the adapter swaps in the
registered scorer for `target_domain`. An explicit `success_fn=…`
construction-time override still wins.

### Why this matters

Before Day-6, cross-domain transfer (the Day-9+ scope) would have
needed every gate caller to manually wire the right scorer for each
target. Now the adapter pulls it from the registry — register a
browser scorer once, every transfer-target=browser call uses it.

### Tests

`tests/test_success_fn_registry.py` (6 cases): gymv auto-registered,
gymv scorer runs, unknown domain falls back to default, registration
is overwriting, FewShotAdapter consults the registry on default
scorer, explicit scorer overrides the registry.

---

## 5. What's deferred

* **Day-6c — Action-level `ReplayValidator`.** The existing dry-run
  `ReplayValidator` (`harness/replay_validator.py`) operates at the
  adapter level and is sufficient for Day-5/6 milestones. The
  step-by-step replay-against-`seed.steps` walk that PLAN-UNIFIED-SKILL-GATE
  §7.1 ultimately wants is a deeper refactor (it needs
  `SkillEpisode.steps[i]` ↔ `skill.protocol[k]` indexing, which
  intersects with §10 of the harness/README "what's missing today").
  Punted to Day 7 alongside the gate-runner work.

* **`AdaptResult` lifecycle persistence.** The driver appends to
  `verified_tasks` *in-memory only* — the eligibility-set diff is
  visible in the run, but committing that back to the on-disk skill
  bank requires a `SkillLifecycleManager` transition (PLAN-SKILL-BANK
  §4.2). That belongs in the Day-7 lifecycle work and the Day-8 spec
  alignment.

* **`mid:OPTIMIZE` and `COMPARE/MERGE` pass-rate semantics.** Their
  1.00 cross-task pass-rate is real-but-trivial — the lifted
  protocols are observational, so the success_fn defers to
  `outcome.success`. The right tightening is a "per-episode any-effect
  was observed" sub-condition. That's a refinement of
  `make_per_step_success_fn`'s threshold logic and is tracked
  separately from the §22 milestone gate.

---

## 6. Files and artifacts

* **New files:**
  * `harness/few_shot_demos_gymv.py`
  * `labeling_supplement/_phase4_transfer_cycle.py`
  * `tests/test_few_shot_demos_gymv.py`
  * `tests/test_success_fn_registry.py`
* **Modified files:**
  * `labeling/_protocol_lift.py` — Day-5a whitelist + word-set matcher.
  * `labeling_supplement/_phase2_real_env_skill_smoke.py` — added
    `DEFAULT_ACTIONS_ROOT` for the transfer driver.
  * `harness/few_shot_adapter.py` — Day-5 intra-domain task transfer
    + Day-6 registry consult.
  * `harness/gym_schema_producer.py` — candy_crush + super_mario
    producers.
  * `harness/gymv_success.py` — Day-6 success_fn registry.
  * `harness/__init__.py` — exports.
  * `tests/test_protocol_lift.py` — 4 new cases.
  * `tests/test_gym_schema_producer.py` — 5 new cases.
* **Empirical artifacts:**
  * `labeling_supplement/harness_io_out/_phase4_transfer_twenty_forty_eight_to_tetris_*.json`
  * `labeling_supplement/harness_io_out/_phase4_transfer_twenty_forty_eight_to_twenty_forty_eight_*.json`
  * `labeling_supplement/harness_io_out/_phase4_transfer_tetris_to_twenty_forty_eight_*.json`
  * `labeling_supplement/harness_io_out/_phase2_tetris_*.json` (Day-5a
    rigor evidence).

---

## 7. Test counts

| Test file | Day-4 | Day-5/6 | Δ |
| --- | --- | --- | --- |
| `test_protocol_lift.py` | 33 | 37 | +4 (Day-5a) |
| `test_gym_schema_producer.py` | 13 | 18 | +5 (Day-6a) |
| `test_few_shot_demos_gymv.py` | 0 | 6 | +6 (new, Day-5b) |
| `test_success_fn_registry.py` | 0 | 6 | +6 (new, Day-6b) |

**Full suite: 342 passing, 1 pre-existing failure unrelated to
Day-5/6 (`tests/test_schema_predicates.py::TestRobustness::test_extra_whitespace_tolerated`).**
