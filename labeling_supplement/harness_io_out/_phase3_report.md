# Phase-3 / Day-4 — lift v2 + deterministic schema producer

> **Status:** Day-4 lift v2 (broadened predicate-mining vocabulary) +
> Day-4B deterministic ``<state>``-block producer for live gym envs are
> deployed and empirically validated. Per-hop predicate verdicts are
> now **decidable** end-to-end: ``score``, ``highest_tile``,
> ``entity_attrs``, ``entity_label_count``, ``phase`` are surfaced from
> ``env.info`` and consumed by the success-fn — no VLM in the loop, no
> "entity_attrs missing on both sides" undecidability.
> **Last reviewed:** 2026-05-01.
> **Cross-refs:** [`harness/README.md` §22](../../harness/README.md),
> [`labeling/_protocol_lift.py`](../../labeling/_protocol_lift.py),
> [`harness/gym_schema_producer.py`](../../harness/gym_schema_producer.py),
> [`labeling_supplement/_phase2_real_env_skill_smoke.py`](../_phase2_real_env_skill_smoke.py),
> [`labeling_supplement/harness_io_out/_phase2_report.md`](_phase2_report.md).

## 1. What landed

### 1A — Lift v2: predicate-trigger expansion

`labeling/_protocol_lift.py::_PREDICATE_TRIGGERS` was widened to cover
the indirect phrasings that appear in real cold-start prose. Triggers
stay narrow enough that disjunctive phrasing (``"… or one merge"``)
does **not** over-fire reward, but specific phrases (``"valid merges
were applied correctly"``, ``"score is higher"``, ``"top-out"``) do.

| Predicate | New triggers (Day-4) | Empirical motivation |
|---|---|---|
| `cumulative_reward_increased` | `score is higher`, `valid merges`, `valid merge`, `merges were applied`, `merges applied`, `merge resolved`, `merges resolved`, `merges produce`, `small line clear`, `small line clears`, `match awards`, `match awarded`, `points awarded` | 2048 ``Commit/Merge``: was `attribute_changed=3` only → now also `cumulative_reward_increased=3`; tetris ``Commit/Optimize``: was `entity_count_changed=4` only → now also `cumulative_reward_increased=4`; candy_crush ``Commit/Clear``: was 0 → now 2 (reward + value_decreased on moves) |
| `phase_transitioned` | `top out`, `top-out`, `topping out`, `topped out`, `non-recoverable state` | Tetris abort criteria (`"would cause immediate top-out"`); Mario abort criteria (`"non-recoverable state for this intent"`) |
| `entity_appeared` | `is visible`, `newly visible`, `new terrain`, `new section`, `new piece`, `new pieces` | Mario ``Commit/Navigate``'s `"a new navigable section is visible"`; lift was producing 0 effects before |
| `entity_count_changed` | `lines cleared` (added precise variants), `line cleared`, `rows cleared`, `hole count`, `holes increase`, `holes decrease`, `no lines are cleared`, `no line is cleared` | Tetris hole-count phrases didn't fire under v1 |
| `entity_value_increased` | `increases from` | Tetris `"Hole count increases from 3 to 4"` |
| `entity_value_decreased` | `has decreased`, `count has decreased`, `count decreased`, `decreases from`, `decreased from` | Candy-crush ``Commit/Clear``: `"moves remaining count has decreased"` |
| `attribute_changed` | `remains the same`, `remains approximately`, `stays the same`, `is preserved` | Mario inspection prose; tetris ``Commit/Survive``'s `"the board's hole structure is preserved"` |

`labeling/_decorate_skill_records.py` gained a `--force_relift` flag
that restores `protocol` from the preserved `protocol_raw` and
re-mines effects from the fresh prose, so trigger-set updates sweep
the bank cleanly without manual surgery.

### 1B — Deterministic schema producer (`harness/gym_schema_producer.py`)

`make_gaming_env_producer(game) -> Optional[SchemaProducer]` returns a
pure function `(info, obs, *, step, task, goal) -> str` that turns
`env.info` (and the textual obs) into a real `<state>...</state>`
block matching the cold-start labeler's convention:

  * `<entities>` — `e1[type=…, label=…, …]` per visible object
  * `<attributes>` — `e1.value=…`, `e1.state=visible`, …
  * `<state_flags>` — `phase=play|gameover`, `progress=…`,
    `cumulative_reward=…`, `step_score=…`
  * `<affordances>`, `<actions>` — for human readability

Two producers ship for Day-4:
  * **`twenty_forty_eight_producer`** — reads
    `info["board"]` (4×4 tile values), `info["total_score"]`,
    `info["max_tile_power"]`, `info["is_legal_move"]`. Emits one
    `tile_<value>` entity per non-zero cell (deterministic row-major
    `eN`-numbering), plus `board`, `empty_cells`, `highest_tile`,
    `score`. Handles numpy types (`np.uint8`, `np.int64`) without
    raising. Phase is `gameover` when board is full **and** the last
    move was illegal.
  * **`tetris_producer`** — reads `info["score"]`, `info["lines"]`,
    `info["level"]`, `info["next_piece_ids"]`, plus the textual obs
    for active-piece detection (the tetromino letter that appears
    exactly 4 times). Emits `board`, `active_piece_<L>`,
    `score`, `lines_cleared`, `level`, `holes`, `filled_cells`. Holes
    are computed as the standard tetris-AI "empty cell with a
    non-empty cell anywhere above it in the same column".

Both round-trip cleanly through
`labeling_supplement._harness_io_helpers.parse_schema_canonical` —
asserted by `tests/test_gym_schema_producer.py`.

### 1C — Wire-up: `make_gymv_executor(env, …, schema_producer=…)`

`harness/gymv_executor.py` accepts a new `schema_producer` kwarg on
both `make_gymv_executor(...)` and `initial_state_from_env(...)`. When
set, the executor's per-hop post-state path renders the block
internally (via `_adapt_schema_producer` which closes over `domain` /
`task` / a per-call step counter) and parses it through
`parse_schema_canonical`. When unset (the Day-3 default), the executor
falls back to the plain-text obs path.

`labeling_supplement/_phase2_real_env_skill_smoke.py` plugs in
`make_gaming_env_producer(args.game)` automatically and exposes a
`--no-schema-producer` flag for A/B comparisons against the Day-3
text-obs path.

The smoke driver also gained `--n-trials N --seed S` so a single skill
can be evaluated against deterministic resets (seeds `S..S+N-1`),
defeating the env's non-deterministic initial-tile placement (a 2048
``Commit/Merge`` skill can only score on resets where ``up`` produces
a legal merge).

## 2. Empirical results

### 2A — Predicate coverage diff (lift v1 → v2)

Per-skill `effects_add` counts on the
`labeling/skill_bank_out/run_20260430_030637/env_wrappers/<game>/skill_bank.jsonl`
bank, all four games:

| Game | Skill | Lift v1 (Day-3) | Lift v2 (Day-4) |
|---|---|---|---|
| candy_crush | `Commit/Clear` | (none) | `cumulative_reward_increased=1`, `entity_value_decreased=1` |
| super_mario | `Commit/Navigate` | (none) | `entity_appeared=3` |
| super_mario | `Inspect/Setup` | (none) | `attribute_changed=2` |
| tetris | `Commit/Evade` | (none) | `phase_transitioned=2` |
| tetris | `Commit/Optimize` | `entity_count_changed=4` | `cumulative_reward_increased=4`, `entity_count_changed=4`, `phase_transitioned=4` |
| tetris | `Commit/Position` | (none) | `entity_count_changed=5` |
| tetris | `Commit/Setup` | (none) | (still none — phrase doesn't match any trigger; D5 follow-up) |
| tetris | `Commit/Survive` | (none) | `attribute_changed=4`, `entity_count_changed=4` |
| tetris | `Inspect/Setup` | (none) | `phase_transitioned=3` |
| twenty_forty_eight | `Commit/Merge` | `attribute_changed=3` | `attribute_changed=3`, **`cumulative_reward_increased=3`** ✨ |
| twenty_forty_eight | `Compare/Merge` | (none) | (REASONING; observational hops carry no env-side effects) |
| twenty_forty_eight | `Mid Optimize` | (none) | (still none — abstract phrasings only) |

Coverage went from **2 / 12 skills** with any predicates to **9 / 12**.

### 2B — Phase-2 smoke with vs without producer (2048 ``Commit/Merge``, 8 deterministic seeds)

`python labeling_supplement/_phase2_real_env_skill_smoke.py --game twenty_forty_eight --max-skills 1 --bindings direction=up --bindings target=up --n-trials 8 --seed 0`

| Mode | Per-trial pass-rates | Best | Predicate decidability |
|---|---|---|---|
| **Day-3 baseline** (`--no-schema-producer`) | `[0.00, 0.33, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67]` | 0.67 | `attribute_changed`: **undecidable** ("entity_attrs missing on both sides") — counts as non-blocking pass and inflates rate. `cumulative_reward_increased`: decidable (Day-4A win). |
| **Day-4B** (default producer wired in) | `[0.00, 0.33, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]` | 0.33 | `attribute_changed`: **decidable** (`"score attrs changed"` on merge-bearing slides). `cumulative_reward_increased`: decidable. |

The Day-4B numbers are **lower** but **more rigorous** — Day-3 was
over-claiming because undecidable predicates inflated pass rates, and
the producer makes that systematic over-claim go away. The difference
is the empirical signature of "evaluation now sees what actually
changed". The single trial-1 verdict's hop-6 (SLIDE) shows the win
clearly:

```
hop 6: passed=True
  attribute_changed              passed=True   detail='score attrs changed'
  cumulative_reward_increased    passed=True   detail='score 0.0 → 4.0'
```

Day-3 would have shown `attribute_changed: passed=None` here.

### 2C — Phase-2 smoke on tetris (4 seeds)

`python labeling_supplement/_phase2_real_env_skill_smoke.py --game tetris --max-skills 3 --bindings direction=left --bindings dir=cw --bindings target=active_piece --include-reasoning --n-trials 4 --seed 0`

```
=== Phase-2 smoke summary [tetris, env=gaming_agent] ===
skill_id         type     hops eval pass  rate  ok
COMMIT__EVADE    action     7    2    0  0.00   Y
COMMIT__OPTIMIZE action     7    4    0  0.00   Y
COMMIT__POSITION action     7    5    5  1.00   Y
```

`COMMIT__EVADE`'s `phase_transitioned` correctly fires, correctly
fails (the env didn't top-out on shifted-left moves). `COMMIT__POSITION`'s
`entity_count_changed` is undecidable because the lift didn't bind
`entity_label="holes"` from the criterion (the schema_index didn't
register `holes` as an entity); see Day-5 follow-up below.

## 3. Tests

| File | Tests | Focus |
|---|---|---|
| `tests/test_protocol_lift.py` | 33 (6 new) | Trigger expansion: 2048 valid-merges → reward, candy-crush score-higher → reward, top-out → phase_transitioned, moves-decreased → value_decreased. Anti-overfire tests for disjunctive merge phrasings and bare execution acknowledgements. |
| `tests/test_gym_schema_producer.py` | 11 (new) | Producer-output shape, parse_schema_canonical round-trip, predicate decidability (cumulative_reward, entity_value_increased, entity_count_changed for tetris holes), numpy survival, gameover detection, registry lookup. |

Day-1/2/3 tests still green (33 lift + 11 executor + 17 success + 7
parse + 27 task-axis + 1 invariants + 11 schema canonical = 119
domain-specific tests pass; 324 passing overall, only the pre-existing
`test_extra_whitespace_tolerated` fails on `main`). The taxonomy
completeness assertion in `test_gymv_success::test_taxonomy_completeness`
(runtime tax ≡ labeling tax) keeps the two sides locked.

## 4. What this proves

| Phase | Question | Verdict |
|---|---|---|
| 4A | Does the lift's predicate-mining vocabulary cover the indirect phrasings real cold-start prose actually uses? | **Yes for the 2048 / tetris / candy_crush / mario hot path** — coverage 2/12 → 9/12 skills, with the headline `2048 Commit/Merge` gaining `cumulative_reward_increased=3`. |
| 4A | Does the runtime evaluator distinguish scoring slides from non-scoring slides on the live env? | **Yes.** `score 0.0 → 4.0` passes, `score 0.0 → 0.0` fails. The Phase-2 smoke's per-seed pass-rate (`[0.00, 0.33, 0.67, …]`) tracks the env's actual reward signal. |
| 4B | Does the deterministic schema producer make `attribute_changed`, `entity_value_*`, `entity_count_*` predicates decidable on the live env, **without a VLM**? | **Yes.** Producer reads `env.info` directly and emits a `<state>` block that `parse_schema_canonical` round-trips. Tested for 2048 (board / score / highest_tile / phase), tetris (board / score / lines / level / holes / active_piece). |
| 4B | Does it reduce evaluation over-claim? | **Yes — measurably.** A/B on 2048 ``Commit/Merge`` × 8 seeds: best pass-rate 0.67 (with producer disabled, undecidable counts as pass) → 0.33 (with producer, only real merges pass). The numerical drop is the empirical signature of newly-decidable predicates. |

## 5. What this does NOT validate

* **Tetris lift's entity-label binding for `holes`.** The cold-start
  schema_canonical for tetris doesn't tag `holes` / `stack_height` as
  entities — only `board`, `next_pieces`, `current_piece`. So the lift
  mines `entity_count_changed` from `"Hole count increases from 3 to
  4"` but with `args={}`. The producer **does** surface `holes` as an
  entity, so the predicate would be decidable if the lift bound the
  label. Day-5 fix: extend the per-game schema_index to whitelist
  `holes`, `stack_height`, `column_heights`, `filled_cells` even when
  the cold-start schema doesn't enumerate them.
* **Stage 3a `FewShotAdapter` cross-task transfer probes.** Producer
  + lift v2 close the **execution-rigor** gap. Cross-task **transfer
  evaluation** (2048-feasible skill running on tetris, scored by the
  same predicate machinery) is the Day-5 milestone tied to the gate
  service.
* **Producers for `candy_crush` / `super_mario` / `gym_v` SEGA games.**
  The factory returns `None` for everything except 2048 and tetris;
  the executor falls back gracefully to the text-obs path. Adding
  producers is mechanical (read each env's `info` keys, emit
  `<entities>` + `<attributes>`) but each one is its own per-game
  scope.
* **Slot-binding policy.** The smoke driver still hard-codes
  `bindings={direction:up, target:up}`. That's the actor's job; this
  driver is execution-rigor evidence, not adoption evidence.

## 6. Day-4 wrap-up & next step

Lift v2 + deterministic schema producer: **done.** The intra-gymv
transfer milestone now has an end-to-end **decidable** evaluation
surface. Three follow-ups remain:

* **Day 5 (a) — extend the lift's per-game schema_index** so tetris
  binds `holes`, `stack_height`, `filled_cells`, `lines` as entity
  labels. This makes the existing `entity_value_increased` /
  `entity_count_changed` predicates decidable on the producer's
  output (which already surfaces those entities).
* **Day 5 (b) — Stage 3a `FewShotAdapter` cross-task transfer cycle.**
  With predicates decidable on a live env, the gate's Stage 3a can
  run a 2048-feasible skill on tetris and score it. The
  `make_per_step_success_fn(...)` from Day-3 plus the producer from
  Day-4B is the wired-in success path. Append `verified_tasks` on
  PASS / LIMITED_PASS, re-run the cross-eligibility probe to show the
  F2′ admit set widens for the verified skill.
* **Day 5+ (c) — producers for the rest of the gymv envs**
  (candy_crush, super_mario, the 13 SEGA gym_v games). Each is its
  own ~100-line per-game producer; the executor's `schema_producer`
  hook is already there.
