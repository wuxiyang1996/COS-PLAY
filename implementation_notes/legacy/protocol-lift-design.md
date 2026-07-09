# Protocol lift — design (Day 1, Phase 0)

> **Status:** design only. Implementation lands in
> [`labeling/_decorate_skill_records.py`](../../labeling/_decorate_skill_records.py)
> (extension, not replacement). Targets the
> [`harness/README.md` §21](../../harness/README.md) blocker called out by the
> [`harness/README.md` audit](../../harness/README.md), and unblocks the
> [§Intra-gymv transfer](../../harness/README.md) milestone.
> **Last reviewed:** 2026-04-30.

## 1. Problem

Cold-start `protocol` is a dict of prose-string lists:

```json
{
  "preconditions": ["…"],
  "steps": ["Read the current board and enumerate the four possible slide directions: up, down, left, right.", "…"],
  "success_criteria": ["…"],
  "abort_criteria": ["…"]
}
```

The harness consumes `protocol` via `iter_hops`:

```40:44:Multi-hop-Reasoning-VLM-Agent/harness/adapters/_common.py
def iter_hops(skill: SkillRecord) -> Iterator[Tuple[int, Dict[str, Any]]]:
    for i, hop in enumerate(skill.protocol):
        if not isinstance(hop, dict):
            continue
        yield i, hop
```

and `normalize_hop_action`:

```61:63:Multi-hop-Reasoning-VLM-Agent/harness/adapters/_common.py
def normalize_hop_action(hop: Dict[str, Any]) -> str:
    """Map a hop's `op` / `action` field to a canonical action_type."""
    return str(hop.get("action") or hop.get("op") or hop.get("type") or "STEP").upper()
```

There are two distinct loaded-shapes today depending on the entry path, and each has its own residual gap:

| Path | Loader | `skill.protocol` after load | `iter_hops` yields | `normalize_hop_action` returns |
|---|---|---|---|---|
| Direct from `skill_bank.jsonl` (e.g. `lifecycle.ingest_draft` on the raw entry) | passthrough | the raw `{"steps": [...], …}` dict | **0 hops** (dict keys fail `isinstance(hop, dict)`) | n/a — never called |
| Via `labeling_supplement._harness_io_helpers.record_from_bank_entry` (the dump-driver / probe path) | `_wrap_protocol_steps` | `[{"action": "EXEC", "payload": {}, "notes": "<prose>"}, …]` (one row per prose sentence) | **N hops** (all dicts) | always `"EXEC"` |

So the audit's "zero hops" framing (`harness/README.md` §21) is correct only for the raw-load path. The dump-driver / Phase-0 path already has a *shape lift* deployed via `_harness_io_helpers._wrap_protocol_steps` — every prose step becomes a typed dict, `iter_hops` yields N hops, but every hop normalises to `"EXEC"` with empty payload. **The shape gap is closed; the semantic gap is open.**

The semantic gap has three components:

1. **No real verb.** Every hop is `"EXEC"`. `GymvAdapter`'s real executor (Day 6–7) has nothing to dispatch on — it can't tell `SLIDE` from `INSPECT` from `STOP`.
2. **No payload slots.** `payload={}` always. There are no `${direction}` / `${target_entity}` placeholders for `HopBindings.resolve` to fill.
3. **No effects.** Cold-start `contract.eff_add` / `eff_del` are empty for every skill in `run_20260430_030637` (one exception: `COMPARE/MERGE` ships world-blob strings that aren't a typed predicate language). Without typed effects the success_fn (Day 4–5) has nothing to test against.

The lift must close all three, in the same pass, idempotently. Day-1 Phase-0 measurement (`labeling_supplement/_phase0_cross_eligibility_probe.py`) confirmed that the eligibility filter admits 100 % of cross-game skills today — that's a *separate* problem (§22 task axis, Day 2) and **does not depend on this lift**. Eligibility never reads protocol semantics; it only checks `(status, domain, adapter, can_handle)`. The lift's value is downstream of eligibility, in `run_skill` and `success_fn`.

## 2. Source data is richer than the prose suggests

The cold-start corpus carries a structured-state-schema for *every* actor step under `episode_*.json:experiences[i].metadata.schema_canonical`. It is uniform across the four `env_wrappers` games. Excerpt from a 2048 episode 0:

```
<entities>
e1[type=region,  label=board,         ontology=container_entity]
e2[type=object,  label=tile_2,        ontology=selectable_entity]
e4[type=region,  label=empty_cells,   ontology=navigable_region]
e5[type=text,    label=highest_tile,  ontology=goal_indicator]

<affordances>
e1.affords=[inspect]
e2.affords=[select, track, compare]
e4.affords=[approach]
e5.affords=[read]
```

…and from a tetris episode 0:

```
<entities>
e1[type=region, label=playfield,       ontology=container_entity]
e2[type=object, label=active_piece_S,  ontology=tracked_entity]

<affordances>
e1.affords=[inspect]
e2.affords=[move, rotate, drop]
```

Two facts the lift exploits:

1. **The verb taxonomy already exists on disk.** Every game's `<affordances>` section lists the verbs the schema thinks each entity supports. The intersection across env_wrappers games is `{inspect, read}`. The union is `{inspect, read, select, track, compare, approach, move, rotate, drop, slide, swap, click, type, place}`. This is the *abstract verb taxonomy* — no LLM call required to mine it.
2. **The slot ontology already exists too.** The per-entity `ontology=` tag (`container_entity / selectable_entity / navigable_region / goal_indicator / tracked_entity`) gives the slot type system. A typed-hop slot is `${role:type}` where `type ∈ {selectable_entity, …}` and `role` is the placeholder name.

The lift's job is to normalise each prose step against this dual table.

## 3. Target shape

A `SkillRecord.protocol` row after the lift:

```json
{
  "op": "SLIDE",
  "payload": {"direction": "${dir}"},
  "preconditions": [
    {"type": "domain_eq", "value": "gymv"},
    {"type": "task_eq",   "value": "twenty_forty_eight"}
  ],
  "effects_add": [{"type": "tile_count_increase", "key": "highest_tile"}],
  "effects_del": [],
  "evidence_role": "COMMIT",
  "notes": "<original prose step kept verbatim for diffing>"
}
```

Three deliberate choices:

- **Key name `op`** (also accepts `action` / `type`) — pinned by `normalize_hop_action`. The user-suggested `verb` would silently fall through to `"STEP"`. This is the load-bearing key.
- **`${slot}` placeholders only inside `payload`** — pinned by `HopBindings.resolve` in `_common.py`. Anything outside `payload` is not slot-resolved.
- **Effects on the hop, not just the contract.** Cold-start contracts ship empty `eff_add` / `eff_del`. The success_fn (Day 4–5) needs typed effect predicates *per hop* so it can localise which step in a multi-hop skill failed. The decorator should populate both: per-hop `effects_add / effects_del` and the contract roll-up.

## 4. Verb taxonomy (gymv-only, v0)

Mined from (a) the union of `<affordances>` across the four env_wrappers games, and (b) **empirical sweep of every prose step** in `run_20260430_030637` (80 prose steps across 18 skills, 4 games). The taxonomy is split by hop role — env, gather, reason, control — because the executor for each class is different (env-class hops dispatch to `GymvAdapter._executor`; the others can be evaluated in-harness without a real env step).

### 4.1 Verb table

| Bucket | Verb | Lemma triggers | Slot signature | Intended evidence role |
|---|---|---|---|---|
| env-mutating | `SELECT` | select, choose, pick | `target: selectable_entity` | COMMIT |
| env-mutating | `SWAP` | swap, exchange | `lhs, rhs: selectable_entity` | COMMIT |
| env-mutating | `SLIDE` | slide, shift, swipe | `direction: enum{up,down,left,right}` | COMMIT |
| env-mutating | `MOVE` | move, drift, navigate, walk, step | `target: tracked_entity, direction: enum` | COMMIT |
| env-mutating | `ROTATE` | rotate, turn, spin | `target: tracked_entity, dir: enum{cw,ccw}` | COMMIT |
| env-mutating | `DROP` | drop, land | `target: tracked_entity` | COMMIT |
| env-mutating | `PLACE` | place, position, set, lock | `target: tracked_entity, anchor: container_entity` | COMMIT |
| env-mutating | `APPROACH` | approach, head, begin (movement) | `target: navigable_region` | COMMIT |
| env-mutating | `EXECUTE` | execute, perform, apply, do, merge, compress, clear | (catch-all; payload free-form) | COMMIT |
| gather | `READ` | read, report | `target: goal_indicator` | GATHER |
| gather | `INSPECT` | inspect, examine, look, scan, observe, identify, list, enumerate, find, assess | `target: container_entity` | GATHER |
| gather | `TRACK` | track, monitor | `target: tracked_entity` | GATHER |
| reason | `COMPARE` | compare | `lhs, rhs: any` | REASON |
| reason | `EVALUATE` | determine, evaluate, decide, score, rank, compute | `subject, criterion: any` | REASON |
| reason | `SIMULATE` | simulate, predict | `move, base_state: any` | REASON |
| reason | `PREFER` | prefer, favor | `chosen, alternatives: any` | REASON |
| reason | `PENALIZE` | penalize, avoid, discount | `subject, criterion: any` | REASON |
| reason | `VERIFY` | verify, check, confirm, ensure | `predicate: <effect_predicate>` | VERIFY |
| control | `STOP` | stop, abort, terminate | (no slots) | (passthrough) |
| control | `CONTINUE` | continue, resolve | (no slots) | (passthrough) |
| control | `KEEP` | keep, maintain, let | `invariant: <effect_predicate>` | (passthrough) |

### 4.2 Coverage on cold-start corpus (measured 2026-04-30)

Naive first-word match against an earlier 12-verb table covered only 38.8 % of prose steps. With (a) leading-subordinator stripping (`if / when / after / for / while / during / unless / …`) and (b) a downstream walk that finds the first matching verb anywhere in the sentence (not just the head), the table above covers **74 / 80 = 92.5 %** of prose steps in `run_20260430_030637`:

| Bucket | Lemma → verb match rate | Verbs ranked by use |
|---|---|---|
| First-word match (after subordinator strip) | 55 / 80 = 68.8 % | EXECUTE (13), INSPECT (9), SELECT (6), PREFER (6), CONTINUE (5), KEEP (5), READ (4), EVALUATE (4), VERIFY (4), STOP (3), SLIDE (3), PLACE (3), ROTATE (2), DROP (2), TRACK (2), MOVE (1), PENALIZE (1), COMPARE (1) |
| Downstream-walk rescue | 19 / 80 = 23.8 % (cumulative 92.5 %) | (same buckets; rescued counts add to those) |
| Unmatched (fall through to `EXEC`) | 6 / 80 = 7.5 % | first-words: `lock`, `let`, `assess`, `begin`, `soon` (4 of these are now triggers above; the residue after the next pass is 2/80 ≈ 2.5 %) |

Key implementation notes that fall out of the measurement:

- **Tetris and super_mario contribute most of the "rescued" matches.** Their prose is more narrative ("Lock the Z even if this creates one additional hole, …" — `lock` not in the v0 head set). Adding `lock → PLACE`, `assess → INSPECT`, `let → KEEP`, `begin → APPROACH/EXECUTE` lifts coverage to ≥ 97 %.
- **`EXECUTE` is the highest-frequency verb.** That's because cold-start prose often uses the umbrella "Execute the move" / "Apply the slide" before naming the specific env action. Day-2 implementation should treat `EXECUTE` as a *defer-to-payload* verb — the payload's `direction` / `target` slot disambiguates which real env op to call.
- **`COMPARE` is rare (1/80) but `EVALUATE` is frequent (4/80).** They're distinct: `COMPARE` is binary (lhs vs rhs, no decision); `EVALUATE` ranks N options. Don't fold them.
- **`PREFER` (6/80) and `PENALIZE` (1/80) are reasoning-only.** They produce a softmax over options; they don't step the env. The executor for these reads `state.facts` and returns a scored list — no real-env binding required, which means they work today even before `GymvAdapter.set_executor` lands.

### 4.3 Out of scope for v0

- **Cross-domain verbs.** `CLICK / TYPE / SCROLL / NAVIGATE_URL` (browser), keyboard / mouse primitives (osworld), frame-time queries (video), MCQ-style answers (visual_reasoning) are all `harness/README.md` §16 follow-ups. The decorator must refuse to emit these for `applicable_domains=["gymv"]` skills.
- **Compound verbs.** "compress and merge" (2048 internal) decomposes into two hops at extraction time; the lift does not synthesize compound atomics.
- **Trainable verb classifier.** The 92.5 % v0 floor is sufficient given the corpus size (80 prose steps). If the corpus grows past ~10 games or ~40 skills, a small `cold_start_labeling/`-style LLM call to disambiguate the `EXEC` residue may pay off — but that's not Day-2 work.

## 5. Effect predicates (gymv-only, v0)

Mined from `metadata.schema_canonical` deltas between consecutive `experiences[i]` and `experiences[i+1]`, projected onto a small typed predicate language:

| Predicate | Args | Fires when |
|---|---|---|
| `entity_value_increased` | `entity_label`, `min_delta=1` | the value attribute of an entity with `label==entity_label` strictly increases by ≥ `min_delta` between the two schemas |
| `entity_value_decreased` | `entity_label`, `min_delta=1` | …strictly decreases |
| `entity_count_changed` | `entity_label`, `delta` | the count of entities with that label changes by exactly `delta` |
| `entity_appeared` | `entity_label` | label absent in prev schema's `<entities>`, present in next |
| `entity_disappeared` | `entity_label` | label present in prev schema's `<entities>`, absent in next |
| `attribute_changed` | `entity_label`, `attribute` | the named attribute of an entity changed value |
| `cumulative_reward_increased` | `min_delta=0.0` | `step.reward > min_delta` (read directly from the actor record) |
| `phase_transitioned` | `from`, `to` | `<state_flags>.phase` moved `from → to` |

Concrete instances per game (sampled from the four bank rows):

| Skill | Effect |
|---|---|
| `twenty_forty_eight / COMMIT/MERGE` | `entity_value_increased(highest_tile)` ∨ `entity_count_changed(empty_cells, +1)` ∨ `cumulative_reward_increased(0.0)` |
| `twenty_forty_eight / mid:OPTIMIZE` | `cumulative_reward_increased(0.0)` ∧ ¬`phase_transitioned(early, gameover)` |
| `tetris / COMMIT/SETUP` | `attribute_changed(active_piece_*, pos)` |
| `tetris / COMMIT/OPTIMIZE` | `entity_value_increased(lines)` |
| `tetris / COMMIT/EVADE` | ¬`phase_transitioned(early, gameover)` |
| `tetris / COMMIT/SURVIVE` | `attribute_changed(active_piece_*, value)` (next-piece visible) ∧ ¬`phase_transitioned(*, gameover)` |
| `candy_crush / COMMIT/CLEAR` | `entity_count_changed(<color_label>, ≤ -3)` ∨ `cumulative_reward_increased(0.0)` |
| `super_mario / COMMIT/NAVIGATE` | `attribute_changed(mario, pos)` |

### 5.1 Predicate inputs — what the evaluator actually reads

The natural-sounding "consult `state.facts[entity_label]`" turns out to **not** work against `StateSchema` as currently parsed. Empirical check on a 2048 step:

```
state.task         = "make_gaming_env/twenty_forty_eight"
state.domain       = "gymv"
state.facts.keys() = ['goal']                            ← only the natural-language goal
state.elements[0]  = {'id':'e1','type':'region','label':'board','ontology':'container_entity', …}
                                                          ← entity values like e5.value=2 are NOT here
```

`labeling_supplement._harness_io_helpers.parse_schema_canonical` parses the `<entities>` block (into `state.elements`) and the `goal` line (into `state.facts['goal']`), but **does not parse the `<attributes>` block at all** — `e5.value=2` etc. are dropped. The same parser also has a small bug: the regex splits attribute payloads on `,`, so `pos=0,1,1,1` only retains the first value.

The Day-4 success-fn evaluator therefore has two concrete options. The lift design is agnostic between them; pick whichever lands faster:

- **Option A (preferred): extend `parse_schema_canonical` to also populate `state.facts` from the `<attributes>` block.** The decorator emits predicates that reference `state.facts[entity_label][attribute]` (or just `state.facts[entity_label]` when there's a unique value attribute). Pure additive; same parser everyone goes through.
- **Option B: read predicate inputs from `step.state` (the dict-as-string).** No parser change, but each evaluator becomes a separate ad-hoc dict-of-dict walk. Brittle once schemas evolve.

Both options leave `cumulative_reward_increased` unchanged — that one reads `step.reward` from the rollout record directly, not the schema.

### 5.2 RewardLogger and where the predicate verdict lands

The harness's reward sink today carries scalar fields:

```24:34:Multi-hop-Reasoning-VLM-Agent/harness/reward_logger.py
@dataclass
class RewardLogEntry:
    episode_id: str
    skill_id: str
    skill_version: str
    domain: str
    success: bool
    score: Optional[float]
    cost: Dict[str, float] = field(default_factory=dict)
    parent_run_id: Optional[str] = None
    transfer_label: Optional[str] = None
    timestamp: float = 0.0
```

There is no `r_env / r_cost / r_total` decomposition in the live `RewardLogger` — that's `harness/README.md` audit §10 ("structured `reward_components`") which is listed as missing. For Day-4 the success-fn populates `success: bool` (predicate aggregate) and `score: Optional[float]` (a 0–1 fraction of effects fired), and optionally `transfer_label` for cross-task probes. Adding a structured `r_follow / r_env / r_cost / r_total` decomposition is **separate audit work** and explicitly out of scope for the lift.

### 5.3 Notes on the predicate set itself

- `entity_value_increased` etc. are **type-checked** at decoration time — the decorator refuses to emit a predicate referencing an entity label that does not appear in any cold-start episode's schema for the same game.
- The eight predicates cover the full union of cold-start `eff_add` rows in `run_20260430_030637`. The single bank row that has populated effects today (`COMPARE/MERGE` in 2048) emits world-blob strings (`world.{'board'=[[…]], 'highest_tile':128, …}`); those become `attribute_changed(board, value)` after the lift.

## 6. The transformer

Lives at `labeling/_decorate_skill_records.py`. Pure offline / deterministic / idempotent. Single new function `_lift_protocol_to_typed_hops(skill_record_dict, *, schema_index)` invoked from `_decorate_skill_record`:

```python
def _lift_protocol_to_typed_hops(skill, *, schema_index):
    """Return (typed_protocol, contract_eff_add, contract_eff_del).

    schema_index is built once per game by scanning every
    experience.metadata.schema_canonical for the entities / affordances /
    state_flags vocabulary. Pure deterministic mapping; no LLM.
    """
    prose_steps   = (skill.get("protocol") or {}).get("steps") or []
    preconditions = (skill.get("protocol") or {}).get("preconditions") or []
    success_crit  = (skill.get("protocol") or {}).get("success_criteria") or []
    abort_crit    = (skill.get("protocol") or {}).get("abort_criteria") or []
    role          = (skill.get("evidence_role") or "COMMIT").upper()
    game          = (skill.get("provenance") or {}).get("source_name")

    typed: List[Dict[str, Any]] = []
    for prose in prose_steps:
        verb, payload, slot_types, mode = _classify_prose_step(
            prose, role, schema_index[game],
        )
        eff_add, eff_del = _mine_effects_for_step(
            prose, success_crit, abort_crit, schema_index[game],
        )
        typed.append({
            "op": verb,                 # ∈ Section 4 taxonomy ∪ {"EXEC"}
            "payload": payload,         # `${slot}` placeholders for bind-time resolution
            "slot_types": slot_types,   # parallel dict: slot_name → schema ontology type
            "preconditions": _classify_preconditions(preconditions, schema_index[game]),
            "effects_add": eff_add,
            "effects_del": eff_del,
            "evidence_role": role,
            "notes": prose,             # diffable original prose (kept verbatim)
            "lift_mode": mode,          # "first" | "rescued" | "fallback_exec" — for coverage metrics
        })
    contract_add = sorted({e["type"] for h in typed for e in h["effects_add"]})
    contract_del = sorted({e["type"] for h in typed for e in h["effects_del"]})
    return typed, contract_add, contract_del
```

`_classify_prose_step` is the deterministic verb-classifier validated empirically in §4.2:

1. **Tokenise** the prose to lowercase ASCII words.
2. **Strip leading subordinators** (`if / when / after / before / during / while / for / because / unless / until / as / though / given / the / a / an / this / that / any / no / with / without / in / on / of / by`). Continue until the first content token.
3. **First-word match.** If the head token is in the lemma-trigger table (Section 4.1), return `(verb, mode="first")`. This catches 68.8 % of cold-start prose.
4. **Downstream-walk rescue.** Otherwise scan the rest of the tokens; the first match wins. This catches another 23.8 % (cumulative 92.5 %).
5. **Fallback.** If still no match, return `("EXEC", payload={"raw": prose}, mode="fallback_exec")`. Tally on `_lifecycle_meta.json: lift_fallback_exec: int` so coverage is observable run-over-run.

Slot population is shallow in v0:

- For env-mutating verbs, the decorator looks for entity references in the prose (matching against `schema_index[game].entity_labels`) and emits `payload={"target": "${target}"}` with `slot_types={"target": "<ontology>"}`. Direction enums (`up / down / left / right / cw / ccw`) are extracted by regex against a small list.
- For reasoning / control verbs, payload is `{}` and the verb itself is the load-bearing field.
- Effects are mined from the `<success_criteria>` / `<abort_criteria>` lists, not from the prose step — those lists already carry actionable phrases like "highest tile increases" / "stack reaches the top".

This matches the existing decorator's idempotence rule: rows that already carry `protocol: List[Dict]` are passed through untouched, so re-running the decorator on a partially-lifted bank is a no-op. The two distinct loaded-shapes from §1 collapse to the same shape after the lift, by construction.

## 7. Tests (live alongside the transformer)

| Test | Asserts |
|---|---|
| `test_iter_hops_yields_at_least_one_per_skill` | Every row of every existing bank, after lift, yields `len(list(iter_hops(record))) ≥ 1`. Today this is 0 for every cold-start skill. |
| `test_normalize_hop_action_is_in_taxonomy` | Every yielded hop normalises to a verb ∈ Section-4 taxonomy ∪ `{EXEC}`. The `EXEC` count is logged but not failing. |
| `test_payload_slots_resolve` | Every `${slot}` in `payload` resolves against `state.facts.keys()` for at least one cold-start episode of the same game. |
| `test_effects_round_trip` | For every `(skill, episode)` pair, evaluating `effects_add / effects_del` on consecutive `(state_t, state_t+1)` produces a non-zero satisfaction rate when the actor's `selected_skill_id == skill.id`. (Sanity floor — not a quality bar.) |
| `test_decorator_is_idempotent` | Running the decorator twice on the same bank produces an identical jsonl byte-for-byte. |

## 8. Acceptance gate for moving to Day 2

Calibrated by the §4.2 empirical sweep (74/80 = 92.5 % already achieved by the classifier):

- `test_iter_hops_yields_at_least_one_per_skill` passes for all 18 skills in `run_20260430_030637`. (Sanity floor; the existing `_wrap_protocol_steps` workaround already meets this. The lift's job is to replace `EXEC` with real verbs.)
- `_lifecycle_meta.json: lift_fallback_exec_pct ≤ 10 %` (i.e., ≥ 90 % of prose steps mapped to a non-`EXEC` verb). The §4.2 sweep shows 7.5 % fallback today; a small expansion of the lemma triggers (`lock`, `assess`, `let`, `begin`) drops that to ≤ 2.5 %, comfortably under the 10 % gate.
- `test_effects_round_trip` shows non-zero satisfaction for each of the four games on at least one `(skill, episode)` pair. Predicate inputs read either via the parser-extension (Option A in §5.1) or directly from `step.state` (Option B); both paths are gate-acceptable.
- **Predicate-input parser path is decided.** Either Option A or Option B from §5.1 is committed in code with a one-line note in `parse_schema_canonical`'s docstring saying which it is.

If any of these fail, the v0 taxonomy / parser path is too small — extend Section 4 / 5, don't lower the gate. The 10 % gate is calibrated against measured 7.5 %; raising the gate to 5 % is the next-tier acceptance, achievable by adding the 4 missing lemma triggers and the `soon`-style adverb to skip-leaders.

## 9. What this design explicitly does NOT include

- **No browser / osworld / video / visual_reasoning verbs.** Out of scope; gymv-only.
- **No LLM call.** The decoration is purely deterministic against the schema vocabulary. The cold-start LLM already produced the prose; lifting it is mechanical.
- **No protocol semantics beyond "verb + payload + effects".** No goals, sub-goals, branching, or loops. Hops are flat and ordered. Branching belongs to the actor; the harness only re-checks effects per hop.
- **No `feasible_tasks` / `verified_tasks`.** That is the Day 2 task-axis change, not part of the protocol lift.
- **No trainable verb classifier.** Section 4's lemma-match is sufficient given the four-game cold-start corpus's small verb vocabulary. If the corpus grows past ~10 games, revisit.

## 10. Why this is the right Day-1 design lock

1. **Verb taxonomy is empirically mined and measured, not invented.** §4.2 reports 92.5 % coverage (74/80) on the actual cold-start corpus with leading-subordinator stripping plus downstream-walk rescue. The unmatched residue is 5 lemmas, all real verbs that drop into the table with one line each. Zero design freedom for someone to bikeshed later.
2. **Slot ontology and effect predicates are mined from `schema_canonical`,** also pre-existing on disk. The decorator type-checks predicates against the per-game entity vocabulary at decoration time so a typo can't ship.
3. **The transformer extends an existing idempotent file** (`labeling/_decorate_skill_records.py`); no new pipeline, no new run-id, no migration.
4. **Phase-0 measurement is reproducible and the cross-contamination signal is decoupled from this lift.** The lift's value is downstream of eligibility (in `run_skill` and `success_fn`); the §22 task-axis fix is what drives cross-game admission to 0. Both can land independently in any order.
5. **The lift's success criterion is single-valued and instrumented.** `lift_fallback_exec_pct` lands in `_lifecycle_meta.json` per run; that one number tells the next reviewer whether the lemma table needs more triggers.
6. **The two-loaded-shapes ambiguity from §1 collapses after the lift.** Rows that already carry `protocol: List[Dict]` are passed through untouched; rows with the raw `{"steps": [...]}` shape are lifted. After one decorator pass, every row is in the same shape regardless of entry path, and the dump-driver workaround (`_wrap_protocol_steps`) becomes a no-op fallback.
