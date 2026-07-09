# `decision_agents/` — TO-DO LIST

Engineering follow-ups from the visual-state → action / skill-query review.
Each item links the design observation to a concrete code anchor and an
actionable next step. Ordered by **impact × effort** (top = highest leverage).

---

## TL;DR of the review

The schema-native actor (`ActorAgent.step`) **does not consume pixels** for
game / web / OS tasks — it expects a pre-parsed `<state>…</state>` block
on `info["schema_text"]` and outsources visual→state conversion to whatever
runs upstream:

```text
actor_agent.py:348-358   step(observation: str, schema_text: Optional[str], …)
```

Pixels only re-enter through (a) SFT/GRPO subclasses that attach the image
as a side-channel content-part inside `_call_llm`, and (b) `VRHarness`
perception ops (`LOOK / CROP / READ_TEXT / COUNT / SEGMENT`) that emit
`info["schema_delta"]` for the next turn.

For the four 2D games this repo targets (2048, Candy Crush, Tetris, Super
Mario), `GymHarness.step` is a pass-through that does **no** perception —
so any visual-grounding error in the upstream schema is silently inherited
by the actor, the predicate dict, the skill query, and the action prompt.

The work below closes that integration gap.

---

## 1. Wire `canonical_schema.py` as the production schema source for game tasks  ★★★

**Why:** Today `GymHarness` ships no schema producer. The actor only works if
something upstream populated `info["schema_text"]`. The deterministic
canonical generator already exists and is byte-stable per env state, so we
can give the actor a reproducible source-of-truth without paying for an LLM
call per step.

**Anchors:**

```text
visual_grounding_tests/canonical_schema.py:715        make_canonical_schema(...)
visual_grounding_tests/canonical_schema.py:736        canonical_label_hint(...)
visual_grounding_tests/canonical_schema.py:806        MAX_ENTITIES_BY_GAME
decision_agents/core/harness_gym.py:67-79             GymHarness.step (pass-through)
decision_agents/actor_agent.py:1115-1138              runner reads schema_from_info
```

**Action items:**

- [ ] Add `env_wrappers/canonical_schema_wrapper.py` (gym `ObservationWrapper`)
      that calls `make_canonical_schema(game, info, …)` after every `step()`
      and stores the result on `info["schema_text"]`.
- [ ] Add a constructor flag on `make_gaming_env(..., emit_schema=True)` so
      callers opt in.
- [ ] Once the wrapper is in, retire the `schema_from_info` lambda fallback
      in `actor_agent.py:1116-1118` and require schemas explicitly — silent
      fallback to `_compact_from_text` hides bugs.

**Acceptance test:** `scripts/qwen3_decision_agent.py --game twenty_forty_eight`
runs end-to-end with the deterministic schema, byte-identical across seeds.

---

## 2. Make `schema_delta` perception a base-class hook for every harness  ★★★

**Why:** The `_merge_schema_delta` machinery is general-purpose but only
fires inside `VRHarness.step`. `GymHarness` has no plug for "if confidence
is low, run a `LOOK` op next turn", so the agent has zero recovery
mechanism if the upstream schema is wrong.

**Anchors:**

```text
decision_agents/actor_agent.py:518-592          _merge_schema_delta (general)
decision_agents/core/harness_vr.py:254-343      VRHarness emits schema_delta
decision_agents/core/harness_gym.py:67-79       GymHarness — no schema_delta path
```

**Action items:**

- [ ] Promote `maybe_emit_schema_delta(state) -> List[Entity]` to a base
      method on `Harness`. Default returns `[]`.
- [ ] In `GymHarness`, override it to: re-run `make_canonical_schema` on
      the new `info`, diff its entity bag against `state.schema.entities`,
      and emit only the entities that changed (an O(N) keys-bag delta).
- [ ] In `VRHarness`, refactor the existing op-driven delta path to use
      the same hook so both harnesses converge on one contract.

**Acceptance test:** When the upstream LLM schema misses a freshly-spawned
`tile_2`, the next `actor.step` sees it via `schema_delta` without re-emitting
the full `<state>` block.

---

## 3. Use `_eval_agreement.py` as an in-loop drift monitor, not just an offline tool  ★★

**Why:** We just measured 0.77–1.00 entity-IoU between the text-head and
image-head schemas on 2048. That number IS the contract surface area the
actor has to be robust to. Today it only runs offline against pre-saved
`steps.jsonl` files; it should also fire per-step in production with a
threshold that triggers a fallback (canonical schema or a `LOOK` op).

**Anchors:**

```text
visual_grounding_tests/_eval_agreement.py:43-65     parse / keys / jaccard
decision_agents/actor_agent.py:382-394              schema parse + delta merge
```

**Action items:**

- [ ] Extract `parse() / keys() / jaccard()` into a tiny
      `decision_agents/core/perception/agreement.py` helper.
- [ ] In the runner (`actor_agent.py:1115-1145`) compare the upstream LLM
      schema against the canonical schema (when both present); if
      `1 - IoU > drift_threshold`, log a warning and prefer the canonical
      block for that step.
- [ ] Surface the drift number on `ActorDecision.diagnostics` so it ends
      up in the per-episode log alongside `parse_path` and `queried_skill`.

**Acceptance test:** A unit test feeds two schemas with synthetic noise
(extra `Perf 0.00` HUD entity) and asserts the runner picks the canonical
block whenever IoU drops below the threshold.

---

## 4. Implement real perception backends — retire the Mock* stubs  ★★

**Why:** README explicitly defers Grounding-DINO / SAM-2 / PaddleOCR to
"Phase 8.1". Today `LOOK("red cup")` returns `None` whenever no detector is
bound, which silently degrades to a scratchpad write. That's invisible
failure: tests pass, agent does nothing useful.

**Anchors:**

```text
decision_agents/core/perception/detector.py:22-60     RegionDetector protocol
decision_agents/core/perception/segmenter.py          Segmenter protocol
decision_agents/core/perception/ocr.py                OCREngine protocol
decision_agents/core/harness_vr.py:431-461            _do_look (returns None on no detector)
```

**Action items:**

- [ ] Add `GroundingDinoDetector(RegionDetector)` adapter in
      `core/perception/grounding_dino.py`. Use the existing `image_bytes` +
      open-vocab query contract; cache via the existing `cache.py`.
- [ ] Add `Sam2Segmenter(Segmenter)` adapter.
- [ ] Add `PaddleOCRReader(OCREngine)` adapter.
- [ ] Make `_do_look` raise (or surface a warning on
      `info["perception_warning"]`) when no detector is bound — silent
      `None` is the worst possible default.

**Acceptance test:** `tests/test_perception_real_backends.py` exercises
each adapter on a fixed PNG and asserts a non-empty result with bbox
inside the image bounds.

---

## 5. Implement `BrowserHarness.step` and `OSWorldHarness.step`  ★★

**Why:** Both currently raise `NotImplementedError`. The README claims
five task families; in practice today it's one and a half (game + VR with
mock perception). Anyone running `harness_browser.py` or `harness_osworld.py`
hits a wall on the first action.

**Anchors:**

```text
decision_agents/core/harness_browser.py    BrowserHarness.step → NotImplementedError
decision_agents/core/harness_osworld.py    OSWorldHarness.step → NotImplementedError
```

**Action items:**

- [ ] Wire `BrowserHarness.step` to the existing `browsergym_wrapper`
      (already imports cleanly per `environment.yml`).
- [ ] Wire `OSWorldHarness.step` to `osworld_wrapper`.
- [ ] Implement `valid_actions()` for both: scrape from BrowserGym /
      OSWorld action spaces; cap at `MAX_VALID_ACTIONS_IN_PROMPT`.
- [ ] Implement `action_kind()` so the cost shaper distinguishes
      deliberation vs. primitive (mirrors `VRHarness`).

**Acceptance test:** `tests/test_harness_browser_smoke.py` reaches a
search box on `example.com` and types a query.

---

## 6. Add per-action `action_kind` to `GymHarness` for cost-aware shaping  ★

**Why:** `GymHarness.action_kind` always returns `"primitive"`
(`harness_gym.py:102-104`), so per-action reward shaping is flat. The
`RewardComputer` is happy to charge `r_cost` differently per action
class — we're just not feeding it the signal.

**Anchors:**

```text
decision_agents/core/harness_gym.py:102-104           always returns "primitive"
decision_agents/reward_func.py                        RewardConfig.cost_for_action_kind
decision_agents/core/harness_vr.py                    has real action_kind taxonomy
```

**Action items:**

- [ ] Classify actions per-game: `"primitive"` (move/swap), `"deliberate"`
      (re-plan / undo), `"meta"` (query_skill / query_mem). Tetris
      `hard_drop` is primitive; 2048 `up/down/left/right` is primitive;
      candy_crush `swap (i,j)` is primitive — but `LOOK`-style "scan
      board" ops (when added) are deliberate.
- [ ] Update `RewardConfig.cost_for_action_kind` defaults for game tasks
      so `QUERY_SKILL` / `QUERY_MEM` cost > 0.

---

## 7. Measure marginal contribution of each block in the action prompt  ★

**Why:** The action prompt stacks task + intention + state-summary +
entity-block + active-skill-block + scratchpad + recent-actions +
numbered-valid-actions, then asks for `THOUGHT:`/`ACTION:` in 200 tokens.
On smaller models this is noisy. Worth measuring.

**Anchors:**

```text
decision_agents/actor_agent.py:888-925     _build_action_prompt
```

**Action items:**

- [ ] Add a `prompt_blocks: Set[str]` config knob on `ActorAgent`
      (`{"task", "intention", "summary", "entities", "skill", "scratchpad", "history"}`).
- [ ] Run an ablation on each game: drop one block at a time, log
      success rate, average reward, and parse-path distribution.
- [ ] Document which blocks are load-bearing for which model class
      (gpt-5 vs. gpt-4o vs. Qwen3-VL-8B) in the README.

---

## 8. Let the actor request perception explicitly before committing to an action  ★

**Why:** Today perception runs once per env step, before action selection,
regardless of what the agent wants to look at. The interesting failure mode
on 2D games is "the LLM is unsure where the next 4-tile is" — which is a
`LOOK` opportunity that the architecture supports but doesn't expose to
the actor as a deliberate sub-step in the game harness.

**Anchors:**

```text
decision_agents/actor_agent.py:471-498         action selection (single call)
decision_agents/core/harness_vr.py:254-343     VRHarness already supports LOOK ops
```

**Action items:**

- [ ] Add a two-pass action mode (config-gated):
      pass 1 outputs `LOOK_AT: <slot>` *or* `ACTION: <verb>`;
      if `LOOK_AT`, run perception, merge `schema_delta`, prompt again.
- [ ] Track in `ActorDecision.diagnostics` how often the agent chose to
      look first vs. acted directly — this is the empirical evidence that
      the visual head adds value over the canonical schema.

---

## 9. Schema-grammar drift safety net  ★

**Why:** `schema_parser.py:40-70` imports the canonical regexes from
`vlm_wrapper.schema` but ships local copies as a fallback. If
`vlm_wrapper` changes the grammar (adds a section, renames a tag), the
fallback path silently parses to the old schema and predicates go
stale.

**Anchors:**

```text
decision_agents/schema_parser.py:40-70    import + local-copy fallbacks
vlm_wrapper/schema.py                     authoritative grammar
```

**Action items:**

- [ ] Add a unit test `tests/test_schema_grammar_in_sync.py` that imports
      the live regex constants from `vlm_wrapper.schema` and asserts
      byte-equality with the local fallback strings — fails CI on drift.
- [ ] Or, drop the local fallback entirely and require `vlm_wrapper` as
      a hard dep; the offline-friendliness comment is no longer worth
      the silent-stale-parser risk.

---

## 10. Documentation: keep the README honest about which paths actually run  ★

**Why:** The README's "five task families" claim is accurate as a
roadmap but not as today's runtime. New contributors are likely to pick
up `harness_browser.py`, hit `NotImplementedError`, and bounce. The README
already has an open-work table — it just needs to be linked from the
top.

**Action items:**

- [ ] Add a "Status today" section at the top of `decision_agents/README.md`
      listing each task family with a ✅ / ⚠️ / ❌ marker and a one-liner.
- [ ] Cross-link items 4 / 5 / 8 of this TO-DO list as the path to
      promoting ⚠️ / ❌ rows to ✅.

---

## Out of scope for this list (handled elsewhere)

- **Skill creation / synthesis** — read-only inside `decision_agents/`. Lives
  in the sibling `skill_agents` package (`segment` / `contract` /
  `curator` LoRAs).
- **Schema-producer LLM heads** — handled in `visual_grounding_tests/`
  (`generate_envwrappers_text_schema.py`,
  `generate_envwrappers_image_schema.py`,
  `generate_gymv_text_schema.py`, `generate_gymv_image_schema.py`).
- **Skill-bank retrieval engine** — `skill_agents.query.SkillQueryEngine`,
  used through the `SkillProvider` protocol seam.

---

## Suggested ordering

If the team has limited cycles, do them in this order:

1. **(1) canonical-schema wrapper** — unblocks every game-task test and
   makes the agent reproducible.
2. **(3) drift monitor** — once (1) lands, this becomes the audit channel
   that proves the LLM heads are doing useful work (or aren't).
3. **(2) base-class `schema_delta` hook** — generalises the recovery
   path to all harnesses.
4. **(4) real perception backends** — the next correctness leap for VR / video.
5. Everything else as bandwidth permits.
