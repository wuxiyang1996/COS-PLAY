# Phase-0 baseline — twenty_forty_eight ↔ tetris (2026-04-30)

> Day-1 deliverable per
> [`implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md`](../../implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md)
> §7 Phase 0. Drives the
> [`harness/README.md` audit §22](../../harness/README.md) measurement
> baseline.

## 1. What we ran

Two runs against `labeling/skill_bank_out/run_20260430_030637` and
`labeling/skill_actions_out/run_20260430_064325`, sources
`twenty_forty_eight` + `tetris`, 3 episodes × 50 steps per source.

| Run | Driver | Bank input | Candidate set fed to filter |
|---|---|---|---|
| `run_phase0_20260501_012106` | `bash labeling_supplement/run_dump_harness_io.sh --surface online` | per-source bank (3 / 6 skills) | actor's `retrieved_skill_ids` per step (cold-start RAG) |
| `run_phase0_cross_20260501_012205` | same driver, with `--bank-run` pointing at a fused bank dir we built (`labeling_supplement/harness_io_out/_fused_bank_2048_tetris`, both games' `skill_bank.jsonl` concatenated, 9 skills) | fused bank | actor's `retrieved_skill_ids` per step |
| `_phase0_cross_eligibility_probe.py` | standalone, bypasses actor RAG | fused bank | **all 9 skills** |

The third run is the one that exercises §22 — the dump driver as written feeds the actor's per-step RAG-retrieved IDs as candidates, and that RAG was built per-game so cross-game IDs are never present. Without bypassing the RAG, the dump driver cannot exhibit the §22 failure mode by itself.

## 2. Result

Eligibility narrowing on the 9-skill fused bank, with all 9 skills as candidates at every step:

| Actor episodes | Steps | Same-game skills eligible (per step) | **Cross-game skills eligible (per step)** |
|---|---|---|---|
| `twenty_forty_eight` | 150 | 2/3 always (`COMMIT/MERGE`, `mid:OPTIMIZE`) | **6/6 always — every tetris skill** |
| `tetris` | 127 | 6/6 always | **2/3 always — both 2048 ACTION skills** |

Per-skill admission count ([labeling_supplement/harness_io_out/_phase0_cross_eligibility_probe.json](_phase0_cross_eligibility_probe.json)):

```
[twenty_forty_eight]  150 steps
  [SAME ] COMMIT__MERGE                  150 / 150
  [SAME ] mid:OPTIMIZE                   150 / 150
  [CROSS] COMMIT__EVADE       (tetris)   150 / 150
  [CROSS] COMMIT__OPTIMIZE    (tetris)   150 / 150
  [CROSS] COMMIT__POSITION    (tetris)   150 / 150
  [CROSS] COMMIT__SETUP       (tetris)   150 / 150
  [CROSS] COMMIT__SURVIVE     (tetris)   150 / 150
  [CROSS] INSPECT__SETUP      (tetris)   150 / 150

[tetris]  127 steps
  [SAME ] COMMIT__EVADE                  127 / 127
  [SAME ] COMMIT__OPTIMIZE               127 / 127
  [SAME ] COMMIT__POSITION               127 / 127
  [SAME ] COMMIT__SETUP                  127 / 127
  [SAME ] COMMIT__SURVIVE                127 / 127
  [SAME ] INSPECT__SETUP                 127 / 127
  [CROSS] COMMIT__MERGE       (2048)     127 / 127
  [CROSS] mid:OPTIMIZE        (2048)     127 / 127
```

The only skill that is ever *filtered out* across both games is `COMPARE/MERGE` (2048), and only because it's tagged `evidence_role=REASON` → `SkillType.REASONING`, and the gymv adapter is only registered for `(gymv, ACTION) / (gymv, MIXED) / (gymv, GROUNDING)`. **There is zero task-axis filtering.** The audit's §22 prediction holds at 100 %.

## 3. What this means concretely

1. **§22 is the right next thing to fix.** Adding `feasible_tasks` to `SkillRecord` and a `target_task` parameter to `EligibilityFilter` is sufficient to drive the cross-game admission rate to 0. The lift in [`implementation_notes/legacy/protocol-lift-design.md`](../../implementation_notes/legacy/protocol-lift-design.md) §9 explicitly defers this — they're separate Day-1 / Day-2 work items.
2. **§21 (protocol prose → typed hops) is masked, not solved, by the dump driver's `_wrap_protocol_steps` workaround.** The workaround makes `iter_hops` yield N hops per skill, but every hop normalises to `"EXEC"` (no real verb). At eligibility time this doesn't matter — `EligibilityFilter.filter` doesn't read protocol semantics. At `harness.run_skill` time it would matter; we did not run `--run-skill` in Phase 0 because the result would be vacuous.
3. **The same-game eligibility numbers are honest baselines, but their *interpretation* differs by game.** Per-step inspection of the dump output shows:

```
[tetris ep0]   retrieved_skill_ids size hist: {5: 43}   ← actor RAG returns top-5 of 6
               eligible           size hist: {5: 43}   ← filter passes all 5
[2048   ep0]   retrieved_skill_ids size hist: {3: 50}   ← actor RAG returns all 3 in bank
               eligible           size hist: {2: 50}   ← filter drops 1 (REASONING type)
```

   So for **tetris** the eligibility filter is passing every retrieved skill — the "missing" skill never reached the filter at all (it's an actor-RAG top-K=5 setting, not a filter veto). For **2048** the filter genuinely drops `COMPARE/MERGE` because it's `evidence_role=REASON → SkillType.REASONING` and no `(gymv, REASONING)` adapter is registered. The 7 disagreement steps in 2048 (95.3 % agreement) are all the actor selecting `COMPARE/MERGE` against the filter's veto — a real filter-vs-actor disagreement that survives §22.

4. **Day-2's task-axis fix is expected to preserve the same-game numbers and drive cross-game numbers to 0.** Same-game stays at `(2, 6)`; cross-game drops from `(6, 2)` to `(0, 0)` unless `target_task ∈ verified_tasks` (which is empty at Day 2 by construction). The 7 2048 vetoes are independent of §22 — they're a `SkillType.REASONING`-vs-no-adapter issue that an adapter registration fix or a `(gymv, REASONING)` stub would address; not in scope for the lift or task axis.

## 4. What this measurement *did not* answer

The probe measures **admission** (does the filter let cross-game skills through?), not **agreement-impact** (would a cross-game admission steal the actor's selected skill?). The dump driver can't measure (b) because the actor's selection is fixed in the per-step record — it was made against the actor's original (single-game) RAG retrieval, not against the fused bank. To answer (b) honestly, Day 2 needs either:

- **(b.i)** re-run the cold-start actor against the fused bank (expensive — full inference over the 6 episodes again with cross-game RAG), or
- **(b.ii)** compute "if eligibility were the only filter, what does the actor's top-K look like?" — i.e., re-rank the actor's per-step retrieved candidates against the eligibility-filtered set and check whether the top-1 matches the actor's recorded `selected_skill_id`. Cheaper.

Day-2 should commit to (b.ii) with a small extension to `_phase0_cross_eligibility_probe.py`. (b.i) is a Day-5 follow-up.

## 5. What this measurement *might* under-count when extended

When the probe is extended past 2048+tetris (to all 4 env_wrappers games), `safe_skill_id` collisions surface — `INSPECT/SETUP` exists in both `tetris` and `super_mario` and after `/`→`__` they collide. The hardened probe (post-Day-1 review) detects this and emits a warning, dropping the second occurrence. For 2048+tetris specifically there is no collision and the Day-1 numbers are unaffected.

## 6. Reproducibility

```bash
cd Multi-hop-Reasoning-VLM-Agent

# Same-bank smoke (per-source, per-source bank).
bash labeling_supplement/run_dump_harness_io.sh \
    --surface online \
    --sources twenty_forty_eight tetris \
    --max-episodes 3 --max-steps 50 \
    --output-dir labeling_supplement/harness_io_out/run_phase0_<ts> \
    --parallel 2

# Fused-bank smoke (each source sees both games' skills, but candidates
# still come from actor RAG — so cross-game IDs never reach the filter).
BANK_SRC=labeling/skill_bank_out/run_20260430_030637
FUSED=labeling_supplement/harness_io_out/_fused_bank_2048_tetris
mkdir -p "$FUSED"/env_wrappers/{twenty_forty_eight,tetris}
cat "$BANK_SRC"/env_wrappers/{twenty_forty_eight,tetris}/skill_bank.jsonl \
    > "$FUSED"/env_wrappers/twenty_forty_eight/skill_bank.jsonl
cp  "$FUSED"/env_wrappers/twenty_forty_eight/skill_bank.jsonl \
    "$FUSED"/env_wrappers/tetris/skill_bank.jsonl
bash labeling_supplement/run_dump_harness_io.sh \
    --bank-run "$FUSED" --surface online \
    --sources twenty_forty_eight tetris \
    --max-episodes 3 --max-steps 50 \
    --output-dir labeling_supplement/harness_io_out/run_phase0_cross_<ts> \
    --parallel 2

# Real cross-eligibility probe — bypasses actor RAG, all 9 skills are
# candidates at every step. This is the §22 measurement.
python labeling_supplement/_phase0_cross_eligibility_probe.py
```
