# Single MDP vs. two-MDP — latency, games-only, and cross-task transfer

> **Status:** decided. Single-MDP / Harness-driven design (shipped patch #13)
> stays. Do **not** reintroduce the outer/inner-hop framing.
> **Last reviewed:** 2026-04-29.
> **Cross-refs:** [`decision_agents/README.md`](../../decision_agents/README.md)
> ("How the actor agent works", "Honest assessment", "Migration status"),
> [`decision_agents/actor_agent.py`](../../decision_agents/actor_agent.py)
> (`ActorAgent.step`, `_pick_action`, `_infer_intention`).

This memo records the design discussion behind keeping the unified
single-MDP / per-task-`Harness` actor instead of returning to the
deleted two-MDP framing (outer `env.step` MDP + inner `HopPolicy` MDP).

Three questions were considered, in order:

1. Does running two MDPs make inference meaningfully slower?
2. Is two-MDP worth doing on the **game** legs we actually train on?
3. Does **cross-task skill transfer** (game-trained skills → web / OS /
   VR / video) flip the answer?

The answer to all three is the same — **single MDP wins** — but for
distinct reasons, recorded here so the trade-off doesn't have to be
re-litigated.

---

## 1. Latency cost of a two-MDP design

### Wiring summary

| Design | LLM calls per outer step | Notes |
|---|---|---|
| **Two-MDP (deleted)** — outer `env.step` + inner `HopPolicy` | `1 + N`, with `N` ≤ `max_hops_per_step` (typ. 4–8) | Inner hops fired meaningfully on VR / video; on game / web / OS the inner loop short-circuited via `EXECUTE` after the first hop. |
| **Single MDP (shipped)** | 2 at the upper bound (`_infer_intention` + `_pick_action`); 1 if `_infer_intention` is folded or skipped | `harness.step` is **not** an LLM call — VR's `LOOK / CROP / READ_TEXT / COUNT / SEGMENT` go to detector / OCR / segmenter backends (Mocks today, Grounding-DINO / SAM-2 / PaddleOCR in Phase 8.1). |

### Wall-clock comparison (Qwen3-VL-8B / vLLM / A100, ~5–8 s/decode)

| Setting | LLM calls / outer step | Per-step latency | Per-episode (200-step Tetris) |
|---|---|---|---|
| Two-MDP (`N=4`) on VR / video | 5 | ~25–40 s | n/a (game) |
| Two-MDP (`N=8`) on VR / video | 9 | ~45–70 s | n/a (game) |
| Two-MDP on game / web / OS (inner short-circuited) | 1–2 | ~6–14 s | ~25 min |
| **Single MDP, all task families** | 1–2 | ~6–14 s | ~25 min |

### Cost model

```
T_step ≈ T_decode × (1 + N_inner)          [two-MDP]
T_step ≈ T_decode × (1 or 2)               [single MDP]
```

Where `N_inner` = how many inner hops actually fire. The deleted
framing was only "really" two-MDP on VR / video; for the legs we
optimise (games + GRPO budget) it never bought anything.

GRPO budget impact at `N=4`, ~hundreds of episodes/epoch: **~5× the
rollout cost** for what was, on game data, a no-op inner loop. The
cost compounds: 5× wall-clock means either 5× hardware budget or 5×
fewer rollouts (= worse advantage estimates = worse training).

### Caveats — when "two MDPs" doesn't actually slow you down

The label "two MDPs" is overloaded. Latency depends on **how many
decoder calls per outer step**, not on framing:

1. **Two LoRAs on one backbone (`skill_selection` + `action_taking`)** —
   what we ship today. `skill_selection` only fires on `should_reselect`,
   ~1/15 steps. Mean ≈ `1.07` LLM calls / step. vLLM multi-LoRA
   hot-swap keeps both adapters resident; switch is ~0 ms. **Not slow.**
2. **Two separate small models running in parallel** (e.g. a 0.6B
   intention model alongside the 8B action model) — free in wall-clock
   with overlap. Not done today, cheap if needed.
3. **A second sequential LLM call per step** (e.g. verifier / planner) —
   adds ~5–8 s/step on the same backbone. Only worth it if it lets you
   cut the action LoRA's `max_tokens` proportionally.
4. **A nested LLM call inside `harness.step`** — re-creates the deleted
   hop-policy cost structure: **`N×` slower**. *This is the trap to
   avoid.* The Phase-8 perception ops (`LOOK / READ_TEXT / SEGMENT /
   COUNT / CROP`) deliberately use **non-LLM** backends so harness ops
   never multiply decode cost. Resist routing reasoning through
   `harness.step`; route it through `_pick_action` so the GRPO action
   LoRA bears the cost and gets the credit.

---

## 2. Is two-MDP worth doing on games?

**No.** The information-acquisition argument that justified the inner
loop on VR / video does not apply to games.

### Why games don't need an inner loop

The hop loop was designed for tasks where each `LOOK / READ_TEXT /
RETRIEVE` *acquires evidence the outer step didn't have*. Games don't
have that bottleneck:

- **2048 / Tetris / Candy Crush** — `info["schema"]` already contains
  the whole board, the next pieces, score, legal moves. Nothing for
  `LOOK` to "discover."
- **Super Mario (Orak)** — frame already encodes mario pos, enemies,
  items; the schema extracts them. `READ_TEXT` would fire on the HUD,
  which is already parsed.
- **Avalon / Diplomacy** — there is hidden info (other agents' roles /
  orders), but that is **adversarial uncertainty**, not visual
  grounding. No tool call recovers it. What helps is opponent
  modelling, which the inner-hop vocabulary does not contain.

So on games, an inner MDP collapses to *more CoT, billed as MDP steps*
— exactly what the README's "Honest assessment" calls out.

### What actually moves the needle on games (already shipped)

In rough order of impact:

1. **Skill bank + `skill_selection` LoRA.** Tetris benefits hugely
   from "T-spin setup", "burn", "flat-stack". 2048 from "corner-anchor",
   "ladder". This is COS-PLAY's per-game gain.
2. **Better game-specific facts in the schema** —
   `decision_agents.agent_helper.extract_game_facts` is the leverage
   point for Tetris (`stack_h / holes / bumpiness / next-3`) and
   2048 (`highest / empty / merges`).
3. **Bigger `max_tokens` on `_pick_action`.** Cheap reasoning depth:
   `1×` decode for `~3×` reasoning vs. `N×` decode for nested calls.
   Same effective CoT, no MDP machinery.
4. **GRPO `r_env` densification.** Per-step env reward from line
   clears / merges / score deltas is dense enough that hop-level
   credit is unnecessary.
5. **Anti-repetition** — handles the failure mode an inner MDP would
   actually catch (the agent stuck in a 2-action loop).

### Apparent two-MDP cases that are really single-MDP

- **Diplomacy negotiation phases** — better expressed as a phase-keyed
  prompt template inside `_pick_action` than a nested MDP. Same
  compute, no nesting.
- **Avalon vote/quest reasoning** — `run_qwen3_avalon_episode` already
  runs all 5 players as Qwen3 in parallel via `ThreadPoolExecutor`.
  The actor's `infer_intention → action` chain is already two LLM
  calls; that's enough.

### If we ever do need lookahead on a game

The right move is **MCTS / beam search at inference with a forward
model** (cheap for 2048 / Tetris where the env is simulatable, harder
for Mario / Avalon). That gives genuine multi-step lookahead instead
of LLM-call multiplication. Not on the build path; recorded so the
fork point is documented.

---

## 3. What about cross-task skill transfer?

This is the angle that makes the inner-MDP design *most* tempting —
"the agent doesn't yet know how a game-skill applies to a new task,
so it needs to probe." The case **strengthens** the single-MDP
position rather than weakening it. Three reasons.

### What "skill" actually is in the transfer pipeline

`build_skill_bank_gymv.py` exports skills as
`(strategic_description, protocol_steps, eff_add / eff_del,
preconditions, success / abort criteria)`, embedded by
`Qwen3-Embedding-0.6B` for retrieval. So a transferred skill is **a
vector + a piece of structured guidance pasted into the next prompt**.
It is *not* a callable sub-policy. The `action_taking` LoRA decides
each primitive in the new task; the skill is RAG context.

That single fact kills the inner-MDP argument. An inner MDP buys
"more decoder calls to deliberate." Transfer doesn't need more
deliberation; it needs **a better retrieval embedding** and **a
uniform action vocabulary across tasks**. Both are already in the
single-MDP design.

### What does and doesn't transfer (concretely)

| Game skill (your bank) | Transfer target | Mechanism that actually transfers it | Inner MDP needed? |
|---|---|---|---|
| Tetris `maintain_flat_stack` | OS file management ("consolidate similar files") | Embedding similarity on `strategic_description`; action LoRA generalises | **No** |
| 2048 `corner_anchor` | Web ("park required fields first") | Same | **No** |
| Avalon `deduce_role_from_voting` | VR ("deduce identity from partial views") | Maps to flat `LOOK + COUNT + COMPARE + NOTE + VERIFY` actions in `VRHarness` | **No** — flat harness ops |
| Mario `predict_enemy_trajectory` | Video ("predict event sequence") | Maps to flat `TRACK + TIMELINE + WINDOW` ops in `VideoHarness` | **No** — flat ops |
| Diplomacy `negotiate_then_betray` | Web (multi-step booking dialogs) | Per-phase prompt routing inside `_pick_action` | **No** — one LLM call, branched prompt |
| Tetris `t_spin_setup` | (anything else) | Doesn't transfer (game-mechanics-specific) | **No** — irrelevant |

The cases that **look** like they need a second MDP (Avalon→VR,
Mario→Video) are exactly the ones the unified-harness design absorbs
as flat actions in `valid_actions(state)`. The action LoRA picks from
`LOOK / COUNT / COMPARE / TRACK / WINDOW / ...` directly; no nesting.

### Why two MDPs would actively hurt cross-task transfer

1. **Skill-bank embeddings stop matching.** Today the bank embeds
   `(state_summary, intention, strategic_description)` and retrieves
   over `(target_state_summary, target_intention)`. With a second MDP
   you also have to embed inner-MDP states — mid-CoT scratchpads
   without a stable shape. Game-trained outer-skills won't retrieve
   cleanly when the agent is in an inner-MDP step on a new task.
   You'd need a **second skill bank** for inner skills, doubling the
   labelling cost (`build_skill_bank_gymv.py` would need a sibling
   `build_inner_skill_bank.py`) and halving per-bank sample size.
2. **The dropped `hop_select` LoRA returns.** Patch #13 explicitly
   drops `hop_select` because hop and action decisions roll into one
   `action_taking` LoRA — which is what makes cross-task transfer
   *cheap*. Two MDPs reintroduces a separate hop-policy decoder
   trained per task family (or a wide shared one, which is harder).
   Either way the SFT data on games (where the inner loop never
   fires) doesn't help train it. We lose the
   "GPT-5.4 → Qwen3.5-9B SFT cold-start works for both LoRAs from
   the same trace" property — the entire point of how the cold-start
   collectors and `build_skill_bank_gymv.py` are wired today.
3. **`harness.action_kind` bucketing breaks.** Today every per-task
   action maps to one of `vr_look / vr_retrieve / vr_note / vr_answer
   / video_*` for uniform `r_cost`. Cross-task transfer rides on
   this: the same `RewardConfig` knobs apply on game and on web. Two
   MDPs adds an inner-action-kind dimension that has to be
   recalibrated per task family.

### What does enable game→other-task transfer (already shipped)

- **`SUBGOAL_TAGS` is task-agnostic.** Tetris `[CLEAR]` and a
  webform `[CLEAR]` share an embedding neighbourhood. The real
  cross-task hook for the skill bank.
- **`extract_game_facts` → schema → embedding** runs on every task.
  Schema-parser quality is the bottleneck for transfer, not MDP shape.
- **One `action_taking` LoRA across 5 task families.** The cross-task
  sharing mechanism. Don't fork it.
- **`EpisodicMemoryStore` is task-blind.** A Mario "got hit by enemy
  after standing still" memory can retrieve into a web episode where
  the agent is stalling. Free transfer, single-MDP.

### Escalation order if transfer underperforms

If empirical transfer is bad, fix it in this order **before** even
considering an inner MDP:

1. **Improve `strategic_description` quality** in the skill bank
   (better SEGMENT teacher prompts in `build_skill_bank_gymv.py`).
2. **Add an abstraction layer to skills** — store both a
   task-specific protocol and a task-agnostic "pattern" tag (e.g.
   `gather_evidence_then_decide`, `consolidate_then_clear`); embed on
   the pattern. **One extra field on `Skill`, not a second MDP.**
3. **Two-call inference** (predict-pattern → act) — two LLM calls in
   *one* MDP step, not two MDPs.

Exhaust all three before considering a real inner MDP. Even then,
expect MCTS-with-a-forward-model to beat it on games and tool-augmented
harness ops to beat it on VR / video.

### Suggested transfer stress test

Before any architectural change, verify whether transfer is actually
the bottleneck:

1. Freeze the per-game banks produced by `build_skill_bank_gymv.py`.
2. Run `ActorAgent` on a small VR / web slice with
   `SkillBankProvider` pointing at the **game banks**.
3. Measure (a) skill-retrieval quality (does anything fire? are the
   firings sensible?) and (b) action-LoRA accuracy on tasks where
   retrieval fires.

If (a) is bad → fix descriptions / pattern tags, not MDP shape.
That's almost certainly the bottleneck.

---

## TL;DR

- **Latency:** two-MDP costs `1 + N` LLM calls per outer step; single
  MDP costs 1–2. On VR / video at `N=8` that's ~5–9× slower; on game /
  web / OS the inner loop never fired meaningfully so the cost was
  ~0–10%. Either way, no upside on games.
- **Games:** the inner-MDP value prop (acquire evidence the outer
  step lacks) does not apply. Schema already contains the board.
  Reach for `max_tokens` and skill bank quality before reaching for
  a second MDP. If you ever need real lookahead, do MCTS with a
  forward model, not nested LLM calls.
- **Cross-task transfer:** transfer rides on embedding similarity +
  uniform action vocabulary, not on nested deliberation. Two MDPs
  would force a second skill bank, reintroduce the dropped
  `hop_select` LoRA, and break the single-corpus SFT cold-start that
  makes the pipeline cheap. **Single MDP makes transfer cheaper, not
  more expensive.**

The trap to avoid in code: **never put an LLM call inside
`harness.step`.** That re-creates the deleted hop-policy cost
structure and undoes the unification. Route reasoning through
`_pick_action` so the GRPO action LoRA bears the cost and gets the
credit.
