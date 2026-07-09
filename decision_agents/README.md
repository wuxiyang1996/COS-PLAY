# Decision Agent

The Decision Agent module from the **COS-PLAY** co-evolution framework (COLM 2026). Implements the three-stage decision loop described in Section 4.1 of the paper: **skill retrieval** → **intention update** → **action execution**, with composite reward shaping (r_total = r_env + λ_f · r_follow + r_cost).

Three actor flavours and one legacy agent ship here:

| Agent | Backbone | Input | Use when |
|-------|----------|-------|----------|
| `ActorAgent` (base, schema-native) | text-only via `API_func.ask_model` (defaults to `Qwen/Qwen3.5-9B`) | Parsed `<state>…</state>` schema from `vlm_wrapper` | You want the unmodified loop from [`plans/02-action-agent/PLAN-ACTION-AGENT.md`](../plans/02-action-agent/PLAN-ACTION-AGENT.md) §2.3 — no images, no recorder. Kept stable so the existing test suite and offline rollouts keep working. |
| `GPT4oCollectorActor` ([`SFT/`](SFT/)) | **`gpt-5.5`** (the SFT teacher; class name retained for back-compat) with multimodal chat completions | Schema + screenshot | You're **gathering SFT data** to fine-tune the Qwen3.5-9B student. Writes per-step rows in the exact layout `trainer/SFT/data_loader.py` consumes. |
| `QwenVLActor` ([`grpo/`](grpo/)) | **`Qwen/Qwen3.5-9B`** (LoRA-trained actor) via `trainer.coevolution.vllm_client.AsyncVLLMClient` (multi-LoRA hot-swap) | Schema + screenshot | You're running **online inference or GRPO+LoRA training**. Emits `trainer.common.metrics.RolloutStep` records the GRPO trainer ingests directly. |
| `VLMDecisionAgent` (legacy, text-native) | text-only | Raw observation text | You don't yet have VLM grounding wired in. Kept for backward compatibility with `scripts/qwen3_decision_agent.py` and `inference/run_qwen3_8b_eval.py`. |

The two new flavours **subclass `ActorAgent`** and override exactly one
seam (`_call_llm`) plus a bit of bookkeeping. They reuse the entire
schema-parse → intention → reselect → harness-driven action selection →
entity-resolve → anti-repetition pipeline unchanged. There is **no
fork** of the per-step contract; only the LLM backend and the per-step
artefact differ. The MDP itself is **single-level** — one COS-PLAY-style
loop, the per-task `Harness` decides what an action *is* and what
`step(action)` *does* (game env, browser, OS, scratchpad, frame cursor).

Why split this way: the SFT collector and the GRPO actor have
fundamentally different deployment shapes (sync OpenAI vs async vLLM,
filesystem JSONL vs in-memory `RolloutRecord`), and the
`vlm_wrapper/README.md` distillation plan keeps them strictly
separate (`gpt-5.5` SFT teacher → `Qwen/Qwen3.5-9B` student). Folding
them under one class would force every caller to depend on both stacks.

Sub-package map:

```
decision_agents/
├─ actor_agent.py        ← unified single-MDP ActorAgent (Harness-driven)
├─ agent.py              ← VLMDecisionAgent / LLMDecisionAgent (text-only)
├─ schema_parser.py      ← shared StateSchema / Entity / parse_state_schema
├─ skill_interface.py    ← SkillProvider seam
├─ skill_tracker.py      ← SkillTracker (slot coverage, reselect-on-stall)
├─ reward_func.py        ← RewardComputer (r_env + r_follow + r_cost,
│                          incl. per-action-kind costs for VR/Video)
├─ agent_helper.py       ← infer_intention, EpisodicMemoryStore
├─ core/                 ← shared scaffolding for the unified MDP
│   ├─ multimodal.py        VisualInput, build_*_messages, load_image_as_data_url
│   ├─ harness.py           Harness Protocol, HarnessState, ACTION_KIND_* tags
│   ├─ harness_gym.py       GymHarness (game / Gymnasium-shaped envs)
│   ├─ harness_browser.py   BrowserHarness (Playwright / BrowserGym; step stub)
│   ├─ harness_osworld.py   OSWorldHarness (desktop primitives + bash; step stub)
│   ├─ harness_vr.py        VRHarness (visual reasoning; LOOK/RETRIEVE/NOTE/ANSWER)
│   └─ harness_video.py     VideoHarness (frame cursor + VR ops + NEXT_FRAME/JUMP/...)
├─ SFT/                  ← gpt-5.5 (SFT teacher) data-collection actor (see SFT/README.md)
│   ├─ actor_gpt4o.py        ← class kept under historical name; `model="gpt-5.5"` by default
│   ├─ sft_recorder.py
│   └─ run_collect.py
└─ grpo/                 ← Qwen/Qwen3.5-9B + GRPO + LoRA (see grpo/README.md)
    ├─ actor_qwen_vl.py
    └─ rollout_logger.py
```

> **Removed:** `inner_mdp.py` (HopAction / HopPolicy / HeuristicHopPolicy /
> HopStep / HopTrace / parse_hop_action) was deleted in the unified-harness
> migration — its operators relocated into `VRHarness` / `VideoHarness` action
> vocabularies. Importing the old names emits a `DeprecationWarning` via
> `decision_agents.__getattr__` for one release.

---

## How the actor agent works

`ActorAgent` is the **single** schema-native decision class shared
across the GPT-4o SFT collector ([`SFT/`](SFT/)) and the Qwen3-VL-8B
GRPO actor ([`grpo/`](grpo/)), and shared across **all five task
families** the project targets: game agents, web agents, OS agents,
visual reasoning, and video understanding. It implements **one** MDP
— the COS-PLAY skill-augmented decision loop, generalised so the
"environment" each task plugs in is just another **harness**.

### Single MDP, five tasks, one harness contract

The actor's MDP signature is task-independent:

```
state_t   = (observation_t, intention_t, active_skill_t, scratchpad_t,
             valid_actions_t)
action_t  ~  π(state_t)                          # ActorAgent._pick_action
(obs_{t+1}, r_t, done_t, info_{t+1}) = harness.step(action_t)
r_total_t = r_env_t  +  w · r_follow_t  +  r_cost_t
```

Per-step pipeline (uniform across all 5 tasks — exactly the COS-PLAY
recipe `select_skill → update_intention → take_action`):

```
┌──────────── ActorAgent.step (one MDP step, every task) ─────────────┐
│ 1. parse_observation       → schema_t           (vlm_wrapper)       │
│ 2. compact_summary                                                  │
│ 3. infer_intention         → "[TAG] subgoal"   (SUBGOAL_TAGS)       │
│ 4. should_reselect?        → skill_provider.select(state, harness)  │
│ 5. _pick_action(harness.valid_actions(state))   ← LLM (_call_llm)   │
│ 6. resolve_entity_action + anti-repetition                          │
│ 7. harness.step(action_t)  → (obs, reward, done, info)              │
│ 8. observe_result          → r_env+r_follow+r_cost,                 │
│                              memory.add, tracker.update             │
└─────────────────────────────────────────────────────────────────────┘
```

What changes between tasks is **only the harness** — which decides
what an `observation` looks like, what `valid_actions` are, and what
`harness.step(action)` does to the world (or to the scratchpad).

### The five task harnesses

| Task | Harness | Observation | `valid_actions(state)` | `harness.step` mutates | Terminal reward |
|------|---------|-------------|-----------------------|-----------------------|-----------------|
| **Game agent** | `LMGameHarness` / `OrakHarness` / `AgentEvolverHarness` | game frame + structured state | env primitives (`up`, `down`, `key:A`, `click(e5)`, …) | game state | env score |
| **Web agent** | `BrowserHarness` (Playwright + BrowserGym) | DOM + viewport screenshot | `click(bid)`, `type(bid, str)`, `scroll(dy)`, `goto(url)`, `key(...)`, `ANSWER(str)` | live page | task success |
| **OS agent** | `OSWorldHarness` | desktop screenshot + a11y tree | `xdotool click`, `key(...)`, `bash(cmd)`, `read(path)`, `ANSWER(str)` | OS state | task success |
| **Visual reasoning** | `VRHarness(image, question)` | image + scratchpad | `LOOK(region)`, `CROP(bbox)`, `COUNT(class, region)`, `COMPARE(a,b,attr)`, `READ_TEXT(bbox)`, `RETRIEVE(query)`, `ANSWER(str)` | scratchpad only (image is read-only) | answer correctness |
| **Video understanding** | `VideoHarness(clip, question)` | current frame(s) + scratchpad + cursor `t` | `NEXT_FRAME`, `JUMP(t)`, `WINDOW(t1,t2)`, `FOCUS(bbox,t)`, `TRACK(eid)`, `RETRIEVE(query)`, `ANSWER(str)` | scratchpad + frame cursor (clip is read-only) | answer correctness |

Game / web / OS are **mutable-world** harnesses (50–500 steps,
dense + terminal reward). VR / video are **read-only-world**
harnesses (5–30 steps, sparse + terminal reward) where the harness
actions are reasoning operators that grow the scratchpad until the
agent emits `ANSWER`. To `ActorAgent`, both look identical: pick one
action from `valid_actions_t`, get back `(obs, reward, done)`, repeat.

### Dual-axis intention vocabulary (current canonical)

The runtime agent and the offline labeler share a single **dual-axis**
labeling scheme.  Each step is annotated along **two orthogonal axes**
exported from `decision_agents.agent_helper`:

| Axis | Vocabulary | Tokens | Purpose |
|------|------------|--------|---------|
| `operator` | `INTENT_OPERATORS` | `INSPECT / TRACK / COMPARE / COMMIT / VERIFY / RECOVER` (6) | Cognitive mode — *what is the agent doing with attention this step?*  Domain-agnostic transfer signal. Future-aligned with the two-MDP inner-hop alphabet `{GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE}` from `plans/02-action-agent/PLAN-ACTION-AGENT.md §5.3`. |
| `subgoal`  | `UNIFIED_SUBGOALS` | `SETUP / NAVIGATE / POSITION / CLEAR / MERGE / COLLECT / BUILD / ATTACK / DEFEND / EVADE / OPTIMIZE / SURVIVE / EXPLORE / EXECUTE` (14) | Game achievement — *what concrete goal is being pursued this step?*  Domain anchor.  One alphabet for both `gym_v` (action games) and `env_wrappers` (puzzle / strategy). |
| Action set (per-task) | enumerated by `harness.valid_actions(state)` | (variable) | Candidate set passed to `_call_llm` and matched against by the multi-strategy action parser. |

The two axes are written into `Experience.intentions` as
`"[OPERATOR/SUBGOAL] note"` (e.g. `"[COMMIT/EVADE] sidestep left to
avoid bullets"`).  The downstream `parse_intention_tag` returns the
operator (cross-domain transfer signal); the new `parse_intention_tags`
returns both axes.  Legacy single-tag intentions
(`[CLEAR] match three reds`, `[COMMIT] advance toward orb`) are still
accepted — the missing axis is reconstructed via the
`SUBGOAL_TO_OPERATOR` / `OPERATOR_TO_SUBGOAL` alias maps in
`agent_helper.py`, so older banks load unchanged.

For backward compatibility, the legacy 13-tag `SUBGOAL_TAGS` is kept as
an alias of `UNIFIED_SUBGOALS` minus `EVADE` (which is split out from
`DEFEND` to give dodge-style movement a discriminative anchor for
shoot-em-ups and fighting games).

| Outer step (used today)            | ↔   | Inner hop (planned, two-MDP)             |
|------------------------------------|-----|------------------------------------------|
| `INSPECT` (parse / RAG)             | ~   | `GROUND`, `RETRIEVE`                      |
| `VERIFY`  (check result)            | ~   | `CHECK`                                   |
| `COMMIT`  (act on goal)             | ~   | `COMMIT`, `EXECUTE`                       |
| `COMPARE` (weigh options)           | —   | (outer-step only)                         |
| `TRACK`   (follow change)           | —   | (outer-step only)                         |
| `RECOVER` (defensive react)         | —   | (outer-step only)                         |

**VERIFY discipline.** VERIFY is reserved for steps with **no new
directional intent** (NOOP / idle / repeating the same input only to
confirm registration).  The "X had no effect, try Y" pattern is COMMIT
(a *new* directional decision), with `EXPLORE` as the typical subgoal.
Without this discipline the labeller collapses to ~85 % VERIFY on retro
action games — see
[`labeling/readme.md`](../labeling/readme.md#vocabularies--dual-axis-one-alphabet-for-both-corpora)
for the full diagnosis and the smoke-test calibration that fixes it.

### What happens on a single MDP step

1. **Parse** `harness.parse_observation(obs)` → schema (entities, relations, targets, uncertainty).
2. **Compactify** to a `summary` line (key=value, capped length).
3. **`infer_intention(summary, task)`** → `"[TAG] subgoal phrase"`, tag constrained to `SUBGOAL_TAGS`.
4. **Skill tracking** — `SkillTracker.should_reselect(schema)` decides whether to keep the active skill or query the `SkillProvider` (skill bank with the current harness in scope). Reselects are charged `query_skill_cost`.
5. **Action selection** (`_pick_action(valid_actions)`) — strict priority:
   1. The active skill's current protocol step, if it matches a valid action.
   2. **`_call_llm(prompt, images=…)`** — text-only by default; vision-aware in the SFT/GRPO subclasses. The reply is decoded by a multi-strategy parser (`exact → numbered → entity_ref → edit_distance → token_overlap → loose → trailing-digit`) and the chosen branch is logged as `parse_path`.
   3. Deterministic fallback (first valid action, else `"no-op"`).
6. **Resolve entity references** (`click(e5) → click(bid_42)`) via the schema.
7. **Anti-repetition** — if the last *N* actions are identical and all rewards ≤ 0, swap to a different valid action.
8. **`harness.step(action_t)`** → `(obs, reward, done, info)`. For mutable-world harnesses this advances the env; for VR/video it updates the scratchpad (and possibly the frame cursor).
9. Return an `ActorDecision` (action, resolved action, intention, summary, parse path, `queried_skill`, `queried_mem`, …).

After the harness step the runner calls `observe_result(...)`, which:

- updates `last_actions` / `last_rewards` / `progress_notes`,
- forwards the step to `SkillTracker.record_step` and `RewardComputer.compute_reward` (`r_env + w_follow · r_follow + r_cost` shaping; per-action-type costs come from `RewardConfig`),
- writes `(state_summary, action, next_state_summary, done)` into `EpisodicMemoryStore` so future `RETRIEVE` actions in any task can hit this transition,
- finalizes the active skill on episode end with `success` / `timeout` so the `SkillProvider` can update its statistics.

This is the **same** loop COS-PLAY runs — there is no second MDP, no
"inner hop" phase. What COS-PLAY did for one game environment, this
project does for any harness that implements the seven-method
contract: `reset`, `step`, `valid_actions`, `parse_observation`,
`is_done`, `compute_reward`, `summarize_state`.

### What's trained vs hand-coded

| Component | State today | Trained how |
|-----------|-------------|-------------|
| Schema parser, skill tracker, scratchpad, anti-repetition, reward computer, harness contract | Hand-coded | n/a (deterministic) |
| `infer_intention` (subgoal labelling) | LLM call constrained to `SUBGOAL_TAGS` | rides on top of the action LoRA's intention-conditioned training data |
| `SkillProvider` (skill retrieval) | `SkillBankProvider` over the live skill bank | `skill_selection` LoRA (SFT cold-start → GRPO); the bank itself is co-evolved by the Skill Bank agent (`segment` / `contract` / `curator` LoRAs) |
| Action selection (`_pick_action` LLM branch) | GPT-4o (SFT collector) or Qwen3-VL-8B (GRPO student) | `action_taking` LoRA — SFT cold-starts → GRPO refines on harness reward (env score for game/web/OS; correctness for VR/video) |
| Per-task harness (game / web / OS / VR / video) | Hand-coded action set + step rules | n/a (the harness *defines* the MDP; it isn't trained) |
| Episodic memory | Embedder-backed `EpisodicMemoryStore` | Frozen embedder (Qwen3-Embedding-0.6B); store is filled at runtime |

So inside the actor agent **GRPO trains exactly two LoRA adapters** —
`skill_selection` and `action_taking` — both cold-started by SFT on
GPT-4o teacher data. The same two LoRAs serve all five task families;
only the harness (and therefore the `valid_actions`) changes.

### Migration status (shipped)

The unified-MDP design above is **shipped** as of 2026-04. The
two-MDP framing (outer `env.step` + inner `HopPolicy`) has been
collapsed into a single COS-PLAY-style loop whose action vocabulary
and `step(action)` semantics are owned by an injected `Harness`.
What landed:

| Module | What changed |
|--------|--------------|
| `actor_agent.py` `step` | `_run_inner_mdp` deleted; `_pick_action` chooses directly from `harness.valid_actions(state)`. Legacy `hop_policy=` / `max_hops_per_step=` kwargs are still accepted (with a `DeprecationWarning`) so existing constructors compile unchanged. |
| `inner_mdp.py` | **Deleted.** Operators relocated: `GROUND` → `VRHarness.step(LOOK)`, `RETRIEVE` → `VRHarness.step(RETRIEVE)`, `CONCLUDE` → `VRHarness.step(NOTE)`, `EXECUTE` → folded into the env-action loop. `HopAction` / `HopPolicy` / etc. raise `AttributeError` with a `DeprecationWarning` via the package shim. |
| `core/` | + `Harness` protocol + `HarnessState` + 5 reference impls (`GymHarness`, `BrowserHarness`, `OSWorldHarness`, `VRHarness`, `VideoHarness`) + `parse_op_call` helper. |
| `info["valid_actions"]` source | `harness.valid_actions(state)` is the single source of truth (legacy `info["valid_actions"]` / `schema.actions` still feed `GymHarness.valid_actions` so the back-compat path is byte-identical). |
| `reward_func.RewardConfig` | + `vr_look_cost / vr_retrieve_cost / vr_note_cost / vr_answer_cost` and `video_next_frame_cost / video_jump_cost / video_focus_cost / video_track_cost` (all default `0.0`, looked up via `harness.action_kind(action)`). |
| GRPO LoRAs | Two only (`skill_selection`, `action_taking`). The earmarked `hop_select` LoRA is dropped — hop decisions are now action decisions and roll into `action_taking`. |

#### Migration of inner-MDP operators

| Old hop op (`inner_mdp.HopAction`) | New harness action | Side effect |
|---|---|---|
| `GROUND(slot)`     | `VRHarness.step(LOOK(region))`     | `scratchpad.grounded_slots[slot] = "observed"`; `tracker.clear_ground_flag` |
| `RETRIEVE(query)`  | `VRHarness.step(RETRIEVE(query))`  | `memory.query(query)` → top-3 hits into `scratchpad.memory_hits` |
| `CONCLUDE(text)`   | `VRHarness.step(NOTE(text))`       | `scratchpad.notes.append(text)` |
| `EXECUTE(action)`  | (folded into env action loop)      | n/a — env action goes straight through `_pick_action` |
| `CHECK / VERIFY`   | `VRHarness.step(COUNT(arg))` / `VRHarness.step(COMPARE(arg))` | recorded as a diagnostic note on the scratchpad |
| n/a                | `VRHarness.step(ANSWER(text))`     | `done=True`; `r_env = +1` iff `text == gold_answer` |

`VideoHarness` adds `NEXT_FRAME / JUMP(t) / WINDOW(t1,t2) / FOCUS(bbox,t) / TRACK(eid)` on top of the same scratchpad ops.

### TL;DR

One MDP. One per-step pipeline (`select_skill → update_intention →
take_action`, COS-PLAY style). Five task families, each plugged in
through a `Harness` that decides what an action is and what
`step(action)` does. One universal vocabulary (`SUBGOAL_TAGS`). Two
GRPO LoRAs (`skill_selection`, `action_taking`). One LLM seam
(`_call_llm`) that swaps GPT-4o (SFT collection) for Qwen3-VL-8B
(GRPO inference / training). Everything else — schema parsing, skill
tracking, reward shaping, anti-repetition, episodic memory, entity
resolution — is deterministic scaffolding that the LoRA never has to
relearn.

---

## Honest assessment — does this actually do multi-hop reasoning?

The MDP shape and the seams are right. The hop *bodies* are not. Today
`VRHarness.step(LOOK(...))` performs a dict write
(`scratchpad.grounded_slots[slot] = "observed"`); it does not call a
detector or a segmenter and it does not return new pixels into the
next prompt. The same is true for `CROP / READ_TEXT / COUNT / COMPARE`.
Concretely:

| Op (today) | What `step` does | New visual evidence in `obs_{t+1}`? |
|---|---|---|
| `LOOK(slot)` / `CROP(arg)` / `READ_TEXT(arg)` | `scratchpad.grounded_slots[slot] = "observed"` | **No** |
| `COUNT(arg)` / `COMPARE(arg)` | `scratchpad.notes.append(f"{op}({arg})")` (echoes the action back) | **No** |
| `RETRIEVE(query)` | `EpisodicMemoryStore.query(query, k=3)` | Only if the store was pre-populated; for a fresh VR episode it's empty |
| `NOTE(text)` | `scratchpad.notes.append(text)` | **No** — records the LLM's own text |
| `ANSWER(text)` | exact-match against `gold_answer` for `r=+1` | n/a (terminal) |
| `NEXT_FRAME / JUMP(t)` (video) | advances an integer cursor | **No** — the next prompt still sees the clip the caller passed in |

So at 8 hops × 1 LLM call per hop, you're paying 8× inference for one
chain-of-thought pass — that's CoT in MDP clothing, not multi-hop
reasoning in the sense the QA literature uses it. We document this
plainly so we don't fool ourselves with the framing.

**What's right** (so the next mile is short): one hop = one MDP step
(credit can attach per hop), the scratchpad is the only mutable state
(perfect anchor for tool-output caching), `harness.step` is the single
chokepoint (swapping the body for real tool calls doesn't touch
`ActorAgent`, the LoRA, or the prompt builder), `action_kind → cost`
plumbing already exists (real tools cost real reward), and
`bind_actor` already wires tool outputs onto the actor's scratchpad.

**Four concrete pieces still missing** to make hops carry real
evidence:

1. **`harness.step` must call real vision tools and feed bytes back.**
   `LOOK("red cup")` should run an open-vocab detector
   (Grounding-DINO 1.5 Edge), crop, and append the cropped image to
   the next prompt's `images=[…]`; `READ_TEXT(bbox)` should run OCR;
   `COUNT(class)` should run a detector and return an integer;
   `COMPARE(a,b,attr)` should compute on cached attributes. The
   `vlm_wrapper/` tool-calling scaffold is ready — it isn't wired in.
2. **The schema must grow per hop, not be frozen at episode start.**
   After `LOOK`, the new entity (with `bbox` + attrs from the tool)
   should be appended so step `t+1`'s prompt can reference it by `e_42`.
   Today the schema is parsed once; the LLM has to redo grounding
   from words on every call.
3. **Hop-credit reward shaping.** Today only `ANSWER` produces
   `r_env`; costs are flat negatives. GRPO will mostly learn "answer
   fast". Partial credit on intermediate hops (`+0.2` when `LOOK`'s
   bbox IoU with the gold-region > 0.5; `+0.2` when `RETRIEVE`
   returns the gold passage in top-3; `+0.1` for a connected
   evidence chain) is needed for the LoRA to actually learn the hop
   policy. VisualToolBench / TIR-Bench / ScienceQA-IMG / NExT-QA all ship the
   region/event-level annotations to supervise this.
4. **Video-specific: a frame backbone.** Pre-computed per-frame
   embeddings (CLIP / SigLIP / InternVideo) cached in the harness so
   `JUMP(t)` actually swaps the rendered image, `WINDOW(t1,t2)`
   returns sub-sampled keyframes, and `TRACK(eid)` runs SAM-Track /
   DEVA between frames. Without this, video understanding degrades
   to "show all frames, run CoT, answer".

Items #1–#2 are entirely inside `core/harness_vr.py` and
`core/harness_video.py` — no actor / LoRA / GRPO churn. #3 is one new
field on `RewardComputer.compute_reward`. #4 is a frame-cache class.

### Proposed extended tool catalog (Phase 8)

To make multi-hop genuinely earn its name, we group tools into three
roles. Each row is a candidate harness op; each carries the backing
model, side-effect on the scratchpad / schema, the action kind for
cost lookup, and where it applies (VR / Video / both). "Status"
shows whether the op exists today (✅), is a stub (🟡), or is a
proposed addition (⬜).

#### A. Perception — turn pixels into structured evidence

These ops produce **new bytes** (crops, masks, OCR strings, depth,
pose) and append a typed `Entity` to the schema. They are the
foundation: every other tool consumes their output.

| Op | Backing model | Returns | Schema side effect | `action_kind` | Status |
|----|---------------|---------|--------------------|---------------|--------|
| `LOOK(text)` / `REFER(text)` | Grounding-DINO 1.5 Edge (open weights, Apache-2.0) — early text↔image fusion handles referring expressions; OWLv2 fallback for low-resource | `{bbox, conf}` | adds `Entity(label=text, pos=bbox, extra={"source_op":"LOOK","conf":…})` | `vr_look` | ✅ plumbing (Mock backend; real model lazy-loads in Phase 8.1) |
| `CROP(bbox)` | pillow / torchvision | `image_crop` | appends crop to `info["images"]`; mints `Entity(label=…, pos=bbox, extra={"source_op":"CROP"})` | `vr_look` | ✅ plumbing |
| `SEGMENT(target)` | SAM-2 / SEEM | pixel mask + bbox | `Entity.attributes["area_px"] / ["seg_score"]`; bbox refined to mask extent | `vr_look` | ✅ plumbing (Mock backend; real model lazy-loads in Phase 8.1) |
| `READ_TEXT(bbox)` | PaddleOCR / TrOCR | text + per-char conf | `Entity.value=text`; `Entity.attributes["text"] / ["ocr_score"]` | `vr_look` | ✅ plumbing (Mock backend; real model lazy-loads in Phase 8.1) |
| `DEPTH(bbox\|point)` | Depth-Anything-V2 | scalar depth | `Entity.attrs["depth_m"] = …` | `vr_look` | ⬜ |
| `POSE(person_id)` | ViTPose / HMR2 | 17 kpts (2D) / SMPL (3D) | `Entity.attrs["pose_kpts"] = …` | `vr_look` | ⬜ |
| `GAZE(person_id)` | Gaze360 / MCGaze | 3D gaze vector | `Entity.attrs["gaze_dir"] = …` | `vr_look` | ⬜ |
| `ATTRIBUTES(bbox)` | CLIP / SigLIP zero-shot | `{color, material, texture, …}` dict | merges into `Entity.attrs` | `vr_look` | ⬜ |
| `CLASSIFY(bbox, taxonomy)` | CLIP / fine-grained head | top-k labels + probs | `Entity.attrs["class_topk"] = …` | `vr_look` | ⬜ |
| `COUNT(class, region)` | Grounding-DINO 1.5 Edge + class-agnostic NMS | integer | `Entity(label="count(query)", attributes={"count":N,"query":…})`; also `info["count"]` | `vr_look` | ✅ plumbing (Mock backend; real model lazy-loads in Phase 8.1) |
| `READ_CHART(bbox)` | DePlot / ChartQA-VLM | structured table (CSV-ish) | `Entity(kind="chart", table=…)` | `vr_look` | ⬜ |
| `MEASURE(bbox_a, bbox_b)` | pure geometry (cached bboxes) | `{px_dist, ratio_h, ratio_w, IoU}` | `Entity(kind="measurement", …)` | `vr_look` | ⬜ |
| `OPTICAL_FLOW(t1, t2)` | RAFT / SEA-RAFT | dense flow tensor (downsampled) | `Entity(kind="flow", t1, t2, magnitude)` | `video_focus` | ⬜ video |

Design rules: every perception op writes a typed entity into the
schema with a stable `eid`, so subsequent ops (and the LLM) can refer
to it by name; raw tensors are kept in the harness's per-episode
`_evidence_cache` keyed by `eid` and only the lightweight summary
(bbox, conf, attrs) goes into the prompt.

#### B. Evidence finding — locate, retrieve, compare cached evidence

These ops never touch raw pixels themselves; they query what the
perception ops already cached, retrieve external knowledge, or check
similarity / coreference.

| Op | Backend | Returns | Side effect | `action_kind` | Status |
|----|---------|---------|-------------|---------------|--------|
| `RETRIEVE(query)` | `EpisodicMemoryStore` (cosine + keyword) | top-k memory rows | `scratchpad.memory_hits ←` | `vr_retrieve` | ✅ |
| `MATCH(eid_a, eid_b)` | SigLIP cos-sim on cached crops | similarity ∈ [0,1] | note + `Entity.attrs["match_with"]` | `vr_retrieve` | ⬜ |
| `KB_LOOKUP(entity, attr)` | Wikidata SPARQL / Wikipedia RAG | structured fact | `Entity.attrs[attr] = value, source=…` | `vr_retrieve` | ⬜ |
| `SEARCH_WEB(query)` | Tavily / SerpAPI | top-k snippets | `scratchpad.memory_hits ←` (with `source=web`) | `vr_retrieve` | ⬜ |
| `LOCATE(scene_text)` | CLIP-text → region heatmap argmax | bbox or `None` | `Entity(kind="region")` | `vr_look` | ⬜ |
| `COREF(question_phrase)` | sentence-transformer over schema entities | `eid` or `None` | `scratchpad.coref[phrase] = eid` | `vr_retrieve` | ⬜ |
| `COMPARE(eid_a, eid_b, attr)` | symbolic over cached attrs | `>`, `<`, `==`, `unknown` | `Entity(kind="comparison", outcome=…)` | `vr_look` | 🟡 stub |
| `RECALL_FRAME(t)` | scratchpad temporal index | rerenders frame `t` | re-attaches frame to `obs_{t+1}.images` | `video_focus` | ⬜ video |
| `FIND_EVENT(predicate)` | Moment-DETR / UniVTG | `(t_start, t_end)` | `Entity(kind="event", span=…)` | `video_jump` | ⬜ video |

Design rules: every retrieval op writes a *traceable* evidence row
(`{source, query, hit, conf}`) so reward shaping can credit the right
hop and post-hoc analysis can show provenance for each ANSWER.

#### C. Reasoning-inspire — compose, verify, gate the answer

These ops manipulate evidence rather than acquire it. They are what
turns a pile of detected entities into a defensible answer chain.

| Op | What it does | Side effect | `action_kind` | Status |
|----|--------------|-------------|---------------|--------|
| `NOTE(text)` | append free-form claim | `scratchpad.notes ←` | `vr_note` | ✅ |
| `PLAN(subgoal)` | decompose remaining hops into a checklist | `scratchpad.plan = [step, …]`; `tracker` consumes for skill alignment | `vr_note` | ⬜ |
| `HYPOTHESIZE(prop)` | post a candidate answer for testing | `scratchpad.hypotheses.append({prop, status="open"})` | `vr_note` | ⬜ |
| `VERIFY(prop)` | check `prop` against scratchpad evidence (NLI / symbolic) | flips hypothesis status to `confirmed / refuted / undetermined` | `vr_note` | ⬜ |
| `CONTRAST(eid_a, eid_b)` | enumerate diffs from cached attrs | `scratchpad.notes ←` "diff: color, size" | `vr_note` | ⬜ |
| `EXPLAIN()` | generate justification chain from notes + grounded slots | `scratchpad.justification ←` | `vr_note` | ⬜ |
| `COMPUTE(expr)` | sandboxed Python `eval` for arithmetic / IoU / area | result note; `Entity(kind="computed", value=…)` | `vr_note` | ⬜ |
| `BACKTRACK(k)` | undo last `k` notes / hypotheses (when `VERIFY` rejects) | rolls back scratchpad slice | `vr_note` | ⬜ |
| `CONFIDENCE_GATE(τ)` | refuse `ANSWER` until mean(grounded conf) ≥ τ | sets `scratchpad.gate_open = bool` | `vr_note` | ⬜ |
| `CAUSAL(event_a, event_b)` | infer "before/after/causes" between two grounded events | note with relation tag | `vr_note` | ⬜ video |
| `TIMELINE()` | sort grounded events by frame index | `scratchpad.timeline = [(t, eid), …]` | `vr_note` | ⬜ video |
| `SUMMARIZE_WINDOW(t1,t2)` | one-line caption per window from cached frames | `Entity(kind="window_summary", span, text)` | `video_focus` | ⬜ video |
| `EVENT_SEGMENT()` | divide clip into events (Mask2Former-VOS / TAL) | `Entity(kind="event", span)` × N | `video_jump` | ⬜ video |
| `TRACK(eid)` | SAM-Track / DEVA across frames | `Entity.attrs["trajectory"] = [(t, bbox), …]` | `video_track` | 🟡 stub |
| `BEFORE / AFTER(event_eid)` | grab frame immediately before/after a grounded event | re-attaches frame to `obs_{t+1}.images` | `video_focus` | ⬜ video |

Design rules: reasoning ops never produce new pixels — they
**combine** existing evidence. `VERIFY` and `CONFIDENCE_GATE` are
what turn the agent from "answer when token budget runs out" into
"answer when the evidence supports it".

#### Credit-assignment hooks per role

Each role aligns naturally with one reward signal:

| Role | Reward shaping signal | Where to wire it |
|------|----------------------|------------------|
| **Perception** | IoU(bbox, gold_region) > τ → `+δ_perceive` | `RewardComputer.compute_reward(perception_iou=…)` |
| **Evidence** | `gold_passage ∈ retrieved_topk` → `+δ_retrieve` | `RewardComputer.compute_reward(retrieval_hit=bool)` |
| **Reasoning** | `VERIFY` confirms before `ANSWER`, OR `EXPLAIN` chain covers all gold predicates → `+δ_reason` | `RewardComputer.compute_reward(verified=bool, chain_coverage=float)` |
| **Terminal** | exact / fuzzy match on `ANSWER(text)` → `r_env=+1` | unchanged |

The four signals are independent so GRPO can credit each hop
separately. Without per-role signals, the LoRA only ever sees the
terminal reward and falls back to "skip the hops, answer fast".

#### What stays universal vs what specialises

Even with this expanded catalog, **only the harness changes per task
family**. `ActorAgent`, the schema parser, the skill tracker, the
prompt builder, the action parser, the SFT recorder, the GRPO
rollout logger, the two LoRAs — all stay byte-identical. The
`Harness` protocol absorbs the entire perception/evidence/reasoning
expansion via:

- `harness.valid_actions(state)` — enumerates the per-task subset of
  the catalog above (e.g. `BrowserHarness` won't expose `DEPTH`).
- `harness.step(action)` — owns the tool calls + cache + schema
  augmentation.
- `harness.action_kind(action)` — maps every new op to one of the
  existing `vr_look / vr_retrieve / vr_note / vr_answer / video_*`
  cost buckets so `RewardConfig` doesn't need an entry per tool.

So Phase 8 is bounded: ~600 LOC of harness extensions + a frame-cache
class + 3 new optional fields on `RewardComputer`. Everything else
already exists.

### Design choice — VR / Video are the weak leg by construction

This project optimises for **best-unified-agent across five task
families**, not for best-on-any-one-benchmark. The Harness contract
fits perception–action loops with mutable worlds (game / web / OS)
naturally; it fits narrative inference (VR / video) only loosely.
Rather than fork the architecture per task family — which would buy
points on one benchmark at the cost of the unification thesis — we
accept VR / video as the **weakest of the five legs by design**.

Concretely:

- VR / Video keep first-class harness support and the full action
  vocabularies; they prove the unified contract works for read-only
  worlds.
- They get the **cross-cutting** perception ops
  (`LOOK / READ_TEXT / SEGMENT / CROP`) when those land — those four
  also serve OS / web / game grounding so they earn their cost on
  every leg, not just two.
- They **do not** get the Video-Holmes-tuned additions (audio
  modality, person Re-ID, action recognition, camera-language ops,
  abduction / induction / counterfactual). Those would let a
  specialist climb the Video-Holmes leaderboard but would not
  generalise to the other four legs, so they're out of scope here.
- We measure VR / Video against a small smoke-test slice (~200
  questions per benchmark — TIR-Bench / NExT-QA / Video-Holmes), not the
  full leaderboards. The bar is **"within 5pts of the same backbone
  in fat-context mode"** — i.e. prove the harness overhead doesn't
  actively hurt — not "top the board".

The full perception / evidence / reasoning catalog above stays in
this README as a documented frontier for whoever wants to build a
video-specialist fork. It is explicitly **not** on this project's
build path. Phase 8 ships the four cross-cutting perception ops and
stops there.

#### Primary scorecards (where we actually claim numbers)

| Leg | Primary benchmark | Bar |
|-----|-------------------|-----|
| Game | LMGame-Bench (2048 / Tetris / Candy Crush), AgentEvolver (Avalon / Diplomacy), Orak (Mario) | Match Qwen3-8B baseline at lower inference cost |
| Web | WebArena / VisualWebArena / BrowserGym MiniWoB++ | Match GPT-4o-Web on a multi-step subset |
| OS | OSWorld (subset) | Beat Qwen2.5-VL baseline; match GPT-4o-OS on common workflows |
| VR (smoke) | TIR-Bench / VisualToolBench (~500 q) | Within 5pts of Qwen3-VL-8B fat-context on the same questions |
| Video (smoke) | NExT-QA / Video-Holmes (~200 q each) | Within 5pts of fat-context — *not* leaderboard-chasing |

---

## ActorAgent — schema-native decision agent

`ActorAgent` implements §1 of `PLAN-ACTION-AGENT.md` end-to-end:

```
<state> schema (from vlm_wrapper)
    ↓
parse_state_schema → StateSchema          # decision_agents/schema_parser.py
    ↓
compact_summary + infer_intention         # replaces raw-text compression
    ↓
SkillTracker.should_reselect              # decision_agents/skill_tracker.py
    ↓ (if reselect)
SkillProvider.select(query, schema, ...)  # decision_agents/skill_interface.py
    ↓
SkillTracker.activate → slot coverage     # PLAN §10
    ↓
action prompt → LLM                       # schema + skill +
    ↓                                       harness.valid_actions(state)
resolve_entity_action(click(e5))          # PLAN §7 Phase 3
    ↓
anti-repetition guard
    ↓
harness.step(action)                      # GymHarness / VRHarness / ...
    ↓                                       (driven by the runner)
SkillTracker.record_step + RewardComputer  # incl. harness.action_kind cost
    ↓
SkillProvider.record_outcome (on skill end)
```

### Dependency injection

Every piece with a `…Provider` / `…Policy` name is injectable so later phases of the plan can swap implementations without changing `ActorAgent`:

| Interface | Default | Swap-in for |
|-----------|---------|-------------|
| `SkillProvider` | `NullSkillProvider` (skill-free) or `SkillBankProvider(bank)` (RAG) | Trained Skill-Use Agent (Agent 2) |
| `Harness` (see "How the actor agent works" above) | `GymHarness` / `BrowserHarness` / `OSWorldHarness` / `VRHarness` / `VideoHarness` (all shipped under `core/`; the browser + OSWorld `step()` raises `NotImplementedError` until the env layer is plugged in, but their `valid_actions` are real) | Per-task action sets and `step` semantics across all 5 task families |
| `RewardComputer` | `reward_func.RewardComputer` (now reads `Harness.action_kind(action)` so VR/video per-action costs flow into `r_cost`) | Extended reward decomposition (PLAN §6) |

### Skill interface contract

`SkillProvider` is the seam between the actor and anything that knows about skills. Three methods:

| Method | Purpose |
|--------|---------|
| `select(query, state_summary, structured_state, current_predicates, top_k) -> list[SkillGuidance]` | Return candidate skills for the current state. |
| `record_outcome(skill_id, outcome, reward, steps_taken, info)` | Called after every skill attempt terminates (`success` / `abort` / `stall` / `switch` / `timeout`). |
| `available_skills() -> list[str]` | Enumerate what the provider can return. |

`SkillGuidance` bundles everything the actor renders into the prompt: name, strategy, protocol steps, preconditions, success/abort criteria, required/optional slots (→ drive the GROUND-insertion rule), `eff_add` / `eff_del` effects (→ feed `r_follow`), and a fallback `micro_plan`.

```python
from decision_agents import ActorAgent, SkillBankProvider, run_actor_episode
from skill_agents.skill_bank.bank import SkillBankMVP
from skill_agents.query import SkillQueryEngine

bank = SkillBankMVP("path/to/bank.jsonl"); bank.load()
engine = SkillQueryEngine(bank=bank)

agent = ActorAgent(
    model="Qwen/Qwen3.5-9B",                    # = BACKBONE_MODEL
    skill_provider=SkillBankProvider(engine),   # or NullSkillProvider() for baseline
)

episode = run_actor_episode(env, agent=agent, task="Clear the board", max_steps=200)
```

### Where schemas come from

The runner expects the env (or a wrapper around it) to place the `<state>` text on `info["schema"]` (or `info["schema_text"]`). Override via `schema_from_info=…` when integrating a different wrapper. When the key is missing the actor falls back to the raw-text path (Phase 1), so you can drop `ActorAgent` into existing envs before finishing the VLM wiring.

### Files

| File / sub-package | What it does |
|--------------------|--------------|
| `actor_agent.py` | `ActorAgent`, `ActorDecision`, `ActorState`, `run_actor_episode`. Owns the per-step pipeline; exposes a single `_call_llm(prompt, images=None, ...)` seam that the SFT and GRPO actors override. |
| `schema_parser.py` | `StateSchema`, `Entity`, `Targets`, `StateFlags`, `Relation`, `Hop`, `Answer`, `ResolvedAction`, `parse_state_schema`, `resolve_entity_action` |
| `skill_interface.py` | `SkillProvider` protocol, `SkillGuidance`, `NullSkillProvider`, `SkillBankProvider` |
| `skill_tracker.py` | `SkillTracker`, `ActivationCheck`, `TrackerState` — lifecycle + slot-coverage (PLAN §10) |
| [`core/`](core/) | `Harness` Protocol + `HarnessState` + 5 reference impls (`GymHarness`, `BrowserHarness`, `OSWorldHarness`, `VRHarness`, `VideoHarness`); `parse_op_call`; `VisualInput`, `build_openai_vision_messages`, `build_qwen_vl_messages`, `load_image_as_data_url` — multimodal scaffolding shared by the SFT/GRPO actors. |
| [`SFT/`](SFT/) | `GPT4oCollectorActor`, `SFTRecorder`, `SFTRecord`, `run_collect` CLI — see [`SFT/README.md`](SFT/README.md). |
| [`grpo/`](grpo/) | `QwenVLActor`, `GRPORolloutLogger`, `DEFAULT_QWEN_VL_MODEL` — see [`grpo/README.md`](grpo/README.md). |

---

## Two specialised flavours (SFT collection ⇄ GRPO inference/training)

The two sub-packages [`SFT/`](SFT/) and [`grpo/`](grpo/) implement the
distillation pipeline laid out in [`vlm_wrapper/README.md`](../vlm_wrapper/README.md):
**GPT-4o teacher → SFT cold-start → Qwen3-VL-8B-Instruct student → GRPO+LoRA**.
Both subclass [`ActorAgent`](actor_agent.py); they reuse the entire
schema-parse → intention → reselect → harness-driven action selection →
entity-resolve → anti-repetition pipeline unchanged. Only the LLM
backend and the per-step artefact differ.

### Pipeline at a glance

```
   ┌─────────────────────────────────────────────────────────────┐
   │ Stage 1 — Data collection (offline, GPT-4o teacher)         │
   │                                                             │
   │   harness  ──► GPT4oCollectorActor.step(image, schema)      │
   │     │           │    ▲                                      │
   │     │           │    └─ _call_llm = OpenAI chat completions │
   │     │           │        (multimodal: [text, image_url, …]) │
   │     │           ▼                                           │
   │     │      SFTRecorder.record_action_taking(...)            │
   │     │           │                                           │
   │     ▼           ▼                                           │
   │  harness.step(action)                                       │
   │  (GymHarness for game, BrowserHarness for web,              │
   │   VRHarness for visual reasoning, …)                        │
   │                                                             │
   │   <out>/<game>/{skill_selection,action_taking}.jsonl        │
   │   exact format trainer/SFT/data_loader.py reads             │
   └────────────────────────────┬────────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Stage 2 — Cold-start SFT (trainer/SFT/train.py)             │
   │   trains LoRA adapters: skill_selection, action_taking      │
   │   on top of Qwen/Qwen3-VL-8B-Instruct                       │
   │   output: runs/sft_coldstart/decision/<adapter>/            │
   └────────────────────────────┬────────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Stage 3 — Online GRPO (trainer/coevolution/grpo_training.py)│
   │                                                             │
   │   harness  ──► QwenVLActor.step(image, schema)              │
   │     │           │    ▲                                      │
   │     │           │    └─ _call_llm = AsyncVLLMClient.        │
   │     │           │         generate_chat                     │
   │     │           │         adapter="action_taking"           │
   │     │           ▼                                           │
   │     │      GRPORolloutLogger.log_step(decision, rr)         │
   │     ▼           │                                           │
   │  harness.step(action) ──► (obs, reward, done, info)         │
   │                                                             │
   │   RolloutStep[r_env, r_follow, r_cost, r_total,             │
   │               action_kind ← harness.action_kind(action),    │
   │               parse_path, queried_skill, queried_mem, …]    │
   │   → RolloutRecord (trainer.common.metrics)                  │
   │   → DecisionGRPOTrainer                                     │
   └─────────────────────────────────────────────────────────────┘
```

The same `<state>` schema, the same `harness.valid_actions(state)`,
and the same prompt shape flow through both flavours. That guarantees
the SFT records the GPT-4o teacher writes are 1:1 alignable with the
rollouts the Qwen3-VL student produces — which is the whole point of
the distillation: **what the student saw at GRPO time and what the
teacher saw at labeling time must come from the same actor pipeline,
binding the same Harness**.

### Flavour comparison

| Aspect | `GPT4oCollectorActor` ([`SFT/`](SFT/)) | `QwenVLActor` ([`grpo/`](grpo/)) |
|--------|---------------------------------------|----------------------------------|
| Backbone | GPT-4o (OpenAI / OpenRouter) | `Qwen/Qwen3-VL-8B-Instruct` (vLLM) |
| LLM call | sync `openai.chat.completions.create(...)` | async `AsyncVLLMClient.generate_chat(...)` |
| LoRA adapters | n/a (frozen teacher) | multi-LoRA hot-swap; `adapter="action_taking"` |
| Per-step artefact | JSONL row → `<out>/<game>/<adapter>.jsonl` | `RolloutStep` → `RolloutRecord` |
| Consumed by | `trainer.SFT.data_loader.load_decision_adapter_data` | `trainer.coevolution.grpo_training.DecisionGRPOTrainer` |
| Vision input | optional but recommended | optional but recommended |
| Inference cost | API-billed; teacher only | self-hosted; rollout-rate friendly (≈5–8 s/step on A100) |
| Where it lives at runtime | data-collection notebooks / batch scripts | online co-evolution loop, eval drivers |
| When to use | bootstrapping the SFT corpus, gold-label cross-validation | every online rollout once SFT cold-start has converged |

### `GPT4oCollectorActor` quick start

```python
from decision_agents import GPT4oCollectorActor, SFTRecorder, VisualInput

recorder = SFTRecorder()                     # default path → trainer/SFT
actor = GPT4oCollectorActor(
    recorder=recorder,
    game="tetris",
    model="gpt-4o",
)

obs, info = env.reset()
done = False
while not done:
    decision = actor.step(
        observation=str(obs),
        schema_text=info.get("schema_text"),
        valid_actions=info.get("valid_actions"),
        task="Clear lines as fast as possible.",
        images=[VisualInput(image_path=info["screenshot"])],   # optional
    )
    obs, reward, term, trunc, info = env.step(decision.action)
    done = bool(term or trunc)
    actor.observe_result(decision, reward=reward, done=done)
recorder.write_manifest()                    # _manifest.json with row counts
```

CLI entrypoint:

```bash
python -m decision_agents.SFT.run_collect \
    --env-factory my_envs.tetris:make_env \
    --game tetris --episodes 50 --max-steps 200 \
    --image-info-key screenshot --schema-info-key schema_text
```

The output JSONL row matches `trainer/SFT/data_loader.py` field by
field (`prompt`, `completion`, `intention`, `active_skill`) plus an
extra `image` block silently passed through, so the existing
cold-start trainer ingests these artefacts without any conversion. See
[`SFT/README.md`](SFT/README.md) for the full row schema and the
GPT-4o vision routing logic.

### `QwenVLActor` quick start

```python
from decision_agents import (
    QwenVLActor, GRPORolloutLogger, DEFAULT_QWEN_VL_MODEL, VisualInput,
)
from trainer.coevolution.vllm_client import AsyncVLLMClient

vllm = AsyncVLLMClient(
    base_url="http://localhost:8000/v1",
    model=DEFAULT_QWEN_VL_MODEL,             # "Qwen/Qwen3-VL-8B-Instruct"
)
logger = GRPORolloutLogger(env_name="tetris", game_name="tetris")

actor = QwenVLActor(
    vllm_client=vllm,
    rollout_logger=logger,
    adapter="action_taking",                 # LoRA from runs/sft_coldstart/
)

obs, info = env.reset()
logger.start_episode(seed=42)
done = False
while not done:
    decision = actor.step(
        observation=str(obs),
        schema_text=info.get("schema_text"),
        valid_actions=info.get("valid_actions"),
        images=[VisualInput(image_path=info["screenshot"])],
    )
    obs, reward, term, trunc, info = env.step(decision.action)
    done = bool(term or trunc)
    actor.observe_result(decision, reward=reward, done=done)
record = logger.finalize_episode(score=info.get("score", 0.0), won=info.get("won", False))
# record is a trainer.common.metrics.RolloutRecord ready for DecisionGRPOTrainer.
```

For runners already on an event loop (e.g. the async co-evolution
collector), use `await actor.step_async(...)` instead — the LLM call
stays non-blocking. See [`grpo/README.md`](grpo/README.md) for the
LoRA adapter routing and the full `RolloutStep` field mapping.

### Multimodal scaffolding ([`core/`](core/))

Both flavours go through the same `VisualInput` and message builders,
so the screenshot the teacher sees and the screenshot the student sees
are byte-identical (after data-URL normalisation):

```python
from decision_agents import VisualInput, build_qwen_vl_messages

img = VisualInput(image_path="rollouts/.../step_0007.png",
                  caption="browser viewport @ 1280x720")

messages = build_qwen_vl_messages(
    prompt="<full action prompt with schema + valid actions>",
    images=[img],
    system="You are an Actor Agent ...",
)
# → [{"role": "system", ...},
#    {"role": "user", "content": [
#         {"type": "text", "text": "<prompt>"},
#         {"type": "text", "text": "browser viewport @ 1280x720"},
#         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
#    ]}]
```

`build_openai_vision_messages` is a thin alias kept separate so future
GPT-4o-only tweaks don't bleed into the Qwen path (and vice versa).

### Lazy imports

`from decision_agents import GPT4oCollectorActor` (or `QwenVLActor`)
goes through a PEP 562 `__getattr__` shim, so plain
`import decision_agents` does **not** pull in `openai` or
`trainer.coevolution.vllm_client`. Existing offline tests keep
running with neither dependency installed.

---

## ActorAgent — open work (post-harness migration)

The unified-MDP / `Harness` migration (patch #13 below) closed most
of the original gap list against `plans/02-action-agent/PLAN-ACTION-AGENT.md`.
What remains is a short list of additive, non-breaking improvements.

**Status legend:** ✅ shipped · 🟡 partial · ⬜ pending.

### Still open

| # | Status | Gap | Plan ref | Fix |
|---|--------|-----|----------|-----|
| 1 | ⬜ | **`r_follow` uses text substring matching even when a schema is present.** | §4 r_follow | Thread an optional `schema` into `RewardComputer.compute_reward` and match `eff_add` against `state_flags` / `entities_by_ontology` / `relations`. |
| 2 | 🟡 | **Action parsing fallbacks aren't shared with `VLMDecisionAgent`.** | §1 step 6 | Lift `_extract_action_from_reply` into a shared `ActionParser` and add an optional RAG-embedding `ActionEmbeddingMatcher` (already used in `scripts/qwen3_decision_agent.py`). |
| 3 | ⬜ | **No `ContinueSwitchPolicy` seam for Agent 2 GRPO.** | §6 | Abstract `SkillTracker.should_reselect` into a pluggable policy so the trained Skill-Use Agent can replace the rule-based default. |
| 4 | ⬜ | **Pipeline-orchestrator log shapes are missing.** | PLAN-PIPELINE-ORCHESTRATOR §2.1/§2.2 | Accept a `TraceContext` dataclass in `ActorAgent.step`; thread `run_id` / `episode_id` / `step_id` / `span_id` / `schema_hash` through `decision.to_dict()`. |
| 5 | ⬜ | **Budget control is only a step cap.** | PLAN-PIPELINE-ORCHESTRATOR §7 | Accept an optional `BudgetCounter`; decrement in `_select_skill`, `_pick_action`, `harness.step`, `memory.query`. On exhaustion, degrade to the deterministic fallbacks already present. |
| 6 | ⬜ | **Entity-referenced actions aren't prompted for browser/OSWorld.** | §7 Phase 3 | When `harness.domain in {"browser", "osworld"}`, append an "Entity-referenced actions you may also emit" section to the prompt, enumerated from `schema.interactive_entities()`. |
| 7 | ⬜ | **Anti-repetition is deterministic** — plan §1 step 7 says "randomly pick". | §1 step 7 | Seed with `Random(hash(episode_id))` or rotate by `len(last_actions)` to avoid 2-action limit cycles. |

### Harness `step()` implementations (primary-scorecard blockers)

Three of the five harnesses currently raise `NotImplementedError` from
`step()` (or rely on a pre-wired Gym env). Until they actually drive
their respective environments end-to-end, the unification thesis only
covers two of the five legs in practice (game + VR). These items
unblock the **primary scorecards** (WebArena, OSWorld) and are the
real long pole for "best-unified-agent across five task families".

| # | Status | Item | Estimate | Notes |
|---|--------|------|----------|-------|
| H1 | ⬜ | **`BrowserHarness.step` over Playwright + BrowserGym.** Wire `click(bid)` / `type(bid, str)` / `scroll(dy)` / `goto(url)` / `key(...)` / `ANSWER(str)` to BrowserGym primitives. Surface DOM + viewport screenshot in observations. | 3-5 days | Unblocks WebArena / VisualWebArena / MiniWoB++ rollouts. Use `bid` from a11y tree first; perception fallback (H4) is a refinement. |
| H2 | ⬜ | **`OSWorldHarness.step` over OSWorld + xdotool.** Wire `xdotool click(x,y)` / `key(...)` / `bash(cmd)` / `read(path)` / `ANSWER(str)` to the OSWorld VM. Surface desktop screenshot + a11y tree. | 3-5 days | Unblocks OSWorld benchmark. Same shape as H1. |
| H3 | ⬜ | **`VideoHarness.JUMP / WINDOW / TRACK` actually rerender frames.** Today `NEXT_FRAME` advances an integer cursor but the next prompt still sees whatever the caller passed. Plug a frame-cache class so `JUMP(t)` swaps the rendered image and `WINDOW(t1, t2)` returns sub-sampled keyframes. | 2-3 days | Prerequisite for any video smoke-test that's better than fat-context CoT. |
| H4 | ⬜ | **Perception fallback for `BrowserHarness.valid_actions` / `OSWorldHarness.valid_actions`.** When a target has no `bid` (image-only button, captcha) or sparse a11y tree, run `LOOK("submit button")` + `READ_TEXT` and synthesize click targets from detector bboxes. | 1-2 days *after* H1 / H2 land and the perception package (Phase 8.0) is shipped | This is "perception earning its keep on a primary scorecard" — the actual cross-cutting payoff. |

Build order chosen for the current sprint:

1. **Phase 8.0 (in progress)** — perception package + `VRHarness.step` rewire. Lands the cross-cutting infrastructure (Protocols / `EvidenceCache` / Mocks / schema-delta merge) with VR as the only consumer for now. Honest scope: only advances the weak leg, but the package is reused by H4.
2. **H1 (next)** — `BrowserHarness.step` over Playwright + BrowserGym. First mutable-world leg beyond game; unblocks the web scorecard.
3. **H2** — `OSWorldHarness.step`. Second mutable-world leg.
4. **H4** — perception fallback wired into Browser / OSWorld `valid_actions`. Now perception ops earn their cost on three legs (VR + web + OS), validating the cross-cutting claim.
5. **Phase 8.1** — replace mock backends with real `GroundingDINODetector` / `SAM2Segmenter` / `PaddleOCREngine`, gated behind lazy imports.
6. **H3** — video frame-cache + rerender. Lower priority since video is explicitly the weak leg.

### Already shipped (recap)

The bullets below were closed during the harness migration and earlier
patch sets; the patch-set log at the bottom of this README has the full
diffs.

- ✅ Multi-strategy action parser with `parse_path` logging.
- ✅ Intention inference re-ordered to run **before** reselect / action.
- ✅ `r_cost` events for `QUERY_SKILL` / `QUERY_MEM` / `CALL_SKILL` / `SKILL_SWITCH`.
- ✅ Per-action-kind costs (VR/Video) via `RewardConfig.cost_for_action_kind` + `Harness.action_kind(action)`.
- ✅ `progress_notes` rendered into the action prompt.
- ✅ `ActorDecision.to_dict` re-exposes `reasoning` + `parse_path` + `queried_*`.
- ✅ Schema's own `goal` / `task` falls through `_build_action_prompt`.
- ✅ Slot-coverage seeding from `ActivationCheck.missing_slots`; `tracker.clear_ground_flag` called by `VRHarness.step(LOOK)`.
- ✅ Inner-MDP hops folded into harness actions (deleted `inner_mdp.py`, ~430 LOC).

### Patch-set log

Shipped so far (all additive — no existing-caller API breaks):

1. ✅ Extended `RewardComputer` with orthogonal `queried_skill` / `queried_mem` cost events.
2. ✅ Fixed `f"get_slot_bindings"` typo in `skill_interface.py`.
3. ✅ Added `InnerScratchpad` dataclass and plumbed it onto `ActorState`.
4. ✅ Extended `ActorDecision` with `queried_skill`, `queried_mem`, `parse_path`; re-exposed `reasoning` in `to_dict`.
5. ✅ Re-ordered `ActorAgent.step` — intention inference now runs before reselect.
6. ✅ Refactored `_run_inner_mdp` + new `_apply_hop_side_effect` to actually execute hops (GROUND → `tracker.clear_ground_flag`, RETRIEVE → `self.memory.query`, CONCLUDE → notes).
7. ✅ Multi-strategy action parser (exact → numbered → entity-ref → edit distance → token overlap → loose → trailing-digit), with a `parse_path` log tag.
8. ✅ `_build_action_prompt` now renders `Inner reasoning so far`, `Recent progress`, and falls back to `schema.goal` when the caller didn't pass a `task`.
9. ✅ `observe_result` + `run_actor_episode` thread the new flags into `Experience.extras` (`queried_skill`, `queried_mem`, `parse_path`, `scratchpad`, `reasoning`).
10. ✅ Removed the dead `_ = json` sentinel; logged the `_build_default_memory` fallback path.
11. ✅ Tests: 13 new cases covering reselect cost, scratchpad grounding, RETRIEVE-hop memory wiring, `to_dict` shape, and the parser pipeline.
12. ✅ **`HopPolicy` now consumes the outer-step intention.** The `SUBGOAL_TAGS` `[TAG]` from `infer_intention` is forwarded to `select_next_hop(... intention=...)` via signature introspection (backward-compatible with policies that pre-date the kwarg). `HeuristicHopPolicy` adds one intention-aware rule (`EXPLORE` on an empty trace → one `GROUND("scene")` before `EXECUTE`); the rest of the heuristic stays schema-driven. 4 new tests in `decision_agents/tests/test_intention_dispatch.py` lock in the contract. *(Now superseded by the unified-harness pivot — see #13.)*
13. ✅ **Unified single-MDP / per-task-harness pivot — shipped.** Collapsed the two-MDP framing (`outer env-step MDP + inner HopPolicy MDP`) into a single COS-PLAY-style MDP whose action set is defined by a per-task `Harness`. Five harnesses ship under `decision_agents/core/`: `GymHarness` (game), `BrowserHarness` (web; `step` stub), `OSWorldHarness` (OS; `step` stub), `VRHarness` (visual reasoning), `VideoHarness` (video understanding). `inner_mdp.py` (HopAction / HopPolicy / HopStep / HopTrace / HeuristicHopPolicy / parse_hop_action — ~430 LOC) was deleted; its operators relocated as first-class actions inside `VRHarness` / `VideoHarness` (`LOOK / RETRIEVE / NOTE / ANSWER` and `NEXT_FRAME / JUMP / WINDOW / FOCUS / TRACK`). `ActorAgent.step` now consumes `harness.valid_actions(state)` directly; `_run_inner_mdp` + `_apply_hop_side_effect` deleted. `RewardConfig` gained 8 optional per-action-kind cost fields (defaults `0.0`) looked up via `harness.action_kind(action)` so VR / video tasks can shape away over-deliberation without affecting game / web / OS `r_total`. The legacy `run_actor_episode(env, agent)` path keeps working byte-identical — it auto-binds a `GymHarness` over the env. Deprecated `hop_policy=` / `max_hops_per_step=` kwargs and the `decision_agents.HopAction` / `HopPolicy` / etc. names emit a one-shot `DeprecationWarning` for one release of grace. The `hop_select` LoRA is dropped — GRPO trains exactly two LoRAs (`skill_selection`, `action_taking`) for all 5 tasks. Tests: deleted `test_intention_dispatch.py` (4 cases) + 6 inner-MDP cases in `test_actor_agent.py`; added `test_harness.py` (32 cases across all 5 harnesses + `parse_op_call`), `test_actor_with_vr_harness.py` (5 end-to-end cases), and `test_actor_back_compat.py` (12 cases pinning the legacy entry point + the deprecation contract). 79 tests total, all passing.
14. ✅ **Phase 8.0 — perception plumbing + VR harness rewire.** Added `decision_agents/core/perception/` sub-package: `RegionDetector` / `Segmenter` / `OCREngine` Protocols (`runtime_checkable`), `Detection` / `Segmentation` / `OCRResult` frozen dataclasses, `MockRegionDetector` / `MockSegmenter` / `MockOCR` deterministic stand-ins (no GPU / no `transformers` dep), and a per-episode LRU `EvidenceCache` keyed by `(image_hash, op, args_blob)` with hit/miss stats. `VRHarness` constructor now accepts optional `detector / segmenter / ocr / cache`; `LOOK / CROP / READ_TEXT / COUNT / SEGMENT` actually call the backends, mint `Entity` rows, and surface them on `info["schema_delta"]`; `CROP` additionally appends the cropped region as a `VisualInput` on `info["images"]`. `ActorAgent.step` gained `_merge_schema_delta` (accepts `list[Entity]` *or* `list[dict]`) which folds harness-emitted entities into the current schema *before* `_pick_action` so the next prompt sees the fresh entity. New `SEGMENT(eid)` op surfaced in `valid_actions`. The harness keeps Phase-7 backward compatibility — when no backends are bound, all ops degrade gracefully into the original scratchpad-only behaviour. Real backends (Grounding-DINO 1.5 Edge, SAM-2, PaddleOCR) are deferred to Phase 8.1 and will load lazily so `import decision_agents` stays fast. New tests: `test_perception.py` (27 cases — Protocols + Mocks + cache hit/miss/eviction + bbox geometry), `test_vr_harness_with_perception.py` (27 cases — image-byte loading, schema_delta emission per op, cache reuse, ANSWER scoring, schema-delta merge semantics including dict coercion + affords dedup). Total: 133 tests, all passing.

Still open (see tables above): #5-sharedActionParser, #6, #7, #9, #10, VERIFY semantics, anti-repetition randomness, plus H1–H4 (Browser/OSWorld `step()` implementations, video frame-cache, perception fallback for sparse a11y trees), and Phase 8.1 (real Grounding-DINO / SAM-2 / PaddleOCR backends behind lazy imports). (Items #1-OptionB / #8 / `hop_select` LoRA are now obsolete — superseded by #13.)

---

## Legacy VLMDecisionAgent (text-native, Pipeline A / B)

**Two model backends:**

- **`gpt-5.5`** (training-free; previously called `gpt-5.4`/`gpt-4o`) — used
  for cold-start data generation and labeling via OpenRouter / OpenAI API.
  Tracks `BACKBONE_SFT_TEACHER_MODEL` from `common/models.py`.
- **`Qwen/Qwen3.5-9B`** (GRPO-trained with LoRA adapters) — served via vLLM
  for decision agent inference and evaluation.  Tracks `BACKBONE_MODEL`.
  The deferred Qwen3-8B track remains reachable through
  `inference/run_qwen3_8b_eval.{py,sh}` and `scripts/qwen3_*.py`.

Both share the same code path; `API_func.ask_model` routes to the correct API based on the model name. Skill bank loading and querying are identical for both backends.

## Supported games

**6 games** across three environment stacks (matching `cold_start/`):

| # | Stack | Game | Registry Key |
|---|-------|------|-------------|
| 1 | LMGame-Bench | **2048** | `twenty_forty_eight` |
| 2 | LMGame-Bench | **Candy Crush** | `candy_crush` |
| 3 | LMGame-Bench | **Tetris** | `tetris` |
| 4 | AgentEvolver | **Avalon** | `avalon` |
| 5 | AgentEvolver | **Diplomacy** | `diplomacy` |
| 6 | Orak | **Super Mario** | `super_mario` |

---

## Decision agent pipelines

Two script-level pipelines drive the decision agent at inference time. Both use `Qwen/Qwen3-8B` served via vLLM and share the same core helpers from `decision_agents/`, but differ in skill-bank integration depth and game coverage.

### Pipeline A — `scripts/qwen3_decision_agent.py` (with skill select)

Skill-bank-guided decision agent with protocol-aware lifecycle management.

**Per-step loop:**

1. **`get_state_summary()`** — deterministic + LLM state compression into `key=value` format (≤400 chars)
2. **`infer_intention()`** — Qwen3-8B produces a `[TAG] subgoal phrase` from summary + context (last actions, task)
3. **Skill re-selection check** (`_SkillTracker.should_reselect()`) — triggers re-query when: no active skill, duration exceeded, zero-reward stall (≥4 steps with reward ≤0), abort/success criteria keyword-matched in current state
4. **`get_skill_guidance()`** — queries `SkillQueryEngine` (RAG mode) using `game_name + intention + state_text[:1500]` as query, with `structured_state` converted to `{predicate: float}` for applicability scoring. Returns skill_id, skill_name, execution_hint, protocol (steps, preconditions, success/abort criteria)
   - If re-selecting and the same skill returns, `_try_alternate_skill()` randomly picks a different skill_id
   - Sets protocol on `_SkillTracker` for step tracking and prompt injection
5. **`qwen3_action()`** — builds prompt: system prompt + `format_skill_guidance_for_prompt()` (active skill name, strategy, plan steps with `>>` marker at current step, preconditions, done-when, abort-if) + recent actions/rewards context + numbered action list → Qwen3-8B via vLLM
6. **`parse_qwen_response()`** — multi-strategy action extraction: exact match → numbered selection → substring → edit distance → token overlap → **RAG embedding semantic match** (`ActionEmbeddingMatcher` using `Qwen3-Embedding-0.6B`) as final fallback
7. **`_apply_anti_repetition()`** — if same action repeated N times with 0 reward, randomly pick an alternative
8. **`env.step(action)`**
9. **`_SkillTracker.update()`** — advance protocol step index, track reward-on-skill, switch count
10. **Build `Experience`** with: state, action, reward, next_state, done, intentions, tasks, sub_tasks (active skill), summary_state, available_actions

**Key features:**

- Protocol-aware skill lifecycle (find-apply loop with duration caps, stall detection, criteria matching)
- RAG `ActionEmbeddingMatcher` for semantic action fallback
- Anti-repetition guard
- Per-game skill bank loading (`bank_dir/<game_name>/`)
- Output: `test_rollout/decision_agent/<game>/<timestamp>/`

**Usage:**

```bash
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
export VLLM_BASE_URL="http://localhost:8000/v1"

python -m scripts.qwen3_decision_agent --games twenty_forty_eight --episodes 3
python -m scripts.qwen3_decision_agent --one_per_game --gpu 0 -v
python -m scripts.qwen3_decision_agent --no-bank --episodes 3        # baseline without skill bank
python -m scripts.qwen3_decision_agent --bank /path/to/bank --episodes 3
```

### Pipeline B — `inference/run_qwen3_8b_eval.py` (without skill select)

General-purpose evaluation script across multiple benchmarks, with optional skill bank support but no skill lifecycle tracking.

**Per-step loop:**

1. **`get_state_summary()`** — same deterministic + LLM state compression
2. **`infer_intention()`** — same Qwen3-8B intention inference
3. **`get_skill_guidance()`** — optional (via `--bank` flag), simpler query using `state[:500]`, no intention/structured_state scoring, no re-selection logic
4. **`qwen3_agent_action()`** — builds prompt: system prompt + skill guidance text + user template (comma-separated actions) → Qwen3-8B via vLLM
5. **`_parse_qwen_response()`** — simpler parsing: exact match (case-insensitive) → `extract_action()` fallback → first valid action (no fuzzy/edit-distance/RAG)
6. **`env.step(action)`**
7. **Generate experience summary** via LLM: a "short strategic note" from state + action (extra LLM call per step)
8. **Build `Experience`** with same rich fields

**Game-specific episode runners:**

| Runner | Games | Features |
|--------|-------|----------|
| `run_qwen3_episode()` | 2048, Tetris, Candy Crush | Standard LMGame-Bench loop |
| `run_qwen3_avalon_episode()` | Avalon | Multi-agent (all players = Qwen3), `ThreadPoolExecutor` parallel queries |
| `run_qwen3_diplomacy_episode()` | Diplomacy | 7 powers, order parsing, SC delta tracking, 20-phase cap |
| `run_qwen3_orak_episode()` | Super Mario | Orak env wrappers |

**Key features:**

- Multi-benchmark (LMGame-Bench + AgentEvolver + Orak)
- Resume interrupted runs (`--resume`)
- Per-experience LLM summary generation
- Output: `output/<model_slug>/<game>/<timestamp>/`

**Usage:**

```bash
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
export VLLM_BASE_URL="http://localhost:8000/v1"

python -m inference.run_qwen3_8b_eval --games twenty_forty_eight --episodes 3
python -m inference.run_qwen3_8b_eval --episodes 10                   # all 6 games
python -m inference.run_qwen3_8b_eval --resume                       # resume interrupted run
python -m inference.run_qwen3_8b_eval --bank path/to/bank.jsonl      # with optional skill bank
python -m inference.run_qwen3_8b_eval --list-games                   # show available games
```

### Pipeline comparison

| Aspect | `qwen3_decision_agent.py` | `run_qwen3_8b_eval.py` |
|--------|---------------------------|------------------------|
| Skill bank | Required (per-game, query engine, tracker) | Optional (`--bank` flag) |
| Skill lifecycle | `_SkillTracker` with reselect, alternate, protocol steps | Single query per step, no tracking |
| Skill query key | `game + intention + state[:1500]` | `state[:500]` |
| Applicability scoring | Yes (structured_state → predicate floats) | No (pass_rate proxy only) |
| Action parsing | Fuzzy + edit distance + RAG embedding | Exact match + `extract_action()` |
| Anti-repetition | Yes | No |
| Action format in prompt | Numbered list | Comma-separated |
| Game coverage | LMGame-Bench | LMGame-Bench + Avalon + Diplomacy + Orak |
| Experience summary | State summary only | Extra LLM call for strategic note |
| Output dir | `test_rollout/decision_agent/` | `output/<model>/` |
| Resume support | No | Yes |

---

## Skill selection (RAG mode)

Skill selection is RAG-based by default. When `SkillQueryEngine` initializes, it auto-loads the `Qwen3-Embedding-0.6B` embedder and pre-embeds all skill descriptions. The TF-IDF keyword fallback in `agent_helper._rank_skills_by_relevance()` only fires if the query engine fails to initialize.

### How `select_skill_from_bank()` routes

The function tries four paths in order, stopping at the first success:

1. **`SkillQueryEngine.select()`** — richest path (RAG relevance + applicability + structured guidance)
2. **`SkillQueryEngine.query_for_decision_agent()`** — convenience wrapper that delegates to `select()` when state is available
3. **`SkillBankAgent.select_skill()`** — alternative agent-level selection
4. **TF-IDF keyword fallback** via `_rank_skills_by_relevance()` — only when no query engine is available

### `SkillQueryEngine.select()` scoring

Each candidate skill is scored on three axes and combined into a final confidence:

| Component | Weight | Source |
|-----------|--------|--------|
| Retrieval relevance | 40% | RAG embedding cosine similarity + keyword Jaccard |
| Execution applicability | 35% | Effect compatibility against current state predicates |
| Historical pass rate | 25% | Success rate from past executions |

Skills are sorted by confidence and top-k returned as `SkillSelectionResult` objects containing: `skill_id`, `skill_name`, `why_selected`, `relevance`, `applicability_score`, `confidence`, `expected_effects`, `preconditions`, `termination_hint`, `failure_modes`, `execution_hint`, `micro_plan`, `contract`, `pass_rate`.

---

## Files

| File / sub-package | What it does |
|--------------------|--------------|
| `agent.py` | `VLMDecisionAgent` (LLM decision agent), `run_tool()`, `run_episode_vlm_agent()`, tool handlers (e.g. `TOOL_SELECT_SKILL` → `active_skill_plan` from protocol steps) |
| `agent_helper.py` | `get_state_summary()`, `build_rag_summary()`, `extract_game_facts()`, `infer_intention()`, `EpisodicMemoryStore`, `skill_bank_to_text()`, `query_skill_bank()` / `select_skill_from_bank()`, `_get_protocol_for_skill()` |
| `reward_func.py` | `RewardConfig`, `RewardResult`, `RewardComputer`, `compute_reward()` (r_follow uses skill contract `eff_add`) |
| `dummy_agent.py` | Baseline `language_agent_action()` + game detection + action extraction for all 6 supported games (LMGame-Bench, AgentEvolver, Orak) |
| `__init__.py` | Re-exports the above; lazy `__getattr__` for `GPT4oCollectorActor` / `QwenVLActor` etc. |
| [`core/`](core/) | Multimodal scaffolding (`VisualInput`, `build_*_messages`) — used by the SFT / GRPO actors above. |
| [`SFT/`](SFT/) | GPT-4o data-collection actor that writes `trainer/SFT`-compatible JSONL. See [`SFT/README.md`](SFT/README.md). |
| [`grpo/`](grpo/) | Qwen3-VL-8B online actor + GRPO rollout logger. See [`grpo/README.md`](grpo/README.md). |

---

## Quick start — run a full episode

`run_episode_vlm_agent()` returns an **`Episode`** object (from `data_structure.experience`) with fully-populated `Experience` objects per step.

```python
from decision_agents import VLMDecisionAgent, run_episode_vlm_agent, RewardConfig

episode = run_episode_vlm_agent(
    env,
    model="Qwen/Qwen3.5-9B",  # = BACKBONE_MODEL; pass "gpt-5.5" for the
                              # training-free SFT-teacher cold-start path
    task="Complete level 1",
    max_steps=200,
    verbose=True,
)

print(episode.get_length())
print([e.reward for e in episode.experiences])
print([e.reward_details for e in episode.experiences])
print(episode.metadata["cumulative_reward"])
print(episode.experiences[-1].done)

exp = episode.experiences[0]
print(exp.summary_state)   # key=value format
print(exp.intentions)      # [TAG] phrase
print(exp.sub_tasks)       # active skill ID
print(exp.reward_details)  # full reward breakdown dict
```

### With a skill bank and custom reward config

```python
from decision_agents import (
    VLMDecisionAgent,
    run_episode_vlm_agent,
    EpisodicMemoryStore,
    RewardConfig,
)
from skill_agents.skill_bank.bank import SkillBankMVP

bank = SkillBankMVP("path/to/bank.jsonl")
bank.load()

from rag import get_text_embedder
memory = EpisodicMemoryStore(max_entries=500, embedder=get_text_embedder())

reward_cfg = RewardConfig(
    w_follow=0.1,
    query_mem_cost=-0.05,
    query_skill_cost=-0.05,
    call_skill_cost=-0.02,
    skill_switch_cost=-0.10,
)

agent = VLMDecisionAgent(
    model="Qwen/Qwen3.5-9B",       # = BACKBONE_MODEL
    skill_bank=bank,
    memory=memory,
    reward_config=reward_cfg,
    retrieval_budget_n=10,
    skill_abort_k=5,
)

episode = run_episode_vlm_agent(env, agent=agent, task="Clear all boxes", max_steps=500, verbose=True)
```

---

## Step-by-step control (manual loop)

```python
from decision_agents import VLMDecisionAgent

agent = VLMDecisionAgent(model="Qwen/Qwen3.5-9B")  # = BACKBONE_MODEL
obs, info = env.reset()

last_tool_name = None
last_tool_result = None

for t in range(200):
    decision = agent.step(str(obs), info, last_tool_name, last_tool_result)
    tool = decision["tool"]
    args = decision["args"]

    if tool == "take_action":
        obs, reward, term, trunc, info = env.step(args["action"])
        agent.update_from_tool_result("take_action", args["action"], str(obs))
        if term or trunc:
            break
    elif tool == "reward":
        rr = agent.reward_computer.compute_reward(r_env=reward, action_type="primitive", observation=str(obs))
        agent.update_from_tool_result("reward", rr, str(obs))
    else:
        from decision_agents import run_tool
        result = run_tool(tool, args, agent, str(obs), info)
        agent.update_from_tool_result(tool, result, str(obs))

    last_tool_name = tool
    last_tool_result = decision.get("result")
```

---

## Skill bank: protocol store vs contract

The skill bank stores each skill as a **Skill** object with two logical parts (see `skill_agents.stage3_mvp.schemas`):

- **Protocol store** — What the decision agent sees: `name`, `strategic_description`, `tags`, `protocol` (steps, preconditions, success_criteria, abort_criteria, expected_duration), `confidence`. Used by `skill_bank_to_text()`, `query_skill_bank()`, and to set `active_skill_plan` from `protocol.steps`.
- **Contract** — Effects (`eff_add`, `eff_del`, `eff_event`) used for segmentation, verification, and **reward shaping**. The agent still gets the contract via `bank.get_contract(skill_id)` when computing r_follow.

So: the agent **plans** from protocols (when present) and is **rewarded** for making progress on the contract's eff_add predicates.

---

## Helper functions

### `get_state_summary(observation, structured_state=None, *, max_chars=400, use_llm_fallback=False, llm_callable=None)`

Produces a compact `key=value` state summary optimised for LLM context windows, retrieval, skill-bank indexing, and trajectory segmentation. Summaries are **never** raw observation text and always ≤ 400 characters.

**Priority order:**
1. `structured_state` → `compact_structured_state()` (preferred; wrapper-produced dict)
2. `observation` → `compact_text_observation()` (deterministic boilerplate removal + clause compression)
3. LLM fallback (optional, disabled by default)

```python
from decision_agents import get_state_summary

summary = get_state_summary(
    obs_text,
    structured_state=info.get("structured_state"),
)
# → "game=tetris | phase=midgame | stack_h=14 | holes=32 | next=T,Z,I,J | level=1"
```

**Supported wrappers with `build_structured_state_summary()`:**

| Wrapper | Key fields | Example |
|---------|-----------|---------|
| GamingAgent (LMGame-Bench) | game, step, self, objective, critical, affordance | `game=2048 \| self=highest:256 \| objective=merge tiles` |
| Avalon | game, phase, self, progress, critical, objective | `game=avalon \| phase=team_vote \| self=role:Percival(G)` |
| Diplomacy | game, phase, self, resources, critical, objective | `game=diplomacy \| phase=S1902M \| self=power:FRANCE centers:5` |
| Orak (Mario) | game, step, self, objective, critical, affordance | `game=super_mario \| self=pos:(120,80) \| objective=reach flag` |

### `build_rag_summary(state, game_name, *, step_idx, total_steps, reward, max_chars)`

Fully deterministic (no LLM) `key=value` summary optimised for RAG embedding retrieval. Combines game-aware fact extraction with phase estimation and reward.

```python
from decision_agents.agent_helper import build_rag_summary

summary = build_rag_summary(
    state_text,
    game_name="tetris",
    step_idx=50,
    total_steps=86,
    reward=1.0,
)
# → "game=tetris | phase=midgame | step=50/86 | stack_h=14 | holes=32 | next=T,Z,I,J | level=1 | reward=+1"
```

Uses `extract_game_facts()` internally — game-specific parsers for Tetris (stack_h, holes, piece, next), 2048 (highest, empty, tiles, merges), Candy Crush (score, moves, pairs), Super Mario (mario position, enemies, items), Avalon (phase, role, quest), and Diplomacy (phase, power, centers, units).

### `infer_intention(summary_or_observation, game=None, model=None, context=None)`

Returns a `[TAG] subgoal phrase` (≤15 words) describing the agent's
current subgoal. Constrained to the legacy game-tactical vocabulary:

```
SUBGOAL_TAGS = SETUP | CLEAR | MERGE | ATTACK | DEFEND | NAVIGATE | POSITION |
               COLLECT | BUILD | SURVIVE | OPTIMIZE | EXPLORE | EXECUTE   (13)
```

```python
from decision_agents import infer_intention

intention = infer_intention(
    summary,
    context={
        "last_actions": ["up", "left"],
        "progress_notes": ["pushed box onto goal"],
        "task": "push all boxes to goals",
    },
)
# e.g. "[NAVIGATE] Push remaining box right toward goal tile"
```

For the cross-domain `INTENT_OPERATORS` vocabulary (gym-v / future
two-MDP), use the data-layer labeler `labeling/label_intentions_gpt54.py`
instead — it consumes `(metadata.schema, action)` and emits a
`[OPERATOR] note` rewrite of `Experience.intentions` so the segmenter
sees a real categorical signal.

```
INTENT_OPERATORS = INSPECT | TRACK | COMPARE | COMMIT | VERIFY | RECOVER  (6)
```

| Outer-step operator | Cognitive mode |
|---------------------|----------------|
| `INSPECT`           | parse / understand current state — menus, transitions, opening screens, RAG retrieval |
| `TRACK`             | follow or wait on a state change you do not control |
| `COMPARE`           | explicitly weigh options before commit |
| `COMMIT`            | take the chosen action with goal-progressing intent |
| `VERIFY`            | check the result of a recent action against expectation |
| `RECOVER`           | reactive / defensive response to surprise or failure |

Both vocabularies live in `agent_helper.py` and are bridged by the
alias maps `SUBGOAL_TO_OPERATOR` / `OPERATOR_TO_SUBGOAL` and by the
segmenter's `parse_intention_tag` (in
`skill_agents/boundary_proposal/signal_extractors.py`), so a unified
consumer can normalise either prefix to either alphabet.

### `EpisodicMemoryStore`

RAG-embedding retrieval memory for the `query_memory` tool. When an embedder is supplied (or auto-loaded from `rag/`), memories are embedded on `add` and queries use cosine similarity blended with keyword overlap.

```python
from decision_agents import EpisodicMemoryStore
from rag import get_text_embedder

mem = EpisodicMemoryStore(
    max_entries=500,
    embedder=get_text_embedder(),
    embedding_weight=0.7,
)

mem.add_experience(
    state_summary="game=tetris | stack_h=14 | holes=32 | next=T,Z,I,J | level=1",
    action="rotate_cw",
    next_state_summary="game=tetris | stack_h=14 | holes=30 | next=Z,I,J,S | level=1",
    done=False,
)

results = mem.query("game=tetris | stack_h=high | holes=many", k=3)
```

### `skill_bank_to_text(skill_bank)` and `query_skill_bank(skill_bank, state, task, ...)`

**`skill_bank_to_text(skill_bank)`** — Formats the skill bank for agent prompts. When a skill has a protocol (name, strategic_description, steps), the string shows those; otherwise it falls back to effect counts.

**`query_skill_bank(skill_bank, state, task, ...)`** — Alias for `select_skill_from_bank`. Picks the best-matching skill for the current state/task and returns it with a protocol dict (steps, preconditions, success_criteria, expected_duration).

---

## Reward function

### Standalone usage

```python
from decision_agents import RewardComputer, RewardConfig

cfg = RewardConfig(w_follow=0.1, skill_switch_cost=-0.10)
rc = RewardComputer(cfg)

rr = rc.compute_reward(
    r_env=1.0,
    action_type="primitive",
    observation="checkpoint area",
    active_skill_id="nav_to_cp",
    skill_contract=contract,
)
print(rr)
# RewardResult(r_env=1.0000, r_follow=0.0500, r_cost=0.0000, r_total=1.0050)
```

### RewardConfig defaults

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `w_follow` | 0.1 | Weight on r_follow in r_total |
| `query_mem_cost` | -0.05 | Cost per QUERY_MEM action |
| `query_skill_cost` | -0.05 | Cost per QUERY_SKILL action |
| `call_skill_cost` | -0.02 | Cost per CALL_SKILL action |
| `skill_switch_cost` | -0.10 | Penalty when active skill changes |
| `follow_predicate_bonus` | 0.05 | Bonus per newly satisfied eff_add predicate |
| `follow_completion_bonus` | 0.20 | Bonus when all eff_add predicates satisfied |
| `follow_no_progress_penalty` | -0.01 | Penalty per step with no predicate progress |

### Reward components

- **r_env**: Raw environment reward passed through.
- **r_follow**: Skill-following shaping (termination-free). Checks how many `eff_add` predicates from the active skill's contract appear in the observation. Awards bonuses for newly satisfied predicates and a completion bonus when all are met.
- **r_cost**: Negative costs for queries, skill calls, and skill switching.
- **r_total**: `r_env + w_follow * r_follow + r_cost`.

---

## Dummy agent (baseline)

The original single-call LLM agent, for comparison or simple use:

```python
from decision_agents import language_agent_action

action = language_agent_action(
    state_nl=observation_text,
    game="gamingagent",
    model="Qwen/Qwen3.5-9B",  # = BACKBONE_MODEL; or "gpt-5.5"
)
```

Supports all 6 games: 2048, Candy Crush, Tetris (LMGame-Bench), Avalon, Diplomacy (AgentEvolver), Super Mario (Orak).

---

## Per-step loop (LLMDecisionAgent protocol)

Every timestep the runner executes:

1. **`get_state_summary`** — required; runner computes it before action (returns `key=value` facts).
2. **(Optional)** **`select_skill`** — choose a skill when no active skill, skill exhausted, or agent is stuck. Returns full structured guidance (protocol steps, preconditions, termination hints, failure modes). Budget-limited to once every N steps unless stuck.
3. **`take_action`** — required; exactly one environment action. Agent has intention (from previous step), fresh state summary, and any active skill guidance in context.
4. **`get_intention`** — required; runner updates intention after observing action result (returns `[TAG] subgoal phrase`).
5. **`reward`** — required; compute `(r_env, r_follow, r_cost, r_total)` for logging/training.

### Format consistency

The agent prompt uses consistent formats across cold-start labeling and runtime inference:
- **Intention**: `"[TAG] subgoal phrase"` (e.g., `"[CLEAR] Reduce holes before stack overflows"`)
- **State summary**: `"key=value"` pairs (e.g., `"game=tetris | phase=endgame | stack_h=15 | holes=42"`)
- **Memory results**: `key=value` summaries from `EpisodicMemoryStore`
