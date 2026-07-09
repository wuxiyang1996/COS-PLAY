# Multi-Hop Visual Reasoning as Long-Horizon Interactions

## Core Idea

Multi-hop visual reasoning can be reframed as a **long-horizon interaction** problem.
Instead of performing all reasoning hops inside a single LLM inference pass, each hop becomes an explicit step in an MDP — turning the reasoning trace into a proper trajectory that can be optimized end-to-end with RL.

### Current Granularity

|                    | Multi-hop VLM Agent                                              | COS-PLAY (Game Agent)                                      |
|--------------------|------------------------------------------------------------------|-------------------------------------------------------------|
| **One "step"**     | Screenshot → (all hops internally in one LLM call) → env action  | State summary → skill select → action                       |
| **"Reasoning"**    | Happens *inside* a single inference pass (hop1→hop2→…→action)    | Happens *across* steps (skill guides action over many steps) |
| **Horizon**        | Short (hops are hidden, only env actions count as steps)         | Long (dozens to hundreds of env steps)                      |

### The Reframe

Treat each reasoning hop as an explicit step in an MDP, not as a hidden computation inside a single LLM call.

---

## Two-Level MDP

```
┌─────────────────────────────────────────────────────┐
│  OUTER MDP (environment level)                       │
│  State: screenshot + task description                │
│  Action: click / type / game-move                    │
│  Reward: task success (sparse)                       │
│                                                      │
│  ┌───────────────────────────────────────────────┐   │
│  │  INNER MDP (reasoning level)                  │   │
│  │  State: visual schema + hop trace so far       │   │
│  │  Action: reasoning operation (one of):         │   │
│  │    - GROUND(entity_query)  → find entities     │   │
│  │    - CHECK(relation/attr)  → verify condition  │   │
│  │    - RETRIEVE(memory_key)  → query skill bank  │   │
│  │    - CONCLUDE(intermediate)→ commit subgoal    │   │
│  │    - EXECUTE(env_action)   → exit inner loop   │   │
│  │  Reward: shaped from hop quality + outer reward│   │
│  └───────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

Each outer step contains 1..N inner reasoning steps. The agent **learns when to stop reasoning and act**.

---

## Architecture

```
Screenshot ──→ VLM Schema Gen ──→ <state> entities/relations/flags
                                        │
                                        ▼
                              ┌── Reasoning Loop ──┐
                              │                     │
                              │  hop_t: SELECT reasoning action  ◄── Skill guidance
                              │    │                │
                              │    ├─ GROUND(query) → update entity set
                              │    ├─ CHECK(pred)   → update state_flags
                              │    ├─ RETRIEVE(key) → pull from memory/skill bank
                              │    ├─ CONCLUDE(sub) → commit intermediate
                              │    └─ EXECUTE(act)  → break loop, send to env
                              │                     │
                              │  hop_t+1: ...       │
                              └─────────────────────┘
                                        │
                                        ▼
                              Environment action ──→ new screenshot ──→ repeat
```

Each episode trajectory becomes:

```
(schema_0, GROUND, schema_0') → (schema_0', CHECK, schema_0'') → (schema_0'', EXECUTE(click), schema_1) → ...
```

This is a proper long-horizon trajectory that the skill discovery pipeline can segment, label, and learn from.

---

## Mapping Existing Components

### Hop trace → inner action vocabulary

The existing VLM Agent hop trace already defines the inner action space:

```
hop1 = locate relevant input fields [e2,e5]       → GROUND action
hop2 = check required constraints from task goal   → CHECK action
hop3 = detect blocker or missing prerequisite [e4] → CHECK action
hop4 = select feasible next action path            → CONCLUDE action
output = click/input action                        → EXECUTE action
```

### COS-PLAY machinery → training infrastructure

| COS-PLAY Component                                 | Role in Unified System                                                        |
|-----------------------------------------------------|-------------------------------------------------------------------------------|
| **Skill Bank**                                      | Stores reusable **hop chain templates** (not just action patterns)            |
| **Skill protocols** (trigger, steps, abort/success) | Become **reasoning protocols** — when to ground, when to check, when to conclude |
| **GRPO training**                                   | Optimizes the full reasoning chain end-to-end                                 |
| **Co-evolution loop**                               | Discovers new reasoning patterns from trajectories                            |
| **RAG + embeddings**                                | Retrieves relevant reasoning templates for current visual state               |

---

## Why This Works

1. **Adaptive reasoning depth** — RL learns when 1 hop is enough vs. when 5 are needed. No more fixed hop chains.

2. **Transferable reasoning skills** — the same abstract pattern works across domains:
   - **Games**: piece blocked → clear obstacle → retry move
   - **Web**: form invalid → fill missing field → resubmit
   - **Visual QA**: weak evidence → gather another anchor → conclude

3. **Credit assignment through the reasoning chain** — GRPO optimizes not just "which action to take" but "which reasoning steps lead to good actions."

4. **Skills capture *how to think*, not just *what to do*** — example:

   ```
   skill: constraint_satisfaction
   trigger: state_flags.error != null OR target.blocker != null
   protocol:
     hop1: GROUND(blocker entity)
     hop2: CHECK(what constraint is violated)
     hop3: RETRIEVE(similar past resolution)
     hop4: CONCLUDE(subgoal = resolve blocker first)
     hop5: EXECUTE(action addressing blocker)
   ```

5. **Unifies all environments** — games, browser, OSWorld, and visual QA all become instances of the same observe-reason-act loop at different horizon lengths.

---

## Design Decisions

### 1. Re-observation between reasoning hops

- **Option A (default):** Hops operate on the same `<state>` schema, only updating an internal scratchpad. Cheaper, faster.
- **Option B (selective):** GROUND actions can trigger re-rendering or zooming into a region. More expensive but handles visual detail.
- **Recommendation:** A for games/web, B for visual QA where fine-grained visual grounding matters.

### 2. Reward for inner hops

Terminal reward from the outer env is sparse and far away. Options:

- **Schema consistency reward** — does the hop produce a valid schema update?
- **Hop trace quality reward** — GPT-4o judges full hop trace (used for GRPO).
- **Progress shaping** — reward for reducing uncertainty or resolving blockers.
- **Recommendation:** GRPO with trajectory-level reward; let the policy learn which hops matter. This aligns with the existing `r_follow` component.

### 3. Inner loop length

- Hard cap (e.g., max 8 hops per outer step) + learned EXECUTE decision.
- The existing `_SkillTracker` abort/success criteria naturally handles this — if the skill protocol says "max 5 hops," the tracker enforces it.

### 4. LoRA adapter layout

| Adapter          | Purpose                                                       |
|------------------|---------------------------------------------------------------|
| `schema_gen`     | Screenshot → `<state>` schema (Qwen3-VL)                     |
| `hop_select`     | Schema + trace → next reasoning action (replaces `action_taking`) |
| `skill_select`   | Schema → which reasoning skill to invoke                      |
| `segment`        | Trajectory → skill boundary detection                         |
| `contract`       | Segment → effects contract                                    |

---

## Transferable Skill Families Under This Model

| Family                               | Game                            | Web                               | Visual Reasoning                         |
|--------------------------------------|---------------------------------|-----------------------------------|------------------------------------------|
| **Locate → filter → select**         | Candidate moves → best legal    | UI candidates → relevant control  | Objects → attributes → answer target     |
| **Blocker → prerequisite → replan**  | Deadlock → missing setup        | Disabled control → missing field  | Weak evidence → gather anchor            |
| **History → hidden state → act**     | Dialogue → alliance/threat      | Prior pages → next step           | Prior frames → disambiguate              |
| **Compare under future constraint**  | Move preserving structure       | Path lowering risk/steps          | Candidate consistent with constraints    |

Under the long-horizon framing, each family is a reusable **multi-step reasoning policy** — not a single-call chain-of-thought template, but an actual policy that can be trained, composed, and transferred.

---

## Open Questions

1. **Starting environment** — begin with BrowserGym (where multi-hop reasoning is most natural) or go cross-domain from the start?

2. **Inner MDP vs. structured chain-of-thought** — should hops be truly separate LLM calls (more controllable, more training signal) or structured output within a single call (simpler) with added structure?

3. **Passive reasoning evaluation** — M3-Agent's multi-turn retrieval + reasoning over long-term memory is structurally close to this inner MDP. Keep long-video QA benchmarks in scope as a "passive reasoning" evaluation, or stay focused on interactive environments?
