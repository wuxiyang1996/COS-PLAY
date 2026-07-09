# PLAN: Action Agent (Decision Agent)

**Scope:** The decision-making agent that consumes structured `<state>` schemas from the [Visual Grounding](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md) pipeline and selects/executes environment actions guided by skills from the [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md).

**Scope boundaries (deliberate).** The Action Agent is **domain-general** across game / webagent / os-agent / video-understanding / visual reasoning — the inner-action alphabet and the three-layer actor (§5) carry across all of them (see [§5.3a Cross-domain semantics](#53a-cross-domain-semantics-of-inner-actions)). The skills it invokes are **general protocols feasible across all five target domains** ([Skill Bank §0.1](../03-skill-bank/PLAN-SKILL-BANK.md#01-general-protocol-invariant-no-domain-specific-skill-families)); the agent does not consume domain-specific skill families. The **first evaluation arena** for the agent and its retrieved protocols is **short-video evidence-grounded reasoning** (Video-Holmes-style) — that determines which adapters and eval slices are wired first, not what kind of skills exist. The agent operates entirely over an episode-local trajectory: `RETRIEVE` targets only the skill bank, and any evidence the agent cites lives inside the current episode's structured `<state>`, hop trace, and intermediate belief state.

**Upstream:** Structured schema from visual grounding (vlm_wrapper heads); skill guidance from Skill Bank.
**Downstream:** Environment actions; experience trajectories fed back to Skill Bank and GRPO training.

---

## 1. Architecture overview

The action agent implements the three-stage decision loop from COS-PLAY (COLM 2026):

```
structured_state (from VLM grounding)
    ↓
skill retrieval (from Skill Bank)  ←──── re-select on stall/completion/abort
    ↓
intention update (subgoal inference)
    ↓
action execution (from valid action set)
    ↓
reward computation (r_env + r_follow + r_cost)
    ↓
experience → Skill Bank ingestion + GRPO buffer
```

### Per-step loop

The actor is decomposed into three explicit layers. Rule-heavy scaffolding runs first; the lightweight inner MDP only activates when local uncertainty cannot be resolved deterministically; the final action head always terminates the step.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer A — Skill Continuation Gate  (mostly rule-heavy)          │
│  schema update → intention inference → continue / reselect /     │
│  no-skill decision; stall, duration, success/abort checks        │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓  (only if local uncertainty remains)
┌─────────────────────────────────────────────────────────────────┐
│  Layer B — Lightweight Inner MDP  (typed, ≤3 hops)               │
│  GROUND? CHECK? RETRIEVE? COMMIT? — resolve local uncertainty    │
│  using a short typed trace, then exit to Layer C                 │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer C — Final Action Head                                     │
│  action selection from valid list, parsing / fallback chain,     │
│  anti-repetition, env.step, experience build                     │
└─────────────────────────────────────────────────────────────────┘
```

Concretely, each step executes:

1. **`get_state_summary()`** — deterministic + LLM state compression into `key=value` format (≤400 chars). Prefers `structured_state` from VLM grounding when available. *(Layer A)*
2. **`infer_intention()`** — LLM produces a `[TAG] subgoal phrase` from summary + context (last actions, task). Tags: SETUP, CLEAR, MERGE, ATTACK, DEFEND, NAVIGATE, POSITION, COLLECT, BUILD, SURVIVE, OPTIMIZE, EXPLORE, EXECUTE. *(Layer A)*
3. **Skill continuation gate** — rule-heavy check using `_SkillTracker`: triggers re-query when no active skill, duration exceeded, zero-reward stall (≥4 steps), abort/success criteria matched in current state, or invalid context for the active skill. *(Layer A)*
4. **`get_skill_guidance()`** — on reselect only: queries `SkillQueryEngine` (RAG mode) using `game_name + intention + state_text[:1500]`, with `structured_state` converted to `{predicate: float}` for applicability scoring. Returns skill_id, protocol, execution_hint, failure_modes. *(Layer A)*
5. **Lightweight inner MDP** (invoked only if Layer A leaves local uncertainty) — short typed hop sequence from `{GROUND, CHECK, RETRIEVE, COMMIT}`, capped at 0–2 hops by default and 3 hops under uncertainty. `hop_select` is a constrained typed router (see §5), not a free-form planner. *(Layer B)*
6. **`action()`** — builds prompt: system prompt + skill guidance + recent actions/rewards + valid action list → LLM → action. *(Layer C)*
7. **`parse_response()`** — multi-strategy action extraction: exact match → numbered selection → substring → edit distance → token overlap → RAG embedding semantic match as final fallback. *(Layer C)*
8. **Anti-repetition** — if same action repeated N times with 0 reward, randomly pick alternative. *(Layer C)*
9. **`env.step(action)`** *(Layer C)*
10. **`_SkillTracker.update()`** — advance protocol step index, track reward-on-skill, switch count.
11. **Build `Experience`** — state, action, reward, next_state, done, intentions, tasks, sub_tasks (active skill), summary_state, available_actions.

**Boundary rule.** Heavy reasoning tasks — failure diagnosis, counterfactual analysis, skill composition, cross-domain transfer mapping, new skill invention, reflective repair planning — **never** run inside this per-step loop. They are reserved for the offline 32B/72B synthesis/reflection tier (see §2, §6). The inner MDP is the actor's guardrail, not its brain.

---

## 1a. Actor Role and Boundary

The Actor Agent should follow the COS-PLAY Decision Agent pattern rather than introducing a separate controller. It takes schema-based state as input, builds a compact state summary and intention, optionally retrieves or continues a skill, and then outputs either a primitive action, a skill-conditioned action, or a typed reasoning step. Reasoning steps are bounded intermediate decisions within the same online control loop rather than a separate long-horizon planner.

For this project, the Actor remains the **online policy**. It is responsible for deciding, at each step, whether to continue the current skill, switch to another eligible skill, act without a skill, or emit a typed reasoning step before acting. This responsibility should not be delegated entirely to the [Harness](../05-harness/PLAN-HARNESS.md), even when the Harness is stronger.

The main reason is that skill continuation, skill switching, no-skill fallback, reasoning-step emission, and primitive action selection all belong to the same policy space. Moving final skill choice into the Harness would break the COS-PLAY-style decision loop and turn the Harness into a hidden policy model rather than a runtime support module.

### 1a.1 Actor input / output contract

The Actor takes schema-based state as the primary input. Textual rendering may be used only as an auxiliary prompt representation.

Suggested actor input:

```python
ActorInput = {
    "episode_id": str,
    "step_idx": int,
    "schema_state": dict,
    "state_text": str,
    "valid_actions": list[str],
    "recent_actions": list[str],
    "recent_rewards": list[float],
    "task_spec": dict,
    "active_skill": dict | None,
    "eligible_skills": list[dict],   # filtered by Harness
    "local_reasoning_trace": list[dict],
}
```

The Actor output should allow three step types:

```python
ActorOutput = {
    "step_type": "primitive_action" | "skill_conditioned_action" | "reasoning_step",
    "state_summary": str,
    "intention": str,
    "selected_skill_id": str | None,
    "reasoning_step": dict | None,
    "action": str | None,
    "evidence_warrant": list[str],
    "notes": dict,
}
```

### 1a.2 Actor decision scope

The Actor is the **final policy-level decision maker** for:

- whether to continue the current skill,
- whether to switch to another eligible skill,
- whether to act without a skill,
- whether to emit a typed reasoning step first,
- which final primitive action to take.

The Actor should **not** directly perform:

- skill promotion,
- bank mutation,
- transfer validation,
- replay validation,
- rollback,
- long-horizon reflection,
- memory retrieval.

Those belong to other modules (see [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md), [Harness](../05-harness/PLAN-HARNESS.md), [Pipeline Orchestrator](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md), [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)).

### 1a.3 Why final skill selection stays in the Actor

Although the system uses a strong frozen Harness, **final online skill choice should remain in the Actor**.

Reasons:

- The Actor is the trainable policy, so it must learn skill continuation, switching, no-skill fallback, and reasoning/action coordination.
- Skill choice and reasoning-step choice are coupled. Separating them across different controllers would fragment the online policy space.
- If final skill choice is fully delegated to the frozen Harness, the Actor becomes a weak executor instead of a real decision agent. This would create train-test mismatch if the goal is to improve the Actor itself.

Therefore, the Actor must remain the final decider over:

- continue skill,
- switch skill,
- no skill,
- reasoning step,
- primitive action.

### 1a.4 Typed reasoning step as a first-class Actor output

To support schema-based multimodal reasoning, the Actor may emit a **typed reasoning step** before action. This extends the COS-PLAY decision pattern without replacing it.

Supported reasoning step types align with the inner-hop vocabulary in [§5.3](#53-reduced-hop-vocabulary):

- `GROUND`
- `CHECK`
- `COMPARE`
- `VERIFY`
- `RETRIEVE`
- `COMMIT`

These reasoning steps must be:

- bounded,
- structured,
- episode-local,
- validator-friendly,
- non-memory-based.

A reasoning step should never become unrestricted free-form long reasoning.

### 1a.5 Actor runtime loop

The Actor loop should follow:

1. summarize schema state,
2. infer current intention,
3. consult tracker for continue / reselect / no-skill,
4. **receive eligible skill candidates filtered by the Harness**,
5. choose among:
   - continue current skill,
   - switch to an eligible skill,
   - no skill,
   - reasoning step,
   - primitive action,
6. if a reasoning step is chosen, update local context and loop once more (subject to the §5.4 hop cap),
7. if an action is chosen, execute and log experience.

This preserves the COS-PLAY decision skeleton while extending it to typed reasoning steps.

### 1a.6 Actor–Harness interaction

The Actor should **not** query the raw [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) directly for unrestricted online use. Instead:

- Skill Bank retrieves top-k candidate skills.
- Harness filters and validates them into `eligible_skills`.
- Actor performs final policy choice over those eligible skills.

Thus the Actor consumes a **constrained candidate set** rather than an unconstrained bank.

This preserves Actor flexibility while using the Harness to enforce runtime safety and feasibility. The Harness may additionally veto an Actor-proposed skill at invocation time (see [PLAN-HARNESS.md §1a](../05-harness/PLAN-HARNESS.md#1a-harness-role-as-frozen-72b-runtime-layer)); on veto, the Actor must fall back to another eligible skill, no-skill mode, a reasoning step, or a primitive action.

### 1a.7 Training implication

The trainable Actor should still learn:

- when to continue a skill,
- when to switch,
- when to avoid skill use,
- when to issue a reasoning step,
- how to choose final actions.

This is essential if the project's central policy is meant to live in the smaller trainable model rather than in the frozen Harness.

---

## 2. Tiered model architecture

### Design choice: 8B-only vs. tiered 32B/72B + 8B

Two options were considered:

| | Option A: 8B + GRPO LoRA everywhere | Option B: 32B/72B inference + 8B trained executor |
|---|---|---|
| **Decision loop** | 8B (trained) | 8B (trained) — identical |
| **Skill Crafter / Failure Reflector** | 8B (trained) | 32B/72B (inference-only) |
| **Per-step latency** | Fast | Fast (same 8B in the hot path) |
| **Skill/protocol quality** | Limited by 8B reasoning ceiling | Strong — 32B/72B excels at composition, diagnosis, analogy |
| **Trainability** | Fully end-to-end | Real-time loop fully trainable; offline components frozen |
| **Cold-start** | Weak until GRPO converges | 32B/72B generates high-quality trajectories from day one |

**Decision: Option B (tiered).** The pipeline's offline components — Skill Crafter (composition, generalization, hypothesis), Failure Reflector (backward trace analysis, diagnosis, recovery), and cold-start generation — require multi-step counterfactual reasoning, cross-domain analogy, and structured diagnosis that 8B models consistently struggle with. These components run *between* episodes, not per-step, so the larger model adds zero latency to the decision loop.

### 7B/8B capability assessment

Not all parts of the system demand the same reasoning capability. The following assessment maps each component to a model-size tier based on task complexity, scaffolding, and failure risk.

**Safe for 7B/8B** — bounded work under strong scaffolding:

| Component | Why 7B/8B is sufficient |
|---|---|
| Online action execution (per-step loop §1) | Compressed key=value state, intention tag prediction, retrieval from RAG, action from valid list, robust parsing with fallbacks. Bounded decisions under scaffolding. |
| Skill selection with strong retrieval (§3) | Not pure generation — retrieval + scoring with relevance, applicability, and historical pass rate, plus protocol-aware lifecycle rules (stall detection, success/abort matching, reselect triggers). |
| Protocol following / reactive control | `_SkillTracker` offloads control structure: duration caps, reward stall detection, protocol step tracking, reselect triggers. Model is not inventing control policy from scratch. |
| Short typed inner-hop routing (§5) | Constrained typed hops from `{GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE}`, capped at 0–2 by default and 3 under uncertainty. Only invoked when Layer A leaves local uncertainty; hops operate on the same schema and the action vocabulary is typed and limited. |

**Borderline for 7B/8B** — works only with strong constraints:

| Component | Risk | Mitigation |
|---|---|---|
| Intention inference | Noisy in ambiguous scenes or long histories | Fix the number of tags, keep state summary clean |
| `hop_select` in the inner MDP | Free-form hop generation drifts badly on 7B/8B — the actor starts imitating a full planner | Treat `hop_select` as a constrained typed next-hop router (not a reasoning generator): it predicts `(NEXT_HOP_TYPE, TARGET)` from a fixed 5-token vocabulary, with a hard 0–3 hop cap and a rule-heavy pre-gate (§5). |
| `schema_gen` | Can work for stable interfaces or grid games; becomes bottleneck with messy screenshots, subtle occlusions, or social-video details | Use cascaded head escalation (§12 in Visual Grounding) as fallback |

**Not 7B/8B-safe as currently written** — requires 32B/72B or redesign:

| Component | Why 7B/8B fails | Plan reference |
|---|---|---|
| Failure reflection (localize + diagnose) | Requires backward trace analysis, hypothesis generation, verification, counterfactual confirmation. Even 32B/72B needs 3–5 passes. | [Skill Crafter §6](../04-skill-crafter/PLAN-SKILL-CRAFTER.md#6-failure-reflection--reasoning-recovery) |
| Skill composition | Verifying precondition/postcondition compatibility across multiple skills requires combinatorial checking with multi-pass reasoning. | [Skill Crafter §3](../04-skill-crafter/PLAN-SKILL-CRAFTER.md#3-skill-composer) |
| Skill hypothesis / new skill invention | Best-of-N proposal generation + scoring — too high-variance for 7B; a weaker model here pollutes the bank with bad abstractions. | [Skill Crafter §5](../04-skill-crafter/PLAN-SKILL-CRAFTER.md#5-skill-hypothesizer) |
| Cross-domain transfer | Schema-slot mapping is ambiguous and needs identify-map-instantiate-sanity-check passes; 7B tends to fake analogical reasoning. | [Skill Crafter §4](../04-skill-crafter/PLAN-SKILL-CRAFTER.md#4-skill-generalizer) |
| Cold-start trajectory generation | 8B-only option is weak until GRPO converges; tiered option uses larger models for high-quality traces from day one. | §2 tier table above |

**Key risk:** If 7B/8B is made to do everything, the system may "work" in demos but fail silently in the parts that matter most for self-evolution. The biggest risk is not obvious crashes — it is the model producing plausible but wrong diagnoses, bad skill abstractions, and brittle transferred skills that then contaminate the bank.

### Three-agent role split

The system decomposes into three logical agents with distinct model assignments. The "skills agent" is split conceptually into a **skill-use agent** (online, in the loop) and a **skill-evolution agent** (offline, between episodes) — these are very different jobs.

```
┌──────────────────────────────────────────────────────────────┐
│  Agent 1: Actor / Decision Agent  (Qwen3-8B, GRPO-trained)  │
│  • schema consumption                                        │
│  • intention inference                                       │
│  • skill continuation gate (rule-heavy) + skill selection    │
│  • typed next-hop routing (lightweight inner MDP, ≤3 hops)   │
│  • action execution + anti-repetition                        │
│  • protocol following                                        │
│  • RL adaptation in the real-time loop                       │
│  Timescale: updates every training iteration                 │
└──────────────────────────────────────────────────────────────┘
                        ↕ skill guidance, experience
┌──────────────────────────────────────────────────────────────┐
│  Agent 2: Skill-Use / Operational Agent  (8B or rule-heavy)  │
│  • RAG skill retrieval (SkillQueryEngine)                    │
│  • applicability scoring from structured predicates          │
│  • _SkillTracker lifecycle management                        │
│  • protocol step tracking                                    │
│  • stall detection and reselect triggers                     │
│  • pass-rate lookup                                          │
│  Timescale: updates every few training iterations            │
└──────────────────────────────────────────────────────────────┘
                        ↕ candidate skills, patches, trajectories
┌──────────────────────────────────────────────────────────────┐
│  Agent 3: Skill Synthesis / Reflection Agent                 │
│  (Qwen3-32B/72B, inference-only, frozen)                    │
│                                                              │
│  Roles:                                                      │
│  • Failure reflector (localize + diagnose)                   │
│  • Skill composer (effect chaining)                          │
│  • Skill hypothesizer (failure → new skills)                 │
│  • Cross-domain transfer mapper                              │
│  • Offline trajectory generator (cold-start)                 │
│  • Verification / judge for candidate skills and traces      │
│                                                              │
│  Key design:                                                 │
│  • Outputs are treated as CANDIDATE patches/skills,          │
│    admitted only after multi-pass verification and           │
│    held-out replay checks.                                   │
│  • NOT an unchecked writer — a proposal-and-verification     │
│    engine.                                                   │
│  Timescale: proposals every N episodes, with acceptance gate │
└──────────────────────────────────────────────────────────────┘
```

**Acceptance gate for synthesis-reflection outputs:** Every synthesized or revised skill must pass:
1. Contract completeness checks against the schema
2. Retrieval compatibility with current bank entries
3. Replay or held-out verification on stored trajectories
4. Non-regression filtering (new patches must not lower pass rate on prior successful contexts)

This maps to the existing Tier 1 / Tier 2 split but makes the boundary explicit: **Tier 2 = Agent 1 + Agent 2 (trained, online)**, **Tier 1 = Agent 3 (frozen, offline)**.

### Three model tiers

```
┌─────────────────────────────────────────────┐
│  Tier 0: GPT-5.4 / frontier (API, offline)   │
│  • Initial cold-start labeling               │
│  • Reward model / judge for GRPO             │
└──────────────────┬──────────────────────────┘
                   ↓ bootstrap data
┌─────────────────────────────────────────────┐
│  Tier 1: Qwen3-32B/72B (inference, offline)  │
│  • Skill Composer: effect chaining           │
│  • Skill Hypothesizer: failure → new skills  │
│  • Skill Generalizer: cross-domain transfer  │
│  • Failure Reflector: localize + diagnose    │
│  • Trajectory generation for GRPO buffer     │
│  • Skill verification judging                │
└──────────────────┬──────────────────────────┘
                   │ skills, protocols, recovery
                   │ patches, training signal
                   ↓
┌─────────────────────────────────────────────┐
│  Tier 2: Qwen3-8B + GRPO LoRA (trained)     │
│  • schema_gen: screenshot → <state> schema   │
│  • skill_select: schema + intention → skill  │
│  • hop_select: typed next-hop router (0–3)   │
│  • Action execution + anti-repetition        │
│  • SkillTracker protocol following           │
└─────────────────────────────────────────────┘
```

### Why the 8B real-time loop stays fast

The 32B/72B never enters the per-step loop. Per-step operations (schema_gen, skill_select on reselect, typed hop_select ×0–3, action execution) are all served by the 8B on vLLM, identical to Option A. The 32B/72B runs as a separate offline process:

| Offline operation | When it runs | Frequency |
|---|---|---|
| Failure Reflection (localize, diagnose) | After a failed episode ends | ~1× per failed episode |
| Skill Composition (effect chaining) | Periodic batch job | Every N episodes |
| Skill Hypothesis (from failure patterns) | When the failure-pattern store accumulates | Rare |
| Cross-domain transfer | When adding a new domain | One-time |
| Cold-start trajectory generation | Before GRPO training begins | One-time |

Offline reflection can be pipelined: the 8B rolls out episode batch K+1 while the 32B/72B reflects on batch K. GRPO gradient computation dominates wall-clock time, so the 32B/72B reflection effectively hides inside existing training overhead.

### Multi-run reasoning for 32B/72B

Even at 32B/72B scale, the offline tasks (failure diagnosis, skill composition, cross-domain transfer) are too complex for single-pass inference. The 32B/72B must run **multiple reasoning passes** per task:

| Offline task | Why single-pass is insufficient | Multi-run strategy |
|---|---|---|
| **Failure localization** | Backward trace analysis requires re-evaluating each hop against its context; a single pass often fixates on the symptom step, not the root cause | Pass 1: identify symptom step. Pass 2: re-evaluate each prior hop with targeted prompts. Pass 3: confirm root cause via counterfactual ("would fixing step K have prevented the failure?") |
| **Failure diagnosis** | Diagnosis involves hypothesis generation + verification; one-shot diagnoses frequently hallucinate causal links | Pass 1: generate candidate explanations. Pass 2: verify each against context_snapshot. Pass 3: select most consistent explanation |
| **Skill composition** | Effect chaining across 2+ skills requires verifying precondition/postcondition compatibility, which involves combinatorial checking | Pass 1: propose candidate compositions. Pass 2: verify effect chain validity per pair. Pass 3: generate protocol + test expectations |
| **Skill hypothesis** | Novel skill proposals from failure patterns are speculative; most initial proposals are low-quality | Best-of-N sampling (N=3–5): generate N proposals, score by contract completeness + novelty, keep top-K |
| **Cross-domain transfer** | Schema-slot mapping between domains is ambiguous; naive mappings fail | Pass 1: identify shared structural slots. Pass 2: propose mapping candidates. Pass 3: instantiate and sanity-check with target-domain examples |

This multi-run design means each offline task costs ~3–5× the tokens of a single 32B/72B call, but these costs are amortized across many episodes and remain far cheaper than putting the 32B/72B in the per-step loop. The quality improvement from multi-run (especially for failure localization and diagnosis) is critical — single-pass failure reflection would produce noisy recovery patches that degrade rather than improve skills.

**Budget guideline:** ~500–2000 tokens per offline reasoning pass × 3–5 passes × a few tasks per episode batch = on the order of 10K–50K tokens of 32B/72B inference per training iteration. Negligible compared to GRPO rollout costs.

### Routing

All three tiers share the same code path; `API_func.ask_model` routes to the correct backend:

| Tier | Routing |
|---|---|
| Tier 0: GPT-5.4 | OpenRouter / OpenAI API |
| Tier 1: Qwen3-32B/72B | vLLM serving (offline batch, quantized) |
| Tier 2: Qwen3-8B | vLLM serving (real-time, LoRA hot-swap) |

---

## 3. Skill-guided decision making

### Skill selection (RAG mode)

`select_skill_from_bank()` tries four paths in order:

1. **`SkillQueryEngine.select()`** — richest path (RAG relevance + applicability + structured guidance)
2. **`SkillQueryEngine.query_for_decision_agent()`** — convenience wrapper
3. **`SkillBankAgent.select_skill()`** — agent-level selection
4. **TF-IDF keyword fallback** — only when no query engine available

### Scoring

Each candidate skill is scored on three axes:

| Component | Weight | Source |
|-----------|--------|--------|
| Retrieval relevance | 40% | RAG embedding cosine similarity + keyword Jaccard |
| Execution applicability | 35% | Effect compatibility against current state predicates |
| Historical pass rate | 25% | Success rate from past executions |

### Protocol-aware lifecycle

The `_SkillTracker` manages:
- Protocol step tracking with `>>` marker at current step
- Duration caps per skill
- Stall detection (≥4 steps with reward ≤0)
- Success/abort criteria keyword matching in current state
- Automatic alternate skill selection when same skill re-selected after failure

---

## 4. Reward computation

**r_total = r_env + w_follow × r_follow + r_cost**

| Component | Source | Purpose |
|-----------|--------|---------|
| r_env | Raw environment reward | Task progress |
| r_follow | Skill contract eff_add predicate matching | Skill-following shaping |
| r_cost | Query/call costs + skill switch penalty | Discourage excessive tool use |

### r_follow details

Checks how many `eff_add` predicates from the active skill's contract appear in the observation:
- **+0.05** per newly satisfied predicate
- **+0.20** when all eff_add predicates satisfied
- **-0.01** per step with no predicate progress

### r_cost defaults

| Parameter | Default |
|-----------|---------|
| `query_mem_cost` | -0.05 |
| `query_skill_cost` | -0.05 |
| `call_skill_cost` | -0.02 |
| `skill_switch_cost` | -0.10 |

---

## 5. Lightweight inner MDP (typed local control)

We keep the inner MDP, but shrink it into a **lightweight typed control loop** — not a second planner. The online 8B actor benefits from structured local reasoning, but must not perform open-ended multi-step planning: short typed inner-hop reasoning is safe for 7B/8B, free-form hop selection is not.

**Philosophy.** The inner MDP is the actor's *guardrail*, not its brain. The real heavy reasoning brain stays in the offline 32B/72B synthesis/reflection tier (§2, §6).

### 5.1 What stays, what shrinks, what moves out

| Keep | Shrink | Move to offline 32B/72B |
|---|---|---|
| Structured state | Number of online hops | Failure diagnosis |
| Intention inference | Hop vocabulary size | Multi-step causal analysis |
| Skill retrieval / continuation | Freedom of `hop_select` | New skill invention |
| `_SkillTracker` lifecycle | Online reasoning depth | Cross-skill composition |
| Typed internal reasoning steps | Amount of actor-time reasoning | Cross-domain transfer mapping |
| Replayable internal traces |   | Reflective repair planning |

### 5.2 Three-layer actor decomposition

The actor is explicitly three layers with clean responsibilities; see also the per-step loop in §1.

**Layer A — Skill Continuation Gate** (mostly rule-heavy, lightweight scoring only).
Decides keep current skill / reselect / no-skill. Owns:
- no active skill → reselect
- max duration exceeded → reselect
- repeated no-progress / stall (≥4 steps, reward ≤0) → reselect
- success criterion matched → retire skill
- abort criterion matched → retire + blocklist for the step
- invalid context for active skill → forced reselect

If Layer A produces a confident action path (e.g. high-confidence state + high-confidence skill with populated slots), it bypasses Layer B entirely and goes straight to Layer C.

**Layer B — Lightweight inner MDP** (only when local uncertainty remains).
Resolves local uncertainty via short typed hops:
- do I need to verify a condition against current evidence? → CHECK
- do I need a reusable skill / protocol? → RETRIEVE (from the skill bank)
- is the state stale or insufficient? → GROUND (optional)
- can I commit this local sub-decision? → COMMIT

This is the *only* place the reduced hop vocabulary lives.

**Layer C — Final Action Head**.
Selects the concrete environment action from the valid list. Retains parsing, fallback chain, and anti-repetition safeguards. Always terminates the step.

### 5.3 Reduced hop vocabulary

The inner action space is a small, fixed, typed vocabulary — no free-form operations:

| Hop | Meaning | Typical target |
|---|---|---|
| `GROUND` *(optional)* | Refresh or refine the structured state | `entity_query`, `scene`, `region:e5` |
| `CHECK` | Verify a condition relevant to the current skill or subgoal | `can_continue_current_skill`, predicate name |
| `RETRIEVE` | Fetch a reusable skill / protocol from the skill bank | `skill_bank:<intention>`, `skill_bank:locate_filter_select` |
| `COMMIT` | Finalize the local sub-decision (internal state update) | subgoal id |
| `EXECUTE` | Choose and emit the environment action; exits the inner loop | env action |

`EXECUTE` is always the terminal hop of Layer B; everything else is non-terminal. `COMMIT` replaces the older, looser `CONCLUDE` token — the name makes clear it is a local finalization, not a free-form conclusion.

#### 5.3-bis Inner-hop ↔ `evidence_role` contract (enforced by the Harness)

Every skill in the bank declares an `evidence_role` ([PLAN-SKILL-BANK.md §0.3 Clause B](../03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills)). The Action Agent **may not** invoke a skill whose `evidence_role` does not match the inner hop under which it is being invoked; mismatches are raised as `contract-violation: skill-role-mismatch` by the Harness (see [PLAN-HARNESS.md §10 Gate G0](../05-harness/PLAN-HARNESS.md#10-promotion-gates)).

| Inner hop | Allowed `evidence_role` | Required episode fields at hop exit |
|-----|-----|-----|
| `GROUND` | `GATHER` | `evidence_out ≠ ∅` (new grounding records written to `<state>.evidence_refs`) |
| `RETRIEVE` | `GATHER` | `evidence_out ≠ ∅` (retrieved skill/protocol record treated as evidence-of-selection; empty retrieval ⇒ `INSUFFICIENT` and no COMMIT) |
| `CHECK` | `VERIFY` | `evidence_in ≠ ∅`; `verify_verdict ∈ {PASS, FAIL, INSUFFICIENT}` written back to `<state>` |
| `COMMIT` | `REASON` or `COMMIT` | `REASON` ⇒ `reason_warrant ⊆ evidence_in`, non-empty; `COMMIT` ⇒ `evidence_warrant ≠ ∅` |
| `EXECUTE` | `COMMIT` | `evidence_warrant ≠ ∅`; the environment action (or final answer for QA domains) must be paired with this warrant in `ActionRecord` |

This is the Action Agent side of the evidence-driven invariant: it guarantees that every hop in a trace leaves an evidence footprint — empty `evidence_in ∪ evidence_out` at the end of an inner episode is a contract violation, not merely a low-reward episode.

### 5.3a. Cross-domain semantics of inner actions

The inner-action alphabet is domain-general: the same five hops carry different surface forms across target domains while keeping one typed meaning. This is what makes inner traces transferable (see [Skill Bank §1.5](../03-skill-bank/PLAN-SKILL-BANK.md#15-cross-task-transfer-objective)).

| Hop | Cross-domain semantics | Game | Webagent | OS-agent | Video understanding | Visual reasoning |
|-----|------------------------|------|----------|----------|----------------------|-------------------|
| `GROUND` | Localize the relevant entity / region / control / frame / moment | unit / tile / legal-move object | UI control / DOM node | window / file / desktop object | clip frame / temporal moment | object / region / text span |
| `CHECK` | Verify a claim or constraint against grounded evidence | predicate on game state | attribute on DOM element | attribute on desktop object | predicate on frame contents | predicate on region contents |
| `RETRIEVE` | Fetch a reusable skill / protocol from the bank | skill over game entities | skill over UI elements | skill over desktop objects | skill over temporal evidence | skill over visual evidence |
| `COMMIT` | Lock an intermediate belief or answer candidate | subgoal / chosen move | chosen control | chosen window / object | chosen evidence frame | chosen answer candidate |
| `EXECUTE` | Emit a domain action — or, for QA domains, emit the final answer | env action | click / type / navigate | invoke / click / keystroke | emit answer with evidence chain | emit answer with region citation |

**Implication.** Online the actor stays domain-general at the *hop level*; domain specialization lives only in (a) the adapter that resolves `TARGET` and (b) Layer C's final action/answer head.

### 5.4 Hop depth policy (strict caps)

The actor does **not** perform long hop chains online.

| Regime | Depth | Typical trace |
|---|---|---|
| Default | **0–2 hops** | high-confidence state + skill → `EXECUTE` |
|   |   | clear skill, uncertain condition → `CHECK → EXECUTE` |
| Uncertain | **≤3 hops** | weak skill match → `RETRIEVE → COMMIT → EXECUTE` |
|   |   | stale / insufficient state → `GROUND → CHECK → EXECUTE` |
| Forbidden | — | open-ended or long online hop chains |

The cap is enforced hard (depth counter in `_SkillTracker`). When the cap is reached without `EXECUTE`, Layer C is invoked anyway with whatever evidence is available; this is deliberate — online time is not the place to reason further.

### 5.5 `hop_select` as a typed next-hop router

`hop_select` is **not** a mini planner and **not** a free-form reasoning generator. It is a constrained router that, at each inner step, predicts one typed operation and its target from a fixed schema:

```
NEXT_HOP = CHECK
TARGET   = can_continue_current_skill
```

or

```
NEXT_HOP = RETRIEVE
TARGET   = skill_bank:locate_filter_select
```

This keeps the inner MDP compatible with 8B and avoids the drift risk noted in §2 (borderline table). No free-form natural-language reasoning trace is emitted online; the online trace is short, typed, and targeted. Long-form natural-language reasoning is reserved for the offline tier.

### 5.6 Component boundaries

To keep the three typed heads from entangling:

| Head | Input | Output | Scope |
|---|---|---|---|
| `schema_gen` | observation | structured `<state>` | perception only |
| `skill_select` | structured state + intention | skill / no-skill | skill retrieval + scoring |
| `hop_select` | structured state + short typed trace | `(NEXT_HOP, TARGET)` from 5-token vocabulary | local uncertainty routing only |

Anything that doesn't fit cleanly in one of these boxes is a sign it belongs in Layer A (rules) or in the offline tier.

### 5.7 Re-observation between hops

- **Option A (default):** Hops operate on the same `<state>` schema, only updating an internal scratchpad. Cheaper, faster.
- **Option B (selective):** `GROUND` can trigger re-rendering or zooming into a region. More expensive but handles visual detail.
- **Recommendation:** A for games/web, B for visual QA where fine-grained visual grounding matters.

### 5.8 Reward for inner hops

- Schema consistency reward (does the hop produce a valid schema update?)
- Hop trace quality reward (offline judge rates the short typed trace, for GRPO)
- Progress shaping (reward for reducing uncertainty or resolving blockers)
- Cost penalty for exceeding the 0–2 hop default (pushes the policy toward shorter traces)
- **Recommendation:** GRPO with trajectory-level reward + explicit per-hop cost; aligns with the existing `r_follow` and `r_cost`.

### 5.9 LoRA adapter layout

| Adapter | Purpose |
|---|---|
| `schema_gen` | Screenshot → `<state>` schema (Qwen3-VL) |
| `skill_select` | Structured state + intention → skill / no-skill |
| `hop_select` | Structured state + short typed trace → `(NEXT_HOP, TARGET)` (constrained router) |
| `segment` | Trajectory → skill boundary detection |
| `contract` | Segment → effects contract |

### 5.10 Episode trajectory format

Each episode is a long-horizon trajectory of outer env steps, each containing 0–3 typed inner hops:

```
(schema_0, CHECK, schema_0') → (schema_0', EXECUTE(click), schema_1)
(schema_1, RETRIEVE, schema_1') → (schema_1', COMMIT, schema_1'') → (schema_1'', EXECUTE(move), schema_2)
(schema_2, EXECUTE(wait), schema_3)
```

This remains segmentable by the Skill Bank — short typed hop sub-chains become discoverable local skills — without inflating the online planning surface.

### 5.11 What is explicitly removed from the online actor

The following are *not* done by the online 8B actor and are the responsibility of the offline 32B/72B tier (see §2, §6):

- failure reflection / localization / diagnosis
- counterfactual analysis
- skill repair drafting
- cross-domain transfer mapping
- composition of new multi-skill procedures
- long-form natural-language reasoning traces

This separation is strict: keeping it intact is what makes the online loop trainable and stable.

---

## 6. Co-evolution & GRPO decomposition

### Asymmetric co-evolution

The three agents co-evolve, but not symmetrically. The core loop:

1. **Actor improves** using the current skill bank (GRPO on action execution, skill selection, hop selection).
2. **Skill bank improves** using trajectories from the current actor (operational updates to skills, protocols, pass rates).
3. **Synthesis-reflection agent proposes** bank updates from accumulated failures and successes (candidate skills, revised protocols, recovery patches).
4. **Verified updates** are fed back to the actor.

```
Actor rolls out with current bank
        ↓
Collect success/failure traces + skill-use statistics
        ↓
Frozen 32B/72B proposes diagnoses, new skills, compositions, protocol patches
        ↓
Verifier checks with replay, contract consistency, held-out contexts, non-regression
        ↓
Only accepted artifacts enter the bank or the training buffer
        ↓
Train smaller specialized heads/adapters on accepted artifacts (if narrow subtask becomes important)
        ↓
Repeat
```

### GRPO decomposition across agents

| Agent | GRPO? | Policy outputs | Reward signal |
|---|---|---|---|
| **Actor** (8B) | **Yes** — primary GRPO target | selected skill, next reasoning hop, action | r_env + r_follow + r_cost; bonuses for valid action formatting, not stalling, respecting active-skill protocol |
| **Skill-use / operational** (8B) | **Selective** — only for sequential bank-management decisions | continue/switch skill, accept/reject candidate segment as skill instance, merge/split/retire/keep, protocol revision choice from candidate set | Downstream actor improvement, skill reuse rate, contract satisfaction rate, reduced stall / fewer useless switches, bank compactness regularization |
| **Synthesis-reflection** (32B/72B) | **Not initially** — frozen inference-only | N/A (outputs are candidate proposals, not RL actions) | N/A initially; if adapted later, narrow task-specific rewards only |

**What NOT to GRPO on the skill-use side:** Simple retrieval, applicability scoring, pass-rate lookup, and `_SkillTracker` lifecycle logic are already algorithmic or scorer-based. GRPO adds value only for the sequential decision components (when to switch, accept, merge, split, refine).

### Timescale separation

All three agents must NOT co-evolve at full speed simultaneously — that creates instability where the actor chases a moving skill bank while the reflector changes supervision.

| Timescale | Agent | Update cadence |
|---|---|---|
| **Fast** | Actor | Every training iteration |
| **Medium** | Skill-bank operational | Every few training iterations |
| **Slow** | Synthesis-reflection | Proposals every N episodes, with acceptance gating |

**Concrete schedule:**
- 5–10 actor GRPO update cycles, then 1 offline skill-bank update cycle.
- Skill-bank refinement batch runs after the actor's update converges enough that traces are meaningful.
- Synthesis-reflection runs after failed episodes, periodically for effect chaining, when the failure-pattern store accumulates, when adding new domains, and before GRPO for cold-start trajectory generation.

### Training schedule

**Phase 0: Bootstrap**
- Use frontier model (GPT-5.4) or 32B/72B for seed traces and labels.
- Initialize a small skill bank.
- Train actor with GRPO on seed data.
- Keep synthesis-reflection frozen.

**Phase 1: Actor–skill bank co-evolution**
- Alternate: K rollout/update cycles for actor, then 1 offline skill-bank update cycle.
- Actor GRPO trains at least two LoRAs: `skill_select`, `action_execute`. Optionally a third for `hop_select` if committing to the inner MDP.
- Skill-bank GRPO trains: SEGMENT, CONTRACT, CURATOR LoRAs (see [Skill Bank §7](../03-skill-bank/PLAN-SKILL-BANK.md#7-grpo-co-evolution)).

**Phase 2: Gated synthesis-reflection**
- Every N failed episodes or every M training iterations: run reflector on recent failures.
- Propose revised protocols, new skills, composition candidates, recovery patches.
- Verify before admitting to bank.
- Only accepted artifacts enter the bank.

**Phase 3: Optional teacher adaptation**
- Only after the frozen synthesis-reflection agent becomes the real bottleneck.
- Train on narrow tasks only: failure localization, protocol revision, contract writing, candidate ranking/judging.
- Not broad end-to-end GRPO. SFT or preference-style adaptation first.
- See [Skill Crafter §2](../04-skill-crafter/PLAN-SKILL-CRAFTER.md#2-architecture) for the phased teacher adaptation policy.

### Frozen teacher improvement channels

The synthesis-reflection agent (32B/72B) can improve over time WITHOUT weight updates through five channels:

1. **Better input distribution** — as the 8B actor and skill bank improve, the synthesis agent sees cleaner trajectories, more reusable segments, better failure logs, and richer skill statistics.
2. **Better evidence organization, replay validation, and transfer routing** — the system is: frozen LLM + Crafter-private failure-pattern store + skill bank snapshots + replay slices + verification logs + proposal archive. As these artifact stores get better at *organizing evidence*, *validating replay*, and *routing transfer candidates*, the teacher improves without weight updates.
3. **Better inference procedure** — upgrade the multi-pass reasoning procedure: better decomposition, proposal-then-verify, best-of-N, counterfactual replay, stricter acceptance filters.
4. **Better verification and selection** — as the actor and bank mature, downstream usefulness can be measured more reliably. The teacher doesn't need to be smarter if the gatekeeping becomes smarter.
5. **Distillation into smaller specialized modules** — train smaller adapters or submodules on the outputs that pass verification: a small failure-localizer, a contract writer, a protocol patch ranker, a transfer-mapping scorer.

**Principle:** Actor improvement = mostly weight updates. Synthesis improvement = mostly system-level updates at first. Train the actor, not the judge, until there is evidence the judge is the limiting factor.

---

## 7. Integration with Visual Grounding

The action agent currently consumes text observations. The integration path:

1. **Phase 1 (current):** `get_state_summary()` compresses raw text observations. Single-call reasoning.
2. **Phase 2 (schema input):** `get_state_summary()` receives the `<state>` schema directly from the VLM grounding pipeline, which replaces text-based compression with structured grounding.
3. **Phase 3 (lightweight inner MDP):** Action agent implements the three-layer design in §5. The `<state>` schema becomes the inner MDP state. `GROUND` hops call vlm_wrapper tools. Entity references enable grounded actions ("click e5" instead of "click the red jacket"). Online hop depth is capped at 0–2 by default (≤3 under uncertainty).

### Schema as inner MDP state

The `<state>` schema is the state representation for the inner reasoning MDP:
- `<entities>` + `<relations>` — what the agent knows about the scene (updated by GROUND/CHECK hops)
- `<uncertainty>` — drives whether to gather more info (GROUND) or act (EXECUTE)
- `<targets>` (target, blocker, candidate_set) — narrows the inner action space
- `<state_flags>` (progress, error, dialog_open) — lifecycle decisions, including when to `COMMIT` or abort

---

## 8. Two pipeline variants

### Pipeline A — `qwen3_decision_agent.py` (full skill lifecycle)

- Skill bank required (per-game, query engine, tracker)
- `_SkillTracker` with reselect, alternate, protocol steps
- Fuzzy + edit distance + RAG embedding action parsing
- Anti-repetition guard
- Output: `test_rollout/decision_agent/<game>/<timestamp>/`

### Pipeline B — `run_qwen3_8b_eval.py` (lightweight evaluation)

- Skill bank optional (`--bank` flag)
- Single query per step, no tracking
- Exact match + `extract_action()` parsing
- Multi-benchmark (LMGame-Bench + AgentEvolver + Orak)
- Resume support
- Output: `output/<model>/<game>/<timestamp>/`

---

## 9. Supported environments

| # | Stack | Game | Registry Key |
|---|-------|------|-------------|
| 1 | LMGame-Bench | 2048 | `twenty_forty_eight` |
| 2 | LMGame-Bench | Candy Crush | `candy_crush` |
| 3 | LMGame-Bench | Tetris | `tetris` |
| 4 | AgentEvolver | Avalon | `avalon` |
| 5 | AgentEvolver | Diplomacy | `diplomacy` |
| 6 | Orak | Super Mario | `super_mario` |

---

## 10. Uncertainty-driven GROUND triggering

The inner MDP agent actively manages information completeness through the GROUND action.  The `<uncertainty>` section in the schema is the communication channel between grounding and reasoning.

### When to GROUND vs. act

The `hop_select` adapter learns this trade-off end-to-end via GRPO:

| Schema signal | Inner MDP response | Rationale |
|---------------|-------------------|-----------|
| `uncertainty.e5.label=high` + skill needs `$target` | `GROUND(e5)` → re-detect | Skill can't execute without confident target |
| `uncertainty.e3.pos=medium` + action is coarse | `EXECUTE(action)` directly | Position uncertainty tolerable for coarse actions |
| `blocker=null` + skill is `blocker_prerequisite_replan` | `GROUND(scene)` → find blockers | Skill explicitly requires blocker identification |
| `candidate_set=[]` + skill is `locate_filter_select` | `GROUND(candidates)` → populate set | Empty candidate set means grounding was incomplete |
| All slots populated, uncertainty low | Skip GROUND → `CHECK` or `EXECUTE` | Grounding was sufficient, proceed |

### Slot coverage check before skill execution

Before the `_SkillTracker` activates a skill, it checks that the skill's required slots (from `SlotBinding`) are populated in the current state.  Missing slots trigger a GROUND hop rather than skill failure:

```
SkillTracker.activate(skill, current_schema)
  → check slot_bindings against <targets> + <entities>
  → if $target missing: insert GROUND(target_query) as hop 0
  → if $blocker missing but skill needs it: insert GROUND(blocker_query) as hop 0
  → proceed with skill protocol from hop 1
```

This means **grounding doesn't need to be perfect** — it needs to be *good enough* for the reasoning layer to identify what's missing and fill it in.  The inner MDP reward naturally optimises this: unnecessary GROUND hops waste budget, but missing critical information causes task failure.

See [Visual Grounding §12](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md#12-schema-completeness-guarantee-grounding--reasoning-contract) for the full 4-layer guarantee. See also [Visual Skills](../01-visual-grounding/PLAN-VISUAL-SKILLS.md) for an optional extension where recurring multi-step grounding patterns (disambiguation, target recovery, evidence collection) are captured as transferable grounding skills that `hop_select` can invoke as reusable templates.

---

## 11. TODO

| Task | Priority | Status |
|------|----------|--------|
| Integrate VLM schema as primary state input | P0 | Not started |
| Implement three-layer actor (Skill Continuation Gate → Lightweight Inner MDP → Final Action Head) | P0 | Not started |
| Rule-heavy skill continuation gate (Layer A) wrapping `_SkillTracker` as pre-gate before inner MDP | P0 | Not started |
| `hop_select` as typed next-hop router with 5-token vocabulary {GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE} | P0 | Not started |
| Hard online hop-depth cap (0–2 default, ≤3 under uncertainty) with per-hop cost in reward | P0 | Not started |
| Implement three-agent role split (actor / skill-use / synthesis-reflection routing) | P0 | Not started |
| GRPO decomposition: actor LoRAs (skill_select, action_execute, hop_select-router) | P0 | Not started |
| Acceptance gate for synthesis-reflection outputs (contract check, replay, non-regression) | P0 | Not started |
| Entity-referenced actions (click e5 instead of click(400,510)) | P1 | Not started |
| Inner hop reward shaping (schema consistency + progress) | P1 | Not started |
| Slot coverage check in _SkillTracker before skill activation | P1 | Not started |
| Uncertainty-driven GROUND insertion (hop 0 when slots missing) | P1 | Not started |
| Timescale separation: implement fast/medium/slow update cadence | P1 | Not started |
| Skill-use GRPO: continue/switch, accept/reject, merge/split decisions | P1 | Not started |
| Extend to BrowserGym action space | P1 | Not started |
| Extend to OSWorld action space | P2 | Not started |
| Video-based decision making (temporal action selection) | P2 | Not started |
| Learned EXECUTE timing (when to stop reasoning and act) | P2 | Not started |
| Phase 3 teacher adaptation: narrow SFT/preference tuning if frozen teacher bottlenecks | P2 | Not started |

---

## 12. Implementation

| File | Purpose |
|------|---------|
| `decision_agents/agent.py` | `VLMDecisionAgent`, `run_tool()`, `run_episode_vlm_agent()` |
| `decision_agents/agent_helper.py` | `get_state_summary()`, `infer_intention()`, `select_skill_from_bank()` |
| `decision_agents/reward_func.py` | `RewardComputer`, `compute_reward()` |
| `decision_agents/dummy_agent.py` | Baseline agent for comparison |
| `scripts/qwen3_decision_agent.py` | Pipeline A (full skill lifecycle) |
| `inference/run_qwen3_8b_eval.py` | Pipeline B (lightweight evaluation) |
