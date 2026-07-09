# decision_agents

LLM decision-making agent that plays video games step-by-step. Supports **skill-bank retrieval** (RAG-based protocol-driven plans), **episodic memory**, **intention inference**, and **composite reward shaping**.

**Two model backends:**

- **GPT-5.4** (training-free) — used for cold-start data generation and labeling via OpenRouter / OpenAI API.
- **Qwen3-8B** (GRPO-trained) — served via vLLM for decision-agent inference and evaluation.

Both share the same code path; `API_func.ask_model` routes to the correct API based on the model name. Skill bank loading and querying (`load_skill_bank`, `select_skill_from_bank`, `skill_bank_to_text`) are identical for both backends.

## Supported games

**8 games** across three environment stacks (matching `cold_start/`):

| # | Stack | Game | Registry Key |
|---|-------|------|-------------|
| 1 | LMGame-Bench | **2048** | `twenty_forty_eight` |
| 2 | LMGame-Bench | **Sokoban** | `sokoban` |
| 3 | LMGame-Bench | **Candy Crush** | `candy_crush` |
| 4 | LMGame-Bench | **Tetris** | `tetris` |
| 5 | AgentEvolver | **Avalon** | `avalon` |
| 6 | AgentEvolver | **Diplomacy** | `diplomacy` |
| 7 | Orak | **Super Mario** | `super_mario` |
| 8 | Orak | **Pokemon Red** | `pokemon_red` |

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

### Pipeline B — `scripts/run_qwen3_8b_eval.py` (without skill select)

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
| `run_qwen3_sokoban_episode()` | Sokoban | `SokobanNLWrapper` with grid prompts, periodic reflection every N steps |
| `run_qwen3_avalon_episode()` | Avalon | Multi-agent (all players = Qwen3), `ThreadPoolExecutor` parallel queries |
| `run_qwen3_diplomacy_episode()` | Diplomacy | 7 powers, order parsing, SC delta tracking, 20-phase cap |
| `run_qwen3_orak_episode()` | Super Mario, Pokemon Red | Orak env wrappers |

**Key features:**

- Multi-benchmark (LMGame-Bench + AgentEvolver + Orak)
- Resume interrupted runs (`--resume`)
- Per-experience LLM summary generation
- Output: `output/<model_slug>/<game>/<timestamp>/`

**Usage:**

```bash
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
export VLLM_BASE_URL="http://localhost:8000/v1"

python -m scripts.run_qwen3_8b_eval --games twenty_forty_eight --episodes 3
python -m scripts.run_qwen3_8b_eval --episodes 10                   # all available games
python -m scripts.run_qwen3_8b_eval --resume                       # resume interrupted run
python -m scripts.run_qwen3_8b_eval --bank path/to/bank.jsonl      # with optional skill bank
python -m scripts.run_qwen3_8b_eval --list-games                   # show available games
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
| Game coverage | LMGame-Bench (+ Sokoban special) | LMGame-Bench + Avalon + Diplomacy + Orak |
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

Three scores per skill, combined into a final confidence:

**Retrieval relevance** (per-skill):
- 60% RAG embedding cosine similarity (query vs skill description embedding)
- 40% keyword Jaccard, further split: 35% `strategic_tokens` (name + description) + 35% `id_tokens` (skill_id) + 30% `effect_tokens` (eff_add/del/event)

Skill descriptions are built from: `skill_id + name + strategic_description + preconditions[:5] + protocol.steps[:3] + eff_add + eff_del + eff_event`.

**Execution applicability** (per-skill):
- Uses `_effects_compat_score()` against current state predicates
- For each `eff_add`: true at end → +1, contradicted → -1, missing → -0.5
- For each `eff_del`: false at end → +1, still true → -1, missing → -0.5
- Normalized to [-1, +1]
- Without state info, falls back to `pass_rate - 0.5`

**Combined confidence:**
- 40% retrieval relevance + 35% applicability (normalized to [0,1]) + 25% historical pass rate

Skills are sorted by confidence and top-k returned as `SkillSelectionResult` objects containing: `skill_id`, `skill_name`, `why_selected`, `relevance`, `applicability_score`, `confidence`, `expected_effects`, `preconditions`, `termination_hint`, `failure_modes`, `execution_hint`, `micro_plan`, `contract`, `pass_rate`.

---

## Files

| File | What it does |
|------|-------------|
| `agent.py` | `VLMDecisionAgent` (LLM decision agent), `run_tool()`, `run_episode_vlm_agent()`, tool handlers (e.g. `TOOL_SELECT_SKILL` → `active_skill_plan` from protocol steps) |
| `agent_helper.py` | `get_state_summary()`, `build_rag_summary()`, `extract_game_facts()`, `infer_intention()`, `EpisodicMemoryStore`, `skill_bank_to_text()`, `query_skill_bank()` / `select_skill_from_bank()`, `_get_protocol_for_skill()` |
| `reward_func.py` | `RewardConfig`, `RewardResult`, `RewardComputer`, `compute_reward()` (r_follow uses skill contract `eff_add`) |
| `dummy_agent.py` | Baseline `language_agent_action()` + game detection + action extraction for all 8 supported games (LMGame-Bench, AgentEvolver, Orak) |
| `__init__.py` | Re-exports the above |

---

## Quick start — run a full episode

`run_episode_vlm_agent()` returns an **`Episode`** object (from `data_structure.experience`) with fully-populated `Experience` objects per step.

```python
from decision_agents import VLMDecisionAgent, run_episode_vlm_agent, RewardConfig

episode = run_episode_vlm_agent(
    env,
    model="Qwen/Qwen3-8B",   # or "gpt-5.4" for training-free cold-start
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
    model="Qwen/Qwen3-8B",
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

agent = VLMDecisionAgent(model="Qwen/Qwen3-8B")
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
# → "game=sokoban | self=Player at (2,3) | objective=push box to goal | boxes=3 goals=3"
```

**Supported wrappers with `build_structured_state_summary()`:**

| Wrapper | Key fields | Example |
|---------|-----------|---------|
| GamingAgent (LMGame-Bench) | game, step, self, objective, critical, affordance | `game=sokoban \| self=Player at (2,3) \| objective=push box` |
| Avalon | game, phase, self, progress, critical, objective | `game=avalon \| phase=team_vote \| self=role:Percival(G)` |
| Diplomacy | game, phase, self, resources, critical, objective | `game=diplomacy \| phase=S1902M \| self=power:FRANCE centers:5` |
| Orak (Mario / Pokemon Red) | game, step, self, objective, critical, affordance | `game=super_mario \| self=pos:(120,80) \| objective=reach flag` |

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

Uses `extract_game_facts()` internally — game-specific parsers for Tetris (stack_h, holes, piece, next), 2048 (highest, empty, tiles, merges), Candy Crush (score, moves, pairs), Sokoban (boxes, goals, worker), Super Mario (x_pos, y_pos, coins, time), Avalon (phase, role, quest), and Diplomacy (phase, power, centers, units).

### `infer_intention(summary_or_observation, game=None, model=None, context=None)`

Returns a `[TAG] subgoal phrase` (≤15 words) describing the agent's current subgoal. Tags:

```
SETUP | CLEAR | MERGE | ATTACK | DEFEND | NAVIGATE | POSITION |
COLLECT | BUILD | SURVIVE | OPTIMIZE | EXPLORE | EXECUTE
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
    game="sokoban",
    model="Qwen/Qwen3-8B",    # or "gpt-5.4"
)
```

Supports all 8 games: 2048, Sokoban, Candy Crush, Tetris (LMGame-Bench), Avalon, Diplomacy (AgentEvolver), Super Mario, Pokemon Red (Orak).

---

## Per-step loop (LLMDecisionAgent protocol)

Every timestep the runner executes:

1. **`get_state_summary`** — required; runner computes it before action (returns `key=value` facts).
2. **(Optional)** **`select_skill`** — choose a skill when no active skill, skill exhausted, or agent is stuck. Returns full structured guidance (protocol steps, preconditions, termination hints, failure modes). Budget-limited to once every N steps unless stuck.
3. **`take_action`** — required; exactly one environment action. Agent has intention (from previous step), fresh state summary, and any active skill guidance in context.
4. **`get_intention`** — required; runner updates intention after observing action result (returns `[TAG] subgoal phrase`).
5. **`reward`** — required; compute `(r_env, r_follow, r_cost, r_total)` for logging/training.

### Format consistency

The agent prompt explicitly labels both formats so Qwen agents learn them:
- **Intention**: `"intention ([TAG] subgoal): [CLEAR] Reduce holes before stack overflows"`
- **State summary**: `"state_summary (key=value): game=tetris | phase=endgame | stack_h=15 | holes=42"`
- **Memory results**: returned as key=value summaries from `EpisodicMemoryStore`

This matches the labeling pipeline in `labeling/label_episodes_gpt54.py`, ensuring cold-start data and runtime data share identical formats for RAG retrieval and skill extraction.
