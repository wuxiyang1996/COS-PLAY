# skill_agents (GRPO Skill Bank Pipeline)

Build and maintain a **Skill Bank** from long-horizon game trajectories: segment trajectories into skills, learn symbolic contracts (effects), and serve queries for the [decision_agents](../decision_agents/README.md) VLM agent. The pipeline supports **GRPO-trained LoRA adapters** that wrap existing LLM call points: each call produces G samples, is scored with CPU-only rewards, and the best sample is returned so the EM pipeline runs unchanged while adapters improve over time.

**Model convention:** This project uses **Qwen3-8B** for all skill-bank components (vLLM serving, LoRA adapters, boundary/protocol/contract/curator calls). All configs and code references use Qwen3-8B. See [TODO_Lists/SKILLBANK_GRPO_PLAN.md](../TODO_Lists/SKILLBANK_GRPO_PLAN.md) for the full GRPO architecture.

**Model-agnostic design:** The pipeline and decision agent use the same skill-bank functions regardless of LLM backend. GPT and Qwen differ only in which API `ask_model` calls; set `PipelineConfig.llm_model` and/or `extractor_model` (e.g. `Qwen/Qwen3-8B` or `gpt-4o-mini`) for protocol synthesis and boundary proposal.

**Reasoning-model compatibility:** When using Qwen3 or other reasoning models that emit `<think>` blocks, all LLM call sites are wrapped via [`_llm_compat.py`](_llm_compat.py): prompts get `/no_think` appended and responses are stripped of think tags. See [Reasoning-model compatibility](#reasoning-model-compatibility) below.

**Transient vLLM / HTTP failures:** Contract and curator calls use [`_llm_retry.py`](_llm_retry.py) (`sync_ask_with_retry`) so bursts of `Connection error` or empty responses retry with backoff instead of immediately returning `None` (which GRPO logs as `0/4 non-empty` and reward `0.0`). Env: `SKILLBANK_LLM_RETRIES` (default 5), `SKILLBANK_LLM_RETRY_DELAY_S` (default 1.0), `VLLM_OPENAI_MAX_RETRIES` in [`API_func.py`](../API_func.py) `ask_vllm` (default 3).

**Flat GRPO rewards → no gradient:** If every completion in a group gets the same reward, plain normalization gives **advantage 0 for all** (``r - mean`` is zero). Co-evolution and [`GRPOLoRATrainer`](grpo/trainer.py) therefore use [`compute_grpo_group_advantages`](grpo/advantage_utils.py): when variance is ~0, a small **completion tiebreak** (zero-mean from completion text) is added so LoRA still gets a learning signal. Disable with `GRPO_TIEBREAK_SCALE=0`; tune magnitude with default `0.08`.

**GRPO training text vs parse:** Contract and curator returns are wrapped as [`SkillBankLLMOutput`](grpo/grpo_outputs.py) (a ``dict`` with the parsed JSON keys only). The **raw model string** lives in ``_grpo_raw_completion``; [`GRPOCallWrapper`](grpo/wrapper.py) stores that in ``completions`` for FSDP log-prob training (not ``str(dict)``). Segment uses ``PreferenceListWithRollouts.raw_rollouts`` joined with ``\\n---\\n``.

---

## Overview

| Component | Purpose |
|-----------|--------|
| **SkillBankAgent** | Full pipeline: ingest episodes → segment → learn contracts → maintain bank → query. |
| **SkillQueryEngine** | Rich retrieval + skill selection: keyword, effect-based, and state-aware search. |
| **SkillBankMVP** | Persistent storage (JSONL) for skill contracts and verification reports. Provides `compat_fn` for Stage 3 → Stage 2 feedback. |
| **NewPoolManager** | Rich tracking of `__NEW__` segments; clusters graduate via **proto-skill staging** (materialize → verify → promote). |
| **ProtoSkillManager** | Staging area for new skills: materialize creates proto-skills; verify + promote graduate them to the real bank. |
| **GRPO wrappers** | Wrap existing LLM calls (Stage 2 preferences, Stage 3 contract, Stage 4 filter): G samples, CPU-only reward, best returned to pipeline. |
| **tool_call_reward** | Reward for tool calls (query_skill / query_memory / call_skill) for agentic RL. |
| **LLM curator** | `bank_maintenance/llm_curator.py` — single-turn LLM filter (approve/veto/defer) for Stage 4 candidates. GRPO-wrapped via `enable_curator_grpo()`. |
| **LLM retrieval** | `skill_bank/llm_retrieval.py` — RETRIEVAL LoRA adapter for query rewriting + skill ranking in decision agents. |
| **Sub-episode quality** | `quality/sub_episode_evaluator.py` — Stage 4.5 quality scoring (outcome, follow-through, consistency, compactness). |
| **PLAN.md** | Operating plan (stages, constraints, data model). |

Subpackages implement each stage:

- **boundary_proposal** — Stage 1: high-recall candidate cut points + boundary plausibility scoring. Phase transitions (from `phase_detector`) are injected as boundary events. *Not GRPO-wrapped* (boundaries are algorithmic from predicates + signals).
- **infer_segmentation** — Stage 2: skill labelling with preference learning + contract feedback + **phase-aware intention-fit scoring**. Skills are compound labels (`"endgame:MERGE"`, `"opening:POSITION"`). **GRPO:** SEGMENT LoRA wraps `collect_segment_preferences()`; reward = `SegmentationDiagnostics`.
- **stage3_mvp** — Stage 3: effects-only contract learn/verify/refine. **GRPO:** CONTRACT LoRA wraps `llm_summarize_contract()`; reward = `verify_effects_contract().overall_pass_rate`.
- **quality** — Stage 4.5: sub-episode quality evaluation, drift detection, and protocol synthesis. Scores outcome_reward, follow_through, consistency, compactness on `SubEpisodeRef` metadata.
- **bank_maintenance** — Stage 4: propose candidates → **filter (CURATOR LoRA)** → execute. Refine/merge/split/materialize/promote; proto-skill lifecycle respected.
- **skill_evaluation** — Quality assessment (optional LLM judge).

---

## GRPO Skill Bank Pipeline

**Design: wrap, don't rebuild.** Existing LLM functions are wrapped with a generic GRPO sampler. Each call produces G samples, evaluates them with a downstream metric (CPU-only), stores (prompt, completions, rewards) in a buffer, and returns the best sample to the pipeline. The EM pipeline runs unchanged.

### Two-phase architecture

| Phase | When | What happens |
|-------|------|---------------|
| **Phase 1 — Rollout** | During EM | Pipeline calls LLM as normal → `GRPOCallWrapper` generates G samples (temperature=0.7) → reward computed per sample → data stored in `GRPOBuffer` → best sample returned to pipeline. |
| **Phase 2 — Training** | After EM step | `GRPOLoRATrainer` reads buffer → recomputes log_probs (gradients on) → group-normalized rewards → GRPO policy gradient → updates LoRA adapter → clears buffer. |

### Three GRPO targets (3 LoRA adapters on Qwen3-8B)

| Adapter | Stage | Wrapped function | Reward |
|---------|-------|------------------|--------|
| **CONTRACT** (P0) | 3 | `llm_summarize_contract()` in `stage3_mvp/llm_contract.py` | `verify_effects_contract().overall_pass_rate` on holdout |
| **CURATOR** (P1) | 4 | `filter_candidates()` in `bank_maintenance/llm_curator.py` — approve/veto/defer bank mutations | `bank_quality_delta` = q_filtered − q_all |
| **SEGMENT** (P1) | 2 | `collect_segment_preferences()` in `infer_segmentation/llm_teacher.py` | `SegmentationDiagnostics` (margin, confidence, compat) |

Stage 1 (boundary) and RETRIEVAL are **not** GRPO-wrapped: boundaries are algorithmic; skill selection is trained by the decision-agent GRPO trainer via episode return.

### Stage 4: propose → filter → execute

Stage 4 uses a **propose-filter-execute** flow:

1. **Propose:** Algorithm builds `SkillProfile` per skill and proposes candidate actions (refine, merge, split, materialize, promote) with triggers from `bank_maintenance/`.
2. **Filter:** One LLM call (`filter_candidates()`) with the CURATOR LoRA — returns approve/veto/defer per candidate. Conflict resolution and deferral expiry (e.g. auto-approve after 3 defers) apply.
3. **Execute:** Approved actions run via production `bank_maintenance/` (refine = weaken + strengthen; merge/split with alias map and local re-decode; materialize → proto-skill staging; promote after verification).

New skills enter through **proto-skill staging:** `__NEW__` clusters → materialize → proto-skill in staging → verify → promote to real bank. See [TODO_Lists/SKILLBANK_GRPO_PLAN.md](../TODO_Lists/SKILLBANK_GRPO_PLAN.md) for full lifecycle and guardrails.

### Infrastructure (not GRPO)

- `run_sub_episode_quality_check()` — Stage 4.5 data quality gate (outcome, follow-through, consistency, compactness scoring on `SubEpisodeRef` metadata).
- `distill_execution_hints()` — deterministic extraction from sub-episodes.
- `update_protocols()` — LLM synthesis (summarization, not GRPO).
- `run_until_stable()` / convergence, alias map application, proto-skill formation/verification logic.

---

## Quick start

### 1. Build a skill bank from raw episodes

```python
from data_structure.experience import Episode
from skill_agents import SkillBankAgent, PipelineConfig

# Config (optional; defaults work)
config = PipelineConfig(
    bank_path="data/skill_bank.jsonl",
    env_name="llm+overcooked",      # or "llm", "overcooked", etc.
    preference_store_path="data/preferences.json",
    report_dir="data/reports",
    min_instances_per_skill=5,
    # Contract feedback: Stage 3 → Stage 2 closed loop
    contract_feedback_mode="weak",   # "off" | "weak" | "strong"
    contract_feedback_strength=0.3,
    # NEW pool management
    new_pool_min_cluster_size=5,
    new_pool_min_consistency=0.5,
)

agent = SkillBankAgent(config=config)
agent.load()   # load existing bank if path exists

# Episodes: list of Episode (each has .experiences, .task)
episodes = [ep1, ep2, ep3]

# Ingest: Stage 1+2 (segment) + Stage 3 (learn contracts)
agent.ingest_episodes(episodes, env_name="llm+overcooked")

# Optional: iterate until stable (Stage 3 → Stage 4 → materialize NEW)
agent.run_until_stable(max_iterations=3)

agent.save()
```

### 2. Query skills (for decision_agents or scripts)

```python
# Natural-language key (scene / objective / entities)
results = agent.query_skill("navigate to pot and place onion", top_k=3)
for r in results:
    print(r["skill_id"], r["score"], r.get("micro_plan"))

# Rich skill selection (preferred for decision agents):
# separates retrieval relevance from execution applicability
results = agent.select_skill(
    query="navigate to pot",
    current_state={"near_pot": 0.1, "holding_onion": 0.9},
    top_k=3,
)
for r in results:
    print(r["skill_id"], r["relevance"], r["applicability"], r["confidence"])

# Effect-based: find skills that achieve desired state changes
results = agent.query_by_effects(
    desired_add={"at_pot", "holding_onion"},
    desired_del={"at_spawn"},
    top_k=3,
)

# List all skills
summary = agent.list_skills()

# Full detail for one skill
detail = agent.get_skill_detail("nav_to_pot")
```

### 3. Use with the decision agent

Pass the **SkillBankAgent** (or a plain **SkillBankMVP**) as the decision agent's skill bank. The decision agent calls `select_skill_from_bank` (via `run_tool(TOOL_SELECT_SKILL, ...)`); the helper uses the richest available API (SkillQueryEngine when present, else name match). The same path is used for both GPT and Qwen — only the `model` string changes, and `API_func.ask_model` routes to the correct backend.

```python
from decision_agents import VLMDecisionAgent, run_episode_vlm_agent, RewardConfig
from skill_agents import SkillBankAgent

# Build or load skill bank
skill_agent = SkillBankAgent(bank_path="data/skill_bank.jsonl")
skill_agent.load()

# Decision agent: pass any model name (GPT, Qwen, etc.); same code path
vlm_agent = VLMDecisionAgent(
    model="gpt-4o-mini",      # or "Qwen/Qwen3-8B", etc.
    skill_bank=skill_agent,   # or skill_agent.bank for plain bank
    reward_config=RewardConfig(w_follow=0.1),
)

result = run_episode_vlm_agent(env, agent=vlm_agent, max_steps=500, verbose=True)
```

### 4. Tool-call reward (agentic RL)

Use `compute_tool_call_reward` to get a reward signal for each tool call (query_skill, query_memory, call_skill). Combine with env reward and decision_agents' `reward` tool for RL training.

```python
from skill_agents import compute_tool_call_reward, ToolCallRewardConfig, SkillBankMVP

bank = SkillBankMVP("data/skill_bank.jsonl")
bank.load()

# After the decision agent calls query_skill and you have the outcome:
reward_result = compute_tool_call_reward(
    tool_name="query_skill",
    tool_args={"key": "navigate to pot and place onion"},
    context_observation="chef near pot, holding onion",
    outcome_observation="onion in pot, soup cooking",
    skill_bank=bank,
    retrieved_skill_id="nav_to_pot",
    retrieved_result={"skill_id": "nav_to_pot", "score": 0.85},
    config=ToolCallRewardConfig(w_relevance=1.0, w_utility=1.0),
)
print(reward_result.r_total)   # for RL loss / value target
print(reward_result.to_dict()) # r_relevance, r_utility, details
```

For a full episode of tool calls, use `compute_episode_tool_call_returns(tool_call_trajectory, skill_bank=bank)` to get a list of per-step rewards.

### How tool-call reward is computed

**r_total = w_relevance × r_relevance + w_utility × r_utility** (for agentic RL: reward for "was this tool call good?").

- **r_relevance**
  - **query_skill**: If you have a retrieval **score** (e.g. from `SkillQueryEngine.query(key)` or `retrieved_result["score"]`), **r_relevance = score × relevance_scale** (default scale 0.5). If no score is passed but `skill_bank` and query `key` are provided, the code builds a `SkillQueryEngine`, runs `query(key, top_k=1)`, and uses the top result's score. Otherwise **r_relevance = default_query_reward** (0).
  - **query_memory**: If you pass **retrieval_quality** in [0,1], **r_relevance = retrieval_quality × relevance_scale**; else **r_relevance = default_memory_reward** (0).

- **r_utility** (for **query_skill** and **call_skill**)
  - The skill's **eff_add** set is taken from the contract (or from the bank using `retrieved_skill_id`). Each predicate is **satisfied** if all of its tokens (length ≥ 2) appear in the **outcome_observation** text (case-insensitive). Then **r_utility = (# satisfied) × utility_per_predicate** (default 0.1), plus **utility_full_completion** (default 0.3) when all **eff_add** predicates are satisfied. No state across steps—purely outcome-based.

Predicate satisfaction is implemented by tokenizing the predicate and the outcome string and checking containment (see `_predicates_satisfied_in_text` in `tool_call_reward.py`). This reward is separate from the decision agent's **r_follow** (which is stateful and uses the current observation); use both for RL (e.g. env + decision reward + tool-call reward).

---

## Main APIs

### SkillBankAgent

| Method | Description |
|--------|-------------|
| `load()` | Load bank and optional preference store from disk. |
| `save()` | Persist bank, preferences, iteration history. |
| `segment_episode(episode, env_name=..., skill_names=...)` | Run Stage 1+2 on one episode; returns `(SegmentationResult, list[SubTask_Experience])`. Accumulates segment records. |
| `ingest_episodes(episodes, ...)` | Segment each episode then run Stage 3 (contract learning). Returns list of `(result, sub_episodes)` per episode. |
| `run_contract_learning()` | Stage 3: learn/verify/refine effects contracts from accumulated segments. |
| `update_protocols()` | LLM protocol synthesis for skills that have contracts but no protocol yet. Returns count of updated skills. |
| `distill_execution_hints(min_successful=3)` | Deterministic extraction of execution hints from successful sub-episodes. Returns count of updated skills. |
| `run_sub_episode_quality_check()` | Stage 4.5: score sub-episodes on outcome, follow-through, consistency, compactness. Returns list of quality reports. |
| `run_bank_maintenance(...)` | Stage 4 core: split / merge / refine skills. Returns `BankMaintenanceResult`. |
| `form_proto_skills()` | Stage 4 materialize: cluster `__NEW__` segments → create proto-skills in staging area. |
| `verify_proto_skills()` | Stage 4 verify: run contract verification on unverified proto-skills. |
| `promote_proto_skills()` | Stage 4 promote: graduate verified proto-skills to real bank skills. |
| `materialize_new_skills()` | Stage 4 materialize (legacy): promote `__NEW__` via `NewPoolManager` clustering + consistency checks. |
| `run_evaluation(episode_outcomes=...)` | Run skill quality evaluation (optional). |
| `run_full_iteration(episodes=...)` | One pass: (optional ingest) → Stage 3 → Stage 4 (split/merge/refine/materialize/promote) → snapshot. |
| `run_until_stable(max_iterations=...)` | Iterate until convergence; then `save()`. |
| `query_skill(key, top_k=3)` | Keyword-style retrieval; returns list of `{skill_id, score, contract, micro_plan}`. |
| `select_skill(query, current_state=..., top_k=3)` | **Rich skill selection**: relevance + applicability + confidence. Preferred for decision agents. |
| `query_by_effects(desired_add=..., desired_del=..., top_k=3)` | Effect-based retrieval. |
| `list_skills()` | Compact list of all skills. |
| `get_skill_detail(skill_id)` | Full contract + report for one skill. |
| `add_skill(skill_id, eff_add, eff_del, eff_event)` | Manually add a skill. |
| `remove_skill(skill_id)` | Remove a skill. |
| `update_skill(skill_id, ...)` | Update contract fields and bump version. |
| `skill_ids` | *Property.* List of all skill IDs in the bank. |
| `get_contract(skill_id)` | Get the `SkillEffectsContract` for a specific skill. |
| `get_bank()` | Get the underlying `SkillBankMVP` instance. |
| `segments` | *Property.* All accumulated `SegmentRecord` objects. |
| `iteration_history` | *Property.* List of `IterationSnapshot` objects from past iterations. |

### SkillQueryEngine

Use when you already have a **SkillBankMVP** and want retrieval without the full pipeline.
The engine auto-loads a RAG `TextEmbedder` (Qwen3-Embedding-0.6B) for
cosine-similarity scoring, blended with keyword Jaccard:

```python
from skill_agents import SkillQueryEngine
from skill_agents.skill_bank.bank import SkillBankMVP

bank = SkillBankMVP("bank.jsonl")
bank.load()

engine = SkillQueryEngine(bank)            # auto-loads RAG embedder

# Simple retrieval (backward compatible)
results = engine.query("place onion in pot", top_k=3)
detail = engine.get_detail("place_onion")
list_all = engine.list_all()

# Rich skill selection (preferred for decision agents):
# separates retrieval relevance from execution applicability
selections = engine.select(
    query="place onion in pot",
    current_state={"near_pot": 0.8, "holding_onion": 0.9},
    top_k=3,
)
for s in selections:
    print(s.skill_id, s.relevance, s.applicability, s.confidence)
    print("  matched:", s.matched_effects, "missing:", s.missing_effects)

# Format expected by decision_agents run_tool(QUERY_SKILL)
decision_result = engine.query_for_decision_agent(
    "place onion",
    current_state={"near_pot": 0.8},  # optional: enables applicability scoring
    top_k=1,
)
# → {"skill_id": "...", "relevance": 0.8, "applicability": 0.6, "confidence": 0.7, ...}
```

### PipelineConfig

All fields (see `pipeline.PipelineConfig` dataclass). **Model convention:** use `Qwen/Qwen3-8B` for GRPO and co-evolution.

| Field | Default | Meaning |
|-------|--------|--------|
| | | **Paths** |
| `bank_path` | `None` | JSONL path for the skill bank. |
| `preference_store_path` | `None` | JSON path for the preference store (Stage 2). |
| `report_dir` | `None` | Directory for evaluation reports. |
| | | **Stage 1: boundary proposal** |
| `env_name` | `"llm"` | Signal extraction: `"llm"`, `"llm+overcooked"`, `"overcooked"`, etc. |
| `game_name` | `"generic"` | Actual game identifier for phase detection (e.g. `"twenty_forty_eight"`, `"super_mario"`). |
| `merge_radius` | `3` | Merge boundary candidates within this many steps. |
| `extractor_model` | `None` | LLM for boundary proposal (e.g. `Qwen/Qwen3-8B`, `gpt-4o-mini`). |
| | | **Stage 2: segmentation** |
| `segmentation_method` | `"dp"` | Decoder method: `"dp"` (Viterbi) or `"beam"`. |
| `preference_iterations` | `3` | Active-learning rounds. |
| `margin_threshold` | `1.0` | Segments with margin below this get preference queries. |
| `max_queries_per_iter` | `5` | Max pairwise queries per active-learning iteration. |
| `new_skill_penalty` | `2.0` | Penalty for proposing a new skill label. |
| `consistency_penalty` | `0.3` | Tag-change penalty in boundary scoring. |
| `min_segment_length` | `3` | Min segment length (steps). |
| `min_skill_length` | `2` | Min skill length (steps). |
| `boundary_score_threshold` | `0.3` | Boundary plausibility score cutoff. |
| `contract_feedback_mode` | `"off"` | Stage 3 → Stage 2 contract feedback: `"off"`, `"weak"`, `"strong"`. |
| `contract_feedback_strength` | `0.3` | Weight of contract compatibility term in Stage 2 scoring. |
| | | **Stage 3: contract learning** |
| `eff_freq` | `0.8` | Min frequency for a literal to be in a contract. |
| `min_instances_per_skill` | `5` | Skip skills with fewer instances. |
| `start_end_window` | `5` | Steps from segment start/end for predicate extraction. |
| | | **NEW pool management** |
| `new_pool_min_cluster_size` | `5` | Min cluster size for NEW → materialize (proto-skill staging). |
| `new_pool_min_consistency` | `0.5` | Min effect pattern consistency for materialize. |
| `new_pool_min_distinctiveness` | `0.25` | Min Jaccard distance from existing skills. |
| | | **Stage 4: bank maintenance** |
| `split_pass_rate_threshold` | `0.7` | Pass rate below this triggers a split check. |
| `child_pass_rate_threshold` | `0.8` | Required pass rate for split children. |
| `merge_jaccard_threshold` | `0.85` | Effect Jaccard similarity above this triggers merge. |
| `merge_embedding_threshold` | `0.90` | Embedding cosine similarity above this triggers merge. |
| `min_child_size` | `3` | Min instances per child after split. |
| `min_new_cluster_size` | `5` | Min cluster size for Stage 4 materialize. |
| | | **Convergence** |
| `max_iterations` | `5` | Cap for `run_until_stable()`. |
| `convergence_margin_std` | `0.5` | Margin std below this → converged. |
| `convergence_new_rate` | `0.05` | NEW rate below this → converged. |
| | | **LLM** |
| `llm_model` | `None` | LLM for protocol synthesis and pipeline LLM calls. |
| `max_concurrent_llm_calls` | `None` | Cap for local GPU concurrency (e.g. `1`). |

**GRPO:** Per-stage hyperparameters (group_size, clip_ratio, kl_coeff, lr) live in `trainer/common/configs/skillbank_grpo.yaml`; see [SKILLBANK_GRPO_PLAN.md](../TODO_Lists/SKILLBANK_GRPO_PLAN.md).

---

## Data flow

1. **Episode** (from env rollouts or demos) has `experiences` and `task`. Use `Episode` from `data_structure.experience`.
2. **Phase detection** (`phase_detector`): per-step game-phase labels are computed from state features (board occupancy, position progress, etc.). Combined with intention tags to create **compound skill labels** (`"endgame:MERGE"`, `"early_level:NAVIGATE"`). See [Phase detection](#phase-detection-preprocessor) below.
3. **Stage 1** (boundary_proposal): extract signals → propose candidate cut points **C**. Phase transitions are injected as boundary events. *Not GRPO-wrapped.*
4. **Stage 2** (infer_segmentation): decode over **C** with preference-learned scorer → segments + compound skill labels (including `__NEW__`). The `intention_fit` term matches compound labels. Contract feedback uses `bank.compat_fn`. **GRPO:** SEGMENT LoRA wraps `collect_segment_preferences()`; reward from `SegmentationDiagnostics` after scorer rebuild + decode. Entry points: `infer_and_segment()` for online pipeline, `infer_and_segment_offline()` for batch re-segmentation, `grpo_scorer_factory()` for GRPO reward evaluation.
5. **Stage 3** (stage3_mvp): for each non-NEW skill, learn effects contract, verify, refine; persist to bank. **GRPO:** CONTRACT LoRA wraps `llm_summarize_contract()`; reward = `verify_effects_contract().overall_pass_rate`. Contracts feed back into Stage 2 via `compat_fn`.
6. **Stage 4.5** (quality): `run_sub_episode_quality_check()` scores sub-episodes on outcome_reward, follow_through, consistency, compactness. Low-quality segments can be dropped before bank maintenance.
7. **Stage 4** (bank_maintenance): **propose** candidates (refine/merge/split/materialize/promote) from `SkillProfile` + triggers → **filter** via `filter_candidates()` in `llm_curator.py` (CURATOR LoRA: approve/veto/defer) → **execute** approved actions. Refine = weaken + strengthen; merge/split use alias map and local re-decode. **GRPO:** CURATOR LoRA reward = `bank_quality_delta`.
8. **Proto-skill staging:** `__NEW__` → `NewPoolManager` clusters → materialize → proto-skill in staging → verify → promote to real bank. Proto-skills participate in Stage 2 decoding before promotion.
9. **Post-processing:** `distill_execution_hints()` extracts hints from successful sub-episodes; `update_protocols()` synthesizes structured execution protocols via LLM.
10. **Query / Select:** decision agent uses `select_skill(query, current_state)` or `query_skill(key)` / `query_by_effects(...)`.

### Stage 2 scoring formula

The `SegmentScorer` combines six terms to score each candidate segment `(i, j)` with skill `k`:

```
Score(i, j, k | k_prev) =
    w_bf  * behavior_fit(obs, act, k)       [LLM preferences — Bradley-Terry]
  + w_if  * intention_fit(k, i, j)          [per-step tag agreement]
  + w_dp  * duration_prior(j-i+1, k)       [Gaussian]
  + w_tp  * transition_prior(k, k_prev)     [LLM preferences — Bradley-Terry]
  + w_cc  * contract_compat(k, P_i, P_j)   [Stage 3 feedback]
  + w_bp  * boundary_preference(i, j)
```

| Term | Weight | Source | Purpose |
|------|--------|--------|---------|
| `behavior_fit` | 1.0 | LLM teacher rankings → Bradley-Terry | Does the segment's behavior match this skill? |
| `intention_fit` | **2.0** | Per-step phase:tag compound labels | Fraction of steps in `[i,j]` whose compound label matches skill `k`, scaled by segment length. Prevents label collapse and distinguishes same-tag skills across game phases. |
| `duration_prior` | 0.3 | Gaussian (configurable) | Is the segment length reasonable for this skill? |
| `transition_prior` | 1.0 | LLM teacher rankings → Bradley-Terry | Is `k` a natural successor to `k_prev`? |
| `contract_compat` | 0.0 (off) | Stage 3 effects contracts | Do the state changes match the skill's learned contract? |
| `boundary_preference` | 0.5 | Boundary plausibility scorer | Is this a good place to cut? |

The `intention_fit` term uses **compound labels** produced by the [phase detection preprocessor](#phase-detection-preprocessor). Each step gets a label like `"endgame:MERGE"` (combining game phase + intention tag), so that the same tactical intent in different game phases becomes a distinct skill. When no tags are available, `intention_fit` returns 0 and the scorer degrades gracefully to LLM-only mode. Partial matching is supported: a raw tag skill like `"MERGE"` still matches the tag portion of compound labels.

Vocabulary merging ensures the decoder always sees all compound labels alongside existing bank skills: `_seed_skills_from_intentions()` runs phase detection and extracts unique compound labels from each episode, merging them into the skill vocabulary before decoding. The preference store re-collects LLM rankings whenever new skills appear (`unseen_skills` check).

**GRPO co-evolution:** Each EM step runs Phase 1 (wrappers generate G samples, store in buffer, return best); then Phase 2 (`GRPOLoRATrainer` one step per adapter, clear buffer). Protocol synthesis (`update_protocols()`) remains plain LLM inference when `llm_model` / `extractor_model` is set.

---

## Phase detection preprocessor

Per-step intention tags (`[MERGE]`, `[NAVIGATE]`) capture **tactical** intent but not **strategic** context. The same tactic in different game phases represents a different skill — e.g. `MERGE` on an empty 2048 board (opening) vs. a nearly-full board (endgame) require different strategies.

The **phase detector** (`infer_segmentation/phase_detector.py`) adds a phase label to each timestep and combines it with the intention tag to create **compound skill labels**:

```
raw tag:      MERGE          MERGE          MERGE
phase:        opening        midgame        endgame
compound:     opening:MERGE  midgame:MERGE  endgame:MERGE   ← 3 distinct skills
```

### Game-specific extractors

Each game has a dedicated extractor that parses structured state features:

| Game | State Feature | Phases |
|------|--------------|--------|
| **2048** | Board occupancy + highest tile | `opening`, `midgame`, `endgame` |
| **Tetris** | Board fill ratio | `opening`, `midgame`, `endgame` |
| **Super Mario** | Mario x-position (level progress) | `early_level`, `mid_level`, `late_level` |
| **Sokoban** | Boxes on goal positions | `setup`, `solving`, `finishing` |
| **Pokemon Red** | State text keywords (battle/route/menu) | `battle`, `exploration`, `overworld`, `menu` |
| **Avalon** | Round signals (vote/quest/assassin) | `team_building`, `quest`, `endgame` |
| **Diplomacy** | Turn/season signals | `opening`, `orders`, `retreat`, `adjustment` |
| **Candy Crush** | Temporal position (no strong state signal) | `early`, `mid`, `late` |

### Generic fallback

If a game-specific extractor returns only 1 unique phase (uninformative), the detector falls back to **temporal-third detection**: it splits the episode into thirds and checks whether the tag distributions differ meaningfully across them. If they do, it labels steps as `early`/`mid`/`late`; otherwise it returns `"mid"` everywhere (which `make_compound_label` strips, preserving raw tags).

### Integration points

The phase detector feeds into three places:

1. **Skill seeding** (`pipeline.py → _seed_skills_from_intentions`): seeds compound labels (`"endgame:MERGE"`) instead of raw tags, giving the decoder a richer vocabulary.
2. **Intention-fit scoring** (`episode_adapter.py → _build_intention_fit_fn`): the scorer matches compound labels against skills, so `endgame:MERGE` scores high in late-episode segments and 0 in early ones.
3. **Boundary proposal** (`boundary_proposal/episode_adapter.py → propose_from_episode`): phase transition timesteps are injected as boundary events, ensuring the decoder has cut points where game phases change.

The `game_name` field in `PipelineConfig` controls which game-specific extractor is used (separate from `env_name` which controls signal extraction strategy).

### Impact

| Game | Before (raw tags) | After (compound labels) |
|------|-------------------|------------------------|
| 2048 | 1 skill (`MERGE`) | 9 skills (`opening:MERGE`, `midgame:MERGE`, `endgame:MERGE`, `endgame:SURVIVE`, ...) |
| Super Mario | 1 skill (`NAVIGATE`) | 4 skills (`early:NAVIGATE`, `NAVIGATE`, `CLEAR`, `late:CLEAR`) |
| Candy Crush | 1 skill | 2 skills |
| Sokoban | 2 skills | 3 skills |
| Pokemon Red | 2 skills | 3 skills |
| Tetris | 4 skills | 4 skills (maintained) |

---

## Reasoning-model compatibility

Reasoning models (e.g. **Qwen3-8B**, QwQ) default to an internal "thinking" mode that emits `<think>…</think>` blocks before the actual answer. Those blocks consume the `max_tokens` budget and often leave little or no room for the structured output (JSON, rankings, protocols) the pipeline needs.

The module [`skill_agents/_llm_compat.py`](_llm_compat.py) provides:

| Function | Purpose |
|----------|--------|
| `strip_think_tags(text)` | Remove `<think>` blocks from LLM output. |
| `is_reasoning_model(model_name)` | Detect Qwen3 / QwQ-style model names. |
| `wrap_ask_for_reasoning_models(ask_fn, model_hint=...)` | Wrap any `ask_model`-style callable: appends `/no_think` to prompts for reasoning models and strips think tags from responses. |

All LLM entry points in skill_agents use this wrapper:

- **infer_segmentation/llm_teacher.py** — segment/transition rankings, skill naming
- **pipeline.py** — protocol synthesis (`_llm_synthesize_protocol`)
- **boundary_proposal/llm_extractor.py** — predicate extraction, boundary significance
- **skill_bank/llm_retrieval.py** — RETRIEVAL adapter
- **stage3_mvp/llm_contract.py** — CONTRACT adapter
- **bank_maintenance/llm_curator.py** — CURATOR adapter
- **skill_evaluation/evaluators.py** — LLM judge

Protocol and naming prompts have also been tightened (game-AI expert roles, concrete steps, tag-specific execution-hint failure modes) so that with the full token budget, outputs are concrete and game-specific rather than generic.

---

## Running skill extraction (with or without GRPO)

The extraction script `extract_skillbank_grpo_gpt54.py` supports three modes. A shell wrapper `extract_skillbank/run_extract_skillbank_grpo.sh` sets up `PYTHONPATH` and API keys automatically.

### Mode 1: API-based LLM teacher (default — GPT-5.4)

```bash
export OPENROUTER_API_KEY="sk-or-..."
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

# All games
python -m skill_agents_grpo.extract_skillbank.extract_skillbank_grpo_gpt54

# Quick test: one episode per game
python -m skill_agents_grpo.extract_skillbank.extract_skillbank_grpo_gpt54 --one_per_game -v

# Specific games
python -m skill_agents_grpo.extract_skillbank.extract_skillbank_grpo_gpt54 --games twenty_forty_eight tetris

# Via shell wrapper (handles PYTHONPATH + API key check)
bash extract_skillbank/run_extract_skillbank_grpo.sh --games tetris
bash extract_skillbank/run_extract_skillbank_grpo.sh --one_per_game -v
bash extract_skillbank/run_extract_skillbank_grpo.sh --dry_run
```

### Mode 2: Local model as LLM teacher (Qwen3-8B, no GRPO)

Uses Qwen3-8B (or any HuggingFace model) for all LLM teacher calls instead of the API. The model is loaded once and registered as the shared instance; all LLM call sites auto-discover it.

```bash
python -m skill_agents_grpo.extract_skillbank.extract_skillbank_grpo_gpt54 \
    --local_model Qwen/Qwen3-8B \
    --games twenty_forty_eight --one_per_game -v
```

### Mode 3: Local model + GRPO training

Enables the full GRPO loop: for each LLM teacher call, G candidate ranking sets are sampled at elevated temperature, each is evaluated by rebuilding the scorer and running the Viterbi decoder, the best is returned to the pipeline, and all samples are stored for LoRA fine-tuning. After each game, a GRPO training step updates the LoRA adapters.

```bash
python -m skill_agents_grpo.extract_skillbank.extract_skillbank_grpo_gpt54 \
    --local_model Qwen/Qwen3-8B \
    --use_grpo \
    --grpo_group_size 4 \
    --adapter_dir output/lora_adapters \
    --games twenty_forty_eight -v
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | `labeling/output/gpt54` | Directory with labeled game sub-folders. |
| `--output_dir` | `extract_skillbank/output/gpt54_skillbank_grpo` | Output directory. |
| `--games` | all | Only process these games (space-separated). |
| `--model` | `gpt-5.4` | LLM model for skill naming/description. |
| `--max_episodes` | all | Max episodes per game. |
| `--one_per_game` | off | Process only the first episode per game. |
| `--resegment` | off | Re-run pipeline against seeded bank (second pass, doubles LLM cost). |
| `--skip_archetypes` | off | Skip cross-game archetype aggregation. |
| `--save_annotated` | off | Save annotated episodes (with skills field) to output dir. |
| `--dry_run` | off | Preview what would be processed without running extraction. |
| `--resume` | off | Resume from last checkpoint (skip completed games/episodes). |
| `-v` / `--verbose` | off | Print per-step details. |
| `--local_model` | None | HuggingFace model id or path (e.g. `Qwen/Qwen3-8B`). |
| `--use_grpo` | off | Enable GRPO sampling + LoRA training. |
| `--grpo_group_size` | 4 | Number of samples per LLM call (G). |
| `--adapter_dir` | `<output>/lora_adapters` | Load/save LoRA adapters here. |
| `--grpo_train_every` | 0 | Train every N episodes (0 = once per game). |

The GRPO loop per game:
1. `GRPOOrchestrator.enable_wrappers()` — monkey-patches `collect_segment_preferences`, `llm_summarize_contract`, `filter_candidates`
2. Run `extract_skills_for_game()` — pipeline calls go through GRPO wrappers
3. `GRPOOrchestrator.train_step()` — reads buffer, computes advantages, PPO-clip loss on LoRA params
4. `GRPOOrchestrator.disable_wrappers()` — restores original functions
5. Save updated LoRA adapters to `--adapter_dir`

Trained adapters persist across runs: pass the same `--adapter_dir` on the next invocation to resume training from where you left off.

---

## Training integration (GRPO co-evolution)

During co-evolution training, the skill bank pipeline runs with **GRPO wrappers** active. The **SkillBankCoEvolutionCallback** (in `trainer/decision/coevolution_callback.py`) drives the loop; the GRPO pipeline lives under `trainer/skillbank/grpo/`.

### Co-evolution loop (wrapper architecture)

1. **Decision agent GRPO** (existing): collect rollouts, update full policy (including skill selection).
2. At bank-update cadence: trajectories are ingested and the **EM pipeline runs with GRPO wrappers**:
   - **Stage 2:** `collect_segment_preferences()` → SEGMENT wrapper (G samples, `SegmentationDiagnostics` reward, buffer, best returned).
   - **Stage 3:** `llm_summarize_contract()` → CONTRACT wrapper (G samples, `verify_effects_contract` reward, buffer, best returned).
   - **Stage 4:** `propose_candidates()` → `filter_candidates()` (CURATOR wrapper) → `execute_approved()`. Filter: G samples, `bank_quality_delta` reward, buffer, best returned.
3. Infrastructure steps (no GRPO): `run_sub_episode_quality_check()`, `distill_execution_hints()`, `update_protocols()`.
4. **Phase 2:** `GRPOLoRATrainer.train_step(grpo_buffer)` — one gradient step per adapter; buffer cleared.
5. SkillEval gating: if bank quality drops, rollback; otherwise bank is hot-swapped into workers.

Shared infrastructure: `GRPOCallWrapper`, `GRPOBuffer`, `GRPOLoRATrainer`; `MultiLoraSkillBankLLM.log_probs()` for training. Config: `trainer/common/configs/skillbank_grpo.yaml` and `skillbank_em.yaml`.

### Segmentation persistence

The `SegmentationStore` keeps per-trajectory segmentations updated during training. When EM re-segments with a newer bank, the store replaces the old entry so replay/evaluation/visualization see the latest segmentation.

### Tool-call reward in training

`tool_call_reward` is integrated via `TrainRewardShaper` (in `trainer/decision/reward_shaping.py`). For QUERY_SKILL, QUERY_MEM, CALL_SKILL the shaper adds `r_tool`:

`r_total = r_env + w_follow × r_follow + r_cost + tool_call_reward_weight × r_tool`

See [trainer/README.md](../trainer/README.md) for co-evolution setup and [TODO_Lists/SKILLBANK_GRPO_PLAN.md](../TODO_Lists/SKILLBANK_GRPO_PLAN.md) for the full GRPO plan (rewards, Stage 4 propose-filter-execute, proto-skill lifecycle, guardrails).

---

## Cold-start I/O recording for Qwen+GRPO

Every LLM-calling function that will be replaced or augmented by Qwen+GRPO records its full prompt/response. These records serve as:

1. **Supervised fine-tuning data** for Qwen3-8B cold-start (before any GRPO training).
2. **Reference outputs** for GRPO reward comparison (teacher vs. student).

### Two recording systems

| System | File | Source modules | When it fires |
|--------|------|---------------|---------------|
| **Teacher I/O** | `teacher_io_coldstart.jsonl` | `infer_segmentation/llm_teacher.py` | Always (API fallback) |
| **Cold-start I/O** | `coldstart_io_all.jsonl` | `boundary_proposal`, `stage3_contract`, `bank_curator`, `skill_retrieval`, `pipeline` | API calls always; LoRA adapter calls when configured |

Both are written per-game to the game's output directory. `coldstart_io_all.jsonl` is **incrementally flushed per-episode** (append mode) so data survives crashes.

### Recorded functions and sample formats

#### 1. `segment_ranking` (llm_teacher — SEGMENTATION adapter)

Rank candidate skills for a trajectory segment.

```json
{
  "function": "segment_ranking",
  "prompt": "You are an expert at recognizing skills...\nA trajectory segment spans timesteps 11 to 12...",
  "response": "{\"ranking\":[\"late:DEFEND\",\"POSITION\",...],\"reasoning\":\"...\"}",
  "parsed": {"ranking": ["late:DEFEND", "POSITION", "..."], "reasoning": "..."},
  "model": "gpt-5.4",
  "temperature": 0.3,
  "max_tokens": 1000,
  "elapsed_s": 3.42,
  "segment_start": 11,
  "segment_end": 12,
  "skill_names": ["CLEAR", "POSITION", "early:CLEAR", "late:DEFEND"]
}
```

#### 2. `transition_ranking` (llm_teacher — SEGMENTATION adapter)

Rank likely next skills given the previous skill.

```json
{
  "function": "transition_ranking",
  "prompt": "The agent just finished executing skill: \"early:CLEAR\"...",
  "response": "{\"ranking\":[\"early:POSITION\",...],\"reasoning\":\"...\"}",
  "parsed": {"ranking": ["early:POSITION", "early:SETUP", "..."], "reasoning": "..."},
  "model": "gpt-5.4",
  "temperature": 0.3,
  "elapsed_s": 2.3,
  "skill_names": ["CLEAR", "POSITION", "early:CLEAR", "early:POSITION"],
  "prev_skill": "early:CLEAR"
}
```

#### 3. `pairwise_choice` (llm_teacher — SEGMENTATION adapter)

A/B choice for uncertain segment assignments.

```json
{
  "function": "pairwise_choice",
  "prompt": "Segment: timesteps 0 to 13\nObservations: [...]...\nSkill A: POSITION\nSkill B: DEFEND",
  "response": "{\"choice\":\"A\",\"evidence\":\"...\"}",
  "parsed": {"choice": "A", "evidence": "The segment is dominated by selecting quest teams..."},
  "model": "gpt-5.4",
  "elapsed_s": 2.62,
  "segment_end": 13,
  "skill_a": "POSITION",
  "skill_b": "DEFEND"
}
```

#### 4. `skill_naming` (llm_teacher — SEGMENTATION adapter)

Name + description for new skill clusters. Fires when new skills emerge.

```json
{
  "function": "skill_naming",
  "prompt": "Name this skill cluster based on the following segments...",
  "response": "{\"name\":\"Guard Endgame Reads\",\"description\":\"...\"}",
  "parsed": {"name": "Guard Endgame Reads", "description": "..."},
  "model": "gpt-5.4",
  "elapsed_s": 1.8,
  "skill_names": ["DEFEND"]
}
```

#### 5. `boundary_proposal:predicate_extraction` (BOUNDARY adapter)

Extract structured predicates from state descriptions.

```json
{
  "module": "boundary_proposal",
  "function": "predicate_extraction",
  "prompt": "You are analyzing a game agent's trajectory...\nTimesteps:\n  t=0: game=2048 | highest_tile=2 ...",
  "response": "[{\"phase\":\"opening\",\"highest_tile\":\"2\"}, ...]",
  "parsed": {"n_predicates": 30},
  "model": "qwen3-14b",
  "temperature": 0.2,
  "max_tokens": 3000,
  "elapsed_s": 1.2,
  "segment_start": 0,
  "segment_end": 30,
  "n_steps": 30
}
```

#### 6. `boundary_proposal:boundary_significance` (BOUNDARY adapter)

Filter which predicate changes are real skill boundaries.

```json
{
  "module": "boundary_proposal",
  "function": "boundary_significance",
  "prompt": "You are analyzing state transitions...\nPairs:\n  t=5: {\"phase\":\"opening\"} -> {\"phase\":\"midgame\"}...",
  "response": "[true, false, true]",
  "parsed": {"significances": [true, false, true]},
  "model": "qwen3-14b",
  "temperature": 0.1,
  "max_tokens": 2000,
  "elapsed_s": 0.8,
  "n_steps": 3
}
```

#### 7. `stage3_contract:contract_summary` (CONTRACT adapter)

Generate effect summary (eff_add / eff_del / description) for a skill.

```json
{
  "module": "stage3_contract",
  "function": "contract_summary",
  "prompt": "You are analyzing skill effects...\nSkill: endgame:MERGE\nNumber of instances: 5...",
  "response": "{\"eff_add\":[\"phase=endgame\"],\"eff_del\":[\"phase=midgame\"],\"description\":\"Transition to endgame\"}",
  "parsed": {"eff_add": ["phase=endgame"], "eff_del": ["phase=midgame"], "description": "Transition to endgame"},
  "temperature": 0.1,
  "elapsed_s": 1.5,
  "skill_id": "endgame:MERGE",
  "extra": {"n_instances": 5}
}
```

#### 8. `bank_curator:filter_candidates` (CURATOR adapter)

Approve / veto / defer bank maintenance actions.

```json
{
  "module": "bank_curator",
  "function": "filter_candidates",
  "prompt": "You are a skill bank maintenance curator...\nAction 0: REFINE on endgame:MERGE...",
  "response": "{\"decisions\":[{\"idx\":0,\"verdict\":\"approve\",\"reason\":\"clear evidence\"}]}",
  "parsed": {"decisions": [{"idx": 0, "verdict": "approve", "reason": "clear evidence"}]},
  "temperature": 0.2,
  "elapsed_s": 2.0,
  "extra": {"n_candidates": 3}
}
```

#### 9. `skill_retrieval:retrieve_skills` (RETRIEVAL adapter)

Query rewriting + skill ranking for decision agents.

```json
{
  "module": "skill_retrieval",
  "function": "retrieve_skills",
  "prompt": "You are a skill retrieval assistant...\nAgent's current goal: merge tiles...",
  "response": "{\"rewritten_query\":\"endgame merge\",\"ranking\":[\"endgame:MERGE\"],\"reasoning\":\"best fit\"}",
  "parsed": {"rewritten_query": "endgame merge", "ranking": ["endgame:MERGE"], "reasoning": "best fit"},
  "temperature": 0.1,
  "elapsed_s": 0.5,
  "skill_names": ["endgame:MERGE", "midgame:MERGE"],
  "extra": {"query": "merge tiles", "top_k": 5}
}
```

#### 10. `pipeline:predicate_extraction`

Same as #5 but called from `SkillBankAgent._ensure_predicates_extracted` (Stage 3).

```json
{
  "module": "pipeline",
  "function": "predicate_extraction",
  "prompt": "You are analyzing a game agent's trajectory...",
  "response": "[{\"phase\":\"opening\"}]",
  "parsed": {"n_predicates": 1},
  "model": "gpt-5.4",
  "temperature": 0.2,
  "max_tokens": 3000,
  "elapsed_s": 1.1,
  "segment_start": 0,
  "segment_end": 30,
  "n_steps": 30
}
```

#### 11. `pipeline:protocol_synthesis`

Generate structured execution protocol for a skill.

```json
{
  "module": "pipeline",
  "function": "protocol_synthesis",
  "prompt": "You are a game-AI protocol designer...\nSkill: endgame:MERGE\nEffects: ...",
  "response": "{\"preconditions\":[\"high tile\"],\"steps\":[\"merge tiles\"],\"success_criteria\":[\"reached 2048\"],\"abort_criteria\":[\"no moves\"]}",
  "parsed": {"preconditions": ["high tile"], "steps": ["merge tiles"], "success_criteria": ["reached 2048"], "abort_criteria": ["no moves"]},
  "model": "gpt-5.4",
  "temperature": 0.2,
  "max_tokens": 800,
  "elapsed_s": 1.8,
  "skill_id": "endgame:MERGE"
}
```

### Coverage matrix

| # | Module | Function | LoRA Adapter | Fires w/o LoRA? | Output file |
|---|--------|----------|--------------|-----------------|-------------|
| 1 | `llm_teacher` | `segment_ranking` | SEGMENTATION | Yes (API fallback) | `teacher_io_coldstart.jsonl` |
| 2 | `llm_teacher` | `transition_ranking` | SEGMENTATION | Yes | `teacher_io_coldstart.jsonl` |
| 3 | `llm_teacher` | `pairwise_choice` | SEGMENTATION | Yes | `teacher_io_coldstart.jsonl` |
| 4 | `llm_teacher` | `skill_naming` | SEGMENTATION | Yes | `teacher_io_coldstart.jsonl` |
| 5 | `boundary_proposal` | `predicate_extraction` | BOUNDARY | Yes (API fallback) | `coldstart_io_all.jsonl` |
| 6 | `boundary_proposal` | `boundary_significance` | BOUNDARY | Yes (API fallback) | `coldstart_io_all.jsonl` |
| 7 | `stage3_contract` | `contract_summary` | CONTRACT | No (LoRA only) | `coldstart_io_all.jsonl` |
| 8 | `bank_curator` | `filter_candidates` | CURATOR | No (LoRA only) | `coldstart_io_all.jsonl` |
| 9 | `skill_retrieval` | `retrieve_skills` | RETRIEVAL | No (LoRA only) | `coldstart_io_all.jsonl` |
| 10 | `pipeline` | `predicate_extraction` | — | Yes (API) | `coldstart_io_all.jsonl` |
| 11 | `pipeline` | `protocol_synthesis` | — | Yes (API) | `coldstart_io_all.jsonl` |

Functions 7–9 only produce records when `MultiLoraSkillBankLLM` is configured (Qwen3-8B + LoRA adapters). The rest fire via the API fallback.

### Implementation

The centralized recording module is `coldstart_io.py`:

```python
from skill_agents_grpo.coldstart_io import record_io, ColdStartRecord, flush, reset

# Record after each LLM call
record_io(ColdStartRecord(
    module="boundary_proposal",
    function="predicate_extraction",
    prompt=prompt,
    response=response,
    parsed=parsed_result,
    model=model_name,
    elapsed_s=elapsed,
))

# Flush (returns list[dict] and clears buffer)
records = flush()

# Non-destructive read
records = get_records()
```

The main pipeline (`extract_skillbank_grpo_gpt54.py`):
- Resets the buffer at the start of each game.
- Flushes incrementally after each episode (append to `coldstart_io_all.jsonl` with `episode_index` and `game` tags).
- Flushes remaining records at the end of the pipeline.
- Prints per-module record counts in the summary.

---

## Subpackage docs

- [boundary_proposal/README.md](boundary_proposal/README.md) — Stage 1 signals and `segment_episode` / `propose_from_episode`.
- [infer_segmentation/README.md](infer_segmentation/README.md) — Stage 2 preference learning and decoders.
- [infer_segmentation/OVERVIEW.md](infer_segmentation/OVERVIEW.md) — Stage 2 architectural overview.
- [stage3_mvp/README.md](stage3_mvp/README.md) — Stage 3 effects-only contract learning.
- [bank_maintenance/README.md](bank_maintenance/README.md) — Stage 4 split/merge/refine.
- [skill_bank/README.md](skill_bank/README.md) — Persistent storage and NEW pool management.
- [skill_evaluation/README.md](skill_evaluation/README.md) — Quality assessment.
- [contract_verification/README.md](contract_verification/README.md) — Legacy full Pre/Eff/Inv contract verification.
- [lora/README.md](lora/README.md) — Multi-LoRA model (Qwen3-8B + adapters).
- [PLAN.md](PLAN.md) — Full operating plan (constraints, thresholds, module map).
- [PIPELINE_CALL_FLOW.md](PIPELINE_CALL_FLOW.md) — How each function is called from the agent framework.
- [CHANGELOG_REFACTOR.md](CHANGELOG_REFACTOR.md) — Refactoring history and migration notes.
- [TODO_Lists/SKILLBANK_GRPO_PLAN.md](../TODO_Lists/SKILLBANK_GRPO_PLAN.md) — GRPO wrapper architecture, 3 LoRA adapters, Stage 4 propose-filter-execute, proto-skill staging, guardrails.

---

## File layout

```
skill_agents_grpo/
├── README.md                 # This file
├── PLAN.md                   # SkillBank Agent operating plan
├── PIPELINE_CALL_FLOW.md     # How each function is called from the agent framework
├── CHANGELOG_REFACTOR.md     # Refactoring history and migration notes
├── __init__.py               # SkillBankAgent, SkillQueryEngine, NewPoolManager, etc.
├── _llm_compat.py            # Reasoning-model compatibility (strip_think_tags, /no_think wrapper)
├── coldstart_io.py           # Centralized cold-start I/O recording for all GRPO-connected functions
├── default_predicates.py     # Shared default predicate extractor (no-op fallback)
├── pipeline.py               # SkillBankAgent orchestrator (contract feedback, NEW pool, proto-skill flow)
├── query.py                  # SkillQueryEngine + SkillSelectionResult (retrieval + selection policy)
├── tool_call_reward.py       # Reward for tool calls (agentic RL)
├── skill_bank/
│   ├── bank.py               # SkillBankMVP persistence + compat_fn (Stage 3→2 feedback)
│   ├── new_pool.py           # NewPoolManager: NEW tracking; ProtoSkillManager staging
│   └── llm_retrieval.py      # LLM-assisted skill retrieval via RETRIEVAL LoRA adapter
├── boundary_proposal/        # Stage 1 (not GRPO-wrapped)
│   ├── proposal.py           # BoundaryCandidate, ProposalConfig, propose_boundary_candidates()
│   ├── llm_extractor.py      # LLM predicate extraction + boundary significance
│   ├── episode_adapter.py    # propose_from_episode(): adapts Episodes for boundary proposal
│   ├── changepoint.py        # Embedding changepoint detection
│   ├── boundary_preference.py # Boundary plausibility scoring
│   ├── signal_extractors.py  # Extract signals from trajectories
│   └── example_toy.py        # Toy example / integration test
├── infer_segmentation/       # Stage 2 — preference learning + phase-aware intention-fit scoring
│   ├── config.py             # ScorerWeights (incl. intention_fit=2.0), SegmentationConfig
│   ├── scorer.py             # SegmentScorer: 6-term composite (behavior_fit, intention_fit, ...)
│   ├── dp_decoder.py         # Viterbi HSMM decoder
│   ├── beam_decoder.py       # Beam search decoder
│   ├── episode_adapter.py    # infer_and_segment(), infer_and_segment_offline(), grpo_scorer_factory()
│   ├── phase_detector.py     # Phase detection: game-specific extractors + generic fallback
│   ├── llm_teacher.py        # LLM teacher: rankings → pairwise prefs; TeacherIORecord cold-start
│   ├── preference.py         # PreferenceStore, PreferenceScorer (Bradley-Terry)
│   ├── diagnostics.py        # SegmentationResult, SegmentDiagnostic
│   └── example_toy.py        # Toy example / integration test
├── stage3_mvp/               # Stage 3 — CONTRACT LoRA wraps llm_summarize_contract()
│   ├── run_stage3_mvp.py     # Stage 3 orchestrator: run_stage3_mvp()
│   ├── config.py             # Stage3MVPConfig
│   ├── schemas.py            # SkillEffectsContract, VerificationReport, Protocol, Skill, SegmentRecord
│   ├── contract_learn.py     # learn_effects_contract()
│   ├── contract_verify.py    # verify_effects_contract()
│   ├── contract_refine.py    # refine_effects_contract()
│   ├── effects_compute.py    # compute_effects()
│   ├── llm_contract.py       # llm_summarize_contract() — wrapped by CONTRACT LoRA
│   ├── segment_summarize.py  # summarize_segment()
│   ├── extract_predicates.py # default_extract_predicates()
│   └── predicate_vocab.py    # Predicate vocabulary management
├── quality/                  # Stage 4.5 — sub-episode quality evaluation
│   └── sub_episode_evaluator.py  # score_sub_episode(), run_quality_check_batch()
├── bank_maintenance/         # Stage 4 — propose → filter (CURATOR LoRA) → execute
│   ├── run_bank_maintenance.py   # Stage 4 orchestrator: run_bank_maintenance()
│   ├── config.py             # BankMaintenanceConfig
│   ├── schemas.py            # SkillProfile, BankDiffReport
│   ├── split.py              # check_split_triggers(), execute_split()
│   ├── merge.py              # retrieve_merge_candidates(), execute_merge()
│   ├── refine.py             # check_refine_triggers(), refine_skill()
│   ├── llm_curator.py        # filter_candidates() — CURATOR LoRA approve/veto/defer
│   ├── local_redecode.py     # redecode_windows(), build_redecode_requests()
│   ├── indices.py            # EffectInvertedIndex, MinHashLSH, EmbeddingANN
│   ├── duration_model.py     # DurationModelStore
│   └── example_toy.py        # Toy example / integration test
├── contract_verification/    # Legacy full Pre/Eff/Inv contract verification
├── skill_evaluation/         # Quality evaluation
│   ├── run_evaluation.py     # run_skill_evaluation()
│   ├── evaluators.py         # evaluate_coherence(), evaluate_discriminability(), etc.
│   ├── config.py             # SkillEvaluationConfig
│   ├── schemas.py            # EvaluationSummary, SkillQualityReport
│   └── example_toy.py        # Toy example / integration test
├── grpo/                     # GRPO infrastructure
│   ├── orchestrator.py       # GRPOOrchestrator: enable/disable wrappers, train_step
│   ├── wrapper.py            # GRPOCallWrapper (G samples, reward, store, return best)
│   ├── trainer.py            # GRPOLoRATrainer (log_probs, advantages, PPO-clip LoRA update)
│   ├── rewards.py            # contract_reward(), curator_reward(), segmentation_reward()
│   ├── buffer.py             # GRPOBuffer (adapter, prompt, completions, rewards)
│   └── config.py             # Per-stage GRPO hyperparameters (StageGRPOConfig, GRPOConfig)
├── lora/                     # Multi-LoRA model (GRPO-capable)
│   ├── model.py              # MultiLoraSkillBankLLM: generate() + log_probs() for GRPO training
│   ├── config.py             # MultiLoraConfig (Qwen3-8B base + adapter paths)
│   └── skill_function.py     # SkillFunction enum (SEGMENT, CONTRACT, CURATOR, ...)
└── extract_skillbank/        # Extraction scripts
    ├── extract_skillbank_grpo_gpt54.py  # Main script: --local_model, --use_grpo flags
    ├── run_extract_skillbank_grpo.sh    # Shell wrapper (PYTHONPATH + API key setup)
    └── output/               # Per-game outputs (skill_bank.jsonl, catalogs, cold-start I/O)
```
