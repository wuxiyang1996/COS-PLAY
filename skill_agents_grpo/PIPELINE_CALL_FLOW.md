# Skill Agents — How Each Function Is Called in the Agent Framework

This document describes how each component of `skill_agents` is invoked from the **SkillBankAgent** pipeline and related callers.

---

## Does the Skill Bank Agent have parameters we update to improve quality?

Yes, in two ways — but only one of them is gradient-based.

| What | Updated? | How | Improves |
|------|----------|-----|----------|
| **PreferenceScorer** (in infer_segmentation) | **Yes** | **Trained** on LLM preference data (Bradley–Terry): gradient steps on pairwise preferences so that the scorer ranks skills the way the LLM did. | Quality of **segment → skill** labelling (which segment gets which skill). |
| **PipelineConfig** (and stage configs) | **Manually / tuned** | Thresholds and penalties (e.g. `eff_freq`, `new_skill_penalty`, `margin_threshold`, `split_pass_rate_threshold`, …) are set in config. They are **not** learned by backprop; you can improve quality by **tuning** them (grid search, Bayesian opt, or downstream reward). | Quality of boundaries, contract learning, split/merge/refine decisions. |
| **Bank content** (skills + contracts) | **Yes** | Updated by the pipeline / Hard-EM (add, remove, refine skills and contracts). This is **data**, not neural parameters. | Which skills exist and what they do. |

So: we **do** update something to improve how well the agent performs its functions — the **PreferenceScorer** is trained to improve segmentation; the **bank** is updated to reflect better skills and contracts; and **config** can be tuned to improve behavior, but it is not currently “trained” as parameters.

---

## 1. Entry point: `SkillBankAgent` (`pipeline.py`)

The single entry point is **`SkillBankAgent`**. It holds the bank (`SkillBankMVP`), the segment cache, and (lazily) the query engine.

| Agent method | What it does |
|--------------|--------------|
| `agent.load()` | Loads existing bank from `config.bank_path` |
| `agent.ingest_episodes(episodes, ...)` | Stage 1+2 per episode, then Stage 3 once |
| `agent.segment_episode(episode, ...)` | Stage 1+2 only (one episode) |
| `agent.run_contract_learning()` | Stage 3 on all cached segments |
| `agent.run_bank_maintenance()` | Stage 4 (split/merge/refine) |
| `agent.materialize_new_skills()` | Promote `__NEW__` clusters to real skills |
| `agent.run_evaluation()` | Run skill_evaluation (LLM judge) |
| `agent.run_full_iteration(episodes?)` | Stage 3 → Stage 4 → materialize → snapshot |
| `agent.run_until_stable(max_iterations)` | Loop `run_full_iteration` until convergence |
| `agent.query_skill(key)`, `agent.query_by_effects(...)` | Query API for decision_agents |

---

## 2. Stage 1 + Stage 2: per-episode segmentation

**Called from:** `SkillBankAgent.segment_episode()` (and thus from `ingest_episodes()`).

**Flow:**

1. **`segment_episode()`** builds Stage 2 config and proposal config, gets/creates `PreferenceStore`, then:
   - If there are known skills: calls **`infer_and_segment()`**
   - Else: calls **`infer_and_segment_offline()`**

2. **`infer_and_segment()`** (in `infer_segmentation/episode_adapter.py`):
   - **Stage 1:** `propose_from_episode(episode, ...)` → `candidate_centers_only(...)` → candidate cut indices.
   - **Stage 2:**  
     - Cold start: `collect_segment_preferences()` (+ optional `collect_transition_preferences()`), store in `PreferenceStore`.  
     - Build scorer from preferences, **`infer_segmentation(candidates, T, skill_names, ...)`** (DP or beam decode).  
     - Optional active loop: `collect_uncertain_preferences()` → retrain scorer → decode again.
   - Converts result to `SubTask_Experience` list.

3. **`segment_episode()`** then calls **`_cache_trajectory()`**: stores observations in `_observations_by_traj`, converts segments to **`SegmentRecord`** and appends to `_all_segments` (and `_new_pool` when label is `__NEW__`).

**Summary:** **boundary_proposal** is used via `propose_from_episode` + `candidate_centers_only`; **infer_segmentation** via `infer_and_segment` / `infer_and_segment_offline` (which use scorer, decoders, and LLM teacher).

---

## 3. Stage 3: contract learning

**Called from:** `SkillBankAgent.run_contract_learning()` (after `ingest_episodes`) and from `SkillBankAgent.materialize_new_skills()` for each new-skill cluster.

- **`run_contract_learning()`** builds **`SegmentSpec`** list from `_all_segments` (excluding NEW), builds **`Stage3MVPConfig`** from `PipelineConfig`, then calls:
  - **`run_stage3_mvp(specs, self._observations_by_traj, config, bank=self.bank)`**  
  from `stage3_mvp/run_stage3_mvp.py`.
- **`materialize_new_skills()`** groups `_new_pool` by effect signature, for each qualifying cluster builds specs with a new skill id and calls **`run_stage3_mvp(...)`** again, then optionally **`suggest_skill_name()`** (LLM) and **`bank.add_or_update(contract)`**.

**Summary:** **stage3_mvp** is used only through **`run_stage3_mvp()`**; the pipeline does **not** call **contract_verification**’s `run_stage3` (that’s the full Pre/Eff/Inv pipeline; the agent uses the effects-only MVP).

---

## 4. Stage 4: bank maintenance

**Called from:** `SkillBankAgent.run_bank_maintenance()` (and thus from `run_full_iteration()`).

- **`run_bank_maintenance()`** builds **`BankMaintenanceConfig`** from `PipelineConfig`, then:
  - **`run_bank_maintenance(bank=self.bank, all_segments=self._all_segments, config=..., traj_lengths=self._traj_lengths, report_path=...)`**  
  from `bank_maintenance/run_bank_maintenance.py`.
- If the result has **`alias_map`** (e.g. after merges), **`_apply_alias_map()`** updates `_all_segments` and `_new_pool` skill labels.

**Summary:** **bank_maintenance** is used only through **`run_bank_maintenance()`**.

---

## 5. Skill evaluation (LLM judge)

**Called from:** `SkillBankAgent.run_evaluation()` (optional; not inside `run_full_iteration`).

- **`run_evaluation()`** builds **`SkillEvaluationConfig`**, then:
  - **`run_skill_evaluation(bank=self.bank, all_segments=self._all_segments, config=..., report_path=...)`**  
  from `skill_evaluation/run_evaluation.py`.

**Summary:** **skill_evaluation** is used only through **`run_skill_evaluation()`**.

---

## 6. Skill bank

**Used by the agent as state, not as a separate “step”:**

- **`SkillBankAgent`** constructs **`SkillBankMVP(path=config.bank_path)`** and keeps it in **`self.bank`**.
- Stage 3 and materialize write via **`run_stage3_mvp(..., bank=self.bank)`** and **`bank.add_or_update()`**.
- Stage 4 and evaluation read/update via **`run_bank_maintenance(bank=self.bank, ...)`** and **`run_skill_evaluation(bank=self.bank, ...)`**.
- **`skill_bank/bank.py`** is the only place that implements this storage (JSONL, versioning).

---

## 7. Query API (for decision_agents)

**Called from:** `SkillBankAgent.query_skill()`, `query_by_effects()`, `list_skills()`, `get_skill_detail()`.

- Each of these uses **`_get_query_engine()`**, which lazily builds **`SkillQueryEngine(self.bank)`** from **`skill_agents/query.py`**.
- So **query** is used only through **`SkillQueryEngine`** (embedding + keyword + effect retrieval over `SkillBankMVP`). The agent invalidates the engine (`_invalidate_query_engine()`) after any change to the bank so the next query gets a fresh index.

---

## 8. Tool-call reward (RL)

**Not called from `SkillBankAgent`.** It’s for **decision_agents** (or another trainer) to score tool use:

- **`tool_call_reward.compute_tool_call_reward(...)`** takes `skill_bank` (or an object with `.bank`), optional **`SkillQueryEngine`**, tool name/args, context/outcome observations, and optionally `retrieved_skill_id`, and returns relevance/utility rewards for RL. So **skill_bank** and **query** are used here only as inputs to the reward function.

---

## End-to-end call flow (diagram)

```
SkillBankAgent
├── load()                    → bank.load()
├── ingest_episodes(episodes)
│   ├── for each episode: segment_episode(ep)
│   │   ├── infer_and_segment() or infer_and_segment_offline()
│   │   │   ├── boundary_proposal: propose_from_episode(), candidate_centers_only()
│   │   │   ├── infer_segmentation: collect_*_preferences(), infer_segmentation() (scorer + DP/beam)
│   │   │   └── _segments_to_sub_episodes()
│   │   └── _cache_trajectory()  → _all_segments, _observations_by_traj
│   └── run_contract_learning()
│       └── stage3_mvp: run_stage3_mvp(specs, observations, config, bank)
├── run_full_iteration(episodes?)
│   ├── [if episodes] ingest_episodes()   # as above
│   ├── [else] run_contract_learning()
│   ├── run_bank_maintenance()            → bank_maintenance.run_bank_maintenance(bank, segments, ...)
│   ├── materialize_new_skills()         → run_stage3_mvp() per cluster, suggest_skill_name(), bank.add_or_update()
│   └── _take_snapshot()
├── run_until_stable()                    → loop run_full_iteration()
├── run_evaluation()                     → skill_evaluation.run_skill_evaluation(bank, segments, ...)
├── query_skill() / query_by_effects()    → SkillQueryEngine(bank).query() / .query_by_effects()
└── add_skill / remove_skill / update_skill  → bank.add_or_update() / .remove() + _invalidate_query_engine()
```

---

## Summary table

| Component | Called from agent? | Entry point / usage |
|-----------|--------------------|----------------------|
| **boundary_proposal** | Yes (via segment_episode) | `propose_from_episode()`, `candidate_centers_only()` |
| **infer_segmentation** | Yes (via segment_episode) | `infer_and_segment()`, `infer_and_segment_offline()` |
| **stage3_mvp** | Yes | `run_stage3_mvp()` in run_contract_learning + materialize_new_skills |
| **contract_verification** | No | Full Pre/Eff/Inv pipeline; agent uses MVP only |
| **bank_maintenance** | Yes | `run_bank_maintenance()` in run_full_iteration |
| **skill_evaluation** | Yes (optional) | `run_skill_evaluation()` via run_evaluation() |
| **skill_bank** | Yes (as state) | `self.bank` (SkillBankMVP); load/save/add/remove |
| **query** | Yes | `SkillQueryEngine(self.bank)` behind query_skill / query_by_effects |
| **tool_call_reward** | No (external) | Used by decision_agents / trainer for RL reward |

---

## Tool-calling vs programmatic: what the LLM actually calls

**Only the query API is exposed as tools that the decision agent (LLM) calls during rollouts.** The rest of skill_agents is run **programmatically** by the pipeline or by the trainer, not as LLM tool invocations.

| Invocation style | What | Where |
|------------------|------|--------|
| **Tool-calling (LLM)** | The decision agent chooses an action each step. When it outputs **QUERY_SKILL(key)** or **QUERY_MEM(key)** or **CALL_SKILL(skill_id)**, the runner/env executes that as a “tool”: e.g. `run_tool(TOOL_QUERY_SKILL, {"key": "..."}, agent, ...)` → `query_skill_bank(agent.skill_bank, key)` → `skill_bank.query_skill(key)` (which uses `SkillQueryEngine` under the hood). So **only** `query_skill` / the bank’s query API is a tool the LLM invokes. |
| **Programmatic** | **Segment_episode**, **run_contract_learning**, **run_bank_maintenance**, **materialize_new_skills**, **run_evaluation** are **not** tools. They are called by `SkillBankAgent` (or by the trainer’s co-evolution / Hard-EM path) in a fixed order. The LLM never “calls” these; they run in the background to build and maintain the bank from trajectories. |

So: **tool-calling** = the decision agent’s retrieval actions (query_skill, query_memory, call_skill) executed by the env/runner. **Programmatic** = the skill_agents pipeline (Stage 1–4, evaluation) run by Python code.

---

## How training happens

### Decision agent (LLM) training

1. **Rollouts**  
   The policy (LLM) generates actions per step. Actions can be primitive (e.g. “move north”) or retrieval: **QUERY_SKILL(key)**, **QUERY_MEM(key)**, **CALL_SKILL(skill_id)**. These are parsed and executed:
   - In **VERL**: `GameAIVecEnv.step(actions)` handles retrieval in `_handle_retrieval()` — for QUERY_SKILL it uses `SkillQueryEngine(self.skill_bank).query(key)` and appends results to the observation context.
   - In **standalone** runner: `run_tool(TOOL_QUERY_SKILL, args, agent, ...)` → `query_skill_bank(agent.skill_bank, key)`.

2. **Reward**  
   The reward used for GRPO/PPO is **not** from `skill_agents.tool_call_reward`. It comes from **`decision_agents.reward_func.RewardComputer`** (used by `TrainRewardShaper` in the trainer and by the `reward` tool in the agent):
   - **r_env**: raw environment reward from `env.step`.
   - **r_follow**: skill-following shaping (progress toward active skill’s eff_add predicates).
   - **r_cost**: fixed costs for query_skill, query_memory, call_skill, and skill switching.
   - **r_total** = r_env + w_follow * r_follow + r_cost.

   So the decision agent is trained to maximise this r_total; the negative **query_skill_cost** discourages unnecessary queries. **`compute_tool_call_reward`** in skill_agents is an optional extra (relevance/utility) and is not wired into the default trainer reward.

3. **Optimization**  
   Rollouts (with tool calls and rewards) are fed into the GRPO or VERL GiGPO/PPO trainer; the policy is updated to increase return.

### Skill bank updates (no LLM training)

The **skill bank** is not updated by the LLM. It is updated in two possible ways:

- **SkillBankAgent pipeline (skill_agents)**  
  Some caller (e.g. a script or co-evolution driver) has **Episode** objects and calls:
  - `agent.ingest_episodes(episodes)` → segment_episode (Stage 1+2) per episode, then run_contract_learning (Stage 3);
  - optionally `agent.run_until_stable()` → run_bank_maintenance (Stage 4), materialize_new_skills, etc.

- **Trainer co-evolution / Hard-EM (trainer/skillbank)**  
  After (or alongside) decision agent training, rollouts are converted to trajectories and fed into the **Hard-EM** SkillBank trainer (`trainer.skillbank.em_trainer`, `launch_coevolution`). That path uses its own stages (propose cuts, decode, contracts, update, skilleval) and may use a **SkillBankCoEvolutionCallback** in VERL to periodically update the bank and hot-swap it into the env (`GameAIVecEnv.update_skill_bank(new_bank)`).

So: **training** = training the decision agent’s policy on rollouts where the LLM can use **query_skill** (and other tools); **skill bank updates** = separate, programmatic pipeline or Hard-EM that builds/refines the bank from trajectories.

---

## How the skill bank is updated — and where the LLM is used

The **skill bank** is updated by the pipeline (or Hard-EM), but **an LLM is used inside** that pipeline. It does not update the bank by gradient training; it provides **semantic inputs** that the pipeline turns into segment labels and names.

| Step | Who decides | Role of the LLM |
|------|-------------|------------------|
| **Which segments exist** | Stage 1 (boundary_proposal) | Optional: hybrid extractors can use an LLM to get predicates from text; boundaries are then proposed from signals. |
| **Which skill label each segment gets** | Stage 2 (infer_segmentation) | **LLM is the preference teacher:** the pipeline asks the LLM to **rank** skills for each segment (e.g. “move > gather > attack”). Those rankings become pairwise preferences; a **PreferenceScorer** is trained (e.g. Bradley–Terry) on them; DP/beam decode uses that scorer to assign a skill to each segment. So the **segment–skill mapping** that feeds the bank is **LLM-guided** (via preferences), not from gradient updates. |
| **What the contract is (eff_add, eff_del)** | Stage 3 (stage3_mvp) | **No LLM.** Contracts are computed from predicate deltas and aggregation (stats, frequency thresholds). |
| **Naming a new skill** | materialize_new_skills | **LLM:** when a cluster of `__NEW__` segments is promoted to a skill, the pipeline can call **suggest_skill_name()** so the LLM proposes a human-readable **name** and **description**. |
| **Quality / refinement priorities** | skill_evaluation (optional) | **LLM-as-judge** scores dimensions (coherence, discriminability, etc.). Output can guide what to refine; it does not directly add/remove skills. |

So: **the skill bank agent is updated by a pipeline**, and that pipeline **uses an LLM** to (1) guide segment labelling via preference rankings, (2) optionally name new skills, and (3) optionally evaluate quality. The **content** of contracts and the **decision** of which segments become which skills come from the pipeline logic; the **LLM supplies the semantic/preference signals** that drive that logic.

---

## In the trainer: how the skill bank is updated (no LLM inside the update)

The **trainer** can also update the skill bank, via **co-evolution** (`trainer/launch_coevolution.py`) or a VERL **co-evolution callback**. In that path the bank is updated by **trainer/skillbank** (Hard-EM), not by **skill_agents** pipeline.

| Step | What happens | LLM? |
|------|----------------|------|
| **Data** | Rollouts come from the **decision agent (LLM)** during GRPO. So the **trajectories** that feed the bank update are **generated by the LLM**. | Yes (LLM produces rollouts) |
| **Ingest** | `ingest_rollouts(rollouts)` turns `RolloutRecord`s into `TrajectoryForEM` (frames, actions, etc.). | No |
| **Stage 0** | Enrich frames with predicates (booleanize from observations). | No |
| **Stage 1** | Propose cuts (boundaries) from predicate change, surprisal, min segment width. Rule-based. | No |
| **Stage 2** | **Decode**: effect-matching DP — score each segment against each skill’s contract by effect overlap; assign best skill or `__NEW__`. No preference teacher, no LLM. | **No** |
| **Stage 3** | Learn contracts from decoded segments (aggregate effects, verify). Uses `skill_agents.stage3_mvp` schemas only. | No |
| **Stage 4** | Refine, materialize NEW, merge, split. New skills get synthetic IDs (e.g. `S_new_*`); the trainer does **not** call `suggest_skill_name` (LLM). | **No** |
| **Gating** | SkillEval on holdout (pass rate, new rate, margin); commit or rollback. | No |

So **in the trainer**, the skill bank is updated from **LLM-generated rollouts**, but the **update logic itself does not call an LLM**. Segment labels come from **effect-matching DP**; contracts from **aggregation**; new-skill names are **synthetic**. The only LLM in the loop is the **decision agent** being trained, whose behavior produces the trajectories; the bank is then updated algorithmically by **trainer/skillbank** (Hard-EM). If you want LLM-guided segment labelling or LLM naming of new skills in the trainer, you would need to plug in **skill_agents** (e.g. `SkillBankAgent.ingest_episodes` with Episode objects built from rollouts) or add LLM calls into the trainer’s stages.
