# skill_agents (GRPO Skill Bank Pipeline) — **LEGACY COS-PLAY TRACK**

> **Status — legacy / reference.** This module is the **COS-PLAY co-evolution
> Qwen GRPO skill-bank pipeline** retained for reference and for the
> deferred Qwen3-8B / 32B / 72B tracks. **It is no longer the live
> skill-agent implementation.** The canonical, plan-driven pipeline now
> lives at the repo root under [`harness/`](../harness/),
> [`orchestrator/`](../orchestrator/), [`crafter/`](../crafter/), and the
> new top-level [`skill_bank/`](../skill_bank/) (split-storage +
> `SkillLifecycleManager`).  Library defaults now follow the project-wide
> three-tier stack from [`common/models.py`](../common/models.py): the
> trained skill-bank backbone is `Qwen/Qwen3.5-9B`
> (`BACKBONE_MODEL`), the frozen control-plane backbone +
> LLM-as-judge is `Qwen/Qwen3.5-35B-A3B`
> (`BACKBONE_TEACHER_MODEL` = `BACKBONE_JUDGE_MODEL` — same weights,
> two roles), and the SFT cold-start data teacher is `gpt-5.5`
> (`BACKBONE_SFT_TEACHER_MODEL`). See the
> top-level [`readme.md`](../readme.md), the plan corpus in
> [`plans/`](../plans/), and
> [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) for the live build.
>
> Read this README only if you are working on the deferred Qwen GRPO track
> (`skill_agents/lora/`, `scripts/qwen3_*.py`, `inference/run_qwen3_8b_eval.py`,
> `inference/run_academic_benchmarks.py`) or the legacy migration glue
> scheduled in `skill_bank/legacy_bridge.py`.

---

## The new skill-agent pipeline (where to look instead)

The current build replaces the single GRPO Skill Bank Pipeline described
below with a four-layer, gate-bound architecture. The same conceptual stages
(segment → contract → curate → retrieve) still hold, but ownership and
mechanical invariants are different:

| Concern | Legacy here (COS-PLAY) | **Live build** |
| --- | --- | --- |
| Per-invocation skill runtime | implicit in `pipeline.py` | [`harness/`](../harness/) — `SkillHarness`, `AdapterRegistry`, `EligibilityFilter`, `ReplayValidator`, `RewardLogger`, `SkillAdapter` + `adapters/{gymv,browser,osworld,video,visual_reasoning}` |
| Skill storage + lifecycle | `skill_bank/bank.py` (`SkillBankMVP`, single JSONL) | [`skill_bank/`](../skill_bank/) — split stores (`draft / candidate / active / archive`), `SkillLifecycleManager` (the *only* writer), `SkillRepository`, bank-write isolation invariant |
| Stage 4 mutations | `bank_maintenance/` propose → CURATOR-LoRA filter → execute | [`crafter/`](../crafter/) — `Composer`, `Generalizer`, `Hypothesizer`, `FailureDiagnoser`, `FailureMemory`, `SkillCrafterService` (proposals are **DRAFT only**; never touches `active_store`) |
| End-to-end control plane | `extract_skillbank_grpo_gpt54.py` driver script | [`orchestrator/`](../orchestrator/) — `EpisodeRunner`, atomic `ArtifactStore`, `BudgetController`, `GateService` (stages 0–4), `PromotionOrchestrator`, `SnapshotManager` |
| Promotion control | `min_instances_per_skill` + verification rate | **Unified Skill Gate** — canonical `SkillStatus` (`draft → candidate → shadow → provisional → active`, plus `deprecated / rejected / rolled_back`), six-gate stack `static → replay → shadow → transfer → non-regression`. See [`plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md). |
| Skill records | `SkillEffectsContract` + `VerificationReport` (effects-only) | `SkillRecord` + `SkillEpisode` + `GateVerdict` + `SkillEvaluationRecord` + `BankMutationProposal` + `FailureTrace` + `RunRelease` in [`data_structure/extensions/`](../data_structure/extensions/) |
| Backbone model | Qwen3-8B + 3 GRPO LoRA adapters (CONTRACT, CURATOR, SEGMENT) | **`Qwen/Qwen3.5-9B`** for the trained actor + skill-bank adapters (`BACKBONE_MODEL`), **`Qwen/Qwen3.5-35B-A3B`** for both the frozen crafter / harness / orchestrator backbone AND the LLM-as-judge for skill evaluation / promotion gates (`BACKBONE_TEACHER_MODEL` = `BACKBONE_JUDGE_MODEL` — same weights, two roles), and **`gpt-5.5`** for SFT cold-start data labeling only (`BACKBONE_SFT_TEACHER_MODEL`). The Qwen3-VL Phase-F teachers + 32B / 72B tracks remain deferred (overridable via `VLM_AGENT_BACKBONE_*` env vars; e.g. `VLM_AGENT_BACKBONE_JUDGE_MODEL=gpt-5.5` to swap to an off-distribution oracle). |

### Mechanically-enforced invariants the new pipeline adds

These are **not** present in this legacy module; they are enforced by code
(not reviewer discipline) in the new build, with tests in
`tests/test_invariants.py`:

1. **G0 evidence-driven** — every successful non-`ACTION` skill must record
   `evidence_in` / `evidence_out` / `evidence_warrant`. `SkillEpisode.finalize`
   raises on empty evidence; `SkillLifecycleManager` rejects `ACTIVE`
   promotion when `expected_evidence_roles` is empty.
2. **No-memory** — no cross-episode storage. `SkillEpisodeStep.__post_init__`
   rejects any action type starting with `QUERY_MEM` / `WRITE_MEM`.
3. **General-protocol** — every `ACTIVE` skill must be feasible in ≥ 2
   domains (`feasible_domains < 2` is rejected at promotion).
4. **Bank-write isolation** — only `SkillLifecycleManager` may mutate any
   skill store; direct `SkillStore.put` / `remove` raises `StoreLockedError`.
5. **Gate-bound promotion** — `PromotionOrchestrator.promote` rejects `FAIL`
   verdicts and content-hash drift; refuses `ACTIVE` on `LIMITED_PASS`.
6. **Crafter scope** — `SkillCrafterService` materialises only `DRAFT`
   records; static dependency rule keeps `crafter/` from importing
   `skill_bank/stores`.
7. **Source-domain / transfer-target asymmetry** — every `ACTIVE` skill must
   declare ≥ 1 `source_domains` entry (currently `{"gymv"}`) **and** ≥ 1
   `verified_domains` entry from `{browser, osworld, video, visual_reasoning}`.
8. **`verified_domains` is gate-owned** — only
   `SkillLifecycleManager.record_transfer_verification` may mutate
   `verified_domains` / `adapter_history`, called from
   `PromotionOrchestrator._record_transfer_verifications` *before* the
   status transition.

### Bridge from this module into the new pipeline

Two top-level files in this folder act as the bridge into the
transferable-skill model used by the new pipeline (these are **not** in the
upstream COS-PLAY repo):

- [`skill_template.py`](skill_template.py) — `TransferableSkill`, `SlotBinding`,
  `ReasoningProtocol`, `HopStep`, `AbstractPredicate`, `FAMILY_PROTOCOLS`.
  Defines the domain-agnostic skill representation built around the inner
  MDP action vocabulary (`GROUND`, `CHECK`, `RETRIEVE`, `CONCLUDE`,
  `EXECUTE`) and the canonical shared slots (`target`, `blocker`,
  `constraint`, `candidate_set`, `history_anchor`).
- [`extract_transferable.py`](extract_transferable.py) — five-stage extractor
  (predicate normalisation → structural clustering → template abstraction →
  transferability scoring → export) that mines per-game `SkillBankMVP`
  banks for cross-domain reusable templates and writes
  `transferable_skills.jsonl` + `transfer_index.json`.

Migration glue scheduled (per `IMPLEMENTATION-STATUS.md`):

- `skill_bank/legacy_bridge.py` — one-way migration of Stage-3
  `skill_agents/skill_bank` records into the new `SkillRecord` format.
- `decision_agents.skill_interface.HarnessSkillProvider` — wraps
  `SkillHarness.select_eligible_skills` so the legacy COS-PLAY Actor can
  consume the new harness instead of querying the legacy bank directly.

---

## Overlap with the live build (`orchestrator/` + `skill_bank/` + `crafter/`)

This module *does* perform skill-bank management and skill promotion. So
does the live build. The two are deliberately overlapping — the live
build is the successor — but they are **isolated at runtime** and
cannot interfere with each other. Three points to keep straight:

### 1. The overlap is functional, not behavioural

Each row of the side-by-side table above is one concern that exists in
both trees. The legacy pipeline owns its own loop end-to-end; the live
build owns the same loop end-to-end via different files:

| Concern | Legacy (here) | Live build |
| --- | --- | --- |
| Skill execution at invocation | `skill_agents/pipeline.py` | `harness/skill_harness.py` |
| Skill storage | `skill_agents/skill_bank/bank.py::SkillBankMVP` (single JSONL) | `skill_bank/` (split `draft / candidate / active / archive`, `SkillLifecycleManager` as the only writer) |
| Bank mutations | `skill_agents/bank_maintenance/` (propose → CURATOR-LoRA filter → execute) | `crafter/` (proposals are DRAFT only, never touch `active_store`) |
| Promotion gating | `min_instances_per_skill` + verification rate, applied directly inside `bank_maintenance/` | `orchestrator/gate_service.py` (canonical `static → replay → shadow → transfer → non-regression` gate stack) + `orchestrator/promotion_orchestrator.py` (transactional promote / rollback) |
| End-to-end driver | `skill_agents/extract_skillbank/extract_skillbank_grpo_gpt54.py` | `orchestrator/runner.py::EpisodeRunner` |

So `skill_agents/bank_maintenance/` is the conceptual predecessor of
`crafter/` + `orchestrator.PromotionOrchestrator`, and
`skill_agents/skill_bank/` is the conceptual predecessor of the new
top-level `skill_bank/`.

### 2. Why this is not a real conflict today

The live build *never imports* `skill_agents`. The dependency edges go
only one way:

- `skill_agents` is imported only by the legacy COS-PLAY entrypoints:
  `decision_agents/` (the legacy Actor), `scripts/qwen3_*.py`,
  `inference/run_qwen3_8b_eval.py`, `inference/run_inference.py`,
  `inference/run_diplomacy_discrete_eval.py`,
  `labeling/extract_skillbank_gpt54.py`. None of these run by default —
  the library defaults in `common/models.py` pin GPT-4o.
- The live test suite (`tests/test_smoke.py`,
  `tests/test_invariants.py`, `tests/test_few_shot_transfer.py`,
  `tests/test_crafter_repair.py`, `tests/test_backbone_model.py`) does
  not depend on this module.
- The two banks share *no records*. `SkillBankMVP` writes its own JSONL
  with `SkillEffectsContract` + `VerificationReport`; the live
  `skill_bank/` writes `SkillRecord` / `SkillEpisode` / `GateVerdict` /
  `SkillEvaluationRecord` from `data_structure/extensions/`. A record
  from one cannot be loaded by the other.
- Bank-write isolation in the live build is mechanically enforced —
  `SkillStore.put` / `remove` raises `StoreLockedError` unless called
  through `SkillLifecycleManager` (invariant 4 in the root
  [`readme.md`](../readme.md)). Nothing in `skill_agents/` can promote
  a record into the live ACTIVE store even if the two trees were wired
  together by accident.

### 3. The two pieces of glue that *do* cross the boundary

Both bridge *out of* this module into the live build, never the other
way:

- [`skill_template.py`](skill_template.py) +
  [`extract_transferable.py`](extract_transferable.py) — the
  `TransferableSkill` / `SlotBinding` / `ReasoningProtocol` /
  `HopStep` / `AbstractPredicate` / `FAMILY_PROTOCOLS` types are
  **already consumed by the live build**, specifically by
  [`crafter/generalizer.py`](../crafter/generalizer.py). They live
  under `skill_agents/` for historical reasons but are part of the
  live transferable-skill substrate (root [`readme.md`](../readme.md)
  §"Generalizer — proposing transferable templates"). Do not delete
  them.
- The two pending bridges (`skill_bank/legacy_bridge.py` and
  `decision_agents.skill_interface.HarnessSkillProvider`) listed under
  *Bridge from this module into the new pipeline* above. Until they
  land, the legacy COS-PLAY Actor + bank stays self-contained, and the
  live build stays self-contained.

### What to do about the overlap

Three options, in increasing order of effort:

| Option | What it means | When to pick it |
| --- | --- | --- |
| **Leave as-is** | Treat `skill_agents/` as deferred Qwen 8B/32B/72B reference. It does not run unless `VLM_AGENT_BACKBONE_MODEL=Qwen/...` is set or a `scripts/qwen3_*` entrypoint is invoked. | Default. The banner at the top of this file already warns readers off it. |
| **Land the bridges** | Implement `skill_bank/legacy_bridge.py` and `HarnessSkillProvider`. The legacy bank's records become readable as `SkillRecord` in the live flow; the legacy Actor consumes the live harness. | When you actually want to migrate Stage-3 records out of the legacy JSONL or rewire the legacy Actor. Tracked in [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md). |
| **Delete the overlap** | Move `skill_template.py` + `extract_transferable.py` under `crafter/` (they are already its dependencies), then archive the rest of `skill_agents/` outside the repo or under a `legacy/` namespace. | When the deferred Qwen GRPO tracks are dropped from the project plan altogether. |

For the live pipeline, the short answer to *"is `skill_agents/` still
performing skill-bank management and promoting skills?"* is **no, not
into the live build**. The legacy pipeline can still do those things
inside its own JSONL bank when invoked through a Qwen entrypoint, but
none of those writes can reach the live `skill_bank/` stores or the
gate.

---

## Legacy reference (COS-PLAY GRPO Skill Bank Pipeline)

> Everything below describes the **deferred Qwen3-8B GRPO track**. It remains
> the source of truth for `skill_agents/lora/`, `scripts/qwen3_*.py`,
> `inference/run_qwen3_8b_eval.py`, and the per-game extraction scripts under
> `extract_skillbank/`. None of it runs by default — to re-enable, pass
> `model="Qwen/Qwen3-8B"` explicitly or set `VLM_AGENT_BACKBONE_MODEL`.

Build and maintain a **Skill Bank** from long-horizon game trajectories: segment trajectories into skills, learn symbolic contracts (effects), and serve queries for the [decision_agents](../decision_agents/README.md) VLM agent. The pipeline supports **GRPO-trained LoRA adapters** that wrap existing LLM call points: each call produces G samples, is scored with CPU-only rewards, and the best sample is returned so the EM pipeline runs unchanged while adapters improve over time.

**Model convention (legacy section):** The deferred GRPO track described
below uses **Qwen3-8B** for all skill-bank components (vLLM serving, LoRA
adapters, boundary/protocol/contract/curator calls). All configs and code
references in this section keep `Qwen3-8B` for back-compat. The live build
uses **`Qwen/Qwen3.5-9B`** (`BACKBONE_MODEL`) instead — see the banner at
the top of this file.

**Model-agnostic design:** The pipeline and decision agent use the same skill-bank functions regardless of LLM backend. GPT and Qwen differ only in which API `ask_model` calls; set `PipelineConfig.llm_model` and/or `extractor_model` (e.g. `Qwen/Qwen3.5-9B` for the live actor backbone, `Qwen/Qwen3-8B` for the deferred track, or `gpt-5.5-mini` for an SFT-teacher path) for protocol synthesis and boundary proposal.

**Reasoning-model compatibility:** When using Qwen3 or other reasoning models that emit `<think>` blocks, all LLM call sites are wrapped via [`_llm_compat.py`](_llm_compat.py): prompts get `/no_think` appended and responses are stripped of think tags. See [Reasoning-model compatibility](#reasoning-model-compatibility) below.

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

New skills enter through **proto-skill staging:** `__NEW__` clusters → materialize → proto-skill in staging → verify → promote to real bank.

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
    env_name="llm+avalon",          # or "llm", "avalon", etc.
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
agent.ingest_episodes(episodes, env_name="llm+avalon")

# Optional: iterate until stable (Stage 3 → Stage 4 → materialize NEW)
agent.run_until_stable(max_iterations=3)

agent.save()
```

### 2. Query skills (for decision_agents or scripts)

```python
# Natural-language key (scene / objective / entities)
results = agent.query_skill("propose team and vote on quest", top_k=3)
for r in results:
    print(r["skill_id"], r["score"], r.get("micro_plan"))

# Rich skill selection (preferred for decision agents):
# separates retrieval relevance from execution applicability
results = agent.select_skill(
    query="propose team for quest",
    current_state={"is_leader": True, "quest_round": 2},
    top_k=3,
)
for r in results:
    print(r["skill_id"], r["relevance"], r["applicability"], r["confidence"])

# Effect-based: find skills that achieve desired state changes
results = agent.query_by_effects(
    desired_add={"team_proposed", "quest_active"},
    desired_del={"waiting_for_leader"},
    top_k=3,
)

# List all skills
summary = agent.list_skills()

# Full detail for one skill
detail = agent.get_skill_detail("propose_team")
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
    tool_args={"key": "propose team and vote on quest"},
    context_observation="leader phase, round 2, score 1-0",
    outcome_observation="team approved, quest succeeded",
    skill_bank=bank,
    retrieved_skill_id="propose_team",
    retrieved_result={"skill_id": "propose_team", "score": 0.85},
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
results = engine.query("propose team for quest", top_k=3)
detail = engine.get_detail("propose_team")
list_all = engine.list_all()

# Rich skill selection (preferred for decision agents):
# separates retrieval relevance from execution applicability
selections = engine.select(
    query="propose team for quest",
    current_state={"is_leader": True, "quest_round": 2},
    top_k=3,
)
for s in selections:
    print(s.skill_id, s.relevance, s.applicability, s.confidence)
    print("  matched:", s.matched_effects, "missing:", s.missing_effects)

# Format expected by decision_agents run_tool(QUERY_SKILL)
decision_result = engine.query_for_decision_agent(
    "propose team",
    current_state={"is_leader": True},  # optional: enables applicability scoring
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
| `env_name` | `"llm"` | Signal extraction: `"llm"`, `"llm+avalon"`, `"avalon"`, etc. |
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

**GRPO:** Per-stage hyperparameters (group_size, clip_ratio, kl_coeff, lr) live in `trainer/common/configs/skillbank_grpo.yaml`.

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
| Tetris | 4 skills | 4 skills (maintained) |

### Two-vocabulary segmentation (legacy `[CLEAR]` + cross-domain `[COMMIT]`)

The `[TAG]` parser at `boundary_proposal/signal_extractors.py::parse_intention_tag`
recognises **both** intention vocabularies side-by-side, with no
config flag or per-corpus switch:

* `SUBGOAL_TAGS` — the legacy 13-token game-tactical alphabet
  (`SETUP / CLEAR / MERGE / ATTACK / DEFEND / NAVIGATE / POSITION /
  COLLECT / BUILD / SURVIVE / OPTIMIZE / EXPLORE / EXECUTE`) emitted by
  `decision_agents.agent_helper.infer_intention` for env_wrappers
  episodes.
* `INTENT_OPERATORS` — the new 6-token cross-domain alphabet
  (`INSPECT / TRACK / COMPARE / COMMIT / VERIFY / RECOVER`) emitted by
  `labeling/label_intentions_gpt54.py` for gym-v Temporal ROMs and any
  other corpus where the actor writes free-form English without a
  `[TAG]` prefix. Future-aligned with the two-MDP inner-hop alphabet
  (see `decision_agents/README.md`).

Concretely:

* `parse_intention_tag(intention)` returns the canonical token from
  whichever vocabulary matched (so a 2048 episode produces
  `"endgame:MERGE"` compound labels while a gym-v episode produces
  `"opening:COMMIT"` / `"endgame:RECOVER"`); the rest of the pipeline
  treats them identically.
* `extract_skillbank_grpo_gpt54.py` accepts both via the union frozenset
  `_SUBGOAL_TAG_SET = frozenset(SUBGOAL_TAGS) | frozenset(INTENT_OPERATORS)`,
  so a single skill bank can mix segments tagged in either vocabulary.
* `labeling/unify_skill_index.py` aggregates over both corpora and
  records the originating `corpus` (`gym_v` vs `env_wrappers`) on every
  skill row, so the cross-corpus consumer can filter or rebalance by
  vocabulary if needed.

Phase × tag compounding still works for either vocabulary
(`temporal_thirds` is the generic fallback when the corpus lacks a
game-specific phase extractor — gym-v Temporal ROMs always hit this
path), so a gym-v skill seed looks like `early:INSPECT`,
`mid:COMMIT`, `late:RECOVER` and slots into the same decoder
machinery.

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

The extraction script `extract_skillbank_grpo_gpt54.py` supports three modes. A shell wrapper `extract_skillbank/run_extract_skillbank_grpo.sh` sets up `PYTHONPATH` and checks required environment variables automatically.

### Mode 1: API-based LLM teacher (default — GPT-5.4)

```bash
export OPENROUTER_API_KEY="sk-or-..."
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

# All games
python -m skill_agents.extract_skillbank.extract_skillbank_grpo_gpt54

# Quick test: one episode per game
python -m skill_agents.extract_skillbank.extract_skillbank_grpo_gpt54 --one_per_game -v

# Specific games
python -m skill_agents.extract_skillbank.extract_skillbank_grpo_gpt54 --games twenty_forty_eight tetris

# Via shell wrapper (handles PYTHONPATH + API key check)
bash extract_skillbank/run_extract_skillbank_grpo.sh --games tetris
bash extract_skillbank/run_extract_skillbank_grpo.sh --one_per_game -v
bash extract_skillbank/run_extract_skillbank_grpo.sh --dry_run
```

### Mode 2: Local model as LLM teacher (Qwen3-8B, no GRPO)

Uses Qwen3-8B (or any HuggingFace model) for all LLM teacher calls instead of the API. The model is loaded once and registered as the shared instance; all LLM call sites auto-discover it.

```bash
python -m skill_agents.extract_skillbank.extract_skillbank_grpo_gpt54 \
    --local_model Qwen/Qwen3-8B \
    --games twenty_forty_eight --one_per_game -v
```

### Mode 3: Local model + GRPO training

Enables the full GRPO loop: for each LLM teacher call, G candidate ranking sets are sampled at elevated temperature, each is evaluated by rebuilding the scorer and running the Viterbi decoder, the best is returned to the pipeline, and all samples are stored for LoRA fine-tuning. After each game, a GRPO training step updates the LoRA adapters.

```bash
python -m skill_agents.extract_skillbank.extract_skillbank_grpo_gpt54 \
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

During co-evolution training, the skill bank pipeline runs with **GRPO wrappers** active. The co-evolution orchestrator (`trainer/coevolution/orchestrator.py`) drives the loop; GRPO training lives in `trainer/coevolution/grpo_training.py`.

### Co-evolution loop (wrapper architecture)

1. **Decision agent GRPO** (existing): collect rollouts, update full policy (including skill selection).
2. At bank-update cadence: trajectories are ingested and the **EM pipeline runs with GRPO wrappers**:
   - **Stage 2:** `collect_segment_preferences()` → SEGMENT wrapper (G samples, `SegmentationDiagnostics` reward, buffer, best returned).
   - **Stage 3:** `llm_summarize_contract()` → CONTRACT wrapper (G samples, `verify_effects_contract` reward, buffer, best returned).
   - **Stage 4:** `propose_candidates()` → `filter_candidates()` (CURATOR wrapper) → `execute_approved()`. Filter: G samples, `bank_quality_delta` reward, buffer, best returned.
3. Infrastructure steps (no GRPO): `run_sub_episode_quality_check()`, `distill_execution_hints()`, `update_protocols()`.
4. **Phase 2:** `GRPOLoRATrainer.train_step(grpo_buffer)` — one gradient step per adapter; buffer cleared.
5. SkillEval gating: if bank quality drops, rollback; otherwise bank is hot-swapped into workers.

Shared infrastructure: `GRPOCallWrapper`, `GRPOBuffer`, `GRPOLoRATrainer`; `MultiLoraSkillBankLLM.log_probs()` for training.

### Segmentation persistence

The `SegmentationStore` keeps per-trajectory segmentations updated during training. When EM re-segments with a newer bank, the store replaces the old entry so replay/evaluation/visualization see the latest segmentation.

### Tool-call reward in training

`tool_call_reward` is integrated via the co-evolution episode runner (`trainer/coevolution/episode_runner.py`). For QUERY_SKILL, QUERY_MEM, CALL_SKILL the reward includes `r_tool`:

`r_total = r_env + w_follow × r_follow + r_cost + tool_call_reward_weight × r_tool`

See [trainer/README.md](../trainer/README.md) for co-evolution setup.

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
from skill_agents.coldstart_io import record_io, ColdStartRecord, flush, reset

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

### Legacy (this folder)

- [boundary_proposal/README.md](boundary_proposal/README.md) — Stage 1 signals and `segment_episode` / `propose_from_episode`.
- [infer_segmentation/README.md](infer_segmentation/README.md) — Stage 2 preference learning and decoders.
- [infer_segmentation/OVERVIEW.md](infer_segmentation/OVERVIEW.md) — Stage 2 architectural overview.
- [stage3_mvp/README.md](stage3_mvp/README.md) — Stage 3 effects-only contract learning.
- [bank_maintenance/README.md](bank_maintenance/README.md) — Stage 4 split/merge/refine.
- [skill_bank/README.md](skill_bank/README.md) — Legacy persistent storage and NEW pool management (single JSONL `SkillBankMVP`).
- [skill_evaluation/README.md](skill_evaluation/README.md) — Quality assessment.
- [lora/README.md](lora/README.md) — Multi-LoRA model (Qwen3-8B + adapters).
- [PIPELINE_CALL_FLOW.md](PIPELINE_CALL_FLOW.md) — How each function is called from the agent framework.

### New pipeline (top-level packages — preferred)

- [`../harness/README.md`](../harness/README.md) — Per-invocation skill runtime, eligibility filter, adapters, replay validator, reward logger. *Phase A MVP delivered.*
- [`../orchestrator/README.md`](../orchestrator/README.md) — End-to-end control plane: `EpisodeRunner`, `ArtifactStore`, `BudgetController`, `GateService`, `PromotionOrchestrator`, `SnapshotManager`. *Phase B MVP delivered.*
- [`../crafter/README.md`](../crafter/README.md) — Slow-timescale typed proposal layer: composer, generalizer, hypothesizer, failure diagnoser, failure memory. Outputs **DRAFT only**. *Phase C MVP delivered.*
- [`../skill_bank/`](../skill_bank/) — New split-storage skill bank (`draft / candidate / active / archive`) with `SkillLifecycleManager` as the only writer. *Replaces this folder's `skill_bank/bank.py` for the live build.*
- [`../common/`](../common/) — Canonical enums, ID helpers, `<state>` schema, `BACKBONE_MODEL` registry.
- [`../data_structure/extensions/`](../data_structure/extensions/) — `SkillEpisode`, `SkillRecord`, `GateVerdict`, `SkillEvaluationRecord`, `BankMutationProposal`, `FailureTrace`, `RunRelease`.
- [`../plans/README.md`](../plans/README.md) — Full plan corpus (`00-system` … `09-implementation`, plus **`legacy/`** for finished edit passes) that the new pipeline implements.
- [`../IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) — Live delivered/pending tracker per Phase A → F.

---

## File layout

```
skill_agents/
├── README.md                 # This file (legacy; new pipeline lives at repo root)
├── PLAN.md                   # SkillBank Agent operating plan (legacy)
├── PIPELINE_CALL_FLOW.md     # How each function is called from the agent framework
├── __init__.py               # SkillBankAgent, SkillQueryEngine, NewPoolManager, TransferableSkill*, etc.
├── _llm_compat.py            # Reasoning-model compatibility (strip_think_tags, /no_think wrapper)
├── _llm_retry.py             # LLM retry wrapper with backoff
├── coldstart_io.py           # Centralized cold-start I/O recording for all GRPO-connected functions
├── default_predicates.py     # Shared default predicate extractor (no-op fallback)
├── pipeline.py               # SkillBankAgent orchestrator (contract feedback, NEW pool, proto-skill flow)
├── query.py                  # SkillQueryEngine + SkillSelectionResult (retrieval + selection policy)
├── tool_call_reward.py       # Reward for tool calls (agentic RL)
├── skill_template.py         # BRIDGE → new pipeline: TransferableSkill, SlotBinding,
│                             # ReasoningProtocol, HopStep, AbstractPredicate, FAMILY_PROTOCOLS.
│                             # Domain-agnostic skill representation over inner-MDP actions
│                             # {GROUND, CHECK, RETRIEVE, CONCLUDE, EXECUTE} and shared slots
│                             # {target, blocker, constraint, candidate_set, history_anchor}.
├── extract_transferable.py   # BRIDGE → new pipeline: TransferableSkillExtractor,
│                             # extract_transferable_skills(). Five-stage extractor (predicate
│                             # normalisation → structural clustering → template abstraction →
│                             # transferability scoring → export) that mines per-game
│                             # SkillBankMVP banks for cross-domain reusable templates.
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
├── skill_evaluation/         # Quality evaluation
│   ├── run_evaluation.py     # run_skill_evaluation()
│   ├── evaluators.py         # evaluate_coherence(), evaluate_discriminability(), etc.
│   ├── config.py             # SkillEvaluationConfig
│   ├── schemas.py            # EvaluationSummary, SkillQualityReport
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
    ├── run_extract_skillbank_grpo.sh    # Shell wrapper (PYTHONPATH + env var setup)
    └── output/               # Per-game outputs (skill_bank.jsonl, catalogs, cold-start I/O)
```
