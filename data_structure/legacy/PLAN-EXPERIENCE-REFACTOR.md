# PLAN: Refactor `experience.py` for a No-Memory, Evidence-Grounded Skill Runtime

**Scope.** Refactor `data_structure/experience.py` from a mostly COS-PLAY-style transition buffer into a **no-memory runtime data structure** that supports transferable skills across game / webagent / os-agent / video-understanding / visual reasoning domains, with **short-video evidence-grounded reasoning** as the first execution focus.

**Upstream invariants this plan honors.**
- **No-memory contract** — no episodic / semantic memory subsystem, no memory retrieval APIs, no `MemoryRetrievalRequest`. All runtime context lives **inside** `Experience` / `Episode` / `SubTask_Experience`. See [`../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md` §4](../../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) ("Evidence & trace bookkeeping (no-memory contract)") and [`../plans/legacy/10-edits/PLAN-EDITS-HARNESS-CONTROL-PLANE.md` Revision note 1](../../plans/legacy/10-edits/PLAN-EDITS-HARNESS-CONTROL-PLANE.md).
- **Evidence-driven invariant** — every reasoning step must carry `evidence_in ∪ evidence_out ≠ ∅` and an `evidence_role ∈ {GATHER, VERIFY, REASON, COMMIT}` (mapped here onto `step_type`). See [`../plans/03-skill-bank/PLAN-SKILL-BANK.md` §0.3](../../plans/03-skill-bank/PLAN-SKILL-BANK.md) and [`../plans/legacy/10-edits/PLAN-EDITS-HARNESS-CONTROL-PLANE.md` Revision note 2](../../plans/legacy/10-edits/PLAN-EDITS-HARNESS-CONTROL-PLANE.md).
- **Inner primitives unchanged** — `GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE` are the canonical typed inner-loop step kinds (with optional `PRIMITIVE` for raw low-level actions).

**Non-goals.**
- No new modules, no new agents, no new buffers beyond extending the existing ones.
- No introduction of episodic / semantic memory, RAG store, or memory query API.
- No long-horizon video assumptions; first-track is short-video (Video-Holmes-style) evidence-grounded reasoning.
- No renaming of `<state>` slots or inner primitives.

---

## 1. Goal

Turn `experience.py` into the runtime substrate that explicitly carries:

1. A **short typed trace** of inner-hop reasoning steps (`GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE | PRIMITIVE`).
2. An **evidence ledger** — every step can declare what evidence it used, what claim it supports, and whether the claim is `candidate | verified | contradicted | insufficient`.
3. **Active skill execution context** — which skill is in flight, what phase it is in, and how its contract is progressing.
4. **Transfer / verification metadata** on segmented sub-trajectories so the skill bank can reuse them as skill candidates with provenance.

Episode-local only. **Not** a long-horizon memory agent.

---

## 2. Design principles

| Principle | Statement |
|-----------|-----------|
| **A. No memory subsystem** | Do not introduce episodic / semantic memory, memory retrieval APIs, or memory write/query structures. All relevant runtime context lives inside `Experience` / `Episode` / `SubTask_Experience`. |
| **B. Evidence is first-class** | Every reasoning step can say *what evidence it used*, *what claim it supports*, and *whether that claim is verified, contradicted, or still insufficient*. |
| **C. Skills are transferable** | Sub-trajectories carry `source_domain`, `candidate_target_domains`, `verified_domains`, `adapter_id`, `transfer_status`, `failure_mode`. |
| **D. Episode-local, not long-horizon** | This repo only maintains the current rollout trace, the local evidence chain, and active skill execution state. |

---

## 3. Proposed roles of each class

### 3.1 `Experience`
A single typed reasoning/control step. **Not** only a `state → action → next_state` transition. It is simultaneously:
- one atomic step in a reasoning/control trajectory,
- one unit of evidence usage,
- one unit of claim progression.

### 3.2 `Episode`
Full rollout / trajectory container. The carrier of:
- the episode-level trace (ordered `Experience`s),
- final answer + answer-support chain,
- aggregated reward, claims, and evidence bookkeeping.

### 3.3 `SubTask_Experience`
A segmented local trajectory that may become a:
- skill candidate,
- verification unit,
- transfer candidate,
- failure-reflection unit.

---

## 4. Refactor plan for `Experience`

Anchor: current implementation is `Experience.__init__` at `experience.py:24-72` and `to_dict` / `from_dict` at `experience.py:176-226`.

### 4.1 Keep these existing fields

These cross-domain fields are still useful and stay as-is:

`state`, `action`, `reward`, `next_state`, `done`, `raw_state`, `raw_next_state`, `available_actions`, `reward_details`, `interface`.

### 4.2 Replace old `action_type` semantics

Currently `action_type` (`experience.py:70-72`) accepts `"primitive" | "QUERY_MEM" | "QUERY_SKILL" | "CALL_SKILL"`. The `QUERY_MEM` value violates the no-memory contract. Replace `action_type` with a new typed-step vocabulary stored on a new field `step_type`:

| `step_type` | Meaning |
|-------------|---------|
| `GROUND` | localize the relevant entity / region / control / frame / moment |
| `CHECK` | verify a claim or constraint against evidence |
| `RETRIEVE` | retrieve a reusable skill or protocol from the skill bank (no memory store) |
| `COMMIT` | finalize an intermediate belief or final answer candidate |
| `EXECUTE` | emit a domain action or output |
| `PRIMITIVE` | direct low-level action when no typed step applies |

`action_type` is **kept temporarily as a legacy alias** (mirrors `step_type` on read; emits a deprecation warning when set to `QUERY_MEM`). Removed in Phase 3.

### 4.3 Add domain-general runtime fields

Add to `Experience.__init__`:

| Field | Type | Notes |
|-------|------|-------|
| `domain_type` | `Literal["game","web","os","video","visual_reasoning"]` | Required at construction time. |
| `task_type` | `Optional[str]` | Free-form within domain (e.g. `"video_holmes_qa"`, `"web_form_fill"`). |
| `step_type` | `Optional[str]` | One of §4.2. |
| `active_skill` | `Optional[str]` | `skill_id` of the skill currently being invoked, if any. |
| `skill_phase` | `Optional[str]` | Phase within that skill's contract (e.g. `"locate"`, `"compose"`, `"verify"`). |
| `trace_parent_id` | `Optional[str]` | `idx` (or UUID) of the parent step when the inner-hop loop recurses. Lets the orchestrator render parent/child spans. |

### 4.4 Add evidence fields

These implement the evidence ledger required by the evidence-driven invariant:

| Field | Type | Notes |
|-------|------|-------|
| `evidence_refs` | `List[str]` | IDs of evidence items consumed by this step (`evidence_in`). |
| `evidence_items` | `List[dict]` | New evidence objects produced or attached at this step (`evidence_out`). Each item has `{evidence_id, kind, source, payload, confidence}`. |
| `supports_claims` | `List[str]` | Claim IDs this step supports (or refutes when paired with `claim_status`). |
| `claim_status` | `Optional[Literal["candidate","verified","contradicted","insufficient"]]` | Status this step assigns to its supported claim(s). |
| `evidence_confidence` | `Optional[float]` | Aggregate confidence of the supporting evidence in `[0, 1]`. |

**Invariant check (lightweight, in-class).** When `step_type ∈ {GROUND, CHECK, REASON, COMMIT}` (mapped to evidence roles `GATHER, VERIFY, REASON, COMMIT`), `Experience.validate_evidence_contract()` must return `True`, i.e. `evidence_refs ∪ evidence_items ≠ ∅`. Failure is surfaced by the Harness gate (G0); this method just exposes the predicate.

### 4.5 Replace long-task framing with skill-execution framing

Current `tasks` and `sub_tasks` (`experience.py:57-58`) are long-horizon-flavored. Add the skill-execution fields and **mark the old ones legacy**:

| New field | Replaces |
|-----------|----------|
| `goal` | `tasks` |
| `subgoal` | `sub_tasks` |
| `active_skill` (also added in §4.3) | the implicit "current strategy" notion |
| `skill_phase` (also added in §4.3) | — |

Backward compatibility: keep `tasks` / `sub_tasks` in `__init__` and `to_dict` / `from_dict` for one release with a deprecation comment. `from_dict` populates `goal` / `subgoal` from the legacy fields when the new fields are absent.

### 4.6 Function-level changes

| Function | Current behavior | New behavior |
|----------|------------------|--------------|
| `generate_summary()` (`experience.py:74-99`) | Game-oriented "strategic note". | Domain-aware summary generator that branches on `domain_type`: game → strategic local move; web → UI interaction / blocker / next control; os → window/object/interaction; video → evidence-bearing temporal event; visual_reasoning → object/region clue summary. |
| `generate_summary_state()` (`experience.py:101-128`) | Compact `key=value` state. | Keep compact format; expose **candidate sets**, **constraints**, **uncertainty**, **evidence anchors**, **current relevant entities** as named keys when present. |
| `generate_intentions()` (`experience.py:130-166`) | Game-heavy `[TAG] subgoal` taxonomy. | Replace tag set with the domain-general reasoning/action vocabulary: `LOCATE | VERIFY | DISAMBIGUATE | SELECT | PLAN | CHECK | GROUND | EXECUTE`. Keep `[TAG] phrase` output shape so downstream consumers don't break. |

---

## 5. Refactor plan for `Episode`

Anchor: `Episode.__init__` at `experience.py:232-263`; `separate_into_sub_episodes` at `experience.py:301-327`.

### 5.1 Keep current container role

`episode_id`, `experiences`, `task`, `metadata`, `summary`, reward aggregation (`get_reward`, `get_total_reward`), outcome aggregation (`set_outcome`).

### 5.2 Add domain-level metadata

| Field | Notes |
|-------|-------|
| `domain_type` | `"game" | "web" | "os" | "video" | "visual_reasoning"`. |
| `benchmark_name` | e.g. `"video_holmes"`, `"webarena"`, `"osworld"`. |
| `input_modality` | `"pixels" | "dom" | "frames" | "text" | "mixed"`. |
| `output_modality` | `"action" | "answer" | "answer+chain"`. |
| `episode_status` | `"running" | "completed" | "aborted" | "failed_contract"`. |

`env_name` / `game_name` (`experience.py:248-251`) are kept as legacy; new code reads `domain_type` + `benchmark_name`.

### 5.3 Add evidence-grounded answer bookkeeping

| Field | Notes |
|-------|-------|
| `final_answer` | The episode's final answer/output (`Optional[Any]`). |
| `root_claims` | `List[str]` of top-level claim IDs whose support resolves to `final_answer`. |
| `answer_support_chain` | Ordered list of `(experience_idx, evidence_refs, claim_id)` triples that walk from grounding → verification → commit for the final answer. |
| `episode_claim_graph` | `dict[claim_id → {status, supported_by_refs, contradicted_by_refs, parent_claims}]`. |

These fields are filled at finalize time by the Action Agent / Harness; this class only provides the container and a `validate_answer_chain()` predicate (returns `True` iff every `root_claim` resolves to `verified` in `episode_claim_graph` with a non-empty support chain).

### 5.4 Expand the role of `metadata`

Use `metadata` explicitly for: `model_version`, `rollout_source`, `adapter_ids`, `transfer_mode`, `budget_stats` (token / time / step budgets consumed), `replay_split` / `partition` info. Add a typed helper `Episode.set_metadata(**kwargs)` that validates these well-known keys without locking out free-form ones.

### 5.5 Rewrite `separate_into_sub_episodes()`

Current implementation (`experience.py:301-327`) collects unique `sub_tasks` values in order and slices. This is fragile: it requires monotone `sub_tasks` changes, doesn't stabilize segment labels, ignores `step_type`, and produces no segment-level evidence bookkeeping.

Rewrite as a **two-stage** routine.

**Stage 1 — identify segment boundaries.** Produce explicit boundary tuples `(start_idx, end_idx, segment_label)` using a configurable boundary policy:

- `policy="subgoal_change"` (default, current behavior generalized): boundary whenever `subgoal` changes (with `sub_tasks` as a legacy fallback).
- `policy="active_skill"`: boundary on `active_skill` changes — produces one segment per skill invocation.
- `policy="commit"`: boundary at every `step_type == "COMMIT"` step.
- `policy="claim_resolution"`: boundary at every transition from `claim_status == "candidate"` to `verified | contradicted`.

Stage 1 returns `List[Tuple[int, int, str]]` and is independently testable.

**Stage 2 — build `SubTask_Experience` from boundaries.** For each `(start_idx, end_idx, label)`:
1. Collect segment experiences `experiences[start_idx:end_idx]`.
2. Collect outcome lookahead `experiences[end_idx:end_idx+outcome_length]`.
3. Bind the stable `segment_label`.
4. Compute evidence stats: `segment_evidence_refs`, `segment_claims`, evidence sufficiency, cumulative reward.
5. Return a `SubTask_Experience` with all transfer / verification fields initialized to defaults (see §6).

Signature: `Episode.separate_into_sub_episodes(outcome_length: int = 5, policy: str = "subgoal_change") -> List[SubTask_Experience]`.

---

## 6. Refactor plan for `SubTask_Experience`

Anchor: current implementation at `experience.py:365-539`.

### 6.1 Keep current segmentation purpose

`sub_task`, `final_goal`, `sub_task_experience`, `outcome_experiences`, `quality_score`, `outcome_classification`, `seg_id`, `episode_id`, `rollout_source`, `summary`, `outcome_summary`, `length`, `cumulative_reward`, `to_sub_episode_ref()` (`experience.py:484-502`).

### 6.2 Upgrade into a skill-candidate unit

Add the following fields with sensible defaults:

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `source_domain` | `str` | inherited from parent `Episode.domain_type` | Where the segment was produced. |
| `candidate_target_domains` | `List[str]` | `[]` | Domains the Skill Crafter has proposed transferring this skill to. |
| `verified_domains` | `List[str]` | `[]` | Domains where transfer has been verified by the Harness. |
| `adapter_id` | `Optional[str]` | `None` | Adapter / transfer artifact id. |
| `transfer_status` | `Literal["none","proposed","verifying","verified","rejected"]` | `"none"` | |
| `verification_status` | `Literal["unverified","verified","contradicted","needs_more_evidence"]` | `"unverified"` | |
| `skill_candidate_type` | `Optional[str]` | `None` | e.g. `"protocol"`, `"primitive_composition"`, `"verifier"`. |
| `failure_mode` | `Optional[str]` | `None` | Aligned with Harness diagnostics: `slot_binding_failed`, `adapter_execution_mismatch`, `evidence_insufficient`, `temporal_mismatch`, `ui_grounding_mismatch`, `desktop_object_mismatch`, `overconfident_commit`, `contract_mismatch`. |

### 6.3 Add segment-level evidence fields

| Field | Type | Notes |
|-------|------|-------|
| `segment_evidence_refs` | `List[str]` | Union of `evidence_refs` across the segment's `Experience`s. |
| `segment_claims` | `List[str]` | Union of `supports_claims` across the segment. |
| `segment_contract_progress` | `dict` | Per-phase progress of the active skill's contract within this segment. |
| `evidence_sufficiency` | `Literal["sufficient","partial","insufficient"]` | Coarse summary used by the Harness gate G0. |

These fields are computed by Stage 2 of the new `separate_into_sub_episodes()` (§5.5) and are recomputable via `SubTask_Experience.recompute_evidence_stats()`.

### 6.4 `to_sub_episode_ref()` updates

Extend the returned `SubEpisodeRef` payload (in `skill_agents.stage3_mvp.schemas`) with:
- `source_domain`, `verified_domains`, `transfer_status`,
- `evidence_sufficiency`, `failure_mode`.

These are additive fields — existing skill-bank consumers continue to work if they ignore unknown keys.

---

## 7. Buffer-level changes

Anchor: `Experience_Replay_Buffer` (`experience.py:545-582`), `Episode_Buffer` (`experience.py:587-690`), `Tool_Buffer` (`experience.py:695-745`).

The current buffers are simple FIFO replay containers. Extend them with **domain-aware retrieval** (no memory subsystem — these remain in-process Python containers).

### 7.1 Filtering and retrieval

Add `filter(...)` / `query(...)` methods that accept any combination of:

- `domain_type`,
- `step_type`,
- `active_skill`,
- `outcome_classification`,
- `quality_score` (range),
- `failure_mode`,
- `verified_domains` (set membership).

Return type: `List[Experience]` for the experience buffer, `List[Episode]` for the episode buffer, `List[SubTask_Experience]` for the tool buffer.

### 7.2 Sampling modes

Extend `sample_*` with a `mode` argument:

| Mode | Behavior |
|------|----------|
| `"uniform"` (default) | Current `random.sample`. |
| `"high_quality"` | Filter by `quality_score >= threshold` then sample uniformly. |
| `"failure_replay"` | Sample where `failure_mode is not None`. |
| `"transfer_success"` | Sample where `transfer_status == "verified"` and `verified_domains` non-empty. |
| `"transfer_failure"` | Sample where `transfer_status == "rejected"` or `failure_mode in {"adapter_execution_mismatch","slot_binding_failed"}`. |
| `"domain_balanced"` | Stratified sampling so each `domain_type` contributes ~equally. |

These directly support: skill bank curation, harness validation, failure reflection, and transfer benchmarking.

### 7.3 Persistence

`Episode_Buffer.save_to_json` / `load_from_json` (`experience.py:639-690`) keep working. Bump `to_dict()` to include the new `Episode` and `Experience` fields (gracefully via `dict.get(...)` defaults in `from_dict`).

---

## 8. How this encodes the no-memory plan

Instead of a memory subsystem we use:

| Mechanism | Where it lives |
|-----------|----------------|
| **A. Short typed trace** | The ordered `Experience` list inside `Episode.experiences`, with `step_type` typing every entry. |
| **B. Evidence ledger** | `Experience.evidence_refs` / `evidence_items` / `supports_claims` / `claim_status` plus `Episode.answer_support_chain` and `episode_claim_graph`. |
| **C. Active skill context** | `Experience.active_skill` / `skill_phase` plus segment-level `SubTask_Experience` skill-candidate metadata and `segment_contract_progress`. |

This gives structured local context, verifiable evidence chains, and transferable skill units **without** any memory subsystem.

---

## 9. Implementation order

### Phase 1 — Minimal structural cleanup (lands first; unblocks the Harness G0 gate)
1. Replace old `action_type` values; add `step_type` field and the `QUERY_MEM` deprecation alias.
2. Add `domain_type`, `task_type`, `active_skill`, `skill_phase`, `trace_parent_id` to `Experience`.
3. Add `evidence_refs`, `evidence_items`, `supports_claims`, `claim_status`, `evidence_confidence` + `validate_evidence_contract()` to `Experience`.
4. Rewrite `Episode.separate_into_sub_episodes()` as the two-stage routine of §5.5 (default `policy="subgoal_change"` keeps current behavior).
5. Update `to_dict` / `from_dict` on all three classes to round-trip new fields.

### Phase 2 — Make it transfer-aware
1. Add `source_domain`, `candidate_target_domains`, `verified_domains`, `adapter_id`, `transfer_status`, `verification_status`, `skill_candidate_type`, `failure_mode` to `SubTask_Experience`.
2. Extend buffers with the §7.1 filtering and §7.2 sampling modes.
3. Add `domain_type`, `benchmark_name`, `input_modality`, `output_modality`, `episode_status` to `Episode` and the typed `Episode.set_metadata` helper.

### Phase 3 — Make it skill-bank ready
1. Add `final_answer`, `root_claims`, `answer_support_chain`, `episode_claim_graph` + `validate_answer_chain()` to `Episode`.
2. Add segment-level evidence fields (`segment_evidence_refs`, `segment_claims`, `segment_contract_progress`, `evidence_sufficiency`) and `recompute_evidence_stats()` to `SubTask_Experience`.
3. Extend `to_sub_episode_ref()` payload with the new transfer / evidence fields.
4. Remove the legacy `action_type` field and the legacy `tasks` / `sub_tasks` slots (after one release).

---

## 10. Field-by-field cheat sheet (for the patch)

### `Experience` — additions / replacements
```text
+ domain_type:           Literal["game","web","os","video","visual_reasoning"]
+ task_type:             Optional[str]
+ step_type:             Optional[Literal["GROUND","CHECK","RETRIEVE","COMMIT","EXECUTE","PRIMITIVE"]]
+ active_skill:          Optional[str]
+ skill_phase:           Optional[str]
+ trace_parent_id:       Optional[str]
+ evidence_refs:         List[str]              = []
+ evidence_items:        List[dict]             = []
+ supports_claims:       List[str]              = []
+ claim_status:          Optional[Literal["candidate","verified","contradicted","insufficient"]]
+ evidence_confidence:   Optional[float]
+ goal:                  Optional[str]          # replaces tasks
+ subgoal:               Optional[str]          # replaces sub_tasks
~ action_type:           legacy alias of step_type (DeprecationWarning on QUERY_MEM)
- (eventually remove)    tasks, sub_tasks, action_type
```

### `Episode` — additions
```text
+ domain_type, benchmark_name, input_modality, output_modality, episode_status
+ final_answer, root_claims, answer_support_chain, episode_claim_graph
+ set_metadata(**kwargs)                 # validates well-known keys
+ validate_answer_chain() -> bool
~ separate_into_sub_episodes(outcome_length=5, policy="subgoal_change")
```

### `SubTask_Experience` — additions
```text
+ source_domain, candidate_target_domains, verified_domains
+ adapter_id, transfer_status, verification_status
+ skill_candidate_type, failure_mode
+ segment_evidence_refs, segment_claims, segment_contract_progress, evidence_sufficiency
+ recompute_evidence_stats()
~ to_sub_episode_ref()                   # extend payload with the new fields
```

### Buffers — additions
```text
+ Experience_Replay_Buffer.filter(**criteria), .query(...), .sample_experience(batch_size, mode=...)
+ Episode_Buffer.filter(...), .sample_episode(batch_size, mode=...)
+ Tool_Buffer.filter(...), .sample_tool(batch_size, mode=...)
```

---

## 11. Verification checklist

After Phase 1 lands:
- [ ] `rg "QUERY_MEM" Multi-hop-Reasoning-VLM-Agent/` returns only the deprecation alias path.
- [ ] Every `Experience` produced by the Action Agent in a Video-Holmes rollout has `domain_type == "video"` and a non-`None` `step_type`.
- [ ] For every `Experience` with `step_type ∈ {GROUND, CHECK, COMMIT}`, `validate_evidence_contract()` returns `True`.
- [ ] `Episode.separate_into_sub_episodes(policy="subgoal_change")` reproduces the previous segmentation on a fixture episode.
- [ ] `Episode.to_dict()` round-trips through `Episode.from_dict()` without information loss for all new fields.

After Phase 2:
- [ ] `Experience_Replay_Buffer.sample_experience(batch_size=8, mode="failure_replay")` returns only experiences whose owning `SubTask_Experience` has a non-`None` `failure_mode`.
- [ ] `Episode_Buffer.filter(domain_type="video", benchmark_name="video_holmes")` returns the expected subset on a mixed-domain fixture.

After Phase 3:
- [ ] For every successful Video-Holmes episode, `Episode.validate_answer_chain()` returns `True`, and `episode_claim_graph[root_claim].status == "verified"`.
- [ ] `to_sub_episode_ref()` payload includes `evidence_sufficiency` and `transfer_status`; the Skill Bank consumer reads them without code changes (additive contract).

---

## 12. One-paragraph project description

Refactor `experience.py` from a COS-PLAY-style transition buffer into a **no-memory, evidence-grounded trajectory structure**. In the new design, `Experience` represents typed reasoning/control steps (`GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE | PRIMITIVE`) with explicit evidence and claim fields; `Episode` carries rollout-level trace, final-answer, and answer-support-chain bookkeeping; and `SubTask_Experience` serves as the intermediate unit for skill extraction, transfer validation, and failure-aware refinement across game, webagent, os-agent, video-understanding, and visual reasoning tasks. The refactor lands in three phases — structural cleanup, transfer-awareness, then skill-bank readiness — and explicitly avoids introducing any episodic / semantic memory subsystem.
