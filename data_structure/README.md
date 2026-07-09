# `data_structure/` — No-Memory, Evidence-Grounded Trajectory Substrate

This package defines the runtime data structures for a **no-memory,
evidence-grounded skill runtime** that is shared across game, webagent,
os-agent, video-understanding, and visual-reasoning tasks. Short-video
evidence-grounded reasoning (Video-Holmes-style) is the first execution focus.

The design rationale is in [`legacy/PLAN-EXPERIENCE-REFACTOR.md`](legacy/PLAN-EXPERIENCE-REFACTOR.md)
(archived after the refactor landed); this README documents *what's actually here* so you can use these classes from
the action agent, harness, skill bank, and offline labeling pipelines without
re-reading the plan.

> **Invariants enforced by this package.**
>
> 1. **No memory subsystem.** No episodic / semantic memory store, no memory
>    retrieval API, no `QUERY_MEM` step kind. All runtime context lives inside
>    `Experience` / `Episode` / `SubTask_Experience`.
> 2. **Evidence-driven.** Every reasoning step can carry evidence references,
>    supported claims, and a claim status. `Experience.validate_evidence_contract()`
>    exposes the lightweight predicate the Harness gate G0 enforces.
> 3. **Skills are transferable.** `SubTask_Experience` carries source / target
>    domain, adapter id, transfer + verification status, and failure mode.
> 4. **Episode-local.** Nothing here grows beyond the current rollout.

---

## 1. Module overview

| Class | One-line purpose | File anchor |
|-------|------------------|-------------|
| `Experience` | One typed reasoning/control step. | `experience.py` |
| `Episode` | Full rollout: typed trace + answer support chain. | `experience.py` |
| `SubTask_Experience` | Segmented sub-trajectory; skill-candidate unit. | `experience.py` |
| `Experience_Replay_Buffer` | FIFO buffer of individual experiences with filter/query/sample modes. | `experience.py` |
| `Episode_Buffer` | FIFO buffer of complete episodes; JSON persistence. | `experience.py` |
| `Tool_Buffer` | FIFO buffer of `SubTask_Experience` skill candidates. | `experience.py` |

Three module-level constants are also exported:

```python
from data_structure.experience import STEP_TYPES, DOMAIN_TYPES, CLAIM_STATUSES
```

| Constant | Values |
|----------|--------|
| `STEP_TYPES` | `("GROUND", "CHECK", "RETRIEVE", "COMMIT", "EXECUTE", "PRIMITIVE")` |
| `DOMAIN_TYPES` | `("game", "web", "os", "video", "visual_reasoning")` |
| `CLAIM_STATUSES` | `("candidate", "verified", "contradicted", "insufficient")` |

---

## 2. `Experience` — one typed reasoning/control step

Beyond the classic `state → action → next_state` transition, an `Experience`
is also one unit of evidence usage and one unit of claim progression.

### 2.1 Field reference

| Group | Field | Type | Notes |
|-------|-------|------|-------|
| **Core transition** | `state`, `action`, `reward`, `next_state`, `done` | mixed | Required at construction. |
| | `sub_task_done` | `Optional[bool]` | Set externally when a sub-task ends mid-episode. |
| | `idx` | `Optional[int]` | Index inside the owning `Episode`; assigned by the rollout collector. |
| | `raw_state`, `raw_next_state` | `Optional[Any]` | Pre-NL-conversion observations. |
| | `available_actions` | `Optional[List[str]]` | Valid actions at this step. |
| | `interface` | `Optional[dict]` | External evaluator configs / results. |
| | `reward_details` | `Optional[dict]` | `{r_env, r_follow, r_cost, r_total}`. |
| **Domain context** | `domain_type` | `Optional[str]` | One of `DOMAIN_TYPES`. |
| | `task_type` | `Optional[str]` | Free-form within domain (e.g. `"video_holmes_qa"`). |
| | `step_type` | `Optional[str]` | One of `STEP_TYPES`. **Property with validation.** |
| | `active_skill` | `Optional[str]` | `skill_id` of the skill currently in flight. |
| | `skill_phase` | `Optional[str]` | Phase within that skill's contract (`"locate"`, `"verify"`, …). |
| | `trace_parent_id` | `Optional[str]` | Parent step id for inner-hop recursion. |
| **Skill-execution framing** | `goal` | `Optional[str]` | Replaces the legacy long-horizon `tasks`. |
| | `subgoal` | `Optional[str]` | Replaces the legacy `sub_tasks`. |
| **Evidence ledger** | `evidence_refs` | `List[str]` | IDs of evidence consumed (`evidence_in`). |
| | `evidence_items` | `List[dict]` | New evidence produced or attached (`evidence_out`); each item is `{evidence_id, kind, source, payload, confidence}`. |
| | `supports_claims` | `List[str]` | Claim IDs this step supports / refutes. |
| | `claim_status` | `Optional[str]` | One of `CLAIM_STATUSES`. |
| | `evidence_confidence` | `Optional[float]` | Aggregate confidence in `[0, 1]`. |
| **Summarization** | `summary`, `summary_state`, `intentions` | `Optional[str]` | Filled lazily by the helpers below. |
| **Legacy** | `tasks`, `sub_tasks`, `action_type` | mixed | Kept for back-compat. New code should use `goal` / `subgoal` / `step_type`. |

### 2.2 The `step_type` property

`step_type` is a Python property with a validating setter:

- Accepts only values from `STEP_TYPES`. Unknown values raise a
  `UserWarning` but are still stored (forward-compat for new step kinds).
- Setting `step_type = "QUERY_MEM"` raises a `DeprecationWarning` and is
  **silently dropped** — this enforces the no-memory contract.
- Whenever `step_type` is set, the legacy `action_type` is mirrored to the
  same value so older consumers keep working.

### 2.3 The evidence-driven invariant

```python
exp = Experience(state=..., action=..., reward=..., next_state=..., done=False,
                 step_type="CHECK", evidence_refs=["frame_42"])
assert exp.validate_evidence_contract() is True
```

`validate_evidence_contract()` returns `True` iff:

- `step_type ∉ {GROUND, CHECK, COMMIT}`, **or**
- `evidence_refs ∪ evidence_items ≠ ∅`.

`RETRIEVE`, `EXECUTE`, and `PRIMITIVE` steps are not required to carry
evidence at this layer. The Harness G0 gate is what raises the gate-level
violation when this returns `False`.

### 2.4 Summarization helpers

| Method | Behavior |
|--------|----------|
| `generate_summary_state()` | Compact `key=value` snapshot. Tries `decision_agents.agent_helper.build_rag_summary` first (0 LLM tokens); falls back to an LLM prompt that asks for `candidates=`, `constraints=`, `uncertainty=`, `evidence_anchors=`, `entities=` keys when present. |
| `generate_summary()` | Domain-aware short note. Branches on `domain_type` (game / web / os / video / visual_reasoning) for the focus phrase. |
| `generate_intentions(history=None)` | Produces a `[TAG] phrase` string using the domain-general tag set: `LOCATE | VERIFY | DISAMBIGUATE | SELECT | PLAN | CHECK | GROUND | EXECUTE`. |
| `initialize_intentions_and_summary(history=None)` | Idempotent helper that fills `intentions` and `summary` if missing. |

### 2.5 Serialization

`to_dict()` only emits the new fields when they carry information (keeps
JSON compact and back-compatible). `from_dict()` reads everything via
`.get(...)` so old payloads load fine; legacy `action_type` values that
fall inside `STEP_TYPES` are promoted to `step_type` on load.

---

## 3. `Episode` — full rollout container

Carries the typed trace (ordered `Experience`s), aggregated reward/outcome,
domain-level metadata, and the **answer-grounded support chain** that
explains the final answer.

### 3.1 Field reference

| Group | Field | Notes |
|-------|-------|-------|
| **Identity** | `episode_id` | Auto-generated UUID4 if not provided. |
| | `task` | Human-readable task description. |
| **Legacy domain ids** | `env_name`, `game_name` | Kept for back-compat; new code reads `domain_type` / `benchmark_name`. |
| **Trace** | `experiences: List[Experience]` | The typed trace itself. |
| **Aggregation** | `outcome`, `summary` | Filled by `set_outcome()` / `generate_summary()`. |
| **Domain metadata** | `domain_type`, `benchmark_name` | e.g. `"video"`, `"video_holmes"`. |
| | `input_modality`, `output_modality` | e.g. `"frames"`, `"answer+chain"`. |
| | `episode_status` | `"running" \| "completed" \| "aborted" \| "failed_contract"`. |
| **Answer chain** | `final_answer` | The episode's final answer/output. |
| | `root_claims: List[str]` | Top-level claim IDs whose support resolves to `final_answer`. |
| | `answer_support_chain: List[Tuple[int, List[str], str]]` | Ordered `(experience_idx, evidence_refs, claim_id)` triples. |
| | `episode_claim_graph: Dict[str, dict]` | `claim_id → {status, supported_by_refs, contradicted_by_refs, parent_claims}`. |
| **Free-form** | `metadata: Optional[dict]` | Use `set_metadata(**kwargs)` for the well-known keys. |

### 3.2 Aggregation methods

| Method | Behavior |
|--------|----------|
| `get_reward()` | Sum of `reward` across experiences. |
| `get_total_reward()` | Sum of `reward_details["r_total"]` (falls back to `reward`). |
| `get_length()` | Number of experiences. |
| `set_outcome()` | `outcome = experiences[-1].done`. |
| `set_metadata(**kwargs)` | Merge into `metadata`; warns on near-misses of well-known keys. |

Well-known metadata keys: `model_version`, `rollout_source`, `adapter_ids`,
`transfer_mode`, `budget_stats`, `replay_split`, `partition`.

### 3.3 Answer-chain validation

```python
episode.validate_answer_chain()  # True iff every root_claim resolves to "verified"
                                  #         and has a non-empty support chain.
```

### 3.4 Two-stage segmentation

`separate_into_sub_episodes()` is now a two-stage routine:

```python
boundaries = episode.segment_boundaries(policy="subgoal_change")
# boundaries: List[(start_idx, end_idx, segment_label)]

sub_episodes = episode.separate_into_sub_episodes(
    outcome_length=5,
    policy="subgoal_change",  # default; back-compat with old behavior
)
```

Supported boundary policies:

| Policy | Boundary rule |
|--------|---------------|
| `"subgoal_change"` *(default)* | Whenever `subgoal` (or legacy `sub_tasks`) changes. |
| `"active_skill"` | Whenever `active_skill` changes — one segment per skill invocation. |
| `"commit"` | Each segment ends at (and includes) a `step_type == "COMMIT"` step. |
| `"claim_resolution"` | Each segment ends when `claim_status` transitions to `verified` or `contradicted`. |

Each produced `SubTask_Experience` has its evidence stats recomputed via
`recompute_evidence_stats()` before being returned.

---

## 4. `SubTask_Experience` — segmented local trajectory

The intermediate unit consumed by skill extraction, transfer validation, and
failure-aware refinement. Holds the actual `Experience` objects during
processing; persists into the skill bank only as a lightweight
`SubEpisodeRef` pointer (see `to_sub_episode_ref()`).

### 4.1 Field reference

| Group | Field | Notes |
|-------|-------|-------|
| **Identity** | `sub_task`, `final_goal`, `seg_id`, `episode_id`, `rollout_source` | Pointer fields. |
| | `segment_label` | Stable label assigned at segmentation time. |
| **Contents** | `sub_task_experience: List[Experience]` | The segment's experiences. |
| | `outcome_experiences: Optional[List[Experience]]` | Lookahead experiences after the segment. |
| | `length`, `cumulative_reward` | Cached aggregates. |
| **Quality** | `quality_score: float`, `outcome_classification: str` | Filled by the skill agent quality pipeline. |
| **Skill candidate** | `source_domain` | Inherited from the parent `Episode.domain_type`. |
| | `candidate_target_domains: List[str]` | Domains the Skill Crafter has proposed transferring to. |
| | `verified_domains: List[str]` | Domains where transfer has been verified by the Harness. |
| | `adapter_id: Optional[str]` | Adapter / transfer artifact id. |
| | `transfer_status` | `"none" \| "proposed" \| "verifying" \| "verified" \| "rejected"`. |
| | `verification_status` | `"unverified" \| "verified" \| "contradicted" \| "needs_more_evidence"`. |
| | `skill_candidate_type` | e.g. `"protocol"`, `"primitive_composition"`, `"verifier"`. |
| | `failure_mode` | One of the Harness diagnostics: `slot_binding_failed`, `adapter_execution_mismatch`, `evidence_insufficient`, `temporal_mismatch`, `ui_grounding_mismatch`, `desktop_object_mismatch`, `overconfident_commit`, `contract_mismatch`. |
| **Segment evidence** | `segment_evidence_refs: List[str]` | Deduplicated union of step evidence refs. |
| | `segment_claims: List[str]` | Deduplicated union of supported claims. |
| | `segment_contract_progress: Dict[str, int]` | Count map of `skill_phase → occurrences` within the segment. |
| | `evidence_sufficiency` | `"insufficient" \| "partial" \| "sufficient"`. |

### 4.2 Methods

| Method | Behavior |
|--------|----------|
| `recompute_evidence_stats()` | Roll up evidence refs, claims, phase progress, and a coarse sufficiency tag from member experiences. Called automatically by `Episode.separate_into_sub_episodes`. |
| `generate_summary()` / `generate_outcome_summary()` / `sub_task_labeling()` | LLM-backed summarization helpers. |
| `initialize_sub_task_experience()` | Idempotent fill of summary / outcome_summary / label. |
| `to_sub_episode_ref()` | Produce a lightweight `SubEpisodeRef` pointer for skill-bank storage; additively attaches `source_domain`, `verified_domains`, `transfer_status`, `evidence_sufficiency`, `failure_mode`. |

---

## 5. Buffers

All three buffers share the same shape: FIFO eviction at `buffer_size`,
plus `filter(**criteria)` / `query(**criteria)` / `sample_*(batch_size, mode=...)`.

### 5.1 Filter / query criteria

`filter(**criteria)` matches every key/value via `getattr(item, key)`.
The criteria values can be scalars or one of these structured operators:

```python
buf.filter(domain_type="video")                          # scalar equality
buf.filter(quality_score=("ge", 0.7))                    # ≥
buf.filter(verified_domains=("contains", "web"))         # collection membership
buf.filter(step_type=("in", {"GROUND", "CHECK"}))        # set membership
```

### 5.2 Sample modes

Every buffer's `sample_*(batch_size, mode=...)` accepts:

| Mode | Behavior |
|------|----------|
| `"uniform"` *(default)* | `random.sample` from the (optionally filtered) pool. |
| `"high_quality"` | Filter by `quality_score >= threshold` (or `r_total` for `Episode_Buffer`), then uniform. |
| `"failure_replay"` | Pool is restricted to entries with a `failure_mode` (or `episode_status ∈ {aborted, failed_contract}` for episodes). |
| `"transfer_success"` | `transfer_status == "verified"` and `verified_domains` non-empty. |
| `"transfer_failure"` | `transfer_status == "rejected"` or known transfer-failure `failure_mode`. |
| `"domain_balanced"` | Stratified sampling so each `domain_type` contributes ≈ equally. |

Extra `**criteria` are AND-ed on top of the mode's own filter.

### 5.3 `Episode_Buffer` persistence

```python
buf = Episode_Buffer(buffer_size=1000)
buf.add_episode(episode)
buf.save_to_json("path/to/episodes.json")

reloaded = Episode_Buffer.load_from_json("path/to/episodes.json")
```

`save_to_json` writes `{"episodes": [...], "buffer_size": ..., "num_episodes": ...}`.
`load_from_json` defaults `buffer_size` to whatever the file recorded
(`1000` if absent) unless explicitly overridden.

---

## 6. End-to-end usage example

```python
from data_structure.experience import (
    Experience, Episode, SubTask_Experience,
    Experience_Replay_Buffer, Episode_Buffer, Tool_Buffer,
)

# 1. Construct typed reasoning steps as the rollout proceeds.
ground_step = Experience(
    state=frame_text, action="locate(suspect_hand, t=42)",
    reward=0.0, next_state=frame_text, done=False,
    domain_type="video", task_type="video_holmes_qa",
    step_type="GROUND",
    active_skill="vh.locate_hand",
    skill_phase="locate",
    evidence_refs=["frame_42"],
    evidence_items=[{"evidence_id": "bbox_7", "kind": "bbox",
                     "source": "grounder", "payload": [120, 80, 180, 140],
                     "confidence": 0.91}],
    supports_claims=["c.suspect_grabs_object"],
    claim_status="candidate",
    evidence_confidence=0.91,
    goal="Identify what the suspect is holding",
    subgoal="Locate the suspect's hand in frame 42",
)
assert ground_step.validate_evidence_contract()

check_step = Experience(
    state=..., action="verify(bbox_7 in alibi_window)",
    reward=0.0, next_state=..., done=False,
    domain_type="video", step_type="CHECK",
    evidence_refs=["bbox_7", "alibi_window"],
    supports_claims=["c.suspect_grabs_object"],
    claim_status="verified",
    evidence_confidence=0.96,
    subgoal="Verify the bbox is inside the alibi window",
)

commit_step = Experience(
    state=..., action="answer('a watch')",
    reward=1.0, next_state=..., done=True,
    domain_type="video", step_type="COMMIT",
    evidence_refs=["bbox_7", "alibi_window"],
    supports_claims=["c.suspect_grabs_object"],
    claim_status="verified",
    subgoal="Commit final answer",
)

# 2. Wrap them in an Episode with the answer support chain.
episode = Episode(
    experiences=[ground_step, check_step, commit_step],
    task="What is the suspect holding?",
    domain_type="video", benchmark_name="video_holmes",
    input_modality="frames", output_modality="answer+chain",
    final_answer="a watch",
    root_claims=["c.suspect_grabs_object"],
    answer_support_chain=[
        (0, ["frame_42"], "c.suspect_grabs_object"),
        (1, ["bbox_7", "alibi_window"], "c.suspect_grabs_object"),
        (2, ["bbox_7", "alibi_window"], "c.suspect_grabs_object"),
    ],
    episode_claim_graph={
        "c.suspect_grabs_object": {
            "status": "verified",
            "supported_by_refs": ["bbox_7", "alibi_window"],
            "contradicted_by_refs": [],
            "parent_claims": [],
        },
    },
    episode_status="completed",
)
episode.set_outcome()
assert episode.validate_answer_chain()

# 3. Segment for skill extraction (per-skill segments).
candidates = episode.separate_into_sub_episodes(
    outcome_length=2,
    policy="active_skill",
)
for cand in candidates:
    print(cand.segment_label, cand.evidence_sufficiency,
          cand.segment_evidence_refs)

# 4. Push into buffers.
ep_buf = Episode_Buffer(buffer_size=1000)
ep_buf.add_episode(episode)

tool_buf = Tool_Buffer(buffer_size=512)
tool_buf.add_tools(candidates)

# 5. Domain-aware retrieval / sampling.
video_episodes = ep_buf.filter(domain_type="video", benchmark_name="video_holmes")
verified_skills = tool_buf.sample_tool(
    batch_size=8, mode="transfer_success", source_domain="video",
)

# 6. Persistence.
ep_buf.save_to_json("/tmp/episodes.json")
reloaded = Episode_Buffer.load_from_json("/tmp/episodes.json")
```

---

## 7. Back-compatibility notes

The refactor is purely additive at the wire level:

- All previous constructor kwargs of `Experience`, `Episode`, and
  `SubTask_Experience` still work with the same defaults.
- `to_dict()` only emits new fields when they carry information; legacy
  payloads round-trip through `from_dict()` without changes.
- The legacy `action_type` slot is preserved. When loading, legacy
  `action_type` values that fall inside `STEP_TYPES` are promoted to
  `step_type` automatically. Setting `action_type = "QUERY_MEM"` on a fresh
  object is rejected (no-memory contract).
- Legacy `tasks` / `sub_tasks` are kept on `Experience`; new code should
  read/write `goal` / `subgoal` instead. They're auto-mirrored when only the
  legacy field is supplied.
- Legacy `env_name` / `game_name` are kept on `Episode`; new code should
  read `domain_type` / `benchmark_name`.

---

## 8. Cross-references

- [`legacy/PLAN-EXPERIENCE-REFACTOR.md`](legacy/PLAN-EXPERIENCE-REFACTOR.md) — the
  detailed refactor spec this implementation landed (archived for provenance).
- [`../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)
  §4 — the no-memory evidence & trace bookkeeping contract these structures
  serve.
- [`../plans/05-harness/PLAN-HARNESS.md`](../plans/05-harness/PLAN-HARNESS.md) §5 / §10 — the
  `SkillEpisode` extension and the six-gate set (G0 = evidence-driven contract)
  that consumes `validate_evidence_contract()` and the answer support chain.
- [`../plans/03-skill-bank/PLAN-SKILL-BANK.md`](../plans/03-skill-bank/PLAN-SKILL-BANK.md) §0.3 / §4 — the
  evidence-driven invariant and the skill-bank consumer of
  `to_sub_episode_ref()`.
