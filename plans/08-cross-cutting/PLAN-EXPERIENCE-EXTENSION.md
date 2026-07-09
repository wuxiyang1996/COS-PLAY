# PLAN: Experience Extension Layer — System-Facing Records on Top of `data_structure/`

> **Scope.** This plan defines a thin **extension layer** that sits on top of the existing
> `data_structure/` package (`Experience`, `Episode`, `SubTask_Experience`, replay buffers,
> typed step kinds). It is **not** a redesign of the rollout substrate. The substrate is
> already correct — episode-local, no-memory, evidence-grounded. What is missing is a small
> set of **system-contract records** that the Pipeline Orchestrator, Harness,
> Visual Grounding, evaluation, and skill-bank/crafter workflows need in order to share
> governance, provenance, and routing metadata without polluting `experience.py`.
>
> **Hard non-goal.** No memory subsystem, no `QUERY_MEM`-style recall, no cross-episode
> mutable state. The extension records are append-only **side-tables** keyed off
> `episode_id` / `experience.idx`; they never feed back into the rollout substrate.

---

## 1. Why the current substrate is already correct and should not be replaced

The classes in `data_structure/experience.py` already enforce the four invariants the
project commits to in `plans/README.md` and `data_structure/README.md`:

1. **No memory subsystem.** `STEP_TYPES` deliberately omits `QUERY_MEM`; the
   `Experience.step_type` setter rejects forbidden legacy values; nothing in the package
   exposes a recall API.
2. **Evidence-driven steps.** `Experience.evidence_refs` / `evidence_items` /
   `supports_claims` / `claim_status` / `evidence_confidence` are first-class, and
   `validate_evidence_contract()` is the predicate the Harness `G0` gate evaluates per step.
3. **Transferable skills.** `SubTask_Experience` already carries
   `source_domain`, `candidate_target_domains`, `verified_domains`, `adapter_id`,
   `transfer_status`, `verification_status`, `failure_mode`, plus segment-level
   `segment_evidence_refs` / `segment_claims` / `evidence_sufficiency`.
4. **Episode-local.** `Episode` carries `final_answer`, `root_claims`,
   `answer_support_chain`, and `episode_claim_graph`; nothing in the substrate persists
   beyond the rollout.

In addition, the substrate already supports:

- Typed inner-loop step vocabulary (`GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE | PRIMITIVE`).
- Two-stage segmentation (`segment_boundaries` → `separate_into_sub_episodes`) with
  policies (`subgoal_change`, `active_skill`, `commit`, `claim_resolution`) that the
  Harness and Skill Bank both consume.
- Per-step / per-segment / per-episode `to_dict` ↔ `from_dict` round-trips, with
  back-compat preservation of `action_type`, `tasks`, `sub_tasks`, `env_name`, `game_name`.
- Filter / query / sample-mode surfaces on `Experience_Replay_Buffer`, `Episode_Buffer`,
  and `Tool_Buffer` (uniform, high_quality, failure_replay, transfer_success,
  transfer_failure, domain_balanced).

**Conclusion.** The rollout substrate is the right place for typed steps, evidence
ledgers, claim graphs, skill-candidate fields, and JSON persistence of the trajectory
itself. We must not bloat it with run-level governance or per-invocation skill verdicts;
those belong in a separate extension layer.

---

## 2. What is still missing at the system-contract level

Across `../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`, `../05-harness/PLAN-HARNESS.md`,
`../01-visual-grounding/PLAN-VISUAL-GROUNDING.md`, `../03-skill-bank/PLAN-SKILL-BANK.md`, `../04-skill-crafter/PLAN-SKILL-CRAFTER.md`, and
`../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`, the same five gaps recur:

| Gap | What is missing | Why the substrate cannot host it |
|-----|-----------------|----------------------------------|
| **G-1. Run-level governance** | A single record per rollout that pins `policy_version`, `grounding_version`, `harness_version`, `skill_bank_version`, `gate_version`, budget summary, promotion / rollback decision, audit refs. | An `Episode` is one rollout instance; governance metadata is **about** the rollout, not part of the trace. Putting it on `Episode` couples training-cadence concerns to the trajectory schema and breaks back-compat for legacy payloads. |
| **G-2. Grounding provenance** | A per-step / per-span record that pins which grounding head fired (Path A/B/C), the `<state>` schema version + hash, field completeness, missing fields, entity/relation counts, per-field confidences, tool-trace refs, validator results. | `Experience.evidence_items` carries *what* evidence exists, not *how it was produced* or *whether the schema is complete*. `PLAN-VISUAL-GROUNDING.md §3a` declares `GroundingRecord` to be the canonical `evidence_out` for `GATHER` skills, but there is no concrete container for it. |
| **G-3. Skill invocation contract** | A per-invocation record the Harness can write: slot binding, precondition / evidence / adapter checks, fit / risk scores, veto decision, shadow-validation status, harness verdict. | `SubTask_Experience` is the *outcome* of segmentation; a skill *invocation session* is a Harness micro-runtime artifact whose lifecycle is per-call, not per-segment. The Actor decides; the Harness records. |
| **G-4. Governable failure routing** | A normalized record that points at any source artifact (an `Experience`, a segment, a `GroundingRecord`, a `SkillInvocationRecord`) and routes the failure to the right consumer (Skill Crafter, Visual Grounding, Orchestrator rollback, human audit) with severity / recoverability / trigger source. | Failures today are scattered: `Experience.reward_details["failure_mode"]`, `SubTask_Experience.failure_mode`, `Episode.episode_status`, plus various ad-hoc fields. There is no single typed surface the Orchestrator can consume. |
| **G-5. Final-answer support package** | A standardized record that bundles the `final_answer`, the support claim ids, evidence refs, support strength, judge verdict, and joint success — what evaluation actually scores against. | `Episode.answer_support_chain` is a list of `(idx, refs, claim_id)` triples; it is the *ledger*, not the *answer package*. Evaluators currently re-derive the package from `Episode` fields, which makes joint-success metrics fragile. |

The five extension records below close exactly these five gaps. Each one is **append-only**,
keyed off ids that already exist in the substrate, and serializable as standalone JSON
sidecar payloads.

---

## 3. Proposed extension objects

All extension records share a small set of conventions:

- They are **dataclasses** (Python `@dataclass`), not full classes.
- They reference `Episode` / `Experience` / `SubTask_Experience` **by id** (never embed a copy).
- They are **append-only**: once written, fields are immutable; corrections create a new
  record that supersedes the prior one via `audit_refs`.
- `to_dict()` / `from_dict()` are symmetric and produce JSON-safe payloads.
- Every record carries `record_type` (literal), `record_version` (semver string),
  and `created_at` (ISO-8601 UTC) so legacy loaders can branch safely.

### A. `RunRecord` — run-level governance for one rollout

**Purpose.** Capture run-level governance information for one rollout instance without
polluting `Episode`. One `RunRecord` per `episode_id`. This is what the Orchestrator
reads to decide promotion / rollback and what audit pipelines join against.

**Required fields.**

| Field | Type | Notes |
|-------|------|-------|
| `run_id` | `str` (UUID4) | Stable id for this run. |
| `episode_id` | `str` | Foreign key into the substrate. |
| `policy_version` | `str` (semver) | Actor / decision-policy version (`agent_version` in orchestrator vocabulary). |
| `grounding_version` | `str` (semver) | Visual Grounding parser/adapter version. |
| `harness_version` | `str` (semver) | `SkillHarness` runtime version. |
| `skill_bank_version` | `str` (semver) | Active store snapshot id. |
| `gate_version` | `str` (semver) | Gate-stack version (G0 … G5 spec hash). |
| `budget_summary` | `dict` | `{tokens_in, tokens_out, latency_ms, tool_calls, $cost}` aggregates. |
| `promotion_decision` | `Optional[str]` | `"promote" \| "hold" \| "reject" \| "n/a"`. Only set when the run feeds a promotion attempt. |
| `rollback_decision` | `Optional[str]` | `"none" \| "snapshot" \| "skill_subset" \| "policy_subset"`. |
| `audit_refs` | `List[str]` | Pointers into the audit log (artifact uris, log ids). |
| `record_type` | `Literal["RunRecord"]` | |
| `record_version` | `str` | |
| `created_at` | `str` (ISO-8601) | |

**Optional fields.** `notes: Optional[str]`, `parent_run_id: Optional[str]` (when a run
is a re-run of another), `tags: List[str]`, `dataset_split: Optional[str]`,
`benchmark_name: Optional[str]` (denormalized convenience copy of `Episode.benchmark_name`).

**Who creates it.** The Pipeline Orchestrator at the **end** of a run (not during).
**Who reads it.** Orchestrator (promotion / rollback), Harness (non-regression checks
across versions), evaluation dashboards, ablation pipelines.

**Append-only.** Yes. Corrections append a new `RunRecord` with the same `episode_id`
and a new `run_id`; `audit_refs` of the new record point at the prior `run_id`.

**Reference style.** By id (`episode_id`). Never embeds the `Episode`.

**Serialization.** Standalone JSON; one file per `run_id` under
`runs/<date>/<run_id>.json`, or a JSONL stream `runs/<date>/runs.jsonl`. Both must
round-trip via `RunRecord.from_dict(json.loads(...))`.

---

### B. `GroundingRecord` — grounding provenance and confidence

**Purpose.** Capture grounding provenance and confidence per `GROUND` step (or per
schema-emitting span). This is the canonical `evidence_out` payload for `GATHER`-role
skills, as `PLAN-VISUAL-GROUNDING.md §3a` already declares.

**Required fields.**

| Field | Type | Notes |
|-------|------|-------|
| `grounding_id` | `str` (UUID4) | Stable id; usable in `Experience.evidence_refs`. |
| `episode_id` | `str` | |
| `step_idx` | `Optional[int]` | `Experience.idx` of the producing step. Present when the grounding maps to one step. |
| `span_id` | `Optional[str]` | Used when the grounding spans multiple steps (e.g. video keyframe sweep). One of `step_idx` / `span_id` must be set. |
| `domain_type` | `str` | One of `DOMAIN_TYPES` (game / web / os / video / visual_reasoning). |
| `task_type` | `str` | Domain-local task tag. |
| `grounding_path` | `Literal["A", "B", "C"]` | Routing path used (Path A heuristic, Path B vision, Path C tool loop). |
| `schema_version` | `str` (semver) | Canonical `<state>` schema version. |
| `schema_hash` | `str` (sha256 hex) | Hash of the emitted schema string. |
| `field_completeness` | `float` in `[0, 1]` | Fraction of required schema fields populated. |
| `missing_fields` | `List[str]` | Names of schema fields the grounder failed to fill. |
| `entity_count` | `int` | |
| `relation_count` | `int` | |
| `state_confidence` | `float` in `[0, 1]` | Aggregate confidence of the structured state. |
| `field_confidences` | `Dict[str, float]` | Per-field confidence map. |
| `tool_trace_refs` | `List[str]` | Pointers to tool-call traces used during grounding (Path C). |
| `validator_results` | `Dict[str, bool]` | Per-validator pass/fail (e.g. `{"schema_lint": true, "bbox_in_frame": true}`). |
| `record_type` | `Literal["GroundingRecord"]` | |
| `record_version` | `str` | |
| `created_at` | `str` (ISO-8601) | |

**Optional fields.** `latency_ms: Optional[int]`, `parser_lora_id: Optional[str]`,
`raw_state_ref: Optional[str]` (pointer into the rollout's raw-state blob store),
`omniparser_version: Optional[str]`.

**Who creates it.** The Visual Grounding component (the `schema_gen` LoRA adapter,
or the Path C tool-loop runtime), once per `GROUND` step / span.

**Who reads it.** Harness `G0` evidence-driven gate (treats it as `evidence_out`
for `GATHER` skills); Skill Bank for grounding-aware retrieval; evaluation for
schema-completeness metrics; Skill Crafter for `evidence_starved` failure analysis.

**Append-only.** Yes. A re-grounding produces a new `GroundingRecord` (new
`grounding_id`); the old one is preserved.

**Reference style.** By id. Episodes refer to a `GroundingRecord` via
`Experience.evidence_refs` and `Experience.evidence_items[].evidence_id`. The record
itself stores the schema *blob hash*, not the blob.

**Serialization.** Standalone JSON; one record per `grounding_id` under
`grounding/<episode_id>/<grounding_id>.json` or as a JSONL stream
`grounding/<episode_id>.jsonl`. The schema string itself, when persisted, lives in a
sibling blob file keyed by `schema_hash`.

---

### C. `SkillInvocationRecord` — Harness-readable invocation session

**Purpose.** Expose a single per-invocation skill session that the Harness owns end to
end: which skill was selected, how it was bound, whether the gates passed, and what the
final verdict was. One `SkillInvocationRecord` per `(episode_id, subtask_id|segment_id,
skill_id, attempt)`.

**Required fields.**

| Field | Type | Notes |
|-------|------|-------|
| `invocation_id` | `str` (UUID4) | Stable id for this invocation attempt. |
| `episode_id` | `str` | |
| `subtask_id` | `Optional[str]` | Foreign key into the segment that hosted the invocation, when known. |
| `segment_id` | `Optional[str]` | Foreign key when invocation predates segmentation. One of `subtask_id` / `segment_id` must be set. |
| `skill_id` | `str` | |
| `skill_version` | `str` (semver) | |
| `adapter_id` | `Optional[str]` | Domain adapter used to bind the skill at this invocation. |
| `source_domain` | `str` | Domain the skill was originally extracted from. |
| `target_domain` | `str` | Domain the invocation is running in. |
| `slot_binding` | `Dict[str, str]` | Slot-name → entity-id map (`target`, `blocker`, `constraint`, …). |
| `binding_ok` | `bool` | Result of the slot-binding gate. |
| `precondition_ok` | `bool` | Result of the precondition check. |
| `evidence_ok` | `bool` | Result of `G0` evidence-driven check. |
| `adapter_ok` | `bool` | Result of adapter-execution check. |
| `fit_score` | `float` in `[0, 1]` | Skill-Bank-side selection score. |
| `risk_score` | `float` in `[0, 1]` | Harness-side risk estimate. |
| `veto` | `bool` | `True` iff Harness vetoed the invocation. |
| `veto_reason` | `Optional[str]` | One of the canonical Harness diagnostics (see `PLAN-HARNESS.md §10a`). |
| `shadow_validation_status` | `Literal["n/a", "pending", "passed", "failed"]` | Set when the invocation is in shadow mode. |
| `harness_verdict` | `Literal["accept", "reject", "shadow", "abstain"]` | Final verdict. |
| `record_type` | `Literal["SkillInvocationRecord"]` | |
| `record_version` | `str` | |
| `created_at` | `str` (ISO-8601) | |

**Optional fields.** `started_at`, `finished_at`, `latency_ms`, `tokens_in`,
`tokens_out`, `cost`, `prior_invocation_id` (for retried invocations),
`evidence_warrant_refs: List[str]` (for `COMMIT`-role invocations).

**Who creates it.** The Harness (`SkillHarness` micro-runtime) on every skill invocation
attempt — including vetoed ones.

**Who reads it.** Orchestrator (promotion / rollback decisions feed off acceptance
rates), Skill Bank lifecycle manager (shadow→active transitions), Skill Crafter
(failure-mode mining), evaluation (skill-attribution analysis).

**Append-only.** Yes. A retry yields a new `invocation_id` and points at the prior via
`prior_invocation_id`.

**Reference style.** By id (`episode_id`, `subtask_id`/`segment_id`, `skill_id`).
Never embeds the `SubTask_Experience` or `Skill`.

**Serialization.** Standalone JSON; one record per `invocation_id` under
`invocations/<episode_id>/<invocation_id>.json`, or JSONL `invocations/<episode_id>.jsonl`.

---

### D. `FailureRoutingRecord` — making failures governable

**Purpose.** Normalize the existing scattered failure surfaces (`Experience.reward_details
["failure_mode"]`, `SubTask_Experience.failure_mode`, `Episode.episode_status`,
Harness diagnostics) into a single typed record the Orchestrator can route to the
correct downstream owner.

**Required fields.**

| Field | Type | Notes |
|-------|------|-------|
| `routing_id` | `str` (UUID4) | |
| `episode_id` | `str` | |
| `source_record_type` | `Literal["Experience", "SubTask_Experience", "Episode", "GroundingRecord", "SkillInvocationRecord", "AnswerSupportRecord"]` | Which substrate / extension object emitted the failure. |
| `source_record_id` | `str` | The id of that record (e.g. `f"{episode_id}#{idx}"` for `Experience`, `seg_id` for segments, `invocation_id` for invocations, etc.). |
| `failure_type` | `str` | A canonical label: one of the Harness diagnostics (`slot_binding_failed`, `adapter_execution_mismatch`, `evidence_insufficient`, `temporal_mismatch`, `ui_grounding_mismatch`, `desktop_object_mismatch`, `overconfident_commit`, `contract_mismatch`, `opaque_skill_violation`, `evidence_interface_mismatch`, `evidence_starved`, …) plus extension labels (`schema_incomplete`, `judge_negative`, `budget_exceeded`). |
| `severity` | `Literal["info", "warn", "error", "critical"]` | |
| `recoverability` | `Literal["recoverable", "rerun_required", "rollback_required", "fatal"]` | |
| `trigger_source` | `Literal["harness_gate", "validator", "judge", "budget_controller", "adapter_runtime", "schema_validator", "human_audit"]` | What raised the failure. |
| `route_to` | `Literal["skill_crafter", "visual_grounding", "orchestrator_rollback", "skill_bank_curator", "human_audit", "drop"]` | Where the failure should be routed. |
| `reason` | `str` | Short human-readable explanation (≤ 240 chars). |
| `created_at` | `str` (ISO-8601) | |
| `record_type` | `Literal["FailureRoutingRecord"]` | |
| `record_version` | `str` | |

**Optional fields.** `evidence_refs: List[str]` (relevant evidence ids that motivated
the routing), `related_routing_ids: List[str]` (other routing records that originate
from the same root cause), `parent_failure_id: Optional[str]`.

**Who creates it.** Whoever **detects** the failure: Harness gates (most cases), the
schema validator (Visual Grounding), the judge (evaluation), the budget controller
(Orchestrator), or human audit. Detection is decoupled from routing — the Orchestrator
applies a routing rule to set `route_to`.

**Who reads it.** Skill Crafter (failure replay queues), Visual Grounding (re-train
queue), Orchestrator (rollback / promotion-hold), human-audit dashboards.

**Append-only.** Yes. A re-investigation appends a new `FailureRoutingRecord` whose
`parent_failure_id` points at the original.

**Reference style.** By id only. The failure record never embeds the failing
artifact — it points at it.

**Serialization.** Standalone JSON / JSONL under `failures/<date>/`. Routing rules
live elsewhere (Orchestrator config); the record only stores the **applied** route.

---

### E. `AnswerSupportRecord` — final answer + support package

**Purpose.** Standardize the answer-side payload that evaluation actually scores
against. Today, evaluators stitch this together from `Episode.final_answer`,
`Episode.root_claims`, `Episode.answer_support_chain`, `Episode.episode_claim_graph`,
plus per-step evidence. `AnswerSupportRecord` is the single record an evaluator can
read to compute joint success.

**Required fields.**

| Field | Type | Notes |
|-------|------|-------|
| `answer_record_id` | `str` (UUID4) | |
| `episode_id` | `str` | |
| `final_answer` | `Any` | The committed answer (string, structured, or modality-specific payload). |
| `answer_type` | `Literal["mcq", "free_text", "structured", "action", "binary", "numeric"]` | |
| `support_claim_ids` | `List[str]` | The `root_claims` whose verification underwrites the answer. |
| `support_evidence_refs` | `List[str]` | The minimal evidence-ref set that supports the claims. |
| `support_strength` | `float` in `[0, 1]` | Aggregate strength across `support_claim_ids` (e.g. mean of per-claim `evidence_confidence`). |
| `judge_verdict` | `Literal["correct", "partial", "incorrect", "abstain", "ungraded"]` | Filled by the evaluation judge. |
| `joint_success` | `bool` | `True` iff `judge_verdict == "correct"` **and** every `support_claim_id` is `verified` in `Episode.episode_claim_graph` **and** `support_strength ≥ θ` (`θ` is judge-config). |
| `record_type` | `Literal["AnswerSupportRecord"]` | |
| `record_version` | `str` | |
| `created_at` | `str` (ISO-8601) | |

**Optional fields.** `judge_id: Optional[str]`, `judge_version: Optional[str]`,
`judge_notes: Optional[str]`, `gold_answer: Optional[Any]` (for benchmarks where it is
known), `metric_breakdown: Optional[Dict[str, float]]` (per-rubric scores).

**Who creates it.** The Pipeline Orchestrator at the end of every rollout that produces
an answer; the `judge_verdict` and `joint_success` fields are filled by the evaluation
judge in a follow-up write.

**Who reads it.** Evaluation dashboards (joint-success metric), Orchestrator (promotion
gates that condition on joint success), Skill Crafter (negative example mining when
`joint_success == False` despite `judge_verdict == "correct"`).

**Append-only.** Yes. A re-judge appends a new `AnswerSupportRecord` (new
`answer_record_id`) with `audit_refs` (in optional metadata) pointing at the prior.

**Reference style.** By id. Pulls `support_claim_ids` and `support_evidence_refs` from
the `Episode`; never embeds the `Episode`.

**Serialization.** Standalone JSON; one record per `answer_record_id`, sidecar to the
episode under `answers/<episode_id>/<answer_record_id>.json`.

---

## 4. Ownership rules — which module writes which object

The plan must be **mechanically enforceable**: every extension record has exactly one
producer. Other modules may read freely.

| Record | Sole producer | Primary readers |
|--------|---------------|-----------------|
| `RunRecord` | **Pipeline Orchestrator** (end-of-run finalizer). | Orchestrator, Harness (non-regression), evaluation, ablation pipelines. |
| `GroundingRecord` | **Visual Grounding** (`schema_gen` adapter or Path C tool-loop runtime). | Harness `G0` gate, Skill Bank (retrieval), evaluation, Skill Crafter (`evidence_starved` analysis). |
| `SkillInvocationRecord` | **`SkillHarness`** (per-invocation runtime, `harness/` package). | Orchestrator, Skill Bank lifecycle manager, Skill Crafter, evaluation. |
| `FailureRoutingRecord` | **Detector** writes the record body; **Orchestrator** sets `route_to` via routing rules. (Two-step write is enforced by leaving `route_to` `Optional` until the Orchestrator stamps it.) | Skill Crafter, Visual Grounding, Orchestrator rollback, human audit. |
| `AnswerSupportRecord` | **Pipeline Orchestrator** writes the answer-side fields; **Evaluation Judge** fills `judge_verdict` / `joint_success` in a follow-up write. | Evaluation, Orchestrator promotion gates, Skill Crafter. |

**Rule.** No module other than the listed producer may construct or mutate the record.
Readers consume `to_dict()` / `from_dict()` payloads only.

This boundary mirrors the `Actor / Harness / Skill Bank / Orchestrator` four-way
separation already pinned in `PLAN-PIPELINE-ORCHESTRATOR.md §0a`.

---

## 5. Serialization / backward-compatibility strategy

The substrate's serialization rules are already correct (additive `to_dict`,
forgiving `from_dict`). The extension layer extends them as follows.

1. **Sidecar files, not embedded fields.** Extension records are persisted as sidecar
   JSON / JSONL files alongside the rollout payload. They are **not** appended to
   `Episode.to_dict()`. This keeps `Episode_Buffer.save_to_json` /
   `Episode_Buffer.load_from_json` byte-compatible with existing payloads — old loaders
   keep working, new loaders can opt in.

2. **Versioned record envelopes.** Every record carries `record_type` and
   `record_version`. Loaders branch on `(record_type, record_version)` and tolerate
   unknown new fields via `.get(...)` with defaults — the same discipline `Experience.from_dict`
   already follows.

3. **Foreign-key discipline.** Records reference the substrate by id only:
   `episode_id`, `Experience.idx`, `seg_id`. No record ever copies a substrate object.
   This keeps the substrate single-sourced and lets loaders resolve cross-record
   joins lazily.

4. **Optional re-projection helpers.** A small `data_structure/extensions/loader.py`
   may expose helpers like:

   ```python
   load_run_bundle(episode_id) -> RunBundle
   #   = (Episode, RunRecord, list[GroundingRecord], list[SkillInvocationRecord],
   #      list[FailureRoutingRecord], list[AnswerSupportRecord])
   ```

   `RunBundle` is a pure namespace — it does **not** alter `Episode` or any extension
   record; it just colocates them for evaluators that want one entry point.

5. **Legacy payload compatibility.**
   - Episodes written before this layer existed have **no** sidecars; loaders must
     return empty lists and not raise.
   - Episodes loaded with sidecars present must round-trip through
     `from_dict` ↔ `to_dict` for every record.
   - Any extension record loaded from a future `record_version` higher than the
     loader supports must trigger a `UserWarning` and be skipped rather than crash.

6. **Hashing for non-regression.** The Orchestrator and Harness should be able to
   recompute a deterministic content hash over `(Episode.to_dict(), RunRecord.to_dict(),
   sorted([rec.to_dict() for rec in extension_records]))` for non-regression checks.
   To make this stable, every record's `to_dict()` must use sorted keys and stable
   numeric formatting.

---

## 6. Minimal implementation order

Each phase is independently shippable and adds at most one new file under
`data_structure/`. None of them modifies `experience.py`.

| Phase | Deliverable | Why first / next |
|-------|-------------|------------------|
| **P0** | `data_structure/extensions/__init__.py` + `data_structure/extensions/_base.py` (record envelope, version constants, ISO-8601 helpers, `RecordError`). | Establishes the shared envelope so every later record inherits the same versioning + serialization rules. Zero behavior change. |
| **P1** | `data_structure/extensions/grounding_record.py` (`GroundingRecord`). | Visual Grounding is the **upstream** producer; the Harness `G0` gate already expects `GroundingRecord` to exist (see `PLAN-VISUAL-GROUNDING.md §3a`). Shipping this first unblocks evidence-driven gating in the harness. |
| **P2** | `data_structure/extensions/skill_invocation_record.py` (`SkillInvocationRecord`). | The Harness micro-runtime needs a place to write per-invocation verdicts. Required before the Skill Bank can implement shadow→active transitions cleanly. |
| **P3** | `data_structure/extensions/failure_routing_record.py` (`FailureRoutingRecord`). | Once `GroundingRecord` and `SkillInvocationRecord` exist, both can emit failure records. Routing rules can land in the Orchestrator without further substrate changes. |
| **P4** | `data_structure/extensions/answer_support_record.py` (`AnswerSupportRecord`). | Builds on `Episode.answer_support_chain`; can land independently and is consumed by evaluation immediately. |
| **P5** | `data_structure/extensions/run_record.py` (`RunRecord`). | Run-level governance ties everything together; landing it last lets it reference the IDs / record types already shipped in P1–P4. |
| **P6** | `data_structure/extensions/loader.py` (`RunBundle`, `load_run_bundle`, hash helpers). | Thin convenience layer for evaluators / dashboards. No new semantics. |

**Test surface (per phase).** Each record ships with:

- A `from_dict(to_dict(rec)) == rec` round-trip test.
- A "load from a future `record_version`" test that asserts a `UserWarning` and skip.
- A "load from a payload missing optional fields" test that asserts no exception.
- A "loader with no sidecar files" test that asserts empty-list returns.

---

## 7. Risks / anti-goals

**Anti-goals (these the plan explicitly rejects).**

1. **No memory subsystem.** No record may be queried by content from outside the
   originating episode. Extension records exist to make episode-local reasoning
   *governable*, not to reintroduce recall.
2. **No mutation of the substrate.** `experience.py` is not edited by this plan. All
   new behavior lives under `data_structure/extensions/`.
3. **No embedded copies of substrate objects.** Records reference by id; storage stays
   single-sourced.
4. **No cross-record schema unification.** Each record owns its own fields. We do not
   introduce a "universal record" envelope beyond `record_type` / `record_version` /
   `created_at`.
5. **No bypass of the Harness gates.** Extension records report what gates decided;
   they do not become gates themselves.

**Risks and how the plan mitigates them.**

| Risk | Mitigation |
|------|------------|
| Sidecar drift (records and `Episode` going out of sync). | Foreign-key discipline + the `RunBundle` loader's hash helpers (P6). Orchestrator non-regression check refuses to promote when the bundle hash drifts unexpectedly. |
| Producer overlap (two modules trying to write the same record). | Section 4's "sole producer" rule, enforced by code-owner boundaries: each record file lives next to no producer code, and producers are listed in module docstrings. PR review gates writes from non-owner modules. |
| Schema sprawl (new optional fields creeping in). | All fields go through `record_version` bumps. The `from_dict` test "missing optional" is the gate that catches accidental required-field promotions. |
| Performance regression on large rollouts. | Sidecars are JSONL when appropriate; `RunBundle` loads lazily. The hot path (`Episode_Buffer.save_to_json` / `load_from_json`) is unchanged. |
| Confusion between `SubTask_Experience.failure_mode` and `FailureRoutingRecord`. | `SubTask_Experience.failure_mode` remains the *segment-level* substrate field used by skill curation. `FailureRoutingRecord` is the *governance* record used by the Orchestrator. The two coexist and reference each other by id; one is not a replacement for the other. |
| Misuse of `RunRecord` to store memory. | Field whitelist in §3.A is closed; `notes` is free-form but bounded; new fields require a `record_version` bump and a code-review check that the field is governance, not retrieval. |

---

## 8. Recommended concrete file layout

The extension layer adds one subpackage under `data_structure/`. Nothing in
`experience.py` is touched.

```
data_structure/
├── experience.py                 # unchanged — substrate
├── README.md                     # unchanged
├── legacy/                       # unchanged
└── extensions/                   # NEW — system-facing extension layer
    ├── __init__.py               # re-exports the five record classes
    ├── _base.py                  # record envelope, version consts, ISO-8601 helpers
    ├── run_record.py             # RunRecord
    ├── grounding_record.py       # GroundingRecord
    ├── skill_invocation_record.py# SkillInvocationRecord
    ├── failure_routing_record.py # FailureRoutingRecord
    ├── answer_support_record.py  # AnswerSupportRecord
    └── loader.py                 # RunBundle + load_run_bundle + hash helpers
```

**Public surface (`data_structure/extensions/__init__.py`).**

```python
from data_structure.extensions._base import (
    RecordError,
    EXTENSION_RECORD_VERSION,
    iso_now,
)
from data_structure.extensions.run_record import RunRecord
from data_structure.extensions.grounding_record import GroundingRecord
from data_structure.extensions.skill_invocation_record import SkillInvocationRecord
from data_structure.extensions.failure_routing_record import FailureRoutingRecord
from data_structure.extensions.answer_support_record import AnswerSupportRecord
from data_structure.extensions.loader import RunBundle, load_run_bundle, bundle_hash

__all__ = [
    "RecordError", "EXTENSION_RECORD_VERSION", "iso_now",
    "RunRecord", "GroundingRecord", "SkillInvocationRecord",
    "FailureRoutingRecord", "AnswerSupportRecord",
    "RunBundle", "load_run_bundle", "bundle_hash",
]
```

**On-disk layout (per rollout dataset).**

```
<dataset_root>/
├── episodes/                     # existing — Episode_Buffer.save_to_json output
│   └── episodes.json
├── runs/                         # NEW — RunRecord sidecars
│   └── <date>/<run_id>.json
├── grounding/                    # NEW — GroundingRecord sidecars
│   └── <episode_id>.jsonl
├── invocations/                  # NEW — SkillInvocationRecord sidecars
│   └── <episode_id>.jsonl
├── failures/                     # NEW — FailureRoutingRecord sidecars
│   └── <date>/failures.jsonl
└── answers/                      # NEW — AnswerSupportRecord sidecars
    └── <episode_id>/<answer_record_id>.json
```

Old datasets that contain only `episodes/` continue to load. New consumers that need
the system-contract view call `load_run_bundle(episode_id)` to get the `Episode`
plus all sidecars as a `RunBundle` namespace.

---

## 9. Cross-references

- `data_structure/experience.py` — the substrate this plan extends without touching.
- `data_structure/README.md` §1–§5 — substrate field reference and invariants.
- `plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md` §0a, §2.2, §4 — Actor/Harness/Bank/Orchestrator
  boundary; `SkillEpisode` first-class artifact; episode-local evidence & trace bookkeeping.
- `plans/05-harness/PLAN-HARNESS.md` §5.1, §10, §10a — `SkillEpisode` extension; six-gate stack;
  domain-specific transfer-failure diagnostics consumed by `FailureRoutingRecord.failure_type`.
- `plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md` §3a — `GroundingRecord` as canonical `evidence_out`
  for `GATHER` skills; Path A/B/C routing.
- `plans/03-skill-bank/PLAN-SKILL-BANK.md` §0.3, §4 — evidence-driven invariant + skill record fields
  the `SkillInvocationRecord` references by id.
- `plans/04-skill-crafter/PLAN-SKILL-CRAFTER.md` §2.5, §6.2 — typed proposals + `evidence_starved`
  failure category that `FailureRoutingRecord` routes to the Crafter.
- `plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md` — gate execution / promotion / rollback that
  consumes `RunRecord` and `SkillInvocationRecord`.
