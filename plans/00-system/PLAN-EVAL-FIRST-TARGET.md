# PLAN: First Evaluation Target — Short-Video Evidence-Grounded Reasoning

**Status:** First end-to-end evaluation contract for the project.
**Owner:** Pipeline Orchestrator + Harness (consumers of `EpisodeTrace` / `SkillEpisode`).
**Companions:** [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) §6, [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md), [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md), [README.md](../README.md).

---

## 1. Scope and motivation

The project's first execution focus is **short-video evidence-grounded reasoning** ([README §Current execution focus](../README.md#current-execution-focus)). Every other plan describes *what the system is*; this plan describes *how we judge whether it works* on its first arena.

Existing module-level metrics (grounding accuracy, gate pass rate, contract validity) are necessary but not sufficient: they do not tell us whether the assembled system answers a video question correctly **and** can show why. This plan defines that joint contract.

The protocol here is the **task-level evaluation contract**. All other evaluation harnesses (grounding milestones, gate non-regression slices, transfer benchmarks) feed into or specialize from this one. It must stay simple enough to run end-to-end every release.

Concrete goals:

1. Pin the input/output contract for short-video evidence-grounded reasoning.
2. Define a small set of axes that separately measure Actor quality, Harness filtering/veto quality, system performance, skill-use efficiency, reasoning-step usefulness, and transfer robustness ([PLAN-PIPELINE-ORCHESTRATOR.md §0a.5](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#0a5-evaluation-implication)).
3. Give one canonical reported number — **Joint Success Rate** — that cannot be gamed by either answer-only or evidence-only optimization.
4. Specify a minimal, easy-to-follow judge protocol.
5. Ship a benchmark-slice plan small enough to run weekly and detailed enough to localize regressions.

---

## 2. Task definition

### 2.1 Input

Each evaluation instance provides:

| Field | Description |
|-------|-------------|
| `clip` | A short video or clip (seconds to ~minutes; long-video is out of scope, see §11). |
| `question` | A natural-language question or task objective grounded in the clip. |
| `candidate_answers` *(optional)* | Multiple-choice options when the slice is MCQ. Absent for open-ended slices. |
| `metadata` | `slice_id`, `difficulty`, `hop_type` (single-hop / multi-hop), `evidence_type` (visual / temporal / social). |

### 2.2 System output (per instance)

The system MUST emit:

| Field | Type | Description |
|-------|------|-------------|
| `final_answer` | string \| option_id | The committed answer. |
| `answer_type` | enum {`mcq`, `open_short`, `open_free`} | Determines the automatic-evaluation rule (§7.1). |
| `support_package` | object | Minimal artifact backing the answer; see §2.3. |
| `evidence_refs` | list of `EvidenceRef` | Within-episode references (frame ids, time spans, region ids, tool-call ids) per [PLAN-PIPELINE-ORCHESTRATOR.md §4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping). |
| `reasoning_trace_summary` | object | Minimal typed summary of the inner-MDP hops that led to the answer. |

### 2.3 `support_package` (minimal contract)

```text
support_package = {
  claim:               <string>          # restated final claim
  evidence_warrant:    [EvidenceRef]     # non-empty; mirrors ActionRecord.evidence_warrant
  reasoning_steps:     [HopSummary]      # ordered, typed (GROUND/CHECK/RETRIEVE/COMMIT)
  uncertainty:         <float, 0..1>     # self-reported confidence
  budget_used:         { hops, ground_calls, tool_calls, tokens }
}
```

`reasoning_trace_summary` is the same `reasoning_steps` list rolled up to one record per hop, kept short (one line per hop) so it is human-skimmable.

The contract ties directly to existing typed records ([PLAN-PIPELINE-ORCHESTRATOR.md §2.2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#22-required-record-types)): `evidence_warrant` reuses `ActionRecord.evidence_warrant`; `reasoning_steps` is built from `InnerHopRecord`. Nothing new needs to be stored on disk — the eval driver projects existing trace fields into this view.

---

## 3. Required system outputs

The eval driver consumes one JSONL record per instance:

```json
{
  "instance_id": "...",
  "slice_id": "...",
  "final_answer": "...",
  "answer_type": "mcq",
  "support_package": { ... },
  "evidence_refs": [ ... ],
  "reasoning_trace_summary": [ ... ],
  "telemetry": {
    "outer_steps": int,
    "inner_hops": int,
    "ground_calls": int,
    "tool_calls": int,
    "wall_clock_ms": float,
    "estimated_cost_usd": float,
    "actor_skill_decisions": [ ... ],
    "harness_filter_events": [ ... ],
    "harness_veto_events": [ ... ]
  }
}
```

`actor_skill_decisions`, `harness_filter_events`, and `harness_veto_events` are required so the §4 axes can attribute behavior to the Actor vs. the Harness ([PLAN-PIPELINE-ORCHESTRATOR.md §0a](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#0a-actorharnessskill-bankorchestrator-boundary)).

Records missing `support_package`, `evidence_refs`, or `final_answer` are auto-scored as **failure** on every axis except answer correctness — they are not silently dropped.

---

## 4. Evaluation axes

| Axis | Question it answers | Primary metric |
|------|--------------------|----------------|
| **Answer correctness** | Did the system get the right answer? | Answer Accuracy |
| **Evidence validity** | Does the cited evidence exist, support the answer, and cover key claims? | Evidence Support Rate |
| **Joint success** | Did the system get the answer right *and* show valid evidence? | **Joint Success Rate** *(headline)* |
| **Reasoning efficiency** | Did the system use the inner MDP minimally? | Avg Hops, Reasoning Overhead |
| **Tool / grounding efficiency** | Did the system avoid unnecessary tool / grounding calls? | Avg Ground Calls, Avg Tool Calls |
| **Failure quality** | When wrong, is the failure interpretable and recoverable? | Failure Taxonomy distribution (§6) |

Three secondary axes derive from the four-way module separation and are reported alongside the headline:

| Secondary axis | Question it answers | Reported as |
|----------------|--------------------|-------------|
| **Actor quality** | Does the Actor make better decisions over time? | Skill-decision quality, fallback rate |
| **Harness filtering / veto quality** | Does the Harness block bad invocations without over-blocking? | Filter precision, veto precision/recall |
| **Transfer robustness** | Do skills exercised in short-video survive cross-domain replay? | Transfer pass rate from §6.3 of the orchestrator |

Each evaluation run must report **all six primary axes** plus the three secondary axes. Reporting any subset alone is non-conforming.

---

## 5. Joint success definition (headline)

**Joint Success Rate** is the single reported headline number:

```
JointSuccess(instance) := AnswerCorrect(instance) AND EvidenceValid(instance)
JointSuccessRate        := mean over instances of JointSuccess(instance)
```

Where:

- `AnswerCorrect` is determined by the §7.1 automatic answer evaluator for the instance's `answer_type`.
- `EvidenceValid` is determined by the §7.2 evidence judge protocol over `support_package` + `evidence_refs`.

**Constraint:** the evaluation cannot report Answer Accuracy or Evidence Support Rate without also reporting Joint Success Rate. This prevents the project from drifting into "the answer is right but the chain is fabricated" or "the chain is rich but the answer is wrong."

### 5.1 Other required headline numbers

Always reported with Joint Success:

- **Answer Accuracy** — `mean(AnswerCorrect)`.
- **Evidence Support Rate** — `mean(EvidenceValid)`.
- **Joint Success Rate** — as above.

These three together form the **eval triple** that every release prints first.

---

## 6. Failure taxonomy

Every failed instance is labeled with exactly one primary failure class (the most upstream applicable cause). Secondary labels are allowed for diagnostic dashboards but do not change the count.

| Code | Class | Definition |
|------|-------|-----------|
| `F1` | answer_wrong + evidence_wrong | Both the final answer and the cited evidence are incorrect. |
| `F2` | answer_wrong + evidence_insufficient | Answer is wrong; evidence is missing, partial, or does not cover the key claim. |
| `F3` | answer_correct + evidence_missing | Answer is correct but `evidence_warrant` is empty or unverifiable. Counts as failure for evidence/joint axes. |
| `F4` | answer_correct + evidence_mismatched | Answer is correct but the cited evidence does not actually support it (lucky guess, decorative citation). |
| `F5` | grounding_incomplete | Required entities/regions were never grounded; downstream `CHECK` / `COMMIT` is unanchored. |
| `F6` | over_grounding / unnecessary_tool_use | Tool / grounding calls exceeded a documented threshold without changing the answer. |
| `F7` | budget_exhaustion / runaway_reasoning | Hop, token, or wall-clock budget hit before `EXECUTE`; or inner-MDP loop did not converge. |

`F3` and `F4` are the most important to surface: they indicate the system is "right for the wrong reasons" and would silently regress without this taxonomy.

The failure-class distribution is the **failure-quality** metric (§4). A run is regarded as healthier when the same Joint Success Rate has fewer `F3` / `F4` cases.

---

## 7. Judge protocol

Three layers, used in this order. Each later layer is allowed to *flip* an `EvidenceValid` verdict but never an `AnswerCorrect` verdict.

### 7.1 Automatic answer evaluation

| `answer_type` | Rule |
|---------------|------|
| `mcq` | Exact match on `option_id`. |
| `open_short` | Normalized string match (lowercase, strip punctuation, alias list per slice). |
| `open_free` | LLM-judge over `(question, gold_answer, system_answer)` returning `correct ∈ {0, 1}` with rationale. Single deterministic prompt; temperature 0. |

Automatic answer evaluation must produce a verdict for every instance and is the only authority on `AnswerCorrect`.

### 7.2 LLM-as-judge for evidence support

For each instance the judge receives:

- `question`
- `final_answer`
- `support_package.claim`
- `support_package.evidence_warrant` rendered as resolvable references (frame thumbnails, time-stamped clip windows, DOM snippets — whatever the adapter exposes)
- the `reasoning_trace_summary`

The judge produces:

```text
EvidenceVerdict = {
  exists:        bool   # do all evidence_refs resolve to a real artifact?
  supports:      bool   # do they support the final answer?
  covers_claim:  bool   # do they jointly cover every key sub-claim of the answer?
  defects:       [enum]  # subset of {hallucinated, mismatched, insufficient, off_topic}
  notes:         string
}
EvidenceValid := exists AND supports AND covers_claim AND defects == []
```

The judge prompt is fixed per release; its hash is logged with `eval_suite_id` ([PLAN-PIPELINE-ORCHESTRATOR.md §3a.2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a2-batch-evaluation-schedule)) so verdicts are reproducible.

### 7.3 Optional human audit subset

A fixed-size random subset (default **5%** of evaluated instances, minimum 50) is routed for human audit per release. Human verdicts:

- override the LLM judge for that subset (and the override is recorded);
- are aggregated into a per-release **judge-vs-human agreement** number;
- trigger judge-prompt revision when agreement drops below 0.85.

Human audit does not gate releases; it gates the judge.

---

## 8. Minimal benchmark slices

Keep this small. The first version uses one source benchmark (Video-Holmes) plus light internal annotation. It does **not** sprawl across multiple benchmarks (§11).

### 8.1 Difficulty slices

| Slice | Definition |
|-------|-----------|
| `easy` | Single-hop, direct visual evidence, ≤ 30 s clip. |
| `medium` | Two-hop reasoning OR temporal evidence, 30–120 s clip. |
| `hard` | Three+ hops, social / intentional reasoning OR multi-evidence aggregation, ≤ a few minutes clip. |

### 8.2 Hop-structure slices

| Slice | Definition |
|-------|-----------|
| `single_hop` | Answer derivable from one grounded evidence reference. |
| `multi_hop` | Answer requires composing ≥ 2 evidence references (chain or aggregation). |

### 8.3 Evidence-type slices

| Slice | Definition |
|-------|-----------|
| `direct_visual` | Evidence is what is visible in a frame (object, attribute, action). |
| `temporal` | Evidence is what happens across frames (order, change, duration). |
| `social_reasoning` | Evidence requires inferring intention, attention, role, or social relation. |

Each instance carries one label per dimension. Slices are intentionally orthogonal so a single instance contributes to one cell per dimension.

### 8.4 Slice sizing (first version)

| Dimension | Minimum size per slice |
|-----------|------------------------|
| Difficulty | 100 instances |
| Hop structure | 100 instances |
| Evidence type | 100 instances |

Total target: ~300–600 unique instances reused across overlapping slice labels. Small on purpose: the goal is fast, weekly evaluation, not a leaderboard run.

---

## 9. Reporting template

Each release prints **one canonical table** at the top of the evaluation report:

| Setting | Answer Acc | Evidence Support | **Joint Success** | Avg Hops | Avg Ground Calls | Avg Tool Calls | Cost ($/inst) | Latency (s/inst) |
|---------|-----------|------------------|-------------------|----------|------------------|----------------|---------------|------------------|
| `overall` | | | | | | | | |
| `easy` | | | | | | | | |
| `medium` | | | | | | | | |
| `hard` | | | | | | | | |
| `single_hop` | | | | | | | | |
| `multi_hop` | | | | | | | | |
| `direct_visual` | | | | | | | | |
| `temporal` | | | | | | | | |
| `social_reasoning` | | | | | | | | |

Rules:

- The `overall` row is the **headline**. Joint Success is bolded.
- Slice rows are sorted as listed above; do not reorder per release.
- Failure-class distribution (§6) appears in a second table directly below this one.
- Actor / Harness / transfer secondary axes appear as a third table.
- Anything not in the canonical table is appendix-only.

Every cell must come from the same `eval_suite_id` and `bank_snapshot_id` so the table is reproducible.

---

## 10. Phase-wise rollout

Three phases, smallest viable first.

### Phase E0 — Walking skeleton (first release)

- Implement `eval/` driver that consumes a JSONL of instances and writes the §3 record format.
- Implement §7.1 automatic answer evaluator for `mcq` only.
- Implement §7.2 LLM-judge with the fixed prompt; no human audit yet.
- Slices: `easy`, `medium`, `hard` only (single dimension).
- Headline: report the eval triple (Answer Accuracy, Evidence Support Rate, Joint Success Rate) plus failure taxonomy.
- Goal: prove the pipeline can produce a non-empty, reproducible Joint Success Rate.

### Phase E1 — Full first target

- Add `open_short` and `open_free` answer types to §7.1.
- Add hop-structure and evidence-type slices.
- Add the canonical table (§9) and the secondary axes (Actor / Harness / transfer).
- Wire `eval_suite_id` into [PLAN-PIPELINE-ORCHESTRATOR.md §3a.2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a2-batch-evaluation-schedule) so non-regression checks reference this suite.
- Goal: this plan is fully realized end-to-end.

### Phase E2 — Audit + transfer hardening

- Add §7.3 human audit subset and judge-agreement tracking.
- Add per-skill transfer rows (skills exercised on video that were originally crafted for game / web / os / visual reasoning) to surface transfer robustness.
- Wire failure-class distribution into the Crafter failure-cluster export ([PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md)).
- Goal: the eval becomes a closed loop with self-improvement signals.

Subsequent expansions (additional benchmarks, long-video, memory) are explicitly out of scope here; see §11.

---

## 11. Non-goals

Stated to prevent scope drift in the first version of this evaluation:

- **No memory evaluation.** Cross-episode memory, persistent stores, and long-term retention are not measured. The orchestrator's only state-keeping surface is episode-local ([PLAN-PIPELINE-ORCHESTRATOR.md §4](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#4-episode-local-evidence--trace-bookkeeping)).
- **No long-video setting.** Hour-scale or multi-session video is out of scope. Clips are short by construction (§8.1).
- **No multi-benchmark sprawl.** First version pins to one source benchmark family (Video-Holmes-style) plus light internal annotation. Adding more benchmarks is an explicit later phase, not a default.
- **No leaderboard chasing.** Slice sizes are intentionally small (§8.4) so this can run weekly. Larger runs are a later concern.
- **No new metric invention without a vote.** New metrics may be added to the appendix but cannot enter the canonical table without explicit owner sign-off, to keep the headline stable across releases.

---

## 12. Related documents

| Document | Relationship |
|----------|--------------|
| [README.md](../README.md) | Project framing; pins short-video as first arena. |
| [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) | §6 evaluation matrix; §0a four-way separation; §4 episode-local bookkeeping. |
| [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) | `SkillEpisode` records that feed Actor / Harness / transfer secondary axes. |
| [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) | Inner-MDP records that populate `reasoning_trace_summary` and Avg Hops. |
| [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) | Consumes failure taxonomy distributions for repair and synthesis. |
| [PLAN-VISUAL-GROUNDING.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md) | Source of `EvidenceRef` objects underlying §7.2. |
