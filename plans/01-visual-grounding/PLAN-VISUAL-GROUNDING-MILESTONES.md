# PLAN: Visual Grounding — Execution Milestones

**Purpose:** Turn the design in [PLAN-VISUAL-GROUNDING.md](PLAN-VISUAL-GROUNDING.md) into a concrete week-by-week build-out with routing rules, training order, and ablation checkpoints. This document answers "what do we actually build next and in what order."

**Depends on:** [Visual Grounding](PLAN-VISUAL-GROUNDING.md) (design), [Action Agent](../02-action-agent/PLAN-ACTION-AGENT.md) (consumer), [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) (consumer).

---

## 0. One-sentence commitment

Train a single Qwen3-VL-8B + LoRA (`schema_gen`) to produce a unified `<state>` schema from a screenshot across all 5 categories, use specialist tools only for uncertain cases, and keep 32B/72B as the slow offline teacher and reflection layer.

---

## 1. System architecture at inference

```
screenshot / frame(s)
    ↓
┌──────────────────────────────────────────────────────┐
│  Stage 1  Universal input contract                   │
│  • image(s) + goal + optional history                │
│  • ground(GroundingRequest) dispatches by domain     │
└──────────────────┬───────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────────┐
│  Stage 2  Direct parse (8B VLM + schema_gen LoRA)    │
│  • draft <state> schema                              │
│  • confidence / uncertainty flags                    │
│  • missing-field markers                             │
│  • optional tool plan                                │
└──────────────────┬───────────────────────────────────┘
                   ↓ Path A (accept) or Path B (repair)
┌──────────────────────────────────────────────────────┐
│  Stage 3  Tool-assisted repair (same 8B VLM)         │
│  • visual tools: detect_objects, spatial_query, etc. │
│  • video tools: get_frame, track_object, etc.        │
│  • cross-frame tools for temporal tasks              │
└──────────────────┬───────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────────┐
│  Stage 4  Schema validation                          │
│  • required slots present                            │
│  • format / coordinate / coverage checks             │
│  • action-relevant field availability                │
└──────────────────┬───────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────────┐
│  Stage 5  Action-agent consumption                   │
│  • state summary → intention → skill retrieval       │
│  • inner MDP hops → EXECUTE → env.step()             │
│  • GRPO training buffer                              │
└──────────────────────────────────────────────────────┘
```

---

## 2. Model roles

| Role | Model | When it runs | What it does |
|------|-------|:------------:|--------------|
| **Main online VLM** | Qwen3-VL-8B + `schema_gen` LoRA | Every step | Screenshot → schema; tool-calling decisions; schema repair after tool output |
| **Specialist tool backends** | YOLO, Florence-2, OCR, GroundingDINO | Called by 8B VLM | Element detection, region captioning, text extraction, open-vocabulary detection |
| **Slow teacher** | Qwen3-32B/72B or GPT-5.4 | Offline / rare | Label generation for hard cases, failure reflection, skill synthesis, protocol refinement |

**What NOT to put online:** 72B grounding everywhere, tool-calling for every example, five separate category-specific pipelines, multiple small VLMs competing in the hot path.

---

## 3. Routing policy

Three paths, decided after Stage 2 by the semantic validator (§12 Layer 1 in [Visual Grounding](PLAN-VISUAL-GROUNDING.md)):

### Path A — Accept direct parse

Use the 8B VLM output directly when **all** of:

- Required fields present (`<targets> target=` is set for env tasks)
- Entity minimum met (≥3 game, ≥5 browser, ≥1 image QA)
- ≤50 % of entities flagged `uncertainty=high`
- Every required section has ≥1 content line
- Schema passes `validate_schema()` format checks

**Target:** ≥90 % of steps take Path A after Phase 1 training.

### Path B — Tool repair

Call tools when **any** of:

- Key entities missing (entity minimum not met)
- OCR uncertain (text entities with `uncertainty=high`)
- Fine spatial relations matter (skill requires `$blocker` or precise `pos`)
- Target is ambiguous (`candidate_set` empty but skill expects candidates)
- Temporal evidence required (video domain, multi-frame needed)

The 8B VLM emits a tool plan; `run_tool_loop()` executes it; the VLM merges tool outputs into the schema and re-validates.

**Target:** <10 % of steps need Path B after Phase 2 training.

### Path C — Offline / slow escalation

Escalate to 32B/72B only when:

- Repeated failures on the same pattern (≥3 consecutive validation failures)
- Schema repeatedly fails validation after tool repair
- Hard case is valuable enough to cache as new supervision
- Reflection agent identifies a systematic grounding gap

Escalation outputs are stored as training examples and fed back into the next SFT round. This path should fire rarely (<1 % of online steps).

### Domain-aware starting head

| Domain | Default starting path | Rationale |
|--------|:--------------------:|-----------|
| gymv | Head 1 (heuristic) → validator → Path A or escalate | Free text state available |
| browser | Head 1 (heuristic) → validator → Path A or escalate | AXTree/DOM available |
| desktop | Head 3 (OmniParser) → 8B VLM → validator | No text state; local detection cheap |
| image_qa | 8B VLM direct → validator → Path B if needed | GroundingDINO auto-selected |
| video_qa | 8B VLM + video tool loop (always Path B) | Temporal search required |

### Environment / data status (snapshot 2026-04-20)

Tracks runtime readiness for the five domains. See
[`install/INSTALL_BENCHMARKS.md`](../../install/INSTALL_BENCHMARKS.md) for
setup commands. Each runtime is cloned from source and lives in its own
conda env because their transitive pins conflict (see table note below).

| Domain     | Conda env        | Upstream source                                                           | Runtime      | Benchmark       | Data  |
|------------|------------------|---------------------------------------------------------------------------|:------------:|-----------------|:-----:|
| gymv       | `gymv`           | [ModalMinds/gym-v](https://github.com/ModalMinds/gym-v)                   | **Ready**    | 179 procedurally-generated visual envs (single-turn / games / spatial / temporal) | bundled in the repo |
| browser    | `browsergym`     | [ServiceNow/BrowserGym](https://github.com/ServiceNow/BrowserGym)         | **Ready**    | MiniWoB++ / WebArena / VisualWebArena / AssistantBench | tasks bundled; WebArena/VWA need self-hosted sites |
| desktop    | `osworld`        | [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)                   | **Ready**    | OSWorld (Office, Daily, Professional) | Docker image / VMware VM pulled separately |
| image_qa   | `vlm_benchmarks` | HF VisualToolBench + TIR-Bench via `datasets`                             | **Ready**    | HF cache                   | Pull `ScaleAI/VisualToolBench` + `Agents-X/TIR-Bench` (see INSTALL_BENCHMARKS.md §4) |
| video_qa   | `vlm_benchmarks` | pip-installed Video-Holmes/SIV-Bench loaders + `decord`/`av`              | **Ready**    | Video-Holmes               | **Downloaded** — 503 cropped clips + 1 837 test Qs at `data/Video-Holmes/Benchmark/` |

**Why three runtime envs, not one?** The three interactive runtimes have
hard-pinned dependency sets that cannot co-resolve:

| Package      | gym-v        | BrowserGym | OSWorld / desktop-env |
|--------------|--------------|------------|-----------------------|
| `gymnasium`  | `>=1.2.2`    | `>=0.27`   | `~=0.28.1`            |
| `playwright` | —            | `==1.44`   | unpinned (often 1.5x) |
| `transformers` | —          | —          | `~=4.35.2`            |
| `torch`      | —            | (VWA scorer)| `~=2.5.0`            |
| `tqdm`       | —            | `>=4.66.2` (workarena only) | `~=4.65.0` |

Gymnasium 0.28 → 1.x is an API-breaking transition (signatures, wrapper
order, termination semantics), so gym-v and OSWorld cannot share an
interpreter.  BrowserGym-core accepts both ranges, but we keep it in its
own env so the playwright-1.44 and libwebarena pins don't ripple into
the grounding stack.  The `vlm_benchmarks` env covers image_qa and
video_qa plus the full `vlm_wrapper` grounding pipeline.

---

## 4. Two regimes for the 5 categories

### Regime A — Static / single-frame

For browser, desktop, images, most games:

```
screenshot → 8B VLM → optional tools → schema
```

### Regime B — Temporal

For video or temporally heavy tasks:

```
sample frames / temporal navigation
    → localize candidate moments (find_moment, detect_scene_changes)
    → run frame grounding on those moments
    → store evidence in schema <evidence> section
```

The same `ground()` entry point handles both; the domain field selects the tool registry (visual-only vs video_visual).

---

## 5. Training plan

### Phase 0 — Supervision collection (Weeks 1–2)

**Goal:** Build the grounding supervision dataset before any VLM training.

| Source | Role | Module |
|--------|------|--------|
| Head 1 heuristic | Primary labels for gymv + browser (free, fast, deterministic) | `gymv_heuristic.py`, `browser_heuristic.py` |
| Head 3 OmniParser | Element-level labels with bboxes for desktop + cluttered UIs | `grounding.py`, `grounding_browsergym.py` |
| Head 2 GPT-4o | Vision labels for image/video QA and hard cases | `gymv_adapter.py`, `browser_adapter.py` |
| Tool loop traces | Multi-hop evidence chains → SFT data for tool-use training | `tool_loop.py` |
| 32B/72B teacher | Hard-case labels where heuristic + GPT-4o disagree | Offline batch |

**Collection pipeline per step:**

```
1. Collect native observation (screenshot + text state if available)
2. Run domain adapter → raw structured schema
3. Run heuristic head on same observation → cross-validate
4. Flag disagreements → send disagreements to GPT-4o or 32B teacher
5. Apply truncation/pruning to fit token budget (300–800 tokens)
6. Validate format: all tags present, entity refs consistent
7. Store pair: input = image + system_prompt; output = schema text
```

**Data budget:** ~3–5 K labeled examples per domain. GPT-4o labeling cost: ~$30–50 per domain.

**Deliverables:**
- [ ] Collection script that runs Gym-V episodes and stores (frame, schema) pairs
- [ ] Collection script that runs BrowserGym episodes and stores (screenshot, schema) pairs
- [ ] Cross-validation harness: heuristic vs GPT-4o agreement rate
- [ ] Dataset split: 80 % train / 10 % val / 10 % test per domain

### Phase 1 — Direct grounding SFT (Weeks 3–5)

**Goal:** Train the 8B VLM to produce a valid schema from a screenshot alone.

**Training order (easiest → hardest):**

| Order | Domain | Why first/later | Expected examples |
|-------|--------|-----------------|:-----------------:|
| 1 | Gym-V games (2048, Sokoban, Minesweeper) | Grid layouts, ≤20 entities, clean heuristic labels | 3–5 K |
| 2 | MiniWoB++ (simple browser) | Small pages, clear elements, easy validation | 3–5 K |
| 3 | WebArena (complex browser) | Cluttered real-web pages, many entities | 3–5 K |
| 4 | Image QA (VisualToolBench, TIR-Bench) | HF image benchmarks, no env context | 3–5 K |
| 5 | Video QA (SIV-Bench, Video-Holmes) | Temporal, multi-frame — hardest | 2–3 K |

**What to train:**
- Qwen3-VL-8B base + `schema_gen` LoRA
- Input: screenshot (single image) + system prompt with schema spec
- Output: full `<state>…</state>` tagged text
- Loss: standard next-token SFT on the schema output

**Metrics:**
- Field accuracy per section (entities, attributes, relations, state_flags, targets)
- Format compliance rate (regex parse success)
- Entity coverage (IoU against heuristic ground truth)
- Target-slot accuracy (target, blocker correct)

**Validation checkpoint (end of Phase 1):**
- Does the VLM produce schemas the skill pipeline can consume?
- Does structured schema beat raw wrapper text for skill retrieval relevance?
- Path A acceptance rate ≥70 % on gymv, ≥50 % on browser

### Phase 2 — Tool-use SFT (Weeks 5–7)

**Goal:** Train the 8B VLM to decide when tools are needed and how to use them.

**What to train (on top of Phase 1 checkpoint):**
- When direct parse is sufficient vs when tools are needed
- Which tool to call (from the visual / video / cross-frame registries)
- How to incorporate tool outputs back into the schema
- Multi-hop chaining: `detect_objects → spatial_query → count_objects`

**Training data:** Tool-loop traces from Phase 0 collection. Each trace is an SFT example:

```
Input:  screenshot + system prompt + tool definitions
Output: [tool_call: detect_objects(...)] ... [tool_call: spatial_query(...)] ... <state>...</state>
```

**Training order:**
1. Single-tool calls (detect_objects, read_text_region) on cases where direct parse failed
2. Two-hop chains (detect → spatial_query) on relation-heavy examples
3. Multi-hop chains (3+ tools) on video QA and complex image QA
4. Tool-refusal learning: examples where direct parse was correct and tools were unnecessary

**Validation checkpoint (end of Phase 2):**
- Path A acceptance rate ≥90 % on gymv, ≥80 % on browser
- Path B (tool repair) recovers ≥80 % of remaining failures
- Tool-call precision: ≥85 % of tool calls are actually needed
- Overall schema completeness ≥95 % across all domains

### Phase 3 — Actor training (Weeks 7–9)

**Goal:** Train the action agent on top of the grounded schema.

**Prerequisite:** Phase 2 schema quality is good enough that the actor rarely sees broken schemas.

**What to train:**
- `hop_select` LoRA: typed next-hop router — schema + short typed trace → `(NEXT_HOP, TARGET)` from `{GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE}`, with a 0–3 hop cap (see [Action Agent §5](../02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control))
- `skill_select` LoRA: schema → which skill to invoke
- GRPO with trajectory-level reward: `r_env + r_follow + r_cost`

**Integration path (from [Action Agent §7](../02-action-agent/PLAN-ACTION-AGENT.md)):**
1. `get_state_summary()` receives `<state>` schema directly from VLM grounding
2. Entity-referenced actions: `click(e5)` instead of `click(400,510)`
3. `<uncertainty>` section drives GROUND hop insertion

**Validation checkpoint:**
- Actor performance on gymv games with schema input vs raw text input
- Inner MDP hop efficiency: average hops per outer step, unnecessary GROUND rate
- Skill retrieval quality: does structured schema improve applicability scoring?

### Phase 4 — Co-evolution (Weeks 9+)

**Goal:** Close the loop so all components improve together.

```
Actor produces trajectories
    ↓
Skill Bank updates from trajectories (SEGMENT, CONTRACT, CURATOR)
    ↓
Synthesis-reflection agent (32B/72B frozen) studies failures
    ↓
Better grounding traces + repaired protocols feed back
    ↓
Next SFT round for schema_gen incorporates hard-case labels
    ↓
Repeat
```

**Timescale separation (from [Action Agent §6](../02-action-agent/PLAN-ACTION-AGENT.md)):**
- Fast: Actor GRPO every training iteration
- Medium: Skill-bank operational updates every few iterations
- Slow: Synthesis-reflection proposals every N episodes, acceptance-gated

---

## 6. Semantic schema validator (implements §12 Layer 1)

**Status:** TODO — first module to build.

**Implementation target:** `vlm_wrapper/schema.py` → `semantic_validate(schema, domain) → ValidationResult`

| Check | Threshold | On failure |
|-------|-----------|------------|
| Slot population | `<targets> target=` set (not `null`) for env tasks | Re-ground or escalate |
| Entity minimum | ≥3 (games), ≥5 (browser), ≥1 (image QA) | Re-ground or escalate |
| Uncertainty budget | ≤50 % of entities with `uncertainty=high` | Escalate head |
| Section content | Every required section has ≥1 content line | Escalate head |
| Relation coverage | ≥1 relation if ≥3 entities | Warning (soft) |
| Coordinate consistency | `pos=` values within image bounds | Warning |
| Entity reference integrity | All `eN` references in relations/targets exist in `<entities>` | Error |

`ValidationResult` fields: `valid: bool`, `warnings: list[str]`, `errors: list[str]`, `missing_slots: list[str]`, `escalation_recommended: bool`.

---

## 7. Cascaded head escalation (implements §12 Layer 2)

**Status:** TODO — build after validator.

**Implementation target:** `vlm_wrapper/ground.py` → wrap `ground()` with escalation logic.

```
Domain-default head (see §3 table)
    ↓ semantic_validate()
    ↓ passes? → return schema (Path A)
    ↓ fails?
Next head in chain
    ↓ semantic_validate()
    ↓ passes? → return schema
    ↓ fails?
Tool loop (Path B)
    ↓ semantic_validate()
    ↓ passes? → return schema
    ↓ still fails?
Return best attempt + high uncertainty flags (Path C candidate)
```

**Escalation chain by domain:**

| Domain | Head 1 | Head 2 | Head 3 | Tool loop |
|--------|:------:|:------:|:------:|:---------:|
| gymv | Heuristic | 8B VLM | — | On demand |
| browser | Heuristic | 8B VLM | OmniParser | On demand |
| desktop | OmniParser | 8B VLM | — | On demand |
| image_qa | — | 8B VLM | — | Always |
| video_qa | — | — | — | Always |

**Target:** <10 % escalation rate from default head to next head.

---

## 8. Week-by-week schedule

### Weeks 1–2: Foundation

| # | Task | Module | Deliverable |
|---|------|--------|-------------|
| 1.1 | Semantic schema validator | `vlm_wrapper/schema.py` | `semantic_validate()` with all 7 checks |
| 1.2 | Cascaded head escalation | `vlm_wrapper/ground.py` | Escalation wrapper around `ground()` |
| 1.3 | Gym-V data collection script | `labeling/` | Runs episodes, stores (frame, heuristic_schema, gpt4o_schema) triples |
| 1.4 | BrowserGym data collection script | `labeling/` | Same for browser benchmarks |
| 1.5 | Cross-validation harness | `labeling/` | Agreement rate between Head 1 and Head 2 |

**Exit criteria:** Validator passes on ≥95 % of heuristic-generated schemas. Collection scripts produce ≥500 examples per domain. Escalation chain works end-to-end on synthetic examples.

### Weeks 3–4: Gym-V SFT

| # | Task | Module | Deliverable |
|---|------|--------|-------------|
| 2.1 | Collect 3–5 K Gym-V training pairs | `labeling/` | Dataset in SFT format |
| 2.2 | Train `schema_gen` LoRA on Gym-V | `trainer/SFT/` | Checkpoint with field accuracy ≥85 % |
| 2.3 | Evaluate Path A acceptance rate | `ablation_study/` | ≥70 % of schemas pass validator without tools |
| 2.4 | **Ablation A1:** schema vs raw text for skill retrieval | `ablation_study/` | Retrieval relevance comparison |

### Weeks 4–5: Browser SFT

| # | Task | Module | Deliverable |
|---|------|--------|-------------|
| 3.1 | Collect 3–5 K browser training pairs (MiniWoB++ → WebArena) | `labeling/` | Dataset |
| 3.2 | Fine-tune `schema_gen` on browser data (on top of Gym-V checkpoint) | `trainer/SFT/` | Checkpoint |
| 3.3 | Cross-domain eval: does Gym-V training help browser grounding? | `ablation_study/` | Transfer metric |
| 3.4 | **Ablation A2:** single-domain vs multi-domain SFT | `ablation_study/` | Accuracy comparison |

### Weeks 5–7: Tool-use SFT

| # | Task | Module | Deliverable |
|---|------|--------|-------------|
| 4.1 | Collect tool-loop traces for failed direct-parse examples | `labeling/` | Tool-trace SFT dataset |
| 4.2 | Train tool-calling on top of Phase 1 checkpoint | `trainer/SFT/` | Checkpoint with tool precision ≥85 % |
| 4.3 | Add image QA data (VTB + TIR-Bench) | `labeling/` | Benchmark loader + dataset |
| 4.4 | Add video QA data (SIV-Bench + Video-Holmes) | `labeling/` | Benchmark loader + dataset |
| 4.5 | **Ablation A3:** tool-use vs direct-only parse | `ablation_study/` | Path B recovery rate |
| 4.6 | **Ablation A4:** GroundingDINO vs OmniParser for natural images | `ablation_study/` | Detection backend comparison |

### Weeks 7–9: Actor integration

| # | Task | Module | Deliverable |
|---|------|--------|-------------|
| 5.1 | Wire `<state>` schema into `get_state_summary()` | `decision_agents/agent_helper.py` | Schema-as-input mode |
| 5.2 | Entity-referenced action format | `decision_agents/agent.py` | `click(e5)` parsing |
| 5.3 | Lightweight inner-MDP `hop_select` adapter (typed router, ≤3 hops) | `decision_agents/` | `{GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE}` loop |
| 5.4 | GRPO training with schema input | `trainer/` | Actor checkpoint |
| 5.5 | **Ablation A5:** actor with schema vs actor with raw text | `ablation_study/` | Task success rate comparison |

### Weeks 9+: Co-evolution

| # | Task | Module | Deliverable |
|---|------|--------|-------------|
| 6.1 | Skill bank ingestion of schema-based trajectories | `skill_agents/` | Segmentation + contract learning on schema traces |
| 6.2 | Synthesis-reflection integration | `skill_agents/crafter/` | 32B/72B proposes, acceptance gate filters |
| 6.3 | Hard-case feedback loop | `labeling/` + `trainer/SFT/` | Failed schemas → teacher labels → retrain |
| 6.4 | Multi-domain co-training | `trainer/` | Single `schema_gen` LoRA across all domains |

---

## 9. Ablation plan

| ID | Question | Comparison | When | Key metric |
|----|----------|------------|------|------------|
| A1 | Does structured schema improve skill retrieval? | Schema input vs raw `obs.text` for `SkillQueryEngine.select()` | Week 4 | Retrieval relevance (cosine sim), applicability score |
| A2 | Does multi-domain SFT help or hurt? | Gym-V only vs Gym-V + browser joint training | Week 5 | Per-domain field accuracy, cross-domain transfer |
| A3 | Does tool-use training improve schema quality? | Direct-only 8B vs direct + tool-repair 8B | Week 7 | Path A/B acceptance rates, entity coverage |
| A4 | Which detection backend for natural images? | GroundingDINO vs OmniParser on VTB/TIR-Bench | Week 6 | mAP, entity grounding accuracy |
| A5 | Does grounded schema improve the actor? | Actor + schema vs actor + raw text (same GRPO budget) | Week 9 | Task success rate, reward per episode, hop efficiency |
| A6 | How much does cascaded escalation cost? | Single-head only vs cascaded chain | Week 3 | Schema completeness, avg latency per step |
| A7 | Is the 8B VLM sufficient or do we need larger? | 8B direct parse vs 32B direct parse (offline eval) | Week 5 | Field accuracy ceiling, format compliance |

---

## 10. Implementation status (from [Visual Grounding §16](PLAN-VISUAL-GROUNDING.md))

Snapshot 2026-04-21.  Newly-completed rows (this iteration) link to
their source files so the next pair of eyes can audit them quickly:

- Benchmark loaders → `visual_reasoning_wrapper/benchmarks/{visual_toolbench.py, tir_bench.py, siv_bench.py}`
  (registered in `visual_reasoning_wrapper/benchmarks/__init__.py`).
- Evaluation harness → `vlm_wrapper/eval/{metrics.py, harness.py, run_eval.py}`,
  CLI: `python -m vlm_wrapper.eval.run_eval --benchmark visual_toolbench|tir_bench|video_holmes|siv_bench`.
- Phase-0 data collection → `labeling/grounding/{collect_gymv.py,
  collect_browser.py, cross_validate.py}` — outputs land in
  `labeling/output/grounding/{gymv,browser}/triples.jsonl` with
  `hard_cases.jsonl` next to them.
- `schema_gen` LoRA scaffolding →
  `trainer/SFT/schema_gen/{config.py, data_loader.py, train.py}`; run
  `python -m trainer.SFT.schema_gen.train --inspect_only` to dry-run
  the dataset assembly, drop `--inspect_only` to launch the real SFT.
- `<evidence_refs>` + `GroundingRecord` → `vlm_wrapper/schema.py`
  (`_SECTION_EVIDENCE_REFS`, `parse_evidence_refs`, `validate_evidence_refs`).

| Component | Status | Milestone target |
|-----------|--------|:----------------:|
| Unified pipeline `ground()` | **Done** | — |
| Adaptive schema (evidence + answer) | **Done** | — |
| GroundingDINO backend | **Done** | — |
| Visual / video / cross-frame tool registries | **Done** | — |
| OmniParser grounding | **Done** | — |
| BrowserGym grounding adapter | **Done** | — |
| Tool-calling loop | **Done** | — |
| Heuristic adapters | **Done** | — |
| Vision adapters (Head 2) | **Done** | — |
| Schema utilities | **Done** | — |
| Demo script | **Done** | — |
| **Semantic schema validator** | **Done** | Week 1 |
| **Cascaded head escalation** | **Done** | Week 2 |
| **Benchmark loaders** — VisualToolBench / TIR-Bench | **Done** | Week 6 |
| **Benchmark loaders** — Video-Holmes | **Done** | Week 6 |
| **Benchmark loaders** — SIV-Bench | **Done** | Week 6 |
| **Evaluation harness** | **Done** | Week 4 |
| **Re-observation (Option B) for GROUND hops** | **Done** | Week 8 |
| **Qwen3-VL-8B training pipeline** | **Scaffolded** | Week 3 |
| **Data collection scripts** (Gym-V, BrowserGym) | **Done** | Week 1 |
| **Actor schema integration** | **Done** | Week 7 |
| **`<evidence_refs>` schema section + `GroundingRecord`** | **Done** | Week 1 |
| **Heuristic ↔ vision cross-validation harness** | **Done** | Week 2 |

### Five-domain pipeline feasibility (snapshot 2026-04-19)

Inference-time pipeline is green on all five milestone tasks. Each row is wired end-to-end into `vlm_wrapper.ground.cascaded_ground()` and covered by a live test.

| Domain | Chain (`_ESCALATION_CHAINS`) | Tool registry | Required sections | Entity min | Benchmark data | Live test |
|---|---|---|---|---:|---|---|
| `gymv` | `vlm → tool_loop` | `tools_visual + tools_gymv` | `entities, attributes, state_flags, targets, actions` | 3 | env-provided | `gymv_wrapper/tests/test_gpt4o_parsers.py::test_live_gymv_schema` + `vlm_wrapper/tests/test_gpt4o_parsers.py::test_live_gymv_tool_loop_schema` |
| `browser` | `vlm → omniparser → tool_loop` | `tools_visual + tools_browser` | `entities, attributes, state_flags, targets, actions` | 5 | env-provided | `browsergym_wrapper/tests/test_gpt4o_parsers.py::test_live_browser_schema` |
| `desktop` | `omniparser → vlm → tool_loop` | `tools_visual + tools_osworld` | `entities, attributes, state_flags, targets, actions` | 5 | env-provided | `osworld_wrapper/tests/test_gpt4o_parsers.py::test_live_desktop_schema` |
| `image_qa` | `vlm → tool_loop` | `tools_visual` (GroundingDINO-preferred) | `entities, attributes, state_flags, targets, evidence, answer` | 1 | HF VTB + TIR-Bench ✅ | `visual_reasoning_wrapper/tests/test_gpt4o_parsers.py::test_live_tir_bench_schema` |
| `video_qa` | `tool_loop` | `tools_video_visual` (temporal + visual + cross-frame) | `entities, state_flags, targets, evidence, answer` | 1 | `data/Video-Holmes/Benchmark/` ✅ | `visual_reasoning_wrapper/tests/test_gpt4o_parsers.py::test_live_video_holmes_schema` |

**Design choices locked in (by code + test, not just plan):**
- `desktop` has no text-heuristic head (a11y trees are too noisy; OmniParser + VLM covers it).
- `video_qa` starts at `tool_loop` because single-shot VLM over a frame grid can't call `sample_frames` / `find_moment` / `track_object`.  The `vlm` head is still reachable via `chain=["vlm", "tool_loop"]`.
- `browser` OmniParser fallback (when `context["obs"]` lacks `screenshot`) now appends an explicit warning to `GroundingResult.warnings` so the cascade telemetry records the degraded mode — it no longer silently runs image-only.
- Head 2 (`_attempt_vlm`) for `gymv` / `browser` / `desktop` delegates to the same `generate_label` adapter that data-collection scripts call directly, so there is exactly one prompt + one validator on both the cascade path and the labeling path.

---

## 11. Risks specific to the execution plan

| Risk | Impact | Mitigation |
|------|--------|------------|
| 8B VLM can't reach ≥85 % field accuracy on browser | Blocks Phase 1 exit | Fall back to cascaded escalation; increase tool-use reliance |
| Tool-use SFT degrades direct-parse quality | Regression on Path A rate | Train with mixed batches (direct + tool-use); monitor Path A rate every checkpoint |
| Heuristic labels have systematic biases | Training data quality | Cross-validate every label with GPT-4o; flag and exclude disagreements above threshold |
| Actor doesn't benefit from schema (A5 ablation negative) | Questions the whole approach | Check whether the schema format or the schema content is the problem; try simpler key=value format |
| Co-evolution destabilizes both grounding and actor | Training instability | Enforce timescale separation: freeze `schema_gen` LoRA during actor GRPO phases |
| Video QA data is insufficient for temporal grounding | Weak Regime B performance | Supplement with synthetic temporal tasks from Gym-V replays |

---

## 12. Dependencies and prerequisites

| Prerequisite | Needed by | Status |
|-------------|-----------|--------|
| vLLM serving with LoRA hot-swap for Qwen3-VL-8B | Phase 1 SFT eval | Setup needed |
| GPU allocation for OmniParser + GroundingDINO | Phase 0 collection | Available (existing code works) |
| GPT-4o / GPT-5.4 API access | Phase 0 label generation, Phase 0 cross-validation | Available |
| Gym-V environment installation | Phase 0 collection | Assumed ready |
| BrowserGym environment installation | Phase 0 collection | Assumed ready |
| VTB / TIR-Bench / SIV-Bench / Video-Holmes datasets | Phase 2 benchmark eval | HF + local video paths (INSTALL_BENCHMARKS.md) |

---

## 13. Success criteria

| Milestone | Criterion | Target |
|-----------|-----------|--------|
| Phase 0 complete | ≥3 K validated (frame, schema) pairs per domain | 2 domains |
| Phase 1 complete | 8B VLM field accuracy ≥85 % on Gym-V, ≥75 % on browser | Path A ≥70 % |
| Phase 2 complete | Tool-call precision ≥85 %; overall schema completeness ≥95 % | Path A ≥90 % |
| Phase 3 complete | Actor with schema outperforms actor with raw text on ≥3 games | Ablation A5 positive |
| Phase 4 started | Co-evolution loop runs without divergence for ≥5 iterations | Stable metrics |
