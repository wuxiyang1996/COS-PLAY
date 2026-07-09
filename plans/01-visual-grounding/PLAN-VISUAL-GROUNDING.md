# PLAN: Visual Grounding

**Scope:** VLM visual parser — pixels → structured `<state>` schema — across games, browser, desktop, images, and video. Includes grounding heads, observation schema, adapters, training pipeline, benchmark evaluation, and multi-hop tool-calling reasoning.

**Upstream:** Raw observations from three interactive runtimes — [Gym-V](https://github.com/ModalMinds/gym-v) (179 procedurally-generated visual environments with a Gymnasium-compatible API, across single-turn reasoning / multi-turn games / spatial navigation / retro arcade), [BrowserGym](https://github.com/ServiceNow/BrowserGym) (MiniWoB++ / WebArena / VisualWebArena / AssistantBench), [OSWorld](https://github.com/xlang-ai/OSWorld) (desktop tasks over Office/Daily/Professional suites) — plus image benchmarks on HuggingFace (**VisualToolBench**, **TIR-Bench**) and local video benchmarks (**Video-Holmes**, **SIV-Bench**). Each runtime is installed in its own conda env because of hard-pinned dependency conflicts (gymnasium 1.2+ for Gym-V vs 0.28 for OSWorld, transformers 4.35 for OSWorld vs 4.51+ for the grounding pipeline); see [`install/INSTALL_BENCHMARKS.md`](../../install/INSTALL_BENCHMARKS.md).
**Downstream:** [Action Agent](../02-action-agent/PLAN-ACTION-AGENT.md) consumes the structured schema; [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) uses schemas for contract learning and retrieval.

---

## 1. Goal

Train a VLM to produce structured scene summaries from visual observations (pixels only), using the native text state that Gym-V and BrowserGym already provide as free supervision. The trained VLM either grounds visually or calls env APIs for information not observable in pixels. Output plugs directly into the existing skill retrieval → reasoning → action pipeline.

**Key insight:** Both domains already emit rich text state (Gym-V wrapper captions/rules, BrowserGym AXTree/DOM). This text state is the training label — no manual annotation needed. The VLM learns to see what the text state already knows.

---

## 2. Collect paired training data from environments

Every episode step produces a free (visual input, text state) pair.

**Gym-V — [ModalMinds/gym-v](https://github.com/ModalMinds/gym-v)**

- Visual input: raw frame / short frame stack from any of the 179 envs.
- Text state: captions, rules, state descriptions, and structured metadata produced by Gym-V's composable observation wrappers (`Observation(image, text, metadata)`). Agents receive the same `obs["agent_0"].text` that we use as the training label.
- Start with the richest-text wrappers (games + single-turn reasoning); expand into spatial / temporal as label quality allows. Difficulty presets (levels 0/1/2) give curriculum knobs for free.
- Env: `gymv` (gymnasium ≥1.2.2).

**BrowserGym — [ServiceNow/BrowserGym](https://github.com/ServiceNow/BrowserGym)**

- Visual input: screenshot.
- Text state: AXTree / DOM (element types, labels, form state, validation, hierarchy).
- Available across MiniWoB++, WebArena, VisualWebArena, AssistantBench, and the WorkArena fork (optional `--no-deps` install).
- Env: `browsergym` (playwright==1.44).

**OSWorld — [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)**

- Visual input: desktop screenshot at 1920×1080.
- Text state: a11y tree extracted from the guest VM via `desktop-env`'s pyautogui / lxml adapters.
- 369 tasks across Office / Daily / Professional suites; tasks drive VMware, VirtualBox, Docker, or AWS backends.
- Env: `osworld` (gymnasium~=0.28.1, transformers~=4.35.2).

No labeling pipeline needed — run episodes, collect pairs.

---

## 3. Canonical schema

Map heterogeneous text state formats into one shared target format before using them as training labels.

**This is the main design work.** The schema should be derived from what the environments actually emit, not designed in the abstract.

Candidate fields (refined empirically during normalization):

- `domain`, `task`, `goal`
- `entities` (with type, attributes, source)
- `relations` (spatial, functional, grouping)
- `state_flags` (dialog_open, input_missing, page_changed, ...)
- `salient_targets` (target, blocker, constraint)
- `uncertainty` (per-field confidence where grounding is ambiguous)

Shared slot names across domains (for downstream skill transfer): `target`, `blocker`, `constraint`, `candidate_set`, `history_anchor`.

Schema will iterate — expect 3-4 revisions as real data reveals missing or useless fields.

### 3a. Shared output schema (both domains emit this)

```text
<state>
domain={browser|gymv}
task={task_id or env_id}
goal={goal string, ≤60 tokens}
step={int}

<entities>
e1[type={element|object|region|text}, label={str}, bid={str|null}, pos={x,y,w,h|null}]
e2[...]
...

<attributes>
e1.state={visible|hidden|disabled|focused|checked|...}
e1.value={str|null}
e2.state=...
...

<relations>
contains(e1,e2)
adjacent(e3,e4)
blocks(e5,e1)
grouped(e2,e3,e4)
...

<state_flags>
progress={float 0-1 or null}
phase={early|mid|late|null}
error={str|null}
dialog_open={bool}
input_pending={bool}

<targets>
target={eid}
blocker={eid|null}
constraint={str|null}
candidate_set=[eid,eid,...]

<uncertainty>
e1.label={high|medium|low}
e3.pos={medium}

<actions>
a1={action_string}
a2={action_string}
...
</state>
```

**Why this format:**
- Each `<section>` tag is a clear generation boundary — the model learns section ordering.
- Entity references (`e1`, `e2`) are short tokens the model can reuse across sections without repeating long names.
- No nested braces/brackets beyond one level — critical for 8B model reliability.
- Total token count for a typical web page: ~400–600 tokens. For a game frame: ~200–400 tokens.

**Schema as inner MDP state:** Under the two-level MDP (see [Action Agent §5](../02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control)), this schema is the state representation for the inner reasoning MDP. Each GROUND/CHECK hop updates entities, relations, or uncertainty. The `<targets>` and `<uncertainty>` sections drive the agent's decision to continue reasoning (more hops) or act (EXECUTE). Shared slot names (`target`, `blocker`, `constraint`, `candidate_set`, `history_anchor`) are the vocabulary that makes reasoning skills transferable across domains.

**`GroundingRecord` as canonical `evidence_out`.** Under the evidence-driven invariant ([PLAN-SKILL-BANK.md §0.3](../03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills)), a `GroundingRecord` emitted by any grounding head (§4 Head 1 / Head 2 / Head 3, or by a tool-calling loop) is the **canonical `evidence_out`** for a `GATHER`-role skill ([PLAN-VISUAL-SKILLS.md §2](PLAN-VISUAL-SKILLS.md#2-two-kinds-of-skill-effects)). It carries:

- `evidence_id` — fresh unique ID appended to `<state>.evidence_refs` (new `<evidence_refs>` section; additive to the schema above)
- `source` — `heuristic | vision | omniparser | tool:<tool_id>`
- `kind` — `entity | region | frame | temporal_window | text_span | dom_node | desktop_object`
- `anchor` — what the grounding is anchored to (image bbox, clip frame id, DOM bid, etc.)
- `confidence` — mirrors `<uncertainty>` for this record
- `verified_by` — optional back-reference to a `VERIFY`-role episode that checked this record

A `GroundingRecord` that is written to `<state>.evidence_refs` is what the Harness counts as `evidence_out` at Gate G0. A grounding call that updates only `<entities>` / `<attributes>` without emitting a corresponding `evidence_refs` entry **does not** satisfy Gate G0, because downstream `REASON` / `COMMIT` skills cannot cite it as warrant. This is deliberate: it forces grounding to be addressable evidence, not ambient state mutation.

### 3b. Design constraints for small VLM (Qwen3-VL-8B)

- Target structured summary: **300–800 tokens** (output side).
- Input side: 1 image + short system prompt + optional history ≈ 1–2K text tokens + image tokens.
- Flat tagged format (not deeply-nested JSON) — easier for 8B models to produce reliably.
- Every field has a fixed tag name — parse with simple regex, no JSON decoder needed at inference.
- Fields are **ordered** (spatial → semantic → action) so the model can generate left-to-right without backtracking.

### 3c. Schema design decisions for 8B model

| Decision | Choice | Rationale |
|---|---|---|
| Format | Tagged text, not JSON | 8B models produce fewer bracket-matching errors with flat tags |
| Entity limit | ≤ 25 (browser), ≤ 20 (game) | Keeps output < 600 tokens; 8B accuracy degrades with long structured output |
| Position format | `x,y,w,h` integers | Pixel coords (browser) or grid coords (game); no floats to reduce token count |
| Relation format | `verb(eid,eid)` | Lisp-like prefix is shorter and more regular than natural language |
| Action format | Raw env action strings | No abstraction layer — model outputs what the environment accepts |
| Section ordering | spatial → semantic → action | Matches human scan order; model can attend to earlier sections when generating later ones |
| Uncertainty | Per-entity, 3 levels | `high/medium/low` — finer granularity isn't reliable at 8B |
| Candidate actions | Top 3–5 | Keeps action section short; full action space in system prompt |

### 3d. Schema versioning

- **v0.1** — initial schema, tested on 2048 + Sokoban + MiniWoB++
- **v0.2** — revised after first training run (expect field additions/removals)
- **v1.0** — stable schema validated across ≥ 3 Gym-V categories + ≥ 1 BrowserGym benchmark

Schema version goes in the system prompt so the model can be trained on mixed versions during transition.

---

## 4. Three grounding heads

### Head 1 — Heuristic (text-in → schema-out)

Fast, free, deterministic. Parses native text state (obs.text, AXTree/DOM) into the schema with regex/tree-walking. Good for: real-time RL rollouts, cheap baselines, validation.

- **Implementation:** `vlm_wrapper/gymv_heuristic.py`, `vlm_wrapper/browser_heuristic.py`

### Head 2 — Vision (image-in → schema-out)

Sends the screenshot to GPT-4o (or any vision LLM) and receives the schema. The image is the primary input; native text is optional grounding context. Good for: training-label generation, Qwen3-VL-8B distillation.

- **Implementation:** `vlm_wrapper/gymv_adapter.py`, `vlm_wrapper/browser_adapter.py`

### Head 3 — OmniParser-v2 (image-in → local models → schema-out)

Uses Microsoft's OmniParser-v2 (YOLO icon detector + Florence-2 icon captioner + OCR) to parse a GUI screenshot into structured elements with bounding boxes — all locally, no API calls. ~0.6s/frame on GPU.

- **Implementation:** `vlm_wrapper/grounding.py`, `vlm_wrapper/grounding_browsergym.py`

### Tool-calling loop (multi-hop visual reasoning / inner MDP)

VLM sees screenshot + system prompt + tool definitions → calls tools to gather ground-truth data → produces final schema. The trace becomes SFT data.

Under the two-level MDP (see [Action Agent §5](../02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control)), each tool call in the loop maps to an inner MDP action:

| Tool call pattern | Inner MDP action | Schema update |
|-------------------|-----------------|---------------|
| `detect_objects`, `visual_search` | GROUND(query) | Add/update entities |
| `spatial_query`, `check_relation` | CHECK(predicate) | Update relations, state_flags |
| `describe_region`, `classify_scene` | GROUND(detail) | Refine attributes, reduce uncertainty |
| `get_frame`, `sample_frames` | GROUND(temporal) | Add temporal entities/events |
| Final schema output | COMMIT or EXECUTE | Complete schema for downstream |

**Design decision — re-observation between hops:**
- **Option A (default):** Hops operate on the same `<state>` schema, only updating an internal scratchpad. Cheaper, faster. Used for games/web.
- **Option B (selective):** GROUND actions can trigger re-rendering or zooming into a region. More expensive but handles fine-grained visual detail. Used for visual QA and video benchmarks.

- **Implementation:** `vlm_wrapper/tool_loop.py`

| Head | Input | Grounding method | Latency | Best for |
|------|-------|------------------|---------|----------|
| Head 1 | Text state | Regex/tree-walking | ~ms | Interactive envs with text state |
| Head 2 | Screenshot → VLM | VLM directly produces schema | ~2-4s | General images, static QA |
| Head 3 | Screenshot → YOLO+OCR+Florence-2 | Local model ensemble | ~0.6s GPU | UI screenshots, precise bbox |
| Tool loop | Screenshot → VLM calls tools | Multi-hop tool calling | ~5-15s | Complex reasoning with evidence chains |

---

## 5. Domain adapters

### 5a. BrowserGym adapter

Source observation fields from `BrowserEnv._get_obs()`:

| BrowserGym field | Type | Maps to schema |
|---|---|---|
| `obs["axtree_object"]` | merged AXTree dict | `<entities>`, `<attributes>`, `<relations>` |
| `obs["dom_object"]` | DOM snapshot dict | backup for AXTree gaps |
| `obs["extra_element_properties"]` | dict[bid → {visibility, bbox, clickable, set_of_marks}] | `e*.pos`, `e*.state` |
| `obs["screenshot"]` | np.array (H,W,3) | VLM visual input |
| `obs["focused_element_bid"]` | str | `<state_flags> focused_element=` |
| `obs["url"]` | str | `task=` context |
| `obs["goal"]` / `obs["goal_object"]` | str / list[dict] | `goal=` |
| `obs["last_action_error"]` | str | `<state_flags> error=` |

### 5b. Gym-V adapter

Source observation fields from `gym_v.core.Observation`:

| Gym-V field | Type | Maps to schema |
|---|---|---|
| `obs.image` | PIL Image | VLM visual input |
| `obs.text` | str | parse into `<entities>`, `<attributes>`, `<state_flags>` |
| `obs.metadata` | dict | additional context for label enrichment |
| `env.description` | str (property) | `goal=`, game rules → training system prompt |
| `info["history"]` | list[{obs, action}] | step count, prior context |
| `info["invalid_action"]` | bool | `<state_flags> error=` |

### 5c. OSWorld adapter

Uses Head 3 (OmniParser-v2) directly on desktop screenshots. Domain = `"desktop"`.

- **Implementation:** `vlm_wrapper/grounding_browsergym.py → grounding_osworld_obs_to_schema()`

---

## 6. VLM training pipeline

### Stage A — Pure visual grounding

Supervised distillation: VLM sees only the visual input, produces the normalized text state.

- **Training signal:** field-level match against normalized text state labels.
- **Metrics:** field accuracy, relation accuracy, target-slot accuracy, format compliance.
- **No action prediction.** The VLM is a parser.

Start with Qwen-VL or similar and fine-tune on the paired data.

### Stage B — Learn when to call APIs vs. ground visually

Some text state fields are not visually observable. The VLM needs to learn what it can see (produce from pixels) vs. what it can't see (emit an API call).

1. First train pure visual grounding (Stage A) — no API calls.
2. Then add tool-use training — teach the model to request non-visual info via env API.

### Training label generation pipeline

```
For each episode step in {Gym-V, BrowserGym}:
  1. Collect native observation
  2. Run domain adapter → raw structured summary
  3. Apply truncation/pruning to fit token budget
  4. Validate format: regex check all tags present, entity refs consistent
  5. Store as training pair: input = image + system_prompt; output = schema text
```

### Recommended training sequence

1. **Gym-V games first** (2048, Sokoban, Minesweeper) — grid layouts, limited entities, clean GPT-4o labels. Expect >90% field accuracy within ~3-5K examples per game.
2. **Validate with heuristic head** — for every GPT-4o label, run the heuristic head on the same observation. Flag disagreements to catch GPT-4o hallucinations before they enter training data.
3. **Add tool-use training** — after basic SFT, add a second stage where the model learns to emit tool calls for position queries and relation checks.
4. **Browser: MiniWoB++ → WebArena** — simple pages first, validate schema and entity prioritization, then scale to complex real-web pages. Use API escalation for the hard tail during data collection.
5. **Benchmark evaluation** — VisualToolBench, TIR-Bench (image), SIV-Bench, Video-Holmes (video).  Selected for transferable visual reasoning + tool use, not passive VQA alone.
6. **Expected data budget:** ~3-5K labeled examples per domain. At ~$0.01/example with GPT-4o, that's $30-50 per domain.

### Can Qwen3-VL-8B learn this task?

**Yes.** The task is structured template generation (not open-ended reasoning). Gym-V benchmarks show Qwen3-VL-8B scores 16.5 average zero-shot on harder tasks (logic, algorithms, multi-turn strategy). Structured scene parsing is strictly easier, and we fine-tune rather than zero-shot. Distillation from GPT-4o to 8B is well-established for structured output.

---

## 7. Rollout order

**Phase 1 — Gym-V (controlled lab)**
1. Collect paired data from richest wrapper environments.
2. Normalize wrapper text into schema (this designs the schema).
3. Train VLM on (frame → normalized summary).
4. Plug into existing skill pipeline. Measure: does structured format improve retrieval?

**Phase 2 — BrowserGym (noisy deployment target)**
1. Collect paired data across benchmarks.
2. Normalize AXTree/DOM into the same schema.
3. Fine-tune or adapt the Gym-V-trained VLM.
4. Test whether game-learned visual grounding transfers to web UIs.

**Phase 3 — API calling (non-visual grounding)**
1. Add tool-use training on both domains.
2. Measure: does API fallback recover the information lost by visual-only grounding?

---

## 8. HallusionBench — canonical schema for visual reasoning QA

The canonical schema applies to HallusionBench without structural changes. HallusionBench is static image-QA (~1,100 samples, yes/no answers). The schema is a scene representation, not an environment interface.

**Hop trace** maps directly to hallucination reasoning:

```text
trigger=question asks whether the image contains triangles
hop1=locate relevant entities [e1, e3]
hop2=ground entity attributes from pixels [e1.shape → ambiguous]
hop3=check text overlay against visual evidence
hop4=resolve conflict: visual grounding vs language prior
output=answer=no
grounding=[e1, e2, e3]
```

Each hop forces explicit grounding with source attribution. Hallucination failure modes become auditable.

---

## 9. Tool registries for multi-hop reasoning

### Two detection backends

| Backend | Model | Detects | Best for | Tool |
|---------|-------|---------|----------|------|
| OmniParser-v2 | YOLO + Florence-2 + OCR | UI elements: buttons, icons, text fields, menus | GUI screenshots (games, browser, desktop) | `detect_objects` |
| GroundingDINO | `IDEA-Research/grounding-dino-base` | Arbitrary objects from text query | Natural images (VTB, TIR-Bench, photos, video frames) | `grounded_detect` |

**`detect_objects`** runs OmniParser by default (GUI domains) or GroundingDINO (natural-image domains, controlled by `prefer_gdino` flag — set automatically by `ground()` based on domain).

**`grounded_detect`** is always available as a query-driven tool: the VLM describes what to find in plain English ("red sphere", "person sitting on chair"), and GroundingDINO returns bounding boxes. This is the key capability for multi-hop reasoning on non-GUI images.

### Visual tools (single frame)

`build_visual_registry(image, prefer_gdino=False)` → 10 tools:

| Tool | Purpose |
|------|---------|
| `detect_objects` | OmniParser-v2 (GUI) or GroundingDINO (natural) element detection |
| `grounded_detect` | **Open-vocabulary query-driven detection** (GroundingDINO) — finds arbitrary objects by description |
| `describe_region` | Florence-2 caption for a crop |
| `visual_search` | Text-query search over detected elements |
| `count_objects` | Count elements by type or description |
| `classify_scene` | Scene type classification |
| `spatial_query` | Spatial relations between two elements |
| `measure_distance` | Pixel distance between points |
| `extract_colors` | Dominant colors in a region |
| `read_text_region` | OCR on a region |

### Video tools (temporal navigation)

`build_video_registry(frames, fps)` → 8 tools:

| Tool | Purpose |
|------|---------|
| `get_frame` | Retrieve frame by index/timestamp |
| `sample_frames` | Uniformly sample N frames |
| `compare_frames` | Pixel diff between two frames |
| `detect_scene_changes` | Find scene boundaries |
| `get_video_info` | Video metadata |
| `read_text_in_frame` | OCR on a specific frame |
| `temporal_navigate` | Move to a different point in time |
| `list_valid_actions` | Available navigation actions |

### Cross-frame tools (video + visual)

`build_video_visual_registry(frames, fps)` → combines all above + 6 cross-frame tools:

| Tool | Purpose |
|------|---------|
| `track_object` | Track element across frames |
| `summarize_clip` | Timeline of visual changes |
| `find_moment` | Find frame where event occurs |
| `detect_activity` | Classify activity in frame range |
| `compare_elements` | Semantic diff between two frames |
| `detect_objects_at_frame` | Detection on a specific frame |

---

## 10. Benchmarks — selected for transferable visual reasoning skills

The goal is to learn visual reasoning skills that transfer across games, web agents, OS agents, and video understanding.  Each benchmark trains a specific subset of skills; together they cover the full skill set.

### Why these 4 benchmarks and not others

**ToolVQA was dropped.**  Its hard part is tool orchestration and external knowledge retrieval (Calculator, GoogleSearch) — not visual reasoning.  A "call GoogleSearch after OCR" skill doesn't transfer to game playing or web navigation.

The 4 remaining benchmarks were chosen because every skill they test maps directly to an interactive-agent skill:

| Visual reasoning skill | Learned from | Transfers to |
|------------------------|-------------|-------------|
| Entity grounding (find things in pixels) | TIR-Bench tasks, VTB crops | Game tiles, browser buttons, desktop icons |
| Spatial reasoning (left-of, above, between, contains) | TIR spatial / math / jigsaw families | Game board layouts, web page structure, OS window arrangement |
| Attribute recognition (color, material, state, value) | VTB + TIR attribute probes | Element state (disabled, checked, focused), tile values, score displays |
| Relation identification (blocks, contains, adjacent) | TIR compositional items | Same relation verbs in `<state>` schema across all domains |
| Temporal entity tracking (track across frames) | SIV-Bench (people across video clips) | Game replays, browser session recordings, OS task execution |
| Multi-hop evidence chaining (clue → clue → answer) | Video-Holmes (suspense film reasoning) | Multi-step task decomposition, skill-bank retrieval chains |

**The `<state>` schema is the transfer mechanism.**  If the VLM learns to produce correct entities/relations/evidence from TIR-Bench / VisualToolBench, that same skill produces correct entities/relations/evidence from browser screenshots — because the schema is identical.

### Image-based

**VisualToolBench** — Tool-enabled perception, transformation, and reasoning (arXiv:2510.12712). HuggingFace `ScaleAI/VisualToolBench`.
- Skills trained: when to crop/zoom, chain detectors with edits, answer under rubric-style constraints.
- Dataset card: https://huggingface.co/datasets/ScaleAI/VisualToolBench

**TIR-Bench** — Thirteen thinking-with-images task families (arXiv:2511.01833). HuggingFace `Agents-X/TIR-Bench`.
- Skills trained: spatial, symbolic, OCR, jigsaw, contrast, and other agentic image manipulations.
- Dataset card: https://huggingface.co/datasets/Agents-X/TIR-Bench

### Video-based

**SIV-Bench** — Social interaction understanding.  2,792 real-world video clips, 8,792 QAs.
- Skills trained: temporal entity tracking, interaction detection, state change recognition across frames.
- Why transferable: tracking entities over time and detecting state changes is exactly what game/browser/OS agents need during multi-step execution.
- Dataset: https://huggingface.co/datasets/Fancylalala/SIV-Bench
- Code: https://github.com/kfq20/SIV-Bench
- Project page: https://kfq20.github.io/sivbench/

**Video-Holmes** — Multi-step clue-chaining. 270 manually annotated suspense short films, 1,837 questions.
- Skills trained: multi-hop evidence chaining, temporal navigation, grounded reasoning across distant frames.
- Why transferable: building an evidence chain from scattered visual clues is the same skill as multi-step task decomposition — the hops in Video-Holmes map 1:1 to inner-MDP hops in the action agent.
- Repo: https://github.com/TencentARC/Video-Holmes
- Dataset: https://huggingface.co/datasets/TencentARC/Video-Holmes
- Project page: https://video-holmes.github.io/Page.github.io/
- Download:
  ```
  git clone https://github.com/TencentARC/Video-Holmes.git
  cd Video-Holmes
  pip install huggingface_hub
  python download.py --hf_token YOUR_HUGGINGFACE_ACCESS_TOKEN
  unzip Benchmark/videos.zip -d Benchmark/
  unzip Benchmark/annotations.zip -d Benchmark/
  ```

### Per-benchmark grounding strategy

All benchmarks use the unified `ground()` entry point.  Image benchmarks auto-select GroundingDINO; video benchmarks get the full video_visual registry.

| Benchmark | Modality | `ground()` domain | Detection backend | Multi-hop tool chain |
|-----------|:--------:|:-----------------:|:-----------------:|----------------------|
| VisualToolBench | Image | `"image_qa"` | GroundingDINO + tools | `zoom_region` / `grounded_detect` / `describe_region` chains (rubric gold) |
| TIR-Bench | Image | `"image_qa"` | GroundingDINO | `grounded_detect` → `spatial_query` → task-specific tools |
| SIV-Bench | Video | `"video_qa"` | GroundingDINO | `grounded_detect` → `track_object` → `find_moment` → `detect_activity` |
| Video-Holmes | Video | `"video_qa"` | GroundingDINO | `detect_scene_changes` → `grounded_detect` → `track_object` → `describe_region` |

---

## 11. Grounding-aware schema extensions for benchmarks

```text
<evidence>
hop1.tool=detect_objects
hop1.result_ref=e1,e3
hop1.frame=null                  # null for image benchmarks
hop1.confidence=high

<answer>
answer={predicted answer}
grounding=[e1,e3]
evidence_chain=[hop1,hop2]
confidence=high
```

For video benchmarks, add temporal grounding: `hop1.frame=42`, `hop1.timestamp=14.0`.

---

## 12. Schema completeness guarantee (grounding → reasoning contract)

Skills live in the reasoning/action layers, not in grounding.  Grounding is perception (SFT/distillation); reasoning is strategy (skills + GRPO).  But the reasoning layer depends on grounding producing a *complete enough* schema.  Four mechanisms guarantee this:

### Layer 1 — Semantic schema validator (static gate)

Extends the current `validate_schema()` (which only checks tag presence) with semantic checks run *before* the reasoning layer sees the schema:

| Check | Threshold | On failure |
|-------|-----------|------------|
| **Slot population** | `<targets>` has `target=` set (not `null` for env tasks) | Re-ground or escalate head |
| **Entity minimum** | ≥3 entities (games), ≥5 entities (browser), ≥1 (image QA) | Re-ground or escalate head |
| **Uncertainty budget** | ≤50% of entities with `uncertainty=high` | Escalate head |
| **Section content** | Every required section has ≥1 content line, not just tag | Escalate head |
| **Relation coverage** | ≥1 relation if ≥3 entities present | Warning (soft) |

### Layer 2 — Cascaded head escalation (automatic quality recovery)

When the semantic validator fails, automatically try the next head:

```
Head 1: Heuristic (fast, free)
    ↓ fails validator?
Head 2: Vision / GPT-4o (~2-4s)
    ↓ fails validator?
Head 3: Tool loop / multi-hop (~5-15s)
    ↓ still fails?
Return best attempt + high uncertainty flags → reasoning layer decides
```

Escalation is domain-aware: games start at Head 1, image QA starts at Head 2, video QA starts at Head 3 (tool loop always).  The escalation rate target is <10% from Head 1 → Head 2.

### Layer 3 — Uncertainty-driven GROUND triggering (runtime feedback loop)

The inner MDP's `GROUND` action IS the feedback loop.  The reasoning agent doesn't passively receive a schema — it actively extends it when needed:

```
Schema arrives with <uncertainty> e5.label=high
    ↓
Inner MDP: skill "blocker_prerequisite_replan" triggers
    ↓
hop1: GROUND(e5)  →  calls detect_objects/describe_region  →  resolves uncertainty
hop2: CHECK(constraint)  →  now has enough info
hop3: EXECUTE(action)
```

The `<uncertainty>` section is the communication channel between grounding and reasoning.  Grounding flags what it's unsure about; reasoning decides whether to investigate or act despite uncertainty.  **This is why grounding doesn't need skills** — the reasoning skills already include GROUND as their first hop when information is missing.

The `hop_select` LoRA adapter (see [Action Agent §5](../02-action-agent/PLAN-ACTION-AGENT.md#5-lightweight-inner-mdp-typed-local-control)) learns this trade-off end-to-end via GRPO: when is more grounding worth the cost versus acting on partial info?

### Layer 4 — Skill-level slot coverage (before skill execution)

When `SkillQueryEngine.select()` computes applicability, it checks whether the skill's required slots are populated in the current state.  A skill requiring `$blocker` won't fire if `blocker=null`.  This prevents skills from executing on incomplete information and implicitly triggers re-grounding when the agent picks skills that require more detailed perception.

### Design decision: skills for reasoning only, not grounding

| Layer | Learn skills? | Training paradigm | Rationale |
|-------|:---:|---|---|
| **Visual grounding** (perception) | No | SFT / distillation | Perception has correct answers, not strategic choices |
| **Inner MDP reasoning** | **Yes** | Skills + GRPO | Strategic hop selection under uncertainty |
| **Outer MDP action** | **Yes** | Skills + GRPO | Strategic action selection guided by skills |
| **Grounding → reasoning evidence** | Yes (indirect) | Extraction pipeline | Mine hop patterns from grounding traces into reasoning skill templates |

Grounding tool-loop traces (e.g. `detect_objects → spatial_query → count_objects` on image QA) are mined as evidence for reasoning skill templates via the transferable skill extraction pipeline (see [Skill Bank §9](../03-skill-bank/PLAN-SKILL-BANK.md#9-transferable-skill-extraction)), but the templates are consumed by the reasoning layer, not the grounding layer.

**Optional extension — grounding strategies as skills:** Multi-step grounding patterns (disambiguation, target recovery, evidence collection) that recur across domains can optionally be captured as transferable grounding skills. These sit between perception tools and reasoning skills and use belief/binding-effect contracts rather than world-effect contracts. See [Visual Skills](PLAN-VISUAL-SKILLS.md) for the full design. This extension does not change the core principle above — atomic perception tools remain tools, not skills.

---

## 13. Validation checkpoints

- **After Phase 1:** Does the VLM produce structured summaries that the skill pipeline can use? Does structured format beat raw wrapper text for retrieval?
- **After Phase 2:** Does the same schema work for web pages? Does the VLM generalize from games to web?
- **After Phase 3:** Does API calling recover meaningful information?
- **After schema guarantee:** Does the semantic validator + cascaded escalation achieve ≥95% schema completeness on the first pass?  Does the uncertainty channel correctly signal when GROUND hops are needed?

---

## 14. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Gym-V wrapper quality is uneven across 179 envs | Start with richest environments; expand selectively |
| Schema designed from Gym-V doesn't fit BrowserGym | Keep shared slots minimal; allow domain-specific extensions |
| VLM structured output drifts or breaks format | Output validation layer; constrained decoding (vLLM) → ~98% compliance |
| API-calling adds latency and complexity | Measure visual-only ceiling first |
| Structured summaries don't actually help downstream | Test early (Phase 1 checkpoint) |
| Entity position accuracy from pixels | Tool-use delegation: VLM identifies entities, tools return exact coordinates from env APIs |
| Entity coverage on cluttered web pages | Cascaded approach: 8B model first, escalate to API on low coverage (<10% escalation rate expected) |
| Relations require game semantics, not just vision | Game rules in prompt context + tool delegation for non-visual logic |
| Video temporal reasoning is expensive | Multi-hop tool loop with frame sampling; detect_scene_changes narrows search before per-frame analysis |

---

## 15. Unified grounding pipeline (`vlm_wrapper/ground.py`)

All domains share a single code path: `ground(GroundingRequest) → GroundingResult`.

```python
from vlm_wrapper import ground, GroundingRequest

# Image QA (VisualToolBench, TIR-Bench)
result = ground(GroundingRequest(
    images=pil_image,
    goal="How many red spheres are left of the blue cube?",
    domain="image_qa",
))
result.schema   # <state>...</state> with <evidence> + <answer>
result.answer   # "3"
result.evidence # [HopTrace(tool='detect_objects', ...), HopTrace(tool='spatial_query', ...)]

# Game (Gym-V)
result = ground(GroundingRequest(
    images=frame,
    goal="Reach 2048",
    domain="gymv",
    context={"obs_text": obs.text, "description": env.description},
))
result.schema   # <state>...</state> with <targets> + <actions>

# Browser (BrowserGym)
result = ground(GroundingRequest(
    images=screenshot,
    goal="Find cheapest laptop",
    domain="browser",
    context={"obs": browsergym_obs},
))

# Desktop (OSWorld)
result = ground(GroundingRequest(
    images=screenshot,
    goal="Open Spotify settings",
    domain="desktop",
    context={"a11y_tree_xml": a11y_xml},
))

# Video QA (SIV-Bench, Video-Holmes)
result = ground(GroundingRequest(
    images=frame_list,
    goal="When does the character first appear?",
    domain="video_qa",
    context={"fps": 24.0},
))
result.answer   # "at 14.0 seconds"
result.evidence # [HopTrace(tool='find_moment', frame=42, timestamp=14.0, ...)]
```

**All tasks are interactive multi-hop reasoning.** The tool-calling loop IS the interaction — an image-QA item where the VLM calls `detect_objects` → `spatial_query` → `count_objects` → emits answer is structurally identical to a game agent calling `list_entities` → `check_relation` → emitting action. Every domain gets the same core schema (entities, attributes, relations, state_flags, targets, uncertainty, evidence). The only variation is the terminal section: `<actions>` for env tasks, `<answer>` for QA, or both.

**How it works:**

1. `ground()` resolves domain → output_mode (actions / answer / both)
2. Auto-composes the right `ToolRegistry`: vision tools + domain tools
3. Builds an adaptive system prompt: shared core + terminal section
4. Runs `run_tool_loop()` — same interactive loop for ALL tasks
5. Parses the result into a universal `GroundingResult`

| Domain | Output mode | Auto-composed registry | Schema (core + terminal) |
|--------|:-----------:|----------------------|--------------------------|
| gymv | actions | visual + gymv | core + `<actions>` |
| browser | actions | visual + browser | core + `<actions>` |
| desktop | actions | visual + osworld | core + `<actions>` |
| image_qa | answer | visual only | core + `<answer>` |
| video_qa | answer | video_visual (all tools) | core + `<answer>` |

Core = entities, attributes, relations, state_flags, targets, uncertainty, **evidence** (always present — the tool-calling hops are the evidence chain for every task type).

**The per-domain wrappers still exist** for backward compatibility, but new code should use `ground()`.

---

## 16. Implementation status

| Component | Module | Status |
|-----------|--------|--------|
| **Unified pipeline** | `vlm_wrapper/ground.py` | **Done** |
| Adaptive schema (evidence + answer) | `vlm_wrapper/schema.py` | **Done** |
| **GroundingDINO backend** | `visual_reasoning_wrapper/tools_visual.py` | **Done** (`grounded_detect` + dual-backend `detect_objects`) |
| Visual tool registry | `visual_reasoning_wrapper/tools_visual.py` | Done (10 tools) |
| Video tool registry | `visual_reasoning_wrapper/tools_video.py` | Done (8 tools) |
| Cross-frame registry | `visual_reasoning_wrapper/tools_video_visual.py` | Done (6 tools) |
| OmniParser grounding | `vlm_wrapper/grounding.py` | Done |
| BrowserGym grounding adapter | `vlm_wrapper/grounding_browsergym.py` | Done |
| Tool-calling loop | `vlm_wrapper/tool_loop.py` | Done |
| Heuristic adapters | `vlm_wrapper/gymv_heuristic.py`, `browser_heuristic.py` | Done |
| Vision adapters (Head 2) | `vlm_wrapper/gymv_adapter.py`, `browser_adapter.py` | Done |
| Schema utilities | `vlm_wrapper/schema.py` | Done |
| Demo script | `vlm_wrapper/demo_visual_grounding.py` | Done |
| Semantic schema validator (§12 Layer 1) | `vlm_wrapper/schema.py` (`semantic_validate`, `ValidationResult`) | **Done** |
| Cascaded head escalation (§12 Layer 2) | `vlm_wrapper/ground.py` (`cascaded_ground`) | **Done** |
| Benchmark loaders | — | **TODO** |
| Evaluation harness | — | **TODO** |
| Re-observation (Option B) for GROUND hops | `vlm_wrapper/tool_loop.py` (`allow_reobservation`) + `tools_visual.zoom_region` | **Done** |
| Qwen3-VL-8B training pipeline | — | **TODO** |

---

## 17. Reference

- vlm_wrapper repo: https://github.com/wuxiyang1996/Multi-hop-Reasoning-VLM-Agent/tree/main/vlm_wrapper
- System plan index: [`plans/README.md`](../README.md)
