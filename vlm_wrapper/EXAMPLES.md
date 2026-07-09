# `vlm_wrapper` — Schema & Tool-Calling Examples

This document shows **real, end-to-end outputs** from `vlm_wrapper.ground.cascaded_ground()`: the `<state>` schema produced for each of the five supported domains, together with the multi-hop tool-calling traces where they apply. Every schema here is a verbatim copy of the file written to `out/schemas/<case>.schema.txt` by the official smoke test:

```bash
python scripts/test_vlm_parsers.py                             # full 5-case cascade (VLM-first)
python scripts/test_vlm_parsers.py --cases gymv --gymv-head tool_loop   # multi-hop tool calling
```

---

## 1. How to read a schema

The shared schema has a fixed set of sections. Every head (VLM, OmniParser, tool-loop, opt-in heuristic) emits the **same** grammar so downstream code never has to branch on the producer.

| Section | Purpose |
|---|---|
| `<entities>` | `e_i[type=…, label=…, bid=…, pos=x,y,w,h, ontology=…]` — objects the agent can reason about. `bid` is the browser/desktop element id (null elsewhere). `ontology` is the cross-domain bridge used by skill transfer. |
| `<attributes>` | `e_i.key=val` — per-entity facts (state, value, text content, …). |
| `<affordances>` | Per-entity list of abstract operators (`select`, `inspect`, `open`, `read`, …) — drives skill-applicability matching. |
| `<relations>` | `rel(e_i, e_j, …)` — `adjacent`, `contains`, `grouped`, `blocks`, `merge_candidate`, … |
| `<state_flags>` | `progress`, `phase`, `scene_type` (canonical: `main_menu`, `game_play`, `form_entry`, `modal_dialog`, `results_view`, `video_segment`, `image_qa`, `landing_page`, …), `error`, `dialog_open`, `input_pending`. |
| `<targets>` | `target=` (eid), `blocker`, `constraint`, `candidate_set=[…]`, `history_anchor` (eid carried from previous step). |
| `<uncertainty>` | `e_i.field=low|mid|high` — self-reported confidence per field. |
| `<evidence>` | Multi-hop reasoning trace: `hop_k.abstract_op ∈ {GROUND, CHECK, RETRIEVE, CONCLUDE, VERIFY}`, `hop_k.tool`, `hop_k.result_ref`, `hop_k.frame/timestamp/confidence`. |
| `<actions>` *(env modes)* | `a_i=<action>` — proposed moves, copied verbatim from `valid_actions` for gymv. |
| `<answer>` *(QA modes)* | `answer=`, `grounding=[e_i,…]`, `evidence_chain=[hop_k,…]`, `confidence`. |

The canonical enums are exported from `vlm_wrapper.schema`:

```python
from vlm_wrapper.schema import (
    ONTOLOGY_TYPES, ABSTRACT_OPERATORS, INNER_MDP_OPS,
    SCENE_TYPES, ENTITY_TYPES,
)
```

---

## 2. Default pipeline — VLM-first

`cascaded_ground()` runs the VLM-first chain per domain (no heuristic on the default path). A real 5-case run reports:

```
  [OK] gymv            6.6s | rounds=1 head_used=vlm
  [OK] browser         3.9s | rounds=1 head_used=vlm
  [OK] desktop         6.4s | rounds=1 head_used=vlm
  [OK] tir_bench      15.3s | answer=A  ground_truth=A  rounds=1 head_used=vlm
  [OK] video_holmes   14.1s | answer=C    ground_truth=F   rounds=5 head_used=tool_loop
```

Chain used (from `vlm_wrapper.ground._ESCALATION_CHAINS`):

```python
"gymv":     ["vlm", "tool_loop"]
"browser":  ["vlm", "omniparser", "tool_loop"]
"desktop":  ["omniparser", "vlm", "tool_loop"]
"image_qa": ["vlm", "tool_loop"]
"video_qa": ["tool_loop"]
```

---

## 3. Example schemas — one per domain (default, VLM-first)

### 3.1 Gym-V (2048 frame, `head_used=vlm`)

```
<state>
domain=gymv
task=Game2048-v0
goal=Reach 2048
step=0

<entities>
e1[type=element, label=tile_2, bid=null, pos=0,0,1,1, ontology=tracked_entity]
e2[type=element, label=tile_4, bid=null, pos=0,1,1,1, ontology=tracked_entity]
e3[type=element, label=empty_tile, bid=null, pos=0,2,1,1, ontology=tracked_entity]
…
e16[type=element, label=empty_tile, bid=null, pos=3,3,1,1, ontology=tracked_entity]
</entities>

<state_flags>
progress=0.0
phase=early
scene_type=game_play
…

<targets>
target=e1
candidate_set=[e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16]

<evidence>
hop1.abstract_op=GROUND
hop1.tool=null
hop1.result_ref=e1,e2,…,e16
hop1.confidence=high

<actions>
a1=[Up]
a2=[Down]
a3=[Left]
a4=[Right]
</state>
```

Full file: [`out/schemas/gymv.schema.txt`](../out/schemas/gymv.schema.txt). Actions are copied verbatim from `valid_actions` injected into the request context — no `slide_left`-style hallucinations.

### 3.2 Browser (Wikipedia main page, `head_used=vlm`)

```
<state>
domain=browser
task=wiki.main_page.demo
goal=Identify the first section heading on the Wikipedia main page.
step=0

<entities>
e1[type=text, label="Welcome to Wikipedia", bid=null, pos=50,100,200,30, ontology=textual_anchor]
e2[type=text, label="From today's featured article", bid=null, pos=50,200,300,30, ontology=textual_anchor]
e3[type=text, label="In the news", bid=null, pos=50,300,150,30, ontology=textual_anchor]
e4[type=text, label="The Great Barrier Reef is the world's largest coral reef system…", bid=null, pos=50,230,500,30, ontology=textual_anchor]
…

<relations>
contains(e2,e4)
contains(e3,e5)
contains(e3,e6)

<state_flags>
scene_type=landing_page

<targets>
target=e2
candidate_set=[e2,e3]

<evidence>
hop1.abstract_op=GROUND
hop1.tool=null
hop1.result_ref=e2
hop1.confidence=high

<actions>
a1=read(e2)
a2=inspect(e3)
a3=compare(e2,e3)
</state>
```

Full file: [`out/schemas/browser.schema.txt`](../out/schemas/browser.schema.txt). `bid=null` here because the screenshot was captured outside of BrowserGym — when an AXTree is available it populates the browser element id.

### 3.3 Desktop (OSWorld-style screenshot, `head_used=vlm` after OmniParser CUDA fallback)

```
<state>
domain=desktop
task=osworld.demo.open-files
goal=Open the Files application.  Plan the first pyautogui action you would take from this desktop state.

<entities>
e1[type=element, label=Home, bid=null, pos=0,0,100,100, ontology=selectable_entity]
e2[type=element, label=Trash, bid=null, pos=0,100,100,100, ontology=selectable_entity]
e3[type=element, label=Files, bid=null, pos=0,200,100,100, ontology=selectable_entity]
e4[type=element, label=Firefox, bid=null, pos=0,300,100,100, ontology=selectable_entity]
e5[type=element, label=Terminal, bid=null, pos=0,400,100,100, ontology=selectable_entity]
e6[type=element, label=Terminal window, bid=null, pos=100,0,600,400, ontology=container_entity]
…

<state_flags>
scene_type=main_menu

<targets>
target=e3
candidate_set=[e3]

<evidence>
hop1.abstract_op=GROUND
hop1.tool=list_entities
hop1.result_ref=e1,e2,e3,e4,e5,e6,e7,e8,e9
hop2.abstract_op=CHECK
hop2.tool=query_entity_pos
hop2.result_ref=e3

<actions>
a1=click(50,250)
</state>
```

Escalation trace: `omniparser=FAIL (CUDA kernel unavailable) → vlm=OK`. The cascade handled the hardware mismatch transparently. Full file: [`out/schemas/desktop.schema.txt`](../out/schemas/desktop.schema.txt).

### 3.4 TIR-Bench (image QA, `head_used=vlm`)

Question: *Are there any other things that are the same shape as the big metallic object?*

```
<state>
domain=image_qa
task=tir_bench.refcoco.1

<entities>
e1[type=object, label=big metallic cube, bid=null, pos=null, ontology=tracked_entity]
e2[type=object, label=small metallic cube, bid=null, pos=null, ontology=tracked_entity]
e3[type=object, label=purple sphere, bid=null, pos=null, ontology=tracked_entity]
e4[type=object, label=large brown cylinder, bid=null, pos=null, ontology=tracked_entity]
e5[type=object, label=small green cylinder, bid=null, pos=null, ontology=tracked_entity]
e6[type=object, label=large gray cube, bid=null, pos=null, ontology=tracked_entity]
</entities>

<relations>
grouped(e1,e2,e6)

<state_flags>
scene_type=image_qa

<targets>
target=e1
candidate_set=[e1,e2,e3,e4,e5,e6]

<evidence>
hop1.abstract_op=GROUND
hop1.tool=list_entities
hop1.result_ref=[e1,e2,e3,e4,e5,e6]
hop1.confidence=high
hop2.abstract_op=CHECK
hop2.tool=check_relation
hop2.result_ref=[e1,e2,e6]
hop2.confidence=high

<answer>
answer=yes
grounding=[e1,e2,e6]
evidence_chain=[hop1,hop2]
confidence=high
</state>
```

Full file: [`out/schemas/tir_bench.schema.txt`](../out/schemas/tir_bench.schema.txt). The VLM chains visual tools over the benchmark image; the excerpt illustrates the same `<state>` / `<evidence>` / `<answer>` contract used across domains.

### 3.5 Video-Holmes (video QA, `head_used=tool_loop`)

Question: *What is the relationship between the man with a black beard and Benjamin?* (6 frames from a 4-minute clip).

Schema (abbrev.):

```
<state>
domain=video_qa
task=video_holmes.test.fH6bbNJJfqk.Q1

<entities>
e1[type=text, label=Hii bro, did you reach, bid=null, pos=37,73,142,20, ontology=textual_anchor]
e2[type=text, label=Yes bro, the flight was, bid=null, pos=71,117,72,14, ontology=textual_anchor]
e3[type=text, label=On time, bid=null, pos=71,133,48,14, ontology=textual_anchor]

<state_flags>
scene_type=video_segment

<evidence>
hop1.abstract_op=GROUND
hop1.tool=detect_scene_changes
hop2.abstract_op=GROUND
hop2.tool=sample_frames
hop3.abstract_op=RETRIEVE
hop3.tool=read_text_in_frame
hop3.result_ref=e1,e2,e3
hop3.frame=1
hop3.timestamp=0.04
hop3.confidence=high

<answer>
answer=C
grounding=[e1,e2,e3]
evidence_chain=[hop1,hop2,hop3]
confidence=high
</state>
```

Full file: [`out/schemas/video_holmes.schema.txt`](../out/schemas/video_holmes.schema.txt). Video-QA's default chain is tool-loop-only — see §4.2 for the full trace.

---

## 4. Tool-calling traces

Two kinds of multi-hop tool-calling are exercised in the pipeline:

- **Domain tools** (gymv / browser / desktop): structured queries over the environment's ground-truth state — `list_entities`, `query_entity_pos`, `check_relation`, `get_state_flags`, `list_valid_actions`, `get_grid_state`, `count_merge_candidates`, `check_deadlock`, `spatial_analysis`.
- **Visual tools** (any image / video): calls out to OmniParser-v2, GroundingDINO, Florence-2, EasyOCR, video decoders — `detect_objects_at_frame`, `grounded_detect`, `describe_region`, `zoom_region`, `read_text_in_frame`, `sample_frames`, `detect_scene_changes`, `visual_search`, `count_objects`, `classify_scene`, `spatial_query`, `measure_distance`, `extract_colors`.

### 4.1 Gym-V — VLM + tool-calling (`--gymv-head tool_loop`)

Invocation (hides `obs_text` from the VLM prompt but keeps it wired to the tool handlers, so GPT-4o has to actually call tools to ground the schema):

```bash
python scripts/test_vlm_parsers.py --cases gymv --gymv-head tool_loop
```

Scorecard:

```
head_used: tool_loop  chain: tool_loop=OK
tool_trace: 2 call(s)  [get_grid_state×1, list_valid_actions×1]
validation: valid=True  entities=16  high_uncert=0.0  escalation=False  missing_slots=[]
```

Actual tool calls GPT-4o issued (from `out/schemas/gymv.raw.json`):

```json
[
  {
    "call": {"name": "get_grid_state", "arguments": {}},
    "result": {
      "available": true, "rows": 4, "cols": 4,
      "grid": [["2","4","0","0"],["0","0","0","0"],
               ["0","0","0","0"],["0","0","0","0"]]
    },
    "reobserved": false
  },
  {
    "call": {"name": "list_valid_actions", "arguments": {}},
    "result": {
      "actions": ["[Up]", "[Down]", "[Left]", "[Right]"],
      "count": 4
    },
    "reobserved": false
  }
]
```

Resulting `<evidence>` block (the tool trace is faithfully cited, with `functions.` prefixes normalised during reconciliation):

```
hop1.abstract_op=GROUND
hop1.tool=functions.get_grid_state
hop1.result_ref=e1,e2,e3,…,e16
hop1.confidence=high
hop2.abstract_op=GROUND
hop2.tool=functions.list_valid_actions
hop2.result_ref=null
hop2.confidence=high
```

Full files: [`out/schemas/gymv.schema.txt`](../out/schemas/gymv.schema.txt), [`out/schemas/gymv.raw.json`](../out/schemas/gymv.raw.json).

Registered gymv tools (see `vlm_wrapper/tools_gymv.py::build_gymv_registry`):

| Tool | Purpose |
|---|---|
| `list_entities(filter_type?, max_results?)` | Enumerate entities with their positions / attributes. |
| `query_entity_pos(entity_label)` | Look up the `(row,col)` of one or more entities by label. |
| `check_relation(entity_a, entity_b, relation)` | `adjacent`, `same_row`, `same_column`, `blocks`, `merge_candidate`, … |
| `get_state_flags()` | Parse `progress` / `phase` / `error` / dialog flags. |
| `list_valid_actions()` | Return the env's action vocabulary. |
| `get_grid_state()` | Full 2D grid for grid-based games (2048, Sokoban, Minesweeper). |
| `check_deadlock()` | Corner-deadlock detection for Sokoban-family games. |
| `spatial_analysis(entity_label?)` | Manhattan distances, nearest neighbours, row/col alignment. |
| `count_merge_candidates()` | Adjacent mergeable pairs (2048, Threes). |

### 4.2 Video-Holmes — multi-hop video reasoning (`head_used=tool_loop`, 5 rounds)

Full tool trace for the schema shown in §3.5:

| Round | Tool | Arguments | Result (summary) |
|---|---|---|---|
| 1 | `detect_scene_changes` | `start_idx=0, end_idx=5, threshold=0.15` | 3 shot boundaries at frames 1, 3, 4 |
| 2 | `sample_frames` | `n=3, start_sec=0.04, end_sec=0.17` | 3 frames around the first shot |
| 3 | `detect_objects_at_frame` | `frame_index=0, conf=0.05, max=30` | **error** — CUDA kernel unavailable (RTX 5090 vs. sm_90 torch) |
| 3 | `detect_objects_at_frame` | `frame_index=1, …` | error (same) |
| 3 | `detect_objects_at_frame` | `frame_index=2, …` | error (same) |
| 4 | `read_text_in_frame` | `frame_index=1` | 7 OCR snippets: *Hii bro, did you reach*, *Yes bro, the*, *flight was*, *On time*, … |

The VLM recovered from three GPU-side tool failures in round 3 by falling through to `read_text_in_frame` — a classic multi-hop-with-reobservation pattern. The `<evidence>` block records the three hops that mattered (`detect_scene_changes → sample_frames → read_text_in_frame`) and drops the failed ones; `reconcile_evidence_with_tool_trace()` then verifies every cited hop is backed by a real tool call.

### 4.3 Visual tool catalogue (browser / desktop / video / image)

Built via `visual_reasoning_wrapper.tools_visual.build_visual_registry(image)` or `build_video_registry(frames)`. All are available to any `tool_loop` invocation:

| Tool | Backend | Purpose |
|---|---|---|
| `detect_objects` | OmniParser-v2 (YOLO+Florence-2+OCR) | UI element detection with bboxes and interactability |
| `grounded_detect` | GroundingDINO | Open-vocabulary detection for natural images |
| `describe_region` | Florence-2 | Dense caption for a cropped region |
| `zoom_region` | PIL resampling → re-observation | VLM asks for a crop; next turn receives an upscaled image |
| `read_text_region` | EasyOCR | Text in a specific bbox |
| `visual_search` / `count_objects` / `classify_scene` | GroundingDINO / Florence-2 | Higher-level convenience wrappers |
| `spatial_query` / `measure_distance` / `extract_colors` | Pure-Python geometry & color | Pixel-space spatial reasoning |
| `sample_frames` / `detect_scene_changes` / `detect_objects_at_frame` / `read_text_in_frame` | decord + OmniParser + EasyOCR | Video-only variants |

---

## 5. Opt-in alternative — the heuristic head

The `obs.text` / AXTree heuristic is **implemented** but **NOT on the default cascade** (see `_ESCALATION_CHAINS` in `vlm_wrapper/ground.py`). Keeping it off the main path prevents a regex shortcut from silently masking VLM grounding bugs.

### When to use it

- **Text-only regressions** / unit tests that must avoid GPT-4o API calls.
- **Real-time RL baselines** where `<1 ms` latency matters more than visual fidelity.
- **New environments** where obs-text is already structured and you want a schema in place before tuning the VLM prompt.
- **Cost floors**: offline batch labelling when a teacher-rated heuristic schema is "good enough" for a subset of states.

### Direct call

```python
from vlm_wrapper import gymv_heuristic_schema, browser_heuristic_schema

schema = gymv_heuristic_schema(
    obs_text="| 2 | 4 | 0 | 0 |\n| 0 | 16 | 8 | 0 |\n| 0 | 0 | 0 | 0 |\n| 0 | 0 | 0 | 0 |",
    description="You are playing 2048.  Slide tiles.  Valid moves: [Up], [Down], [Left], [Right].",
    task_id="Game2048-v0", step=5,
)
```

Real output (same environment, heuristic head only — no GPT-4o, no screenshot):

```
<state>
domain=gymv
task=Game2048-v0
step=5

<entities>
e1[type=object, label=tile_2,  bid=null, pos=0,0,1,1, ontology=selectable_entity]
e2[type=object, label=tile_4,  bid=null, pos=0,1,1,1, ontology=selectable_entity]
e3[type=object, label=tile_16, bid=null, pos=1,1,1,1, ontology=selectable_entity]
e4[type=object, label=tile_8,  bid=null, pos=1,2,1,1, ontology=selectable_entity]
e5[type=region, label=empty,   bid=null, pos=null,    ontology=navigable_region]

<attributes>
e1.value=2
e2.value=4
e3.value=16
e4.value=8
e5.cells=12

<affordances>
e1.affords=[select, compare]
e2.affords=[select, compare]
e3.affords=[select, compare]
e4.affords=[select, compare]
e5.affords=[approach]

<relations>
adjacent(e1,e2)
adjacent(e2,e3)
adjacent(e3,e4)

<state_flags>
progress=null
phase=mid
scene_type=game_play

<targets>
target=e1
candidate_set=[e1,e2,e3,e4,e5]

<actions>
a1=[Up]
a2=[Down]
a3=[Left]
a4=[Right]
</state>
```

Notes:

- No `<evidence>` block — the heuristic is single-hop by construction; it cannot cite tools it didn't call.
- `ontology=selectable_entity` differs from the VLM's `tracked_entity` because the heuristic keys off label type, not visual context. Both are valid ontology tags.
- The heuristic fills `<affordances>` from a fixed mapping — a useful side-benefit for downstream skill mining.

### Opt-in cascade (heuristic first, escalate on failure)

```python
from vlm_wrapper.ground import (
    GroundingRequest, cascaded_ground, _HEURISTIC_CHAINS,
)

result = cascaded_ground(
    GroundingRequest(
        images=frame, goal="Reach 2048", domain="gymv",
        context={
            "obs_text": obs_text,
            "description": game_rules,
            "valid_actions": ["[Up]", "[Down]", "[Left]", "[Right]"],
        },
    ),
    chain=_HEURISTIC_CHAINS["gymv"],   # ["heuristic", "vlm", "tool_loop"]
)
print(result.head_used)          # "heuristic" if it passed validation, else "vlm" / "tool_loop"
print(result.escalation_trace)   # full per-head ValidationResult trail
```

Equivalent CLI shortcut for the smoke test:

```bash
python scripts/test_vlm_parsers.py --cases gymv --gymv-head heuristic
python scripts/test_vlm_parsers.py --cases browser --browser-head heuristic
```

### Contract guarantees

- The heuristic emits the **same** `<state>…</state>` grammar as every other head (same required sections, same entity/attribute/relation syntax, same `scene_type` enum).
- `semantic_validate(schema, domain=…)` is run on the heuristic's output identically to the VLM's — `_HEURISTIC_CHAINS` will therefore escalate to VLM (or tool-loop) whenever the heuristic misses a required slot or falls below the entity-count floor.
- `reconcile_evidence_with_tool_trace()` is a no-op for heuristic schemas (empty trace, empty evidence) — no false-positive fabrication warnings.

---

## 6. Reproducing every schema in this doc

```bash
# Full VLM-first cascade (writes one schema per case)
python scripts/test_vlm_parsers.py

# Multi-hop tool calling for gym-v
python scripts/test_vlm_parsers.py --cases gymv --gymv-head tool_loop

# Opt-in heuristic-first (fastest, no API calls)
python scripts/test_vlm_parsers.py --cases gymv --gymv-head heuristic

# Pytest mirror of the tool-calling regression
pytest vlm_wrapper/tests/test_gpt4o_parsers.py::test_live_gymv_tool_loop_schema -m live
```

All outputs land under `out/schemas/` as `<case>.schema.txt` (rendered `<state>` block), `<case>.raw.json` (full adapter dict including `tool_trace` and `escalation_trace`), and `summary.json` (one-line-per-case roll-up).
