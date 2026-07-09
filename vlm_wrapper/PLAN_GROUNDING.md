# Visual Grounding Integration Plan for vlm_wrapper

## Context

The vlm_wrapper currently has two heads:
- **Head 1 (Heuristic)**: AXTree/obs.text -> schema (free, instant, needs text state)
- **Head 2 (Vision)**: Screenshot -> GPT-4o API -> schema (costs money, 1-3s latency)

We add **Head 3 (Grounding Model)**: Screenshot -> local grounding model -> schema. Free, local, vision-only. No AXTree or obs.text required.

---

## Model Landscape

### Tier 1: Most practical to try now

| Model | Size | Domain fit | What it does | Integration effort |
|---|---|---|---|---|
| **Florence-2** (Microsoft) | 0.2B / 0.8B | Both games + browser | Multi-task: open-vocab detection, phrase grounding, OCR with regions, captioning. All in one model. | Low -- `pip install transformers`, runs on CPU or small GPU |
| **Qwen2.5-VL-7B** | 7B | Both | Built-in bounding box grounding with absolute pixel coords. Already in our model family. | Medium -- needs ~16GB VRAM, but we're already targeting Qwen3-VL-8B |
| **OmniParser V2** (Microsoft) | YOLO + Florence | Browser-focused | Purpose-built screen parser: detects UI elements, extracts icon semantics, outputs structured elements with bboxes. 0.6s/frame on A100. | Medium -- clone repo + download weights |

### Tier 2: Best-in-class but heavier

| Model | Size | Domain fit | What it does |
|---|---|---|---|
| **MolmoPoint-GUI-8B** (Allen AI, March 2026) | 8B | Browser/GUI | Latest SOTA for GUI pointing (61.1% ScreenSpotPro). Uses novel 3-token coarse-to-fine grounding. |
| **UGround-V1** (OSU NLP, ICLR'25) | 2B / 7B / 72B | Browser/GUI | Trained on 10M GUI elements from 1.3M screenshots. SOTA on ScreenSpot-Pro. |
| **UI-TARS** (ByteDance) | various | Browser/GUI | Full GUI agent with built-in grounding. Beats Claude/GPT-4o on OSWorld. |

### Tier 3: General-purpose object detection

| Model | Size | Domain fit | What it does |
|---|---|---|---|
| **Grounding DINO 1.6 Pro** | ~0.9B | Games mostly | Open-vocab object detection from text prompts ("player", "box", "wall"). 55.4 AP on COCO. |
| **SAM 2** (Meta) | various | Both (segmentation) | Segment Anything -- gives masks, not labels. Good companion to a detector. |

### Already available (no model needed)

**Set-of-Mark (SoM)** -- BrowserGym already supports this via `extra_element_properties[bid]["set_of_marks"]`. It overlays numbered marks on screenshots so any VLM can reference elements by number. This is a prompting technique, not a model, and it's already wired into the heuristic head.

---

## Recommended starting point: Florence-2 (0.8B)

Why Florence-2 first:

1. **Covers both domains** -- games and browser, unlike OmniParser (browser-only) or Grounding DINO (better for games)
2. **Tiny model** (0.8B) -- runs on CPU, fast inference, no GPU drama
3. **Multi-task in one forward pass** -- open-vocab detection gives entity bboxes, OCR gives text content, captioning gives scene description. All three feed directly into schema fields.
4. **Lowest integration effort** -- just `pip install transformers`, load model, call with task prompts
5. **Natural stepping stone** -- once Florence-2 works as Head 3, we can swap in OmniParser for browser or Grounding DINO for games as upgrades

**Future upgrades (not in this scope):**
- OmniParser V2 -- browser-specific, uses YOLO + Florence internally
- Grounding DINO -- better for game object detection from text prompts
- MolmoPoint-GUI-8B -- SOTA GUI pointing (March 2026)
- UGround -- SOTA GUI element grounding (ICLR'25)
- Qwen2.5-VL-7B -- built-in grounding, stays in Qwen family

---

## Architecture

### How Head 3 fits with existing heads

```
                  +-----------------+       +-----------------+
                  | Game Frame      |       | Web Screenshot  |
                  | (Gym-V)         |       | (BrowserGym)    |
                  +--------+--------+       +--------+--------+
                           |                         |
          +----------------+-------------------------+----------------+
          |                |                         |                |
          v                v                         v                v
   +-----------+   +-----------+              +-----------+   +-----------+
   | Head 1    |   | Head 3    |              | Head 1    |   | Head 3    |
   | Heuristic |   | Florence-2|              | Heuristic |   | Florence-2|
   | obs.text  |   | pixels    |              | AXTree    |   | pixels    |
   +-----------+   +-----------+              +-----------+   +-----------+
          |                |                         |                |
          v                v                         v                v
          +----------------+-------------------------+----------------+
                                     |
                                     v
                         +------------------------+
                         | Canonical Schema       |
                         | <state>...</state>     |
                         +------------------------+
```

Head 2 (GPT-4o API) also feeds into the same schema but is omitted for clarity -- it's the expensive cloud option.

### File layout

```
vlm_wrapper/
├── schema.py                  # (existing) shared schema spec
├── gymv_heuristic.py          # (existing) Head 1 -- Gym-V
├── browser_heuristic.py       # (existing) Head 1 -- BrowserGym
├── gymv_adapter.py            # (existing) Head 2 -- Gym-V
├── browser_adapter.py         # (existing) Head 2 -- BrowserGym
├── grounding.py               # NEW -- Head 3: Florence-2 grounding model core
├── grounding_browsergym.py    # NEW -- Head 3 adapter for BrowserGym obs
├── grounding_gymv.py          # NEW -- Head 3 adapter for Gym-V obs
├── example_grounding.py       # NEW -- demo comparing all 3 heads
└── __init__.py                # UPDATE -- export Head 3 functions
```

---

## New files

### 1. `grounding.py` -- Core Florence-2 wrapper

- Lazy-load `microsoft/Florence-2-large` (or `-base`) on first call
- Expose task-specific functions that map to schema fields:
  - `detect_entities(image, text_prompt) -> list[Entity]` -- uses `<OPEN_VOCABULARY_DETECTION>` or `<CAPTION_TO_PHRASE_GROUNDING>`
  - `extract_text_regions(image) -> list[TextRegion]` -- uses `<OCR_WITH_REGION>`
  - `caption_scene(image) -> str` -- uses `<MORE_DETAILED_CAPTION>`
- Each returns structured dicts with `label`, `bbox`, `confidence`
- Device auto-detection (CUDA if available, else CPU)
- Florence-2 outputs `<loc_X>` tokens for bounding boxes (normalized 0-999 coordinates) -- we convert to absolute pixel coords matching the schema's `pos=x,y,w,h` format

### 2. `grounding_browsergym.py` -- BrowserGym adapter

- `grounding_obs_to_schema(image, goal, step, task_id) -> str`
- Calls `detect_entities()` with prompts derived from goal + generic UI terms ("button", "link", "input", "text", "image", "menu")
- Calls `extract_text_regions()` to get text content for value fields
- Maps Florence-2 outputs into the canonical `<state>...</state>` schema
- Same signature pattern as `browser_heuristic.obs_to_schema()` for easy comparison

### 3. `grounding_gymv.py` -- Gym-V adapter

- `grounding_frame_to_schema(image, goal, task_id, step, game_type_hint) -> str`
- Uses `detect_entities()` with game-relevant prompts ("tile", "player", "box", "wall", "target", etc.)
- For grid games: post-process detections into grid coordinates
- Maps outputs into the same `<state>...</state>` schema

### 4. `example_grounding.py` -- Comparison demo

- Reuses the synthetic shopping screenshot from `example_browsergym.py`
- Runs all 3 heads on the same image:
  - Head 1 (heuristic) -- from AXTree
  - Head 3 (grounding) -- from pixels via Florence-2
  - Optionally Head 2 (vision) -- from pixels via GPT-4o
- Prints schemas side-by-side and compares entity counts, labels, positions
- No BrowserGym install required

### 5. `__init__.py` update

- Add exports: `grounding_browser_schema`, `grounding_gymv_schema`
- Guard behind `try/except ImportError` (transformers may not be installed)

---

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Coordinate format | Florence-2 normalized (0-999) -> absolute pixel `x,y,w,h` | Matches existing schema `pos=` convention |
| Entity type mapping | Florence-2 labels -> `element` (UI), `object` (game), `text` (OCR) | Consistent with Head 1 and 2 output |
| Entity cap | <=25 (browser), <=20 (game) | Same limits as other heads for 8B model compatibility |
| Model loading | Lazy on first call, cached globally | Avoids startup cost when Head 3 not used |
| Fallback | Head 3 is additive, not a replacement | Works when text state unavailable; Head 1 still best for real-time RL |

---

## Execution order

1. Create `grounding.py` with Florence-2 lazy loader, `detect_entities()`, `extract_text_regions()`, `caption_scene()`
2. Create `grounding_browsergym.py` adapter mapping detections to canonical schema
3. Create `grounding_gymv.py` adapter mapping detections to canonical schema
4. Create `example_grounding.py` comparing Head 1 vs Head 3 on synthetic screenshot
5. Update `__init__.py` to export Head 3 functions
6. `pip install` Florence-2 deps and run example end-to-end
