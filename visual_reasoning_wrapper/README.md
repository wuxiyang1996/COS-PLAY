# `visual_reasoning_wrapper`

Visual-reasoning slice of the wrapper: image / video tool registries **and** the
public benchmarks the agent is evaluated on. Every other adapter (Gym-V,
BrowserGym, OSWorld) reuses these tools — they are not benchmark-private.

## Layout

```
visual_reasoning_wrapper/
├── __init__.py            # XSkill-aligned design notes + PRIMARY_VISUAL_REASONING_BENCHMARKS
├── tools_visual.py        # Single-frame observation tools (detect_objects, describe_region, …)
├── tools_video.py         # Temporal navigation tools (get_frame, detect_scene_changes, …)
├── tools_video_visual.py  # Cross-frame tools (track_object, find_moment, summarize_clip)
├── tools_reasoning.py     # Symbolic reasoning tools (count_value, compute_ratio,
│                          # compare_values, verify_claim) + the per-registry
│                          # _DerivationLog that renders <derivations>
├── question_router.py     # Question-class router that injects required reasoning
│                          # tools into the goal text per benchmark sample
├── skill_executor.py      # VisualReasoningExecutor: HopExecutor that wires the
│                          # visual + reasoning tool registries into
│                          # harness.adapters.VisualReasoningAdapter
└── benchmarks/            # Loaders + parse_*_sample for the standard 2×2 set
    ├── README.md
    ├── _hf_images.py
    ├── visual_toolbench.py   # HF ScaleAI/VisualToolBench
    ├── tir_bench.py          # HF Agents-X/TIR-Bench
    ├── video_holmes.py       # local data/Video-Holmes
    └── siv_bench.py          # local data/SIV-Bench
```

The package's `__init__.py` only carries documentation and the
`VisualReasoningBenchmark` dataclass / `PRIMARY_VISUAL_REASONING_BENCHMARKS`
tuple. Tool registries and benchmark loaders are imported directly:

```python
from visual_reasoning_wrapper.tools_visual import build_visual_registry
from visual_reasoning_wrapper.tools_video_visual import (
    build_video_visual_registry,
)
from visual_reasoning_wrapper.benchmarks.tir_bench import (
    iter_tir_bench_samples, parse_tir_bench_sample,
)
```

The convenience symbols (`build_visual_registry`, `build_video_registry`,
`build_video_visual_registry`, all `iter_*` / `parse_*` helpers) are also
re-exported at the top level via `vlm_wrapper/__init__.py`.

## Tools

The registries split cleanly into **observation tools** (look at pixels)
and **reasoning tools** (consume already-grounded values and record a typed
derivation step).  Both registry builders expose `include_reasoning=True`
by default so every benchmark gets the symbolic-derivation suite for free.

* **`tools_visual.build_visual_registry(image, *, prefer_gdino=False, include_reasoning=True)`**
  — single frame observation: `detect_objects`, `grounded_detect`,
  `describe_region`, `zoom_region`, `visual_search`, `count_objects`,
  `classify_scene`, `spatial_query`, `measure_distance`, `extract_colors`,
  `read_text_region`. With `include_reasoning=True` the registry also
  carries the four reasoning tools and a fresh `derivation_log` attribute.
* **`tools_video.build_video_registry(frames=…, fps=…, current_index=…)`** —
  temporal navigation: `get_frame`, `compare_frames`,
  `detect_scene_changes`, `sample_frames`, `read_text_in_frame`,
  `find_moment`, `summarize_clip`, …
* **`tools_video_visual.build_video_visual_registry(frames=…, …, include_reasoning=True)`**
  — cross-frame superset (`track_object`, `find_moment`,
  `summarize_clip`, `detect_objects_at_frame`, `describe_frame`, …) merged
  on top of the merged video + visual registries; the reasoning toolset
  is appended once at the top so derivations are not double-registered.
* **`tools_reasoning.build_reasoning_registry(log=None)`** — domain-agnostic
  symbolic tools the VLM uses to *commit* a numeric step instead of
  describing it inline:

  | Tool | When to call | Records |
  |------|--------------|---------|
  | `count_value(value, label, refs)` | "how many", "count of …" | `kind=COUNT` row |
  | `compute_ratio(numerator, denominator, label, refs, unit)` | "what proportion / fraction / percent" | `kind=RATIO` row |
  | `compare_values(a, b, op, label_a, label_b)` | "which is larger / closer / earlier" | `kind=COMPARE` row |
  | `verify_claim(claim, evidence_refs, confidence)` | final reasoning step before `<answer>` | `kind=VERIFY` row |

  Each call appends to a per-registry `_DerivationLog` and returns a
  stable `derivation_id` (`d1`, `d2`, …) the model cites in
  `evidence_chain=` and `<derivations>`.

* **`question_router.classify_question(question, *, modality)`** — assigns
  the prompt to one or more classes (`count`, `ratio`, `compare`,
  `spatial`, `ocr`, `identity`, `temporal`, `social`, `verify`,
  `answer`) and emits a small bullet-list block that the benchmark
  parsers append to the goal so the VLM is told which reasoning tools
  it MUST call before writing `<answer>`.

### How the loop fits together

1. Benchmark parser builds the goal text and calls
   `cascaded_ground` with `chain=["tool_loop"]` for image benches and
   the default `["tool_loop"]` for video benches.
2. `vlm_wrapper.ground._build_registry` constructs the per-domain
   registry with the reasoning tools merged in. The merged registry
   carries a `derivation_log` attribute (preserved by
   `ToolRegistry.merge`).
3. The VLM grounds entities with observation tools, then calls one or
   more reasoning tools to commit each numeric / verbal derivation.
4. After the loop, `ground()` reads `registry.derivation_log` and
   stitches a `<derivations>` block into the schema if the model
   omitted it. The validator then cross-checks `<answer>
   evidence_chain=` against the derivation ids.

The new schema section is documented in
[`vlm_wrapper/schema.py`](../vlm_wrapper/schema.py) — search for
`_SECTION_DERIVATIONS` / `DERIVATION_KINDS` / `_DERIVATION_KIND_RE`.

## Executing transferred skills (`skill_executor.py`)

The wrapper does not just *generate* schemas — it also serves as a
real `HopExecutor` for the harness `VisualReasoningAdapter`, so a
`SkillRecord` mined on `gymv` can be re-run on a `visual_reasoning`
sample without going through the LLM tool-loop a second time.

Inner-MDP action → tool mapping:

| `hop["action"]` | Tool dispatched | Evidence role |
|-----------------|-----------------|---------------|
| `GROUND` (with `query`) | `grounded_detect` | `GATHER` |
| `GROUND` (no `query`) | `detect_objects` | `GATHER` |
| `RETRIEVE` (bbox / `entity_index`) | `describe_region` (or `read_text_region` if `use_ocr=True`) | `GATHER` |
| `CHECK` with `kind=COUNT/RATIO/COMPARE` | `count_value` / `compute_ratio` / `compare_values` | `REASON` |
| `CHECK` with `element_a/element_b` | `spatial_query` | `REASON` |
| `VERIFY` | `verify_claim` | `VERIFY` |
| `COMMIT` | `verify_claim` (final) | `VERIFY` + `COMMIT` |
| `EXECUTE` | no-op (image QA has no env effects) | `COMMIT` |

Wiring it onto an adapter:

```python
from PIL import Image
from harness.adapters.visual_reasoning_adapter import (
    VisualReasoningAdapter,
    bind_visual_reasoning_executor,
)

adapter = VisualReasoningAdapter()
img = Image.open("frame.png")
executor = bind_visual_reasoning_executor(adapter, image=img)

# adapter.run(skill, ctx) now dispatches each hop to a real tool;
# executor.derivation_log carries every typed derivation row.
```

The executor reads ``${slot}`` substitutions performed by
``HopBindings.resolve_dict`` and surfaces unbound slots as a hop
failure (``ok=False``) so the harness aborts rather than silently
sending placeholder strings to a tool.

## Benchmarks

The standard coverage is a 2 image + 2 video set chosen to exercise the
agent's actual action space (`inspect`, `crop`, `zoom`, `compare`, `track`,
`OCR`, `verify`, `answer`) instead of only static visual-question answering.

| Key | Modality | Loader | Source links | Why it is covered |
|-----|----------|--------|--------------|-------------------|
| `visual_toolbench` | image | `benchmarks/visual_toolbench.py` | [HF dataset](https://huggingface.co/datasets/ScaleAI/VisualToolBench), [paper](https://arxiv.org/abs/2510.12712) | Tool-enabled image perception, transformation, and reasoning; closer to "think with images" than CLEVR/GQA-style QA. |
| `tir_bench` | image | `benchmarks/tir_bench.py` | [repo](https://github.com/agents-x-project/TIR-Bench), [HF dataset](https://huggingface.co/datasets/Agents-X/TIR-Bench), [paper](https://arxiv.org/abs/2511.01833) | Agentic thinking-with-images across 13 task families; stresses image operations inside a reasoning chain. |
| `video_holmes` | video | `benchmarks/video_holmes.py` | [repo](https://github.com/TencentARC/Video-Holmes), [project page](https://video-holmes.github.io/Page.github.io/), [HF dataset](https://huggingface.co/datasets/TencentARC/Video-Holmes), [paper](https://arxiv.org/abs/2505.21374) | Short-video multi-hop evidence-chain reasoning over clues scattered across clips. |
| `siv_bench` | video | `benchmarks/siv_bench.py` | [project page](https://kfq20.github.io/sivbench/), [HF dataset](https://huggingface.co/datasets/Fancylalala/SIV-Bench), [paper](https://arxiv.org/abs/2506.05425) | Social scene, state, intent, and interaction reasoning; complements Video-Holmes with social/latent-state tasks. |

The image pair intentionally replaces CLEVR + GQA for this wrapper: CLEVR/GQA are
useful static VQA checks, but they do not directly evaluate whether the agent
selects and uses visual tools during reasoning. For video, SIV-Bench is preferred
over a generic Video-MME short subset because it tests interaction, intent, and
social-state inference rather than broad video QA alone.

Download instructions and HF cache tips: [`benchmarks/README.md`](benchmarks/README.md).

## Why one folder

XSkill (arXiv:2603.12056) frames a multimodal agent as **observation → reasoning
+ tool calls → structured state**.  The tool registries and the benchmarks that
exercise them are two halves of the same loop, so they live together; pulling
either side into the agent is one import (`visual_reasoning_wrapper`).

The rest of `vlm_wrapper/` (`tool_loop.py`, `ground.py`, `schema.py`, the env
adapters and the heuristic / OmniParser heads) plugs into these registries —
e.g. `tool_loop.visual_generate_label_with_tools` calls
`visual_reasoning_wrapper.tools_visual.build_visual_registry` and
`tool_loop.video_visual_generate_label_with_tools` calls
`visual_reasoning_wrapper.tools_video_visual.build_video_visual_registry`.

## Evaluation

Wired up in [`../vlm_wrapper/eval/run_eval.py`](../vlm_wrapper/eval/run_eval.py) — pass
`--benchmark {visual_toolbench,tir_bench,video_holmes,siv_bench}`. See
[`../vlm_wrapper/eval/README.md`](../vlm_wrapper/eval/README.md) for the harness API and metrics.

## Cold-start actor pipeline

A second consumer of these benchmarks lives under
[`../cold_start/generate_cold_start_actor_visual_reasoning.py`](../cold_start/generate_cold_start_actor_visual_reasoning.py).
It runs a **two-stage gpt-5.5 pipeline** that mirrors the env-rollout
cold-start (`generate_cold_start_actor.py` family) but on the four
visual-reasoning benchmarks:

1. **Visual schema (vision call)** — gpt-5.5 consumes the image (or
   ``--num_frames`` uniformly-sampled video frames) and emits a
   ``<state>...</state>`` schema using ``vlm_wrapper.schema``
   (`build_system_prompt(domain="image_qa"|"video_qa")` +
   `build_user_message`).
2. **Actor agent (text call)** — gpt-5.5 reads the schema + the
   question + the valid action space and picks one answer via OpenAI
   function calling (`choose_answer`):
   * MCQ video benchmarks → letter `A..F` (Holmes) / `A..L` (SIV);
   * free-form image benchmarks → concise answer string.

Per-sample records (schema + raw VLM output + actor answer + gold +
correctness) land under
``<repo>/Cold-start-out-visual-reasoning/<run_id>/<benchmark>/``.

### Running it

```bash
# All four benchmarks, 5 test cases each (the canonical sweep)
bash cold_start/run_coldstart_actor_visual_reasoning.sh

# Free-form benchmarks (visual_toolbench, tir_bench) graded by an
# LLM-as-judge (gpt-5.5).  Verdicts cached on disk under
# <benchmark>/judge_cache/, so re-runs are free.
bash cold_start/run_coldstart_actor_visual_reasoning.sh \
    --num_test_cases 5 --judge -v

# Video-only sweep with extra frames per clip (helps temporal-ordering
# questions on Video-Holmes)
bash cold_start/run_coldstart_actor_visual_reasoning.sh \
    --benchmarks video_holmes siv_bench \
    --num_frames 12 --num_test_cases 5 -v
```

### LLM-as-judge (`--judge`, free-form benchmarks only)

Naïve substring matching is too strict for VTB / TIR-Bench: VTB ships
multi-paragraph rubric-style golds and TIR-Bench occasionally asks for
numeric answers expressed differently from the gold (e.g. `"1.14%"`
vs `"1.15%"`).  Officially VTB grades by rubric and TIR-Bench by
task-specific metric — both use a judge model under the hood.  When
`--judge` is passed:

* For each sample on `visual_toolbench` / `tir_bench` (free-form only),
  a small gpt-5.5 judge call grades `(question, gold, predicted)` with
  a `correct | incorrect | unscoreable` verdict + a 2-sentence reason.
* The verdict is cached at
  ``<output_dir>/<benchmark>/judge_cache/<sample_id>.<key>.json``,
  keyed by `sha256(gold || predicted || judge_model)`.  Re-runs that
  hit the same `(gold, predicted, judge_model)` triple skip the API
  call entirely.
* Per-sample JSON keeps both signals: `correct` (judge-derived when
  enabled, otherwise string-match) and `correct_strmatch` (always
  the diagnostic substring match), plus the full `judge: {…}` block
  with verdict / reason / cached / model.
* MCQ benchmarks (`video_holmes`, `siv_bench`) skip the judge — letter
  equality already grades exactly and a judge call adds zero signal.

Empirically this lifts VTB accuracy on a 5-sample sweep from `0/5`
(string-match) to roughly `1–2/5` (judge), matching the rate at which
gpt-5.5 actually answers the multi-fact rubric prompts correctly.

### SIV-Bench MCQ width (A..L, not A..E)

Heads-up for anyone consuming `iter_siv_bench_samples`: the loader was
previously hard-coded to a 5-letter option set (`A..E`) per the paper,
but the released TSV ships some sub-tasks (notably **Relation
Inference**) with up to **12 options** (`A. service, B. grandparent-
child, … L. boss-employee`).  The cold-start sweep surfaced this as a
silent gold-truncation: rows whose `correct_answer_index='L'` looked
like 5-option rows whose gold was an unreachable letter, and the
actor was mathematically forced to pick wrong.

`benchmarks/siv_bench.py` now carries the full `A..L` letter span end-
to-end (`_ANSWER_LETTERS`, `_parse_options`, `_split_letter_prefixed_options`,
`_normalise_answer_letter`).  Existing 4–5-option rows still parse
identically; over-wide rows (10–12 options) now expose the full
candidate set on `SIVBenchSample.options`.

If you are downstream of `SIVBenchSample.options` and previously
assumed the dict had at most 5 keys (e.g. when iterating MCQ buttons
in a UI), update the upper bound to 12.

### Per-sample JSON shape

```jsonc
{
  "benchmark": "visual_toolbench",
  "sample_id": "68658a711603983919432621",
  "question": "How many rebars …",
  "gold_answer": "There are 6 rebar locations …",
  "answer": "5 rebars; ~4021 mm²",
  "correct": false,                  // judge-derived when --judge, else string-match
  "correct_strmatch": false,         // always the diagnostic substring match
  "judge": {                         // only present when --judge fired on this sample
    "verdict": "incorrect",
    "reason": "Pred says 5 rebars at 32 mm; gold says 6 rebars at 25 mm.",
    "cached": false,
    "judge_model": "openai/gpt-5.5"
  },
  "schema": "<state> … </state>",    // raw VLM output from the vision stage
  "schema_source": "vlm",            // 'vlm' | 'fenced' | 'truncated' | 'untagged' | 'no_image'
  "schema_recovery": "strict",
  "schema_finish_reason": "stop",
  "frames_saved": [...],
  "video_meta": {...}                // only on video benchmarks
}
```
