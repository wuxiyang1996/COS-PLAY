# `browsergym_wrapper`

BrowserGym-specific visual grounding: heuristics, VLM adapter,
OmniParser-v2 glue, and a tool registry — all in one place.

This package centralises everything that depends on
[ServiceNow/BrowserGym](https://github.com/ServiceNow/BrowserGym) (AXTree,
`extra_element_properties`, `browsergym_id`, set-of-marks, etc.) so the
cross-domain code in `vlm_wrapper/` can stay environment-agnostic.

> **TL;DR** — give it a BrowserGym observation dict (or just a screenshot)
> and get the canonical `<state>…</state>` schema back, via any of the three
> heads (heuristic / vision LLM / local OmniParser-v2).

---

## What's inside

| File | What it does |
|------|--------------|
| `__init__.py`              | Re-exports the public API (`generate_label`, `browser_obs_to_schema`, `obs_to_schema`, `build_browser_registry`, `grounding_image_to_schema`, `grounding_obs_to_schema`, …). The OmniParser head is imported lazily so a thin install without torch/transformers still works. |
| `adapter.py`               | Vision head: `generate_label(image, …)` calls a VLM (GPT-4o by default) and returns the `<state>` schema. Also exposes `browser_obs_to_schema(obs, …)` which unpacks a BrowserGym `obs` dict. |
| `heuristic.py`             | Text head: `obs_to_schema(obs, …)` walks the AXTree + `extra_element_properties` deterministically into a schema. Fast, free, no API. |
| `grounding.py`             | OmniParser-v2 head: `grounding_image_to_schema(image, …)` and `grounding_obs_to_schema(obs, …)` run YOLO + OCR + Florence-2 locally (via `vlm_wrapper.grounding`) and emit the schema. Hosts the shared `_elements_to_schema` helper that `osworld_wrapper.grounding` reuses with `domain="desktop"`. |
| `tools.py`                 | Tool-calling registry for multi-turn visual reasoning (query bbox, search elements, get page info, get SoM elements, etc.). Build with `build_browser_registry(obs)`. |
| `example.py`               | `python -m browsergym_wrapper.example` — synthetic shopping-page obs + screenshot for end-to-end smoke tests. **No browser launched** (PIL-drawn fake). |
| `example_grounding.py`     | `python -m browsergym_wrapper.example_grounding` — Head 3 demo: OmniParser-v2 on the synthetic screenshot. Compare-mode runs Head 1 vs Head 3 side by side. |
| `demo_visual_grounding.py` | Full Mode A / B / C driver: direct grounding (no API), VLM tool-calling loop (needs API key), and a three-way comparison (heuristic + grounding + tool loop). |
| `test_schema_gen.py`       | `python -m browsergym_wrapper.test_schema_gen` — boots **real** `browsergym/openended` on Google + Wikipedia (headless Chromium via Playwright) and sends each rendered page to GPT-4o through `adapter.generate_label`. Requires the full `browsergym` env. |
| `tests/`                   | `pytest` suite with offline (always-run) and `@pytest.mark.live` GPT-4o regression tests for the adapter + legacy `vlm_wrapper.browser_adapter` shim. `tests/conftest.py` autoloads `.env` so `OPENAI_API_KEY` / `VLM_TEST_API_KEY` are picked up without `python-dotenv`. |

The legacy modules under `vlm_wrapper/` (`browser_adapter.py`,
`browser_heuristic.py`, `grounding_browsergym.py`, `tools_browser.py`,
`example_browsergym.py`) are now thin compatibility shims that re-export
from this package, so existing imports keep working.

---

## Install

The package itself is pure Python; **how much you install depends on which
heads you want to run.** Three install levels, lightest to heaviest:

### Level 1 — heuristic + synthetic demos only (no BrowserGym)

Works inside any of the repo's conda envs (`game-ai-agent`, `osworld`, …).
Only needs `numpy`, `Pillow`, and `vlm_wrapper.schema`. This is enough to
run `example.py` and `obs_to_schema(...)` against a hand-built obs dict —
no Playwright, no Chromium, no BrowserGym install.

```bash
# Already covered by the main env; nothing extra to do:
conda activate game-ai-agent
python -m browsergym_wrapper.example          # Head 1 only
```

### Level 2 — heuristic + vision (GPT-4o adapter, no real browser)

Adds `openai` + a working API key. Still no browser; the screenshot is the
synthetic PIL one from `example.build_browsergym_obs()`.

```bash
conda activate game-ai-agent
export OPENAI_API_KEY=sk-...                  # or OPENROUTER_API_KEY
python -m browsergym_wrapper.example --vision
```

### Level 3 — full BrowserGym (real Chromium-rendered pages)

Required for `test_schema_gen.py`,
`visual_grounding_tests/generate_browsergym_schema.py --urls ...`, and
anything else that calls `gym.make("browsergym/openended", ...)`. Use the
dedicated installer in `install/`:

```bash
# Creates the `browsergym` conda env from install/browsergym.environment.yml,
# clones ServiceNow/BrowserGym, pip-installs the sub-packages editable,
# installs Playwright system deps + Chromium, and runs install/browsergym_smoke.py.
bash Multi-hop-Reasoning-VLM-Agent/install/install_browsergym.sh

conda activate browsergym
python install/browsergym_smoke.py            # one-liner OK/FAIL/WARN per dep
```

What that buys you:

- `playwright==1.44` (BrowserGym-core hard pin) + headless Chromium binary
- `browsergym.{core,miniwob,webarena,visualwebarena,assistantbench,experiments}` editable
- `libwebarena 0.0.5`, `libvisualwebarena 0.0.15`, `torch 2.4.1+cu121` (visualwebarena scoring model)
- `openai`, `anthropic`, `google-genai` for the agent-side smoke tests

WorkArena is not bundled (its `browsergym-workarena` package pins `tqdm` incompatibly):

```bash
pip install --no-deps browsergym-workarena    # optional, on top
```

For the broader environment matrix and the full benchmark layout (WebArena
hosts, VisualWebArena assets, MiniWoB++ HTML server) see
[`install/INSTALL_BENCHMARKS.md`](../install/INSTALL_BENCHMARKS.md) §2.

#### Headed mode (visible Chromium window)

`test_schema_gen.py` hard-codes `headless=True`. To watch the browser:

- For `visual_grounding_tests/generate_browsergym_schema.py`, pass `--no_headless`.
- For `test_schema_gen.py`, edit the `headless=True` kwarg in
  `capture_browser_obs()` (no CLI flag yet).
- On a headless server, `Xvfb :99 -screen 0 1920x1080x24 & export DISPLAY=:99`
  before launching.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `playwright._impl._errors.Error: Executable doesn't exist` | `python -m playwright install chromium` (the installer does this; re-run if you skipped it). |
| `BrowserType.launch: Host system is missing dependencies` | Re-run `python -m playwright install-deps chromium` with sudo, or apt-install the libs listed in [`install/README.md`](../install/README.md) §"Troubleshooting → BrowserGym". |
| `ModuleNotFoundError: browsergym.core` | You're not in the `browsergym` conda env. `conda activate browsergym`. |
| OmniParser head silently disabled (`grounding_*` not exported) | Heavyweight extras (torch, transformers, ultralytics, easyocr) not installed in the current env — that's by design. Run the OmniParser head from the `vlm_benchmarks` env or install the extras into `browsergym`. |

---

## Quick start

### Head 1 — Heuristic (AXTree → schema)

```python
from browsergym_wrapper import obs_to_schema

schema_str = obs_to_schema(obs, step=3, task_id="webarena.shopping.143")
```

### Head 2 — Vision (screenshot → VLM → schema)

```python
from browsergym_wrapper import browser_obs_to_schema

result = browser_obs_to_schema(
    obs,
    step=3,
    task_id="webarena.shopping.143",
    model="gpt-4o",                # or rely on $VLM_LABEL_MODEL
    api_key="sk-...",              # or rely on the OpenAI client default
)
print(result["schema"])
```

### Head 3 — OmniParser-v2 (screenshot → local models → schema)

```python
from browsergym_wrapper import grounding_obs_to_schema, grounding_image_to_schema

result = grounding_obs_to_schema(obs, step=3, task_id="webarena.shopping.143")

result = grounding_image_to_schema(pil_image, goal="Find cheapest laptop", step=2)
```

### Tool-calling registry (multi-turn grounding)

```python
from browsergym_wrapper import build_browser_registry
from vlm_wrapper.tool_loop import browser_generate_label_with_tools

registry = build_browser_registry(obs)
result = browser_generate_label_with_tools(
    obs["screenshot"], goal=obs["goal"], task_id="…",
    registry=registry, model="gpt-4o", api_key="sk-...",
)
print(result["schema"])      # <state>…</state>
print(result["tool_trace"])  # [{call, result}, …] — SFT training data
```

---

## Run the demos

### Synthetic shopping page (no BrowserGym install required)

`example.py` builds a fake obs dict (AXTree + bounding boxes + PIL-drawn
screenshot) so you can exercise both Head 1 and Head 2 end-to-end without
booting a browser:

```bash
python -m browsergym_wrapper.example                                    # Head 1 only
python -m browsergym_wrapper.example --vision                           # Head 1 + Head 2 (needs OPENAI_API_KEY)
python -m browsergym_wrapper.example --vision \                         # OpenRouter
    --api-key "$OPENROUTER_KEY" \
    --base-url https://openrouter.ai/api/v1 \
    --model openai/gpt-4.1
```

### OmniParser-v2 grounding on a screenshot (Head 3)

```bash
python -m browsergym_wrapper.example_grounding                          # Head 3 only
python -m browsergym_wrapper.example_grounding --compare                # Head 1 vs Head 3
python -m browsergym_wrapper.example_grounding --image path/screen.png  # custom screenshot
python -m browsergym_wrapper.example_grounding --save-annotated out.png
python -m browsergym_wrapper.example_grounding --paddleocr --no-caption # faster variant
```

### Full Mode A / B / C comparison

```bash
# Mode A — direct grounding only (no API)
python -m browsergym_wrapper.demo_visual_grounding

# Mode B — VLM tool-calling loop (needs API key)
python -m browsergym_wrapper.demo_visual_grounding --vlm-tools \
    --api-key sk-... --model openai/gpt-4.1

# Mode C — heuristic + grounding + tool loop, side by side
python -m browsergym_wrapper.demo_visual_grounding --compare-all \
    --api-key sk-... --model openai/gpt-4.1
```

### Real BrowserGym pages (Chromium-rendered)

Needs Level 3 install above (`browsergym` conda env). Boots
`browsergym/openended`, navigates to the URL, captures the rendered
screenshot + AXTree, and sends it to GPT-4o for schema generation:

```bash
conda activate browsergym
python -m browsergym_wrapper.test_schema_gen                            # default: Google + Wikipedia
python -m browsergym_wrapper.test_schema_gen \
    --url https://en.wikipedia.org/wiki/Reinforcement_learning \
    --goal "Find the section about temporal difference learning" \
    --save-images
```

API key is loaded from `api_keys.open_router_api_key` (next to repo root)
or set `OPENROUTER_API_KEY` / `OPENAI_API_KEY`.

---

## Tests

```bash
# Offline only (always run; verifies imports + adapter contract)
pytest browsergym_wrapper/tests/ -q

# Including the live GPT-4o adapter test
#   (set OPENAI_API_KEY or VLM_TEST_API_KEY first; .env is auto-loaded)
pytest -m "live or not live" browsergym_wrapper/tests/ -q
```

Live-test schemas are saved to `out/schemas/browser.schema.txt` for
inspection.
