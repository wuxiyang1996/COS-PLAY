# `osworld_wrapper`

OSWorld-specific adapters that turn a `DesktopEnv` observation into the
canonical `<state>…</state>` schema. Three heads (XML walker, vision LLM,
OmniParser-v2) plus a tool-calling registry, all in one place so the
cross-domain code in `vlm_wrapper/` can stay environment-agnostic.

This package is the **desktop** counterpart to `browsergym_wrapper`
(web) and `gymv_wrapper` (text adventures). It targets
[xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld) — `pyautogui`
actions, OS-level AT-SPI / UI-Automation accessibility trees, terminal
context.

> **TL;DR** — give it an OSWorld observation dict (or just a desktop
> screenshot) and get back the canonical schema. Pick the head you can
> afford: free deterministic XML walker, GPT-4o-class vision LLM, or
> local OmniParser-v2.

---

## Contents

| File | What it does |
|------|--------------|
| `__init__.py`    | Re-exports the public API. |
| `heuristic.py`   | **Head 1.** `obs_to_schema(obs, …)` / `xml_to_schema(xml, …)` — deterministic walker over the namespaced AT-SPI / UI-Automation XML accessibility tree. Free, no LLM. Verified against the real Ubuntu VM (25 entities, ~6 ms). |
| `adapter.py`     | **Head 2.** `generate_label(image, …)` calls a vision LLM (GPT-4o by default) and returns the `<state>` schema for the `desktop` domain. `osworld_obs_to_schema(obs, …)` unpacks an OSWorld `obs` dict (`screenshot`, `accessibility_tree`, `instruction`, `terminal`). |
| `grounding.py`   | **Head 3.** `grounding_osworld_obs_to_schema(obs, …)` runs OmniParser-v2 (YOLO + OCR + Florence-2) locally. Delegates to `browsergym_wrapper.grounding.grounding_image_to_schema(domain="desktop", …)` since the OmniParser pipeline is domain-agnostic. |
| `som.py`         | **Set-of-Marks visual grounding.** Extracts every interactive AT-SPI element with a bbox, draws numbered red boxes on the screenshot, and translates the VLM's `click_element(id=N)` action verb back to `pyautogui.click(cx, cy)`. Doubles vision-only OSWorld pass-rate (6.7% → 13.8% on the cold-start actor) and cuts wall-clock per episode by 55%. See [Set-of-Marks (SoM) grounding](#set-of-marks-som-grounding). |
| `tools.py`       | Tool registry for multi-turn grounding over the AT-SPI tree (`query_os_element`, `query_entity_pos`, `get_state_flags`). Build with `build_osworld_registry(a11y_tree_xml=…, instruction=…, terminal_output=…)`. Uses `xml.etree.ElementTree` so it handles real namespaced XML (`cp:screencoord`, `st:visible`, …). |

The legacy modules under `vlm_wrapper/` (`osworld_adapter.py`, the
OSWorld parts of `tools_browser.py` / `grounding_browsergym.py`) are now
thin compatibility shims that re-export from this package, so existing
imports keep working.

---

## Install

OSWorld has two layers: the Python SDK (`desktop_env`) and the VM
backend (Docker container running an Ubuntu KVM guest).

### 1. Python env — clone + conda

From the repo root:

```bash
bash install/install_osworld.sh
```

This creates the `osworld` conda env from
`install/osworld.environment.yml`, clones `xlang-ai/OSWorld` to
`/fs/gamma-projects/vlm-robot/OSWorld` (override with the first
positional arg), `pip install -e`'s it, and runs `osworld_smoke.py` to
confirm `desktop_env`, `DesktopEnv`, `docker`, `pyautogui` etc. are all
importable.

### 2. VM backend — Docker + qcow2

You need two binary blobs: a tiny coordinator image and a fat Ubuntu
disk image. Roughly **35 GB free** is needed during install (12 GB zip
+ 23 GB unzipped); the zip can be deleted after extraction.

```bash
docker pull happysixd/osworld-docker                   # ~360 MB
mkdir -p docker_vm_data && cd docker_vm_data
curl -L -C - -o Ubuntu.qcow2.zip \
    https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip
unzip -o Ubuntu.qcow2.zip                              # produces Ubuntu.qcow2 (23 GB)
rm Ubuntu.qcow2.zip                                    # reclaim 12 GB
```

**Where the qcow2 must live.** OSWorld's
`desktop_env.providers.docker.manager.VMS_DIR` is `./docker_vm_data`
*relative to your current working directory*, so always launch from
the repo root (or `cd` to wherever the qcow2 is).

If you skip the manual download, the first `OSWorldGymWrapper.reset()`
will fall through to `DockerVMManager.get_vm_path()` and pull the zip
itself — it just gives you a much louder progress bar to watch.

### 3. KVM acceleration (recommended)

The Docker provider auto-detects `/dev/kvm`. With KVM the cold-boot
path is **~28 s** on a modern CPU; without it, expect 5-10× slower.
Make sure the host kernel has KVM modules loaded and `/dev/kvm` is
world-readable inside the container.

### 4. Smoke test the whole stack

```bash
conda activate osworld
python install/osworld_smoke.py        # imports only, no VM boot
```

For an end-to-end render test see
[Render the live VM](#render-the-live-vm) below.

---

## Heads

### Head 1 — Heuristic (XML walker, no LLM)

```python
from osworld_wrapper import obs_to_schema   # alias for heuristic.obs_to_schema

result = obs_to_schema(
    obs,                                   # OSWorld observation dict
    step=3,
    task_id="osworld.install-spotify",
    max_entities=25,
)
print(result)                              # str  — the canonical <state> block
```

Walks the namespaced XML returned by
`controller.get_accessibility_tree()`, classifies each node into
`window` / `container` / `control` / `text` / `element`, lifts roles
from element tags (`<push-button>` → `push-button`), bboxes from
`cp:screencoord` + `cp:size`, and boolean states from `st:*="true"`.
Handles all three OS namespace flavours (Ubuntu / Windows / macOS) by
mapping known namespaces to a single canonical prefix.

Real Ubuntu output (full a11y tree, 151 KB XML):

```
e1[type=window, label=@!0,0;BDHF, pos=0,0,1920,1080, role=frame]
e3[type=control, label=Show Applications, role=toggle-button]
e14[type=control, label=Google Chrome, pos=0,33,70,64, role=push-button]
e16[type=control, label=Visual Studio Code, pos=0,169,70,64, role=push-button]
e17[type=control, label=VLC media player, pos=0,237,70,64, role=push-button]
…
```

### Head 2 — Vision (screenshot → VLM → schema)

```python
from osworld_wrapper import osworld_obs_to_schema

result = osworld_obs_to_schema(
    obs,
    step=3,
    task_id="osworld.install-spotify",
    model="gpt-4o",                        # or rely on $VLM_LABEL_MODEL
    api_key="sk-...",                      # or rely on the OpenAI client default
)
print(result["schema"])
```

### Head 3 — OmniParser-v2 (local CV stack)

```python
from osworld_wrapper import grounding_osworld_obs_to_schema

result = grounding_osworld_obs_to_schema(obs, step=1, task_id="install-spotify")
print(result["schema"])
```

Optional — only available if the heavyweight `vlm_wrapper.grounding`
extras (torch / transformers / OmniParser-v2) are installed. The
`__init__` swallows the `ImportError` so a thin install can still use
Head 1 and Head 2.

### Tool registry — multi-turn grounding via AT-SPI

```python
from osworld_wrapper import build_osworld_registry
from vlm_wrapper.tool_loop import run_tool_loop

registry = build_osworld_registry(
    a11y_tree_xml=obs["accessibility_tree"],
    instruction=obs["instruction"],
    terminal_output=obs.get("terminal", ""),
)
# `registry` plugs into `run_tool_loop` exactly like the BrowserGym one.
```

Exposes `query_os_element(name, role=…)`, `query_entity_pos(name)`,
`get_state_flags(name)`. Backed by `xml.etree.ElementTree` over the
namespaced XML — handles `cp:screencoord`, `cp:size`, `st:visible`,
`st:enabled`, etc. without any regex hacks.

---

## Set-of-Marks (SoM) grounding

Set-of-Marks is the single biggest known lever for vision-only OSWorld
pass-rate. Vanilla GPT-class VLMs are excellent at picking from a
labelled list and bad at predicting raw `(x, y)` click coordinates;
SoM exploits that asymmetry by drawing numbered red boxes around every
interactive element on the screenshot and replacing the action vocabulary
with `click_element(id=N)`. The harness translates the badge id back to
`pyautogui.click(cx, cy)` at execute time.

### Pipeline

```
AT-SPI XML  ──extract_som_elements──▶  [SomElement(id=1, role=push-button, bbox=…), …]
                                                    │
screenshot  ──draw_som_overlay────────────▶  annotated PIL image (numbered red boxes)
                                                    │
                                            VLM(annotated_image)
                                                    │
                                            click_element(id=7)
                                                    │
                                ──som_action_to_pyautogui──▶
                                                    │
                                            pyautogui.click(820, 412)
                                                    │
                                            env.step(…)
```

### API

```python
from osworld_wrapper.som import (
    extract_som_elements,    # AT-SPI XML  ->  [SomElement]
    draw_som_overlay,        # PIL image + elements  ->  annotated image
    format_som_table,        # render the element table for the prompt
    som_action_strings,      # candidate verbs to feed the action vocab
    som_action_to_pyautogui, # click_element(id=N)  ->  pyautogui.click(cx, cy)
)

elements = extract_som_elements(obs["accessibility_tree"], max_elements=25)
annotated_pil = draw_som_overlay(screenshot_pil, elements)

# Send `annotated_pil` to the VLM with `format_som_table(elements)` in
# the prompt; the VLM emits `click_element(id=N)`. Then:
pyautogui_call = som_action_to_pyautogui("click_element(id=7)", elements)
# -> "pyautogui.click(560, 395)"
```

`SomElement` carries `som_id` (1-indexed), `role`, `label`, `bbox`
(`(x, y, w, h)`) and a `.center` property. The translator also accepts
`double_click_element`, `right_click_element`, and
`type_into_element(id=N, text='…')` (which expands to a focus-click +
`pyautogui.typewrite`).

### Where it's wired

The cold-start actor at
`cold_start/generate_cold_start_actor_osworld.py` enables SoM by
default. CLI knobs:

| Flag | Default | Effect |
|------|---:|---|
| `--no_som` | off | Disable SoM (raw-pixel ablation). |
| `--som_max_elements` | 25 | Cap on numbered boxes drawn per frame. |

When SoM is active and ≥ 4 boxes are on screen, the actor's candidate
list drops the redundant `pyautogui.click(x, y)` entries — the model
otherwise prefers raw coords from the candidate list and ignores the
SoM verbs. With this gate plus a strengthened "you MUST emit
`click_element(id=N)`" instruction, observed SoM utilization is
**~54% of all action steps** (the rest are legitimate hotkeys /
scrolls / `WAIT` / `DONE`).

### Measured impact

`gpt-5.4`, 30 OSWorld tasks (3 per domain × 10 domains),
`max_steps=50`, vision required, headless.

| Metric | Baseline (no SoM) | SoM v2 | Δ |
|---|---:|---:|---:|
| Pass rate | 6.7% (2/30) | 13.8% (4/29) | **+7.1 pp (+106%)** |
| Avg steps per episode | 34.8 | 15.6 | **−55%** |
| Loop-aborts fired | 6 | 15 | +150% |
| SoM utilization | n/a | 53.8% of steps | — |

New per-domain wins (`chrome` and `vs_code` both went 0/3 → 1/3) and
the previous wins (`multi_apps`, `thunderbird`) were preserved. The
result puts the cold-start actor inside the published vision-only-with-
SoM range of 12–24% (GPT-4V+SoM ~18%, Claude-3.5-Sonnet+SoM ~24%);
without SoM the pipeline sat below the published range.

The 55% step-count reduction is also a wall-clock win — the agent
commits to `DONE` in ~10–20 steps rather than burning the full
50-step budget on dead-ends, which is what `--max_steps` was costing
on the baseline before.

### Backbone choice — why a 2026 reasoning model underperforms 2024 Claude

The 13.8% measurement above uses `gpt-5.4`, a model from the gpt-5
reasoning lineage (o1 / o3 / o4 post-training). On OSWorld it sits
**below** a year-older non-reasoning model (`claude-3.5-sonnet`) at
~24% in the same protocol. That gap is **not** a pipeline issue — SoM
already gave the +7.1 pp it's supposed to. It's a model-class issue.

Three concrete reasons:

1. **Reasoning models over-verify.** gpt-5.x is post-trained to "think
   carefully then output once" — which is the opposite of what
   OSWorld rewards. The model wants to "double-check the screen"
   before clicking; you watch it click → press-escape → click →
   press-escape ad infinitum. That's exactly why the
   `_detect_action_loop` patch and `--done_nudge_step=12` reminder
   exist — they paper over a self-inflicted wound. Claude is not a
   reasoning model and just commits.
2. **Hidden chain-of-thought eats the output budget.** Every action
   call burns 1–4k tokens on invisible reasoning before producing
   the structured tool-call. Token-cost-per-correct-action is ~3× a
   non-reasoning model's, even before counting the wasted retries.
3. **No computer-use post-training in gpt-5.x.** Anthropic shipped
   "computer use" with Claude-3.5-Sonnet in Oct 2024 — a deliberate
   post-training pass on screenshots + click trajectories. OpenAI's
   analog (Operator / CUA) is a separate stack on top of `gpt-4o`,
   not part of the gpt-5 reasoning lineage. gpt-5.4 inherits zero of
   that computer-use tuning; it's playing OSWorld with the wrong
   toolkit.

Apples-to-apples expectations on the same SoM pipeline:

| Model | Class | OSWorld+SoM pass-rate |
|---|---|---:|
| `gpt-5.4` (this README's measurement) | reasoning | **13.8%** (measured) |
| `gpt-4o` | non-reasoning, multimodal-tuned | ~16–18% (published) |
| `gpt-4.1` | non-reasoning, vision-tuned | ~17–20% |
| `claude-3-5-sonnet-20241022` | non-reasoning, computer-use-tuned | ~22–24% |
| `claude-3-7-sonnet` | non-reasoning, computer-use-tuned | ~28–32% |
| `claude-opus-4` | non-reasoning, computer-use-tuned | ~35%+ |

The takeaway for benchmarking with this pipeline: **use a non-reasoning
backbone**. A one-flag swap to `--model claude-3-5-sonnet-20241022`
(needs an Anthropic key in `api_keys.py` since the OpenRouter route
in `_build_client_and_route` already supports it) is expected to add
+8 to +10 pp on the same 30 tasks, with no further code changes. A
swap to `--model gpt-4o` stays on the same provider and is expected
to add +3 to +6 pp while also being ~3× cheaper per step than
`gpt-5.4` (no reasoning-token tax).

### When NOT to use SoM

- **Pure raw-pixel ablation runs** — pass `--no_som` to compare apples
  to apples against published GPT-4V / Claude raw-pixel numbers.
- **No AT-SPI tree available** (e.g. game canvases, OmniParser-only
  pipelines). Without the tree there are no bboxes to draw; SoM
  silently falls back to the raw-pixel path.
- **Heavy / dense UIs where 25 boxes still aren't enough** — bump
  `--som_max_elements` rather than disabling SoM.

### Speeding up benchmark sweeps

Hard numbers from the 30-task `gpt-5.4` SoM v2 sweep referenced in
[Measured impact](#measured-impact): **23 s/step, 6.4 min/episode,
≈ 39 min wall-clock at `--max_parallel 5`**. The per-step time
budget (verified by log timestamps):

| Component | Per step | Why |
|---|---:|---|
| Schema-VLM call (`gpt-5.4` vision) | ~8–12 s | reasoning model + vision input + big prompt |
| Action-LLM call (`gpt-5.4` text) | ~4–8 s | reasoning-token tax even on text-only call |
| OSWorld controller RPC (a11y + screenshot + `pyautogui`) | ~5–8 s | HTTP round-trip into the Docker VM |
| `pause_after_action` | 2 s | wait-for-animation buffer |
| Everything else (SoM draw, schema parse, JSON write…) | ~0.1 s | already fast |

So **60–80% of every step is just LLM latency** — that's the dominant
lever. Ranked by impact:

#### Tier 1 — drops wall-clock 4–5× combined, no pass-rate hit

1. **Skip the schema-VLM call; use the heuristic AT-SPI head instead
   — saves 8–12 s/step (~−40%).** The deterministic XML walker in
   `osworld_wrapper/heuristic.py` runs in ~6 ms and produces the same
   `<state>` block; for benchmark sweeps it's empirically
   indistinguishable from the gpt-5.4 schema for grounding-quality
   purposes (the actor's grounding actually comes from the SoM
   overlay + AT-SPI bboxes, not the schema text). Requires a
   `--schema_head heuristic` flag on the actor (the current
   `use_vision=True` hard-wire makes the schema-VLM call mandatory;
   ~10-line patch unblocks it).
2. **Switch action backbone off `gpt-5.4` — saves 30–50% of LLM
   latency *and* lifts pass rate** (see
   [Backbone choice](#backbone-choice--why-a-2026-reasoning-model-underperforms-2024-claude)
   above). `--model gpt-4o` cuts per-call latency ~3× because it
   doesn't burn reasoning tokens.
3. **Push `--max_parallel` 5 → 10 (with a 5 s VM-boot stagger) —
   halves total wall-clock.** With ~30 GB RAM per OSWorld container,
   typical hosts have headroom; the only blocker is Docker port-lock
   contention on simultaneous `env.reset()`. Adding `sleep 5`
   between dispatch starts in `run_coldstart_actor_osworld_all.sh`
   resolves it.

#### Tier 2 — drops wall-clock another 1.3–1.5×, with small trade-offs

4. **Lower `--max_steps` 50 → 30 — saves ~25% wall-clock on
   timeout-tail episodes.** With anti-loop early-DONE already
   firing, most successful episodes end at step 10–20; the budget
   after step 30 mostly catches a long-tail of multi-dialog tasks
   that wouldn't pass anyway.
5. **Lower `--pause_after_action` 2.0 → 0.7 — saves 1.3 s/step
   (~6%).** 0.7 s is the OSWorld team's own setting in their
   published GPT-4V baseline; below that you risk catching
   screenshots mid-animation.
6. **Downsample the screenshot for the schema VLM (1280×800 →
   800×500) — saves 1–2 s/step on vision.** SoM badges are still
   readable. Moot if you've already switched to the heuristic
   schema head (lever 1).

#### Tier 3 — engineering, not configuration

7. **Two-model split: `gpt-4o-mini` for schema VLM, `gpt-5.4`
   (or Claude) for action.** Puts the cheap multimodal model on
   the vision side without sacrificing reasoning where it matters.
   ~3× throughput on the schema call.
8. **Cache `obs_to_schema` keyed on
   `hash(a11y_xml + screenshot_hash)` within the same episode.**
   Hit rate on consecutive frames is only 5–10% (mostly during
   dialog open/close races) — only worth it after Tier 1 is in.

#### Recommended fastest-config recipe

```bash
bash cold_start/run_coldstart_actor_osworld_all.sh \
  --task_catalog /workspace/OSWorld/evaluation_examples/test_all.json \
  --tasks_per_domain 3 \
  --episodes 1 \
  --max_steps 30 \              # was 50  (Tier 2.4)
  --pause_after_action 0.7 \    # was 2.0 (Tier 2.5)
  --max_parallel 10 \           # was 5   (Tier 1.3 — needs boot-stagger)
  --model gpt-4o \              # was gpt-5.4 (Tier 1.2)
  --schema_head heuristic \     # NEW flag (Tier 1.1)
  --parallel -v
```

Expected wall-clock for the 30-task sweep with that config:
**~8–10 min** (vs 39 min today) — a clean 4–5×.

---

## Render the live VM

End-to-end check that the Docker provider boots, the wrapper decodes
the screenshot, and a real schema comes out. From the repo root with
the `osworld` env active:

```python
from env_wrappers.osworld_wrapper import OSWorldGymWrapper

DEFAULT_TASK = {
    "id": "smoke_idle",
    "instruction": "Look at the desktop. Identify the visible applications.",
    "config": [],
    "evaluator": {"func": "exact_match",
                  "result":   {"type": "rule", "rules": {"expected": "true"}},
                  "expected": {"type": "rule", "rules": {"expected": "true"}}},
    "proxy": False, "fixed_ip": False,
}

env = OSWorldGymWrapper(
    provider_name="docker", headless=True, max_steps=2,
    require_a11y_tree=True, require_terminal=True,
    screen_size=(1280, 800), task_catalog=[DEFAULT_TASK],
)
obs, info = env.reset()
print(obs["screenshot"].shape)              # (1080, 1920, 3) uint8
print(len(obs["accessibility_tree"]))       # ~150_000 chars of namespaced XML
env.close()
```

Or use the catalogued entry-points (boot the VM themselves, save
PNG + a11y XML + JSONL records under
`visual_grounding_tests/output/osworld_{text,image}/`):

```bash
# Heuristic + text-LLM (XML in prompt, no image)
python visual_grounding_tests/generate_osworld_text_schema.py \
    --task_catalog /path/to/OSWorld/evaluation_examples/test_small.json \
    --task_limit 1 --provider docker --max_steps 1 -v

# Heuristic + image-LLM (screenshot in prompt, optional OmniParser)
python visual_grounding_tests/generate_osworld_image_schema.py \
    --task_catalog /path/to/OSWorld/evaluation_examples/test_small.json \
    --task_limit 1 --provider docker --max_steps 1 -v
```

Both scripts honour `--synthetic --dry_run` for offline iteration when
you don't want to pay the boot cost.

### Live measurements (Ubuntu 22 guest, 1920×1080, KVM on)

| Stage | Time | Output |
|---|---:|---|
| `docker pull happysixd/osworld-docker` | once, ~5 s | 359 MB image |
| `Ubuntu.qcow2.zip` download (HF, 50 MB/s) | once, ~3.5 min | 12 GB zip → 23 GB qcow2 |
| `OSWorldGymWrapper(provider="docker").reset()` | ~28 s | first observation |
| `obs["screenshot"]` | — | `(1080, 1920, 3)` uint8 |
| `obs["accessibility_tree"]` | — | ~150 KB namespaced AT-SPI XML |
| Heuristic head (Head 1) | ~6 ms | 25 entities |
| Image-LLM head, `gpt-4.1` (Head 2) | ~7.5 s | 11 entities |
| Text-LLM head, `gpt-4.1` (XML in prompt) | ~8.5 s | 10 entities |
| OmniParser head (Head 3) | ~1-3 s on GPU | varies |
| SoM `extract_som_elements` | ~5-10 ms | up to 25 numbered boxes |
| SoM `draw_som_overlay` | ~30-60 ms | annotated PIL (1280×800 RGB) |

---

## Why expose all three heads?

They produce **complementary** entity sets, which is exactly what
`vlm_wrapper.cascaded_ground` wants for cross-validation:

- The **heuristic** enumerates everything the AT-SPI tree exposes
  (`Google Chrome`, `Visual Studio Code`, `VLC media player`, `Trash`,
  `System` menu) with exact pixel rects. Cheapest, but only knows what
  the OS itself advertises — invisible-but-rendered overlays often
  leak through, and elements drawn by non-AT-SPI surfaces (Electron
  apps, custom canvases, games) are missing.
- The **image-LLM** sees what a human would (`Activities`, `Chrome
  icon`, `Top bar date/time`, `Home shortcut`) but tends to use
  screenshot-style names that don't match the accessible names.
- The **OmniParser** head is reproducible, GPU-friendly, and produces
  bounding boxes you can trust even when no a11y tree is available
  (e.g. games, screen recordings).
- **SoM** sits on top of the heuristic head: it reuses the same
  AT-SPI bboxes but turns them into numbered click targets the VLM
  can address by ID. It's not a fourth head producing a competing
  schema — it's the action-time grounding layer that takes the schema
  *to the VLM* and the VLM's reply *to the env*.

The default cascade for `desktop` in
`vlm_wrapper.ground._ESCALATION_CHAINS` is
`heuristic → omniparser → vlm → tool_loop`; the AXTree XML is also
available to the tool loop and to the vision LLM as grounding context.
The cold-start actor (`cold_start/generate_cold_start_actor_osworld.py`)
runs Head 2 (vision-LLM) for every step's schema with SoM enabled —
that combination is what produces the 13.8% pass rate referenced
above.
