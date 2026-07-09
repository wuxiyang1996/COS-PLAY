# Visual-Grounding Benchmarks — Installation Guide

This document covers the **five task domains** targeted by the
`vlm_wrapper` grounding pipeline (see
[`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md`](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md) §2 and
[`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md`](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) §3).

For training / inference of the actor + skill stack, see the original
[`install/README.md`](README.md). This file is scoped to what's needed
to **drive the benchmark runtimes** and **run the grounding pipeline**
on each domain.

---

## TL;DR — three conda envs, one per runtime family

The three interactive runtimes have hard-pinned dependency sets that
cannot co-resolve (gymnasium 0.28 vs 1.2+, transformers 4.35 vs 4.51+,
playwright pinned to 1.44 for BrowserGym-core). They each get their own
env. The four **visual-reasoning benchmarks** (TIR-Bench, VisualToolBench,
Video-Holmes, SIV-Bench) — both their **loaders + sample iterators** and
the API-driven `cascaded_ground` parser — run **inside `game-ai-agent`**.
The optional `vlm_benchmarks` env is only required for the **heavy
in-process grounding heads** (GroundingDINO, OmniParser-v2, Florence-2,
EasyOCR, YOLO) used by the offline labeling pipeline.

> **Note on gym-v:** in the new install flow gym-v lives **inside the
> `game-ai-agent` env** (it co-resolves cleanly with `gymnasium 1.3` +
> `numpy 2.x` + `torch 2.11+cu130`). See [`README.md`](README.md) §1. We
> still keep [`install_gymv.sh`](install_gymv.sh) /
> [`gymv.environment.yml`](gymv.environment.yml) for users who want a
> standalone `gymv` env (e.g. VLM-eval-only setups), but for the full
> training stack you do not need a separate env.

| Domain     | Conda env        | Upstream source                                                     | Purpose                                                                     |
|------------|------------------|---------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `gymv`     | `game-ai-agent`  | [ModalMinds/gym-v](https://github.com/ModalMinds/gym-v)             | 179 procedurally-generated visual envs (incl. 13 `Temporal/*` retro games)  |
| `image_qa` | `game-ai-agent`  | HF `datasets` + cache, local mirror                                 | VisualToolBench + TIR-Bench loaders (`visual_reasoning_wrapper.benchmarks`) |
| `video_qa` | `game-ai-agent`  | local on-disk videos                                                | Video-Holmes + SIV-Bench loaders (decord-based frame sampling)              |
| `browser`  | `browsergym`     | [ServiceNow/BrowserGym](https://github.com/ServiceNow/BrowserGym)   | MiniWoB++ / WebArena / VisualWebArena / AssistantBench (2,063 tasks)        |
| `desktop`  | `osworld`        | [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)             | OSWorld desktop runtime (369 Ubuntu + 49 Windows tasks)                     |
| `grounding`*| `vlm_benchmarks`*| —                                                                  | *Optional* heavy grounding heads (GroundingDINO, OmniParser-v2, Florence-2, EasyOCR) |

\* `vlm_benchmarks` is only needed if you want to run the **offline
grounding labeling pipeline** locally. For evaluating the four
visual-reasoning benchmarks against an OpenAI/Anthropic/Google API,
`game-ai-agent` is sufficient — its `cascaded_ground` calls hit the
remote VLM directly.

### Incompatibility matrix (why we still need separate envs)

| Package        | game-ai-agent     | BrowserGym    | OSWorld         | vlm_benchmarks |
|----------------|-------------------|---------------|-----------------|----------------|
| `gymnasium`    | `1.3.x`           | `>=0.27`      | `~=0.28.1`      | any            |
| `playwright`   | —                 | `==1.44`      | unpinned        | latest         |
| `transformers` | `5.6.x`           | `4.57.x`      | `~=4.35.2`      | `>=4.51,<4.56` |
| `torch`        | `2.11.0+cu130`    | `2.4.1+cu121` | `~=2.5.0+cu124` | `2.4.1+cu121`  |
| `numpy`        | `2.x`             | `1.26.4`      | `1.24.4`        | `<2.0`         |
| `tqdm`         | latest            | `>=4.66.2` *  | `~=4.65.0`      | latest         |
| `decord`       | `0.6.x`           | —             | —               | `0.6.x`        |

\* `browsergym-workarena` only — install it on top of the `browsergym`
env with `pip install --no-deps browsergym-workarena`.

`game-ai-agent` is the unified training/inference + game-environment env
defined in [`README.md`](README.md). It's listed here only to show why it
cannot share dependencies with BrowserGym / OSWorld. You install it via
`install_main_env.sh`, not from this file.

---

## 0. Prerequisites

```bash
# Miniconda3 / Anaconda with conda on PATH
conda --version

# Workspace root
cd /fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent
```

For HuggingFace-gated datasets/models (Video-Holmes, OmniParser-v2):

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```

---

## 1. `gymv` — ModalMinds/gym-v

> **Recommended path:** gym-v is now installed automatically inside the
> `game-ai-agent` env by `install_main_env.sh` (with the 13 `Temporal/*`
> Sega Genesis envs enabled when you pass a ROM zip). See
> [`README.md`](README.md) §1. The standalone `gymv` env below is only
> needed for VLM-eval-only or grounding-only setups where you don't want
> the full training stack.

### Install — unified path (recommended)

```bash
# from cos-play parent dir
bash Multi-hop-Reasoning-VLM-Agent/install/install_main_env.sh \
     "" \
     /path/to/Mega_Drive_Mini_Full_Set.zip
# gym-v is installed editable into game-ai-agent at the same time
conda activate game-ai-agent
python -c "import gym_v, gym_v.envs; print('Temporal/*:', sum(1 for k in gym_v.registry if k.startswith('Temporal/')))"
```

### Install — standalone `gymv` env (optional)

```bash
bash install/install_gymv.sh            # creates env, clones gym-v, pip install -e .[games,spatial], smoke-tests
```

Or manually:

```bash
git clone https://github.com/ModalMinds/gym-v.git /fs/gamma-projects/vlm-robot/gym-v
conda env create -f install/gymv.environment.yml
conda activate gymv
pip install -e "/fs/gamma-projects/vlm-robot/gym-v[games,spatial]"
# optional: .[temporal] (stable-retro), .[vlmeval], .[reasoning-gym]
python install/gymv_smoke.py
```

### What's bundled

All 179 envs ship with the source install. Default enabled groups:

| Group      | Packages                              | Envs                                                                      |
|------------|---------------------------------------|---------------------------------------------------------------------------|
| `games`    | textarena, pettingzoo[classic]        | Chess, TicTacToe, TextArena games                                         |
| `spatial`  | minigrid, miniworld                   | 2D + 3D navigation                                                        |
| `temporal` | stable-retro                          | 13 Sega Genesis games — auto-applied when you pass a ROM zip to `install_main_env.sh` (see [`gymv_temporal_patch/`](gymv_temporal_patch/)) |
| `vlmeval`  | VLMEvalKit (git)                      | VLM evaluation benchmarks                                                 |

### Quick-start

```python
import gym_v
env = gym_v.make("Games/TicTacToe-v0")
obs, info = env.reset(seed=0)
# obs = {"agent_0": Observation(image=PIL.Image, text="...", metadata={})}
env.close()

# Retro / Temporal/* envs (after gymv_temporal_patch + ROMs imported):
import gym_v.envs                               # triggers Temporal/* registration
env = gym_v.make("Temporal/StreetsOfRage2-v0")  # NOTE: gym-v's Env doesn't accept render_mode=
obs, info = env.reset(seed=0)
env.close()
```

---

## 2. `browser` — ServiceNow/BrowserGym

### Install

```bash
bash install/install_browsergym.sh      # creates env, clones repo, pip installs each sub-package editable, playwright install-deps + install
```

Or manually:

```bash
git clone https://github.com/ServiceNow/BrowserGym.git /fs/gamma-projects/vlm-robot/BrowserGym
conda env create -f install/browsergym.environment.yml
conda activate browsergym
cd /fs/gamma-projects/vlm-robot/BrowserGym
pip install -e ./browsergym/core \
            -e ./browsergym/miniwob \
            -e ./browsergym/webarena \
            -e ./browsergym/visualwebarena \
            -e ./browsergym/assistantbench \
            -e ./browsergym/experiments
python -m playwright install-deps chromium    # system libs (needs sudo on first install)
python -m playwright install chromium         # browser binary
```

### Task counts (after install)

```bash
conda run -n browsergym python -c "
import browsergym.miniwob, browsergym.webarena, browsergym.visualwebarena, browsergym.assistantbench
import gymnasium
from collections import Counter
ids = [k for k in gymnasium.envs.registry if k.startswith('browsergym/')]
b = Counter(k.split('/')[1].split('.')[0] for k in ids)
for k,v in sorted(b.items(), key=lambda x: -x[1]): print(f'  {k:20s} {v}')
print(f'  total: {len(ids)}')
"
```

Expected: 910 visualwebarena + 812 webarena + 215 assistantbench + 125 miniwob + 1 openended = **2,063 tasks**.

### Optional WorkArena

```bash
conda activate browsergym
pip install --no-deps browsergym-workarena   # needs playwright==1.44.0 too; compatible
```

### Self-hosted sites

MiniWoB++ HTML pages ship with the pip package. WebArena/VisualWebArena
ship only task definitions — the actual Shopping / Reddit / GitLab /
Classifieds Docker images must be hosted separately
(see <https://github.com/web-arena-x/webarena#setup-instructions>).
Visual grounding label collection can run against any Playwright page
without the hosted sites.

---

## 3. `desktop` — xlang-ai/OSWorld

### Install

```bash
bash install/install_osworld.sh         # creates env, clones OSWorld, pip install -r requirements.txt, pip install -e .
```

Or manually:

```bash
git clone https://github.com/xlang-ai/OSWorld.git /fs/gamma-projects/vlm-robot/OSWorld
conda env create -f install/osworld.environment.yml
conda activate osworld
cd /fs/gamma-projects/vlm-robot/OSWorld
pip install -r requirements.txt
pip install -e .
python install/osworld_smoke.py
```

### Task counts (after install)

```bash
ls /workspace/OSWorld/evaluation_examples
# README.md examples examples_windows settings test_all.json test_infeasible.json test_nogdrive.json test_small.json
find /workspace/OSWorld/evaluation_examples/examples         -maxdepth 2 -name '*.json' | wc -l   # → 369  (Ubuntu)
find /workspace/OSWorld/evaluation_examples/examples_windows -maxdepth 2 -name '*.json' | wc -l   # →  49  (Windows)
```

### VM backend

To actually run desktop tasks you need one of:

| Backend  | Setup                                                                           |
|----------|---------------------------------------------------------------------------------|
| Docker   | `docker pull happysixd/osworld-docker` (requires KVM on host for performance)   |
| VMware   | Install VMware Workstation Pro, configure `vmrun`; see OSWorld's `INSTALL_VMWARE.md` |
| AWS      | See OSWorld's `SETUP_GUIDELINE.md` — Host-Client architecture, parallel eval    |

The `grounding_osworld_obs_to_schema` converter in `vlm_wrapper` works
against any captured screenshot+a11y dict, so offline schema labeling
does not require a VM.

---

## 4. `image_qa` + `video_qa` — runs inside `game-ai-agent`

The four visual-reasoning benchmarks live in
[`visual_reasoning_wrapper/benchmarks/`](../visual_reasoning_wrapper/benchmarks/)
and run from `game-ai-agent` directly — no separate env is required for
loading samples, decoding image/video frames, or calling
`cascaded_ground` against an external VLM API.

| Key                | Modality | Module                                                         | Local source                                                               |
|--------------------|----------|----------------------------------------------------------------|----------------------------------------------------------------------------|
| `tir_bench`        | image    | `visual_reasoning_wrapper.benchmarks.tir_bench`                | `data/datasets/TIR-Bench/TIR-Bench.json` (local mirror) — 1 215 questions  |
| `visual_toolbench` | image    | `visual_reasoning_wrapper.benchmarks.visual_toolbench`         | HF Hub stream (`ScaleAI/VisualToolBench`, ~7.3 GB) — 1 204 single-turn rows |
| `video_holmes`     | video    | `visual_reasoning_wrapper.benchmarks.video_holmes`             | `data/Video-Holmes/Benchmark/` — 1 837 MCQs over 503 cropped clips         |
| `siv_bench`        | video    | `visual_reasoning_wrapper.benchmarks.siv_bench`                | `data/SIV-Bench/{origin,w_sub,wo_sub,SIV-Bench-QA.tsv}` — 8 728 MCQs / 2 792 clips |

`requirements.txt` now installs `datasets`, `Pillow`, `opencv-python`,
and `decord` into `game-ai-agent`, which is everything the four
loaders need.

### Disk layout (relative to repo root)

The `default_*_root()` helpers in each loader resolve to
`<repo>/data/<bench>`:

```
data/
├── datasets/
│   ├── TIR-Bench/                         # 1.7 GB
│   │   ├── TIR-Bench.json                 # ← read by tir_bench._load_dataset()
│   │   ├── data/                          # raw images per task family
│   │   └── README.md
│   └── VisualToolBench/                   # 7.3 GB (optional local mirror; loader streams from HF)
│       ├── test.parquet
│       └── manifest.json
├── Video-Holmes/                          # 4.1 GB
│   ├── README.md
│   └── Benchmark/                         # ← required wrapper expected by video_holmes._benchmark_dir()
│       ├── test_Video-Holmes.json
│       ├── train_Video-Holmes.json
│       ├── annotations/<vid>.json         # 270 files
│       ├── annotation_training/<vid>.json # 246 files
│       └── videos/videos_cropped/<vid>.mp4 # 503 clips
└── SIV-Bench/                             # 42 GB
    ├── SIV-Bench-QA.tsv                   # 8 728 rows
    ├── origin/<category>/<vid>.mp4        # canonical clip set
    ├── w_sub/<category>/<vid>.mp4         # subtitles burnt-in
    └── wo_sub/<category>/<vid>.mp4        # subtitles removed
```

The repo's `data/` directory is gitignored, and you can keep the actual
data on a faster scratch disk and symlink it in:

```bash
cd /path/to/Multi-hop-Reasoning-VLM-Agent
ln -s /scratch/visual-reasoning-data data        # or any path that holds the four sub-dirs
```

### Download (HuggingFace)

Three of the four datasets are public; Video-Holmes is gated and needs an
HF token (after clicking *Agree and access repository* once on
<https://huggingface.co/datasets/TencentARC/Video-Holmes>).

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1                 # 5–10× faster on fast networks

cd /path/to/Multi-hop-Reasoning-VLM-Agent
mkdir -p data/datasets
pip install -U "huggingface_hub[cli]"

# image benchmarks (small, public)
hf download Agents-X/TIR-Bench         --repo-type dataset --local-dir data/datasets/TIR-Bench           # ~1.2 GB
hf download ScaleAI/VisualToolBench    --repo-type dataset --local-dir data/datasets/VisualToolBench     # ~7.3 GB (optional mirror)

# video benchmarks (large)
hf download Fancylalala/SIV-Bench      --repo-type dataset --local-dir data/SIV-Bench                    # ~45 GB
hf download TencentARC/Video-Holmes    --repo-type dataset --local-dir data/Video-Holmes                 # ~4 GB (gated)
```

After Video-Holmes finishes, the HF snapshot lays it out flat. Move the
five entries into the `Benchmark/` subdir the loader expects:

```bash
cd data/Video-Holmes
mkdir -p Benchmark/videos
mv test_Video-Holmes.json train_Video-Holmes.json annotations annotation_training Benchmark/
mv videos_cropped Benchmark/videos/
```

If `hf download` aborts mid-stream with `brotli: decoder process called
with data when 'can_accept_more_data()' is False`, the bundled
`brotlicffi` decoder is incompatible with `httpx`. Swap it for
`brotli`:

```bash
pip uninstall -y brotlicffi && pip install --no-cache-dir brotli
```

`hf_transfer` partial files are kept on disk, so the retry resumes
where it stopped.

If `hf download` fails with **permission denied** under a shared default
cache, point Hugging Face at a repo-local cache:

```bash
export HF_HOME="$(pwd)/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_XET=1
```

### Smoke test (no API call)

```bash
conda activate game-ai-agent
PYTHONPATH=/path/to/Multi-hop-Reasoning-VLM-Agent python - <<'PY'
from visual_reasoning_wrapper.benchmarks import (
    iter_tir_bench_samples, iter_visual_toolbench_samples,
    iter_video_holmes_samples, iter_siv_bench_samples,
)
from visual_reasoning_wrapper.benchmarks.video_holmes import sample_video_frames

t = next(iter_tir_bench_samples(limit=1)); print("TIR-Bench:", t.task, t.sample_id)
v = next(iter_visual_toolbench_samples(limit=1)); print("VisualToolBench:", v.sample_id, v.turncase)
h = next(iter_video_holmes_samples(split="test", limit=1)); print("Video-Holmes:", h.video_id, h.question_type)
s = next(iter_siv_bench_samples(limit=1)); print("SIV-Bench:", s.video_id, s.dimension)

frames, fps, meta = sample_video_frames(h.video_path, num_frames=4)
print(f"  decord backend: {meta['backend']}, fps={fps:.1f}, decoded={len(frames)} PIL frames")
PY
```

Expected output (paths abbreviated):

```
TIR-Bench: refcoco 1
VisualToolBench: 68658a711603983919432621 single-turn
Video-Holmes: fH6bbNJJfqk SR
SIV-Bench: boss-employee/video_141 Environment Perception
  decord backend: decord, fps=24.0, decoded=4 PIL frames
```

### Optional: `vlm_benchmarks` env for in-process grounding heads

If you want to run the **offline grounding labeling pipeline** (Florence-2
captioning, GroundingDINO, OmniParser-v2 icon detection, EasyOCR text
spotting, YOLO bounding boxes) **inside the same Python process** instead
of via a remote VLM API, install the heavy `vlm_benchmarks` env:

```bash
conda env create -f install/vlm_benchmarks.environment.yml
conda activate vlm_benchmarks
python -m playwright install chromium     # optional; used by demo/labeling scripts
pip install -e .                           # makes vlm_wrapper importable
python install/vlm_benchmarks_smoke.py
```

This env pins `transformers >=4.51,<4.56` and `torch 2.4.1+cu121` —
incompatible with `game-ai-agent`'s `transformers 5.6.x` /
`torch 2.11+cu130`, which is why it's a separate env.

---

## 5. OmniParser-v2 weights (optional, used by Head 3)

OmniParser-v2 is lazy-loaded by `vlm_wrapper.grounding`. First call
downloads ~1.5 GB of weights to `~/.cache/omniparser-v2/` (override with
`OMNIPARSER_CACHE_DIR`). Already covered by the `ultralytics`,
`easyocr`, and `supervision` pins in the `vlm_benchmarks` env.

---

## 6. End-to-end verification

```bash
# Unified path: gym-v + visual-reasoning loaders both live in game-ai-agent
conda run -n game-ai-agent   python install/gymv_smoke.py
conda run -n game-ai-agent   python -c "
from visual_reasoning_wrapper.benchmarks import (
    iter_tir_bench_samples, iter_visual_toolbench_samples,
    iter_video_holmes_samples, iter_siv_bench_samples,
)
print('TIR     ', next(iter_tir_bench_samples(limit=1)).task)
print('VTB     ', next(iter_visual_toolbench_samples(limit=1)).sample_id)
print('VHolmes ', next(iter_video_holmes_samples(limit=1)).video_id)
print('SIV     ', next(iter_siv_bench_samples(limit=1)).video_id)
"
conda run -n browsergym      python install/browsergym_smoke.py
conda run -n osworld         python install/osworld_smoke.py

# Optional grounding heads (only if you installed the heavy env)
conda run -n vlm_benchmarks  python install/vlm_benchmarks_smoke.py

# Standalone gymv env (only if you ran install_gymv.sh)
conda run -n gymv            python install/gymv_smoke.py
```

Each script prints one `[OK] / [FAIL] / [WARN]` line per check.

---

## 7. Mapping the install to the plan

| Plan item                                                                       | Status after this doc       |
|---------------------------------------------------------------------------------|-----------------------------|
| §3 escalation chain — `gymv`                                                    | Runtime ready (inside `game-ai-agent`; standalone `gymv` env still available) |
| §3 escalation chain — `image_qa`                                                | Runtime ready (`game-ai-agent`); TIR-Bench local mirror + VisualToolBench HF stream |
| §3 escalation chain — `video_qa`                                                | Runtime ready (`game-ai-agent`); Video-Holmes + SIV-Bench on disk, `decord` decode verified |
| §3 escalation chain — `browser`                                                 | Runtime ready (`browsergym` env) |
| §3 escalation chain — `desktop`                                                 | Runtime ready (`osworld` env); VM backend pulled separately |
| §3 grounding-head pipeline                                                      | Optional (`vlm_benchmarks` env) — only required for in-process Florence-2 / GroundingDINO / OmniParser-v2 / EasyOCR / YOLO |
| Phase 0 — Gym-V data collection                                                 | Unblocked                   |
| Phase 0 — BrowserGym data collection                                            | Unblocked                   |
| Phase 1 — image-QA SFT                                                          | Unblocked after HF cache / `datasets` smoke |
| Phase 1 — video-QA SFT                                                          | Unblocked                   |

---

## 8. Troubleshooting

### `gym-v` install fails with `gymnasium` not found

`gymnasium>=1.2.2` only exists on PyPI as of 2025-Q4. Upgrade pip
(`pip install -U pip`) and re-run inside the `gymv` env.

### BrowserGym pages render blank, or `playwright._impl._errors.Error: Host system is missing dependencies to run browsers`

System libraries are missing. Install them once with sudo:

```bash
conda activate browsergym
python -m playwright install-deps chromium     # apt-installs libnspr4, libnss3, libatk*, libcups2, etc.
python -m playwright install chromium
```

The unattended `install_browsergym.sh` script does both steps; this
trips up only manual installs.

### OSWorld's `pip install -r requirements.txt` drags in ~1.5 GB of wheels

That's expected — OSWorld bundles Office-document processors (pymupdf,
python-docx, python-pptx, openpyxl, formulas, odfpy) plus audio libs
(librosa, mutagen, pyacoustid) for the eval checkers. If you only need
the `DesktopEnv` runtime and don't plan to run the evaluator locally,
`pip install desktop-env` is enough (done automatically by
`install/osworld.environment.yml`).

### `huggingface_hub` 401 on Video-Holmes

The repo is gated. Set `HF_TOKEN` in the **same shell** that runs
`download.py`, and click "Agree and access repository" on
<https://huggingface.co/datasets/TencentARC/Video-Holmes>.

### `decord` import fails on Linux

Use the prebuilt wheel: `pip install decord==0.6.0`. If that still
fails, substitute with `pip install av` (the `sample_video_frames`
helper in `visual_reasoning_wrapper.benchmarks.video_holmes` falls
back to `pyav` and finally to `cv2`, all of which are installed in
`game-ai-agent`).

### `visual_reasoning_wrapper.benchmarks` loaders raise `FileNotFoundError`

Each loader resolves data via `default_*_root()` which returns
`<repo>/data/<bench>`. Either copy/symlink the data into
`<repo>/data/...`, or pass an explicit root:

```python
iter_video_holmes_samples(split="test", video_holmes_root="/scratch/Video-Holmes")
iter_siv_bench_samples(siv_root="/scratch/SIV-Bench")
```

Video-Holmes specifically expects a `Benchmark/` wrapper subdir;
the canonical layout is shown in §4.
