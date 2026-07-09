# `visual_reasoning_wrapper.benchmarks` — image & video QA loaders

Streaming loaders and `parse_*_sample` helpers for four public benchmarks used with the `image_qa` / `video_qa` tool loops (`vlm_wrapper.tool_loop`). Each module returns a uniform result dict (schema, answer, tool trace, metadata) after running the configured vision model through `cascaded_ground` / parsers.

**Legacy note:** CLEVR and GQA loaders were removed in favour of **VisualToolBench** and **TIR-Bench** (tool-centric and thinking-with-images image benchmarks on HuggingFace).

**Design matrix** — see [`../__init__.py`](../__init__.py) (`PRIMARY_VISUAL_REASONING_BENCHMARKS`).

## Benchmarks

| Key | Modality | Module | Upstream links | Local source |
|-----|----------|--------|----------------|--------------|
| `visual_toolbench` | Image | `visual_toolbench.py` | [HF dataset](https://huggingface.co/datasets/ScaleAI/VisualToolBench), [paper](https://arxiv.org/abs/2510.12712) | HF cache or optional `data/datasets/VisualToolBench/` mirror |
| `tir_bench` | Image | `tir_bench.py` | [repo](https://github.com/agents-x-project/TIR-Bench), [HF dataset](https://huggingface.co/datasets/Agents-X/TIR-Bench), [paper](https://arxiv.org/abs/2511.01833) | HF cache or optional `data/datasets/TIR-Bench/` mirror |
| `video_holmes` | Video | `video_holmes.py` | [repo](https://github.com/TencentARC/Video-Holmes), [project page](https://video-holmes.github.io/Page.github.io/), [HF dataset](https://huggingface.co/datasets/TencentARC/Video-Holmes), [paper](https://arxiv.org/abs/2505.21374) | `data/Video-Holmes/Benchmark/` |
| `siv_bench` | Video | `siv_bench.py` | [project page](https://kfq20.github.io/sivbench/), [HF dataset](https://huggingface.co/datasets/Fancylalala/SIV-Bench), [paper](https://arxiv.org/abs/2506.05425) | `data/SIV-Bench/` (see `default_siv_bench_root()` in code) |

These are the covered benchmarks for the visual-reasoning wrapper:

* **Image:** VisualToolBench + TIR-Bench, because both evaluate tool-enabled /
  thinking-with-images behavior rather than only static visual QA.
* **Video:** Video-Holmes + SIV-Bench, because the pair covers multi-hop clue
  localization plus social-state / intent / interaction reasoning.

## Download HF-hosted benchmarks

Image loaders call `datasets.load_dataset(...)` and use the HuggingFace **cache** by default (`~/.cache/huggingface/`). SIV-Bench is also hosted on HF, but the video loader expects its TSV and clips under `data/SIV-Bench/`. To keep a **full on-disk copy** under the repo (ignored by git — see root `.gitignore` `data/`):

```bash
cd /fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent
mkdir -p data/datasets

# Option A — huggingface-cli (recommended; resumable)
pip install -U "huggingface_hub[cli]"
hf download Agents-X/TIR-Bench         --repo-type dataset --local-dir data/datasets/TIR-Bench
hf download ScaleAI/VisualToolBench    --repo-type dataset --local-dir data/datasets/VisualToolBench
hf download Fancylalala/SIV-Bench      --repo-type dataset --local-dir data/SIV-Bench

# Option B — Python (same result)
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
root = Path("data/datasets")
root.mkdir(parents=True, exist_ok=True)
snapshot_download("Agents-X/TIR-Bench", repo_type="dataset", local_dir=str(root / "TIR-Bench"))
snapshot_download("ScaleAI/VisualToolBench", repo_type="dataset", local_dir=str(root / "VisualToolBench"))
siv_root = Path("data/SIV-Bench")
snapshot_download("Fancylalala/SIV-Bench", repo_type="dataset", local_dir=str(siv_root))
print("Saved image datasets to", root.resolve(), "and SIV-Bench to", siv_root.resolve())
PY
```

Set `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` if your account must authenticate for any dataset.

**Cluster tip:** if downloads fail with permission errors under a global HF path, force a writable cache in the repo (also gitignored):

```bash
cd /path/to/Multi-hop-Reasoning-VLM-Agent
export HF_HOME="$(pwd)/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_XET=1   # optional: avoids xet log/tmp issues on some NFS mounts
```

Rough sizes: TIR-Bench ~1.2 GB; VisualToolBench ~7.5 GB; SIV-Bench ~45 GB — ensure disk space before downloading.

## Python API

```python
from visual_reasoning_wrapper.benchmarks import (
    iter_visual_toolbench_samples,
    parse_visual_toolbench_sample,
    iter_tir_bench_samples,
    parse_tir_bench_sample,
    iter_video_holmes_samples,
    parse_video_holmes_sample,
    iter_siv_bench_samples,
    parse_siv_bench_sample,
)
```

Docstrings in `__init__.py` list exports and dependencies (`Pillow`, `datasets` for image HF rows; `decord` or `opencv-python` for video).

## Reasoning-tool integration

Each `parse_*_sample` now wires the question through
`visual_reasoning_wrapper.question_router.classify_question` before
calling `cascaded_ground`. The router classifies the prompt into one
or more *question classes* — `count`, `ratio`, `compare`, `spatial`,
`ocr`, `identity`, `temporal`, `social`, `verify`, plus the always-on
`answer` — and maps each class onto the reasoning tools the model
must call before emitting `<answer>`:

| Class | Required reasoning tool | Derivation `kind=` |
|-------|-------------------------|---------------------|
| `count`   | `count_value`     | `COUNT`   |
| `ratio`   | `compute_ratio`   | `RATIO`   |
| `compare` | `compare_values`  | `COMPARE` |
| `verify`  | `verify_claim`    | `VERIFY`  |
| `answer`  | `verify_claim`    | `VERIFY`  |

The router also surfaces an "observation tools to ground inputs"
suggestion so the prompt nudges the VLM toward the right perception
primitive (e.g. `read_text_region` for OCR questions,
`measure_distance` for spatial comparisons).

The selected tools, classes, and derivation kinds flow into the
returned schema as:

* a `<derivations>` block whose `dN.kind=` matches one of
  `COUNT | RATIO | COMPARE | VERIFY` (added automatically by
  `vlm_wrapper.ground._ensure_derivations_block` if the VLM forgets
  it),
* `<answer> evidence_chain=[hop1,d1,d2,…]` referencing the typed
  derivation rows,
* `result["sample"]["meta_data"]` plus the per-sample `context`
  recording `question_classes`, `required_reasoning_tools`, and
  `derivation_kinds` for downstream analysis.

`vlm_wrapper.schema.semantic_validate` cross-checks that every `dN`
cited in `evidence_chain=` exists in `<derivations>`; a missing block
when the answer references one is a hard error and triggers
escalation.

## CLI evaluation

Batch scoring and JSONL logging: [`../../vlm_wrapper/eval/run_eval.py`](../../vlm_wrapper/eval/run_eval.py) (see [`../../vlm_wrapper/eval/README.md`](../../vlm_wrapper/eval/README.md)).

## Environment

Install and paths: parent [`../README.md`](../README.md) and [`../../install/INSTALL_BENCHMARKS.md`](../../install/INSTALL_BENCHMARKS.md).
