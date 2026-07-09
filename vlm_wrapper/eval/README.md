# `vlm_wrapper.eval` — batch harness & metrics

## Role

- **`harness.py`** — `run_eval()` iterates a sample stream, calls your **grounder** (usually `parse_*_sample` from `visual_reasoning_wrapper.benchmarks`), records per-sample results, optional JSONL, and aggregates [`metrics`](metrics.py) (answer accuracy, format compliance, entity IoU when gold boxes exist, tool-related stats).
- **`run_eval.py`** — CLI front-end: picks a benchmark via `--benchmark`, builds the iterator + grounder, prints a JSON summary and optional report file.
- **`metrics.py`** — Pure helpers used by the harness; safe to import for custom eval scripts.

## CLI

Run from the **Multi-hop-Reasoning-VLM-Agent** repo root with `vlm_benchmarks` (or equivalent) activated:

```bash
python -m vlm_wrapper.eval.run_eval \
  --benchmark tir_bench \
  --limit 200 \
  --model gpt-4o \
  --output runs/eval/tir_bench.jsonl
```

Supported `--benchmark` values: `visual_toolbench`, `tir_bench`, `video_holmes`, `siv_bench`.

Common flags:

| Flag | Purpose |
|------|---------|
| `--split` | Video splits (`test` / `train`). TIR-Bench is `test` only (other values are ignored with a warning). |
| `--limit` | Cap number of samples |
| `--model` | Vision model id (defaults to `VLM_LABEL_MODEL` or `gpt-4o`) |
| `--api_key` / `--base_url` | API credentials (defaults from env) |
| `--num_frames` | Video benchmarks only — frames fed to the grounder |
| `--subtitle` | SIV-Bench condition (default `origin`) |
| `--output` | Per-sample JSONL path |
| `--report` | Aggregated metrics JSON (defaults to `<output>.report.json` if `--output` set) |
| `--max_entities` / `--max_rounds` | Tool-loop caps passed through to parsers |

## Programmatic use

```python
from vlm_wrapper.eval.harness import run_eval
from visual_reasoning_wrapper.benchmarks.tir_bench import (
    iter_tir_bench_samples,
    parse_tir_bench_sample,
)

def grounder(sample):
    return parse_tir_bench_sample(sample, model="gpt-4o", api_key="...")

report = run_eval(
    samples=iter_tir_bench_samples(split="test", limit=50),
    grounder=grounder,
    domain="image_qa",
    gold_extractor=lambda s: {"answer": s.answer},
    sample_id_fn=lambda s: s.sample_id,
)
print(report.metrics.to_dict())
```

## Related docs

- Benchmark details: [`../benchmarks/README.md`](../benchmarks/README.md)
- Full pipeline and env setup: [`../README.md`](../README.md)
