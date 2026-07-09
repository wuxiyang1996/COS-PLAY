# vLLM split-base topology — 9B actor + 35B-A3B grounding tower

> **Status:** S2 closure for **T2.8** (pre-training-readiness-audit §0.1).
> **Default deployment:** option (ii) — `schema_gen` is **offline-only**;
> the trainer's `VLLMServerManager` runs a single base (`Qwen/Qwen3.5-9B`)
> across 4 GPUs.
> **First written:** 2026-05-02.
> **Last reviewed:** 2026-05-02.
> **Owner:** trainer / deployment.
> **Cross-refs:**
> [`trainer/coevolution/vllm_server.py`](../../trainer/coevolution/vllm_server.py),
> [`inference/serve_qwen35_35b_a3b.sh`](../../inference/serve_qwen35_35b_a3b.sh),
> [`trainer/coevolution/config.py`](../../trainer/coevolution/config.py)
> (`model_name`, `inference_only_model`, `vllm_gpu_ids`),
> [`pre-training-readiness-audit.md` §0.1 T2.8](../pre-training-readiness-audit.md),
> [`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md` §13](../../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md).

---

## 1. The constraint

vLLM bakes the base model into the worker process at start time. A single
worker can hot-swap LoRA adapters via `/v1/load_lora_adapter`, but **cannot**
swap to a different base mid-run. We have two bases in play:

| Base | Used by | Footprint (bf16) | Where the LoRA lives |
|---|---|---|---|
| `Qwen/Qwen3.5-9B` (dense, 32 layers, 8.7 B params) | actor + skill bank | ~18 GB weights | `runs/sft_coldstart/{decision,skillbank}/<adapter>/` |
| `Qwen/Qwen3.5-35B-A3B` (MoE, 35 B total / ~3 B active) | `schema_gen` (visual grounding control plane) | ~70 GB weights | `runs/sft_schema_gen/schema_gen_<ts>/adapter/` |

Therefore the deployment **always** uses *two* vLLM servers — never one. The
choice is whether the second one runs concurrently with training.

## 2. Default topology (training time)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  H200 host (8 GPUs, 141 GB each)                                        │
├──────────────────────────────────┬──────────────────────────────────────┤
│  GPUs 0-3 — vLLM (actor)         │  GPUs 4-7 — FSDP (training)          │
│  base = Qwen/Qwen3.5-9B          │  base = Qwen/Qwen3.5-9B + LoRA       │
│  TP = 1 each (4 instances)       │  GRPO inner loop (action_taking,     │
│  ports 8000-8003                 │  skill_selection, segment, contract, │
│  hot-reloads after each step     │  curator)                            │
│  ~18 GB weights, ~115 GB KV      │                                      │
├──────────────────────────────────┴──────────────────────────────────────┤
│  GPU set unused at training time:                                       │
│  schema_gen 35B-A3B grounding LoRA — see §3 (offline-only by default)   │
└─────────────────────────────────────────────────────────────────────────┘
```

Wired in [`trainer/coevolution/config.py`](../../trainer/coevolution/config.py):

```python
@dataclass
class CoEvolutionConfig:
    model_name: str = "Qwen/Qwen3.5-9B"          # the *trained* base
    vllm_gpu_ids: List[int] = [0, 1, 2, 3]       # 9B actor, TP=1 each
    inference_only_model: Optional[str] = "Qwen/Qwen3.5-35B-A3B"
        # informational; the trainer does NOT auto-start a 35B-A3B server.
```

The four 9B vLLM workers share an HF cache, listen on `8000-8003`, and are
managed by `VLLMServerManager`. After every GRPO step the trainer writes the
updated adapter to disk and calls `manager.reload_adapters({...})` — each
worker hot-swaps its in-memory LoRA without restart.

## 3. `schema_gen` — offline-only by default

The `schema_gen` adapter targets `Qwen3.5-35B-A3B`. With the 9B actor
occupying GPUs 0-3 and FSDP training occupying GPUs 4-7, **no GPUs are free
during training** to host the 35B-A3B base. Three deployment options:

### Option (ii) — offline-only (recommended; default)

`schema_gen` is invoked **outside** the GRPO loop:

* **Cold-start ingest** (`labeling/build_schema_gen_triples.py`,
  `labeling_supplement/decide_promotion_gpt54.py`): the script starts
  [`inference/serve_qwen35_35b_a3b.sh`](../../inference/serve_qwen35_35b_a3b.sh)
  on a free machine (or kills the trainer first), runs grounding once over
  the corpus, persists results to disk, then tears the server down.
* **Post-training eval** (T1.1′ exact-match probe, T2.3 E0 driver): same —
  kill the trainer, free all 8 H200s, launch with `TENSOR_PARALLEL=8` for
  ~2× throughput.
* **Live actor**: never. The actor consumes the *cached* schema output as
  a regular tool call, never round-trips through 35B-A3B at training time.

Pros: zero training-time vLLM contention, no extra GPUs reserved, both
bases get TP they need.
Cons: schema must be pre-cached for every cold-start episode (it is —
`labeling/output/grounding/{gymv,env_wrappers,...}/`), and any
schema-fresh evaluation requires a deploy / redeploy step.

### Option (i) — co-resident (not recommended; reserved future capacity)

Reserve some GPUs for a second `VLLMServerManager` running `Qwen3.5-35B-A3B`:

```python
# Hypothetical CoEvolutionConfig extension; NOT implemented today.
inference_only_model_gpu_ids: List[int] = [4, 5, 6, 7]   # carve from FSDP
inference_only_tp: int = 4                                # TP=4 ZeRO-3
```

Required changes (none of these landed; deferred):

1. New `VLLMServerManager` instance for the 35B base; `start()` /
   `reload_adapters()` parallel to the 9B manager.
2. Reduce `vllm_gpu_ids` and FSDP gpu set; rebalance batch sizes.
3. Pass through the schema-gen adapter path on every reload.
4. Network endpoint indirection (`schema_gen_endpoint_url`) the actor
   calls instead of the cached schema output.

Cost: 4 H200s permanently bound to the 35B base (TP=4 fits ~18 GB/GPU
weights + ~110 GB KV); halves FSDP capacity. Unblocks live re-grounding
mid-episode but doubles deployment complexity. Not justified for the
fast-loop bring-up.

### Option (iii) — host-on-demand (debug only)

The trainer pauses, `inference/serve_qwen35_35b_a3b.sh` spins up on the
training GPU set briefly for one re-grounding pass, gets killed, training
resumes. Acceptable for debug; unacceptable for production (vLLM cold start
~2-3 min on 35B-A3B).

## 4. Why the two bases can't share a worker

| Reason | Implication |
|---|---|
| vLLM loads base weights at process start; the underlying class is decided then. | A worker started with `--model Qwen/Qwen3.5-9B` cannot accept a 35B-A3B request. |
| `Qwen3_5MoeForConditionalGeneration` (35B-A3B) and `Qwen3_5ForCausalLM` (9B) have different layer types and different LoRA `target_modules` (see [`trainer/SFT/lora_targets.py`](../../trainer/SFT/lora_targets.py) — T2.11). | A LoRA trained for 9B *will not* apply to a 35B-A3B base; the projection-name keys don't match. |
| MoE expert-parallel (`--enable-expert-parallel`) needs `TENSOR_PARALLEL ≥ 2`. | The trainer's TP=1 / one-instance-per-GPU layout is incompatible with 35B-A3B; it would need TP=4 or TP=8. |

## 5. Hot-reload contract (training time)

After each GRPO step, the trainer:

1. Writes new LoRA weights to `adapter_dir/<adapter_name>/`.
2. Calls `manager.reload_adapters({"action_taking": <path>,
   "skill_selection": <path>, ...})`.
3. Each of the 4 vLLM instances issues
   `POST /v1/load_lora_adapter` with `name=<adapter_name>` +
   `lora_path=<path>` (overwriting the named slot).
4. Returns within ~100 ms; no process restart, no KV cache invalidation
   (vLLM reuses cache across LoRA swaps for the same base).

**This contract only applies to LoRAs that target `model_name`.** A LoRA
targeting `inference_only_model` (i.e., `schema_gen`) cannot be hot-loaded
into the trainer's manager and must be served by a separate process — see
§3.

## 6. Operational checklist

| Task | Procedure |
|---|---|
| Start training | `scripts/run_coevolution.py` — auto-starts the 9B `VLLMServerManager` on `vllm_gpu_ids`. The 35B-A3B server is **not** auto-started. |
| Run schema-gen probe (T1.1′) on a fresh checkpoint | `kill` the trainer; `bash inference/serve_qwen35_35b_a3b.sh`; run `python evaluation/probe_schema_gen_exact_match.py`; `Ctrl-C` the server; restart trainer. |
| Run E0 release scoreboard (T2.3) | Same as schema-gen probe. The driver assumes the actor is offline and the 35B-A3B server is the sole vLLM endpoint. |
| Live actor inference (post-launch) | The 9B `VLLMServerManager` continues to run after training stops; point eval clients at `http://localhost:8000-8003/v1`. |
| Side-by-side teacher comparison | Start `serve_qwen35_35b_a3b.sh` on a separate host; set `VLLM_BASE_URL` in the eval driver to the teacher's endpoint. |

## 7. Implications for the audit's open items

* **T1.1′** — exact-match probe needs the 35B-A3B server. **Run after
  training completes**, not during. Driver knows this (no auto-start
  code path).
* **T2.3** — release scoreboard runs against the same 35B-A3B server, then
  emits `releases/<release_id>/scoreboard.md`. Driver reuses
  `serve_qwen35_35b_a3b.sh`.
* **Phase-2 SFT** — when/if `schema_gen` retraining is triggered (NORTHSTAR
  §5.4), it runs through `trainer/SFT/schema_gen/run_schema_gen.sh` against
  the 35B-A3B base on all 8 GPUs (DeepSpeed ZeRO-3) — the trainer is killed
  during this window. Same operational pattern as cold-start ingest.

## 8. Future work — promotion to option (i)

The code path is structurally simple (a second `VLLMServerManager` on a
different GPU set). It is deferred because:

1. The fast-loop bring-up only needs the actor + cached schema; live
   re-grounding is not on the critical path.
2. Halving the FSDP GPU set increases per-step cost; effective batch must
   be retuned — burns iteration time.
3. The 35B-A3B base needs ≥4 GPUs at TP=4 (or 8 at TP=8). 4 GPUs at TP=2
   doesn't fit. So option (i) means dedicating 4 of 8 H200s.

When it is needed (multi-domain transfer with frequent re-grounding, or a
Phase-2 SFT loop that reads live grounding outputs), the work is roughly
half a day:

* Add `inference_only_model_*` knobs to `CoEvolutionConfig`.
* Instantiate a second `VLLMServerManager` from the orchestrator if those
  knobs are set.
* Add `schema_gen_endpoint_url` field on the actor's tool-call shim.
* Document the GPU split in this file (§3 option (i) gets promoted).

This memo will be revised at that time.
