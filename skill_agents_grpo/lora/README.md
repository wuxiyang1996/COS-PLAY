# Multi-LoRA Skill Bank Agent (GRPO Edition)

## 3 GRPO-trained LoRA adapters

This module uses one shared Qwen3-8B backbone with **3 active LoRA adapters**, each trained via GRPO:

| Function      | What it does                                          | Adapter    | GRPO reward |
|--------------|-------------------------------------------------------|------------|-------------|
| **segment**  | Rank/label skill candidates for trajectory segments   | `segment`  | `SegmentationDiagnostics` (scorer rebuild + decode) |
| **contract** | Summarize skill effects and contracts                 | `contract` | `verify_effects_contract().overall_pass_rate` |
| **curator**  | Approve/veto/defer bank maintenance mutations         | `curator`  | `bank_quality_delta` |

Two functions from `skill_agents` are **not** GRPO-trained here:
- **boundary** — skipped because the LLM extracts predicates, not boundaries; reward is too indirect.
- **retrieval** — skipped because reward requires full environment rollouts; the existing decision-agent GRPO trainer already handles skill selection.

Their `SkillFunction` enum values are retained for backward compatibility.

Each adapter learns function-specific patterns without interference:

- **Efficiency** — the base model is loaded once; switching adapters is a lightweight `set_adapter()` call.
- **Independent training** — each adapter can be updated on its own data/schedule.
- **GRPO training** — `log_probs()` enables per-token gradient computation for policy-gradient updates on LoRA weights.

## Architecture

```
┌──────────────────────────────────────────┐
│        Qwen3-8B  (shared base)          │
│                                          │
│  ┌──────────┐  ┌──────────┐             │
│  │ segment   │  │ contract │             │
│  │  LoRA     │  │  LoRA    │             │
│  │  (GRPO)   │  │  (GRPO)  │             │
│  └──────────┘  └──────────┘             │
│  ┌──────────┐                            │
│  │ curator   │                           │
│  │  LoRA     │                           │
│  │  (GRPO)   │                           │
│  └──────────┘                            │
└──────────────────────────────────────────┘
          │
    SkillFunction enum selects
    which adapter is active
```

## Two modes of operation

### 1. Inference: `generate()`
Standard text generation under `torch.inference_mode()`. Used during EM pipeline rollouts.

### 2. GRPO training: `log_probs()`
Computes per-token log-probabilities with `torch.enable_grad()`. Used by `GRPOLoRATrainer` to compute policy-gradient loss and update LoRA adapter weights.

```python
# Inference (rollout phase)
text = llm.generate(SkillFunction.CONTRACT, prompt, temperature=0.7)

# Training (gradient phase)
lp = llm.log_probs(SkillFunction.CONTRACT, prompt, completion)  # (n_tokens,) with grad
loss = -lp.mean()  # simplified — real GRPO uses advantages + clipping
loss.backward()     # gradients flow into CONTRACT LoRA params only
```

## How adapter routing works

1. `SkillFunction` enum defines 3 active functions (`SEGMENT`, `CONTRACT`, `CURATOR`) plus 2 legacy values (`BOUNDARY`, `RETRIEVAL`).
2. `MultiLoraSkillBankLLM` loads the base model + configured adapters.
3. Calling `llm.generate(SkillFunction.CONTRACT, prompt)` does:
   - `model.set_adapter("contract")` — activates the correct LoRA
   - Runs inference on the shared backbone + contract LoRA
   - Returns the generated text
4. Existing call sites (`llm_teacher.py`, `llm_contract.py`, `llm_curator.py`) auto-discover the shared instance via `get_shared_instance()` and route to the correct adapter.
5. If no LoRA is configured, they fall back to the API-based `ask_model`.

## Quick start

### Inference with adapters

```python
from skill_agents_grpo.lora import MultiLoraSkillBankLLM, MultiLoraConfig, SkillFunction

cfg = MultiLoraConfig(
    base_model_name_or_path="Qwen/Qwen3-8B",
    adapter_paths={
        "segment":   "runs/lora_adapters/segment",
        "contract":  "runs/lora_adapters/contract",
        "curator":   "runs/lora_adapters/curator",
    },
)

llm = MultiLoraSkillBankLLM(cfg)
llm.load()

# Register so existing code auto-discovers it
MultiLoraSkillBankLLM.set_shared_instance(llm)

# Generate with a specific adapter
out = llm.generate(SkillFunction.CONTRACT, "Summarize effects of ...")
```

### With GRPO training

```python
from skill_agents_grpo.grpo import GRPOOrchestrator, GRPOConfig

orch = GRPOOrchestrator(llm=llm, config=GRPOConfig())

# Phase 1: enable wrappers → run EM pipeline → samples collected in buffer
orch.enable_wrappers(
    contract_holdout_instances=holdout,
    contract_verify_config=stage3_config,
)
run_em_pipeline(...)  # calls llm_summarize_contract() etc. — wrappers intercept

# Phase 2: train adapters from collected samples
stats = orch.train_step()

orch.disable_wrappers()
```

### From YAML config

```python
import yaml
from skill_agents_grpo.lora import MultiLoraSkillBankLLM, MultiLoraConfig

with open("configs/skillbank_lora.yaml") as f:
    raw = yaml.safe_load(f)

cfg = MultiLoraConfig.from_dict(raw["lora"])
llm = MultiLoraSkillBankLLM(cfg)
llm.load()
MultiLoraSkillBankLLM.set_shared_instance(llm)
```

### Backward compatibility (no LoRA)

If you don't set a shared instance, all LLM calls fall through to the existing `API_func.ask_model` with no changes needed.

## GRPO wrapper architecture

The GRPO system wraps existing LLM call points rather than building separate training loops:

| Stage | Wrapped function | Reward signal | Compute cost |
|-------|-----------------|---------------|-------------|
| 3 CONTRACT | `llm_summarize_contract()` | `verify_effects_contract().overall_pass_rate` | ~820ms/skill |
| 4 CURATOR | `filter_candidates()` | `bank_quality_delta` | ~3s/EM iter |
| 2 SEGMENT | `collect_segment_preferences()` | `SegmentationDiagnostics` | ~12.5s/episode |

Each wrapper:
1. Generates G samples at higher temperature (rollout phase, inference mode)
2. Evaluates each with a CPU-only reward function
3. Stores (prompt, completions, rewards) in `GRPOBuffer`
4. Returns the best sample to the EM pipeline (unchanged behavior)
5. After the EM step, `GRPOLoRATrainer` recomputes log-probs with gradients and updates LoRA weights

## File layout

```
skill_agents_grpo/lora/
├── __init__.py          # Exports: SkillFunction, MultiLoraConfig, MultiLoraSkillBankLLM
├── skill_function.py    # SkillFunction enum (3 active + 2 legacy)
├── config.py            # MultiLoraConfig, LoraTrainingConfig (Qwen3-8B defaults)
├── model.py             # MultiLoraSkillBankLLM (generate + log_probs)
└── README.md            # This file

skill_agents_grpo/grpo/
├── __init__.py          # Exports: GRPOBuffer, GRPOCallWrapper, GRPOLoRATrainer, GRPOConfig
├── buffer.py            # GRPOSample + GRPOBuffer (partitioned by adapter)
├── wrapper.py           # GRPOCallWrapper (generic function wrapper)
├── trainer.py           # GRPOLoRATrainer (advantage computation + gradient updates)
├── rewards.py           # contract_reward, curator_reward, segmentation_reward
├── config.py            # GRPOConfig + StageGRPOConfig
└── orchestrator.py      # GRPOOrchestrator (top-level enable/disable/train)
```

## Tests

```bash
cd Game-AI-Agent
python -m pytest tests/test_lora_dispatch.py -v
```
