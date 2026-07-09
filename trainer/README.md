# Training Infrastructure

## Quick Start — Co-Evolution Training

The primary training path is the **async co-evolution loop** in `trainer/coevolution/`. It runs two agents in alternating phases with cross-system overlap, GRPO training on 5 LoRA adapters, and full W&B logging.

```bash
# 1. Start vLLM server (GPUs 0-3, 5 LoRA adapters)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/launch_vllm_coevolution.sh

# 2. Run co-evolution (GRPO on GPUs 4-7)
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
python scripts/run_coevolution.py \
    --total-steps 100 \
    --episodes-per-game 8 \
    --checkpoint-interval 5 \
    --wandb-project game-ai-coevolution

# Quick test (3 games, no GRPO, no W&B)
python scripts/run_coevolution.py \
    --games tetris twenty_forty_eight sokoban \
    --total-steps 3 --no-grpo --no-wandb

# Resume from checkpoint
python scripts/run_coevolution.py --resume
```

---

## Repo Layout

```
trainer/
  README.md
  coevolution/                       ← PRIMARY: async co-evolution loop
    __init__.py
    config.py                        # CoEvolutionConfig: games, GPUs, checkpointing, W&B
    vllm_client.py                   # AsyncVLLMClient: async wrapper for vLLM multi-LoRA API
    episode_runner.py                # run_episode_async(): async port of run_episode()
    rollout_collector.py             # collect_rollouts(): LPT scheduling + semaphore
    skillbank_pipeline.py            # AsyncSkillBankPipeline: async Stage 1-4 wrapper
    grpo_training.py                 # DecisionGRPOTrainer + SkillBankGRPOTrainer
    checkpoint.py                    # save/load/find checkpoints (bank + 5 adapters)
    orchestrator.py                  # co_evolution_loop(): Phase A+B+C main loop

  common/                            ← DEPRECATED (used by legacy modules below)
    configs/
      decision_grpo.yaml             # GRPO hyperparams, costs, shaping weights
      skillbank_em.yaml              # EM hyperparams, gating thresholds
    logging.py                       # TrainLogger (W&B wrapper)
    eval_harness.py                  # Fixed-seed evaluation harness
    seeds.py                         # Deterministic seed management
    metrics.py                       # RolloutRecord schema, metric aggregation

  decision/                          ← DEPRECATED (replaced by coevolution/)
    env_wrapper.py                   # EnvWrapper: retrieval-as-action
    policy_interface.py              # PolicyInterface: logprob extraction
    reward_shaping.py                # Reward shaping with tool-call reward
    rollout_collector.py             # Synchronous rollout collection
    grpo_trainer.py                  # GRPO training loop + GameAITrainer (VERL)
    replay_buffer.py                 # Episode replay buffer
    launch_train.py                  # CLI entry point (standalone / VERL)
    coevolution_callback.py          # SkillBank co-evolution callback (VERL)

  skillbank/                         ← DEPRECATED (replaced by coevolution/)
    ingest_rollouts.py               # Convert rollouts → trajectory objects
    em_trainer.py                    # Hard-EM loop driver
    stages/                          # Stage 0-4 pipeline
    learners/                        # Optional supervised learners
    bank_io/                         # VersionedBankStore, indices, diff logger
    lora/                            # LoRA SFT training scripts (still used by tests)

  launch_coevolution.py              ← DEPRECATED (replaced by coevolution/orchestrator.py)
```

### Deprecation notes

The `common/`, `decision/`, `skillbank/`, and `launch_coevolution.py` modules are the **legacy synchronous training path**. They are kept for backward compatibility because:

- `cold_start/load_rollouts.py` imports `trainer.common.metrics.RolloutRecord`
- `tests/test_lora_dispatch.py` imports `trainer.skillbank.lora.data_builder`
- `scripts/skillbank_agent_train.sh` imports from `trainer.skillbank.lora`
- `install_game_ai_agent_env.sh` verification checks reference them

The new `trainer/coevolution/` package is **completely self-contained** and does not import from any of these legacy modules. All new development should use `trainer/coevolution/`.

---

## Co-Evolution Architecture

Two agents share one Qwen3-8B base model served through a single vLLM instance with **5 LoRA adapters** loaded simultaneously:

| Adapter | Agent | Purpose | GRPO reward signal |
|---------|-------|---------|-------------------|
| `action_taking` | Decision | Choose game action | Step reward from env |
| `skill_selection` | Decision | Pick skill from bank | Step reward from env |
| `segment` | Skill Bank | Assign skill labels | Contract pass rate + follow score |
| `contract` | Skill Bank | Learn effect contracts | Holdout verification pass rate |
| `curator` | Skill Bank | Refine/merge/split skills | Bank quality delta |

**Not GRPO-trained:** `boundary` (base model, reward too indirect), `retrieval` (legacy/planned).

### The Loop

```
Step 0 (cold start):
    Bank = empty
    Decision agent collects rollouts WITHOUT skill selection
    (action_taking LoRA only, no skill_selection calls)

Step 1:
    Skill bank processes Step 0 rollouts → Bank_v1

Step 2:
    Decision agent collects rollouts WITH skill selection using Bank_v1
    (both skill_selection + action_taking LoRAs active)

Step 3:
    Skill bank processes Step 2 rollouts → Bank_v2
    GRPO updates all 5 LoRAs

    ... repeat Step 2-3 ...
```

### Three Phases per Step

```
Phase A + B (overlapped):
  ┌───────────────────────────────────────────────────────────┐
  │  collect_rollouts()                                        │
  │  ├── LPT schedule: super_mario → pokemon → ... → candy    │
  │  ├── asyncio.Semaphore(40) caps concurrency                │
  │  ├── run_episode_async() × 64 coroutines                   │
  │  │   ├── summary_state    (deterministic, 0 calls)         │
  │  │   ├── R1: summary_prose ║ skill_selection (parallel)    │
  │  │   ├── R2: subgoal + action  (action_taking, merged)     │
  │  │   └── env.step()       (ThreadPoolExecutor)             │
  │  └── on_episode_done → asyncio.Queue                       │
  │                              │                             │
  │  skill_bank_consumer()  ◄────┘  cross-system overlap       │
  │  └── micro-batch → Stage 1+2 (ThreadPoolExecutor)         │
  └───────────────────────────────────────────────────────────┘
                              │
Phase B finalize:             ▼
  ┌───────────────────────────────────────────────────────────┐
  │  sb_pipeline.finalize_update()                             │
  │  ├── Stage 3: contract learning  (contract LoRA)           │
  │  ├── Stage 4: bank maintenance   (curator LoRA)            │
  │  └── Proto-skill materialization                           │
  └───────────────────────────────────────────────────────────┘
                              │
Phase C (parallel on GPUs 4-7):
  ┌──────────────────────┐  ┌──────────────────────┐
  │ Decision GRPO        │  │ Skill Bank GRPO      │
  │ GPUs 4-5             │  │ GPUs 6-7             │
  │ • skill_selection    │  │ • segment            │
  │ • action_taking      │  │ • contract           │
  │                      │  │ • curator             │
  └──────────────────────┘  └──────────────────────┘
```

### GPU Allocation (8 GPUs)

| GPUs | Role | What runs |
|------|------|-----------|
| 0-3 | Inference | vLLM server (TP=4), 5 LoRAs, prefix caching, chunked prefill |
| 4-5 | Training | Decision agent GRPO (skill_selection + action_taking) |
| 6-7 | Training | Skill bank GRPO (segment + contract + curator) |

Inference and training never compete for the same GPUs — vLLM serves continuously during Phase A+B while GRPO runs on separate devices in Phase C.

---

## Key Features

### LPT Scheduling (`rollout_collector.py`)

Games are sorted by descending duration and interleaved round-robin:
- Longest games (super_mario 500 steps) start first
- Shortest games (candy_crush 50 steps) finish early → feed skill bank pipeline while long games run
- Maximizes vLLM GPU utilization and enables cross-system overlap

### Cross-System Overlap (`orchestrator.py`)

As short-game episodes complete, their trajectories immediately enter the skill bank pipeline via `asyncio.Queue`. By the time super_mario finishes, 6/8 games are already through Stage 1+2. Effective Phase B overhead: ~30s (instead of ~4 min serial).

### Cold-Start Handling (`episode_runner.py`)

Step 0 passes `skill_bank=None` → `get_top_k_skill_candidates()` returns `[]` → no `skill_selection` LoRA call → only `action_taking` fires. GRPO records contain `action_taking` data only (no `skill_selection` samples). Subsequent steps automatically enable full skill selection when the bank becomes populated.

### Merged Subgoal + Action Call (`episode_runner.py`)

Intention (subgoal) and action selection are merged into a single LLM call,
reducing serial rounds from 3 to 2 per game step (~33% faster Phase A):

```
BEFORE (3 serial rounds):          AFTER (2 serial rounds):
  R1: summary ║ skill (parallel)     R1: summary ║ skill (parallel)
  R2: intention (base, 40 tok)       R2: subgoal+action (action_taking, 256 tok)
  R3: action (action_taking, 512)    env.step()
  env.step()
```

The model outputs `SUBGOAL: [TAG] phrase\nREASONING: ...\nACTION: N` in one
generation. The subgoal is parsed out and tracked identically to before.

### Token Budget Tuning (`episode_runner.py`)

`max_tokens` for each LLM call is set based on measured output lengths across all 8 games
(~37,776 steps from 61 cold-start episodes per game):

| Call | `max_tokens` | Actual p99 | Actual max | Headroom |
|------|-------------|-----------|-----------|----------|
| `summary_prose` (base) | 25 | ~15 tok | ~20 tok | 1.25× |
| `skill_selection` | 128 | ~80 tok | ~91 tok | 1.4× |
| `action_taking` (merged) | 256 | ~100 tok | ~130 tok | 2.0× |

All calls use stop sequences (`\n`, `\n\nAvailable`, etc.) so outputs terminate
naturally well before hitting the token limit. The completions API (raw prompt)
is used — Qwen3's `<think>` mode is not activated, so no hidden token overhead.

Diplomacy has the largest prompts (~5K words) but its actions are presented as
discrete numbered choices via `_DiplomacyAdapter`, so output length is the same
"SUBGOAL: ... REASONING: ... ACTION: N" format as all other games.

### Stuck Detection (`episode_runner.py`)

Early episode termination if the last N steps (default 15) have zero cumulative reward. Saves vLLM tokens on hopeless episodes without affecting good runs.

### Checkpointing (`checkpoint.py`)

Every `checkpoint_interval` steps (default 5) and at step 0:
- Skill bank state (`skill_bank.jsonl`)
- All 5 LoRA adapter weights
- Step metadata (bank version, metrics, timing)
- Auto-cleanup keeps last 10 checkpoints

### W&B Logging (`orchestrator.py`)

Logged every step:
- Per-game rewards (mean, max, min, steps)
- Aggregate reward across all games
- Skill bank size and growth
- Per-adapter GRPO loss and sample counts
- Phase timing breakdown (A+B, B finalize, C)
- vLLM call counts and token usage

---

## Module Reference

### `trainer/coevolution/config.py`

```python
@dataclass
class CoEvolutionConfig:
    games: List[str]                # Default: all 8 skill bank games
    episodes_per_game: int = 8
    max_concurrent_episodes: int = 40
    total_steps: int = 30
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3-8B"
    temperature: float = 0.3
    max_tokens: int = 512
    grpo_enabled: bool = True
    grpo_decision_devices: List[int] = [4, 5]
    grpo_skillbank_devices: List[int] = [6, 7]
    checkpoint_dir: str = "runs/coevolution/checkpoints"
    checkpoint_interval: int = 5
    wandb_enabled: bool = True
    wandb_project: str = "game-ai-coevolution"
    resume_from_step: Optional[int] = None
```

### `trainer/coevolution/vllm_client.py`

Async wrapper over vLLM's OpenAI-compatible API. Routes requests to the correct LoRA adapter via the `model` field. Tracks call counts and token usage for logging.

### `trainer/coevolution/episode_runner.py`

Async port of `scripts/qwen3_decision_agent.run_episode()`. Returns `EpisodeResult` with `grpo_records: List[GRPORecord]` for both `action_taking` and `skill_selection` adapters.

### `trainer/coevolution/rollout_collector.py`

LPT-ordered scheduling with `asyncio.Semaphore` concurrency cap. Calls `on_episode_done` callback for cross-system overlap.

### `trainer/coevolution/skillbank_pipeline.py`

Wraps `skill_agents_grpo.pipeline.SkillBankAgent` for async operation. Receives episodes incrementally during rollout collection, then finalizes with contract learning + bank maintenance.

### `trainer/coevolution/grpo_training.py`

Two independent trainers (`DecisionGRPOTrainer`, `SkillBankGRPOTrainer`) that wrap `skill_agents_grpo.grpo.GRPOOrchestrator`. Run concurrently on separate GPU groups via `asyncio.gather`.

### `trainer/coevolution/checkpoint.py`

Saves/loads full snapshots: bank state + all 5 adapter weights + metadata. Auto-detects latest checkpoint for resume.

### `trainer/coevolution/orchestrator.py`

The main `co_evolution_loop()` coroutine. Manages Phase A (rollouts with cross-system overlap), Phase B (skill bank finalize), Phase C (GRPO training), checkpointing, and W&B logging.

---

## Estimated Timeline

| Steps | Wall time (8× A100) | Notes |
|-------|---------------------|-------|
| 1 step | ~12-20 min | Phase A ~8-14 min (56 episodes), Phase B ~2 min, Phase C ~3 min |
| 30 steps | ~6-10 hours | Default setting |
| 100 steps | ~20-33 hours | Recommended for convergence |

Phase A is the bottleneck — limited by 200-step games (2048, tetris, sokoban,
avalon). Each game step requires 2 sequential LLM rounds (summary ∥ skill
selection, then merged subgoal + action). With 48-56 concurrent async
episodes and 4 GPUs on vLLM (TP=4), typical throughput is ~10-13 LLM
calls/sec globally.

---

## Legacy Modules (deprecated)

The following modules are the original synchronous training infrastructure. They are **not used** by `trainer/coevolution/` and are kept only for backward compatibility with `cold_start/`, `tests/`, and shell scripts.

### `trainer/launch_coevolution.py`

Old synchronous co-evolution loop. Replaced by `trainer/coevolution/orchestrator.py`.

### `trainer/decision/`

Old synchronous decision agent training: `GRPOTrainer`, `LLMPolicy`, `collect_batch`, `ReplayBuffer`, `EnvWrapper`. Also contains unused VERL scaffolding (`GameAITrainer`, `VERLActorProxy`) behind `try/except ImportError` guards — VERL is not installed.

### `trainer/skillbank/`

Old Hard-EM pipeline driver: `EMTrainer`, `VersionedBankStore`, stages 0-4, `ingest_rollouts`. The `lora/` subdirectory (standalone LoRA SFT scripts) is still referenced by `tests/test_lora_dispatch.py`.

### `trainer/common/`

Shared types (`RolloutRecord`, `RolloutStep`, `DecisionMetrics`, `SkillBankMetrics`), `TrainLogger`, `SeedManager`, evaluation harness, and YAML config files. Still imported by `cold_start/load_rollouts.py`.
