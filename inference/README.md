# Inference & Evaluation

All post-training inference and evaluation scripts for the **COS-PLAY**
co-evolution framework (COLM 2026).  For **training** scripts, see
[`scripts/`](../scripts/README.md).

---

## Directory Contents

| File | Purpose |
|------|---------|
| `run_decision_agent.py` | Core single-episode runner (API) |
| `run_inference.py` | Multi-game batch runner with skill bank & co-evolution checkpoint resolution |
| `run_inference.sh` | Shell wrapper for `run_inference.py` |
| `run_qwen3_8b_eval.py` | Main evaluation harness — all 6 games, vLLM + LoRA |
| `run_qwen3_8b_eval.sh` | Shell wrapper: launches vLLM, handles orak-mario env split |
| `run_qwen3_avalon_matched.py` | Avalon eval with training-matched prompt format |
| `run_diplomacy_discrete_eval.py` | Diplomacy eval with discrete action selection |
| `run_academic_benchmarks.py` | Academic benchmarks (MMLU-Pro, Math-500) — LoRA vs base |
| `run_single_player_inference.sh` | One-command eval for 2048 / Tetris / Candy Crush |
| `run_avalon_inference.sh` | One-command eval for Avalon (best / da / matched) |
| `run_diplomacy_inference.sh` | One-command eval for Diplomacy (da / discrete) |
| `infer_super_mario_best.sh` | One-command eval for Super Mario (orak-mario env) |

---

## Quick Start — Best-Checkpoint Evaluation

Each shell script launches a local vLLM server with the best LoRA adapter,
then runs evaluation episodes:

```bash
# Single-player games (tetris, 2048, candy_crush)
bash inference/run_single_player_inference.sh --game tetris
bash inference/run_single_player_inference.sh --game 2048
bash inference/run_single_player_inference.sh --game candy_crush

# Avalon (3 variants)
bash inference/run_avalon_inference.sh --variant best      # self-play, best checkpoint
bash inference/run_avalon_inference.sh --variant da         # vs GPT-5.4 opponents
bash inference/run_avalon_inference.sh --variant matched    # training-matched prompt format

# Diplomacy (2 variants)
bash inference/run_diplomacy_inference.sh --variant da       # vs GPT-5.4 opponents
bash inference/run_diplomacy_inference.sh --variant discrete # discrete action format

# Super Mario (separate env: orak-mario)
bash inference/infer_super_mario_best.sh

# Common overrides (all scripts):
EPISODES=16 bash inference/run_single_player_inference.sh --game tetris
EVAL_GPUS=0 bash inference/run_avalon_inference.sh --variant da
NO_SERVER=1 VLLM_BASE_URL=http://localhost:8022/v1 \
    bash inference/run_single_player_inference.sh --game tetris
```

## Main Evaluation Harness

`run_qwen3_8b_eval.py` is the comprehensive evaluation script supporting all
6 games.  The shell wrapper `run_qwen3_8b_eval.sh` handles vLLM launch and
conda env switching for Super Mario:

```bash
# Run via shell wrapper (recommended)
bash inference/run_qwen3_8b_eval.sh --games tetris avalon super_mario

# Run directly (requires vLLM already running)
python -m inference.run_qwen3_8b_eval --games twenty_forty_eight --episodes 3
python -m inference.run_qwen3_8b_eval --episodes 10    # all 6 games
python -m inference.run_qwen3_8b_eval --resume          # resume interrupted run
```

## Multi-Game Batch Runner

`run_inference.py` provides a higher-level API with skill bank loading,
co-evolution checkpoint resolution, and metrics aggregation:

```bash
bash inference/run_inference.sh \
    --model Qwen/Qwen3.5-9B \
    --bank path/to/bank.jsonl \
    --game tetris --game avalon \
    --episodes 10
```

> The `--model` default tracks the project-wide actor backbone
> (`BACKBONE_MODEL = "Qwen/Qwen3.5-9B"` in `common/models.py`).  Pass
> `--model Qwen/Qwen3-8B` to drive the deferred Qwen3-8B inference
> entrypoints (`inference/run_qwen3_8b_eval.{py,sh}`,
> `inference/run_academic_benchmarks.py`).

## Academic Benchmarks (catastrophic forgetting)

Evaluates Qwen3-8B on MMLU-Pro and Math-500 before/after game-RL fine-tuning:

```bash
python -m inference.run_academic_benchmarks --gpu 0 \
    --adapter_path runs/.../best/adapters/decision/action_taking
```

## Python API (single episode)

```python
from inference import run_inference, rollout_to_episode

episode = run_inference(
    env,
    task="Complete level 1",
    max_steps=500,
    verbose=True,
)
```

## Storage Format

- **Episode**: `experiences` (list of `Experience`), `task`, `outcome`,
  `summary`, `metadata`.
- **Experience**: `state`, `action`, `reward`, `next_state`, `done`, optional
  `intentions`, `tasks`, `sub_tasks`, `summary`, `summary_state`, `idx`,
  `reward_details`.
- **save_path**: If provided, each episode is appended as one JSON line (JSONL).

## Dependencies

- **Local inference:** `decision_agents`, `data_structure.experience`,
  `env_wrappers`
