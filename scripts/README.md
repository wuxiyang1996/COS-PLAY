# scripts -- Training

Entry-point scripts for the COS-PLAY training pipeline: SFT cold-start,
co-evolution training, skill extraction, and GRPO training data generation.

For **inference and evaluation**, see [`inference/`](../inference/README.md).

---

## Pipeline Overview

```
1. SFT Cold-Start        run_sft_coldstart.sh
2. Skill Extraction       run_qwen3_skillbank_agent.sh
3. Co-Evolution Training  run_coevolution.py  (+ per-game .sh wrappers)
4. Decision Agent (GRPO)  qwen3_decision_agent.py  (training data generation)
```

---

## SFT Cold-Start (Step 1)

Trains all 5 LoRA adapters from teacher-labelled cold-start data.

```bash
bash scripts/run_sft_coldstart.sh
```

## Skill Bank Extraction (Step 2)

Extracts skills from SFT-teacher (`gpt-5.5`) rollouts using the
project-wide skill-bank backbone (`Qwen/Qwen3.5-9B` —
`BACKBONE_MODEL` in `common/models.py`).  The legacy entrypoint
keeps its `qwen3_*` filename for back-compat; override the model
via the `MODEL` env var if you want to run the deferred Qwen3-8B
track instead.

```bash
bash scripts/run_qwen3_skillbank_agent.sh
```

## Co-Evolution Training (Step 3)

| Script | Game(s) | Notes |
|--------|---------|-------|
| `run_coevolution.py` | Any (CLI args) | Core training launcher |
| `run_all.sh` | All 5 games | Curriculum training (sequential phases) |
| `run_2048.sh` | 2048 | SFT warm-start |
| `run_tetris.sh` | Tetris | Stability-focused GRPO |
| `run_super_mario.sh` | Super Mario | Orak subprocess env + Xvfb |
| `run_avalon.sh` | Avalon | Self-play, conservative GRPO |
| `run_diplomacy.sh` | Diplomacy | 7-player FSDP tuning |
| `train_avalon_vs_gpt5mini.sh` | Avalon | GPT-5-mini as external opponent |
| `train_diplomacy_vs_gpt5mini.sh` | Diplomacy | GPT-5-mini as external opponent |

```bash
# Single game
bash scripts/run_2048.sh

# Full curriculum (all games sequentially)
bash scripts/run_all.sh

# Direct CLI (advanced)
python scripts/run_coevolution.py \
    --games tetris candy_crush \
    --total-steps 50 \
    --episodes-per-game 8
```

## Decision Agent with GRPO Data (Step 4)

Runs the dual-LoRA decision agent pipeline and records GRPO training data
(skill_selection + action_taking adapters).

```bash
bash scripts/run_qwen3_decision_agent.sh
```
