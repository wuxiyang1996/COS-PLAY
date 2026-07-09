# Ablation Study

Ablation experiments for Table 2 of the paper: evaluating the contribution of LoRA fine-tuning (SFT vs co-evolution) and skill bank quality (none / initial / best) on Diplomacy, Avalon, and Super Mario.

## Ablation Matrix

Each script takes `--adapter` and `--bank` flags to select one cell of the matrix:

| Adapter \ Bank | `none` | `first` (step 0) | `best` (final) |
|---------------|--------|-------------------|-----------------|
| `base` (vanilla Qwen/Qwen3.5-9B) | `--adapter base` | — | — |
| `sft` (SFT cold-start LoRA) | `--adapter sft --bank none` | `--adapter sft --bank first` | `--adapter sft --bank best` |
| `coevo` (co-evolution LoRA) | `--adapter coevo --bank none` | — | `--adapter coevo --bank best` |

## Usage

```bash
# Diplomacy (28 episodes: 4/power × 7, vs GPT-5.4)
bash ablation_study/run_diplomacy_ablation.sh --adapter base
bash ablation_study/run_diplomacy_ablation.sh --adapter sft --bank none
bash ablation_study/run_diplomacy_ablation.sh --adapter sft --bank first
bash ablation_study/run_diplomacy_ablation.sh --adapter sft --bank best
bash ablation_study/run_diplomacy_ablation.sh --adapter coevo --bank none
bash ablation_study/run_diplomacy_ablation.sh --adapter coevo --bank best   # full system

# Avalon (40 episodes: 8/player × 5, vs GPT-5.4)
bash ablation_study/run_avalon_ablation.sh --adapter coevo --bank best

# Super Mario (8 episodes, no bank variants)
bash ablation_study/run_super_mario_ablation.sh --adapter base
bash ablation_study/run_super_mario_ablation.sh --adapter sft

# Run all ablations for a game sequentially
bash ablation_study/run_all_ablations.sh --game diplomacy
bash ablation_study/run_all_ablations.sh --game avalon
bash ablation_study/run_all_ablations.sh --game all

# Common overrides
EVAL_GPUS=0 bash ablation_study/run_diplomacy_ablation.sh --adapter coevo --bank best
NO_SERVER=1 VLLM_BASE_URL=http://localhost:8025/v1 \
    bash ablation_study/run_diplomacy_ablation.sh --adapter sft --bank none
```

## Files

| File | Purpose |
|------|---------|
| `run_diplomacy_ablation.sh` | Diplomacy ablation (adapter × bank, vLLM + per-power eval) |
| `run_avalon_ablation.sh` | Avalon ablation (adapter × bank, vLLM + per-role eval) |
| `run_super_mario_ablation.sh` | Super Mario ablation (adapter only, Xvfb + orak-mario) |
| `run_all_ablations.sh` | Orchestrator: runs all valid combos for a game sequentially |
