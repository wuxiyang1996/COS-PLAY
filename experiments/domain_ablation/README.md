# Domain Ablations: ALFWorld, WebShop, Candy Crush

This folder collects the domain-specific ablation entry points that otherwise
live in different runtime families:

- `candy_crush`: contract and segmentation training ablations.
- `webshop`: BrowserGym/WebShop evaluation ablations.
- `alfworld`: text-household runtime/protocol ablations.

The goal is to keep one paper-facing experiment folder while preserving the
runtime-specific implementation boundaries.

## Quick Commands

```bash
# Candy Crush contract/segmentation ablations, sequential 4xA100 setting.
bash experiments/domain_ablation/run_domain_ablation.sh --domain candy_crush

# WebShop held-out evaluation on the first 50 registered goals.
bash experiments/domain_ablation/run_domain_ablation.sh --domain webshop

# ALFWorld text protocol ablation using the ALFWorld conda env.
bash experiments/domain_ablation/run_domain_ablation.sh --domain alfworld
```

Run all three sequentially:

```bash
bash experiments/domain_ablation/run_domain_ablation.sh --domain all
```

## Domain Notes

### Candy Crush

Delegates to `scripts/run_candy_crush_contract_ablation.sh`, which runs:

- `full`
- `no_effect_contract`
- `raw_delta_contract`
- `heuristic_only_segmentation`

The defaults match the local paper setting: 10 total steps, 8 episodes per
step, checkpoint every step, and 4xA100 split as vLLM GPUs `0 1` and GRPO GPUs
`2 3`.

### WebShop

Delegates to `scripts.skillbridge_eval.eval_browsergym` with inline
`browsergym/webshop.N` tasks. This requires:

- the `browsergym` env for BrowserGym/Playwright,
- the `webshop` env/server from `install/install_webshop.sh`,
- a vLLM endpoint serving the actor model if using model-backed evaluation.

Variants:

- `full`: normal BrowserGym actor path.
- `no_vision`: forwards `--no_vision` to the BrowserGym cold-start actor.

### ALFWorld

Runs a light text-env ablation using `env_wrappers.alfworld_nl_wrapper`:

- `with_admissible`: observations include admissible action text.
- `without_admissible`: observations omit admissible action text.

The evaluator records random-policy rollouts over admissible commands. This is
intended as the runtime/protocol ablation scaffold; model-backed ALFWorld actor
training can be added later without changing this folder layout.

## Output Layout

Defaults:

```text
runs/domain_ablation/
  candy_crush/
  webshop/
  alfworld/
```

Override with:

```bash
RUN_ROOT=/path/to/runs bash experiments/domain_ablation/run_domain_ablation.sh --domain all
```

