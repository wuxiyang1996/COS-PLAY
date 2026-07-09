# Inference: run decision agent and store rollouts in data_structure format.
# Use run_inference for local/single-episode; use run_verl_inference for VERL (vLLM/sglang) eval.

from .run_decision_agent import (
    rollout_to_episode,
    run_inference,
)
from .run_verl_inference import run_verl_inference

__all__ = [
    "rollout_to_episode",
    "run_inference",
    "run_verl_inference",
]

# Academic benchmark evaluation (MMLU-Pro, Math-500) available via:
#   python -m inference.run_academic_benchmarks [--help]
