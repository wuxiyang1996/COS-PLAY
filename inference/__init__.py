# Inference & Evaluation: all post-training scripts live here.
#
# Core API (single-episode runner):
#   from inference import run_inference, rollout_to_episode
#
# CLI entry points:
#   python -m inference.run_inference          (multi-game batch runner)
#   python -m inference.run_qwen3_8b_eval      (main evaluation harness)
#   python -m inference.run_academic_benchmarks  (MMLU-Pro, Math-500)

from .run_decision_agent import (
    rollout_to_episode,
    run_inference,
)

__all__ = [
    "rollout_to_episode",
    "run_inference",
]
