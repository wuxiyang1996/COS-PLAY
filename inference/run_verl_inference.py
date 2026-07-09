# VERL-based inference: run decision agent via verl-agent (vLLM/sglang) and optionally save rollouts.
#
# This runs the VERL trainer in evaluation-only mode (no training steps):
#   total_epochs=0, val_before_train=True
# So the validation rollout loop runs once and metrics/rollouts are produced using
# the same env and reward as VERL training.
#
# Usage:
#   python -m inference.run_verl_inference [Hydra overrides...]
#   python -m inference.run_verl_inference trainer.total_epochs=0 trainer.val_before_train=True
#
# Requires: verl-agent at ../verl-agent and Game-AI-Agent on PYTHONPATH.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_verl_inference(extra_overrides: list[str] | None = None) -> int:
    """Run VERL Game-AI in inference-only mode (validation rollouts, no training).

    Requires verl-agent at ../verl-agent relative to Game-AI-Agent repo root.
    Overrides are passed to verl.trainer.main_gameai (Hydra).
    """
    repo_root = Path(__file__).resolve().parent.parent
    verl_agent_root = repo_root.parent / "verl-agent"
    if not verl_agent_root.is_dir():
        print(
            f"verl-agent not found at {verl_agent_root}. "
            "Clone it next to Game-AI-Agent and install: pip install -e .",
            file=sys.stderr,
        )
        return 1

    env = os.environ.copy()
    path_parts = [str(repo_root), str(verl_agent_root)]
    if env.get("PYTHONPATH"):
        path_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(path_parts)

    default_overrides = [
        "trainer.total_epochs=0",
        "trainer.val_before_train=True",
        "env.env_name=gameai",
        "env.seed=42",
        "reward_model.reward_manager=gameai",
    ]
    overrides = default_overrides + (extra_overrides or [])
    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_gameai",
        *overrides,
    ]
    print("Running VERL inference (eval-only):", " ".join(cmd))
    return subprocess.run(cmd, env=env, cwd=str(repo_root.parent)).returncode


def main() -> int:
    _, *overrides = sys.argv
    return run_verl_inference(overrides)


if __name__ == "__main__":
    sys.exit(main())
