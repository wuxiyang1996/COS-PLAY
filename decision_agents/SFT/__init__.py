"""GPT-4o-driven actor + per-step SFT recorder.

This sub-package is the **data-collection** flavour of the Actor Agent.
It keeps every piece of the schema-native pipeline that
:class:`~decision_agents.actor_agent.ActorAgent` ships, but:

* swaps the LLM call to GPT-4o vision (chat completions with
  ``[text, image_url]`` content parts) so the actor sees the same
  screenshot the future Qwen3-VL student will see, and
* writes a per-step JSONL row in the format
  :mod:`trainer.SFT.data_loader` already understands.

The artefacts land at::

    <out_dir>/<game>/skill_selection.jsonl
    <out_dir>/<game>/action_taking.jsonl

…which is exactly the directory layout
``trainer.SFT.config.SFTConfig.decision_data_dir`` points at, so the
existing cold-start trainer ingests the data without any conversion.

Why GPT-4o
----------
Per ``vlm_wrapper/README.md`` the pipeline distils a GPT-4o-driven
cascade into a Qwen3-VL-8B student.  Stage A (schema-only SFT) and the
action-taking decision adapter both need a strong teacher to label
``(image, schema, valid_actions) → (subgoal, reasoning, action)``.
GPT-4o is the cheapest reachable model that hits acceptable label
quality on ``gymv`` / ``browser`` / ``desktop`` simultaneously, and
``API_func.ask_gpt`` already routes it via OpenRouter.

Companion sub-package: :mod:`decision_agents.grpo` (Qwen3-VL-8B
inference + GRPO+LoRA training).
"""

from __future__ import annotations

from decision_agents.SFT.actor_gpt4o import GPT4oCollectorActor
from decision_agents.SFT.sft_recorder import (
    SFTRecorder,
    SFTRecord,
    DEFAULT_SFT_OUTPUT_DIR,
)

__all__ = [
    "GPT4oCollectorActor",
    "SFTRecorder",
    "SFTRecord",
    "DEFAULT_SFT_OUTPUT_DIR",
]
