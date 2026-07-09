"""``schema_gen`` SFT pipeline — Qwen3-VL-8B visual grounding adapter.

Implements PLAN-VISUAL-GROUNDING-MILESTONES §5.1 Phase-1: the
visual-grounding LoRA that turns ``(image, goal, [optional context])``
into a fully-populated ``<state>…</state>`` schema.

Layout::

    config.py        # ``SchemaGenConfig`` — Qwen3-VL-8B + LoRA + paths
    data_loader.py   # builds an HF Dataset from the Phase-0 triples
    train.py         # SFTTrainer / TRL entry point

The trainer reads triples produced by
``labeling/grounding/collect_{gymv,browser}.py`` plus the per-benchmark
parsers under ``visual_reasoning_wrapper/benchmarks/`` and emits a LoRA adapter at
``runs/sft_schema_gen/<run_id>/`` that the inference cascade
(``vlm_wrapper.ground.cascaded_ground``) can plug straight into the
Path-A "vision head".
"""

from .config import SchemaGenConfig
from .data_loader import (
    SchemaGenSample,
    iter_browser_triples,
    iter_gymv_triples,
    iter_image_qa_triples,
    iter_video_qa_triples,
    load_schema_gen_dataset,
)

__all__ = [
    "SchemaGenConfig",
    "SchemaGenSample",
    "iter_browser_triples",
    "iter_gymv_triples",
    "iter_image_qa_triples",
    "iter_video_qa_triples",
    "load_schema_gen_dataset",
]
