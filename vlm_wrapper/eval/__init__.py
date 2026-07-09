"""Evaluation harness for the visual-grounding pipeline.

Implements PLAN-VISUAL-GROUNDING-MILESTONES §8 Week 4 deliverable —
the metrics + driver code that turns a stream of (input, gold,
predicted-schema) triples into the numbers the milestone exit criteria
are measured against:

* **Field accuracy** per schema section (entities, attributes,
  relations, state_flags, targets, evidence, answer)
* **Format compliance** rate (regex parse success + ``validate_schema``)
* **Entity coverage** — IoU of predicted entity bboxes vs heuristic
  / scene-graph ground truth
* **Target-slot accuracy** (target / blocker correctly identified)
* **Path A / B / C breakdown** (cascaded escalation telemetry)
* **Tool-call precision / recall** vs the heuristic-derived oracle

Layout::

    metrics.py     # pure-function metrics over (gold, pred) schemas
    harness.py     # ``run_eval`` driver — iterates a sample stream and
                   # produces per-sample + aggregated reports
    run_eval.py    # CLI entry point: ``python -m vlm_wrapper.eval.run_eval``

Designed to plug straight into the benchmark loaders
(``visual_reasoning_wrapper.benchmarks.*``) and the cascaded grounding pipeline
(``vlm_wrapper.ground.cascaded_ground``) — no benchmark-specific
branching in the harness itself.
"""

from .metrics import (
    EvalMetrics,
    PerSampleResult,
    compute_answer_accuracy,
    compute_entity_iou,
    compute_field_accuracy,
    compute_format_compliance,
    compute_path_breakdown,
    compute_target_accuracy,
    compute_tool_precision,
    summarise_metrics,
)
from .harness import (
    EvalReport,
    run_eval,
)

__all__ = [
    "EvalMetrics",
    "PerSampleResult",
    "EvalReport",
    "compute_answer_accuracy",
    "compute_entity_iou",
    "compute_field_accuracy",
    "compute_format_compliance",
    "compute_path_breakdown",
    "compute_target_accuracy",
    "compute_tool_precision",
    "summarise_metrics",
    "run_eval",
]
