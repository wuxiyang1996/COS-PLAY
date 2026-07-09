"""evaluation/ — Eval E0 driver, answer evaluator and canonical scoreboard.

Spec: ``PLAN-SYSTEM-NORTHSTAR.md`` §4 / §7.3 — every release MUST emit
``releases/<release_id>/scoreboard.md`` with the 10-column canonical
table + 4 companion tables, all sharing one ``eval_suite_id`` +
``bank_snapshot_id``.

Public surface::

    from evaluation import (
        AnswerEvaluator,
        EvalInstance,
        EvalDriver,
        DriverResult,
        Scoreboard,
        ScoreboardAssembler,
        compute_joint_success,
        write_scoreboard_md,
    )
"""

from __future__ import annotations

from evaluation.answer_evaluator import (
    AnswerEvaluator,
    EvalInstance,
    FailureClass,
    compute_joint_success,
)
from evaluation.driver import DriverResult, EvalDriver
from evaluation.scoreboard import (
    Scoreboard,
    ScoreboardAssembler,
    write_scoreboard_md,
)

__all__ = [
    "AnswerEvaluator",
    "DriverResult",
    "EvalDriver",
    "EvalInstance",
    "FailureClass",
    "Scoreboard",
    "ScoreboardAssembler",
    "compute_joint_success",
    "write_scoreboard_md",
]
