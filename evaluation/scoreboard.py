"""evaluation/scoreboard.py - canonical scoreboard assembler.

Spec: PLAN-SYSTEM-NORTHSTAR.md section 4.

Reads a flat list of EvalInstance records (typically produced by
EvalDriver) and emits the canonical Markdown scoreboard for a release.

  - section 4.1 - 10-column canonical table (10 settings x 10 columns).
  - section 4.2 #1 - Failure-taxonomy distribution (F1-F7).
  - section 4.2 #2 - Module quality strip (caller-supplied).
  - section 4.2 #3 - Promotion / rollback ledger.
  - section 4.2 #4 - Few-shot transfer table (per target_domain).

The output file is conventionally
``releases/<release_id>/scoreboard.md`` and the path is recorded on
RunRelease.scoreboard_path.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evaluation.answer_evaluator import (
    FAILURE_CLASSES_ORDERED,
    AnswerEvaluator,
    EvalInstance,
    FailureClass,
    compute_joint_success,
)


__all__ = [
    "CANONICAL_SETTINGS",
    "CANONICAL_COLUMNS",
    "Scoreboard",
    "ScoreboardAssembler",
    "write_scoreboard_md",
]


CANONICAL_SETTINGS: Tuple[str, ...] = (
    "overall",
    "easy",
    "medium",
    "hard",
    "single_hop",
    "multi_hop",
    "direct_visual",
    "temporal",
    "social_reasoning",
    "cross_domain_transfer",
)


CANONICAL_COLUMNS: Tuple[str, ...] = (
    "Answer Acc",
    "Evidence Support",
    "Joint Success",
    "Path A",
    "Binding Success",
    "Transfer Pass",
    "Rollback Rate",
    "Avg Tool Calls",
    "Cost ($/inst)",
    "Latency (s/inst, p50/p95)",
)


@dataclass(frozen=True)
class Scoreboard:
    """Canonical-table payload for one (eval_suite_id, bank_snapshot_id)."""

    eval_suite_id: str
    bank_snapshot_id: str
    release_id: str = ""
    rows: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    failure_distribution: Mapping[str, int] = field(default_factory=dict)
    transfer_table: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    module_quality: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    promotion_ledger: Mapping[str, Any] = field(default_factory=dict)
    n_instances: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_suite_id": self.eval_suite_id,
            "bank_snapshot_id": self.bank_snapshot_id,
            "release_id": self.release_id,
            "n_instances": self.n_instances,
            "rows": {k: dict(v) for k, v in self.rows.items()},
            "failure_distribution": dict(self.failure_distribution),
            "transfer_table": {k: dict(v) for k, v in self.transfer_table.items()},
            "module_quality": {k: dict(v) for k, v in self.module_quality.items()},
            "promotion_ledger": dict(self.promotion_ledger),
        }


class ScoreboardAssembler:
    """Aggregates per-instance records into a Scoreboard."""

    def __init__(
        self,
        *,
        eval_suite_id: str,
        bank_snapshot_id: str,
        release_id: str = "",
    ) -> None:
        self._suite_id = eval_suite_id
        self._snapshot_id = bank_snapshot_id
        self._release_id = release_id
        self._evaluator = AnswerEvaluator()

    def assemble(
        self,
        instances: Iterable[EvalInstance],
        *,
        module_quality: Optional[Mapping[str, Mapping[str, Any]]] = None,
        promotion_ledger: Optional[Mapping[str, Any]] = None,
    ) -> Scoreboard:
        items: List[EvalInstance] = [
            self._evaluator.evaluate(ins) for ins in instances
        ]

        rows = self._row_aggregates(items)
        failure_distribution = self._failure_distribution(items)
        transfer_table = self._transfer_table(items)

        return Scoreboard(
            eval_suite_id=self._suite_id,
            bank_snapshot_id=self._snapshot_id,
            release_id=self._release_id,
            rows=rows,
            failure_distribution=failure_distribution,
            transfer_table=transfer_table,
            module_quality=dict(module_quality or {}),
            promotion_ledger=dict(promotion_ledger or {}),
            n_instances=len(items),
        )

    def _row_aggregates(
        self, items: Sequence[EvalInstance]
    ) -> Dict[str, Dict[str, Any]]:
        rows: Dict[str, Dict[str, Any]] = {}
        for setting in CANONICAL_SETTINGS:
            if setting == "overall":
                bucket = list(items)
            elif setting == "cross_domain_transfer":
                bucket = [i for i in items if i.transfer]
            else:
                bucket = [i for i in items if i.setting == setting]
            rows[setting] = self._aggregate_bucket(bucket)
        return rows

    def _aggregate_bucket(self, bucket: Sequence[EvalInstance]) -> Dict[str, Any]:
        if not bucket:
            return {col: None for col in CANONICAL_COLUMNS} | {"n": 0}

        n = len(bucket)
        answer_acc = sum(1 for i in bucket if i.answer_correct) / n
        evidence = sum(1 for i in bucket if i.evidence_supported) / n
        joint = sum(1 for i in bucket if compute_joint_success(i)) / n
        path_a = sum(1 for i in bucket if i.path_a) / n
        binding = sum(1 for i in bucket if i.binding_success) / n

        transfer_eligible = [i for i in bucket if i.transfer_pass is not None]
        if transfer_eligible:
            transfer_pass: Optional[float] = sum(
                1 for i in transfer_eligible if i.transfer_pass
            ) / len(transfer_eligible)
        else:
            transfer_pass = None

        rollback = sum(1 for i in bucket if i.rolled_back) / n
        avg_tool_calls = statistics.mean(i.tool_calls for i in bucket)
        cost = statistics.mean(i.cost_usd for i in bucket)

        latencies = sorted(i.latency_ms for i in bucket)
        p50 = latencies[len(latencies) // 2] / 1000.0
        p95_idx = max(0, int(round(0.95 * (len(latencies) - 1))))
        p95 = latencies[p95_idx] / 1000.0

        return {
            "n": n,
            "Answer Acc": answer_acc,
            "Evidence Support": evidence,
            "Joint Success": joint,
            "Path A": path_a,
            "Binding Success": binding,
            "Transfer Pass": transfer_pass,
            "Rollback Rate": rollback,
            "Avg Tool Calls": avg_tool_calls,
            "Cost ($/inst)": cost,
            "Latency (s/inst, p50/p95)": (p50, p95),
        }

    def _failure_distribution(
        self, items: Sequence[EvalInstance]
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {fc.value: 0 for fc in FAILURE_CLASSES_ORDERED}
        for ins in items:
            if compute_joint_success(ins):
                continue
            cls = ins.failure_class
            if cls in counts:
                counts[cls] += 1
        return counts

    def _transfer_table(
        self, items: Sequence[EvalInstance]
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        eligible = [i for i in items if i.transfer]
        targets = sorted({i.target_domain for i in eligible if i.target_domain})
        for tgt in targets:
            tgt_items = [i for i in eligible if i.target_domain == tgt]
            n = len(tgt_items)
            with_pass = [i for i in tgt_items if i.transfer_pass is not None]
            if with_pass:
                pass_rate: Optional[float] = sum(
                    1 for i in with_pass if i.transfer_pass
                ) / len(with_pass)
            else:
                pass_rate = None
            coverage = sum(1 for i in tgt_items if i.binding_success) / n if n else 0.0
            adapt_cost = (
                statistics.mean(i.cost_usd for i in tgt_items) if n else 0.0
            )
            regression_rate = (
                sum(1 for i in tgt_items if i.rolled_back) / n if n else 0.0
            )
            out[tgt] = {
                "n": n,
                "K_shot_pass_rate": pass_rate,
                "transfer_skill_coverage": coverage,
                "adaptation_cost": adapt_cost,
                "target_domain_regression_rate": regression_rate,
            }
        return out


def write_scoreboard_md(
    scoreboard: Scoreboard,
    out_path: str,
    *,
    release_notes: str = "",
) -> str:
    """Write the canonical scoreboard markdown to out_path.

    Returns the absolute path of the written file. A JSON sidecar at
    ``<out_path>.json`` round-trips losslessly.
    """

    abs_out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)

    lines: List[str] = []
    lines.append(f"# Scoreboard - {scoreboard.release_id or '(no release id)'}")
    lines.append("")
    lines.append(f"- `eval_suite_id`: `{scoreboard.eval_suite_id}`")
    lines.append(f"- `bank_snapshot_id`: `{scoreboard.bank_snapshot_id}`")
    lines.append(f"- `n_instances`: {scoreboard.n_instances}")
    lines.append("")

    lines.extend(_render_canonical_table(scoreboard))
    lines.append("")
    lines.extend(_render_failure_table(scoreboard))
    lines.append("")
    lines.extend(_render_module_quality_table(scoreboard))
    lines.append("")
    lines.extend(_render_promotion_ledger(scoreboard))
    lines.append("")
    lines.extend(_render_transfer_table(scoreboard))

    if release_notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append(release_notes.strip())

    text = "\n".join(lines).rstrip() + "\n"
    with open(abs_out, "w", encoding="utf-8") as fh:
        fh.write(text)

    sidecar = abs_out + ".json"
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(scoreboard.to_dict(), fh, indent=2, default=_json_safe)

    return abs_out


def _render_canonical_table(sb: Scoreboard) -> List[str]:
    header = "| Setting | " + " | ".join(CANONICAL_COLUMNS) + " |"
    sep = "|---" + "|---" * len(CANONICAL_COLUMNS) + "|"
    out = ["## Canonical table (PLAN-SYSTEM-NORTHSTAR section 4.1)", "", header, sep]
    for setting in CANONICAL_SETTINGS:
        row = sb.rows.get(setting, {})
        cells = [_setting_label(setting)]
        for col in CANONICAL_COLUMNS:
            cells.append(_format_cell(col, row.get(col), row.get("n", 0)))
        out.append("| " + " | ".join(cells) + " |")
    return out


def _render_failure_table(sb: Scoreboard) -> List[str]:
    out = ["## Failure-taxonomy distribution (PLAN-EVAL-FIRST-TARGET section 6)", ""]
    out.append("| Class | Description | Count | Share |")
    out.append("|---|---|---|---|")
    total = sum(sb.failure_distribution.values()) or 1
    for fc in FAILURE_CLASSES_ORDERED:
        n = sb.failure_distribution.get(fc.value, 0)
        share = n / total
        out.append(
            f"| `{fc.value}` | {_failure_class_doc(fc)} | {n} | {share:.1%} |"
        )
    return out


def _render_module_quality_table(sb: Scoreboard) -> List[str]:
    out = ["## Module quality strip (PLAN-SYSTEM-NORTHSTAR section 4.2 #2)", ""]
    out.append("| Module | Metric | Value |")
    out.append("|---|---|---|")
    rows = sb.module_quality or {
        "Actor": {"decision_top_1": None},
        "Harness": {"filter_precision": None, "veto_precision_recall": None},
        "Grounding": {"schema_completeness": None},
    }
    for module, metrics in rows.items():
        for metric, value in metrics.items():
            out.append(f"| {module} | {metric} | {_format_scalar(value)} |")
    return out


def _render_promotion_ledger(sb: Scoreboard) -> List[str]:
    out = ["## Promotion / rollback ledger (PLAN-SYSTEM-NORTHSTAR section 4.2 #3)", ""]
    out.append("| Field | Value |")
    out.append("|---|---|")
    fields = (
        "promotions",
        "rollbacks",
        "promotion_precision",
        "top_rollback_reasons",
    )
    for f in fields:
        out.append(f"| `{f}` | {_format_scalar(sb.promotion_ledger.get(f))} |")
    return out


def _render_transfer_table(sb: Scoreboard) -> List[str]:
    out = ["## Few-shot transfer table (PLAN-SYSTEM-NORTHSTAR section 4.2 #4)", ""]
    if not sb.transfer_table:
        out.append("_no transfer probes in this evaluation suite_")
        return out
    out.append(
        "| target_domain | n | K-shot pass rate | "
        "transfer skill coverage | adaptation cost | "
        "target-domain regression rate |"
    )
    out.append("|---|---|---|---|---|---|")
    for tgt, row in sorted(sb.transfer_table.items()):
        line = (
            f"| {tgt} | {row.get('n', 0)} | "
            f"{_format_scalar(row.get('K_shot_pass_rate'))} | "
            f"{_format_scalar(row.get('transfer_skill_coverage'))} | "
            f"{_format_scalar(row.get('adaptation_cost'))} | "
            f"{_format_scalar(row.get('target_domain_regression_rate'))} |"
        )
        out.append(line)
    return out


def _setting_label(setting: str) -> str:
    if setting == "overall":
        return "**overall**"
    return f"`{setting}`"


def _format_cell(column: str, value: Any, n_in_bucket: int) -> str:
    if n_in_bucket == 0 or value is None:
        return "n/a"
    if column == "Joint Success":
        return f"**{value:.3f}**"
    if column == "Latency (s/inst, p50/p95)":
        p50, p95 = value
        return f"{p50:.2f} / {p95:.2f}"
    if column == "Avg Tool Calls":
        return f"{value:.2f}"
    if column == "Cost ($/inst)":
        return f"${value:.4f}"
    return f"{value:.3f}"


def _format_scalar(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _failure_class_doc(fc: FailureClass) -> str:
    table = {
        FailureClass.F1: "answer_wrong + evidence_wrong",
        FailureClass.F2: "answer_wrong + evidence_insufficient",
        FailureClass.F3: "answer_correct + evidence_missing",
        FailureClass.F4: "answer_correct + evidence_mismatched",
        FailureClass.F5: "grounding_incomplete",
        FailureClass.F6: "over_grounding / unnecessary_tool_use",
        FailureClass.F7: "budget_exhaustion / runaway_reasoning",
    }
    return table[fc]


def _json_safe(o: Any) -> Any:
    if isinstance(o, tuple):
        return list(o)
    return str(o)
