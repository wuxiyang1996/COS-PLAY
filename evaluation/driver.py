"""evaluation/driver.py - Eval E0 driver (T2.3).

Reads or accepts EvalInstance records that share one eval_suite_id and
bank_snapshot_id, classifies them via AnswerEvaluator, writes a
per-instance JSONL trace, builds a Scoreboard via ScoreboardAssembler,
and writes the suite-level scoreboard JSON consumed by Stage-4
non-regression (orchestrator/eval_suite.py, T2.2). Optionally emits
releases/<release_id>/scoreboard.md.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

from evaluation.answer_evaluator import AnswerEvaluator, EvalInstance
from evaluation.scoreboard import (
    Scoreboard,
    ScoreboardAssembler,
    write_scoreboard_md,
)


__all__ = [
    "DriverResult",
    "EvalDriver",
    "load_instances_jsonl",
]


@dataclass
class DriverResult:
    eval_suite_id: str
    bank_snapshot_id: str
    n_instances: int
    overall_joint_success: float
    instances_path: str
    suite_scoreboard_path: str
    scoreboard: Scoreboard
    scoreboard_md_path: Optional[str] = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "eval_suite_id": self.eval_suite_id,
            "bank_snapshot_id": self.bank_snapshot_id,
            "n_instances": self.n_instances,
            "overall_joint_success": self.overall_joint_success,
            "instances_path": self.instances_path,
            "suite_scoreboard_path": self.suite_scoreboard_path,
            "scoreboard_md_path": self.scoreboard_md_path,
            "extras": dict(self.extras),
        }


class EvalDriver:
    def __init__(
        self,
        *,
        eval_suite_id: str,
        bank_snapshot_id: str,
        out_dir: str,
        suites_root: Optional[str] = None,
        release_id: str = "",
    ) -> None:
        self._suite_id = eval_suite_id
        self._snapshot_id = bank_snapshot_id
        self._out_dir = os.path.abspath(out_dir)
        self._suites_root = suites_root
        self._release_id = release_id
        self._evaluator = AnswerEvaluator()

    def run(
        self,
        *,
        instances: Optional[Iterable[EvalInstance]] = None,
        runtime_hook: Optional[Callable[[], Iterable[EvalInstance]]] = None,
        scoreboard_md_path: Optional[str] = None,
        release_notes: str = "",
        module_quality: Optional[Mapping[str, Mapping[str, Any]]] = None,
        promotion_ledger: Optional[Mapping[str, Any]] = None,
    ) -> DriverResult:
        if instances is None and runtime_hook is None:
            raise ValueError(
                "EvalDriver.run: pass either instances=... or runtime_hook=..."
            )
        if instances is not None and runtime_hook is not None:
            raise ValueError(
                "EvalDriver.run: pass either instances=... or "
                "runtime_hook=..., not both."
            )

        raw: List[EvalInstance] = list(
            instances if instances is not None else runtime_hook()
        )
        classified = [self._evaluator.evaluate(ins) for ins in raw]

        os.makedirs(self._out_dir, exist_ok=True)
        instances_path = os.path.join(self._out_dir, "instances.jsonl")
        with open(instances_path, "w", encoding="utf-8") as fh:
            for ins in classified:
                fh.write(json.dumps(ins.to_dict()) + "\n")

        assembler = ScoreboardAssembler(
            eval_suite_id=self._suite_id,
            bank_snapshot_id=self._snapshot_id,
            release_id=self._release_id,
        )
        sb = assembler.assemble(
            classified,
            module_quality=module_quality,
            promotion_ledger=promotion_ledger,
        )

        suite_sb_path = self._write_suite_scoreboard(classified, sb)

        md_path: Optional[str] = None
        if scoreboard_md_path:
            md_path = write_scoreboard_md(
                sb, scoreboard_md_path, release_notes=release_notes
            )

        overall = sb.rows.get("overall", {}).get("Joint Success") or 0.0

        return DriverResult(
            eval_suite_id=self._suite_id,
            bank_snapshot_id=self._snapshot_id,
            n_instances=len(classified),
            overall_joint_success=float(overall),
            instances_path=instances_path,
            suite_scoreboard_path=suite_sb_path,
            scoreboard=sb,
            scoreboard_md_path=md_path,
            extras={"release_id": self._release_id} if self._release_id else {},
        )

    def _write_suite_scoreboard(
        self, items: Sequence[EvalInstance], sb: Scoreboard
    ) -> str:
        if self._suites_root is None:
            target_dir = os.path.join(
                self._out_dir, "suites", self._suite_id, "scoreboards"
            )
        else:
            target_dir = os.path.join(
                self._suites_root, self._suite_id, "scoreboards"
            )
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, self._snapshot_id + ".json")

        score = sb.rows.get("overall", {}).get("Joint Success") or 0.0

        suite_metrics: dict = {}
        for setting, row in sb.rows.items():
            if not row or row.get("n", 0) == 0:
                continue
            for col in (
                "Answer Acc",
                "Evidence Support",
                "Joint Success",
                "Path A",
                "Binding Success",
                "Rollback Rate",
                "Avg Tool Calls",
                "Cost ($/inst)",
            ):
                value = row.get(col)
                if value is None:
                    continue
                key = setting + "." + _metric_key(col)
                suite_metrics[key] = float(value)
            tp = row.get("Transfer Pass")
            if tp is not None:
                suite_metrics[setting + ".transfer_pass"] = float(tp)

        payload = {
            "bank_snapshot_id": self._snapshot_id,
            "suite_id": self._suite_id,
            "score": float(score),
            "metrics": suite_metrics,
            "evaluated_at_utc": _utcnow(),
            "n_instances": len(items),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path


def load_instances_jsonl(path: str) -> List[EvalInstance]:
    out: List[EvalInstance] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(EvalInstance.from_dict(json.loads(line)))
    return out


def _metric_key(column: str) -> str:
    s = column.lower().replace(" ", "_")
    s = s.replace("(", "").replace(")", "")
    s = s.replace("$/inst", "usd_per_instance")
    s = s.replace("$/", "usd_per_")
    s = s.replace("/", "_")
    return s


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
