"""T2.3 - tests for evaluation/{answer_evaluator,scoreboard,driver}.py.

Pins:
  * AnswerEvaluator classifies F1..F7 in the priority documented in
    PLAN-EVAL-FIRST-TARGET section 6.
  * Joint Success = answer_correct AND evidence_supported.
  * ScoreboardAssembler emits the canonical 10-row x 10-column table
    (PLAN-SYSTEM-NORTHSTAR section 4.1) with empty buckets reported as
    n/a (never silently zero).
  * write_scoreboard_md emits the markdown body and a JSON sidecar
    that round-trips Scoreboard.to_dict.
  * EvalDriver writes (a) instances.jsonl, (b) the suite scoreboard
    JSON consumable by orchestrator/eval_suite.py (T2.2), and (c) the
    optional release markdown.
  * RunRelease.scoreboard_path is included in to_json() and the
    content_hash.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest

from data_structure.extensions.run_release import RunRelease
from evaluation import (
    AnswerEvaluator,
    EvalDriver,
    EvalInstance,
    FailureClass,
    Scoreboard,
    ScoreboardAssembler,
    compute_joint_success,
    write_scoreboard_md,
)
from evaluation.scoreboard import CANONICAL_COLUMNS, CANONICAL_SETTINGS
from orchestrator.eval_suite import EvalSuiteLoader


def _ins(
    *,
    instance_id: str = "i",
    setting: str = "easy",
    domain: str = "gymv",
    answer_correct: bool = True,
    evidence_supported: bool = True,
    **extra,
) -> EvalInstance:
    return EvalInstance(
        instance_id=instance_id,
        setting=setting,
        domain=domain,
        answer_correct=answer_correct,
        evidence_supported=evidence_supported,
        **extra,
    )


# --- AnswerEvaluator ---------------------------------------------------


def test_joint_success_definition() -> None:
    assert compute_joint_success(_ins()) is True
    assert compute_joint_success(
        _ins(answer_correct=False, evidence_supported=True)
    ) is False
    assert compute_joint_success(
        _ins(answer_correct=True, evidence_supported=False)
    ) is False


def test_evaluator_clears_failure_class_on_joint_success() -> None:
    ev = AnswerEvaluator()
    out = ev.evaluate(_ins(failure_class="F1"))
    assert out.failure_class is None


def test_evaluator_classifies_f3_when_answer_right_evidence_missing() -> None:
    ev = AnswerEvaluator()
    out = ev.evaluate(
        _ins(answer_correct=True, evidence_supported=False)
    )
    assert out.failure_class == FailureClass.F3.value


def test_evaluator_classifies_f4_when_evidence_present_but_mismatched() -> None:
    ev = AnswerEvaluator()
    out = ev.evaluate(
        _ins(
            answer_correct=True,
            evidence_supported=False,
            extras={"evidence_present": True},
        )
    )
    assert out.failure_class == FailureClass.F4.value


def test_evaluator_classifies_f5_grounding_then_f7_budget() -> None:
    ev = AnswerEvaluator()
    f5 = ev.evaluate(_ins(answer_correct=False, grounding_complete=False))
    assert f5.failure_class == FailureClass.F5.value
    f7 = ev.evaluate(
        _ins(
            answer_correct=False,
            grounding_complete=False,
            budget_exhausted=True,
        )
    )
    assert f7.failure_class == FailureClass.F7.value  # F7 wins


def test_evaluator_classifies_f1_f2_f6_for_wrong_answers() -> None:
    ev = AnswerEvaluator()
    f1 = ev.evaluate(
        _ins(
            answer_correct=False,
            evidence_supported=False,
            extras={"evidence_present": True},
        )
    )
    assert f1.failure_class == FailureClass.F1.value

    f6 = ev.evaluate(
        _ins(answer_correct=False, evidence_supported=False, over_grounded=True)
    )
    assert f6.failure_class == FailureClass.F6.value

    f2 = ev.evaluate(_ins(answer_correct=False, evidence_supported=False))
    assert f2.failure_class == FailureClass.F2.value


# --- ScoreboardAssembler ----------------------------------------------


def _mixed_bucket() -> List[EvalInstance]:
    out: List[EvalInstance] = []
    for k in range(8):
        out.append(
            _ins(
                instance_id=f"easy-{k}",
                setting="easy",
                answer_correct=True,
                evidence_supported=True,
                path_a=True,
                tool_calls=2,
                cost_usd=0.001,
                latency_ms=120,
            )
        )
    for k in range(2):
        out.append(
            _ins(
                instance_id=f"easy-fail-{k}",
                setting="easy",
                answer_correct=False,
                evidence_supported=False,
                path_a=False,
                tool_calls=4,
                cost_usd=0.005,
                latency_ms=400,
            )
        )
    for k in range(5):
        out.append(
            _ins(
                instance_id=f"medium-{k}",
                setting="medium",
                answer_correct=k < 3,
                evidence_supported=k < 3,
                tool_calls=3,
                cost_usd=0.002,
                latency_ms=200,
            )
        )
    out.append(
        _ins(
            instance_id="transfer-1",
            setting="single_hop",
            transfer=True,
            target_domain="browser",
            transfer_pass=True,
            tool_calls=1,
            cost_usd=0.001,
            latency_ms=100,
        )
    )
    out.append(
        _ins(
            instance_id="transfer-2",
            setting="single_hop",
            transfer=True,
            target_domain="browser",
            transfer_pass=False,
            answer_correct=False,
            evidence_supported=False,
            tool_calls=2,
            cost_usd=0.002,
            latency_ms=160,
        )
    )
    return out


def test_assembler_emits_all_canonical_settings_and_columns() -> None:
    items = _mixed_bucket()
    sb = ScoreboardAssembler(
        eval_suite_id="suite-x",
        bank_snapshot_id="snap-1",
        release_id="rel-1",
    ).assemble(items)
    assert sb.n_instances == len(items)
    assert set(sb.rows.keys()) == set(CANONICAL_SETTINGS)
    for row in sb.rows.values():
        for col in CANONICAL_COLUMNS:
            assert col in row


def test_assembler_overall_aggregates_all_instances() -> None:
    items = _mixed_bucket()
    sb = ScoreboardAssembler(
        eval_suite_id="suite-x", bank_snapshot_id="snap-1"
    ).assemble(items)
    overall = sb.rows["overall"]
    expected_joint = sum(1 for i in items if compute_joint_success(i)) / len(items)
    assert overall["Joint Success"] == pytest.approx(expected_joint)
    assert overall["n"] == len(items)


def test_assembler_empty_bucket_reports_none() -> None:
    items = _mixed_bucket()
    sb = ScoreboardAssembler(
        eval_suite_id="suite-x", bank_snapshot_id="snap-1"
    ).assemble(items)
    hard = sb.rows["hard"]
    assert hard["n"] == 0
    for col in CANONICAL_COLUMNS:
        assert hard[col] is None


def test_assembler_failure_distribution_matches_classifier() -> None:
    items = _mixed_bucket()
    sb = ScoreboardAssembler(
        eval_suite_id="suite-x", bank_snapshot_id="snap-1"
    ).assemble(items)
    total_fail = sum(sb.failure_distribution.values())
    expected = sum(1 for i in items if not compute_joint_success(i))
    assert total_fail == expected


def test_assembler_transfer_table_keys_by_target_domain() -> None:
    items = _mixed_bucket()
    sb = ScoreboardAssembler(
        eval_suite_id="suite-x", bank_snapshot_id="snap-1"
    ).assemble(items)
    assert "browser" in sb.transfer_table
    row = sb.transfer_table["browser"]
    assert row["n"] == 2
    assert row["K_shot_pass_rate"] == pytest.approx(0.5)


# --- write_scoreboard_md ------------------------------------------------


def test_write_scoreboard_md_emits_canonical_columns(tmp_path: Path) -> None:
    items = _mixed_bucket()
    sb = ScoreboardAssembler(
        eval_suite_id="suite-x",
        bank_snapshot_id="snap-1",
        release_id="rel-1",
    ).assemble(items)
    out = tmp_path / "scoreboard.md"
    written = write_scoreboard_md(sb, str(out), release_notes="hello")
    assert os.path.exists(written)
    body = out.read_text(encoding="utf-8")
    for col in CANONICAL_COLUMNS:
        assert col in body
    for setting in CANONICAL_SETTINGS:
        assert "`" + setting + "`" in body or setting == "overall"
    assert "**overall**" in body
    assert "## Failure-taxonomy distribution" in body
    sidecar = out.with_suffix(".md.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["release_id"] == "rel-1"


# --- EvalDriver ---------------------------------------------------------


def test_driver_writes_instances_and_suite_scoreboard(tmp_path: Path) -> None:
    items = _mixed_bucket()
    out_dir = tmp_path / "eval_out"
    suites_root = tmp_path / "suites"
    drv = EvalDriver(
        eval_suite_id="suite-x",
        bank_snapshot_id="snap-1",
        out_dir=str(out_dir),
        suites_root=str(suites_root),
        release_id="rel-1",
    )
    md_path = tmp_path / "rel-1" / "scoreboard.md"
    res = drv.run(
        instances=items,
        scoreboard_md_path=str(md_path),
        release_notes="hello",
    )
    assert res.n_instances == len(items)
    assert os.path.isfile(res.instances_path)
    assert os.path.isfile(res.suite_scoreboard_path)
    assert res.scoreboard_md_path == str(md_path.resolve())

    sb_payload = json.loads(Path(res.suite_scoreboard_path).read_text(encoding="utf-8"))
    assert sb_payload["bank_snapshot_id"] == "snap-1"
    assert sb_payload["suite_id"] == "suite-x"
    assert "overall.joint_success" in sb_payload["metrics"]


def test_driver_output_loadable_via_orchestrator_loader(tmp_path: Path) -> None:
    items_pre = [
        _ins(instance_id=f"e-{k}", setting="easy", answer_correct=True, evidence_supported=True)
        for k in range(7)
    ] + [
        _ins(instance_id=f"e-{k}", setting="easy", answer_correct=False, evidence_supported=False)
        for k in range(3)
    ]
    items_post = [
        _ins(instance_id=f"e-{k}", setting="easy", answer_correct=True, evidence_supported=True)
        for k in range(9)
    ] + [
        _ins(instance_id=f"e-{k}", setting="easy", answer_correct=False, evidence_supported=False)
        for k in range(1)
    ]

    suites_root = tmp_path / "suites"
    suite_dir = suites_root / "suite-x"
    suite_dir.mkdir(parents=True)
    (suite_dir / "suite.yaml").write_text(
        "\n".join(
            [
                "suite_id: suite-x",
                'version: "1.0.0"',
                'description: "t"',
                "datasets:",
                "  - bench: gymv",
                "    split: holdout",
                "    n: 10",
                "metric_keys:",
                "  - overall.joint_success",
                "  - easy.joint_success",
                "",
            ]
        ),
        encoding="utf-8",
    )

    EvalDriver(
        eval_suite_id="suite-x",
        bank_snapshot_id="snap-pre",
        out_dir=str(tmp_path / "out_pre"),
        suites_root=str(suites_root),
    ).run(instances=items_pre)

    EvalDriver(
        eval_suite_id="suite-x",
        bank_snapshot_id="snap-post",
        out_dir=str(tmp_path / "out_post"),
        suites_root=str(suites_root),
    ).run(instances=items_post)

    suite = EvalSuiteLoader(str(suites_root)).build_eval_suite(
        "suite-x",
        pre_snapshot_id="snap-pre",
        post_snapshot_id="snap-post",
    )
    assert suite.pre_score == pytest.approx(0.7)
    assert suite.post_score == pytest.approx(0.9)
    assert suite.delta() == pytest.approx(0.2)


def test_driver_rejects_mixed_input_shapes(tmp_path: Path) -> None:
    drv = EvalDriver(
        eval_suite_id="s", bank_snapshot_id="snap", out_dir=str(tmp_path)
    )
    with pytest.raises(ValueError):
        drv.run()
    with pytest.raises(ValueError):
        drv.run(instances=[], runtime_hook=lambda: [])


# --- RunRelease ---------------------------------------------------------


def test_run_release_includes_scoreboard_path() -> None:
    rel = RunRelease(
        eval_suite_id="suite-x",
        scoreboard_path="releases/rel-1/scoreboard.md",
    )
    payload = rel.to_json()
    assert payload["eval_suite_id"] == "suite-x"
    assert payload["scoreboard_path"] == "releases/rel-1/scoreboard.md"
    same = RunRelease(
        release_id=rel.release_id,
        eval_suite_id="suite-x",
        scoreboard_path="releases/rel-1/scoreboard.md",
    )
    diff = RunRelease(
        release_id=rel.release_id,
        eval_suite_id="suite-x",
        scoreboard_path="releases/rel-2/scoreboard.md",
    )
    assert rel.content_hash() == same.content_hash()
    assert rel.content_hash() != diff.content_hash()
