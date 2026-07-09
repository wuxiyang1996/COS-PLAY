"""T2.2 - tests for orchestrator/eval_suite.py.

Pins:
  * EvalSuiteSpec parses suite.yaml (PyYAML or minimal-YAML fallback).
  * EvalSuiteLoader discovers suites + per-snapshot scoreboards under a
    registry root and rejects mismatched files.
  * EvalSuiteLoader.build_eval_suite returns the runtime EvalSuite shape
    with delta/pre/post metric keys populated.
  * GateService.evaluate accepts eval_suite= and rejects callers that
    mix it with the legacy scalar pair.
  * The shipped gymv-smoke-v1 suite under evaluation/suites/ parses
    cleanly through the loader.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest

from common.enums import GateStage, SkillSourceType, SkillStatus, SkillType
from data_structure.extensions.bank_mutation_proposal import HypothesisProposal
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.gate_runner import EvalSuite as RuntimeEvalSuite
from harness.skill_harness import HarnessConfig, SkillHarness
from orchestrator.eval_suite import (
    EvalSuite,
    EvalSuiteLoader,
    EvalSuiteSpec,
    Scoreboard,
    default_suites_root,
    load_eval_suite,
    load_eval_suite_spec,
    load_scoreboard,
)
from orchestrator.gate_service import GateService


def _write_spec_yaml(suite_dir: Path, suite_id: str, version: str = "1.0.0") -> Path:
    suite_dir.mkdir(parents=True, exist_ok=True)
    p = suite_dir / "suite.yaml"
    p.write_text(
        "\n".join(
            [
                f"suite_id: {suite_id}",
                f'version: "{version}"',
                'description: "test"',
                "datasets:",
                "  - bench: gymv",
                "    split: holdout",
                "    n: 50",
                "metric_keys:",
                "  - pass_rate",
                "  - joint_success",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return p


def _write_scoreboard(
    suite_dir: Path,
    *,
    snapshot_id: str,
    suite_id: str,
    score: float,
    metrics: Mapping[str, float],
) -> Path:
    sb_dir = suite_dir / "scoreboards"
    sb_dir.mkdir(parents=True, exist_ok=True)
    p = sb_dir / f"{snapshot_id}.json"
    p.write_text(
        json.dumps(
            {
                "bank_snapshot_id": snapshot_id,
                "suite_id": suite_id,
                "score": score,
                "metrics": dict(metrics),
                "evaluated_at_utc": "2026-05-02T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return p


@pytest.fixture
def suites_root(tmp_path: Path) -> Path:
    root = tmp_path / "suites"
    suite_dir = root / "gymv-test-v1"
    _write_spec_yaml(suite_dir, "gymv-test-v1", version="1.0.0")
    _write_scoreboard(
        suite_dir,
        snapshot_id="snap-pre",
        suite_id="gymv-test-v1",
        score=0.40,
        metrics={"pass_rate": 0.40, "joint_success": 0.30},
    )
    _write_scoreboard(
        suite_dir,
        snapshot_id="snap-post",
        suite_id="gymv-test-v1",
        score=0.55,
        metrics={"pass_rate": 0.55, "joint_success": 0.50},
    )
    return root


def test_default_suites_root_resolves_under_repo() -> None:
    root = default_suites_root()
    assert os.path.basename(root) == "suites"
    if os.path.isdir(root):
        assert "gymv-smoke-v1" in EvalSuiteLoader().list_suites()


def test_load_spec_parses_yaml(suites_root: Path) -> None:
    spec = EvalSuiteLoader(str(suites_root)).load_spec("gymv-test-v1")
    assert isinstance(spec, EvalSuiteSpec)
    assert spec.suite_id == "gymv-test-v1"
    assert spec.version == "1.0.0"
    assert spec.metric_keys == ("pass_rate", "joint_success")
    assert len(spec.datasets) == 1
    assert spec.datasets[0]["bench"] == "gymv"
    assert spec.datasets[0]["split"] == "holdout"
    assert spec.datasets[0]["n"] == 50


def test_load_spec_supports_json(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suites" / "json-suite-v1"
    suite_dir.mkdir(parents=True)
    payload: Dict[str, Any] = {
        "suite_id": "json-suite-v1",
        "version": "0.1.0",
        "description": "json variant",
        "datasets": [{"bench": "gymv", "split": "holdout", "n": 12}],
        "metric_keys": ["pass_rate"],
    }
    (suite_dir / "suite.json").write_text(json.dumps(payload), encoding="utf-8")
    spec = EvalSuiteLoader(str(tmp_path / "suites")).load_spec("json-suite-v1")
    assert spec.metric_keys == ("pass_rate",)
    assert spec.description == "json variant"


def test_load_spec_rejects_directory_mismatch(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suites" / "right-name-v1"
    _write_spec_yaml(suite_dir, "wrong-name-v1", version="1.0.0")
    with pytest.raises(ValueError, match="does not match"):
        EvalSuiteLoader(str(tmp_path / "suites")).load_spec("right-name-v1")


def test_load_spec_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EvalSuiteLoader(str(tmp_path)).load_spec("missing")


def test_load_scoreboard_round_trip(suites_root: Path) -> None:
    sb = EvalSuiteLoader(str(suites_root)).load_scoreboard(
        "gymv-test-v1", "snap-pre"
    )
    assert isinstance(sb, Scoreboard)
    assert sb.suite_id == "gymv-test-v1"
    assert sb.bank_snapshot_id == "snap-pre"
    assert sb.score == pytest.approx(0.40)
    assert sb.metrics["pass_rate"] == pytest.approx(0.40)


def test_load_scoreboard_rejects_suite_mismatch(suites_root: Path) -> None:
    sb_dir = suites_root / "gymv-test-v1" / "scoreboards"
    bad = sb_dir / "snap-bad.json"
    bad.write_text(
        json.dumps(
            {
                "bank_snapshot_id": "snap-bad",
                "suite_id": "wrong-suite",
                "score": 0.42,
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        EvalSuiteLoader(str(suites_root)).load_scoreboard(
            "gymv-test-v1", "snap-bad"
        )


def test_list_suites_and_scoreboards(suites_root: Path) -> None:
    loader = EvalSuiteLoader(str(suites_root))
    assert loader.list_suites() == ["gymv-test-v1"]
    assert loader.list_scoreboards("gymv-test-v1") == ["snap-post", "snap-pre"]


def test_build_eval_suite_returns_runtime_payload(suites_root: Path) -> None:
    loader = EvalSuiteLoader(str(suites_root))
    suite = loader.build_eval_suite(
        "gymv-test-v1",
        pre_snapshot_id="snap-pre",
        post_snapshot_id="snap-post",
    )
    assert isinstance(suite, RuntimeEvalSuite)
    assert isinstance(suite, EvalSuite)
    assert suite.suite_id == "gymv-test-v1"
    assert suite.pre_score == pytest.approx(0.40)
    assert suite.post_score == pytest.approx(0.55)
    assert suite.delta() == pytest.approx(0.15)
    assert suite.metrics["pre.pass_rate"] == pytest.approx(0.40)
    assert suite.metrics["post.pass_rate"] == pytest.approx(0.55)
    assert suite.metrics["delta.pass_rate"] == pytest.approx(0.15)
    assert suite.metrics["delta.joint_success"] == pytest.approx(0.20)


def test_module_helpers_match_loader(suites_root: Path) -> None:
    spec = load_eval_suite_spec("gymv-test-v1", suites_root=str(suites_root))
    sb = load_scoreboard(
        "gymv-test-v1", "snap-pre", suites_root=str(suites_root)
    )
    suite = load_eval_suite(
        "gymv-test-v1",
        pre_snapshot_id="snap-pre",
        post_snapshot_id="snap-post",
        suites_root=str(suites_root),
    )
    assert spec.suite_id == "gymv-test-v1"
    assert sb.bank_snapshot_id == "snap-pre"
    assert suite.delta() == pytest.approx(0.15)


def test_shipped_gymv_smoke_v1_parses() -> None:
    root = default_suites_root()
    if not os.path.isdir(os.path.join(root, "gymv-smoke-v1")):
        pytest.skip("gymv-smoke-v1 not shipped on this checkout")
    spec = EvalSuiteLoader().load_spec("gymv-smoke-v1")
    assert spec.suite_id == "gymv-smoke-v1"
    assert spec.version == "1.0.0"
    assert any(d.get("bench") == "gymv" for d in spec.datasets)
    assert "gymv_holdout.pass_rate" in spec.metric_keys


def _fresh_skill() -> SkillRecord:
    return SkillRecord(
        skill_id="s",
        name="s",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.MINED,
        status=SkillStatus.PROVISIONAL,
        feasible_domains=["gymv", "browser"],
        source_domains=["gymv"],
        protocol=[{"action": "STEP", "payload": {}}],
        contract=SkillContract(),
    )


def _hypothesis_for(skill: SkillRecord) -> HypothesisProposal:
    return HypothesisProposal(
        proposal_id="prop-test",
        rationale="t",
        name=skill.name,
        novel_protocol=list(skill.protocol),
        contract=skill.contract,
    )


def _gate_service() -> GateService:
    registry = AdapterRegistry()
    return GateService(harness=SkillHarness(registry, config=HarnessConfig()))


def test_gate_service_accepts_eval_suite_kw(suites_root: Path) -> None:
    suite = load_eval_suite(
        "gymv-test-v1",
        pre_snapshot_id="snap-pre",
        post_snapshot_id="snap-post",
        suites_root=str(suites_root),
    )
    skill = _fresh_skill()
    gs = _gate_service()
    rec = gs.evaluate(
        proposal=_hypothesis_for(skill),
        skill=skill,
        eval_suite=suite,
    )
    sv = rec.verdict.stage_for(GateStage.NON_REGRESSION)
    assert sv is not None
    assert sv.metrics["pre"] == pytest.approx(0.40)
    assert sv.metrics["post"] == pytest.approx(0.55)


def test_gate_service_rejects_mixed_stage4_inputs(suites_root: Path) -> None:
    suite = load_eval_suite(
        "gymv-test-v1",
        pre_snapshot_id="snap-pre",
        post_snapshot_id="snap-post",
        suites_root=str(suites_root),
    )
    skill = _fresh_skill()
    gs = _gate_service()
    with pytest.raises(ValueError, match="eval_suite.*baseline_score"):
        gs.evaluate(
            proposal=_hypothesis_for(skill),
            skill=skill,
            eval_suite=suite,
            baseline_score=0.4,
            post_score=0.5,
        )
