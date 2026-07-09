"""Smoke tests for the visual_reasoning HopExecutor.

These exercise the binding seam between
:class:`visual_reasoning_wrapper.skill_executor.VisualReasoningExecutor`
and
:class:`harness.adapters.visual_reasoning_adapter.VisualReasoningAdapter`
without spinning up any actual vision models — observation tools are
stubbed, but the *real* pure-Python reasoning tools are kept so the
typed derivation log behaviour is also validated.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import pytest
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.enums import EVIDENCE_ROLES, SkillSourceType, SkillType
from common.state_schema import StateSchema
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness.adapters.visual_reasoning_adapter import (
    VisualReasoningAdapter,
    bind_visual_reasoning_executor,
)
from harness.skill_adapter import AdapterRunContext
from visual_reasoning_wrapper import skill_executor as ve
from visual_reasoning_wrapper.tools_reasoning import build_reasoning_registry
from vlm_wrapper.tools import ToolDef, ToolRegistry


# --------------------------------------------------------------------- helpers

def _make_stub_visual_registry(
    image: Image.Image,  # noqa: ARG001 - signature parity with tools_visual.build_visual_registry
    *,
    prefer_gdino: bool = True,  # noqa: ARG001
    include_reasoning: bool = True,  # noqa: ARG001
) -> ToolRegistry:
    """Drop-in replacement for ``build_visual_registry`` for tests.

    Registers stub observation tools that mirror the real signatures
    plus the *real* reasoning tools (pure Python) so derivation IDs
    and the verify path are exercised end-to-end.
    """
    reg = ToolRegistry(domain="visual")

    def _detect_objects(*, max_elements: int = 25, confidence_threshold: float = 0.2):
        return {
            "elements": [
                {
                    "id": "e1",
                    "label": "red cube",
                    "bbox": {"x": 10, "y": 10, "w": 40, "h": 40},
                    "confidence": 0.9,
                },
                {
                    "id": "e2",
                    "label": "blue sphere",
                    "bbox": {"x": 60, "y": 20, "w": 30, "h": 30},
                    "confidence": 0.8,
                },
            ][:max_elements],
        }

    def _grounded_detect(*, query: str, confidence_threshold: float = 0.2, max_results: int = 10):
        return {
            "query": query,
            "elements": [
                {
                    "id": "e1",
                    "label": query,
                    "bbox": {"x": 10, "y": 10, "w": 40, "h": 40},
                    "confidence": 0.85,
                },
            ][:max_results],
        }

    def _describe_region(*, x: int, y: int, w: int, h: int):
        return {
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "caption": f"object at ({x},{y}) of size {w}x{h}",
        }

    def _read_text_region(*, x: int = 0, y: int = 0, w: int = 0, h: int = 0):
        return {
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "text": "stub-ocr-text",
            "lines": ["stub-ocr-text"],
        }

    def _spatial_query(*, element_a: str, element_b: str):
        return {
            "element_a": element_a,
            "element_b": element_b,
            "relation": "left_of",
            "distance_px": 25,
        }

    reg.register(
        ToolDef(name="detect_objects", description="stub", parameters={"type": "object", "properties": {}}),
        lambda **kw: _detect_objects(**kw),
    )
    reg.register(
        ToolDef(name="grounded_detect", description="stub", parameters={"type": "object", "properties": {}}),
        lambda **kw: _grounded_detect(**kw),
    )
    reg.register(
        ToolDef(name="describe_region", description="stub", parameters={"type": "object", "properties": {}}),
        lambda **kw: _describe_region(**kw),
    )
    reg.register(
        ToolDef(name="read_text_region", description="stub", parameters={"type": "object", "properties": {}}),
        lambda **kw: _read_text_region(**kw),
    )
    reg.register(
        ToolDef(name="spatial_query", description="stub", parameters={"type": "object", "properties": {}}),
        lambda **kw: _spatial_query(**kw),
    )

    reasoning_reg, log = build_reasoning_registry()
    merged = reg.merge(reasoning_reg)
    merged.derivation_log = log  # type: ignore[attr-defined]
    return merged


@pytest.fixture
def stub_image() -> Image.Image:
    return Image.new("RGB", (128, 128), color=(255, 255, 255))


@pytest.fixture
def adapter_with_stub_executor(monkeypatch: pytest.MonkeyPatch, stub_image: Image.Image):
    """Adapter with a stub-backed VisualReasoningExecutor wired in."""
    monkeypatch.setattr(ve, "build_visual_registry", _make_stub_visual_registry)
    adapter = VisualReasoningAdapter()
    executor = bind_visual_reasoning_executor(adapter, image=stub_image)
    return adapter, executor


def _ctx(task: str = "describe the red cube", **bindings: Any) -> AdapterRunContext:
    return AdapterRunContext(
        state=StateSchema(task=task, domain="visual_reasoning"),
        bindings=dict(bindings),
        budget={"hops": 16, "ms": 60_000.0},
    )


def _skill(protocol: List[Dict[str, Any]], expected_roles: List[str]) -> SkillRecord:
    return SkillRecord.new(
        name="ground_check_verify_commit",
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.CRAFTED,
        feasible_domains=["gymv", "visual_reasoning"],
        source_domains=["gymv"],
        transfer_target_domains=["visual_reasoning"],
        protocol=protocol,
        contract=SkillContract(expected_evidence_roles=expected_roles),
    )


# --------------------------------------------------------------------- tests

class TestExecutorWiring:
    def test_bind_attaches_executor(self, adapter_with_stub_executor) -> None:
        adapter, executor = adapter_with_stub_executor
        assert isinstance(executor, ve.VisualReasoningExecutor)
        assert adapter._executor is executor  # type: ignore[attr-defined]

    def test_bind_requires_set_executor(self, stub_image: Image.Image) -> None:
        with pytest.raises(TypeError, match="set_executor"):
            ve.bind_executor(object(), image=stub_image)


class TestActionToToolMapping:
    def test_ground_with_query_uses_grounded_detect(self, adapter_with_stub_executor) -> None:
        _, executor = adapter_with_stub_executor
        out = executor("GROUND", {"query": "red cube"}, _ctx())
        assert out["ok"] is True
        assert out["observation"]["tool"] == "grounded_detect"
        assert {ev.role for ev in out["evidence"]} == {"GATHER"}

    def test_ground_without_query_falls_back_to_detect_objects(self, adapter_with_stub_executor) -> None:
        _, executor = adapter_with_stub_executor
        out = executor("GROUND", {"max_elements": 5}, _ctx())
        assert out["observation"]["tool"] == "detect_objects"
        assert out["observation"]["result"]["elements"][0]["id"] == "e1"

    def test_retrieve_uses_describe_region_for_bbox(self, adapter_with_stub_executor) -> None:
        _, executor = adapter_with_stub_executor
        out = executor("RETRIEVE", {"x": 10, "y": 10, "w": 40, "h": 40}, _ctx())
        assert out["observation"]["tool"] == "describe_region"
        assert {ev.role for ev in out["evidence"]} == {"GATHER"}

    def test_retrieve_with_use_ocr_switches_to_read_text_region(
        self, adapter_with_stub_executor
    ) -> None:
        _, executor = adapter_with_stub_executor
        out = executor(
            "RETRIEVE",
            {"x": 10, "y": 10, "w": 40, "h": 40, "use_ocr": True},
            _ctx(),
        )
        assert out["observation"]["tool"] == "read_text_region"

    def test_retrieve_via_entity_index_after_ground(self, adapter_with_stub_executor) -> None:
        _, executor = adapter_with_stub_executor
        executor("GROUND", {}, _ctx())
        out = executor("RETRIEVE", {"entity_index": 0}, _ctx())
        assert out["ok"] is True
        assert out["observation"]["tool"] == "describe_region"

    def test_check_count_emits_reason_evidence_and_derivation(
        self, adapter_with_stub_executor
    ) -> None:
        _, executor = adapter_with_stub_executor
        out = executor(
            "CHECK",
            {"kind": "COUNT", "value": 3, "label": "red cubes", "refs": "e1,e2,e3"},
            _ctx(),
        )
        assert out["ok"] is True
        assert out["observation"]["tool"] == "count_value"
        roles = {ev.role for ev in out["evidence"]}
        assert "REASON" in roles
        # The derivation log should now have one COUNT row.
        log_rows = list(executor.derivation_log)
        assert any(r.kind == "COUNT" for r in log_rows)

    def test_check_ratio_uses_compute_ratio(self, adapter_with_stub_executor) -> None:
        _, executor = adapter_with_stub_executor
        out = executor(
            "CHECK",
            {"kind": "RATIO", "numerator": 3, "denominator": 10, "label": "red/total"},
            _ctx(),
        )
        assert out["observation"]["tool"] == "compute_ratio"
        assert out["observation"]["result"]["fraction"] == pytest.approx(0.3)

    def test_check_compare_uses_compare_values(self, adapter_with_stub_executor) -> None:
        _, executor = adapter_with_stub_executor
        out = executor(
            "CHECK",
            {
                "kind": "COMPARE",
                "a": 100,
                "b": 50,
                "op": ">",
                "label_a": "red_area",
                "label_b": "blue_area",
            },
            _ctx(),
        )
        assert out["observation"]["tool"] == "compare_values"
        assert out["observation"]["result"]["result"] is True

    def test_check_without_kind_falls_back_to_spatial_query(
        self, adapter_with_stub_executor
    ) -> None:
        _, executor = adapter_with_stub_executor
        out = executor(
            "CHECK",
            {"element_a": "e1", "element_b": "e2"},
            _ctx(),
        )
        assert out["ok"] is True
        assert out["observation"]["tool"] == "spatial_query"

    def test_verify_commits_via_verify_claim(self, adapter_with_stub_executor) -> None:
        _, executor = adapter_with_stub_executor
        out = executor(
            "VERIFY",
            {"claim": "C", "evidence_refs": "e1,d1", "confidence": "high"},
            _ctx(),
        )
        assert out["observation"]["tool"] == "verify_claim"
        assert {ev.role for ev in out["evidence"]} == {"VERIFY"}

    def test_commit_emits_both_verify_and_commit_evidence(
        self, adapter_with_stub_executor
    ) -> None:
        _, executor = adapter_with_stub_executor
        out = executor(
            "COMMIT",
            {"claim": "yes", "evidence_refs": ["d1"], "confidence": "medium"},
            _ctx(),
        )
        roles = {ev.role for ev in out["evidence"]}
        assert {"VERIFY", "COMMIT"}.issubset(roles)

    def test_execute_is_no_op_with_commit_evidence(self, adapter_with_stub_executor) -> None:
        _, executor = adapter_with_stub_executor
        out = executor("EXECUTE", {}, _ctx())
        assert out["ok"] is True
        assert out["observation"]["note"].startswith("no-op")
        assert {ev.role for ev in out["evidence"]} == {"COMMIT"}


class TestErrorPaths:
    def test_unbound_slot_aborts_hop(self, adapter_with_stub_executor) -> None:
        _, executor = adapter_with_stub_executor
        out = executor("GROUND", {"query": "${target}"}, _ctx())
        assert out["ok"] is False
        assert "unbound slot" in out["reason"]
        assert out["evidence"] == []

    def test_unknown_action_returns_failure(self, adapter_with_stub_executor) -> None:
        _, executor = adapter_with_stub_executor
        out = executor("CONTEMPLATE", {}, _ctx())
        assert out["ok"] is False
        assert "no visual_reasoning tool mapping" in out["reason"]

    def test_check_without_kind_or_spatial_args_fails(
        self, adapter_with_stub_executor
    ) -> None:
        _, executor = adapter_with_stub_executor
        out = executor("CHECK", {"value": 3}, _ctx())
        assert out["ok"] is False
        assert "CHECK hop" in out["reason"]

    def test_verify_without_claim_fails(self, adapter_with_stub_executor) -> None:
        _, executor = adapter_with_stub_executor
        out = executor("VERIFY", {"evidence_refs": "e1"}, _ctx())
        assert out["ok"] is False
        assert "claim" in out["reason"]


class TestEndToEndAdapterRun:
    """A 4-hop ``GROUND → CHECK → VERIFY → COMMIT`` skill must run cleanly."""

    def test_protocol_runs_with_full_evidence_roles(
        self, adapter_with_stub_executor
    ) -> None:
        adapter, executor = adapter_with_stub_executor
        skill = _skill(
            protocol=[
                {"action": "GROUND", "payload": {"query": "${target}"}},
                {
                    "action": "CHECK",
                    "payload": {
                        "kind": "COUNT",
                        "value": 1,
                        "label": "${target}",
                        "refs": "e1",
                    },
                },
                {
                    "action": "VERIFY",
                    "payload": {
                        "claim": "yes",
                        "evidence_refs": "e1,d1",
                        "confidence": "high",
                    },
                },
                {
                    "action": "COMMIT",
                    "payload": {
                        "claim": "yes",
                        "evidence_refs": "e1,d1",
                    },
                },
            ],
            expected_roles=["GATHER", "REASON", "VERIFY", "COMMIT"],
        )
        ctx = _ctx(target="red cube")
        result = adapter.run(skill, ctx)

        assert result.success is True
        assert result.contract_satisfied is True
        assert len(result.steps) == 4
        roles = {ev.role for ev in result.new_evidence}
        for r in EVIDENCE_ROLES:
            assert r in roles, f"missing role {r!r} from {roles}"

        # Two derivations: one COUNT (CHECK) + one VERIFY-class row from
        # the COMMIT/VERIFY hops.  The exact count is implementation-
        # dependent, but at least one COUNT and at least one VERIFY row
        # must show up so downstream <derivations> rendering is non-empty.
        kinds = {row.kind for row in executor.derivation_log}
        assert "COUNT" in kinds
        assert "VERIFY" in kinds

    def test_unbound_slot_aborts_run(self, adapter_with_stub_executor) -> None:
        adapter, _ = adapter_with_stub_executor
        skill = _skill(
            protocol=[
                {"action": "GROUND", "payload": {"query": "${target}"}},
            ],
            expected_roles=["GATHER"],
        )
        ctx = _ctx()  # no bindings → ${target} stays literal
        result = adapter.run(skill, ctx)
        assert result.success is False
        assert "unbound slot" in (result.abort_reason or "")
