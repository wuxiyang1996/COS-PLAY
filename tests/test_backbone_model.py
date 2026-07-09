"""Backbone model invariant test.

Pins the project-wide three-tier backbone stack (current phase) and
verifies every key surface defaults to the right tier:

* Actor + Skill-Bank → ``Qwen/Qwen3.5-9B`` (``BACKBONE_MODEL``)
* Crafter / Harness / Orchestrator + LLM-as-judge →
  ``Qwen/Qwen3.5-35B-A3B`` (``BACKBONE_TEACHER_MODEL`` and
  ``BACKBONE_JUDGE_MODEL`` — same weights, different roles, see
  ``common/models.py`` for the rationale).
* SFT cold-start data generation → ``gpt-5.5``
  (``BACKBONE_SFT_TEACHER_MODEL``) — kept on the frontier model
  because cold-start labels are baked once into SFT adapters and
  never re-run during training.

The 8B / 32B / 72B Qwen tracks remain deferred and must not appear as a
*runtime default* anywhere we control.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.models import (
    BACKBONE_JUDGE_MODEL,
    BACKBONE_MODEL,
    BACKBONE_SFT_TEACHER_MODEL,
    BACKBONE_TEACHER_MODEL,
    DEFERRED_MODELS,
    assert_default_backbone,
    is_deferred,
)


ACTOR_MODEL = "Qwen/Qwen3.5-9B"
CONTROL_PLANE_MODEL = "Qwen/Qwen3.5-35B-A3B"
SFT_TEACHER_MODEL = "gpt-5.5"
# Off-distribution oracle used when overriding the judge for paper /
# formal eval runs where within-Qwen-family bias must be controlled.
# Never a runtime default — kept here only for the override-path test.
OFF_DISTRIBUTION_JUDGE = "gpt-5.5"


class TestBackboneModelDefaults:
    def test_backbone_is_qwen35_9b(self) -> None:
        """Actor + skill-bank policy backbone."""
        assert BACKBONE_MODEL == ACTOR_MODEL, BACKBONE_MODEL
        assert_default_backbone()

    def test_teacher_is_qwen35_35b_a3b(self) -> None:
        """Crafter / harness / orchestrator control-plane backbone."""
        assert BACKBONE_TEACHER_MODEL == CONTROL_PLANE_MODEL, BACKBONE_TEACHER_MODEL

    def test_judge_is_qwen35_35b_a3b(self) -> None:
        """LLM-as-judge for the eval driver — defaults to the local
        35B-A3B teacher backbone (same weights as
        ``BACKBONE_TEACHER_MODEL``, different role).  Eliminates judge
        API spend at the cost of a within-Qwen-family bias the
        spot-check protocol in ``common/models.py`` covers."""
        assert BACKBONE_JUDGE_MODEL == CONTROL_PLANE_MODEL, BACKBONE_JUDGE_MODEL

    def test_sft_teacher_is_gpt5_5(self) -> None:
        """Cold-start data generation teacher (feeds ``trainer/SFT``).

        Stays on the frontier model because cold-start labels are baked
        once into SFT adapters and never re-run during training; paying
        API cost once for a stronger teacher is the right trade.
        """
        assert BACKBONE_SFT_TEACHER_MODEL == SFT_TEACHER_MODEL, BACKBONE_SFT_TEACHER_MODEL

    def test_actor_distinct_from_teacher_and_judge(self) -> None:
        """Actor must be distinct from the control plane / judge (the
        actor is LoRA-trained, the others are frozen)."""
        assert BACKBONE_MODEL != BACKBONE_TEACHER_MODEL
        assert BACKBONE_MODEL != BACKBONE_JUDGE_MODEL

    def test_teacher_and_judge_share_backbone(self) -> None:
        """Pin the design decision (2026-05-03) that judge and teacher
        share the same 35B-A3B backbone weights so a single vLLM server
        can satisfy both roles. If you bump one, bump the other or
        explicitly diverge them and update this test with the
        rationale."""
        assert BACKBONE_TEACHER_MODEL == BACKBONE_JUDGE_MODEL == CONTROL_PLANE_MODEL

    def test_deferred_models_are_not_default(self) -> None:
        assert BACKBONE_MODEL not in DEFERRED_MODELS
        assert BACKBONE_TEACHER_MODEL not in DEFERRED_MODELS
        assert BACKBONE_JUDGE_MODEL not in DEFERRED_MODELS
        assert BACKBONE_SFT_TEACHER_MODEL not in DEFERRED_MODELS
        assert is_deferred("Qwen/Qwen3-8B")
        assert is_deferred("Qwen/Qwen2.5-72B")
        assert not is_deferred(ACTOR_MODEL)
        assert not is_deferred(CONTROL_PLANE_MODEL)


class TestOrchestratorConfigUsesBackbone:
    def test_teacher_config_default(self) -> None:
        from orchestrator.config import TeacherConfig

        # Crafter teacher is the frozen control-plane backbone.
        assert TeacherConfig().model_name == BACKBONE_TEACHER_MODEL == CONTROL_PLANE_MODEL

    def test_judge_config_default(self) -> None:
        from orchestrator.config import JudgeConfig

        assert JudgeConfig().model_name == BACKBONE_JUDGE_MODEL == CONTROL_PLANE_MODEL

    def test_orchestrator_config_default(self) -> None:
        from orchestrator.config import OrchestratorConfig

        cfg = OrchestratorConfig()
        # ``backbone_model`` is the actor/policy default.
        assert cfg.backbone_model == BACKBONE_MODEL == ACTOR_MODEL
        # Teacher pulls from the control-plane backbone.
        assert cfg.teacher.model_name == BACKBONE_TEACHER_MODEL == CONTROL_PLANE_MODEL
        # Judge shares the 35B-A3B control-plane backbone (same weights,
        # different role) — see ``common/models.py``.
        assert cfg.judge.model_name == BACKBONE_JUDGE_MODEL == CONTROL_PLANE_MODEL


class TestCrafterUsesBackbone:
    def test_crafter_service_teacher_default(self, tmp_path) -> None:
        from crafter import SkillCrafterService
        from orchestrator import ArtifactStore
        from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore
        from skill_bank.stores import StoreName

        repo = SkillRepository(
            draft_store=SkillStore(StoreName.DRAFT, str(tmp_path / "draft")),
            candidate_store=SkillStore(StoreName.CANDIDATE, str(tmp_path / "candidate")),
            active_store=SkillStore(StoreName.ACTIVE, str(tmp_path / "active")),
            archive_store=SkillStore(StoreName.ARCHIVE, str(tmp_path / "archive")),
        )
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(lifecycle=lifecycle, artifact_store=artifacts)
        # The crafter holds the canonical teacher model on its private slot;
        # by default this is ``BACKBONE_TEACHER_MODEL`` (Qwen3.5-35B-A3B).
        assert crafter._teacher == BACKBONE_TEACHER_MODEL  # noqa: SLF001


class TestDecisionAgentDefaults:
    """Ensure live decision-agent surfaces also default to the actor
    backbone (``Qwen/Qwen3.5-9B``).

    The legacy ``decision_agents`` package has optional heavy deps
    (``google.genai``, ``vllm``, …) so we check the source text rather
    than the imported module — the check stays meaningful in lean test
    environments and remains tied to the literal that ships.
    """

    def test_actor_agent_default(self) -> None:
        path = os.path.join(_ROOT, "decision_agents", "actor_agent.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        assert 'DEFAULT_MODEL: str = "Qwen/Qwen3.5-9B"' in src, (
            "decision_agents.actor_agent.DEFAULT_MODEL is not Qwen/Qwen3.5-9B; "
            "see common/models.py for the project-wide actor backbone."
        )

    def test_agent_helper_default(self) -> None:
        path = os.path.join(_ROOT, "decision_agents", "agent_helper.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        assert 'DEFAULT_LLM_MODEL: str = "Qwen/Qwen3.5-9B"' in src

    def test_vlm_decision_agent_default(self) -> None:
        path = os.path.join(_ROOT, "decision_agents", "agent.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        assert 'DEFAULT_MODEL = "Qwen/Qwen3.5-9B"' in src


class TestApiFuncRoutingDefault:
    """The central LLM router must default to the actor backbone
    (``Qwen/Qwen3.5-9B``) when called with ``model=None``."""

    def test_ask_model_defaults_to_actor_backbone(self) -> None:
        path = os.path.join(_ROOT, "API_func.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        assert 'model = "Qwen/Qwen3.5-9B"' in src, (
            "API_func.ask_model lost its Qwen/Qwen3.5-9B default."
        )


class TestEnvOverridePath:
    """Confirm the ``VLM_AGENT_BACKBONE_MODEL`` override path is honored
    at import time without breaking when unset."""

    def test_override_present_when_env_set(self, monkeypatch) -> None:
        monkeypatch.setenv("VLM_AGENT_BACKBONE_MODEL", "Qwen/Qwen3.5-9B-Instruct")
        import importlib

        import common.models as m

        importlib.reload(m)
        try:
            assert m.BACKBONE_MODEL == "Qwen/Qwen3.5-9B-Instruct"
        finally:
            monkeypatch.delenv("VLM_AGENT_BACKBONE_MODEL", raising=False)
            importlib.reload(m)

    def test_judge_override(self, monkeypatch) -> None:
        """Judge default is now ``Qwen/Qwen3.5-35B-A3B``;
        ``VLM_AGENT_BACKBONE_JUDGE_MODEL`` is the documented escape
        hatch back to an off-distribution oracle (e.g. ``gpt-5.5``)
        for paper / formal eval runs."""
        monkeypatch.setenv("VLM_AGENT_BACKBONE_JUDGE_MODEL", "gpt-5.5")
        import importlib

        import common.models as m

        importlib.reload(m)
        try:
            assert m.BACKBONE_JUDGE_MODEL == "gpt-5.5"
        finally:
            monkeypatch.delenv("VLM_AGENT_BACKBONE_JUDGE_MODEL", raising=False)
            importlib.reload(m)


class TestSkillEvalJudgeWiring:
    """``LLMJudgeConfig`` must default to the canonical
    ``BACKBONE_JUDGE_MODEL`` (not ``None``, which silently falls back to
    the 9B actor and produces a self-judging pathology). The
    ``VLM_AGENT_BACKBONE_JUDGE_MODEL`` override must propagate end-to-end
    so a single env var swaps the judge from the default local 35B-A3B
    to an off-distribution oracle (gpt-5.5) without code changes —
    see ``.env.example`` and ``skill_agents/skill_evaluation/config.py``.
    """

    def test_default_judge_is_canonical_backbone(self) -> None:
        from skill_agents.skill_evaluation.config import LLMJudgeConfig

        assert LLMJudgeConfig().model == BACKBONE_JUDGE_MODEL == CONTROL_PLANE_MODEL

    def test_judge_override_to_off_distribution_oracle_skill_eval(
        self, monkeypatch,
    ) -> None:
        monkeypatch.setenv(
            "VLM_AGENT_BACKBONE_JUDGE_MODEL", OFF_DISTRIBUTION_JUDGE,
        )
        import importlib

        import common.models as m
        import skill_agents.skill_evaluation.config as cfg

        importlib.reload(m)
        importlib.reload(cfg)
        try:
            assert m.BACKBONE_JUDGE_MODEL == OFF_DISTRIBUTION_JUDGE
            assert cfg.LLMJudgeConfig().model == OFF_DISTRIBUTION_JUDGE
        finally:
            monkeypatch.delenv("VLM_AGENT_BACKBONE_JUDGE_MODEL", raising=False)
            importlib.reload(m)
            importlib.reload(cfg)

    def test_judge_override_to_off_distribution_oracle_orchestrator(
        self, monkeypatch,
    ) -> None:
        monkeypatch.setenv(
            "VLM_AGENT_BACKBONE_JUDGE_MODEL", OFF_DISTRIBUTION_JUDGE,
        )
        import importlib

        import common.models as m
        import orchestrator.config as ocfg

        importlib.reload(m)
        importlib.reload(ocfg)
        try:
            assert ocfg.JudgeConfig().model_name == OFF_DISTRIBUTION_JUDGE
        finally:
            monkeypatch.delenv("VLM_AGENT_BACKBONE_JUDGE_MODEL", raising=False)
            importlib.reload(m)
            importlib.reload(ocfg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
