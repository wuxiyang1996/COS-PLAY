"""Unit tests for the §5.5 ablation CLI flags (block B).

Each flag is exercised by either invoking the affected component
directly with the relevant kwarg, or by inspecting the parsed CLI
namespace and config that ``run_coevolution.py`` would build.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make ``scripts/`` importable for the CLI parser test.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


# ── B1: --harness-mode ─────────────────────────────────────────────────


def _make_state(task: str = "gymv_columns") -> "StateSchema":
    from common.state_schema import StateSchema
    return StateSchema(domain="gymv", task=task)


def test_b1_plain_text_skills_bypasses_filter():
    """B1: harness_mode='plain-text-skills' → all candidates admitted."""
    from trainer.coevolution._harness_hook import SkillHarnessHook

    h = SkillHarnessHook(domain="gymv", records={})
    h._mode = "plain-text-skills"

    candidates = [
        {"skill_id": "skill-a", "name": "A"},
        {"skill_id": "skill-b", "name": "B"},
    ]
    state = _make_state()
    filtered, diag = h.filter_candidates(candidates, state)
    assert len(filtered) == 2
    assert diag["mode"] == "plain-text-skills"
    assert diag["n_admitted"] == 2


def test_b1_full_mode_runs_eligibility():
    """B1: default 'full' mode does not set the plain-text marker."""
    from trainer.coevolution._harness_hook import SkillHarnessHook

    h = SkillHarnessHook(domain="gymv", records={})
    assert h._mode == "full"

    state = _make_state()
    # Empty bank cache → all candidates "unknown" (passed through unchanged)
    filtered, diag = h.filter_candidates(
        [{"skill_id": "x"}], state,
    )
    assert "mode" not in diag  # only set by plain-text-skills branch


def test_b1_validate_choice_bypasses_in_plain_text_mode():
    """B1: plain-text-skills also bypasses validate_invocation."""
    from trainer.coevolution._harness_hook import SkillHarnessHook

    h = SkillHarnessHook(domain="gymv", records={})
    h._mode = "plain-text-skills"
    state = _make_state()
    ok, d = h.validate_choice("skill-x", state)
    assert ok is True
    assert d["status"] == "plain_text_skills_bypass"


# ── B2: --no-crafter ──────────────────────────────────────────────────


def test_b2_config_default_crafter_enabled():
    from trainer.coevolution.config import CoEvolutionConfig
    cfg = CoEvolutionConfig()
    assert cfg.crafter_enabled is True


def test_b2_config_no_crafter_flag():
    from trainer.coevolution.config import CoEvolutionConfig
    cfg = CoEvolutionConfig(crafter_enabled=False)
    assert cfg.crafter_enabled is False


# ── B3: --promotion-bypass-mode ───────────────────────────────────────


def test_b3_default_gated():
    from trainer.coevolution.config import CoEvolutionConfig
    cfg = CoEvolutionConfig()
    assert cfg.promotion_bypass_mode == "gated"


def test_b3_permissive_evaluation_all_pass():
    """B3 (driver-side): _build_permissive_evaluation returns PASS for every stage."""
    from common.enums import SkillSourceType, SkillType
    from data_structure.extensions.skill_record import (
        SkillContract, SkillRecord,
    )
    from labeling_supplement.decide_promotion_gpt54 import (
        _build_permissive_evaluation,
    )

    rec = SkillRecord.new(
        name="s1",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.MINED,
        feasible_domains=["gymv"],
        feasible_tasks=[],
        protocol=[{"action": "EXEC", "payload": {}, "notes": "step"}],
        contract=SkillContract(),
    )

    # Minimal stub proposal — _build_permissive_evaluation only reads
    # ``proposal.proposal_id``.
    proposal = MagicMock()
    proposal.proposal_id = "p1"

    ev = _build_permissive_evaluation(
        proposal=proposal, skill=rec, judge_model="test-model",
    )
    assert ev.verdict.final_verdict.value == "pass"
    assert all(s.verdict.value == "pass" for s in ev.verdict.stages)


# ── B4: --intention-trigger ───────────────────────────────────────────


def test_b4_default_trigger_every_step():
    from trainer.coevolution.config import CoEvolutionConfig
    cfg = CoEvolutionConfig()
    # 'every-step' matches historical pre-block-B behaviour;
    # changing the default would silently mutate baseline numbers.
    assert cfg.intention_trigger == "every-step"


def test_b4_episode_runner_kwarg_signature():
    """B4: run_episode_async accepts intention_trigger kwarg."""
    import inspect
    from trainer.coevolution.episode_runner import run_episode_async
    sig = inspect.signature(run_episode_async)
    assert "intention_trigger" in sig.parameters
    assert sig.parameters["intention_trigger"].default == "every-step"


# ── B5: --actor-bank-cap-K ────────────────────────────────────────────


def test_b5_default_no_cap():
    from trainer.coevolution.config import CoEvolutionConfig
    cfg = CoEvolutionConfig()
    assert cfg.actor_bank_cap_k == 0


def test_b5_query_engine_select_signature_has_bank_cap():
    """B5: SkillQueryEngine.select() accepts bank_cap_k kwarg."""
    import inspect
    from skill_agents.query import SkillQueryEngine
    sig = inspect.signature(SkillQueryEngine.select)
    assert "bank_cap_k" in sig.parameters
    assert sig.parameters["bank_cap_k"].default == 0


def test_b5_get_top_k_skill_candidates_signature():
    import inspect
    from scripts.qwen3_decision_agent import get_top_k_skill_candidates
    sig = inspect.signature(get_top_k_skill_candidates)
    assert "bank_cap_k" in sig.parameters
    assert sig.parameters["bank_cap_k"].default == 0


# ── End-to-end CLI parser check ────────────────────────────────────────


def test_cli_accepts_all_block_b_flags(monkeypatch):
    """Every block-B flag is recognised by ``run_coevolution.py`` argparse."""
    import scripts.run_coevolution as rc

    test_argv = [
        "run_coevolution.py",
        "--total-steps", "1",
        "--games", "gymv_columns",
        "--harness-mode", "plain-text-skills",
        "--no-crafter",
        "--promotion-bypass-mode", "permissive",
        "--intention-trigger", "sharp-shift",
        "--actor-bank-cap-k", "10",
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    # Lift the parser from main() so we don't kick off any training.
    parser = argparse.ArgumentParser()
    # Easiest: re-import and just call _build_parser if it exists; else
    # build the parser by reaching into rc.
    if hasattr(rc, "_build_parser"):
        parser = rc._build_parser()
    else:
        # main() builds the parser locally — we can monkey-patch
        # ``parser.parse_args`` to be a no-op.
        parser = None

    if parser is not None:
        args = parser.parse_args(test_argv[1:])
        assert args.harness_mode == "plain-text-skills"
        assert args.crafter_enabled is False
        assert args.promotion_bypass_mode == "permissive"
        assert args.intention_trigger == "sharp-shift"
        assert args.actor_bank_cap_k == 10
    else:
        pytest.skip("run_coevolution.py does not expose _build_parser")
