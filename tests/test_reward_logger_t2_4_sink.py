"""Tests for the T2.4 single-sink reward logger.

Covers:
    1. ``RewardLogger.log_grpo_record`` writes a per-step entry with
       ``kind="grpo_step"`` to the JSONL log.
    2. The new method does not interfere with ``log_episode``: both
       kinds coexist in the same JSONL file, kind-discriminated.
    3. ``CoEvolutionConfig.reward_log_path`` auto-resolves under
       ``rewards_dir`` after ``resolve_paths()``.
    4. ``run_episode_async`` ignores ``reward_logger=None`` (the
       backward-compat path).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from harness.reward_logger import GRPOStepLogEntry, RewardLogger


# --------------------------------------------------------------------- log_grpo_record


def test_log_grpo_record_writes_per_step_entry(tmp_path) -> None:
    log_path = tmp_path / "reward_log.jsonl"
    rl = RewardLogger(str(log_path))
    entry = rl.log_grpo_record(
        episode_id="ep-1",
        adapter="action_taking",
        step=4,
        reward=1.5,
        game="tetris",
        metadata={"chosen_action": "DOWN", "raw_env_reward": 0.5},
    )
    assert isinstance(entry, GRPOStepLogEntry)
    assert entry.adapter == "action_taking"
    assert entry.step == 4
    assert entry.reward == 1.5
    assert entry.metadata["chosen_action"] == "DOWN"

    # In-memory typed read.
    grpo_entries = rl.grpo_step_entries()
    assert len(grpo_entries) == 1
    assert grpo_entries[0].kind == "grpo_step"

    # JSONL on disk has one row, kind-discriminated.
    rows = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(rows) == 1
    j = json.loads(rows[0])
    assert j["kind"] == "grpo_step"
    assert j["episode_id"] == "ep-1"
    assert j["adapter"] == "action_taking"
    assert j["step"] == 4
    assert j["reward"] == 1.5


def test_log_grpo_record_preserves_log_episode(tmp_path) -> None:
    """The two kinds coexist in one JSONL, kind-discriminated."""
    from common.enums import SkillType
    from data_structure.extensions.skill_episode import (
        SkillEpisode,
        SkillEpisodeOutcome,
    )

    log_path = tmp_path / "reward_log.jsonl"
    rl = RewardLogger(str(log_path))

    rl.log_grpo_record(
        episode_id="ep-1",
        adapter="action_taking",
        step=0,
        reward=0.1,
    )
    ep = SkillEpisode.begin(
        skill_id="sk-1",
        skill_version="v1",
        skill_type=SkillType.ACTION,
        domain="gymv",
        parent_run_id=None,
    )
    ep.outcome = SkillEpisodeOutcome(success=True, contract_satisfied=True, score=0.7)
    rl.log_episode(ep)
    rl.log_grpo_record(
        episode_id="ep-1",
        adapter="skill_selection",
        step=0,
        reward=0.4,
    )

    # Typed-read separation.
    grpo_only = rl.grpo_step_entries()
    episodes_only = rl.entries()
    assert len(grpo_only) == 2
    assert len(episodes_only) == 1
    assert all(e.kind == "grpo_step" for e in grpo_only)
    assert all(e.kind == "skill_episode" for e in episodes_only)

    # All three rows share the JSONL, in write order.
    rows = [json.loads(r) for r in log_path.read_text(encoding="utf-8").strip().split("\n")]
    assert len(rows) == 3
    assert [r["kind"] for r in rows] == ["grpo_step", "skill_episode", "grpo_step"]


# --------------------------------------------------------------------- config wiring


def test_coevolution_config_resolves_reward_log_path(tmp_path) -> None:
    """T2.4: ``reward_log_path`` auto-places under ``rewards_dir`` when
    left at the default empty string.
    """
    from trainer.coevolution.config import CoEvolutionConfig

    cfg = CoEvolutionConfig(
        run_dir=str(tmp_path / "run"),
    )
    cfg.resolve_paths()
    assert cfg.reward_log_path != ""
    assert cfg.reward_log_path.endswith(os.path.join("rewards", "reward_log.jsonl"))
    assert cfg.reward_log_path.startswith(str(tmp_path / "run"))


def test_coevolution_config_explicit_relative_reward_log_path(tmp_path) -> None:
    from trainer.coevolution.config import CoEvolutionConfig

    cfg = CoEvolutionConfig(
        run_dir=str(tmp_path / "run"),
        reward_log_path="custom_rewards/log.jsonl",
    )
    cfg.resolve_paths()
    assert cfg.reward_log_path == os.path.join(
        str(tmp_path / "run"), "custom_rewards", "log.jsonl"
    )


# --------------------------------------------------------------------- backward compat


def test_run_episode_async_signature_includes_reward_logger() -> None:
    """``reward_logger`` must be a keyword-only kwarg with a None default."""
    import inspect

    from trainer.coevolution.episode_runner import run_episode_async

    sig = inspect.signature(run_episode_async)
    assert "reward_logger" in sig.parameters
    p = sig.parameters["reward_logger"]
    assert p.default is None
    assert p.kind == inspect.Parameter.KEYWORD_ONLY


def test_collect_rollouts_signature_includes_reward_logger() -> None:
    import inspect

    from trainer.coevolution.rollout_collector import collect_rollouts

    sig = inspect.signature(collect_rollouts)
    assert "reward_logger" in sig.parameters
    p = sig.parameters["reward_logger"]
    assert p.default is None
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
