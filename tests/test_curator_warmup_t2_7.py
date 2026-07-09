"""Tests for T2.7 — curator overfit mitigation (warmup ramp).

Covers:
    1. Default state is no-op identity scaling (weight=1.0, warmup=0).
    2. ``set_curator_warmup`` updates the configured fields.
    3. The ramp returns ``weight * min(1, step / warmup)``.
    4. ``CoEvolutionConfig.curator_weight`` and
       ``CoEvolutionConfig.curator_warmup_steps`` are wired through.
"""

from __future__ import annotations

import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture(autouse=True)
def _reset_curator_warmup_state():
    """Restore the module-level state after each test."""
    from skill_agents.bank_maintenance import llm_curator

    snap = dict(llm_curator._curator_warmup_state)
    yield
    llm_curator._curator_warmup_state.clear()
    llm_curator._curator_warmup_state.update(snap)


# --------------------------------------------------------------------- defaults


def test_default_warmup_state_is_identity() -> None:
    from skill_agents.bank_maintenance.llm_curator import (
        _curator_reward_weight,
        get_curator_warmup_state,
    )

    state = get_curator_warmup_state()
    assert state["weight"] == 1.0
    assert state["warmup_steps"] == 0
    assert state["current_step"] == 0
    assert _curator_reward_weight() == 1.0


# --------------------------------------------------------------------- ramp math


def test_ramp_zero_warmup_returns_full_weight() -> None:
    from skill_agents.bank_maintenance.llm_curator import (
        _curator_reward_weight,
        set_curator_warmup,
    )

    set_curator_warmup(weight=0.5, warmup_steps=0, current_step=42)
    assert _curator_reward_weight() == 0.5


def test_ramp_linear_within_warmup_window() -> None:
    from skill_agents.bank_maintenance.llm_curator import (
        _curator_reward_weight,
        set_curator_warmup,
    )

    set_curator_warmup(weight=1.0, warmup_steps=10, current_step=0)
    assert _curator_reward_weight() == 0.0
    set_curator_warmup(current_step=5)
    assert _curator_reward_weight() == pytest.approx(0.5)
    set_curator_warmup(current_step=10)
    assert _curator_reward_weight() == 1.0
    set_curator_warmup(current_step=20)
    # Saturates at the configured weight (no overshoot).
    assert _curator_reward_weight() == 1.0


def test_ramp_with_weight_below_one_caps_at_weight() -> None:
    from skill_agents.bank_maintenance.llm_curator import (
        _curator_reward_weight,
        set_curator_warmup,
    )

    set_curator_warmup(weight=0.3, warmup_steps=4, current_step=0)
    assert _curator_reward_weight() == 0.0
    set_curator_warmup(current_step=2)
    assert _curator_reward_weight() == pytest.approx(0.15)
    set_curator_warmup(current_step=4)
    assert _curator_reward_weight() == pytest.approx(0.3)
    set_curator_warmup(current_step=99)
    assert _curator_reward_weight() == pytest.approx(0.3)


def test_set_curator_warmup_partial_update() -> None:
    from skill_agents.bank_maintenance.llm_curator import (
        get_curator_warmup_state,
        set_curator_warmup,
    )

    set_curator_warmup(weight=0.7, warmup_steps=20)
    set_curator_warmup(current_step=10)
    s = get_curator_warmup_state()
    assert s["weight"] == 0.7
    assert s["warmup_steps"] == 20
    assert s["current_step"] == 10


# --------------------------------------------------------------------- config wiring


def test_coevolution_config_has_curator_weight_knobs() -> None:
    from trainer.coevolution.config import CoEvolutionConfig

    cfg = CoEvolutionConfig()
    assert hasattr(cfg, "curator_weight")
    assert hasattr(cfg, "curator_warmup_steps")
    # Defaults preserve pre-T2.7 behaviour.
    assert cfg.curator_weight == 1.0
    assert cfg.curator_warmup_steps == 0


def test_coevolution_config_overrides() -> None:
    from trainer.coevolution.config import CoEvolutionConfig

    cfg = CoEvolutionConfig(
        curator_weight=0.4,
        curator_warmup_steps=64,
    )
    assert cfg.curator_weight == 0.4
    assert cfg.curator_warmup_steps == 64
