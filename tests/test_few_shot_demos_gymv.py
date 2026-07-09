"""Tests for `harness.few_shot_demos_gymv`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.few_shot_demos_gymv import (
    build_demos_from_episode_file,
    build_demos_from_episodes,
)


def _write_episode(tmp_path: Path, *, game: str, name: str, exps: list) -> Path:
    p = tmp_path / "env_wrappers" / game / name
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_id": name,
        "env_name": f"make_gaming_env/{game}",
        "game_name": game,
        "experiences": exps,
        "task": f"make_gaming_env/{game}",
    }
    p.write_text(json.dumps(payload))
    return p


def _exp(*, schema: str, action: str, reward: float, idx: int = 0,
         is_noop: bool = False) -> dict:
    return {
        "step_id": idx,
        "action": action,
        "reward": reward,
        "metadata": {
            "schema_canonical": schema,
            "is_noop": is_noop,
        },
    }


_SC_TETRIS = """<state>
domain=gymv
task=make_gaming_env/tetris
goal=Survive
step=0

<entities>
e1[type=region, label=board, bid=null, pos=0,0, size=10x20]
e2[type=goal_indicator, label=score, bid=null, pos=null, size=null]

<attributes>
e2.value=0
e2.kind=count

<state_flags>
phase=playing
</state>
"""


def test_build_demos_from_episode_file_skips_noop(tmp_path: Path) -> None:
    ep = _write_episode(tmp_path, game="tetris", name="episode_000.json", exps=[
        _exp(schema=_SC_TETRIS, action="left", reward=1.0, idx=0, is_noop=True),
        _exp(schema=_SC_TETRIS, action="right", reward=2.0, idx=1, is_noop=False),
        _exp(schema=_SC_TETRIS, action="hard_drop", reward=4.0, idx=2),
    ])
    demos = build_demos_from_episode_file(ep, game="tetris", max_demos=5)
    actions = [d.bindings["direction"] for d in demos]
    assert actions == ["right", "hard_drop"]
    assert demos[0].state.task == "make_gaming_env/tetris"
    assert demos[0].state.facts.get("score") == 0.0
    assert demos[0].expected["reward"] == 2.0


def test_build_demos_from_episode_file_caps_at_max(tmp_path: Path) -> None:
    ep = _write_episode(tmp_path, game="tetris", name="episode_000.json", exps=[
        _exp(schema=_SC_TETRIS, action="left", reward=0.0, idx=i)
        for i in range(5)
    ])
    demos = build_demos_from_episode_file(ep, game="tetris", max_demos=2)
    assert len(demos) == 2


def test_build_demos_from_episode_file_skips_invalid_schema(tmp_path: Path) -> None:
    ep = _write_episode(tmp_path, game="tetris", name="episode_000.json", exps=[
        _exp(schema="(no schema block)", action="left", reward=0.0, idx=0),
        _exp(schema=_SC_TETRIS, action="right", reward=1.0, idx=1),
    ])
    demos = build_demos_from_episode_file(ep, game="tetris", max_demos=5)
    # The malformed step is silently dropped; the valid one survives.
    assert len(demos) == 1
    assert demos[0].bindings["direction"] == "right"


def test_build_demos_from_episodes_walks_root(tmp_path: Path) -> None:
    _write_episode(tmp_path, game="tetris", name="episode_000.json", exps=[
        _exp(schema=_SC_TETRIS, action="left", reward=0.0, idx=0),
        _exp(schema=_SC_TETRIS, action="right", reward=1.0, idx=1),
    ])
    _write_episode(tmp_path, game="tetris", name="episode_001.json", exps=[
        _exp(schema=_SC_TETRIS, action="hard_drop", reward=4.0, idx=0),
    ])
    demos = build_demos_from_episodes(
        tmp_path, corpus="env_wrappers", game="tetris",
        max_episodes=5, max_demos_per_episode=2,
    )
    assert len(demos) == 3
    notes = [d.notes for d in demos]
    assert any("episode_000" in n for n in notes)
    assert any("episode_001" in n for n in notes)


def test_build_demos_from_episodes_handles_missing_root(tmp_path: Path) -> None:
    demos = build_demos_from_episodes(
        tmp_path / "does_not_exist", corpus="env_wrappers", game="tetris",
    )
    assert demos == []


def test_build_demos_carries_expected_reward_and_action(tmp_path: Path) -> None:
    ep = _write_episode(tmp_path, game="tetris", name="episode_000.json", exps=[
        _exp(schema=_SC_TETRIS, action="hard_drop", reward=8.0, idx=0),
    ])
    demos = build_demos_from_episode_file(ep, game="tetris", max_demos=1)
    assert demos[0].expected["reward"] == 8.0
    assert demos[0].expected["action"] == "hard_drop"
    assert demos[0].expected["episode"] == "episode_000"
    assert demos[0].bindings["direction"] == "hard_drop"
    assert demos[0].bindings["target"] == "hard_drop"
