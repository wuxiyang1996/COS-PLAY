"""Regression tests for the Day-3 `parse_schema_canonical` extension.

PLAN-HARNESS §22 (Day-1 design note → Day-3 implementation): the parser
gap on the `<attributes>` block was the load-bearing reason
`state.facts` was empty for everything except `goal`. The harness's
gymv success_fn cannot evaluate predicates like
`cumulative_reward_increased` or `entity_value_increased` without those
attribute values.

These tests pin the new contract:

  facts["score"]              ← entity labelled "score" → value
  facts["highest_tile"]       ← entity labelled "highest_tile" → value
  facts["entity_attrs"]       ← {label → {field → value}}
  facts["entity_label_count"] ← {label → count}
  facts["phase"]              ← <state_flags>.phase
  facts["progress"]           ← <state_flags>.progress
  facts["goal"]               ← preserved from pre-Day-3 contract

Plus back-compat: schemas without `<attributes>` (e.g. cold-start dumps
made before that block was emitted) still produce a valid StateSchema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labeling_supplement._harness_io_helpers import (
    parse_schema_canonical,
    parse_step_state,
)


_SAMPLE_2048 = """<state>
domain=gymv
task=make_gaming_env/twenty_forty_eight
goal=Play 2048 on a 4x4 grid
step=0

<entities>
e1[type=region, label=board, ontology=container_entity]
e2[type=object, label=tile_2, ontology=selectable_entity]
e3[type=object, label=tile_2, ontology=selectable_entity]
e4[type=region, label=empty_cells, ontology=navigable_region]
e5[type=text, label=highest_tile, ontology=goal_indicator]
e6[type=text, label=score, ontology=goal_indicator]

<attributes>
e1.state=visible
e2.state=visible
e2.value=2
e3.state=visible
e3.value=2
e4.value=14
e5.value=2
e6.value=0

<state_flags>
phase=play
progress=0.05
</state>"""


_LEGACY_NO_ATTRIBUTES = """<state>
domain=gymv
task=tetris
goal=Stack pieces

<entities>
e1[type=object, label=active_piece]
</state>"""


def test_attributes_block_promotes_hot_path_scalars() -> None:
    s = parse_schema_canonical(_SAMPLE_2048)
    assert s.facts.get("score") == 0
    assert s.facts.get("highest_tile") == 2
    # Numeric values must be parsed (not strings).
    assert isinstance(s.facts["score"], int)
    assert isinstance(s.facts["highest_tile"], int)


def test_state_flags_block_promotes_phase_and_progress() -> None:
    s = parse_schema_canonical(_SAMPLE_2048)
    assert s.facts.get("phase") == "play"
    assert s.facts.get("progress") == pytest.approx(0.05)


def test_entity_attrs_index_keyed_by_label() -> None:
    s = parse_schema_canonical(_SAMPLE_2048)
    attrs = s.facts.get("entity_attrs") or {}
    assert "tile_2" in attrs
    # The schema repeats `tile_2` across e2 / e3; last write wins on the
    # label-keyed index, but both should at least surface a numeric value.
    assert isinstance(attrs["tile_2"].get("value"), int)
    assert attrs["highest_tile"].get("value") == 2


def test_entity_label_count_counts_repeats() -> None:
    s = parse_schema_canonical(_SAMPLE_2048)
    counts = s.facts.get("entity_label_count") or {}
    assert counts.get("tile_2") == 2
    assert counts.get("board") == 1
    assert counts.get("score") == 1


def test_legacy_schema_without_attributes_still_parses() -> None:
    """Back-compat: when the `<attributes>` block is missing the parser
    must NOT raise and must NOT fabricate hot-path scalars."""

    s = parse_schema_canonical(_LEGACY_NO_ATTRIBUTES)
    assert s.domain == "gymv"
    assert s.task == "tetris"
    assert "score" not in s.facts
    assert "highest_tile" not in s.facts
    # entity_label_count is built off entities, so it should still exist.
    assert (s.facts.get("entity_label_count") or {}).get("active_piece") == 1


def test_parse_step_state_flows_through() -> None:
    """`parse_step_state` should pick `metadata.schema_canonical` and
    delegate to `parse_schema_canonical`, so the new fields are
    visible to the harness io helpers without further plumbing."""

    step = {"metadata": {"schema_canonical": _SAMPLE_2048}}
    s = parse_step_state(step)
    assert s.facts.get("score") == 0
    assert s.facts.get("phase") == "play"
    assert "entity_label_count" in s.facts


# ───────────── property-style guard against the real corpus ─────────────


def test_real_2048_episode_fields() -> None:
    """Property-style smoke against the real cold-start corpus to make
    sure the parser doesn't choke on any quirk we missed."""

    ep_path = Path(
        "/workspace/Multi-hop-Reasoning-VLM-Agent"
        "/labeling/skill_actions_out/run_20260430_064325"
        "/env_wrappers/twenty_forty_eight/episode_000.json"
    )
    if not ep_path.exists():
        pytest.skip(f"corpus missing: {ep_path}")
    ep = json.loads(ep_path.read_text())
    seen_score = False
    for step in ep.get("experiences", [])[:8]:
        s = parse_step_state(step)
        assert s.domain == "gymv"
        assert s.task.endswith("twenty_forty_eight")
        if isinstance(s.facts.get("score"), int):
            seen_score = True
            assert s.facts["score"] >= 0
    assert seen_score, "no parseable score fact across first 8 steps"
