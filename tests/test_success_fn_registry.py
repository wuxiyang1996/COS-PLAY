"""Day-6: tests for the domain-keyed `SuccessFn` registry in
`harness.gymv_success`.

The registry lets `FewShotAdapter` look up a domain-specific scorer
(by `target_domain`) instead of being fixed at construction time. This
test pins:

  * `gymv` is auto-registered to `make_per_step_success_fn` at import.
  * `success_fn_for_domain("gymv")` returns a working scorer.
  * `success_fn_for_domain(<unknown>)` falls back to
    `default_success_fn`.
  * `register_success_fn(...)` overwrites cleanly.
  * `FewShotAdapter` consults the registry when constructed with the
    default scorer; an explicit scorer override still wins.
"""
from __future__ import annotations

from typing import Any

import pytest

from common.enums import SkillType
from common.state_schema import StateSchema
from data_structure.extensions.skill_episode import SkillEpisode, SkillEpisodeOutcome
from harness.few_shot_adapter import (
    FewShotAdapter,
    FewShotDemo,
    default_success_fn,
)
from harness.gymv_success import (
    register_success_fn,
    registered_success_fn_domains,
    success_fn_for_domain,
)


def _stub_episode(success: bool = True) -> SkillEpisode:
    out = SkillEpisodeOutcome(
        success=success,
        contract_satisfied=success,
        score=1.0 if success else 0.0,
        abort_reason=None,
    )
    return SkillEpisode(
        episode_id="ep-1",
        skill_id="stub",
        skill_version="v1",
        skill_type=SkillType.ACTION,
        domain="gymv",
        parent_run_id=None,
        steps=[],
        outcome=out,
        initial_state={},
        cost={"tokens": 0.0, "ms": 0.0},
    )


def test_gymv_is_registered_at_import() -> None:
    domains = registered_success_fn_domains()
    assert "gymv" in domains


def test_success_fn_for_domain_gymv_runs() -> None:
    fn = success_fn_for_domain("gymv")
    # Stub episode with no per_hop_effects → the gymv scorer falls
    # back to outcome.success (`require_episode_success=True`).
    ep = _stub_episode(success=True)
    score = fn(ep, FewShotDemo(state=StateSchema(domain="gymv", task="x")))
    assert score == 1.0


def test_success_fn_for_domain_unknown_falls_back_to_default() -> None:
    """An unregistered target domain falls back to
    `default_success_fn` (success ⇔ outcome.success +
    contract_satisfied).

    Phase-5 Stages 1-4 registered scorers for ``visual_reasoning`` /
    ``video`` / ``osworld`` / ``browser``, so this test now uses a
    deliberately-unknown name (``__no_such_domain__``) to keep
    asserting the registry's fallback contract.
    """
    fn = success_fn_for_domain("__no_such_domain__")
    assert fn is default_success_fn


def test_register_success_fn_overwrites() -> None:
    sentinel = lambda *a, **kw: 0.5  # noqa: E731
    register_success_fn("test_domain", lambda **_kw: sentinel)
    fn = success_fn_for_domain("test_domain")
    assert fn is sentinel
    # Re-registering replaces the previous factory.
    new_sentinel = lambda *a, **kw: 0.25  # noqa: E731
    register_success_fn("test_domain", lambda **_kw: new_sentinel)
    fn2 = success_fn_for_domain("test_domain")
    assert fn2 is new_sentinel


def test_few_shot_adapter_consults_registry_on_default_scorer() -> None:
    """When `FewShotAdapter` is built with no explicit scorer (=>
    `default_success_fn`), `adapt(target_domain="gymv", …)` should
    swap in the registered gymv scorer."""

    captured = {"called_with": None}

    def fake_gymv_factory(**_kw):
        def _scorer(episode: SkillEpisode, demo: Any) -> float:
            captured["called_with"] = (episode.episode_id, getattr(demo, "notes", ""))
            return 1.0
        return _scorer

    # Save and restore the production factory.
    from harness.gymv_success import _DOMAIN_SUCCESS_FN_FACTORIES
    original = _DOMAIN_SUCCESS_FN_FACTORIES.get("gymv")
    register_success_fn("gymv", fake_gymv_factory)
    try:
        # Spin up a minimal harness mock that just records the run.
        class _StubHarness:
            adapter_registry = type("R", (), {
                "get": staticmethod(lambda *a, **kw: object()),
            })()

            def run_skill(self, skill, state, parent_run_id=None, bindings=None):
                return _stub_episode(success=True)

        from common.enums import SkillSourceType, SkillStatus
        from data_structure.extensions.skill_record import SkillRecord
        skill = SkillRecord(
            skill_id="s",
            name="s",
            skill_type=SkillType.ACTION,
            source_type=SkillSourceType.MINED,
            status=SkillStatus.PROVISIONAL,
            source_domains=["gymv"],
            feasible_domains=["gymv"],
        )
        adapter = FewShotAdapter(harness=_StubHarness())
        demo = FewShotDemo(
            state=StateSchema(domain="gymv", task="tetris"),
            notes="trip-wire",
        )
        r = adapter.adapt(
            skill=skill,
            target_domain="gymv",
            demos=[demo],
            target_task="tetris",
        )
        assert r.n_success == 1
        assert captured["called_with"] is not None
        assert captured["called_with"][1] == "trip-wire"
    finally:
        if original is not None:
            register_success_fn("gymv", original)


def test_few_shot_adapter_explicit_scorer_overrides_registry() -> None:
    """Passing `success_fn=...` to `FewShotAdapter()` skips the
    registry — it's an explicit per-instance override."""

    def explicit_scorer(_episode: SkillEpisode, _demo: Any) -> float:
        return 0.0  # always fail

    class _StubHarness:
        adapter_registry = type("R", (), {
            "get": staticmethod(lambda *a, **kw: object()),
        })()

        def run_skill(self, skill, state, parent_run_id=None, bindings=None):
            return _stub_episode(success=True)

    from common.enums import SkillSourceType, SkillStatus
    from data_structure.extensions.skill_record import SkillRecord
    skill = SkillRecord(
        skill_id="s",
        name="s",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.MINED,
        status=SkillStatus.PROVISIONAL,
        source_domains=["gymv"],
        feasible_domains=["gymv"],
    )
    adapter = FewShotAdapter(harness=_StubHarness(), success_fn=explicit_scorer)
    demo = FewShotDemo(state=StateSchema(domain="gymv", task="tetris"))
    r = adapter.adapt(
        skill=skill, target_domain="gymv", demos=[demo], target_task="tetris",
    )
    # Even though the gymv scorer would have returned 1.0, our
    # explicit override returns 0.0 → no successes.
    assert r.n_success == 0
