"""End-to-end ActorAgent ↔ VRHarness tests.

The unified single-MDP migration moves the legacy inner-MDP operators
(``GROUND / RETRIEVE / CONCLUDE / ANSWER``) into the
:class:`VRHarness` action vocabulary.  These tests confirm the actor
binds the harness, threads its action vocabulary through
:meth:`ActorAgent.step`, and reaches an ``ANSWER`` within the
harness's deliberation budget — with the per-action-kind cost flowing
into ``r_cost`` exactly as
``decision_agents/README.md`` "Migration of inner-MDP operators"
specifies.

The actor is run with ``ask_model`` patched to ``None`` (see
``conftest.py``), so it falls back to the deterministic "first valid
action" path.  The first action in :meth:`VRHarness.valid_actions` is
``ANSWER("<text>")``, which means a one-step rollout is enough to hit
``done=True`` and exercise the whole reward path.
"""

from __future__ import annotations

from typing import List

import pytest

from decision_agents.actor_agent import ActorAgent
from decision_agents.core import VRHarness, VisualInput
from decision_agents.reward_func import RewardConfig
from decision_agents.skill_interface import NullSkillProvider


@pytest.fixture
def fake_image() -> VisualInput:
    return VisualInput(image_url="https://example.com/scene.png", caption="scene")


# ══════════════════════════════════════════════════════════════════════
# Single-step end-to-end
# ══════════════════════════════════════════════════════════════════════


class TestActorWithVRHarness:
    def test_step_uses_harness_valid_actions(self, fake_image: VisualInput) -> None:
        h = VRHarness(image=fake_image, question="how many?", gold_answer="3")
        agent = ActorAgent(skill_provider=NullSkillProvider(), harness=h)
        agent.reset()
        h.reset()

        decision = agent.step(observation="how many?", task="VR Q&A")

        # The harness's valid_actions are surfaced into the decision.
        assert decision.valid_actions
        assert decision.valid_actions[0].startswith("ANSWER(")
        # Without an LLM the deterministic fallback picks ``valid[0]``,
        # i.e. an ANSWER action — which is enough to exercise the
        # action_kind plumbing.
        assert decision.action.startswith("ANSWER(")
        assert decision.action_kind == "vr_answer"

    def test_observe_result_charges_action_kind_cost(self, fake_image: VisualInput) -> None:
        # Opt-in: assign a non-zero ANSWER cost so the per-action-kind
        # plumbing actually shows up in r_cost.
        cfg = RewardConfig(vr_answer_cost=-0.25)
        h = VRHarness(image=fake_image, question="how many?", gold_answer="3")
        agent = ActorAgent(
            skill_provider=NullSkillProvider(),
            harness=h,
            reward_config=cfg,
        )
        agent.reset()
        obs, info = h.reset()

        decision = agent.step(observation=obs, task="VR Q&A")
        next_obs, reward, done, env_info = h.step(decision.action)
        rr = agent.observe_result(
            decision,
            reward=reward,
            next_observation=next_obs,
            done=done,
        )

        # ANSWER terminates the harness episode.
        assert done is True
        # ``r_env`` reflects whether the answer matched the gold.  The
        # deterministic fallback emits the placeholder ``"<text>"`` so
        # the gold check fails — r_env is 0.
        assert rr.r_env == pytest.approx(0.0)
        # The opt-in ``vr_answer_cost`` flowed through.
        assert rr.r_cost <= -0.25 + 1e-9

    def test_correct_answer_scores_one(self, fake_image: VisualInput) -> None:
        """A direct harness.step('ANSWER("3")') with matching gold scores +1.

        We bypass the actor's deterministic fallback (which emits the
        prompt placeholder) by feeding the harness a fully-rendered
        ANSWER action directly — the goal here is to lock in the
        harness↔actor reward contract, not the LLM behaviour.
        """
        h = VRHarness(image=fake_image, question="how many?", gold_answer="3")
        agent = ActorAgent(skill_provider=NullSkillProvider(), harness=h)
        agent.reset()
        h.reset()

        # Build a decision out-of-band so we can pass a known-good answer.
        decision = agent.step(observation="how many?", task="VR Q&A")
        # Replace the action with the correct answer for the harness step.
        next_obs, reward, done, info = h.step('ANSWER("3")')

        rr = agent.observe_result(
            decision,
            reward=reward,
            next_observation=next_obs,
            done=done,
        )
        assert done is True
        assert reward == pytest.approx(1.0)
        assert rr.r_env == pytest.approx(1.0)

    def test_max_steps_terminates_within_budget(self, fake_image: VisualInput) -> None:
        """An open-ended VR rollout terminates within the harness budget.

        With ``max_steps=3`` the harness force-terminates after three
        outer steps even if the actor never emits ANSWER.  Combined
        with the deterministic ``valid[0] == ANSWER`` fallback this
        means the rollout actually terminates on step 1 — but the
        invariant we want to lock in is "rollout ≤ max_steps", not the
        exact step count.
        """
        h = VRHarness(
            image=fake_image, question="q", gold_answer="3", max_steps=3
        )
        agent = ActorAgent(skill_provider=NullSkillProvider(), harness=h)
        agent.reset()
        h.reset()

        steps = 0
        done = False
        obs = "q"
        while not done and steps < 5:  # safety cap above max_steps
            decision = agent.step(observation=obs, task="VR Q&A")
            obs, reward, done, info = h.step(decision.action)
            agent.observe_result(
                decision, reward=reward, next_observation=obs, done=done
            )
            steps += 1
        assert done is True
        assert steps <= 3

    def test_bind_actor_wires_scratchpad(self, fake_image: VisualInput) -> None:
        """``ActorAgent.bind_harness`` should hand the actor's scratchpad
        to the harness so VR ops mutate the shared state.
        """
        h = VRHarness(image=fake_image, question="q")
        agent = ActorAgent(skill_provider=NullSkillProvider(), harness=h)
        agent.reset()
        h.reset()

        # The harness's ``scratchpad`` is now the actor's scratchpad.
        assert h.scratchpad is agent.state.scratchpad
        h.step("LOOK(scene)")
        assert "scene" in agent.state.scratchpad.grounded_slots
