# Reward tool for the VLM decision agent.
#
# Computes reward from two sources after every take_action:
#   (a) r_env   — raw environment reward from env.step
#   (b) r_follow — skill-following shaping reward (termination-free;
#                  measures progress toward the active skill's "late states")
# Plus a cost term:
#   (c) r_cost  — negative costs for retrieval queries, skill calls, and skill switching
# Combined:
#   (d) r_total = r_env + w_follow * r_follow + r_cost

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """All weights and cost constants for the reward tool."""

    # Weight on skill-following shaping reward (keep small relative to r_env).
    w_follow: float = 0.1

    # Fixed negative costs (added to r_cost each time the action type occurs).
    query_mem_cost: float = -0.05
    query_skill_cost: float = -0.05
    call_skill_cost: float = -0.02
    skill_switch_cost: float = -0.10

    # r_follow: per-predicate bonus when a skill eff_add literal becomes true.
    follow_predicate_bonus: float = 0.05
    # r_follow: bonus when *all* eff_add literals are satisfied (skill "completed").
    follow_completion_bonus: float = 0.20
    # r_follow: small penalty per step while following a skill with no predicate progress.
    follow_no_progress_penalty: float = -0.01

    # ── Per-action-kind costs (Harness.action_kind) ──────────────────
    # Defaults are 0.0 so r_total for game / web / OS tasks
    # ("primitive" kind) is unchanged unless callers opt in.  The VR /
    # Video harnesses surface non-trivial action kinds (LOOK / RETRIEVE
    # / NOTE / ANSWER, NEXT_FRAME / JUMP / FOCUS / TRACK) so RL signals
    # can shape away over-deliberation without touching r_env.

    # Visual-reasoning harness actions
    vr_look_cost: float = 0.0
    vr_retrieve_cost: float = 0.0
    vr_note_cost: float = 0.0
    vr_answer_cost: float = 0.0

    # Video-understanding harness actions
    video_next_frame_cost: float = 0.0
    video_jump_cost: float = 0.0
    video_focus_cost: float = 0.0
    video_track_cost: float = 0.0

    def cost_for_action_kind(self, kind: str) -> float:
        """Return the per-action-kind cost (``0.0`` for unknown kinds).

        Mirrors the ``ACTION_KIND_*`` constants in
        :mod:`decision_agents.core.harness`.  Kept on the config so
        callers can sub-class :class:`RewardConfig` and override the
        whole table for an experiment without monkeypatching
        :class:`RewardComputer`.
        """
        if not kind:
            return 0.0
        # Map kind tag → field name on the config.  Anything unknown
        # falls through to 0.0 — mutable-world envs (gym / web / OS)
        # report ``"primitive"`` and rely on the legacy
        # ``call_skill_cost`` / ``skill_switch_cost`` / etc. fields.
        table = {
            "vr_look":          self.vr_look_cost,
            "vr_retrieve":      self.vr_retrieve_cost,
            "vr_note":          self.vr_note_cost,
            "vr_answer":        self.vr_answer_cost,
            "video_next_frame": self.video_next_frame_cost,
            "video_jump":       self.video_jump_cost,
            "video_focus":      self.video_focus_cost,
            "video_track":      self.video_track_cost,
        }
        return float(table.get(kind, 0.0))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RewardResult:
    """Output of one reward(...) call."""

    r_env: float = 0.0
    r_follow: float = 0.0
    r_cost: float = 0.0
    r_total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "r_env": self.r_env,
            "r_follow": self.r_follow,
            "r_cost": self.r_cost,
            "r_total": self.r_total,
        }

    def __repr__(self) -> str:
        return (
            f"RewardResult(r_env={self.r_env:.4f}, r_follow={self.r_follow:.4f}, "
            f"r_cost={self.r_cost:.4f}, r_total={self.r_total:.4f})"
        )


# ---------------------------------------------------------------------------
# Reward computer (stateful — tracks active skill, satisfied predicates, etc.)
# ---------------------------------------------------------------------------

class RewardComputer:
    """
    Computes (r_env, r_follow, r_cost, r_total) after every transition.

    Maintains per-episode state:
      - active_skill_id: which skill is currently being followed (or None)
      - satisfied_preds: set of eff_add predicates already achieved this skill run
      - steps_in_skill: how many consecutive steps the current skill has been active
      - prev_skill_id: skill id from previous step (for switching cost detection)

    Usage (runner calls after every take_action):
        rr = computer.compute_reward(
            r_env=reward_from_env,
            action_type="primitive",        # or "QUERY_MEM", "QUERY_SKILL", "CALL_SKILL"
            observation=obs_text,
            active_skill_id="navigate_corridor",
            skill_contract=contract,        # SkillEffectsContract or None
        )
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.cfg = config or RewardConfig()
        self.reset()

    def reset(self) -> None:
        """Reset per-episode state."""
        self._active_skill_id: Optional[str] = None
        self._prev_skill_id: Optional[str] = None
        self._satisfied_preds: Set[str] = set()
        self._steps_in_skill: int = 0
        self._cumulative: RewardResult = RewardResult()
        self._history: List[RewardResult] = []
        self._skill_period_r_follow: float = 0.0

    # ── Main API ─────────────────────────────────────────────────────

    def compute_reward(
        self,
        r_env: float = 0.0,
        action_type: str = "primitive",
        observation: str = "",
        active_skill_id: Optional[str] = None,
        skill_contract: Any = None,
        queried_skill: bool = False,
        queried_mem: bool = False,
        action_kind: str = "",
    ) -> RewardResult:
        """
        Compute the composite reward for the last transition.

        Args:
            r_env: Environment reward from env.step.
            action_type: One of "primitive", "QUERY_MEM", "QUERY_SKILL", "CALL_SKILL".
            observation: Current observation text (used to check predicate satisfaction).
            active_skill_id: Skill currently being followed, or None.
            skill_contract: SkillEffectsContract (or any object with .eff_add set), or None.
            queried_skill: True when this step invoked the skill-bank / provider
                retrieval API (reselect event). Triggers ``query_skill_cost`` on
                top of whatever ``action_type`` contributes. PLAN-ACTION-AGENT §4.
            queried_mem: True when this step's harness action is a retrieval-class
                action (e.g. ``VRHarness.step(RETRIEVE(q))``).  Triggers
                ``query_mem_cost`` on top of the per-action-kind cost.
            action_kind: Output of ``Harness.action_kind(action)``.  Looked
                up via :meth:`RewardConfig.cost_for_action_kind` to add a
                per-action-type penalty (e.g. ``vr_look_cost``).  Defaults
                to ``""`` so legacy callers without a harness see r_total
                unchanged.

        Returns:
            RewardResult with r_env, r_follow, r_cost, r_total.
        """
        r_cost = self._compute_cost(
            action_type,
            active_skill_id,
            queried_skill=queried_skill,
            queried_mem=queried_mem,
            action_kind=action_kind,
        )
        r_follow = self._compute_follow(observation, active_skill_id, skill_contract)
        r_total = r_env + self.cfg.w_follow * r_follow + r_cost

        result = RewardResult(
            r_env=r_env,
            r_follow=r_follow,
            r_cost=r_cost,
            r_total=r_total,
        )
        self._history.append(result)
        self._cumulative.r_env += r_env
        self._cumulative.r_follow += r_follow
        self._cumulative.r_cost += r_cost
        self._cumulative.r_total += r_total

        self._prev_skill_id = active_skill_id
        return result

    # ── Cost component ───────────────────────────────────────────────

    def _compute_cost(
        self,
        action_type: str,
        active_skill_id: Optional[str],
        *,
        queried_skill: bool = False,
        queried_mem: bool = False,
        action_kind: str = "",
    ) -> float:
        """Negative cost for retrieval / skill-call / skill-switching.

        ``action_type`` supplies the *primary* cost bucket, but a single
        step can legitimately query both the skill bank AND memory while
        still calling a skill — the ``queried_skill`` / ``queried_mem``
        booleans let the caller add those costs on top without clobbering
        the primary bucket.

        ``action_kind`` is the harness's per-action-type tag (e.g.
        ``"vr_look"``, ``"video_jump"``).  It adds a separate cost on
        top of the action-type bucket so VR / Video tasks can shape
        away over-deliberation while game / web / OS tasks (which
        report ``"primitive"``) are unaffected.
        """
        cost = 0.0
        at = action_type.upper() if action_type else ""

        if "QUERY_MEM" in at:
            cost += self.cfg.query_mem_cost
        elif "QUERY_SKILL" in at:
            cost += self.cfg.query_skill_cost
        elif "CALL_SKILL" in at:
            cost += self.cfg.call_skill_cost

        # Orthogonal query events reported by the actor (PLAN-ACTION-AGENT §4
        # — reselect and RETRIEVE-class actions incur costs regardless of
        # whether a skill is currently active).
        if queried_skill:
            cost += self.cfg.query_skill_cost
        if queried_mem:
            cost += self.cfg.query_mem_cost

        if active_skill_id != self._prev_skill_id and self._prev_skill_id is not None:
            cost += self.cfg.skill_switch_cost

        # Harness-level per-action cost (defaults to 0.0 for unknown /
        # primitive kinds so back-compat callers see r_total unchanged).
        if action_kind:
            cost += self.cfg.cost_for_action_kind(action_kind)

        return cost

    # ── Skill-following shaping reward ───────────────────────────────

    def _compute_follow(
        self,
        observation: str,
        active_skill_id: Optional[str],
        skill_contract: Any,
    ) -> float:
        """
        Termination-free skill-following reward.

        Checks which eff_add predicates from the active skill's contract are now
        satisfied (keyword-present in observation). Awards:
          - per-predicate bonus for each newly satisfied predicate
          - completion bonus when all are satisfied
          - small penalty per step with no new predicate progress
        """
        if active_skill_id is None or skill_contract is None:
            self._active_skill_id = None
            self._satisfied_preds = set()
            self._steps_in_skill = 0
            return 0.0

        if active_skill_id != self._active_skill_id:
            self._active_skill_id = active_skill_id
            self._satisfied_preds = set()
            self._steps_in_skill = 0
            self._skill_period_r_follow = 0.0

        self._steps_in_skill += 1

        eff_add: Set[str] = getattr(skill_contract, "eff_add", set()) or set()
        if not eff_add:
            return 0.0

        obs_lower = observation.lower() if observation else ""
        newly_satisfied: Set[str] = set()
        for pred in eff_add:
            if pred in self._satisfied_preds:
                continue
            tokens = pred.lower().replace("_", " ").split()
            if all(tok in obs_lower for tok in tokens if len(tok) >= 2):
                newly_satisfied.add(pred)

        r = 0.0
        if newly_satisfied:
            r += len(newly_satisfied) * self.cfg.follow_predicate_bonus
            self._satisfied_preds |= newly_satisfied
        else:
            r += self.cfg.follow_no_progress_penalty

        if self._satisfied_preds == eff_add:
            r += self.cfg.follow_completion_bonus

        self._skill_period_r_follow += r
        return r

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def cumulative(self) -> RewardResult:
        """Cumulative reward across the episode so far."""
        return self._cumulative

    @property
    def history(self) -> List[RewardResult]:
        """Per-step reward history."""
        return list(self._history)

    @property
    def active_skill_id(self) -> Optional[str]:
        return self._active_skill_id

    @property
    def satisfied_predicates(self) -> Set[str]:
        return set(self._satisfied_preds)

    @property
    def steps_in_skill(self) -> int:
        return self._steps_in_skill

    @property
    def skill_period_r_follow(self) -> float:
        """Cumulative r_follow for the current skill period (reset on switch)."""
        return self._skill_period_r_follow


# ---------------------------------------------------------------------------
# Convenience: standalone reward(...) function matching the tool spec
# ---------------------------------------------------------------------------

def compute_reward(
    r_env: float = 0.0,
    action_type: str = "primitive",
    observation: str = "",
    active_skill_id: Optional[str] = None,
    skill_contract: Any = None,
    computer: Optional[RewardComputer] = None,
    config: Optional[RewardConfig] = None,
    queried_skill: bool = False,
    queried_mem: bool = False,
) -> RewardResult:
    """
    Stateless convenience wrapper. If no ``computer`` is provided, creates a
    fresh one (no cross-step memory). For proper episode tracking, pass a
    persistent ``RewardComputer`` instance.
    """
    if computer is None:
        computer = RewardComputer(config=config)
    return computer.compute_reward(
        r_env=r_env,
        action_type=action_type,
        observation=observation,
        active_skill_id=active_skill_id,
        skill_contract=skill_contract,
        queried_skill=queried_skill,
        queried_mem=queried_mem,
    )
