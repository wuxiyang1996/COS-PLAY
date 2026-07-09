"""
Configuration for Stage 2 InferSegmentation.

Dataclasses for scorer weights, decoder parameters, the NEW-skill channel,
LLM preference-teacher settings, and preference-learning hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class ContractFeedbackConfig:
    """Controls the Stage 3 → Stage 2 contract feedback loop.

    Three modes selected by ``mode``:
      - ``"off"``:    no contract feedback (``contract_compat`` weight = 0).
      - ``"weak"``:   light bias from contracts (default ``strength`` = 0.3).
      - ``"strong"``: heavy reliance on contracts (default ``strength`` = 1.0).

    When ``mode != "off"``, the ``contract_compat`` weight in ``ScorerWeights``
    is automatically set to ``strength`` unless manually overridden.
    """

    mode: str = "off"  # "off" | "weak" | "strong"
    strength: float = 0.3  # effective weight when mode != "off"
    p_thresh: float = 0.5
    missing_penalty: float = -0.5
    contradiction_penalty: float = -1.0


@dataclass
class ScorerWeights:
    """Relative weights for each term in the segment score decomposition.

    ``intention_fit`` was previously 2.0, which made the tag-matching
    signal dominate — segments always preferred existing `:SETUP` (or
    similar) skills and ``__NEW__`` could never compete (it scores 0
    on intention_fit).  Lowered to 1.0 so behaviour_fit and transition
    priors have a fair vote alongside tags.
    """

    behavior_fit: float = 1.0
    intention_fit: float = 1.0
    duration_prior: float = 0.3
    transition_prior: float = 1.0
    contract_compat: float = 0.0  # set by ContractFeedbackConfig or manually
    boundary_preference: float = 0.5  # Phase 3: "cut here" vs "do not cut here"


@dataclass
class DurationPriorConfig:
    """Parameters for the per-skill duration prior log p(l | k).

    When ``adaptive`` is True (default), ``default_mean`` and
    ``default_std`` are auto-tuned per episode based on the game and
    trajectory length via :func:`get_duration_prior_for_game`.
    """

    default_mean: float = 8.0
    default_std: float = 5.0
    min_length: int = 2
    max_length: int = 200
    adaptive: bool = True


# ── Game-aware duration priors ───────────────────────────────────────
# Hand-tuned (mean, std) for segment length by game.  Longer / more
# strategic games get larger means so the decoder produces fewer,
# chunkier segments that capture real skill-level strategies.

GAME_DURATION_PRIORS: Dict[str, Tuple[float, float]] = {
    "tetris": (8.0, 5.0),
    "candy_crush": (12.0, 6.0),
    "twenty_forty_eight": (18.0, 10.0),
    "2048": (18.0, 10.0),
    "sokoban": (14.0, 7.0),
    "avalon": (12.0, 6.0),
    "diplomacy": (18.0, 10.0),
    "pokemon_red": (12.0, 7.0),
    "pokemon": (12.0, 7.0),
    "super_mario": (10.0, 6.0),
}


def get_duration_prior_for_game(
    game_name: str,
    episode_length: Optional[int] = None,
) -> DurationPriorConfig:
    """Return a :class:`DurationPriorConfig` tuned for *game_name*.

    Lookup order:
      1. ``GAME_DURATION_PRIORS`` hand-tuned table.
      2. Adaptive heuristic: target ~10-12 segments per episode so each
         segment captures a meaningful strategic chunk.
      3. Fallback defaults (mean=8, std=5).
    """
    if game_name in GAME_DURATION_PRIORS:
        mean, std = GAME_DURATION_PRIORS[game_name]
        if episode_length and episode_length > 0:
            ratio = episode_length / (mean * 12)
            if ratio > 1.5:
                mean = max(mean, episode_length / 12.0)
                std = max(std, mean * 0.55)
    elif episode_length and episode_length > 0:
        mean = max(5.0, episode_length / 12.0)
        std = max(3.0, mean * 0.55)
    else:
        return DurationPriorConfig()

    return DurationPriorConfig(
        default_mean=mean,
        default_std=std,
        min_length=2,
        max_length=max(200, int(mean * 6)),
        adaptive=False,
    )


@dataclass
class NewSkillConfig:
    """Parameters for the special NEW-skill channel.

    ``background_log_prob`` is the per-step log-probability used for
    ``__NEW__`` segments' behavior_fit.  The old value of -3.0 was so
    harsh that __NEW__ could never beat even a mediocre known-skill
    match, effectively freezing the skill bank after cold start.
    Reduced to -0.5 so the decoder can label genuinely novel segments.
    """

    enabled: bool = True
    penalty: float = 5.0  # alpha: subtracted from background score
    background_log_prob: float = -0.5  # fallback log-prob per step


@dataclass
class LLMTeacherConfig:
    """Settings for the LLM preference teacher (provides rankings, not scores)."""

    model: Optional[str] = None  # None = use ask_model default (gpt-4o)
    temperature: float = 0.3
    max_tokens: int = 1000
    # Number of worker threads (None or 1 = sequential).
    max_workers: Optional[int] = 4
    # Cap on concurrent LLM calls (e.g. GPU inference).
    # Limits per-episode concurrency; the global cap in llm_teacher.py
    # limits cross-episode concurrency.  Together they prevent flooding
    # the vLLM servers when many episodes are segmented in parallel.
    max_concurrent_llm_calls: Optional[int] = 4


@dataclass
class PreferenceLearningConfig:
    """Hyperparameters for the preference-learning loop."""

    num_iterations: int = 3
    margin_threshold: float = 1.0  # query segments with margin below this
    max_queries_per_iter: int = 5
    training_epochs: int = 20
    learning_rate: float = 0.1
    collect_transitions: bool = True  # also collect transition preferences


@dataclass
class UncertainLabelConfig:
    """Phase 3: uncertain-label path thresholds.

    Instead of forcing every segment into a known skill, three outcomes:
      - confident known skill (margin >= confident_margin)
      - low-confidence known skill (margin < confident_margin but >= uncertain_margin)
      - NEW / unknown (assigned __NEW__)

    Low-confidence segments go to an uncertain pool for reconsideration.
    """

    confident_margin: float = 2.0
    uncertain_margin: float = 1.0
    reconsider_after_n_updates: int = 3


@dataclass
class DecoderConfig:
    """Parameters shared by Viterbi DP and beam decoders."""

    top_m_skills: Optional[int] = None  # per segment, only score top-M skills (None = all)
    top_r_transitions: Optional[int] = None  # per skill, only consider top-R next skills

    # Beam-specific
    beam_width: int = 16
    beam_max_segments: Optional[int] = None  # early-stop after this many segments

    # Diagnostics
    top_k_diagnostics: int = 3  # how many skill candidates to return per segment


@dataclass
class SegmentationConfig:
    """Top-level configuration for InferSegmentation (Stage 2)."""

    weights: ScorerWeights = field(default_factory=ScorerWeights)
    duration: DurationPriorConfig = field(default_factory=DurationPriorConfig)
    new_skill: NewSkillConfig = field(default_factory=NewSkillConfig)
    llm_teacher: LLMTeacherConfig = field(default_factory=LLMTeacherConfig)
    preference: PreferenceLearningConfig = field(default_factory=PreferenceLearningConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    contract_feedback: ContractFeedbackConfig = field(default_factory=ContractFeedbackConfig)
    uncertain_label: UncertainLabelConfig = field(default_factory=UncertainLabelConfig)

    method: str = "dp"  # "dp" (Viterbi) or "beam"

    def __post_init__(self) -> None:
        """Apply contract feedback mode to scorer weights if not manually set."""
        cf = self.contract_feedback
        if cf.mode == "off":
            pass  # keep contract_compat at whatever the user set (default 0.0)
        elif cf.mode == "weak":
            if self.weights.contract_compat == 0.0:
                self.weights.contract_compat = cf.strength
        elif cf.mode == "strong":
            if self.weights.contract_compat == 0.0:
                self.weights.contract_compat = max(cf.strength, 1.0)
