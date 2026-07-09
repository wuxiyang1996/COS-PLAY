"""
Configuration for Stage 3 MVP: effects-only contract learning.

Flat config with sensible defaults for predicate booleanization,
effect frequency thresholds, and verification criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass
class Stage3MVPConfig:
    """All tunables for the effects-only MVP pipeline."""

    # Window size (w) for start/end predicate smoothing
    start_end_window: int = 5

    # Probability threshold to booleanize vision/HUD predicates
    p_thresh_vision: float = 0.7

    # Minimum frequency across instances to include an effect literal
    eff_freq: float = 0.8

    # Minimum segment instances required before learning a contract
    min_instances_per_skill: int = 5

    # Hard cap on effect literals per contract
    max_effects_per_skill: int = 50

    # UI predicates use OR-aggregation within the window
    ui_or_mode: bool = True

    # Predicate namespaces to keep
    keep_types: Set[str] = field(default_factory=lambda: {"ui", "hud", "world"})

    # Minimum predicate reliability to include in effect computation
    reliability_min_for_effects: float = 0.7

    # Fraction of contract literals an instance must satisfy to "pass"
    instance_pass_literal_frac: float = 0.7

    # Top-N worst segments to include in verification reports
    max_worst_segments: int = 10

    # LLM model for contract summary generation (e.g. "Qwen/Qwen3-8B").
    # Empty string = use ask_model default routing.
    model: str = ""
