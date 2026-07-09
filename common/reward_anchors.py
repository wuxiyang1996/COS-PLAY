"""Teacher-anchored reward normalization (training_notes §4.5).

Per-game raw rewards span ~3 orders of magnitude across the 8 Gym-V
Temporal games (Strider 0–112 → SpaceHarrierII 14 k–29 k). Within a
phase this is a non-issue — GRPO normalizes advantages inside the
rollout group, so absolute reward magnitude doesn't reach the
optimizer. **Across phases**, however, every aggregate metric (W&B
mean, Layer-D dashboard transfer matrix, best-checkpoint selection,
curriculum phase-success thresholds) is reward-magnitude-biased toward
whichever phase has the largest absolute reward.

This module exposes the **teacher-anchored, additive normalization
layer** locked in §4.5:

    r_norm[game] = clip(r_raw[game] / r_teacher_anchor[game], 0.0, 2.0)

* ``r_teacher_anchor[game]`` is auto-derived from
  ``Cold-start-out-gymv/latest/<env_folder>/rollout_summary.json``
  when available, otherwise falls back to a hardcoded table baked
  from the new 4-backbone teacher data (see §4.5 Anchor table).
* ``r_norm = 1.0`` ⟺ matches teacher; ``0.5`` ⟺ half teacher; the
  ``2.0`` ceiling stops a lucky-spike episode from owning the
  dashboard.
* ``None`` (anchor missing or zero) ≠ ``0`` so dashboards can
  distinguish "no anchor" from "scored zero".

The normalized value is *additive* to existing reward fields — see
``harness.RewardLogger`` (RewardLogEntry / GRPOStepLogEntry have a new
``reward_normalized`` field) and ``trainer.coevolution.orchestrator``
(W&B emit splits ``reward/raw/{game}/...`` from
``reward/normalized/{game}/...``).

GRPO advantages are **NOT** touched — group normalization already
handles within-batch variance (§4.5 "Where it's NOT applied").
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Static fallback anchors ──────────────────────────────────────────
#
# Frozen on 2026-05-03 PM from the new
# ``Cold-start-out-gymv/gpt54_skip8_e16_s80_*`` 4-backbone teacher
# table (GPT-5.4 / Claude-4.6-Sonnet / Gemini-3.1-Pro / Qwen3-VL-235B,
# 16 episodes × frame_skip=8). Each anchor is the **max across the 4
# frontier rows** so the normalized reward of 1.0 represents
# "matches the strongest frontier teacher", and clip(2.0) leaves room
# for actor outperformance.
#
# Paper Table-3 anchors for the 4 env_wrappers games are populated
# from ``baselines/`` runs on first call to ``resolve_anchors`` — they
# stay ``None`` here so a missing baseline produces the proper
# "no anchor" signal in W&B / dashboards rather than a silent default.
TEACHER_REWARD_ANCHORS: Dict[str, Optional[float]] = {
    # Gym-V Temporal/* — refreshed 2026-05-03 PM from new SFT data.
    "gymv_thunder_force_iii": 750.0,    # Qwen3-VL frontier upper-CI
    "gymv_altered_beast": 425.0,        # Gemini frontier upper-CI
    "gymv_columns": 160.8,              # GPT-5.4 frontier upper-CI
    "gymv_dynamite_headdy": 100.0,      # GPT-5.4 / Claude tied
    "gymv_space_harrier_ii": 29431.0,   # Claude — scale-jump outlier
    "gymv_streets_of_rage_2": 408.8,    # Gemini frontier upper-CI
    "gymv_airstriker": 97.5,            # Gemini frontier upper-CI
    "gymv_strider": 112.5,              # Gemini frontier upper-CI
                                        # (NOTE: GPT-5.4 / Claude / Qwen-VL all scored 0
                                        # on Strider — partial-signal "rescue" target)
    # env_wrappers / paper Table-3 games — anchor from
    # baselines/<game>/ runs, populated on first ``resolve_anchors``
    # call. ``None`` here means "no anchor" (vs ``0.0`` which would
    # poison the divisor).
    "tetris": None,
    "candy_crush": None,
    "twenty_forty_eight": None,
    "super_mario": None,
}

# Slug → cold-start sub-directory (Temporal/<Foo>-v0 → Temporal_<Foo>-v0).
# Kept in sync with ``env_wrappers/gymv_temporal_nl_wrapper.GYMV_TEMPORAL_GAMES``.
_GYMV_COLD_START_DIRS: Dict[str, str] = {
    "gymv_thunder_force_iii": "Temporal_ThunderForceIII-v0",
    "gymv_altered_beast": "Temporal_AlteredBeast-v0",
    "gymv_columns": "Temporal_Columns-v0",
    "gymv_dynamite_headdy": "Temporal_DynamiteHeaddy-v0",
    "gymv_space_harrier_ii": "Temporal_SpaceHarrierII-v0",
    "gymv_streets_of_rage_2": "Temporal_StreetsOfRage2-v0",
    "gymv_airstriker": "Temporal_Airstriker-v0",
    "gymv_strider": "Temporal_Strider-v0",
}

# Default cold-start root, relative to repo root.
_DEFAULT_COLD_START_ROOT = "Cold-start-out-gymv/latest"

# Clip ceiling — see §4.5 "Interpretation".
_CLIP_FLOOR = 0.0
_CLIP_CEIL = 2.0

# Which field of ``rollout_summary.json`` to use as the auto-derived
# anchor. Default is ``"max_reward"`` to stay numerically consistent
# with the static fallback table above (which is "max across 4 frontier
# teachers"). Override at runtime via the ``COEVO_REWARD_ANCHOR_FIELD``
# env var or the ``anchor_field=`` kwarg to :func:`auto_derive_anchors`
# / :func:`resolve_anchors`.
#
# Trade-off:
#   * ``"max_reward"`` (default) — strict; ``r_norm = 1.0`` ⟺ "matches
#     the teacher's *best* episode". Aligns with the static fallback
#     so auto-derive vs. fallback values are directly comparable in
#     W&B. May be too tight on noisy gymv signals where the teacher
#     itself has high variance.
#   * ``"mean_reward"`` — lenient; ``r_norm = 1.0`` ⟺ "matches the
#     teacher's *typical* episode". More forgiving for noisy gymv
#     reward but inconsistent with the static fallback when the
#     auto-derive subset partially covers the registry.
_DEFAULT_ANCHOR_FIELD = "max_reward"
_ANCHOR_FIELD_ENV_VAR = "COEVO_REWARD_ANCHOR_FIELD"
_VALID_ANCHOR_FIELDS = ("max_reward", "mean_reward")


def _resolve_anchor_field(explicit: Optional[str]) -> str:
    """Resolve the ``rollout_summary.json`` field name used for
    auto-derived anchors.

    Order: explicit kwarg > ``COEVO_REWARD_ANCHOR_FIELD`` env var >
    :data:`_DEFAULT_ANCHOR_FIELD`. Unknown values fall back to the
    default with a warning.
    """

    candidate = explicit or os.environ.get(_ANCHOR_FIELD_ENV_VAR) or _DEFAULT_ANCHOR_FIELD
    if candidate not in _VALID_ANCHOR_FIELDS:
        logger.warning(
            "reward_anchors: unknown anchor_field=%r (expected one of %s); "
            "falling back to %r.",
            candidate, _VALID_ANCHOR_FIELDS, _DEFAULT_ANCHOR_FIELD,
        )
        return _DEFAULT_ANCHOR_FIELD
    return candidate


# ── Public API ───────────────────────────────────────────────────────


def auto_derive_anchors(
    cold_start_root: Optional[str] = None,
    *,
    anchor_field: Optional[str] = None,
) -> Dict[str, float]:
    """Read per-game teacher anchors from the SFT cold-start summaries.

    Returns a dict ``{slug: anchor}`` for games where
    ``<cold_start_root>/<env_folder>/rollout_summary.json`` exists and
    contains a numeric, **strictly positive** value at the configured
    field (default ``max_reward`` — see ``_DEFAULT_ANCHOR_FIELD``).
    Slugs with missing files, missing fields, or zero/negative
    rewards are simply omitted — callers can then layer the
    static fallback table on top via :func:`resolve_anchors`.

    Args:
      cold_start_root: Path to the cold-start ``latest`` dir
        (e.g. ``Cold-start-out-gymv/latest``). When ``None``, reads
        the ``COEVO_COLD_START_ROOT`` env var, then falls back to
        ``Cold-start-out-gymv/latest`` relative to ``cwd``. Missing
        roots produce an empty dict (not an exception) so anchors
        gracefully degrade to the static fallback in dev / test.
      anchor_field: Override the ``rollout_summary.json`` field name
        used as the per-game anchor. ``None`` ⇒ read the
        ``COEVO_REWARD_ANCHOR_FIELD`` env var, then fall back to
        ``"max_reward"``. Valid values: ``"max_reward"``,
        ``"mean_reward"``.
    """

    root = _resolve_cold_start_root(cold_start_root)
    if root is None:
        return {}

    field_name = _resolve_anchor_field(anchor_field)

    anchors: Dict[str, float] = {}
    for slug, env_dir in _GYMV_COLD_START_DIRS.items():
        summary_path = root / env_dir / "rollout_summary.json"
        try:
            value = _read_anchor_from_summary(summary_path, field_name=field_name)
        except OSError:
            continue
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "reward_anchors.auto_derive: malformed %s (%s) — skipping",
                summary_path, exc,
            )
            continue
        if value is None or value <= 0.0:
            continue
        anchors[slug] = float(value)

    if anchors:
        logger.info(
            "reward_anchors.auto_derive: %d gymv anchors loaded from %s "
            "(field=%s)",
            len(anchors), root, field_name,
        )
    return anchors


def resolve_anchors(
    *,
    cold_start_root: Optional[str] = None,
    overrides: Optional[Dict[str, Optional[float]]] = None,
    fallback: Optional[Dict[str, Optional[float]]] = None,
    anchor_field: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """Build the active anchor table by layering: fallback → auto → overrides.

    Order (later wins):
      1. ``fallback``  — defaults to :data:`TEACHER_REWARD_ANCHORS`.
      2. Auto-derived  — :func:`auto_derive_anchors` against
         ``cold_start_root`` using ``anchor_field``. Auto values
         override the fallback only when strictly positive (so a stale
         or zero summary cannot silently zero out the static anchor).
      3. ``overrides`` — caller-supplied values take precedence.
         Pass ``None`` for a slug here to *explicitly* mark it as
         "no anchor" (downstream produces ``reward_normalized=None``).

    The returned dict always uses ``Optional[float]`` to encode the
    "no anchor" state — see :func:`normalize_reward`.
    """

    fallback_table = dict(fallback if fallback is not None else TEACHER_REWARD_ANCHORS)
    auto = auto_derive_anchors(
        cold_start_root=cold_start_root,
        anchor_field=anchor_field,
    )
    for slug, value in auto.items():
        if value is not None and value > 0.0:
            fallback_table[slug] = float(value)
    if overrides:
        for slug, value in overrides.items():
            fallback_table[slug] = value
    return fallback_table


def normalize_reward(
    raw_reward: Optional[float],
    game: str,
    *,
    anchors: Optional[Dict[str, Optional[float]]] = None,
    clip_floor: float = _CLIP_FLOOR,
    clip_ceil: float = _CLIP_CEIL,
) -> Optional[float]:
    """Apply the teacher-anchored normalization to a single reward.

    Returns ``None`` when:

      * ``raw_reward is None`` (no reward observed),
      * ``game`` has no anchor (``anchors[game] is None`` or missing), or
      * the anchor is not strictly positive (would-be /0).

    Otherwise returns ``clip(raw_reward / anchor, clip_floor, clip_ceil)``.

    This is the single point of truth for normalization — RewardLogger
    and W&B emit code both call this helper so the numerical
    semantics never diverge.
    """

    if raw_reward is None:
        return None
    table = anchors if anchors is not None else TEACHER_REWARD_ANCHORS
    anchor = table.get(game)
    if anchor is None:
        return None
    if not (anchor > 0.0):
        return None
    norm = float(raw_reward) / float(anchor)
    if norm < clip_floor:
        return clip_floor
    if norm > clip_ceil:
        return clip_ceil
    return norm


def normalize_per_game(
    rewards: Dict[str, Optional[float]],
    *,
    anchors: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Optional[float]]:
    """Vectorised :func:`normalize_reward` for ``{game: raw}`` dicts.

    Returns a parallel ``{game: norm_or_None}`` dict.
    """

    return {
        game: normalize_reward(raw, game, anchors=anchors)
        for game, raw in rewards.items()
    }


# ── Internals ────────────────────────────────────────────────────────


def _resolve_cold_start_root(explicit: Optional[str]) -> Optional[Path]:
    """Resolve the cold-start ``latest`` dir, returning ``None`` if absent."""

    candidate = explicit or os.environ.get("COEVO_COLD_START_ROOT") or _DEFAULT_COLD_START_ROOT
    path = Path(candidate)
    if not path.is_absolute():
        # Relative to the current working directory — orchestrator runs
        # always cd into ``Multi-hop-Reasoning-VLM-Agent`` first.
        path = Path.cwd() / path
    if not path.exists():
        return None
    return path


def _read_anchor_from_summary(
    summary_path: Path,
    *,
    field_name: str = _DEFAULT_ANCHOR_FIELD,
) -> Optional[float]:
    """Return the requested anchor field from ``rollout_summary.json``.

    Raises ``OSError`` if the file is missing (caller handles).
    Raises ``ValueError`` for malformed JSON / unexpected schema.
    """

    if not summary_path.is_file():
        raise OSError(f"missing: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"summary not a dict: {summary_path}")
    value = data.get(field_name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"non-numeric {field_name}={value!r} in {summary_path}"
        ) from exc


def known_anchored_games() -> Iterable[str]:
    """Return the list of slugs that have a static fallback anchor.

    Used for sanity checks and W&B namespace pre-population.
    """

    return tuple(TEACHER_REWARD_ANCHORS.keys())


def anchor_for(game: str, *, anchors: Optional[Dict[str, Optional[float]]] = None) -> Optional[float]:
    """Return the active anchor for *game* (or ``None`` if missing)."""

    table = anchors if anchors is not None else TEACHER_REWARD_ANCHORS
    return table.get(game)


__all__ = [
    "TEACHER_REWARD_ANCHORS",
    "anchor_for",
    "auto_derive_anchors",
    "known_anchored_games",
    "normalize_per_game",
    "normalize_reward",
    "resolve_anchors",
]
