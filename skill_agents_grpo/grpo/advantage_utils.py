"""GRPO group advantage computation with reward-tie handling.

If every sample in a group gets the **same** reward (common for segment
decode saturation, or contract/curator ties), then ``r - mean`` is zero for
all ``r`` and normalized advantages are **all zero** → no policy gradient.

We detect near-zero variance and add a tiny **completion-identity** signal
(zero-mean, unit-scale) so different completions still receive distinct
advantages.  Scale is small so real reward spread still dominates when it
exists.

Environment
-----------
GRPO_TIEBREAK_SCALE
    Magnitude of completion tiebreak when rewards are tied (default ``0.08``).
    Set ``0`` to disable.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional


def _completion_zero_mean_unit(completions: List[str]) -> List[float]:
    """Deterministic pseudo-random vector, zero mean, ~unit variance."""
    n = len(completions)
    raw: List[float] = []
    for c in completions:
        s = c if isinstance(c, str) else ""
        h = 0
        limit = min(len(s), 4000)
        for i in range(limit):
            h = (h * 131 + ord(s[i])) & 0xFFFFFFFF
        raw.append((h % 10007) / 10007.0)
    m = sum(raw) / n
    centered = [x - m for x in raw]
    v = sum(x * x for x in centered) / n
    sdev = max(v**0.5, 1e-8)
    return [x / sdev for x in centered]


def compute_grpo_group_advantages(
    rewards: List[float],
    completions: Optional[List[str]] = None,
    *,
    std_floor: float = 0.1,
    tiebreak_scale: Optional[float] = None,
    var_eps: float = 1e-14,
) -> List[float]:
    """Zero-mean, scaled advantages for one GRPO group.

    Matches co-evolution behavior: ``std = max(sqrt(var), std_floor)``.

    When reward variance is ~0 and *completions* is provided (same length as
    *rewards*), perturbs effective rewards by
    ``tiebreak_scale * z_i`` with ``z`` zero-mean unit-ish from completion
    text, then re-normalizes.  This preserves learning signal when the
    reward function is flat across the group.
    """
    if not rewards:
        return []
    finite = [r for r in rewards if math.isfinite(r)]
    if not finite:
        return [0.0] * len(rewards)
    fallback = sum(finite) / len(finite)
    sanitized = [r if math.isfinite(r) else fallback for r in rewards]
    n = len(sanitized)
    if n == 1:
        return [0.0]

    if tiebreak_scale is None:
        tiebreak_scale = float(os.environ.get("GRPO_TIEBREAK_SCALE", "0.08"))

    mean = sum(sanitized) / n
    var = sum((r - mean) ** 2 for r in sanitized) / n
    r_span = max(sanitized) - min(sanitized)

    adjusted = list(sanitized)
    tied = (var <= var_eps) or (r_span <= 1e-12)
    if (
        tied
        and completions is not None
        and len(completions) == n
        and tiebreak_scale > 0.0
    ):
        z = _completion_zero_mean_unit([c or "" for c in completions])
        adjusted = [sanitized[i] + tiebreak_scale * z[i] for i in range(n)]
        mean = sum(adjusted) / n
        var = sum((r - mean) ** 2 for r in adjusted) / n

    std = max(var**0.5, std_floor)
    return [(r - mean) / std for r in adjusted]
