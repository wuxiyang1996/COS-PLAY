"""
Duration model p(ℓ | k) for Bank Maintenance.

Two representations:
  - ``DurationHistogram``: binned counts with Laplace smoothing.
  - ``DurationLogNormal``: MLE fit of log-normal (mean/var of log-lengths).

Both expose ``log_prob(length)`` so Stage 2 can use them as priors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DurationHistogram:
    """Fixed-bin histogram for segment durations."""

    n_bins: int = 20
    min_len: int = 1
    max_len: int = 2000
    smoothing: float = 1.0

    counts: List[float] = field(default_factory=list)
    total: float = 0.0
    _bin_width: float = 0.0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0.0] * self.n_bins
        self._bin_width = (self.max_len - self.min_len) / self.n_bins

    def _bin_index(self, length: int) -> int:
        clamped = max(self.min_len, min(length, self.max_len - 1))
        idx = int((clamped - self.min_len) / self._bin_width)
        return min(idx, self.n_bins - 1)

    def add(self, length: int, weight: float = 1.0) -> None:
        self.counts[self._bin_index(length)] += weight
        self.total += weight

    def add_batch(self, lengths: List[int]) -> None:
        for l in lengths:
            self.add(l)

    def log_prob(self, length: int) -> float:
        """Smoothed log-probability of *length*."""
        idx = self._bin_index(length)
        numerator = self.counts[idx] + self.smoothing
        denominator = self.total + self.smoothing * self.n_bins
        if denominator <= 0:
            return -math.log(self.n_bins)
        return math.log(numerator / denominator)

    def mean_var(self) -> Tuple[float, float]:
        """Compute weighted mean and variance from bin centres."""
        if self.total <= 0:
            return (0.0, 0.0)
        centres = [
            self.min_len + (i + 0.5) * self._bin_width
            for i in range(self.n_bins)
        ]
        mean = sum(c * n for c, n in zip(centres, self.counts)) / self.total
        var = (
            sum(n * (c - mean) ** 2 for c, n in zip(centres, self.counts))
            / self.total
        )
        return (mean, var)

    def to_dict(self) -> dict:
        return {
            "n_bins": self.n_bins,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "smoothing": self.smoothing,
            "counts": self.counts,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DurationHistogram:
        hist = cls(
            n_bins=d["n_bins"],
            min_len=d["min_len"],
            max_len=d["max_len"],
            smoothing=d.get("smoothing", 1.0),
        )
        hist.counts = d["counts"]
        hist.total = d["total"]
        return hist


@dataclass
class DurationLogNormal:
    """MLE log-normal fit for segment durations.

    Stores sufficient statistics so updates are O(1) per new observation.
    """

    _sum_log: float = 0.0
    _sum_log_sq: float = 0.0
    _n: int = 0

    @property
    def mu(self) -> float:
        return self._sum_log / self._n if self._n > 0 else 0.0

    @property
    def sigma_sq(self) -> float:
        if self._n < 2:
            return 1.0
        mean = self.mu
        return self._sum_log_sq / self._n - mean * mean

    def add(self, length: int) -> None:
        if length < 1:
            return
        log_l = math.log(length)
        self._sum_log += log_l
        self._sum_log_sq += log_l * log_l
        self._n += 1

    def add_batch(self, lengths: List[int]) -> None:
        for l in lengths:
            self.add(l)

    def log_prob(self, length: int) -> float:
        if length < 1 or self._n < 2:
            return 0.0
        log_l = math.log(length)
        mu = self.mu
        s2 = max(self.sigma_sq, 1e-6)
        return -0.5 * math.log(2 * math.pi * s2) - log_l - (
            (log_l - mu) ** 2
        ) / (2 * s2)

    def mean_var(self) -> Tuple[float, float]:
        mu = self.mu
        s2 = max(self.sigma_sq, 1e-6)
        mean = math.exp(mu + s2 / 2)
        var = (math.exp(s2) - 1) * math.exp(2 * mu + s2)
        return (mean, var)

    def to_dict(self) -> dict:
        return {
            "sum_log": self._sum_log,
            "sum_log_sq": self._sum_log_sq,
            "n": self._n,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DurationLogNormal:
        obj = cls()
        obj._sum_log = d["sum_log"]
        obj._sum_log_sq = d["sum_log_sq"]
        obj._n = d["n"]
        return obj


# ── Per-skill duration store ─────────────────────────────────────────


class DurationModelStore:
    """Manages per-skill duration models (histogram + optional log-normal)."""

    def __init__(
        self,
        n_bins: int = 20,
        min_len: int = 1,
        max_len: int = 2000,
        smoothing: float = 1.0,
    ) -> None:
        self._params = dict(
            n_bins=n_bins, min_len=min_len, max_len=max_len, smoothing=smoothing,
        )
        self._histograms: Dict[str, DurationHistogram] = {}
        self._lognormals: Dict[str, DurationLogNormal] = {}

    def update(self, skill_id: str, lengths: List[int]) -> None:
        if skill_id not in self._histograms:
            self._histograms[skill_id] = DurationHistogram(**self._params)
            self._lognormals[skill_id] = DurationLogNormal()
        self._histograms[skill_id].add_batch(lengths)
        self._lognormals[skill_id].add_batch(lengths)

    def replace(self, skill_id: str, lengths: List[int]) -> None:
        """Replace (not accumulate) duration model for a skill."""
        self._histograms[skill_id] = DurationHistogram(**self._params)
        self._lognormals[skill_id] = DurationLogNormal()
        self._histograms[skill_id].add_batch(lengths)
        self._lognormals[skill_id].add_batch(lengths)

    def log_prob(self, skill_id: str, length: int) -> float:
        hist = self._histograms.get(skill_id)
        if hist is None:
            return 0.0
        return hist.log_prob(length)

    def mean_var(self, skill_id: str) -> Tuple[float, float]:
        hist = self._histograms.get(skill_id)
        if hist is None:
            return (0.0, 0.0)
        return hist.mean_var()

    def remove(self, skill_id: str) -> None:
        self._histograms.pop(skill_id, None)
        self._lognormals.pop(skill_id, None)

    def rename(self, old_id: str, new_id: str) -> None:
        if old_id in self._histograms:
            self._histograms[new_id] = self._histograms.pop(old_id)
        if old_id in self._lognormals:
            self._lognormals[new_id] = self._lognormals.pop(old_id)

    @property
    def skill_ids(self) -> List[str]:
        return list(self._histograms.keys())

    def to_dict(self) -> dict:
        return {
            "params": self._params,
            "histograms": {
                sid: h.to_dict() for sid, h in self._histograms.items()
            },
            "lognormals": {
                sid: ln.to_dict() for sid, ln in self._lognormals.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> DurationModelStore:
        store = cls(**d.get("params", {}))
        for sid, hd in d.get("histograms", {}).items():
            store._histograms[sid] = DurationHistogram.from_dict(hd)
        for sid, ld in d.get("lognormals", {}).items():
            store._lognormals[sid] = DurationLogNormal.from_dict(ld)
        return store
