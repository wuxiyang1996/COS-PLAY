"""
Fast candidate-retrieval indices for Bank Maintenance.

Three index types (all avoid O(n^2) comparisons):
  A) ``EffectInvertedIndex`` — predicate → set(skill_ids)
  B) ``MinHashLSH``         — approximate Jaccard on effect sets
  C) ``EmbeddingANN``       — optional ANN on centroid embeddings

Only skills in the index are searched; indices are updated incrementally.
"""

from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

from skill_agents.bank_maintenance.schemas import SkillProfile

# ═════════════════════════════════════════════════════════════════════
# A) Effect Inverted Index
# ═════════════════════════════════════════════════════════════════════


class EffectInvertedIndex:
    """Maps each effect predicate to the skill_ids that include it."""

    def __init__(self) -> None:
        self._index: Dict[str, Set[str]] = defaultdict(set)

    def add(self, skill_id: str, effects: Iterable[str]) -> None:
        for pred in effects:
            self._index[pred].add(skill_id)

    def remove(self, skill_id: str) -> None:
        for pred_set in self._index.values():
            pred_set.discard(skill_id)

    def update_skill(self, skill_id: str, effects: Iterable[str]) -> None:
        self.remove(skill_id)
        self.add(skill_id, effects)

    def candidates_for(
        self,
        effects: Iterable[str],
        min_shared: int = 2,
        exclude: Optional[Set[str]] = None,
    ) -> List[Tuple[str, int]]:
        """Return (skill_id, n_shared) pairs with >= *min_shared* overlap."""
        counts: Dict[str, int] = defaultdict(int)
        for pred in effects:
            for sid in self._index.get(pred, ()):
                if exclude and sid in exclude:
                    continue
                counts[sid] += 1
        return [
            (sid, cnt) for sid, cnt in counts.items() if cnt >= min_shared
        ]

    def build_from_profiles(self, profiles: Dict[str, SkillProfile]) -> None:
        self._index.clear()
        for sid, prof in profiles.items():
            self.add(sid, prof.all_effects)


# ═════════════════════════════════════════════════════════════════════
# B) MinHash LSH (pure-Python, no external deps)
# ═════════════════════════════════════════════════════════════════════

_MAX_HASH = (1 << 32) - 1


def _murmur_ish(item: str, seed: int) -> int:
    """Cheap hash: SHA-256 truncated to 32-bit with a seed."""
    h = hashlib.sha256(f"{seed}:{item}".encode()).digest()
    return struct.unpack("<I", h[:4])[0]


class MinHashSignature:
    """Fixed-size MinHash signature for a set of strings."""

    __slots__ = ("values",)

    def __init__(self, values: List[int]) -> None:
        self.values = values

    def jaccard_estimate(self, other: MinHashSignature) -> float:
        if len(self.values) != len(other.values):
            raise ValueError("Signatures must have equal length")
        matches = sum(a == b for a, b in zip(self.values, other.values))
        return matches / len(self.values)


class MinHashLSH:
    """Locality-Sensitive Hashing for approximate Jaccard nearest neighbours.

    Uses a simple banding scheme: *num_perm* hash functions split into *bands*
    bands of *rows* rows each.  Two sets that agree on all rows in any one
    band are candidates.
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.50) -> None:
        self.num_perm = num_perm
        self._bands, self._rows = self._optimal_banding(num_perm, threshold)
        self._signatures: Dict[str, MinHashSignature] = {}
        self._buckets: List[Dict[int, Set[str]]] = [
            defaultdict(set) for _ in range(self._bands)
        ]

    @staticmethod
    def _optimal_banding(num_perm: int, threshold: float) -> Tuple[int, int]:
        """Choose (bands, rows) so that the S-curve inflection ≈ threshold."""
        best = (1, num_perm)
        best_diff = float("inf")
        for b in range(1, num_perm + 1):
            r = num_perm // b
            if r == 0:
                continue
            inflection = (1.0 / b) ** (1.0 / r)
            diff = abs(inflection - threshold)
            if diff < best_diff:
                best_diff = diff
                best = (b, r)
        return best

    def _compute_signature(self, items: Iterable[str]) -> MinHashSignature:
        vals = [_MAX_HASH] * self.num_perm
        for item in items:
            for i in range(self.num_perm):
                h = _murmur_ish(item, i)
                if h < vals[i]:
                    vals[i] = h
        return MinHashSignature(vals)

    def _band_hashes(self, sig: MinHashSignature) -> List[int]:
        hashes: List[int] = []
        for b in range(self._bands):
            start = b * self._rows
            end = start + self._rows
            band_tuple = tuple(sig.values[start:end])
            hashes.append(hash(band_tuple))
        return hashes

    def add(self, skill_id: str, effects: Iterable[str]) -> None:
        sig = self._compute_signature(effects)
        self._signatures[skill_id] = sig
        for b, bh in enumerate(self._band_hashes(sig)):
            self._buckets[b][bh].add(skill_id)

    def remove(self, skill_id: str) -> None:
        sig = self._signatures.pop(skill_id, None)
        if sig is None:
            return
        for b, bh in enumerate(self._band_hashes(sig)):
            self._buckets[b][bh].discard(skill_id)

    def update_skill(self, skill_id: str, effects: Iterable[str]) -> None:
        self.remove(skill_id)
        self.add(skill_id, effects)

    def query(
        self, effects: Iterable[str], exclude: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Return candidate skill_ids that are likely high-Jaccard."""
        sig = self._compute_signature(effects)
        candidates: Set[str] = set()
        for b, bh in enumerate(self._band_hashes(sig)):
            candidates.update(self._buckets[b].get(bh, ()))
        if exclude:
            candidates -= exclude
        return candidates

    def candidate_pairs(self) -> Set[FrozenSet[str]]:
        """Return all candidate pairs that share at least one LSH bucket."""
        pairs: Set[FrozenSet[str]] = set()
        for bucket_dict in self._buckets:
            for bucket in bucket_dict.values():
                items = sorted(bucket)
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        pairs.add(frozenset((items[i], items[j])))
        return pairs

    def build_from_profiles(self, profiles: Dict[str, SkillProfile]) -> None:
        self._signatures.clear()
        self._buckets = [defaultdict(set) for _ in range(self._bands)]
        for sid, prof in profiles.items():
            self.add(sid, prof.all_effects)


# ═════════════════════════════════════════════════════════════════════
# C) Embedding ANN (lightweight wrapper; uses brute-force or FAISS)
# ═════════════════════════════════════════════════════════════════════


class EmbeddingANN:
    """Optional approximate nearest-neighbour search on centroid embeddings.

    Falls back to brute-force cosine if FAISS is unavailable.
    """

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._vecs: List[List[float]] = []

    def add(self, skill_id: str, centroid: List[float]) -> None:
        self._ids.append(skill_id)
        self._vecs.append(centroid)

    def remove(self, skill_id: str) -> None:
        if skill_id in self._ids:
            idx = self._ids.index(skill_id)
            self._ids.pop(idx)
            self._vecs.pop(idx)

    def update_skill(self, skill_id: str, centroid: List[float]) -> None:
        self.remove(skill_id)
        self.add(skill_id, centroid)

    def query(
        self,
        centroid: List[float],
        k: int = 5,
        exclude: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return top-k (skill_id, cosine_similarity) by brute-force."""
        if not self._vecs:
            return []
        results: List[Tuple[str, float]] = []
        for sid, vec in zip(self._ids, self._vecs):
            if exclude and sid in exclude:
                continue
            sim = self._cosine(centroid, vec)
            results.append((sid, sim))
        results.sort(key=lambda x: -x[1])
        return results[:k]

    def build_from_profiles(self, profiles: Dict[str, SkillProfile]) -> None:
        self._ids.clear()
        self._vecs.clear()
        for sid, prof in profiles.items():
            if prof.embedding_centroid is not None:
                self.add(sid, prof.embedding_centroid)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
