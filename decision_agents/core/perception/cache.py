"""Per-episode evidence cache.

The harness owns one :class:`EvidenceCache` per episode.  All
perception-op results — detector hits, segmentation masks, OCR
strings — are keyed by ``(image_hash, op, args_blob)`` so identical
queries on the same image collapse to one backend call.

This is a hard correctness *and* performance requirement:

* **Correctness** — when ``LOOK("close button")`` runs at step 3 and
  ``LOOK("close button")`` runs again at step 5, both hops must
  resolve to the same bbox (same ``eid``).  The detector itself is
  deterministic but Grounding-DINO inference is non-trivial — we
  skip the second call.
* **Performance** — Grounding-DINO 1.5 Edge is ~150 ms / query on an
  A100.  At one ``LOOK`` per step on a 200-step Browser/OS rollout,
  uncached repetition would add ~30 s wall-time per episode.  With
  cache, the second call is a dict lookup.

The cache is **per-episode**: ``Harness.reset()`` clears it.  Two
parallel rollouts on the same image must get separate caches so a
mutating environment (browser scroll, game frame advance) doesn't
serve stale crops to the other rollout.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any, Optional

_LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_SIZE: int = 256


# ──────────────────────────────────────────────────────────────────────
# Hash helpers
# ──────────────────────────────────────────────────────────────────────


def hash_image_bytes(image_bytes: bytes) -> str:
    """Return a short SHA1 hex prefix (16 chars) for ``image_bytes``.

    16 hex chars = 64 bits of entropy — collision-safe for any
    realistic per-episode rollout (a million distinct frames still
    has < 10⁻⁶ collision probability).  Short to keep cache keys
    cheap in logs.
    """
    return hashlib.sha1(image_bytes).hexdigest()[:16]


def serialise_args(args: Any) -> str:
    """Render ``args`` to a deterministic string key.

    ``json.dumps(sort_keys=True)`` for dicts/lists; ``str(args)`` for
    primitives.  Tuples become lists (JSON has no tuple).  The result
    is used as the cache key suffix, never deserialised — collisions
    only matter in the structural sense.
    """
    if args is None:
        return ""
    if isinstance(args, (str, int, float, bool)):
        return str(args)
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(args)


# ──────────────────────────────────────────────────────────────────────
# EvidenceCache
# ──────────────────────────────────────────────────────────────────────


class EvidenceCache:
    """Bounded LRU cache keyed by ``(image_hash, op, args_blob)``.

    Parameters
    ----------
    max_entries
        Hard cap before LRU eviction kicks in.  Default 256.  Sized
        for typical 30-200-step rollouts where a single image is
        queried 5-30 times across the episode.

    Behaviour
    ---------
    Insertion-ordered ``OrderedDict`` under the hood; on each ``get``
    the entry is moved to the end (LRU touch).  ``put`` evicts the
    oldest entry when the cap is hit.  ``clear`` resets the whole
    cache — called by ``Harness.reset()``.
    """

    def __init__(self, max_entries: int = DEFAULT_CACHE_SIZE) -> None:
        self.max_entries = max(1, int(max_entries))
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def make_key(image_hash: str, op: str, args: Any) -> str:
        """Compose the canonical cache key.

        Public so harness implementations and tests can construct
        keys without going through ``get`` / ``put`` (useful for
        cache-pre-warming experiments).
        """
        return f"{image_hash}::{op}::{serialise_args(args)}"

    def get(
        self,
        image_hash: str,
        op: str,
        args: Any = None,
    ) -> Optional[Any]:
        """Return the cached value or ``None``; bumps hit/miss counters."""
        key = self.make_key(image_hash, op, args)
        if key in self._store:
            self._hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self._misses += 1
        return None

    def put(
        self,
        image_hash: str,
        op: str,
        args: Any,
        value: Any,
    ) -> None:
        """Store ``value``; evict LRU entry when over ``max_entries``."""
        key = self.make_key(image_hash, op, args)
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            evicted_key, _ = self._store.popitem(last=False)
            _LOGGER.debug("EvidenceCache evicted %s", evicted_key)

    def clear(self) -> None:
        """Drop every entry; reset hit/miss counters.

        Called by ``Harness.reset()`` so a fresh episode never sees
        evidence carried over from the previous one.
        """
        self._store.clear()
        self._hits = 0
        self._misses = 0

    # ── observability ────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def stats(self) -> dict:
        """Return a small dict suitable for logging onto rollout records."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }


__all__ = [
    "EvidenceCache",
    "DEFAULT_CACHE_SIZE",
    "hash_image_bytes",
    "serialise_args",
]
